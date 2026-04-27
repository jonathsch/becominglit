from typing import Dict, Optional

import gsplat
import torch
import torch.nn.functional as thf
from omegaconf import DictConfig
from torch import nn
from torchvision.utils import make_grid

import becominglit.nn.layers as la
from becominglit.util import envmap, image, sh, vecmath
from becominglit.util.geometry import GeometryModule, depth2normals
from becominglit.util.lbs import batch_rodrigues
from becominglit.util.pbr import polface_specular_bsdf


class FlameEncoder(nn.Module):
    def __init__(self, input_ch: int):
        super().__init__()
        self.uv_enc = nn.Sequential(*la.make_linear(input_ch, 256 * 8 * 8, "wn", nn.LeakyReLU(0.2, inplace=True)))
        self.apply(lambda m: la.glorot(m, 0.2))

    def forward(self, f_flame: torch.Tensor):
        preds = {}

        # UV encoding
        uv_embed = self.uv_enc(f_flame).view(-1, 256, 8, 8)  # [B, 256 * 8 * 8] -> [B, 256, 8, 8]
        preds.update(embs_uv=uv_embed, embs=f_flame)

        return preds


class NeuralBSDF(nn.Module):
    def __init__(
        self,
        sh_degree: int = 6,
        feat_ch: int = 32,
        out_ch: int = 1,
        color_ch: int = 1,
    ):
        super().__init__()
        self.sh_degree = sh_degree
        self.feat_ch = feat_ch
        self.out_ch = out_ch

        self.n_sh_coeff = color_ch * (sh_degree + 1) ** 2
        self.input_dim = self.n_sh_coeff + feat_ch

        self.net = nn.Sequential(
            *la.make_linear(self.input_dim, 64, "wn", nn.LeakyReLU(0.2, inplace=True)),
            *la.make_linear(64, 64, "wn", nn.LeakyReLU(0.2, inplace=True)),
            *la.make_linear(64, out_ch, "wn", nn.Softplus()),
        )

        self.apply(lambda m: la.glorot(m, 0.2))
        la.glorot(self.net[-1], 1.0)

    def forward(
        self,
        sh_coeff: torch.Tensor,
        features: torch.Tensor,
    ):
        """
        Args:
            sh_coeff: Incident light SH coefficients in local frame, shape (B, N, n_sh_coeff)
            features: Per-gaussian features of shape (B, N, feat_ch)
        """
        B, N, _ = features.shape
        x = torch.cat([sh_coeff.repeat(1, N, 1), features], dim=-1)
        x = self.net(x.reshape(-1, self.input_dim))
        return x.view(B, N, self.out_ch)


class BecomingLitModel(nn.Module):
    def __init__(
        self,
        flame_config: DictConfig,
        diff_sh_deg: int = 6,
        init_albedo: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__()
        self.uv_size = 512
        self.num_prims = self.uv_size**2
        self.diff_sh_deg = diff_sh_deg

        # define static primitive attributes
        if init_albedo is not None:
            init_albedo = image.scale_img_hwc(init_albedo, (self.uv_size, self.uv_size))
            init_albedo = image.srgb2linear(init_albedo)
        else:
            init_albedo = torch.rand(self.uv_size, self.uv_size, 3)

        self.means = nn.Parameter(torch.zeros(self.uv_size, self.uv_size, 3))
        self.albedo = nn.Parameter(init_albedo)
        self.roughness = nn.Parameter(torch.zeros(self.uv_size, self.uv_size, 1))

        # flame function
        self.flame_mod = GeometryModule(flame_config, uv_size=self.uv_size)

        # expression encoder
        enc_in_ch = flame_config["expr_params"] + 3 + 6
        self.expr_enc = FlameEncoder(enc_in_ch)

        # view encoder
        self.view_enc = nn.Sequential(*la.make_linear(3, 8, "wn", nn.LeakyReLU(0.2, inplace=True)))

        # geometry module
        geomod_ch = 11 + 32  # gaussian primitives + features
        self.geo_mod = nn.Sequential(
            *la.make_conv_trans(256, 256, 4, 2, 1, "wn", nn.LeakyReLU(0.2, inplace=True), ub=(16, 16)),
            *la.make_conv_trans(256, 128, 4, 2, 1, "wn", nn.LeakyReLU(0.2, inplace=True), ub=(32, 32)),
            *la.make_conv_trans(128, 128, 4, 2, 1, "wn", nn.LeakyReLU(0.2, inplace=True), ub=(64, 64)),
            *la.make_conv_trans(128, 64, 4, 2, 1, "wn", nn.LeakyReLU(0.2, inplace=True), ub=(128, 128)),
            *la.make_conv_trans(64, 32, 4, 2, 1, "wn", nn.LeakyReLU(0.2, inplace=True), ub=(256, 256)),
            *la.make_conv_trans(32, geomod_ch, 4, 2, 1, "wn", ub=(512, 512)),
        )

        # specular module
        specmod_ch = 32 + 8  # features from geomod + view_enc
        spec_ch = 4  # d_nml + spec_vis
        self.spec_mod = nn.Sequential(
            *la.make_conv(specmod_ch, 32, 4, 2, 1, "wn", nn.LeakyReLU(0.2, inplace=True), ub=(256, 256)),
            *la.make_conv_trans(32, spec_ch, 4, 2, 1, "wn", ub=(512, 512)),
        )

        # diffuse bsdf module
        self.bsdf_mod = NeuralBSDF(sh_degree=self.diff_sh_deg)

        self.apply(lambda m: la.glorot(m, 0.2))
        la.glorot(self.geo_mod[-1], 1.0)
        la.glorot(self.spec_mod[-1], 1.0)

    def forward(
        self,
        flame_params: Dict[str, torch.Tensor],
        Rt: torch.Tensor,
        K: torch.Tensor,
        width: int,
        height: int,
        light_pos: torch.Tensor,
        light_intensity: torch.Tensor,
        is_fullylit_frame: bool,
        n_lights: torch.Tensor,
        flame_verts: Optional[torch.Tensor] = None,
        envmap_mip: Optional[torch.Tensor] = None,
        envbg: Optional[torch.Tensor] = None,
        lightrot: Optional[torch.Tensor] = None,
        fg_lut: Optional[torch.Tensor] = None,
        render_auxiliary: bool = False,
        **kwargs,
    ):
        batch_size = Rt.shape[0]
        preds = {}

        cam_to_world = torch.linalg.inv(Rt)  # [B, 4, 4]
        campos = cam_to_world[:, :3, 3]  # [B, 3]

        # head pose
        head_pose = torch.eye(4).to(Rt).unsqueeze(0).repeat(batch_size, 1, 1)
        head_pose[:, :3, :3] = batch_rodrigues(flame_params["rotation"])
        head_pose[:, :3, 3] = flame_params["translation"]

        # cam and lights in head space
        Rt_headrel = Rt @ head_pose
        headrel_campos = ((campos - head_pose[:, :3, 3])[:, None] @ head_pose[:, :3, :3])[:, 0]
        headrel_light_pos = (light_pos - head_pose[:, None, :3, 3]) @ head_pose[:, :3, :3]  # [B, L, 3]
        headrel_light_dir = thf.normalize(headrel_light_pos, p=2, dim=-1)
        sh_coeffs = sh.dir2sh_torch(self.diff_sh_deg, headrel_light_dir)
        headrel_light_sh = (sh_coeffs[:, :, None] * light_intensity[..., None]).sum(dim=1)  # [B, K, 3]

        # encode FLAME expression
        enc_input = torch.cat(
            [
                flame_params["expr"],
                flame_params["jaw_pose"],
                flame_params["eyes_pose"],
            ],
            dim=-1,
        )  # [B, enc_in_ch]
        enc_preds = self.expr_enc(enc_input)
        preds.update(**enc_preds)
        f_flame = enc_preds["embs_uv"]  # [B, 256, 8, 8]

        # main geometry module
        f_geo = self.geo_mod(f_flame)

        # encode view direction
        headrel_viewdir = thf.normalize(headrel_campos, dim=-1)  # [B, 3]
        f_view = self.view_enc(headrel_viewdir)[:, :, None, None].expand(-1, -1, 512, 512)  # [B, 8, 512, 512]
        f_dec = f_geo[:, 11:]  # [B, 32, 512, 512] # TODO: Find a better name for this variable
        embds_view = torch.cat([f_dec, f_view], dim=1)  # [B, 32 + 32 + 8, 512, 512]

        # specular module
        f_spec = self.spec_mod(embds_view).permute(0, 2, 3, 1)  # [B, 4, 512, 512]

        # gaussian primitives
        f_prim = f_geo[:, :11].permute(0, 2, 3, 1)
        posed_prims = self.warp_cano_to_posed(flame_params, flame_verts, f_prim, f_spec)
        preds.update(**posed_prims)

        mask = self.flame_mod.valid_mask.squeeze(-1)  # [U, V]
        pos = posed_prims["means"][:, mask, :].contiguous()  # [B, N, 3]
        nrm = posed_prims["normals"][:, mask, :].contiguous()  # [B, N, 3]
        quats = posed_prims["quats"][:, mask, :].contiguous()
        scales = posed_prims["scales"][:, mask, :].contiguous()
        opacities = posed_prims["opacities"][:, mask, :].contiguous()
        albedo = self.albedo[mask, :].contiguous().repeat(batch_size, 1, 1)
        f_dec = f_dec.permute(0, 2, 3, 1)[:, mask, :].contiguous()  # [B, N, 32]

        # roughness
        sigma = self.roughness[None].repeat(batch_size, 1, 1, 1)  # [B, N, 1]
        sigma = sigma[:, mask, :].contiguous()  # [B, N, 1]
        sigma = (torch.exp(sigma) * 0.1).clamp(min=0.01, max=1.0)

        # view-dependent specular visibility
        spec_vis = torch.sigmoid(f_spec[..., :1])[:, mask, :].contiguous()  # [B, N, 1]

        if envmap_mip is None:
            # point light shading
            diffuse = albedo * self.bsdf_mod.forward(headrel_light_sh.mean(dim=-2, keepdim=True), f_dec).repeat(1, 1, 3)
            wo_local = thf.normalize(headrel_campos[:, None] - pos, dim=-1)  # [B, N, 3]
            wi_local = thf.normalize(headrel_light_pos[:, None] - pos[:, :, None], dim=-1)  # [B, N, L, 3]
            specular = polface_specular_bsdf(spec_vis, sigma, wo_local, wi_local, light_intensity, nrm)
        else:
            # envmap shading
            ref_dirs = torch.einsum("bxy,bny->bnx", lightrot, nrm)
            light_col = envmap.env_diffuse(envmap_mip, ref_dirs)
            reflectance = self.bsdf_mod.forward(headrel_light_sh.mean(dim=-2, keepdim=True), f_dec).repeat(1, 1, 3)
            diffuse = albedo * reflectance * light_col  # [B, N, 3]

            wo_local = thf.normalize(headrel_campos[:, None] - pos, dim=-1)
            reflvec = vecmath.safe_normalize(vecmath.reflect(wo_local, nrm))
            reflvec = torch.einsum("bxy,bny->bnx", lightrot, reflvec)
            specular = envmap.env_specular(envmap_mip, fg_lut, reflvec, nrm, sigma, wo_local) * spec_vis  # [B, N, 3]

        preds["diff_color"] = diffuse
        color = diffuse.clamp_min(0.0) + specular

        # render
        if render_auxiliary:
            render_attr = torch.cat([color, albedo, diffuse, specular, nrm, sigma, spec_vis], dim=-1)
        else:
            render_attr = color
        render_features, render_alpha, _ = gsplat.rasterization(
            means=pos,
            quats=quats,
            scales=scales,
            opacities=opacities.squeeze(-1),
            colors=render_attr,
            viewmats=Rt_headrel[:, None],
            Ks=K[:, None],
            width=width,
            height=height,
            render_mode="RGB+D",
        )
        render_features = render_features.squeeze(1).permute(0, 3, 1, 2)  # [B, C, H, W]
        render_alpha = render_alpha.squeeze(1).permute(0, 3, 1, 2)  # [B, 1, H, W]
        render_rgb = render_features[:, :3]  # [B, 3, H, W]
        render_depth = render_features[:, -1:]  # [B, 1, H, W]
        preds.update(rgb=render_rgb, alpha=render_alpha, depth=render_depth)

        if envmap_mip is not None:
            render_rgb = envmap.compose_envmap(render_rgb, render_alpha, envbg, K, Rt)
            preds["rgb"] = render_rgb

        if render_auxiliary:
            render_albedo = render_features[:, 3:6]  # [B, 3, H, W]
            render_diffuse = render_features[:, 6:9]  # [B, 3, H, W]
            render_specular = render_features[:, 9:12]  # [B, 3, H, W]
            render_normals = render_features[:, 12:15]  # [B, 3, H, W]
            render_roughness = render_features[:, 15:16]  # [B, 1, H, W]
            render_spec_vis = render_features[:, 16:17]  # [B, 1, H, W]
            preds.update(
                albedo=render_albedo,
                render_diffuse=render_diffuse,
                render_specular=render_specular,
                render_normals=render_normals,
                render_roughness=render_roughness,
                render_spec_vis=render_spec_vis,
            )

        return preds

    def warp_cano_to_posed(self, flame_params, verts, f_geo, f_spec):
        # base geometry
        if verts is None:
            verts = self.flame_mod.forward(flame_params)  # [B, V, 3]
        tbn_frame_uv = self.flame_mod.get_tbn(verts)

        # Position
        pos_base = self.flame_mod.get_base_pos(verts)  # [B, U, V, 3]
        static_offset = self.means[None]  # [1, U, V, 3]
        gb_pos = pos_base + (tbn_frame_uv @ static_offset[..., None]).squeeze(-1) + f_geo[..., :3]  # [B, U, V, 3]

        # Rotation
        gb_quats = f_geo[..., 3:7]  # [B, U, V, 4]

        # Scales
        scales = nn.functional.softplus(f_geo[..., 7:10])  # [B, U, V, 3]
        gb_scales = scales.clamp(0.1, 20.0)

        # Opacities
        opacities = torch.sigmoid(f_geo[..., 10:11])

        # Normals
        local_normals = f_spec[..., 1:]
        gb_normals = thf.normalize(tbn_frame_uv[..., 2] + local_normals, dim=-1)  # [B, U, V, 3]

        return {
            "means": gb_pos,
            "scales": gb_scales,
            "scales_preclip": scales,
            "quats": gb_quats,
            "opacities": opacities,
            "normals": gb_normals,
            "normal_offsets": local_normals,
            "mesh_verts": verts,
            "mesh_normals": tbn_frame_uv[..., 2],
        }


class BecomingLitSummary:
    @torch.no_grad()
    def __call__(self, preds, batch):
        summary = {}

        summary["rgb"] = image.linear2srgb(preds["rgb"])
        summary["gt_rgb"] = image.linear2srgb(batch["image"])
        summary["alpha"] = preds["alpha"]
        summary["normal"] = preds["render_normals"] * 0.5 + 0.5
        summary["render_albedo"] = image.linear2srgb(preds["albedo"])
        summary["roughness"] = preds["render_roughness"]
        summary["render_spec_vis"] = preds["render_spec_vis"]
        summary["diffuse"] = image.linear2srgb(preds["render_diffuse"])
        summary["specular"] = image.linear2srgb(preds["render_specular"])
        summary["uv_normal"] = preds["normals"].permute(0, 3, 1, 2) * 0.5 + 0.5

        summary["depth_nml"] = (
            summary["alpha"] * (0.5 * -depth2normals(preds["depth"], batch["focal"], batch["princpt"]) + 0.5)
            + (1.0 - summary["alpha"]) * 0.5
        )

        for k, v in summary.items():
            summary[k] = make_grid(v.detach()).clamp(0, 1).mul(255).byte().cpu().permute(1, 2, 0).numpy()

        return summary
