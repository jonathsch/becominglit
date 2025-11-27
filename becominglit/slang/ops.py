import os

import numpy as np
import slangtorch
import torch

slang_bsdf = slangtorch.loadModule(os.path.join(os.path.dirname(__file__), "pbr.slang"))

#######################################
# Specular BSDF
#######################################


class _slang_specular_bsdf_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, albedo, roughness, spec_vis, pos, nrm, view_pos, light_pos, light_intensity, num_lights):
        ctx.save_for_backward(albedo, roughness, spec_vis, pos, nrm, view_pos, light_pos, light_intensity, num_lights)
        return slang_bsdf.specular_fwd(
            albedo, roughness, spec_vis, pos, nrm, view_pos, light_pos, light_intensity, num_lights, albedo.dim()
        )

    @staticmethod
    def backward(ctx, grad_out):
        albedo, roughness, spec_vis, pos, nrm, view_pos, light_pos, light_intensity, num_lights = ctx.saved_tensors
        (
            albedo_grad,
            roughness_grad,
            spec_vis_grad,
            pos_grad,
            nrm_grad,
        ) = slang_bsdf.specular_bwd(
            albedo,
            roughness,
            spec_vis,
            pos,
            nrm,
            view_pos,
            light_pos,
            light_intensity,
            num_lights,
            albedo.dim(),
            grad_out,
        )
        return albedo_grad, roughness_grad, spec_vis_grad, pos_grad, nrm_grad, None, None, None, None


def specular_bsdf_slang(albedo, roughness, spec_vis, pos, nrm, view_pos, light_pos, light_intensity, num_lights):
    assert albedo.dim() == 3 and albedo.shape[-1] == 3, f"albedo must be of shape (B, N, 3), got {albedo.shape}"
    assert roughness.dim() == 3 and roughness.shape[-1] == 1, (
        f"roughness must be of shape (B, N, 1), got {roughness.shape}"
    )
    assert spec_vis.dim() == 3 and spec_vis.shape[-1] == 1, f"spec_vis must be of shape (B, N, 1), got {spec_vis.shape}"
    assert pos.dim() == 3 and pos.shape[-1] == 3, f"pos must be of shape (B, N, 3), got {pos.shape}"
    assert nrm.dim() == 3 and nrm.shape[-1] == 3, f"nrm must be of shape (B, N, 3), got {nrm.shape}"
    assert view_pos.dim() == 2 and view_pos.shape[-1] == 3, f"view_pos must be of shape (B, 3), got {view_pos.shape}"
    assert light_pos.dim() == 3 and light_pos.shape[-1] == 3, (
        f"light_pos must be of shape (B, L, 3), got {light_pos.shape}"
    )
    assert (
        light_intensity.dim() == 3
        and light_intensity.shape[-1] == 3
        and light_intensity.shape[-2] == light_pos.shape[-2]
    ), f"light_intensity must be of shape (B, L, 1), got {light_intensity.shape}"
    assert num_lights.shape[0] == albedo.shape[0], f"num_lights must be of shape (B,), got {num_lights.shape}"

    albedo = albedo.contiguous()
    roughness = roughness.contiguous()
    spec_vis = spec_vis.contiguous()
    pos = pos.contiguous()
    nrm = nrm.contiguous()
    view_pos = view_pos.contiguous()
    light_pos = light_pos.contiguous()
    light_intensity = light_intensity.contiguous()
    num_lights = num_lights.reshape(-1, 1).contiguous()

    return _slang_specular_bsdf_func.apply(
        albedo, roughness, spec_vis, pos, nrm, view_pos, light_pos, light_intensity, num_lights
    )


#######################################
# Cubemap operations
#######################################

slang_cubemap = slangtorch.loadModule(os.path.join(os.path.dirname(__file__), "cubemap.slang"))


class _diffuse_cubemap_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        out = slang_cubemap.diffuse_cubemap_fwd(cubemap)
        ctx.save_for_backward(cubemap)
        return out

    @staticmethod
    def backward(ctx, dout):
        (cubemap,) = ctx.saved_variables
        cubemap_grad = slang_cubemap.diffuse_cubemap_bwd(cubemap, dout)
        return cubemap_grad, None


def diffuse_cubemap(cubemap, use_python=False):
    if use_python:
        assert False
    else:
        out = _diffuse_cubemap_func.apply(cubemap)
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of diffuse_cubemap contains inf or NaN"
    return out


class _specular_cubemap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap, roughness, costheta_cutoff, bounds):
        out = slang_cubemap.specular_cubemap_fwd(cubemap, bounds, roughness, costheta_cutoff)
        ctx.save_for_backward(cubemap, bounds)
        ctx.roughness, ctx.theta_cutoff = roughness, costheta_cutoff
        return out

    @staticmethod
    def backward(ctx, dout):
        cubemap, bounds = ctx.saved_variables
        cubemap_grad = slang_cubemap.specular_cubemap_bwd(cubemap, bounds, ctx.roughness, ctx.theta_cutoff, dout)
        return cubemap_grad, None, None, None


# Compute the bounds of the GGX NDF lobe to retain "cutoff" percent of the energy
def __ndfBounds(res, roughness, cutoff):
    def ndf_blinn_phong(alpha, cos_theta):
        return 0.5 * (alpha + 2.0) * (1.0 / np.pi) * np.power(cos_theta, alpha)

    def ndf_blinn_phong_mix(roughness, cos_theta):
        return roughness * ndf_blinn_phong(12.0, cos_theta) + (1.0 - roughness) * ndf_blinn_phong(48.0, cos_theta)

    # Sample out cutoff angle
    nSamples = 1000000
    costheta = np.cos(np.linspace(0, np.pi / 2.0, nSamples))
    D = np.cumsum(ndf_blinn_phong_mix(roughness, costheta))
    idx = np.argmax(D >= D[..., -1] * cutoff)

    # Brute force compute lookup table with bounds
    bounds = slang_cubemap.specular_bounds(res, costheta[idx])

    return costheta[idx], bounds


__ndfBoundsDict = {}


def specular_cubemap(cubemap, roughness, cutoff=0.99, use_python=False):
    assert cubemap.shape[0] == 6 and cubemap.shape[1] == cubemap.shape[2], "Bad shape for cubemap tensor: %s" % str(
        cubemap.shape
    )

    if use_python:
        assert False
    else:
        key = (cubemap.shape[1], roughness, cutoff)
        if key not in __ndfBoundsDict:
            __ndfBoundsDict[key] = __ndfBounds(*key)
        out = _specular_cubemap.apply(cubemap, roughness, *__ndfBoundsDict[key])
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of specular_cubemap contains inf or NaN"
    return out[..., 0:3] / out[..., 3:]
