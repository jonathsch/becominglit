import os
from typing import Any, Dict, List, Mapping

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as thf
from torch import nn

from becominglit.slang.ops import specular_cubemap
from becominglit.util import envmap, image, vecmath
from becominglit.util.torchutils import to_device


class PointLightPathDecorator(torch.nn.Module):
    def __init__(
        self,
        mod: nn.Module,
        cycle: int = 256,
        light_rotate_axis: int = 1,
    ) -> None:
        super().__init__()
        self.mod = mod

        self.cycle = cycle
        self.light_rotate_axis = light_rotate_axis
        self.register_buffer("light_intensity", torch.tensor([1, 1, 1]).reshape(1, 3) * 1.5)

    def forward(self, **data: Dict[str, Any]) -> Dict[str, Any]:
        device = data["campos"].device
        batch_size = data["campos"].size(0)

        light_pos = []

        for i in range(batch_size):
            index = data["index"][i] % self.cycle

            angle = (abs(index) / self.cycle) * 2 * np.pi
            if self.light_rotate_axis == 0:
                cur_lpos = np.asarray([0.0, 1100.0 * np.sin(angle), 1100.0 * np.cos(angle)]).astype(np.float32)
            elif self.light_rotate_axis == 1:
                cur_lpos = np.asarray([-1.0 * 1100.0 * np.sin(angle), 300.0, 1100.0 * np.abs(np.cos(angle))]).astype(
                    np.float32
                )
            else:
                cur_lpos = np.asarray([1100.0 * np.cos(angle), 1100.0 * np.sin(angle), 0.0]).astype(np.float32)

            light_pos.append(torch.from_numpy(cur_lpos).to(device))

        light_intensity = self.light_intensity.clone().unsqueeze(0).repeat(batch_size, 1, 1).float().to(device)
        data["light_intensity"] = light_intensity

        light_pos = torch.stack(light_pos, dim=0).float().to(device)
        data["light_pos"] = light_pos
        data["n_lights"] = torch.ones(batch_size, device=device).int()
        data["is_fullylit_frame"] = torch.zeros(1).to(device).bool()

        return self.mod(**data)


class EnvLightSpinDecorator(nn.Module):
    def __init__(
        self,
        mod: nn.Module,
        envmap_path: os.PathLike,
        assets: Mapping[str, torch.Tensor],
        envmap_dist: float = 10000.0,
        env_scale: float = 18.0,
        cycle: int = 256,
        ydown: bool = True,
    ) -> None:
        super().__init__()
        self.mod = mod
        self.envmap_path = envmap_path
        self.envmap_dist = envmap_dist
        self.env_scale = env_scale
        self.cycle = cycle
        self.ydown = ydown

        self.LIGHT_MIN_RES = 16 * 2

        self.envmap_base = torch.as_tensor(iio.imread(envmap_path)).float()  # HDR map
        cubemap = envmap.latlong_to_cubemap(self.envmap_base.cuda(), [512, 512])
        self.cubemap_mip = self.build_mips(cubemap)

        self.register_buffer("light_pos", assets["light_positions"])
        self.register_buffer("fg_lut", assets["fg_lut"])

        L = 16
        theta, phi = np.meshgrid(
            (np.arange(L, dtype=np.float32) + 0.5) * np.pi / L,
            (np.arange(-L, L, dtype=np.float32) + 0.5) * np.pi / L,
            indexing="ij",
        )
        sph = np.stack(
            [np.sin(theta) * np.sin(phi), np.cos(theta), -np.sin(theta) * np.cos(phi)],
            axis=0,
        ).reshape((3, -1))
        self.register_buffer("sphvec", torch.from_numpy(sph))

    def build_mips(self, cubemap_base, cutoff=0.99) -> List[torch.Tensor]:
        specular = [cubemap_base]
        while specular[-1].shape[1] > self.LIGHT_MIN_RES:
            specular += [image.avg_pool_nhwc(specular[-1], (2, 2))]

        for idx in range(len(specular) - 1):
            roughness = (idx / (len(specular) - 2)) * (
                envmap.MAX_ROUGHNESS - envmap.MIN_ROUGHNESS
            ) + envmap.MIN_ROUGHNESS
            specular[idx] = specular_cubemap(specular[idx], roughness, cutoff)
        specular[-1] = specular_cubemap(specular[-1], 1.0, cutoff)

        return specular

    def forward(self, **data: Dict[str, Any]) -> Dict[str, Any]:
        device = data["campos"].device
        batch_size = data["campos"].size(0)

        light_intensity = []
        envbg = []
        light_dir = []
        envmaps = []

        lightrots = torch.zeros(batch_size, 3, 3).float().to(device)

        for i in range(batch_size):
            index = data["index"][i]
            rot_y = 2.0 * np.pi * index / self.cycle
            axis = torch.Tensor([0.0, 1.0, 0.0])
            quat = torch.Tensor(axis.tolist() + [rot_y]).float()
            quat = thf.normalize(quat[0:3], dim=0) * quat[3]
            rot_mat = vecmath.rvec_to_R(quat)
            new_env = envmap.rotate_envmap_mat(self.envmap_base.permute(2, 0, 1).cpu(), rot_mat)

            lightrots[i] = rot_mat
            perc90 = np.percentile(self.envmap_base.data.cpu().numpy(), 90)
            envbg.append(new_env / (perc90 if perc90 > 0 else new_env.max().item()) * 255)

            for i in range(10):
                new_env = thf.avg_pool2d(new_env[None], 5, stride=1, padding=1)[0]

            new_env = thf.interpolate(new_env[None], (16, 32), mode="bilinear", antialias=True)[0]

            new_env_sin = (
                new_env * torch.sin((torch.arange(new_env.shape[1]) + 0.5) * np.pi / new_env.shape[1])[None, :, None]
            )
            new_env = self.env_scale * new_env / new_env_sin.sum()

            envmaps.append(new_env)

            light_intensity.append(new_env.view(3, -1).t())
            light_dir.append(self.sphvec.view(3, -1).t())

        envbg = torch.stack(envbg, dim=0).float().to(device)
        light_intensity = torch.stack(light_intensity, dim=0).float().to(device)
        light_dir = torch.stack(light_dir, dim=0).float().to(device)

        data["envmap_mip"] = to_device(self.cubemap_mip, device)
        data["lightrot"] = lightrots
        data["envbg"] = envbg / 255.0
        data["n_lights"] = data["light_intensity"].shape[1] * torch.ones(batch_size, 1).to(device)
        data["is_fullylit_frame"] = torch.zeros(1).to(device)
        data["light_intensity"] = torch.ones_like(data["light_intensity"]) * (0.5 / 3.0)
        data["fg_lut"] = self.fg_lut.to(device)

        out = self.mod(**data)
        return out
