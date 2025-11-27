import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as thf

from becominglit.util import vecmath

MAX_ROUGHNESS = 0.9
MIN_ROUGHNESS = 0.08


def rotx(theta: float) -> np.ndarray:
    """
    Produces a counter-clockwise 3D rotation matrix around axis X with angle `theta` in radians.
    """
    return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]], dtype="float32")


def roty(theta: float) -> np.ndarray:
    """
    Produces a counter-clockwise 3D rotation matrix around axis Y with angle `theta` in radians.
    """
    return np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]], dtype="float32")


def rotz(theta: float) -> np.ndarray:
    """
    Produces a counter-clockwise 3D rotation matrix around axis Z with angle `theta` in radians.
    """
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]], dtype="float32")


def rotate_envmap(image: torch.Tensor, rot_x: float = 0, rot_y: float = 0, rot_z: float = 0) -> torch.Tensor:
    # image: envmap, tensor 3 x H x W
    # rot_x, rot_y, rot_z: in radians
    # return: 3 x H x W

    height = image.shape[1]
    width = image.shape[2]
    theta, phi = torch.meshgrid(
        (torch.arange(height, dtype=torch.float32) + 0.5) * 3.1415926 / height,
        (torch.arange(-width // 2, width // 2, dtype=torch.float32) + 0.5) * 3.1415926 * 2 / width,
        indexing="ij",
    )

    # H x W x 3
    vec = torch.stack(
        [torch.sin(theta) * torch.sin(phi), torch.cos(theta), torch.sin(theta) * torch.cos(phi)], dim=-1
    ).to(image.device)
    rot_mat = rotz(rot_z).dot(roty(rot_y)).dot(rotx(rot_x))
    rot_mat = torch.from_numpy(rot_mat).T
    rot_mat = rot_mat.contiguous().to(image.device)

    # rot vec
    vec = torch.matmul(vec, rot_mat[None, :, :])
    vec = torch.clamp(vec, -1, 1)

    u = (1 / np.pi) * torch.atan2(vec[:, :, 0], vec[:, :, 2])  # range: [-1, 1]
    v = (1 / np.pi) * torch.acos(vec[:, :, 1])  # range: [0, 1]
    v = 2 * v - 1.0

    coords = torch.stack([u, v], -1)

    new_image = thf.grid_sample(image[None, :, :, :], coords[None, :, :, :], padding_mode="border")
    new_image = new_image[0]

    return new_image


def rotate_envmap_mat(image: torch.Tensor, rot_mat: torch.Tensor) -> torch.Tensor:
    # image: 3 x H x W tensor
    # rot_mat: (3,3), rotation matrix

    height = image.shape[1]
    width = image.shape[2]
    theta, phi = torch.meshgrid(
        (torch.arange(height, dtype=torch.float32) + 0.5) * 3.1415926 / height,
        (torch.arange(-width // 2, width // 2, dtype=torch.float32) + 0.5) * 3.1415926 * 2 / width,
        indexing="ij",
    )

    # H x W x 3
    vec = torch.stack(
        [torch.sin(theta) * torch.sin(phi), torch.cos(theta), torch.sin(theta) * torch.cos(phi)], dim=-1
    ).to(image.device)

    rot_mat = rot_mat.T.float().contiguous()
    vec = torch.matmul(vec, rot_mat[None, :, :])
    vec = torch.clamp(vec, -1, 1)

    u = (1 / np.pi) * torch.atan2(vec[:, :, 0], vec[:, :, 2])  # range: [-1, 1]
    v = (1 / np.pi) * torch.acos(vec[:, :, 1])  # range: [0, 1]
    v = 2 * v - 1.0

    coords = torch.stack([u, v], -1)
    new_image = thf.grid_sample(image[None, :, :, :], coords[None, :, :, :], padding_mode="border")
    new_image = new_image[0]

    return new_image


def cube_to_dir(s, x, y):
    if s == 0:
        rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1:
        rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2:
        rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3:
        rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4:
        rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5:
        rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)


def latlong_to_cubemap(latlong_map, res):
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device="cuda")
    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
            indexing="ij",
        )
        v = vecmath.safe_normalize(cube_to_dir(s, gx, gy))

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        H, W, C = latlong_map.shape[:3]
        cubemap[s, ...] = dr.texture(latlong_map.view(1, H, W, C), texcoord[None, ...], filter_mode="linear")[0]
    return cubemap


def get_mip(roughness, num_levels):
    return torch.where(
        roughness < MAX_ROUGHNESS,
        (torch.clamp(roughness, MIN_ROUGHNESS, MAX_ROUGHNESS) - MIN_ROUGHNESS)
        / (MAX_ROUGHNESS - MIN_ROUGHNESS)
        * (num_levels - 2),
        (torch.clamp(roughness, MAX_ROUGHNESS, 1.0) - MAX_ROUGHNESS) / (1.0 - MAX_ROUGHNESS) + num_levels - 2,
    )


def env_diffuse(preconv_envmap, dirs):
    return dr.texture(
        preconv_envmap[-1][None, ...], dirs[:, :, None, :].contiguous(), filter_mode="linear", boundary_mode="cube"
    ).squeeze(2)


def env_specular(preconv_envmap, fg_lut, refl_dirs, nrm, roughness, wo):
    n_dot_v = torch.clamp(vecmath.dot(wo, nrm), min=1e-4)
    fg_uv = torch.cat((n_dot_v, roughness), dim=-1)
    fg_lookup = dr.texture(fg_lut, fg_uv[:, :, None, :], filter_mode="linear", boundary_mode="clamp")
    miplevel = get_mip(roughness, len(preconv_envmap))
    spec = (
        dr.texture(
            preconv_envmap[0][None, ...],
            refl_dirs[:, :, None, :].contiguous(),
            mip=list(m[None, ...] for m in preconv_envmap[1:]),
            mip_level_bias=miplevel[..., None, 0],
            filter_mode="linear-mipmap-linear",
            boundary_mode="cube",
        )
    ) * 0.1

    return (spec * (fg_lookup[..., 0:1] + fg_lookup[..., 1:2])).squeeze(2)


def envmap_to_image(
    w: int,
    h: int,
    envbg: torch.Tensor,
    princpt: torch.Tensor,
    focal: torch.Tensor,
    camrot: torch.Tensor = None,
    focal_scale: float = 0.2,
    blurbg: bool = True,
    D: torch.Tensor = None,
) -> torch.Tensor:
    py, px = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing="ij")
    pixelcoords = torch.stack([px, py], -1)[None].expand(princpt.shape[0], -1, -1, -1).to(envbg.device)

    if D is not None:
        focal_np = focal.float().data.cpu().numpy()
        princpt_np = princpt.float().data.cpu().numpy()
        D_np = D.float().data.cpu().numpy()
        pix_coords = pixelcoords[0].float().data.cpu().numpy()

        pixel_coords = []
        for bi in range(princpt.shape[0]):
            Ki = np.concatenate([focal_np[bi], princpt_np[bi][:, None]], -1)
            Ki = np.concatenate([Ki, np.array([[0, 0, 1]])], -2)
            Di = D_np[bi]

            pixelcoords_undistorted = cv2.fisheye.undistortPoints(pix_coords, Ki, Di, P=Ki)  # h * w * 2

        pixel_coords.append(pixelcoords_undistorted)
        pixelcoords = torch.from_numpy(np.stack(pixel_coords)).contiguous().to(envbg.device)

    # convert pixel coords into directions
    raydir = pixelcoords - princpt[:, None, None, :]
    raydir[..., 0] /= focal[:, None, None, 0, 0] * focal_scale
    raydir[..., 1] /= focal[:, None, None, 1, 1] * focal_scale
    raydir = torch.cat([raydir, torch.ones_like(raydir[:, :, :, 0:1])], dim=-1)
    if camrot is not None:
        raydir = torch.einsum("bxy,bhwx->bhwy", camrot, raydir)
    raydir = thf.normalize(raydir, dim=-1)
    u = (1 / np.pi) * torch.atan2(raydir[..., 0], raydir[..., 2])  # range: [-1, 1]
    v = (1 / np.pi) * torch.acos(raydir[..., 1])  # range: [0, 1]
    v = 2 * v - 1.0
    uv = torch.stack([u, v], -1)
    envbg = thf.grid_sample(envbg, uv, mode="bicubic", padding_mode="border", align_corners=True)

    if blurbg:
        blurkernel = torch.exp(-(torch.linspace(-4.0, 4.0, 101) ** 2))
        blurkernel = blurkernel[:, None] * blurkernel[None, :]
        blurkernel = blurkernel / torch.sum(blurkernel)
        blurkernel = blurkernel[None, None, :, :].repeat(3, 1, 1, 1).to(princpt.device)
        envbg = thf.conv2d(envbg, weight=blurkernel, stride=1, padding=50, groups=3)
        envbg = thf.interpolate(envbg, size=(h, w))

    return envbg


def envmap_to_mirrorball(w, h, env, camrot=None):
    py, px = torch.meshgrid(torch.linspace(-1.0, 1.0, h), torch.linspace(-1.0, 1.0, w))
    pixelcoords = torch.stack([px, py], -1)[None].expand(env.shape[0], -1, -1, -1).to(env.device)
    zsq = pixelcoords.pow(2).sum(-1, keepdim=True)
    mask = (zsq < 1.0).float()[:, None, :, :, 0]
    nz = -(1.0 - zsq).clamp(min=0.0).sqrt()
    nml = torch.cat([pixelcoords, nz], -1)
    ref = -2.0 * nz * nml
    ref[..., 2] = 1.0 + ref[..., 2]
    if camrot is not None:
        ref = torch.einsum("bxy,bhwx->bhwy", camrot, ref)
    envball = 255.0 * (0.5 * ref.permute(0, 3, 1, 2) + 0.5)
    u = (1 / np.pi) * torch.atan2(ref[..., 0], ref[..., 2])  # range: [-1, 1]
    v = (1 / np.pi) * torch.acos(ref[..., 1])  # range: [0, 1]
    v = 2 * v - 1.0
    uv = torch.stack([u, v], -1)
    envball = thf.grid_sample(env, uv, mode="bicubic", padding_mode="border", align_corners=True)

    return torch.cat([envball, mask], 1)


def compose_envmap(render, alpha, envbg, K, Rt):
    envbg = torch.stack(
        [rotate_envmap(envbg[i], rot_y=np.pi) for i in range(envbg.shape[0])]
    )  # rotate 180 degree to match coord system

    env_mirror = envmap_to_mirrorball(200, 200, envbg, Rt[:, :3, :3])
    # to offset mugsy color correction
    env_mirror[:, 0] = env_mirror[:, 0]
    env_mirror[:, 2] = env_mirror[:, 2]

    mirror_img = torch.zeros_like(render)
    mirror_alpha = torch.zeros_like(alpha)
    mirror_alpha[:, :, -200:, -200:] = env_mirror[:, 3:]
    mirror_img[:, :, -200:, -200:] = env_mirror[:, :3]

    envbg = envmap_to_image(render.shape[-1], render.shape[-2], envbg, K[:, :2, 2], K, Rt[:, :3, :3])
    # to offset mugsy color correction
    envbg[:, 0] = envbg[:, 0]
    envbg[:, 2] = envbg[:, 2]
    render = render + (1.0 - alpha) * envbg.clamp(0, 1.0)
    render = (1.0 - mirror_alpha) * render + mirror_alpha * mirror_img

    return render
