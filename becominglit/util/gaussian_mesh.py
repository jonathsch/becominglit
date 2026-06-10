"""Convert 3D Gaussian primitives to ellipsoid meshes for visualization.

Each Gaussian is represented as an icosphere scaled by its per-axis standard
deviation, rotated by its orientation quaternion and translated to its mean.
The resulting (verts, faces) can be rendered with the existing
``becominglit.util.mesh.NVDiffRenderer`` to visualize the Gaussian point cloud
as solid geometry.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_to_matrix
from pytorch3d.utils import ico_sphere


@lru_cache(maxsize=8)
def _unit_icosphere(level: int, device_str: str):
    """Cached unit-sphere icosphere (verts on the unit sphere, triangle faces)."""
    mesh = ico_sphere(level)
    verts = mesh.verts_packed().to(device_str)  # [Vs, 3], on the unit sphere
    faces = mesh.faces_packed().to(device_str)  # [Fs, 3], long
    return verts, faces


def gaussians_to_ellipsoid_mesh(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    level: int = 0,
    scale_mult: float = 1.0,
    opacity: torch.Tensor = None,
    opacity_thresh: float = None,
):
    """Convert a set of 3D Gaussians into a single triangle mesh of ellipsoids.

    Args:
        means:  [N, 3] Gaussian centers.
        scales: [N, 3] per-axis standard deviations (same units as ``means``).
        quats:  [N, 4] orientation quaternions in (w, x, y, z); need not be normalized.
        level:  icosphere subdivision level (0 -> 12 verts / 20 faces, 1 -> 42 / 80).
        scale_mult: ellipsoid semi-axes = ``scale_mult * scales`` (1.0 == 1 sigma).
        opacity: optional [N] or [N, 1]; together with ``opacity_thresh`` drops
                 low-opacity Gaussians before meshing.
        opacity_thresh: if set (with ``opacity``), keep only Gaussians with
                 ``opacity > opacity_thresh``.

    Returns:
        verts: [M * Vs, 3] vertex positions.
        faces: [M * Fs, 3] triangle indices (long), where M is the number of
               Gaussians kept after optional opacity filtering.
    """
    if opacity is not None and opacity_thresh is not None:
        keep = opacity.reshape(-1) > opacity_thresh
        means, scales, quats = means[keep], scales[keep], quats[keep]

    device = means.device
    v_unit, f_unit = _unit_icosphere(int(level), str(device))  # [Vs, 3], [Fs, 3]
    n_verts = v_unit.shape[0]
    n_gauss = means.shape[0]

    rot = quaternion_to_matrix(F.normalize(quats, dim=-1))             # [N, 3, 3]
    scaled = v_unit[None] * (scale_mult * scales)[:, None, :]          # [N, Vs, 3]
    verts = torch.einsum("nij,nvj->nvi", rot, scaled) + means[:, None, :]  # [N, Vs, 3]
    verts = verts.reshape(n_gauss * n_verts, 3)

    offsets = (torch.arange(n_gauss, device=device) * n_verts)[:, None, None]  # [N, 1, 1]
    faces = (f_unit[None] + offsets).reshape(-1, 3)                    # [N * Fs, 3]
    return verts, faces


@torch.no_grad()
def render_gaussian_ellipsoids(
    renderer,
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    Rt: torch.Tensor,
    K: torch.Tensor,
    image_size,
    opacity: torch.Tensor = None,
    opacity_thresh: float = None,
    level: int = 0,
    scale_mult: float = 1.0,
    scene_scale: float = 1.0,
    opencv_to_opengl: bool = True,
    colors: torch.Tensor = None,
    background_color=(1.0, 1.0, 1.0),
):
    """Build an ellipsoid mesh from a Gaussian set and render it from camera (Rt, K).

    Args:
        renderer: a ``becominglit.util.mesh.NVDiffRenderer``.
        means/scales/quats: [N, ...] for a single instance.
        Rt: [1, 4, 4] world-to-camera (same space as ``means``).
        K:  [1, 3, 3] intrinsics matching ``image_size``.
        image_size: (height, width).
        scene_scale: uniformly rescale the scene (verts + camera translation) by
            this factor before rendering. A similarity transform leaves the
            projected image unchanged but lets the geometry fall within the
            renderer's near/far clip planes (e.g. use 1e-3 for millimeter scenes,
            since the renderer's default far plane is 100).
        opencv_to_opengl: if True, convert the (OpenCV: x right, y down, z forward)
            camera to the OpenGL convention (y up, z backward) that NVDiffRenderer
            expects, by negating the camera y and z axes. Without this the geometry
            ends up behind the camera and nothing is rasterized.
        colors: optional [N, 3] color, one per Gaussian. When given, every ellipsoid
            is rendered with its own (normal-shaded) color; otherwise a uniform
            normal-shaded surface is used. Filtered together with the Gaussians.

    Returns:
        rgb: [1, H, W, 3] float in [0, 1] (normal-shaded ellipsoids).
    """
    # Filter here (rather than inside gaussians_to_ellipsoid_mesh) so per-Gaussian
    # colors stay aligned with the kept Gaussians.
    if opacity is not None and opacity_thresh is not None:
        keep = opacity.reshape(-1) > opacity_thresh
        means, scales, quats = means[keep], scales[keep], quats[keep]
        if colors is not None:
            colors = colors[keep]

    verts, faces = gaussians_to_ellipsoid_mesh(means, scales, quats, level=level, scale_mult=scale_mult)
    if opencv_to_opengl or scene_scale != 1.0:
        Rt = Rt.clone()
    if scene_scale != 1.0:
        verts = verts * scene_scale
        Rt[..., :3, 3] = Rt[..., :3, 3] * scene_scale
    if opencv_to_opengl:
        flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device=Rt.device, dtype=Rt.dtype))
        Rt = flip @ Rt

    if colors is None:
        out = renderer.render_without_texture(
            verts[None], faces, Rt, K, image_size, background_color=list(background_color)
        )
    else:
        verts_per_ellipsoid = verts.shape[0] // max(means.shape[0], 1)
        v_color = colors.repeat_interleave(verts_per_ellipsoid, dim=0)  # [M * Vs, 3]
        out = renderer.render_v_color(
            verts[None], v_color[None], faces, Rt, K, image_size, background_color=list(background_color)
        )
    return out["rgba"][..., :3].clamp(0.0, 1.0)
