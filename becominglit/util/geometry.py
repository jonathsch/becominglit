from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as thf
from pytorch3d.renderer import rasterize_meshes
from pytorch3d.structures import Meshes
from torch import nn

from becominglit.util import vecmath
from becominglit.util.flame import FlameHead, FlameMask


def xyz2normals(xyz: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert XYZ image to normal image

    Args:
        xyz: torch.Tensor
        [B, 3, H, W] XYZ image

    Returns:
        torch.Tensor: [B, 3, H, W] image of normals
    """

    nrml = torch.zeros_like(xyz)
    xyz = torch.cat((xyz[:, :, :1, :] * 0, xyz[:, :, :, :], xyz[:, :, :1, :] * 0), dim=2)
    xyz = torch.cat((xyz[:, :, :, :1] * 0, xyz[:, :, :, :], xyz[:, :, :, :1] * 0), dim=3)
    U = (xyz[:, :, 2:, 1:-1] - xyz[:, :, :-2, 1:-1]) / -2
    V = (xyz[:, :, 1:-1, 2:] - xyz[:, :, 1:-1, :-2]) / -2

    nrml[:, 0, ...] = U[:, 1, ...] * V[:, 2, ...] - U[:, 2, ...] * V[:, 1, ...]
    nrml[:, 1, ...] = U[:, 2, ...] * V[:, 0, ...] - U[:, 0, ...] * V[:, 2, ...]
    nrml[:, 2, ...] = U[:, 0, ...] * V[:, 1, ...] - U[:, 1, ...] * V[:, 0, ...]
    veclen = torch.norm(nrml, dim=1, keepdim=True).clamp(min=eps)
    return nrml / veclen


def depth2xyz(depth, focal, princpt) -> torch.Tensor:
    """Convert depth image to XYZ image using camera intrinsics

    Args:
        depth: torch.Tensor
        [B, 1, H, W] depth image

        focal: torch.Tensor
        [B, 2, 2] camera focal lengths

        princpt: torch.Tensor
        [B, 2] camera principal points

    Returns:
        torch.Tensor: [B, 3, H, W] XYZ image
    """

    b, h, w = depth.shape[0], depth.shape[2], depth.shape[3]
    ix = (torch.arange(w, device=depth.device).float()[None, None, :] - princpt[:, None, None, 0]) / focal[
        :, None, None, 0, 0
    ]
    iy = (torch.arange(h, device=depth.device).float()[None, :, None] - princpt[:, None, None, 1]) / focal[
        :, None, None, 1, 1
    ]
    xyz = torch.zeros((b, 3, h, w), device=depth.device)
    xyz[:, 0, ...] = depth[:, 0, :, :] * ix
    xyz[:, 1, ...] = depth[:, 0, :, :] * iy
    xyz[:, 2, ...] = depth[:, 0, :, :]
    return xyz


# pyre-fixme[2]: Parameter must be annotated.
def depth2normals(depth, focal, princpt) -> torch.Tensor:
    """Convert depth image to normal image using camera intrinsics

    Args:
        depth: torch.Tensor
        [B, 1, H, W] depth image

        focal: torch.Tensor
        [B, 2, 2] camera focal lengths

        princpt: torch.Tensor
        [B, 2] camera principal points

    Returns:
        torch.Tensor: [B, 3, H, W] normal image
    """

    return xyz2normals(depth2xyz(depth, focal, princpt))


def make_uv_face_index(
    vt: torch.Tensor,
    vti: torch.Tensor,
    uv_shape: Union[Tuple[int, int], int],
    flip_uv: bool = True,
    device: Optional[Union[str, torch.device]] = None,
):
    """Compute a UV-space face index map identifying which mesh face contains each
    texel. For texels with no assigned triangle, the index will be -1."""

    if isinstance(uv_shape, int):
        uv_shape = (uv_shape, uv_shape)

    if device is not None:
        if isinstance(device, str):
            dev = torch.device(device)
        else:
            dev = device
        assert dev.type == "cuda"
    else:
        dev = torch.device("cuda")

    vt = 1.0 - vt.clone()

    if flip_uv:
        vt = vt.clone()
        vt[:, 1] = 1 - vt[:, 1]
    vt_pix = 2.0 * vt.to(dev) - 1.0
    vt_pix = torch.cat([vt_pix, torch.ones_like(vt_pix[:, 0:1])], dim=1)
    meshes = Meshes(vt_pix[np.newaxis], vti[np.newaxis].to(dev))
    with torch.no_grad():
        face_index, _, _, _ = rasterize_meshes(meshes, uv_shape, faces_per_pixel=1, z_clip_value=0.0, bin_size=0)
        face_index = face_index[0, ..., 0]
    return face_index


def make_uv_vert_index(
    vt: torch.Tensor,
    vi: torch.Tensor,
    vti: torch.Tensor,
    uv_shape: Union[Tuple[int, int], int],
    flip_uv: bool = True,
):
    """Compute a UV-space vertex index map identifying which mesh vertices
    comprise the triangle containing each texel. For texels with no assigned
    triangle, all indices will be -1.
    """
    face_index_map = make_uv_face_index(vt, vti, uv_shape, flip_uv).to(vi.device)
    vert_index_map = vi[face_index_map.clamp(min=0)]
    vert_index_map[face_index_map < 0] = -1
    return vert_index_map.long()


def bary_coords(points: torch.Tensor, triangles: torch.Tensor, eps: float = 1.0e-6):
    """Computes barycentric coordinates for a set of 2D query points given
    coordintes for the 3 vertices of the enclosing triangle for each point."""
    x = points[:, 0] - triangles[2, :, 0]
    x1 = triangles[0, :, 0] - triangles[2, :, 0]
    x2 = triangles[1, :, 0] - triangles[2, :, 0]
    y = points[:, 1] - triangles[2, :, 1]
    y1 = triangles[0, :, 1] - triangles[2, :, 1]
    y2 = triangles[1, :, 1] - triangles[2, :, 1]
    denom = y2 * x1 - y1 * x2
    n0 = y2 * x - x2 * y
    n1 = x1 * y - y1 * x

    # Small epsilon to prevent divide-by-zero error.
    denom = torch.where(denom >= 0, denom.clamp(min=eps), denom.clamp(max=-eps))

    bary_0 = n0 / denom
    bary_1 = n1 / denom
    bary_2 = 1.0 - bary_0 - bary_1

    return torch.stack((bary_0, bary_1, bary_2))


def make_uv_barys(
    vt: torch.Tensor,
    vti: torch.Tensor,
    uv_shape: Union[Tuple[int, int], int],
    flip_uv: bool = True,
):
    """Compute a UV-space barycentric map where each texel contains barycentric
    coordinates for that texel within its enclosing UV triangle. For texels
    with no assigned triangle, all 3 barycentric coordinates will be 0.
    """
    if isinstance(uv_shape, int):
        uv_shape = (uv_shape, uv_shape)

    if flip_uv:
        # Flip here because texture coordinates in some of our topo files are
        # stored in OpenGL convention with Y=0 on the bottom of the texture
        # unlike numpy/torch arrays/tensors.
        vt = vt.clone()
        vt[:, 1] = 1 - vt[:, 1]

    face_index_map = make_uv_face_index(vt, vti, uv_shape, flip_uv=False).to(vt.device)
    vti_map = vti.long()[face_index_map.clamp(min=0)]
    uv_tri_uvs = vt[vti_map].permute(2, 0, 1, 3)

    uv_grid = torch.meshgrid(
        torch.linspace(0.5, uv_shape[0] - 0.5, uv_shape[0]) / uv_shape[0],
        torch.linspace(0.5, uv_shape[1] - 0.5, uv_shape[1]) / uv_shape[1],
        indexing="ij",
    )
    uv_grid = torch.stack(uv_grid[::-1], dim=2).to(uv_tri_uvs)

    bary_map = bary_coords(uv_grid.view(-1, 2), uv_tri_uvs.view(3, -1, 2))
    bary_map = bary_map.permute(1, 0).view(uv_shape[0], uv_shape[1], 3)
    bary_map[face_index_map < 0] = 0
    return face_index_map, bary_map


def values_to_uv(values, index_img, bary_img):
    uv_size = index_img.shape[0]
    index_mask = torch.all(index_img != -1, dim=-1)
    idxs_flat = index_img[index_mask].to(torch.int64)
    bary_flat = bary_img[index_mask].to(torch.float32)
    # NOTE: here we assume
    values_flat = torch.sum(values[:, idxs_flat].permute(0, 3, 1, 2) * bary_flat, dim=-1).permute(0, 2, 1)
    values_uv = torch.zeros(
        values.shape[0],
        uv_size,
        uv_size,
        values.shape[-1],
        dtype=values.dtype,
        device=values.device,
    )
    values_uv[:, index_mask, :] = values_flat
    return values_uv


def face_values_to_uv(values, index_img, bary_img):
    uv_size = index_img.shape[0]
    index_mask = index_img != -1
    idxs_flat = index_img[index_mask].to(torch.int64)
    values_flat = values[:, idxs_flat]
    values_uv = torch.zeros(
        values.shape[0],
        uv_size,
        uv_size,
        values.shape[-1],
        dtype=values.dtype,
        device=values.device,
    )
    values_uv[:, index_mask, :] = values_flat
    return values_uv


def compute_vertex_normals(verts, faces):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    v_normals = torch.zeros_like(verts)
    N = verts.shape[0]
    v_normals.scatter_add_(1, i0[..., None].repeat(N, 1, 3), face_normals)
    v_normals.scatter_add_(1, i1[..., None].repeat(N, 1, 3), face_normals)
    v_normals.scatter_add_(1, i2[..., None].repeat(N, 1, 3), face_normals)

    v_normals = torch.where(
        vecmath.dot(v_normals, v_normals) > 1e-20,
        v_normals,
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device="cuda"),
    )
    v_normals = vecmath.safe_normalize(v_normals)
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_normals))
    return v_normals


def compute_vertex_tangents(verts, faces, vert_uvs, tex_idx, vertex_normals):
    r"""Compute vertex tangents.

    The vertex tangents are useful to apply normal maps during rendering.

    .. seealso::

        https://en.wikipedia.org/wiki/Normal_mapping#Calculating_tangent_space

    Args:
       faces (torch.LongTensor): unbatched triangle mesh faces, of shape
                                 :math:`(\text{num_faces}, 3)`.
       face_vertices (torch.Tensor): unbatched triangle face vertices, of shape
                                     :math:`(\text{num_faces}, 3, 3)`.
       face_uvs (torch.Tensor): unbatched triangle UVs, of shape
                                :math:`(\text{num_faces}, 3, 2)`.
       vertex_normals (torch.Tensor): unbatched vertex normals, of shape
                                      :math:`(\text{num_vertices}, 3)`.

    Returns:
       (torch.Tensor): The vertex tangents, of shape :math:`(\text{num_vertices, 3})`
    """
    # This function is strongly inspired by
    # https://github.com/NVlabs/nvdiffrec/blob/main/render/mesh.py#L203
    tangents = torch.zeros_like(vertex_normals)

    v0 = verts[..., faces[..., 0], :]
    v1 = verts[..., faces[..., 1], :]
    v2 = verts[..., faces[..., 2], :]
    tri_xyz = torch.stack([v0, v1, v2], dim=-2)  # [B, N, 3, 3]

    vt0 = vert_uvs[..., tex_idx[..., 0], :]
    vt1 = vert_uvs[..., tex_idx[..., 1], :]
    vt2 = vert_uvs[..., tex_idx[..., 2], :]
    tri_uv = torch.stack([vt0, vt1, vt2], dim=-2)  # [B, N, 3, 2]

    face_uvs0, face_uvs1, face_uvs2 = torch.split(tri_uv, 1, dim=-2)
    fv0, fv1, fv2 = torch.split(tri_xyz, 1, dim=-2)
    uve1 = face_uvs1 - face_uvs0
    uve2 = face_uvs2 - face_uvs0
    pe1 = (fv1 - fv0).squeeze(-2)
    pe2 = (fv2 - fv0).squeeze(-2)

    nom = pe1 * uve2[..., 1] - pe2 * uve1[..., 1]
    denom = uve1[..., 0] * uve2[..., 1] - uve1[..., 1] * uve2[..., 0]
    # Avoid division by zero for degenerated texture coordinates
    tang = nom / torch.where(denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6))
    vn_idx = torch.split(faces, 1, dim=-1)
    indexing_dim = 0 if tri_xyz.ndim == 3 else 1
    # TODO(cfujitsang): optimizable?
    for i in range(3):
        idx = vn_idx[i].repeat(1, 3)
        if indexing_dim == 1:
            idx = idx[None].repeat(tangents.shape[0], 1, 1)
        tangents.scatter_add_(indexing_dim, idx.long(), tang)
    # Normalize and make sure tangent is perpendicular to normal
    tangents = torch.nn.functional.normalize(tangents, dim=-1)
    tangents = torch.nn.functional.normalize(
        tangents - torch.sum(tangents * vertex_normals, dim=-1, keepdim=True) * vertex_normals, dim=-1
    )

    # bitangents = torch.cross(vertex_normals, tangents, dim=-1)
    # bitangents = bitangents / torch.norm(bitangents, dim=-1, keepdim=True).clamp(min=1e-5).clamp(min=1e-5)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(tangents))
        # assert torch.all(torch.isfinite(bitangents))

    return tangents  # , bitangents


class GeometryModule(nn.Module):
    def __init__(self, flame_config, uv_size: int):
        super().__init__()

        self.flame_model = FlameHead(**flame_config)
        # self.flame_render_mod = RenderFlameNormalsModule(self.flame_model.faces)

        flame_mask = FlameMask()
        v_neck = flame_mask.get_vid_by_region(["neck_lower", "boundary"])
        v_mask_neck = torch.ones(self.flame_model.v_template.shape[0])
        v_mask_neck[v_neck] = 0.0

        v_eyes = flame_mask.get_vid_by_region(["eyeballs", "left_eyelid", "right_eyelid"])
        v_mask_eyes = torch.ones(self.flame_model.v_template.shape[0])
        v_mask_eyes[v_eyes] = 0.0

        # Create UV index map
        index_image_vert = make_uv_vert_index(
            self.flame_model.verts_uvs,
            self.flame_model.faces,
            self.flame_model.textures_idx,
            uv_shape=uv_size,
            flip_uv=True,
        ).cpu()
        index_image_face = make_uv_face_index(
            self.flame_model.verts_uvs, self.flame_model.textures_idx, uv_shape=uv_size, flip_uv=True
        ).cpu()
        valid_mask = index_image_vert[..., :1] != -1
        face_index, bary_image = make_uv_barys(
            self.flame_model.verts_uvs, self.flame_model.textures_idx, uv_shape=uv_size, flip_uv=True
        )
        self.register_buffer(
            "uv_mask_neck", values_to_uv(v_mask_neck[None, :, None], index_image_vert, bary_image).bool()
        )
        self.register_buffer(
            "uv_mask_eyes", values_to_uv(v_mask_eyes[None, :, None], index_image_vert, bary_image).bool()
        )
        self.register_buffer("index_image", index_image_vert)
        self.register_buffer("index_image_face", index_image_face)
        self.register_buffer("valid_mask", valid_mask)
        self.register_buffer("face_index", face_index)
        self.register_buffer("bary_image", bary_image)

    def forward(self, flame_params):
        # Forward FLAME model
        # Forward without R and t
        verts, verts_cano, lmks = self.flame_model.forward(
            shape=flame_params["shape"],
            expr=flame_params["expr"],
            # rotation=flame_params["rotation"],
            rotation=torch.zeros_like(flame_params["rotation"]),
            neck=flame_params["neck_pose"],
            jaw=flame_params["jaw_pose"],
            eyes=flame_params["eyes_pose"],
            # translation=flame_params["translation"],
            translation=torch.zeros_like(flame_params["translation"]),
            zero_centered_at_root_node=True,
            # zero_centered_at_root_node=False,
            return_verts_cano=True,
            # static_offset=flame_params["static_offset"],
            static_offset=torch.zeros_like(flame_params["static_offset"]),
        )

        # meters to millimeters
        verts = verts * 1000.0
        return verts, verts_cano, lmks

    def get_tbn(self, verts):
        # Compute vertex normals, and tangents
        faces = self.flame_model.faces
        v_nrm = compute_vertex_normals(verts, faces)  # [B, V, 3]
        v_tng = compute_vertex_tangents(verts, faces, self.flame_model.verts_uvs, self.flame_model.textures_idx, v_nrm)

        # Comptute TBN frame for every texel in UV map
        normals_uv = thf.normalize(values_to_uv(v_nrm, self.index_image, self.bary_image), dim=-1)  # [B, U, V, 3]
        tangents_uv = thf.normalize(values_to_uv(v_tng, self.index_image, self.bary_image), dim=-1)  # [B, U, V, 3]
        tangents_uv = thf.normalize(
            tangents_uv - vecmath.dot(tangents_uv, normals_uv) * normals_uv, dim=-1
        )  # NOTE: Asserts that normals and tangents are orthogonal after interpolation
        bitangent_uv = torch.cross(normals_uv, tangents_uv, dim=-1)  # [B, U, V, 3]
        tbn_frame_uv = torch.stack([tangents_uv, bitangent_uv, normals_uv], dim=-1)  # [B, U, V, 3, 3]

        return tbn_frame_uv

    def render_normals(self, v, vn, K, Rt, width, height):
        return self.flame_render_mod(v, vn, K, Rt, width, height)

    def get_base_pos(self, verts):
        return values_to_uv(verts, self.index_image, self.bary_image)
