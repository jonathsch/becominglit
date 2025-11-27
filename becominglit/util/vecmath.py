import torch


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, -1, keepdim=True)


def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2 * dot(x, n) * n - x


def length(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))  # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN


def safe_normalize(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return x / length(x, eps)


def to_hvec(x: torch.Tensor, w: float) -> torch.Tensor:
    return torch.nn.functional.pad(x, pad=(0, 1), mode="constant", value=w)


def rvec_to_R(rvec: torch.Tensor) -> torch.Tensor:
    """Computes the rotation matrix R from a tensor of Rodrigues vectors.

    n = ||rvec||
    rn = rvec/||rvec||
    N = [rn]_x = [[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]]
    R = I + sin(n)*N + (1-cos(n))*N*N
    """
    n = rvec.norm(dim=-1, p=2).clamp(min=1e-6)[..., None, None]
    rn = rvec / n[..., :, 0]
    zero = torch.zeros_like(n[..., 0, 0])
    N = torch.stack(
        (
            zero,
            -rn[..., 2],
            rn[..., 1],
            rn[..., 2],
            zero,
            -rn[..., 0],
            -rn[..., 1],
            rn[..., 0],
            zero,
        ),
        -1,
    ).view(rvec.shape[:-1] + (3, 3))
    R = (
        torch.eye(3, dtype=n.dtype, device=n.device).view([1] * (rvec.dim() - 1) + [3, 3])
        + torch.sin(n) * N
        + ((1 - torch.cos(n)) * N) @ N
    )
    return R
