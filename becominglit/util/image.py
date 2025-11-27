import torch


def linear2srgb(img: torch.Tensor, gamma: float = 2.4) -> torch.Tensor:
    linear_part = img * 12.92
    exp_part = 1.055 * (torch.max(img, torch.tensor(0.0031308)) ** (1 / gamma)) - 0.055
    return torch.where(img <= 0.0031308, linear_part, exp_part)


def srgb2linear(img: torch.Tensor, gamma: float = 2.4) -> torch.Tensor:
    linear_part = img / 12.92
    exp_part = (torch.max(img, torch.tensor(0.04045)) + 0.055) / 1.055
    return torch.where(img <= 0.04045, linear_part, exp_part**gamma)


def srgb2linear_cc(img: torch.Tensor, ccm: torch.Tensor, gamma: float = 2.4) -> torch.Tensor:
    """
    Convert sRGB image to linear RGB and apply given color correction matrix (CCM)."""
    B, C, H, W = img.shape
    img = srgb2linear(img, gamma)
    img = img.permute(0, 2, 3, 1)  # NCHW -> NHWC
    img = (img.view(B, -1, C) @ ccm.transpose(-1, -2)).view(B, H, W, C)  # Apply CCM
    return img.permute(0, 3, 1, 2)  # NHWC -> NCHW


def color_correct_srgb(img: torch.Tensor, ccm: torch.Tensor, black_level: float = 0.0) -> torch.Tensor:
    """
    Apply color correction matrix to sRGB images.

    Args:
        f: Input image tensor with shape (B, 3, H, W,).
        ccm: Color correction matrix with shape (B, 3, 3).
    """
    B, C, H, W = img.shape
    img = srgb2linear(img)
    img = torch.clamp_min(img - (black_level / 255.0), 0.0)
    img = img.permute(0, 2, 3, 1)  # NCHW -> NHWC
    img = (img.view(B, -1, C) @ ccm.transpose(-1, -2)).view(B, H, W, C)  # Apply CCM
    img = linear2srgb(img).clamp(0.0, 1.0)
    return img.permute(0, 3, 1, 2)


# ----------------------------------------------------------------------------
# Image scaling
# ----------------------------------------------------------------------------


def scale_img_hwc(x: torch.Tensor, size, mag="bilinear", min="area") -> torch.Tensor:
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]


def scale_img_nhwc(x: torch.Tensor, size, mag="bilinear", min="area") -> torch.Tensor:
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), (
        "Trying to magnify image in one dimension and minify in the other"
    )
    y = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]:  # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else:  # Magnification
        if mag in ("bilinear", "bicubic"):
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC


def scale_img_chw(x: torch.Tensor, size, mag="bilinear", min="area") -> torch.Tensor:
    return scale_img_nchw(x[None, ...], size, mag, min)[0]


def scale_img_nchw(x: torch.Tensor, size, mag="bilinear", min="area") -> torch.Tensor:
    if x.shape[1] > size[0] and x.shape[2] > size[1]:  # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(x, size, mode=min)
    else:  # Magnification
        if mag in ("bilinear", "bicubic"):
            y = torch.nn.functional.interpolate(x, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(x, size, mode=mag)
    return y


def avg_pool_nhwc(x: torch.Tensor, size) -> torch.Tensor:
    y = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    y = torch.nn.functional.avg_pool2d(y, size)
    return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC
