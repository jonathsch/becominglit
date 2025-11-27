import torch

from becominglit.util.vecmath import dot, safe_normalize


def shade_lambert_python(kd, pos, nrm, light_pos, light_intensity):
    """
    Args:
        kd: (B, N, 3)
        pos: (B, N, 3)
        nrm: (B, N, 3)
        light_pos: (B, L, 3)
        light_intensity: (B, L, 3)
    """
    wi = torch.nn.functional.normalize(light_pos[:, None, :, :] - pos[:, :, None, :], dim=-1)  # [B, N, L, 3]
    ndotl = torch.clamp((nrm[:, :, None, :] * wi).sum(dim=-1) / torch.pi, 0, 1)  # [B, L, N]
    diff = kd[:, :, None, :] * ndotl[:, :, :, None]  # [B, N, L, 3]
    # light_dist = (light_pos[:, None, :, :] - pos[:, :, None, :]).norm(dim=-1)  # [B, N, L]
    # light_intensity = light_intensity[:, None, :, :] / light_dist[:, :, :, None] ** 2  # [B, N, L, 3]
    out = diff * light_intensity[:, None, :]
    return out.sum(dim=-2)


def shade_lambert_python_deferred(kd, pos, nrm, light_pos, light_intensity):
    """
    Args:
        kd: (B, H, W, 3)
        pos: (B, H, W, 3)
        nrm: (B, H, W, 3)
        light_pos: (B, L, 3)
        light_intensity: (B, L, 3)
    """
    wi = torch.nn.functional.normalize(light_pos[:, None, None, :, :] - pos[..., None, :], dim=-1)  # [B, H, W, L, 3]
    ndotl = torch.clamp((nrm[..., None, :] * wi).sum(dim=-1) / torch.pi, 0, 1)  # [B, H, W, L]
    diff = kd[..., None, :] * ndotl[..., None]  # [B, N, L, 3]
    # light_dist = (light_pos[:, None, :, :] - pos[:, :, None, :]).norm(dim=-1)  # [B, N, L]
    # light_intensity = light_intensity[:, None, :, :] / light_dist[:, :, :, None] ** 2  # [B, N, L, 3]
    return (diff * light_intensity[:, None, None, :]).sum(dim=-2)


def shade_diffuse_polface(kd, pos, nrm, light_pos, light_intensity):
    DIFFUSE_SCALE_FACTOR = 28.0 / (23.0 * torch.pi)
    F0 = 0.04
    wi = torch.nn.functional.normalize(light_pos[:, None, :, :] - pos[:, :, None, :], dim=-1)  # [B, N, L, 3]
    ndotl = dot(nrm[:, :, None, :], wi).clamp_min(0.0)  # [B, N, L]
    fresnel = torch.pow(1.0 - torch.pow(1 - ndotl / 2.0, 5), 2)
    diffuse = DIFFUSE_SCALE_FACTOR * (1 - F0) * (fresnel * light_intensity[:, None, :, :]).sum(dim=-2)
    return kd * diffuse


################################################################################
# Frostbite diffuse
################################################################################


def bsdf_frostbite(nrm, wi, wo, linearRoughness):
    wiDotN = dot(wi, nrm)
    woDotN = dot(wo, nrm)

    h = safe_normalize(wo + wi)
    wiDotH = dot(wi, h)

    energyBias = 0.5 * linearRoughness
    energyFactor = 1.0 - (0.51 / 1.51) * linearRoughness
    f90 = energyBias + 2.0 * wiDotH * wiDotH * linearRoughness
    f0 = 1.0

    wiScatter = bsdf_fresnel_shlick(f0, f90, wiDotN)
    woScatter = bsdf_fresnel_shlick(f0, f90, woDotN)
    res = wiScatter * woScatter * energyFactor
    return torch.where((wiDotN > 0.0) & (woDotN > 0.0), res, torch.zeros_like(res))


################################################################################
# PBR's implementation of GGX specular
################################################################################

specular_epsilon = 1e-4


def _bsdf_lambert(nrm, wi):
    return torch.clamp(dot(nrm, wi), min=0.0) / torch.pi


def bsdf_fresnel_shlick(f0, f90, cosTheta):
    _cosTheta = torch.clamp(cosTheta, min=specular_epsilon, max=1.0 - specular_epsilon)
    return f0 + (f90 - f0) * (1.0 - _cosTheta) ** 5.0


def bsdf_ndf_ggx(alphaSqr, cosTheta):
    _cosTheta = torch.clamp(cosTheta, min=specular_epsilon, max=1.0 - specular_epsilon)
    d = (_cosTheta * alphaSqr - _cosTheta) * _cosTheta + 1
    return alphaSqr / (d * d * torch.pi)


def ndf_blinn_phong(alpha, cos_theta):
    return 0.5 * (alpha + 2.0) * (1.0 / torch.pi) * torch.pow(cos_theta, alpha)


def bsdf_ndf_blinn_phong_mix(blend, cos_theta):
    return blend * ndf_blinn_phong(12.0, cos_theta) + (1.0 - blend) * ndf_blinn_phong(48.0, cos_theta)


def bsdf_lambda_ggx(alphaSqr, cosTheta):
    _cosTheta = torch.clamp(cosTheta, min=specular_epsilon, max=1.0 - specular_epsilon)
    cosThetaSqr = _cosTheta * _cosTheta
    tanThetaSqr = (1.0 - cosThetaSqr) / cosThetaSqr
    res = 0.5 * (torch.sqrt(1 + alphaSqr * tanThetaSqr) - 1.0)
    return res


def bsdf_masking_smith_ggx_correlated(alphaSqr, cosThetaI, cosThetaO):
    lambdaI = bsdf_lambda_ggx(alphaSqr, cosThetaI)
    lambdaO = bsdf_lambda_ggx(alphaSqr, cosThetaO)
    return 1 / (1 + lambdaI + lambdaO)


def bsdf_pbr_specular(gain, col, nrm, wo, wi, alpha, min_roughness=0.01):
    _alpha = torch.clamp(alpha, min=min_roughness * min_roughness, max=1.0)
    alphaSqr = _alpha * _alpha

    h = safe_normalize(wo + wi)
    woDotN = dot(wo, nrm)
    wiDotN = dot(wi, nrm)
    woDotH = dot(wo, h)
    nDotH = dot(nrm, h)

    D = bsdf_ndf_ggx(alphaSqr, nDotH)
    G = bsdf_masking_smith_ggx_correlated(alphaSqr, woDotN, wiDotN)
    F = bsdf_fresnel_shlick(col, 1, woDotH)

    w = gain * F * D * G * 0.25 / torch.clamp(woDotN, min=specular_epsilon)

    frontfacing = (woDotN > specular_epsilon) & (wiDotN > specular_epsilon)
    return torch.where(frontfacing, w, torch.zeros_like(w))


def pbr_python(kd, arm, pos, nrm, view_pos, light_pos, light_intensity, spec_nrm=None):
    """
    Args:
        kd: (B, N, 3)
        ks: (B, N, 3)
        pos: (B, N, 3)
        nrm: (B, N, 3)
        view_pos: (B, 3)
        light_pos: (B, L, 3)
        light_intensity: (B, L, 3)
    """
    wi = safe_normalize(light_pos[:, None, :, :] - pos[:, :, None, :])  # [B, N, L, 3]
    wo = safe_normalize(view_pos[:, None, :] - pos)  # [B, N, 3]

    spec_str = arm[..., 0:1]  # x component
    roughness = arm[..., 1:2]  # y component
    # metallic = arm[..., 2:3]  # z component
    # ks = 0.04 * (1.0 - metallic) + kd * metallic
    # kd = kd * (1.0 - metallic)
    ks = torch.ones_like(kd)
    alpha = roughness**2

    # relative_light_intensity = (
    #     light_intensity[:, None, :, :] / (light_pos[:, None, :, :] - pos[:, :, None, :]).norm(dim=-1, keepdim=True) ** 2
    # )

    diffuse = kd * (_bsdf_lambert(nrm[:, :, None, :], wi) * light_intensity[:, None, :, :]).sum(dim=-2)  # [B, N, 3]
    spec_nrm = nrm if spec_nrm is None else spec_nrm
    specular = (
        bsdf_pbr_specular(ks[:, :, None, :], spec_nrm[:, :, None, :], wo[:, :, None, :], wi, alpha[:, :, None, :])
        * light_intensity[:, None, :, :]
    ).sum(dim=-2) * spec_str  # [B, N, 3]

    return diffuse + specular, (diffuse, specular)


def pbr_python_deferred(kd, arm, pos, nrm, view_pos, light_pos, light_intensity, spec_nrm=None):
    """
    Args:
        kd: (B, H, W, 3)
        arm: (B, H, W, 3)
        pos: (B, H, W, 3)
        nrm: (B, H, W, 3)
        view_pos: (B, 3)
        light_pos: (B, L, 3)
        light_intensity: (B, L, 3)
    """
    wi = safe_normalize(light_pos[:, None, None, :, :] - pos[:, :, :, None, :])  # [B, H, W, L, 3]
    wo = safe_normalize(view_pos[:, None, None, :] - pos)  # [B, H, W, 3]

    spec_str = arm[..., 0:1]  # x component
    roughness = arm[..., 1:2]  # y component
    metallic = arm[..., 2:3]  # z component
    spec_col = 0.04 * (1.0 - metallic) + kd * metallic
    diff_col = kd * (1.0 - metallic)
    alpha = roughness**2

    diffuse = diff_col * (
        _bsdf_lambert(nrm[..., None, :], wi).expand(-1, -1, -1, -1, 3) * light_intensity[:, None, None, ...]
    ).sum(dim=-2)  # [B, H, W, 3]
    spec_nrm = nrm if spec_nrm is None else spec_nrm
    specular = (
        bsdf_pbr_specular(spec_col[..., None, :], spec_nrm[..., None, :], wo[..., None, :], wi, alpha[..., None, :])
        * light_intensity[:, None, None, ...]
    ).sum(dim=-2) * spec_str  # [B, H, W, 3]
    return diffuse + specular, (diffuse, specular)


def polface_diffuse(kd, nrm, wi):
    return ((28 * kd) / (23 * torch.pi)) * 0.96 * (1.0 - (1.0 - dot(nrm, wi) / 2) ** 5) ** 2


def polface_specular(ks, nrm, wo, wi, roughness):
    _alpha = torch.clamp(roughness**2, min=0.08**2)
    alpha_sqr = _alpha * _alpha

    h = safe_normalize(wo + wi)
    woDotN = dot(wo, nrm)
    wiDotN = dot(wi, nrm)
    woDotH = dot(wo, h)
    nDotH = dot(nrm, h)

    D = bsdf_ndf_blinn_phong_mix(roughness, nDotH)
    G = bsdf_masking_smith_ggx_correlated(alpha_sqr, woDotN, wiDotN)
    F = bsdf_fresnel_shlick(1.0, 1, woDotH)

    w = ks * F * D * G * 0.25 / torch.clamp(woDotN, min=specular_epsilon)

    frontfacing = (woDotN > specular_epsilon) & (wiDotN > specular_epsilon)
    return torch.where(frontfacing, w, torch.zeros_like(w))


def polface_spec_ndf(roughness, wo, light_pos, light_intensity, nrm):
    """
    Args:
        roughness: (B, N, 1)
        wo: (B, 3)
        light_pos: (B, L, 3)
        nrm: (B, N, 3)
    """
    wi = safe_normalize(light_pos[:, None, :, :] - wo[:, :, None, :])  # [B, N, L, 3]
    h = safe_normalize(wo[:, :, None] + wi)
    nDotH = dot(nrm[:, :, None, :], h)  # [B, N, L]
    D = bsdf_ndf_blinn_phong_mix(roughness[:, :, None], nDotH) * light_intensity[:, None]  # [B, N, L]
    return D.sum(dim=-2)  # [B, N]


def disney_spec_ndf(roughness, wo, wi, light_intensity, nrm):
    alpha = torch.clamp(roughness, min=0.01, max=1.0) ** 2
    h = safe_normalize(wo[:, :, None] + wi)
    nDoth = dot(nrm[:, :, None, :], h)  # [B, N, L]
    D = bsdf_ndf_ggx(alpha[:, :, None] ** 2, nDoth) * light_intensity[:, None]  # [B, N, L, 3]
    return D.sum(dim=-2)  # [B, N, 3]


def polface_specular_bsdf(gain, roughness, wo, wi, light_intensity, nrm):
    # wi = safe_normalize(light_pos[:, None, :, :] - pos[:, :, None, :])  # [B, N, L, 3]
    spec = (
        polface_specular(gain[:, :, None], nrm[:, :, None, :], wo[:, :, None, :], wi, roughness[:, :, None, :])
        * light_intensity[:, None, :, :]
    )  # [B, N, L, 3]
    return spec.sum(dim=-2)  # [B, N, 3]
