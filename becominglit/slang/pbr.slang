import utils;

static const float F0 = 0.04f;
static const float DIFFUSE_SCALE_FACTOR = 28.f / (23.f * PI);
static const float SPECULAR_EPSILON = 1e-4f;
static const float MIN_ROUGHNESS = 0.1f;
static const float MIN_ALPHA = MIN_ROUGHNESS * MIN_ROUGHNESS;
static const float eps = 1e-5f;

[BackwardDifferentiable]
float fresnelPolface(const float f0, const float cosTheta)
{
    return f0 + (1 - f0) * pow(1 - cosTheta, 5);
}

[BackwardDifferentiable]
float ndfBlinnPhong(const float alpha, const float cosTheta)
{
    let power = pow(cosTheta, alpha);
    return 0.5 * (alpha + 2.f) * INV_PI * power;
}

[BackwardDifferentiable]
float ndfBlinnPhongMix(const float blend, const float cosTheta)
{
    return blend * ndfBlinnPhong(12.f, cosTheta) + (1.f - blend) * ndfBlinnPhong(48.f, cosTheta);
}

[BackwardDifferentiable]
float lambdaBlinnPhong(const float alpha, const float cosTheta)
{
    var res = 0.0f;

    let cosThetaSqr = cosTheta * cosTheta;
    let tanThetaSqr = (1.f - cosThetaSqr) / cosThetaSqr;
    let tanTheta = sqrt(tanThetaSqr);
    let a = sqrt(0.5 * alpha + 1) / (tanTheta + eps);

    if (a >= 1.6f)
    {
        let aSqr = a * a;
        res = (1.0f - 1.259f * a + 0.396f * aSqr) / (3.535f * a + 2.181f * aSqr);
    }
    return res;
}

[BackwardDifferentiable]
float maskingBlinnPhongCorrelated(const float alpha, const float cosThetaI, const float cosThetaO)
{
    let lambdaI = lambdaBlinnPhong(alpha, cosThetaI);
    let lambdaO = lambdaBlinnPhong(alpha, cosThetaO);
    return 1.0f / (1.0f + lambdaI + lambdaO);
}

[BackwardDifferentiable]
float3 specularBSDF(const float3 albedo,
                    const float roughnessBlend,
                    const float specVis,
                    const float3 pos,
                    const float3 nrm,
                    no_diff const float3 light_intensity,
                    no_diff const float3 view_pos,
                    no_diff const float3 light_pos)
{
    float3 wi = safeNormalize(light_pos - pos);
    float3 wo = safeNormalize(view_pos - pos);

    let woDotN = dot(wo, nrm);
    let wiDotN = dot(wi, nrm);
    bool frontfacing = (woDotN > SPECULAR_EPSILON) & (wiDotN > SPECULAR_EPSILON);

    float refl = 0.0f;
    if (frontfacing)
    {
        float3 h = safeNormalize(wi + wo);
        float woDotH = dot(wo, h);
        float nDotH = dot(nrm, h);
        float D = ndfBlinnPhongMix(roughnessBlend, nDotH);
        float G = maskingBlinnPhongCorrelated(12.0f, woDotN, wiDotN);
        let F = fresnelPolface(F0, woDotN);
        refl = F * D * G * 0.25 / (woDotN + eps);
    }

    return specVis * refl * light_intensity;
}

///////////////////////////////////////////////////////////////////////////////
// CUDA kernels
///////////////////////////////////////////////////////////////////////////////

[CudaKernel]
void specular_bsdf_fwd_kernel(TensorView<float3> albedo,          // B x N x 3
                              TensorView<float> roughness,        // B x N, x 1
                              TensorView<float> specVis,          // B x N x 1
                              TensorView<float3> pos,             // B x N x 3
                              TensorView<float3> nrm,             // B x N x 3
                              TensorView<float3> view_pos,        // B x 3
                              TensorView<float3> light_pos,       // B x L x 3
                              TensorView<float3> light_intensity, // B x L x 1
                              TensorView<int> numLights,          // B x 1
                              TensorView<float3> out)
{
    let idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (idx.x > out.size(1) || idx.y > out.size(0))
    {
        return;
    }

    let idxr = uint2(idx.y, idx.x);
    float3 v_pos = view_pos[idx.y];
    int nLights = numLights[idx.y];

    float3 spec_acum = float3(0.0f);
    for (uint i = 0; i < nLights; i++)
    {
        float3 l_pos = light_pos[idx.y, i];
        float3 l_int = light_intensity[idx.y, i];
        float3 specular = specularBSDF(albedo[idxr], roughness[idxr], specVis[idxr], pos[idxr], nrm[idxr], l_int, v_pos, l_pos);
        spec_acum += select(isfinite(specular), specular, float3(0.0f));
    }

    out[idxr] = spec_acum;
}

[CudaKernel]
void polface_bn_bwd_kernel(TensorView<float3> albedo,   // B x N x 3
                           TensorView<float> roughness, // B x N, x 1
                           TensorView<float> specVis,   // B x N x 1
                           TensorView<float3> pos,
                           TensorView<float3> nrm,
                           TensorView<float3> view_pos,
                           TensorView<float3> light_pos,
                           TensorView<float3> light_intensity,
                           TensorView<int> numLights,
                           TensorView<float3> albedo_grad,
                           TensorView<float> roughness_grad,
                           TensorView<float> specVis_grad,
                           TensorView<float3> pos_grad,
                           TensorView<float3> nrm_grad,
                           TensorView<float3> grad_out)
{
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (idx.x > grad_out.size(1) || idx.y > grad_out.size(0))
    {
        return;
    }

    uint2 idxr = uint2(idx.y, idx.x);

    float3 v_pos = view_pos[idx.y];

    let n_lights = numLights[idx.y];
    float3 albedo_accum = float3(0.0f);
    float roughness_accum = 0.0f;
    float specVis_accum = 0.0f;
    float3 pos_accum = float3(0.0f);
    float3 nrm_accum = float3(0.0f);

    for (int i = 0; i < n_lights; ++i)
    {
        let l_pos = light_pos[idx.y, i];
        let l_int = light_intensity[idx.y, i];
        if (!(length(l_int) > 0.0f))
        {
            continue;
        }

        var dp_albedo = diffPair(albedo[idxr]);
        var dp_roughness = diffPair(roughness[idxr]);
        var dp_specVis = diffPair(specVis[idxr]);
        var dp_pos = diffPair(pos[idxr]);
        var dp_nrm = diffPair(nrm[idxr]);

        __bwd_diff(specularBSDF)(dp_albedo, dp_roughness, dp_specVis, dp_pos, dp_nrm, l_int, v_pos, l_pos, grad_out[idxr]);

        albedo_accum += dp_albedo.d;
        roughness_accum += dp_roughness.d;
        specVis_accum += dp_specVis.d;
        pos_accum += dp_pos.d;
        nrm_accum += dp_nrm.d;
    }

    albedo_grad[idxr] = select(isfinite(albedo_accum), albedo_accum, float3(0.0f));
    roughness_grad[idxr] = select(isfinite(roughness_accum), roughness_accum, 0.0f);
    specVis_grad[idxr] = select(isfinite(specVis_accum), specVis_accum, 0.0f);
    pos_grad[idxr] = select(isfinite(pos_accum), pos_accum, float3(0.0f));
    nrm_grad[idxr] = select(isfinite(nrm_accum), nrm_accum, float3(0.0f));
}

///////////////////////////////////////////////////////////////////////////////
// Torch entry point
///////////////////////////////////////////////////////////////////////////////

[TorchEntryPoint]
TorchTensor<float3> specular_fwd(TorchTensor<float3> albedo,
                                TorchTensor<float> roughness,
                                TorchTensor<float> specVis,
                                TorchTensor<float3> pos,
                                TorchTensor<float3> nrm,
                                TorchTensor<float3> view_pos,
                                TorchTensor<float3> light_pos,
                                TorchTensor<float3> light_intensity,
                                TorchTensor<int> numLights,
                                const int dims)
{
    TorchTensor<float3> result = TorchTensor<float3>.emptyLike(albedo);

    uint3 dims = uint3(albedo.size(1), albedo.size(0), 1);
    let blockSize = uint3(64, 1, 1);
    let blockCount = getLaunchGridSize(blockSize, dims);
    __dispatch_kernel(specular_bsdf_fwd_kernel, blockCount, blockSize)(albedo, roughness, specVis, pos, nrm, view_pos, light_pos, light_intensity, numLights, result);

    return result;
}

struct PbrBwdReturnType
{
    TorchTensor<float3> albedo_grad;
    TorchTensor<float> roughness_grad;
    TorchTensor<float> specVis_grad;
    TorchTensor<float3> pos_grad;
    TorchTensor<float3> nrm_grad;
}

[TorchEntryPoint]
PbrBwdReturnType specular_bwd(TorchTensor<float3> albedo,
                             TorchTensor<float> roughness,
                             TorchTensor<float> specVis,
                             TorchTensor<float3> pos,
                             TorchTensor<float3> nrm,
                             TorchTensor<float3> view_pos,
                             TorchTensor<float3> light_pos,
                             TorchTensor<float3> light_intensity,
                             TorchTensor<int> numLights,
                             const int dims,
                             TorchTensor<float3> grad_out)
{
    PbrBwdReturnType rt;
    rt.albedo_grad = TorchTensor<float3>.zerosLike(albedo);
    rt.roughness_grad = TorchTensor<float>.zerosLike(roughness);
    rt.specVis_grad = TorchTensor<float>.zerosLike(specVis);
    rt.pos_grad = TorchTensor<float3>.zerosLike(pos);
    rt.nrm_grad = TorchTensor<float3>.zerosLike(nrm);

    uint3 dims = uint3(grad_out.size(1), grad_out.size(0), 1);
    let blockSize = uint3(64, 1, 1);
    let blockCount = getLaunchGridSize(blockSize, dims);
    __dispatch_kernel(polface_bn_bwd_kernel, blockCount, blockSize)(albedo, roughness, specVis, pos, nrm, view_pos, light_pos, light_intensity, numLights,
                                                                    rt.albedo_grad, rt.roughness_grad, rt.specVis_grad, rt.pos_grad, rt.nrm_grad, grad_out);

    return rt;
}
