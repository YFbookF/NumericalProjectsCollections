//https://github.com/mmp/pbrt-v4
Float DielectricBxDF::PDF(Vector3f wo, Vector3f wi, TransportMode mode,
                          BxDFReflTransFlags sampleFlags) const {
    if (eta == 1 || mfDistrib.EffectivelySmooth())
        return 0;
    // Evaluate sampling PDF of rough dielectric BSDF
    // Compute generalized half vector _wm_
    Float cosTheta_o = CosTheta(wo), cosTheta_i = CosTheta(wi);
    bool reflect = cosTheta_i * cosTheta_o > 0;
    float etap = 1;
    if (!reflect)
        etap = cosTheta_o > 0 ? eta : (1 / eta);
    Vector3f wm = wi * etap + wo;
    CHECK_RARE(1e-5f, LengthSquared(wm) == 0);
    if (cosTheta_i == 0 || cosTheta_o == 0 || LengthSquared(wm) == 0)
        return {};
    wm = FaceForward(Normalize(wm), Normal3f(0, 0, 1));

    // Discard backfacing microfacets
    if (Dot(wm, wi) * cosTheta_i < 0 || Dot(wm, wo) * cosTheta_o < 0)
        return {};

    // Determine Fresnel reflectance of rough dielectric boundary
    Float R = FrDielectric(Dot(wo, wm), eta);
    Float T = 1 - R;

    // Compute probabilities _pr_ and _pt_ for sampling reflection and transmission
    Float pr = R, pt = T;
    if (!(sampleFlags & BxDFReflTransFlags::Reflection))
        pr = 0;
    if (!(sampleFlags & BxDFReflTransFlags::Transmission))
        pt = 0;
    if (pr == 0 && pt == 0)
        return {};

    // Return PDF for rough dielectric
    Float pdf;
    if (reflect) {
        // Compute PDF of rough dielectric reflection
        pdf = mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm)) * pr / (pr + pt);

    } else {
        // Compute PDF of rough dielectric transmission
        Float denom = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap);
        Float dwm_dwi = AbsDot(wi, wm) / denom;
        pdf = mfDistrib.PDF(wo, wm) * dwm_dwi * pt / (pr + pt);
    }
    return pdf;
}