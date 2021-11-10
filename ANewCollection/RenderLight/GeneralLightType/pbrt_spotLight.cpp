// SpotLight Method Definitions
SpotLight::SpotLight(const Transform &renderFromLight,
                     const MediumInterface &mediumInterface, Spectrum Iemit, Float scale,
                     Float totalWidth, Float falloffStart)
    : LightBase(LightType::DeltaPosition, renderFromLight, mediumInterface),
      Iemit(LookupSpectrum(Iemit)),
      scale(scale),
      cosFalloffEnd(std::cos(Radians(totalWidth))),
      cosFalloffStart(std::cos(Radians(falloffStart))) {
    CHECK_LE(falloffStart, totalWidth);
}

Float SpotLight::PDF_Li(LightSampleContext, Vector3f, bool allowIncompletePDF) const {
    return 0.f;
}

SampledSpectrum SpotLight::I(Vector3f w, SampledWavelengths lambda) const {
    return SmoothStep(CosTheta(w), cosFalloffEnd, cosFalloffStart) * scale *
           Iemit->Sample(lambda);
}

SampledSpectrum SpotLight::Phi(SampledWavelengths lambda) const {
    return scale * Iemit->Sample(lambda) * 2 * Pi *
           ((1 - cosFalloffStart) + (cosFalloffStart - cosFalloffEnd) / 2);
}

pstd::optional<LightBounds> SpotLight::Bounds() const {
    Point3f p = renderFromLight(Point3f(0, 0, 0));
    Vector3f w = Normalize(renderFromLight(Vector3f(0, 0, 1)));
    Float phi = scale * Iemit->MaxValue() * 4 * Pi;
    Float cosTheta_e = std::cos(std::acos(cosFalloffEnd) - std::acos(cosFalloffStart));
    // Allow a little slop here to deal with fp round-off error in the computation of
    // cosTheta_p in the importance function.
    if (cosTheta_e == 1 && cosFalloffEnd != cosFalloffStart)
        cosTheta_e = 0.999f;
    return LightBounds(Bounds3f(p, p), w, phi, cosFalloffStart, cosTheta_e, false);
}

pstd::optional<LightLeSample> SpotLight::SampleLe(Point2f u1, Point2f u2,
                                                  SampledWavelengths &lambda,
                                                  Float time) const {
    // Choose whether to sample spotlight center cone or falloff region
    Float p[2] = {1 - cosFalloffStart, (cosFalloffStart - cosFalloffEnd) / 2};
    Float sectionPDF;
    int section = SampleDiscrete(p, u2[0], &sectionPDF);

    // Sample chosen region of spotlight cone
    Vector3f wLight;
    Float pdfDir;
    if (section == 0) {
        // Sample spotlight center cone
        wLight = SampleUniformCone(u1, cosFalloffStart);
        pdfDir = UniformConePDF(cosFalloffStart) * sectionPDF;

    } else {
        // Sample spotlight falloff region
        Float cosTheta = SampleSmoothStep(u1[0], cosFalloffEnd, cosFalloffStart);
        DCHECK(cosTheta >= cosFalloffEnd && cosTheta <= cosFalloffStart);
        Float sinTheta = SafeSqrt(1 - Sqr(cosTheta));
        Float phi = u1[1] * 2 * Pi;
        wLight = SphericalDirection(sinTheta, cosTheta, phi);
        pdfDir = SmoothStepPDF(cosTheta, cosFalloffEnd, cosFalloffStart) * sectionPDF /
                 (2 * Pi);
    }

    // Return sampled spotlight ray
    Ray ray =
        renderFromLight(Ray(Point3f(0, 0, 0), wLight, time, mediumInterface.outside));
    return LightLeSample(I(wLight, lambda), ray, 1, pdfDir);
}

void SpotLight::PDF_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    Float p[2] = {1 - cosFalloffStart, (cosFalloffStart - cosFalloffEnd) / 2};
    *pdfPos = 0;
    // Find spotlight directional PDF based on $\cos \theta$
    Float cosTheta = CosTheta(renderFromLight.ApplyInverse(ray.d));
    if (cosTheta >= cosFalloffStart)
        *pdfDir = UniformConePDF(cosFalloffStart) * p[0] / (p[0] + p[1]);
    else
        *pdfDir = SmoothStepPDF(cosTheta, cosFalloffEnd, cosFalloffStart) * p[1] /
                  ((p[0] + p[1]) * (2 * Pi));
}

std::string SpotLight::ToString() const {
    return StringPrintf(
        "[ SpotLight %s Iemit: %s cosFalloffStart: %f cosFalloffEnd: %f ]",
        BaseToString(), Iemit, cosFalloffStart, cosFalloffEnd);
}

SpotLight *SpotLight::Create(const Transform &renderFromLight, Medium medium,
                             const ParameterDictionary &parameters,
                             const RGBColorSpace *colorSpace, const FileLoc *loc,
                             Allocator alloc) {
    Spectrum I = parameters.GetOneSpectrum("I", &colorSpace->illuminant,
                                           SpectrumType::Illuminant, alloc);
    Float sc = parameters.GetOneFloat("scale", 1);

    Float coneangle = parameters.GetOneFloat("coneangle", 30.);
    Float conedelta = parameters.GetOneFloat("conedeltaangle", 5.);
    // Compute spotlight rendering to light transformation
    Point3f from = parameters.GetOnePoint3f("from", Point3f(0, 0, 0));
    Point3f to = parameters.GetOnePoint3f("to", Point3f(0, 0, 1));

    Transform dirToZ = (Transform)Frame::FromZ(Normalize(to - from));
    Transform t = Translate(Vector3f(from.x, from.y, from.z)) * Inverse(dirToZ);
    Transform finalRenderFromLight = renderFromLight * t;

    sc /= SpectrumToPhotometric(I);

    Float phi_v = parameters.GetOneFloat("power", -1);
    if (phi_v > 0) {
        Float cosFalloffEnd = std::cos(Radians(coneangle));
        Float cosFalloffStart = std::cos(Radians(coneangle - conedelta));
        Float k_e =
            2 * Pi * ((1 - cosFalloffStart) + (cosFalloffStart - cosFalloffEnd) / 2);
        sc *= phi_v / k_e;
    }

    return alloc.new_object<SpotLight>(finalRenderFromLight, medium, I, sc, coneangle,
                                       coneangle - conedelta);
}