// PointLight Method Definitions
// pbrt
SampledSpectrum PointLight::Phi(SampledWavelengths lambda) const {
    return 4 * Pi * scale * I->Sample(lambda);
}

pstd::optional<LightBounds> PointLight::Bounds() const {
    Point3f p = renderFromLight(Point3f(0, 0, 0));
    Float phi = 4 * Pi * scale * I->MaxValue();
    return LightBounds(Bounds3f(p, p), Vector3f(0, 0, 1), phi, std::cos(Pi),
                       std::cos(Pi / 2), false);
}

pstd::optional<LightLeSample> PointLight::SampleLe(Point2f u1, Point2f u2,
                                                   SampledWavelengths &lambda,
                                                   Float time) const {
    Point3f p = renderFromLight(Point3f(0, 0, 0));
    Ray ray(p, SampleUniformSphere(u1), time, mediumInterface.outside);
    return LightLeSample(scale * I->Sample(lambda), ray, 1, UniformSpherePDF());
}

void PointLight::PDF_Le(const Ray &, Float *pdfPos, Float *pdfDir) const {
    *pdfPos = 0;
    *pdfDir = UniformSpherePDF();
}

std::string PointLight::ToString() const {
    return StringPrintf("[ PointLight %s I: %s scale: %f ]", BaseToString(), I, scale);
}

PointLight *PointLight::Create(const Transform &renderFromLight, Medium medium,
                               const ParameterDictionary &parameters,
                               const RGBColorSpace *colorSpace, const FileLoc *loc,
                               Allocator alloc) {
    Spectrum I = parameters.GetOneSpectrum("I", &colorSpace->illuminant,
                                           SpectrumType::Illuminant, alloc);
    Float sc = parameters.GetOneFloat("scale", 1);

    sc /= SpectrumToPhotometric(I);

    Float phi_v = parameters.GetOneFloat("power", -1);
    if (phi_v > 0) {
        Float k_e = 4 * Pi;
        sc *= phi_v / k_e;
    }

    Point3f from = parameters.GetOnePoint3f("from", Point3f(0, 0, 0));
    Transform tf = Translate(Vector3f(from.x, from.y, from.z));
    Transform finalRenderFromLight(renderFromLight * tf);

    return alloc.new_object<PointLight>(finalRenderFromLight, medium, I, sc);
}
