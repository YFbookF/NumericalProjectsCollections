//pbrt
// DistantLight Method Definitions
SampledSpectrum DistantLight::Phi(SampledWavelengths lambda) const {
    return scale * Lemit->Sample(lambda) * Pi * Sqr(sceneRadius);
}

pstd::optional<LightLeSample> DistantLight::SampleLe(Point2f u1, Point2f u2,
                                                     SampledWavelengths &lambda,
                                                     Float time) const {
    // Choose point on disk oriented toward infinite light direction
    Vector3f w = Normalize(renderFromLight(Vector3f(0, 0, 1)));
    Frame wFrame = Frame::FromZ(w);
    Point2f cd = SampleUniformDiskConcentric(u1);
    Point3f pDisk = sceneCenter + sceneRadius * wFrame.FromLocal(Vector3f(cd.x, cd.y, 0));

    // Compute _DistantLight_ light ray
    Ray ray(pDisk + sceneRadius * w, -w, time);

    return LightLeSample(scale * Lemit->Sample(lambda), ray, 1 / (Pi * Sqr(sceneRadius)),
                         1);
}

void DistantLight::PDF_Le(const Ray &, Float *pdfPos, Float *pdfDir) const {
    *pdfPos = 1 / (Pi * sceneRadius * sceneRadius);
    *pdfDir = 0;
}

std::string DistantLight::ToString() const {
    return StringPrintf("[ DistantLight %s Lemit: %s scale: %f ]", BaseToString(), Lemit,
                        scale);
}

DistantLight *DistantLight::Create(const Transform &renderFromLight,
                                   const ParameterDictionary &parameters,
                                   const RGBColorSpace *colorSpace, const FileLoc *loc,
                                   Allocator alloc) {
    Spectrum L = parameters.GetOneSpectrum("L", &colorSpace->illuminant,
                                           SpectrumType::Illuminant, alloc);
    Float sc = parameters.GetOneFloat("scale", 1);

    Point3f from = parameters.GetOnePoint3f("from", Point3f(0, 0, 0));
    Point3f to = parameters.GetOnePoint3f("to", Point3f(0, 0, 1));

    Vector3f w = Normalize(from - to);
    Vector3f v1, v2;
    CoordinateSystem(w, &v1, &v2);
    Float m[4][4] = {v1.x, v2.x, w.x, 0, v1.y, v2.y, w.y, 0,
                     v1.z, v2.z, w.z, 0, 0,    0,    0,   1};
    Transform t(m);
    Transform finalRenderFromLight = renderFromLight * t;

    // Scale the light spectrum to be equivalent to 1 nit
    sc /= SpectrumToPhotometric(L);

    // Adjust scale to meet target illuminance value
    // Like for IBLs we measure illuminance as incident on an upward-facing
    // patch.
    Float E_v = parameters.GetOneFloat("illuminance", -1);
    if (E_v > 0)
        sc *= E_v;

    return alloc.new_object<DistantLight>(finalRenderFromLight, L, sc);
}