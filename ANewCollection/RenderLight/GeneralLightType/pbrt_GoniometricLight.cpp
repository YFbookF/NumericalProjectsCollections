//pbrt
// GoniometricLight Method Definitions
GoniometricLight::GoniometricLight(const Transform &renderFromLight,
                                   const MediumInterface &mediumInterface, Spectrum Iemit,
                                   Float scale, Image im, Allocator alloc)
    : LightBase(LightType::DeltaPosition, renderFromLight, mediumInterface),
      Iemit(LookupSpectrum(Iemit)),
      scale(scale),
      image(std::move(im)),
      distrib(alloc) {
    CHECK_EQ(1, image.NChannels());
    CHECK_EQ(image.Resolution().x, image.Resolution().y);
    // Compute sampling distribution for _GoniometricLight_
    Array2D<Float> d = image.GetSamplingDistribution();
    distrib = PiecewiseConstant2D(d);

    imageBytes += image.BytesUsed() + distrib.BytesUsed();
}

pstd::optional<LightLiSample> GoniometricLight::SampleLi(LightSampleContext ctx,
                                                         Point2f u,
                                                         SampledWavelengths lambda,
                                                         bool allowIncompletePDF) const {
    Point3f p = renderFromLight(Point3f(0, 0, 0));
    Vector3f wi = Normalize(p - ctx.p());
    SampledSpectrum L =
        I(renderFromLight.ApplyInverse(-wi), lambda) / DistanceSquared(p, ctx.p());
    return LightLiSample(L, wi, 1, Interaction(p, &mediumInterface));
}

Float GoniometricLight::PDF_Li(LightSampleContext, Vector3f,
                               bool allowIncompletePDF) const {
    return 0.f;
}

SampledSpectrum GoniometricLight::Phi(SampledWavelengths lambda) const {
    Float sumY = 0;
    for (int y = 0; y < image.Resolution().y; ++y)
        for (int x = 0; x < image.Resolution().x; ++x)
            sumY += image.GetChannel({x, y}, 0);
    return scale * Iemit->Sample(lambda) * 4 * Pi * sumY /
           (image.Resolution().x * image.Resolution().y);
}

pstd::optional<LightBounds> GoniometricLight::Bounds() const {
    Float sumY = 0;
    for (int y = 0; y < image.Resolution().y; ++y)
        for (int x = 0; x < image.Resolution().x; ++x)
            sumY += image.GetChannel({x, y}, 0);
    Float phi = scale * Iemit->MaxValue() * 4 * Pi * sumY /
                (image.Resolution().x * image.Resolution().y);

    Point3f p = renderFromLight(Point3f(0, 0, 0));
    // Bound it as an isotropic point light.
    return LightBounds(Bounds3f(p, p), Vector3f(0, 0, 1), phi, std::cos(Pi),
                       std::cos(Pi / 2), false);
}

pstd::optional<LightLeSample> GoniometricLight::SampleLe(Point2f u1, Point2f u2,
                                                         SampledWavelengths &lambda,
                                                         Float time) const {
    // Sample direction and PDF for ray leaving goniometric light
    Float pdf;
    Point2f uv = distrib.Sample(u1, &pdf);
    Vector3f wLight = EqualAreaSquareToSphere(uv);
    Float pdfDir = pdf / (4 * Pi);

    Ray ray =
        renderFromLight(Ray(Point3f(0, 0, 0), wLight, time, mediumInterface.outside));
    return LightLeSample(I(wLight, lambda), ray, 1, pdfDir);
}

void GoniometricLight::PDF_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    *pdfPos = 0.f;
    Vector3f wLight = Normalize(renderFromLight.ApplyInverse(ray.d));
    Point2f uv = EqualAreaSphereToSquare(wLight);
    *pdfDir = distrib.PDF(uv) / (4 * Pi);
}

std::string GoniometricLight::ToString() const {
    return StringPrintf("[ GoniometricLight %s Iemit: %s scale: %f ]", BaseToString(),
                        Iemit, scale);
}

GoniometricLight *GoniometricLight::Create(const Transform &renderFromLight,
                                           Medium medium,
                                           const ParameterDictionary &parameters,
                                           const RGBColorSpace *colorSpace,
                                           const FileLoc *loc, Allocator alloc) {
    Spectrum I = parameters.GetOneSpectrum("I", &colorSpace->illuminant,
                                           SpectrumType::Illuminant, alloc);
    Float sc = parameters.GetOneFloat("scale", 1);

    Image image(alloc);

    std::string texname = ResolveFilename(parameters.GetOneString("filename", ""));
    if (texname.empty())
        Warning(loc, "No \"filename\" parameter provided for goniometric light.");
    else {
        ImageAndMetadata imageAndMetadata = Image::Read(texname, alloc);

        if (imageAndMetadata.image.HasAnyInfinitePixels())
            ErrorExit(
                loc,
                "%s: image has infinite pixel values and so is not suitable as a light.",
                texname);
        if (imageAndMetadata.image.HasAnyNaNPixels())
            ErrorExit(loc,
                      "%s: image has not-a-number pixel values and so is not suitable as "
                      "a light.",
                      texname);

        if (imageAndMetadata.image.Resolution().x !=
            imageAndMetadata.image.Resolution().y)
            ErrorExit("%s: image resolution (%d, %d) is non-square. It's unlikely "
                      "this is an equal-area environment map.",
                      texname, imageAndMetadata.image.Resolution().x,
                      imageAndMetadata.image.Resolution().y);

        ImageChannelDesc rgbDesc = imageAndMetadata.image.GetChannelDesc({"R", "G", "B"});
        ImageChannelDesc yDesc = imageAndMetadata.image.GetChannelDesc({"Y"});

        if (rgbDesc) {
            if (yDesc)
                ErrorExit("%s: has both \"R\", \"G\", and \"B\" or \"Y\" "
                          "channels.",
                          texname);
            image = Image(imageAndMetadata.image.Format(),
                          imageAndMetadata.image.Resolution(), {"Y"},
                          imageAndMetadata.image.Encoding(), alloc);
            for (int y = 0; y < image.Resolution().y; ++y)
                for (int x = 0; x < image.Resolution().x; ++x)
                    image.SetChannel(
                        {x, y}, 0,
                        imageAndMetadata.image.GetChannels({x, y}, rgbDesc).Average());
        } else if (yDesc)
            image = imageAndMetadata.image;
        else
            ErrorExit(loc,
                      "%s: has neither \"R\", \"G\", and \"B\" or \"Y\" "
                      "channels.",
                      texname);
    }

    sc /= SpectrumToPhotometric(I);

    Float phi_v = parameters.GetOneFloat("power", -1);
    if (phi_v > 0) {
        Float sumY = 0;
        for (int y = 0; y < image.Resolution().y; ++y)
            for (int x = 0; x < image.Resolution().x; ++x)
                sumY += image.GetChannel({x, y}, 0);
        Float k_e = 4 * Pi * sumY / (image.Resolution().x * image.Resolution().y);
        sc *= phi_v / k_e;
    }

    const Float swapYZ[4][4] = {1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1};
    Transform t(swapYZ);
    Transform finalRenderFromLight = renderFromLight * t;

    return alloc.new_object<GoniometricLight>(finalRenderFromLight, medium, I, sc,
                                              std::move(image), alloc);
}