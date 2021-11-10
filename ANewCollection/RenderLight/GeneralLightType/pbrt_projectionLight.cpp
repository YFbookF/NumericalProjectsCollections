//pbrt
// ProjectionLight Method Definitions
ProjectionLight::ProjectionLight(Transform renderFromLight,
                                 MediumInterface mediumInterface, Image im,
                                 const RGBColorSpace *imageColorSpace, Float scale,
                                 Float fov, Allocator alloc)
    : LightBase(LightType::DeltaPosition, renderFromLight, mediumInterface),
      image(std::move(im)),
      imageColorSpace(imageColorSpace),
      scale(scale),
      distrib(alloc) {
    // _ProjectionLight_ constructor implementation
    // Initialize _ProjectionLight_ projection matrix
    Float aspect = Float(image.Resolution().x) / Float(image.Resolution().y);
    if (aspect > 1)
        screenBounds = Bounds2f(Point2f(-aspect, -1), Point2f(aspect, 1));
    else
        screenBounds = Bounds2f(Point2f(-1, -1 / aspect), Point2f(1, 1 / aspect));
    screenFromLight = Perspective(fov, hither, 1e30f /* yon */);
    lightFromScreen = Inverse(screenFromLight);

    // Compute projection image area _A_
    Float opposite = std::tan(Radians(fov) / 2);
    A = 4 * Sqr(opposite) * (aspect > 1 ? aspect : 1 / aspect);

    // Compute sampling distribution for _ProjectionLight_
    ImageChannelDesc channelDesc = image.GetChannelDesc({"R", "G", "B"});
    if (!channelDesc)
        ErrorExit("Image used for ProjectionLight does not have R, G, B channels.");
    CHECK_EQ(3, channelDesc.size());
    CHECK(channelDesc.IsIdentity());
    auto dwdA = [&](const Point2f &p) {
        Vector3f w = Vector3f(lightFromScreen(Point3f(p.x, p.y, 0)));
        return Pow<3>(CosTheta(Normalize(w)));
    };
    Array2D<Float> d = image.GetSamplingDistribution(dwdA, screenBounds);
    distrib = PiecewiseConstant2D(d, screenBounds);

    imageBytes += image.BytesUsed() + distrib.BytesUsed();
}

pstd::optional<LightLiSample> ProjectionLight::SampleLi(LightSampleContext ctx, Point2f u,
                                                        SampledWavelengths lambda,
                                                        bool allowIncompletePDF) const {
    // Return sample for incident radiance from _ProjectionLight_
    Point3f p = renderFromLight(Point3f(0, 0, 0));
    Vector3f wi = Normalize(p - ctx.p());
    Vector3f wl = renderFromLight.ApplyInverse(-wi);
    SampledSpectrum Li = I(wl, lambda) / DistanceSquared(p, ctx.p());
    if (!Li)
        return {};
    return LightLiSample(Li, wi, 1, Interaction(p, &mediumInterface));
}

Float ProjectionLight::PDF_Li(LightSampleContext, Vector3f,
                              bool allowIncompletePDF) const {
    return 0.f;
}

std::string ProjectionLight::ToString() const {
    return StringPrintf("[ ProjectionLight %s scale: %f A: %f ]", BaseToString(), scale,
                        A);
}

SampledSpectrum ProjectionLight::I(Vector3f w, const SampledWavelengths &lambda) const {
    // Discard directions behind projection light
    if (w.z < hither)
        return SampledSpectrum(0.f);

    // Project point onto projection plane and compute RGB
    Point3f ps = screenFromLight(Point3f(w));
    if (!Inside(Point2f(ps.x, ps.y), screenBounds))
        return SampledSpectrum(0.f);
    Point2f uv = Point2f(screenBounds.Offset(Point2f(ps.x, ps.y)));
    RGB rgb;
    for (int c = 0; c < 3; ++c)
        rgb[c] = image.LookupNearestChannel(uv, c);

    // Return scaled wavelength samples corresponding to RGB
    RGBIlluminantSpectrum s(*imageColorSpace, ClampZero(rgb));
    return scale * s.Sample(lambda);
}

SampledSpectrum ProjectionLight::Phi(SampledWavelengths lambda) const {
    SampledSpectrum sum(0.f);
    for (int y = 0; y < image.Resolution().y; ++y)
        for (int x = 0; x < image.Resolution().x; ++x) {
            // Compute change of variables factor _dwdA_ for projection light pixel
            Point2f ps = screenBounds.Lerp(Point2f((x + 0.5f) / image.Resolution().x,
                                                   (y + 0.5f) / image.Resolution().y));
            Vector3f w = Vector3f(lightFromScreen(Point3f(ps.x, ps.y, 0)));
            w = Normalize(w);
            Float dwdA = Pow<3>(CosTheta(w));

            // Update _sum_ for projection light pixel
            RGB rgb;
            for (int c = 0; c < 3; ++c)
                rgb[c] = image.GetChannel({x, y}, c);
            RGBIlluminantSpectrum s(*imageColorSpace, ClampZero(rgb));
            sum += s.Sample(lambda) * dwdA;
        }
    // Return final power for projection light
    return scale * A * sum / (image.Resolution().x * image.Resolution().y);
}

pstd::optional<LightBounds> ProjectionLight::Bounds() const {
    Float sum = 0;
    for (int v = 0; v < image.Resolution().y; ++v)
        for (int u = 0; u < image.Resolution().x; ++u)
            sum += std::max({image.GetChannel({u, v}, 0), image.GetChannel({u, v}, 1),
                             image.GetChannel({u, v}, 2)});
    Float phi = scale * sum / (image.Resolution().x * image.Resolution().y);

    Point3f pCorner(screenBounds.pMax.x, screenBounds.pMax.y, 0);
    Vector3f wCorner = Normalize(Vector3f(lightFromScreen(pCorner)));
    Float cosTotalWidth = CosTheta(wCorner);

    Point3f p = renderFromLight(Point3f(0, 0, 0));
    Vector3f w = Normalize(renderFromLight(Vector3f(0, 0, 1)));
    return LightBounds(Bounds3f(p, p), w, phi, std::cos(0.f), cosTotalWidth, false);
}

pstd::optional<LightLeSample> ProjectionLight::SampleLe(Point2f u1, Point2f u2,
                                                        SampledWavelengths &lambda,
                                                        Float time) const {
    // Sample light space ray direction for projection light
    Float pdf;
    Point2f ps = distrib.Sample(u1, &pdf);
    if (pdf == 0)
        return {};
    Vector3f w = Vector3f(lightFromScreen(Point3f(ps.x, ps.y, 0)));

    // Compute PDF for sampled projection light direction
    Float cosTheta = CosTheta(Normalize(w));
    CHECK_GT(cosTheta, 0);
    Float pdfDir = pdf * screenBounds.Area() / (A * Pow<3>(cosTheta));

    // Compute radiance and return projection light sample
    Point2f p = Point2f(screenBounds.Offset(ps));
    RGB rgb;
    for (int c = 0; c < 3; ++c)
        rgb[c] = image.LookupNearestChannel(p, c);
    SampledSpectrum L =
        scale * RGBIlluminantSpectrum(*imageColorSpace, rgb).Sample(lambda);
    Ray ray = renderFromLight(
        Ray(Point3f(0, 0, 0), Normalize(w), time, mediumInterface.outside));
    return LightLeSample(L, ray, 1, pdfDir);
}

void ProjectionLight::PDF_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    *pdfPos = 0;
    // Transform ray direction to light space and reject invalid ones
    Vector3f w = Normalize(renderFromLight.ApplyInverse(ray.d));
    if (w.z < hither) {
        *pdfDir = 0;
        return;
    }

    // Compute screen space coordinates for direction and test against bounds
    Point3f ps = screenFromLight(Point3f(w));
    if (!Inside(Point2f(ps.x, ps.y), screenBounds)) {
        *pdfDir = 0;
        return;
    }

    *pdfDir = distrib.PDF(Point2f(ps.x, ps.y)) * screenBounds.Area() /
              (A * Pow<3>(CosTheta(w)));
}

ProjectionLight *ProjectionLight::Create(const Transform &renderFromLight, Medium medium,
                                         const ParameterDictionary &parameters,
                                         const FileLoc *loc, Allocator alloc) {
    Float scale = parameters.GetOneFloat("scale", 1);
    Float power = parameters.GetOneFloat("power", -1);
    Float fov = parameters.GetOneFloat("fov", 90.);

    std::string texname = ResolveFilename(parameters.GetOneString("filename", ""));
    if (texname.empty())
        ErrorExit(loc, "Must provide \"filename\" to \"projection\" light source");

    ImageAndMetadata imageAndMetadata = Image::Read(texname, alloc);
    if (imageAndMetadata.image.HasAnyInfinitePixels())
        ErrorExit(
            loc, "%s: image has infinite pixel values and so is not suitable as a light.",
            texname);
    if (imageAndMetadata.image.HasAnyNaNPixels())
        ErrorExit(
            loc,
            "%s: image has not-a-number pixel values and so is not suitable as a light.",
            texname);

    const RGBColorSpace *colorSpace = imageAndMetadata.metadata.GetColorSpace();

    ImageChannelDesc channelDesc = imageAndMetadata.image.GetChannelDesc({"R", "G", "B"});
    if (!channelDesc)
        ErrorExit(loc, "Image provided to \"projection\" light must have R, G, "
                       "and B channels.");
    Image image = imageAndMetadata.image.SelectChannels(channelDesc, alloc);

    scale /= SpectrumToPhotometric(&colorSpace->illuminant);
    if (power > 0) {
        Bounds2f screenBounds;
        Float A;
        Transform lightFromScreen, screenFromLight;
        Float hither = 1e-3f;
        // Initialize _ProjectionLight_ projection matrix
        Float aspect = Float(image.Resolution().x) / Float(image.Resolution().y);
        if (aspect > 1)
            screenBounds = Bounds2f(Point2f(-aspect, -1), Point2f(aspect, 1));
        else
            screenBounds = Bounds2f(Point2f(-1, -1 / aspect), Point2f(1, 1 / aspect));
        screenFromLight = Perspective(fov, hither, 1e30f /* yon */);
        lightFromScreen = Inverse(screenFromLight);

        // Compute projection image area _A_
        Float opposite = std::tan(Radians(fov) / 2);
        A = 4 * Sqr(opposite) * (aspect > 1 ? aspect : 1 / aspect);

        Float sum = 0;
        RGB luminance = colorSpace->LuminanceVector();
        for (int y = 0; y < image.Resolution().y; ++y)
            for (int x = 0; x < image.Resolution().x; ++x) {
                Point2f ps = screenBounds.Lerp(
                    {(x + .5f) / image.Resolution().x, (y + .5f) / image.Resolution().y});
                Vector3f w = Vector3f(lightFromScreen(Point3f(ps.x, ps.y, 0)));
                w = Normalize(w);
                Float dwdA = Pow<3>(w.z);

                for (int c = 0; c < 3; ++c)
                    sum += image.GetChannel({x, y}, c) * luminance[c] * dwdA;
            }
        scale *= power / (A * sum / (image.Resolution().x * image.Resolution().y));
    }

    Transform flip = Scale(1, -1, 1);
    Transform renderFromLightFlipY = renderFromLight * flip;

    return alloc.new_object<ProjectionLight>(
        renderFromLightFlipY, medium, std::move(image), colorSpace, scale, fov, alloc);
}