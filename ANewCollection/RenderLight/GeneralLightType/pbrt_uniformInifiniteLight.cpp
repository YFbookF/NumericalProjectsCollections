// UniformInfiniteLight Method Definitions
UniformInfiniteLight::UniformInfiniteLight(const Transform &renderFromLight,
                                           Spectrum Lemit, Float scale)
    : LightBase(LightType::Infinite, renderFromLight, MediumInterface()),
      Lemit(LookupSpectrum(Lemit)),
      scale(scale) {}

SampledSpectrum UniformInfiniteLight::Le(const Ray &ray,
                                         const SampledWavelengths &lambda) const {
    return scale * Lemit->Sample(lambda);
}

pstd::optional<LightLiSample> UniformInfiniteLight::SampleLi(
    LightSampleContext ctx, Point2f u, SampledWavelengths lambda,
    bool allowIncompletePDF) const {
    if (allowIncompletePDF)
        return {};
    // Return uniform spherical sample for uniform infinite light
    Vector3f wi = SampleUniformSphere(u);
    Float pdf = UniformSpherePDF();
    return LightLiSample(scale * Lemit->Sample(lambda), wi, pdf,
                         Interaction(ctx.p() + wi * (2 * sceneRadius), &mediumInterface));
}

Float UniformInfiniteLight::PDF_Li(LightSampleContext ctx, Vector3f w,
                                   bool allowIncompletePDF) const {
    if (allowIncompletePDF)
        return 0;
    return UniformSpherePDF();
}

SampledSpectrum UniformInfiniteLight::Phi(SampledWavelengths lambda) const {
    return 4 * Pi * Pi * Sqr(sceneRadius) * scale * Lemit->Sample(lambda);
}

pstd::optional<LightLeSample> UniformInfiniteLight::SampleLe(Point2f u1, Point2f u2,
                                                             SampledWavelengths &lambda,
                                                             Float time) const {
    // Sample direction for uniform infinite light ray
    Vector3f w = SampleUniformSphere(u1);

    // Compute infinite light sample ray
    Frame wFrame = Frame::FromZ(-w);
    Point2f cd = SampleUniformDiskConcentric(u2);
    Point3f pDisk = sceneCenter + sceneRadius * wFrame.FromLocal(Vector3f(cd.x, cd.y, 0));
    Ray ray(pDisk + sceneRadius * -w, w, time);

    // Compute probabilities for uniform infinite light
    Float pdfPos = 1 / (Pi * Sqr(sceneRadius));
    Float pdfDir = UniformSpherePDF();

    return LightLeSample(scale * Lemit->Sample(lambda), ray, pdfPos, pdfDir);
}

void UniformInfiniteLight::PDF_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    *pdfDir = UniformSpherePDF();
    *pdfPos = 1 / (Pi * Sqr(sceneRadius));
}

std::string UniformInfiniteLight::ToString() const {
    return StringPrintf("[ UniformInfiniteLight %s Lemit: %s ]", BaseToString(), Lemit);
}

// ImageInfiniteLight Method Definitions
ImageInfiniteLight::ImageInfiniteLight(Transform renderFromLight, Image im,
                                       const RGBColorSpace *imageColorSpace, Float scale,
                                       std::string filename, Allocator alloc)
    : LightBase(LightType::Infinite, renderFromLight, MediumInterface()),
      image(std::move(im)),
      imageColorSpace(imageColorSpace),
      scale(scale),
      distribution(alloc),
      compensatedDistribution(alloc) {
    // ImageInfiniteLight constructor implementation
    // Initialize sampling PDFs for image infinite area light
    ImageChannelDesc channelDesc = image.GetChannelDesc({"R", "G", "B"});
    if (!channelDesc)
        ErrorExit("%s: image used for ImageInfiniteLight doesn't have R, G, B "
                  "channels.",
                  filename);
    CHECK_EQ(3, channelDesc.size());
    CHECK(channelDesc.IsIdentity());
    if (image.Resolution().x != image.Resolution().y)
        ErrorExit("%s: image resolution (%d, %d) is non-square. It's unlikely "
                  "this is an equal area environment map.",
                  filename, image.Resolution().x, image.Resolution().y);
    Array2D<Float> d = image.GetSamplingDistribution();
    Bounds2f domain = Bounds2f(Point2f(0, 0), Point2f(1, 1));
    distribution = PiecewiseConstant2D(d, domain, alloc);

    // Initialize compensated PDF for image infinite area light
    Float average = std::accumulate(d.begin(), d.end(), 0.) / d.size();
    for (Float &v : d)
        v = std::max<Float>(v - average, 0);
    if (std::all_of(d.begin(), d.end(), [](Float v) { return v == 0; }))
        std::fill(d.begin(), d.end(), Float(1));
    compensatedDistribution = PiecewiseConstant2D(d, domain, alloc);
}

Float ImageInfiniteLight::PDF_Li(LightSampleContext ctx, Vector3f w,
                                 bool allowIncompletePDF) const {
    Vector3f wLight = renderFromLight.ApplyInverse(w);
    Point2f uv = EqualAreaSphereToSquare(wLight);
    Float pdf = 0;
    if (allowIncompletePDF)
        pdf = compensatedDistribution.PDF(uv);
    else
        pdf = distribution.PDF(uv);
    return pdf / (4 * Pi);
}

SampledSpectrum ImageInfiniteLight::Phi(SampledWavelengths lambda) const {
    // We're computing fluence, then converting to power...
    SampledSpectrum sumL(0.);

    int width = image.Resolution().x, height = image.Resolution().y;
    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            RGB rgb;
            for (int c = 0; c < 3; ++c)
                rgb[c] = image.GetChannel({u, v}, c, WrapMode::OctahedralSphere);
            sumL +=
                RGBIlluminantSpectrum(*imageColorSpace, ClampZero(rgb)).Sample(lambda);
        }
    }
    // Integrating over the sphere, so 4pi for that.  Then one more for Pi
    // r^2 for the area of the disk receiving illumination...
    return 4 * Pi * Pi * Sqr(sceneRadius) * scale * sumL / (width * height);
}

pstd::optional<LightLeSample> ImageInfiniteLight::SampleLe(Point2f u1, Point2f u2,
                                                           SampledWavelengths &lambda,
                                                           Float time) const {
    // Sample infinite light image and compute ray direction _w_
    Float mapPDF;
    pstd::optional<Point2f> uv = distribution.Sample(u1, &mapPDF);
    if (!uv)
        return {};
    Vector3f wLight = EqualAreaSquareToSphere(*uv);
    Vector3f w = -renderFromLight(wLight);

    // Compute infinite light sample ray
    Frame wFrame = Frame::FromZ(-w);
    Point2f cd = SampleUniformDiskConcentric(u2);
    Point3f pDisk = sceneCenter + sceneRadius * wFrame.FromLocal(Vector3f(cd.x, cd.y, 0));
    Ray ray(pDisk + sceneRadius * -w, w, time);

    // Compute _ImageInfiniteLight_ ray PDFs
    Float pdfDir = mapPDF / (4 * Pi);
    Float pdfPos = 1 / (Pi * Sqr(sceneRadius));

    return LightLeSample(ImageLe(*uv, lambda), ray, pdfPos, pdfDir);
}

void ImageInfiniteLight::PDF_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    Vector3f wl = -renderFromLight.ApplyInverse(ray.d);
    Float mapPDF = distribution.PDF(EqualAreaSphereToSquare(wl));
    *pdfDir = mapPDF / (4 * Pi);
    *pdfPos = 1 / (Pi * Sqr(sceneRadius));
}

std::string ImageInfiniteLight::ToString() const {
    return StringPrintf("[ ImageInfiniteLight %s scale: %f ]", BaseToString(), scale);
}

// PortalImageInfiniteLight Method Definitions
PortalImageInfiniteLight::PortalImageInfiniteLight(
    const Transform &renderFromLight, Image equalAreaImage,
    const RGBColorSpace *imageColorSpace, Float scale, const std::string &filename,
    std::vector<Point3f> p, Allocator alloc)
    : LightBase(LightType::Infinite, renderFromLight, MediumInterface()),
      image(alloc),
      imageColorSpace(imageColorSpace),
      scale(scale),
      filename(filename),
      distribution(alloc) {
    ImageChannelDesc channelDesc = equalAreaImage.GetChannelDesc({"R", "G", "B"});
    if (!channelDesc)
        ErrorExit("%s: image used for PortalImageInfiniteLight doesn't have R, "
                  "G, B channels.",
                  filename);
    CHECK_EQ(3, channelDesc.size());
    CHECK(channelDesc.IsIdentity());

    if (equalAreaImage.Resolution().x != equalAreaImage.Resolution().y)
        ErrorExit("%s: image resolution (%d, %d) is non-square. It's unlikely "
                  "this is an equal area environment map.",
                  filename, equalAreaImage.Resolution().x, equalAreaImage.Resolution().y);

    if (p.size() != 4)
        ErrorExit("Expected 4 vertices for infinite light portal but given %d", p.size());
    for (int i = 0; i < 4; ++i)
        portal[i] = p[i];

    // PortalImageInfiniteLight constructor conclusion
    // Compute frame for portal coordinate system
    Vector3f p01 = Normalize(portal[1] - portal[0]);
    Vector3f p12 = Normalize(portal[2] - portal[1]);
    Vector3f p32 = Normalize(portal[2] - portal[3]);
    Vector3f p03 = Normalize(portal[3] - portal[0]);
    // Do opposite edges have the same direction?
    if (std::abs(Dot(p01, p32) - 1) > .001 || std::abs(Dot(p12, p03) - 1) > .001)
        Error("Infinite light portal isn't a planar quadrilateral");
    // Sides perpendicular?
    if (std::abs(Dot(p01, p12)) > .001 || std::abs(Dot(p12, p32)) > .001 ||
        std::abs(Dot(p32, p03)) > .001 || std::abs(Dot(p03, p01)) > .001)
        Error("Infinite light portal isn't a planar quadrilateral");
    portalFrame = Frame::FromXY(p03, p01);

    // Resample environment map into rectified image
    image = Image(PixelFormat::Float, equalAreaImage.Resolution(), {"R", "G", "B"},
                  equalAreaImage.Encoding(), alloc);
    ParallelFor(0, image.Resolution().y, [&](int y) {
        for (int x = 0; x < image.Resolution().x; ++x) {
            // Resample _equalAreaImage_ to compute rectified image pixel $(x,y)$
            // Find $(u,v)$ coordinates in equal-area image for pixel
            Point2f uv((x + 0.5f) / image.Resolution().x,
                       (y + 0.5f) / image.Resolution().y);
            Vector3f w = RenderFromImage(uv);
            w = Normalize(renderFromLight.ApplyInverse(w));
            Point2f uvEqui = EqualAreaSphereToSquare(w);

            for (int c = 0; c < 3; ++c) {
                Float v =
                    equalAreaImage.BilerpChannel(uvEqui, c, WrapMode::OctahedralSphere);
                image.SetChannel({x, y}, c, v);
            }
        }
    });

    // Initialize sampling distribution for portal image infinite light
    auto duv_dw = [&](Point2f p) {
        Float duv_dw;
        (void)RenderFromImage(p, &duv_dw);
        return duv_dw;
    };
    Array2D<Float> d = image.GetSamplingDistribution(duv_dw);
    distribution = WindowedPiecewiseConstant2D(d, alloc);
}

SampledSpectrum PortalImageInfiniteLight::Phi(SampledWavelengths lambda) const {
    // We're really computing fluence, then converting to power, for what
    // that's worth..
    SampledSpectrum sumL(0.);

    for (int y = 0; y < image.Resolution().y; ++y) {
        for (int x = 0; x < image.Resolution().x; ++x) {
            RGB rgb;
            for (int c = 0; c < 3; ++c)
                rgb[c] = image.GetChannel({x, y}, c);

            Point2f st((x + 0.5f) / image.Resolution().x,
                       (y + 0.5f) / image.Resolution().y);
            Float duv_dw;
            (void)RenderFromImage(st, &duv_dw);

            sumL +=
                RGBIlluminantSpectrum(*imageColorSpace, ClampZero(rgb)).Sample(lambda) /
                duv_dw;
        }
    }

    return scale * Area() * sumL / (image.Resolution().x * image.Resolution().y);
}

SampledSpectrum PortalImageInfiniteLight::Le(const Ray &ray,
                                             const SampledWavelengths &lambda) const {
    pstd::optional<Point2f> uv = ImageFromRender(Normalize(ray.d));
    pstd::optional<Bounds2f> b = ImageBounds(ray.o);
    if (!uv || !b || !Inside(*uv, *b))
        return SampledSpectrum(0.f);
    return ImageLookup(*uv, lambda);
}

SampledSpectrum PortalImageInfiniteLight::ImageLookup(
    Point2f uv, const SampledWavelengths &lambda) const {
    RGB rgb;
    for (int c = 0; c < 3; ++c)
        rgb[c] = image.LookupNearestChannel(uv, c);
    RGBIlluminantSpectrum spec(*imageColorSpace, ClampZero(rgb));
    return scale * spec.Sample(lambda);
}

pstd::optional<LightLiSample> PortalImageInfiniteLight::SampleLi(
    LightSampleContext ctx, Point2f u, SampledWavelengths lambda,
    bool allowIncompletePDF) const {
    // Sample $(u,v)$ in potentially visible region of light image
    pstd::optional<Bounds2f> b = ImageBounds(ctx.p());
    if (!b)
        return {};
    Float mapPDF;
    pstd::optional<Point2f> uv = distribution.Sample(u, *b, &mapPDF);
    if (!uv)
        return {};

    // Convert portal image sample point to direction and compute PDF
    Float duv_dw;
    Vector3f wi = RenderFromImage(*uv, &duv_dw);
    if (duv_dw == 0)
        return {};
    Float pdf = mapPDF / duv_dw;
    CHECK(!IsInf(pdf));

    // Compute radiance for portal light sample and return _LightLiSample_
    SampledSpectrum L = ImageLookup(*uv, lambda);
    Point3f pl = ctx.p() + 2 * sceneRadius * wi;
    return LightLiSample(L, wi, pdf, Interaction(pl, &mediumInterface));
}

Float PortalImageInfiniteLight::PDF_Li(LightSampleContext ctx, Vector3f w,
                                       bool allowIncompletePDF) const {
    // Find image $(u,v)$ coordinates corresponding to direction _w_
    Float duv_dw;
    pstd::optional<Point2f> uv = ImageFromRender(w, &duv_dw);
    if (!uv || duv_dw == 0)
        return 0;

    // Return PDF for sampling $(u,v)$ from reference point
    pstd::optional<Bounds2f> b = ImageBounds(ctx.p());
    if (!b)
        return {};
    Float pdf = distribution.PDF(*uv, *b);
    return pdf / duv_dw;
}

pstd::optional<LightLeSample> PortalImageInfiniteLight::SampleLe(
    Point2f u1, Point2f u2, SampledWavelengths &lambda, Float time) const {
    Float mapPDF;
    Bounds2f b(Point2f(0, 0), Point2f(1, 1));
    pstd::optional<Point2f> uv = distribution.Sample(u1, b, &mapPDF);
    if (!uv)
        return {};

    // Convert infinite light sample point to direction
    // Note: ignore WorldToLight since we already folded it in when we
    // resampled...
    Float duv_dw;
    Vector3f w = -RenderFromImage(*uv, &duv_dw);
    if (duv_dw == 0)
        return {};

    // Compute PDF for sampled infinite light direction
    Float pdfDir = mapPDF / duv_dw;

#if 0
    // Just sample within the portal.
    // This works with the light path integrator, but not BDPT :-(
    Point3f p = portal[0] + u2[0] * (portal[1] - portal[0]) +
        u2[1] * (portal[3] - portal[0]);
    // Compute _PortalImageInfiniteLight_ ray PDFs
    Ray ray(p, w, time);

    // Cosine to account for projected area of portal w.r.t. ray direction.
    Normal3f n = Normal3f(portalFrame.z);
    Float pdfPos = 1 / (Area() * AbsDot(n, w));
#else
    // Compute infinite light sample ray
    Frame wFrame = Frame::FromZ(-w);
    Point2f cd = SampleUniformDiskConcentric(u2);
    Point3f pDisk = sceneCenter + sceneRadius * wFrame.FromLocal(Vector3f(cd.x, cd.y, 0));
    Ray ray(pDisk + sceneRadius * -w, w, time);

    Float pdfPos = 1 / (Pi * Sqr(sceneRadius));
#endif

    SampledSpectrum L = ImageLookup(*uv, lambda);

    return LightLeSample(L, ray, pdfPos, pdfDir);
}

void PortalImageInfiniteLight::PDF_Le(const Ray &ray, Float *pdfPos,
                                      Float *pdfDir) const {
    // TODO: negate here or???
    Vector3f w = -Normalize(ray.d);
    Float duv_dw;
    pstd::optional<Point2f> uv = ImageFromRender(w, &duv_dw);

    if (!uv || duv_dw == 0) {
        *pdfPos = *pdfDir = 0;
        return;
    }

    Bounds2f b(Point2f(0, 0), Point2f(1, 1));
    Float pdf = distribution.PDF(*uv, b);

#if 0
    Normal3f n = Normal3f(portalFrame.z);
    *pdfPos = 1 / (Area() * AbsDot(n, w));
#else
    *pdfPos = 1 / (Pi * Sqr(sceneRadius));
#endif

    *pdfDir = pdf / duv_dw;
}

std::string PortalImageInfiniteLight::ToString() const {
    return StringPrintf("[ PortalImageInfiniteLight %s filename:%s scale: %f portal: %s "
                        " portalFrame: %s ]",
                        BaseToString(), filename, scale, portal, portalFrame);
}