// DiffuseAreaLight Method Definitions
DiffuseAreaLight::DiffuseAreaLight(const Transform &renderFromLight,
                                   const MediumInterface &mediumInterface, Spectrum Le,
                                   Float scale, const Shape shape, FloatTexture alpha,
                                   Image im, const RGBColorSpace *imageColorSpace,
                                   bool twoSided)
    : LightBase(
          [](FloatTexture alpha) {
              // Special case handling for area lights with constant zero-valued alpha
              // textures to allow invisible area lights: we will null out the alpha
              // texture below so that as far as the DiffuseAreaLight is concerned, there
              // is no alpha texture and the light is fully emissive. However, such lights
              // will never be intersected by rays (because their associated primitives
              // still have the alpha texture), so we mark them as DeltaPosition lights
              // here so that MIS isn't used for direct illumination. Thus, light sampling
              // is the only strategy used and we get an unbiased (if potentially high
              // variance) estimate.

              const FloatConstantTexture *fc =
                  alpha.CastOrNullptr<FloatConstantTexture>();
              if (fc && fc->Evaluate(TextureEvalContext()) == 0)
                  return LightType::DeltaPosition;
              return LightType::Area;
          }(alpha),
          renderFromLight, mediumInterface),
      shape(shape),
      alpha(type == LightType::Area ? alpha : nullptr),
      area(shape.Area()),
      twoSided(twoSided),
      Lemit(LookupSpectrum(Le)),
      scale(scale),
      image(std::move(im)),
      imageColorSpace(imageColorSpace) {
    ++numAreaLights;

    if (image) {
        ImageChannelDesc desc = image.GetChannelDesc({"R", "G", "B"});
        if (!desc)
            ErrorExit("Image used for DiffuseAreaLight doesn't have R, G, B "
                      "channels.");
        CHECK_EQ(3, desc.size());
        CHECK(desc.IsIdentity());
        CHECK(imageColorSpace);
    } else {
        CHECK(Le);
    }

    // Warn if light has transformation with non-uniform scale, though not
    // for Triangles or bilinear patches, since this doesn't matter for them.
    if (renderFromLight.HasScale() && !shape.Is<Triangle>() && !shape.Is<BilinearPatch>())
        Warning("Scaling detected in rendering to light space transformation! "
                "The system has numerous assumptions, implicit and explicit, "
                "that this transform will have no scale factors in it. "
                "Proceed at your own risk; your image may have errors.");
}

pstd::optional<LightLiSample> DiffuseAreaLight::SampleLi(LightSampleContext ctx,
                                                         Point2f u,
                                                         SampledWavelengths lambda,
                                                         bool allowIncompletePDF) const {
    // Sample point on shape for _DiffuseAreaLight_
    ShapeSampleContext shapeCtx(ctx.pi, ctx.n, ctx.ns, 0 /* time */);
    pstd::optional<ShapeSample> ss = shape.Sample(shapeCtx, u);
    if (!ss || ss->pdf == 0 || LengthSquared(ss->intr.p() - ctx.p()) == 0)
        return {};
    DCHECK(!IsNaN(ss->pdf));
    ss->intr.mediumInterface = &mediumInterface;

    // Check sampled point on shape against alpha texture, if present
    if (AlphaMasked(ss->intr))
        return {};

    // Return _LightLiSample_ for sampled point on shape
    Vector3f wi = Normalize(ss->intr.p() - ctx.p());
    SampledSpectrum Le = L(ss->intr.p(), ss->intr.n, ss->intr.uv, -wi, lambda);
    if (!Le)
        return {};
    return LightLiSample(Le, wi, ss->pdf, ss->intr);
}

Float DiffuseAreaLight::PDF_Li(LightSampleContext ctx, Vector3f wi,
                               bool allowIncompletePDF) const {
    ShapeSampleContext shapeCtx(ctx.pi, ctx.n, ctx.ns, 0 /* time */);
    return shape.PDF(shapeCtx, wi);
}

SampledSpectrum DiffuseAreaLight::Phi(SampledWavelengths lambda) const {
    SampledSpectrum L(0.f);
    if (image) {
        // Compute average light image emission
        for (int y = 0; y < image.Resolution().y; ++y)
            for (int x = 0; x < image.Resolution().x; ++x) {
                RGB rgb;
                for (int c = 0; c < 3; ++c)
                    rgb[c] = image.GetChannel({x, y}, c);
                L += RGBIlluminantSpectrum(*imageColorSpace, ClampZero(rgb))
                         .Sample(lambda);
            }
        L *= scale / (image.Resolution().x * image.Resolution().y);

    } else
        L = Lemit->Sample(lambda) * scale;
    return Pi * (twoSided ? 2 : 1) * area * L;
}

pstd::optional<LightBounds> DiffuseAreaLight::Bounds() const {
    // Compute _phi_ for diffuse area light bounds
    Float phi = 0;
    if (image) {
        // Compute average _DiffuseAreaLight_ image channel value
        // Assume no distortion in the mapping, FWIW...
        for (int y = 0; y < image.Resolution().y; ++y)
            for (int x = 0; x < image.Resolution().x; ++x)
                for (int c = 0; c < 3; ++c)
                    phi += image.GetChannel({x, y}, c);
        phi /= 3 * image.Resolution().x * image.Resolution().y;

    } else
        phi = Lemit->MaxValue();
    phi *= scale * area * Pi;

    DirectionCone nb = shape.NormalBounds();
    return LightBounds(shape.Bounds(), nb.w, phi, nb.cosTheta, std::cos(Pi / 2),
                       twoSided);
}

pstd::optional<LightLeSample> DiffuseAreaLight::SampleLe(Point2f u1, Point2f u2,
                                                         SampledWavelengths &lambda,
                                                         Float time) const {
    // Sample a point on the area light's _Shape_
    pstd::optional<ShapeSample> ss = shape.Sample(u1);
    if (!ss)
        return {};
    ss->intr.time = time;
    ss->intr.mediumInterface = &mediumInterface;

    // Check sampled point on shape against alpha texture, if present
    if (AlphaMasked(ss->intr))
        return {};

    // Sample a cosine-weighted outgoing direction _w_ for area light
    Vector3f w;
    Float pdfDir;
    if (twoSided) {
        // Choose side of surface and sample cosine-weighted outgoing direction
        if (u2[0] < 0.5f) {
            u2[0] = std::min(u2[0] * 2, OneMinusEpsilon);
            w = SampleCosineHemisphere(u2);
        } else {
            u2[0] = std::min((u2[0] - 0.5f) * 2, OneMinusEpsilon);
            w = SampleCosineHemisphere(u2);
            w.z *= -1;
        }
        pdfDir = CosineHemispherePDF(std::abs(w.z)) / 2;

    } else {
        w = SampleCosineHemisphere(u2);
        pdfDir = CosineHemispherePDF(w.z);
    }
    if (pdfDir == 0)
        return {};

    // Return _LightLeSample_ for ray leaving area light
    const Interaction &intr = ss->intr;
    Frame nFrame = Frame::FromZ(intr.n);
    w = nFrame.FromLocal(w);
    SampledSpectrum Le = L(intr.p(), intr.n, intr.uv, w, lambda);
    return LightLeSample(Le, intr.SpawnRay(w), intr, ss->pdf, pdfDir);
}

void DiffuseAreaLight::PDF_Le(const Interaction &intr, Vector3f w, Float *pdfPos,
                              Float *pdfDir) const {
    CHECK_NE(intr.n, Normal3f(0, 0, 0));
    *pdfPos = shape.PDF(intr);
    *pdfDir = twoSided ? (CosineHemispherePDF(AbsDot(intr.n, w)) / 2)
                       : CosineHemispherePDF(Dot(intr.n, w));
}

std::string DiffuseAreaLight::ToString() const {
    return StringPrintf("[ DiffuseAreaLight %s Lemit: %s scale: %f shape: %s alpha: %s "
                        "twoSided: %s area: %f image: %s ]",
                        BaseToString(), Lemit, scale, shape, alpha,
                        twoSided ? "true" : "false", area, image);
}

DiffuseAreaLight *DiffuseAreaLight::Create(const Transform &renderFromLight,
                                           Medium medium,
                                           const ParameterDictionary &parameters,
                                           const RGBColorSpace *colorSpace,
                                           const FileLoc *loc, Allocator alloc,
                                           const Shape shape, FloatTexture alphaTex) {
    Spectrum L = parameters.GetOneSpectrum("L", nullptr, SpectrumType::Illuminant, alloc);
    Float scale = parameters.GetOneFloat("scale", 1);
    bool twoSided = parameters.GetOneBool("twosided", false);

    std::string filename = ResolveFilename(parameters.GetOneString("filename", ""));
    Image image(alloc);
    const RGBColorSpace *imageColorSpace = nullptr;
    if (!filename.empty()) {
        if (L)
            ErrorExit(loc, "Both \"L\" and \"filename\" specified for DiffuseAreaLight.");
        ImageAndMetadata im = Image::Read(filename, alloc);

        if (im.image.HasAnyInfinitePixels())
            ErrorExit(
                loc,
                "%s: image has infinite pixel values and so is not suitable as a light.",
                filename);
        if (im.image.HasAnyNaNPixels())
            ErrorExit(loc,
                      "%s: image has not-a-number pixel values and so is not suitable as "
                      "a light.",
                      filename);

        ImageChannelDesc channelDesc = im.image.GetChannelDesc({"R", "G", "B"});
        if (!channelDesc)
            ErrorExit(loc,
                      "%s: Image provided to \"diffuse\" area light must have "
                      "R, G, and B channels.",
                      filename);
        image = im.image.SelectChannels(channelDesc, alloc);

        imageColorSpace = im.metadata.GetColorSpace();
    } else if (!L)
        L = &colorSpace->illuminant;

    // scale so that radiance is equivalent to 1 nit
    scale /= SpectrumToPhotometric(L ? L : &colorSpace->illuminant);

    Float phi_v = parameters.GetOneFloat("power", -1.0f);
    if (phi_v > 0) {
        // k_e is the emissive power of the light as defined by the spectral
        // distribution and texture and is used to normalize the emitted
        // radiance such that the user-defined power will be the actual power
        // emitted by the light.
        Float k_e;
        // Get the appropriate luminance vector from the image colour space
        RGB lum = imageColorSpace->LuminanceVector();
        // we need to know which channels correspond to R, G and B
        // we know that the channelDesc is valid as we would have exited in the
        // block above otherwise
        ImageChannelDesc channelDesc = image.GetChannelDesc({"R", "G", "B"});
        if (image) {
            k_e = 0;
            // Assume no distortion in the mapping, FWIW...
            for (int y = 0; y < image.Resolution().y; ++y)
                for (int x = 0; x < image.Resolution().x; ++x) {
                    for (int c = 0; c < 3; ++c)
                        k_e += image.GetChannel({x, y}, c) * lum[c];
                }
            k_e /= image.Resolution().x * image.Resolution().y;
        }

        k_e *= (twoSided ? 2 : 1) * shape.Area() * Pi;

        // now multiply up scale to hit the target power
        scale *= phi_v / k_e;
    }

    return alloc.new_object<DiffuseAreaLight>(renderFromLight, medium, L, scale, shape,
                                              alphaTex, std::move(image), imageColorSpace,
                                              twoSided);
}