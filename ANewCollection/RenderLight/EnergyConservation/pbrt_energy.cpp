
///////////////////////////////////////////////////////////////////////////
// Energy Conservation Tests   https://github.com/mmp/pbrt-v4

static void TestEnergyConservation(
    std::function<BSDF*(const SurfaceInteraction&, Allocator)> createBSDF,
    const char* description) {
    RNG rng;

    // Create BSDF, which requires creating a Shape, casting a Ray that
    // hits the shape to get a SurfaceInteraction object.
    auto t = std::make_shared<const Transform>(RotateX(-90));
    auto tInv = std::make_shared<const Transform>(Inverse(*t));

    bool reverseOrientation = false;
    std::shared_ptr<Disk> disk =
        std::make_shared<Disk>(t.get(), tInv.get(), reverseOrientation, 0., 1., 0, 360.);
    Point3f origin(0.1, 1,
                   0);  // offset slightly so we don't hit center of disk
    Vector3f direction(0, -1, 0);
    Ray r(origin, direction);
    auto si = disk->Intersect(r);
    ASSERT_TRUE(si.has_value());
    BSDF* bsdf = createBSDF(si->intr, Allocator());

    for (int i = 0; i < 10; ++i) {
        Point2f uo{rng.Uniform<Float>(), rng.Uniform<Float>()};
        Vector3f woL = SampleUniformHemisphere(uo);
        Vector3f wo = bsdf->LocalToRender(woL);

        const int nSamples = 16384;
        SampledSpectrum Lo(0.f);
        for (int j = 0; j < nSamples; ++j) {
            Float u = rng.Uniform<Float>();
            Point2f ui{rng.Uniform<Float>(), rng.Uniform<Float>()};
            pstd::optional<BSDFSample> bs = bsdf->Sample_f(wo, u, ui);
            if (bs)
                Lo += bs->f * AbsDot(bs->wi, si->intr.n) / bs->pdf;
        }
        Lo /= nSamples;

        EXPECT_LT(Lo.MaxComponentValue(), 1.01)
            << description << ": Lo = " << Lo << ", wo = " << wo;
    }
}