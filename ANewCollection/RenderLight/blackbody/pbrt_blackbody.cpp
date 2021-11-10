//https://github.com/mmp/pbrt-v4
TEST(Spectrum, Blackbody) {
    // Relative error.
    auto err = [](Float val, Float ref) { return std::abs(val - ref) / ref; };

    // Planck's law.
    // A few values via
    // http://www.spectralcalc.com/blackbody_calculator/blackbody.php
    // lambda, T, expected radiance
    Float v[][3] = {
        {483, 6000, 3.1849e13},
        {600, 6000, 2.86772e13},
        {500, 3700, 1.59845e12},
        {600, 4500, 7.46497e12},
    };
    int n = PBRT_ARRAYSIZE(v);
    for (int i = 0; i < n; ++i) {
        Float lambda = v[i][0], T = v[i][1], LeExpected = v[i][2];
        EXPECT_LT(err(Blackbody(lambda, T), LeExpected), .001);
    }

    // Use Wien's displacement law to compute maximum wavelength for a few
    // temperatures, then confirm that the value returned by Blackbody() is
    // consistent with this.
    for (Float T : {2700, 3000, 4500, 5600, 6000}) {
        Float lambdaMax = 2.8977721e-3 / T * 1e9;
        Float lambda[3] = {Float(.99 * lambdaMax), lambdaMax, Float(1.01 * lambdaMax)};
        EXPECT_LT(Blackbody(lambda[0], T), Blackbody(lambda[1], T));
        EXPECT_GT(Blackbody(lambda[1], T), Blackbody(lambda[2], T));
    }
}