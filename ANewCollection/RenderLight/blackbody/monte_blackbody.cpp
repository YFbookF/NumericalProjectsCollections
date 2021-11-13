//https://github.com/linusmossberg/monte-carlo-ray-tracer
 // Normalized tristimulus of blackbody with temperature T
        inline glm::dvec3 blackbody(double T)
        {
            // Radiant emittance of blackbody for wavelength w
            auto B = [T](double w)
            {
                w *= 1e-9;

                constexpr double c = 2.99792458e+8; // Speed of light
                constexpr double h = 6.626176e-34;  // Planck's constant
                constexpr double k = 1.380662e-23;  // Boltzmann constant

                return (C::TWO_PI * h * pow2(c)) / (std::pow(w, 5) * (std::exp((h * c / k) / (T * w)) - 1.0));
            };

            glm::dvec3 result(0.0);
            for (double w = CMF.a + 0.5 * CMF.dw; w < CMF.b; w += CMF.dw)
            {
                result += B(w) * CMF(w);
            }
            return result / result.y;
        }