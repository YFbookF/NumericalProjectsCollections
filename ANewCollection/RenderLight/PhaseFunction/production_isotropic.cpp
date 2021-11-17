//Production Volume Rendering SIGGRAPH 2017 Course
//Isotropic phase function
class IsotropicPhaseBSDF : public BSDF
{
public:
    IsotropicPhaseBSDF(ShadingContext &ctx) : BSDF(ctx) {}
    virtual void EvaluateSample(RendererServices &rs, const Vector &sampleDirection, Color &L, float &pdf)
    {
        pdf = 0.25 / M_PI;
        L = Color(pdf);
    }
    virtual void GenerateSample(RendererServices &rs, Vector &sampleDirection, Color &L,
                                float &pdf)
    {
        float xi = rs.GenerateRandomNumber();
        sampleDirection.z = xi * 2.0 - 1.0;                           // cosTheta
        float sinTheta = 1.0 - sampleDirection.z * sampleDirection.z; // actually square of
        sinTheta if (sinTheta > 0.0)
        {
            sinTheta = std::sqrt(sinTheta);
            xi = rs.GenerateRandomNumber();
            float phi = xi * 2.0 * M_PI;
            sampleDirection.x = sinTheta * cosf(phi);
            sampleDirection.y = sinTheta * sinf(phi);
        }
        else sampleDirection.x = sampleDirection.y = 0.0;
        pdf = 0.25 / M_PI;
        L = Color(pdf);
    }
};