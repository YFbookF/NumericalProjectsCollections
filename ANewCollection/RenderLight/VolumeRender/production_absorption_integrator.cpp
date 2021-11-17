//Production Volume Rendering SIGGRAPH 2017 Course
//Homogeneous absorption integrator
class BeersLawVolume : public Volume
{
public:
    BeersLawVolume(const Color &absorption, ShadingContext &ctx) : Volume(ctx), m_absorption(absorption) {}
    virtual bool Integrate(RendererServices &rs, const Ray &wi, Color &L, Color &transmittance, Color &weight, Point &P, Ray &wo, Geometry &g)
    {
        if (!rs.GetNearestHit(Ray(m_ctx.GetP(), wi.dir), P, g))
            return false;
        L = Color(0.0);
        transmittance = Transmittance(rs, P, m_ctx.GetP());
        weight = Color(1.0);
        wo = Ray(P, wi.dir);
        return true;
    }
    virtual Color Transmittance(RendererServices &rs, const Point &P0, const Point &P1)
    {
        float distance = Vector(P0 - P1).Length();
        return Color(exp(m_absorption.r * -distance), exp(m_absorption.g * -distance),
                     exp(m_absorption.b * -distance));
    }

protected:
    const Color m_absorption;
};