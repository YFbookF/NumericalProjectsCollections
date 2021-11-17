//Production Volume Rendering SIGGRAPH 2017 Course
//Heterogeneous absorption integrator
class BeersLawHeterogeneousVolume : public Volume
{
public:
    BeersLawHeterogeneousVolume(Color &maxAbsorption, int absorptionProperty,
                                ShadingContext &ctx) : Volume(ctx), m_maxAbsorption(maxAbsorption.ChannelAvg()),
                                                       m_absorptionProperty(absorptionProperty) {}
    virtual bool Integrate(RendererServices &rs, const Ray &wi, Color &L, Color &transmittance, float &weight, Point &P, Ray &wo, Geometry &g)
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
        Vector dir = Vector(P1 - P0) / distance;
        bool terminated = false;
        float t = 0;
        do
        {
            float zeta = rs.GenerateRandomNumber();
            t = t - log(1 - zeta) / m_maxAbsorption;
            if (t > distance)
            {
                break; // Did not terminate in the volume
            }
            // Update the shading context
            Point P = P0 + t * dir;
            m_ctx.SetP(P);
            m_ctx.RecomputeInputs();
            // Recompute the local absorption after updating the shading context
            Color absorption = m_ctx.GetColorProperty(m_absorptionProperty);
            float xi = rs.GenerateRandomNumber();
            if (xi < (absorption.ChannelAvg() / m_maxAbsorption))
                terminated = true;
        } while (!terminated);
        if (terminated)
            return Color(0.0);
        else
            return Color(1.0);
    }

protected:
    const float m_maxAbsorption;
    const int m_absorptionProperty;
};