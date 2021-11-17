//Production Volume Rendering SIGGRAPH 2017 Course
// Single scaering heterogeneous integrator
class SingleScatterHeterogeneousVolume : public Volume
{
public:
    SingleScatterHeterogeneousVolume(int scatteringAlbedoProperty, Color &maxExtinction,
                                     int extinctionProperty, ShadingContext &ctx) : Volume(ctx), m_scatteringAlbedoProperty(scatteringAlbedoProperty),
                                                                                    m_maxExtinction(maxExtinction.ChannelAvg()),
                                                                                    m_extinctionProperty(extinctionProperty) {}
    virtual bool Integrate(RendererServices &rs, const Ray &wi, Color &L, Color &transmittance, Color &weight, Point &P, Ray &wo, Geometry &g)
    {
        Point P0 = m_ctx.GetP();
        if (!rs.GetNearestHit(Ray(P0, wi.dir), P, g))
            return false;
        float distance = Vector(P - P0).Length();
        bool terminated = false;
        float t = 0;
        do
        {
            float zeta = rs.GenerateRandomNumber();
            t = t - log(1 - zeta) / m_maxExtinction;
            if (t > distance)
            {
                break; // Did not terminate in the volume
            }
            // Update the shading context
            Point Pl = P0 + t * wi.dir;
            m_ctx.SetP(Pl);
            m_ctx.RecomputeInputs();
            // Recompute the local extinction after updating the shading context
            Color extinction = m_ctx.GetColorProperty(m_extinctionProperty);
            float xi = rs.GenerateRandomNumber();
            if (xi < (extinction.ChannelAvg() / m_maxExtinction))
                terminated = true;
        } while (!terminated);
        if (terminated)
        {
            // The shading context has already been advanced to the
            // scatter location. Compute direct lighting after
            // evaluating the local scattering albedo and extinction
            Color scatteringAlbedo = m_ctx.GetColorProperty(m_scatteringAlbedoProperty);
            Color extinction = m_ctx.GetColorProperty(m_extinctionProperty);
            IsotropicPhaseBSDF phaseBSDF(m_ctx);
            L = Color(0.0);
            Color lightL, bsdfL, beamTransmittance;
            float lightPdf, bsdfPdf;
            Vector sampleDirection;
            rs.GenerateLightSample(m_ctx, sampleDirection, lightL, lightPdf,
                                   beamTransmittance);
            phaseBSDF.EvaluateSample(rs, sampleDirection, bsdfL, bsdfPdf);
            L += lightL * bsdfL * beamTransmittance * scatteringAlbedo * extinction *
                 rs.MISWeight(1, lightPdf, 1, bsdfPdf) / lightPdf;
            phaseBSDF.GenerateSample(rs, sampleDirection, bsdfL, bsdfPdf);
            rs.EvaluateLightSample(m_ctx, sampleDirection, lightL, lightPdf,
                                   beamTransmittance);
            L += lightL * bsdfL * beamTransmittance * scatteringAlbedo * extinction *
                 rs.MISWeight(1, lightPdf, 1, bsdfPdf) / bsdfPdf;
            transmittance = Color(0.0f);
            Color pdf = extinction; // Should be extinction * Tr, but Tr is 1
            weight = 1.0 / pdf;
        }
        else
        {
            transmittance = Color(1.0f);
            weight = Color(1.0f);
        }
        wo = Ray(P, wi.dir);
        return true;
    }