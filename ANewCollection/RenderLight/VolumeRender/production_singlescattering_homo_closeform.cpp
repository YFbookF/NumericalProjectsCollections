
//Production Volume Rendering SIGGRAPH 2017 Course
//Single scaering homogeneous integrator using closed form tracking
class SingleScatterHomogeneousVolume : public Volume
{
public:
    SingleScatterHomogeneousVolume(Color &scatteringAlbedo, Color &extinction,
                                   ShadingContext &ctx) : Volume(ctx), m_scatteringAlbedo(scatteringAlbedo), m_extinction(extinction) {}
    virtual bool Integrate(RendererServices &rs, const Ray &wi, Color &L, Color &transmittance, Color &weight, Point &P, Ray &wo, Geometry &g)
    {
        if (!rs.GetNearestHit(Ray(m_ctx.GetP(), wi.dir), P, g))
            return false;
        // Transmittance over the entire interval
        transmittance = Transmittance(rs, P, m_ctx.GetP());
        // Compute sample location for scattering, based on the PDF
        // normalized to the total transmission
        float xi = rs.GenerateRandomNumber();
        float scatterDistance = -logf(1.0f - xi * (1.0f - transmittance.ChannelAvg())) /
                                m_extinction.ChannelAvg();
        // Set up shading context to be at the scatter location
        Point Pscatter = m_ctx.GetP() + scatterDistance * wi.dir;
        m_ctx.SetP(Pscatter);
        m_ctx.RecomputeInputs();
        // Compute direct lighting with light sampling and phase function sampling
        IsotropicPhaseBSDF phaseBSDF(m_ctx);
        L = Color(0.0);
        Color lightL, bsdfL, beamTransmittance;
        float lightPdf, bsdfPdf;
        Vector sampleDirection;
        rs.GenerateLightSample(m_ctx, sampleDirection, lightL, lightPdf, beamTransmittance);
        phaseBSDF.EvaluateSample(rs, sampleDirection, bsdfL, bsdfPdf);
        L += lightL * bsdfL * beamTransmittance * rs.MISWeight(1, lightPdf, 1, bsdfPdf) /
             lightPdf;
        phaseBSDF.GenerateSample(rs, sampleDirection, bsdfL, bsdfPdf);
        rs.EvaluateLightSample(m_ctx, sampleDirection, lightL, lightPdf, beamTransmittance);
        L += lightL * bsdfL * beamTransmittance * rs.MISWeight(1, lightPdf, 1, bsdfPdf) /
             bsdfPdf;
        Color Tr(exp(m_extinction.r * -scatterDistance), exp(m_extinction.g * -scatterDistance), exp(m_extinction.b * -scatterDistance));
        L *= (m_extinction * m_scatteringAlbedo * Tr);
        // This weight is 1 over the PDF normalized to the total transmission
        weight = (1 - transmittance) / (Tr * m_extinction);
        wo = Ray(P, wi.dir);
        return true;
    }
    virtual Color Transmittance(RendererServices &rs, const Point &P0, const Point &P1)
    {
        float distance = Vector(P0 - P1).Length();
        return Color(exp(m_extinction.r * -distance), exp(m_extinction.g * -distance),
                     exp(m_extinction.b * -distance));
    }

protected:
    const Color m_scatteringAlbedo;
    const Color m_extinction;
};