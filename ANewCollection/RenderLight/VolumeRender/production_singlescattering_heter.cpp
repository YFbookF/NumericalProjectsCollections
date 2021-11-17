
//Production Volume Rendering SIGGRAPH 2017 Course
//Single scaering heterogenous integrator extended with MIS
Color SingleScatterHeterogeneousVolume::directLighting(RendererServices &rs)
{
    Color scatteringAlbedo = m_ctx.GetColorProperty(m_scatteringAlbedoProperty);
    Color extinction = m_ctx.GetColorProperty(m_extinctionProperty);
    IsotropicPhaseBSDF phaseBSDF(scatteringAlbedo, m_ctx);
    Color L = Color(0.0);
    Color lightL, bsdfL, beamTransmittance;
    float lightPdf, bsdfPdf;
    Vector sampleDirection;
    rs.GenerateLightSample(m_ctx, sampleDirection, lightL, lightPdf, beamTransmittance);
    phaseBSDF.EvaluateSample(rs, sampleDirection, bsdfL, bsdfPdf);
    L += lightL * bsdfL * beamTransmittance * extinction * rs.MISWeight(1, lightPdf, 1, bsdfPdf) / lightPdf;
    phaseBSDF.GenerateSample(rs, sampleDirection, bsdfL, bsdfPdf);
    rs.EvaluateLightSample(m_ctx, sampleDirection, lightL, lightPdf, beamTransmittance);
    L += lightL * bsdfL * beamTransmittance * extinction * rs.MISWeight(1, lightPdf, 1, bsdfPdf) / bsdfPdf;
    return L;
}
virtual bool SingleScatterHeterogeneousVolume::Integrate(RendererServices &rs, const Ray &wi, Color &L, Color &transmittance, Color &weight, Point &P, Ray &wo, Geometry &g)
{
    Point P0 = m_ctx.GetP(), Pl;
    if (!rs.GetNearestHit(Ray(P0, wi.dir), P, g))
        return false;
    Color extinction;
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
        Pl = P0 + t * wi.dir;
        m_ctx.SetP(Pl);
        m_ctx.RecomputeInputs();
        // Recompute the local extinction after updating the shading context
        extinction = m_ctx.GetColorProperty(m_extinctionProperty);
        float xi = rs.GenerateRandomNumber();
        if (xi < (extinction.ChannelAvg() / m_maxExtinction))
            terminated = true;
    } while (!terminated);
    // Generate a uniform sampling location
    float uniformDistance = rs.GenerateRandomNumber() * distance;
    L = Color(0.0);
    // Lighting and transmittance estimator from delta tracking
    float uniformPDF = 1.0 / distance, deltaPDF, w;
    if (terminated)
    {
        extinction = m_ctx.GetColorProperty(m_extinctionProperty);
        deltaPDF = extinction.ChannelAvg(); // extinction x transmittance, which is one
        w = rs.MISWeight(1, deltaPDF, 1, uniformPDF) / deltaPDF;
        L += directLighting(rs) * w;
        transmittance = Color(0.0f);
    }
    else
        transmittance = Color(1.0f);
    // Lighting and transmittance estimator from uniform sampling
    Pl = P0 + uniformDistance * wi.dir;
    m_ctx.SetP(Pl);
    m_ctx.RecomputeInputs();
    if ((terminated && t > uniformDistance) || !terminated)
    {
        extinction = m_ctx.GetColorProperty(m_extinctionProperty);
        deltaPDF = extinction.ChannelAvg();
    }
    else
        deltaPDF = 0.0;
    w = rs.MISWeight(1, deltaPDF, 1, uniformPDF) / uniformPDF;
    L += directLighting(rs) * w;
    transmittance += Color(0.0f) * w;
    weight = Color(1.0);
    wo = Ray(P, wi.dir);
    return true;
}