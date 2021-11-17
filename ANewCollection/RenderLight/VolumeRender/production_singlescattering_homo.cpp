
//Production Volume Rendering SIGGRAPH 2017 Course
//Single scaering homogeneous integrator extended for overlapping volumes
cclass SingleScatterHomogeneousVolume : public Volume
{
public:
    SingleScatterHomogeneousVolume(Color & scatteringAlbedo, Color & extinction,
                                   ShadingContext & ctx) : Volume(ctx), m_scatteringAlbedo(scatteringAlbedo), m_extinction(extinction) {}
    virtual bool Integrate(RendererServices & rs, const Ray &wi, Color &L, Color &transmittance, Color &weight, Point &P, Ray &wo, Geometry &g)
    {
        if (!rs.GetNearestHit(Ray(m_ctx.GetP(), wi.dir), P, g))
            return false;
        // Sum the extinction over all volumes. Build a CDF of their densities
        float distance = Vector(P - m_ctx.GetP()).Length();
        std::vector<Volume *> &volumes = m_ctx.GetAllVolumes();
        std::vector<float> cdf;
        Color summedExtinction(0.0);
        for (auto v = volumes.begin(); v != volumes.end(); ++v)
        {
            summedExtinction += (*v)->GetExtinction();
            cdf.push_back(summedExtinction.ChannelAvg());
        }
        for (auto f = cdf.begin(); f != cdf.end(); ++f)
        {
            *f /= summedExtinction.ChannelAvg();
        }
        // Transmittance over the entire interval, computed over all volumes
        transmittance = Color(exp(summedExtinction.r * -distance), exp(summedExtinction.g * -distance), exp(summedExtinction.b * -distance));
        // Compute sample location for scattering, based on the PDF
        // normalized to the total transmission
        float xi = rs.GenerateRandomNumber();
        float scatterDistance = -logf(1.0f - xi * (1.0f - transmittance.ChannelAvg())) /
                                summedExtinction.ChannelAvg();
        // Set up shading context to be at the scatter location
        Point Pscatter = m_ctx.GetP() + scatterDistance * wi.dir;
        m_ctx.SetP(Pscatter);
        m_ctx.RecomputeInputs();
        // Pick one of the overlapping volumes
        float zeta = rs.GenerateRandomNumber();
        unsigned index;
        for (index = 0; index < cdf.size(); ++index)
        {
            if (zeta > cdf[index])
                break;
        }
        // Compute direct lighting with light sampling and phase function sampling
        BSDF *phaseBSDF = volumes[index]->CreateBSDF(m_ctx);
        L = Color(0.0);
        Color lightL, bsdfL, beamTransmittance;
        float lightPdf, bsdfPdf;
        Vector sampleDirection;
        rs.GenerateLightSample(m_ctx, sampleDirection, lightL, lightPdf, beamTransmittance);
        phaseBSDF->EvaluateSample(rs, sampleDirection, bsdfL, bsdfPdf);
        L += lightL * bsdfL * beamTransmittance * rs.MISWeight(1, lightPdf, 1, bsdfPdf) /
             lightPdf;
        phaseBSDF->GenerateSample(rs, sampleDirection, bsdfL, bsdfPdf);
        rs.EvaluateLightSample(m_ctx, sampleDirection, lightL, lightPdf, beamTransmittance);
        L += lightL * bsdfL * beamTransmittance * rs.MISWeight(1, lightPdf, 1, bsdfPdf) /
             bsdfPdf;
        Color Tr(exp(m_extinction.r * -scatterDistance), exp(m_extinction.g * -scatterDistance), exp(m_extinction.b * -scatterDistance));
        L *= (volumes[index]->GetScatteringAlbedo() * volumes[index]->GetExtinction() * Tr);
        weight = Color(1.0);
        wo = Ray(P, wi.dir);
        return true;
    }
    virtual bool CanOverlap() const { return true; }
    virtual int GetOverlapPriority() const { return 1; }
    virtual Color GetExtinction() const { return m_extinction; }
    virtual Color GetScatteringAlbedo() const { return m_scatteringAlbedo; }
    virtual BSDF *CreateBSDF(ShadingContext & ctx) const { return new IsotropicPhaseBSDF(ctx); }
    virtual Color Transmittance(RendererServices & rs, const Point &P0, const Point &P1)
    {
        float distance = Vector(P0 - P1).Length();
        return Color(exp(m_extinction.r * -distance), exp(m_extinction.g * -distance),
                     exp(m_extinction.b * -distance));
    }

protected:
    const Color m_scatteringAlbedo;
    const Color m_extinction;
};