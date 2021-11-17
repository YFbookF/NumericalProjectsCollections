
//Production Volume Rendering SIGGRAPH 2017 Course
//Calculating transmiance with ratio tracking
virtual Color BeersLawHeterogeneousVolume::Transmittance(RendererServices &rs, const Point &P0, const Point &P1)
{
    float distance = Vector(P0 - P1).Length();
    Vector dir = Vector(P1 - P0) / distance;
    float t = 0;
    Color transmittance = Color(1.0);
    do
    {
        float zeta = rs.GenerateRandomNumber();
        t = t - log(1 - zeta) / m_maxExtinction;
        if (t > distance)
        {
            break; // Did not terminate in the volume
        }
        // Update the shading context
        Point P = P0 + t * dir;
        m_ctx.SetP(P);
        m_ctx.RecomputeInputs();
        // Recompute the local extinction after updating the shading context
        Color extinction = m_ctx.GetColorProperty(m_extinctionProperty);
        transmittance *= Color(1.0) - (extinction / m_maxExtinction);
    } while (true);
    return transmittance;
}