
//Production Volume Rendering SIGGRAPH 2017 Course
//Delta tracking using bounded subintervals from a k-d tree
bool takeInitialStep(RendererServices &rs, RtInt rangeCounts, float const *rangeDists,
                     float const *maxDensities, float *accumDistance, int *currentSeg)
{
    bool finished = false;
    do
    {
        // Find first interesting segment
        int i = *currentSeg;
        float maxRayDensity = maxDensities[i];
        // Skip over empty segments entirely
        if (maxRayDensity == 0)
        {
            if (i == rangeCounts - 1)
            {
                // We're done with all the segments
                finished = true;
                break;
            }
            else
            {
                *accumDistance = rangeDists[i];
                *currentSeg = i + 1;
                continue;
            }
        }
        float xi = rs.GenerateRandomNumber();
        // Woodcock tracking: pick distance based on maximum density
        *accumDistance += -logf(1 - xi) / maxRayDensity;
        if (*accumDistance >= rangeDists[i])
        {
            // Skipped past the end of the current segment
            if (i == rangeCounts - 1)
            {
                finished = true;
                break;
            }
            else
            {
                // Went past the segments with no interaction. Move on to
                // the next segment, resetting the accumulated distance to
                // the beginning of that segment
                *accumDistance = rangeDists[i];
                *currentSeg = i + 1;
            }
        }
        else
            // Check for actual interaction
            break;
    } while (true);
    return finished;
}
virtual void Transmittance(RendererServices &rs, const Point &P0, const Point &P1,
                           Color &transmittance)
{
    float distance = Vector(P0 - P1).Length();
    Vector dir = Vector(P1 - P0) / distance;
    bool terminated = false;
    float t = 0;
    int currentSeg = 0;
    int rangeCounts;
    float *rangeDists, *maxDensities;
    intersectSegmentAgainstKDTree(P0, P1, &rangeCounts, &rangeDists, &maxDensities);
    do
    {
        if (takeInitialStep(rs, rangeCounts, rangeDists, maxDensities, &t, &currentSeg))
            break; // Did not terminate in the volume
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
        transmittance = Color(0.0f);
    else
        transmittance = Color(1.0f);
}