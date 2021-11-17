//Production Volume Rendering SIGGRAPH 2017 Course
//forward path tracing
Color L = Color(0.0);
Color throughput = Color(1.0);
Ray ray = pickCameraDirection();
if (rs.GetNearestHit(ray, P, g))
    continue;
int j = 0;
while (j < maxPathLength)
{
    ShadingContext *ctx = rs.CreateShadingContext(P, ray, g);
    Material *m = g.GetMaterial();
    BSDF *bsdf = m->CreateBSDF(*ctx);
    // Perform direct lighting on the surface
    L += throughput * directLighting();
    // Compute direction of indirect ray
    float pdf;
    Color Ls;
    Vector sampleDirection;
    bsdf->GenerateSample(rs, sampleDirection, Ls, pdf);
    throughput *= (Ls / pdf);
    Ray nextRay(ray);
    nextRay.org = P;
    nextRay.dir = sampleDirection;

    Volume *volume = 0;
    if (m->HasVolume())
    {
        // Did we go transmit through the surface? V is the
        // direction away from the point P on the surface.
        float VdotN = ctx->GetV().Dot(ctx->GetN());
        float dirDotN = sampleDirection.Dot(ctx->GetN());
        bool transmit = (VdotN < 0.0) != (dirDotN < 0.0);
        if (transmit)
        {
            // We transmitted through the surface. Check dot
            // product between the sample direction and the
            // surface normal N to see whether we entered or
            // exited the volume media
            bool entered = dirDotN < 0.0f;
            if (entered)
            {
                nextRay.EnterMaterial(m);
            }
            else
            {
                nextRay.ExitMaterial(m);
            }
        }
        volume = nextRay.GetVolume(*ctx);
    }
    if (volume)
    {
        Color Lv;
        Color transmittance;
        float weight;
        if (!volume->Integrate(rs, nextRay, Lv, transmittance, weight, P, nextRay,
                               g))
            break;
        L += weight * throughput * Lv;
        throughput *= transmittance;
    }
    else
    {
        if (!rs.GetNearestHit(nextRay, P, g))
            break;
    }
    if (!rs.GetNearestHit(nextRay, P, g))
        break;
    ray = nextRay;
    j++;
}