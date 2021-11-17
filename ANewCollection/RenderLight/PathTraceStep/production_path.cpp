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
    if (!rs.GetNearestHit(nextRay, P, g))
        break;
    ray = nextRay;
    j++;
}