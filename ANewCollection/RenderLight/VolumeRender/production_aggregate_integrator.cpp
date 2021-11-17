
//Production Volume Rendering SIGGRAPH 2017 Course
//Pseudo-interface for aggregate-based integrator
Color WoodcockIntegrator::integrate(const Ray &ray, const VolumeAggregate &aggregate)
{
    Iterator iter = aggregate.iterator(ray);
    for (; !iter.empty(); ++iter)
    {
        // Skip empty space
        if (iter.numVolumes() == 0)
        {
            continue;
        }
        // Track current position and end of segment
        float t = iter.start(), tEnd = iter.end();
        while (t < tEnd)
        {
            float xiStep = rng(), xiInteract = rng();
            // Take a step
            t += deltaStep(xiStep, iter.maxExtinction());
            // Did we escape the segment?
            if (t > tEnd)
            {
                break;
            }
            Point p = ray(t);
            float extinction = sumVolumeExtinction(iter.volumes(), p);
            if (xiInteract < extinction / iter.maxExtinction())
            {
                Color L = computeDirectLighting(p);
                if (ray.depth < MAX_DEPTH)
                {
                    Ray indirectRay = setupIndirectRay(p);
                    L += integrate(indirectRay);
                }
                // Terminate since a real collision was found
                return L;
            }
        }
    }
};