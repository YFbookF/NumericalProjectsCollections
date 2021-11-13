//https://github.com/linusmossberg/monte-carlo-ray-tracer
bool Integrator::absorb(const Ray &ray, glm::dvec3 &throughput) const
{
    double survive = glm::compMax(throughput) * ray.refraction_scale;

    if (survive == 0.0) return true;

    if (ray.diffuse_depth > min_ray_depth || ray.depth > min_priority_ray_depth)
    {
        survive = std::min(0.95, survive);

        if (!Random::trial(survive))
        {
            return true;
        }
        throughput /= survive;
    }
    return false;
}