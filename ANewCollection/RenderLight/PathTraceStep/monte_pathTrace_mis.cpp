//https://github.com/linusmossberg/monte-carlo-ray-tracer
glm::dvec3 PathTracer::sampleRay(Ray ray)
{
    glm::dvec3 radiance(0.0), throughput(1.0);
    RefractionHistory refraction_history(ray);
    glm::dvec3 bsdf_absIdotN;
    LightSample ls;

    while (true)
    {
        Intersection intersection = scene.intersect(ray);

        if (!intersection)
        {
            return radiance + scene.skyColor(ray) * throughput;
        }

        Interaction interaction(intersection, ray, refraction_history.externalIOR(ray));

        radiance += Integrator::sampleEmissive(interaction, ls) * throughput;
        radiance += Integrator::sampleDirect(interaction, ls) * throughput;

        if (!interaction.sampleBSDF(bsdf_absIdotN, ls.bsdf_pdf, ray))
        {
            return radiance;
        }

        throughput *= bsdf_absIdotN / ls.bsdf_pdf;

        if (absorb(ray, throughput))
        {
            return radiance;
        }

        refraction_history.update(ray);
    }
}

glm::dvec3 Scene::skyColor(const Ray& ray) const
{
    double fy = (1.0 + std::asin(glm::dot(glm::dvec3(0.0, 1.0, 0.0), ray.direction)) / C::PI) / 2.0;
    return glm::mix(glm::dvec3(1.0, 0.5, 0.0), glm::dvec3(0.0, 0.5, 1.0), fy);
}

/********************************************************************
Adds emittance from interaction surface if applicable, or samples 
the emissive using the BSDF from the previous interaction using MIS.
********************************************************************/
glm::dvec3 Integrator::sampleEmissive(const Interaction& interaction, const LightSample &ls) const
{
    if (interaction.material->emissive && !interaction.inside)
    {
        if (interaction.ray.depth == 0 || interaction.ray.dirac_delta)
        {
            return interaction.material->emittance;
        }
        if(ls.light == interaction.surface)
        {
            double cos_light_theta = glm::dot(interaction.out, interaction.normal);
            double light_pdf = pow2(interaction.t) / (interaction.surface->area() * cos_light_theta);
            double mis_weight = powerHeuristic(ls.bsdf_pdf, light_pdf);
            return mis_weight * interaction.material->emittance / ls.select_probability;
        }
    }
    return glm::dvec3(0.0);
}

inline double powerHeuristic(double a_pdf, double b_pdf)
{
    double a_pdf2 = a_pdf * a_pdf;
    return a_pdf2 / (a_pdf2 + b_pdf * b_pdf);
}
