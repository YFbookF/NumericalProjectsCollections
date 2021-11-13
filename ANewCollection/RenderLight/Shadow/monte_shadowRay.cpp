//https://github.com/linusmossberg/monte-carlo-ray-tracer
/**************************************************************************
Samples a light source using MIS. The BSDF is sampled using MIS later 
in the next interaction in sampleEmissive, if the ray hits the same light.
**************************************************************************/
glm::dvec3 Integrator::sampleDirect(const Interaction& interaction, LightSample& ls) const
{
    if (scene.emissives.empty() || interaction.material->dirac_delta)
    {
        ls.light = nullptr;
        return glm::dvec3(0.0);
    }

    // Pick one light source and divide with probability of selecting light source
    ls.light = scene.selectLight(ls.select_probability);

    glm::dvec3 light_pos = ls.light->operator()(Random::unit(), Random::unit());
    Ray shadow_ray(interaction.position + interaction.normal * C::EPSILON, light_pos);

    double cos_light_theta = glm::dot(-shadow_ray.direction, ls.light->normal(light_pos));

    if (cos_light_theta <= 0.0)
    {
        return glm::dvec3(0.0);
    }

    double cos_theta = glm::dot(shadow_ray.direction, interaction.normal);
    if (cos_theta <= 0.0)
    {
        if (interaction.material->opaque || cos_theta == 0.0)
        {
            return glm::dvec3(0.0);
        }
        else
        {
            // Try transmission
            shadow_ray = Ray(interaction.position - interaction.normal * C::EPSILON, light_pos);
        }
    }

    Intersection shadow_intersection = scene.intersect(shadow_ray);

    if (!shadow_intersection || shadow_intersection.surface != ls.light)
    {
        return glm::dvec3(0.0);
    }    

    double light_pdf = pow2(shadow_intersection.t) / (ls.light->area() * cos_light_theta);

    double bsdf_pdf;
    glm::dvec3 bsdf_absIdotN;
    if (!interaction.BSDF(bsdf_absIdotN, shadow_ray.direction, bsdf_pdf))
    {
        return glm::dvec3(0.0);
    }

    double mis_weight = powerHeuristic(light_pdf, bsdf_pdf);

    return mis_weight * bsdf_absIdotN * ls.light->material->emittance / (light_pdf * ls.select_probability);
}
