//https://github.com/linusmossberg/monte-carlo-ray-tracer
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
