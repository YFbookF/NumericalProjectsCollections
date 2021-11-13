//https://github.com/linusmossberg/monte-carlo-ray-tracer
glm::dvec3 Material::specularReflection(const glm::dvec3& wi, const glm::dvec3& wo, double& PDF) const
{
    if (wi.z < 0.0)
    {
        PDF = 0.0;
        return glm::dvec3(0.0);
    }
    if (rough_specular)
    {
        return specular_reflectance * GGX::reflection(wi, wo, a, PDF);
    }
    else
    {
        PDF = 1.0;
        return specular_reflectance / std::abs(wi.z);
    }        
}