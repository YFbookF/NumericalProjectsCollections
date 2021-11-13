//https://github.com/linusmossberg/monte-carlo-ray-tracer
glm::dvec3 Material::diffuseReflection(const glm::dvec3& wi, const glm::dvec3& wo, double& PDF) const
{
    if (wi.z < 0.0)
    {
        PDF = 0.0;
        return glm::dvec3(0.0);
    }

    PDF = wi.z * C::INV_PI;
    return rough ? OrenNayar(wi, wo) : lambertian();
}

glm::dvec3 Material::lambertian() const
{
    return reflectance * C::INV_PI;
}

// Avoids trigonometric functions for increased performance.
glm::dvec3 Material::OrenNayar(const glm::dvec3& wi, const glm::dvec3& wo) const
{
    // equivalent to dot(normalize(i.x, i.y, 0), normalize(o.x, o.y, 0)).
    // i.e. remove z-component (normal) and get the cos angle between vectors with dot
    double cos_delta_phi = glm::clamp((wi.x*wo.x + wi.y*wo.y) / 
                           std::sqrt((pow2(wi.x) + pow2(wi.y)) * 
                           (pow2(wo.x) + pow2(wo.y))), 0.0, 1.0);

    // D = sin(alpha) * tan(beta), i.z = dot(i, (0,0,1))
    double D = std::sqrt((1.0 - pow2(wi.z)) * (1.0 - pow2(wo.z))) / std::max(wi.z, wo.z);

    // A and B are pre-computed in constructor.
    return lambertian() * (A + B * cos_delta_phi * D);
}