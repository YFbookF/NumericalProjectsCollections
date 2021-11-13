//https://github.com/linusmossberg/monte-carlo-ray-tracer
glm::dvec3 Material::specularTransmission(const glm::dvec3& wi, const glm::dvec3& wo, double n1, 
                                          double n2, double& PDF, bool inside, bool flux) const
{
    if (wi.z > 0.0)
    {
        PDF = 0.0;
        return glm::dvec3(0.0);
    }

    glm::dvec3 btdf = !inside ? transmittance : glm::dvec3(1.0);
    if (rough_specular)
    {
        btdf *= GGX::transmission(wi, wo, n1, n2, a, PDF);
        if (flux) btdf *= pow2(n2 / n1);
    }
    else
    {
        PDF = 1.0;
        btdf *= transmittance / std::abs(wi.z);
        if (!flux) btdf *= pow2(n1 / n2);
    }
    return btdf;
}
