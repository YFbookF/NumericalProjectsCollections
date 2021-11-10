https://github.com/ZheyuanXie/Project3-CUDA-Path-Tracer
/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 * 
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
        PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        unsigned int& seed)
{
    //pathSegment.diffuse = false;
    pathSegment.specular = false;
    pathSegment.ray.origin = intersect + 1e-4f * normal;    // New ray shoot from intersection point
    pathSegment.remainingBounces--;                         // Decrease bounce counter

    if (m.hasRefractive) {                                  // Refreaction
        float eta = 1.0f / m.indexOfRefraction;
        float unit_projection = glm::dot(pathSegment.ray.direction, normal);
        if (unit_projection > 0) {
            eta = 1.0f / eta;
        }

        // Schlick's approximation
        float R0 = powf((1.0f - eta) / (1.0f + eta), 2.0f);
        float R = R0 + (1 - R0) * powf(1 - glm::abs(unit_projection), 5.0f);
        if (R < nextRand(seed)) {
            // Refracting Light
            pathSegment.ray.direction = glm::refract(pathSegment.ray.direction, normal, eta);
            normal = -normal;
        }
        else {
            // Reflecting Light
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
            pathSegment.color *= m.specular.color;
            pathSegment.specular = true;
        }
    } else if (nextRand(seed) < m.hasReflective) {                  // Specular
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.color *= m.specular.color;
        pathSegment.specular = true;
    } else {                                                        // Diffusive
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, seed);
        pathSegment.diffuse = true;
    }
}
