// https://github.com/sunwj/SunVolumeRender
// Created by 孙万捷 on 16/6/26.
//

#ifndef SUNVOLUMERENDER_LAMBERT_H
#define SUNVOLUMERENDER_LAMBERT_H

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include <curand_kernel.h>

#include "../sampling.h"

__inline__ __device__ float lambert_brdf_f(const glm::vec3& wi, const glm::vec3& wo)
{
    return 1.f / float(M_PI);
}

__inline__ __device__ void lambert_brdf_sample_f(const glm::vec3& wo, const glm::vec3& normal, glm::vec3* wi, float* pdf, curandState& rng)
{
    *wi = cosine_weightd_sample_hemisphere(rng, normal);
    *pdf = fabsf(glm::dot(*wi, normal)) / float(M_PI);
}

#endif //SUNVOLUMERENDER_LAMBERT_H
