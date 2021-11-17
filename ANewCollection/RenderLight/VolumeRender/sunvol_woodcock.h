//
// Created by 孙万捷 on 16/5/22.https://github.com/sunwj/SunVolumeRender
//

#ifndef SUNVOLUMERENDER_WOODCOCK_TRACKING_H
#define SUNVOLUMERENDER_WOODCOCK_TRACKING_H

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include <curand.h>
#include <curand_kernel.h>

#include "cuda_ray.h"
#include "cuda_volume.h"
#include "cuda_transfer_function.h"
https://github.com/sunwj/SunVolumeRender
#define BASE_SAMPLE_STEP_SIZE 1.f

__inline__ __device__ float sample_distance(const cudaRay& ray, const cudaVolume& volume, const cudaTransferFunction& tf, curandState& rng)
{
    float tNear, tFar;
    if(volume.Intersect(ray, &tNear, &tFar))
    {
        ray.tMin = tNear < 0.f ? 1e-6 : tNear;
        ray.tMax = tFar;
        auto t = ray.tMin;

        float sigmaMax = tf.GetMaxOpacity();
        float invSigmaMax = 1.f / sigmaMax;
        float invSigmaMaxSampleInterval = 1.f / (sigmaMax * BASE_SAMPLE_STEP_SIZE);
        while(true)
        {
            t += -logf(1.f - curand_uniform(&rng)) * invSigmaMaxSampleInterval;
            if(t > ray.tMax)
                return -FLT_MAX;

            auto ptInWorld = ray.PointOnRay(t);
            auto intensity = volume(ptInWorld);
            auto color_opacity = tf(intensity);
            auto sigma_t = color_opacity.w;

            if(curand_uniform(&rng) < sigma_t * invSigmaMax || t > ray.tMax)
                break;
        }

        return t;
    }

    return -FLT_MAX;
}

#endif //SUNVOLUMERENDER_WOODCOCK_TRACKING_H
