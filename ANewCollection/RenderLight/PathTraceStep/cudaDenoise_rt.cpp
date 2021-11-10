//https://github.com/ZheyuanXie/Project3-CUDA-Path-Tracer
// compute shadow ray by randomly sampling in a unit circle centered at the light source
__host__ __device__
void computeShadowRay(Ray& shadowRay, glm::vec3 originPos, Geom& light, float lightRadius, float& shadowRayExpectDist, unsigned int& seed) {
    glm::vec3 directionToCenter = glm::normalize(light.translation - originPos);
    glm::quat rot = glm::rotation(glm::vec3(0.0f, 0.0f, 1.0f), directionToCenter);
    float theta = 2 * PI * nextRand(seed);
    glm::vec3 sampleDirection = glm::rotate(rot, glm::vec3(cosf(theta), sinf(theta), 0.0f));
    float sampleRadius = nextRand(seed) * lightRadius;

    glm::vec3 samplePoint = light.translation + sampleDirection * sampleRadius;
    shadowRayExpectDist = glm::l2Norm(samplePoint - originPos);

    shadowRay.origin = originPos;
    shadowRay.direction = glm::normalize(samplePoint - originPos);
}

// do ray tracing kernel
__global__ void rt(int frame, int num_paths, int max_depth,
    PathSegment * pathSegments, ShadeableIntersection * intersections, 
    Geom * geoms, int geoms_size, Triangle* triangles, Material * materials, GBufferTexel * gbuffer, glm::vec3 * image,
    bool trace_shadowray, bool reduce_var, float sintensity, float lightSampleRadius, bool denoise, bool sepcolor,
    /* BVH & Texture */ BoundingBox * boudings, BVH_ArrNode * bvhnodes, Texture* texts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        PathSegment& segment = pathSegments[idx];
        ShadeableIntersection& intersection = intersections[idx];
        glm::vec3 accumulatedColor(0.0f);

        // compute first intersection and populate g-buffer
        bool hit = computeIntersection(segment.ray, intersection, geoms, geoms_size, triangles,
                                       /* BVH */ boudings, bvhnodes);
        Material &material = materials[intersection.materialId];
        gbuffer[idx].position = segment.ray.origin + intersection.t * segment.ray.direction;
        gbuffer[idx].normal = intersection.surfaceNormal;
        gbuffer[idx].geomId = intersection.geomId;
        gbuffer[idx].albedo = material.texid == -1 ?
                              material.color :
                              texts[material.texid].getColor(intersection.uv);
        gbuffer[idx].ialbedo = glm::vec3(1.0f);

        for (int depth = 1; depth <= max_depth; depth++) {
            if (!hit) break;

            unsigned int seed = initRand(idx, frame + depth, 16);
            Material &material = materials[intersection.materialId];

            if (material.emittance > 0.0f) {  // Hit light (terminate ray)
                if (!trace_shadowray || !reduce_var || !segment.diffuse) {
                    accumulatedColor += segment.color * material.color * material.emittance;
                }
                break;
            }
            else {                            // Hit material (scatter ray)
                glm::vec3 intersectionPos = segment.ray.origin + intersection.t * segment.ray.direction;
                glm::vec3 &intersectionNormal = intersection.surfaceNormal;
                bool materialIsDiffuse = material.hasReflective < 1e-6 && material.hasRefractive < 1e-6;

                // Apply color
                if (denoise && sepcolor) {
                    if (depth > 1) {
                        segment.color *= material.texid != -1 ? 
                                         texts[material.texid].getColor(intersection.uv) : 
                                         material.color;
                    }
                }
                else {
                    segment.color *= material.texid != -1 ? 
                                     texts[material.texid].getColor(intersection.uv) : 
                                     material.color;
                }
                glm::clamp(segment.color, glm::vec3(0.0f), glm::vec3(1.0f));

                // Trace shadow ray
                if (trace_shadowray && materialIsDiffuse) {
                    // TODO: pick random light
                    int lightIdx = 0;
                    Geom& light = geoms[lightIdx];

                    // generate shadow ray
                    Ray shadowRay;
                    float shadowRayExpectDist = 0.0f;
                    computeShadowRay(shadowRay, intersectionPos + 1e-4f * intersectionNormal, light, lightSampleRadius, shadowRayExpectDist, seed);

                    // compute shadow ray intersection
                    ShadeableIntersection shadowRayIntersection;
                    bool shadowRayHit = computeIntersection(shadowRay, shadowRayIntersection, geoms, geoms_size, triangles,
                                                            /* BVH */ boudings, bvhnodes);

                    // compute color
                    if (shadowRayIntersection.geomId == lightIdx) {
                        Material shadowRayMaterial = materials[shadowRayIntersection.materialId];
                        if (shadowRayMaterial.emittance > 0.0f) {
                            glm::vec3 shadowRayIntersectionPos = shadowRay.origin + shadowRay.direction * shadowRayIntersection.t;
                            float diffuse = glm::max(0.0f, glm::dot(shadowRay.direction, intersectionNormal));
                            float shadowIntensity = sintensity / pow(shadowRayExpectDist, 2.0f);
                            accumulatedColor += segment.color
                                                * shadowRayMaterial.emittance * shadowRayMaterial.color
                                                * shadowIntensity * diffuse;
                        }
                    }
                }

                // Bounce ray and compute intersection
                if (depth < max_depth) {
                    scatterRay(segment, intersectionPos, intersectionNormal, material, seed);
                    hit = computeIntersection(segment.ray, intersection, geoms, geoms_size, triangles,
                                              /* BVH */ boudings, bvhnodes);
                }
            }
        }
        if (denoise) {
            image[segment.pixelIndex] = accumulatedColor;
        } else {
            image[segment.pixelIndex] = image[segment.pixelIndex] * (float)frame / (float)(frame + 1) + accumulatedColor / (float)(frame + 1);
        }
    }
}
