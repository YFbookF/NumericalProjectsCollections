// https://github.com/xelatihy/yocto-gl/
// Eyelight for quick previewing.
static trace_result trace_eyelight(const scene_data& scene,
    const scene_bvh& bvh, const trace_lights& lights, const ray3f& ray_,
    rng_state& rng, const trace_params& params) {
  // initialize
  auto radiance   = vec3f{0, 0, 0};
  auto weight     = vec3f{1, 1, 1};
  auto ray        = ray_;
  auto hit        = false;
  auto hit_albedo = vec3f{0, 0, 0};
  auto hit_normal = vec3f{0, 0, 0};
  auto opbounce   = 0;

  // trace  path
  for (auto bounce = 0; bounce < max(params.bounces, 4); bounce++) {
    // intersect next point
    auto intersection = intersect_scene(bvh, scene, ray);
    if (!intersection.hit) {
      if (bounce > 0 || !params.envhidden)
        radiance += weight * eval_environment(scene, ray.d);
      break;
    }

    // prepare shading point
    auto outgoing = -ray.d;
    auto position = eval_shading_position(scene, intersection, outgoing);
    auto normal   = eval_shading_normal(scene, intersection, outgoing);
    auto material = eval_material(scene, intersection);

    // handle opacity
    if (material.opacity < 1 && rand1f(rng) >= material.opacity) {
      if (opbounce++ > 128) break;
      ray = {position + ray.d * 1e-2f, ray.d};
      bounce -= 1;
      continue;
    }

    // set hit variables
    if (bounce == 0) {
      hit        = true;
      hit_albedo = material.color;
      hit_normal = normal;
    }

    // accumulate emission
    auto incoming = outgoing;
    radiance += weight * eval_emission(material, normal, outgoing);

    // brdf * light
    radiance += weight * pif *
                eval_bsdfcos(material, normal, outgoing, incoming);

    // continue path
    if (!is_delta(material)) break;
    incoming = sample_delta(material, normal, outgoing, rand1f(rng));
    if (incoming == vec3f{0, 0, 0}) break;
    weight *= eval_delta(material, normal, outgoing, incoming) /
              sample_delta_pdf(material, normal, outgoing, incoming);
    if (weight == vec3f{0, 0, 0} || !isfinite(weight)) break;

    // setup next iteration
    ray = {position, incoming};
  }

  return {radiance, hit, hit_albedo, hit_normal};
}