https://github.com/RickyMexx/yocto-gl-vpt
// New Path tracing.
static vec4f trace_path(const ptr::scene* scene, const ray3f& ray_,
			  rng_state& rng, const trace_params& params) {
	// initialize
	auto radiance     = zero3f;
	auto weight       = vec3f{1, 1, 1};
	auto ray          = ray_;
	auto volume_stack = std::vector<vsdf>{};
	auto hit          = false;
    
	for (auto bounce = 0; bounce < params.bounces; bounce++) {
		// intersect next point
		auto intersection = intersect_scene_bvh(scene, ray);
		if (!intersection.hit) {
			radiance += weight * eval_environment(scene, ray);
			break;
		}
		
		auto in_volume = false;
		auto incoming  = ray.d;
		auto outgoing  = -ray.d;
		auto position  = zero3f;
		
		// Handle volumes      
		if (!volume_stack.empty()) {
			auto &vsdf = volume_stack.back();
			// Handle heterogeneous volumes
      if(vsdf.htvolume) {
        if (params.vpt == DELTA) {
          // Delta tracking
          auto [t, w] = eval_delta_tracking(vsdf, intersection.distance, rng, ray);
          weight *= w;
          position = ray.o + t * ray.d;
          if (t < intersection.distance) {
            in_volume = true;
            incoming = sample_scattering(vsdf, outgoing, rand1f(rng), rand2f(rng));
            if (vsdf.event == EVENT_ABSORB) {
              auto er = zero3f;
              if (has_vpt_emission(vsdf.object))
                er = math::blackbody_to_rgb(eval_vpt_emission(vsdf, position) * 40e3);
              radiance += weight * er * vsdf.object->radiance_mult;
              break;
            }
          }
        } else if (params.vpt == SPTRK) {
          // Spectral tracking
          auto [t, w] = eval_spectral_tracking(vsdf, intersection.distance, rng, ray);
          weight *= w;
          position = ray.o + t * ray.d;
          // Handle an interaction with a medium
          if (t < intersection.distance) {
            in_volume = true;	    
            if (vsdf.event == EVENT_SCATTER)
              incoming = sample_scattering(vsdf, outgoing, rand1f(rng), rand2f(rng));
            if (vsdf.event == EVENT_ABSORB) {
              auto er = zero3f;
              if (has_vpt_emission(vsdf.object))
                er = math::blackbody_to_rgb(eval_vpt_emission(vsdf, position) * 40e3);
              radiance += weight * er * vsdf.object->radiance_mult;
              break;
            }
          }
        } else if(params.vpt == SPMIS) {
          // Spectral MIS
          auto [t, w] = eval_unidirectional_spectral_mis(vsdf, intersection.distance, rng, ray);
          weight *= w;
          position = ray.o + t * ray.d;
          // Handle an interaction with a medium
          if (t < intersection.distance) {
            in_volume = true;	    
            if (vsdf.event == EVENT_SCATTER)
              incoming = sample_scattering(vsdf, outgoing, rand1f(rng), rand2f(rng));
            if (vsdf.event == EVENT_ABSORB) {
              auto er = zero3f;
              if (has_vpt_emission(vsdf.object))
                er = math::blackbody_to_rgb(eval_vpt_emission(vsdf, position) * 40e3);
              radiance += weight * er * vsdf.object->radiance_mult;
              break;
            }
          }
        }
      } else {
        auto  distance = sample_transmittance(vsdf.density, intersection.distance, rand1f(rng), rand1f(rng));
        weight *= eval_transmittance(vsdf.density, distance) /
                  sample_transmittance_pdf(vsdf.density, distance, intersection.distance);
        in_volume             = distance < intersection.distance;
        intersection.distance = distance;
      }
    }
		
		if (!in_volume) {
			// prepare shading point
			auto object   = scene->objects[intersection.object];
			auto element  = intersection.element;
			auto uv       = intersection.uv;
			position = eval_position(object, element, uv);
			auto normal   = eval_shading_normal(object, element, uv, outgoing);
			auto emission = eval_emission(object, element, uv, normal, outgoing);
			auto brdf     = eval_brdf(object, element, uv, normal, outgoing);

			// handle opacity
			if (brdf.opacity < 1 && rand1f(rng) >= brdf.opacity) {
				ray = {position + ray.d * 1e-2f, ray.d};
				bounce -= 1;
				continue;
			}
			hit = true;
			// accumulate emission
			radiance += weight * eval_emission(emission, normal, outgoing);
			// next direction
			incoming = ray.d;
			if (!is_delta(brdf)) {
				if (rand1f(rng) < 0.5f) {
					incoming = sample_brdfcos(brdf, normal, outgoing, rand1f(rng), rand2f(rng));
				} else {
					incoming = sample_lights(scene, position, rand1f(rng), rand1f(rng), rand2f(rng));
				}
				weight *= eval_brdfcos(brdf, normal, outgoing, incoming) /
					(0.5f * sample_brdfcos_pdf(brdf, normal, outgoing, incoming) +
						0.5f * sample_lights_pdf(scene, position, incoming));
			} else {
				incoming = sample_delta(brdf, normal, outgoing, rand1f(rng));
				weight *= eval_delta(brdf, normal, outgoing, incoming) /
					sample_delta_pdf(brdf, normal, outgoing, incoming);
			} 

			// update volume stack
			if ((has_volume(object) || has_vpt_volume(object)) 
					&&  dot(normal, outgoing) * dot(normal, incoming) < 0) {
				
        if(has_vpt_volume(object)) bounce -= 1; // fix for heterogeneous volumes
				
        if (volume_stack.empty()) {
					auto volpoint = eval_vsdf(object, element, uv);
					volume_stack.push_back(volpoint);
				} else {
					volume_stack.pop_back();
				}
			} 
		} else {
      // prepare shading point
      outgoing = -ray.d;
      position = ray.o + ray.d * intersection.distance;
      auto& vsdf     = volume_stack.back();
      // handle opacity
      hit = true;
      if(!vsdf.htvolume) {
        // next direction
        incoming = zero3f;
        if (rand1f(rng) < 0.5f) {
          incoming = sample_scattering(vsdf, outgoing, rand1f(rng), rand2f(rng));
        } else {
          incoming = sample_lights(
              scene, position, rand1f(rng), rand1f(rng), rand2f(rng));
        }
        weight *= eval_scattering(vsdf, outgoing, incoming) /
                  (0.5f * sample_scattering_pdf(vsdf, outgoing, incoming) +
                      0.5f * sample_lights_pdf(scene, position, incoming));

      }
    }

		// check weight
		if (weight == zero3f || !isfinite(weight)) break;
		// russian roulette
		if (bounce > 3) {
			auto rr_prob = min((float)0.99, max(weight));
			if (rand1f(rng) >= rr_prob) break;
			weight *= 1 / rr_prob;
		} 

		// setup next iteration
		ray = {position, incoming};
	}
	return {radiance, hit ? 1.0f : 0.0f};
}