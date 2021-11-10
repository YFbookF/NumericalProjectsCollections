//https://github.com/sergeneren/Volumetric-Path-Tracer
//////////////////////////////////////////////////////////////////////////
// Rendering Integrators 
//////////////////////////////////////////////////////////////////////////

// PBRT Volume Integrator
__device__ inline float3 vol_integrator(
	Rand_state rand_state,
	const light_list lights,
	float3 ray_pos,
	float3 ray_dir,
	float& tr,
	const Kernel_params kernel_params,
	const GPU_VDB* gpu_vdb,
	const sphere ref_sphere,
	OCTNode* root,
	const AtmosphereParameters atmosphere)
{

	float3 L = BLACK;
	float3 beta = WHITE;
	float3 env_pos = ray_pos;
	bool mi;
	float t, tmax;
	int obj;

	if (root->bbox.Intersect(ray_pos, ray_dir, t, tmax)) { // found an intersection
		ray_pos += ray_dir * (t + EPS);
		for (int depth = 1; depth <= kernel_params.ray_depth; depth++) {
			mi = false;

			beta *= sample(rand_state, ray_pos, ray_dir, mi, obj, tr, kernel_params, gpu_vdb, ref_sphere, root);
			if (isBlack(beta)) break;

			if (mi) { // medium interaction 
				L += beta * uniform_sample_one_light(kernel_params, lights, ray_pos, ray_dir, rand_state, gpu_vdb, ref_sphere, root, atmosphere) + estimate_emission(rand_state, ray_pos, ray_dir, kernel_params, gpu_vdb, root);
				sample_hg(ray_dir, rand_state, kernel_params.phase_g1);
			}

		}

		ray_dir = normalize(ray_dir);
	}

	if (length(beta) > 0.9999f) ray_pos = env_pos;

	L += beta * sample_atmosphere(kernel_params, atmosphere, ray_pos, ray_dir);

	tr = fminf(tr, 1.0f);
	return L;
}
// From Ray Tracing Gems Vol-28
__device__ inline float3 direct_integrator(
	Rand_state rand_state,
	float3 ray_pos,
	float3 ray_dir,
	float& tr,
	const Kernel_params kernel_params,
	const GPU_VDB* gpu_vdb,
	const light_list lights,
	const sphere& ref_sphere,
	OCTNode* root,
	const AtmosphereParameters atmosphere)
{
	float3 L = BLACK;
	float3 beta = WHITE;
	bool mi = false;
	float3 env_pos = ray_pos;
	float t_min;
	int obj;
	// TODO use bvh to determine if we intersect volume or geometry

	for (int ray_depth = 1; ray_depth <= kernel_params.ray_depth; ray_depth++) {

		obj = get_closest_object(ray_pos, ray_dir, root, ref_sphere, t_min);

		if (obj == 1) {
			ray_pos += ray_dir * (t_min + EPS);
			for (int volume_depth = 1; volume_depth <= kernel_params.volume_depth; volume_depth++) {
				mi = false;

				beta *= sample(rand_state, ray_pos, ray_dir, mi, obj, tr, kernel_params, gpu_vdb, ref_sphere, root);
				if (isBlack(beta) || obj == 2) break;

				if (mi) { // medium interaction 
					sample_hg(ray_dir, rand_state, kernel_params.phase_g1);
				}

			}
			if (mi) {
				L += estimate_sun(kernel_params, rand_state, ray_pos, ray_dir, gpu_vdb, ref_sphere, root, atmosphere) * beta;
				if(lights.num_lights>0) L += estimate_point_light(kernel_params, lights, rand_state, ray_pos, ray_dir, gpu_vdb, ref_sphere, root) * beta;
			}

			if (kernel_params.emission_scale > 0 && mi) {
				L += estimate_emission(rand_state, ray_pos, ray_dir, kernel_params, gpu_vdb, root);
			}
		}
		obj = get_closest_object(ray_pos, ray_dir, root, ref_sphere, t_min);
		if (obj == 2) {

			ray_pos += ray_dir * t_min;
			float3 normal = normalize((ray_pos - ref_sphere.center) / ref_sphere.radius);
			float3 nl = dot(normal, ray_dir) < 0 ? normal : normal * -1;

			float phi = 2 * M_PI * rand(&rand_state);
			float r2 = rand(&rand_state);
			float r2s = sqrtf(r2);

			float3 w = normalize(nl);
			float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
			float3 v = cross(w, u);

			float3 hemisphere_dir = normalize(u * cosf(phi) * r2s + v * sinf(phi) * r2s + w * sqrtf(1 - r2));
			float3 ref = reflect(ray_dir, nl);
			ray_dir = lerp(ref, hemisphere_dir, ref_sphere.roughness);

			float3 light_dir = degree_to_cartesian(kernel_params.azimuth, kernel_params.elevation);

			ray_pos += normal * EPS;

			beta *= ref_sphere.color;

			float3 v_tr = Tr(rand_state, ray_pos, light_dir, kernel_params, gpu_vdb, ref_sphere, root);
			L += kernel_params.sun_color * kernel_params.sun_mult * v_tr * fmaxf(dot(light_dir, normal), .0f) * beta;
			env_pos = ray_pos;
		}

	}

	if (kernel_params.environment_type == 0) {

		L += sample_atmosphere(kernel_params, atmosphere, env_pos, ray_dir) * beta * kernel_params.sky_mult * kernel_params.sky_color;

	}
	else {

		const float4 texval = tex2D<float4>(
			kernel_params.env_tex,
			atan2f(ray_dir.z, ray_dir.x) * (float)(0.5 / M_PI) + 0.5f,
			acosf(fmaxf(fminf(ray_dir.y, 1.0f), -1.0f)) * (float)(1.0 / M_PI));
		L += make_float3(texval.x, texval.y, texval.z) * kernel_params.sky_color * beta * isotropic();
	}



	tr = fminf(tr, 1.0f);
	return L;

}