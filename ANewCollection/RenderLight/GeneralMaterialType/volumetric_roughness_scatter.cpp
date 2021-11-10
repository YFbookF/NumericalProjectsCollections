//https://github.com/sergeneren/Volumetric-Path-Tracer
	__device__ virtual bool scatter(float3& ray_pos, float3& ray_dir, float t_min, float3& normal, float3& atten, Rand_state rand_state) const {

		ray_dir = normalize(ray_dir);
		ray_pos += ray_dir * t_min;
		normal = normalize((ray_pos - center) / radius);
		float3 nl = dot(normal, ray_dir) < 0 ? normal : normal * -1;

		float phi = 2 * M_PI * rand(&rand_state);
		float r2 = rand(&rand_state);
		float r2s = sqrtf(r2);

		float3 w = normalize(nl);
		float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
		float3 v = cross(w, u);

		float3 hemisphere_dir = normalize(u * cosf(phi) * r2s + v * sinf(phi) * r2s + w * sqrtf(1 - r2));
		float3 ref = reflect(ray_dir, nl);
		ray_dir = lerp(ref, hemisphere_dir, roughness);

		ray_pos += ray_dir * 0.1;

		atten *= color;

		return true;
	}