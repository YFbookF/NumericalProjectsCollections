//https://github.com/sergeneren/Volumetric-Path-Tracer
__device__ inline float3 render_earth(float3 ray_pos, float3 ray_dir, const Kernel_params kernel_params, const AtmosphereParameters atmosphere) {

	float3 earth_center = make_float3(.0f, -atmosphere.bottom_radius, .0f);
	float3 sun_direction = degree_to_cartesian(kernel_params.azimuth, kernel_params.elevation);

	float3 p = ray_pos - earth_center;
	float p_dot_v = dot(p, ray_dir);
	float p_dot_p = dot(p, p);
	float ray_earth_center_squared_distance = p_dot_p - p_dot_v * p_dot_v;
	float distance_to_intersection = -p_dot_v - sqrt(earth_center.y * earth_center.y - ray_earth_center_squared_distance);

	float ground_alpha = 0.0;
	float3 ground_radiance = make_float3(0.0);

	if (distance_to_intersection > 0.0) {
		float3 point = ray_pos + ray_dir * distance_to_intersection;
		float3 normal = normalize(point - earth_center);

		// Compute the radiance reflected by the ground.
		float3 sky_irradiance;
		float3 sun_irradiance = GetSunAndSkyIrradiance(atmosphere, point - earth_center, normal, sun_direction, sky_irradiance);
		ground_radiance = atmosphere.ground_albedo * (1.0 / M_PI) * (sun_irradiance + sky_irradiance);

		float3 transmittance;
		float3 in_scatter = GetSkyRadianceToPoint(atmosphere, ray_pos - earth_center, point - earth_center, .0f, sun_direction, transmittance);
		ground_radiance = ground_radiance * transmittance + in_scatter;
		ground_alpha = 1.0;
	}

	float3 transmittance_sky;
	float3 radiance_sky = GetSkyRadiance(atmosphere, ray_pos - earth_center, ray_dir, .0f, sun_direction, transmittance_sky);

	float2 sun_size = make_float2(tanf(atmosphere.sun_angular_radius), cosf(atmosphere.sun_angular_radius));

	if (dot(ray_dir, sun_direction) > sun_size.y) {
		radiance_sky = radiance_sky + transmittance_sky * GetSolarRadiance(atmosphere);
	}

	ground_radiance = lerp(radiance_sky, ground_radiance, ground_alpha);

	return ground_radiance;

}
