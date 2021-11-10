//https://github.com/sergeneren/Volumetric-Path-Tracer
__device__  float3 GetSkyRadiance(const AtmosphereParameters atmosphere, float3 camera, float3 view_ray, float shadow_length, float3 sun_direction, float3& transmittance)
{
	// Compute the distance to the top atmosphere boundary along the view ray,
	// assuming the viewer is in space (or NaN if the view ray does not intersect
	// the atmosphere).
	float r = length(camera);
	float rmu = dot(camera, view_ray);
	float distance_to_top_atmosphere_boundary = -rmu -
		sqrt(rmu * rmu - r * r + atmosphere.top_radius * atmosphere.top_radius);
	// If the viewer is in space and the view ray intersects the atmosphere, move
	// the viewer to the top atmosphere boundary (along the view ray):
	if (distance_to_top_atmosphere_boundary > 0.0 * m) {
		camera = camera + view_ray * distance_to_top_atmosphere_boundary;
		r = atmosphere.top_radius;
		rmu += distance_to_top_atmosphere_boundary;
	}
	else if (r > atmosphere.top_radius) {
		// If the view ray does not intersect the atmosphere, simply return 0.
		transmittance = make_float3(1.0f);
		return make_float3(0.0f * watt_per_square_meter_per_sr_per_nm());
	}
	// Compute the r, mu, mu_s and nu parameters needed for the texture lookups.
	float mu = rmu / r;
	float mu_s = dot(camera, sun_direction) / r;
	float nu = dot(view_ray, sun_direction);
	bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);

	transmittance = ray_r_mu_intersects_ground ? make_float3(0.0f) : GetTransmittanceToTopAtmosphereBoundary(atmosphere, r, mu);
	float3 single_mie_scattering;
	float3 scattering;
	if (shadow_length == 0.0 * m) {
		scattering = GetCombinedScattering(atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground, single_mie_scattering);
	}
	else {
		// Case of light shafts (shadow_length is the total float noted l in our
		// paper): we omit the scattering between the camera and the point at
		// distance l, by implementing Eq. (18) of the paper (shadow_transmittance
		// is the T(x,x_s) term, scattering is the S|x_s=x+lv term).
		float d = shadow_length;
		float r_p = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
		float mu_p = (r * mu + d) / r_p;
		float mu_s_p = (r * mu_s + d * nu) / r_p;

		scattering = GetCombinedScattering(atmosphere, r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground, single_mie_scattering);
		float3 shadow_transmittance = GetTransmittance(atmosphere, r, mu, shadow_length, ray_r_mu_intersects_ground);
		scattering = scattering * shadow_transmittance;
		single_mie_scattering = single_mie_scattering * shadow_transmittance;
	}

	float3 sky_radiance = scattering * RayleighPhaseFunction(nu) + single_mie_scattering * MiePhaseFunction(atmosphere.mie_phase_function_g, nu);

	if (atmosphere.use_luminance != 0) sky_radiance *= atmosphere.sky_spectral_radiance_to_luminance;
	return sky_radiance;
}

__device__  float3 GetSkyRadianceToPoint(const AtmosphereParameters atmosphere, float3 camera, float3 point, float shadow_length, float3 sun_direction, float3& transmittance)
{
	// Compute the distance to the top atmosphere boundary along the view ray,
	// assuming the viewer is in space (or NaN if the view ray does not intersect
	// the atmosphere).
	float3 view_ray = normalize(point - camera);
	float r = length(camera);
	float rmu = dot(camera, view_ray);
	float distance_to_top_atmosphere_boundary = -rmu - sqrt(rmu * rmu - r * r + atmosphere.top_radius * atmosphere.top_radius);
	// If the viewer is in space and the view ray intersects the atmosphere, move
	// the viewer to the top atmosphere boundary (along the view ray):
	if (distance_to_top_atmosphere_boundary > 0.0 * m) {
		camera = camera + view_ray * distance_to_top_atmosphere_boundary;
		r = atmosphere.top_radius;
		rmu += distance_to_top_atmosphere_boundary;
	}

	// Compute the r, mu, mu_s and nu parameters for the first texture lookup.
	float mu = rmu / r;
	float mu_s = dot(camera, sun_direction) / r;
	float nu = dot(view_ray, sun_direction);
	float d = length(point - camera);
	bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);

	transmittance = GetTransmittance(atmosphere, r, mu, d, ray_r_mu_intersects_ground);

	float3 single_mie_scattering;
	float3 scattering = GetCombinedScattering(atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground, single_mie_scattering);

	// Compute the r, mu, mu_s and nu parameters for the second texture lookup.
	// If shadow_length is not 0 (case of light shafts), we want to ignore the
	// scattering along the last shadow_length meters of the view ray, which we
	// do by subtracting shadow_length from d (this way scattering_p is equal to
	// the S|x_s=x_0-lv term in Eq. (17) of our paper).
	d = max(d - shadow_length, 0.0 * m);
	float r_p = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
	float mu_p = (r * mu + d) / r_p;
	float mu_s_p = (r * mu_s + d * nu) / r_p;

	float3 single_mie_scattering_p;
	float3 scattering_p = GetCombinedScattering(atmosphere, r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground, single_mie_scattering_p);

	// Combine the lookup results to get the scattering between camera and point.
	float3 shadow_transmittance = transmittance;
	if (shadow_length > 0.0 * m) {
		// This is the T(x,x_s) term in Eq. (17) of our paper, for light shafts.
		shadow_transmittance = GetTransmittance(atmosphere, r, mu, d, ray_r_mu_intersects_ground);
	}
	scattering = scattering - shadow_transmittance * scattering_p;
	single_mie_scattering =
		single_mie_scattering - shadow_transmittance * single_mie_scattering_p;
#ifdef COMBINED_SCATTERING_TEXTURES
	single_mie_scattering = GetExtrapolatedSingleMieScattering(atmosphere, make_float4(scattering, single_mie_scattering.x));
#endif

	// Hack to avoid rendering artifacts when the sun is below the horizon.
	single_mie_scattering = single_mie_scattering * smoothstep(float(0.0), float(0.01), mu_s);

	float3 sky_radiance = scattering * RayleighPhaseFunction(nu) + single_mie_scattering * MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
	if (atmosphere.use_luminance != 0) sky_radiance *= atmosphere.sky_spectral_radiance_to_luminance;
	return sky_radiance;
}

__device__  float3 GetSunAndSkyIrradiance(const AtmosphereParameters atmosphere, float3 point, float3 normal, float3 sun_direction, float3& sky_irradiance)
{
	float r = length(point);
	float mu_s = dot(point, sun_direction) / r;

	// Indirect irradiance (approximated if the surface is not horizontal).
	sky_irradiance = GetIrradiance(atmosphere, r, mu_s) * (1.0 + dot(normal, point) / r) * 0.5;
	float3 sun_irradiance = atmosphere.solar_irradiance * GetTransmittanceToSun(atmosphere, r, mu_s) * max(dot(normal, sun_direction), 0.0);

	if (atmosphere.use_luminance != 0) {

		sky_irradiance *= atmosphere.sky_spectral_radiance_to_luminance;
		sun_irradiance *= atmosphere.sun_spectral_radiance_to_luminance;
	}
	// Direct irradiance.
	return sun_irradiance;
}

__device__ float3 GetSolarRadiance(const AtmosphereParameters atmosphere) {

	float3 solar_radiance = atmosphere.solar_irradiance / (M_PI * atmosphere.sun_angular_radius * atmosphere.sun_angular_radius);
	if (atmosphere.use_luminance != 0) solar_radiance *= atmosphere.sun_spectral_radiance_to_luminance;
	return solar_radiance;
}

// Light Samplers 

__device__ inline float3 sample_atmosphere(
	const Kernel_params& kernel_params,
	const AtmosphereParameters& atmosphere,
	const float3 ray_pos, const float3 ray_dir)
{

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

	float3 exposure = atmosphere.use_luminance == 0 ? make_float3(atmosphere.exposure) : make_float3(atmosphere.exposure) * 1e-5;

	ground_radiance = powf(make_float3(1.0f) - expf(-ground_radiance / atmosphere.white_point * exposure), make_float3(1.0 / 2.2));

	return ground_radiance;

	/* old sky
	float azimuth = atan2f(-ray_dir.z, -ray_dir.x) * INV_2_PI + 0.5f;
	float elevation = acosf(fmaxf(fminf(ray_dir.y, 1.0f), -1.0f)) * INV_PI;
	const float4 texval = tex2D<float4>( kernel_params.sky_tex, azimuth, elevation);
	return make_float3(texval.x, texval.y, texval.z);
	*/
}
__device__ inline float3 estimate_sun(
	Kernel_params kernel_params,
	Rand_state& randstate,
	const float3& ray_pos,
	float3& ray_dir,
	const GPU_VDB* gpu_vdb,
	const sphere& ref_sphere,
	OCTNode* root,
	const AtmosphereParameters atmosphere)
{
	float3 Ld = BLACK;
	float3 wi;
	float phase_pdf = .0f;

	// sample sun light with multiple importance sampling

	//Find sun direction 
	wi = degree_to_cartesian(kernel_params.azimuth, kernel_params.elevation);

	// find scattering pdf
	float cos_theta = dot(ray_dir, wi);
	phase_pdf = henyey_greenstein(cos_theta, kernel_params.phase_g1);

	// Check visibility of light source 
	float3 tr = Tr(randstate, ray_pos, wi, kernel_params, gpu_vdb, ref_sphere, root);

	float3 sky_irradiance;
	float3 sun_irradiance = GetSunAndSkyIrradiance(atmosphere, ray_pos, ray_dir, wi, sky_irradiance);

	// Ld = Li * visibility.Tr * scattering_pdf / light_pdf  
	//Ld = (length(sky_irradiance)*sun_irradiance) * tr  * phase_pdf;
	Ld = tr * phase_pdf;

	// No need for sampling BSDF with importance sampling
	// please see: http://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Direct_Lighting.html#fragment-SampleBSDFwithmultipleimportancesampling-0

	return Ld * kernel_params.sun_color * kernel_params.sun_mult;

}
__device__ inline float3 estimate_sky(
	Kernel_params kernel_params,
	Rand_state& randstate,
	const float3& ray_pos,
	float3& ray_dir,
	const GPU_VDB* gpu_vdb,
	const sphere ref_sphere,
	OCTNode* root,
	const AtmosphereParameters atmosphere)
{
	float3 Ld = BLACK;

	for (int i = 0; i < 1; i++) {

		float3 Li = BLACK;
		float3 wi;

		float light_pdf = .0f, phase_pdf = .0f;

		float az = rand(&randstate) * 360.0f;
		float el = rand(&randstate) * 180.0f;

		// Sample light source with multiple importance sampling 

		if (kernel_params.environment_type == 0) {
			light_pdf = draw_sample_from_distribution(kernel_params, randstate, wi);
			Li = sample_atmosphere(kernel_params, atmosphere, ray_pos, wi);
		}
		else {
			light_pdf = sample_spherical(randstate, wi);
			Li = sample_env_tex(kernel_params, wi);
		}

		if (light_pdf > .0f && !isBlack(Li)) {

			float cos_theta = dot(ray_dir, wi);
			phase_pdf = henyey_greenstein(cos_theta, kernel_params.phase_g1);

			if (phase_pdf > .0f) {
				float3 tr = Tr(randstate, ray_pos, wi, kernel_params, gpu_vdb, ref_sphere, root);
				Li *= tr;

				if (!isBlack(Li)) {

					float weight = power_heuristic(1, light_pdf, 1, phase_pdf);
					Ld += Li * phase_pdf * weight / light_pdf;
				}

			}

		}


		// Sample BSDF with multiple importance sampling 
		wi = ray_dir;
		phase_pdf = sample_hg(wi, randstate, kernel_params.phase_g1);
		float3 f = make_float3(phase_pdf);
		if (phase_pdf > .0f) {
			Li = BLACK;
			float weight = 1.0f;
			if (kernel_params.environment_type == 0)
			{
				light_pdf = pdf_li(kernel_params, wi);
			}
			else light_pdf = isotropic();

			if (light_pdf == 0.0f) return Ld;
			weight = power_heuristic(1, phase_pdf, 1, light_pdf);

			float3 tr = Tr(randstate, ray_pos, wi, kernel_params, gpu_vdb, ref_sphere, root);

			if (kernel_params.environment_type == 0)
			{
				Li = sample_atmosphere(kernel_params, atmosphere, ray_pos, wi);
			}
			else Li = sample_env_tex(kernel_params, wi);


			if (!isBlack(Li))
				Ld += Li * tr * weight;
		}


	}

	return Ld;

}