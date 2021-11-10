//https://github.com/sergeneren/Volumetric-Path-Tracer
// Environment light samplers

__device__ inline float draw_sample_from_distribution(
	Kernel_params kernel_params,
	Rand_state rand_state,
	float3& wo) {

	float xi = rand(&rand_state);
	float zeta = rand(&rand_state);

	float pdf = 1.0f;
	int v = 0;
	int res = kernel_params.env_sample_tex_res;

	// Find marginal row number

	// Find interval

	int first = 0, len = res;

	while (len > 0) {

		int half = len >> 1, middle = first + half;

		if (tex_lookup_1d(kernel_params.env_marginal_cdf_tex, middle) <= xi) {
			first = middle + 1;
			len -= half + 1;
		}
		else len = half;

	}
	v = clamp(first - 1, 0, res - 2);



	float dv = xi - tex_lookup_1d(kernel_params.env_marginal_cdf_tex, v);
	float d_cdf_marginal = tex_lookup_1d(kernel_params.env_marginal_cdf_tex, v + 1) - tex_lookup_1d(kernel_params.env_marginal_cdf_tex, v);
	if (d_cdf_marginal > .0f) dv /= d_cdf_marginal;

	// Calculate marginal pdf
	float marginal_pdf = tex_lookup_1d(kernel_params.env_marginal_func_tex, v + dv) / kernel_params.env_marginal_int;

	// calculate Φ (elevation)
	float theta = ((float(v) + dv) / float(res)) * M_PI;

	// v is now our row number. find the conditional value and pdf from v

	int u;
	first = 0, len = res;
	while (len > 0) {

		int half = len >> 1, middle = first + half;

		if (tex_lookup_2d(kernel_params.env_cdf_tex, middle, v) <= zeta) {
			first = middle + 1;
			len -= half + 1;
		}
		else len = half;

	}
	u = clamp(first - 1, 0, res - 2);

	float du = zeta - tex_lookup_2d(kernel_params.env_cdf_tex, u, v);

	float d_cdf_conditional = tex_lookup_2d(kernel_params.env_cdf_tex, u + 1, v) - tex_lookup_2d(kernel_params.env_cdf_tex, u, v);
	if (d_cdf_conditional > 0) du /= d_cdf_conditional;

	//Calculate conditional pdf
	float conditional_pdf = tex_lookup_2d(kernel_params.env_func_tex, u + du, v) / tex_lookup_1d(kernel_params.env_marginal_func_tex, v);

	// Find the θ (azimuth)
	float phi = ((float(u) + du) / float(res)) * M_PI * 2.0f;



	float cos_theta = cosf(theta);
	float sin_theta = sinf(theta);
	float sin_phi = sinf(phi);
	float cos_phi = cosf(phi);

	float3 sundir = normalize(make_float3(sinf(kernel_params.azimuth) * cosf(kernel_params.elevation),
		sinf(kernel_params.azimuth) * sinf(kernel_params.elevation), cosf(kernel_params.azimuth)));

	wo = normalize(make_float3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta));
	pdf = (marginal_pdf * conditional_pdf) / (2 * M_PI * M_PI * sin_theta);
	//if (kernel_params.debug) printf("\n%f	%f	%f	%d	%d", ((float(u) + du) / float(res)), ((float(v) + dv) / float(res)), pdf, u, v);
	//if (kernel_params.debug) printf("\n%f	%f	%f	%f", wo.x, wo.y,wo.z, dot(wo, sundir));
	return pdf;
}