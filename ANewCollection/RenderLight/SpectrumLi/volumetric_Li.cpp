//https://github.com/sergeneren/Volumetric-Path-Tracer
__device__ inline float pdf_li(
	Kernel_params kernel_params,
	float3 wi)
{
	float theta = acosf(clamp(wi.y, -1.0f, 1.0f));
	float phi = atan2f(wi.z, wi.x);
	float sin_theta = sinf(theta);

	if (sin_theta == .0f) return .0f;
	float2 polar_pos = make_float2(phi * INV_2_PI, theta * INV_PI) / (2.0f * M_PI * M_PI * sin_theta);
	return draw_pdf_from_distribution(kernel_params, polar_pos);

}