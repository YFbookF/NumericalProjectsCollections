
//https://github.com/sergeneren/Volumetric-Path-Tracer
//Phase functions pdf 
__device__ inline float draw_pdf_from_distribution(Kernel_params kernel_params, float2 point)
{
	int res = kernel_params.env_sample_tex_res;

	int iu = clamp(int(point.x * res), 0, res - 1);
	int iv = clamp(int(point.y * res), 0, res - 1);

	float conditional = tex_lookup_2d(kernel_params.env_func_tex, iu, iv);
	float marginal = tex_lookup_1d(kernel_params.env_marginal_func_tex, iv);

	return conditional / marginal;
}