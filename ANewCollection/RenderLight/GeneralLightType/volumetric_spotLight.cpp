//https://github.com/sergeneren/Volumetric-Path-Tracer
__device__ inline float henyey_greenstein(
	float cos_theta,
	float g)
{

	float denominator = 1 + g * g - 2 * g * cos_theta;

	return M_PI_4 * (1 - g * g) / (denominator * sqrtf(denominator));

}
__device__ inline float power_heuristic(int nf, float fPdf, int ng, float gPdf)
{
	float f = nf * fPdf, g = ng * gPdf;
	return (f*f) / (f*f + g * g);
}

class point_light : public light {

public:

	__host__ __device__ point_light(){}
	__host__ __device__ point_light(float3 p, float3 cl, float pow) {
		pos = p;
		color = cl;
		power = pow;
	}

	__device__ float3 Le(Rand_state &randstate, float3 ray_pos, float3 ray_dir, float phase_g1, float3 tr, float max_density, float density_mult, float tr_depth) const {

		float3 Ld = make_float3(.0f);
		float3 wi;
		float phase_pdf = .0f;
		float eq_pdf = .0f;
		
		
		// Sample point light with phase pdf  
		wi = normalize(pos - ray_pos);
		float cos_theta = dot(ray_dir, wi);
		phase_pdf = henyey_greenstein(cos_theta, phase_g1);
		float sqr_dist = length(pos * pos - ray_pos * ray_pos);
		float falloff = 1 / sqr_dist;

		float3 Li = color * power * tr  * phase_pdf * falloff;
		
		return Li;

		// Sample point light with equiangular pdf

		float delta = dot(pos - ray_pos, ray_dir);
		float D = length(ray_pos + ray_dir * delta - pos);

		float inv_max_density = 1.0f / max_density;
		float inv_density_mult = 1.0f / density_mult;

		float max_t = .0f;
		max_t -= logf(1 - rand(&randstate)) * inv_max_density * inv_density_mult * tr_depth;

		float thetaA = atan2f(.0f - delta, D);
		float thetaB = atan2f(max_t - delta, D);

		float t = D * tanf(lerp(thetaA, thetaB, rand(&randstate)));

		eq_pdf = D / ((thetaB - thetaA) * (D*D + t * t));
		float3 Leq = color * power * tr  * eq_pdf * falloff;

		float weight = power_heuristic(1, phase_pdf, 1, eq_pdf);

		Ld = (Li + Leq) * weight;
		
		return Ld;

	}

	__host__ __device__ int get_type() const { return POINT_LIGHT; }

};
