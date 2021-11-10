//https://github.com/sergeneren/Volumetric-Path-Tracer
__device__ inline float henyey_greenstein(
	float cos_theta,
	float g)
{

	float denominator = 1 + g * g - 2 * g * cos_theta;

	return M_PI_4 * (1 - g * g) / (denominator * sqrtf(denominator));

}
//kernel.cu
__device__ inline float sample_hg(
	float3& wo,
	Rand_state& randstate,
	float g)
{

	float cos_theta;

	if (fabsf(g) < EPS) cos_theta = 1 - 2 * rand(&randstate);
	else {
		float sqr_term = (1 - g * g) / (1 - g + 2 * g * rand(&randstate));
		cos_theta = (1 + g * g - sqr_term * sqr_term) / (2 * g);
	}
	float sin_theta = sqrtf(fmaxf(.0f, 1.0f - cos_theta * cos_theta));
	float phi = (float)(2.0 * M_PI) * rand(&randstate);
	float3 v1, v2;
	coordinate_system(wo * -1.0f, v1, v2);
	wo = spherical_direction(sin_theta, cos_theta, phi, v1, v2, wo);
	return henyey_greenstein(-cos_theta, g);
}


__device__ inline float sample_double_hg(
	float3& wi,
	Rand_state randstate,
	float f,
	float g1,
	float g2)
{
	wi *= -1.0f;
	float3 v1 = wi, v2 = wi;
	float cos_theta1, cos_theta2;


	if (f > 0.9999f) {

		cos_theta1 = sample_hg(v1, randstate, g1);
		wi = v1;
		return henyey_greenstein(cos_theta1, g1);
	}
	else if (f < EPS)
	{
		cos_theta2 = sample_hg(v2, randstate, g2);
		wi = v2;
		return henyey_greenstein(cos_theta2, g2);
	}
	else {

		cos_theta1 = sample_hg(v1, randstate, g1);
		cos_theta2 = sample_hg(v2, randstate, g2);

		wi = lerp(v1, v2, 1 - f);
		float cos_theta = lerp(cos_theta1, cos_theta2, 1 - f);
		return double_henyey_greenstein(cos_theta, f, g1, g2);
	}

}
