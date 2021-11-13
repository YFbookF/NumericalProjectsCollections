//https://github.com/peterkutz/GPUPathTracer
__host__ __device__
	float3
	computeReflectionDirection(const float3 &normal, const float3 &incident)
{
	return 2.0 * dot(normal, incident) * normal - incident;
}

__host__ __device__
	float3
	computeTransmissionDirection(const float3 &normal, const float3 &incident, float refractiveIndexIncident, float refractiveIndexTransmitted)
{
	// Snell's Law:
	// Copied from Photorealizer.

	float cosTheta1 = dot(normal, incident);

	float n1_n2 = refractiveIndexIncident / refractiveIndexTransmitted;

	float radicand = 1 - pow(n1_n2, 2) * (1 - pow(cosTheta1, 2));
	if (radicand < 0)
		return make_float3(0, 0, 0); // Return value???????????????????????????????????????
	float cosTheta2 = sqrt(radicand);

	if (cosTheta1 > 0)
	{ // normal and incident are on same side of the surface.
		return n1_n2 * (-1 * incident) + (n1_n2 * cosTheta1 - cosTheta2) * normal;
	}
	else
	{ // normal and incident are on opposite sides of the surface.
		return n1_n2 * (-1 * incident) + (n1_n2 * cosTheta1 + cosTheta2) * normal;
	}
}
