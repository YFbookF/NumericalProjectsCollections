//https://github.com/peterkutz/GPUPathTracer
__host__ __device__
	Color
	computeTransmission(Color absorptionCoefficient, float distance)
{
	Color transmitted;
	transmitted.x = pow((float)E, (float)(-1 * absorptionCoefficient.x * distance));
	transmitted.y = pow((float)E, (float)(-1 * absorptionCoefficient.y * distance));
	transmitted.z = pow((float)E, (float)(-1 * absorptionCoefficient.z * distance));
	return transmitted;
}
