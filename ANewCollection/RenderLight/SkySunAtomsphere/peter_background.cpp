//https://github.com/peterkutz/GPUPathTracer
__host__ __device__
	Color
	computeBackgroundColor(const float3 &direction)
{
	float position = (dot(direction, normalize(make_float3(-0.5, 0.5, -1.0))) + 1) / 2;
	Color firstColor = make_float3(0.15, 0.3, 0.5); // Bluish.
	Color secondColor = make_float3(1.0, 1.0, 1.0); // White.
	Color interpolatedColor = (1 - position) * firstColor + position * secondColor;
	float radianceMultiplier = 1.0;
	return interpolatedColor * radianceMultiplier;
}
