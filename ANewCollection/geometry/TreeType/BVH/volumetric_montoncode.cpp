//https://github.com/sergeneren/Volumetric-Path-Tracer
__global__ void ComputeMortonCodes(const GPU_VDB* volumes, int numTriangles, AABB sceneBounds, MortonCode* mortonCodes) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	
	if (i < numTriangles) {

		// Compute volume centroid
		float3 centroid = volumes[i].Bounds().Centroid();

		// Normalize triangle centroid to lie within [0,1] of scene bounding box
		float x = (centroid.x - sceneBounds.pmin.x) / (sceneBounds.pmax.x - sceneBounds.pmin.x);
		float y = (centroid.y - sceneBounds.pmin.y) / (sceneBounds.pmax.y - sceneBounds.pmin.y);
		float z = (centroid.z - sceneBounds.pmin.z) / (sceneBounds.pmax.z - sceneBounds.pmin.z);

		// Compute morton code
		mortonCodes[i] = ComputeMortonCode(x, y, z);
	}
	
}
/**
* Expand bits, used in Morton code calculation
*/
__device__ MortonCode bitExpansion(MortonCode i) {
	i = (i * 0x00010001u) & 0xFF0000FFu;
	i = (i * 0x00000101u) & 0x0F00F00Fu;
	i = (i * 0x00000011u) & 0xC30C30C3u;
	i = (i * 0x00000005u) & 0x49249249u;
	return i;
}

/**
* Compute morton code given volume centroid scaled to [0,1] of scene bounding box
*/
__device__ MortonCode ComputeMortonCode(float x, float y, float z) {

	x = min(max(x * 1024.0f, 0.0f), 1023.0f);
	y = min(max(y * 1024.0f, 0.0f), 1023.0f);
	z = min(max(z * 1024.0f, 0.0f), 1023.0f);
	MortonCode xx = bitExpansion((MortonCode)x);
	MortonCode yy = bitExpansion((MortonCode)y);
	MortonCode zz = bitExpansion((MortonCode)z);
	return xx * 4 + yy * 2 + zz;

}