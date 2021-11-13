//https://github.com/henrikdahlberg/GPUPathTracer
/**
* Longest common prefix for Morton code
*/
__device__ int LongestCommonPrefix(int i, int j, int numTriangles,
								   MortonCode* mortonCodes, int* triangleIDs) {
	if (i < 0 || i > numTriangles - 1 || j < 0 || j > numTriangles - 1) {
		return -1;
	}

	MortonCode mi = mortonCodes[i];
	MortonCode mj = mortonCodes[j];

	if (mi == mj) {
		return __clzll(mi ^ mj) + __clzll(triangleIDs[i] ^ triangleIDs[j]);
	}
	else {
		return __clzll(mi ^ mj);
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
* Compute morton code given triangle centroid scaled to [0,1] of scene bounding box
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