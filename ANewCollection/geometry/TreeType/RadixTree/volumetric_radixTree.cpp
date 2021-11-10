//https://github.com/sergeneren/Volumetric-Path-Tracer
__global__ void BuildRadixTree(BVHNode* radixTreeNodes, BVHNode* radixTreeLeaves, MortonCode* mortonCodes, int* volumeIds, int numVolumes) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < numVolumes - 1) {
		// Run radix tree construction algorithm
		// Determine direction of the range (+1 or -1)
		int d = LongestCommonPrefix(i, i + 1, numVolumes, mortonCodes, volumeIds) -
			LongestCommonPrefix(i, i - 1, numVolumes, mortonCodes, volumeIds) >= 0 ? 1 : -1;

		// Compute upper bound for the length of the range
		int deltaMin = LongestCommonPrefix(i, i - d, numVolumes, mortonCodes, volumeIds);
		//int lmax = 128;
		int lmax = 2;

		while (LongestCommonPrefix(i, i + lmax * d, numVolumes, mortonCodes, volumeIds) > deltaMin) {
			//lmax = lmax * 4;
			lmax = lmax * 2;
		}

		// Find the other end using binary search
		int l = 0;
		int divider = 2;
		for (int t = lmax / divider; t >= 1; divider *= 2) {
			if (LongestCommonPrefix(i, i + (l + t) * d, numVolumes, mortonCodes, volumeIds) > deltaMin) {
				l = l + t;
			}
			if (t == 1) break;
			t = lmax / divider;
		}

		int j = i + l * d;

		// Find the split position using binary search
		int deltaNode = LongestCommonPrefix(i, j, numVolumes, mortonCodes, volumeIds);
		int s = 0;
		divider = 2;
		for (int t = (l + (divider - 1)) / divider; t >= 1; divider *= 2) {
			if (LongestCommonPrefix(i, i + (s + t) * d, numVolumes, mortonCodes, volumeIds) > deltaNode) {
				s = s + t;
			}
			if (t == 1) break;
			t = (l + (divider - 1)) / divider;
		}

		int gamma = i + s * d + min(d, 0);

		//printf("i:%d, d:%d, deltaMin:%d, deltaNode:%d, lmax:%d, l:%d, j:%d, gamma:%d. \n", i, d, deltaMin, deltaNode, lmax, l, j, gamma);

		// Output child pointers
		BVHNode* current = radixTreeNodes + i;

		if (min(i, j) == gamma) {
			current->leftChild = radixTreeLeaves + gamma;
			(radixTreeLeaves + gamma)->parent = current;
		}
		else {
			current->leftChild = radixTreeNodes + gamma;
			(radixTreeNodes + gamma)->parent = current;
		}

		if (max(i, j) == gamma + 1) {
			current->rightChild = radixTreeLeaves + gamma + 1;
			(radixTreeLeaves + gamma + 1)->parent = current;
		}
		else {
			current->rightChild = radixTreeNodes + gamma + 1;
			(radixTreeNodes + gamma + 1)->parent = current;
		}

		current->minId = min(i, j);
		current->maxId = max(i, j);
	}
}
