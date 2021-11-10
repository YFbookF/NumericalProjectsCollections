//https://github.com/sergeneren/Volumetric-Path-Tracer
__global__ void ConstructBVH(BVHNode* BVHNodes, BVHNode* BVHLeaves, int* nodeCounter, GPU_VDB* volumes, int* volumeIDs, int numVolumes) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < numVolumes) {
		BVHNode* leaf = BVHLeaves + i;

		int volumeIdx = volumeIDs[i];
		// Handle leaf first
		leaf->volIndex = volumeIdx;
		leaf->boundingBox = volumes[volumeIdx].Bounds();

		BVHNode* current = leaf->parent;
		int currentIndex = current - BVHNodes;

		int res = atomicAdd(nodeCounter + currentIndex, 1);

		// Go up and handle internal nodes
		while (true) {
			if (res == 0) {
				return;
			}
			AABB leftBoundingBox = current->leftChild->boundingBox;
			AABB rightBoundingBox = current->rightChild->boundingBox;

			// Compute current bounding box
			current->boundingBox = UnionB(leftBoundingBox, rightBoundingBox);

			// If current is root, return
			if (current == BVHNodes) {
				return;
			}
			current = current->parent;
			currentIndex = current - BVHNodes;
			res = atomicAdd(nodeCounter + currentIndex, 1);
		}
	}
}