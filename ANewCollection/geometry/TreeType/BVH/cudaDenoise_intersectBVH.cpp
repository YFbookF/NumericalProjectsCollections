//////////https://github.com/ZheyuanXie/Project3-CUDA-Path-Tracer
#define MAX_BVH_DEPTH 64
__host__ __device__ bool IntersectBVH(const Ray &ray, ShadeableIntersection * isect,
	int & hit_tri_index,
	const BVH_ArrNode *bvh_nodes,
	const Triangle* primitives) {

	if (bvh_nodes == nullptr) { return false; }

	bool hit = false;
	int isDirNeg[3] = { ray.direction.x < 0.f, ray.direction.y < 0.f, ray.direction.z < 0.f };
	glm::vec3 invdir(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z);

	int toVisitOffset = 0, curr_ind = 0;
	int needToVisit[MAX_BVH_DEPTH];
	while (true) {
		const BVH_ArrNode *node = &bvh_nodes[curr_ind];
		float temp_t = 0.f;
		//check bounding box intersection
		if (node->bounds.AABBIntersect2(ray, invdir)) {
			if (node->primitive_count > 0) {												// leaf node
				for (int i = 0; i < node->primitive_count; i++) {							// intersect test with each triangle in the nodes
					ShadeableIntersection inter;
					if (primitives[node->primitivesOffset + i].Intersect(ray, &inter)) {	// triangles intersect test
						hit = true;
						if (isect->t == -1.0f) {
							(*isect) = inter;
							hit_tri_index = primitives[node->primitivesOffset + i].id;
						}
						else {
							if (inter.t < isect->t) {
								(*isect) = inter;
								hit_tri_index = primitives[node->primitivesOffset + i].id;
							}
						}
					}
				}
				if (toVisitOffset == 0) { break; }
				curr_ind = needToVisit[--toVisitOffset];
			}
			else {
				// Trick: learn from hanming zhang, if toVisitOffset reaches maximum
				// we don't want add more index to needToVisit Array
				// we just give up this interior node and handle previous nodes instead 
				if (toVisitOffset == MAX_BVH_DEPTH) {
					curr_ind = needToVisit[--toVisitOffset];
					continue;
				}
				// add index to nodes to visit
				if (isDirNeg[node->axis]) {
					needToVisit[toVisitOffset++] = curr_ind + 1;
					curr_ind = node->rightchildoffset;
				}
				else {
					needToVisit[toVisitOffset++] = node->rightchildoffset;
					curr_ind = curr_ind + 1;
				}
			}
		}
		else {// do not hit anything
			if (toVisitOffset == 0) { break; }
			curr_ind = needToVisit[--toVisitOffset];
		}
	}
	return hit;
}
