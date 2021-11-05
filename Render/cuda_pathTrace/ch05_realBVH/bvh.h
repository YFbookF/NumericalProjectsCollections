#ifndef __BVH_H_
#define __BVH_H_
#include <list>
#include <vector>
#include "linear_algebra.h"
struct BVHNode
{
	Vector3f bb_min;
	Vector3f bb_max;
	bool isLeaf;
	// no leaf
	BVHNode* _left;
	BVHNode* _right;
	// leaf
	std::list<const Triangle*> _triangles;

	unsigned int box_count;
	unsigned int tri_count;
};

struct BVHNodeCache
{
	Vector3f bb_min;
	Vector3f bb_max;
	bool isLeaf;
	unsigned int _left;
	unsigned int _right;

	unsigned int box_count;
	unsigned int tri_count;
	unsigned int tri_idx;
};

extern BVHNodeCache* BVHCahce;

void CreateBVH();


#endif // !__BVH_H_


