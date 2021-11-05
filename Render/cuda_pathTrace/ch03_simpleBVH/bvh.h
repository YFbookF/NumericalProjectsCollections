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
};

BVHNode* BVHRoot;
BVHNode* CreateBVH();


#endif // !__BVH_H_


