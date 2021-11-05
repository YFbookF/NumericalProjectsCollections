#pragma once
#include<vector>
#include "bvh.h"
#include "cuda_render.h"
struct BBoxTemp
{
	Vector3f bb_min;
	Vector3f bb_max;
	const Triangle* _pTri;
};

BVHNode* CreateBVHChild(std::vector<BBoxTemp> work_bb)
{
	unsigned int work_bb_size = work_bb.size();
	if (work_bb_size < 2)
	{
		BVHNode* leaf = new BVHNode;
		leaf->isLeaf = true;
		for (std::vector<BBoxTemp>::iterator it = work_bb.begin();
			it != work_bb.end(); it++)
		{
			leaf->_triangles.push_back(it->_pTri);
		}
		return leaf;
		
	}
	unsigned int mid_size = floor(work_bb_size * 0.5);
	std::vector<BBoxTemp> work_left;
	std::vector<BBoxTemp> work_right;
	for (int i = 0; i < work_bb_size; i++)
	{
		if (i <= mid_size)
		{
			work_left.push_back(work_bb[i]);
		}
		else
		{
			work_right.push_back(work_bb[i]);
		}
	}
	BVHNode* inner = new BVHNode;
	inner->isLeaf = false;
	inner->_left = CreateBVHChild(work_left);
	inner->_right = CreateBVHChild(work_right);
	return inner;

}

BVHNode* CreateBVH()
{
	std::vector<BBoxTemp> work_bb;

	for (unsigned int j = 0; j < triangle_num; j++)
	{
		BBoxTemp bbt;
		const Triangle& triangle_cur = scene_triangles[j];
		bbt._pTri = &triangle_cur;

	}

	BVHRoot = CreateBVHChild(work_bb);
}