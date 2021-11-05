#pragma once
#include<vector>
#include "bvh.h"
#include "cuda_render.h"

BVHNode* BVHRoot;
BVHNodeCache* BVHCahce;

extern unsigned int vertices_num;
extern Vertex* scene_vertices_pos;
extern unsigned int triangle_num;
extern Triangle* scene_triangles;

struct BBoxTemp
{
	Vector3f bb_min;
	Vector3f bb_max;
	const Triangle* _pTri;
	BBoxTemp() :
		bb_min(WORLD_MAX, WORLD_MAX, WORLD_MAX),
		bb_max(-WORLD_MAX, -WORLD_MAX, -WORLD_MAX),
		_pTri(NULL)
	{
	}
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
		leaf->box_count = 1;
		leaf->tri_count = work_bb_size;
		return leaf;
		
	}
	unsigned int mid_size = floor((work_bb_size + 1.0) * 0.5);
	std::vector<BBoxTemp> work_left;
	std::vector<BBoxTemp> work_right;

	Vector3f left_bb_min(WORLD_MAX, WORLD_MAX, WORLD_MAX);
	Vector3f left_bb_max(-WORLD_MAX, -WORLD_MAX, -WORLD_MAX);

	Vector3f right_bb_min(WORLD_MAX, WORLD_MAX, WORLD_MAX);
	Vector3f right_bb_max(-WORLD_MAX, -WORLD_MAX, -WORLD_MAX);

	for (int i = 0; i < work_bb_size; i++)
	{
		if (i < mid_size)
		{
			work_left.push_back(work_bb[i]);
			left_bb_min = min3f(left_bb_min, work_bb[i].bb_min);
			left_bb_max = max3f(left_bb_max, work_bb[i].bb_max);
		}
		else
		{
			work_right.push_back(work_bb[i]);
			right_bb_min = min3f(right_bb_min, work_bb[i].bb_min);
			right_bb_max = max3f(right_bb_max, work_bb[i].bb_max);
		}
	}
	BVHNode* inner = new BVHNode;
	inner->isLeaf = false;
	inner->_left = CreateBVHChild(work_left);
	inner->_right = CreateBVHChild(work_right);

	inner->_left->bb_min = left_bb_min;
	inner->_left->bb_max = left_bb_max;
	inner->_right->bb_min = right_bb_min;
	inner->_right->bb_max = right_bb_max;
	inner->box_count = inner->_left->box_count + inner->_right->box_count + 1;
	inner->tri_count = inner->_left->tri_count + inner->_right->tri_count;

	return inner;

}

unsigned int node_idx = 0;
unsigned int tri_idx = 0;

void populateCache(BVHNode* root)
{
	int node_current = node_idx;
	BVHCahce[node_current].bb_min = root->bb_min;
	BVHCahce[node_current].bb_max = root->bb_max;
	BVHCahce[node_current].box_count = root->box_count;
	BVHCahce[node_current].tri_idx = -1;
	if (root->isLeaf)
	{
		BVHCahce[node_current].isLeaf = true;
		BVHCahce[node_current].tri_idx = tri_idx;
		tri_idx += 1;
	}
	else
	{
		node_idx += 1;
		unsigned int idxLeft = node_idx;
		populateCache(root->_left);
		node_idx += 1;
		unsigned int idxRight = node_idx;
		populateCache(root->_right);
		BVHCahce[node_current].isLeaf = false;
		BVHCahce[node_current]._left = idxLeft;
		BVHCahce[node_current]._right = idxRight;
	}
}

void CreateBVH()
{
	std::vector<BBoxTemp> work_bb;

	Vector3f root_bb_min(WORLD_MAX, WORLD_MAX, WORLD_MAX);
	Vector3f root_bb_max(-WORLD_MAX, -WORLD_MAX, -WORLD_MAX);

	for (unsigned int j = 0; j < 8; j++)
	{
		//printf_s("%f %f %f\n", scene_vertices_pos[j].x, scene_vertices_pos[j].y, scene_vertices_pos[j].z);
	}

	for (unsigned int j = 0; j < 12; j++)
	{
		//printf_s("%d %d %d\n", scene_triangles[j]._idx1, scene_triangles[j]._idx2, scene_triangles[j]._idx3);
	}

	for (unsigned int j = 0; j < triangle_num; j++)
	{
		BBoxTemp bbt;
		const Triangle& triangle_cur = scene_triangles[j];
		bbt._pTri = &triangle_cur;

		bbt.bb_min = min3f(min3f(bbt.bb_min, scene_vertices_pos[triangle_cur._idx1 - 1]),
			(min3f(scene_vertices_pos[triangle_cur._idx2 - 1], scene_vertices_pos[triangle_cur._idx3 - 1])));
		bbt.bb_max = max3f(max3f(bbt.bb_max, scene_vertices_pos[triangle_cur._idx1 - 1]),
			(max3f(scene_vertices_pos[triangle_cur._idx2 - 1], scene_vertices_pos[triangle_cur._idx3 - 1])));

		//printf_s("min %f %f %f\n", bbt.bb_min.x, bbt.bb_min.y, bbt.bb_min.z);
		//printf_s("max %f %f %f\n", bbt.bb_max.x, bbt.bb_max.y, bbt.bb_max.z);
		work_bb.push_back(bbt);

		root_bb_min = min3f(root_bb_min, bbt.bb_min);
		root_bb_max = max3f(root_bb_max, bbt.bb_max);
		
	}

	BVHRoot = CreateBVHChild(work_bb);
	BVHRoot->bb_min = root_bb_min;
	BVHRoot->bb_max = root_bb_max;

	int tree_size = BVHRoot->box_count;
	BVHCahce = new BVHNodeCache[tree_size];
	populateCache(BVHRoot);
}

