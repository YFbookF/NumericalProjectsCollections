#pragma once
#include "linear_algebra.h"
#include <list>

struct AABB
{
	Vector3f bottom;
	Vector3f top;
};

class Triangle
{
public:
	unsigned int _idx1;
	unsigned int _idx2;
	unsigned int _idx3;
	Vector3f _center;
	Vector3f _normal;
	Vector3f _twoSided;

	Vector3f bb_bottom;
	Vector3f bb_top;
	Triangle();
	~Triangle();


private:

};


class BVHNode
{
	Vector3f minCorner;
	Vector3f maxCorner;
	virtual bool IsLeaf() = 0;

	Vector3f getVertexPos(unsigned _idx)
	{
		return Vector3f(0, 0, 0);
	}

	BVHNode* CreateBVH()
	{
		float WORLD_MAX = 100;
		Vector3f bottom(WORLD_MAX, WORLD_MAX, WORLD_MAX);
		Vector3f top(WORLD_MAX, WORLD_MAX, WORLD_MAX);

		int trianglesNum;
		for (unsigned int j = 0; j < trianglesNum; j++)
		{
			Triangle triangle;
			AABB b;
			b.bottom = min3(min3(b.bottom, getVertexPos(triangle._idx1)),
				min3(getVertexPos(triangle._idx2), getVertexPos(triangle._idx3)));
			b.top = max3(max3(b.bottom, getVertexPos(triangle._idx1)),
				max3(getVertexPos(triangle._idx2), getVertexPos(triangle._idx3)));

		}
	}

	BVHNode* Recurse()
	{
		for (int axis_idx = 0; axis_idx < 3; axis_idx++)
		{
			float start = 0, stop = 0;
			if (fabsf(start - stop) < 1e-4)continue;
			float step = (stop - start) / 1024.;
			float WORLD_MAX = 100;
			for (float testSplit = start + step; testSplit < stop - step; testSplit += step)
			{
				Vector3f lbottom(WORLD_MAX, WORLD_MAX, WORLD_MAX);
				Vector3f ltop(-WORLD_MAX, -WORLD_MAX, -WORLD_MAX);

				Vector3f rbottom(WORLD_MAX, WORLD_MAX, WORLD_MAX);
				Vector3f rtop(-WORLD_MAX, -WORLD_MAX, -WORLD_MAX);

			}
		}
	}
};

class BVHLeaf :BVHNode
{
	std::list<const int*> _triangles;
	bool IsLeaf() { return true; }
}