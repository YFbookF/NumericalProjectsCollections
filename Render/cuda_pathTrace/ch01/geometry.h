#ifndef __GEOMETRY_H_
#define __GEOMETRY_H_

#include "linear_algebra.h"

struct Vertex
{
	Vector3f _pos;
	Vector3f _normal;
	Vertex(float px, float py, float pz, float nx, float ny, float nz):
		_pos(Vector3f(px,py,pz)),_normal(Vector3f(nx,ny,nz))
	{

	}
};

struct Triangle
{
	unsigned int _idx1;
	unsigned int _idx2;
	unsigned int _idx3;

	Vector3f _normal;

	Vector3f bb_bottom;
	Vector3f bb_top;
};

#endif // !__GEOMETRY_H_
