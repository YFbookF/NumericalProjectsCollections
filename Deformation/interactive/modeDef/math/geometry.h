/*********************************************************************
 * geometry.h
 * Authored by Kris Hauser 2002-2003
 *
 * Header for simple geometry objects.
 *
 * Copyright (c) 2002 SquireSoft, used and modified with permission.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *********************************************************************/



#ifndef MATH_GEOMETRY_H
#define MATH_GEOMETRY_H

#include "primitives.h"
#include "mylist.h"


typedef Vector3 Point;

struct AABB
{
	AABB();
	AABB(const Vector3&, const Vector3&);
	bool contains(const Vector3&) const;
	bool intersects(const AABB&) const;

	void minimize();
	void maximize();
	void set_intersect(const AABB&, const AABB&);
	void set_expand(const AABB&, const AABB&);
	void intersect(const AABB&);
	void expand(const AABB&);

	const AABB& operator = (const AABB&);

	Vector3 bmin, bmax;
};

inline AABB AABB_intersect(const AABB& x, const AABB& y)
{
	AABB z;
	z.set_intersect(x,y);
	return z;
}

inline AABB AABB_expand(const AABB& x, const AABB& y)
{
	AABB z;
	z.set_expand(x,y);
	return z;
}


inline void GetMinVector(const Vector3& in, Vector3& out)
{
	if(in.x < out.x) out.x=in.x;
	if(in.y < out.y) out.y=in.y;
	if(in.z < out.z) out.z=in.z;
}

inline void GetMaxVector(const Vector3& in, Vector3& out)
{
	if(in.x > out.x) out.x=in.x;
	if(in.y > out.y) out.y=in.y;
	if(in.z > out.z) out.z=in.z;
}



struct Line
{
	Vector3 source;
	Vector3 direction;
};

typedef Line Ray;

struct Segment
{
	Vector3 A;
	Vector3 B;
};

struct Plane
{
	Vector3 normal;
	float offset;
};

//forms a plane from a point on the plane and two basis Vector3s
void Plane_FromPointBases(const Vector3& a, const Vector3& b1, const Vector3& b2, Plane&);

//forms a plane from three points on the plane
void Plane_FromPoints(const Vector3& a, const Vector3& b, const Vector3& c, Plane&);

//returns the orthogonal distance from the plane to the Vector3
float PlaneOffset(const Vector3& v, const Plane& p);

//projects a Vector3 onto this plane
void PlaneProject(const Vector3& vin, const Plane& p, Vector3& vout);

//transforms a plane
void PlaneTransform(const Plane& pin, const Matrix4& xform, Plane& pout);



struct CollisionData
{
	Vector3 a_pt, b_pt;				//the points of collision on a and b, respectively
	Vector3 a_plane, b_plane;		//the normals of the planes of collision on a and b, respectively
									//(pointing outward)
};


#endif
