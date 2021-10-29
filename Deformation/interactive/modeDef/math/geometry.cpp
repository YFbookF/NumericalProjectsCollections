/*********************************************************************
 * geometry.cpp
 * Authored by Kris Hauser 2002-2003
 *
 * Simple geometry objects.
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


#include "geometry.h"



#define AABB_RANGE_MAX FLT_MAX
#define AABB_RANGE_MIN -FLT_MAX
//#define AABB_RANGE_MAX 1000
//#define AABB_RANGE_MIN -1000

AABB::AABB()
{}

AABB::AABB(const Vector3& a, const Vector3& b)
:bmin(a), bmax(b)
{}

bool AABB::contains(const Vector3& v) const
{
	return (v.x>=bmin.x && v.x<=bmax.x &&
		v.y>=bmin.y && v.y<=bmax.y &&
		v.z>=bmin.z && v.z<=bmax.z);
}

bool AABB::intersects(const AABB& a) const
{
	return (bmin.x <= a.bmax.x && bmax.x >= a.bmin.x &&
		bmin.y <= a.bmax.y && bmax.y >= a.bmin.y &&
		bmin.z <= a.bmax.z && bmax.z >= a.bmin.z);
}

void AABB::minimize()
{
	bmin.x = bmin.y = bmin.z = AABB_RANGE_MAX;
	bmax.x = bmax.y = bmax.z = AABB_RANGE_MIN;
}

void AABB::maximize()
{
	bmin.x = bmin.y = bmin.z = AABB_RANGE_MIN;
	bmax.x = bmax.y = bmax.z = AABB_RANGE_MAX;
}

void AABB::set_intersect(const AABB& a, const AABB& b)
{
	operator = (a);
	GetMaxVector(b.bmin, bmin);
	GetMinVector(b.bmax, bmax);
}

void AABB::set_expand(const AABB& a, const AABB& b)
{
	operator = (a);
	GetMinVector(b.bmin, bmin);
	GetMaxVector(b.bmax, bmax);
}


void AABB::intersect(const AABB& b)
{
	GetMaxVector(b.bmin, bmin);
	GetMinVector(b.bmax, bmax);
}

void AABB::expand(const AABB& b)
{
	GetMinVector(b.bmin, bmin);
	GetMaxVector(b.bmax, bmax);
}

const AABB& AABB::operator = (const AABB& other)
{
	bmin = other.bmin;
	bmax = other.bmax;
	return *this;
}

#include <float.h>
#define min(a,b) (a < b ? a : b)
#define max(a,b) (a > b ? a : b)
#define opposing_signs(a,b) ((a<0) != (b<0))



//forms a plane from a point on the plane and two basis Vector3s
void Plane_FromPointBases(const Vector3& a, const Vector3& b1, const Vector3& b2, Plane& p)
{
	p.normal.setCross(b1,b2);
	normalize(p.normal);
	p.offset = dot(a, p.normal);
}

//forms a plane from three points on the plane
void Plane_FromPoints(const Vector3& a, const Vector3& b, const Vector3& c, Plane& p)
{
	Plane_FromPointBases(a, b-a, c-a, p);
}

//returns the orthogonal distance from the plane to the Vector3
float PlaneOffset(const Vector3& v, const Plane& p)
{
	return dot(v,p.normal) - p.offset;
}

//projects a Vector3 onto this plane
void PlaneProject(const Vector3& vin, const Plane& p, Vector3& vout)
{
	vout = vin - p.normal*(dot(vin,p.normal) - p.offset);
}

//transforms a plane
void PlaneTransform(const Plane& pin, const Matrix4& xform, Plane& pout)
{
	xform.mulVector(pin.normal, pout.normal);

	Vector3 v, v_out;
	v = pin.normal*pin.offset;
	xform.mulPoint(v, v_out);
	pout.offset = dot(v_out, pout.normal);
}

