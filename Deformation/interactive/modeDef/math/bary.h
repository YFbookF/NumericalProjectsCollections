/*********************************************************************
 * bary.h
 * Authored by Kris Hauser 2002-2003
 *
 * Header for functions involving barycentric coordinates of tetrahedra.
 *
 * Copyright 2003, Regents of the University of California 
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



#ifndef BARYCENTRIC_MATH_H
#define BARYCENTRIC_MATH_H

#include "shared/primitives.h"

typedef vec3_t tetrahedron [4];
typedef float_t* tetrahedron_ptr [4];
typedef vec4_t bary_t;

float_t triangle_area(const vec3_t a, const vec3_t b, const vec3_t c);
float_t tet_volume(const vec3_t a, const vec3_t b, const vec3_t c, const vec3_t d);
float_t tet_volume(const tetrahedron);
float_t tet_volume(const tetrahedron_ptr);
float_t get_barycentric_coordinates(const vec3_t point, const tetrahedron e, bary_t bary);
float_t get_barycentric_coordinates(const vec3_t point, const tetrahedron_ptr e, bary_t bary);

void tet_centroid(const tetrahedron e, vec3_t centroid);
void tet_centroid(const tetrahedron_ptr e, vec3_t centroid);

#include "math/primitives.h"

//on the line L(t) = a+tb
//returns the point q s.t. q = L(t0) and |q-p| is minimized
void get_line_closest_point(const Vector3& a, const Vector3& b,
			    const Vector3& p,
			    Vector3& q,
			    double& t0);

//on the segment L(t) = a+tb, t in [0,1]
//returns the point q s.t. q = L(t0) and |q-p| is minimized
void get_segment_closest_point(const Vector3& a, const Vector3& b,
			       const Vector3& p,
			       Vector3& q,
			       double& t0);

//gets the closest point x on the triangle(a,b,c) to the point p
void get_triangle_closest_point(const Vector3& a, const Vector3& b, const Vector3& c,
				const Vector3& p,
				Vector3& x);

//gets the closest point x on the tetrahedron(a,b,c,d) to the point p
void get_tetrahedron_closest_point(const Vector3& a, const Vector3& b, const Vector3& c, const Vector3& d,
				const Vector3& p,
				Vector3& x);



#endif

