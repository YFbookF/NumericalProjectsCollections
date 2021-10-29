/*********************************************************************
 * raytriangle.h
 * Authored by Kris Hauser 2002-2003
 *
 * Ray-triangle intersection tests
 *
 * Copyright (c) 2003, Regents of the University of California
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


#include "primitives.h"


//u,v are the coordinates in the triangle's frame, t is the dist
//non-culling
int ray_triangle_intersect(const vec3_t orig, const vec3_t dir,
                   const vec3_t vert0, const vec3_t vert1, const vec3_t vert2,
		   float_t *t, float_t *u, float_t *v);

//u,v are the coordinates in the triangle's frame, t is the dist
//does backface culling
int ray_triangle_intersect_cull(const vec3_t orig, const vec3_t dir,
                   const vec3_t vert0, const vec3_t vert1, const vec3_t vert2,
		   float_t *t, float_t *u, float_t *v);

//returns the intersection point of the t,u,v values
void intersect_to_intersection(float_t t, float_t u, float_t v,
		const vec3_t vert0, const vec3_t vert1, const vec3_t vert2,
 	        vec3_t pos);
