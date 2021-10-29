/*********************************************************************
 * raytriangle.cpp
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


#include "raytriangle.h"

#define EPSILON 2e-6

//u,v are the coordinates in the triangle's frame, t is the dist
//non-culling
int ray_triangle_intersect(const vec3_t orig, const vec3_t dir,
                   const vec3_t vert0, const vec3_t vert1, const vec3_t vert2,
                   float_t *t, float_t *u, float_t *v)
{
   vec3_t edge1, edge2, tvec, pvec, qvec;
   float_t det,inv_det;

   /* find vectors for two edges sharing vert0 */
   vec3_sub(edge1, vert1, vert0);
   vec3_sub(edge2, vert2, vert0);

   /* begin calculating determinant - also used to calculate U parameter */
   vec3_cross(pvec, dir, edge2);

   /* if determinant is near zero, ray lies in plane of triangle */
   det = vec3_dot(edge1, pvec);

   if (det > -EPSILON && det < EPSILON)
     return 0;
   inv_det = 1.0 / det;

   /* calculate distance from vert0 to ray origin */
   vec3_sub(tvec, orig, vert0);

   /* calculate U parameter and test bounds */
   *u = vec3_dot(tvec, pvec) * inv_det;
   if (*u < 0.0 || *u > 1.0)
     return 0;

   /* prepare to test V parameter */
   vec3_cross(qvec, tvec, edge1);

   /* calculate V parameter and test bounds */
   *v = vec3_dot(dir, qvec) * inv_det;
   if (*v < 0.0 || *u + *v > 1.0)
     return 0;

   /* calculate t, ray intersects triangle */
   *t = vec3_dot(edge2, qvec) * inv_det;

   return 1;
}

//u,v are the coordinates in the triangle's frame, t is the dist
//does backface culling
int ray_triangle_intersect_cull(const vec3_t orig, const vec3_t dir,
                   const vec3_t vert0, const vec3_t vert1, const vec3_t vert2,
                   float_t *t, float_t *u, float_t *v)
{
   vec3_t edge1, edge2, tvec, pvec, qvec;
   float_t det,inv_det;

   /* find vectors for two edges sharing vert0 */
   vec3_sub(edge1, vert1, vert0);
   vec3_sub(edge2, vert2, vert0);

   /* begin calculating determinant - also used to calculate U parameter */
   vec3_cross(pvec, dir, edge2);

   /* if determinant is near zero, ray lies in plane of triangle */
   det = vec3_dot(edge1, pvec);

   if (det < EPSILON)
      return 0;

   /* calculate distance from vert0 to ray origin */
   vec3_sub(tvec, orig, vert0);

   /* calculate U parameter and test bounds */
   *u = vec3_dot(tvec, pvec);
   if (*u < 0.0 || *u > det)
      return 0;

   /* prepare to test V parameter */
   vec3_cross(qvec, tvec, edge1);

    /* calculate V parameter and test bounds */
   *v = vec3_dot(dir, qvec);
   if (*v < 0.0 || *u + *v > det)
      return 0;

   /* calculate t, scale parameters, ray intersects triangle */
   *t = vec3_dot(edge2, qvec);
   inv_det = 1.0 / det;
   *t *= inv_det;
   *u *= inv_det;
   *v *= inv_det;

   return 1;
}


void intersect_to_intersection(float_t t, float_t u, float_t v,
		const vec3_t vert0, const vec3_t vert1, const vec3_t vert2,
		vec3_t pos)
{
  vec3_t edge1, edge2;
  vec3_sub(edge1, vert1, vert0);
  vec3_sub(edge2, vert2, vert0);

  vec3_equal(pos, vert0);
  vec3_madd(pos, edge1, u);
  vec3_madd(pos, edge2, v); 
}
