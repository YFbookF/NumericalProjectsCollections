/*********************************************************************
 * bary.cpp
 * Authored by Kris Hauser 2002-2003
 *
 * Functions involving barycentric coordinates of tetrahedra.
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



#include "bary.h"
#include <assert.h>

float_t triangle_area(const vec3_t a, const vec3_t b, const vec3_t c)
{
	vec3_t i,j,n;
	vec3_sub(i, b, a);
	vec3_sub(j, c, a);
	vec3_cross(n,i,j);
	return vec3_length(n)*0.5;
}

float_t tet_volume(const vec3_t a, const vec3_t b, const vec3_t c, const vec3_t d)
{
  vec3_t x,y,z,tmp;
  vec3_sub(x,b,a);
  vec3_sub(y,c,a);
  vec3_sub(z,d,a);
  vec3_cross(tmp, x, y);
  return fabs(vec3_dot(tmp,z))/6.0;
}

float_t tet_volume(const tetrahedron e)
{
	return tet_volume(e[0],e[1],e[2],e[3]);
}

float_t tet_volume(const tetrahedron_ptr e)
{
	return tet_volume(e[0],e[1],e[2],e[3]);
}

void tet_centroid(const tetrahedron e, vec3_t centroid)
{
  vec3_add(centroid, e[0], e[1]);
  vec3_add(centroid, centroid, e[2]);
  vec3_add(centroid, centroid, e[3]);
  vec3_multiply(centroid, centroid, 0.25);
}

void tet_centroid(const tetrahedron_ptr e, vec3_t centroid)
{
  vec3_add(centroid, e[0], e[1]);
  vec3_add(centroid, centroid, e[2]);
  vec3_add(centroid, centroid, e[3]);
  vec3_multiply(centroid, centroid, 0.25);
}

float_t get_barycentric_coordinates(const vec3_t point, const tetrahedron e, bary_t bary) {
  float_t vol = tet_volume(e[0], e[1], e[2], e[3]);
  //if(vol < EPSILON)
  //vol = bary[0]+bary[1]+bary[2]+bary[2];
  vol = 1.0/vol;

  bary[0] = tet_volume(point, e[1], e[2], e[3])*vol;
  bary[1] = tet_volume(point, e[2], e[3], e[0])*vol;
  bary[2] = tet_volume(point, e[3], e[0], e[1])*vol;
  bary[3] = tet_volume(point, e[0], e[1], e[2])*vol;

  return bary[0] + bary[1] + bary[2] + bary[3];
  //if(bary[0] + bary[1] + bary[2] + bary[3] > 1.0001)
  //return false;
  //return true;
}

float_t get_barycentric_coordinates(const vec3_t point, const tetrahedron_ptr e, bary_t bary) {
  float_t vol = tet_volume(e[0], e[1], e[2], e[3]);
  //if(vol < EPSILON)
  //vol = bary[0]+bary[1]+bary[2]+bary[2];
  vol = 1.0/vol;

  bary[0] = tet_volume(point, e[1], e[2], e[3])*vol;
  bary[1] = tet_volume(point, e[2], e[3], e[0])*vol;
  bary[2] = tet_volume(point, e[3], e[0], e[1])*vol;
  bary[3] = tet_volume(point, e[0], e[1], e[2])*vol;

  return bary[0] + bary[1] + bary[2] + bary[3];
  //if(bary[0] + bary[1] + bary[2] + bary[3] > 1.0001)
  //return false;
  //return true;
}







//on the line L(t) = a+tb
//returns the point q s.t. q = L(t0) and |q-p| is minimized
void get_line_closest_point(const Vector3& a, const Vector3& b,
			    const Vector3& p,
			    Vector3& q,
			    double& t0)
{
  t0 = dot(p-a,b)/dot(b,b);
  q = a + t0*b;
}

//on the segment L(t) = a+tb, t in [0,1]
//returns the point q s.t. q = L(t0) and |q-p| is minimized
void get_segment_closest_point(const Vector3& a, const Vector3& b,
			       const Vector3& p,
			       Vector3& q,
			       double& t0)
{
  t0 = dot(p-a,b)/dot(b,b);
  if(t0 < 0.0) {
    t0 = 0.0;
    q = a;
  }
  else if(t0 > 1.0) {
    t0 = 1.0;
    q = a+b;
  }
  else
    q = a + t0*b;
}



//gets the closest point x on the triangle(a,b,c) to the point p
void get_triangle_closest_point(const Vector3& a, const Vector3& b, const Vector3& c,
				const Vector3& p,
				Vector3& x)
{
  //first get the closest segment
  Vector3 centroid = (a+b+c)/3.0;
  Vector3 va,vb,vc;
  va = a - centroid;
  vb = b - centroid;
  vc = c - centroid;
  
  Vector3 i,j;
  i = b-a;
  j = c-a;
  Vector3 n = cross(i,j);

  Matrix3 m1 (va, vb, n);
  Matrix3 m2 (vb, vc, n);
  Matrix3 m3 (vc, va, n);
  Matrix3 minv;
  double t;

  minv.setInverse(m1);
  i = minv * (p - centroid);
  if(i[0] >= 0.0 && i[1] >= 0.0) { //(a,b) is closest segment 
    if(i[0] + i[1] <= 1.0)  //it's within sub-triangle
      x = centroid + va*i[0] + vb*i[1];
    else //it's outside, so project it onto the segment
      get_segment_closest_point(a, b-a, p, x, t);
    return;
  }

  minv.setInverse(m2);
  i = minv * (p - centroid);
  if(i[0] >= 0.0 && i[1] >= 0.0) { //(b,c) is closest segment 
    if(i[0] + i[1] <= 1.0)  //it's within sub-triangle
      x = centroid + vb*i[0] + vc*i[1];
    else //it's outside, so project it onto the segment
      get_segment_closest_point(b, c-b, p, x, t);
    return;
  }

  minv.setInverse(m3);
  i = minv * (p - centroid);
  //if(i[0] >= 0.0 && i[1] >= 0.0) { //(c,a) is closest segment 
  assert(i[0] >= 0.0 && i[1] >= 0.0);
    if(i[0] + i[1] <= 1.0)  //it's within sub-triangle
      x = centroid + vc*i[0] + va*i[1];
    else //it's outside, so project it onto the segment
      get_segment_closest_point(c, a-c, p, x, t);
  //}
}


//gets the closest point x on the tetrahedron(a,b,c,d) to the point p
void get_tetrahedron_closest_point(const Vector3& a, const Vector3& b, const Vector3& c, const Vector3& d,
				const Vector3& p,
				Vector3& x)
{
  //first get the closest face
  Vector3 centroid = (a+b+c+d)/4.0;
  Vector3 va,vb,vc,vd;
  va = a - centroid;
  vb = b - centroid;
  vc = c - centroid;
  vd = d - centroid;
  
  Matrix3 m;
  Matrix3 minv;
  Vector3 i;
  double t;

  //cout << "p is " << p << ", centroid is " << centroid <<  endl;
  //cout << "p-cent " << p-centroid << endl;

  //cout << va << endl;
  //cout << vb << endl;
  //cout << vc << endl;
  //cout << vd << endl;

  m.set(va, vb, vc);
  minv.setInverse(m);
  i = minv*(p - centroid);
  //cout << p-centroid  << " = " << m*i << endl;
  //cout << "inv 1 (abc): " << i << endl;
  if(i[0] >= 0.0 && i[1] >= 0.0 && i[2] >= 0.0) { //(a,b,c) is closest face 
    if(i[0] + i[1] + i[2] <= 1.0)  //it's within sub-pyramid
      x = p;
    else //it's outside, so project it onto the triangle
      get_triangle_closest_point(a, b, c, p, x);
    return;
  }

  m.set(vb, vc, vd);
  minv.setInverse(m);
  i = minv * (p - centroid);
  //cout << p-centroid  << " = " << m*i << endl;
  //cout << "inv 2 (bcd): " << i << endl;
  if(i[0] >= 0.0 && i[1] >= 0.0 && i[2] >= 0.0) { //(b,c,d) is closest face 
    if(i[0] + i[1] + i[2] <= 1.0)  //it's within sub-pyramid
      x = p;
    else //it's outside, so project it onto the triangle
      get_triangle_closest_point(b, c, d, p, x);
    return;
  }

  m.set(vc, vd, va);
  minv.setInverse(m);
  i = minv * (p - centroid);
  //cout << p-centroid  << " = " << m*i << endl;
  //cout << "inv 3 (cda): " << i << endl;
  if(i[0] >= 0.0 && i[1] >= 0.0 && i[2] >= 0.0) { //(c,d,a) is closest face 
    if(i[0] + i[1] + i[2] <= 1.0)  //it's within sub-pyramid
      x = p;
    else //it's outside, so project it onto the triangle
      get_triangle_closest_point(c, d, a, p, x);
    return;
  }

  m.set(vd, va, vb);
  minv.setInverse(m);
  i = minv * (p - centroid);
  //cout << p-centroid  << " = " << m*i << endl;
  //cout << "inv 4 (dab): " << i << endl;
  if(i[0] >= 0.0 && i[1] >= 0.0 && i[2] >= 0.0) { //(d,a,b) is closest face 
    if(i[0] + i[1] + i[2] <= 1.0)  //it's within sub-pyramid
      x = p;
    else //it's outside, so project it onto the triangle
      get_triangle_closest_point(d, a, b, p, x);
    return;
  }
  printf("Error in get_tetrahedron_closest_point, point not in a division of space?\n");
  assert(false);
}
