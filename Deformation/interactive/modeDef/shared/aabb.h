/*********************************************************************
 * aabb.h
 * Authored by Kris Hauser 2002-2003
 *
 * Functions for an inlined axis-aligned bounding box.
 *
 * Copyright (c) 2003 Regents of the University of California.
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

#include <primitives.h>
#include <float.h>

/*
template<class type>
inline const type& min(const type& a, const type& b) {
  return (a<b?a:b);
}

template<class type>
inline const type& max(const type& a, const type& b) {
  return (a>b?a:b);
  }*/
#define MIN(a,b) (a<b?a:b)
#define MAX(a,b) (a>b?a:b)


#define AABB_DECLARE(name,type) type name[6]

template <class type>
inline type* aabb_min(type b [6])
{
  return &b[0];
}

template <class type>
inline type* aabb_max(type b [6])
{
  return &b[3];
}

template <class type>
inline void aabb_init(type b [6], type min = DBL_MAX, type max = -DBL_MAX)
{
  b[0] = b[1] = b[2] = min;
  b[3] = b[4] = b[5] = max;
}

template <class type>
inline bool aabb_contains_point(const type b [6], const type x[3]) {
  return x[0] >= b[0] && x[0] <= b[3] && 
    x[1] >= bmin[1] && x[1] <= b[4] && 
    x[2] >= bmin[2] && x[2] <= b[5];
}

template <class type>
inline bool aabb_contains(const type b [6], const type b2 [6]) {
  return b2[0] > b[0] && b2[3] < b[3] && 
    b2[1] > b[1] && b2[4] < b[4] && 
    b2[2] > b[2] && b2[5] < b[5];
}

template <class type>
inline void aabb_expand_point(type b [6], const type x[3]) {
  b[0] = MIN(b[0], x[0]);
  b[1] = MIN(b[1], x[1]);
  b[2] = MIN(b[2], x[2]);
  b[3] = MAX(b[3], x[0]);
  b[4] = MAX(b[4], x[1]);
  b[5] = MAX(b[5], x[2]);
}

template <class type>
inline void aabb_expand(type b [6], const type b2 [6]) {
  b[0] = MIN(b[0], b2[0]);
  b[1] = MIN(b[1], b2[1]);
  b[2] = MIN(b[2], b2[2]);
  b[3] = MAX(b[3], b2[3]);
  b[4] = MAX(b[4], b2[4]);
  b[5] = MAX(b[5], b2[5]);
}

#include <stdio.h>

inline void aabb_print(const float b[6], FILE* f = stdout)
{
  fprintf(f, "BMin: (%f %f %f)\n", b[0], b[1], b[2]);
  fprintf(f, "BMax: (%f %f %f)\n", b[3], b[4], b[5]);
}
