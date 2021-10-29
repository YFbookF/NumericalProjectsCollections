/*********************************************************************
 * fastvector.h
 * Authored by Kris Hauser 2002-2003
 *
 * Declares vector_type (an efficient C-style vector)
 * and several inlined helper functions.
 *
 * Copyright (c) 2002 SquireSoft, used with permission.
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



#ifndef FAST_VECTOR_H
#define FAST_VECTOR_H

#include <stdlib.h>
#include <stdio.h>

#include "primitives.h"

typedef Real* vector_type;

inline vector_type vector_create(int n)
{
	if(n != 0)
		return (Real*)malloc(sizeof(Real)*n);
	return NULL;
}

inline void vector_delete(vector_type v)
{
	if(v)
		free(v);
}


inline void vector_equal(vector_type a, const vector_type b, int n)
{
	int i;
	for(i=0;i<n;i++)
		a[i] = b[i];
}

inline vector_type vector_copy(const vector_type a, int n)
{
  vector_type temp = vector_create(n);
  vector_equal(temp, a, n);
  return temp;
}

inline void vector_zero(vector_type a, int n)
{
	int i;
	for(i=0;i<n;i++)
		a[i] = Zero;
}

inline void vector_add(vector_type x, const vector_type a, const vector_type b, int n)
{
	int i;
	for(i=0;i<n;i++)
		x[i] = a[i]+b[i];
}

inline void vector_sub(vector_type x, const vector_type a, const vector_type b, int n)
{
	int i;
	for(i=0;i<n;i++)
		x[i] = a[i]-b[i];
}

inline void vector_multiply(vector_type x, const vector_type a, Real s, int n)
{
	int i;
	for(i=0;i<n;i++)
		x[i] = a[i]*s;
}

inline void vector_madd(vector_type x, const vector_type a, Real s, int n)
{
	int i;
	for(i=0;i<n;i++)
		x[i] += a[i]*s;
}

inline void vector_scale(vector_type a, float s, int n)
{
	int i;
	for(i=0;i<n;i++)
		a[i] *= s;
}

inline void vector_scale(vector_type a, const vector_type b, float s, int n)
{
	vector_multiply(a,b,s,n);
}

inline Real vector_dot(const vector_type a, const vector_type b, int n)
{
	int i;
	Real sum = Zero;
	for(i=0;i<n;i++)
		sum += a[i]*b[i];
	return sum;
}

inline void vector_print(vector_type v, int n)
{
	printf("(");
	for(int j=0; j<n; j++)
	{
		printf("%f\t", v[j]);
	}
	printf(")\n");
}

#endif

