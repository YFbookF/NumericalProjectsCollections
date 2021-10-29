/*********************************************************************
 * primitives.cpp
 * Authored by Kris Hauser 2002-2003
 *
 * Declares some simple linear algebra primitives,
 * using C arrays.
 * NOTE: there's another primitives.h in math/.
 * This one doesn't use classes.
 *
 * Copyright (c) 2002,2003 SquireSoft, used with permission.
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

void matrix4_rotation(matrix4_t x, const vec3_t r)
{
	float_t m = vec3_length(r);
	float_t cm = cos(m);
	float_t sm = sin(m);

	if(m == 0)
	{
		matrix4_identity(x);
		return;
	}

	vec3_t v;
	vec3_multiply(v,r,1.0/m);

	matrix4_t rmat;
	matrix4_cross_product(rmat, v);

	matrix4_multiply(x, rmat, rmat);
	matrix4_scale3(x, cm);

	matrix4_scale3(rmat, sm);
	matrix4_sub(x, rmat, x);

	for(int i=0; i<3; i++)
		for(int j=0; j<3; j++)
			x[i][j] += v[i]*v[j];

	x[3][3] = 1.0;
}

void matrix4_rotation_axis(vec3_t r, const matrix4_t R)
{
	float_t theta = acos((R[0][0] + R[1][1] + R[2][2] - 1.0)*0.5);

	if(theta == 0.0)
	{
		vec3_zero(r);
		return;
	}

	matrix4_t r_cross;
	matrix4_transpose(r_cross, R);
	matrix4_sub(r_cross, R, r_cross);
	matrix4_scale3(r_cross, 0.5*theta/sin(theta));

	r[0] = r_cross[1][2];
	r[1] = r_cross[2][0];
	r[2] = r_cross[0][1];
}


#include <stdio.h>

#define ROTATE1(i) ((i+1)%3)
#define ROTATE2(i) ((i+2)%3)

bool matrix3_invert( matrix4_t x, const matrix4_t a )
{
  matrix4_identity(x);

  float_t fDetInv = 1.0 / ( a[0][0] * ( a[1][1] * a[2][2] - a[1][2] * a[2][1] ) -
                             a[0][1] * ( a[1][0] * a[2][2] - a[1][2] * a[2][0] ) +
                             a[0][2] * ( a[1][0] * a[2][1] - a[1][1] * a[2][0] ) );

  int i,j;

  for(i=0; i<3; i++) {
    for(j=0; j<3; j++)
      {
	x[i][j] = a[ROTATE1(j)][ROTATE1(i)]*a[ROTATE2(j)][ROTATE2(i)] - a[ROTATE1(j)][ROTATE2(i)]*a[ROTATE2(j)][ROTATE1(i)];
	x[i][j] *= fDetInv;
      }
  }
  return true;
}

#define MATSWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}

bool matrix4_invert(matrix4_t matr, const matrix4_t a) {
  matrix4_t ident;
  int i, j, k, l, ll;
  int icol=0, irow=0;
  int indxc[4], indxr[4], ipiv[4];
  float_t big, dum, pivinv, temp;

  matrix4_equal(matr, a);
  matrix4_identity(ident);

   // Gauss-jordan elimination with full pivoting.  Yes, folks, a 
   // GL Matrix4 is inverted like any other, since the identity is 
   // still the identity.
   
   // from numerical recipies in C second edition, pg 39
 
  for(j=0;j<=3;j++) ipiv[j] = 0;
  for(i=0;i<=3;i++) {
    big=0.0;
    for (j=0;j<=3;j++) {
      if(ipiv[j] != 1) {
	for (k=0;k<=3;k++) {
	  if(ipiv[k] == 0) {
	    if(fabs(matr[j][k]) >= big) {
	      big = (float_t) fabs(matr[j][k]);
               irow=j;
               icol=k;
             }
           } else if (ipiv[k] > 1) {
	     printf("Error in gauss-jordan inversion\n");
	     return false;
	   }
         } 
       }
    }
     ++(ipiv[icol]);
     if (irow != icol) {
       for (l=0;l<=3;l++) MATSWAP(matr[irow][l],matr[icol][l]);
       for (l=0;l<=3;l++) MATSWAP(ident[irow][l],ident[icol][l]);
     }
     indxr[i]=irow;
     indxc[i]=icol;
     if(matr[icol][icol] == 0.0f) {
       printf("Error in gauss-jordan inversion\n");
       return false;
     }
     pivinv = 1.0f / matr[icol][icol];
     matr[icol][icol]=1.0f;
     for (l=0;l<=3;l++) matr[icol][l] *= pivinv;
     for (l=0;l<=3;l++) ident[icol][l] *= pivinv;
     for (ll=0;ll<=3;ll++) {
       if (ll != icol) {
         dum=matr[ll][icol];
         matr[ll][icol]=0.0f;
         for (l=0;l<=3;l++) matr[ll][l] -= matr[icol][l]*dum;
         for (l=0;l<=3;l++) ident[ll][l] -= ident[icol][l]*dum;
       }
     }
   }
   for (l=3;l>=0;l--) {
     if (indxr[l] != indxc[l]) {
       for (k=0;k<=3;k++) {
         MATSWAP(matr[k][indxr[l]],matr[k][indxc[l]]);
       }
     }
   }
   return true;
}
