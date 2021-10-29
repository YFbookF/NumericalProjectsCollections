/*********************************************************************
 * misc.cpp
 * Authored by Kris Hauser 2002-2003
 *
 * Miscellaneous math functions.
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


#include "misc.h"
#include <stdio.h>

//returns the # of real roots found (-1 if infinite)
int quadratic(Real a, Real b, Real c, Real& x1, Real& x2) {
  //printf("quadratic %f %f %f\n", a, b, c);
  static const Real Four = 4.0;

  if(a == Zero)
  {
	  if(b == Zero)
	  {
		if(c == Zero)
			return -1;
		return 0;
	  }
	  x1 = -c/b;
	  return 1;
  }

  Real det = b*b-Four*a*c;
  if(det < Zero)
    return 0;
  det = Sqrt(det);
  x1 = (-b + det)/a*Half;
  x2 = (-b - det)/a*Half;
  return 2;
}

Real pythag(Real a, Real b)		//reduce roundoff of large numbers
{
	Real absa,absb;
	absa=Abs(a);
	absb=Abs(b);

	if(absa > absb)
		return absa*Sqrt(One + Sqr(b/a));
	else
		return absb*Sqrt(One + Sqr(a/b));
}

