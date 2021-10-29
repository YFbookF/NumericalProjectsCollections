/*********************************************************************
 * misc.h
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



#ifndef MATH_MISC_H
#define MATH_MISC_H

#include "primitives.h"

int quadratic(Real a, Real b, Real c, Real& x1, Real& x2);
Real pythag(Real a, Real b);

#endif

