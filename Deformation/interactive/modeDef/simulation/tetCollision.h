/*********************************************************************
 * tetCollision.h
 *
 * These functions determine the overlap volume, center, and direction
 * of a pair of possibly colliding objects.  Adapted from code written
 * by James O'Brien.
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



#include "math/primitives.h"
#include "math/geometry.h"

 void computeTetTetOverlap(const Vector3 &v10, 
			   const Vector3 &v11, 
			   const Vector3 &v12, 
			   const Vector3 &v13,
				 
			   const Vector3 &v20, 
			   const Vector3 &v21, 
			   const Vector3 &v22, 
			   const Vector3 &v23, Real &vol, Vector3 &cnt, Vector3 &dir);

void computeTetPlaneOverlap(const Vector3 &v0, 
			    const Vector3 &v1, 
			    const Vector3 &v2, 
			    const Vector3 &v3,
			    const Plane &plane, Real &vol, Vector3 &cnt, Vector3 &dir);

void computeTetSphereOverlap(const Vector3 &v0, 
			     const Vector3 &v1, 
			     const Vector3 &v2, 
			     const Vector3 &v3,
			     const Vector3 &center, const Real radius, 
			     Real &vol, Vector3 &cnt, Vector3 &dir);
