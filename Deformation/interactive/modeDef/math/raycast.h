/*********************************************************************
 * raycast.h
 * Authored by Kris Hauser 2002-2003
 *
 * Borrowed heavily from code written by James O'Brien.
 * Structure to store the results of a ray-casting operation.
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



#include "geometry.h"

class RayCast
{
public:
  inline RayCast();
  inline RayCast(const Ray &ray , Real tMin = 0.0, Real tMax = Inf, long data = 0);
  inline RayCast(const RayCast &that);
  
  inline RayCast &operator=(const RayCast &that); 

  inline Vector3 minPos() const { return ray.source + tMin*ray.direction; }
  inline Vector3 maxPos() const { return ray.source + tMax*ray.direction; }

	Ray ray;
	Real tMin,tMax;
	long data;
};

inline RayCast::RayCast() :
  tMin(Zero),
  tMax(Inf),
  data(0) {
}
  

inline RayCast::RayCast(const Ray &_ray , Real _tMin, Real _tMax, long _data) :
  ray ( _ray),
  tMin(_tMin),
  tMax(_tMax),
  data(_data) {
}

inline RayCast::RayCast(const RayCast &that) :
  ray (that.ray),
  tMin(that.tMin),
  tMax(that.tMax),
  data(that.data) {
}
  
inline RayCast &RayCast::operator=(const RayCast &that) {
  ray  = that.ray ;
  tMin = that.tMin;
  tMax = that.tMax;
  data = that.data;
  return (*this);
}  

