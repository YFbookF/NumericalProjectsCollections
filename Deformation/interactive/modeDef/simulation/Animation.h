/*********************************************************************
 * Animation.h
 * Authored by Kris Hauser 2002-2003
 *
 * Defines a generic keyframed animation path class, as well as
 * rigid-frame animations that use cardinal interpolation.
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


#include "interp.h"

template<class type>
class CKeyframedAnimation
{
 public:
  CKeyframedAnimation();
  virtual ~CKeyframedAnimation();
  virtual void Initialize(int n);
  virtual void Interpolate(double t, type& out) const = 0;
  virtual bool Load(const char* fn);
  virtual bool Save(const char* fn);

  int num_keys;
  double *times;
  type* keys;
};

template<class type>
CKeyframedAnimation<type>::CKeyframedAnimation()
  :num_keys(0), times(NULL), keys(NULL)
{}

template<class type>
CKeyframedAnimation<type>::~CKeyframedAnimation()
{
  if(times)
    delete [] times;
  if(keys)
    delete [] keys;
}

template<class type>
void CKeyframedAnimation<type>::Initialize(int n)
{
  if(times)
    delete [] times;
  if(keys)
    delete [] keys;

  num_keys = n;
  times = new double [n];
  keys = new type [n];
}

template<class type>
bool CKeyframedAnimation<type>::Load(const char* fn)
{
  ifstream in;
  in.open(fn);
  if(!in)
    return false;
  in >> num_keys;
  Initialize(num_keys);
  for(int i=0; i<num_keys; i++) {
    in >> times[i];
    in >> keys[i];
  }
  if(!in)
    return false;
  in.close();
  return true;
}


template<class type>
bool CKeyframedAnimation<type>::Save(const char* fn)
{
  ofstream out;
  out.open(fn);
  if(!out)
    return false;
  out << num_keys << endl << endl;
  for(int i=0; i<num_keys; i++) {
    out << times[i] << endl;
    out << keys[i] << endl << endl;
  }
  if(!out)
    return false;
  out.close();
  return true;
}

template<class type>
class CKeyframedAnimationLinear : public CKeyframedAnimation<type>
{
 public:
  void Interpolate(double t, type& out) const {
    linear_interp_piecewise(times, keys, num_keys, t, out);
  }
};


template<class type>
class CKeyframedAnimationCubic : public CKeyframedAnimation<type>
{
 public:
  void Interpolate(double t, type& out) const {
    cubic_interp_piecewise(times, keys, num_keys, t, out);
  }
};



#include "math/primitives.h"
#include "math/rotation.h"

typedef Vector3 AnimationPosition;
typedef QuaternionRotation AnimationRotation;


typedef CKeyframedAnimationCubic<AnimationPosition> CPositionAnimation;

class CRotationAnimation : public CKeyframedAnimation<AnimationRotation>
{
 public:
  void Interpolate(double t, AnimationRotation& out) const {
    int n = num_keys;
    int j = seek_segment(times,n,t);

    if(j < 0)
      out = keys[0];
    else if(j >= n-1)
      out = keys[n-1];
    else {
      int j_1 = j-1;
      if(j_1 < 0)
	j_1 = 0;
      int j2 = j+2;
      if(j2 > n-1)
	j2 = n-1;
      
      double u = (t-times[j])/(times[j+1]-times[j]);
      SCerp(keys[j_1], keys[j], keys[j+1], keys[j2], out, u);
    }
  }
};

class CFrameAnimation
{
 public: 
  bool Load(const char* fn) {
    //loads from position vector and euler angle (ZYX) vector (in degrees)

    ifstream in;
    in.open(fn);
    if(!in)
      return false;
    in >> position_track.num_keys;
    in >> rotation_track.num_keys;
    position_track.Initialize(position_track.num_keys);
    rotation_track.Initialize(rotation_track.num_keys);

    for(int i=0; i<position_track.num_keys; i++) {
      in >> position_track.times[i];
      in >> position_track.keys[i];
    }
    for(int i=0; i<rotation_track.num_keys; i++) {
      in >> rotation_track.times[i];
      Vector3 r;
      in >> r;
      r *= M_PI/180.0;
      rotation_track.keys[i].setEulerAnglesZYX(r);
    }
    if(!in)
      return false;
    in.close();
    return true;
  }

  void Interp(double t, AnimationPosition& pos, AnimationRotation& rot) const {
    position_track.Interpolate(t, pos);
    rotation_track.Interpolate(t, rot);
  }

  CPositionAnimation position_track;
  CRotationAnimation rotation_track;
};
