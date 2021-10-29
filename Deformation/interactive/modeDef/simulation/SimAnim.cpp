/*********************************************************************
 * SimAnim.cpp
 * Authored by Kris Hauser 2002-2003
 *
 * This is just like a CFrameAnimation but it puts the output of the
 * path into a CRigidState structure, using numerical differentiation
 * to calculate approximate velocities and accelerations. 
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


#include "SimAnim.h"

void QuatToRotation(const Quaternion& q, Vector3& v)
{
  Real angle = Two*Acos(q.w);
  Real imscale = angle/q.imNorm();
  v.set(q.x*imscale, q.y*imscale, q.z*imscale);
}

void DQ_CenteredDifferences(const Quaternion& qminush,
		       const Quaternion& qplush,
		       double h,
		       Vector3& dq)
{
  //assuming unit quaternions
  Quaternion qminush_conj;
  qminush_conj.setConjugate(qminush);
  QuatToRotation(qplush * qminush_conj, dq);
  dq *= 1.0/(2.0*h);
}

void D2Q_CenteredDifferences(const Quaternion& qminush,
			const Quaternion& q,
			const Quaternion& qplush,
			double h,
			Vector3& ddq)
{
  //assuming unit quaternions
  Quaternion q_conj;
  q_conj.setConjugate(q);
  QuatToRotation(qplush * q_conj * q_conj * qminush, ddq);
  ddq *= 1.0/(h*h);
}


void CRigidBodyAnimation::GetState(double t, CRigidState& state) const
{
  AnimationPosition pos;
  AnimationRotation rot;
  Interp(t, pos, rot);

  state.trnVec = pos;
  rot.getMatrix(state.rotMat);

  //finite differencing for vel, accel
  double h = 0.01;
  AnimationPosition pos1, pos2;
  AnimationRotation rot1, rot2;
  Interp(t-h, pos1, rot1);
  Interp(t+h, pos2, rot2);

  state.tVel = (pos2-pos1)/(h*2.0);
  DQ_CenteredDifferences(rot1, rot2, h, state.aVel);
  //leave it in world coords

  state.netFrc = (pos2 - 2*pos + pos1)/(h*h);
  D2Q_CenteredDifferences(rot1, rot, rot2, h, state.netMom);

  //cout << "avel " << state.aVel<< endl;
  //cout << "mom " << state.netMom<< endl;

  //both still in world coords, change em to sim coords
  state.netFrc = state.netFrc * state.rotMat;
  state.netMom = state.netMom * state.rotMat;
}


