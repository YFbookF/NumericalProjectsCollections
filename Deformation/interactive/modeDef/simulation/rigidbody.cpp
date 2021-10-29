/*********************************************************************
 * rigidbody.cpp
 *
 * Rigid body objects.
 * Copied almost verbatim from James O'Brien's SM rigid body code,
 * Adapted to use my math structures.
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



#include "rigidbody.h"
#include <stdlib.h>

//-------------------------------------------------------------------
//-------------------------------------------------------------------

// DynamicsRigidBody


DynamicsRigidBody::DynamicsRigidBody() {
  _pos.setIdentity();
  _IMat   .setIdentity();
  _IMatInv.setIdentity();
  _mass = 1.0;
  _com  = 0.0;
  _tVel = 0.0;
  _aVel = 0.0;
  _frc  = 0.0;
  _mom  = 0.0;
  _posLock = 0;
  _rotLock = 0;
  recomputeVelMat(); 
}

  

double DynamicsRigidBody::update() {
  recomputeVelMat();
  return 0.0;
}



double DynamicsRigidBody::update(double deltaT) {

  if (!_posLock) {
    Vector3 accW =  _pos.R * ( _frc  / _mass );
    Vector3 velW = _tVel + accW * deltaT;

    _pos.t += ( _tVel + velW ) * (0.5 * deltaT);
    _tVel = velW;

  }
  
  if (!_rotLock) {
    Matrix3 pInv;
    pInv.setTranspose(_pos.R);

    Vector3 w  = (pInv*_aVel);   // rot vel in local frame
    Vector3 H  = (_IMat*w   );   // rot momentum in local frame
    Vector3 dH = (_mom-cross(w,H));  

    Vector3 accW = (_pos.R*_IMatInv*dH);
    Vector3 velW = ( _aVel + accW * deltaT );

    MomentRotation dRotV = ((_aVel + velW )*( 0.5 * deltaT ));
    _aVel = velW;

    Matrix3 dRotM;
    dRotV.getMatrix(dRotM);
    Matrix3 newRot;
    newRot.mul(dRotM,_pos.R);

    _pos.t += _pos.R*_com - newRot*_com;
    _pos.R = newRot;

  }

  recomputeVelMat();

  _frc = 0.0;
  _mom = 0.0;

  return deltaT;
}

void DynamicsRigidBody::recomputeVelMat() {

  if (_posLock) {
    _vel.t = 0.0;
  }else{
    _vel.t = _tVel;
  }

  if (_rotLock) {
    _vel.R = 0.0;
  }else{
    _vel.R = _pos.R;
    Matrix3 localRot;
    Vector3 localVel;
    _pos.R.mulTranspose(_aVel, localVel);
    localRot.setCrossProduct(localVel);
    _vel.R *= localRot;

    _vel.t -=  _vel.R*_com;
  }
}
