/*******************************************************************************
* nemesis. an experimental finite element code.                                *
* Copyright (C) 2004-2011 F.E.Karaoulanis [http://www.nemesis-project.org]     *
*                                                                              *
* This program is free software; you can redistribute it and/or modify         *
* it under the terms of the GNU General Public License version 3, as           *
* published by the Free Software Foundation.                                   *
*                                                                              *
* This program is distributed in the hope that it will be useful,              *
* but WITHOUT ANY WARRANTY; without even the implied warranty of               *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
* GNU General Public License for more details.                                 *
*                                                                              *
* You should have received a copy of the GNU General Public License            *
* along with this program.  If not, see < http://www.gnu.org/licenses/>.       *
*******************************************************************************/

// *****************************************************************************
// $LastChangedDate$
// $LastChangedRevision$
// $LastChangedBy$
// $HeadURL$
// Author(s): F.E. Karaoulanis (fkar@nemesis-project.org)
// *****************************************************************************

#include "material/vm.h"
#include "material/von_mises.h"

VonMises::VonMises() {
}
VonMises::VonMises(int id, MultiaxialMaterial* elastic, double s0, double K)
:MultiaxialElastoPlastic(id, elastic) {
  // Material parameters
  MatParams[0]=s0;
  MatParams[1]=K;
  // Yield/potential surfaces
  fSurfaces.push_back(new VM(s0, K));
  gSurfaces.push_back(new VM(s0, K));
  // Material tag
  // myTag = TAG_MATERIAL_MOHR_COULOMB;
  nHardeningVariables = 1;
}
VonMises::~VonMises() {
}
MultiaxialMaterial* VonMises::get_clone() {
  // Material parameters
  int myID    = this->get_id();
  double s0   = MatParams[ 0];
  double K    = MatParams[ 1];
  // Create clone and return
  VonMises* newClone = new VonMises(myID, myElastic, s0, K);
  return newClone;
}
