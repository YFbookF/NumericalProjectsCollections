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

#include "material/hardening.h"

Hardening::Hardening() {
}
double Hardening::get_h(const Vector& v) {
  return sqrt(2./3.*(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]));
}
const Vector& Hardening::get_hds(const Vector& /*sigma*/,
                                const double /*kappa*/) {
  static Vector a(3, 0.);
  return a;
}
double Hardening::get_hdk(const Vector& /*sigma*/, const double /*kappa*/) {
  return 0;
}

