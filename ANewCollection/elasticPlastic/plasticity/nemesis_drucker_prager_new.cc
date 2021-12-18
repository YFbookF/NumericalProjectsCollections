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

#include "material/drucker_prager_new.h"
#include <iostream>
#include <sstream>
#include <string>
#include "material/drucker_prager_ys.h"
#include "material/tension_cutoff_ys.h"

Matrix DruckerPragerNew::C(6, 6, 0.);
Matrix DruckerPragerNew::C3(3, 3, 0.);

DruckerPragerNew::DruckerPragerNew()
    : myElastic(0),
      plastic(false),
      inaccurate(0),
      aTrial(0.),
      aConvg(0.),
      fSurfaces(0),
      gSurfaces(0) {
}

DruckerPragerNew::DruckerPragerNew(int id, MultiaxialMaterial* elastic,
                                   double c, double phi, double psi, double Kc,
                                   double Kphi, double T)
    : MultiaxialMaterial(id, 0., 0.) {
  // Get the elastic part
  myElastic = elastic->get_clone();
  MatParams[30] = myElastic->get_param(30);
  MatParams[31] = myElastic->get_param(31);

  // Material properties
  MatParams[0] = c;
  MatParams[1] = phi;
  MatParams[2] = psi;
  MatParams[3] = Kc;
  MatParams[4] = Kphi;
  MatParams[5] = T;
  // plastic info
  plastic = false;
  inaccurate = 0;
  // Hardening variables
  aTrial = 0.;
  aConvg = 0.;
  // f surfaces
  fSurfaces.resize(2);
  fSurfaces[0] = new DruckerPragerYS(c, phi, Kc, Kphi);
  fSurfaces[1] = new TensionCutOffYS(T);
  // g surfaces
  gSurfaces.resize(2);
  gSurfaces[0]=new DruckerPragerYS(c, psi, Kc, Kphi);
  gSurfaces[1]=new TensionCutOffYS(T);
}
DruckerPragerNew::~DruckerPragerNew() {
  delete myElastic;
}
MultiaxialMaterial* DruckerPragerNew::get_clone() {
  // Material parameters
  int id      = this->get_id();
  double c    = MatParams[ 0];
  double phi  = MatParams[ 1];
  double psi  = MatParams[ 2];
  double Kc   = MatParams[ 3];
  double Kphi = MatParams[ 4];
  double T    = MatParams[ 5];
  // Create clone and return
  DruckerPragerNew* clone =
    new DruckerPragerNew(id, myElastic, c, phi, psi, Kc, Kphi, T);
  return clone;
}


/**
 * Update stresses given a total strain increment.
 * @param De Vector containing total strain increment.
 */
void DruckerPragerNew::set_strain(const Vector& De, const double Dt) {
  // material properties
  double E = myElastic->get_param(0);
  double nu= myElastic->get_param(1);

  // elasticity matrix
  C3(0, 0) =   1/E;
  C3(0, 1) = -nu/E;
  C3(0, 2) = -nu/E;
  C3(1, 0) = -nu/E;
  C3(1, 1) =   1/E;
  C3(1, 2) = -nu/E;
  C3(2, 0) = -nu/E;
  C3(2, 1) = -nu/E;
  C3(2, 2) =   1/E;

  // spectral decomposition
  Vector s(3), De3(3);
  Matrix sV(3, 3), eV(3, 3);
  aTrial = aConvg;
  eTrial = eTotal+De;
  sTrial = sConvg+(this->get_C())*De;
  spectralDecomposition(sTrial, s, sV);
  Vector eTrial3 = C3*s;

  // report(eTrial3, "De", 8, 3);
  // report(s, "De", 8, 3);

  // Find active surfaces
  std::vector<int> activeS;
  for (unsigned i = 0; i < 2; i++)
    if (fSurfaces[i]->get_f(s, aTrial)>1e-9)
      activeS.push_back(i);

  // cout << fSurfaces[0]->get_f(s, aTrial) << ' ' << aTrial << endl;
  // Elastic (quick return)
  if (activeS.empty())
    return;

  if (activeS.size() == 2)
    std::cout << "2 active" << std::endl;

  // Plastic - start iterations ----------------------------------
  int iter = 0;
  Matrix A;
  Vector R;
  Vector x;
  Vector DLambda(2, 0.);
  while (iter < 20) {
    // increase iteration number
    ++iter;

    // build system
    int nA = static_cast<int>(activeS.size());
    A.Resize(3+nA+1, 3+nA+1, 0.);
    R.Resize(3+nA+1, 0.);
    x.Resize(3+nA+1);
    A.Append(C3, 0, 0, 1.0, 1.0);

    R.Append(C3*s-eTrial3, 0, 1.0, 0.0);
    for (int i = 0; i < nA; i++) {
      YS* ys =fSurfaces[activeS[i]];
      YS* ps =gSurfaces[activeS[i]];
      double DL = DLambda[activeS[i]];
      A.Append(DL*(ps->get_d2fdsds(s, aTrial)),   0,   0, 1., 1.);
      A.AppendCol(ps->get_dfds(s, aTrial),        0, 3+i, 1., 0.);
      A.AppendRow(ys->get_dfds(s, aTrial),      3+i,   0, 1., 0.);
      R.Append(DL*(ps->get_dfds(s, aTrial)),           0, 1., 1.);
      R[3+i]=-(ys->get_f(s, aTrial));
      A.AppendCol(DL*(ps->get_f2dkds(s, aTrial)), 0, 3+nA, 1., 1.);
      A(3+i, 3+nA)+=ys->get_dfdk(s, aTrial);
    }
    // hardening
    A.AppendRow(DLambda[0]*(gSurfaces[0]->get_dfds(s, aTrial)),
                            3+nA, 0, 1., 0.);

    A(3+nA, 3+nA-1) = -EL.get_h(gSurfaces[0]->get_dfds(s, aTrial));
    // A(3+nA, 3+nA-1)=-DLambda[0]*sqrt(2./3.);
    A(3+nA, 3+nA) = 1.;

    R[3+nA]=-aTrial+aConvg
            +DLambda[0]*(EL.get_h(gSurfaces[0]->get_dfds(s, aTrial)));
    // R[3+nA]=-aTrial+aConvg+DLambda[0]*sqrt(2./3.);

    // solve
    // report(A, "A", 12, 4);
    // report(R, "R");
    A.Solve(x, R);
    // report(x, "x");

    // update
    s[0]+=x[0];
    s[1]+=x[1];
    s[2]+=x[2];
    for (int i = 0; i < nA; i++)
      DLambda[activeS[i]]+=x[3+i];
    aTrial+=x[3+nA];

    // check f
    int check = 0;
    for (int i = 0; i < nA; i++) {
      if (fSurfaces[activeS[i]]->get_f(s, aTrial)>1e-9)
        check++;
    }

    // check dL < 0
    bool restart = false;
    if (check == 0) {
      for (int i = 0; i < nA; i++) {
        if (DLambda[activeS[i]] < 0.) {
          std::cout << "RESTART " << nA << '\n';
          activeS.erase(activeS.begin()+i);
          restart = true;
        }
      }
    }
    if (restart) {
      DLambda.Clear();
      iter = 0;
    }
    // check
    R.Append(C3*s-eTrial3, 0, 1.0, 0.0);
    for (int i = 0; i < nA; i++) {
      YS* ys =fSurfaces[activeS[i]];
      YS* ps =gSurfaces[activeS[i]];
      double DL = DLambda[activeS[i]];
      R.Append(DL*(ps->get_dfds(s, aTrial)),          0, 1., 1.);
      R[3+i]=-(ys->get_f(s, aTrial));
    }

    if (check == 0 && restart == false) break;
  }
  if (R.Twonorm()>1.e-12) report(R, "error");
  // report(R);
  // cout << fSurfaces[0]->get_f(s, 0.) << endl;
  // cout << fSurfaces[1]->get_f(s, 0.) << endl;
  // coordinate transformation
  sTrial[0] = s[0]*sV(0, 0)*sV(0, 0)
             +s[1]*sV(1, 0)*sV(1, 0)
             +s[2]*sV(2, 0)*sV(2, 0);
  sTrial[1] = s[0]*sV(0, 1)*sV(0, 1)
             +s[1]*sV(1, 1)*sV(1, 1)
             +s[2]*sV(2, 1)*sV(2, 1);
  sTrial[2] = s[0]*sV(0, 2)*sV(0, 2)
             +s[1]*sV(1, 2)*sV(1, 2)
             +s[2]*sV(2, 2)*sV(2, 2);
  sTrial[3] = s[0]*sV(0, 0)*sV(0, 1)
             +s[1]*sV(1, 0)*sV(1, 1)
             +s[2]*sV(2, 0)*sV(2, 1);
  sTrial[4] = s[0]*sV(0, 1)*sV(0, 2)
             +s[1]*sV(1, 1)*sV(1, 2)
             +s[2]*sV(2, 1)*sV(2, 2);
  sTrial[5] = s[0]*sV(0, 0)*sV(0, 2)
             +s[1]*sV(1, 0)*sV(1, 2)
             +s[2]*sV(2, 0)*sV(2, 2);
}
/**
 * Commit material state.
 */
void DruckerPragerNew::Commit() {
  // report(inaccurate);
  inaccurate = 0;
  eTotal = eTrial;  /// @todo
  sConvg = sTrial;
  aConvg = aTrial;
}
/**
 * Get tangent material matrix.
 * @todo fill it
 * @return A reference to the tangent material matrix.
 */
const Matrix& DruckerPragerNew::get_C() {
  return myElastic->get_C();
}
bool DruckerPragerNew::isPlastic() {
  return plastic;
}

void DruckerPragerNew::Save(std::ostream* os) {
  // start saving
  (*os) << "{";
  (*os) << "\"data\":{";
  (*os) << "\"sigm\":"    << sConvg << ',';
  (*os) << "\"epst\":"    << eTotal << ',';
//  (*os) << "\"epsp\":"    << ePConvg << ',';
  (*os) << "\"epsv\":"    << eTotal[0]+eTotal[1]+eTotal[2] << ',';
  (*os) << "\"p\":"       << sConvg.p() << ',';
  (*os) << "\"q\":"       << sConvg.q();
  (*os) << "}";
  // finalize
  (*os) << "}";
}
