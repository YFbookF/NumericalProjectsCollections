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

#include "material/drucker_prager_new2.h"
#include <iostream>
#include <sstream>
#include <string>
#include "material/drucker_prager_ys.h"
#include "material/tension_cutoff_ys.h"

Matrix DruckerPragerNew2::C(6, 6, 0.);
Matrix DruckerPragerNew2::C3(3, 3, 0.);

DruckerPragerNew2::DruckerPragerNew2()
    : myElastic(0),
      plastic(false),
      inaccurate(0),
      aTrial(0.),
      aConvg(0.),
      fSurfaces(0),
      gSurfaces(0) {
}

DruckerPragerNew2::DruckerPragerNew2(int id, MultiaxialMaterial* elastic,
                                   double c, double phi, double psi, double Kc,
                                   double Kphi, double T)
    : MultiaxialMaterial(id, 0., 0.),
      plastic(false),
      inaccurate(0),
      aTrial(0.),
      aConvg(0.) {
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
  // f surfaces
  fSurfaces.resize(2);
  fSurfaces[0] = new DruckerPragerYS(c, phi, Kc, Kphi);
  fSurfaces[1] = new TensionCutOffYS(T);
  // g surfaces
  gSurfaces.resize(2);
  gSurfaces[0] = new DruckerPragerYS(c, psi, Kc, Kphi);
  gSurfaces[1] = new TensionCutOffYS(T);
}
DruckerPragerNew2::~DruckerPragerNew2() {
  delete myElastic;
}
MultiaxialMaterial* DruckerPragerNew2::get_clone() {
  // Material parameters
  int myID    = this->get_id();
  double c    = MatParams[ 0];
  double phi  = MatParams[ 1];
  double psi  = MatParams[ 2];
  double Kc   = MatParams[ 3];
  double Kphi = MatParams[ 4];
  double T    = MatParams[ 5];
  // Create clone and return
  DruckerPragerNew2* newClone =
    new DruckerPragerNew2(myID, myElastic, c, phi, psi, Kc, Kphi, T);
  return newClone;
}


/**
 * Update stresses given a total strain increment.
 * @param De Vector containing total strain increment.
 */
void DruckerPragerNew2::set_strain(const Vector& De, const double Dt) {
  // material properties
  double E = myElastic->get_param(0);
  double nu = myElastic->get_param(1);
  double phi  = MatParams[ 1];
  double Kc   = MatParams[ 3];

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

  Vector snn = sTrial;

  // report(eTrial3, "De", 8, 3);
  // report(s, "De", 8, 3);

  // Find active surfaces
  std::vector<int> activeS;
  for (unsigned i = 0; i < 2; i++)
    if (fSurfaces[i]->get_f(s, aTrial)>1e-9)
      activeS.push_back(i);

  // cout << fSurfaces[0]->get_f(s, aTrial)<<' '<<aTrial << endl;

  // Elastic (quick return)
  if (activeS.empty()) return;
  // check
  if (activeS.size() == 2) std::cout << "2 active" << std::endl;

  // Plastic - start iterations ----------------------------------
  int iter = 0;
  Matrix A;
  Vector R;
  Vector x;
  Vector DLambda(2, 0.);

  // ---------------------------------------------------------------------------
  // Case 1: Kc = 0, No TC
  // ---------------------------------------------------------------------------
  // cout << fSurfaces[0]->get_f(s, aTrial) << endl;
  // cout << fSurfaces[1]->get_f(s, aTrial) << endl;
  if ((fSurfaces[0]->get_f(s, aTrial) > 1e-9) &&
      (fSurfaces[1]->get_f(s, aTrial) < 1e-9) &&
      (Kc < 1e-9)) {
    while (iter < 20) {
      ++iter;
      int nA = 0;
      A.Resize(3+nA+1, 3+nA+1, 0.);
      R.Resize(3+nA+1, 0.);
      x.Resize(3+nA+1);
      A.Append(C3, 0, 0, 1.0, 1.0);

      R.Append(C3*s-eTrial3, 0, 1.0, 0.0);
      YS* ys =fSurfaces[0];
      YS* ps =gSurfaces[0];
      double DL = DLambda[0];
      A.Append(DL*(ps->get_d2fdsds(s, aTrial)),   0,   0, 1., 1.);
      A.AppendCol(ps->get_dfds(s, aTrial),        0,   3, 1., 0.);
      A.AppendRow(ys->get_dfds(s, aTrial),        3,   0, 1., 0.);

      R.Append(DL*(ps->get_dfds(s, aTrial)),           0, 1., 1.);
      R[3]=ys->get_f(s, aTrial);

      A.Solve(x, -R);
      // report(R, "R");

      // update
      s[0]+=x[0];
      s[1]+=x[1];
      s[2]+=x[2];
      DLambda[0]+=x[3];
      // cout << DLambda[0];
      aTrial = 0.;
      if ((R.Twonorm() < 1.e-12) && (fSurfaces[0]->get_f(s, aTrial) < 1e-9))
        break;
    }
  }

  // ---------------------------------------------------------------------------
  // Case 2: Kc != 0, No TC
  // ---------------------------------------------------------------------------
  // cout << fSurfaces[0]->get_f(s, aTrial) << endl;
  // cout << fSurfaces[1]->get_f(s, aTrial) << endl;
  int num_of_iters = 80;
  if ((fSurfaces[0]->get_f(s, aTrial) > 1e-9) &&
      (fSurfaces[1]->get_f(s, aTrial) < 1e-9) &&
      (Kc > 1e-9)) {
    aTrial = aConvg;
    while (iter < num_of_iters) {
      ++iter;
      int nA = 1;
      A.Resize(3+nA+1, 3+nA+1, 0.);
      R.Resize(3+nA+1, 0.);
      x.Resize(3+nA+1);

      YS* ys =fSurfaces[0];
      YS* ps =gSurfaces[0];
      double DL = DLambda[0];
      const Vector &v = ys->get_dfds(s, aTrial);
      double dhdq = sqrt(2./3.*(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]));
      double D = Kc*6*cos(phi)/(sqrt(3.)*(3-sin(phi)));

      // Row 1
      A.Append(C3, 0, 0, 1.0, 1.0);
      A.Append(DL*(ps->get_d2fdsds(s, aTrial)),   0,   0, 1., 1.);
      A(0, 3)=0.;
      A(1, 3)=0.;
      A(2, 3)=0.;
      A.AppendCol(ps->get_dfds(s, aTrial),        0,   4, 1., 0.);
      R.Append(C3*s-eTrial3, 0, 1.0, 0.0);
      R.Append(DL*(ps->get_dfds(s, aTrial)),           0, 1., 1.);

      // Row 2
      A(3, 0)=0.;
      A(3, 1)=0.;
      A(3, 2)=0.;
      A(3, 3)=1./D;
      A(3, 4)=dhdq;
      R[3]=-aTrial+aConvg+DL*dhdq;

      // Row 3
      A.AppendRow(ys->get_dfds(s, aTrial),        4,   0, 1., 0.);
      A(4, 3)=1.;
      A(4, 4)=0.;
      R[4]=ys->get_f(s, aTrial);

      // report(A, "A");
      // report(R, "R");
      A.Solve(x, -R);
      // report(x, "x");

      // update
      s[0]+=x[0];
      s[1]+=x[1];
      s[2]+=x[2];
      aTrial-=1./D*x[3];
      // aTrial-=x[3];
      DLambda[0]+=x[4];
      // cout << DLambda[0];
      // cout << R.Twonorm() << endl;
      if ((R.Twonorm() < 1.e-12) && (fSurfaces[0]->get_f(s, aTrial) < 1e-9))
        break;
    }
  }

  // cout << iter << endl;
  if (iter == num_of_iters) report(iter, "error");

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

  // double Dt =10000.;
  // double eta = 10000.;
  // sTrial=(snn+(Dt/eta)*sTrial)/(1+Dt/eta);
  // aTrial=(ann+(Dt/eta)*aTrial)/(1+Dt/eta);
}
/**
 * Commit material state.
 */
void DruckerPragerNew2::Commit() {
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
const Matrix& DruckerPragerNew2::get_C() {
  return myElastic->get_C();
}

bool DruckerPragerNew2::isPlastic() {
  return plastic;
}

void DruckerPragerNew2::Save(std::ostream* os) {
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
