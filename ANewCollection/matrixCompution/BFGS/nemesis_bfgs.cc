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

#include "algorithm/bfgs.h"
#include "analysis/analysis.h"
#include "control/control.h"
#include "convergence/convergence_norm.h"
#include "numeric/vector.h"
#include "soe/soe.h"

/// @todo: Check initialization list
BFGS::BFGS(int m_)
    : m(m_),
      etaMin(0.),
      etaMax(0.),
      rTol(0.),
      maxIter(0),
      isLineSearchActive(false),
      s(m_),
      y(m_) {
  myTag = TAG_NONE;
}

BFGS::BFGS(int m_, double etaMin_, double etaMax_, double rTol_, int maxIter_)
    : m(m_),
      etaMin(etaMin_),
      etaMax(etaMax_),
      rTol(rTol_),
      maxIter(maxIter_),
      isLineSearchActive(true),
      s(m_),
      y(m_) {
  myTag = TAG_NONE;
}

BFGS::~BFGS() {
}

int BFGS::SolveStep(int /*n*/) {
  int size = pA->get_model()->get_num_eqns();
  Vector resOld(size);
  Vector resNew(size);
  Vector q(size);
  Vector du(size);
  Vector a(m);
  Vector r(m);
  for (int j = 0; j < m; j++) {
    s[j].Resize(size, 0.);
    y[j].Resize(size, 0.);
  }

  // Predictor phase
  pA->get_control()->FormTangent();
  pA->get_control()->Predict();
  pA->get_convergence_norm()->NewStep();
  du = pA->get_soe()->get_X();
  resOld = pA->get_soe()->get_B();
  resOld*=-1.;
  pA->get_control()->FormResidual(pA->get_control()->get_lambda());

  // Corrector phase
  int k = 0;
  int check;
  while ((check = pA->get_convergence_norm()->Update()) > 0) {
    resNew = pA->get_soe()->get_B();
    resNew*=-1.;
    y[k]=resNew-resOld;
    s[k]=du;

    q=-resNew;
    // for (int i = k-1;i >= k-m;i--)          // for l-bfgs
    for (int i = k-1; i >= 0; i--) {
      // cout << "A "<<i<<" "<<k-i-1 << endl;   // for l-bfgs
      // if (i <= 0) continue;                // for l-bfgs
      r[i+1]=1/(y[i+1]*s[i+1]);
      a[i]=r[i+1]*s[i+1]*q;
      q-=a[i]*y[i+1];
    }

    pA->get_soe()->set_B(q);
    pA->get_soe()->solve();
    du = pA->get_soe()->get_X();
    // for (int i = k-m;i <= k-1;i++)          // for l-bfgs
    for (int i = 1; i <= k; i++) {
      // cout << "B "<<i<<" "<<k-i-1 << endl;   // for l-bfgs
      // if (i <= 0) continue;                // for l-bfgs
      double b = r[i]*y[i]*du;
      du+=s[i]*(a[i-1]-b);
    }

    pA->get_soe()->set_X(du);
    pA->get_control()->Correct();
    pA->get_control()->FormResidual(pA->get_control()->get_lambda());
    double s0=-s[k]*resNew;
    double s1=-s[k]*(pA->get_soe()->get_B());
    if (isLineSearchActive) this->LineSearch(s0, s1, du);
    resOld = resNew;
    ++k;
  }
  return check;
}

void BFGS::LineSearch(double s0, double sj, const Vector& du) {
  Vector dx = du;
  int k = 0;
  double eta = 1.0;
  double etaP = 1.0;

  double r0;
  if (!num::tiny(s0))  r0 = fabs(sj/s0);
  else        r0 = 0.;
  double rj = r0;
  while (k < maxIter && rj > rTol) {
    eta = etaP*s0/(s0-sj);
    if (eta < etaMin) eta = etaMin;
    if (eta>etaMax) eta = etaMax;
    if (rj > r0    ) eta = 1.0;
    dx*=eta;
    pA->get_soe()->set_X(dx);
    pA->get_control()->Correct();
    pA->get_control()->FormResidual(pA->get_control()->get_lambda());
    sj = du*(pA->get_soe()->get_B());
    rj = fabs(sj/s0);
    etaP = eta;
    ++k;
  }
  dx = eta*du;
  pA->get_soe()->set_X(dx);
}
