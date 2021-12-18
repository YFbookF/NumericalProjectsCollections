/* This file is part of IGsim, a simple c++ simulation library.
 * 
 * Copyright (C) 2016 Like Ma <milkpku@gmail.com>
 *  https://github.com/milkpku/IGsim
 * This Source Code Form is subject to the terms of the Mozilla Public License 
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can 
 * obtain one at http://mozilla.org/MPL/2.0/.
 * This should *NOT* be contained in a IGSIM_*_H ifdef, since it may be defined
 * differently based on when it is included
 */

#include "neohookean_model.h"

template < 
  typename DerivedF, typename ScalarMu, typename ScalarLam, 
  typename ScalarE>
IGSIM_INLINE void sim::neohookean_model(
  const Eigen::PlainObjectBase<DerivedF>& F,
  const ScalarMu& Mu,
  const ScalarLam& Lam,
  ScalarE& energy)
{
  assert(F.cols() == 3 && F.rows() == 3 
    && "neohookean model only suport 3x3 deformation gradient");
  using namespace std;
  double I1 = F.squaredNorm();
  double J = F.determinant();
  double logJ = log(J);
  
  /* energy(F) = \mu/2 * (I_1 - log I_3 - 3) + \lambda/8 * log^2 I_3 */
  energy = 0.5 * Mu * (I1 - 2 * logJ - 3) + 0.5 * Lam * logJ * logJ;
}


template < 
  typename DerivedF, typename ScalarMu, typename ScalarLam, 
  typename ScalarE, typename DerivedP>
IGSIM_INLINE void sim::neohookean_model(
  const Eigen::PlainObjectBase<DerivedF>& F,
  const ScalarMu& Mu,
  const ScalarLam& Lam,
  ScalarE& energy,
  Eigen::PlainObjectBase<DerivedP>& P)
{
  assert(F.cols() == 3 && F.rows() == 3 
    && "neohookean model only suport 3x3 deformation gradient");
  using namespace std;
  double I1 = F.squaredNorm();
  double J = F.determinant();
  double logJ = log(J);

  /* energy(F) = \mu/2 * (I_1 - log I_3 - 3) + \lambda/8 * log^2 I_3 */
  energy = 0.5 * Mu * (I1 - 2 * logJ - 3) + 0.5 * Lam * logJ * logJ;

  /* P(F) = \mu * (F - F^{-T}) + \lambda/2 * log I_3 * F^{-T} */
  DerivedF FinvT = F.inverse().transpose();
  P = Mu * (F - FinvT) + Lam * logJ * FinvT;
}

template < 
  typename DerivedF, typename ScalarMu, typename ScalarLam, 
  typename ScalarDmu, typename ScalarDlam,
  typename DerivedPmu, typename DerivedPlam>
IGSIM_INLINE void sim::neohookean_model(
  const Eigen::PlainObjectBase<DerivedF>& F,
  const ScalarMu& Mu,
  const ScalarLam& Lam,
  ScalarDmu&  demu,
  ScalarDlam& delam,
  Eigen::PlainObjectBase<DerivedPmu>& dPmu,
  Eigen::PlainObjectBase<DerivedPlam>& dPlam)
{
  assert(F.cols() == 3 && F.rows() == 3 
    && "neohookean model only suport 3x3 deformation gradient");
  using namespace std;
  double I1 = F.squaredNorm();
  double J = F.determinant();
  double logJ = log(J);

  /* energy(F) = \mu/2 * (I_1 - log I_3 - 3) + \lambda/8 * log^2 I_3 */
  demu = 0.5 * (I1 - 2 * logJ - 3); 
  delam = 0.5 * logJ * logJ;

  /* P(F) = \mu * (F - F^{-T}) + \lambda/2 * log I_3 * F^{-T} */
  DerivedF FinvT = F.inverse().transpose();
  dPmu = F - FinvT;
  dPlam = logJ * FinvT;
}

#ifdef IGSIM_STATIC_LIBRARY

#endif
