/* This file is part of IGsim, a simple c++ simulation library.
 *  https://github.com/milkpku/IGsim
 * Copyright (C) 2016 Like Ma <milkpku@gmail.com>
 * 
 * This Source Code Form is subject to the terms of the Mozilla Public License 
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can 
 * obtain one at http://mozilla.org/MPL/2.0/.
 * This should *NOT* be contained in a IGSIM_*_H ifdef, since it may be defined
 * differently based on when it is included
 */

#include "neohookean_model_dPiola.h"

#include <vector>

template < 
  typename DerivedF, typename DeriveddF, 
  typename ScalarMu, typename ScalarLam, 
  typename DeriveddP>
IGSIM_INLINE void sim::neohookean_model_dPiola(
  const Eigen::PlainObjectBase<DerivedF>& F,
  const Eigen::PlainObjectBase<DeriveddF>& dF,
  const ScalarMu& Mu,
  const ScalarLam& Lam,
  Eigen::PlainObjectBase<DeriveddP>& dP)
{
  assert(F.cols() == 3 && F.rows() == 3 
    && "neohookean model only suport 3x3 deformation gradient");
  assert(dF.cols() == 3 && dF.rows() == 3 
    && "neohookean model only suport 3x3 deformation gradient");
  using namespace std;
  double J = F.determinant();
  double logJ = log(J);
  
  DerivedF FinvT = F.inverse().transpose();
  double tr = FinvT.cwiseProduct(dF).sum();
  dP = Mu * dF 
    + (Mu - Lam * logJ) * FinvT * dF.transpose() * FinvT 
    + Lam * tr * FinvT;
}

template <
  typename DerivedF, typename DeriveddF_T, typename DeriveddF_A,
  typename ScalarMu, typename ScalarLam,
  typename DeriveddP_T, typename DeriveddP_A>
IGSIM_INLINE void sim::neohookean_model_dPiola(
  const Eigen::PlainObjectBase<DerivedF>& F,
  const std::vector<DeriveddF_T, DeriveddF_A>& dF,
  const ScalarMu& Mu,
  const ScalarLam& Lam,
  std::vector<DeriveddP_T, DeriveddP_A>& dP)
{
  assert(F.cols() == 3 && F.rows() == 3 
    && "neohookean model only suport 3x3 deformation gradient");
  
  using namespace std;
  double J = F.determinant();
  double logJ = log(J);
  
  DerivedF FinvT = F.inverse().transpose();

  dP.clear();
  dP.reserve(dF.size());
  for (int i = 0; i < dF.size(); i++)
  {
    assert(dF[i].cols() == 3 && dF[i].rows() == 3
      && "neohookean model only suport 3x3 deformation gradient");
    double tr = FinvT.cwiseProduct(dF[i]).sum();
    DeriveddP_T dP_tmp;
    dP_tmp = Mu * dF[i] 
        + (Mu - Lam * logJ) * FinvT * dF[i].transpose() * FinvT 
        + Lam * tr * FinvT;
    dP.push_back(dP_tmp);
  }
}
