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

#include "volume_gradient.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <vector>

template <typename DerivedV, typename DerivedF, typename DerivedG>
IGSIM_INLINE void sim::volume_gradient(
  const Eigen::PlainObjectBase<DerivedV>& V,
  const Eigen::PlainObjectBase<DerivedF>& F,
  Eigen::PlainObjectBase<DerivedG>& G)
{
  assert(V.cols() == 3 && "should be 3D vertices");
  assert(F.cols() == 3 && "should be triangular mesh");

  G.resize(V.rows(), 3);
  G.setZero();
  for (int i = 0; i < F.rows(); ++i)
  {
    /* compute face area normal */
    const Eigen::RowVector3d& a = V.row(F(i, 0));
    const Eigen::RowVector3d& b = V.row(F(i, 1));
    const Eigen::RowVector3d& c = V.row(F(i, 2));
    // here we record 1/3 of face normal, which can be directly added to 
    // each vertices normal
    const Eigen::RowVector3d n = (a-c).cross(b-c) / 6.0;
    
    /* add to corresponding vertex */
    G.row(F(i,0)) += n;
    G.row(F(i,1)) += n;
    G.row(F(i,2)) += n;
  }

}

template <typename DerivedV, typename DerivedF, typename ScalarH>
IGSIM_INLINE void sim::volume_gradient(
  const Eigen::PlainObjectBase<DerivedV>& V,
  const Eigen::PlainObjectBase<DerivedF>& F,
  Eigen::SparseMatrix<ScalarH>& H)
{
  assert(V.cols() == 3 && "should be 3D vertices");
  assert(F.cols() == 3 && "should be triangular mesh");

  /* construct matrix dn = K dr, dn is the difference of facet area normal
   * dn = [dn_x^T, dn_y^T, dn_z^T]^T == dn(N, 3).resize(3N, 1)
   * dr = [dx^T, dy^T, dz^T]^T       == dr(N, 3).resize(3N, 1)
   *
   * for a single triangle, n = (x1 - x0) x (x2 - x0) / 2.
   * n(x0 + dx0) - n(x0) = (x2 - x1) x dx0 / 2. + o(dx0)
   *
   *          |  0  -lz  ly|
   * l x dr = | lz   0  -lx| * dr
   *          |-ly   lx  0 |
   */
  const int N = V.rows();
  H = Eigen::SparseMatrix<ScalarH>(3 * N, 3 * N);
  
  typedef Eigen::Triplet<ScalarH> T;
  std::vector<T> H_coeff;
  H_coeff.clear();
  H_coeff.reserve(F.rows() * 3 * 3 * 9);
  
  /* for each face calculate its influence on H */
  for (int i = 0; i < F.rows(); ++i)
  {
    /* for each vertex of face, calculate its influence on dn */
    for (int j = 0; j < 3; ++j)
    {
      /* calculate (x2 - x1 / 6.0), 2.0 for triangle area and 3.0 for contribution
       * factor from each triangle to its 3 vertices */
      const Eigen::RowVector3d l = (V.row(F(i, (j+2)%3)) - V.row(F(i, (j+1)%3))) / 6.;

      /* restore influence on kth vertex */
      for (int k = 0; k < 3; ++k)
      {
        const int targ = F(i, k);
        const int src = F(i, j);
        H_coeff.push_back(T(targ,       src + N,    -l.z()));
        H_coeff.push_back(T(targ,       src + 2*N,   l.y()));
        H_coeff.push_back(T(targ + N,   src,         l.z()));
        H_coeff.push_back(T(targ + N,   src + 2*N,  -l.x()));
        H_coeff.push_back(T(targ + 2*N, src,        -l.y()));
        H_coeff.push_back(T(targ + 2*N, src + N,     l.x()));
      }
    }
  }

  H.setFromTriplets(H_coeff.begin(), H_coeff.end());
}
