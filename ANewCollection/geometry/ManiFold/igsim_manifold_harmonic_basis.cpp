/* This file is part of IGsim, a simple c++ simulation library.
 * https://github.com/milkpku/IGsim
 * Copyright (C) 2016 Like Ma <milkpku@gmail.com>
 * 
 * This Source Code Form is subject to the terms of the Mozilla Public License 
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can 
 * obtain one at http://mozilla.org/MPL/2.0/.
 * This should *NOT* be contained in a IGSIM_*_H ifdef, since it may be defined
 * differently based on when it is included
 */

#include "manifold_harmonic_basis.h"

#include "manifold_volume.h"

#include <igl/cotmatrix.h>
#include <igl/average_onto_vertices.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <spectra/MatOp/SparseGenRealShiftSolve.h>
#include <spectra/GenEigsRealShiftSolver.h>

#include <type_traits>
#include <iostream>
#include <algorithm>

template <typename DerivedV, typename DerivedF>
IGSIM_INLINE void sim::manifold_harmonic_basis(
  const Eigen::PlainObjectBase<DerivedV>& V,
  const Eigen::PlainObjectBase<DerivedF>& F,
  const int& k,
  Eigen::VectorXcd& S,
  Eigen::MatrixXcd& U)
{
  manifold_harmonic_basis(V, F, k, 0, S, U);
}

template <typename DerivedV, typename DerivedF>
IGSIM_INLINE void sim::manifold_harmonic_basis(
  const Eigen::PlainObjectBase<DerivedV>& V,
  const Eigen::PlainObjectBase<DerivedF>& F,
  const int& k,
  const double& sigma,
  Eigen::VectorXcd& S,
  Eigen::MatrixXcd& U)
{
  using namespace std;
  /* test S, U scalar type, must be complex */

  /* construct Laplacian matrix and mass matrix */
  typedef Eigen::SparseMatrix<double> SpMat;
  SpMat L;
  igl::cotmatrix(V, F, L);
  Eigen::VectorXd Mt, Mv;
  sim::manifold_volume(V, F, Mt);
  igl::average_onto_vertices(V, F, Mt, Mv);

  /* get k smallest eigenvector by Spectra */
  typedef Spectra::SparseGenRealShiftSolve<double> OpType;
  SpMat K = - (Mv.asDiagonal().inverse() * L);
  OpType op(K);
  Spectra::GenEigsRealShiftSolver<double, Spectra::LARGEST_MAGN, OpType>
    eigs(&op, k, min(2 * k + 1, (int)V.rows()), sigma);

  eigs.init();
  int nconv = eigs.compute();

  if (eigs.info() != Spectra::SUCCESSFUL)
  {
    cerr << "Faile to get manifold harmonic basis\n";
    return;
  }

  /* return eigenvalue and eigenvectors */
  S = eigs.eigenvalues();
  U = eigs.eigenvectors();

}
