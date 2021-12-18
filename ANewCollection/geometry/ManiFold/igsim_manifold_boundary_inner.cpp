/* This file is part of IGsim, a simple c++ simulation library.
 *  https://github.com/milkpku/IGsim
 * Copyright (C) 2017 Like Ma <milkpku@gmail.com>
 * 
 * This Source Code Form is subject to the terms of the Mozilla Public License 
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can 
 * obtain one at http://mozilla.org/MPL/2.0/.
 * This should *NOT* be contained in a IGSIM_*_H ifdef, since it may be defined
 * differently based on when it is included
 */

#include "manifold_boundary_inner.h"

#include <igl/sort.h>
#include <igl/sortrows.h>

#include <vector>

template<typename DerivedT, typename DerivedF, typename DeriveduF>
IGSIM_INLINE void sim::manifold_boundary_inner(
  const Eigen::PlainObjectBase<DerivedT>& T,
  Eigen::PlainObjectBase<DerivedF>& F,
  Eigen::PlainObjectBase<DeriveduF>& uF)
{
  typedef Eigen::VectorXi iVec;
  iVec TA, FA, uFA1, uFA2;
  TA = iVec::Zero(T.rows());
  manifold_boundary_inner(T, TA, F, FA, uF, uFA1, uFA2);
}

template<
  typename DerivedT, typename DerivedTA, 
  typename DerivedF, typename DerivedFA>
IGSIM_INLINE void sim::manifold_boundary_inner(
  const Eigen::PlainObjectBase<DerivedT>& T,
  const Eigen::PlainObjectBase<DerivedTA>& TA,
  Eigen::PlainObjectBase<DerivedF>& F,
  Eigen::PlainObjectBase<DerivedFA>& FA)
{
  Eigen::MatrixXi uF;
  DerivedTA uFA1, uFA2;
  manifold_boundary_inner(T, TA, F, FA, uF, uFA1, uFA2);
}

template<
  typename DerivedT, typename DerivedTA,
  typename DerivedF, typename DerivedFA,
  typename DeriveduF, typename DeriveduFA1, typename DeriveduFA2>
IGSIM_INLINE void sim::manifold_boundary_inner(
  const Eigen::PlainObjectBase<DerivedT>& T,
  const Eigen::PlainObjectBase<DerivedTA>& TA,
  Eigen::PlainObjectBase<DerivedF>& F,
  Eigen::PlainObjectBase<DerivedFA>& FA,
  Eigen::PlainObjectBase<DeriveduF>& uF,
  Eigen::PlainObjectBase<DeriveduFA1>& uFA1,
  Eigen::PlainObjectBase<DeriveduFA2>& uFA2)
{
  assert(T.cols() == 3 || T.cols() == 4 && "T should be col 3 or 4");

  /* generate boundary */
  DerivedT bound;
  Eigen::VectorXi bound_to_tet;

  if (T.cols() == 3) // triangular mesh
  {
    bound.resize(T.rows() * 3, 2);
    bound_to_tet.resize(T.rows() * 3);

    for (int i = 0; i < T.rows(); ++i)
    {
      auto t = T.row(i);
      bound.row(3 * i    ) << t(0), t(1);
      bound.row(3 * i + 1) << t(1), t(2);
      bound.row(3 * i + 2) << t(2), t(0);
      bound_to_tet(3 * i    ) = i;
      bound_to_tet(3 * i + 1) = i;
      bound_to_tet(3 * i + 2) = i;
    }
  }
  else if (T.cols() == 4) // tetrahedron mesh
  {
    bound = DerivedT(T.rows() * 4, 3);
    bound_to_tet.resize(T.rows() * 4);
    for (int i = 0; i < T.rows(); ++i)
    {
      auto t = T.row(i);
      bound.row(4 * i    ) << t(0), t(1), t(2);
      bound.row(4 * i + 1) << t(0), t(2), t(3);
      bound.row(4 * i + 2) << t(0), t(3), t(1);
      bound.row(4 * i + 3) << t(3), t(2), t(1);
      bound_to_tet(4 * i    ) = i;
      bound_to_tet(4 * i + 1) = i;
      bound_to_tet(4 * i + 2) = i;
      bound_to_tet(4 * i + 3) = i;
    }
  }

  /* transfer boudary simplex to be undirected */
  DerivedT undir_bound;
  Eigen::MatrixXi IX;
  igl::sort(bound, 2, true, undir_bound, IX);

  /* sort rows to get duplicate */
  Eigen::VectorXi I;
  DerivedT sort_bound;
  igl::sortrows(undir_bound, true, sort_bound, I);

  /* get unique boundary ids and inner facet ids */
  std::vector<int> unique_id;
  unique_id.reserve(I.size());
  std::vector<int> inner_id1, inner_id2;
  inner_id1.reserve(I.size());
  inner_id2.reserve(I.size());

  int ptr = 0;
  while (ptr < I.size() - 1)
  {
    auto b0 = sort_bound.row(ptr);
    auto b1 = sort_bound.row(ptr + 1);

    if (b0.cwiseEqual(b1).all())
    {
      inner_id1.push_back(I(ptr));
      inner_id2.push_back(I(ptr+1));
      ptr += 2;
    }
    else
    {
      unique_id.push_back(I(ptr));
      ptr++;
    }
  }
  if (ptr == I.size() - 1) unique_id.push_back(I(ptr));

  /* build F and FA */
  F.resize(unique_id.size(), bound.cols());
  uF.resize(inner_id1.size(), bound.cols());
  FA.resize(unique_id.size(), TA.cols());
  uFA1.resize(inner_id1.size(), TA.cols());
  uFA2.resize(inner_id2.size(), TA.cols());

  for (int i = 0; i < unique_id.size(); ++i)
  {
    F.row(i) << bound.row(unique_id[i]);
    FA.row(i) << TA.row(bound_to_tet(unique_id[i]));
  }

  for (int i = 0; i < inner_id1.size(); ++i)
  {
    uF.row(i) << bound.row(inner_id1[i]);
    uFA1.row(i) << TA.row(bound_to_tet(inner_id1[i]));
    uFA2.row(i) << TA.row(bound_to_tet(inner_id2[i]));
  }
}
