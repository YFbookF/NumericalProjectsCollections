/* This file is part of IGsim, a simple c++ simulation library.
 * 
 * Copyright (C) 2016 Like Ma <milkpku@gmail.com>
 * 
 * This Source Code Form is subject to the terms of the Mozilla Public License 
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can 
 * obtain one at http://mozilla.org/MPL/2.0/.
 * This should *NOT* be contained in a IGSIM_*_H ifdef, since it may be defined
 * differently based on when it is included
 */

#include "elastic_neohookean.h"
#include "neohookean_model.h"
#include "neohookean_model_dPiola.h"
#include "average_onto_faces_mat.h"

#include <igl/parallel_for.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/StdVector>

/* initialization of Bm and W */
template <
  typename DerivedV, typename DerivedT, 
  typename DerivedBm_T, typename DerivedBm_A, typename DerivedW>
IGSIM_INLINE void sim::elastic_neohookean(
  const Eigen::PlainObjectBase<DerivedV>& V,
  const Eigen::PlainObjectBase<DerivedT>& T,
  std::vector<DerivedBm_T, DerivedBm_A>& Bm,
  Eigen::PlainObjectBase<DerivedW>& W)
{
  assert(V.cols() == 3 && "Vertices dim must be 3");
  assert(T.cols() == 4 && "Tetra dim must be 4");
  typedef Eigen::Matrix3d Mat3;
  
  Bm.clear();
  Bm.reserve(T.rows());
  W = DerivedW::Zero(T.rows());

  for(int i = 0; i < T.rows(); i++)
  {
    Mat3 Ds_t;
    Ds_t << V.row(T(i, 0)), V.row(T(i, 1)), V.row(T(i, 2));
    Ds_t.rowwise() -= V.row(T(i, 3));

    W(i) = Ds_t.determinant() / 6.0;
    DerivedBm_T Bm_tmp = Ds_t.inverse().transpose();
    
    Bm.push_back(Bm_tmp);
  }

  assert( W.minCoeff() > 0 && "negtive volume in W");
}

/* convenient API for energy without start & num */
template <
  typename DerivedV, typename DerivedT,
  typename DerivedBm_T, typename DerivedBm_A, typename DerivedW,
  typename DerivedMu, typename DerivedLam,
  typename DerivedE>
IGSIM_INLINE void sim::elastic_neohookean(
  const Eigen::PlainObjectBase<DerivedV>& V,
  const Eigen::PlainObjectBase<DerivedT>& T, 
  const std::vector<DerivedBm_T, DerivedBm_A>& Bm,
  const Eigen::PlainObjectBase<DerivedW>& W,
  const Eigen::PlainObjectBase<DerivedMu>& _Mu,
  const Eigen::PlainObjectBase<DerivedLam>& _Lam,
  DerivedE&  energy)
{
  int start = 0;
  int num = T.rows();
  elastic_neohookean(V, T, Bm, W, _Mu, _Lam, start, num, energy);
}

/* convenient API for energy, grad without start & num */
template <
  typename DerivedV, typename DerivedT, 
  typename DerivedBm_T, typename DerivedBm_A, typename DerivedW,
  typename DerivedMu, typename DerivedLam,
  typename DerivedE, typename DerivedG>
IGSIM_INLINE void sim::elastic_neohookean(
  const Eigen::PlainObjectBase<DerivedV>& V,
  const Eigen::PlainObjectBase<DerivedT>& T, 
  const std::vector<DerivedBm_T, DerivedBm_A>& Bm,
  const Eigen::PlainObjectBase<DerivedW>& W,
  const Eigen::PlainObjectBase<DerivedMu>& _Mu,
  const Eigen::PlainObjectBase<DerivedLam>& _Lam,
  DerivedE&  energy,
  Eigen::PlainObjectBase<DerivedG>& grad)
{
  int start = 0;
  int num = T.rows();
  elastic_neohookean(V, T, Bm, W, _Mu, _Lam, start, num, energy, grad);
}

/* convenient API for energy, grad and hess without start & num */
template <
  typename DerivedV, typename DerivedT, 
  typename DerivedBm_T, typename DerivedBm_A, typename DerivedW,
  typename DerivedMu, typename DerivedLam,
  typename DerivedE, typename DerivedG, typename ScalarH>
IGSIM_INLINE void sim::elastic_neohookean(
  const Eigen::PlainObjectBase<DerivedV>& V,
  const Eigen::PlainObjectBase<DerivedT>& T, 
  const std::vector<DerivedBm_T, DerivedBm_A>& Bm,
  const Eigen::PlainObjectBase<DerivedW>& W,
  const Eigen::PlainObjectBase<DerivedMu>& _Mu,
  const Eigen::PlainObjectBase<DerivedLam>& _Lam,
  DerivedE&  energy,
  Eigen::PlainObjectBase<DerivedG>& grad,
  Eigen::SparseMatrix<ScalarH>& H)
{
  int start = 0;
  int num = T.rows();
  std::vector<Eigen::Triplet<ScalarH>> hess_vec;
  elastic_neohookean(V, T, Bm, W, _Mu, _Lam, start, num, energy, grad, hess_vec);
  H.setZero();
  H.resize(V.size(), V.size());
  H.setFromTriplets(hess_vec.begin(), hess_vec.end());
  H.makeCompressed();
}

/* main body to compute energy */
template <
  typename DerivedV, typename DerivedT,
  typename DerivedBm_T, typename DerivedBm_A, typename DerivedW,
  typename DerivedMu, typename DerivedLam,
  typename DerivedE>
IGSIM_INLINE void sim::elastic_neohookean(
  const Eigen::PlainObjectBase<DerivedV>& V,
  const Eigen::PlainObjectBase<DerivedT>& T, 
  const std::vector<DerivedBm_T, DerivedBm_A>& Bm,
  const Eigen::PlainObjectBase<DerivedW>& W,
  const Eigen::PlainObjectBase<DerivedMu>& _Mu,
  const Eigen::PlainObjectBase<DerivedLam>& _Lam,
  const int start,
  const int num,
  DerivedE&  energy)
{
  using namespace std;
  assert(V.cols() == 3 && "Vertices dim must be 3");
  assert(T.cols() == 4 && "Tetra dim must be 4");
  assert(Bm.size() == T.rows() && "Bm size missmatch T rows");
  assert(W.size() == T.rows() && "W size missmatch T rows");
  assert(_Mu.size() == _Lam.size() && "Mu size missmatch Lam size");
  
  DerivedMu Mu;
  DerivedLam Lam;
  if (_Mu.size() == V.rows())
  {
    /* average parameter \mu and \lambda from vertices to tetrahedrons */
    Eigen::SparseMatrix<DerivedE> Proj;
    sim::average_onto_faces_mat(V, T, Proj);

    Mu = Proj * _Mu;
    Lam = Proj * _Lam;
  }
  else if (_Mu.size() == T.rows())
  {
    Mu = _Mu;
    Lam = _Lam;
  }
  else
  {
    assert(false && "Mu size missmatch V.rows() and T.rows()");
    return;
  }

  typedef Eigen::Matrix3d Mat3;

  /* loop for of each tet */
  int end = start + num;
  assert(end <= T.rows() && "start + num excceeds max T rows");

  energy = 0;

  /* loop for each tet */
  for (int i = 0; i < end; i++)
  {
    /* compute deformation pos Ds = [v0-v3, v1-v3, v2-v3] */
    Mat3 Ds_t;
    Ds_t << V.row(T(i, 0)), V.row(T(i, 1)), V.row(T(i, 2));
    Ds_t.rowwise() -= V.row(T(i, 3));

    /* compute deformation gradient F */
    Mat3 F;
    F = Ds_t.transpose() * Bm[i];

    /* call neohookean model to get energy */
    DerivedE tmp_e;
    sim::neohookean_model(F, Mu(i), Lam(i), tmp_e);

    /* accumulate energy */
    energy += W(i) * tmp_e;
    if (energy != energy) return;
  }
}

/* main body to compute energy, grad */
template <
  typename DerivedV, typename DerivedT, 
  typename DerivedBm_T, typename DerivedBm_A, typename DerivedW,
  typename DerivedMu, typename DerivedLam,
  typename DerivedE, typename DerivedG>
IGSIM_INLINE void sim::elastic_neohookean(
  const Eigen::PlainObjectBase<DerivedV>& V,
  const Eigen::PlainObjectBase<DerivedT>& T,
  const std::vector<DerivedBm_T, DerivedBm_A>& Bm,
  const Eigen::PlainObjectBase<DerivedW>& W,
  const Eigen::PlainObjectBase<DerivedMu>& _Mu,
  const Eigen::PlainObjectBase<DerivedLam>& _Lam,
  const int start,
  const int num,
  DerivedE& energy,
  Eigen::PlainObjectBase<DerivedG>& grad)
{
  using namespace std;
  assert(V.cols() == 3 && "Vertices dim must be 3");
  assert(T.cols() == 4 && "Tetra dim must be 4");
  assert(Bm.size() == T.rows() && "Bm size missmatch T rows");
  assert(W.size() == T.rows() && "W size missmatch T rows");
  assert(_Mu.size() == _Lam.size() && "Mu size missmatch Lam size");
  
  DerivedMu Mu;
  DerivedLam Lam;
  if (_Mu.size() == V.rows())
  {
    /* average parameter \mu and \lambda from vertices to tetrahedrons */
    Eigen::SparseMatrix<DerivedE> Proj;
    sim::average_onto_faces_mat(V, T, Proj);

    Mu = Proj * _Mu;
    Lam = Proj * _Lam;
  }
  else if (_Mu.size() == T.rows())
  {
    Mu = _Mu;
    Lam = _Lam;
  }
  else
  {
    assert(false && "Mu size missmatch V.rows() and T.rows()");
    return;
  }

  typedef Eigen::Matrix3d Mat3;

  /* loop for of each tet */
  int end = start + num;
  assert(end <= T.rows() && "start + num excceeds max T rows");

  energy = 0;
  grad.resizeLike(V);
  grad.setZero();

  for (int i = start; i < end; i++)
  {
    /* compute deformation pos Ds = [v0-v3, v1-v3, v2-v3] */
    Mat3 Ds_t;
    Ds_t << V.row(T(i, 0)), V.row(T(i, 1)), V.row(T(i, 2));
    Ds_t.rowwise() -= V.row(T(i, 3));

    /* compute deformation gradient F */
    Mat3 F;
    F = Ds_t.transpose() * Bm[i];

    /* call neohookean model to get energy */
    double tmp_e;
    Mat3 P;
    sim::neohookean_model(F, Mu(i), Lam(i), tmp_e, P);

    /* accumulate energy */
    energy += W(i) * tmp_e;
    if (energy != energy) return;

    /* accumulate forces */
    Mat3 _H = W(i) * Bm[i] * P.transpose();  /* change to be row wise force */
    grad.row(T(i, 0)) += _H.row(0);
    grad.row(T(i, 1)) += _H.row(1);
    grad.row(T(i, 2)) += _H.row(2);
    grad.row(T(i, 3)) -= _H.colwise().sum();
  }
}

/* main body to compute energy, grad and hess_vec */
template <
  typename DerivedV, typename DerivedT, 
  typename DerivedBm_T, typename DerivedBm_A, typename DerivedW,
  typename DerivedMu, typename DerivedLam,
  typename DerivedE, typename DerivedG, typename ScalarH>
IGSIM_INLINE void sim::elastic_neohookean(
  const Eigen::PlainObjectBase<DerivedV>& V,
  const Eigen::PlainObjectBase<DerivedT>& T,
  const std::vector<DerivedBm_T, DerivedBm_A>& Bm,
  const Eigen::PlainObjectBase<DerivedW>& W,
  const Eigen::PlainObjectBase<DerivedMu>& _Mu,
  const Eigen::PlainObjectBase<DerivedLam>& _Lam,
  const int start,
  const int num,
  DerivedE& energy,
  Eigen::PlainObjectBase<DerivedG>& grad,
  std::vector<Eigen::Triplet<ScalarH>>& hess_vec)
{
  using namespace std;
  assert(V.cols() == 3 && "Vertices dim must be 3");
  assert(T.cols() == 4 && "Tetra dim must be 4");
  assert(Bm.size() == T.rows() && "Bm size missmatch T rows");
  assert(W.size() == T.rows() && "W size missmatch T rows");
  assert(_Mu.size() == _Lam.size() && "Mu size missmatch Lam size");
  
  DerivedMu Mu;
  DerivedLam Lam;
  if (_Mu.size() == V.rows())
  {
    /* average parameter \mu and \lambda from vertices to tetrahedrons */
    Eigen::SparseMatrix<DerivedE> Proj;
    sim::average_onto_faces_mat(V, T, Proj);

    Mu = Proj * _Mu;
    Lam = Proj * _Lam;
  }
  else if (_Mu.size() == T.rows())
  {
    Mu = _Mu;
    Lam = _Lam;
  }
  else
  {
    assert(false && "Mu size missmatch V.rows() and T.rows()");
    return;
  }

  typedef Eigen::Matrix3d Mat3;

  /* loop for of each tet */
  int end = start + num;
  assert(end <= T.rows() && "start + num excceeds max T rows");

  energy = 0;
  grad.resizeLike(V);
  grad.setZero();
  hess_vec.clear();
  hess_vec.reserve(144 * num);

  for (int i = start; i < end; i++)
  {
    /* compute deformation pos Ds = [v0-v3, v1-v3, v2-v3] */
    Mat3 Ds_t;
    Ds_t << V.row(T(i, 0)), V.row(T(i, 1)), V.row(T(i, 2));
    Ds_t.rowwise() -= V.row(T(i, 3));

    /* compute deformation gradient F */
    Mat3 F;
    F = Ds_t.transpose() * Bm[i];

    /* call neohookean model to get energy */
    double tmp_e;
    Mat3 P;
    sim::neohookean_model(F, Mu(i), Lam(i), tmp_e, P);

    /* accumulate energy */
    energy += W(i) * tmp_e;
    if (energy != energy) return;

    /* accumulate forces */
    Mat3 _H = W(i) * Bm[i] * P.transpose();  /* change to be row wise force */
    grad.row(T(i, 0)) += _H.row(0);
    grad.row(T(i, 1)) += _H.row(1);
    grad.row(T(i, 2)) += _H.row(2);
    grad.row(T(i, 3)) -= _H.colwise().sum();

    /* accumulate stiffness matrix */
    double a = W(i) * Mu(i);
    double J = F.determinant();
    double b = W(i) * (Mu(i) - Lam(i) * std::log(J));
    double c = W(i) * Lam(i);

    Mat3 A = Bm[i] * Bm[i].transpose();
    Mat3 B = Ds_t.inverse();
    double H[12][12];

    /* \par H_kl / \par D_sj = a \delta{s, k} A_jl + b B_kj B_sl + c B_sj B_kl */

    for (int l = 0; l < 3; l++)
      for (int k = 0; k < 3; k++)
        for (int j = 0; j < 3; j++)
          for (int s = 0; s < 3; s++)
            H[3 * l + k][3 * j + s] = b * B(k, j) * B(s, l) + c * B(s, j) * B(k, l);

    for (int l = 0; l < 3; l++)
      for (int s = 0; s < 3; s++)
        for (int j = 0; j < 3; j++)
          H[3 * l + s][3 * j + s] += a * A(j, l);

    for (int s = 0; s < 9; s++)
    {
      H[9][s] = H[s][9] = -(H[s][0] + H[s][3] + H[s][6]);
      H[10][s] = H[s][10] = -(H[s][1] + H[s][4] + H[s][7]);
      H[11][s] = H[s][11] = -(H[s][2] + H[s][5] + H[s][8]);
    }

    H[9][9] = -(H[9][0] + H[9][3] + H[9][6]);
    H[9][10] = H[10][9] = -(H[9][1] + H[9][4] + H[9][7]);
    H[9][11] = H[11][9] = -(H[9][2] + H[9][5] + H[9][8]);
    H[10][10] = -(H[10][1] + H[10][4] + H[10][7]);
    H[10][11] = H[11][10] = -(H[10][2] + H[10][5] + H[10][8]);
    H[11][11] = -(H[11][2] + H[11][5] + H[11][8]);

    int addr[12];

    if (DerivedV::Options == Eigen::ColMajor)
    {
      /* use col-major matrix */
      int num_V = V.rows();
      for (int s = 0; s < 4; s++)
      {
        int ind = T(i, s);
        addr[3 * s] = ind;
        addr[3 * s + 1] = ind + num_V;
        addr[3 * s + 2] = ind + 2 * num_V;
      }
    }
    else
    {
      /* use row-major matrix */
      for (int s = 0; s < 4; s++)
      {
        int ind = T(i, s);
        ind *= 3;
        addr[3 * s] = ind;
        addr[3 * s + 1] = ind + 1;
        addr[3 * s + 2] = ind + 2;
      }
    }

    for (int k = 0; k < 12; k++)
      for (int l = 0; l < 12; l++)
        hess_vec.emplace_back(addr[k], addr[l], H[k][l]);
  }
}

/* main body to compute energy, grad and hess on basis */
template<
  typename DerivedV, typename DerivedT, 
  typename DerivedBm_T, typename DerivedBm_A, typename DerivedW,
  typename DerivedMu, typename DerivedLam, typename DerivedB,
  typename DerivedE, typename DerivedG, typename DerivedH>
IGSIM_INLINE void sim::elastic_neohookean(
  const Eigen::PlainObjectBase<DerivedV>& V,
  const Eigen::PlainObjectBase<DerivedT>& T,
  const std::vector<DerivedBm_T, DerivedBm_A>& Bm,
  const Eigen::PlainObjectBase<DerivedW>& W,
  const Eigen::PlainObjectBase<DerivedMu>& _Mu,
  const Eigen::PlainObjectBase<DerivedLam>& _Lam,
  const Eigen::PlainObjectBase<DerivedB>& basis,
  const int start,
  const int num,
  DerivedE& energy,
  Eigen::PlainObjectBase<DerivedG>& grad,
  Eigen::PlainObjectBase<DerivedH>& hess)
{
  using namespace std;
  assert(V.cols() == 3 && "Vertices dim must be 3");
  assert(T.cols() == 4 && "Tetra dim must be 4");
  assert(Bm.size() == T.rows() && "Bm size missmatch T rows");
  assert(W.size() == T.rows() && "W size missmatch T rows");
  assert(_Mu.size() == _Lam.size() && "Mu size missmatch Lam size");
  assert(T.middleRows(start, num).maxCoeff() * 3 < basis.rows() && "index out of basis range");

  DerivedMu Mu;
  DerivedLam Lam;
  if (_Mu.size() == V.rows())
  {
    /* average parameter \mu and \lambda from vertices to tetrahedrons */
    Eigen::SparseMatrix<DerivedE> Proj;
    sim::average_onto_faces_mat(V, T, Proj);

    Mu = Proj * _Mu;
    Lam = Proj * _Lam;
  }
  else if (_Mu.size() == T.rows())
  {
    Mu = _Mu;
    Lam = _Lam;
  }
  else
  {
    assert(false && "Mu size missmatch V.rows() and T.rows()");
    return;
  }
  typedef Eigen::Matrix3d Mat3;

  /* loop for of each tet */
  int end = start + num;
  assert(end <= T.rows() && "start + num excceeds max T rows");

  elastic_neohookean(V, T, Bm, W, Mu, Lam, start, num, energy, grad);

  hess.setZero(basis.cols(), basis.cols());
  int num_core = 8;
  int block = 1 + num / num_core;
  std::vector<DerivedH> hess_tmp;
  hess_tmp.clear();
  for (int i = 0; i < num_core; i++)
  {
    hess_tmp.push_back(DerivedH::Zero(basis.cols(), basis.cols()));
  }

  igl::parallel_for(num_core, [&V, &T, &Bm, &W, &Mu, &Lam, &basis, &start, &hess, &end, &block, &hess_tmp](const int idx)
    {
      int block_start = start + idx * block;
      int block_end = min(start + idx * block + block, end);
      for (int i = block_start; i < block_end; i++)
      {
        /* compute deformation pos Ds = [v0-v3, v1-v3, v2-v3] */
        Mat3 Ds_t;
        Ds_t << V.row(T(i, 0)), V.row(T(i, 1)), V.row(T(i, 2));
        Ds_t.rowwise() -= V.row(T(i, 3));

        /* compute deformation gradient F */
        Mat3 F;
        F = Ds_t.transpose() * Bm[i];

        /* call neohookean model to get energy */
        double tmp_e;
        Mat3 P;
        sim::neohookean_model(F, Mu(i), Lam(i), tmp_e, P);

        /* accumulate stiffness matrix */
        double a = W(i) * Mu(i);
        double J = F.determinant();
        double b = W(i) * (Mu(i) - Lam(i) * std::log(J));
        double c = W(i) * Lam(i);

        Mat3 A = Bm[i] * Bm[i].transpose();
        Mat3 B = Ds_t.inverse();

        double H[12][12];

        /* \par H_kl / \par D_sj = a \delta{s, k} A_jl + b B_kj B_sl + c B_sj B_kl */

        for (int l = 0; l < 3; l++)
          for (int k = 0; k < 3; k++)
            for (int j = 0; j < 3; j++)
              for (int s = 0; s < 3; s++)
                H[3 * l + k][3 * j + s] = b * B(k, j) * B(s, l) + c * B(s, j) * B(k, l);

        for (int l = 0; l < 3; l++)
          for (int s = 0; s < 3; s++)
            for (int j = 0; j < 3; j++)
              H[3 * l + s][3 * j + s] += a * A(j, l);

        for (int s = 0; s < 9; s++)
        {
          H[9][s] = H[s][9] = -(H[s][0] + H[s][3] + H[s][6]);
          H[10][s] = H[s][10] = -(H[s][1] + H[s][4] + H[s][7]);
          H[11][s] = H[s][11] = -(H[s][2] + H[s][5] + H[s][8]);
        }

        H[9][9] = -(H[9][0] + H[9][3] + H[9][6]);
        H[9][10] = H[10][9] = -(H[9][1] + H[9][4] + H[9][7]);
        H[9][11] = H[11][9] = -(H[9][2] + H[9][5] + H[9][8]);
        H[10][10] = -(H[10][1] + H[10][4] + H[10][7]);
        H[10][11] = H[11][10] = -(H[10][2] + H[10][5] + H[10][8]);
        H[11][11] = -(H[11][2] + H[11][5] + H[11][8]);

        int addr[12];

        if (DerivedV::Options == Eigen::ColMajor)
        {
          /* use col-major matrix */
          int num_V = V.rows();
          for (int s = 0; s < 4; s++)
          {
            int ind = T(i, s);
            addr[3 * s] = ind;
            addr[3 * s + 1] = ind + num_V;
            addr[3 * s + 2] = ind + 2 * num_V;
          }
        }
        else
        {
          /* use row-major matrix */
          for (int s = 0; s < 4; s++)
          {
            int ind = T(i, s);
            ind *= 3;
            addr[3 * s] = ind;
            addr[3 * s + 1] = ind + 1;
            addr[3 * s + 2] = ind + 2;
          }
        }

        DerivedH Hmat(12, 12);
        for (int j = 0; j < 12; j++)
          for (int k = 0; k < 12; k++)
            Hmat(j, k) = H[j][k];

        DerivedB sub_base(12, basis.cols());
        for (int k = 0; k < 12; k++)
          sub_base.row(k) = basis.row(addr[k]);

        DerivedH hess_sub = sub_base.transpose() * Hmat * sub_base;

        /* accumulate energy */
        hess_tmp[idx] += hess_sub;
      }
    }, num_core);

  for (int i = 0; i < num_core; i++)
  {
    hess += hess_tmp[i];
  }
}

/* main body to compute mu\lam gradient */
template <
  typename DerivedV, typename DerivedT,
  typename DerivedBm_T, typename DerivedBm_A, typename DerivedW,
  typename DerivedMu, typename DerivedLam,
  typename DerivedFMu, typename DerivedFLam,
  typename ScalarMu, typename ScalarLam>
IGSIM_INLINE void sim::elastic_neohookean(
  const Eigen::PlainObjectBase<DerivedV>& V,
  const Eigen::PlainObjectBase<DerivedT>& T,
  const std::vector<DerivedBm_T, DerivedBm_A>& Bm,
  const Eigen::PlainObjectBase<DerivedW>& W,
  const Eigen::PlainObjectBase<DerivedMu>& _Mu,
  const Eigen::PlainObjectBase<DerivedLam>& _Lam,
  Eigen::PlainObjectBase<DerivedFMu>& fmu,
  Eigen::PlainObjectBase<DerivedFLam>& flam,
  Eigen::SparseMatrix<ScalarMu>& Kmu,
  Eigen::SparseMatrix<ScalarLam>& Klam)
{
  using namespace std;
  assert(V.cols() == 3 && "Vertices dim must be 3");
  assert(T.cols() == 4 && "Tetra dim must be 4");
  assert(Bm.size() == T.rows() && "Bm size missmatch T rows");
  assert(W.size() == T.rows() && "W size missmatch T rows");
  assert(_Mu.size() == _Lam.size() && "Mu size missmatch Lam size");
  
  DerivedMu Mu;
  DerivedLam Lam;
  if (_Mu.size() == V.rows())
  {
    /* average parameter \mu and \lambda from vertices to tetrahedrons */
    Eigen::SparseMatrix<double> Proj;
    sim::average_onto_faces_mat(V, T, Proj);

    Mu = Proj * _Mu;
    Lam = Proj * _Lam;
  }
  else if (_Mu.size() == T.rows())
  {
    Mu = _Mu;
    Lam = _Lam;
  }
  else
  {
    assert(false && "Mu size missmatch V.rows() and T.rows()");
    return;
  }
  typedef Eigen::Matrix3d Mat3;

  /* initialize fmu and flam */
  fmu.resize(Mu.size());
  flam.resize(Lam.size());

  /* initialize Kmu and Klam */
  typedef Eigen::Triplet<ScalarMu> Tmu;
  Kmu.setZero();
  Kmu.resize(V.size(), Mu.size());
  std::vector<Tmu> Kmu_coeff;
  Kmu_coeff.clear();
  Kmu_coeff.reserve(12 * T.rows());

  typedef Eigen::Triplet<ScalarLam> Tlam;
  Klam.setZero();
  Klam.resize(V.size(), Lam.size());
  std::vector<Tlam> Klam_coeff;
  Klam_coeff.clear();
  Klam_coeff.reserve(12 * T.rows());

  /* loop for of each tet */
  for (int i = 0; i < T.rows(); i++)
  {
    /* compute deformation pos Ds = [v0-v3, v1-v3, v2-v3] */
    Mat3 Ds_t;
    Ds_t << V.row(T(i, 0)), V.row(T(i, 1)), V.row(T(i, 2));
    Ds_t.rowwise() -= V.row(T(i, 3));

    /* compute deformation gradient F */
    Mat3 F;
    F = Ds_t.transpose() * Bm[i];

    /* call neohookean model to get dPmu, dPlam */
    ScalarMu demu;
    ScalarLam delam;
    Mat3 dPmu, dPlam;
    sim::neohookean_model(F, Mu(i), Lam(i), demu, delam, dPmu, dPlam);
    int num_V = V.rows();

    /* fill in fmu and flam */
    fmu(i) = W(i) * demu;
    flam(i) = W(i) * delam;

    ////////////////////// fill in Kmu /////////////////////
    Mat3 _dHmu = - W(i) * dPmu * Bm[i].transpose();
    Mat3 dHmu = _dHmu.transpose();
    /* compute dH -> df, j = 0, 1, 2 */
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 3; k++)
      {
        int targ_addr = k * num_V + T(i, j);
        Kmu_coeff.push_back(Tmu(targ_addr, i, dHmu(j, k)));
      }
    /* compute dH -> df, j = 3 */
    auto dfmu_3 = - dHmu.colwise().sum();
    for (int k = 0; k < 3; k++)
    {
      int targ_addr = k * num_V + T(i, 3);
      Kmu_coeff.push_back(Tmu(targ_addr, i, dfmu_3(k)));
    }
    
    ///////////////////// fill in Klam /////////////////////
    Mat3 _dHlam = - W(i) * dPlam * Bm[i].transpose();
    Mat3 dHlam = _dHlam.transpose();
    /* compute dH -> df, j = 0, 1, 2 */
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 3; k++)
      {
        int targ_addr = k * num_V + T(i, j);
        Klam_coeff.push_back(Tlam(targ_addr, i, dHlam(j, k)));
      }
    /* compute dH -> df, j = 3 */
    auto dflam_3 = - dHlam.colwise().sum();
    for (int k = 0; k < 3; k++)
    {
      int targ_addr = k * num_V + T(i, 3);
      Klam_coeff.push_back(Tlam(targ_addr, i, dflam_3(k)));
    }
  }

  Kmu.setFromTriplets(Kmu_coeff.begin(), Kmu_coeff.end());
  Klam.setFromTriplets(Klam_coeff.begin(), Klam_coeff.end());
}

#ifdef IGSIM_STATIC_LIBRARY
#endif
