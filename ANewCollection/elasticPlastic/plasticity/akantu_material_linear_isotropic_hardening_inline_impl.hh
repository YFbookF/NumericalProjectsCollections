/**
 * @file   material_linear_isotropic_hardening_inline_impl.hh
 * https://github.com/sunscale/Akantu
 * @author Ramin Aghababaei <ramin.aghababaei@epfl.ch>
 * @author Guillaume Anciaux <guillaume.anciaux@epfl.ch>
 * @author Lucas Frerot <lucas.frerot@epfl.ch>
 * @author Benjamin Paccaud <benjamin.paccaud@epfl.ch>
 * @author Daniel Pino Muñoz <daniel.pinomunoz@epfl.ch>
 * @author Nicolas Richart <nicolas.richart@epfl.ch>
 *
 * @date creation: Mon Apr 07 2014
 * @date last modification: Thu Nov 30 2017
 *
 * @brief  Implementation of the inline functions of the material plasticity
 *
 *
 * Copyright (©) 2014-2018 EPFL (Ecole Polytechnique Fédérale de Lausanne)
 * Laboratory (LSMS - Laboratoire de Simulation en Mécanique des Solides)
 *
 * Akantu is free  software: you can redistribute it and/or  modify it under the
 * terms  of the  GNU Lesser  General Public  License as published by  the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * Akantu is  distributed in the  hope that it  will be useful, but  WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See  the GNU  Lesser General  Public License  for more
 * details.
 *
 * You should  have received  a copy  of the GNU  Lesser General  Public License
 * along with Akantu. If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "material_linear_isotropic_hardening.hh"

namespace akantu {
/* -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- */
/// Infinitesimal deformations
template <UInt dim>
inline void MaterialLinearIsotropicHardening<dim>::computeStressOnQuad(
    const Matrix<Real> & grad_u, const Matrix<Real> & previous_grad_u,
    Matrix<Real> & sigma, const Matrix<Real> & previous_sigma,
    Matrix<Real> & inelastic_strain,
    const Matrix<Real> & previous_inelastic_strain, Real & iso_hardening,
    const Real & previous_iso_hardening, const Real & sigma_th,
    const Real & previous_sigma_th) {

  Real delta_sigma_th = sigma_th - previous_sigma_th;

  Matrix<Real> grad_delta_u(grad_u);
  grad_delta_u -= previous_grad_u;

  // Compute trial stress, sigma_tr
  Matrix<Real> sigma_tr(dim, dim);
  MaterialElastic<dim>::computeStressOnQuad(grad_delta_u, sigma_tr,
                                            delta_sigma_th);
  sigma_tr += previous_sigma;

  // We need a full stress tensor, otherwise the VM stress is messed up
  Matrix<Real> sigma_tr_dev(3, 3, 0);
  for (UInt i = 0; i < dim; ++i)
    for (UInt j = 0; j < dim; ++j)
      sigma_tr_dev(i, j) = sigma_tr(i, j);

  sigma_tr_dev -= Matrix<Real>::eye(3, sigma_tr.trace() / 3.0);

  // Compute effective deviatoric trial stress
  Real s = sigma_tr_dev.doubleDot(sigma_tr_dev);
  Real sigma_tr_dev_eff = std::sqrt(3. / 2. * s);

  bool initial_yielding =
      ((sigma_tr_dev_eff - iso_hardening - this->sigma_y) > 0);

  Real dp = (initial_yielding)
                ? (sigma_tr_dev_eff - this->sigma_y - previous_iso_hardening) /
                      (3 * this->mu + this->h)
                : 0;

  iso_hardening = previous_iso_hardening + this->h * dp;

  // Compute inelastic strain (ignore last components in 1-2D)
  Matrix<Real> delta_inelastic_strain(dim, dim, 0.);
  if (std::abs(sigma_tr_dev_eff) >
      sigma_tr_dev.norm<L_inf>() * Math::getTolerance()) {
    for (UInt i = 0; i < dim; ++i)
      for (UInt j = 0; j < dim; ++j)
        delta_inelastic_strain(i, j) = sigma_tr_dev(i, j);
    delta_inelastic_strain *= 3. / 2. * dp / sigma_tr_dev_eff;
  }

  MaterialPlastic<dim>::computeStressAndInelasticStrainOnQuad(
      grad_delta_u, sigma, previous_sigma, inelastic_strain,
      previous_inelastic_strain, delta_inelastic_strain);
}

/* -------------------------------------------------------------------------- */
/// Finite deformations
template <UInt dim>
inline void MaterialLinearIsotropicHardening<dim>::computeStressOnQuad(
    const Matrix<Real> & grad_u, const Matrix<Real> & previous_grad_u,
    Matrix<Real> & sigma, const Matrix<Real> & previous_sigma,
    Matrix<Real> & inelastic_strain,
    const Matrix<Real> & previous_inelastic_strain, Real & iso_hardening,
    const Real & previous_iso_hardening, const Real & sigma_th,
    const Real & previous_sigma_th, const Matrix<Real> & F_tensor) {
  // Finite plasticity
  Real dp = 0.0;
  Real d_dp = 0.0;
  UInt n = 0;

  Real delta_sigma_th = sigma_th - previous_sigma_th;

  Matrix<Real> grad_delta_u(grad_u);
  grad_delta_u -= previous_grad_u;

  // Compute trial stress, sigma_tr
  Matrix<Real> sigma_tr(dim, dim);
  MaterialElastic<dim>::computeStressOnQuad(grad_delta_u, sigma_tr,
                                            delta_sigma_th);
  sigma_tr += previous_sigma;

  // Compute deviatoric trial stress,  sigma_tr_dev
  Matrix<Real> sigma_tr_dev(sigma_tr);
  sigma_tr_dev -= Matrix<Real>::eye(dim, sigma_tr.trace() / 3.0);

  // Compute effective deviatoric trial stress
  Real s = sigma_tr_dev.doubleDot(sigma_tr_dev);
  Real sigma_tr_dev_eff = std::sqrt(3. / 2. * s);

  // compute the cauchy stress to apply the Von-Mises criterion
  Matrix<Real> cauchy_stress(dim, dim);
  Material::StoCauchy<dim>(F_tensor, sigma_tr, cauchy_stress);
  Matrix<Real> cauchy_stress_dev(cauchy_stress);
  cauchy_stress_dev -= Matrix<Real>::eye(dim, cauchy_stress.trace() / 3.0);
  Real c = cauchy_stress_dev.doubleDot(cauchy_stress_dev);
  Real cauchy_stress_dev_eff = std::sqrt(3. / 2. * c);

  const Real iso_hardening_t = previous_iso_hardening;
  iso_hardening = iso_hardening_t;
  // Loop for correcting stress based on yield function

  // F is written in terms of S
  // bool initial_yielding = ( (sigma_tr_dev_eff - iso_hardening -
  // this->sigma_y) > 0) ;
  // while ( initial_yielding && std::abs(sigma_tr_dev_eff - iso_hardening -
  // this->sigma_y) > Math::getTolerance() ) {

  //   d_dp = (sigma_tr_dev_eff - 3. * this->mu *dp -  iso_hardening -
  //   this->sigma_y)
  //     / (3. * this->mu + this->h);

  //   //r = r +  h * dp;
  //   dp = dp + d_dp;
  //   iso_hardening = iso_hardening_t + this->h * dp;

  //   ++n;

  //   /// TODO : explicit this criterion with an error message
  //   if ((std::abs(d_dp) < 1e-9) || (n>50)){
  //     AKANTU_DEBUG_INFO("convergence of increment of plastic strain. d_dp:"
  //     << d_dp << "\tNumber of iteration:"<<n);
  //     break;
  //   }
  // }

  // F is written in terms of cauchy stress
  bool initial_yielding =
      ((cauchy_stress_dev_eff - iso_hardening - this->sigma_y) > 0);
  while (initial_yielding && std::abs(cauchy_stress_dev_eff - iso_hardening -
                                      this->sigma_y) > Math::getTolerance()) {

    d_dp = (cauchy_stress_dev_eff - 3. * this->mu * dp - iso_hardening -
            this->sigma_y) /
           (3. * this->mu + this->h);

    // r = r +  h * dp;
    dp = dp + d_dp;
    iso_hardening = iso_hardening_t + this->h * dp;

    ++n;
    /// TODO : explicit this criterion with an error message
    if ((d_dp < 1e-5) || (n > 50)) {
      AKANTU_DEBUG_INFO("convergence of increment of plastic strain. d_dp:"
                        << d_dp << "\tNumber of iteration:" << n);
      break;
    }
  }

  // Update internal variable
  Matrix<Real> delta_inelastic_strain(dim, dim, 0.);
  if (std::abs(sigma_tr_dev_eff) >
      sigma_tr_dev.norm<L_inf>() * Math::getTolerance()) {

    // /// compute the direction of the plastic strain as \frac{\partial
    // F}{\partial S} = \frac{3}{2J\sigma_{effective}}} Ft \sigma_{dev} F
    Matrix<Real> cauchy_dev_F(dim, dim);
    cauchy_dev_F.mul<false, false>(F_tensor, cauchy_stress_dev);
    Real J = F_tensor.det();
    Real constant = J ? 1. / J : 0;
    constant *= 3. * dp / (2. * cauchy_stress_dev_eff);
    delta_inelastic_strain.mul<true, false>(F_tensor, cauchy_dev_F, constant);

    // Direction given by the piola kirchhoff deviatoric tensor \frac{\partial
    // F}{\partial S} = \frac{3}{2\sigma_{effective}}}S_{dev}
    // delta_inelastic_strain.copy(sigma_tr_dev);
    // delta_inelastic_strain *= 3./2. * dp / sigma_tr_dev_eff;
  }

  MaterialPlastic<dim>::computeStressAndInelasticStrainOnQuad(
      grad_delta_u, sigma, previous_sigma, inelastic_strain,
      previous_inelastic_strain, delta_inelastic_strain);
}

/* -------------------------------------------------------------------------- */

template <UInt dim>
inline void MaterialLinearIsotropicHardening<dim>::computeTangentModuliOnQuad(
    Matrix<Real> & tangent, __attribute__((unused)) const Matrix<Real> & grad_u,
    __attribute__((unused)) const Matrix<Real> & previous_grad_u,
    __attribute__((unused)) const Matrix<Real> & sigma_tensor,
    __attribute__((unused)) const Matrix<Real> & previous_sigma_tensor,
    __attribute__((unused)) const Real & iso_hardening) const {
  // Real r=iso_hardening;

  // Matrix<Real> grad_delta_u(grad_u);
  // grad_delta_u -= previous_grad_u;

  // //Compute trial stress, sigma_tr
  // Matrix<Real> sigma_tr(dim, dim);
  // MaterialElastic<dim>::computeStressOnQuad(grad_delta_u, sigma_tr);
  // sigma_tr += previous_sigma_tensor;

  // // Compute deviatoric trial stress,  sigma_tr_dev
  // Matrix<Real> sigma_tr_dev(sigma_tr);
  // sigma_tr_dev -= Matrix<Real>::eye(dim, sigma_tr.trace() / 3.0);

  // // Compute effective deviatoric trial stress
  // Real s = sigma_tr_dev.doubleDot(sigma_tr_dev);
  // Real sigma_tr_dev_eff=std::sqrt(3./2. * s);

  // // Compute deviatoric stress,  sigma_dev
  // Matrix<Real> sigma_dev(sigma_tensor);
  // sigma_dev -= Matrix<Real>::eye(dim, sigma_tensor.trace() / 3.0);

  // // Compute effective deviatoric stress
  // s =  sigma_dev.doubleDot(sigma_dev);
  // Real sigma_dev_eff = std::sqrt(3./2. * s);

  // Real xr = 0.0;
  // if(sigma_tr_dev_eff > sigma_dev_eff * Math::getTolerance())
  //   xr = sigma_dev_eff / sigma_tr_dev_eff;

  // Real __attribute__((unused)) q = 1.5 * (1. / (1. +  3. * this->mu  /
  // this->h) - xr);

  /*
  UInt cols = tangent.cols();
  UInt rows = tangent.rows();

  for (UInt m = 0; m < rows; ++m) {
    UInt i = VoigtHelper<dim>::vec[m][0];
    UInt j = VoigtHelper<dim>::vec[m][1];

    for (UInt n = 0; n < cols; ++n) {
      UInt k = VoigtHelper<dim>::vec[n][0];
      UInt l = VoigtHelper<dim>::vec[n][1];
      */

  // This section of the code is commented
  // There were some problems with the convergence of plastic-coupled
  // simulations with thermal expansion
  // XXX: DO NOT REMOVE
  /*if (((sigma_tr_dev_eff-iso_hardening-sigmay) > 0) && (xr > 0)) {
    tangent(m,n) =
    2. * this->mu * q * (sigma_tr_dev (i,j) / sigma_tr_dev_eff) * (sigma_tr_dev
    (k,l) / sigma_tr_dev_eff) +
    (i==k) * (j==l) * 2. * this->mu * xr +
    (i==j) * (k==l) * (this->kpa - 2./3. * this->mu * xr);
    if ((m == n) && (m>=dim))
    tangent(m, n) = tangent(m, n) - this->mu * xr;
    } else {*/
  /*
  tangent(m,n) =  (i==k) * (j==l) * 2. * this->mu +
    (i==j) * (k==l) * this->lambda;
  tangent(m,n) -= (m==n) * (m>=dim) * this->mu;
  */
  //}
  // correct tangent stiffness for shear component
  //}
  //}
  MaterialElastic<dim>::computeTangentModuliOnQuad(tangent);
}
} // namespace akantu
