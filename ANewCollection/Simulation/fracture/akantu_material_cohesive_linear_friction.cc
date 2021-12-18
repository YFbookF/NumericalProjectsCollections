/**
 * @file   material_cohesive_linear_friction.cc
 *
 * @author Mauro Corrado <mauro.corrado@epfl.ch>
 *
 * @date creation: Tue Jan 12 2016
 * @date last modification: Wed Feb 21 2018
 *
 * @brief  Linear irreversible cohesive law of mixed mode loading with
 * random stress definition for extrinsic type
 *
 *
 * Copyright (©) 2015-2018 EPFL (Ecole Polytechnique Fédérale de Lausanne)
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

/* -------------------------------------------------------------------------- */
#include "material_cohesive_linear_friction.hh"
#include "solid_mechanics_model_cohesive.hh"

namespace akantu {

/* -------------------------------------------------------------------------- */
template <UInt spatial_dimension>
MaterialCohesiveLinearFriction<spatial_dimension>::
    MaterialCohesiveLinearFriction(SolidMechanicsModel & model, const ID & id)
    : MaterialParent(model, id), residual_sliding("residual_sliding", *this),
      friction_force("friction_force", *this) {
  AKANTU_DEBUG_IN();

  this->registerParam("mu", mu_max, Real(0.), _pat_parsable | _pat_readable,
                      "Maximum value of the friction coefficient");

  this->registerParam("penalty_for_friction", friction_penalty, Real(0.),
                      _pat_parsable | _pat_readable,
                      "Penalty parameter for the friction behavior");

  AKANTU_DEBUG_OUT();
}

/* -------------------------------------------------------------------------- */
template <UInt spatial_dimension>
void MaterialCohesiveLinearFriction<spatial_dimension>::initMaterial() {
  AKANTU_DEBUG_IN();

  MaterialParent::initMaterial();

  friction_force.initialize(spatial_dimension);
  residual_sliding.initialize(1);
  residual_sliding.initializeHistory();

  AKANTU_DEBUG_OUT();
}

/* -------------------------------------------------------------------------- */
template <UInt spatial_dimension>
void MaterialCohesiveLinearFriction<spatial_dimension>::computeTraction(
    __attribute__((unused)) const Array<Real> & normal, ElementType el_type,
    GhostType ghost_type) {
  AKANTU_DEBUG_IN();

  residual_sliding.resize();
  friction_force.resize();

  /// define iterators
  auto traction_it =
      this->tractions(el_type, ghost_type).begin(spatial_dimension);
  auto traction_end =
      this->tractions(el_type, ghost_type).end(spatial_dimension);
  auto opening_it = this->opening(el_type, ghost_type).begin(spatial_dimension);
  auto previous_opening_it =
      this->opening.previous(el_type, ghost_type).begin(spatial_dimension);
  auto contact_traction_it =
      this->contact_tractions(el_type, ghost_type).begin(spatial_dimension);
  auto contact_opening_it =
      this->contact_opening(el_type, ghost_type).begin(spatial_dimension);
  auto normal_it = this->normal.begin(spatial_dimension);
  auto sigma_c_it = this->sigma_c_eff(el_type, ghost_type).begin();
  auto delta_max_it = this->delta_max(el_type, ghost_type).begin();
  auto delta_max_prev_it =
      this->delta_max.previous(el_type, ghost_type).begin();
  auto delta_c_it = this->delta_c_eff(el_type, ghost_type).begin();
  auto damage_it = this->damage(el_type, ghost_type).begin();
  auto insertion_stress_it =
      this->insertion_stress(el_type, ghost_type).begin(spatial_dimension);
  auto res_sliding_it = this->residual_sliding(el_type, ghost_type).begin();
  auto res_sliding_prev_it =
      this->residual_sliding.previous(el_type, ghost_type).begin();
  auto friction_force_it =
      this->friction_force(el_type, ghost_type).begin(spatial_dimension);

  Vector<Real> normal_opening(spatial_dimension);
  Vector<Real> tangential_opening(spatial_dimension);

  if (not this->model->isDefaultSolverExplicit())
    this->delta_max(el_type, ghost_type)
        .copy(this->delta_max.previous(el_type, ghost_type));

  /// loop on each quadrature point
  for (; traction_it != traction_end;
       ++traction_it, ++opening_it, ++normal_it, ++sigma_c_it, ++delta_max_it,
       ++delta_c_it, ++damage_it, ++contact_traction_it, ++insertion_stress_it,
       ++contact_opening_it, ++delta_max_prev_it, ++res_sliding_it,
       ++res_sliding_prev_it, ++friction_force_it, ++previous_opening_it) {

    Real normal_opening_norm, tangential_opening_norm;
    bool penetration;
    this->computeTractionOnQuad(
        *traction_it, *opening_it, *normal_it, *delta_max_it, *delta_c_it,
        *insertion_stress_it, *sigma_c_it, normal_opening, tangential_opening,
        normal_opening_norm, tangential_opening_norm, *damage_it, penetration,
        *contact_traction_it, *contact_opening_it);

    if (penetration) {
      /// the friction coefficient mu increases with the damage. It
      /// equals the maximum value when damage = 1.
      //      Real damage = std::min(*delta_max_prev_it / *delta_c_it,
      //      Real(1.));
      Real mu = mu_max; // * damage;

      /// the definition of tau_max refers to the opening
      /// (penetration) of the previous incremental step
      Real normal_opening_prev_norm =
          std::min(previous_opening_it->dot(*normal_it), Real(0.));

      //      Vector<Real> normal_opening_prev = (*normal_it);
      //      normal_opening_prev *= normal_opening_prev_norm;
      Real tau_max = mu * this->penalty * (std::abs(normal_opening_prev_norm));
      Real delta_sliding_norm =
          std::abs(tangential_opening_norm - *res_sliding_prev_it);

      /// tau is the norm of the friction force, acting tangentially to the
      /// surface
      Real tau = std::min(friction_penalty * delta_sliding_norm, tau_max);

      if ((tangential_opening_norm - *res_sliding_prev_it) < 0.0)
        tau = -tau;

      /// from tau get the x and y components of friction, to be added in the
      /// force vector
      Vector<Real> tangent_unit_vector(spatial_dimension);
      tangent_unit_vector = tangential_opening / tangential_opening_norm;
      *friction_force_it = tau * tangent_unit_vector;

      /// update residual_sliding
      *res_sliding_it =
          tangential_opening_norm - (std::abs(tau) / friction_penalty);
    } else {
      friction_force_it->clear();
    }

    *traction_it += *friction_force_it;
  }

  AKANTU_DEBUG_OUT();
}

/* -------------------------------------------------------------------------- */
template <UInt spatial_dimension>
void MaterialCohesiveLinearFriction<spatial_dimension>::computeTangentTraction(
    const ElementType & el_type, Array<Real> & tangent_matrix,
    __attribute__((unused)) const Array<Real> & normal, GhostType ghost_type) {
  AKANTU_DEBUG_IN();

  /// define iterators
  auto tangent_it = tangent_matrix.begin(spatial_dimension, spatial_dimension);
  auto tangent_end = tangent_matrix.end(spatial_dimension, spatial_dimension);

  auto normal_it = this->normal.begin(spatial_dimension);

  auto opening_it = this->opening(el_type, ghost_type).begin(spatial_dimension);
  auto previous_opening_it =
      this->opening.previous(el_type, ghost_type).begin(spatial_dimension);

  /**
   * NB: delta_max_it points on delta_max_previous, i.e. the
   * delta_max related to the solution of the previous incremental
   * step
   */
  auto delta_max_it = this->delta_max.previous(el_type, ghost_type).begin();
  auto sigma_c_it = this->sigma_c_eff(el_type, ghost_type).begin();
  auto delta_c_it = this->delta_c_eff(el_type, ghost_type).begin();
  auto damage_it = this->damage(el_type, ghost_type).begin();

  auto contact_opening_it =
      this->contact_opening(el_type, ghost_type).begin(spatial_dimension);

  auto res_sliding_prev_it =
      this->residual_sliding.previous(el_type, ghost_type).begin();

  Vector<Real> normal_opening(spatial_dimension);
  Vector<Real> tangential_opening(spatial_dimension);

  for (; tangent_it != tangent_end;
       ++tangent_it, ++normal_it, ++opening_it, ++previous_opening_it,
       ++delta_max_it, ++sigma_c_it, ++delta_c_it, ++damage_it,
       ++contact_opening_it, ++res_sliding_prev_it) {

    Real normal_opening_norm, tangential_opening_norm;
    bool penetration;
    this->computeTangentTractionOnQuad(
        *tangent_it, *delta_max_it, *delta_c_it, *sigma_c_it, *opening_it,
        *normal_it, normal_opening, tangential_opening, normal_opening_norm,
        tangential_opening_norm, *damage_it, penetration, *contact_opening_it);

    if (penetration) {
      //      Real damage = std::min(*delta_max_it / *delta_c_it, Real(1.));
      Real mu = mu_max; // * damage;

      Real normal_opening_prev_norm =
          std::min(previous_opening_it->dot(*normal_it), Real(0.));
      //      Vector<Real> normal_opening_prev = (*normal_it);
      //      normal_opening_prev *= normal_opening_prev_norm;

      Real tau_max = mu * this->penalty * (std::abs(normal_opening_prev_norm));
      Real delta_sliding_norm =
          std::abs(tangential_opening_norm - *res_sliding_prev_it);

      // tau is the norm of the friction force, acting tangentially to the
      // surface
      Real tau = std::min(friction_penalty * delta_sliding_norm, tau_max);

      if (tau < tau_max && tau_max > Math::getTolerance()) {
        Matrix<Real> I(spatial_dimension, spatial_dimension);
        I.eye(1.);

        Matrix<Real> n_outer_n(spatial_dimension, spatial_dimension);
        n_outer_n.outerProduct(*normal_it, *normal_it);

        Matrix<Real> nn(n_outer_n);
        I -= nn;
        *tangent_it += I * friction_penalty;
      }
    }

    // check if the tangential stiffness matrix is symmetric
    //    for (UInt h = 0; h < spatial_dimension; ++h){
    //      for (UInt l = h; l < spatial_dimension; ++l){
    //        if (l > h){
    //          Real k_ls = (*tangent_it)[spatial_dimension*h+l];
    //          Real k_us =  (*tangent_it)[spatial_dimension*l+h];
    //          //          std::cout << "k_ls = " << k_ls << std::endl;
    //          //          std::cout << "k_us = " << k_us << std::endl;
    //          if (std::abs(k_ls) > 1e-13 && std::abs(k_us) > 1e-13){
    //            Real error = std::abs((k_ls - k_us) / k_us);
    //            if (error > 1e-10){
    //	      std::cout << "non symmetric cohesive matrix" << std::endl;
    //	      //  std::cout << "error " << error << std::endl;
    //            }
    //          }
    //        }
    //      }
    //    }
  }

  AKANTU_DEBUG_OUT();
}
/* -------------------------------------------------------------------------- */

INSTANTIATE_MATERIAL(cohesive_linear_friction, MaterialCohesiveLinearFriction);

} // namespace akantu
