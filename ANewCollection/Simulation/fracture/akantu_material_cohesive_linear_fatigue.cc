/**
 * @file   material_cohesive_linear_fatigue.cc
 *
 * @author Marco Vocialta <marco.vocialta@epfl.ch>
 *
 * @date creation: Fri Feb 20 2015
 * @date last modification: Tue Feb 20 2018
 *
 * @brief  See material_cohesive_linear_fatigue.hh for information
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
#include "material_cohesive_linear_fatigue.hh"

namespace akantu {

/* -------------------------------------------------------------------------- */
template <UInt spatial_dimension>
MaterialCohesiveLinearFatigue<spatial_dimension>::MaterialCohesiveLinearFatigue(
    SolidMechanicsModel & model, const ID & id)
    : MaterialCohesiveLinear<spatial_dimension>(model, id),
      delta_prec("delta_prec", *this), K_plus("K_plus", *this),
      K_minus("K_minus", *this), T_1d("T_1d", *this),
      switches("switches", *this), delta_dot_prec("delta_dot_prec", *this),
      normal_regime("normal_regime", *this) {

  this->registerParam("delta_f", delta_f, Real(-1.),
                      _pat_parsable | _pat_readable, "delta_f");

  this->registerParam("progressive_delta_f", progressive_delta_f, false,
                      _pat_parsable | _pat_readable,
                      "Whether or not delta_f is equal to delta_max");

  this->registerParam("count_switches", count_switches, false,
                      _pat_parsable | _pat_readable,
                      "Count the opening/closing switches per element");

  this->registerParam(
      "fatigue_ratio", fatigue_ratio, Real(1.), _pat_parsable | _pat_readable,
      "What portion of the cohesive law is subjected to fatigue");
}

/* -------------------------------------------------------------------------- */
template <UInt spatial_dimension>
void MaterialCohesiveLinearFatigue<spatial_dimension>::initMaterial() {
  MaterialCohesiveLinear<spatial_dimension>::initMaterial();

  // check that delta_f has a proper value or assign a defaul value
  if (delta_f < 0)
    delta_f = this->delta_c_eff;
  else if (delta_f < this->delta_c_eff)
    AKANTU_ERROR("Delta_f must be greater or equal to delta_c");

  delta_prec.initialize(1);
  K_plus.initialize(1);
  K_minus.initialize(1);
  T_1d.initialize(1);
  normal_regime.initialize(1);

  if (count_switches) {
    switches.initialize(1);
    delta_dot_prec.initialize(1);
  }
}

/* -------------------------------------------------------------------------- */
template <UInt spatial_dimension>
void MaterialCohesiveLinearFatigue<spatial_dimension>::computeTraction(
    const Array<Real> & normal, ElementType el_type, GhostType ghost_type) {
  AKANTU_DEBUG_IN();

  /// define iterators
  auto traction_it =
      this->tractions(el_type, ghost_type).begin(spatial_dimension);

  auto opening_it = this->opening(el_type, ghost_type).begin(spatial_dimension);

  auto contact_traction_it =
      this->contact_tractions(el_type, ghost_type).begin(spatial_dimension);

  auto contact_opening_it =
      this->contact_opening(el_type, ghost_type).begin(spatial_dimension);

  auto normal_it = normal.begin(spatial_dimension);

  auto traction_end =
      this->tractions(el_type, ghost_type).end(spatial_dimension);

  const Array<Real> & sigma_c_array = this->sigma_c_eff(el_type, ghost_type);
  Array<Real> & delta_max_array = this->delta_max(el_type, ghost_type);
  const Array<Real> & delta_c_array = this->delta_c_eff(el_type, ghost_type);
  Array<Real> & damage_array = this->damage(el_type, ghost_type);

  auto insertion_stress_it =
      this->insertion_stress(el_type, ghost_type).begin(spatial_dimension);

  Array<Real> & delta_prec_array = delta_prec(el_type, ghost_type);
  Array<Real> & K_plus_array = K_plus(el_type, ghost_type);
  Array<Real> & K_minus_array = K_minus(el_type, ghost_type);
  Array<Real> & T_1d_array = T_1d(el_type, ghost_type);
  Array<bool> & normal_regime_array = normal_regime(el_type, ghost_type);

  Array<UInt> * switches_array = nullptr;
  Array<Real> * delta_dot_prec_array = nullptr;

  if (count_switches) {
    switches_array = &switches(el_type, ghost_type);
    delta_dot_prec_array = &delta_dot_prec(el_type, ghost_type);
  }

  auto * memory_space = new Real[2 * spatial_dimension];
  Vector<Real> normal_opening(memory_space, spatial_dimension);
  Vector<Real> tangential_opening(memory_space + spatial_dimension,
                                  spatial_dimension);

  Real tolerance = Math::getTolerance();

  /// loop on each quadrature point
  for (UInt q = 0; traction_it != traction_end; ++traction_it, ++opening_it,
            ++normal_it, ++contact_traction_it, ++insertion_stress_it,
            ++contact_opening_it, ++q) {

    /// compute normal and tangential opening vectors
    Real normal_opening_norm = opening_it->dot(*normal_it);
    normal_opening = (*normal_it);
    normal_opening *= normal_opening_norm;

    tangential_opening = *opening_it;
    tangential_opening -= normal_opening;

    Real tangential_opening_norm = tangential_opening.norm();

    /**
     * compute effective opening displacement
     * @f$ \delta = \sqrt{
     * \frac{\beta^2}{\kappa^2} \Delta_t^2 + \Delta_n^2 } @f$
     */
    Real delta =
        tangential_opening_norm * tangential_opening_norm * this->beta2_kappa2;

    bool penetration = normal_opening_norm < -tolerance;
    if (this->contact_after_breaking == false &&
        Math::are_float_equal(damage_array(q), 1.))
      penetration = false;

    if (penetration) {
      /// use penalty coefficient in case of penetration
      *contact_traction_it = normal_opening;
      *contact_traction_it *= this->penalty;
      *contact_opening_it = normal_opening;
      /// don't consider penetration contribution for delta
      *opening_it = tangential_opening;
      normal_opening.clear();
    } else {
      delta += normal_opening_norm * normal_opening_norm;
      contact_traction_it->clear();
      contact_opening_it->clear();
    }

    delta = std::sqrt(delta);

    /**
     * Compute traction @f$ \mathbf{T} = \left(
     * \frac{\beta^2}{\kappa} \Delta_t \mathbf{t} + \Delta_n
     * \mathbf{n} \right) \frac{\sigma_c}{\delta} \left( 1-
     * \frac{\delta}{\delta_c} \right)@f$
     */

    // update maximum displacement and damage
    delta_max_array(q) = std::max(delta, delta_max_array(q));
    damage_array(q) = std::min(delta_max_array(q) / delta_c_array(q), Real(1.));

    Real delta_dot = delta - delta_prec_array(q);

    // count switches if asked
    if (count_switches) {
      if ((delta_dot > 0. && (*delta_dot_prec_array)(q) <= 0.) ||
          (delta_dot < 0. && (*delta_dot_prec_array)(q) >= 0.))
        ++((*switches_array)(q));

      (*delta_dot_prec_array)(q) = delta_dot;
    }

    // set delta_f equal to delta_max if desired
    if (progressive_delta_f)
      delta_f = delta_max_array(q);

    // broken element case
    if (Math::are_float_equal(damage_array(q), 1.))
      traction_it->clear();
    // just inserted element case
    else if (Math::are_float_equal(damage_array(q), 0.)) {
      if (penetration)
        traction_it->clear();
      else
        *traction_it = *insertion_stress_it;
      // initialize the 1d traction to sigma_c
      T_1d_array(q) = sigma_c_array(q);
    }
    // normal case
    else {
      // if element is closed then there are zero tractions
      if (delta <= tolerance)
        traction_it->clear();
      // otherwise compute new tractions if the new delta is different
      // than the previous one
      else if (std::abs(delta_dot) > tolerance) {
        // loading case
        if (delta_dot > 0.) {
          if (!normal_regime_array(q)) {
            // equation (4) of the article
            K_plus_array(q) *= 1. - delta_dot / delta_f;
            // equivalent to equation (2) of the article
            T_1d_array(q) += K_plus_array(q) * delta_dot;

            // in case of reloading, traction must not exceed that of the
            // envelop of the cohesive law
            Real max_traction =
                sigma_c_array(q) * (1 - delta / delta_c_array(q));
            bool max_traction_exceeded = T_1d_array(q) > max_traction;
            if (max_traction_exceeded)
              T_1d_array(q) = max_traction;

            // switch to standard linear cohesive law
            if (delta_max_array(q) > fatigue_ratio * delta_c_array(q)) {
              // reset delta_max to avoid big jumps in the traction
              delta_max_array(q) =
                  sigma_c_array(q) /
                  (T_1d_array(q) / delta + sigma_c_array(q) / delta_c_array(q));
              damage_array(q) =
                  std::min(delta_max_array(q) / delta_c_array(q), Real(1.));
              K_minus_array(q) = sigma_c_array(q) / delta_max_array(q) *
                                 (1. - damage_array(q));
              normal_regime_array(q) = true;
            } else {
              // equation (3) of the article
              K_minus_array(q) = T_1d_array(q) / delta;

              // if the traction is following the cohesive envelop, then
              // K_plus has to be reset
              if (max_traction_exceeded)
                K_plus_array(q) = K_minus_array(q);
            }
          } else {
            // compute stiffness according to the standard law
            K_minus_array(q) =
                sigma_c_array(q) / delta_max_array(q) * (1. - damage_array(q));
          }
        }
        // unloading case
        else if (!normal_regime_array(q)) {
          // equation (4) of the article
          K_plus_array(q) +=
              (K_plus_array(q) - K_minus_array(q)) * delta_dot / delta_f;
          // equivalent to equation (2) of the article
          T_1d_array(q) = K_minus_array(q) * delta;
        }

        // applying the actual stiffness
        *traction_it = tangential_opening;
        *traction_it *= this->beta2_kappa;
        *traction_it += normal_opening;
        *traction_it *= K_minus_array(q);
      }
    }

    // update precendent delta
    delta_prec_array(q) = delta;
  }

  delete[] memory_space;
  AKANTU_DEBUG_OUT();
}

/* -------------------------------------------------------------------------- */

INSTANTIATE_MATERIAL(cohesive_linear_fatigue, MaterialCohesiveLinearFatigue);

} // namespace akantu
