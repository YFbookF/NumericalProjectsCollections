/**
 * @file   material_anisotropic_damage_tmpl.hh
 * https://github.com/sunscale/Akantu
 * @author Nicolas Richart
 *
 * @date creation  mar jun 25 2019
 *
 * @brief A Documented file.
 *
 *
 * Copyright (©) 2010-2011 EPFL (Ecole Polytechnique Fédérale de Lausanne)
 * Laboratory (LSMS - Laboratoire de Simulation en Mécanique des Solides)
 *
 * Akantu is free  software: you can redistribute it and/or  modify it under the
 * terms  of the  GNU Lesser  General Public  License as  published by  the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * Akantu is  distributed in the  hope that it  will be useful, but  WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A  PARTICULAR PURPOSE. See  the GNU  Lesser General  Public License  for more
 * details.
 *
 * You should  have received  a copy  of the GNU  Lesser General  Public License
 * along with Akantu. If not, see <http://www.gnu.org/licenses/>.
 *
 */
/* -------------------------------------------------------------------------- */
#include "aka_iterators.hh"
#include "material_anisotropic_damage.hh"
/* -------------------------------------------------------------------------- */

#ifndef __AKANTU_MATERIAL_ANISOTROPIC_DAMAGE_TMPL_HH__
#define __AKANTU_MATERIAL_ANISOTROPIC_DAMAGE_TMPL_HH__

namespace akantu {
struct EmptyIteratorContainer {
  struct iterator {
    auto & operator++() { return *this; }
    Real operator*() { return 0; }
    bool operator!=(const iterator &) const { return true; }
    bool operator==(const iterator &) const { return false; }
  };

  auto begin() { return iterator(); }
  auto end() { return iterator(); }
};
} // namespace akantu

namespace std {
template <> struct iterator_traits<::akantu::EmptyIteratorContainer::iterator> {
  using iterator_category = forward_iterator_tag;
  using value_type = akantu::Real;
  using difference_type = std::ptrdiff_t;
  using pointer = akantu::Real *;
  using reference = akantu::Real &;
};
} // namespace std

namespace akantu {
namespace {
  template <UInt dim, class Op>
  void tensorPlus_(const Matrix<Real> & A, Op && oper) {
    Vector<Real> A_eigs(dim);
    A.eig(A_eigs);

    for (auto & ap : A_eigs) {
      oper(ap);
    }
  }

  template <UInt dim> auto tensorPlus2(const Matrix<Real> & A) {
    Real square = 0;
    tensorPlus_<dim>(A, [&](Real eig) {
      eig = std::max(eig, 0.);
      square += eig * eig;
    });

    return square;
  }

  template <UInt dim> auto tensorPlusTrace(const Matrix<Real> & A) {
    Real trace_plus = 0;
    Real trace_minus = 0;
    tensorPlus_<dim>(A, [&](Real eig) {
      trace_plus += std::max(eig, 0.);
      trace_minus += std::min(eig, 0.);
    });

    return std::make_pair(trace_plus, trace_minus);
  }

  template <UInt dim, class Op>
  auto tensorPlusOp(const Matrix<Real> & A, Matrix<Real> & A_directions,
                    Op && oper, bool sorted = false) {
    Vector<Real> A_eigs(dim);
    Matrix<Real> A_diag(dim, dim);
    A.eig(A_eigs, A_directions, sorted);

    for (auto && data : enumerate(A_eigs)) {
      auto i = std::get<0>(data);
      A_diag(i, i) = oper(std::max(std::get<1>(data), 0.), i);
    }

    return A_directions * A_diag * A_directions.transpose();
  }

  template <UInt dim, class Op>
  auto tensorPlus(const Matrix<Real> & A, Matrix<Real> & A_directions,
                  bool sorted = false) {
    return tensorPlusOp<dim>(A, A_directions, [](Real x, Real) { return x; },
                             sorted);
  }

  template <UInt dim, class Op>
  auto tensorPlusOp(const Matrix<Real> & A, Op && oper) {
    Matrix<Real> A_directions(dim, dim);
    return tensorPlusOp<dim>(A, A_directions, std::forward<Op>(oper));
  }

  template <UInt dim> auto tensorPlus(const Matrix<Real> & A) {
    return tensorPlusOp<dim>(A, [](Real x, Real) { return x; });
  }

  template <UInt dim> auto tensorSqrt(const Matrix<Real> & A) {
    return tensorPlusOp<dim>(A, [](Real x, UInt) { return std::sqrt(x); });
  }

} // namespace

/* -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- */
template <UInt dim, template <UInt> class EquivalentStrain,
          template <UInt> class DamageThreshold, template <UInt> class Parent>
MaterialAnisotropicDamage<dim, EquivalentStrain, DamageThreshold, Parent>::
    MaterialAnisotropicDamage(SolidMechanicsModel & model, const ID & id)
    : Parent<dim>(model, id), damage("damage_tensor", *this),
      elastic_stress("elastic_stress", *this),
      equivalent_strain("equivalent_strain", *this),
      trace_damage("trace_damage", *this), equivalent_strain_function(*this),
      damage_threshold_function(*this) {
  this->registerParam("Dc", Dc, _pat_parsable, "Critical damage");

  this->damage.initialize(dim * dim);
  this->elastic_stress.initialize(dim * dim);
  this->equivalent_strain.initialize(1);

  this->trace_damage.initialize(1);
  this->trace_damage.initializeHistory();
}

/* -------------------------------------------------------------------------- */
template <UInt dim, template <UInt> class EquivalentStrain,
          template <UInt> class DamageThreshold, template <UInt> class Parent>
void MaterialAnisotropicDamage<dim, EquivalentStrain, DamageThreshold, Parent>::
    damageStress(Matrix<Real> & sigma, const Matrix<Real> & sigma_el,
                 const Matrix<Real> & D, Real TrD) {
  // σ_(n + 1) = (1 − D_(n + 1))^(1/2) σ~_(n + 1) (1 − D_(n + 1))^(1 / 2)
  //         - ((1 − D_(n + 1)) : σ~_(n + 1))/ (3 - Tr(D_(n+1))) (1 − D_(n + 1))
  //         + 1/3 (1 - Tr(D_(n+1)) <Tr(σ~_(n + 1))>_+ + <Tr(σ~_(n + 1))>_-) I

  auto one_D = Matrix<Real>::eye(dim) - D;
  auto sqrt_one_D = tensorSqrt<dim>(one_D);

  Real Tr_sigma_plus, Tr_sigma_minus;
  std::tie(Tr_sigma_plus, Tr_sigma_minus) = tensorPlusTrace<dim>(sigma_el);

  auto I = Matrix<Real>::eye(dim);

  sigma = sqrt_one_D * sigma_el * sqrt_one_D -
          (one_D.doubleDot(sigma_el) / (dim - TrD) * one_D) +
          1. / dim * ((1 - TrD) * Tr_sigma_plus - Tr_sigma_minus) * I;
}

/* -------------------------------------------------------------------------- */
template <UInt dim, template <UInt> class EquivalentStrain,
          template <UInt> class DamageThreshold, template <UInt> class Parent>
void MaterialAnisotropicDamage<dim, EquivalentStrain, DamageThreshold,
                               Parent>::computeStress(ElementType type,
                                                      GhostType ghost_type) {

  for (auto && data :
       zip(make_view(this->stress(type, ghost_type), dim, dim),
           make_view(this->gradu(type, ghost_type), dim, dim),
           make_view(this->sigma_th(type, ghost_type)),
           make_view(this->elastic_stress(type, ghost_type), dim, dim),
           make_view(this->equivalent_strain(type, ghost_type)),
           make_view(this->damage(type, ghost_type), dim, dim),
           make_view(this->trace_damage(type, ghost_type)),
           make_view(this->trace_damage.previous(type, ghost_type)),
           equivalent_strain_function, damage_threshold_function)) {
    auto & sigma = std::get<0>(data);
    auto & grad_u = std::get<1>(data);
    auto & sigma_th = std::get<2>(data);
    auto & sigma_el = std::get<3>(data);
    auto & epsilon_hat = std::get<4>(data);
    auto & D = std::get<5>(data);
    auto & TrD_n_1 = std::get<6>(data);
    auto & TrD = std::get<7>(data);
    auto & equivalent_strain_data = std::get<8>(data);
    auto & damage_threshold_data = std::get<9>(data);

    Matrix<Real> Dtmp(dim, dim);
    Real TrD_n_1_tmp;
    Matrix<Real> epsilon(dim, dim);

    // yes you read properly this is a label for a goto
    auto computeDamage = [&]() {
      MaterialElastic<dim>::computeStressOnQuad(grad_u, sigma_el, sigma_th);

      this->template gradUToEpsilon<dim>(grad_u, epsilon);

      // evaluate the damage criteria
      epsilon_hat = equivalent_strain_function(epsilon, equivalent_strain_data);

      // evolve the damage if needed
      auto K_TrD = damage_threshold_function.K(TrD, damage_threshold_data);

      auto f = epsilon_hat - K_TrD;

      // if test function > 0 evolve the damage
      if (f > 0) {
        TrD_n_1_tmp =
            damage_threshold_function.K_inv(epsilon_hat, damage_threshold_data);

        auto epsilon_plus = tensorPlus<dim>(epsilon);
        auto delta_lambda = (TrD_n_1_tmp - TrD) / (epsilon_hat * epsilon_hat);

        Dtmp = D + delta_lambda * epsilon_plus;
        return true;
      }
      return false;
    };

    // compute a temporary version of the new damage
    auto is_damage_updated = computeDamage();

    if (is_damage_updated) {
      /// Check and correct for broken case
      if (Dtmp.trace() > Dc) {
        if (epsilon.trace() > 0) { // tensile loading
          auto kpa = this->kpa;
          auto lambda = this->lambda;

          // change kappa to Kappa_broken = (1-Dc) Kappa
          kpa = (1 - Dc) * kpa;
          this->E = 9 * kpa * (kpa - lambda) / (3 * kpa - lambda);
          this->nu = lambda / (3 * kpa - lambda);
          this->updateInternalParameters();

          computeDamage();
        } else if (std::abs(epsilon.trace()) < 1e-10) { // deviatoric case
          Matrix<Real> n(dim, dim);
          std::vector<UInt> ns;
          tensorPlusOp<dim>(Dtmp, n,
                            [&](Real x, UInt i) {
                              if (x > this->Dc) {
                                ns.push_back(i);
                                return this->Dc;
                              }

                              return x;
                            },
                            true);
        }
      }

      TrD_n_1 = TrD_n_1_tmp;
      D = Dtmp;
    } else {
      TrD_n_1 = TrD;
    }

    // apply the damage to the stress
    damageStress(sigma, sigma_el, D, TrD_n_1);
  }
}

/* -------------------------------------------------------------------------- */
/* EquivalentStrain functions                                                 */
/* -------------------------------------------------------------------------- */
template <UInt dim>
class EquivalentStrainMazars : public EmptyIteratorContainer {
public:
  EquivalentStrainMazars(Material & /*mat*/) {}

  template <class... Other>
  Real operator()(const Matrix<Real> & epsilon, Other &&... /*other*/) {
    Real epsilon_hat = 0.;
    std::tie(epsilon_hat, std::ignore) = tensorPlusTrace<dim>(epsilon);
    return std::sqrt(epsilon_hat);
  }
};

template <UInt dim>
class EquivalentStrainMazarsDruckerPrager : public EquivalentStrainMazars<dim> {
public:
  EquivalentStrainMazarsDruckerPrager(Material & mat)
      : EquivalentStrainMazars<dim>(mat) {
    mat.registerParam("k", k, _pat_parsable, "k");
  }

  template <class... Other>
  Real operator()(const Matrix<Real> & epsilon, Real) {
    Real epsilon_hat = EquivalentStrainMazars<dim>::operator()(epsilon);
    epsilon_hat += k * epsilon.trace();
    return epsilon_hat;
  }

protected:
  Real k;
};

/* -------------------------------------------------------------------------- */
/* DamageThreshold functions                                                  */
/* -------------------------------------------------------------------------- */
template <UInt dim>
class DamageThresholdLinear : public EmptyIteratorContainer {
public:
  DamageThresholdLinear(Material & mat) : mat(mat) {
    mat.registerParam("A", A, _pat_parsable, "A");
    mat.registerParam("K0", K0, _pat_parsable, "K0");
  }

  template <class... Other> Real K(Real x, Other &&... /*other*/) {
    return 1. / A * x + K0;
  }

  template <class... Other> Real K_inv(Real x, Other &&... /*other*/) {
    return A * (x - K0);
  }

private:
  Material & mat;
  Real A;
  Real K0;
};

template <UInt dim> class DamageThresholdTan : public EmptyIteratorContainer {
public:
  DamageThresholdTan(Material & mat) : mat(mat) {
    mat.registerParam("a", a, _pat_parsable, "a");
    mat.registerParam("A", A, _pat_parsable, "A");
    mat.registerParam("K0", K0, _pat_parsable, "K0");
  }

  template <class... Other> Real K(Real x, Other &&... /*other*/) {
    return a * std::tan(std::atan2(x, a) - std::atan2(K0, a));
  }

  template <class... Other> Real K_inv(Real x, Other &&... /*other*/) {
    return a * A * (std::atan2(x, a) - std::atan2(K0, a));
  }

private:
  Material & mat;
  Real a{2.93e-4};
  Real A{5e3};
  Real K0{5e-5};
};

} // namespace akantu

#endif /* __AKANTU_MATERIAL_ANISOTROPIC_DAMAGE_TMPL_HH__ */
