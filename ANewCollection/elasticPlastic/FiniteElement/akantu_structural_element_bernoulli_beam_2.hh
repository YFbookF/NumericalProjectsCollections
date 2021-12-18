/**
 * @file   structural_element_bernoulli_beam_2.hh
 *
 * @author Fabian Barras <fabian.barras@epfl.ch>
 * @author Lucas Frerot <lucas.frerot@epfl.ch>
 * @author Sébastien Hartmann <sebastien.hartmann@epfl.ch>
 * @author Nicolas Richart <nicolas.richart@epfl.ch>
 *
 * @date creation: Wed Oct 11 2017
 * @date last modification: Wed Jan 10 2018
 *
 * @brief  Specific functions for bernoulli beam 2d
 *
 *
 * Copyright (©) 2016-2018 EPFL (Ecole Polytechnique Fédérale de Lausanne)
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
#include "aka_common.hh"
#include "structural_mechanics_model.hh"
/* -------------------------------------------------------------------------- */

#ifndef __AKANTU_STRUCTURAL_ELEMENT_BERNOULLI_BEAM_2_HH__
#define __AKANTU_STRUCTURAL_ELEMENT_BERNOULLI_BEAM_2_HH__

namespace akantu {

/* -------------------------------------------------------------------------- */
template <>
inline void StructuralMechanicsModel::assembleMass<_bernoulli_beam_2>() {
  AKANTU_DEBUG_IN();
  constexpr ElementType type = _bernoulli_beam_2;

  auto & fem = getFEEngineClass<MyFEEngineType>();
  auto nb_element = mesh.getNbElement(type);
  auto nb_nodes_per_element = mesh.getNbNodesPerElement(type);
  auto nb_quadrature_points = fem.getNbIntegrationPoints(type);
  auto nb_fields_to_interpolate = ElementClass<type>::getNbStressComponents();
  auto nt_n_field_size = nb_degree_of_freedom * nb_nodes_per_element;

  Array<Real> n(nb_element * nb_quadrature_points,
                nb_fields_to_interpolate * nt_n_field_size, "N");

  Array<Real> * rho_field =
      new Array<Real>(nb_element * nb_quadrature_points, 1, "Rho");
  rho_field->clear();
  computeRho(*rho_field, type, _not_ghost);

#if 0
  bool sign = true;

  for (auto && ghost_type : ghost_types) {
    fem.computeShapesMatrix(type, nb_degree_of_freedom, nb_nodes_per_element, n,
                            0, 0, 0, sign, ghost_type); // Ni ui -> u
    fem.computeShapesMatrix(type, nb_degree_of_freedom, nb_nodes_per_element, n,
                            1, 1, 1, sign, ghost_type); // Mi vi -> v
    fem.computeShapesMatrix(type, nb_degree_of_freedom, nb_nodes_per_element, n,
                            2, 2, 1, sign, ghost_type); // Li Theta_i -> v
    fem.assembleFieldMatrix(*rho_field, nb_degree_of_freedom, *mass_matrix, n,
                            rotation_matrix, type, ghost_type);
  }
#endif

  delete rho_field;
  AKANTU_DEBUG_OUT();
}

/* -------------------------------------------------------------------------- */
template <>
void StructuralMechanicsModel::computeRotationMatrix<_bernoulli_beam_2>(
    Array<Real> & rotations) {
  auto type = _bernoulli_beam_2;
  auto nodes_it = mesh.getNodes().begin(this->spatial_dimension);

  for (auto && tuple :
       zip(make_view(mesh.getConnectivity(type), 2),
           make_view(rotations, nb_degree_of_freedom, nb_degree_of_freedom))) {

    auto & connec = std::get<0>(tuple);
    auto & R = std::get<1>(tuple);

    Vector<Real> x2 = nodes_it[connec(1)]; // X2
    Vector<Real> x1 = nodes_it[connec(0)]; // X1

    auto le = x1.distance(x2);
    auto c = (x2(0) - x1(0)) / le;
    auto s = (x2(1) - x1(1)) / le;

    /// Definition of the rotation matrix
    R = {{c, s, 0.}, {-s, c, 0.}, {0., 0., 1.}};
  }
}

/* -------------------------------------------------------------------------- */
template <>
void StructuralMechanicsModel::computeTangentModuli<_bernoulli_beam_2>(
    Array<Real> & tangent_moduli) {
  // auto nb_element = getFEEngine().getMesh().getNbElement(_bernoulli_beam_2);
  auto nb_quadrature_points =
      getFEEngine().getNbIntegrationPoints(_bernoulli_beam_2);
  auto tangent_size = 2;

  tangent_moduli.clear();
  auto D_it = tangent_moduli.begin(tangent_size, tangent_size);
  auto el_mat = element_material(_bernoulli_beam_2, _not_ghost).begin();

  for (auto & mat : element_material(_bernoulli_beam_2, _not_ghost)) {
    auto E = materials[mat].E;
    auto A = materials[mat].A;
    auto I = materials[mat].I;
    for (UInt q = 0; q < nb_quadrature_points; ++q, ++D_it) {
      auto & D = *D_it;
      D(0, 0) = E * A;
      D(1, 1) = E * I;
    }
  }
}

} // namespace akantu

#endif /* __AKANTU_STRUCTURAL_ELEMENT_BERNOULLI_BEAM_2_HH__ */
