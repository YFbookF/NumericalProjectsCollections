/**
 * @file   structural_element_kirchhoff_shell.hh
 *
 * @author Fabian Barras <fabian.barras@epfl.ch>
 * @author Lucas Frerot <lucas.frerot@epfl.ch>
 * @author Sébastien Hartmann <sebastien.hartmann@epfl.ch>
 * @author Nicolas Richart <nicolas.richart@epfl.ch>
 * @author Damien Spielmann <damien.spielmann@epfl.ch>
 *
 * @date creation: Wed Oct 11 2017
 * @date last modification: Wed Feb 21 2018
 *
 * @brief  Specific functions for bernoulli kirchhoff shell
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
#ifndef __AKANTU_STRUCTURAL_ELEMENT_BERNOULLI_KIRCHHOFF_SHELL_HH__
#define __AKANTU_STRUCTURAL_ELEMENT_BERNOULLI_KIRCHHOFF_SHELL_HH__

#include "structural_mechanics_model.hh"

namespace akantu {

/* -------------------------------------------------------------------------- */
template <>
inline void
StructuralMechanicsModel::assembleMass<_discrete_kirchhoff_triangle_18>() {
  AKANTU_TO_IMPLEMENT();
}

/* -------------------------------------------------------------------------- */
template <>
void StructuralMechanicsModel::computeTangentModuli<
    _discrete_kirchhoff_triangle_18>(Array<Real> & tangent_moduli) {

  auto tangent_size =
      ElementClass<_discrete_kirchhoff_triangle_18>::getNbStressComponents();
  auto nb_quad =
      getFEEngine().getNbIntegrationPoints(_discrete_kirchhoff_triangle_18);

  auto H_it = tangent_moduli.begin(tangent_size, tangent_size);

  for (UInt mat :
       element_material(_discrete_kirchhoff_triangle_18, _not_ghost)) {
    auto & m = materials[mat];

    for (UInt q = 0; q < nb_quad; ++q, ++H_it) {
      auto & H = *H_it;
      H.clear();
      Matrix<Real> D = {{1, m.nu, 0}, {m.nu, 1, 0}, {0, 0, (1 - m.nu) / 2}};
      D *= m.E * m.t / (1 - m.nu * m.nu);
      H.block(D, 0, 0);                           // in plane membrane behavior
      H.block(D * Math::pow<3>(m.t) / 12., 3, 3); // bending behavior
    }
  }
}

} // namespace akantu

#endif /* __AKANTU_STRUCTURAL_ELEMENT_BERNOULLI_DISCRETE_KIRCHHOFF_TRIANGLE_18_HH__ \
        */
