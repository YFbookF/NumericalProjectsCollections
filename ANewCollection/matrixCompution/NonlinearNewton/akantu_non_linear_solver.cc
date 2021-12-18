/**
 * @file   non_linear_solver.cc
 * https://github.com/sunscale/Akantu
 * @author Nicolas Richart <nicolas.richart@epfl.ch>
 *
 * @date creation: Tue Jul 20 2010
 * @date last modification: Wed Feb 21 2018
 *
 * @brief  Implementation of the base class NonLinearSolver
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
#include "non_linear_solver.hh"
#include "dof_manager.hh"
#include "solver_callback.hh"
/* -------------------------------------------------------------------------- */

namespace akantu {

/* -------------------------------------------------------------------------- */
NonLinearSolver::NonLinearSolver(
    DOFManager & dof_manager,
    const NonLinearSolverType & non_linear_solver_type, const ID & id,
    UInt memory_id)
    : Memory(id, memory_id), Parsable(ParserType::_non_linear_solver, id),
      _dof_manager(dof_manager),
      non_linear_solver_type(non_linear_solver_type) {

  this->registerParam("type", this->non_linear_solver_type, _pat_parsable,
                      "Non linear solver type");
}

/* -------------------------------------------------------------------------- */
NonLinearSolver::~NonLinearSolver() = default;

/* -------------------------------------------------------------------------- */
void NonLinearSolver::checkIfTypeIsSupported() {
  if (this->supported_type.find(this->non_linear_solver_type) ==
          this->supported_type.end() and
      this->non_linear_solver_type != NonLinearSolverType::_auto) {
    AKANTU_EXCEPTION("The resolution method "
                     << this->non_linear_solver_type
                     << " is not implemented in the non linear solver "
                     << this->id << "!");
  }
}

/* -------------------------------------------------------------------------- */
void NonLinearSolver::assembleResidual(SolverCallback & solver_callback) {
  if (solver_callback.canSplitResidual() and
      non_linear_solver_type == NonLinearSolverType::_linear) {
    this->_dof_manager.clearResidual();
    solver_callback.assembleResidual("external");
    this->_dof_manager.assembleMatMulDOFsToResidual("K", -1.);
    solver_callback.assembleResidual("inertial");
  } else {
    solver_callback.assembleResidual();
  }
}

} // namespace akantu
