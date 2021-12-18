/**
 * @file   non_linear_solver_linear.cc
 * https://github.com/sunscale/Akantu
 * @author Nicolas Richart <nicolas.richart@epfl.ch>
 *
 * @date creation: Tue Jul 20 2010
 * @date last modification: Wed Feb 21 2018
 *
 * @brief  Implementation of the default NonLinearSolver
 *
 *
 * Copyright (©)  2010-2018 EPFL (Ecole Polytechnique Fédérale de Lausanne)
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
#include "non_linear_solver_linear.hh"
#include "dof_manager_default.hh"
#include "solver_callback.hh"
/* -------------------------------------------------------------------------- */

namespace akantu {

/* -------------------------------------------------------------------------- */
NonLinearSolverLinear::NonLinearSolverLinear(
    DOFManagerDefault & dof_manager,
    const NonLinearSolverType & non_linear_solver_type, const ID & id,
    UInt memory_id)
    : NonLinearSolver(dof_manager, non_linear_solver_type, id, memory_id),
      dof_manager(dof_manager),
      solver(dof_manager, "J", id + ":sparse_solver", memory_id) {

  this->supported_type.insert(NonLinearSolverType::_linear);
  this->checkIfTypeIsSupported();
}

/* -------------------------------------------------------------------------- */
NonLinearSolverLinear::~NonLinearSolverLinear() = default;

/* ------------------------------------------------------------------------ */
void NonLinearSolverLinear::solve(SolverCallback & solver_callback) {
  solver_callback.beforeSolveStep();
  this->dof_manager.updateGlobalBlockedDofs();

  solver_callback.predictor();

  solver_callback.assembleMatrix("J");

  // Residual computed after J to allow the model to use K to compute the
  // residual
  this->assembleResidual(solver_callback);

  this->solver.solve();

  solver_callback.corrector();
  solver_callback.afterSolveStep(true);
}

/* -------------------------------------------------------------------------- */

} // namespace akantu
