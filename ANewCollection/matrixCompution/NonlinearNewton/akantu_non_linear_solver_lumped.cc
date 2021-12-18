/**
 * @file   non_linear_solver_lumped.cc
 * https://github.com/sunscale/Akantu
 * @author Nicolas Richart <nicolas.richart@epfl.ch>
 *
 * @date creation: Tue Feb 16 2016
 * @date last modification: Wed Jan 31 2018
 *
 * @brief  Implementation of the default NonLinearSolver
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
#include "non_linear_solver_lumped.hh"
#include "communicator.hh"
#include "dof_manager_default.hh"
#include "solver_callback.hh"
#include "solver_vector_default.hh"
/* -------------------------------------------------------------------------- */

namespace akantu {

/* -------------------------------------------------------------------------- */
NonLinearSolverLumped::NonLinearSolverLumped(
    DOFManagerDefault & dof_manager,
    const NonLinearSolverType & non_linear_solver_type, const ID & id,
    UInt memory_id)
    : NonLinearSolver(dof_manager, non_linear_solver_type, id, memory_id),
      dof_manager(dof_manager) {
  this->supported_type.insert(NonLinearSolverType::_lumped);
  this->checkIfTypeIsSupported();

  this->registerParam("b_a2x", this->alpha, 1., _pat_parsmod,
                      "Conversion coefficient between x and A^{-1} b");
}

/* -------------------------------------------------------------------------- */
NonLinearSolverLumped::~NonLinearSolverLumped() = default;

/* ------------------------------------------------------------------------ */
void NonLinearSolverLumped::solve(SolverCallback & solver_callback) {
  solver_callback.beforeSolveStep();
  this->dof_manager.updateGlobalBlockedDofs();
  solver_callback.predictor();

  solver_callback.assembleResidual();

  auto & x = aka::as_type<SolverVectorDefault>(this->dof_manager.getSolution());
  const auto & b = this->dof_manager.getResidual();

  x.resize();

  const auto & blocked_dofs = this->dof_manager.getBlockedDOFs();
  const auto & A = this->dof_manager.getLumpedMatrix("M");

  // alpha is the conversion factor from from force/mass to acceleration needed
  // in model coupled with atomistic \todo find a way to define alpha per dof
  // type
  this->solveLumped(A, x, b, alpha, blocked_dofs);

  this->dof_manager.splitSolutionPerDOFs();

  solver_callback.corrector();
  solver_callback.afterSolveStep(true);
}

/* -------------------------------------------------------------------------- */
void NonLinearSolverLumped::solveLumped(const Array<Real> & A, Array<Real> & x,
                                        const Array<Real> & b, Real alpha,
                                        const Array<bool> & blocked_dofs) {
  for (auto && data :
       zip(make_view(A), make_view(x), make_view(b), make_view(blocked_dofs))) {
    const auto & A = std::get<0>(data);
    auto & x = std::get<1>(data);
    const auto & b = std::get<2>(data);
    const auto & blocked = std::get<3>(data);
    if (not blocked) {
      x = alpha * (b / A);
    }
  }
}

/* -------------------------------------------------------------------------- */

} // namespace akantu
