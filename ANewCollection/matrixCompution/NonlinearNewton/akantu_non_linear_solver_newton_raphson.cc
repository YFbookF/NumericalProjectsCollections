/**
 * @file   non_linear_solver_newton_raphson.cc
 * https://github.com/sunscale/Akantu
 * @author Nicolas Richart <nicolas.richart@epfl.ch>
 *
 * @date creation: Tue Sep 15 2015
 * @date last modification: Wed Feb 21 2018
 *
 * @brief  Implementation of the default NonLinearSolver
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
#include "non_linear_solver_newton_raphson.hh"
#include "communicator.hh"
#include "dof_manager_default.hh"
#include "solver_callback.hh"
#include "solver_vector.hh"
#include "sparse_solver_mumps.hh"
/* -------------------------------------------------------------------------- */

namespace akantu {

/* -------------------------------------------------------------------------- */
NonLinearSolverNewtonRaphson::NonLinearSolverNewtonRaphson(
    DOFManagerDefault & dof_manager,
    const NonLinearSolverType & non_linear_solver_type, const ID & id,
    UInt memory_id)
    : NonLinearSolver(dof_manager, non_linear_solver_type, id, memory_id),
      dof_manager(dof_manager),
      solver(std::make_unique<SparseSolverMumps>(
          dof_manager, "J", id + ":sparse_solver", memory_id)) {

  this->supported_type.insert(NonLinearSolverType::_newton_raphson_modified);
  this->supported_type.insert(NonLinearSolverType::_newton_raphson);
  this->supported_type.insert(NonLinearSolverType::_linear);

  this->checkIfTypeIsSupported();

  this->registerParam("threshold", convergence_criteria, 1e-10, _pat_parsmod,
                      "Threshold to consider results as converged");
  this->registerParam("convergence_type", convergence_criteria_type,
                      SolveConvergenceCriteria::_solution, _pat_parsmod,
                      "Type of convergence criteria");
  this->registerParam("max_iterations", max_iterations, 10, _pat_parsmod,
                      "Max number of iterations");
  this->registerParam("error", error, _pat_readable, "Last reached error");
  this->registerParam("nb_iterations", n_iter, _pat_readable,
                      "Last reached number of iterations");
  this->registerParam("converged", converged, _pat_readable,
                      "Did last solve converged");
  this->registerParam("force_linear_recompute", force_linear_recompute, true,
                      _pat_modifiable,
                      "Force reassembly of the jacobian matrix");
}

/* -------------------------------------------------------------------------- */
NonLinearSolverNewtonRaphson::~NonLinearSolverNewtonRaphson() = default;

/* ------------------------------------------------------------------------ */
void NonLinearSolverNewtonRaphson::solve(SolverCallback & solver_callback) {
  solver_callback.beforeSolveStep();

  this->dof_manager.updateGlobalBlockedDofs();

  solver_callback.predictor();

  if (non_linear_solver_type == NonLinearSolverType::_linear and
      solver_callback.canSplitResidual())
    solver_callback.assembleMatrix("K");

  this->assembleResidual(solver_callback);

  if (this->non_linear_solver_type ==
          NonLinearSolverType::_newton_raphson_modified ||
      (this->non_linear_solver_type == NonLinearSolverType::_linear &&
       this->force_linear_recompute)) {
    solver_callback.assembleMatrix("J");
    this->force_linear_recompute = false;
  }

  this->n_iter = 0;
  this->converged = false;

  this->convergence_criteria_normalized = this->convergence_criteria;

  if (this->convergence_criteria_type == SolveConvergenceCriteria::_residual) {
    this->converged = this->testConvergence(this->dof_manager.getResidual());

    if (this->converged)
      return;

    this->convergence_criteria_normalized =
        this->error * this->convergence_criteria;
  }

  do {
    if (this->non_linear_solver_type == NonLinearSolverType::_newton_raphson)
      solver_callback.assembleMatrix("J");

    this->solver->solve();

    solver_callback.corrector();

    // EventManager::sendEvent(NonLinearSolver::AfterSparseSolve(method));

    if (this->convergence_criteria_type ==
        SolveConvergenceCriteria::_residual) {
      this->assembleResidual(solver_callback);
      this->converged = this->testConvergence(this->dof_manager.getResidual());
    } else {
      this->converged = this->testConvergence(this->dof_manager.getSolution());
    }

    if (this->convergence_criteria_type ==
            SolveConvergenceCriteria::_solution and
        not this->converged)
      this->assembleResidual(solver_callback);
    // this->dump();

    this->n_iter++;
    AKANTU_DEBUG_INFO(
        "[" << this->convergence_criteria_type << "] Convergence iteration "
            << std::setw(std::log10(this->max_iterations)) << this->n_iter
            << ": error " << this->error << (this->converged ? " < " : " > ")
            << this->convergence_criteria);

  } while (not this->converged and this->n_iter <= this->max_iterations);

  // this makes sure that you have correct strains and stresses after the
  // solveStep function (e.g., for dumping)
  if (this->convergence_criteria_type == SolveConvergenceCriteria::_solution)
    this->assembleResidual(solver_callback);

  this->converged =
      this->converged  and not (this->n_iter > this->max_iterations);

  solver_callback.afterSolveStep(this->converged);

  if (not this->converged) {
    AKANTU_CUSTOM_EXCEPTION(debug::NLSNotConvergedException(
        this->convergence_criteria, this->n_iter, this->error));

    AKANTU_DEBUG_WARNING("[" << this->convergence_criteria_type
                             << "] Convergence not reached after "
                             << std::setw(std::log10(this->max_iterations))
                             << this->n_iter << " iteration"
                             << (this->n_iter == 1 ? "" : "s") << "!");
  }

  return;
}

/* -------------------------------------------------------------------------- */
bool NonLinearSolverNewtonRaphson::testConvergence(
    const SolverVector & solver_vector) {
  AKANTU_DEBUG_IN();

  const auto & blocked_dofs = this->dof_manager.getBlockedDOFs();

  const Array<Real> & array(solver_vector);
  UInt nb_degree_of_freedoms = array.size();

  auto arr_it = array.begin();
  auto bld_it = blocked_dofs.begin();

  Real norm = 0.;
  for (UInt n = 0; n < nb_degree_of_freedoms; ++n, ++arr_it, ++bld_it) {
    bool is_local_node = this->dof_manager.isLocalOrMasterDOF(n);
    if ((!*bld_it) && is_local_node) {
      norm += *arr_it * *arr_it;
    }
  }

  dof_manager.getCommunicator().allReduce(norm, SynchronizerOperation::_sum);

  norm = std::sqrt(norm);

  AKANTU_DEBUG_ASSERT(!Math::isnan(norm),
                      "Something went wrong in the solve phase");

  this->error = norm;

  return (error < this->convergence_criteria_normalized);
}

/* -------------------------------------------------------------------------- */

} // namespace akantu
