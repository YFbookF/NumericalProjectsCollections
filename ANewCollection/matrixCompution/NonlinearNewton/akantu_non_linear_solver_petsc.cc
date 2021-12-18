/**
 * @file   non_linear_solver_petsc.cc
 * https://github.com/sunscale/Akantu
 * @author Nicolas Richart 
 *
 * @date creation  Mon Dec 31 2018
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
#include "non_linear_solver_petsc.hh"
#include "dof_manager_petsc.hh"
#include "mpi_communicator_data.hh"
#include "solver_callback.hh"
#include "solver_vector_petsc.hh"
#include "sparse_matrix_petsc.hh"
/* -------------------------------------------------------------------------- */
#include <petscoptions.h>
/* -------------------------------------------------------------------------- */

namespace akantu {

NonLinearSolverPETSc::NonLinearSolverPETSc(
    DOFManagerPETSc & dof_manager,
    const NonLinearSolverType & non_linear_solver_type, const ID & id,
    UInt memory_id)
    : NonLinearSolver(dof_manager, non_linear_solver_type, id, memory_id),
      dof_manager(dof_manager) {
  std::unordered_map<NonLinearSolverType, SNESType>
      petsc_non_linear_solver_types{
          {NonLinearSolverType::_newton_raphson, SNESNEWTONLS},
          {NonLinearSolverType::_linear, SNESKSPONLY},
          {NonLinearSolverType::_gmres, SNESNGMRES},
          {NonLinearSolverType::_bfgs, SNESQN},
          {NonLinearSolverType::_cg, SNESNCG}};

  this->has_internal_set_param = true;

  for (const auto & pair : petsc_non_linear_solver_types) {
    supported_type.insert(pair.first);
  }

  this->checkIfTypeIsSupported();

  auto mpi_comm = dof_manager.getMPIComm();

  PETSc_call(SNESCreate, mpi_comm, &snes);

  auto it = petsc_non_linear_solver_types.find(non_linear_solver_type);
  if (it != petsc_non_linear_solver_types.end()) {
    PETSc_call(SNESSetType, snes, it->second);
  }

  SNESSetFromOptions(snes);
}

/* -------------------------------------------------------------------------- */
NonLinearSolverPETSc::~NonLinearSolverPETSc() {
  PETSc_call(SNESDestroy, &snes);
}

/* -------------------------------------------------------------------------- */
class NonLinearSolverPETScCallback {
public:
  NonLinearSolverPETScCallback(DOFManagerPETSc & dof_manager,
                               SolverVectorPETSc & x)
      : dof_manager(dof_manager), x(x), x_prev(x, "previous_solution") {}

  void corrector() {
    auto & dx = dof_manager.getSolution();
    PETSc_call(VecWAXPY, dx, -1., x_prev, x);

    dof_manager.splitSolutionPerDOFs();
    callback->corrector();

    PETSc_call(VecCopy, x, x_prev);
  }

  void assembleResidual() {
    corrector();
    callback->assembleResidual();
  }

  void assembleJacobian() {
    // corrector();
    callback->assembleMatrix("J");
  }

  void setInitialSolution(SolverVectorPETSc & x) {
    PETSc_call(VecCopy, x, x_prev);
  }

  void setCallback(SolverCallback & callback) { this->callback = &callback; }

private:
  // SNES & snes;
  SolverCallback * callback;
  DOFManagerPETSc & dof_manager;

  SolverVectorPETSc & x;
  SolverVectorPETSc x_prev;
}; // namespace akantu

/* -------------------------------------------------------------------------- */
PetscErrorCode NonLinearSolverPETSc::FormFunction(SNES /*snes*/, Vec /*dx*/,
                                                  Vec /*f*/, void * ctx) {
  auto * _this = reinterpret_cast<NonLinearSolverPETScCallback *>(ctx);
  _this->assembleResidual();
  return 0;
}

/* -------------------------------------------------------------------------- */
PetscErrorCode NonLinearSolverPETSc::FormJacobian(SNES /*snes*/, Vec /*dx*/,
                                                  Mat /*J*/, Mat /*P*/,
                                                  void * ctx) {
  auto * _this = reinterpret_cast<NonLinearSolverPETScCallback *>(ctx);
  _this->assembleJacobian();
  return 0;
}

/* -------------------------------------------------------------------------- */
void NonLinearSolverPETSc::solve(SolverCallback & callback) {
  callback.beforeSolveStep();
  this->dof_manager.updateGlobalBlockedDofs();

  callback.assembleMatrix("J");
  auto & global_x = dof_manager.getSolution();
  global_x.clear();

  if (not x) {
    x = std::make_unique<SolverVectorPETSc>(global_x, "temporary_solution");
  }

  *x = global_x;

  if (not ctx) {
    ctx = std::make_unique<NonLinearSolverPETScCallback>(dof_manager, *x);
  }

  ctx->setCallback(callback);
  ctx->setInitialSolution(global_x);

  auto & rhs = dof_manager.getResidual();
  auto & J = dof_manager.getMatrix("J");

  PETSc_call(SNESSetFunction, snes, rhs, NonLinearSolverPETSc::FormFunction,
             ctx.get());
  PETSc_call(SNESSetJacobian, snes, J, J, NonLinearSolverPETSc::FormJacobian,
             ctx.get());

  rhs.clear();

  callback.predictor();
  callback.assembleResidual();

  PETSc_call(SNESSolve, snes, nullptr, *x);
  PETSc_call(SNESGetConvergedReason, snes, &reason);
  PETSc_call(SNESGetIterationNumber, snes, &n_iter);

  PETSc_call(VecAXPY, global_x, -1.0, *x);

  dof_manager.splitSolutionPerDOFs();
  callback.corrector();

  bool converged = reason >= 0;
  callback.afterSolveStep(converged);

  if (not converged) {
    PetscReal atol, rtol, stol;
    PetscInt maxit, maxf;

    PETSc_call(SNESGetTolerances, snes, &atol, &rtol, &stol, &maxit, &maxf);
    AKANTU_CUSTOM_EXCEPTION(debug::SNESNotConvergedException(
        this->reason, this->n_iter, stol, atol, rtol, maxit));
  }
}

/* -------------------------------------------------------------------------- */
void NonLinearSolverPETSc::set_param(const ID & param,
                                     const std::string & value) {
  std::map<ID, ID> akantu_to_petsc_option = {{"max_iterations", "snes_max_it"},
                                             {"threshold", "snes_stol"}};

  auto it = akantu_to_petsc_option.find(param);
  auto p = it == akantu_to_petsc_option.end() ? param : it->second;

  PetscOptionsSetValue(NULL, p.c_str(), value.c_str());
  SNESSetFromOptions(snes);
  PetscOptionsClear(NULL);
}

/* -------------------------------------------------------------------------- */
void NonLinearSolverPETSc::parseSection(const ParserSection & section) {
  auto parameters = section.getParameters();
  for (auto && param : range(parameters.first, parameters.second)) {
    PetscOptionsSetValue(NULL, param.getName().c_str(),
                         param.getValue().c_str());
  }
  SNESSetFromOptions(snes);
  PetscOptionsClear(NULL);
}

} // namespace akantu
