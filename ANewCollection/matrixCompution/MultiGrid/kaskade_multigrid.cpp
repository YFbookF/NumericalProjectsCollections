/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*  This file is part of the library KASKADE 7                               */
/*    see http://www.zib.de/projects/kaskade7-finite-element-toolbox         */
/*                                                                           */
/*  Copyright (C) 2015-2016 Zuse Institute Berlin                            */
/*                                                                           */
/*  KASKADE 7 is distributed under the terms of the ZIB Academic License.    */
/*    see $KASKADE/academic.txt                                              */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifdef UNITTEST

#include <algorithm>
#include <iostream>

#include <boost/timer/timer.hpp>

#include "dune/grid/config.h"
#include "dune/grid/uggrid.hh"





#include "fem/assemble.hh"
#include "fem/istlinterface.hh"
#include "fem/functional_aux.hh"
#include "fem/fixdune.hh"
#include "fem/variables.hh"
#include "fem/gridmanager.hh"
#include "fem/lagrangespace.hh"
#include "fem/spaces.hh"
#include "fem/variables.hh"
#include "io/vtk.hh"
#include "linalg/direct.hh"
#include "linalg/jacobiPreconditioner.hh"
#include "linalg/threadedMatrix.hh"
#include "mg/additiveMultigrid.hh"
#include "mg/multigrid.hh"
#include "mg/multiplicativeMultigrid.hh"
#include "utilities/gridGeneration.hh" //  createUnitSquare
#include "utilities/linalg/scalarproducts.hh"

template <class RType, class VarSet>
class HeatFunctional : public Kaskade::FunctionalBase<Kaskade::VariationalFunctional>
{
public:
  using Scalar = RType;
  using OriginVars = VarSet;
  using AnsatzVars = VarSet;
  using TestVars = VarSet;

  using Cell = typename AnsatzVars::Grid::template Codim<0>::Entity;

  static constexpr int dim = AnsatzVars::Grid::dimension;
  static constexpr int uIdx = 0;
  static constexpr int uSpaceIdx = boost::fusion::result_of::value_at_c<typename AnsatzVars::Variables,
                                             uIdx>::type::spaceIndex;

  class DomainCache : public Kaskade::CacheBase<HeatFunctional,DomainCache>
  {
  public:
    DomainCache(HeatFunctional const& f,
                typename AnsatzVars::VariableSet const& vars_,
                int flags=7):
      data(vars_), sigmaExterior(f.sigmaExterior)
    {}


    void moveTo(Cell const& cell_)
    {
      cell = &cell_;
    }

    template <class Position, class Evaluators>
    void evaluateAt(Position const& xi, Evaluators const& evaluators)
    {
      u  = boost::fusion::at_c<uIdx>(data.data).value(boost::fusion::at_c<uSpaceIdx>(evaluators));
      du = boost::fusion::at_c<uIdx>(data.data).derivative(boost::fusion::at_c<uSpaceIdx>(evaluators));
      f = 1.0;

      // generate jumping coefficients for challenging multigrid
      auto x = cell->geometry().global(xi);
      x -= 0.5;
      if (x.two_norm()<0.4 && x.two_norm()>0.2)
        sigma = 1;
      else
        sigma = sigmaExterior;
    }

    Scalar
    d0() const
    {
      return sigma*sp(du,du)/2 - f*u;
    }

    template<int row>
    Scalar d1_impl(Kaskade::VariationalArg<Scalar,dim,TestVars::template Components<row>::m> const& arg) const
    {
      return sigma*sp(du,arg.derivative) - f*arg.value;
    }

    template<int row, int col>
    Scalar d2_impl(Kaskade::VariationalArg<Scalar,dim,TestVars::template Components<row>::m> const &arg1,
                   Kaskade::VariationalArg<Scalar,dim,AnsatzVars::template Components<row>::m> const &arg2) const
    {
      return sigma*sp(arg1.derivative,arg2.derivative);
    }

  private:
    typename AnsatzVars::VariableSet const& data;
    Dune::FieldVector<Scalar,AnsatzVars::template Components<uIdx>::m> u, f;
    Dune::FieldMatrix<Scalar,AnsatzVars::template Components<uIdx>::m,dim> du;
    Kaskade::LinAlg::EuclideanScalarProduct sp;
    Cell const* cell;
    double sigma, sigmaExterior;
  };

/// \class BoundaryCache
///
  class BoundaryCache : public Kaskade::CacheBase<HeatFunctional,BoundaryCache>
  {
  public:
    BoundaryCache(HeatFunctional<RType,AnsatzVars> const&,
                  typename AnsatzVars::VariableSet const& vars_,
                  int flags=7):
      data(vars_), penalty(1e9), u(0.), uDirichletBoundaryValue(0.)
    {}

    template <class Position, class Evaluators>
    void evaluateAt(Position const&, Evaluators const& evaluators)
    {
      u = boost::fusion::at_c<uIdx>(data.data).value(boost::fusion::at_c<uSpaceIdx>(evaluators));
    }

    Scalar d0() const
    {
      return penalty*(u-uDirichletBoundaryValue)*(u-uDirichletBoundaryValue)/2;
    }

    template<int row>
    Scalar d1_impl(Kaskade::VariationalArg<Scalar,dim> const& arg) const
    {
      return penalty*(u-uDirichletBoundaryValue)*arg.value;
    }

    template<int row, int col>
    Scalar d2_impl(Kaskade::VariationalArg<Scalar,dim> const &arg1,
                   Kaskade::VariationalArg<Scalar,dim> const &arg2) const
    {
      return penalty*arg1.value*arg2.value;
    }

  private:
    typename AnsatzVars::VariableSet const& data;
    Scalar penalty;
    Dune::FieldVector<Scalar,AnsatzVars::template Components<uIdx>::m> u, uDirichletBoundaryValue;
  };

/// \struct D2
///
  template <int row>
  struct D1: public Kaskade::FunctionalBase<Kaskade::VariationalFunctional>::D1<row>
  {
    static bool const present   = true;
    static bool const constant  = false;

  };

  template <int row, int col>
  struct D2: public Kaskade::FunctionalBase<Kaskade::VariationalFunctional>::D2<row,col>
  {
    static bool const present = true;
    static bool const symmetric = true;
    static bool const lumped = false;
  };

  HeatFunctional(double sExt)
  : sigmaExterior(sExt) {}

  template <class Cell>
  int integrationOrder(Cell const& /* cell */, int shapeFunctionOrder, bool boundary) const
  {
    if (boundary)
      return 2*shapeFunctionOrder;
    else
    {
      int stiffnessMatrixIntegrationOrder = 2*(shapeFunctionOrder-1);
      int sourceTermIntegrationOrder = shapeFunctionOrder;        // as rhs f is constant, i.e. of order 0

      return std::max(stiffnessMatrixIntegrationOrder,sourceTermIntegrationOrder);
    }
  }

  double sigmaExterior;
};



template <class CoefficientVector, class Matrix, class Multigrid, class Function>
void runMultiGrid(Matrix const& A, CoefficientVector const& b, Multigrid& mg, Function& u, int order, std::string base, int steps, bool verbose=false)
{
  using namespace Kaskade;

  auto r = b;
  auto dx = b;
  auto solution = b;
  solution = 0;
  std::vector<double> norms;

  Timings& timer = Timings::instance();

  int i = 0;
  for (i=0; i<steps && (norms.empty() || norms.back()>1e-8); ++i)
  {
    // compute residual r = b-A*u
    r = b;
    timer.start("matrix-vector");
    A.usmv(-1.0,solution,r);
    timer.stop("matrix-vector");

    auto res = r;
    double normr = r.two_norm();
    norms.push_back(normr);
    if (verbose)
      std::cerr << "euclidean residual norm |r|=" << normr << "\n";

    // compute preconditioner
    dx = 0.0;
    timer.start("MG application");
    mg.apply(dx,r);
    timer.stop("MG application");

    // compute step length, using r as temporary
    A.mv(dx,r);
    double omega = (res*dx) / (dx*r);
    dx *= omega;
    solution += dx;

    // output
    boost::fusion::at_c<0>(u.data) = solution;
    if (verbose) writeVTKFile(u,base+"-sol-"+paddedString(i),IoOptions().setOrder(order).setPrecision(8));
    boost::fusion::at_c<0>(u.data) = dx;
    if (verbose) writeVTKFile(u,base+"-cor-"+paddedString(i),IoOptions().setOrder(order).setPrecision(7));
  }

  if (norms.size()>2)
  {
    std::cout << "contraction factor: " << std::sqrt(norms.back()/norms[norms.size()-3]) << "\n";
    std::cout << "residual history: ";
    for (double nrm: norms)
      std::cout << nrm << " ";
    std::cout << "\n";
  }
}

void checkP1(int refinements, int steps)
{
  using namespace Kaskade;

  constexpr int dim = 2;

  using Grid = Dune::UGGrid<dim>;
  using Spaces = boost::fusion::vector<H1Space<Grid> const*>;
  using VariableDescriptions = boost::fusion::vector<Variable<SpaceIndex<0>,Components<1>,VariableId<0>>>;
  using VariableSetDesc = VariableSetDescription<Spaces,VariableDescriptions>;
  using Functional = HeatFunctional<double,VariableSetDesc>;
  using CoefficientVectors = VariableSetDesc::CoefficientVectorRepresentation<0,1>::type;

  std::cerr << "=================================================================\nh multigrid with j=" << refinements << "\n";

  boost::timer::cpu_timer timer;
  GridManager<Grid> gridManager( createUnitSquare<Grid>(0.25) );
  std::cout << "grid creation:   " << timer.format();
  timer.start();
  gridManager.globalRefine(refinements);
  std::cout << "grid refinement: " << timer.format();
  gridManager.enforceConcurrentReads(true);

  // construction of finite element space for the scalar solution u.
  timer.start();
  H1Space<Grid> temperatureSpace(gridManager,gridManager.grid().leafGridView(),1);
  std::cout << "p1space creation:" << timer.format();
  Spaces spaces(&temperatureSpace);
  VariableSetDesc variableSetDesc(spaces,{ "u" });

  Functional F(1e0); // 1e0: constant coefficients, nice problem.
                     // 1e2: jumping coefficients, bad problem

  //construct Galerkin representation
  VariationalFunctionalAssembler<LinearizationAt<Functional>> assembler(spaces);
  VariableSetDesc::VariableSet u(variableSetDesc);
  auto du = u;
  timer.start();
  assembler.assemble(linearization(F,u));
  std::cout << "assembly:        " << timer.format();
  using Matrix = NumaBCRSMatrix<Dune::FieldMatrix<double,1,1>>;
  timer.start();
  auto A = assembler.get<Matrix>(false);
  std::cout << "matrix extract:  " << timer.format();
  using Vector = Dune::BlockVector<Dune::FieldVector<double,1>>;

  // Perform a couple of steps of the
  Vector solution(temperatureSpace.degreesOfFreedom());
  CoefficientVectors rhs(assembler.rhs());
  Vector dx = solution;
  Vector r = boost::fusion::at_c<0>(rhs.data);

  std::cout << "Degrees of freedom: " << temperatureSpace.degreesOfFreedom() << "\n";


  Timings& tmr = Timings::instance();
  tmr.clear();
  // Create multigrid preconditioner
  std::cerr << "\nmultiplicative Jacobi multigrid j=" << refinements << "\n";
  tmr.start("MG Jacobi 2,2 construction");
  auto multMG = makeJacobiMultiGrid(A,gridManager,2,2);
  tmr.stop("MG Jacobi 2,2 construction");
  tmr.start("solution");
  runMultiGrid(A,boost::fusion::at_c<0>(rhs.data),multMG,u,1,"/tmp/mul-",steps);
  tmr.stop("solution");
  std::cout << tmr;
  tmr.clear();

  std::cout << "\nmultiplicative Jacobi multigrid j=" << refinements << " with linesearch\n";
  multMG.setLinesearch(true);
  tmr.start("solution");
  runMultiGrid(A,boost::fusion::at_c<0>(rhs.data),multMG,u,1,"/tmp/mul-",steps);
  tmr.stop("solution");
  std::cout << tmr;
  tmr.clear();

  // Create multigrid preconditioner
  std::cerr << "\nadditive BPX multigrid j=" << refinements << "\n";
  tmr.start("BPX construction");
  auto addMG = makeBPX(A,gridManager);
  tmr.stop("BPX construction");
  tmr.start("solution");
  runMultiGrid(A,boost::fusion::at_c<0>(rhs.data),addMG,u,1,"/tmp/add-",steps);
  tmr.stop("solution");
  std::cout << tmr;
  tmr.clear();

  // Create default multigrid preconditioner
  std::cerr << "\ndefault multigrid j=" << refinements << "\n";
  tmr.start("MG default construction");
  auto defMG = makeMultigrid(A,temperatureSpace);
  tmr.stop("MG default construction");
  tmr.start("solution");
  runMultiGrid(A,boost::fusion::at_c<0>(rhs.data),*defMG,u,1,"/tmp/add-",steps);
  tmr.stop("solution");
  std::cout << tmr;
  tmr.clear();

  // Create algebraic multigrid preconditioner
  std::cerr << "\nalgebraic multigrid j=" << refinements << "\n";
  tmr.start("algebraic MG construction");
  tmr.start("MG stack creation");
  auto mgStack = makeAlgebraicMultigridStack(duplicate(A));
  std::cout << "nr levels: " << mgStack.levels() << "\n";
  //~ std::cout << mgStack << "\n";
  tmr.stop("MG stack creation");
  tmr.start("direct solver creation");
  auto coarsePreconditioner = makeDirectPreconditioner(std::move(mgStack.coarseGridMatrix()));
  tmr.stop("direct solver creation");
  tmr.start("MG creation");
  auto amg = makeMultiplicativeMultiGrid(std::move(mgStack),MakeJacobiSmoother(),moveUnique(std::move(coarsePreconditioner)),3,3);
  tmr.stop("MG creation");
  amg.setSmootherStepSize(0.5);
  tmr.stop("algebraic MG construction");
  tmr.start("solution");
  runMultiGrid(A,boost::fusion::at_c<0>(rhs.data),amg,u,1,"/tmp/add-",steps);
  tmr.stop("solution");
  std::cout << tmr;
  tmr.clear();


  // compare with direct solver
  std::cout << "\ndirect solver j=" << refinements << "\n";
  timer.start();
  DirectSolver<Vector,Vector> direct(A,DirectType::MUMPS);
  std::cout << "Factorization: " << timer.format();
  timer.start();
  direct.apply(dx,r);
  std::cout << "Solve: " << timer.format() << "\n";
}

void checkPp(int refinements, int steps, int order)
{
  using namespace Kaskade;

  std::cout << "=======================================\n";
  std::cout << "running hp multigrid with p=" << order << "\n";
  constexpr int dim = 2;

  using Grid = Dune::UGGrid<dim>;
  using Spaces = boost::fusion::vector<H1Space<Grid> const*>;
  using VariableDescriptions = boost::fusion::vector<Variable<SpaceIndex<0>,Components<1>,VariableId<0>>>;
  using VariableSetDesc = VariableSetDescription<Spaces,VariableDescriptions>;
  using Functional = HeatFunctional<double,VariableSetDesc>;
  using CoefficientVectors = VariableSetDesc::CoefficientVectorRepresentation<0,1>::type;


  Timings& timer = Timings::instance();
  timer.start("grid creation");
  GridManager<Grid> gridManager( createUnitSquare<Grid>(1.0) );
  timer.stop("grid creation");
  timer.start("refinement");
  gridManager.globalRefine(refinements);
  timer.stop("refinement");
  gridManager.enforceConcurrentReads(true);

  // construction of finite element space for the scalar solution u.
  timer.start("P1 space creation");
  H1Space<Grid> p1Space(gridManager,gridManager.grid().leafGridView(),1);
  timer.stop("P1 space creation");
  timer.start("P? space creation");
  H1Space<Grid> temperatureSpace(gridManager,gridManager.grid().leafGridView(),order);
  timer.stop("P? space creation");

  std::cout << "Degrees of freedom: " << temperatureSpace.degreesOfFreedom() << "\n";

  auto assembleMatrix = [&](auto const& space)
  {
    Spaces spaces(&space);
    VariableSetDesc variableSetDesc(spaces,{ "u" });
    Functional F(1);
    VariationalFunctionalAssembler<LinearizationAt<Functional>> assembler(spaces);
    VariableSetDesc::VariableSet u(variableSetDesc);

    assembler.assemble(linearization(F,u));
    using Matrix = NumaBCRSMatrix<Dune::FieldMatrix<double,1,1>>;
    CoefficientVectors r(assembler.rhs());
    return std::make_pair(assembler.get<Matrix>(false),r);
  };

  Spaces spaces(&temperatureSpace);
  VariableSetDesc variableSetDesc(spaces,{ "u" });
  VariableSetDesc::VariableSet u(variableSetDesc);



  //construct Galerkin representation
  timer.start("assembly");
  auto mr = assembleMatrix(temperatureSpace);
  auto A = mr.first;
  auto r = boost::fusion::at_c<0>(mr.second.data);
  timer.stop("assembly");

  std::cout << "Setup:" << timer;
  timer.clear();


  // Create multiplicative multigrid preconditioner
  std::cerr << "\nmultiplicative P multigrid with Jacobi smoother\n";
  timer.start("MG creation");
  auto multMG = makeJacobiPMultiGrid(A,temperatureSpace,2,2);
  timer.stop("MG creation");
  timer.start("solution");
  runMultiGrid(A,r,multMG,u,2,"/tmp/mulp1-"+paddedString(order),steps);
  timer.stop("solution");
  std::cout << timer;
  timer.clear();

  // Create multiplicative multigrid preconditioner
  std::cerr << "\nmultiplicative P multigrid with Schwarz smoother\n";
  timer.start("MG creation");
  auto multMG2 = makeBlockJacobiPMultiGrid(A,temperatureSpace,2,2);
  timer.stop("MG creation");
  timer.start("solution");
  runMultiGrid(A,r,multMG2,u,2,"/tmp/mulp2-"+paddedString(order),steps);
  timer.stop("solution");
  std::cout << timer;
  timer.clear();

  // Create multiplicative multigrid preconditioner
  std::cerr << "\nmultiplicative P multigrid with Schwarz smoother (P1 assembly)\n";
  timer.start("MG creation");
  timer.start("assembly");
  auto cA = assembleMatrix(p1Space).first;
  timer.stop("assembly");
  auto multMG3 = makeBlockJacobiPMultiGrid(A,temperatureSpace,cA,p1Space,2,2);
  timer.stop("MG creation");
  timer.start("solution");
  runMultiGrid(A,r,multMG3,u,2,"/tmp/mulp3-"+paddedString(order),steps);
  timer.stop("solution");
  std::cout << timer;
  timer.clear();

  // Create multigrid preconditioner
  std::cerr << "\nadditive PBPX multigrid\n";
  timer.start("MG creation");
  auto addMG = makePBPX(A,temperatureSpace,p1Space);
  timer.stop("MG creation");
  timer.start("solution");
  runMultiGrid(A,r,addMG,u,2,"/tmp/addp-"+paddedString(order),steps);
  timer.stop("solution");
  std::cout << timer;
  timer.clear();

  // Create default multigrid preconditioner
  std::cerr << "\ndefault multigrid\n";
  timer.start("MG creation");
  auto defMG = makeMultigrid(A,temperatureSpace);
  timer.stop("MG creation");
  timer.start("solution");
  runMultiGrid(A,r,*defMG,u,1,"/tmp/add-",steps);
  timer.stop("solution");
  std::cout << timer;
  timer.clear();

  // compare with direct solver
  std::cout << "\ndirect solver j=" << refinements << "\n";
  timer.start("factorization");
  using Vector = Dune::BlockVector<Dune::FieldVector<double,1>>;
  DirectSolver<Vector,Vector> direct(A,DirectType::MUMPS);
  timer.stop("factorization");
  timer.start("solution");
  Vector dx(temperatureSpace.degreesOfFreedom());
  direct.apply(dx,r);
  timer.stop("solution");
  std::cout << timer;
  timer.clear();
}

int main()
{
  int maxSteps = 100;
  std::cout << "Start multigrid test program" << std::endl;
  for (int j=1; j<12; ++j)
    checkP1(j,maxSteps);
  //~ checkPp(9,maxSteps,2);
  //~ checkPp(8,maxSteps,3);
  //~ checkPp(8,maxSteps,4);
  //~ checkPp(7,maxSteps,5);
  //~ checkPp(7,maxSteps,6);
  return 0;
}



#endif
