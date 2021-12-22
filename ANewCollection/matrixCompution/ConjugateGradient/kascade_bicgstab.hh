#ifndef BICGSTAB_HH
#define BICGSTAB_HH

#include<cmath>
#include<complex>
#include<iostream>
#include<iomanip>
#include<string>

#include "dune/istl/istlexception.hh"
#include "dune/istl/operators.hh"
#include "dune/istl/preconditioners.hh"
#include "dune/istl/scalarproducts.hh"
#include <dune/common/timer.hh>
#include <dune/common/static_assert.hh>

  /** @defgroup ISTL_Solvers Iterative Solvers
      @ingroup ISTL
  */
  /** @addtogroup ISTL_Solvers
    @{
  */


  /** \file
      
  \brief   Define general, extensible interface for inverse operators.
  
  Implementation here covers only inversion of linear operators,
  but the implementation might be used for nonlinear operators
  as well.
  */

  /** 
      \brief Statistics about the application of an inverse operator

      The return value of an application of the inverse
      operator delivers some important information about 
      the iteration.
  */



  // Ronald Kriemanns BiCG-STAB implementation from Sumo
  //! \brief Bi-conjugate Gradient Stabilized (BiCG-STAB)
  template<class X>
  class KBiCGSTABSolver : public InverseOperator<X,X> {
  public:
    //! \brief The domain type of the operator to be inverted.
    typedef X domain_type;
    //! \brief The range type of the operator to be inverted.
    typedef X range_type;
    //! \brief The field type of the operator to be inverted
    typedef typename X::field_type field_type;

    /*! 
      \brief Set up solver.

      \copydoc LoopSolver::LoopSolver(L&,P&,double,int,int)
    */
    template<class L, class P>
    KBiCGSTABSolver (L& op, P& prec,
        double reduction, int maxit, int verbose) : 
      ssp(), _op(op), _prec(prec), _sp(ssp), _reduction(reduction), _maxit(maxit), _verbose(verbose)
    {
        dune_static_assert(static_cast<int>(L::category) == static_cast<int>(P::category), "L and P must be of the same category!");
        dune_static_assert(static_cast<int>(L::category) == static_cast<int>(SolverCategory::sequential), "L must be sequential!");
    }
    /*! 
      \brief Set up solver.

      \copydoc LoopSolver::LoopSolver(L&,S&,P&,double,int,int)
    */
    template<class L, class S, class P>
    KBiCGSTABSolver (L& op, S& sp, P& prec,
        double reduction, int maxit, int verbose) : 
      _op(op), _prec(prec), _sp(sp), _reduction(reduction), _maxit(maxit), _verbose(verbose)
    {
        dune_static_assert( static_cast<int>(L::category) == static_cast<int>(P::category),
                            "L and P must have the same category!");
        dune_static_assert( static_cast<int>(L::category) == static_cast<int>(S::category),
                            "L and S must have the same category!");
    }

    /*! 
      \brief Apply inverse operator.

      \copydoc InverseOperator::apply(X&,Y&,InverseOperatorResult&)
    */
    virtual void apply (X& x, X& b, InverseOperatorResult& res)
    {
    const double EPSILON=1e-80;
    
    double                  it;
    field_type           rho, rho_new(0), alpha, beta, h, omega;
    field_type           norm, norm_old, norm_0;
    
    //
    // get vectors and matrix
    //
    X& r=b;
    X p(r);
    X v(r);
    X t(r);
    X y(r);
    X rt(r);

    //
    // begin iteration
    //
    
    // r = r - Ax; rt = r
    res.clear();                // clear solver statistics
    Timer watch;                // start a timer
    _prec.pre(x,r);             // prepare preconditioner
    if(x.size()!=0)
      _op.applyscaleadd(-1,x,r);  // overwrite b with defect
    else
    {
      x.resize(r.size());
      x=0.0;
    }

    rt=r;

    norm = norm_old = norm_0 = _sp.norm(r);
    
    p=0;
    v=0;

    rho   = 1;
    alpha = 1;
    omega = 1;

    if (_verbose>0)             // printing
    {
      std::cout << "=== KBiCGSTABSolver" << std::endl;
      if (_verbose>1) 
      {
        this->printHeader(std::cout);
        this->printOutput(std::cout,0,norm_0);
        //std::cout << " Iter       Defect         Rate" << std::endl;
        //std::cout << "    0" << std::setw(14) << norm_0 << std::endl;
      }
    }

    if ( norm < (_reduction * norm_0)  || norm<1E-30)
    {
      res.converged = 1;
      _prec.post(x);                  // postprocess preconditioner
      res.iterations = 0;             // fill statistics
      res.reduction = 0;
      res.conv_rate  = 0;
      res.elapsed = watch.elapsed();
      return;
    }

    //
    // iteration
    //
    
    for (it = 0.5; it < _maxit; it+=.5)
    {
      //
      // preprocess, set vecsizes etc.
      //
        
      // rho_new = < rt , r >
      if(!(it < 1)) rho_new = _sp.dot(rt,r);

      // look if breakdown occured
      if (std::abs(rho) <= EPSILON)
            DUNE_THROW(ISTLError,"breakdown in KBiCGSTAB - rho "
                       << rho << " <= EPSILON " << EPSILON
                       << " after " << it << " iterations");
          if (std::abs(omega) <= EPSILON)
            DUNE_THROW(ISTLError,"breakdown in KBiCGSTAB - omega "
                       << omega << " <= EPSILON " << EPSILON
                       << " after " << it << " iterations");

        
      if (it<1)
      p = r;
      else
      {
        beta = ( rho_new / rho ) * ( alpha / omega );       
        p.axpy(-omega,v); // p = r + beta (p - omega*v)
        p *= beta;
        p += r;
      }

      // y = W^-1 * p
      y = 0;
      _prec.apply(y,p);           // apply preconditioner

      if(it < 1)
      {
        rt=r;
        rho_new = _sp.dot(rt,r);
      }

      // v = A * y
      _op.apply(y,v);

      // alpha = rho_new / < rt, v >
      h = _sp.dot(rt,v);

      if ( std::abs(h) < EPSILON )
      DUNE_THROW(ISTLError,"h=0 in KBiCGSTAB");
      
      alpha = rho_new / h;
        
      // apply first correction to x
      // x <- x + alpha y
      x.axpy(alpha,y);

      // r = r - alpha*v
      r.axpy(-alpha,v);
        
      //
      // test stop criteria
      //

      norm = _sp.norm(r);

      if (_verbose>1) // print
      {
        this->printOutput(std::cout,it,norm,norm_old);
      }
        
      if ( norm < (_reduction * norm_0) )
      {
        res.converged = 1;
        break;
      }
      it+=.5;
      
      norm_old = norm;

      // y = W^-1 * r
      y = 0;
      _prec.apply(y,r);

      // t = A * y
      _op.apply(y,t);
      
      // omega = < t, r > / < t, t >
      omega = _sp.dot(t,r)/_sp.dot(t,t);
//      omega = _sp.dot(t,r)/_sp.dot(y,t);
        
      // apply second correction to x
      // x <- x + omega y
      x.axpy(omega,y);
        
      // r = s - omega*t (remember : r = s)
      r.axpy(-omega,t);
        
      rho = rho_new;

      //
      // test stop criteria
      //

      norm = _sp.norm(r);

      if (_verbose > 1)             // print
      {
        this->printOutput(std::cout,it,norm,norm_old);
      }
        
      if ( norm < (_reduction * norm_0)  || norm<1E-30)
      {
        res.converged = 1;
        break;
      }

      norm_old = norm;
    } // end for

    if (_verbose>0)                // printing for non verbose
      this->printOutput(std::cout,it,norm);

    _prec.post(x);                  // postprocess preconditioner
    res.iterations = static_cast<int>(std::ceil(it));              // fill statistics
    res.reduction = norm/norm_0;
    res.conv_rate  = pow(res.reduction,1.0/it);
    res.elapsed = watch.elapsed();
    if (_verbose>0)                 // final print 
              std::cout << "=== rate=" << res.conv_rate
                        << ", T=" << res.elapsed
                        << ", TIT=" << res.elapsed/it
                        << ", IT=" << it << std::endl;
    }

    /*! 
      \brief Apply inverse operator with given reduction factor.

      \copydoc InverseOperator::apply(X&,Y&,double,InverseOperatorResult&)
    */
    virtual void apply (X& x, X& b, double reduction, InverseOperatorResult& res)
    {
      _reduction = reduction;
      (*this).apply(x,b,res);
    }

  private:
    SeqScalarProduct<X> ssp;
    LinearOperator<X,X>& _op;
    Preconditioner<X,X>& _prec;
    ScalarProduct<X>& _sp;
    double _reduction;
    int _maxit;
    int _verbose;
  };


#endif
