#include <vector>

#include <fem/fixdune.hh>
#include <linalg/tcg.hh>

#include "schurblocklu_solve.hh"

/** \ingroup linalg 
 *\ brief Adapter class for DUNE::IterativeSolver
 */

namespace Kaskade
{

void printVec(std::vector<double> const&v, int vend)
{
  int endv=v.size();
  if(vend < v.size()) endv=vend;
  for(int i=0; i< endv; ++i)
  {
    if(i%5 == 0)  std::cout << ".  " << v[i] << std::endl;
    else std::cout << "   " << v[i] << std::endl;

  }
}

template<class Factorization>
void BlockLUFactorization<Factorization>::solve(std::vector<double>const& rhs, std::vector<double>& sol, int nr)
{
  int r=rhs.size()/2/nr;
  std::vector<double> rhs1(r*nr);
  std::vector<double> rhs2(r*nr);
  std::vector<double> sol1(r*nr);
  std::vector<double> sol2(r*nr);
  for(int i=0; i<nr; ++i)
  for(int j=0; j<r; ++j)
  {
    rhs1[i*r+j]=rhs[i*2*r+j];
    rhs2[i*r+j]=-rhs[i*2*r+r+j];
  }
  factoredL->solve(rhs2,sol2,nr,false);  // solve system
  matA.axpy(rhs1,sol2,1.0,nr);
  factoredL->solve(rhs1,sol1,nr,true);  //solve transposed system
  sol.resize(2*r*nr);
  for(int i=0; i<nr; ++i)
  for(int j=0; j<r; ++j)
  {
    sol[i*2*r+j]=-sol2[i*r+j];
    sol[i*2*r+r+j]=sol1[i*r+j];
  }
};


//  UU UY BU* ...  C B* B*
//  YU YY AY* ...  B A  A
//  BU AY 00  ...  B A  A
//  .. .. ..  ...
template<class Factorization>
void DirectBlockSchurSolver<Factorization>::buildNewSchurComplement(SparseLinearSystem const& lin,int task=0)
{
    boost::timer::cpu_timer timer;
    if(report) std::cout << "Schur Complement: " << std::flush;

    MatrixAsTriplet<double> matB, matC;
    rowsB=lin.rows(start2,end3);
    colsBC=lin.cols(start1,end1);
    rowsC=lin.rows(start1,end1);
    if(task==0)
    {
      rows1=lin.rows(start1,end1);
      rows2=lin.rows(start2,end2);
      rows3=lin.rows(start3,end3);
    }

    if(paras.refactorizeOuter || !factorization.get() || B.size()==0)
    {
      if(report) std::cout << "Outer, " << std::flush;
      if(task==0)
      {
        flushFactorization();
      }

      if((task==0 && paras.refactorizeInner) || !factorization.get())
      {
        factorization.reset(new BlockLUFactorization<Factorization>(lin,start2,end2,start3,end3));
        lin.getMatrixBlocks(matANormal,start2,end2, start2, end2);
      }
      else
      {
        MatrixAsTriplet<double> matA;
        lin.getMatrixBlocks(matA,start2,end2, start2, end2);
        factorization->resetBlock22(matA);
      }

  
// Matrix = [C B^T; B A]; rhs=[r2, r1]; sol=[x_2,x_1]
// Compute sol via the Schur complement
// B is stored in column-first formal in a vector B(i,j)=B[i+j*rowsB]
      if(task==0 || B.size()==0)
      {
        matB.resize(0);
        lin.getMatrixBlocks(matB,start2,end3, start1, end1);
        B.resize(0);
        matB.toVector(B);
      }

// Compute A^{-1}B
      AinvB.resize(0);
      factorization->solve(B,AinvB,colsBC);

    }


// Compute C as a full matrix
    matC.resize(0);
    lin.getMatrixBlocks(matC,start1,end1, start1, end1);
    mC.setSize(rowsC, colsBC);
    mC=0.0;
    matC.addToMatrix(mC);

// Compute S := B^T A^{-1}B-C
    for(int i=0; i<rowsC; ++i)
      for(int j=0; j<colsBC; ++j)
      {
        mC[i][j] *= -1.0;
        for(int k=0; k<rowsB; ++k) mC[i][j] += B[i*rowsB+k]*AinvB[j*rowsB+k];
      }
    if(task==0)
    {
      mCNormal.setSize(rowsC, colsBC);
      mCNormal = mC;
    }
    if(report) std::cout << "Finished: " << (double)(timer.elapsed().user)/1e9 << " sec." << std::endl;
}


template<class Factorization>
void DirectBlockSchurSolver<Factorization>::resolve(std::vector<double>& sol, 
                                                    SparseLinearSystem const& lin) const
  {
    boost::timer::cpu_timer timer2;

// Compute xx_1=A^{-1}r_1
    std::vector<double> r,s,t, sol2;
    lin.getRHSBlocks(r,start1,end1);
    lin.getRHSBlocks(s,start2,end2);
    lin.getRHSBlocks(t,start3,end3);

// Compute xx_1=A^{-1}r_1
    std::vector<double> x1,x2,xx1, xx2, r1, r2;

    fwd(xx2,r,s,t);

// Compute x2 := S^{-1}xx_2
    x2.resize(xx2.size(),0.0);

    if(paras.regularizationMethod==BlockSchurParameters::IterateType::CG)
    {
      std::cout << "TCG..." << std::flush;
      MatrixAsTriplet<double> C(mC);
      C *=-1.0;
      std::vector<double> rhs(xx2.size());
      for(int i=0; i<xx2.size(); ++i) rhs[i]=-xx2[i];
      std::vector<double> p(xx2.size());
      int exit = tcg(x2, p, C, rhs,1e-8,100);
      std::cout << "Exit with: " << exit << std::endl;
      if(exit <= 2)
      {
        std::vector<double> x3;
        double sum(0.0),sum2(0.0);
        x3.resize(xx2.size(),0.0);
   
        LeastSquares(SLAPMatrix<double>(mC), xx2, x3);
        for(int i=0; i<x3.size(); ++i) 
        {
          sum += (x3[i]-x2[i])*(x3[i]-x2[i]);
          sum2 += (x3[i])*(x3[i]);
        }
        std::cout << sqrt(sum) << std::endl;
        std::cout << sqrt(sum2) << std::endl;
//        std::swap(x2,x3);
      }
     
    }
    else 
    {
      LeastSquares(SLAPMatrix<double>(mC), xx2, x2);
    }

    bwd(sol,x2,s,t);
  }


template<class Factorization>
void DirectBlockSchurSolver<Factorization>::tsolve(std::vector<double>& sol1,std::vector<double>& sol2, 
                                                   std::vector<double>const &r,std::vector<double>const &s,std::vector<double>const &t) const
{
  std::vector<double> xx2,x2;
  fwd(xx2,r,s,t);

// Compute x2 := S^{-1}xx_2
  x2.resize(xx2.size(),0.0);

   std::cout << "tsolve..." << std::flush;
    
//     MatrixAsTriplet<double> C(mC);

//   C *=-1.0;
//   std::vector<double> rhs(xx2.size());
//   for(int i=0; i<xx2.size(); ++i) rhs[i]=-xx2[i];
 
//   std::vector<double> p(xx2.size(),0.0);
//   int exit = tcg(x2, p, C, rhs,1e-8,100);
//   std::cout << "Exit with: " << exit << std::endl;
  

   
     SLAPMatrix<double> C(mC);

     std::vector<double> eig(mC.N());
     SymmetricEigenvalues(SLAPMatrix<double>(mC),eig);
//     if(report >2) 
       for(int i=0; i<eig.size(); ++i) std::cout << eig[i] << std::endl;    
     double maxeig=-10000000;
     double maxabseig=0;
     for(int i=0; i<eig.size();++i) maxeig = std::max(maxeig,eig[i]);
     for(int i=0; i<eig.size();++i) maxabseig = std::max(maxabseig,std::fabs(eig[i]));
     if(maxeig  > 0) 
     {
       std::cout << "Adding an identity matrix times " <<  1.1*maxeig << std::endl;
       for(int i=0; i<eig.size();++i) C(i,i) += -1.1*maxeig;
     }
   

     LeastSquares(C, xx2, x2);
   
  
  bwd(sol1,x2,s,t);
//   if(exit==2) 
//     bwd(sol2,p,s,t);
//   else
    sol2.resize(0);
}

template<class Factorization>
void DirectBlockSchurSolver<Factorization>::resolveN(std::vector<double>& sol, std::vector<double>const &r,std::vector<double>const &s,std::vector<double>const &t) const
{
  std::vector<double> xx2;
  fwd(xx2,r,s,t);
  std::vector<double> x2(xx2.size(),0.0);
// Compute x2 := S^{-1}xx_2

  LeastSquares(SLAPMatrix<double>(mCNormal), xx2, x2);  

  bwd(sol,x2,s,t);
}

template<class Factorization>
void DirectBlockSchurSolver<Factorization>::ax(std::vector<double>& sol, std::vector<double>const &r) const
{
  std::vector<double> r1(rows1),r2(rows2),r3(rows3,0.0), sol2(rows1+rows2+rows3);
  for(int i=0; i<r1.size(); ++i) r1[i]=r[i];
  for(int i=0; i<r2.size(); ++i) r2[i]=r[i+r1.size()];
  if(r.size() >= r1.size()+r2.size()+r3.size())
    for(int i=0; i<r3.size(); ++i) r3[i]=r[i+r1.size()+r2.size()];
  resolveN(sol2,r1,r2,r3);
  for(int i=0; i<sol.size(); ++i) sol[i]=sol2[i];
}

template<class Factorization>
void DirectBlockSchurSolver<Factorization>::fwd(std::vector<double>& sol, std::vector<double>const &r,std::vector<double>const &s,std::vector<double>const &t) const
{
    std::vector<double> xx1, r1;

    r1.reserve(s.size()+t.size());

    for(int i=0; i<s.size();++i) r1.push_back(s[i]);
    for(int i=0; i<t.size();++i) r1.push_back(t[i]);
// Compute xx_1=A^{-1}r_1

    xx1.resize(r1.size());
    factorization->solve(r1,xx1);

// Compute xx_2 := -r_2+B^T xx_1

    sol.resize(r.size());
    for(int i=0; i<r.size(); ++i)
    {
      sol[i]=-r[i];
      for(int k=0; k<xx1.size(); ++k)
        sol[i]+=B[i*rowsB+k]*xx1[k];
    }

// Compute x2 := S^{-1}xx_2
}

template<class Factorization>
void DirectBlockSchurSolver<Factorization>::bwd
(std::vector<double>& sol, std::vector<double>const &x2,std::vector<double>const &s,std::vector<double>const &t) const
{
  std::vector<double> xx1,x1;

   xx1.reserve(s.size()+t.size());

    for(int i=0; i<s.size();++i) xx1.push_back(s[i]);
    for(int i=0; i<t.size();++i) xx1.push_back(t[i]);

// Compute xx_1 := r_1-B x_2
    for(int i=0; i<xx1.size(); ++i)
      for(int k=0; k<x2.size(); ++k)
        xx1[i]-=B[k*rowsB+i]*x2[k];

// Compute x1 = A^{-1}xx_1
    factorization->solve(xx1,x1);    

// sol=[x2;x1]
    sol.reserve(x2.size()+x1.size());//x2.size()+x1.size());
    sol.resize(x2.size()+x1.size(),0.0);
    int k=0;
    for(int i=0; i< x2.size(); ++i)
    {
      sol[k]=x2[i];
      ++k;
    }
    for(int i=0; i< x1.size(); ++i)
    {
      sol[k]=x1[i];
      ++k;
    }
}


template<class Factorization>
void DirectBlockSchurSolver<Factorization>::resolveAdjAndNormal(std::vector<double>& sol, 
                                                    SparseLinearSystem const& lin) const
  {
    std::cout << "Computing adjoint and normal step..." << std::endl;

// Compute xx_1=A^{-1}r_1
    std::vector<double> r,s,t, sol2, sol3;
    lin.getRHSBlocks(r,start1,end1);
    lin.getRHSBlocks(s,start2,end2);
    lin.getRHSBlocks(t,start3,end3);
    std::vector<double> r0(r.size(),0.0),s0(s.size(),0.0),t0(t.size(),0.0);
    resolveN(sol,r0,s0,t);
    resolveN(sol3,r,s,t0);
    for(int i=r0.size()+s0.size(); i<sol.size();++i) sol[i]=sol3[i];
  }

template<class Factorization>
void DirectBlockSchurSolver<Factorization>::resolveNormal(std::vector<double>& sol, 
                                                          SparseLinearSystem const& lin,
                                                          std::vector<double> const* addrhs)
  {
    std::cout << "Computing simplified normal step..." << std::endl;
    
// Compute xx_1=A^{-1}r_1
    std::vector<double> t;
    std::vector<double> r0(rows1,0.0),s0(rows2,0.0);
    lin.getRHSBlocks(t,start3,end3);
    if(addrhs)
    {
      assert(t.size()==(*addrhs).size());
      for(int i=0; i<t.size();++i) t[i]+= (*addrhs)[i];
    }
    factorization->resetBlock22(matANormal);
    
    resolveN(sol,r0,s0,t);
    for(int i=r0.size()+s0.size(); i<sol.size();++i) sol[i]=0.0;
  }


template<class Factorization>
void DirectBlockSchurSolver<Factorization>::solve(std::vector<double>& sol,
                                                  SparseLinearSystem const& lin)
  {

    buildNewSchurComplement(lin,0);
    if(paras.regularizationMethod==BlockSchurParameters::AddId)
    {
      
      std::vector<double> eig(mC.N());
      SymmetricEigenvalues(SLAPMatrix<double>(mC),eig);
      if(report >2) for(int i=0; i<eig.size(); ++i) std::cout << eig[i] << std::endl;    
      double maxeig=-10000000;
      double maxabseig=0;
      for(int i=0; i<eig.size();++i) maxeig = std::max(maxeig,eig[i]);
      for(int i=0; i<eig.size();++i) maxabseig = std::max(maxabseig,std::fabs(eig[i]));
      if(maxeig  > 0) 
      {
        std::cout << "Adding an identity matrix times " <<  1.1*maxeig << std::endl;
        for(int i=0; i<eig.size();++i) mC[i][i] += -1.1*maxeig;
      }
    }
    resolve(sol,lin);
  }

template<class Factorization>
void DirectBlockSchurSolver<Factorization>::solveAdjAndNormal(std::vector<double>& sol,
                                                  SparseLinearSystem const& lin)
  {

    buildNewSchurComplement(lin,0);
    resolveAdjAndNormal(sol,lin);
  }

template<class Factorization>
void DirectBlockSchurSolver<Factorization>::solveTCG(std::vector<double>& sol1, std::vector<double>& sol2,
                                                     SparseLinearSystem const& linT, SparseLinearSystem const& linN, std::vector<double> const& normalStep, double nu0)
  {
    buildNewSchurComplement(linT,1);
    std::cout << "Computing tangential step..." << std::endl;
    std::vector<double> r(rows1),s(rows2),t(rows3,0.0);
    linN.getRHSBlocks(r,start1,end1);
    linN.getRHSBlocks(s,start2,end2);
    MatrixAsTriplet<double> mu,my;
    linT.getMatrixBlocks(mu,start1,end1,start1,end2);
    linT.getMatrixBlocks(my,start2,end2,start1,end2);
    mu.axpy(r,normalStep,nu0);
    my.axpy(s,normalStep,nu0);
    tsolve(sol1,sol2,r,s,t);
  }

//#include "problems/hyperthermia/fsip_amplituderatio/hypfsip_aux.hh"

template<class Factorization>
void ARDirectBlockSchurSolver<Factorization>::resolve(std::vector<double>& sol, 
                                                    SparseLinearSystem const& lin) const
  {
    if(report) std::cout << std::endl << "Substitution..." << std::endl;
    boost::timer::cpu_timer timer2;

    std::vector<double> x1, r2, xu, x2;
    if(!justsolved)
    {
      linMod->resetLin(lin);
      DBSSolver.resolve(x1,*linMod);
    }
      

    if(!justsolved) for(int i=0; i<lin.cols(0,1); ++i) xu.push_back(x1[i]);
    else  for(int i=0; i<lin.cols(0,1); ++i) xu.push_back(sol[i]);


    MatrixAsTriplet<double> mF(F);

    mF.scaleRows(scaling);

    lin.getRHSBlocks(r2,3,4);
    for(int i=0; i<r2.size();++i)
      r2[i]*=scaling[i];


    mF.axpy(r2,xu,-1.0);

    MatrixAsTriplet<double> mV(Vinv);

    x2.resize(linMod->cols(3,4));

    mV.ax(x2,r2);

//    mV.print(1e-8);
//    printVec(r2);
    

    sol.reserve(lin.size());
    sol.resize(lin.size(),0.0);
    
     if(!justsolved) 
      for(int i=0; i< x1.size(); ++i) sol[i]=x1[i];

    int i0=lin.cols(0,3);
    for(int i=0; i< x2.size(); ++i) 
    {
//      std::cout << x2[i] << std::endl;
      sol[i+i0]=x2[i];
    }
    
    MatrixAsTriplet<double> mata;

    lin.getMatrixBlocks(mata,0,4, 0, 4);
    Factorization factoredL(mata.nrows(),
                            11,
                            mata.ridx, 
                            mata.cidx, 
                            mata.data);

    std::vector<double> rhs,sol2;
    lin.getRHSBlocks(rhs,0,4);
    factoredL.solve(rhs,sol2);
    mata.axpy(rhs,sol2,-1.0);
    for(int i=0;i<rhs.size();++i)
      if(std::fabs(sol[i]-sol2[i]) > 1e-12) std::cout << i << " " << sol[i] << " " << sol2[i] << " " << rhs[i] << std::endl;



//    for(int i=0; i<scaling.size();++i) std::cout << scaling[i] << std::endl;
    if(report)  std::cout <<  "Finished: " << (double)(timer2.elapsed().user)/1e9 << " sec." << std::endl;

  }

template<class Factorization>
void ARDirectBlockSchurSolver<Factorization>::solve(std::vector<double>& sol,
                                                  SparseLinearSystem const& lin)
  {
    DBSSolver.report = report;
    MatrixAsTriplet<double> matV, matF, matFT, matU,matA, matAll;
    flushFactorization();
    if(report) std::cout << "Assembling..." << std::endl;

    lin.getMatrixBlocks(matU,0,1, 0, 1);
    lin.getMatrixBlocks(matF,3,4, 0, 1);
    lin.getMatrixBlocks(matFT,0,1,3,4);
    lin.getMatrixBlocks(matV,3,4, 3, 4);
    lin.getMatrixBlocks(matAll,0,4, 0, 4);
//    matAll.print();
    lin.getRHSBlocks(scaling,4,5);
    matV.scaleRows(scaling);

    
//     typedef Bridge::HypLinearization<Bridge::IPLinearization<Hyperthermia::IPFunc,typename Hyperthermia::IPFunc::AnsatzVars::VariableSet>,
//       Hyperthermia::Variables> HL; 

//     Bridge::Linearization<HL>const & linS(
//     dynamic_cast<Bridge::Linearization<HL>const & >(lin));

// Hyperthermia::Variables o(linS.getOriginImpl());

// std::vector<std::vector<double> > rhsP(51);
// std::vector<double> rhs;

//     linS.getRHSBlocks(rhs,0,4);
//     double h=1e-6;
//     using namespace boost::fusion;

// for(int i=0; i<51; i++)
// {
//   (*at_c<3>(o.vars))[0][i] += h;
//   linS.getLinImpl().init(o);
//     linS.getRHSBlocks(rhsP[i],0,4);
//     (*at_c<3>(o.vars))[0][i] -= h;
// }
// linS.getLinImpl().init(o);


// std::cout << rhs.size() << "[" << std::endl;
// for(int i=0; i<rhs.size(); ++i)
// {
// for(int j=0; j<51; ++j)
// {
//   if(std::fabs(rhsP[j][i]-rhs[i])/h > 1e-10)
//     std::cout << i << " " << j << " "<< (rhsP[j][i]-rhs[i])/h << std::endl;
// }
// }
// std::cout << "]" << std::endl;

    Vinv.setSize(lin.rows(3,4), lin.cols(3,4)); Vinv =0.0;
    F.setSize(lin.rows(3,4), lin.cols(0,1)); F = 0.0;
    FT.setSize(lin.rows(0,1), lin.cols(3,4)); FT=0.0;

    matF.addToMatrix(F);
    matFT.addToMatrix(FT);


    SLAPMatrix<double> VinvS(lin.rows(3,4), lin.cols(3,4)), W(matV);
    std::cout << W.rows() << W.cols() << std::endl;
    pseudoinverse(W,VinvS);
    VinvS.toMatrix(Vinv);

// FTVinvF= F^T * V^{-1} * F
    MatrixAsTriplet<double> Vinvs2(Vinv);
//    Fs.print(1e-12);
//    Vinvs2.print(1e-12);

    MatMult(FTVinv,FT,Vinv);               
    MatMult(FTVinvF,FTVinv,F);

    L.reset(new MatrixAsTriplet<double>(FTVinvF));
    FTVi.reset(new MatrixAsTriplet<double>(FTVinv));

    linMod.reset(new ModifiedSparseSystem(lin,*L,*FTVi,scaling));

    DBSSolver.solve(sol,*linMod);
    
//    justsolved=true;
    resolve(sol,lin);
    justsolved=false;

    MatrixAsTriplet<double> Vinvs(Vinv);
    MatrixAsTriplet<double> Fs(F);
//    Fs.print(1e-12);
//    Vinvs.print(1e-12);
//    L->print(1e-6);
//    FTVi->print(1e-12);

  }



template class BlockLUFactorization<UMFFactorization<double> >;
//template class BlockLUFactorization<PARDISOFactorization<double> >;
//template class BlockLUFactorization<MUMPSFactorization<double> >;
//template class BlockLUFactorization<SUPERLUFactorization<double> >;
template class DirectBlockSchurSolver<UMFFactorization<double> >;
//template class DirectBlockSchurSolver<PARDISOFactorization<double> >;
//template class DirectBlockSchurSolver<MUMPSFactorization<double> >;
//template class DirectBlockSchurSolver<SUPERLUFactorization<double> >;

}  // namespace Kaskade