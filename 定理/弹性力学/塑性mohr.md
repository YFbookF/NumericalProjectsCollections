# MatlabMohr

https://github.com/xz804/MatlabMohrCoulomb

该库附有pdf，解释得很详细

第一步，计算visco-plastic strain rate
$$
\delta \varepsilon^{vp} = \Delta t F \frac{\partial G}{\partial \sigma}
$$
对应代码为

```
deps_pla(:,kk,iel)=dt*F*dgds(:,kk,iel);
```

其中
$$
dt = \frac{4(1+\nu)(1-\nu)}{E(1-2\nu + \sin^2 \phi)}
$$
在biaxaltest.m中是这么写的

```
dt=4*(1+nu)*(1-2*nu)/(E*(1-2*nu+(sind(phi))^2));
```

然而有两步看不懂也找不到代码
$$
\Delta r = \int B^T C \delta \{\varepsilon\} dV
$$
对应代码为

```
           r(eldof) =r(eldof)+Bfem4'*(C*deps_pla(:,kk,iel))*W(kk)*det(J0);  
    end                 % end of looping on GPs           
          f(sctry)= f(sctry)+r(sctry);  
```

## geo

he material point method for geotechnical engineering a practical guide by Alonso, Eduardo E. Fern, James Rohe, Alexander Soga, Kenichi (z-lib.org).pdf

![image-20211218202732065](E:\mycode\collection\定理\弹性力学\image-20211218202732065.png)

感觉找不到啊

## The material point method.

傲娇地加上一个点。

The Material Point Method. A Continuum-Based Particle Method for Extreme Loading Cases- Academic Press  (2016).pdf

![image-20211218202851720](E:\mycode\collection\定理\弹性力学\image-20211218202851720.png)

## moose库

感觉完全不同？

E:\mycode\plastic\moose-next\modules\tensor_mechanics\test\tests\capped_mohr_coulomb\capped_mohr_coulomb.pdf

![image-20211218203004928](E:\mycode\collection\定理\弹性力学\image-20211218203004928.png)

## 物质点法_134

![image-20211218203044436](E:\mycode\collection\定理\弹性力学\image-20211218203044436.png)

## Nemesis库

这已经是组装矩阵了

```
  // coordinate transformation
  sTrial[0] = s[0]*sV(0, 0)*sV(0, 0)
             +s[1]*sV(1, 0)*sV(1, 0)
             +s[2]*sV(2, 0)*sV(2, 0);
  sTrial[1] = s[0]*sV(0, 1)*sV(0, 1)
             +s[1]*sV(1, 1)*sV(1, 1)
             +s[2]*sV(2, 1)*sV(2, 1);
  sTrial[2] = s[0]*sV(0, 2)*sV(0, 2)
             +s[1]*sV(1, 2)*sV(1, 2)
             +s[2]*sV(2, 2)*sV(2, 2);
  sTrial[3] = s[0]*sV(0, 0)*sV(0, 1)
             +s[1]*sV(1, 0)*sV(1, 1)
             +s[2]*sV(2, 0)*sV(2, 1);
  sTrial[4] = s[0]*sV(0, 1)*sV(0, 2)
             +s[1]*sV(1, 1)*sV(1, 2)
             +s[2]*sV(2, 1)*sV(2, 2);
  sTrial[5] = s[0]*sV(0, 0)*sV(0, 2)
             +s[1]*sV(1, 0)*sV(1, 2)
             +s[2]*sV(2, 0)*sV(2, 2);

  // check
  f[0]=(s[0]-s[2])+(s[0]+s[2])*sin(phi)-2*c*cos(phi);
  f[1]=(s[1]-s[2])+(s[1]+s[2])*sin(phi)-2*c*cos(phi);
  f[2]=(s[0]-s[1])+(s[0]+s[1])*sin(phi)-2*c*cos(phi);
```

解方程

```
    A.Resize(3+active.size(), 3+active.size(), 0.);
    x.Resize(3+active.size());
    R.Resize(3+active.size());
    R.Clear();
    A.Append(C3, 0, 0);
    for (unsigned i = 0; i < active.size(); i++) {
      A.AppendCol(dg[active[i]],  0, 3+i);
      A.AppendRow(df[active[i]], 3+i,  0);
      R[3+i]=-f[active[i]];
    }
    // solve
    A.Solve(x, R);
    // check
    bool restart = false;
    for (unsigned i = 0; i < active.size(); i++) {
      if (x[3+i] < 0.) {
        active.erase(active.begin()+i, active.begin()+i+1);
        restart = true;
      }
    }
    if (restart) continue;
    // update
    for (int i = 0; i < 3; i++) s[i]+=x[i];
    break;
```

## MultiSurface

RoundedMohrCoulomb_multi_surface.cpp

```
double RoundedMohrCoulomb_multi_surface::yield_surface_val(DTensor2 const& stress, DTensor2 const& alpha, double yield_sz){


    double pp = -1./3. * (stress(0,0)+stress(1,1)+stress(2,2)) ;
    DTensor2 s(3,3,0.);
    s(i,j) = stress(i,j) + pp * kronecker_delta(i,j) ;

    s(i,j) -= pp * alpha(i,j) ;

    return sqrt( s(i,j) * s(i,j) ) - Rtheta(stress, alpha) * sqrt(2./3.) * yield_sz * (pp - pc);
}

    result(i, j) += Rtheta(stress, curr_alpha) * sqrt(2./27.) * curr_sz * kronecker_delta(i, j);
    static DTensor2 dr_dsigma(3,3,0.);
    dR_dsigma(stress, curr_alpha, dr_dsigma);
    result(i, j) -= dr_dsigma(i,j) * sqrt(2./3.) * curr_sz * pp ;
```

# MPMdev

E:\mycode\materialpoint\mpm-develop\tests\materials\mohr_coulomb_test.cc

```
    // Calculate modulus values
    const double K =
        material->template property<double>("youngs_modulus") /
        (3.0 *
         (1. - 2. * material->template property<double>("poisson_ratio")));
    const double G =
        material->template property<double>("youngs_modulus") /
        (2.0 * (1. + material->template property<double>("poisson_ratio")));
    const double a1 = K + (4.0 / 3.0) * G;
    const double a2 = K - (2.0 / 3.0) * G;
    // Compute elastic tensor
    mpm::Material<Dim>::Matrix6x6 de;
    de.setZero();
    de(0, 0) = a1;
    de(0, 1) = a2;
    de(0, 2) = a2;
```

# NTNU

E:\mycode\Elastic\Uintah_NTNU-main\src\CCA\Components\MPM\Materials\ConstitutiveModel\SoilModels\ShengMohrCoulomb.cc

```
ShengMohrCoulomb::ShengMohrCoulomb(void)
{

	// all the parameters are initialized
	G=10000;		//shear modulus
	K=20000;		//bulk modulus
	E=9*K*G/(G+3*K);		//Young modulus
	Poisson=(3*K-2*G)/(2*G+6*K);	//Poisson ratio

	//Mohr - Coulomb parameters

	Cohesion=0;
	Phi=3.1415/6.0;
	SinPhi=sin(Phi);
	CosPhi=cos(Phi);


	//if the flow rule is not associated; not expected...
	NonAssociated=false;
	Psi=Phi;
	SinPsi=sin(Psi);
	CosPsi=cos(Psi);
	//Rounded Mohr-Coulomb parameters
	Alpha=(3.0-SinPhi)/(3.0+SinPhi);
	Alpha4=pow(Alpha,4); //Alpha^4;

}
```

```
double ShengMohrCoulomb::CalcStressElast (double nu0, double* s0, double* eps0,  double* deps,  double* ds)
{
double K43G=K+4.0*G/3.0;
double K23G=K-2.0*G/3.0;


ds[0]=K43G*deps[0]+K23G*(deps[1]+deps[2]);
ds[1]=K43G*deps[1]+K23G*(deps[0]+deps[2]);
ds[2]=K43G*deps[2]+K23G*(deps[0]+deps[1]);
ds[3]=2*G*deps[3];
ds[4]=2*G*deps[4];
ds[5]=2*G*deps[5];
```

E:\mycode\Elastic\chrono-develop\src\chrono\fea\ChContinuumMaterial.cpp

```
void ChContinuumDruckerPrager::Set_from_MohrCoulomb(double phi, double cohesion, bool inner_approx) {
    if (inner_approx) {
        alpha = (2 * sin(phi)) / (sqrt(3.0) * (3.0 - sin(phi)));
        elastic_yeld = (6 * cohesion * cos(phi)) / (sqrt(3.0) * (3.0 - sin(phi)));
    } else {
        alpha = (2 * sin(phi)) / (sqrt(3.0) * (3.0 + sin(phi)));
        elastic_yeld = (6 * cohesion * cos(phi)) / (sqrt(3.0) * (3.0 + sin(phi)));
    }
}
```

求导

```
            ChStressTensor<> aux_dFdS_C;
            this->ComputeElasticStress(aux_dFdS_C, dFdS);

            ChMatrixNM<double, 1, 1> inner_up;
            inner_up = aux_dFdS_C.transpose() * mincrementstrain;
            ChMatrixNM<double, 1, 1> inner_dw;
            inner_dw = aux_dFdS_C.transpose() * dGdS;

            mplasticstrainflow = dGdS;
            mplasticstrainflow *= inner_up(0) / inner_dw(0);
```

=======================

D:\图形学书籍\论文\Anisotropic Elastoplasticity for Cloth, Knit and Hair Frictional Contact.pdf

![image-20211221224647829](E:\mycode\collection\定理\弹性力学\image-20211221224647829.png)

============================

D:\图形学书籍\图形学书籍\Siggraph\misc\Animating Sand as a Fluid.pdf

![image-20211221233500961](E:\mycode\collection\定理\弹性力学\image-20211221233500961.png)
