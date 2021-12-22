https://github.com/HiroUgoto/CamClay/blob/master/src/camclay.py

https://github.com/Geotechnical-Engineering/camclay/blob/master/MCC.m

```
       dS=D*dStrain;
       S=S+dS;
       strain=strain+dStrain;
       
       depsV = dStrain(1) + dStrain (2) + dStrain (3); % Increamental Volumetric Strain
       depsD = 2./3. * (dStrain(1) - dStrain(3));      % Increamental Deviatoric Strain
       
       %Update Specific Volume
       V=N-(l*log(pc))+(k*log(pc/p(a)));
       
       %Subsequent cycle update
       a=a+1;
       p(a)=(S(1)+S(2)+S(3))/3;
       q(a)=S(1)-S(3);
       u(a)=p0+q(a)/3-p(a);
       
       void(a) = V-1.0;
       epsV(a) = epsV(a-1) + depsV;
       epsD(a) = epsD(a-1) + depsD;
       
       if yield<0, yield=q(a)^2+M^2*p(a)^2-M^2*p(a)*pc;
       else yield=0;
```

https://github.com/najafice/CS-Soil-Mechanics/blob/master/ModifiedCamClay.m

```
epsV(i,1)=((l-k)/(V0*p(i)*(M^2+eta^2)))*((M^2-eta^2)*deltap + (2*eta)*deltaq); % Plastic Volumetric Strain
           epsV(i,2)=(k/(V0*p(i)))*deltap; % Elastic Volumetric Strain
           epsV(i,3)=epsV(i,1)+epsV(i,2); % Total Value of Volumetric Strain
                       
           epsS(i,1)=((l-k)/(V0*p(i)*(M^2+eta^2)))*(((2*eta)*deltap) + (((4*eta^2)/(M^2-eta^2))*deltaq)); % Plastic Shear Strain
           epsS(i,2)=(1/(3*G))*deltaq; % Elastic Shear Strain
           epsS(i,3)=epsS(i,1)+epsS(i,2);
           
 epsV(i,1)=((l-k)/((nu*pprime)*(M^2+eta^2))) * (((M^2-eta^2)*deltap) + ((2*eta)*deltaq)); % Plastic Volumetric Strain
           epsS(i,1)=((l-k)/((nu*pprime)*(M^2+eta^2))) * ( ((2*eta)*deltap) + ((4*eta^2)/(M^2-eta^2))*deltaq ); % Plastic Shear Strain
           
           epsV(i,2)=(k/(V0*p(i)))*deltap; % Elastic Volumetric Strain
           epsS(i,2)=(1/(3*G))*deltaq; % Elastic Shear Strain
```

===========CD-MPM: Continuum Damage Material Point Methods for Dynamic  

![image-20211218210835773](E:\mycode\collection\定理\弹性力学\image-20211218210835773.png)

=================

=================

D:\图形学书籍\图形学书籍\固体物理\Computational Methods in Elasticity and Plasticity Solids and Porous Media by A. Anandarajah (auth.) (z-lib.org).pdf

![image-20211218211824510](E:\mycode\collection\定理\弹性力学\image-20211218211824510.png)

===============ziran

E:\mycode\collection\ANewCollection\elasticPlastic\plasticity\ziran_PlasticityApplier.cpp

```
//Hardening Mode
// 0 -> CD-MPM fake p hardening
// 1 -> our new q hardening
// 2 -> zhao2019 exponential hardening

template <class T>
NonAssociativeCamClay<T>::NonAssociativeCamClay(T logJp, T friction_angle, T beta, T xi, int dim, bool hardeningOn, bool qHard)
    : logJp(logJp)
    , beta(beta)
    , xi(xi)
    , hardeningOn(hardeningOn)
    , qHard(qHard)
{
    T sin_phi = std::sin(friction_angle / (T)180 * (T)3.141592653);
    T mohr_columb_friction = std::sqrt((T)2 / (T)3) * (T)2 * sin_phi / ((T)3 - sin_phi);
    M = mohr_columb_friction * (T)dim / std::sqrt((T)2 / ((T)6 - dim));
}

```

===================suanpan-dev

```
		residual(0) = rel_p * rel_p / square_b + square_qm - a * a;

		if(1 == counter && residual(0) < 0.) return SUANPAN_SUCCESS;

		residual(1) = incre_alpha - 2. * gamma / square_b * rel_p;

		jacobian(0, 0) = -2. * six_shear / denom * square_qm;
		jacobian(1, 0) = -2. * rel_p / square_b;
		jacobian(0, 1) = jacobian(1, 0) * (bulk - da) - 2. * a * da;
		jacobian(1, 1) = 1. - 2. * gamma / square_b * (da - bulk);

		if(!solve(incre, jacobian, residual, solve_opts::equilibrate)) return SUANPAN_FAIL;
```

