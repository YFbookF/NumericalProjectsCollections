axisfem_BiLinearPlasticityModel.cpp

```
    afb::VectorProduct(aux, 1.0, Ce, flowDirectionVec);
    real denominator = afb::VectorScalarProduct(flowDirectionVec, aux);
    denominator += H;
    dLambda = MATH_ABS(numerator / denominator);

    // Determine plastic strain increment (explicit integration)
    afb::ColumnVector dPlasticStrain(6);
    dPlasticStrain = flowDirectionVec;
    dPlasticStrain *= dLambda*dt;
    updatedState.PlasticStrain() += dPlasticStrain;

    // Update effective plastic strain (explicit integration)
    updatedState.EffectivePlasticStrain() += dLambda*dt;

    // Calculate rate of plastic deformation
    afb::SymmetricMatrix Dp(3);
    Dp = flowDirection; Dp *= dLambda;
```

![image-20211218215503877](E:\mycode\collection\定理\弹性力学\image-20211218215503877.png)

![image-20211218215904226](E:\mycode\collection\定理\弹性力学\image-20211218215904226.png)

# J2

![image-20211218205204880](E:\mycode\collection\定理\弹性力学\image-20211218205204880.png)

multisurface

```
    // DTensor2 curr_alpha = iterate_alpha_vec[N_active_ys];
    // double pp = -1./3. * (stress(0,0)+stress(1,1)+stress(2,2)) ;
    // DTensor2 DevStress(3,3,0.);
    // DevStress(i,j) = stress(i,j) + pp * kronecker_delta(i,j) ;
    // DevStress(i,j) -= pp * curr_alpha(i,j) ;
    // double J2bar = 1.5 * DevStress(i,j) * DevStress(i,j);
    // stress_ratio2 = J2bar / pow(abs(I1/3.),2);
```



# D:\图形学书籍\图形学书籍\固体物理\Computational Continuum Mechanics by Ahmed A. Shabana (z-lib.org).pdf

![image-20211220102405376](E:\mycode\collection\定理\弹性力学\image-20211220102405376.png)

![image-20211220102538969](E:\mycode\collection\定理\弹性力学\image-20211220102538969.png)

===========================chrono

```
void ChContinuumPlasticVonMises::ComputePlasticStrainFlow(ChStrainTensor<>& mplasticstrainflow,
                                                          const ChStrainTensor<>& mtotstrain) const {
    double vonm = mtotstrain.GetEquivalentVonMises();
    if (vonm > this->elastic_yeld) {
        ChVoightTensor<> mdev;
        mtotstrain.GetDeviatoricPart(mdev);
        mplasticstrainflow = mdev * ((vonm - this->elastic_yeld) / (vonm));
    } else {
        mplasticstrainflow.setZero();
    }
}
```

