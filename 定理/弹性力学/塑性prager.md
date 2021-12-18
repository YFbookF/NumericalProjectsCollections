# The material point method.

傲娇地加上一个点。

The Material Point Method. A Continuum-Based Particle Method for Extreme Loading Cases- Academic Press  (2016).pdf

![image-20211218204003026](E:\mycode\collection\定理\弹性力学\image-20211218204003026.png)

![image-20211218205257888](E:\mycode\collection\定理\弹性力学\image-20211218205257888.png)

MultiSurface

```
    double den = sqrt(DevStress(i,j) * DevStress(i,j));
    if (den == 0)
    {
        return result; //Elastic
    }
    else
    {
        result(i, j) =
            (
                DevStress(i,j) + curr_alpha(o, t) * kronecker_delta(i, j) * DevStress(o, t) / 3.
            )
            / den;
    }
    result(i, j) += sqrt(2./27.) * curr_sz * kronecker_delta(i, j);

    return result;


    double pp = -1./3. * (stress(0,0)+stress(1,1)+stress(2,2)) ;
    DTensor2 DevStress(3,3,0.);
    DevStress(i,j) = stress(i,j) + pp * kronecker_delta(i,j) ;
    DevStress(i,j) -= pp * curr_alpha(i,j) ;
}
```

# nemesis

又开始计算了

```
void DruckerPragerNew::set_strain(const Vector& De, const double Dt) {
  // material properties
  double E = myElastic->get_param(0);
  double nu= myElastic->get_param(1);

  // elasticity matrix
  C3(0, 0) =   1/E;
  C3(0, 1) = -nu/E;
  C3(0, 2) = -nu/E;
  C3(1, 0) = -nu/E;
  C3(1, 1) =   1/E;
  C3(1, 2) = -nu/E;
  C3(2, 0) = -nu/E;
  C3(2, 1) = -nu/E;
  C3(2, 2) =   1/E;

  // spectral decomposition
  Vector s(3), De3(3);
  Matrix sV(3, 3), eV(3, 3);
  aTrial = aConvg;
  eTrial = eTotal+De;
  sTrial = sConvg+(this->get_C())*De;
  spectralDecomposition(sTrial, s, sV);
  Vector eTrial3 = C3*s;
```

求导

E:\mycode\plastic\nemesis-master\src\material\drucker_prager_ys.cc

```
double DruckerPragerYS::get_f(const Vector& sigma, const double kappa) {
  this->set_sigma(sigma);
  double c = c0+Kc*kappa;
  double phi = phi0+Kphi*kappa;
  double rho = 2*sin(phi)/(sqrt(3.)*(3-sin(phi)));
  double k = 6*c*cos(phi)/(sqrt(3.)*(3-sin(phi)));
  return rho*I1+sqrt(J2)-k;
}

const Vector& DruckerPragerYS::get_dfds(const Vector& sigma,
                                       const double kappa) {
  this->set_sigma(sigma);
  double phi = phi0+Kphi*kappa;
  double rho = 2*sin(phi)/(sqrt(3.)*(3-sin(phi)));

  double C1 = rho;
  double C2 = 1./(2.*sqrt(J2));
  a = a1*C1+a2*C2;
  return a;
}

const Matrix& DruckerPragerYS::get_d2fdsds(const Vector& sigma,
                                          const double /*kappa*/) {
  this->set_sigma(sigma);

  double C2 = 1./(2.*sqrt(J2));
  double C22=-1./(4.*J2*sqrt(J2));
  da = C2*da2+C22*da22;
  return da;
}

```



# OOFEM

```
    // elastic constants
    double eM = LEMaterial.give(Ex, gp);
    double nu = LEMaterial.give(NYxz, gp);
    double gM = eM / ( 2. * ( 1. + nu ) );
    double kM = eM / ( 3. * ( 1. - 2. * nu ) );
    ===============================
     double volConstant = 3. * kM * alphaPsi;
    double devConstant = sqrt(2.) * gM;
    // yield value prime is derivative of yield value with respect to deltaLambda
    double yieldValuePrimeZero = -9. * alpha * alphaPsi * kM - gM;

    auto flowDir = stressDeviator * ( 1. / sqrt(2. * trialStressJTwo) );
    ================================
    deltaLambda += deltaLambdaIncrement;
        tempKappa += kFactor * deltaLambdaIncrement;
        volumetricStress -= volConstant * deltaLambdaIncrement;

        // auto plasticFlow = flowDir * (devConstant * deltaLambdaIncrement)
        //stressDeviator -= plasticFlow;

        stressDeviator += (-devConstant * deltaLambdaIncrement) * flowDir;
   ==================================
           double yieldValuePrime;
        if ( deltaKappa == 0. ) {
            yieldValuePrime = yieldValuePrimeZero
                              - sqrt(2.) / 3. / kM *computeYieldStressPrime(tempKappa, eM);
        } else {
            yieldValuePrime = yieldValuePrimeZero
                              - 2. / 9. / kM / kM *computeYieldStressPrime(tempKappa, eM)
            * deltaVolumetricStress / deltaKappa;
```

# MOOSE

```
void
TensorMechanicsPlasticDruckerPrager::initializeB(Real intnl, int fd, Real & bbb) const
{
  const Real s = (fd == friction) ? std::sin(_mc_phi.value(intnl)) : std::sin(_mc_psi.value(intnl));
  switch (_mc_interpolation_scheme)
  {
    case 0: // outer_tip
      bbb = 2.0 * s / std::sqrt(3.0) / (3.0 - s);
      break;
    case 1: // inner_tip
      bbb = 2.0 * s / std::sqrt(3.0) / (3.0 + s);
      break;
    case 2: // lode_zero
      bbb = s / 3.0;
      break;
    case 3: // inner_edge
      bbb = s / std::sqrt(9.0 + 3.0 * Utility::pow<2>(s));
      break;
    case 4: // native
      const Real c =
          (fd == friction) ? std::cos(_mc_phi.value(intnl)) : std::cos(_mc_psi.value(intnl));
      bbb = s / c;
      break;
  }
}
void
CappedDruckerPragerCosseratStressUpdate::setStressAfterReturn(const RankTwoTensor & 
                  const
{
  // symm_stress is the symmetric part of the stress tensor.
  // symm_stress = (s_ij+s_ji)/2 + de_ij tr(stress) / 3
  //             = q / q_trial * (s_ij^trial+s_ji^trial)/2 + de_ij p / 3
  //             = q / q_trial * (symm_stress_ij^trial - de_ij tr(stress^trial) / 3) + de_ij p / 3
  const Real p_trial = stress_trial.trace();
  RankTwoTensor symm_stress = RankTwoTensor(RankTwoTensor::initIdentity) / 3.0 *
                              (p_ok - (_in_q_trial == 0.0 ? 0.0 : p_trial * q_ok / _in_q_trial));
  if (_in_q_trial > 0)
    symm_stress += q_ok / _in_q_trial * 0.5 * (stress_trial + stress_trial.transpose());
  stress = symm_stress + 0.5 * (stress_trial - stress_trial.transpose());
}
```

## NTNU

```
/* Assumes von Mises plasticity and an associated flow rule.  The back stress
is given by the rate equation D/Dt(beta) = 2/3~gammadot~Hprime~df/dsigma */
void 
PragerKinematicHardening::computeBackStress(const PlasticityState* state,
                                            const double& delT,
                                            const particleIndex idx,
                                            const double& delLambda,
                                            const Matrix3& df_dsigma_normal_new,
                                            const Matrix3& backStress_old,
                                            Matrix3& backStress_new) 
{
  // Get the hardening modulus (constant for Prager kinematic hardening)
  double H_prime = d_cm.beta*d_cm.hardening_modulus;
  double stt = sqrt(2.0/3.0);

  // Compute updated backstress
  backStress_new = backStress_old + df_dsigma_normal_new*(delLambda*H_prime*stt);

  return;
}

```

# PHASE

E:\mycode\Elastic\PhaseFieldBenchmarking-master\Uintah\src\CCA\Components\MPM\Materials\ConstitutiveModel\PlasticityModels\PragerKinematicHardening.cc

```
void 
PragerKinematicHardening::eval_h_beta(const Matrix3& df_dsigma,
                                      const PlasticityState* ,
                                      Matrix3& h_beta)
{
  double H_prime = d_cm.beta*d_cm.hardening_modulus;
  h_beta = df_dsigma*(2.0/3.0*H_prime);
  return;
}
```

# OPENSEES

E:\mycode\Elastic\OpenSees-master\SRC\material\nD\UWmaterials\DruckerPrager.cpp

```
      // epsilon_n1_p_trial = ..._n1_p  = ..._n_p
        mEpsilon_n1_p = mEpsilon_n_p;

		// alpha1_n+1_trial
		mAlpha1_n1 = mAlpha1_n;
		// alpha2_n+1_trial
		mAlpha2_n1 = mAlpha2_n;

        // beta_n+1_trial
        mBeta_n1 = mBeta_n;

        // epsilon_elastic = epsilon_n+1 - epsilon_n_p
		epsilon_e = mEpsilon - mEpsilon_n1_p;

        // trial stress
		mSigma = mCe*epsilon_e;

        // deviator stress tensor: s = 2G * IIdev * epsilon_e
        //I1_trial
		Invariant_1 = ( mSigma(0) + mSigma(1) + mSigma(2) );

        // s_n+1_trial
		s = mSigma - (Invariant_1/3.0)*mI1;

        //eta_trial = s_n+1_trial - beta_n;
		eta = s - mBeta_n;
		
		// compute yield function value (contravariant norm)
        norm_eta = sqrt(eta(0)*eta(0) + eta(1)*eta(1) + eta(2)*eta(2) + 2*(eta(3)*eta(3) + eta(4)*eta(4) + eta(5)*eta(5)));

        // f1_n+1_trial
		f1 = norm_eta + mrho*Invariant_1 - root23*Kiso(mAlpha1_n1);

		// f2_n+1_trial
		f2 = Invariant_1 - T(mAlpha2_n1);
		
		// update elastic bulk and shear moduli 
 		this->updateElasticParam();

		// check trial state
		int count = 1;
		if ((f1<=fTOL) && (f2<=fTOL) || mElastFlag < 2) {

			okay = true;
			// trial state = elastic state - don't need to do any updates.
			mCep = mCe;
			count = 0;

			// set state variables for recorders
            Invariant_ep = 	mEpsilon_n1_p(0)+mEpsilon_n1_p(1)+mEpsilon_n1_p(2);

			norm_ep  = sqrt(mEpsilon_n1_p(0)*mEpsilon_n1_p(0) + mEpsilon_n1_p(1)*mEpsilon_n1_p(1) + mEpsilon_n1_p(2)*mEpsilon_n1_p(2)
                           + 0.5*(mEpsilon_n1_p(3)*mEpsilon_n1_p(3) + mEpsilon_n1_p(4)*mEpsilon_n1_p(4) + mEpsilon_n1_p(5)*mEpsilon_n1_p(5)));
			
			dev_ep = mEpsilon_n1_p - one3*Invariant_ep*mI1;

            norm_dev_ep  = sqrt(dev_ep(0)*dev_ep(0) + dev_ep(1)*dev_ep(1) + dev_ep(2)*dev_ep(2)
                           + 0.5*(dev_ep(3)*dev_ep(3) + dev_ep(4)*dev_ep(4) + dev_ep(5)*dev_ep(5)));

			mState(0) = Invariant_1;
			mState(1) = norm_eta;
       		mState(2) = Invariant_ep;
        	mState(3) = norm_dev_ep;
			mState(4) = norm_ep;
			return;
		}
```

=================

D:\图形学书籍\图形学书籍\固体物理\Computational Methods in Elasticity and Plasticity Solids and Porous Media by A. Anandarajah (auth.) (z-lib.org).pdf

![image-20211218211845242](E:\mycode\collection\定理\弹性力学\image-20211218211845242.png)
$$
\phi(\sigma_{ij},\alpha ,k) = J - \alpha I - k = 0
$$
![image-20211218212255116](E:\mycode\collection\定理\弹性力学\image-20211218212255116.png)

![image-20211218212356159](E:\mycode\collection\定理\弹性力学\image-20211218212356159.png)

![image-20211218212405831](E:\mycode\collection\定理\弹性力学\image-20211218212405831.png)

===================D:\图形学书籍\图形学书籍\固体布料数学\裂缝\Computational Methods for Plasticity Theory and Applications by EA de Souza Neto, Prof. D Periæ, Prof. DRJ Owen (z-lib.org).pdf

![image-20211218212506722](E:\mycode\collection\定理\弹性力学\image-20211218212506722.png)

D:\图形学书籍\图形学书籍\固体物理\Introduction to computational plasticity by Fionn Dunne, Nik Petrinic (z-lib.org).pdf



![image-20211218212906827](E:\mycode\collection\定理\弹性力学\image-20211218212906827.png)

==========CHRONO

```
void ChContinuumDruckerPrager::ComputeReturnMapping(ChStrainTensor<>& mplasticstrainflow,
                                                    const ChStrainTensor<>& mincrementstrain,
                                                    const ChStrainTensor<>& mlastelasticstrain,
                                                    const ChStrainTensor<>& mlastplasticstrain) const {
    ChStrainTensor<> guesselstrain(mlastelasticstrain);
    guesselstrain += mincrementstrain;  // assume increment is all elastic

    ChStressTensor<> mstress;
    this->ComputeElasticStress(mstress, guesselstrain);
    double fprager = this->ComputeYeldFunction(mstress);

    if (fprager > 0) {
        if (mstress.GetInvariant_I1() * this->alpha - sqrt(mstress.GetInvariant_J2()) * this->alpha * this->alpha -
                this->elastic_yeld >
            0) {
            // Case: tentative stress is in polar cone; a singular region where the gradient of
            // the yield function (or flow potential) is not defined. Just project to vertex.
            ChStressTensor<> vertexstress;
            double vertcoord = this->elastic_yeld / (3 * this->alpha);
```



```
//***OBSOLETE***
void ChContinuumDruckerPrager::ComputePlasticStrainFlow(ChStrainTensor<>& mplasticstrainflow,
                                                        const ChStrainTensor<>& mestrain) const {
    ChStressTensor<> mstress;
    this->ComputeElasticStress(mstress, mestrain);
    double prager = mstress.GetInvariant_I1() * this->alpha + sqrt(mstress.GetInvariant_J2());
    if (prager > this->elastic_yeld) {
        ChVoightTensor<> mdev;
        mstress.GetDeviatoricPart(mdev);
        double divisor = 2. * sqrt(mstress.GetInvariant_J2());
        if (divisor > 10e-20)
            mdev *= 1. / divisor;
        mdev.XX() += this->dilatancy;
        mdev.YY() += this->dilatancy;
        mdev.ZZ() += this->dilatancy;
        mplasticstrainflow = mdev;
    } else {
        mplasticstrainflow.setZero();
    }
}
```

