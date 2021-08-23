// helper functions for the Smagorinsky turbulence model by Nils Thuerey
// see http://www.ntoken.com and http://www.matthiasmueller.info/realtimephysics/ for further info
// this code is released under the GPL, see LICENSE.txt for details

const Real CSmago = 0.04;
Real   lesCoeffOffdiag[3][19];
Real   lesCoeffDiag[3][19];

inline Real getLesNoneqTensorCoeff(
		Real df[], 				
		Real feq[] ) {
	Real Qo = 0.0;
	for(int m=0; m< ((3*3)-3)/2 ; m++) { 
		Real qadd = 0.0;
		for(int l=1; l<19; l++) { 
			if(lesCoeffOffdiag[m][l]==0.0) continue;
			qadd += lesCoeffOffdiag[m][l]*(df[l]-feq[l]);
		}
		Qo += (qadd*qadd);
	}
	Qo *= 2.0;
	for(int m=0; m<3; m++) { 
		Real qadd = 0.0;
		for(int l=1; l<19; l++) { 
			if(lesCoeffDiag[m][l]==0.0) continue;
			qadd += lesCoeffDiag[m][l]*(df[l]-feq[l]);
		}
		Qo += (qadd*qadd);
	}
	Qo = sqrt(Qo);
	return Qo;
}

inline Real getLesOmega(Real omega, Real csmago, Real Qo) {
	const Real tau = 1.0/omega;
	const Real nu = (2.0*tau-1.0) * (1.0/6.0);
	const Real C = csmago;
	const Real Csqr = C*C;
	Real S = -nu + sqrt( nu*nu + 18.0*Csqr*Qo ) / (6.0*Csqr);
	return( 1.0/( 3.0*( nu+Csqr*S ) +0.5 ) );
}

// collision with turbulence model
// 
inline void collideSmagorinsky(Real df[], char flag) {
	int l;
	Real rho=df[0]; 
	Real u_x = 0;
	Real u_y = 0;
	Real u_z = 0;
	Real feq[19];
	Real omegaNew;
	Real Qo = 0.0;

	// just like normal collision...
	for(l=1; l<19; l++) { 
		rho += df[l]; 
		u_x  += e_x[l] * df[l]; 
		u_y  += e_y[l] * df[l];  
		u_z  += e_z[l] * df[l];  
	}  

	if (flag == VELOCITY) {
		u_x = LDC_VELOCITY;
	} 

	for(l=0; l<19; l++) { 
		const Real tmp = (e_x[l]*u_x + e_y[l]*u_y + e_z[l]*u_z);
		feq[l] = w[l] * ( rho - 
				(3.0/2.0 * (u_x*u_x + u_y*u_y + u_z*u_z)) +
				(3.0 *     tmp) +
				(9.0/2.0 * tmp*tmp ) );
	}

	// new - compute the new local viscosity
	// according to magnitude of the local Reynolds
	// stress tensor
	Qo = getLesNoneqTensorCoeff(df,feq);
	omegaNew = getLesOmega(omega,CSmago,Qo);

	// relax with new omega...
	for(l=0; l<19; l++) { 
		df[l] = (1.0-omegaNew ) * df[l] + omegaNew * feq[l]; 
	}  
};

// init helper function
inline void initSmagoArrays() {
	// init Smagorinsky model (precompute 
	// stress tensor factors)

	int odm = 0;
	for(int m=0; m< 3; m++) { 
		for(int l=0; l< 19; l++) { 
			lesCoeffDiag[m][l] = 
			lesCoeffOffdiag[m][l] = 0.0;
		}
	}

	for(int m=0; m< 3; m++) { 
		for(int n=0; n< 3; n++) { 
			for(int l=1; l< 19; l++) { 
				Real em;
				switch(m) {
					case 0: em = e_x[l]; break;
					case 1: em = e_y[l]; break;
					case 2: em = e_z[l]; break;
				}
				Real en;
				switch(n) {
					case 0: en = e_x[l]; break;
					case 1: en = e_y[l]; break;
					case 2: en = e_z[l]; break;
				}
				const Real coeff = em*en;
				if(m==n) {
					lesCoeffDiag[m][l] = coeff;
				} else {
					if(m>n) {
						lesCoeffOffdiag[odm][l] = coeff;
					}
				}
			}

			if(m==n) {
			} else {
				if(m>n) odm++;
			}
		}
	}

};



