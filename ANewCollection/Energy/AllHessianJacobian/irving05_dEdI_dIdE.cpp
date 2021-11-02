///////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2002 - 2016, Huamin Wang
//  https://web.cse.ohio-state.edu/~wang.3602/publications.html
//  Descent Methods for Elastic Body Simulation on the GPU
//  All rights reserved.
//  Robust Quasistatic Finite Elements and Flesh Simulation
TYPE I			= Sigma[0]*Sigma[0]+Sigma[1]*Sigma[1]+Sigma[2]*Sigma[2];
        TYPE J			= Sigma[0]*Sigma[1]*Sigma[2];
        TYPE II			= Sigma[0]*Sigma[0]*Sigma[0]*Sigma[0]+Sigma[1]*Sigma[1]*Sigma[1]*Sigma[1]+Sigma[2]*Sigma[2]*Sigma[2]*Sigma[2];
		TYPE III		= J*J;
		TYPE rcbrt_III	= 1.0/cbrt(III);
		TYPE factor_1	= ONE_THIRD*rcbrt_III/III;
		TYPE factor_2	= ONE_THIRD* factor_1/III;
		TYPE dEdI		= 0;
		TYPE dEdII		= 0;
		TYPE dEdIII		= 0;
		TYPE H[3][3];
		memset(&H[0][0], 0, sizeof(TYPE) * 9);
		TYPE energy		= 0;

		
		// StVK
	//	dEdI	+= stiffness_0*(I-3)*0.25-stiffness_1*0.5;
	//	dEdII	+= stiffness_1*0.25;
	//	dEdIII	+= 0;
	//	H[0][0]	+= stiffness_0*0.25;
	//	energy-=stiffness_0*(I-3)*(I-3)*0.125+stiffness_1*(II-2*I+3)*0.25;

        
		//Neo-Hookean
    	dEdI	+= rcbrt_III*stiffness_0;
    	dEdII	+= 0;
    	dEdIII	+= -factor_1*stiffness_0*I;
    	H[0][2]	+= -factor_1*stiffness_0;
    	H[2][2]	+=  factor_2*stiffness_0*I*4;
     
		energy=-(stiffness_0*(I *rcbrt_III-3) + stiffness_1*(J-1)*(J-1));


		// Mooney-Rivlin		
	//	TYPE two_term_a	= stiffness_2*rcbrt_III*I;
	//	TYPE two_term_b	= stiffness_2*rcbrt_III*(I*I-II);
	//	dEdI	+= rcbrt_III*two_term_a;
	//	dEdII	+= -0.5*stiffness_2*rcbrt_III*rcbrt_III;
	//	dEdIII	+= -factor_1*two_term_b;
	//	H[0][0]	+= stiffness_2*rcbrt_III*rcbrt_III;
	//	H[0][2]	+= -factor_1*two_term_a*2;
	//	H[1][2]	+= factor_1*stiffness_2*rcbrt_III;
	//	H[2][2]	+= factor_2*two_term_b*5;
	//	energy-=stiffness_2*(0.5*rcbrt_III*rcbrt_III*(I*I-II)-3);
		
		// Fung
	//	TYPE exp_term	= expf(stiffness_3*(rcbrt_III*I-3))*stiffness_2*stiffness_3;
	//	dEdI	+= exp_term*rcbrt_III;
	//	dEdIII	+= -factor_1*I*exp_term;
	//	H[0][0]	+= rcbrt_III*stiffness_3*rcbrt_III*exp_term;
	//	H[0][2]	+= -factor_1*exp_term*(1+stiffness_3*rcbrt_III*I);
	//	H[2][2]	+= factor_2*I*exp_term* (4 + stiffness_3*I*rcbrt_III);
	//	energy-=stiffness_2*(exp(stiffness_3*(rcbrt_III*I-3))-1);

		// Volume correction
		dEdIII	+=stiffness_1*(J-1)/J;
		H[2][2]	+=stiffness_1/(2*III*J);

		// Make H symmetric
		H[1][0]	= H[0][1];
		H[2][0]	= H[0][2];
		H[2][1]	= H[1][2];

		TYPE P0 = 2 * (dEdI*Sigma[0] + 2 * dEdII*Sigma[0] * Sigma[0] * Sigma[0] + dEdIII*III / Sigma[0]);
		TYPE P1 = 2 * (dEdI*Sigma[1] + 2 * dEdII*Sigma[1] * Sigma[1] * Sigma[1] + dEdIII*III / Sigma[1]);
		TYPE P2 = 2 * (dEdI*Sigma[2] + 2 * dEdII*Sigma[2] * Sigma[2] * Sigma[2] + dEdIII*III / Sigma[2]);