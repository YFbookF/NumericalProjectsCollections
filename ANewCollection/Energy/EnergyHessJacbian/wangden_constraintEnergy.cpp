///////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2002 - 2016, Huamin Wang
//  https://web.cse.ohio-state.edu/~wang.3602/publications.html
//  Descent Methods for Elastic Body Simulation on the GPU
//  All rights reserved.
///////////////////////////////////////////////////////////////////////////////////////////
//  Constraint functions
///////////////////////////////////////////////////////////////////////////////////////////
    TYPE Energy(TYPE I, TYPE II, TYPE III, TYPE stiffness_0, TYPE stiffness_1, TYPE stiffness_2, TYPE stiffness_3)
	{
		TYPE J		= sqrt(III);
		TYPE rcbrtJ = 1.0 / cbrt(J);
		TYPE bar_I	= I *rcbrtJ*rcbrtJ;
		TYPE bar_II = (I*I-II)*rcbrtJ*rcbrtJ*rcbrtJ*rcbrtJ*0.5;
		//return stiffness_0*(bar_I - 3) + stiffness_1*(bar_II - 3);

		//printf("I: %f, %f, %f (%f)\n", I, II, III, expf(stiffness_3*(bar_I-3)));

		return stiffness_0*(bar_I-3) + stiffness_2*(expf(stiffness_3*(bar_I-3))-1);
	}

	TYPE Energy(TYPE Sigma[3], TYPE stiffness_0, TYPE stiffness_1, TYPE stiffness_2, TYPE stiffness_3)
	{
		TYPE J = Sigma[0] * Sigma[1] * Sigma[2];
		TYPE I = Sigma[0] * Sigma[0] + Sigma[1] * Sigma[1] + Sigma[2] * Sigma[2];
		TYPE II = Sigma[0] * Sigma[0] * Sigma[0] * Sigma[0] + Sigma[1] * Sigma[1] * Sigma[1] * Sigma[1] + Sigma[2] * Sigma[2] * Sigma[2] * Sigma[2];
		TYPE III = J*J;
		return Energy(I, II, III, stiffness_0, stiffness_1, stiffness_2, stiffness_3);
	}

	TYPE Numerical_Estimate_dWdI(TYPE I, TYPE II, TYPE III, int id, TYPE stiffness_0, TYPE stiffness_1, TYPE stiffness_2, TYPE stiffness_3)
	{
		TYPE epsilon = 0.01;

		if(id==0)	I  -= epsilon*0.5;
		if(id==1)	II -= epsilon*0.5;
		if(id==2)	III-= epsilon*0.5;
		TYPE value0 = Energy(I, II, III, stiffness_0, stiffness_1, stiffness_2, stiffness_3);

		if(id==0)	I  += epsilon;
		if(id==1)	II += epsilon;
		if(id==2)	III+= epsilon;
		TYPE value1 = Energy(I, II, III, stiffness_0, stiffness_1, stiffness_2, stiffness_3);

		return (value1-value0)/epsilon;
	}

	TYPE Numerical_Estimate_dW2dI2(TYPE I, TYPE II, TYPE III, int id0, int id1, TYPE stiffness_0, TYPE stiffness_1, TYPE stiffness_2, TYPE stiffness_3)
	{
		TYPE epsilon = 0.01;
		if (id0 == 0)	I   -= epsilon*0.5;
		if (id0 == 1)	II  -= epsilon*0.5;
		if (id0 == 2)	III -= epsilon*0.5;
		TYPE value0 = Numerical_Estimate_dWdI(I, II, III, id1, stiffness_0, stiffness_1, stiffness_2, stiffness_3);

		if (id0 == 0)	I   += epsilon;
		if (id0 == 1)	II  += epsilon;
		if (id0 == 2)	III += epsilon;
		TYPE value1 = Numerical_Estimate_dWdI(I, II, III, id1, stiffness_0, stiffness_1, stiffness_2, stiffness_3);

		//printf("value: %f, %f\n", value0, value1);

		return (value1 - value0) / epsilon;
	}

	TYPE Numerical_Estimate_dWdS(TYPE Sigma[3], int id, TYPE stiffness_0, TYPE stiffness_1, TYPE stiffness_2, TYPE stiffness_3)
	{
		TYPE epsilon = 0.00001;

		Sigma[id] -= epsilon*0.5;
		TYPE value0 = Energy(Sigma, stiffness_0, stiffness_1, stiffness_2, stiffness_3);
		Sigma[id] +=epsilon;
		TYPE value1 = Energy(Sigma, stiffness_0, stiffness_1, stiffness_2, stiffness_3);

		return (value1-value0)/epsilon;
	}

    TYPE Numerical_Estimate_K(int t, int i, int j)
    {
        TYPE epsilon = 0.0001;
        int pi      = Tet[t*4+i/3]*3+i%3;
        TYPE old_Xi = X[pi];
        X[pi]+=epsilon*0.5;
        Compute_FM(t, stiffness_0, stiffness_1, stiffness_2, stiffness_3, false);
        TYPE value0=F_Temp[t*12+j];
        X[pi]-=epsilon;
        Compute_FM(t, stiffness_0, stiffness_1, stiffness_2, stiffness_3, false);
        TYPE value1=F_Temp[t*12+j];
        X[pi]=old_Xi;
        return (value0-value1)/epsilon;
    }