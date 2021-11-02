  
///////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2002 - 2016, Huamin Wang
//  https://web.cse.ohio-state.edu/~wang.3602/publications.html
//  Descent Methods for Elastic Body Simulation on the GPU
//  All rights reserved.
  TYPE Compute_FM(int t, TYPE stiffness_0, TYPE stiffness_1, TYPE stiffness_2, TYPE stiffness_3, const bool has_matrix=true)
    {
        stiffness_0=-Vol[t]*stiffness_0;
        stiffness_1=-Vol[t]*stiffness_1;
        stiffness_2=-Vol[t]*stiffness_2;

        //No velocity access in this function
        int p0=Tet[t*4+0]*3;
        int p1=Tet[t*4+1]*3;
        int p2=Tet[t*4+2]*3;
        int p3=Tet[t*4+3]*3;
        
        TYPE Ds[9];
        Ds[0]=X[p1+0]-X[p0+0];
        Ds[3]=X[p1+1]-X[p0+1];
        Ds[6]=X[p1+2]-X[p0+2];
        Ds[1]=X[p2+0]-X[p0+0];
        Ds[4]=X[p2+1]-X[p0+1];
        Ds[7]=X[p2+2]-X[p0+2];
        Ds[2]=X[p3+0]-X[p0+0];
        Ds[5]=X[p3+1]-X[p0+1];
        Ds[8]=X[p3+2]-X[p0+2];
        
        TYPE F[9], U[9], Sigma[3], V[9];
        Matrix_Product_3(Ds, &inv_Dm[t*9], F);
        
        memcpy(U, F, sizeof(TYPE)*9);
        SVD3((TYPE (*)[3])U, Sigma, (TYPE (*)[3])V);
        int small_id;
        if(fabsf(Sigma[0])<fabsf(Sigma[1]) && fabsf(Sigma[0])<fabsf(Sigma[2]))	small_id=0;
        else if(fabsf(Sigma[1])<fabsf(Sigma[2]))                                small_id=1;
        else                                                                    small_id=2;
        if(U[0]*(U[4]*U[8]-U[7]*U[5])+U[3]*(U[7]*U[2]-U[1]*U[8])+U[6]*(U[1]*U[5]-U[4]*U[2])<0)
        {
            U[0+small_id]	=-U[0+small_id];
            U[3+small_id]	=-U[3+small_id];
            U[6+small_id]	=-U[6+small_id];
			Sigma[small_id]	=-Sigma[small_id];
        }
        if(V[0]*(V[4]*V[8]-V[7]*V[5])+V[3]*(V[7]*V[2]-V[1]*V[8])+V[6]*(V[1]*V[5]-V[4]*V[2])<0)
        {
            V[0+small_id]=-V[0+small_id];
            V[3+small_id]=-V[3+small_id];
            V[6+small_id]=-V[6+small_id];
			Sigma[small_id] = -Sigma[small_id];
        }
        
        //SVD3x3((TYPE (*)[3])F, (TYPE (*)[3])U, Sigma, &Q[t*4], (TYPE (*)[3])V, svd_iterations);

		float interpolator0, interpolator1, interpolator2;
		if (Sigma[0] < 1)	interpolator0 = (0.15 - Sigma[0])*10.0;
		if (Sigma[1] < 1)	interpolator1 = (0.15 - Sigma[1])*10.0;
		if (Sigma[2] < 1)	interpolator2 = (0.15 - Sigma[2])*10.0;

		float interpolator = MAX(MAX(interpolator0, interpolator1), interpolator2);
		if(interpolator<0)		interpolator=0;
		if(interpolator>1)		interpolator=1;

		//lambda[t] = MAX(lambda[t], interpolator);
		//lambda[t] = 0;
		//interpolator = lambda[t];

		interpolator = 0;
		stiffness_0 *= (1 - interpolator);
		stiffness_1 *= (1 - interpolator);

		//printf("sigma; %f, %f, %f\n", Sigma[0], Sigma[1], Sigma[2]);


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


        //printf("update: %d\n", has_matrix);
		//printf("check0: %f, %f\n", P0, Numerical_Estimate_dWdS(Sigma, 0, stiffness_0, stiffness_1, stiffness_2, stiffness_3));
		//printf("check1: %f, %f\n", P1, Numerical_Estimate_dWdS(Sigma, 1, stiffness_0, stiffness_1, stiffness_2, stiffness_3));
		//printf("check2: %f, %f\n", P2, Numerical_Estimate_dWdS(Sigma, 2, stiffness_0, stiffness_1, stiffness_2, stiffness_3));
		//printf("C: %f, %f\n", H[0][0], Numerical_Estimate_dW2dI2(I, II, III, 0, 0, stiffness_0, stiffness_1, stiffness_2, stiffness_3));
		//printf("C: %f, %f\n", H[0][2], Numerical_Estimate_dW2dI2(I, II, III, 0, 2, stiffness_0, stiffness_1, stiffness_2, stiffness_3));
		//printf("C: %f, %f\n", H[1][2], Numerical_Estimate_dW2dI2(I, II, III, 1, 2, stiffness_0, stiffness_1, stiffness_2, stiffness_3));
		//printf("C: %f, %f\n", H[2][2], Numerical_Estimate_dW2dI2(I, II, III, 2, 2, stiffness_0, stiffness_1, stiffness_2, stiffness_3));
		//getchar();


        TYPE PV_transpose[9], P[9], force[9];
        PV_transpose[0]=P0*V[0];
        PV_transpose[1]=P0*V[3];
        PV_transpose[2]=P0*V[6];
        PV_transpose[3]=P1*V[1];
        PV_transpose[4]=P1*V[4];
        PV_transpose[5]=P1*V[7];
        PV_transpose[6]=P2*V[2];
        PV_transpose[7]=P2*V[5];
        PV_transpose[8]=P2*V[8];
        Matrix_Product_3(U, PV_transpose, P);
        Matrix_Product_T_3(P, &inv_Dm[t*9], force);
        
        F_Temp[t*12+ 0]=-(force[0]+force[1]+force[2]);
        F_Temp[t*12+ 1]=-(force[3]+force[4]+force[5]);
        F_Temp[t*12+ 2]=-(force[6]+force[7]+force[8]);
        F_Temp[t*12+ 3]=force[0];
        F_Temp[t*12+ 4]=force[3];
        F_Temp[t*12+ 5]=force[6];
        F_Temp[t*12+ 6]=force[1];
        F_Temp[t*12+ 7]=force[4];
        F_Temp[t*12+ 8]=force[7];
        F_Temp[t*12+ 9]=force[2];
        F_Temp[t*12+10]=force[5];
        F_Temp[t*12+11]=force[8];
        

		//Force[p0+0]+=-(force[0]+force[1]+force[2]);
        //Force[p0+1]+=-(force[3]+force[4]+force[5]);
        //Force[p0+2]+=-(force[6]+force[7]+force[8]);
        //Force[p1+0]+=force[0];
        //Force[p1+1]+=force[3];
        //Force[p1+2]+=force[6];
        //Force[p2+0]+=force[1];
        //Force[p2+1]+=force[4];
        //Force[p2+2]+=force[7];
        //Force[p3+0]+=force[2];
        //Force[p3+1]+=force[5];
        //Force[p3+2]+=force[8];
        
        //Copy UV into TU and TV
        memcpy(&TU[t*9], U, sizeof(TYPE)*9);
        memcpy(&TV[t*9], V, sizeof(TYPE)*9);
        

		//Strain limiting part
		TYPE new_R[9];
		Matrix_Product_T_3(U, V, new_R);
		const TYPE* idm = &inv_Dm[t * 9];
		TYPE half_matrix[3][4], result_matrix[3][4];
		half_matrix[0][0] = -idm[0] - idm[3] - idm[6];
		half_matrix[0][1] =  idm[0];
		half_matrix[0][2] =  idm[3];
		half_matrix[0][3] =  idm[6];
		half_matrix[1][0] = -idm[1] - idm[4] - idm[7];
		half_matrix[1][1] =  idm[1];
		half_matrix[1][2] =  idm[4];
		half_matrix[1][3] =  idm[7];
		half_matrix[2][0] = -idm[2] - idm[5] - idm[8];
		half_matrix[2][1] =  idm[2];
		half_matrix[2][2] =  idm[5];
		half_matrix[2][3] =  idm[8];

		Matrix_Substract_3(new_R, F, new_R);
		Matrix_Product(new_R, &half_matrix[0][0], &result_matrix[0][0], 3, 3, 4);

		TYPE pd_stiffness = Vol[t] * stiffness_p * interpolator;
		//Force[p0+0] += result_matrix[0][0] * pd_stiffness;
		//Force[p0+1] += result_matrix[1][0] * pd_stiffness;
		//Force[p0+2] += result_matrix[2][0] * pd_stiffness;
		//Force[p1+0] += result_matrix[0][1] * pd_stiffness;
		//Force[p1+1] += result_matrix[1][1] * pd_stiffness;
		//Force[p1+2] += result_matrix[2][1] * pd_stiffness;
		//Force[p2+0] += result_matrix[0][2] * pd_stiffness;
		//Force[p2+1] += result_matrix[1][2] * pd_stiffness;
		//Force[p2+2] += result_matrix[2][2] * pd_stiffness;
		//Force[p3+0] += result_matrix[0][3] * pd_stiffness;
		//Force[p3+1] += result_matrix[1][3] * pd_stiffness;
		//Force[p3+2] += result_matrix[2][3] * pd_stiffness;
	
		//PD energy
		TYPE pd_energy=0;
		pd_energy+=new_R[0]*new_R[0];
		pd_energy+=new_R[1]*new_R[1];
		pd_energy+=new_R[2]*new_R[2];
		pd_energy+=new_R[3]*new_R[3];
		pd_energy+=new_R[4]*new_R[4];
		pd_energy+=new_R[5]*new_R[5];
		pd_energy+=new_R[6]*new_R[6];
		pd_energy+=new_R[7]*new_R[7];
		pd_energy+=new_R[8]*new_R[8];
		energy+=pd_energy*pd_stiffness*0.5;



		TYPE value0 = half_matrix[0][0]*half_matrix[0][0]+half_matrix[1][0]*half_matrix[1][0]+half_matrix[2][0]*half_matrix[2][0];
		TYPE value1 = half_matrix[0][1]*half_matrix[0][1]+half_matrix[1][1]*half_matrix[1][1]+half_matrix[2][1]*half_matrix[2][1];
		TYPE value2 = half_matrix[0][2]*half_matrix[0][2]+half_matrix[1][2]*half_matrix[1][2]+half_matrix[2][2]*half_matrix[2][2];
		TYPE value3 = half_matrix[0][3]*half_matrix[0][3]+half_matrix[1][3]*half_matrix[1][3]+half_matrix[2][3]*half_matrix[2][3];
		ext_C[Tet[t * 4 + 0]]+=value0*pd_stiffness;
		ext_C[Tet[t * 4 + 1]]+=value1*pd_stiffness;
		ext_C[Tet[t * 4 + 2]]+=value2*pd_stiffness;
		ext_C[Tet[t * 4 + 3]]+=value3*pd_stiffness;



        if(has_matrix==false)	return energy;

        
		//Now compute the stiffness matrix
        TYPE alpha[3][3], beta[3][3], gamma[3][3];
        for(int i=0; i<3; i++)
        for(int j=i; j<3; j++)
        {
            alpha[i][j]=2*dEdI+4*(Sigma[i]*Sigma[i]+Sigma[j]*Sigma[j])*dEdII;
            beta[i][j]=4*Sigma[i]*Sigma[j]*dEdII-2*III*dEdIII/(Sigma[i]*Sigma[j]);
                
            TYPE vi[3]={2*Sigma[i], 4*Sigma[i]*Sigma[i]*Sigma[i], 2*III/Sigma[i]};
            TYPE vj[3]={2*Sigma[j], 4*Sigma[j]*Sigma[j]*Sigma[j], 2*III/Sigma[j]};
            TYPE r[3];
            Matrix_Vector_Product_3(&H[0][0], vj, r);
            gamma[i][j]=DOT(vi, r)+4*III*dEdIII/(Sigma[i]*Sigma[j]);
        }
        
        // Save alpha, beta, and gamma
        memcpy(&T_alpha[t*9], &alpha[0][0], sizeof(TYPE)*9);
        memcpy(&T_beta [t*9], &beta [0][0], sizeof(TYPE)*9);
        memcpy(&T_gamma[t*9], &gamma[0][0], sizeof(TYPE)*9);
        
        T_A[t*9+0]=alpha[0][0]+beta[0][0]+gamma[0][0];
        T_A[t*9+1]=gamma[0][1];
        T_A[t*9+2]=gamma[0][2];
        T_A[t*9+3]=gamma[0][1];
        T_A[t*9+4]=alpha[1][1]+beta[1][1]+gamma[1][1];
        T_A[t*9+5]=gamma[1][2];
        T_A[t*9+6]=gamma[0][2];
        T_A[t*9+7]=gamma[1][2];
        T_A[t*9+8]=alpha[2][2]+beta[2][2]+gamma[2][2];
        
        T_B01[t*4+0]=alpha[0][1];
        T_B01[t*4+1]= beta[0][1];
        T_B01[t*4+2]= beta[0][1];
        T_B01[t*4+3]=alpha[0][1];        
        T_B02[t*4+0]=alpha[0][2];
        T_B02[t*4+1]= beta[0][2];
        T_B02[t*4+2]= beta[0][2];
        T_B02[t*4+3]=alpha[0][2];        
        T_B12[t*4+0]=alpha[1][2];
        T_B12[t*4+1]= beta[1][2];
        T_B12[t*4+2]= beta[1][2];
        T_B12[t*4+3]=alpha[1][2];      

		//Fix...
		//printf("t: %d; %f, %f, %f\n", t, Sigma[0], Sigma[1], Sigma[2]);		
		//eigen_project(&T_A[t*9+0], 3);
		//eigen_project(&T_B01[t*4+0], 2);
		//eigen_project(&T_B02[t*4+0], 2);
		//eigen_project(&T_B12[t*4+0], 2);
		//Fix...



        TYPE dGdF[12][9];    //G is related to force, according to [TSIF05], (g0, g3, g6), (g1, g4, g7), (g2, g5, g8), (g9, g10, g11)
        for(int i=0; i<9; i++)
        {
            TYPE dF[9], temp0[9], temp1[9];
            memset(&dF, 0, sizeof(TYPE)*9);
            dF[i]=1;
            
            Matrix_Product_3(dF, V, temp0);
            Matrix_T_Product_3(U, temp0, temp1);
            
            temp0[0]=T_A[t*9+0]*temp1[0]+T_A[t*9+1]*temp1[4]+T_A[t*9+2]*temp1[8];
            temp0[4]=T_A[t*9+3]*temp1[0]+T_A[t*9+4]*temp1[4]+T_A[t*9+5]*temp1[8];
            temp0[8]=T_A[t*9+6]*temp1[0]+T_A[t*9+7]*temp1[4]+T_A[t*9+8]*temp1[8];

            temp0[1]=T_B01[t*4+0]*temp1[1]+T_B01[t*4+1]*temp1[3];
            temp0[3]=T_B01[t*4+2]*temp1[1]+T_B01[t*4+3]*temp1[3];
            
            temp0[2]=T_B02[t*4+0]*temp1[2]+T_B02[t*4+1]*temp1[6];
            temp0[6]=T_B02[t*4+2]*temp1[2]+T_B02[t*4+3]*temp1[6];
            
            temp0[5]=T_B12[t*4+0]*temp1[5]+T_B12[t*4+1]*temp1[7];
            temp0[7]=T_B12[t*4+2]*temp1[5]+T_B12[t*4+3]*temp1[7];
            
            Matrix_Product_T_3(temp0, V, temp1);
            Matrix_Product_3(U, temp1, temp0);
            Matrix_Product_T_3(temp0, &inv_Dm[t*9], &dGdF[i][0]);
        }
        
        //for(int i=0; i<9; i++)
        //    printf("dPdF %d: %f, %f, %f; %f, %f, %f; %f, %f, %f\n", i,
        //           dPdF[t*81+i*9+0], dPdF[t*81+i*9+1], dPdF[t*81+i*9+2], dPdF[t*81+i*9+3], dPdF[t*81+i*9+4], dPdF[t*81+i*9+5], dPdF[t*81+i*9+6], dPdF[t*81+i*9+7], dPdF[t*81+i*9+8]);
        
        //Transpose dGdF
        TYPE temp;
        for(int i=0; i<9; i++) for(int j=i+1; j<9; j++)
            SWAP(dGdF[i][j], dGdF[j][i]);
        
        for(int j=0; j< 9; j++)
        {
            dGdF[ 9][j]=-dGdF[0][j]-dGdF[1][j]-dGdF[2][j];
            dGdF[10][j]=-dGdF[3][j]-dGdF[4][j]-dGdF[5][j];
            dGdF[11][j]=-dGdF[6][j]-dGdF[7][j]-dGdF[8][j];
        }
        
        TYPE new_idm[4][3];
        new_idm[0][0]=-inv_Dm[t*9+0]-inv_Dm[t*9+3]-inv_Dm[t*9+6];
        new_idm[0][1]=-inv_Dm[t*9+1]-inv_Dm[t*9+4]-inv_Dm[t*9+7];
        new_idm[0][2]=-inv_Dm[t*9+2]-inv_Dm[t*9+5]-inv_Dm[t*9+8];
        new_idm[1][0]= inv_Dm[t*9+0];
        new_idm[1][1]= inv_Dm[t*9+1];
        new_idm[1][2]= inv_Dm[t*9+2];
        new_idm[2][0]= inv_Dm[t*9+3];
        new_idm[2][1]= inv_Dm[t*9+4];
        new_idm[2][2]= inv_Dm[t*9+5];
        new_idm[3][0]= inv_Dm[t*9+6];
        new_idm[3][1]= inv_Dm[t*9+7];
        new_idm[3][2]= inv_Dm[t*9+8];
        
        C_Temp[t*12+ 0]=-Matrix_Product_T(&new_idm[0][0], &dGdF[ 9][0], 4, 3, 3, 0, 0);
        C_Temp[t*12+ 1]=-Matrix_Product_T(&new_idm[0][0], &dGdF[10][0], 4, 3, 3, 0, 1);
        C_Temp[t*12+ 2]=-Matrix_Product_T(&new_idm[0][0], &dGdF[11][0], 4, 3, 3, 0, 2);
        C_Temp[t*12+ 3]=-Matrix_Product_T(&new_idm[0][0], &dGdF[ 0][0], 4, 3, 3, 1, 0);
        C_Temp[t*12+ 4]=-Matrix_Product_T(&new_idm[0][0], &dGdF[ 3][0], 4, 3, 3, 1, 1);
        C_Temp[t*12+ 5]=-Matrix_Product_T(&new_idm[0][0], &dGdF[ 6][0], 4, 3, 3, 1, 2);
        C_Temp[t*12+ 6]=-Matrix_Product_T(&new_idm[0][0], &dGdF[ 1][0], 4, 3, 3, 2, 0);
        C_Temp[t*12+ 7]=-Matrix_Product_T(&new_idm[0][0], &dGdF[ 4][0], 4, 3, 3, 2, 1);
        C_Temp[t*12+ 8]=-Matrix_Product_T(&new_idm[0][0], &dGdF[ 7][0], 4, 3, 3, 2, 2);
        C_Temp[t*12+ 9]=-Matrix_Product_T(&new_idm[0][0], &dGdF[ 2][0], 4, 3, 3, 3, 0);
        C_Temp[t*12+10]=-Matrix_Product_T(&new_idm[0][0], &dGdF[ 5][0], 4, 3, 3, 3, 1);
        C_Temp[t*12+11]=-Matrix_Product_T(&new_idm[0][0], &dGdF[ 8][0], 4, 3, 3, 3, 2);
                
        //Get K matrix per tetrahedron
        /*Matrix_Product_T(&new_idm[0][0], &dGdF[ 0][0], &TK[t*144+ 3*12], 4, 3, 3);
        Matrix_Product_T(&new_idm[0][0], &dGdF[ 1][0], &TK[t*144+ 6*12], 4, 3, 3);
        Matrix_Product_T(&new_idm[0][0], &dGdF[ 2][0], &TK[t*144+ 9*12], 4, 3, 3);
        Matrix_Product_T(&new_idm[0][0], &dGdF[ 3][0], &TK[t*144+ 4*12], 4, 3, 3);
        Matrix_Product_T(&new_idm[0][0], &dGdF[ 4][0], &TK[t*144+ 7*12], 4, 3, 3);
        Matrix_Product_T(&new_idm[0][0], &dGdF[ 5][0], &TK[t*144+10*12], 4, 3, 3);
        Matrix_Product_T(&new_idm[0][0], &dGdF[ 6][0], &TK[t*144+ 5*12], 4, 3, 3);
        Matrix_Product_T(&new_idm[0][0], &dGdF[ 7][0], &TK[t*144+ 8*12], 4, 3, 3);
        Matrix_Product_T(&new_idm[0][0], &dGdF[ 8][0], &TK[t*144+11*12], 4, 3, 3);
        Matrix_Product_T(&new_idm[0][0], &dGdF[ 9][0], &TK[t*144+ 0*12], 4, 3, 3);
        Matrix_Product_T(&new_idm[0][0], &dGdF[10][0], &TK[t*144+ 1*12], 4, 3, 3);
        Matrix_Product_T(&new_idm[0][0], &dGdF[11][0], &TK[t*144+ 2*12], 4, 3, 3);
        
       
        TYPE G[9];
        TYPE x0[3]={1, 0, 0};
        TYPE x1[3]={0, 0, 0};
        TYPE x2[3]={0, 0, 0};
        TYPE x3[3]={0, 0, 0};
        Apply_K(t, x0, x1, x2, x3, G);
        printf("G: %f; %f\n", G[0], C_Temp[t*12+0]);
        
        printf("Center: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n",
               C_Temp[t*12+ 0], C_Temp[t*12+ 1], C_Temp[t*12+ 2],
               C_Temp[t*12+ 3], C_Temp[t*12+ 4], C_Temp[t*12+ 5],
               C_Temp[t*12+ 6], C_Temp[t*12+ 7], C_Temp[t*12+ 8],
               C_Temp[t*12+ 9], C_Temp[t*12+10], C_Temp[t*12+11]);
		*/

		// Evaluate TK
		//printf("value: %f, %f\n", Numerical_Estimate_K(t, 5, 6), TK[t*144+5*12+6]);
		return energy;
    }