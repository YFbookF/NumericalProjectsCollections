///////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2002 - 2016, Huamin Wang
//  https://web.cse.ohio-state.edu/~wang.3602/publications.html
//  Descent Methods for Elastic Body Simulation on the GPU
//  All rights reserved.
//  Robust Quasistatic Finite Elements and Flesh Simulation
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