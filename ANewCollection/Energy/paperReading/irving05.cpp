///////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2002 - 2016, Huamin Wang
//  https://web.cse.ohio-state.edu/~wang.3602/publications.html
//  Descent Methods for Elastic Body Simulation on the GPU
//  All rights reserved.
//  Robust Quasistatic Finite Elements and Flesh Simulation
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