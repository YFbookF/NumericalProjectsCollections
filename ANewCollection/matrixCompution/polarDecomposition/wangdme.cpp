//  Copyright (C) 2002 - 2016, Huamin Wang
//  https://web.cse.ohio-state.edu/~wang.3602/publications.html
//  Descent Methods for Elastic Body Simulation on the GPU
///////////////////////////////////////////////////////////////////////////////////////////
//  Closed-form polar decomposition
///////////////////////////////////////////////////////////////////////////////////////////
template <class TYPE>
void Polar_Decomposition(TYPE F[3][3], TYPE R[3][3])
{
    TYPE C[3][3];
    memset(&C[0][0], 0, sizeof(TYPE)*9);
    for(int i=0; i<3; i++)
    for(int j=0; j<3; j++)
    for(int k=0; k<3; k++)
        C[i][j]+=F[k][i]*F[k][j];
    
    TYPE C2[3][3];
    memset(&C2[0][0], 0, sizeof(TYPE)*9);
    for(int i=0; i<3; i++)
    for(int j=0; j<3; j++)
    for(int k=0; k<3; k++)
        C2[i][j]+=C[i][k]*C[j][k];

    
    TYPE det    =   F[0][0]*F[1][1]*F[2][2]+
                    F[0][1]*F[1][2]*F[2][0]+
                    F[1][0]*F[2][1]*F[0][2]-
                    F[0][2]*F[1][1]*F[2][0]-
                    F[0][1]*F[1][0]*F[2][2]-
                    F[0][0]*F[1][2]*F[2][1];
    
    TYPE I_c    =   C[0][0]+C[1][1]+C[2][2];
    TYPE I_c2   =   I_c*I_c;
    TYPE II_c   =   0.5*(I_c2-C2[0][0]-C2[1][1]-C2[2][2]);
    TYPE III_c  =   det*det;
    TYPE k      =   I_c2-3*II_c;
    
    TYPE inv_U[3][3];
    if(k<1e-10f)
    {
        TYPE inv_lambda=1/sqrt(I_c/3);
        memset(inv_U, 0, sizeof(TYPE)*9);
        inv_U[0][0]=inv_lambda;
        inv_U[1][1]=inv_lambda;
        inv_U[2][2]=inv_lambda;
    }
    else
    {
        TYPE l = I_c*(I_c*I_c-4.5*II_c)+13.5*III_c;
        TYPE k_root = sqrt(k);
        TYPE value=l/(k*k_root);
        if(value<-1.0) value=-1.0;
        if(value> 1.0) value= 1.0;
        TYPE phi = acos(value);
        TYPE lambda2=(I_c+2*k_root*cos(phi/3))/3.0;
        TYPE lambda=sqrt(lambda2);
        
        TYPE III_u = sqrt(III_c);
        if(det<0)   III_u=-III_u;
        TYPE I_u = lambda + sqrt(-lambda2 + I_c + 2*III_u/lambda);
        TYPE II_u=(I_u*I_u-I_c)*0.5;

        
        TYPE U[3][3];
        TYPE inv_rate, factor;
        
        inv_rate=1/(I_u*II_u-III_u);
        factor=I_u*III_u*inv_rate;
        
        memset(U, 0, sizeof(TYPE)*9);
        U[0][0]=factor;
        U[1][1]=factor;
        U[2][2]=factor;
        
        factor=(I_u*I_u-II_u)*inv_rate;
        for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            U[i][j]+=factor*C[i][j]-inv_rate*C2[i][j];
        
        inv_rate=1/III_u;
        factor=II_u*inv_rate;
        memset(inv_U, 0, sizeof(TYPE)*9);
        inv_U[0][0]=factor;
        inv_U[1][1]=factor;
        inv_U[2][2]=factor;
        
        factor=-I_u*inv_rate;
        for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            inv_U[i][j]+=factor*U[i][j]+inv_rate*C[i][j];
    }
    
    memset(&R[0][0], 0, sizeof(TYPE)*9);
    for(int i=0; i<3; i++)
    for(int j=0; j<3; j++)
    for(int k=0; k<3; k++)
        R[i][j]+=F[i][k]*inv_U[k][j];
    
}