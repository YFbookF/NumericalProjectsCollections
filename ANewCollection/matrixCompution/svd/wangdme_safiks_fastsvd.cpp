//  Copyright (C) 2002 - 2016, Huamin Wang
//  https://web.cse.ohio-state.edu/~wang.3602/publications.html
//  Descent Methods for Elastic Body Simulation on the GPU
///////////////////////////////////////////////////////////////////////////////////////////
//  SVD3x3 Fast SVD of a 3x3 matrix, based on Sifakis's paper
///////////////////////////////////////////////////////////////////////////////////////////
#define GAMMA   5.82842712475
#define C_STAR  0.92387953251
#define S_STAR  0.38268343236

template <class TYPE>
void Quaternion_to_Rotation(TYPE q[4], TYPE R[3][3])
{
    TYPE q00=q[0]*q[0];
    TYPE q11=q[1]*q[1];
    TYPE q22=q[2]*q[2];
    TYPE q33=q[3]*q[3];
    TYPE q01=q[0]*q[1];
    TYPE q02=q[0]*q[2];
    TYPE q03=q[0]*q[3];
    TYPE q12=q[1]*q[2];
    TYPE q13=q[1]*q[3];
    TYPE q23=q[2]*q[3];
    R[0][0]=q33+q00-q11-q22;
    R[1][1]=q33-q00+q11-q22;
    R[2][2]=q33-q00-q11+q22;
    R[0][1]=2*(q01-q23);
    R[1][0]=2*(q01+q23);
    R[0][2]=2*(q02+q13);
    R[2][0]=2*(q02-q13);
    R[1][2]=2*(q12-q03);
    R[2][1]=2*(q12+q03);
}


template <class TYPE>
void SVD3x3(TYPE A[3][3], TYPE U[3][3], TYPE S[3], TYPE q[4], TYPE V[3][3], int iterations=8)
{
    //sifakis's paper
    //Part 1: Compute q for V
    TYPE B[3][3];
    Quaternion_to_Rotation(q, V);
    Matrix_Product_3(&A[0][0], &V[0][0], &U[0][0]);
    Matrix_T_Product_3(&U[0][0], &U[0][0], &B[0][0]);
    
    TYPE c_h, s_h, omega;
    TYPE c, s, n;
    bool b;
    TYPE t0, t1, t2, t3, t4, t5;
    TYPE cc, cs, ss, cn, sn;
    TYPE temp_q[4];
    for(int l=0; l<iterations; l++)
    {
        //if(Max(fabs(B[0][1]), fabs(B[0][2]), fabs(B[1][2]))<1e-10f) break;
        
        int i, j, k;
        i=fabs(B[0][1])>fabs(B[0][2])?0:2;
        i=(fabs(B[0][1])>fabs(B[0][2])?fabs(B[0][1]):fabs(B[0][2]))>fabs(B[1][2])?i:1;
        j=(i+1)%3;
        k=3-i-j;
        
        c_h=2*(B[i][i]-B[j][j]);
        s_h=B[i][j];
        b=GAMMA*s_h*s_h<c_h*c_h;
        
        //printf("first c_h: %f; s_h: %f (%f)\n", c_h, s_h, fabs(c_h)+fabs(s_h));
        omega=1.0/(fabs(c_h)+fabs(s_h));
        
        c_h=b?omega*c_h:C_STAR;
        s_h=b?omega*s_h:S_STAR;
        //printf("c_h: %f; s_h: %f (%f)\n", c_h, s_h, omega);
        
        t0=c_h*c_h;
        t1=s_h*s_h;
        n=t0+t1;
        c=t0-t1;
        s=2*c_h*s_h;
        
        //Q[0][0]=c;
        //Q[1][1]=c;
        //Q[2][2]=n;
        //Q[0][1]=-s;
        //Q[1][0]=s;
        //printf("CSN: %f, %f, %f\n", c, s, n);
        //printf("N: %f\n", n);
        //printf("inv: %f\n", inv_length);
        cc=c*c;
        cs=c*s;
        ss=s*s;
        cn=c*n;
        sn=s*n;
        t0=cc*B[i][i]+2*cs*B[i][j]+ss*B[j][j];
        t1=ss*B[i][i]-2*cs*B[i][j]+cc*B[j][j];
        t2=cs*(B[j][j]-B[i][i])+(cc-ss)*B[i][j];
        t3= cn*B[i][k]+sn*B[j][k];
        t4=-sn*B[i][k]+cn*B[j][k];
        t5=B[k][k]*n*n;
        
        B[i][i]=t0;
        B[j][j]=t1;
        B[i][j]=B[j][i]=t2;
        B[i][k]=B[k][i]=t3;
        B[j][k]=B[k][j]=t4;
        B[k][k]=t5;
        
        //Update q
        temp_q[i]=c_h*q[i]+q[j]*s_h;
        temp_q[j]=c_h*q[j]-q[i]*s_h;
        temp_q[k]=c_h*q[k]+q[3]*s_h;
        temp_q[3]=c_h*q[3]-q[k]*s_h;
        memcpy(q, temp_q, sizeof(TYPE)*4);
    }
    
    //part 2: normalize q and obtain V
    TYPE inv_q_length2=1.0/(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    TYPE inv_q_length=sqrt(inv_q_length2);
    q[0]*=inv_q_length;
    q[1]*=inv_q_length;
    q[2]*=inv_q_length;
    q[3]*=inv_q_length;
    S[0]=sqrt(B[0][0])*inv_q_length2;
    S[1]=sqrt(B[1][1])*inv_q_length2;
    S[2]=sqrt(B[2][2])*inv_q_length2;
    
    Quaternion_to_Rotation(q, V);
    
    //Part 3: fix negative S
    int i;
    i=fabs(S[0])<fabs(S[1])?0:1;
    i=(fabs(S[0])<fabs(S[1])?fabs(S[0]):fabs(S[1]))<fabs(S[2])?i:2;
    if(A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])+A[1][0]*(A[2][1]*A[0][2]-A[0][1]*A[2][2])+A[2][0]*(A[0][1]*A[1][2]-A[1][1]*A[0][2])<0)
        S[i]=-S[i];
    
    //Part 4: obtain U
    TYPE rate;
    Matrix_Product_3(&A[0][0], &V[0][0], &U[0][0]);
    rate=1/S[0];
    U[0][0]*=rate;
    U[1][0]*=rate;
    U[2][0]*=rate;
    rate=1/S[1];
    U[0][1]*=rate;
    U[1][1]*=rate;
    U[2][1]*=rate;
    rate=1/S[2];
    U[0][2]*=rate;
    U[1][2]*=rate;
    U[2][2]*=rate;

}