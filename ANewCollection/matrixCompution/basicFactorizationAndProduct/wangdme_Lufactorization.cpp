//  Copyright (C) 2002 - 2016, Huamin Wang
//  https://web.cse.ohio-state.edu/~wang.3602/publications.html
//  Descent Methods for Elastic Body Simulation on the GPU
template <class T> FORCEINLINE
void Matrix_Factorization_3(T* A, T* L, T* U)			//R=L*U, LU factorization
{
	memset(L, 0, sizeof(T)*9);
	memset(U, 0, sizeof(T)*9);
	L[0]=A[0];
	L[3]=A[3];
	L[6]=A[6];
	U[0]=1;
	U[4]=1;
	U[8]=1;
	U[1]=A[1]/L[0];
	U[2]=A[2]/L[0];
	L[4]=A[4]-L[3]*U[1];
	L[7]=A[7]-L[6]*U[1];
	U[5]=(A[5]-L[3]*U[2])/L[4];
	L[8]=A[8]-L[6]*U[2]-L[7]*U[5];
}