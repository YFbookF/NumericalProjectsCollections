//  Copyright (C) 2002 - 2016, Huamin Wang
//  https://web.cse.ohio-state.edu/~wang.3602/publications.html
//  Descent Methods for Elastic Body Simulation on the GPU
template <class T> FORCEINLINE
void Matrix_Factorization_3(T *A, T *R)			//R=chol(A), Chelosky factorization
{
	R[0]=sqrtf(A[0]);
	R[1]=A[1]/R[0];
	R[2]=A[2]/R[0];
	R[3]=0;
	R[4]=sqrtf(A[4]-R[1]*R[1]);
	R[5]=(A[5]-R[1]*R[2])/R[4];
	R[6]=0;
	R[7]=0;
	R[8]=sqrtf(A[8]-R[2]*R[2]-R[5]*R[5]);
}
