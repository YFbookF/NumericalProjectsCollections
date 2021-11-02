///////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2002 - 2016, Huamin Wang
//  https://web.cse.ohio-state.edu/~wang.3602/publications.html
//  Descent Methods for Elastic Body Simulation on the GPU
//  All rights reserved.
	TYPE Get_Energy(TYPE* _X, TYPE t)
	{
		Swap(X, _X);

        TYPE inv_t	= 1/t;

		TYPE _energy[8];

#pragma omp parallel for
		for(int i=0; i<8; i++)
		{
			int t0=tet_number*((i+0)/8.0);
			int t1=tet_number*((i+1)/8.0);
			_energy[i]=Get_Tet_Energy(t0, t1, stiffness_0, stiffness_1, stiffness_2, stiffness_3);
		}

		TYPE energy	= 0;
		for(int i=0; i<8; i++)
			energy+=_energy[i];

		//for(int tet=0; tet<tet_number; tet++)
		//	energy+=Compute_FM(tet, stiffness_0, stiffness_1, stiffness_2, stiffness_3, false);

		for(int i=0; i<number; i++)
		{
			TYPE oc = M[i]*inv_t*inv_t;
            TYPE c  = oc+fixed[i];
         
			energy+=oc*(S[i*3+0]-X[i*3+0])*(S[i*3+0]-X[i*3+0])*0.5;					// kinetic energy
			energy+=oc*(S[i*3+1]-X[i*3+1])*(S[i*3+1]-X[i*3+1])*0.5;					// kinetic energy
			energy+=oc*(S[i*3+2]-X[i*3+2])*(S[i*3+2]-X[i*3+2])*0.5;					// kinetic energy
			energy+=(c-oc)*(fixed_X[i*3+0]-X[i*3+0])*(fixed_X[i*3+0]-X[i*3+0])*0.5;	// fixed energy
			energy+=(c-oc)*(fixed_X[i*3+1]-X[i*3+1])*(fixed_X[i*3+1]-X[i*3+1])*0.5;	// fixed energy
			energy+=(c-oc)*(fixed_X[i*3+2]-X[i*3+2])*(fixed_X[i*3+2]-X[i*3+2])*0.5;	// fixed energy
	        energy+=-gravity*M[i]*X[i*3+1];
		}

		Swap(X, _X);

		return energy;
	}