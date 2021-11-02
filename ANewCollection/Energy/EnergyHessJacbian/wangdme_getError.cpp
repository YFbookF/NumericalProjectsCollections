///////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2002 - 2016, Huamin Wang
//  https://web.cse.ohio-state.edu/~wang.3602/publications.html
//  Descent Methods for Elastic Body Simulation on the GPU
//  All rights reserved.
		TYPE Get_Error(TYPE t)
	{
		TYPE error_sum=0;
		for (int tet = 0; tet < tet_number; tet++)
			Compute_FM(tet, stiffness_0, stiffness_1, stiffness_2, stiffness_3, false);
		for (int i = 0; i < number; i++)
		{
			TYPE oc = M[i] / (t*t);
			TYPE c = oc + fixed[i];

			TYPE force[3] = { 0, 0, 0 };
			for (int index = vtt_num[i]; index < vtt_num[i + 1]; index++)
			{
				force[0] += F_Temp[VTT[index] * 3 + 0];
				force[1] += F_Temp[VTT[index] * 3 + 1];
				force[2] += F_Temp[VTT[index] * 3 + 2];
			}
			B[i * 3 + 0] = oc*(S[i * 3 + 0] - X[i * 3 + 0]) + fixed[i] * (fixed_X[i * 3 + 0] - X[i * 3 + 0]) + force[0];
			B[i * 3 + 1] = oc*(S[i * 3 + 1] - X[i * 3 + 1]) + fixed[i] * (fixed_X[i * 3 + 1] - X[i * 3 + 1]) + force[1] + M[i] * gravity;
			B[i * 3 + 2] = oc*(S[i * 3 + 2] - X[i * 3 + 2]) + fixed[i] * (fixed_X[i * 3 + 2] - X[i * 3 + 2]) + force[2];
			error_sum += B[i * 3 + 0] * B[i * 3 + 0] + B[i * 3 + 1] * B[i * 3 + 1] + B[i * 3 + 2] * B[i * 3 + 2];
		}
		return error_sum;
	}
    