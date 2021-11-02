///////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2002 - 2016, Huamin Wang
//  https://web.cse.ohio-state.edu/~wang.3602/publications.html
//  Descent Methods for Elastic Body Simulation on the GPU
//  All rights reserved.
	void Single_Newton_Update(TYPE t, int iterations)
    {   
		//t=0.0005;
        //Damp the system first
        for(int i=0; i<number; i++) //if(fixed[i]==0)
        {
            V[i*3+0]*=0.999;
            V[i*3+1]*=0.999;
            V[i*3+2]*=0.999;
        }
        
        //Calculate the expected position: S
        for(int i=0; i<number; i++)
        {
            S[i*3+0]=X[i*3+0]+V[i*3+0]*t;
            S[i*3+1]=X[i*3+1]+V[i*3+1]*t;
            S[i*3+2]=X[i*3+2]+V[i*3+2]*t;
            
			X[i * 3 + 0] += (V[i * 3 + 0] + (V[i * 3 + 0] - prev_V[i * 3 + 0])*0)*t;
			X[i * 3 + 1] += (V[i * 3 + 1] + (V[i * 3 + 1] - prev_V[i * 3 + 1])*0)*t;
			X[i * 3 + 2] += (V[i * 3 + 2] + (V[i * 3 + 2] - prev_V[i * 3 + 2])*0)*t;
        }     

        TYPE inv_t = 1/t;

		//Get r
        for(int tet=0; tet<tet_number; tet++)
			Compute_FM(tet, stiffness_0, stiffness_1, stiffness_2, stiffness_3, true);
            
        for(int i=0; i<number; i++)
        {
			TYPE c=M[i]*inv_t*inv_t+fixed[i];
            C[i*3+0]=c;
			C[i*3+1]=c;
			C[i*3+2]=c;
                
			TYPE force[3]={0, 0, 0};
			for(int index=vtt_num[i]; index<vtt_num[i+1]; index++)
			{
				force[0]+=F_Temp[VTT[index]*3+0];
				force[1]+=F_Temp[VTT[index]*3+1];
				force[2]+=F_Temp[VTT[index]*3+2];                    
				C[i*3+0]+=C_Temp[VTT[index]*3+0];
				C[i*3+1]+=C_Temp[VTT[index]*3+1];
				C[i*3+2]+=C_Temp[VTT[index]*3+2];
			}
                
			TYPE oc = M[i]*inv_t*inv_t;
			cg_r[i*3+0]=oc*(S[i*3+0]-X[i*3+0])+fixed[i]*(fixed_X[i*3+0]-X[i*3+0])+force[0];
			cg_r[i*3+1]=oc*(S[i*3+1]-X[i*3+1])+fixed[i]*(fixed_X[i*3+1]-X[i*3+1])+force[1]+M[i]*gravity;
			cg_r[i*3+2]=oc*(S[i*3+2]-X[i*3+2])+fixed[i]*(fixed_X[i*3+2]-X[i*3+2])+force[2];
		}


		TYPE stepping=0.001;
		TYPE omega, rho;

		for(int l=0; l<iterations; l++)
		{
			TYPE error=0;
			for(int i=0; i<number*3; i++)	error+=cg_r[i]*cg_r[i];
			printf("%d: %f\n", l, Get_Energy(X, t));

			for(int i=0; i<number*3; i++)	cg_p[i]=cg_r[i]/C[i];

			for(int i=0; i<number*3; i++)	next_X[i]=X[i]+cg_p[i]*stepping;
			
	/*		if (l == 0)				omega=1;
			if (l == 1)				{ rho = 0.96;	omega = 2 / (2 - rho*rho);		}
			if (l > 1 && l < 7)		{ rho = 0.96;	omega = 4 / (4 - rho*rho*omega);}
			if(l==6)				omega=1;
			if (l == 7)				{ rho = 0.99;	omega = 2 / (2 - rho*rho);		}
			if (l > 7 && l < 12)	{ rho = 0.99;	omega = 4 / (4 - rho*rho*omega);}
			if (l == 11)			omega=1;
			if (l == 12)			{ rho = 0.999;  omega = 2 / (2 - rho*rho);		}	//0.9992
			if (l > 12 && l<20)		{ rho = 0.999;  omega = 4 / (4 - rho*rho*omega);}	//0.9992
			if (l == 20)			omega=1;
			if (l == 21)			{ rho = 0.99996;  omega = 2 / (2 - rho*rho);	}
			if (l > 21)				{ rho = 0.99996;  omega = 4 / (4 - rho*rho*omega);}	//0.9999
*/

			if (l == 0)				omega=1;
			if (l == 1)				{ rho = 0.96;	omega = 2 / (2 - rho*rho);		}
			if (l > 1 && l < 7)		{ rho = 0.96;	omega = 4 / (4 - rho*rho*omega);}
			if(l==6)				omega=1;
			if (l == 7)				{ rho = 0.99;	omega = 2 / (2 - rho*rho);		}
			if (l > 7 && l < 12)	{ rho = 0.99;	omega = 4 / (4 - rho*rho*omega);}
			if (l == 11)			omega=1;
			if (l == 12)			{ rho = 0.991;  omega = 2 / (2 - rho*rho);		}	
			if (l > 12 && l<20)		{ rho = 0.991;  omega = 4 / (4 - rho*rho*omega);}	
			if (l == 20)			omega=1;
			if (l == 21)			{ rho = 0.99;  omega = 2 / (2 - rho*rho);		}
			if (l > 21)				{ rho = 0.99;  omega = 4 / (4 - rho*rho*omega);}	//0.99996

			for(int i=0; i<number*3; i++)
            {
                next_X[i]=omega*(next_X[i]-prev_X[i])+prev_X[i];
                prev_X[i]=X[i];
                X[i]=next_X[i];
				cg_p[i]=X[i]-prev_X[i];
            }	

			//Update r
			A_Times(cg_p, cg_Ap, inv_t);
			for(int i=0; i<number*3; i++)	cg_r[i]-=cg_Ap[i];
		}

		for(int i=0; i<number; i++)
        {
            V[i*3+0]=(X[i*3+0]-S[i*3+0])*inv_t;
            V[i*3+1]=(X[i*3+1]-S[i*3+1])*inv_t;
            V[i*3+2]=(X[i*3+2]-S[i*3+2])*inv_t;
        }

	}