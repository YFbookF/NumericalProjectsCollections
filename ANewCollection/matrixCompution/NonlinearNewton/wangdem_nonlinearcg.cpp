///////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2002 - 2016, Huamin Wang
//  https://web.cse.ohio-state.edu/~wang.3602/publications.html
//  Descent Methods for Elastic Body Simulation on the GPU
//  All rights reserved.
void CG_Update(TYPE t, int iterations)
    {
        //memset(lambda, 0, sizeof(TYPE)*tet_number);

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
            
            //X[i*3+0]+=V[i*3+0]*t;
            //X[i*3+1]+=V[i*3+1]*t;
            //X[i*3+2]+=V[i*3+2]*t;

			X[i * 3 + 0] += (V[i * 3 + 0] + (V[i * 3 + 0] - prev_V[i * 3 + 0])*0)*t;
			X[i * 3 + 1] += (V[i * 3 + 1] + (V[i * 3 + 1] - prev_V[i * 3 + 1])*0)*t;
			X[i * 3 + 2] += (V[i * 3 + 2] + (V[i * 3 + 2] - prev_V[i * 3 + 2])*0)*t;
        }        

        
        TYPE inv_t = 1/t;
        TYPE zr = 0;
        
   /*    for(int tet=0; tet<tet_number; tet++)
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
            B[i*3+0]=oc*(S[i*3+0])+fixed[i]*(fixed_X[i*3+0])+force[0];
            B[i*3+1]=oc*(S[i*3+1])+fixed[i]*(fixed_X[i*3+1])+force[1]+M[i]*gravity;
            B[i*3+2]=oc*(S[i*3+2])+fixed[i]*(fixed_X[i*3+2])+force[2];
            
            //printf("%f, %f, %f, ", B[i*3+0], B[i*3+1], B[i*3+2]);
            //printf("f: %f, %f, %f\n", force[0], force[1], force[2]);
            //printf("sx: %f, %f, %f; %f, %f, %f\n", S[i*3+0]+V[i*3+0]*t, S[i*3+1]+V[i*3+1]*t, S[i*3+2]+V[i*3+2]*t,
            //       X[i*3+0], X[i*3+1], X[i*3+2]);
        }*/

        //for(int i=0; i<number*3; i++)
        //    X[i]=RandomFloat()*2-1;
        
       /* printf("\n");
        printf("x is:\n");
        for(int i=0; i<number*3; i++)
            printf("%f, ", X[i]);
        printf("\n");*/
        //            printf("%f\n", Get_Error(t));
		//			getchar();



        for(int l=0; l<iterations; l++)
        {
			memset(ext_C, 0, sizeof(TYPE)*number);
			memset(lambda, 0, sizeof(TYPE)*tet_number);

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
            
            
			//printf("error: %f, %f\n", Get_Error(t), Get_Energy(X, t));
			//printf("cgr: %f\n", cg_r[10]);
            //A_Times(X, cg_r, inv_t);
            //TYPE diff_error=0;
            //for(int i=0; i<number*3; i++)
            //    diff_error+=X[i]*cg_r[i]-2*B[i]*X[i];
            //for(int i=0; i<number*3; i++)   cg_r[i]=B[i]-cg_r[i];
            
            //for(int i=0; i<number*3; i++)
            //    printf("r %d: %f (%f)\n", i, cg_r[i], X[i]);
            
            for(int k=0; k<1; k++)
            {
            TYPE error_mag=0;
            for(int i=0; i<number; i++)
                error_mag+=cg_r[i*3+0]*cg_r[i*3+0]+cg_r[i*3+1]*cg_r[i*3+1]+cg_r[i*3+2]*cg_r[i*3+2];
            
			TYPE error_mag1=0;
            A_Times(X, cg_Ap, inv_t);
			for(int i=0; i<number*3; i++)
                error_mag1+=-X[i]*cg_Ap[i]-2*cg_r[i]*X[i];
            
			printf("%f\n", Get_Energy(X, t), error_mag);
			//printf("%f\n", error_mag1);
            //getchar();

            //Precondition solve
            for(int i=0; i<number*3; i++)   cg_z[i]=cg_r[i];///(C[i]+ext_C[i/3]); //note here here
            
            //Update p
            TYPE old_zr=zr;
            zr=0;
            for(int i=0; i<number*3; i++)   zr+=cg_z[i]*cg_r[i];
            TYPE beta=0;
			if(l!=0)    
			//if(k!=0)	
				beta=zr/old_zr;			
            
		//	printf("beta: %f (%f, %f)\n", beta, old_zr, zr);

            for(int i=0; i<number*3; i++)   cg_p[i]=cg_z[i]+beta*cg_p[i];
            
            //printf("beta: %f (%f, %f)\n", beta, zr, old_zr);
            
            //for(int i=0; i<number*3; i++)
            //    printf("P: %d, %f\n", i, cg_p[i]);
            
            //get alpha
            A_Times(cg_p, cg_Ap, inv_t);

            //for(int i=0; i<number*3; i++)
            //    printf("P: %d, %f, %f\n", i, cg_p[i], cg_Ap[i]);
		//	printf("PAP: %f, %f\n", cg_p[10], cg_Ap[10]);

            
            TYPE alpha=0;
            for(int i=0; i<number*3; i++)   alpha+=cg_p[i]*cg_Ap[i];    //alpha=pAp
            
		//	printf("alpha: %f\n", alpha);

            //for(int i=0; i<number*3; i++)
            //    printf("value: %f, %f, %f, %f\n", cg_z[i], cg_r[i], cg_p[i], cg_Ap[i]);
            
            //printf("value: %f, %f\n", zr, alpha);

            alpha=zr/alpha;                                             //alpha=zr/pAp
            
            //printf("alpha: %ef\n", alpha);
            
            //Update X
            for(int i=0; i<number*3; i++)
            {
                X[i]=X[i]+alpha*cg_p[i];
                cg_r[i]=cg_r[i]-alpha*cg_Ap[i];
            }

            //for(int i=0; i<number*3; i++)   X[i]=X[i]+alpha*cg_p[i];
		//	printf("end: %f\n", cg_r[10]);
            
            //getchar();
            }
        }
        
        for(int i=0; i<number; i++)
        {
            V[i*3+0]+=(X[i*3+0]-S[i*3+0])*inv_t;
            V[i*3+1]+=(X[i*3+1]-S[i*3+1])*inv_t;
            V[i*3+2]+=(X[i*3+2]-S[i*3+2])*inv_t;
        }
    }
