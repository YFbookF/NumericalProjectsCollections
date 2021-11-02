//  Copyright (C) 2002 - 2016, Huamin Wang
//  https://web.cse.ohio-state.edu/~wang.3602/publications.html
//  Descent Methods for Elastic Body Simulation on the GPU
void eigen_decompose(float a[], int n, float d[], float v[], int *nrot)
{
	int j,iq,ip,i;
	float tresh,theta,tau,t,sm,s,h,g,c;

	float* b=new float[n+1];
	float* z=new float[n+1];
	for (ip=1;ip<=n;ip++) 
	{ 
		for (iq=1;iq<=n;iq++) v[ip*(n+1)+iq]=0.0;
		v[ip*(n+1)+ip]=1.0;
	}
	for (ip=1;ip<=n;ip++) 
	{
		b[ip]=d[ip]=a[ip*(n+1)+ip]; 
		z[ip]=0.0; 
	}
	*nrot=0;
	for (i=1;i<=50;i++) 
	{
		sm=0.0;
		for (ip=1;ip<=n-1;ip++) 
		{ 
			for (iq=ip+1;iq<=n;iq++)
			sm += fabs(a[ip*(n+1)+iq]);
		}
		if (sm == 0.0) 
		{ 
			delete[] z;
			delete[] b;
			return;
		}
		if (i < 4)
			tresh=0.2*sm/(n*n); 
		else
			tresh=0.0;
		for (ip=1;ip<=n-1;ip++) 
		{
			for (iq=ip+1;iq<=n;iq++) 
			{
				g=100.0*fabs(a[ip*(n+1)+iq]);
				if (i > 4 && (float)(fabs(d[ip])+g) == (float)fabs(d[ip])&& (float)(fabs(d[iq])+g) == (float)fabs(d[iq]))
					a[ip*(n+1)+iq]=0.0;
				else if (fabs(a[ip*(n+1)+iq]) > tresh) 
				{
					h=d[iq]-d[ip];
					if ((float)(fabs(h)+g) == (float)fabs(h))
						t=(a[ip*(n+1)+iq])/h; 
					else 
					{
						theta=0.5*h/(a[ip*(n+1)+iq]); 
						t=1.0/(fabs(theta)+sqrt(1.0+theta*theta));
						if (theta < 0.0) t = -t;
					}
					c=1.0/sqrt(1+t*t);
					s=t*c;
					tau=s/(1.0+c);
					h=t*a[ip*(n+1)+iq];
					z[ip] -= h;
					z[iq] += h;
					d[ip] -= h;
					d[iq] += h;
					a[ip*(n+1)+iq]=0.0;
					for (j=1;j<=ip-1;j++) 
					{ 
						ROTATE(a,j,ip,j,iq)
					}
					for (j=ip+1;j<=iq-1;j++) 
					{ 
						ROTATE(a,ip,j,j,iq)
					}
					for (j=iq+1;j<=n;j++) 
					{
						ROTATE(a,ip,j,iq,j)
					}
					for (j=1;j<=n;j++) 
					{
						ROTATE(v,j,ip,j,iq)
					}
					++(*nrot);
				}
			}
		}
		for (ip=1;ip<=n;ip++) 
		{
			b[ip] += z[ip];
			d[ip]=b[ip]; 
			z[ip]=0.0;
		}
	}
	printf("Too many iterations in routine jacobi\n");

}
