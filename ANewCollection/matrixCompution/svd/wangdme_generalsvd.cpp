//  Copyright (C) 2002 - 2016, Huamin Wang
//  https://web.cse.ohio-state.edu/~wang.3602/publications.html
//  Descent Methods for Elastic Body Simulation on the GPU
template <class TYPE>
void SVD3(TYPE u[3][3], TYPE w[3], TYPE v[3][3])
{
	TYPE	anorm,c,f,g,h,s,scale;
	TYPE	x,y,z;
	TYPE	rv1[3];
	g = scale = anorm = 0.0; //Householder reduction to bidiagonal form.

	for(int i=0; i<3; i++) 
	{
		int l=i+1;
		rv1[i]=scale*g;
		g=s=scale=0.0;
		if(i<3) 
		{
			for(int k=i; k<3; k++) scale += fabsf(u[k][i]);
			if(scale!=0) 
			{
				for(int k=i; k<3; k++) 
				{
					u[k][i]/=scale;
					s+=u[k][i]*u[k][i];
				}
				f=u[i][i];
				g=-sqrtf(s)*SIGN(f);
				h=f*g-s;
				u[i][i]=f-g;
				for(int j=l; j<3; j++) 
				{
					s=0;
					for(int k=i;k<3;k++)	s+=u[k][i]*u[k][j];
					f=s/h;
					for(int k=i; k<3; k++)	u[k][j]+=f*u[k][i];
				}
				for(int k=i; k<3; k++)		u[k][i]*=scale;
			}
		}
		w[i]=scale*g;

		g=s=scale=0.0;
		if(i<=2 && i!=2) 
		{
			for(int k=l; k<3; k++)	scale+=fabsf(u[i][k]);
			if(scale!=0) 
			{
				for(int k=l; k<3; k++) 
				{
					u[i][k]/=scale;
					s+=u[i][k]*u[i][k];
				}
				f=u[i][l];
				g=-sqrtf(s)*SIGN(f);
				h=f*g-s;
				u[i][l]=f-g;
				for(int k=l; k<3; k++) rv1[k]=u[i][k]/h;
				for(int j=l; j<3; j++) 
				{
					s=0;
					for(int k=l; k<3; k++)	s+=u[j][k]*u[i][k];
					for(int k=l; k<3; k++)	u[j][k]+=s*rv1[k];
				}
				for(int k=l; k<3; k++) u[i][k]*=scale;
			}
		}
		anorm=MAX(anorm,(fabs(w[i])+fabs(rv1[i])));
	}
	
	for(int i=2, l; i>=0; i--) //Accumulation of right-hand transformations.
	{ 
		if(i<2) 
		{
			if(g!=0) 
			{
				for(int j=l; j<3; j++) //Double division to avoid possible under				
					v[j][i]=(u[i][j]/u[i][l])/g;
				for(int j=l; j<3; j++) 
				{
					s=0;
					for(int k=l; k<3; k++)	s+=u[i][k]*v[k][j];
					for(int k=l; k<3; k++)	v[k][j]+=s*v[k][i];
				}
			}
			for(int j=l; j<3; j++)	v[i][j]=v[j][i]=0.0;
		}
		v[i][i]=1.0;
		g=rv1[i];
		l=i;
	}
	
	for(int i=2; i>=0; i--) //Accumulation of left-hand transformations.
	{	
		int l=i+1;
		g=w[i];
		for(int j=l; j<3; j++) u[i][j]=0;
		if(g!=0)
		{
			g=1/g;
			for(int j=l; j<3; j++) 
			{
				s=0;
				for(int k=l; k<3; k++)	s+=u[k][i]*u[k][j];
				f=(s/u[i][i])*g;
				for(int k=i; k<3; k++)	u[k][j]+=f*u[k][i];
			}
			for(int j=i; j<3; j++)		u[j][i]*=g;
		} 
		else for(int j=i; j<3; j++)		u[j][i]=0.0;
		u[i][i]++;
	}

	for(int k=2; k>=0; k--)				//Diagonalization of the bidiagonal form: Loop over
	{ 
		for(int its=0; its<30; its++)	//singular values, and over allowed iterations.
		{ 
			bool flag=true;
			int  l;
			int	 nm;
			for(l=k; l>=0; l--)			//Test for splitting.
			{ 
				nm=l-1;
				if((TYPE)(fabs(rv1[l])+anorm)==anorm) 
				{
					flag=false;
					break;
				}
				if((TYPE)(fabs(w[nm])+anorm)==anorm)	break;
			}
			if(flag)
			{
				c=0.0; //Cancellation of rv1[l], if l > 0.
				s=1.0;
				for(int i=l; i<k+1; i++) 
				{
					f=s*rv1[i];
					rv1[i]=c*rv1[i];
					if((TYPE)(fabs(f)+anorm) == anorm) break;
					g=w[i];
					h=pythag(f,g);
					w[i]=h;
					h=1.0/h;
					c= g*h;
					s=-f*h;
					for(int j=0; j<3; j++) 
					{
						y=u[j][nm];
						z=u[j][i ];
						u[j][nm]=y*c+z*s;
						u[j][i ]=z*c-y*s;
					}
				}
			}
			z=w[k];
			if(l==k)		// Convergence.
			{	
				if(z<0.0)	// Singular value is made nonnegative.
				{ 
					w[k]=-z;
					for(int j=0; j<3; j++) v[j][k]=-v[j][k];
				}
				break;
			}
			if(its==29) {printf("Error: no convergence in 30 svdcmp iterations");getchar();}
			x=w[l]; //Shift from bottom 2-by-2 minor.
			nm=k-1;
			y=w[nm];
			g=rv1[nm];
			h=rv1[k];
			f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
			g=pythag(f, (TYPE)1.0);
			f=((x-z)*(x+z)+h*((y/(f+fabs(g)*SIGN(f)))-h))/x;
			c=s=1.0; //Next QR transformation:
			
			for(int j=l; j<=nm; j++) 
			{
				int i=j+1;
				g=rv1[i];
				y=w[i];
				h=s*g;
				g=c*g;
				z=pythag(f,h);
				rv1[j]=z;
				c=f/z;
				s=h/z;
				f=x*c+g*s;
				g=g*c-x*s;
				h=y*s;
				y*=c;
				for(int jj=0; jj<3; jj++) 
				{
					x=v[jj][j];
					z=v[jj][i];
					v[jj][j]=x*c+z*s;
					v[jj][i]=z*c-x*s;
				}
				z=pythag(f,h);
				w[j]=z; //Rotation can be arbitrary if z D 0.
				if(z) 
				{
					z=1.0/z;
					c=f*z;
					s=h*z;
				}
				f=c*g+s*y;
				x=c*y-s*g;
				for(int jj=0; jj<3; jj++) 
				{
					y=u[jj][j];
					z=u[jj][i];
					u[jj][j]=y*c+z*s;
					u[jj][i]=z*c-y*s;
				}
			}
			rv1[l]=0.0;
			rv1[k]=f;
			w[k]=x;
		}
	}
}
