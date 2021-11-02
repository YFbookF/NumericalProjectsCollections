//  Copyright (C) 2002 - 2016, Huamin Wang
//  https://web.cse.ohio-state.edu/~wang.3602/publications.html
//  Descent Methods for Elastic Body Simulation on the GPU
///////////////////////////////////////////////////////////////////////////////////////////
//  SVD function <from numerical recipes in C++>
//		Given a matrix a[1..m][1..n], this routine computes its singular value
//		decomposition, A = U.W.VT.  The matrix U replaces a on output.  The diagonal
//		matrix of singular values W is output as a vector w[1..n].  The matrix V (not
//		the transpose VT) is output as v[1..n][1..n].
///////////////////////////////////////////////////////////////////////////////////////////
template <class TYPE> FORCEINLINE
TYPE pythag(TYPE a, TYPE b)
{
	TYPE at = fabs(a), bt = fabs(b), ct, result;
	if (at > bt)       { ct = bt / at; result = at * sqrt(1.0 + ct * ct); }
	else if (bt > 0.0) { ct = at / bt; result = bt * sqrt(1.0 + ct * ct); }
	else result = 0.0;
	return(result);
}	

template <class TYPE>
void SVD(TYPE u[], int m, int n, TYPE w[], TYPE v[])
{
	bool flag;
	int i,its,j,jj,k,l,nm;
	TYPE anorm,c,f,g,h,s,scale,x,y,z;
	TYPE *rv1=new TYPE[n];
	g = scale = anorm = 0.0; //Householder reduction to bidiagonal form.
	for(i=0;i<n;i++) 
	{
		l=i+1;
		rv1[i]=scale*g;
		g=s=scale=0.0;
		if (i < m) 
		{
			for (k=i;k<m;k++) scale += fabs(u[k*n+i]);
			if (scale != 0.0) 
			{
				for (k=i;k<m;k++) 
				{
					u[k*n+i] /= scale;
					s += u[k*n+i]*u[k*n+i];
				}
				f=u[i*n+i];
				g = -sqrt(s)*SIGN(f);
				h=f*g-s;
				u[i*n+i]=f-g;
				for (j=l;j<n;j++) 
				{
					for (s=0.0,k=i;k<m;k++) s += u[k*n+i]*u[k*n+j];
					f=s/h;
					for (k=i;k<m;k++) u[k*n+j] += f*u[k*n+i];
				}
				for (k=i;k<m;k++) u[k*n+i] *= scale;
			}
		}
		w[i]=scale *g;
		g=s=scale=0.0;
		if(i+1 <= m && i+1 != n) 
		{
			for(k=l;k<n;k++) scale += fabs(u[i*n+k]);
			if(scale != 0.0) 
			{
				for (k=l;k<n;k++) 
				{
					u[i*n+k] /= scale;
					s += u[i*n+k]*u[i*n+k];
				}
				f=u[i*n+l];
				g = -sqrt(s)*SIGN(f);
				h=f*g-s;
				u[i*n+l]=f-g;
				for (k=l;k<n;k++) rv1[k]=u[i*n+k]/h;
				for (j=l;j<m;j++) 
				{
					for (s=0.0,k=l;k<n;k++) s += u[j*n+k]*u[i*n+k];
					for (k=l;k<n;k++) u[j*n+k] += s*rv1[k];
				}
				for (k=l;k<n;k++) u[i*n+k] *= scale;
			}
		}
		anorm=MAX(anorm,(fabs(w[i])+fabs(rv1[i])));
	}
	for(i=n-1;i>=0;i--) 
	{ //Accumulation of right-hand transformations.
		if (i < n-1) 
		{
			if (g != 0.0) 
			{
				for (j=l;j<n;j++) //Double division to avoid possible under
					v[j*n+i]=(u[i*n+j]/u[i*n+l])/g;
				for (j=l;j<n;j++) 
				{
					for (s=0.0,k=l;k<n;k++) s += u[i*n+k]*v[k*n+j];
					for (k=l;k<n;k++) v[k*n+j] += s*v[k*n+i];
				}
			}
			for (j=l;j<n;j++) v[i*n+j]=v[j*n+i]=0.0;
		}
		v[i*n+i]=1.0;
		g=rv1[i];
		l=i;
	}
	for(i=MIN(m,n)-1;i>=0;i--) 
	{	//Accumulation of left-hand transformations.
		l=i+1;
		g=w[i];
		for (j=l;j<n;j++) u[i*n+j]=0.0;
		if (g != 0.0) 
		{
			g=1.0/g;
			for (j=l;j<n;j++) 
			{
				for (s=0.0,k=l;k<m;k++) s += u[k*n+i]*u[k*n+j];
				f=(s/u[i*n+i])*g;
				for (k=i;k<m;k++) u[k*n+j] += f*u[k*n+i];
			}
			for (j=i;j<m;j++) u[j*n+i] *= g;
		} 
		else for (j=i;j<m;j++) u[j*n+i]=0.0;
		++u[i*n+i];
	}
	
	for(k=n-1;k>=0;k--) 
	{ //Diagonalization of the bidiagonal form: Loop over
		for (its=0;its<30;its++) 
		{ //singular values, and over allowed iterations.
			flag=true;
			for (l=k;l>=0;l--) 
			{ //Test for splitting.
				nm=l-1;
				if ((TYPE)(fabs(rv1[l])+anorm) == anorm) 
				{
					flag=false;
					break;
				}
				if ((TYPE)(fabs(w[nm])+anorm) == anorm) break;
			}
			if(flag) 
			{
				c=0.0; //Cancellation of rv1[l], if l > 0.
				s=1.0;
				for (i=l;i<k+1;i++) 
				{
					f=s*rv1[i];
					rv1[i]=c*rv1[i];
					if ((TYPE)(fabs(f)+anorm) == anorm) break;
					g=w[i];
					h=pythag(f,g);
					w[i]=h;
					h=1.0/h;
					c=g*h;
					s = -f*h;
					for (j=0;j<m;j++) 
					{
						y=u[j*n+nm];
						z=u[j*n+i];
						u[j*n+nm]=y*c+z*s;
						u[j*n+i]=z*c-y*s;
					}
				}
			}
			z=w[k];
			if (l == k) 
			{	// Convergence.
				if (z < 0.0) 
				{ //Singular value is made nonnegative.
					w[k] = -z;
					for (j=0;j<n;j++) v[j*n+k] = -v[j*n+k];
				}
				break;
			}
			if (its == 29) {printf("Error: no convergence in 30 svdcmp iterations");getchar();}
			x=w[l]; //Shift from bottom 2-by-2 minor.
			nm=k-1;
			y=w[nm];
			g=rv1[nm];
			h=rv1[k];
			f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
			g=pythag(f,1.0);
			f=((x-z)*(x+z)+h*((y/(f+fabs(g)*SIGN(f)))-h))/x;
			c=s=1.0; //Next QR transformation:
			for (j=l;j<=nm;j++) 
			{
				i=j+1;
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
				y *= c;
				for (jj=0;jj<n;jj++) 
				{
					x=v[jj*n+j];
					z=v[jj*n+i];
					v[jj*n+j]=x*c+z*s;
					v[jj*n+i]=z*c-x*s;
				}
				z=pythag(f,h);
				w[j]=z; //Rotation can be arbitrary if z D 0.
				if (z) 
				{
					z=1.0/z;
					c=f*z;
					s=h*z;
				}
				f=c*g+s*y;
				x=c*y-s*g;
				for (jj=0;jj<m;jj++) 
				{
					y=u[jj*n+j];
					z=u[jj*n+i];
					u[jj*n+j]=y*c+z*s;
					u[jj*n+i]=z*c-y*s;
				}
			}
			rv1[l]=0.0;
			rv1[k]=f;
			w[k]=x;
		}
	}
	delete []rv1;
}