====================Nonlinear Material Design Using Principal Stretches  

Separable Material Energy Examples

for assumption 11, The st.Venant-Kirchhoff material is
$$
f(x) = \frac{1}{8}\lambda(x^4 - 6x^2 + 5) + \frac{1}{4}\mu(x^2 - 1)^2 \qquad g(x) = \frac{1}{4}\lambda(x^2 - 1) \qquad h(x) = 0
$$
http://www.github.com/mattoverby/admm-elastic

```
	class StVK : public Spline {
	public:
		StVK( double mu_, double lambda_, double kappa_ ) :
			mu(mu_), lambda(lambda_), kappa(kappa_) {}
		const double mu, lambda, kappa;
		double f(double x) const {
			double x2 = x*x;
			return 0.125*lambda*( x2*x2 - 6.0*x2 + 5.0 ) + 0.25*mu * (x2-1.0)*(x2-1.0);
		}
		double g(double x) const { return 0.25 * lambda * ( x*x - 1.0 ); }
		double h(double x) const { return compress_term(kappa,x); }
		double df(double x) const {
			double x2 = x*x;
			return 0.125*lambda*(4.0*x2*x - 12.0*x) + mu*x*(x2-1.0);
		}
		double dg(double x) const { return 0.5*lambda*x; }
		double dh(double x) const { return d_compress_term(kappa,x); }
	};
```

注意dg主要用来算梯度

```

double SplineTet::SplineProx::energy_density(const Vec3 &x) const {
	return spline->f(x[0]) + spline->f(x[1]) + spline->f(x[2]) +
		spline->g(x[0]*x[1]) + spline->g(x[1]*x[2]) + spline->g(x[2]*x[0]) +
		spline->h(x[0]*x[1]*x[2]); 
}

double SplineTet::SplineProx::value(const Vec3 &x){
	if( x[0]<0.0 || x[1]<0.0 || x[2]<0.0 ){
		// No Mr. Linesearch, you have gone too far!
		return std::numeric_limits<float>::max();
	}
	double t1 = energy_density(x); // U(Dx)
	double t2 = (k*0.5) * (x-x0).squaredNorm(); // quad penalty
	return t1 + t2;
}

double SplineTet::SplineProx::gradient(const Vec3 &x, Vec3 &grad){
	double hprime = spline->dh(x[0]*x[1]*x[2]);
	grad[0] = spline->df(x[0]) + spline->dg(x[0]*x[1])*x[1] + spline->dg(x[2]*x[0])*x[2] + hprime*x[1]*x[2] + k*(x[0]-x0[0]);
	grad[1] = spline->df(x[1]) + spline->dg(x[1]*x[2])*x[2] + spline->dg(x[0]*x[1])*x[0] + hprime*x[2]*x[0] + k*(x[1]-x0[1]);
	grad[2] = spline->df(x[2]) + spline->dg(x[2]*x[0])*x[0] + spline->dg(x[1]*x[2])*x[1] + hprime*x[0]*x[1] + k*(x[2]-x0[2]);
	return value(x);
}


```

这里使用Strain Energy Density用principal stretches 表达
$$
\Psi(\lambda_1,\lambda_2,\lambda_3) = f(\lambda_1) + f(\lambda_2) + f(\lambda_3) + g(\lambda_2\lambda_3) + g(\lambda_1\lambda_2) + g(\lambda_1 \lambda_3) + h(\lambda_1 \lambda_2 \lambda_3)
$$
求导如下
$$
\frac{\partial \Psi}{\partial \lambda_1} = f'(\lambda_1) + g'(\lambda_1\lambda_2)\lambda_2 + g'(\lambda_3\lambda_1)\lambda_3 + h'(\lambda_1\lambda_2\lambda_3)\lambda_2 \lambda_3
$$
