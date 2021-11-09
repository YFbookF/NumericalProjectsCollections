//Projective Dynamics: Fast Simulation of Hyperelastic Models with Dynamic Constraints
//http://www.github.com/mattoverby/admm-elastic
double NeoHookeanTet::NHProx::energy_density(const Vec3 &x) const {
	double J = x[0]*x[1]*x[2];
	double I_1 = x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
	double I_3 = J*J;
	double log_I3 = std::log( I_3 );
	double t1 = 0.5 * mu * ( I_1 - log_I3 - 3.0 );
	double t2 = 0.125 * lambda * log_I3 * log_I3;
	double r = t1 + t2;
	return r;
}

double NeoHookeanTet::NHProx::value(const Vec3 &x){
	if( x[0]<0.0 || x[1]<0.0 || x[2]<0.0 ){
		// No Mr. Linesearch, you have gone too far!
		return std::numeric_limits<float>::max();
	}
	double t1 = energy_density(x); // U(Dx)
	double t2 = (k*0.5) * (x-x0).squaredNorm(); // quad penalty
	return t1 + t2;
}


double NeoHookeanTet::NHProx::gradient(const Vec3 &x, Vec3 &grad){
	double J = x[0]*x[1]*x[2];
	if( J <= 0.0 ){
		throw std::runtime_error("NeoHookeanTet::NHProx::gradient Error: J <= 0");
	} else {
		Eigen::Vector3d x_inv(1.0/x[0],1.0/x[1],1.0/x[2]);
		grad = (mu * (x - x_inv) + lambda * std::log(J) * x_inv) + k*(x-x0);
	}
	return value(x);
}

//
//	St Venant-Kirchhoff
//

double StVKTet::StVKProx::value(const Vec3 &x){
	if( x[0]<0.0 || x[1]<0.0 || x[2]<0.0 ){
		// No Mr. Linesearch, you have gone too far!
		return std::numeric_limits<float>::max();
	}
	double t1 = energy_density(x); // U(Dx)
	double t2 = (k*0.5) * (x-x0).squaredNorm(); // quad penalty
	return t1 + t2;
}

double StVKTet::StVKProx::energy_density(const Vec3 &x) const {
	Vec3 x2( x[0]*x[0], x[1]*x[1], x[2]*x[2] );
	Vec3 st = 0.5 * ( x2 - Vec3(1,1,1) ); // strain tensor
	double st_tr2 = v3trace(st)*v3trace(st);
	double r = ( mu * ddot( st, st ) + ( lambda * 0.5 * st_tr2 ) );
	return r;
}

double StVKTet::StVKProx::gradient(const Vec3 &x, Vec3 &grad){
	Vec3 term1(
		mu * x[0]*(x[0]*x[0] - 1.0),
		mu * x[1]*(x[1]*x[1] - 1.0),
		mu * x[2]*(x[2]*x[2] - 1.0)
	);
	Vec3 term2 = 0.5 * lambda * ( x.dot(x) - 3.0 ) * x;
	grad = term1 + term2 + k*(x-x0);
	return value(x);
}

//
//	Spline Tets
//

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


