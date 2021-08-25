$$
\bold W = \bold U(\frac{1}{2}\nabla _{\bold S_{\bold J}}\mathcal{D}(\bold S_{\bold J})(\bold S_{\bold J}-\bold I)^{-1})^{1/2}\bold U^T = \bold U\bold S_{\bold W}\bold U^T
$$

```
	void operator()(const tbb::blocked_range<size_t>& r) const {
		const double eps = 1e-8;
		double exp_factor = m_state->exp_factor;
		for (size_t i = r.begin(); i != r.end(); ++i) {
			typedef Eigen::Matrix<double, 2, 2> Mat2;
			typedef Eigen::Matrix<double, 2, 1> Vec2;
			Mat2 ji, ri, ti, ui, vi; Vec2 sing; Vec2 closest_sing_vec; Mat2 mat_W;
			Mat2 fGrad; Vec2 m_sing_new;
			double s1, s2;

			ji(0, 0) = Ji(i, 0); ji(0, 1) = Ji(i, 1);
			ji(1, 0) = Ji(i, 2); ji(1, 1) = Ji(i, 3);

			igl::polar_svd(ji, ri, ti, ui, sing, vi);

			s1 = sing(0); s2 = sing(1);

			// Update Weights (currently supports only symmetric dirichlet)
			double s1_g = 2 * (s1 - pow(s1, -3));
			double s2_g = 2 * (s2 - pow(s2, -3));
			m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(s2_g / (2 * (s2 - 1)));

			if (abs(s1 - 1) < eps) m_sing_new(0) = 1; if (abs(s2 - 1) < eps) m_sing_new(1) = 1;
			mat_W = ui * m_sing_new.asDiagonal() * ui.transpose();

			W_11(i) = mat_W(0, 0);
			W_12(i) = mat_W(0, 1);
			W_21(i) = mat_W(1, 0);
			W_22(i) = mat_W(1, 1);

			// 2) Update closest rotations (not rotations in case of conformal energy)
			Ri(i, 0) = ri(0, 0); Ri(i, 1) = ri(1, 0); Ri(i, 2) = ri(0, 1); Ri(i, 3) = ri(1, 1);
		}
	}
```

```
void compute_energies_with_jacobians(const Eigen::MatrixXd& V,
	const Eigen::MatrixXi& F, const Eigen::MatrixXd& Ji, Eigen::MatrixXd& uv, Eigen::VectorXd& areas,
	double& schaeffer_e, double& log_e, double& conf_e, double& norm_arap_e, double& amips, double& exp_symmd, double exp_factor, bool flips_linesearch) {

	int f_n = F.rows();
	//conformal

	schaeffer_e = log_e = conf_e = 0; norm_arap_e = 0; amips = 0;
	Eigen::Matrix<double, 2, 2> ji;
	for (int i = 0; i < f_n; i++) {
		ji(0, 0) = Ji(i, 0); ji(0, 1) = Ji(i, 1);
		ji(1, 0) = Ji(i, 2); ji(1, 1) = Ji(i, 3);

		typedef Eigen::Matrix<double, 2, 2> Mat2;
		typedef Eigen::Matrix<double, 2, 1> Vec2;
		Mat2 ri, ti, ui, vi; Vec2 sing;
		igl::polar_svd(ji, ri, ti, ui, sing, vi);
		double s1 = sing(0); double s2 = sing(1);

		if (flips_linesearch) {
			schaeffer_e += areas(i) * (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2));
			log_e += areas(i) * (pow(log(s1), 2) + pow(log(s2), 2));
			double sigma_geo_avg = sqrt(s1 * s2);
			//conf_e += areas(i) * (pow(log(s1/sigma_geo_avg),2) + pow(log(s2/sigma_geo_avg),2));
			conf_e += areas(i) * ((pow(s1, 2) + pow(s2, 2)) / (2 * s1 * s2));
			norm_arap_e += areas(i) * (pow(s1 - 1, 2) + pow(s2 - 1, 2));
			amips += areas(i) * exp(exp_factor * (0.5 * ((s1 / s2) + (s2 / s1)) + 0.25 * ((s1 * s2) + (1. / (s1 * s2)))));
			exp_symmd += areas(i) * exp(exp_factor * (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2)));
			//amips +=  areas(i) * exp(  0.5*( (s1/s2) +(s2/s1) ) + 0.25*( (s1*s2) + (1./(s1*s2)) )  ) ;
		}
		else {
			if (ui.determinant() * vi.determinant() > 0) {
				norm_arap_e += areas(i) * (pow(s1 - 1, 2) + pow(s2 - 1, 2));
			}
			else {
				// it is the distance form the flipped thing, this is slow, usefull only for debugging normal arap
				vi.col(1) *= -1;
				norm_arap_e += areas(i) * (ji - ui * vi.transpose()).squaredNorm();
			}
		}

	}

}
```

