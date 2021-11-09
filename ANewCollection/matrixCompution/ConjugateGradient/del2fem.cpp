// ---------------------------------------
// Solve Matrix with BiCGSTAB Methods
// ---------------------------------------
bool delfem2::Solve_BiCGSTAB
(double& conv_ratio, int& iteration,
std::vector<double>& u_vec,
const std::vector<double>& A,
const std::vector<double>& y_vec)
{

  //	std::cout.precision(18);

  const double conv_ratio_tol = conv_ratio;
  const unsigned int mx_iter = iteration;

  const unsigned int n = static_cast<unsigned int>(y_vec.size());

  u_vec.assign(n, 0);

  assert(A.size()==n*n);

  std::vector<double>  r_vec = y_vec;
  std::vector<double>  s_vec(n);
  std::vector<double> As_vec(n);
  std::vector<double>  p_vec(n);
  std::vector<double> Ap_vec(n);

  std::vector<double>  r0(n);

  double sq_inv_norm_res;
  {
    double dtmp1 = squaredNorm(r_vec);
    //    std::cout << "Initial Residual: " << sqrt(dtmp1) << std::endl;
    if (dtmp1 < 1.0e-30){
      conv_ratio = 0.0;
      iteration = 0;
      return true;
    }
    sq_inv_norm_res = 1.0/dtmp1;
  }

  r0 = r_vec;
  //  for (int i = 0; i<n; ++i){ r0[i] = std::conj(r_vec[i]); }

  // {p} = {r}
  p_vec = r_vec;

  // calc (r,r0*)
  double r_r0conj = bem::dot(r_vec, r0);

  iteration = mx_iter;
  for (unsigned int iitr = 1; iitr<mx_iter; iitr++){

    // calc {Ap} = [A]*{p}
    matVec(Ap_vec, A, p_vec);

    // calc alpha
    double alpha;
    {
      const double den = bem::dot(Ap_vec, r0);
      alpha = r_r0conj/den;
    }

    // calc s_vector
    for (unsigned int i = 0; i<n; ++i){ s_vec[i] = r_vec[i]-alpha*Ap_vec[i]; }

    // calc {As} = [A]*{s}
    matVec(As_vec, A, s_vec);

    // calc omega
    double omega;
    {
      const double den = squaredNorm(As_vec);
      const double num = bem::dot(As_vec, s_vec);
      omega = num/den;
    }

    // update solution
    for (unsigned int i = 0; i<n; ++i){ u_vec[i] += alpha*p_vec[i]+omega*s_vec[i]; }

    // update residual
    for (unsigned int i = 0; i<n; ++i){ r_vec[i] = s_vec[i]-omega*As_vec[i]; }

    {
      const double sq_norm_res = squaredNorm(r_vec);
      const double sq_conv_ratio = sq_norm_res * sq_inv_norm_res;
       std::cout << iitr << " " << sqrt(sq_conv_ratio) << " " << sqrt(sq_norm_res) << std::endl;
      if (sq_conv_ratio < conv_ratio_tol * conv_ratio_tol){
        conv_ratio = sqrt(sq_norm_res * sq_inv_norm_res);
        iteration = iitr;
        return true;
      }
    }

    // calc beta
    double beta;
    {
      const double tmp1 = bem::dot(r_vec, r0);
      beta = (tmp1/r_r0conj) * (alpha/omega);
      r_r0conj = tmp1;
    }

    // update p_vector
    for (unsigned int i = 0; i<n; ++i){
      p_vec[i] = beta*p_vec[i]+r_vec[i]-(beta*omega)*Ap_vec[i];
    }
  }

  return true;
}
