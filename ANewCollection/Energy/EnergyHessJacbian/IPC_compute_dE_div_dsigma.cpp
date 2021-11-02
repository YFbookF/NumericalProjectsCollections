https://github.com/ipc-sim/IPC
// 注意这是coratation的
template <int dim>
void FixedCoRotEnergy<dim>::compute_dE_div_dsigma(const Eigen::Matrix<double, dim, 1>& singularValues,
    double u, double lambda,
    Eigen::Matrix<double, dim, 1>& dE_div_dsigma) const
{
    const double sigmaProdm1lambda = lambda * (singularValues.prod() - 1.0);
    Eigen::Matrix<double, dim, 1> sigmaProd_noI;
    if constexpr (dim == 2) {
        sigmaProd_noI[0] = singularValues[1];
        sigmaProd_noI[1] = singularValues[0];
    }
    else {
        sigmaProd_noI[0] = singularValues[1] * singularValues[2];
        sigmaProd_noI[1] = singularValues[2] * singularValues[0];
        sigmaProd_noI[2] = singularValues[0] * singularValues[1];
    }

    double _2u = u * 2;
    dE_div_dsigma[0] = (_2u * (singularValues[0] - 1.0) + sigmaProd_noI[0] * sigmaProdm1lambda);
    dE_div_dsigma[1] = (_2u * (singularValues[1] - 1.0) + sigmaProd_noI[1] * sigmaProdm1lambda);
    if constexpr (dim == 3) {
        dE_div_dsigma[2] = (_2u * (singularValues[2] - 1.0) + sigmaProd_noI[2] * sigmaProdm1lambda);
    }
}
template <int dim>
void FixedCoRotEnergy<dim>::compute_d2E_div_dsigma2(const Eigen::Matrix<double, dim, 1>& singularValues,
    double u, double lambda,
    Eigen::Matrix<double, dim, dim>& d2E_div_dsigma2) const
{
    const double sigmaProd = singularValues.prod();
    Eigen::Matrix<double, dim, 1> sigmaProd_noI;
    if constexpr (dim == 2) {
        sigmaProd_noI[0] = singularValues[1];
        sigmaProd_noI[1] = singularValues[0];
    }
    else {
        sigmaProd_noI[0] = singularValues[1] * singularValues[2];
        sigmaProd_noI[1] = singularValues[2] * singularValues[0];
        sigmaProd_noI[2] = singularValues[0] * singularValues[1];
    }

    double _2u = u * 2;
    d2E_div_dsigma2(0, 0) = _2u + lambda * sigmaProd_noI[0] * sigmaProd_noI[0];
    d2E_div_dsigma2(1, 1) = _2u + lambda * sigmaProd_noI[1] * sigmaProd_noI[1];
    if constexpr (dim == 3) {
        d2E_div_dsigma2(2, 2) = _2u + lambda * sigmaProd_noI[2] * sigmaProd_noI[2];
    }

    if constexpr (dim == 2) {
        d2E_div_dsigma2(0, 1) = d2E_div_dsigma2(1, 0) = lambda * ((sigmaProd - 1.0) + sigmaProd_noI[0] * sigmaProd_noI[1]);
    }
    else {
        d2E_div_dsigma2(0, 1) = d2E_div_dsigma2(1, 0) = lambda * (singularValues[2] * (sigmaProd - 1.0) + sigmaProd_noI[0] * sigmaProd_noI[1]);
        d2E_div_dsigma2(0, 2) = d2E_div_dsigma2(2, 0) = lambda * (singularValues[1] * (sigmaProd - 1.0) + sigmaProd_noI[0] * sigmaProd_noI[2]);
        d2E_div_dsigma2(2, 1) = d2E_div_dsigma2(1, 2) = lambda * (singularValues[0] * (sigmaProd - 1.0) + sigmaProd_noI[2] * sigmaProd_noI[1]);
    }
}