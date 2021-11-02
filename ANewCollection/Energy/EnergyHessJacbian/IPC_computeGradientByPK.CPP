//https://github.com/ipc-sim/IPC
template <int dim>
void Energy<dim>::computeGradientByPK(const Mesh<dim>& data,
    int elemI, bool redoSVD,
    AutoFlipSVD<Eigen::Matrix<double, dim, dim>>& svd,
    Eigen::Matrix<double, dim, dim>& F,
    double coef,
    Eigen::Matrix<double, dim*(dim + 1), 1>& gradient) const
{
    const Eigen::Matrix<double, dim, dim>& A = data.restTriInv[elemI];

    if (redoSVD) {
        const Eigen::Matrix<int, 1, dim + 1>& triVInd = data.F.row(elemI);

        Eigen::Matrix<double, dim, dim> Xt;
        Xt.col(0) = (data.V.row(triVInd[1]) - data.V.row(triVInd[0])).transpose();
        Xt.col(1) = (data.V.row(triVInd[2]) - data.V.row(triVInd[0])).transpose();
        if constexpr (dim == 3) {
            Xt.col(2) = (data.V.row(triVInd[3]) - data.V.row(triVInd[0])).transpose();
        }

        F = Xt * A;

        svd.compute(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    }

    Eigen::Matrix<double, dim, dim> P;
    compute_dE_div_dF(F, svd,
        data.u[elemI], data.lambda[elemI], P);

    const double w = coef * data.triArea[elemI];
    P *= w;

    IglUtils::dF_div_dx_mult(P, A, gradient);
}
template <int dim>
void Energy<dim>::computeHessianByPK(const Mesh<dim>& data,
    int elemI, bool redoSVD,
    AutoFlipSVD<Eigen::Matrix<double, dim, dim>>& svd,
    Eigen::Matrix<double, dim, dim>& F,
    double coef,
    Eigen::Matrix<double, dim*(dim + 1), dim*(dim + 1)>& hessian,
    Eigen::Matrix<int, 1, dim + 1>& vInd,
    bool projectSPD,
    bool projectDBC) const
{
    const Eigen::Matrix<int, 1, dim + 1>& triVInd = data.F.row(elemI);

    const Eigen::Matrix<double, dim, dim>& A = data.restTriInv[elemI];

    if (redoSVD) {
        Eigen::Matrix<double, dim, dim> Xt;
        Xt.col(0) = (data.V.row(triVInd[1]) - data.V.row(triVInd[0])).transpose();
        Xt.col(1) = (data.V.row(triVInd[2]) - data.V.row(triVInd[0])).transpose();
        if constexpr (dim == 3) {
            Xt.col(2) = (data.V.row(triVInd[3]) - data.V.row(triVInd[0])).transpose();
        }
        F = Xt * A;
        svd.compute(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    }

    Eigen::Matrix<double, dim * dim, dim * dim> wdP_div_dF;
    const double w = coef * data.triArea[elemI];
    compute_dP_div_dF(svd, data.u[elemI], data.lambda[elemI],
        wdP_div_dF, w, projectSPD);

    Eigen::Matrix<double, dim*(dim + 1), dim * dim> wdP_div_dx;
    IglUtils::dF_div_dx_mult<dim * dim>(wdP_div_dF.transpose(), A, wdP_div_dx, false);
    IglUtils::dF_div_dx_mult<dim*(dim + 1)>(wdP_div_dx.transpose(), A, hessian, true);

    vInd[0] = data.isProjectDBCVertex(triVInd[0], projectDBC) ? (-triVInd[0] - 1) : triVInd[0];
    vInd[1] = data.isProjectDBCVertex(triVInd[1], projectDBC) ? (-triVInd[1] - 1) : triVInd[1];
    vInd[2] = data.isProjectDBCVertex(triVInd[2], projectDBC) ? (-triVInd[2] - 1) : triVInd[2];
    if constexpr (dim == 3) {
        vInd[3] = data.isProjectDBCVertex(triVInd[3], projectDBC) ? (-triVInd[3] - 1) : triVInd[3];
    }
}