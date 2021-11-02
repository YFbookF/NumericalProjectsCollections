//https://github.com/ipc-sim/IPC
template <int dim>
void Energy<dim>::getEnergyValPerElemBySVD(const Mesh<dim>& data, int redoSVD,
    std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
    std::vector<Eigen::Matrix<double, dim, dim>>& F,
    Eigen::VectorXd& energyValPerElem,
    bool uniformWeight) const
{
    energyValPerElem.resize(data.F.rows());
#ifdef USE_TBB
    tbb::parallel_for(0, (int)data.F.rows(), 1, [&](int triI) {
#else
    for (int triI = 0; triI < data.F.rows(); triI++) {
#endif
        if (redoSVD) {
            const Eigen::Matrix<int, 1, dim + 1>& triVInd = data.F.row(triI);
            Eigen::Matrix<double, dim, dim> Xt;
            Xt.col(0) = (data.V.row(triVInd[1]) - data.V.row(triVInd[0])).transpose();
            Xt.col(1) = (data.V.row(triVInd[2]) - data.V.row(triVInd[0])).transpose();
            if constexpr (dim == 3) {
                Xt.col(2) = (data.V.row(triVInd[3]) - data.V.row(triVInd[0])).transpose();
            }
            F[triI] = Xt * data.restTriInv[triI];

            svd[triI].compute(F[triI], Eigen::ComputeFullU | Eigen::ComputeFullV);
            // Eigen::Matrix2d F = Xt * A;
            // fprintf(out, "%le %le %le %le\n", F(0, 0), F(0, 1), F(1, 0), F(1, 1));
        }

        compute_E(svd[triI].singularValues(), data.u[triI], data.lambda[triI], energyValPerElem[triI]);
        if (!uniformWeight) {
            energyValPerElem[triI] *= data.triArea[triI];
        }
    }
#ifdef USE_TBB
    );
#endif
}

template <int dim>
void Energy<dim>::computeEnergyValBySVD(const Mesh<dim>& data, int redoSVD,
    std::vector<AutoFlipSVD<Eigen::Matrix<double, dim, dim>>>& svd,
    std::vector<Eigen::Matrix<double, dim, dim>>& F,
    double coef,
    double& energyVal) const
{
    Eigen::VectorXd energyValPerElem;
    getEnergyValPerElemBySVD(data, redoSVD, svd, F, energyValPerElem);
    energyVal = coef * energyValPerElem.sum();
}











