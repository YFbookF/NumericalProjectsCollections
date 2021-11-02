//https://github.com/ipc-sim/IPC
// 注意这是coratation的
template <int dim>
void FixedCoRotEnergy<dim>::compute_dE_div_dF(const Eigen::Matrix<double, dim, dim>& F,
    const AutoFlipSVD<Eigen::Matrix<double, dim, dim>>& svd,
    double u, double lambda,
    Eigen::Matrix<double, dim, dim>& dE_div_dF) const
{
    Eigen::Matrix<double, dim, dim> JFInvT;
    IglUtils::computeCofactorMtr(F, JFInvT);
    dE_div_dF = (u * 2 * (F - svd.matrixU() * svd.matrixV().transpose()) + lambda * (svd.singularValues().prod() - 1) * JFInvT);
}