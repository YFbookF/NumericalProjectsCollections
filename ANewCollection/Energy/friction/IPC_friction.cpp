//https://github.com/ipc-sim/IPC
template <int dim>
void HalfSpace<dim>::computeFrictionEnergy(const Eigen::MatrixXd& V,
    const Eigen::MatrixXd& Vt, const std::vector<int>& activeSet,
    const Eigen::VectorXd& multipliers,
    double& Ef, double eps2, double coef) const
{
    assert(multipliers.size() == activeSet.size());
    double eps = std::sqrt(eps2);

    if constexpr (dim == 3) {
        //TODO: parallelize
        Ef = 0.0;
        int contactPairI = 0;
        for (const auto& vI : activeSet) {
            Eigen::Matrix<double, dim, 1> VDiff = (V.row(vI) - Vt.row(vI)).transpose();
            VDiff -= Base::velocitydt;
            Eigen::Matrix<double, dim, 1> VProj = VDiff - VDiff.dot(normal) * normal;
            double VProjMag2 = VProj.squaredNorm();
            if (VProjMag2 > eps2) {
                Ef += Base::friction * multipliers[contactPairI] * (std::sqrt(VProjMag2) - eps * 0.5);
            }
            else {
                Ef += Base::friction * multipliers[contactPairI] * VProjMag2 / eps * 0.5;
            }
            ++contactPairI;
        }
        Ef *= coef;
    }
}
template <int dim>
void HalfSpace<dim>::augmentFrictionGradient(const Eigen::MatrixXd& V,
    const Eigen::MatrixXd& Vt, const std::vector<int>& activeSet,
    const Eigen::VectorXd& multipliers,
    Eigen::VectorXd& grad_inc, double eps2, double coef) const
{
    assert(multipliers.size() == activeSet.size());
    double eps = std::sqrt(eps2);

    if constexpr (dim == 3) {
        //TODO: parallelize
        int contactPairI = 0;
        for (const auto& vI : activeSet) {
            Eigen::Matrix<double, dim, 1> VDiff = (V.row(vI) - Vt.row(vI)).transpose();
            VDiff -= Base::velocitydt;
            Eigen::Matrix<double, dim, 1> VProj = VDiff - VDiff.dot(normal) * normal;
            double VProjMag2 = VProj.squaredNorm();
            if (VProjMag2 > eps2) {
                grad_inc.template segment<dim>(vI * dim) += coef * Base::friction * multipliers[contactPairI] / std::sqrt(VProjMag2) * VProj;
            }
            else {
                grad_inc.template segment<dim>(vI * dim) += coef * Base::friction * multipliers[contactPairI] / eps * VProj;
            }
            ++contactPairI;
        }
    }
}
template <int dim>
void HalfSpace<dim>::augmentFrictionHessian(const Mesh<dim>& mesh,
    const Eigen::MatrixXd& Vt, const std::vector<int>& activeSet,
    const Eigen::VectorXd& multipliers,
    LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* H_inc,
    double eps2, double coef, bool projectDBC) const
{
    assert(multipliers.size() == activeSet.size());
    double eps = std::sqrt(eps2);

    //TODO: parallelize
    int contactPairI = 0;
    for (const auto& vI : activeSet) {
        if (projectDBC && mesh.isDBCVertex(vI)) {
            continue;
        }

        double multiplier_vI = coef * Base::friction * multipliers[contactPairI];
        Eigen::Matrix<double, dim, dim> H_vI;

        Eigen::Matrix<double, dim, 1> VDiff = (mesh.V.row(vI) - Vt.row(vI)).transpose();
        VDiff -= Base::velocitydt;
        Eigen::Matrix<double, dim, 1> VProj = VDiff - VDiff.dot(normal) * normal;
        double VProjMag2 = VProj.squaredNorm();
        if (VProjMag2 > eps2) {
            double VProjMag = std::sqrt(VProjMag2);

            H_vI = (VProj * (-multiplier_vI / VProjMag2 / VProjMag)) * VProj.transpose();
            H_vI += (Eigen::Matrix<double, dim, dim>::Identity() - normal * normal.transpose()) * (multiplier_vI / VProjMag);

            IglUtils::makePD(H_vI);
        }
        else {
            H_vI = (Eigen::Matrix<double, dim, dim>::Identity() - normal * normal.transpose()) * (multiplier_vI / eps);
            // already SPD
        }

        int startInd = vI * dim;
        H_inc->addCoeff(startInd, startInd, H_vI(0, 0));
        H_inc->addCoeff(startInd, startInd + 1, H_vI(0, 1));
        H_inc->addCoeff(startInd + 1, startInd, H_vI(1, 0));
        H_inc->addCoeff(startInd + 1, startInd + 1, H_vI(1, 1));
        if constexpr (dim == 3) {
            H_inc->addCoeff(startInd, startInd + 2, H_vI(0, 2));
            H_inc->addCoeff(startInd + 1, startInd + 2, H_vI(1, 2));

            H_inc->addCoeff(startInd + 2, startInd, H_vI(2, 0));
            H_inc->addCoeff(startInd + 2, startInd + 1, H_vI(2, 1));
            H_inc->addCoeff(startInd + 2, startInd + 2, H_vI(2, 2));
        }

        ++contactPairI;
    }
}