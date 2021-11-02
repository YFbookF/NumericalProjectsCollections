//https://github.com/ltt1598/Quasi-Newton-Methods-for-Real-time-Simulation-of-Hyperelastic-Materials
ScalarType TetConstraint::getStressTensorAndEnergyDensity(EigenMatrix3& P, const EigenMatrix3& F, EigenMatrix3& R)
{
	ScalarType e_this = 0;
	switch (m_material_type)
	{
	case MATERIAL_TYPE_COROT:
	{
		EigenMatrix3 U;
		EigenMatrix3 V;
		EigenVector3 SIGMA;
		singularValueDecomp(U, SIGMA, V, F);

		R = U*V.transpose();

		P = 2 * m_mu * (F - R) + m_lambda * ((R.transpose()*F).trace() - 3) * R;

		e_this = m_mu*(F - R).squaredNorm() + 0.5*m_lambda*std::pow((R.transpose()*F).trace() - 3, 2);
	}
	break;
	case MATERIAL_TYPE_StVK:
	{
		EigenMatrix3 I = EigenMatrix3::Identity();
		EigenMatrix3 E = 0.5*(F.transpose()*F - I);
		P = F * (2 * m_mu*E + m_lambda*E.trace() * I);
		e_this = m_mu*E.squaredNorm() + 0.5*m_lambda*std::pow(E.trace(), 2);
		ScalarType J = F.determinant();
		if (J < 1)
		{
			P += -m_kappa / 24 * std::pow((1 - J) / 6, 2) * J * F.inverse().transpose();
			e_this += m_kappa / 12 * std::pow((1 - J) / 6, 3);
		}
	}
	break;
	case MATERIAL_TYPE_NEOHOOKEAN_EXTEND_LOG:
	{
		EigenMatrix3 FtF = F.transpose() * F;
		ScalarType I1 = FtF.trace();
		ScalarType J = F.determinant();

		P = m_mu*F;
		e_this = 0.5*m_mu*(I1 - 3);
		ScalarType logJ;
		const ScalarType& J0 = m_neohookean_clamp_value;
		if (J > J0)
		{
			logJ = std::log(J);
			EigenMatrix3 FinvT = F.inverse().transpose(); // F is invertible because J > 0
			P += m_mu*(-FinvT) + m_lambda*logJ*FinvT;
			e_this += -m_mu*logJ + 0.5*m_lambda*logJ*logJ;
		}
		else
		{
#ifdef LOGJ_QUADRATIC_EXTENSION
			ScalarType fJ = log(J0) + (J - J0) / J0 - 0.5*std::pow((J / J0 - 1), 2);
			ScalarType dfJdJ = 1.0 / J0 - (J - J0) / (J0*J0);
#else
			ScalarType fJ = log(J0) + (J - J0) / J0;
			ScalarType dfJdJ = 1.0 / J0;
#endif
			//ScalarType fJ = log(J);
			//ScalarType dfJdJ = 1 / J;
			EigenMatrix3 FinvT = F.inverse().transpose(); // TODO: here F is nolonger guaranteed to be invertible....
			P += -m_mu * dfJdJ * J * FinvT + m_lambda* fJ * dfJdJ * J * FinvT;
			e_this += -m_mu*fJ + 0.5*m_lambda*fJ*fJ;
		}
	}
	break;
	default:
		break;
	}

	return e_this;
}
void TetConstraint::getStressTensor(EigenMatrix3& P, const EigenMatrix3& F, EigenMatrix3& R)
{
	switch (m_material_type)
	{
	case MATERIAL_TYPE_COROT:
	{
		EigenMatrix3 U;
		EigenMatrix3 V;
		EigenVector3 SIGMA;
		singularValueDecomp(U, SIGMA, V, F);

		R = U*V.transpose();

		P = 2 * m_mu * (F - R) + m_lambda * ((R.transpose()*F).trace() - 3) * R;
	}
	break;
	case MATERIAL_TYPE_StVK:
	{
		EigenMatrix3 I = EigenMatrix3::Identity();
		EigenMatrix3 E = 0.5*(F.transpose()*F - I);
		P = F * (2 * m_mu*E + m_lambda*E.trace() * I);
		ScalarType J = F.determinant();
		if (J < 1)
		{
			P += -m_kappa / 24 * std::pow((1 - J) / 6, 2) * J * F.inverse().transpose();
		}
	}
	break;
	case MATERIAL_TYPE_NEOHOOKEAN_EXTEND_LOG:
	{
		ScalarType J = F.determinant();

		P = m_mu*F;
		ScalarType logJ;
		const ScalarType& J0 = m_neohookean_clamp_value;
		if (J > J0)
		{
			logJ = std::log(J);
			EigenMatrix3 FinvT = F.inverse().transpose(); // F is invertible because J > 0
			P += m_mu*(-FinvT) + m_lambda*logJ*FinvT;
		}
		else
		{
#ifdef LOGJ_QUADRATIC_EXTENSION
			ScalarType fJ = log(J0) + (J - J0) / J0 - 0.5*std::pow((J / J0 - 1), 2);
			ScalarType dfJdJ = 1.0 / J0 - (J - J0) / (J0*J0);
#else
			ScalarType fJ = log(J0) + (J - J0) / J0;
			ScalarType dfJdJ = 1.0 / J0;
#endif
			//ScalarType fJ = log(J);
			//ScalarType dfJdJ = 1 / J;
			EigenMatrix3 FinvT = F.inverse().transpose(); // TODO: here F is nolonger guaranteed to be invertible....
			P += -m_mu * dfJdJ * J * FinvT + m_lambda* fJ * dfJdJ * J * FinvT;
		}
	}
	break;
	default:
		break;
	}
}
void TetConstraint::calculateDPDF(Matrix3333& dPdF, const EigenMatrix3& F)
{
	Matrix3333 dFdF;
	dFdF.SetIdentity();
	switch (m_material_type)
	{
	case MATERIAL_TYPE_COROT:
	{
		EigenMatrix3 U;
		EigenMatrix3 V;
		EigenVector3 SIGMA;
		singularValueDecomp(U, SIGMA, V, F);

		EigenMatrix3 R = U*V.transpose();

		Matrix3333 dRdF;

		// to compute dRdF using derivative of SVD
		for (unsigned int i = 0; i < 3; i++)
		{
			for (unsigned int j = 0; j < 3; j++)
			{
				EigenMatrix3 deltaF = dFdF(i, j);

				EigenMatrix3 A = U.transpose() * deltaF * V;

				// special case for uniform scaling including restpose, restpose is a common pose
				if (abs(SIGMA(0) - SIGMA(1)) < LARGER_EPSILON && abs(SIGMA(0) - SIGMA(2)) < LARGER_EPSILON)
				{
					ScalarType alpha = SIGMA(0); // SIGMA = alpha * Identity
					if (alpha < LARGER_EPSILON) // alpha should be greater than zero because it comes from the ordered SVD
					{
						alpha = LARGER_EPSILON;
					}

					EigenMatrix3 off_diag_A;
					off_diag_A.setZero();
					for (unsigned int row = 0; row != 3; row++)
					{
						for (unsigned int col = 0; col != 3; col++)
						{
							// assign off diagonal values of U^T * delta*F * V
							if (row == col)
								continue;
							else
							{
								off_diag_A(row, col) = A(row, col) / alpha;
							}
						}
					}
					dRdF(i, j) = U*off_diag_A*V.transpose();
				}
				// otherwise TODO: should also discuss the case where 2 sigular values are the same, but since it's very rare, we are gonna just treat it using regularization
				else
				{
					// there are 9 unkown variables, u10, u20, u21, v10, v20, v21, sig00, sig11, sig22
					EigenVector2 unknown_side, known_side;
					EigenMatrix2 known_matrix;
					EigenMatrix3 U_tilde, V_tilde;
					U_tilde.setZero(); V_tilde.setZero();
					EigenMatrix2 reg;
					reg.setZero();
					reg(0, 0) = reg(1, 1) = LARGER_EPSILON;
					for (unsigned int row = 0; row < 3; row++)
					{
						for (unsigned int col = 0; col < row; col++)
						{
							known_side = EigenVector2(A(col, row), A(row, col));
							known_matrix.block<2, 1>(0, 0) = EigenVector2(-SIGMA[row], SIGMA[col]);
							known_matrix.block<2, 1>(0, 1) = EigenVector2(-SIGMA[col], SIGMA[row]);

							if (std::abs(SIGMA[row] - SIGMA[col]) < LARGER_EPSILON) //regularization
							{
								//throw std::exception("Ill-conditioned hessian matrix, using FD hessian.");
								known_matrix += reg;
							}
							else
							{
								assert(std::abs(known_matrix.determinant()) > 1e-6);
							}

							unknown_side = known_matrix.inverse() * known_side;
							EigenVector2 test_vector = known_matrix*unknown_side;
							U_tilde(row, col) = unknown_side[0];
							U_tilde(col, row) = -U_tilde(row, col);
							V_tilde(row, col) = unknown_side[1];
							V_tilde(col, row) = -V_tilde(row, col);
						}
					}
					EigenMatrix3 deltaU = U*U_tilde;
					EigenMatrix3 deltaV = V_tilde*V.transpose();

					dRdF(i, j) = deltaU*V.transpose() + U*deltaV;
				}
			}
		}

		dPdF = 2 * m_mu* (dFdF - dRdF); // first mu term

		Matrix3333 dlambda_term_dR;
		Matrix3333 dlambda_term_dRF;
		Matrix3333 R_kron_R;
		directProduct(R_kron_R, R, R);
		Matrix3333 F_kron_R;
		directProduct(F_kron_R, F, R);
		dlambda_term_dR = F_kron_R + ((R.transpose()*F).trace() - 3) * dFdF/*dFdF = Identity*/;
		dlambda_term_dRF = dlambda_term_dR.Contract(dRdF);

		dPdF = dPdF + (R_kron_R + dlambda_term_dRF) * m_lambda;
	}
	break;
	case MATERIAL_TYPE_StVK:
	{
		EigenMatrix3 E = 0.5 * (F.transpose()*F - EigenMatrix3::Identity());
		EigenMatrix3 deltaE;
		EigenMatrix3 deltaF;
		ScalarType J = F.determinant();
		EigenMatrix3 Finv = F.inverse();
		EigenMatrix3 FinvT = Finv.transpose();

		for (unsigned int i = 0; i < 3; i++)
		{
			for (unsigned int j = 0; j < 3; j++)
			{
				deltaF = dFdF(i, j);
				deltaE = 0.5 * (deltaF.transpose() * F + F.transpose() * deltaF);

				dPdF(i, j) = deltaF * (2 * m_mu * E + m_lambda * E.trace() * EigenMatrix3::Identity()) + F * (2 * m_mu * deltaE + m_lambda*deltaE.trace()*EigenMatrix3::Identity());
				if (J < 1)
				{
					ScalarType one_minus_J_over_six = (1 - J) / 6.0;
					ScalarType one_minus_J_over_six_square = one_minus_J_over_six*one_minus_J_over_six;

					dPdF(i, j) += -m_kappa / 24 * ((-one_minus_J_over_six*J / 3 + one_minus_J_over_six_square)*J*(Finv*deltaF).trace()*FinvT - one_minus_J_over_six_square*J*FinvT*deltaF.transpose()*FinvT);
				}
			}
		}
	}
	break;
	case MATERIAL_TYPE_NEOHOOKEAN_EXTEND_LOG:
	{
		EigenMatrix3 deltaF;
		ScalarType J = F.determinant();
		EigenMatrix3 Finv = F.inverse();  // assert J != 0;
		EigenMatrix3 FinvT = Finv.transpose();
		ScalarType logJ;

		for (unsigned int i = 0; i < 3; i++)
		{
			for (unsigned int j = 0; j < 3; j++)
			{
				deltaF = dFdF(i, j);

				dPdF(i, j) = m_mu * deltaF;

				const ScalarType& J0 = m_neohookean_clamp_value;
				if (J > J0)
				{
					logJ = std::log(J);

					dPdF(i, j) += (m_mu - m_lambda*logJ)*FinvT*deltaF.transpose()*FinvT + m_lambda*(Finv*deltaF).trace() * FinvT;
				}
				else
				{
#ifdef LOGJ_QUADRATIC_EXTENSION
					// quadratic
					ScalarType fJ = log(J0) + (J - J0) / J0 - 0.5*std::pow((J / J0 - 1), 2);
					ScalarType dfJdJ = 1.0 / J0 - (J - J0) / (J0*J0);
					ScalarType d2fJdJ2 = -1.0 / (J0*J0);
#else
					ScalarType fJ = log(J0) + (J - J0) / J0;
					ScalarType dfJdJ = 1.0 / J0;
					ScalarType d2fJdJ2 = 0;
#endif
					EigenMatrix3 FinvTdFTFinvT = FinvT*deltaF.transpose()*FinvT;
					EigenMatrix3 FinvdFtraceFinvT = (Finv*deltaF).trace() * FinvT;

					dPdF(i, j) += -m_mu * (d2fJdJ2*J + dfJdJ) * J * FinvdFtraceFinvT;
					dPdF(i, j) += m_mu * (dfJdJ*J) * FinvTdFTFinvT;
					dPdF(i, j) += m_lambda * (dfJdJ*dfJdJ*J + fJ*(d2fJdJ2*J + dfJdJ)) * J * FinvdFtraceFinvT;
					dPdF(i, j) += -m_lambda * (fJ*dfJdJ*J) * FinvTdFTFinvT;
				}
			}
		}
	}
	break;
	}
}
ScalarType TetConstraint::ComputeLaplacianWeight()
{
	// read section 4.1 in our paper 
	switch (m_material_type)
	{
	case MATERIAL_TYPE_COROT:
		// 2mu (x-1) + lambda (x-1)
		m_laplacian_coeff = 2 * m_mu + m_lambda;
		break;
	case MATERIAL_TYPE_StVK:
		// mu * (x^2  - 1) + 0.5lambda * (x^3 - x)
		// 10% window
		m_laplacian_coeff = 2 * m_mu + 1.0033 * m_lambda;
		//// 20% window
		//m_laplacian_coeff = 2 * m_mu + 1.0126 * m_lambda;
		break;
	case MATERIAL_TYPE_NEOHOOKEAN_EXTEND_LOG:
		// mu * x - mu / x + lambda * log(x) / x
		// 10% window
		m_laplacian_coeff = 2.0066 * m_mu + 1.0122 * m_lambda;
		//// 20% window
		//m_laplacian_coeff = 2.0260 * m_mu + 1.0480 * m_lambda;
		break;
	default:
		break;
	}
	return m_laplacian_coeff;
}

// 0.5*mu*||F-R||^2
ScalarType TetConstraint::EvaluateEnergy(const VectorX& x)
{
	EigenMatrix3 F;
	getDeformationGradient(F, x);

	ScalarType e_this = 0;
	switch (m_material_type)
	{
	case MATERIAL_TYPE_COROT:
	{
		EigenMatrix3 U;
		EigenMatrix3 V;
		EigenVector3 SIGMA;
		singularValueDecomp(U, SIGMA, V, F);

		EigenMatrix3 R = U*V.transpose();

		e_this = m_mu*(F - R).squaredNorm() + 0.5*m_lambda*std::pow((R.transpose()*F).trace() - 3, 2);
	}
	break;
	case MATERIAL_TYPE_StVK:
	{
		EigenMatrix3 I = EigenMatrix3::Identity();
		EigenMatrix3 E = 0.5*(F.transpose()*F - I);
		e_this = m_mu*E.squaredNorm() + 0.5*m_lambda*std::pow(E.trace(), 2);
		ScalarType J = F.determinant();
		if (J < 1)
		{
			e_this += m_kappa / 12 * std::pow((1 - J) / 6, 3);
		}
	}
	break;
	case MATERIAL_TYPE_NEOHOOKEAN_EXTEND_LOG:
	{
		EigenMatrix3 FtF = F.transpose() * F;
		ScalarType I1 = FtF.trace();
		ScalarType J = F.determinant();
		e_this = 0.5*m_mu*(I1 - 3);
		ScalarType logJ;
		const ScalarType& J0 = m_neohookean_clamp_value;
		if (J > J0)
		{
			logJ = std::log(J);
			e_this += -m_mu*logJ + 0.5*m_lambda*logJ*logJ;
		}
		else
		{
#ifdef LOGJ_QUADRATIC_EXTENSION
			ScalarType fJ = log(J0) + (J - J0) / J0 - 0.5*std::pow((J / J0 - 1), 2);
#else
			ScalarType fJ = log(J0) + (J - J0) / J0;
#endif
			e_this += -m_mu*fJ + 0.5*m_lambda*fJ*fJ;
		}
	}
	break;
	}

	e_this *= m_W;

	m_energy = e_this;

	return e_this;
}
