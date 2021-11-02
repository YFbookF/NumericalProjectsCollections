////https://github.com/ltt1598/Quasi-Newton-Methods-for-Real-time-Simulation-of-Hyperelastic-Materials
ScalarType TetConstraint::EvaluateEnergyAndGradient(const VectorX& x)
{
	EigenMatrix3 F;
	getDeformationGradient(F, x);

	EigenMatrix3 P;
	EigenMatrix3 R;
	ScalarType e_this = getStressTensorAndEnergyDensity(P, F, R);
	m_energy = e_this * m_W;

	EigenMatrix3 H = m_W * P * m_Dm_inv.transpose();

	m_g[0] = H.block<3, 1>(0, 0);
	m_g[1] = H.block<3, 1>(0, 1);
	m_g[2] = H.block<3, 1>(0, 2);
	m_g[3] = -m_g[0] - m_g[1] - m_g[2];

	return m_energy;
}
ScalarType TetConstraint::GetEnergyAndGradient(VectorX& gradient)
{
	for (unsigned int i = 0; i < 4; i++)
	{
		gradient.block_vector(m_p[i]) += m_g[i];
	}
	return m_energy;
}

void TetConstraint::EvaluateHessian(const VectorX& x, bool definiteness_fix)
{
	EigenMatrix3 F;
	getDeformationGradient(F, x);

	Matrix3333 dPdF;

	calculateDPDF(dPdF, F);

	Matrix3333 dH;
	dH = m_W * dPdF * m_Dm_inv.transpose();

	//EigenMatrix3 H_blocks[4][4];
	EigenMatrix3 H_one_block;

	// i == 0 to 2 case 
	for (unsigned int i = 0; i < 3; i++)
	{
		// j == 0 to 2 case
		for (unsigned int j = 0; j < 3; j++)
		{
			EigenVector3 v0 = dH(0, j)*m_Dm_inv.transpose().block<3, 1>(0, i);
			EigenVector3 v1 = dH(1, j)*m_Dm_inv.transpose().block<3, 1>(0, i);
			EigenVector3 v2 = dH(2, j)*m_Dm_inv.transpose().block<3, 1>(0, i);

			ThreeVector3ToMatrix3(H_one_block, v0, v1, v2);
			m_H.block<3, 3>(i * 3, j * 3) = H_one_block;
		}
		// j == 3 case
		//H_blocks[i][3] = -H_blocks[i][0] - H_blocks[i][1] - H_blocks[i][2];
		m_H.block<3, 3>(i * 3, 9) = -m_H.block<3, 3>(i * 3, 0) - m_H.block<3, 3>(i * 3, 3) - m_H.block<3, 3>(i * 3, 6);
	}
	// i == 3 case
	for (unsigned int j = 0; j < 4; j++)
	{
		//H_blocks[3][j] = -H_blocks[0][j]-H_blocks[1][j]-H_blocks[2][j];
		m_H.block<3, 3>(9, j * 3) = -m_H.block<3, 3>(0, j * 3) - m_H.block<3, 3>(3, j * 3) - m_H.block<3, 3>(6, j * 3);
	}

	if (definiteness_fix)
	{
		// definiteness fix
		Eigen::EigenSolver<EigenMatrix12> evd;
		evd.compute(m_H);
		EigenMatrix12 Q = evd.eigenvectors().real();
		VectorX LAMBDA = evd.eigenvalues().real();
		//assert(LAMBDA(0) > 0);
		//ScalarType smallest_lambda = LAMBDA(0) * 1e-10;
		ScalarType smallest_lambda = 1e-6;
		for (unsigned int i = 0; i != LAMBDA.size(); i++)
		{
			//assert(LAMBDA(0) > LAMBDA(i));
			if (LAMBDA(i) < smallest_lambda)
			{
				LAMBDA(i) = smallest_lambda;
			}
		}
		m_H = Q * LAMBDA.asDiagonal() * Q.transpose();

		//// debug
		//evd.compute(m_H);
		//Q = evd.eigenvectors().real();
		//LAMBDA = evd.eigenvalues().real();	
	}
}