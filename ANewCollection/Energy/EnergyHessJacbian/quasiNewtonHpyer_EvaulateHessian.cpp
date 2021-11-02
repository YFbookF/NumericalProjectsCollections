//constraint_attachment.cpp
void AttachmentConstraint::EvaluateHessian(const VectorX& x, bool definiteness_fix /* = false */ /* = 1 */)
{
	ScalarType ks = m_stiffness;

	EigenVector3 H_d;
	for (unsigned int i = 0; i != 3; i++)
	{
		H_d(i) = ks;
	}
	m_H = H_d.asDiagonal();
}

void AttachmentConstraint::EvaluateHessian(const VectorX& x, std::vector<SparseMatrixTriplet>& hessian_triplets, bool definiteness_fix)
{
	EvaluateHessian(x, definiteness_fix);
	for (unsigned int i = 0; i != 3; i++)
	{
		hessian_triplets.push_back(SparseMatrixTriplet(3 * m_p0 + i, 3 * m_p0 + i, m_H(i, i)));
	}
}

void AttachmentConstraint::ApplyHessian(const VectorX& x, VectorX& b)
{
	b.block_vector(m_p0) += m_H * x.block_vector(m_p0);
}
//constraint_penalty.cpp
void CollisionSpringConstraint::EvaluateHessian(const VectorX& x, bool definiteness_fix /* = false */ /* = 1 */)
{
	if (IsActive(x))
	{
		ScalarType ks = m_stiffness;

		EigenVector3 H_d;
		for (unsigned int i = 0; i != 3; i++)
		{
			H_d(i) = ks;
		}
		m_H = H_d.asDiagonal();
	}
	else
	{
		m_H.setZero();
	}
}

void CollisionSpringConstraint::EvaluateHessian(const VectorX& x, std::vector<SparseMatrixTriplet>& hessian_triplets, bool definiteness_fix)
{
	EvaluateHessian(x, definiteness_fix);
	for (unsigned int i = 0; i != 3; i++)
	{
		hessian_triplets.push_back(SparseMatrixTriplet(3 * m_p0 + i, 3 * m_p0 + i, m_H(i, i)));
	}
}

void CollisionSpringConstraint::ApplyHessian(const VectorX& x, VectorX& b)
{
	b.block_vector(m_p0) += m_H * x.block_vector(m_p0);
}
//constraint_spring
void SpringConstraint::EvaluateHessian(const VectorX& x, bool definiteness_fix)
{
	EigenVector3 x_ij = x.block_vector(m_p1) - x.block_vector(m_p2);
	ScalarType l_ij = x_ij.norm();
	ScalarType l0 = m_rest_length;
	ScalarType ks = m_stiffness;

	m_H = ks * (EigenMatrix3::Identity() - l0 / l_ij*(EigenMatrix3::Identity() - (x_ij*x_ij.transpose()) / (l_ij*l_ij)));
	//EigenMatrix3 k = ks * (1-l0/l_ij) * EigenMatrix3::Identity();

	if (definiteness_fix)
	{
		// definiteness fix
		Eigen::EigenSolver<EigenMatrix3> evd;
		evd.compute(m_H);
		EigenMatrix3 Q = evd.eigenvectors().real();
		EigenVector3 LAMBDA = evd.eigenvalues().real();
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
	}
}

void SpringConstraint::EvaluateHessian(const VectorX& x, std::vector<SparseMatrixTriplet>& hessian_triplets, bool definiteness_fix)
{
	EvaluateHessian(x, definiteness_fix);

	for (int row = 0; row < 3; row++)
	{
		for (int col = 0; col < 3; col++)
		{
			ScalarType val = m_H(row, col);
			//Update the global hessian matrix
			hessian_triplets.push_back(SparseMatrixTriplet(3 * m_p1 + row, 3 * m_p1 + col, val));
			hessian_triplets.push_back(SparseMatrixTriplet(3 * m_p1 + row, 3 * m_p2 + col, -val));
			hessian_triplets.push_back(SparseMatrixTriplet(3 * m_p2 + row, 3 * m_p1 + col, -val));
			hessian_triplets.push_back(SparseMatrixTriplet(3 * m_p2 + row, 3 * m_p2 + col, val));
		}
	}
}

void SpringConstraint::ApplyHessian(const VectorX& x, VectorX& b)
{
	EigenVector3 p1 = x.block_vector(m_p1);
	EigenVector3 p2 = x.block_vector(m_p2);
	EigenVector3 d = m_H * p1 - m_H * p2;
	b.block_vector(m_p1) += d;
	b.block_vector(m_p2) -= d;
}
//tet
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

void TetConstraint::EvaluateHessian(const VectorX& x, std::vector<SparseMatrixTriplet>& hessian_triplets, bool definiteness_fix)
{
	//// finite difference test
	//std::vector<SparseMatrixTriplet> hessian_triplets_copy = hessian_triplets;
	//EvaluateFiniteDifferentHessian(x, hessian_triplets_copy);

	EvaluateHessian(x, definiteness_fix);

	// set to triplets
	for (unsigned int i = 0; i < 4; i++)
	{
		for (unsigned int j = 0; j < 4; j++)
		{
			//EigenMatrix3 block = H_blocks.block<3, 3>(i * 3, j * 3);
			for (unsigned int row = 0; row < 3; row++)
			{
				for (unsigned int col = 0; col < 3; col++)
				{
					hessian_triplets.push_back(SparseMatrixTriplet(m_p[i] * 3 + row, m_p[j] * 3 + col, m_H(i * 3 + row, j * 3 + col)));
				}
			}
		}
	}
}

void TetConstraint::ApplyHessian(const VectorX& x, VectorX& b)
{
	EigenVector12 x_blocks;
	for (unsigned int i = 0; i != 4; i++)
	{
		x_blocks.block_vector(i) = x.block_vector(m_p[i]);
	}
	EigenVector12 b_blocks = m_H * x_blocks;
	for (unsigned int i = 0; i != 4; i++)
	{
		b.block_vector(m_p[i]) += b_blocks.block_vector(i);
	}
}