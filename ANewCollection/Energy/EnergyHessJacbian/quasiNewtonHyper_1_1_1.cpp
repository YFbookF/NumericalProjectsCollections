//https://github.com/ltt1598/Quasi-Newton-Methods-for-Real-time-Simulation-of-Hyperelastic-Materials
void TetConstraint::EvaluateWeightedLaplacian(std::vector<SparseMatrixTriplet>& laplacian_triplets)
{
	ScalarType ks = m_laplacian_coeff* m_W;

	Matrix3333 Identity_tensor;
	Identity_tensor.SetIdentity();

	Matrix3333 dH;
	dH = ks * Identity_tensor * m_Dm_inv.transpose();

	EigenMatrix3 H_blocks[4][4];

	// i == 0 to 2 case 
	for (unsigned int i = 0; i < 3; i++)
	{
		// j == 0 to 2 case
		for (unsigned int j = 0; j < 3; j++)
		{
			EigenVector3 v0 = dH(0, j)*m_Dm_inv.transpose().block<3, 1>(0, i);
			EigenVector3 v1 = dH(1, j)*m_Dm_inv.transpose().block<3, 1>(0, i);
			EigenVector3 v2 = dH(2, j)*m_Dm_inv.transpose().block<3, 1>(0, i);

			ThreeVector3ToMatrix3(H_blocks[i][j], v0, v1, v2);
		}
		// j == 3 case
		H_blocks[i][3] = -H_blocks[i][0] - H_blocks[i][1] - H_blocks[i][2];
	}
	// i == 3 case
	for (unsigned int j = 0; j < 4; j++)
	{
		H_blocks[3][j] = -H_blocks[0][j] - H_blocks[1][j] - H_blocks[2][j];
	}

	// set to triplets
	for (unsigned int i = 0; i < 4; i++)
	{
		for (unsigned int j = 0; j < 4; j++)
		{
			EigenMatrix3& block = H_blocks[i][j];
			for (unsigned int row = 0; row < 3; row++)
			{
				for (unsigned int col = 0; col < 3; col++)
				{
					laplacian_triplets.push_back(SparseMatrixTriplet(m_p[i] * 3 + row, m_p[j] * 3 + col, block(row, col)));
				}
			}
		}
	}
}