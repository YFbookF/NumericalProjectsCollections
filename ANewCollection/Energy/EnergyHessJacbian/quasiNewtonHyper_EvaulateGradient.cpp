//constraint_attachment.cpp
//
// attachment spring gradient: k*(current_length)*current_direction
void AttachmentConstraint::EvaluateGradient(const VectorX& x, VectorX& gradient)
{
	EigenVector3 g_i = (m_stiffness)*(x.block_vector(m_p0) - m_fixd_point);
	gradient.block_vector(m_p0) += g_i;
}

void AttachmentConstraint::EvaluateGradient(const VectorX& x)
{
	m_g = (m_stiffness)*(x.block_vector(m_p0) - m_fixd_point);
}
//constraint_penalty.cpp
// attachment spring gradient: k*(current_length)*current_direction
void CollisionSpringConstraint::EvaluateGradient(const VectorX& x, VectorX& gradient)
{
	if (IsActive(x))
	{
		EigenVector3 g_i = (m_stiffness)*(x.block_vector(m_p0) - m_fixed_point);
		gradient.block_vector(m_p0) += g_i;
	}
}

void CollisionSpringConstraint::EvaluateGradient(const VectorX& x)
{
	if (IsActive(x))
	{
		m_g = (m_stiffness)*(x.block_vector(m_p0) - m_fixed_point);
	}
	else
	{
		m_g.setZero();
	}
}
//constraint.spring.cpp
// sping gradient: k*(current_length-rest_length)*current_direction;
void SpringConstraint::EvaluateGradient(const VectorX& x, VectorX& gradient)
{
	EigenVector3 x_ij = x.block_vector(m_p1) - x.block_vector(m_p2);
	EigenVector3 g_ij = (m_stiffness)*(x_ij.norm() - m_rest_length)*x_ij.normalized();
	gradient.block_vector(m_p1) += g_ij;
	gradient.block_vector(m_p2) -= g_ij;
}

void SpringConstraint::EvaluateGradient(const VectorX& x)
{
	EigenVector3 x_ij = x.block_vector(m_p1) - x.block_vector(m_p2);
	EigenVector3 g_ij = (m_stiffness)*(x_ij.norm() - m_rest_length)*x_ij.normalized();

	m_g1 = g_ij;
	m_g2 = -g_ij;
}
//constraint_tet.cpp
void TetConstraint::EvaluateGradient(const VectorX& x, VectorX& gradient)
{
	//VectorX gradient1 = gradient;
	//evaluateFDGradient(x, gradient1);

	//gradient = gradient1;
	//return;

	EigenMatrix3 F;
	getDeformationGradient(F, x);

	EigenMatrix3 P;
	EigenMatrix3 R;
	getStressTensor(P, F, R);

	EigenMatrix3 H = m_W * P * m_Dm_inv.transpose();

	EigenVector3 g[4];
	g[0] = H.block<3, 1>(0, 0);
	g[1] = H.block<3, 1>(0, 1);
	g[2] = H.block<3, 1>(0, 2);
	g[3] = -g[0] - g[1] - g[2];

	for (unsigned int i = 0; i < 4; i++)
	{
		gradient.block_vector(m_p[i]) += g[i];
	}
}

void TetConstraint::EvaluateGradient(const VectorX& x)
{
	EigenMatrix3 F;
	getDeformationGradient(F, x);

	EigenMatrix3 P;
	EigenMatrix3 R;
	getStressTensor(P, F, R);

	EigenMatrix3 H = m_W * P * m_Dm_inv.transpose();

	m_g[0] = H.block<3, 1>(0, 0);
	m_g[1] = H.block<3, 1>(0, 1);
	m_g[2] = H.block<3, 1>(0, 2);
	m_g[3] = -m_g[0] - m_g[1] - m_g[2];
}