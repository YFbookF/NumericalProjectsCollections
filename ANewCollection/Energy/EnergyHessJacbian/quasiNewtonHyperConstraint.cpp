//https://github.com/ltt1598/Quasi-Newton-Methods-for-Real-time-Simulation-of-Hyperelastic-Materials
// 0.5*k*(current_length)^2
ScalarType AttachmentConstraint::EvaluateEnergy(const VectorX& x)
{
	ScalarType e_i = 0.5*(m_stiffness)*(x.block_vector(m_p0) - m_fixd_point).squaredNorm();
	m_energy = e_i;

	return e_i;
}

ScalarType AttachmentConstraint::GetEnergy()
{
	return m_energy;
}

// attachment spring gradient: k*(current_length)*current_direction
void AttachmentConstraint::EvaluateGradient(const VectorX& x, VectorX& gradient)
{
	EigenVector3 g_i = (m_stiffness)*(x.block_vector(m_p0) - m_fixd_point);
	gradient.block_vector(m_p0) += g_i;
}
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
