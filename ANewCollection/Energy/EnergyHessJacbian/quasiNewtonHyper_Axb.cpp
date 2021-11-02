    //hessian descent_dir = gradient
	// A          x       =    b
	linearSolve(descent_dir, hessian, gradient);
	
	void Simulation::evaluateGradient(const VectorX& x, VectorX& gradient, bool enable_omp)
{
	ScalarType h_square = m_h*m_h;
	switch (m_integration_method)
	{
	case INTEGRATION_QUASI_STATICS:
		evaluateGradientPureConstraint(x, m_external_force, gradient);
		gradient -= m_external_force;
		break;//DO NOTHING
	case INTEGRATION_IMPLICIT_EULER:
		evaluateGradientPureConstraint(x, m_external_force, gradient);
		gradient = m_mesh->m_mass_matrix * (x - m_y) + h_square*gradient;
		break;
	case INTEGRATION_IMPLICIT_BDF2:
		evaluateGradientPureConstraint(x, m_external_force, gradient);
		gradient = m_mesh->m_mass_matrix * (x - m_y) + (h_square*4.0 / 9.0)*gradient;
		break;
	case INTEGRATION_IMPLICIT_MIDPOINT:
		evaluateGradientPureConstraint((x + m_mesh->m_current_positions) / 2, m_external_force, gradient);
		gradient = m_mesh->m_mass_matrix * (x - m_y) + h_square / 2 * (gradient);
		break;
	case INTEGRATION_IMPLICIT_NEWMARK_BETA:
		evaluateGradientPureConstraint(x, m_external_force, gradient);
		gradient = m_mesh->m_mass_matrix * (x - m_y) + h_square / 4 * (gradient + m_z);
		break;
	}
}
void Simulation::evaluateHessian(const VectorX& x, SparseMatrix& hessian_matrix)
{
	ScalarType h_square = m_h*m_h;
	switch (m_integration_method)
	{
	case INTEGRATION_QUASI_STATICS:
		evaluateHessianPureConstraint(x, hessian_matrix);
		break;//DO NOTHING
	case INTEGRATION_IMPLICIT_EULER:
		evaluateHessianPureConstraint(x, hessian_matrix);
		hessian_matrix = m_mesh->m_mass_matrix + h_square*hessian_matrix;
		break;
	case INTEGRATION_IMPLICIT_BDF2:
		evaluateHessianPureConstraint(x, hessian_matrix);
		hessian_matrix = m_mesh->m_mass_matrix + h_square*4.0 / 9.0*hessian_matrix;
		break;
	case INTEGRATION_IMPLICIT_MIDPOINT:
		evaluateHessianPureConstraint((x + m_mesh->m_current_positions) / 2, hessian_matrix);
		hessian_matrix = m_mesh->m_mass_matrix + h_square / 4 * hessian_matrix;
		break;
	case INTEGRATION_IMPLICIT_NEWMARK_BETA:
		evaluateHessianPureConstraint(x, hessian_matrix);
		hessian_matrix = m_mesh->m_mass_matrix + h_square / 4 * hessian_matrix;
		break;
	}
}

void Simulation::evaluateHessianSmart(const VectorX& x, SparseMatrix& hessian_matrix)
{
	ScalarType h_square = m_h*m_h;
	switch (m_integration_method)
	{
	case INTEGRATION_QUASI_STATICS:
		evaluateHessianPureConstraintSmart(x, hessian_matrix);
		break;//DO NOTHING
	case INTEGRATION_IMPLICIT_EULER:
		evaluateHessianPureConstraintSmart(x, hessian_matrix);
		hessian_matrix = m_mesh->m_mass_matrix + h_square*hessian_matrix;
		break;
	case INTEGRATION_IMPLICIT_BDF2:
		evaluateHessianPureConstraintSmart(x, hessian_matrix);
		hessian_matrix = m_mesh->m_mass_matrix + h_square*4.0 / 9.0*hessian_matrix;
		break;
	case INTEGRATION_IMPLICIT_MIDPOINT:
		evaluateHessianPureConstraintSmart((x + m_mesh->m_current_positions) / 2, hessian_matrix);
		hessian_matrix = m_mesh->m_mass_matrix + h_square / 4 * hessian_matrix;
		break;
	case INTEGRATION_IMPLICIT_NEWMARK_BETA:
		evaluateHessianPureConstraintSmart(x, hessian_matrix);
		hessian_matrix = m_mesh->m_mass_matrix + h_square / 4 * hessian_matrix;
		break;
	}
}

// evaluate hessian
void Simulation::evaluateHessianForCG(const VectorX& x)
{
	VectorX x_evaluated_point;
	switch (m_integration_method)
	{
	case INTEGRATION_QUASI_STATICS:
	case INTEGRATION_IMPLICIT_EULER:
	case INTEGRATION_IMPLICIT_BDF2:
	case INTEGRATION_IMPLICIT_NEWMARK_BETA:
		x_evaluated_point = x;
		break;
	case INTEGRATION_IMPLICIT_MIDPOINT:
		x_evaluated_point = (x + m_mesh->m_current_positions) / 2;
		break;
	}

	if (!m_enable_openmp)
	{
		for (std::vector<Constraint*>::iterator it = m_constraints.begin(); it != m_constraints.end(); ++it)
		{
			(*it)->EvaluateHessian(x_evaluated_point, m_definiteness_fix);
		}

		if (m_processing_collision)
		{
			for (std::vector<CollisionSpringConstraint>::iterator it = m_collision_constraints.begin(); it != m_collision_constraints.end(); ++it)
			{
				it->EvaluateHessian(x_evaluated_point, m_definiteness_fix);
			}
		}
	}
	else
	{
		int i;
#pragma omp parallel
		{
#pragma omp for
			for (i = 0; i < m_constraints.size(); i++)
			{
				m_constraints[i]->EvaluateHessian(x_evaluated_point, m_definiteness_fix);
			}
#pragma omp for
			for (i = 0; i < m_collision_constraints.size(); i++)
			{
				m_collision_constraints[i].EvaluateHessian(x_evaluated_point, m_definiteness_fix);
			}
		}
	}
}
// apply hessian
void Simulation::applyHessianForCG(const VectorX& x, VectorX & b)
{
	ScalarType h_square = m_h*m_h;
	switch (m_integration_method)
	{
	case INTEGRATION_QUASI_STATICS:
		applyHessianForCGPureConstraint(x, b);
		break;//DO NOTHING
	case INTEGRATION_IMPLICIT_EULER:
		applyHessianForCGPureConstraint(x, b);
		b = m_mesh->m_mass_matrix * x + h_square*b;
		break;
	case INTEGRATION_IMPLICIT_BDF2:
		applyHessianForCGPureConstraint(x, b);
		b = m_mesh->m_mass_matrix * x + h_square*4.0 / 9.0*b;
		break;
	case INTEGRATION_IMPLICIT_MIDPOINT:
		applyHessianForCGPureConstraint(x, b);
		b = m_mesh->m_mass_matrix * x + h_square / 4 * b;
		break;
	case INTEGRATION_IMPLICIT_NEWMARK_BETA:
		applyHessianForCGPureConstraint(x, b);
		b = m_mesh->m_mass_matrix * x + h_square/4*b;
		break;
	}

}
