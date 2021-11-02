//https://github.com/ltt1598/Quasi-Newton-Methods-for-Real-time-Simulation-of-Hyperelastic-Materials
bool Simulation::performLBFGSOneIteration(VectorX& x)
{
	bool converged = false;
	ScalarType current_energy;
	VectorX gf_k;

	// set xk and gfk
	if (m_ls_is_first_iteration || !m_enable_line_search)
	{
		current_energy = evaluateEnergyAndGradient(x, gf_k);
	}
	else
	{
		current_energy = m_ls_prefetched_energy;
		gf_k = m_ls_prefetched_gradient;
	}
	//current_energy = evaluateEnergyAndGradient(x, gf_k);

	if (m_lbfgs_need_update_H0) // first iteration
	{
		// clear sk and yk and alpha_k
#ifdef USE_STL_QUEUE_IMPLEMENTATION
		// stl implementation
		m_lbfgs_y_queue.clear();
		m_lbfgs_s_queue.clear();
#else
		// my implementation
		delete m_lbfgs_queue;
		m_lbfgs_queue = new QueueLBFGS(x.size(), m_lbfgs_m);
#endif

		// decide H0 and it's factorization precomputation
		switch (m_lbfgs_H0_type)
		{
		case LBFGS_H0_LAPLACIAN:
			prefactorize();
			break;
		default:
			//prefactorize();
			break;
		}

		g_lbfgs_timer.Resume();
		// store them before wipeout
		m_lbfgs_last_x = x;
		m_lbfgs_last_gradient = gf_k;

		g_lbfgs_timer.Pause();
		// first iteration
		VectorX r;
		LBFGSKernelLinearSolve(r, gf_k, 1);
		g_lbfgs_timer.Resume();

		// update
		VectorX p_k = -r;
		g_lbfgs_timer.Pause();

		if (-p_k.dot(gf_k) < EPSILON_SQUARE || p_k.norm() / x.norm() < LARGER_EPSILON)
		{
			converged = true;
		}

		ScalarType alpha_k = linesearchWithPrefetchedEnergyAndGradientComputing(x, current_energy, gf_k, p_k, m_ls_prefetched_energy, m_ls_prefetched_gradient);
		x += alpha_k * p_k;

		// final touch
		m_lbfgs_need_update_H0 = false;
	}
	else // otherwise
	{
		TimerWrapper t_local;
		TimerWrapper t_global;
		TimerWrapper t_linesearch;
		TimerWrapper t_other;
		bool verbose = false;

		t_other.Tic();
		g_lbfgs_timer.Resume();
		// enqueue stuff
		VectorX s_k = x - m_lbfgs_last_x;
		VectorX y_k = gf_k - m_lbfgs_last_gradient;

#ifdef USE_STL_QUEUE_IMPLEMENTATION
		//stl implementation
		if (m_lbfgs_s_queue.size() > m_lbfgs_m)
		{
			m_lbfgs_s_queue.pop_back();
			m_lbfgs_y_queue.pop_back();
		}
		// enqueue stuff
		m_lbfgs_s_queue.push_front(s_k);
		m_lbfgs_y_queue.push_front(y_k);

		int m_queue_size = m_lbfgs_s_queue.size();
#else
		// my implementation
		if (m_lbfgs_queue->isFull())
		{
			m_lbfgs_queue->dequeue();
		}
		m_lbfgs_queue->enqueue(s_k, y_k);

		int m_queue_size = m_lbfgs_queue->size();
#endif

		// store them before wipeout
		m_lbfgs_last_x = x;
		m_lbfgs_last_gradient = gf_k;
		VectorX q = gf_k;

		// loop 1 of l-BFGS
		std::vector<ScalarType> rho;
		rho.clear();
		std::vector<ScalarType> alpha;
		alpha.clear();
		int m_queue_visit_upper_bound = (m_lbfgs_m < m_queue_size) ? m_lbfgs_m : m_queue_size;
		ScalarType* s_i = NULL;
		ScalarType* y_i = NULL;
		for (int i = 0; i != m_queue_visit_upper_bound; i++)
		{
#ifdef USE_STL_QUEUE_IMPLEMENTATION
			// stl implementation
			ScalarType yi_dot_si = m_lbfgs_y_queue[i].dot(m_lbfgs_s_queue[i]);
			if (yi_dot_si < EPSILON_SQUARE)
			{
				return true;
			}
			ScalarType rho_i = 1.0 / yi_dot_si;
			rho.push_back(rho_i);
			alpha.push_back(rho[i]*m_lbfgs_s_queue[i].dot(q));
			q = q - alpha[i] * m_lbfgs_y_queue[i];
#else
			// my implementation
			m_lbfgs_queue->visitSandY(&s_i, &y_i, i);
			Eigen::Map<const VectorX> s_i_eigen(s_i, x.size());
			Eigen::Map<const VectorX> y_i_eigen(y_i, x.size());
			ScalarType yi_dot_si = (y_i_eigen.dot(s_i_eigen));
			if (yi_dot_si < EPSILON_SQUARE)
			{
				return true;
			}
			ScalarType rho_i = 1.0 / yi_dot_si;
			rho.push_back(rho_i);
			ScalarType alpha_i = rho_i * s_i_eigen.dot(q);
			alpha.push_back(alpha_i);
			q -= alpha_i * y_i_eigen;
#endif
		}
		// compute H0 * q
		g_lbfgs_timer.Pause();
		t_other.Pause();
		t_global.Tic();
		VectorX r;
		// compute the scaling parameter on the fly
		ScalarType scaling_parameter = (s_k.transpose()*y_k).trace() / (y_k.transpose()*y_k).trace();
		if (scaling_parameter < EPSILON) // should not be negative
		{
			scaling_parameter = EPSILON;
		}
		LBFGSKernelLinearSolve(r, q, scaling_parameter);
		t_global.Toc();
		t_other.Resume();
		g_lbfgs_timer.Resume();
		// loop 2 of l-BFGS
		for (int i = m_queue_visit_upper_bound - 1; i >= 0; i--)
		{
#ifdef USE_STL_QUEUE_IMPLEMENTATION
			// stl implementation
			ScalarType beta = rho[i] * m_lbfgs_y_queue[i].dot(r);
			r = r + m_lbfgs_s_queue[i] * (alpha[i] - beta);
#else
			// my implementation
			m_lbfgs_queue->visitSandY(&s_i, &y_i, i);
			Eigen::Map<const VectorX> s_i_eigen(s_i, x.size());
			Eigen::Map<const VectorX> y_i_eigen(y_i, x.size());
			ScalarType beta = rho[i] * y_i_eigen.dot(r);
			r += s_i_eigen * (alpha[i] - beta);
#endif
		}
		// update
		VectorX p_k = -r;
		if (-p_k.dot(gf_k) < EPSILON_SQUARE || p_k.squaredNorm() < EPSILON_SQUARE)
		{
			converged = true;
		}
		g_lbfgs_timer.Pause();
		t_other.Toc();

		t_linesearch.Tic();
		//ScalarType alpha_k = lineSearch(x, gf_k, p_k);
		ScalarType alpha_k = linesearchWithPrefetchedEnergyAndGradientComputing(x, current_energy, gf_k, p_k, m_ls_prefetched_energy, m_ls_prefetched_gradient);
		t_linesearch.Toc();

		x += alpha_k * p_k;

		t_global.Report("Forward Backward Substitution", verbose, TIMER_OUTPUT_MICROSECONDS);
		t_other.Report("Two loop overhead", verbose, TIMER_OUTPUT_MICROSECONDS);
		t_linesearch.Report("Linesearch", verbose, TIMER_OUTPUT_MICROSECONDS);
	}

	return converged;
}

void Simulation::LBFGSKernelLinearSolve(VectorX & r, VectorX rhs, ScalarType scaled_identity_constant) // Ar = rhs
{
	r.resize(rhs.size());
	switch (m_lbfgs_H0_type)
	{
	case LBFGS_H0_IDENTITY:
		r = rhs / scaled_identity_constant;
		break;
	case LBFGS_H0_LAPLACIAN: // h^2*laplacian+mass
	{
		// solve the linear system in reduced dimension because of the pattern of the Laplacian matrix
		// convert to nx3 space
		EigenMatrixx3 rhs_n3(rhs.size()/3, 3);
		Vector3mx1ToMatrixmx3(rhs, rhs_n3);
		// solve using the nxn laplacian
		EigenMatrixx3 r_n3;
		if (m_solver_type == SOLVER_TYPE_CG)
		{
			m_preloaded_cg_solver_1D.setMaxIterations(m_iterative_solver_max_iteration);
			r_n3 = m_preloaded_cg_solver_1D.solve(rhs_n3);
		}
		else
		{
			r_n3 = m_prefactored_solver_1D.solve(rhs_n3);
		}
		// convert the result back
		Matrixmx3ToVector3mx1(r_n3, r);


		////// conventional solve using 3nx3n system
		//if (m_solver_type == SOLVER_TYPE_CG)
		//{
		//	m_preloaded_cg_solver.setMaxIterations(m_iterative_solver_max_iteration);
		//	r = m_preloaded_cg_solver.solve(rhs);
		//}
		//else
		//{
		//	r = m_prefactored_solver.solve(rhs);
		//}
	}
	break;
	default:
		break;
	}
}