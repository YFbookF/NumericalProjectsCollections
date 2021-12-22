	bool ClothSim::LBFGSStep(int k, Vec3x* v_k, const Vec3x* x_k)
	{
		EventTimer timer;

		timer.start();
		evaluateGradient(x_k, v_k);
		std::cout << " evaluateGradient: " << timer.elapsedMilliseconds();

		Scalar one = 1.f, neg_one = -1.f;
		cudaMemset(d_delta_v, 0, m_num_total_nodes * sizeof(Vec3x));
		std::vector<bool> converged(m_num_cloths, false);

		for (int i = 0; i < m_num_cloths; ++i)
		{
			int n_node = getNumNodes(i);
			Scalar* last_g = (Scalar*)&d_last_g[m_offsets[i]];
			Scalar* gradient = (Scalar*)&d_g[m_offsets[i]];
			Scalar* delta_v = (Scalar*)&d_delta_v[m_offsets[i]];

			if (k == 0) // first iteration
			{
				m_lbfgs_g_queue[i].empty();
				m_lbfgs_v_queue[i].empty();
			}
			else
			{
				CublasCaller<Scalar>::scal(m_cublas_handle, 3 * n_node, &neg_one, last_g);
				CublasCaller<Scalar>::axpy(m_cublas_handle, 3 * n_node, &one, gradient, last_g);
				m_lbfgs_g_queue[i].enqueue(last_g);
			}
			CublasCaller<Scalar>::copy(m_cublas_handle, 3 * n_node, gradient, last_g);

			int size = m_lbfgs_g_queue[i].size();
			std::vector<Scalar> pho(size);
			std::vector<Scalar> zeta(size);

			// first loop
			for (int w = size - 1; w >= 0; --w)
			{
				const Scalar* s = m_lbfgs_v_queue[i][w];
				const Scalar* t = m_lbfgs_g_queue[i][w];

				CublasCaller<Scalar>::dot(m_cublas_handle, 3 * n_node, s, t, &pho[w]);
				CublasCaller<Scalar>::dot(m_cublas_handle, 3 * n_node, s, gradient, &zeta[w]);
				zeta[w] /= pho[w];

				Scalar neg_zeta = -zeta[w];
				CublasCaller<Scalar>::axpy(m_cublas_handle, 3 * n_node, &neg_zeta, t, gradient);
			}

			// linear solve
			timer.start();
			m_solvers[i].cholSolve(gradient, delta_v);
			std::cout << " cholSolve: " << timer.elapsedMilliseconds();

			// second loop
			for (int w = 0; w < size; ++w)
			{
				const Scalar* s = m_lbfgs_v_queue[i][w];
				const Scalar* t = m_lbfgs_g_queue[i][w];

				Scalar eta;
				CublasCaller<Scalar>::dot(m_cublas_handle, 3 * n_node, t, delta_v, &eta);
				eta = zeta[w] - eta / pho[w];
				CublasCaller<Scalar>::axpy(m_cublas_handle, 3 * n_node, &eta, s, delta_v);
			}

			// line-search
			CublasCaller<Scalar>::scal(m_cublas_handle, 3 * n_node, &neg_one, delta_v);
			if (m_enable_line_search) lineSearch(i, last_g, delta_v, m_step_size[i]);

			CublasCaller<Scalar>::scal(m_cublas_handle, 3 * n_node, &m_step_size[i], delta_v);
			m_lbfgs_v_queue[i].enqueue(delta_v);
			CublasCaller<Scalar>::axpy(m_cublas_handle, 3 * n_node, &one, delta_v, (Scalar*)v_k);

			Scalar res;
			CublasCaller<Scalar>::nrm2(m_cublas_handle, 3 * n_node, last_g, &res);
			if (sqrt(res) < 1e-6 || m_step_size[i] < 1e-6) converged[i] = true;
			std::cout << "[cloth " << i << "] residual: " << sqrt(res) << " step_size: " << m_step_size[i];
		}

		bool all_converged = true;
		for (const auto& b_con : converged) all_converged = all_converged && b_con;

		return all_converged;
	}