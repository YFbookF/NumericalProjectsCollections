https://github.com/ltt1598/Quasi-Newton-Methods-for-Real-time-Simulation-of-Hyperelastic-Materials

```
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
```

![image-20211029125920052](D:\定理\矩阵计算\BGFS\image-20211029125920052.png)

