//https://github.com/ltt1598/Quasi-Newton-Methods-for-Real-time-Simulation-of-Hyperelastic-Materials
bool Simulation::performNewtonsMethodOneIteration(VectorX& x)
{
	TimerWrapper timer; timer.Tic();
	// evaluate gradient direction
	VectorX gradient;
	evaluateGradient(x, gradient, true);
	//QSEvaluateGradient(x, gradient, m_ss->m_quasi_static);
#ifdef ENABLE_MATLAB_DEBUGGING
	g_debugger->SendVector(gradient, "g");
#endif

	timer.TocAndReport("evaluate gradient", m_verbose_show_converge);
	timer.Tic();

	// evaluate hessian matrix
	SparseMatrix hessian_1;
	evaluateHessian(x, hessian_1);
	//SparseMatrix hessian_2;
	//evaluateHessianSmart(x, hessian_2);

	SparseMatrix& hessian = hessian_1;

#ifdef ENABLE_MATLAB_DEBUGGING
	g_debugger->SendSparseMatrix(hessian_1, "H");
	//g_debugger->SendSparseMatrix(hessian_2, "H2");
#endif

	timer.TocAndReport("evaluate hessian", m_verbose_show_converge);
	timer.Tic();
	VectorX descent_dir;
    //hessian descent_dir = gradient
	// A          x       =    b
	linearSolve(descent_dir, hessian, gradient);
	descent_dir = -descent_dir;

	timer.TocAndReport("solve time", m_verbose_show_converge);
	timer.Tic();

	// line search
	ScalarType step_size = lineSearch(x, gradient, descent_dir);
	//if (step_size < EPSILON)
	//{
	//	std::cout << "correct step size to 1" << std::endl;
	//	step_size = 1;
	//}
	// update x
	x = x + descent_dir * step_size;

	//if (step_size < EPSILON)
	//{
	//	printVolumeTesting(x);
	//}

	timer.TocAndReport("line search", m_verbose_show_converge);
	//timer.Toc();
	//std::cout << "newton: " << timer.Duration() << std::endl;

	if (-descent_dir.dot(gradient) < EPSILON_SQUARE)
		return true;
	else
		return false;
}
