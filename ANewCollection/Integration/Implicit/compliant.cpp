//https://github.com/pielet/Hair-DER CompliantImplicitEuler
bool CompliantImplicitEuler::stepScene()
{
	std::cout << "[pre-compute]" << std::endl;
	scalar t0 = timingutils::seconds();
	scalar t1, ts;
	
	assert(m_x.size() == m_v.size());
	assert(m_x.size() == m_m.size());

	VectorXs dv(m_v.size());
	VectorXs dx(m_x.size());
	VectorXs dx_scripted(m_x.size());

	dv.setZero();
	dx = m_v * m_dt;
	dx_scripted.setZero();  // fixed point will update foreahead using current v
	
	for(int i = 0; i < m_vert_num; ++i)
	{
		if(isFixed(i)) {
			int numdofs = isTip(i) ? 3 : 4;
			dx_scripted.segment( getDof(i), numdofs ) = m_v.segment( getDof(i), numdofs ) * m_dt;
		}
	}
	
	t1 = timingutils::seconds();
	SceneStepper::m_timing_statistics[0] += t1 - t0; // local precomputation
	t0 = t1;
	
	std::cout << "[compute-assist-vars]" << std::endl;
	updateNumConstraints(dx_scripted, dv);
	computeIntegrationVars(dx_scripted, dv);
	
	t1 = timingutils::seconds();
	SceneStepper::m_timing_statistics[1] += t1 - t0; // Jacobian
	ts = t0 = t1;
	
	const int nconstraint = m_lambda.size();
	
	m_A_nz.erase( std::remove_if( m_A_nz.begin(), m_A_nz.end(), [&] ( const Triplets& t ) {
		return t.value() == 0.0;
	}), m_A_nz.end());

	ts = timingutils::seconds();

	m_A.setFromTriplets( m_A_nz.begin(), m_A_nz.end());

	ts = timingutils::seconds();
	
	if (m_J.rows() != nconstraint) m_J.resize( nconstraint, m_dof_num );
	m_J_nz.erase(std::remove_if(m_J_nz.begin(), m_J_nz.end(), [&] ( const Triplets& t ) {
		return  (t.value() == 0.0);// || scene.isFixed(scene.getVertFromDof(t.col()));
	}), m_J_nz.end());
	m_J.setFromTriplets( m_J_nz.begin(), m_J_nz.end() );
	m_J.makeCompressed();
	
	if (m_invC.rows() != nconstraint) m_invC.resize(nconstraint, nconstraint);
	m_invC.setFromTriplets(m_invC_nz.begin(), m_invC_nz.end());
	
	m_JC = m_J.transpose() * m_invC;
	
	m_A += m_JC * m_J;
	
	m_A *= m_dt * m_dt;

	for(int i = 0; i < m_dof_num; ++i)
	{
		if (isFixed(getVertFromDof(i))) {
			m_M_nz[i] = (Triplets(i, i, 1.0));
		} else {
			m_M_nz[i] = (Triplets(i, i, m_m(i)));
		}
	}
	
	m_M.setFromTriplets(m_M_nz.begin(), m_M_nz.end());
	
	m_A += m_M;
	
	m_gradU.setZero();
	accumulateExternalGradU(m_gradU, dx_scripted, dv);
	zeroFixedDoFs(m_gradU);

	m_b = m_v.cwiseProduct(m_m) - m_dt * (m_gradU + (m_JC * m_Phi));

	m_A.makeCompressed();
	
	t1 = timingutils::seconds();
	SceneStepper::m_timing_statistics[2] += t1 - t0; t0 = t1; // matrix composition

	std::cout << "[solve-equations]" << std::endl;
	if(m_model.m_max_iters > 0) {
		scalar nmb = m_b.norm();
		
		m_iterative_solver.compute(m_A);
		m_iterative_solver.setTolerance(m_model.m_criterion / nmb);
		m_iterative_solver.setMaxIterations(m_model.m_max_iters);
		m_vplus = m_iterative_solver.solveWithGuess(m_b, m_v);
		std::cout << "[cg total iter: " << m_iterative_solver.iterations() << ", res: " << (m_iterative_solver.error() * nmb) << "]" << std::endl;
	} else {
		m_solver.compute(m_A);
		m_vplus = m_solver.solve(m_b);
	}
	
	m_lambda = -m_invC * (m_J * m_vplus * m_dt + m_Phi);
	
	t1 = timingutils::seconds();
	SceneStepper::m_timing_statistics[3] += t1 - t0; t0 = t1; // solve equation
	
	// update x, v
	for(int i = 0; i < m_vert_num; ++i){
		if(!isFixed(i)){
			int numdofs = isTip(i) ? 3 : 4;
			m_v.segment(getDof(i), numdofs) = m_vplus.segment(getDof(i), numdofs);
		}
	}
	storeLambda( m_lambda );
	m_x += m_v * m_dt;

	setNextX();

	return true; 
}