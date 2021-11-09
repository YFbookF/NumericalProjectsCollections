Projective Dynamics: Fast Simulation
of Hyperelastic Models with Dynamic Constraints  

http://www.github.com/mattoverby/admm-elastic

18号公式如下
$$
\bold x^{n+1} = argmin(\frac{1}{2\Delta t^2}||\bold x - \tilde {\bold x}||^2_M + \frac{1}{2}||\bold W(\bold D \bold x - \bold z^n + \overline {\bold u}^n||^2) \\
= (\bold M + \Delta t^2 \bold D^T \bold W^T \bold W \bold D)^{-1}(\bold M \bold {\tilde x} + \Delta ^2 \bold D^T \bold W^T \bold W (\bold z^n - \overline {\bold u}))
$$
那么

```
//solver.cpp
solver_termB.noalias() = M_xbar + solver_Dt_Wt_W * ( curr_z - curr_u );
		m_runtime.inner_iters += m_linsolver->solve( curr_x, solver_termB );
```

Energy的定义如下，来源于第27和28式
$$
\bold z^{n+1}_i = argmin(U_i(\bold z_i) + \frac{1}{2}||\bold W_i(\bold D_i \bold x^{n+1} - \bold z_i + \overline {\bold u}_i||^2)\\
\bold {\overline u_i}^{n+1} = \bold {\overline u_i}^n + \bold D _i \bold x^{n+1} - \bold z_i^{n+1}
$$


```
inline void EnergyTerm::update( const SparseMat &D, const VecX &x, VecX &z, VecX &u ){
	int dof = x.rows();
	int dim = get_dim();
	VecX Dix = D.block(g_index,0,dim,dof)*x;
	VecX ui = u.segment(g_index,dim);
	VecX zi = Dix + ui;
	prox( zi );
	ui += (Dix - zi);
	u.segment(g_index,dim) = ui;
	z.segment(g_index,dim) = zi;
}
```

Z 是 ADMM uodate velocity

extended graph coloring algorithm
$$
\bold x_{tmp} = \bold G_i^T(\bold b_i / a_{ii} - \mathcal{A_i \bold x}/a_{ii} + \bold x_i^n - \bold p_i) \qquad \bold x_i^{n+1} = \bold G_i \bold x_{tmp} + \bold p_i
$$
Nodal MultiColorGS.hpp

```
	Vec3 delta_x = Vec3(
		(bi[0] - LUx[0])/aii[0],
		(bi[1] - LUx[1])/aii[1],
		(bi[2] - LUx[2])/aii[2]
	) - p;

	// Solve constrained to a plane
	Mat32 G = orthoG( n );
	Vec2 x_tan = G.transpose() * delta_x;
	new_x = G * x_tan + p;
	return new_x;

```

而且
$$
(\bold M + \Delta t^2 \bold D^T \bold W^T \bold W \bold D)
$$


```
	// Create the Selector+Reduction matrix
	m_W_diag = Eigen::Map<VecX>(&weights[0], weights.size());
	int n_D_rows = weights.size();
	m_D.resize( n_D_rows, dof );
	m_D.setZero();
	m_D.setFromTriplets( triplets.begin(), triplets.end() );
	m_Dt = m_D.transpose();

	// Compute mass matrix
	SparseMat M( dof, dof );
	Eigen::VectorXi nnz = Eigen::VectorXi::Ones( dof ); // non zeros per column
	M.reserve(nnz);
	for( int i=0; i<dof; ++i ){ M.coeffRef(i,i) = m_masses[i]; }

	// Set global matrices
	SparseMat W( n_D_rows, n_D_rows );
	W.reserve(n_D_rows);
	for( int i=0; i<n_D_rows; ++i ){ W.coeffRef(i,i) = m_W_diag[i]; }
	const double dt2 = (m_settings.timestep_s*m_settings.timestep_s);
	solver_Dt_Wt_W = dt2 * m_Dt * W * W;
	solver_termA = M + SparseMat(solver_Dt_Wt_W * m_D);
```

