 https://github.com/mattoverby/mpm-optimization
void Solver::implicit_solve()
{
	using namespace Eigen;

	// Initial guess of grid velocities
	VectorXd v(active_grid.size()*3);
#pragma omp parallel for
	for(int i=0; i<active_grid.size(); ++i)
	{
		for(int j=0; j<3; ++j)
		{
			v[i*3+j] = active_grid[i]->v[j];
		}
	}

	// Minimize
	// 然后就是一些非线性方程的解法
	Objective obj(this);
	optimizer.minimize(obj, v);

	// Copy solver results back to grid
#pragma omp parallel for
	for(int i=0; i<active_grid.size(); ++i)
	{
		for(int j=0; j<3; ++j)
		{
			active_grid[i]->v[j] = v[i*3+j];
		}
		active_grid[i]->v +=  MPM::timestep_s*gravity; // explicitly add gravity
	}

} // end implicit solve