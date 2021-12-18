https://github.com/mattoverby/mpm-optimization
void Solver::explicit_solve()
{
#pragma omp parallel for
	for(int i=0; i<m_particles.size(); ++i){
		Particle *p = m_particles[i];
		Eigen::Matrix3d p_F = p->get_deform_grad();
		p->tempP = p->vol * p->get_piola_stress(p_F) * p->Fe.transpose();
	} // end loop particles

	// Velocity update
#pragma omp parallel for
	for(int i=0; i<active_grid.size(); ++i)
	{
		GridNode *g = active_grid[i];

		Eigen::Vector3d f(0,0,0);
		for(int j=0; j<g->particles.size(); ++j){
			Particle *p = g->particles[j];
			Eigen::Matrix3d p_F = p->get_deform_grad();
			f -= p->tempP*g->dwip[j];
		}

		g->v += (MPM::timestep_s*f)/g->m + MPM::timestep_s*gravity;

	} // end for all active grid

} // end compute forces