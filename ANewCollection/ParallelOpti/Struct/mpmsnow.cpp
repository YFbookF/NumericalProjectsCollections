https://github.com/Azmisov/snow
struct f {
    __host__ __device__
    Grid operator()(const int& idx) {
        return Grid(Eigen::Vector3i(idx % GRID_BOUND_X, idx % (GRID_BOUND_X * GRID_BOUND_Y) / GRID_BOUND_X, idx / (GRID_BOUND_X * GRID_BOUND_Y)));
    }
};

__host__ MPMSolver::MPMSolver(const std::vector<Particle>& _particles) {
    particles.resize(_particles.size());
    thrust::copy(_particles.begin(), _particles.end(), particles.begin());

    grids.resize(GRID_BOUND_X * GRID_BOUND_Y * GRID_BOUND_Z);
    thrust::tabulate(
        thrust::device,
        grids.begin(),
        grids.end(),
        f()
    );
}