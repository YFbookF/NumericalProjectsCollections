https://github.com/Azmisov/snow
// 一定要用trust
__host__ void MPMSolver::initialTransfer() {
    Grid *grid_ptr = thrust::raw_pointer_cast(&grids[0]);

    auto ff = [=] __device__ (Particle& p) {
        float h_inv = 1.0f / PARTICLE_DIAM;
        Eigen::Vector3i pos((p.position * h_inv).cast<int>());

        for (int z = -G2P; z <= G2P; z++) {
            for (int y = -G2P; y <= G2P; y++) {
                for (int x = -G2P; x <= G2P; x++) {
                    auto _pos = pos + Eigen::Vector3i(x, y, z);
                    if (!IN_GRID(_pos)) continue;

                    Eigen::Vector3f diff = (p.position - (_pos.cast<float>() * PARTICLE_DIAM)) * h_inv;
                    int grid_idx = getGridIndex(_pos);
                    float mi = p.mass * weight(diff.cwiseAbs());
                    atomicAdd(&(grid_ptr[grid_idx].mass), mi);
                }
            }
        }
    };

    thrust::for_each(thrust::device, particles.begin(), particles.end(), ff);
}

__host__ void MPMSolver::transferData() {
    Grid *grid_ptr = thrust::raw_pointer_cast(&grids[0]);

    auto ff = [=] __device__ (Particle& p) {
        float h_inv = 1.0f / PARTICLE_DIAM;
        Eigen::Vector3i pos((p.position * h_inv).cast<int>());
        Eigen::Matrix3f volume_stress = -1.0f * p.energyDerivative();

        for (int z = -G2P; z <= G2P; z++) {
            for (int y = -G2P; y <= G2P; y++) {
                for (int x = -G2P; x <= G2P; x++) {
                    auto _pos = pos + Eigen::Vector3i(x, y, z);
                    if (!IN_GRID(_pos)) continue;

                    Eigen::Vector3f diff = (p.position - (_pos.cast<float>() * PARTICLE_DIAM)) * h_inv;
                    auto gw = gradientWeight(diff);
                    int grid_idx = getGridIndex(_pos);

                    Eigen::Vector3f f = volume_stress * gw;

                    float mi = p.mass * weight(diff.cwiseAbs());
                    atomicAdd(&(grid_ptr[grid_idx].mass), mi);
                    atomicAdd(&(grid_ptr[grid_idx].velocity(0)), p.velocity(0) * mi);
                    atomicAdd(&(grid_ptr[grid_idx].velocity(1)), p.velocity(1) * mi);
                    atomicAdd(&(grid_ptr[grid_idx].velocity(2)), p.velocity(2) * mi);
                    atomicAdd(&(grid_ptr[grid_idx].force(0)), f(0));
                    atomicAdd(&(grid_ptr[grid_idx].force(1)), f(1));
                    atomicAdd(&(grid_ptr[grid_idx].force(2)), f(2));
                }
            }
        }
    };

    thrust::for_each(thrust::device, particles.begin(), particles.end(), ff);
}