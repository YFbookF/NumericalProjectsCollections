https://github.com/Azmisov/snow
    // run conjugate residual method
    for (int i = 0; i < SOLVE_MAX_ITERATIONS; i++) {
        thrust::for_each(
            thrust::device,
            grids.begin(),
            grids.end(),
            [=] __device__ (Grid& g) {
                float rar = (g.r).cwiseProduct(g.ar).sum(), apap = (g.ap).cwiseProduct(g.ap).sum();
                float alpha = (apap > 1e-8)? rar / apap: 0.0f;
                g.v += alpha * g.p;
                g.r += (-alpha * g.ap);
                g.rar_tmp = rar;
            }
        );

        computeAr();

        thrust::for_each(
            thrust::device,
            grids.begin(),
            grids.end(),
            [=] __device__ (Grid& g) {
                float beta = (g.rar_tmp > 1e-8)? (g.r).cwiseProduct(g.ar).sum() / g.rar_tmp: 0.0f;
                g.p = g.r + beta * g.p;
                g.ap = g.ar + beta * g.ap;
            }
        );
    }