https://github.com/sxyu/volrend/blob/master/src/cuda/volrend.cu
template<typename scalar_t>
__host__ __device__ __inline__ void rodrigues(
        const scalar_t* __restrict__ aa,
        scalar_t* __restrict__ dir) {
    scalar_t angle = _norm(aa);
    if (angle < 1e-6) return;
    scalar_t k[3];
    for (int i = 0; i < 3; ++i) k[i] = aa[i] / angle;
    scalar_t cos_angle = cos(angle), sin_angle = sin(angle);
    scalar_t cross[3];
    _cross3(k, dir, cross);
    scalar_t dot = _dot3(k, dir);
    for (int i = 0; i < 3; ++i) {
        dir[i] = dir[i] * cos_angle + cross[i] * sin_angle + k[i] * dot * (1.0 - cos_angle);
    }
}
