///////////////////////////////////////////////////////////////////////////////
// Efficient and Accurate Collision Response for Elastically Deformable Models
// [Verschoor et al. 2019]
//https://github.com/ipc-sim/IPC
void compute_Verschoor_point_triangle_constraint(
    const Eigen::Vector3d& v0_t0, // point at start of the timestep
    const Eigen::Vector3d& v1_t0, // triangle point 0 at start of the timestep
    const Eigen::Vector3d& v2_t0, // triangle point 1 at start of the timestep
    const Eigen::Vector3d& v3_t0, // triangle point 2 at start of the timestep
    const Eigen::Vector3d& v0_t1, // point at end of the timestep
    const Eigen::Vector3d& v1_t1, // triangle point 0 at end of the timestep
    const Eigen::Vector3d& v2_t1, // triangle point 1 at end of the timestep
    const Eigen::Vector3d& v3_t1, // triangle point 2 at end of the timestep
    double toi, double& c)
{
    // 1. Find the point of contact using CCD
    if (toi < 0 || toi > 1) {
        c = 1e28;
        return;
    }
    const Eigen::Vector3d vc_toi = (v0_t1 - v0_t0) * toi + v0_t0;
    // 2. Compute barycentric coordinates of contact point ([α₁, α₂, α₃])
    Eigen::Vector3d barycentric_coords;
    const Eigen::Vector3d v1_toi = (v1_t1 - v1_t0) * toi + v1_t0;
    const Eigen::Vector3d v2_toi = (v2_t1 - v2_t0) * toi + v2_t0;
    const Eigen::Vector3d v3_toi = (v3_t1 - v3_t0) * toi + v3_t0;
    barycentric_coordinates(vc_toi, v1_toi, v2_toi, v3_toi, barycentric_coords);
    // 3. Compute contact point's position at the start of the iteration
    const Eigen::Vector3d vc_t1 = barycentric_coords(0) * v1_t1
        + barycentric_coords(1) * v2_t1 + barycentric_coords(2) * v3_t1;
    // 4. Compute the contact normal
    // Is the normal at time of impact?
    // Eigen::Vector3d normal = (v2_toi - v1_toi).cross(v3_toi - v1_toi).normalized();
    // or is the normal at start of iteration?
    Eigen::Vector3d normal = (v2_t1 - v1_t1).cross(v3_t1 - v1_t1).normalized();
    // 5. Compute the distance constraint
    c = normal.dot(v0_t1 - vc_t1); // distance to plane
}
////////////////////////////////////////////////////////////////////////////////
// Parallel contact-aware simulations of deformable particles in 3D Stokes flow
// [Lu et al. 2018]

void compute_STIV_point_triangle_constraint(
    const Eigen::Vector3d& v0_t0, // point at start of the timestep
    const Eigen::Vector3d& v1_t0, // triangle point 0 at start of the timestep
    const Eigen::Vector3d& v2_t0, // triangle point 1 at start of the timestep
    const Eigen::Vector3d& v3_t0, // triangle point 2 at start of the timestep
    const Eigen::Vector3d& v0_t1, // point at end of the timestep
    const Eigen::Vector3d& v1_t1, // triangle point 0 at end of the timestep
    const Eigen::Vector3d& v2_t1, // triangle point 1 at end of the timestep
    const Eigen::Vector3d& v3_t1, // triangle point 2 at end of the timestep
    double toi, double& c)
{
    if (toi < 0 || toi > 1 || !std::isfinite(toi)) {
        c = 1e28;
        return;
    }
    // V(t,X) = (1 - τ) * √(ϵ² + (X_k(t_{n+1}) - X_k(t_{n+1}))⋅n(τ))²) * |t|
    // 0. Avoid zero volume from parallel displacment and normal.
    const double EPS = 1e-3;
    // 1. Compute the displacment of the vertex
    const Eigen::Vector3d u0 = v0_t1 - v0_t0;
    // 2. Compute normal of the triangle at the time of impact
    const Eigen::Vector3d v1_toi = (v1_t1 - v1_t0) * toi + v1_t0;
    const Eigen::Vector3d v2_toi = (v2_t1 - v2_t0) * toi + v2_t0;
    const Eigen::Vector3d v3_toi = (v3_t1 - v3_t0) * toi + v3_t0;
    Eigen::Vector3d n_toi = (v2_toi - v1_toi).cross(v3_toi - v1_toi);
    // 4. Compute the area of triangle
    const double tri_area = n_toi.norm() / 2;
    n_toi.normalize();
    // 5. Compute the STIV
    c = -(1 - toi) * sqrt(EPS * EPS + pow(u0.dot(n_toi), 2)) * tri_area;
    spdlog::debug("STIV={:g}", c);
}