//constraint.cpp
MeshGrad IneqCon::friction(double dt, MeshHess& jac)
{
    if (mu == 0)
        return MeshGrad();
    double fn = abs(energy_grad(value()));
    if (fn == 0)
        return MeshGrad();
    Vec3 v = Vec3(0);
    double inv_mass = 0;
    for (int i = 0; i < 4; i++) {
        if (nodes[i]) {
            v += w[i] * nodes[i]->v;
            if (free[i])
                inv_mass += sq(w[i]) / nodes[i]->m;
        }
    }
    Mat3x3 T = Mat3x3(1) - outer(n, n);
    double vt = norm(T * v);
    double f_by_v = min(mu * fn / vt, 1 / (dt * inv_mass));
    // double f_by_v = mu*fn/max(vt, 1e-1);
    MeshGrad force;
    for (int i = 0; i < 4; i++) {
        if (nodes[i] && free[i]) {
            force.push_back(MeshGradV(nodes[i], -w[i] * f_by_v * T * v));
            for (int j = 0; j < 4; j++) {
                if (free[j]) {
                    jac.push_back(MeshHessV(nodes[i], nodes[j], -w[i] * w[j] * f_by_v * T));
                }
            }
        }
    }
    return force;
}