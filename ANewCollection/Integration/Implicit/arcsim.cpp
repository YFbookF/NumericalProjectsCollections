//physics.cpp
vector<Vec3> implicit_update(vector<Node*>& nodes,
    const vector<Edge*>& edges, const vector<Face*>& faces,
    const vector<Vec3>& fext, const vector<Mat3x3>& Jext,
    const vector<Constraint*>& cons, double dt)
{
    size_t nn = nodes.size();

    // Expand Mx'' + Cx' + Kx = f using backward Euler we have
    // 1/Dt [M + Dt C + Dt^2 (K - J)] Dv = F + (Dt J - C - Dt K) x' - K x
    // where x''(t + Dt) = (x'(t + Dt) - x'(t)) / Dt = Dv / Dt
    // For first step we have
    // M Dv/Dt = F (x + Dt (v + Dv))
    // A Dv = b
    // A = M - Dt^2 DF/Dx
    // b = Dt (F(x) + Dt DF/Dx v))
    SpMat<Mat3x3> A(nn, nn);
    vector<Vec3> b(nn, Vec3(0));

    for (size_t n = 0; n < nn; n++) {
        A(n, n) += Mat3x3(nodes[n]->m) - dt * dt * Jext[n];
        b[n] += dt * (fext[n] + dt * Jext[n] * nodes[n]->v);
    }
    consistency((vector<Vec3>&)fext, "fext");
    consistency(b, "fext");

    add_internal_forces<WS>(faces, edges, A, b, dt);
    consistency(b, "internal forces");

    add_constraint_forces(cons, A, b, dt);
    consistency(b, "constraints");

    add_friction_forces(cons, A, b, dt);
    consistency(b, "friction");

    vector<Vec3> dv = mat3x3_linear_solve(A, b);
    consistency(dv, "linear");

    return dv;
}