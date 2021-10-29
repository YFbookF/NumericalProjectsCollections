//physics.cpp

template <Space s>
double stretching_energy(const Face* face)
{
    Mat3x3 F = deformation_gradient<WS>(face);
    Mat3x3 G = (F.t() * F - Mat3x3(1)) * 0.5;
    Mat3x3 S = material_model(face, G);

    return face->a * 0.5 * inner(S, G);
}

template <Space s>
pair<Mat9x9, Vec9> stretching_force(const Face* face)
{
    const Material* mat = face->material;
    // compute stress, strain
    const Vec3 x[3] = { pos<s>(face->v[0]->node), pos<s>(face->v[1]->node), pos<s>(face->v[2]->node) };
    Mat3x3 F = deformation_gradient<s>(face);
    Mat3x3 G = (F.t() * F - Mat3x3(1)) * 0.5;
    double weakening_mult = 1 / (1 + mat->weakening * face->damage);

    Mat3x3 Y = face->invDm * face->Sp_str;
    Mat3x3 D = Mat3x3::rows(-Y.row(0) - Y.row(1), Y.row(0), Y.row(1));
    Mat<3, 9> DD[3] = { kronecker(rowmat(D.col(0)), Mat3x3(1)),
        kronecker(rowmat(D.col(1)), Mat3x3(1)),
        kronecker(rowmat(D.col(2)), Mat3x3(1)) };
    Vec9 X;
    for (int i = 0; i < 9; i++)
        X[i] = x[i / 3][i % 3];
    Vec3 f[3] = { DD[0] * X, DD[1] * X, DD[2] * X };

    Vec9 grad_f(0);
    Mat9x9 hess_f(0);
    if (mat->use_dde) {
        Vec4 k = stretching_stiffness(reduce_xy(G), mat->dde_stretching) * weakening_mult;

        const Mat<3, 9>& Du = DD[0];
        const Mat<3, 9>& Dv = DD[1];
        Vec9 fuu = Du.t() * f[0], fvv = Dv.t() * f[1], fuv = (Du.t() * f[1] + Dv.t() * f[0]) / 2.;
        grad_f = k[0] * G(0, 0) * fuu + k[2] * G(1, 1) * fvv
            + k[1] * (G(0, 0) * fvv + G(1, 1) * fuu) + 2 * k[3] * G(0, 1) * fuv;
        hess_f = k[0] * (outer(fuu, fuu) + max(G(0, 0), 0.) * Du.t() * Du)
            + k[2] * (outer(fvv, fvv) + max(G(1, 1), 0.) * Dv.t() * Dv)
            + k[1] * (outer(fuu, fvv) + max(G(0, 0), 0.) * Dv.t() * Dv
                         + outer(fvv, fuu) + max(G(1, 1), 0.) * Du.t() * Du)
            + 2. * k[3] * (outer(fuv, fuv));
        // ignoring G(0,1)*(Du.t()*Dv+Dv.t()*Du)/2. term
        // because may not be positive definite
        return make_pair(-face->a * hess_f, -face->a * grad_f);
    } else {
        double gf = -face->a * mat->alt_stretching;
        //Vec9 d_trace = 0.5 * (DD[0].t()*DD[0]+DD[1].t()*DD[1]+DD[2].t()*DD[2]) * X;
        Mat3x3 Gc = max(G, Mat3x3(0)); // posdef
        for (int i = 0; i < 3; i++)
            for (int j = 0; j <= i; j++) {
                Vec9 dG = 0.5 * (DD[i].t() * f[j] + DD[j].t() * f[i]);
                Mat9x9 dG2 = 0.5 * (DD[i].t() * DD[j] + DD[j].t() * DD[i]);
                Vec9 d_trace_c = 0.5 * dG; // posdef

                if (i == j)
                    grad_f += gf * (1.0 - mat->alt_poisson) * G(i, j) * dG + gf * mat->alt_poisson * trace(G) * dG;
                else
                    grad_f += 2.0 * gf * (1.0 - mat->alt_poisson) * G(i, j) * dG;

                if (i == j)
                    hess_f += gf * (1.0 - mat->alt_poisson) * (outer(dG, dG) + Gc(i, j) * dG2) + gf * mat->alt_poisson * (trace(Gc) * dG2 + outer(d_trace_c, dG));
                else
                    hess_f += 2.0 * gf * (1.0 - mat->alt_poisson) * (outer(dG, dG) + Mat9x9(0)); // posdef
            }
        return make_pair(hess_f, grad_f);
    }
}
//dde.cpp
Vec4 stretching_stiffness(const Mat2x2& G, const StretchingSamples& samples)
{
    double a = (G(0, 0) + 0.25) * nsamples;
    double b = (G(1, 1) + 0.25) * nsamples;
    double c = fabsf(G(0, 1)) * nsamples;
    a = clamp(a, 0.0, nsamples - 1 - 1e-5);
    b = clamp(b, 0.0, nsamples - 1 - 1e-5);
    c = clamp(c, 0.0, nsamples - 1 - 1e-5);
    int ai = (int)floor(a);
    int bi = (int)floor(b);
    int ci = (int)floor(c);
    if (ai < 0)
        ai = 0;
    if (bi < 0)
        bi = 0;
    if (ci < 0)
        ci = 0;
    if (ai > nsamples - 2)
        ai = nsamples - 2;
    if (bi > nsamples - 2)
        bi = nsamples - 2;
    if (ci > nsamples - 2)
        ci = nsamples - 2;
    a = a - ai;
    b = b - bi;
    c = c - ci;
    double weight[2][2][2];
    weight[0][0][0] = (1 - a) * (1 - b) * (1 - c);
    weight[0][0][1] = (1 - a) * (1 - b) * (c);
    weight[0][1][0] = (1 - a) * (b) * (1 - c);
    weight[0][1][1] = (1 - a) * (b) * (c);
    weight[1][0][0] = (a) * (1 - b) * (1 - c);
    weight[1][0][1] = (a) * (1 - b) * (c);
    weight[1][1][0] = (a) * (b) * (1 - c);
    weight[1][1][1] = (a) * (b) * (c);
    Vec4 stiffness = Vec4(0);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                for (int l = 0; l < 4; l++) {
                    stiffness[l] += samples.s[ai + i][bi + j][ci + k][l] * weight[i][j][k];
                }
    return stiffness;
}
Vec4 evaluate_stretching_sample(const Mat2x2& _G, const StretchingData& data)
{
    Mat2x2 G = _G;
    G = G * 2. + Mat2x2(1);
    Eig<2> eig = eigen_decomposition(G);
    Vec2 w = Vec2(sqrt(eig.l[0]), sqrt(eig.l[1]));
    Vec2 V = eig.Q.col(0);
    double angle_weight = fabsf(atan2f(V[1], V[0]) / M_PI) * 8;
    if (angle_weight < 0)
        angle_weight = 0;
    if (angle_weight > 4 - 1e-6)
        angle_weight = 4 - 1e-6;
    int angle_id = (int)angle_weight;
    angle_weight = angle_weight - angle_id;
    double strain_value = (w[0] - 1) * 6;
    if (strain_value < 0)
        strain_value = 0;
    if (strain_value > 1 - 1e-6)
        strain_value = 1 - 1e-6;
    int strain_id = (int)strain_value;
    //if(strain_id>1)               strain_id=1;
    strain_id = 0;
    double strain_weight = strain_value - strain_id;
    Vec4 real_elastic;
    real_elastic = data.d[strain_id][angle_id] * (1 - strain_weight) * (1 - angle_weight) + data.d[strain_id + 1][angle_id] * (strain_weight) * (1 - angle_weight) + data.d[strain_id][angle_id + 1] * (1 - strain_weight) * (angle_weight) + data.d[strain_id + 1][angle_id + 1] * (strain_weight) * (angle_weight);
    if (real_elastic[0] < 0)
        real_elastic[0] = 0;
    if (real_elastic[1] < 0)
        real_elastic[1] = 0;
    if (real_elastic[2] < 0)
        real_elastic[2] = 0;
    if (real_elastic[3] < 0)
        real_elastic[3] = 0;
    real_elastic[0] *= 2;
    real_elastic[1] *= 2;
    real_elastic[2] *= 2;
    real_elastic[3] *= 2;
    return real_elastic;
}
void evaluate_stretching_samples(StretchingSamples& samples,
    const StretchingData& data)
{
    for (int i = 0; i < ::nsamples; i++)
        for (int j = 0; j < ::nsamples; j++)
            for (int k = 0; k < ::nsamples; k++) {
                Mat2x2 G;
                G(0, 0) = -0.25 + i / (::nsamples * 1.0);
                G(1, 1) = -0.25 + j / (::nsamples * 1.0);
                G(0, 1) = G(1, 0) = k / (::nsamples * 1.0);
                samples.s[i][j][k] = evaluate_stretching_sample(G, data);
            }
}