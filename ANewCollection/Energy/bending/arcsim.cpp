// plasticity.cpp
void recompute_Sp_bend(Face* face)
{
    Vec3 theta;
    for (int e = 0; e < 3; e++)
        theta[e] = face->adje[e]->theta_ideal;
    face->Sp_bend = edges_to_face(theta, face); // add residual and PS reconstruction
}
Mat3x3 edges_to_face(const Vec3& theta, const Face* face)
{
    Mat3x3 S;
    Vec3 n = normal<MS>(face);
    for (int e = 0; e < 3; e++) {
        //const Edge *edge = face->adje[e];
        Vec3 e_mat = face->v[PREV(e)]->u - face->v[NEXT(e)]->u,
             t_mat = cross(normalize(e_mat), n);
        S -= 1 / 2. * theta[e] * norm(e_mat) * outer(t_mat, t_mat);
    }
    S /= face->a;
    return S;
}
//physics.cpp
double bending_coeff(const Edge* edge, double theta)
{
    const Face *face0 = edge->adjf[0], *face1 = edge->adjf[1];
    double a = face0->a + face1->a;
    double l = norm(edge->n[1]->x - edge->n[0]->x);

    double ke0 = (face0->material->use_dde)
        ? bending_stiffness(edge, 0, face0->material->dde_bending, l, theta)
        : face0->material->alt_bending;
    double ke1 = (face1->material->use_dde)
        ? bending_stiffness(edge, 1, face1->material->dde_bending, l, theta)
        : face1->material->alt_bending;

    double ke = min(ke0, ke1);
    double weakening = max(face0->material->weakening, face1->material->weakening);
    ke *= 1 / (1 + weakening * edge->damage);
    double shape = sq(l) / (2 * a);

    return ke * shape;
}

template <Space s>
double bending_energy(const Edge* edge)
{
    if (!edge->adjf[0] || !edge->adjf[1])
        return 0;

    double theta = dihedral_angle<s>(edge);
    return bending_coeff(edge, theta) * sq(theta - edge->theta_ideal) / 4;
}

template <Space s>
pair<Mat12x12, Vec12> bending_force(const Edge* edge)
{
    const Face *face0 = edge->adjf[0], *face1 = edge->adjf[1];
    if (!face0 || !face1)
        return make_pair(Mat12x12(0), Vec12(0));
    double theta = dihedral_angle<s>(edge);
    Node* op0 = edge_opp_vert(edge, 0)->node;
    Node* op1 = edge_opp_vert(edge, 1)->node;
    Vec3 x0 = pos<s>(edge->n[0]),
         x1 = pos<s>(edge->n[1]),
         x2 = pos<s>(op0),
         x3 = pos<s>(op1);
    double h0 = distance(x2, x0, x1), h1 = distance(x3, x0, x1);
    Vec3 n0 = normal<s>(face0), n1 = normal<s>(face1);
    Vec2 w_f0 = barycentric_weights(x2, x0, x1),
         w_f1 = barycentric_weights(x3, x0, x1);
    Vec12 u = mat_to_vec(Mat3x4(
        -(w_f0[0] * n0 / h0 + w_f1[0] * n1 / h1),
        -(w_f0[1] * n0 / h0 + w_f1[1] * n1 / h1),
        n0 / h0,
        n1 / h1));

    double coeff = bending_coeff(edge, theta);
    double dtheta = theta - edge->theta_ideal;

    // if (op0->index == debug_node || op1->index == debug_node) {
    // if (1) {
    //     double l = norm(edge->n[1]->x - edge->n[0]->x);
    //     double f = norm(n0 / h0 * coeff * dtheta / 2.);
    //     printf("%04d/%04d l: %e  coeff: %e  norm(u): %e  dtheta: %e  f: %e\n",
    //         op0->index, op1->index, l, coeff, norm(n0 / h0), dtheta, f);
    // }

    // // return make_pair(
    // //     -coeff * outer(u, u) * abs(dtheta) * 1e5,
    // //     -coeff * u * sgnsqr(dtheta) / 2. * 1e5);
    // return make_pair(
    //     -coeff * outer(u, u) / 2.  * 0.5,
    //     -coeff * u * sgnsqr(dtheta) / 2. * 1e5);

    // From "Simulation of Clothing with Folds and Wrinkles"

    return make_pair(
        -coeff * outer(u, u) / 2.,
        -coeff * u * dtheta / 2.);
}
//sepstrength.cpp
Mat3x3 compute_sigma(const Face* face)
{ /*
	Mat3x3 F_str = deformation_gradient<WS>(face);
    Mat3x3 G_str = (F_str.t()*F_str - Mat3x3(1)) * 0.5;
    Mat3x3 sigma_str = material_model(face, G_str);

    // add explicit bending
    Mat3x3 F_bend = Mat3x3(1);
    for (int i=0; i<3; i++) 
    	F_bend += face->v[i]->node->curvature * (0.5 * face->material->fracture_bend_thickness);
    Mat3x3 G_bend = (F_bend.t()*F_bend - Mat3x3(1)) * 0.5;
    Mat3x3 sigma_bend = material_model(face, G_bend);

    return get_positive(sigma_str) + get_positive(sigma_bend);*/
    Mat3x3 F_str = deformation_gradient<WS>(face);
    //Mat3x3 G_str = (F_str.t()*F_str - Mat3x3(1)) * 0.5;
    //Mat3x3 sigma_str = material_model(face, G_str);

    // add explicit bending
    Mat3x3 F_bend = Mat3x3(1);
    for (int i = 0; i < 3; i++)
        F_bend += face->v[i]->node->curvature * (0.5 * face->material->fracture_bend_thickness);
    F_bend = F_str * F_bend;
    Mat3x3 G_bend = (F_bend.t() * F_bend - Mat3x3(1)) * 0.5;
    Mat3x3 sigma_bend = material_model(face, G_bend);

    return get_positive(sigma_bend);
}
//dde.cpp
double bending_stiffness(const Edge* edge, int side,
    const BendingData& data, double l, double theta, double initial_angle)
{
    double curv = theta * l / (edge->adjf[0]->a + edge->adjf[1]->a);
    double alpha = curv / 2;
    double value = alpha * 0.2; // because samples are per 0.05 cm^-1 = 5 m^-1
    if (value > 4)
        value = 4;
    int value_i = (int)value;
    if (value_i < 0)
        value_i = 0;
    if (value_i > 3)
        value_i = 3;
    value -= value_i;
    Vec3 du = edge_vert(edge, side, 1)->u - edge_vert(edge, side, 0)->u;
    double bias_angle = (atan2f(du[1], du[0]) + initial_angle) * 4 / M_PI;
    if (bias_angle < 0)
        bias_angle = -bias_angle;
    if (bias_angle > 4)
        bias_angle = 8 - bias_angle;
    if (bias_angle > 2)
        bias_angle = 4 - bias_angle;
    int bias_id = (int)bias_angle;
    if (bias_id < 0)
        bias_id = 0;
    if (bias_id > 1)
        bias_id = 1;
    bias_angle -= bias_id;
    double actual_ke = data.d[bias_id][value_i] * (1 - bias_angle) * (1 - value)
        + data.d[bias_id + 1][value_i] * (bias_angle) * (1 - value)
        + data.d[bias_id][value_i + 1] * (1 - bias_angle) * (value)
        + data.d[bias_id + 1][value_i + 1] * (bias_angle) * (value);
    if (actual_ke < 0)
        actual_ke = 0;
    return actual_ke;
}
