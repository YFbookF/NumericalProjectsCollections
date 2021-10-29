//physics.cpp
Mat3x3 material_model(const Face* face, const Mat3x3& G)
{
    const Material* mat = face->material;
    double weakening_mult = 1 / (1 + mat->weakening * face->damage);
    if (mat->use_dde) {
        Vec4 k = stretching_stiffness(reduce_xy(G), mat->dde_stretching) * weakening_mult;
        Mat3x3 sigma(Vec3(k[0] * G(0, 0) + k[1] * G(1, 1), 0.5 * k[3] * G(0, 1), 0),
            Vec3(0.5 * k[3] * G(1, 0), k[2] * G(1, 1) + k[1] * G(0, 0), 0),
            Vec3(0, 0, 0));
        return sigma;
    } else {
        // Saint Venant-Kirchloff model
        double A = mat->alt_stretching * weakening_mult;
        Mat3x3 S = A * (1.0 - mat->alt_poisson) * G + Mat3x3(A * mat->alt_poisson * trace(G));
        return S;
    }
}