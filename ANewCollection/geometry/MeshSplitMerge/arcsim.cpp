//plasticity.cpp
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

Vec3 face_to_edges(const Mat3x3& S, const Face* face)
{
    const Vec3 n = normal<MS>(face);
    Mat<6, 3> A;

    for (int e = 0; e < 3; e++) {
        Vec3 e_mat = face->v[PREV(e)]->u - face->v[NEXT(e)]->u,
             t_mat = cross(normalize(e_mat), n);
        Mat3x3 Se = -1 / 2. * norm(e_mat) * outer(t_mat, t_mat);
        A.col(e) = Vec<6>(Se(0, 0), Se(1, 1), Se(2, 2), Se(0, 1), Se(0, 2), Se(1, 2));
    }

    Vec<6> y = face->a * Vec<6>(S(0, 0), S(1, 1), S(2, 2), S(0, 1), S(0, 2), S(1, 2));
    return solve_llsq(A, y);
}