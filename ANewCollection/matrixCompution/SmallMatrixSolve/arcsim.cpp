//vectors.hpp
template <>
Vec2 solve_symmetric(const Mat2x2& A, const Vec2& b)
{
    double div = sq(A(0, 1)) - A(1, 1) * A(0, 0);
    if (fabs(div) < 1e-14) {
        cout << A << endl;
        cout << div << endl;
        cout << "singular matrix" << endl;
        exit(1);
    }
    return Vec2(b[1] * A(0, 1) - b[0] * A(1, 1), b[0] * A(0, 1) - b[1] * A(0, 0)) / div;
}

template <>
Vec3 solve_symmetric(const Mat3x3& A, const Vec3& b)
{
    double t13 = A(1, 2) * A(1, 2);
    double t14 = A(0, 2) * A(0, 2);
    double t15 = A(0, 0) * t13;
    double t16 = A(1, 1) * t14;
    double t17 = A(0, 1) * A(0, 1);
    double t18 = A(2, 2) * t17;
    double t21 = A(0, 1) * A(0, 2) * A(1, 2) * 2.0;
    double t22 = A(0, 0) * A(1, 1) * A(2, 2);
    double t19 = t15 + t16 + t18 - t21 - t22;
    if (fabs(t19) == 0) {
        cout << A << endl
             << "singular matrix" << endl;
        exit(1);
    }
    double t20 = 1.0 / t19;
    return Vec3(t20 * (t13 * b[0] + A(0, 2) * (A(1, 1) * b[2] - A(1, 2) * b[1]) - A(0, 1) * (A(1, 2) * b[2] - A(2, 2) * b[1]) - A(1, 1) * A(2, 2) * b[0]),
        t20 * (t14 * b[1] + A(1, 2) * (A(0, 0) * b[2] - A(0, 2) * b[0]) - A(0, 1) * (A(0, 2) * b[2] - A(2, 2) * b[0]) - A(0, 0) * A(2, 2) * b[1]),
        t20 * (t17 * b[2] + A(1, 2) * (A(0, 0) * b[1] - A(0, 1) * b[0]) - A(0, 2) * (A(0, 1) * b[1] - A(1, 1) * b[0]) - A(0, 0) * A(1, 1) * b[2]));
}

template <int m, int n>
Vec<n> solve_llsq(const Mat<m, n>& A, const Vec<m>& b)
{
    Mat<n, n> M;
    Vec<n> y;
    for (int i = 0; i < n; i++) {
        y[i] = dot(b, A.col(i));
        for (int j = 0; j < n; j++)
            M(i, j) = dot(A.col(i), A.col(j));
    }
    if (norm_F(M) == 0) {
        cout << "llsq: normF = 0 " << endl;
        exit(1);
    }
    return solve_symmetric(M, y);
}