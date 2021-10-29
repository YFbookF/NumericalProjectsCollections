//vectors.hpp
template <>
Vec2 eigen_values<2>(const Mat2x2& A)
{
    double a = A(0, 0), b = A(1, 0), d = A(1, 1); // A(1,0) == A(0,1)
    double amd = a - d;
    double apd = a + d;
    double b2 = b * b;
    double det = sqrt(4 * b2 + amd * amd);
    double l1 = 0.5 * (apd + det);
    double l2 = 0.5 * (apd - det);
    return Vec2(l1, l2);
}

template <>
Eig<2> eigen_decomposition<2>(const Mat2x2& A)
{
    // http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/index.html
    // http://en.wikipedia.org/wiki/Eigenvalue_algorithm
    Eig<2> eig;
    double a = A(0, 0), b = A(1, 0), d = A(1, 1); // A(1,0) == A(0,1)
    double amd = a - d;
    double apd = a + d;
    double b2 = b * b;
    double det = sqrt(4 * b2 + amd * amd);
    double l1 = 0.5 * (apd + det);
    double l2 = 0.5 * (apd - det);

    eig.l[0] = l1;
    eig.l[1] = l2;

    double v0, v1, vn;
    if (b) {
        v0 = l1 - d;
        v1 = b;
        vn = sqrt(v0 * v0 + b2);
        eig.Q(0, 0) = v0 / vn;
        eig.Q(1, 0) = v1 / vn;

        v0 = l2 - d;
        vn = sqrt(v0 * v0 + b2);
        eig.Q(0, 1) = v0 / vn;
        eig.Q(1, 1) = v1 / vn;
    } else if (a >= d) {
        eig.Q(0, 0) = 1;
        eig.Q(1, 0) = 0;
        eig.Q(0, 1) = 0;
        eig.Q(1, 1) = 1;
    } else {
        eig.Q(0, 0) = 0;
        eig.Q(1, 0) = 1;
        eig.Q(0, 1) = 1;
        eig.Q(1, 1) = 0;
    }

    return eig;
}
// http://www.mpi-hd.mpg.de/personalhomes/globes/3x3
template <>
Eig<3> eigen_decomposition<3>(const Mat3x3& B)
{
    Eig<3> e;
    Mat3x3& Q = e.Q;
    Mat3x3 A = B;

    double norm; // Squared norm or inverse norm of current eigenvector
    double n0, n1; // Norm of first and second columns of A
    double n0tmp, n1tmp; // "Templates" for the calculation of n0/n1 - saves a few FLOPS
    double thresh; // Small number used as threshold for floating point comparisons
    double error; // Estimated maximum roundoff error in some steps
    double wmax; // The eigenvalue of maximum modulus
    double f, t; // Intermediate storage
    int i, j; // Loop counters

    // Calculate eigenvalues
    dsyevc3(A, e.l);

    wmax = fabs(e.l[0]);
    if ((t = fabs(e.l[1])) > wmax)
        wmax = t;
    if ((t = fabs(e.l[2])) > wmax)
        wmax = t;
    thresh = sqr(8.0 * DBL_EPSILON * wmax);

    // Prepare calculation of eigenvectors
    n0tmp = sqr(A(0, 1)) + sqr(A(0, 2));
    n1tmp = sqr(A(0, 1)) + sqr(A(1, 2));
    Q(0, 1) = A(0, 1) * A(1, 2) - A(0, 2) * A(1, 1);
    Q(1, 1) = A(0, 2) * A(0, 1) - A(1, 2) * A(0, 0);
    Q(2, 1) = sqr(A(0, 1));

    // Calculate first eigenvector by the formula
    //   v[0] = (A - e.l[0]).e1 x (A - e.l[0]).e2
    A(0, 0) -= e.l[0];
    A(1, 1) -= e.l[0];
    Q(0, 0) = Q(0, 1) + A(0, 2) * e.l[0];
    Q(1, 0) = Q(1, 1) + A(1, 2) * e.l[0];
    Q(2, 0) = A(0, 0) * A(1, 1) - Q(2, 1);
    norm = sqr(Q(0, 0)) + sqr(Q(1, 0)) + sqr(Q(2, 0));
    n0 = n0tmp + sqr(A(0, 0));
    n1 = n1tmp + sqr(A(1, 1));
    error = n0 * n1;

    if (n0 <= thresh) // If the first column is zero, then (1,0,0) is an eigenvector
    {
        Q(0, 0) = 1.0;
        Q(1, 0) = 0.0;
        Q(2, 0) = 0.0;
    } else if (n1 <= thresh) // If the second column is zero, then (0,1,0) is an eigenvector
    {
        Q(0, 0) = 0.0;
        Q(1, 0) = 1.0;
        Q(2, 0) = 0.0;
    } else if (norm < sqr(64.0 * DBL_EPSILON) * error) { // If angle between A[0] and A[1] is too small, don't use
        t = sqr(A(0, 1)); // cross product, but calculate v ~ (1, -A0/A1, 0)
        f = -A(0, 0) / A(0, 1);
        if (sqr(A(1, 1)) > t) {
            t = sqr(A(1, 1));
            f = -A(0, 1) / A(1, 1);
        }
        if (sqr(A(1, 2)) > t)
            f = -A(0, 2) / A(1, 2);
        norm = 1.0 / sqrt(1 + sqr(f));
        Q(0, 0) = norm;
        Q(1, 0) = f * norm;
        Q(2, 0) = 0.0;
    } else // This is the standard branch
    {
        norm = sqrt(1.0 / norm);
        for (j = 0; j < 3; j++)
            Q(j, 0) = Q(j, 0) * norm;
    }

    // Prepare calculation of second eigenvector
    t = e.l[0] - e.l[1];
    if (fabs(t) > 8.0 * DBL_EPSILON * wmax) {
        // For non-degenerate eigenvalue, calculate second eigenvector by the formula
        //   v[1] = (A - e.l[1]).e1 x (A - e.l[1]).e2
        A(0, 0) += t;
        A(1, 1) += t;
        Q(0, 1) = Q(0, 1) + A(0, 2) * e.l[1];
        Q(1, 1) = Q(1, 1) + A(1, 2) * e.l[1];
        Q(2, 1) = A(0, 0) * A(1, 1) - Q(2, 1);
        norm = sqr(Q(0, 1)) + sqr(Q(1, 1)) + sqr(Q(2, 1));
        n0 = n0tmp + sqr(A(0, 0));
        n1 = n1tmp + sqr(A(1, 1));
        error = n0 * n1;

        if (n0 <= thresh) // If the first column is zero, then (1,0,0) is an eigenvector
        {
            Q(0, 1) = 1.0;
            Q(1, 1) = 0.0;
            Q(2, 1) = 0.0;
        } else if (n1 <= thresh) // If the second column is zero, then (0,1,0) is an eigenvector
        {
            Q(0, 1) = 0.0;
            Q(1, 1) = 1.0;
            Q(2, 1) = 0.0;
        } else if (norm < sqr(64.0 * DBL_EPSILON) * error) { // If angle between A[0] and A[1] is too small, don't use
            t = sqr(A(0, 1)); // cross product, but calculate v ~ (1, -A0/A1, 0)
            f = -A(0, 0) / A(0, 1);
            if (sqr(A(1, 1)) > t) {
                t = sqr(A(1, 1));
                f = -A(0, 1) / A(1, 1);
            }
            if (sqr(A(1, 2)) > t)
                f = -A(0, 2) / A(1, 2);
            norm = 1.0 / sqrt(1 + sqr(f));
            Q(0, 1) = norm;
            Q(1, 1) = f * norm;
            Q(2, 1) = 0.0;
        } else {
            norm = sqrt(1.0 / norm);
            for (j = 0; j < 3; j++)
                Q(j, 1) = Q(j, 1) * norm;
        }
    } else {
        // For degenerate eigenvalue, calculate second eigenvector according to
        //   v[1] = v[0] x (A - e.l[1]).e[i]
        //
        // This would really get to complicated if we could not assume all of A to
        // contain meaningful values.
        A(1, 0) = A(0, 1);
        A(2, 0) = A(0, 2);
        A(2, 1) = A(1, 2);
        A(0, 0) += e.l[0];
        A(1, 1) += e.l[0];
        for (i = 0; i < 3; i++) {
            A(i, i) -= e.l[1];
            n0 = sqr(A(0, i)) + sqr(A(1, i)) + sqr(A(2, i));
            if (n0 > thresh) {
                Q(0, 1) = Q(1, 0) * A(2, i) - Q(2, 0) * A(1, i);
                Q(1, 1) = Q(2, 0) * A(0, i) - Q(0, 0) * A(2, i);
                Q(2, 1) = Q(0, 0) * A(1, i) - Q(1, 0) * A(0, i);
                norm = sqr(Q(0, 1)) + sqr(Q(1, 1)) + sqr(Q(2, 1));
                if (norm > sqr(256.0 * DBL_EPSILON) * n0) // Accept cross product only if the angle between
                { // the two vectors was not too small
                    norm = sqrt(1.0 / norm);
                    for (j = 0; j < 3; j++)
                        Q(j, 1) = Q(j, 1) * norm;
                    break;
                }
            }
        }

        if (i == 3) // This means that any vector orthogonal to v[0] is an EV.
        {
            for (j = 0; j < 3; j++)
                if (Q(j, 0) != 0.0) // Find nonzero element of v[0] ...
                { // ... and swap it with the next one
                    norm = 1.0 / sqrt(sqr(Q(j, 0)) + sqr(Q((j + 1) % 3, 0)));
                    Q(j, 1) = Q((j + 1) % 3, 0) * norm;
                    Q((j + 1) % 3, 1) = -Q(j, 0) * norm;
                    Q((j + 2) % 3, 1) = 0.0;
                    break;
                }
        }
    }

    // Calculate third eigenvector according to
    //   v[2] = v[0] x v[1]
    Q(0, 2) = Q(1, 0) * Q(2, 1) - Q(2, 0) * Q(1, 1);
    Q(1, 2) = Q(2, 0) * Q(0, 1) - Q(0, 0) * Q(2, 1);
    Q(2, 2) = Q(0, 0) * Q(1, 1) - Q(1, 0) * Q(0, 1);

    // sort eigenvectors
    if (e.l[1] > e.l[0]) {
        swap(e.Q.col(0), e.Q.col(1));
        swap(e.l[0], e.l[1]);
    }
    if (e.l[2] > e.l[0]) {
        swap(e.Q.col(0), e.Q.col(2));
        swap(e.l[0], e.l[2]);
    }
    if (e.l[2] > e.l[1]) {
        swap(e.Q.col(1), e.Q.col(2));
        swap(e.l[1], e.l[2]);
    }

    return e;
}
