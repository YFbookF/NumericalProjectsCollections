
// cache DmInv at startup
std::array<Matrix8x3,8> ComputeHDmInv(const Matrix3x8& Xbar)
{
    std::array<Matrix8x3,8> HDmInv;

    for (int x = 0; x < 8; x++)
    {
        // compute Dm
        const Matrix8x3 H = ComputeH(restGaussPoints[x]);
        const Matrix3 Dm = Xbar * H;
        const Matrix3 DmInverse = Dm.inverse();

        // cache it left-multiplied by H
        HDmInv[x] = H * DmInverse;
    }

    return HDmInv;
}

// Shape function derivative matrix
static Matrix8x3 ComputeH(const Vector3& b)
{
    const Scalar b0Minus = (1.0 - b[0]);
    const Scalar b0Plus  = (1.0 + b[0]);
    const Scalar b1Minus = (1.0 - b[1]);
    const Scalar b1Plus  = (1.0 + b[1]);
    const Scalar b2Minus = (1.0 - b[2]);
    const Scalar b2Plus  = (1.0 + b[2]);

    Matrix8x3 H;

    H(0,0) = -b1Minus * b2Minus;
    H(0,1) = -b0Minus * b2Minus;
    H(0,2) = -b0Minus * b1Minus;

    H(1,0) =  b1Minus * b2Minus;
    H(1,1) = -b0Plus * b2Minus;
    H(1,2) = -b0Plus * b1Minus;

    H(2,0) = -b1Plus * b2Minus;
    H(2,1) =  b0Minus * b2Minus;
    H(2,2) = -b0Minus * b1Plus;

    H(3,0) =  b1Plus * b2Minus;
    H(3,1) =  b0Plus * b2Minus;
    H(3,2) = -b0Plus * b1Plus;

    H(4,0) = -b1Minus * b2Plus;
    H(4,1) = -b0Minus * b2Plus;
    H(4,2) =  b0Minus * b1Minus;

    H(5,0) =  b1Minus * b2Plus;
    H(5,1) = -b0Plus * b2Plus;
    H(5,2) =  b0Plus * b1Minus;

    H(6,0) = -b1Plus * b2Plus;
    H(6,1) =  b0Minus * b2Plus;
    H(6,2) =  b0Minus * b1Plus;

    H(7,0) =  b1Plus * b2Plus;
    H(7,1) =  b0Plus * b2Plus;
    H(7,2) =  b0Plus * b1Plus;

    return H / 8.0;
}
