mpmngf
inline MatrixND<2, real> dR_from_dF(const MatrixND<2, real> &F,
                                    const MatrixND<2, real> &R,
                                    const MatrixND<2, real> &S,
                                    const MatrixND<2, real> &dF)
{
  using Matrix = MatrixND<2, real>;
  using Vector = VectorND<2, real>;

  // set W = R^T dR = [  0    x  ]
  //                  [  -x   0  ]
  //
  // R^T dF - dF^T R = WS + SW
  //
  // WS + SW = [ x(s21 - s12)   x(s11 + s22) ]
  //           [ -x[s11 + s22]  x(s21 - s12) ]
  // ----------------------------------------------------
  Matrix lhs = transposed(R) * dF - transposed(dF) * R;
  real x;
  real abs0 = abs(S[0][0] + S[1][1]);
  real abs1 = abs(S[0][1] - S[1][0]);
  if (abs0 > abs1)
  {
    x = lhs[1][0] / (S[0][0] + S[1][1]);
  }
  else
  {
    x = lhs[0][0] / (S[0][1] - S[1][0]);
  }
  Matrix W = Matrix(Vector(0, -x), Vector(x, 0));
  return R * W;
};