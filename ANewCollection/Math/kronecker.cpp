//acrsim
// A kronecker B = [a11 B, a12 B, ..., a1n B;
//                  a21 B, a22 B, ..., a2n B;
//                   ... ,  ... , ...,  ... ;
//                  am1 B, am2 B, ..., amn B]
template <int m, int n, int p, int q>
Mat<m * p, n * q> kronecker(const Mat<m, n>& A, const Mat<p, q>& B)
{
    Mat<m * p, n * q> C;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < p; k++)
                for (int l = 0; l < q; l++)
                    C(i * p + k, j * q + l) = A(i, j) * B(k, l);
    return C;
}