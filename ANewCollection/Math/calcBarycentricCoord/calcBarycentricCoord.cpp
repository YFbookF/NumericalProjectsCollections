https://github.com/yuki-koyama/real-time-example-based-elastic-deformation
        template <typename Scalar, int NDim, int NSimplex>
        static Vector<Scalar, NSimplex> calcBarycentricCoord(const Vector<Scalar, NDim>  v[NSimplex],
                                                             const Vector<Scalar, NDim>& w,
                                                             Scalar                      coord_sum = 1)
        {
            assert(NSimplex <= NDim + 1);
            /*
             [input]
             v[NSimplex]: simplex vertices
             w: target vector
             [output]
             d[NSimplex]: barycentric coordinate for each simplex vertex
             [equation]----------------------------------------------
             d0 * v0 + ... + dNSimplex * vNSimplex =~ w           (desired target)
             d0      + ... + dNSimplex             = coord_sum       (hard constraint)
             [solution via Lagrangian multiplier]--------------------------
             AtA(ij) = dot(v[i], v[j])
             Atb(i) = dot(v[i], w)
             c = [1 ... 1]
             | AtA ct | = | Atb        |
             | c    0 |   |  coord_sum |
             */
            Matrix<Scalar, NSimplex + 1, NSimplex + 1> A(0.0);
            Vector<Scalar, NSimplex + 1>               b, x;
            for (int i = 0; i < NSimplex; ++i)
            {
                for (int j = i + 1; j < NSimplex; ++j)
                    A(i, j) = A(j, i) = v[i] | v[j];
                A(i, i)        = v[i].lengthSquared();
                A(i, NSimplex) = A(NSimplex, i) = 1;
                b[i]                            = v[i] | w;
            }
            b[NSimplex] = coord_sum;
            A.solve(b, x);
            Vector<Scalar, NSimplex> result;
            memcpy(result.ptr(), x.ptr(), sizeof(Scalar) * NSimplex);
            return result;
        }
    }; // class Util