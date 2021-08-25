Hierarchical Optimization Time Integration for CFL-rate MPM Stepping

```
// strain s is deformation F
template <class T, int dim>
template <class TConst>
bool VonMisesFixedCorotated<T, dim>::projectStrain(TConst& c, Matrix<T, TConst::dim, TConst::dim>& strain)
{
    static_assert(dim == TConst::dim, "Plasticity model has a different dimension as the Constitutive model!");
    ZIRAN_ASSERT(yield_stress >= 0);
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    using MATH_TOOLS::sqr;

    TM U, V;
    TV sigma;
    singularValueDecomposition(strain, U, sigma, V);
    for (int d = 0; d < dim; d++) sigma(d) = std::max((T)1e-4, sigma(d));
    T J = sigma.prod();
    TV tau_trial;
    for (int d = 0; d < dim; d++) tau_trial(d) = 2 * c.mu * (sigma(d) - 1) * sigma(d) + c.lambda * (J - 1) * J;
    T trace_tau = tau_trial.sum();
    TV s_trial = tau_trial - TV::Ones() * (trace_tau / (T)dim);
    T s_norm = s_trial.norm();
    T scaled_tauy = std::sqrt((T)2 / ((T)6 - dim)) * yield_stress;
    if (s_norm - scaled_tauy <= 0) return false;
    T alpha = scaled_tauy / s_norm;
    TV s_new = alpha * s_trial;
    TV tau_new = s_new + TV::Ones() * (trace_tau / (T)dim);
    TV sigma_new;
    for (int d = 0; d < dim; d++) {
        T b2m4ac = sqr(c.mu) - 2 * c.mu * (c.lambda * (J - 1) * J - tau_new(d));
        ZIRAN_ASSERT(b2m4ac >= 0, "Wrong projection ", b2m4ac);
        T sqrtb2m4ac = std::sqrt(b2m4ac);
        T x1 = (c.mu + sqrtb2m4ac) / (2 * c.mu);
        // T x2 = (c.mu - sqrtb2m4ac) / (2 * c.mu);
        // ZIRAN_ASSERT(sqr(x1 - sigma(d)) <= sqr(x2 - sigma(d)));
        sigma_new(d) = x1;
    }
    strain = U * sigma_new.asDiagonal() * V.transpose();
    return true;
}
```

或者是

```
bool SnowPlasticity<T>::projectStrain(TConst& c, Matrix<T, TConst::dim, TConst::dim>& strain)
{
    static const int dim = TConst::dim;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    TM U, V;
    TV sigma;

    // TODO: this is inefficient because next time step updateState will do the svd again!
    singularValueDecomposition(strain, U, sigma, V);

    T Fe_det = (T)1;
    for (int i = 0; i < dim; i++) {
        sigma(i) = std::max(std::min(sigma(i), (T)1 + theta_s), (T)1 - theta_c);
        Fe_det *= sigma(i);
    }

    Eigen::DiagonalMatrix<T, dim, dim> sigma_m(sigma);
    TM Fe = U * sigma_m * V.transpose();
    // T Jp_new = std::max(std::min(Jp * strain.determinant() / Fe_det, max_Jp), min_Jp);
    T Jp_new = Jp * strain.determinant() / Fe_det;
    if (!(Jp_new <= max_Jp))
        Jp_new = max_Jp;
    if (!(Jp_new >= min_Jp))
        Jp_new = min_Jp;

    strain = Fe;
    c.mu *= std::exp(psi * (Jp - Jp_new));
    c.lambda *= std::exp(psi * (Jp - Jp_new));
    Jp = Jp_new;

    return false;
}
```

piola stress

```
template <class T, int _dim>
T LinearCorotated<T, _dim>::psi(const Scratch& s) const
{
    return mu * s.e_hat.squaredNorm() + lambda * 0.5 * s.trace_e_hat * s.trace_e_hat;
}

template <class T, int _dim>
void LinearCorotated<T, _dim>::firstPiola(const Scratch& s, TM& P) const
{
    P.noalias() = (T)2 * mu * R * s.e_hat + lambda * s.trace_e_hat * R;
}

template <class T, int _dim>
void LinearCorotated<T, _dim>::firstPiolaDifferential(const Scratch& s, const TM& dF, TM& dP) const
{
    dP.noalias() = mu * dF + mu * R * dF.transpose() * R + lambda * (R.array() * dF.array()).sum() * R;
}

#if 0
template <class T, int dim>
void LinearCorotated<T, dim>::firstPiolaDerivative(const Scratch& s, Hessian& dPdF) const
{
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j) {
            int row_idx = i + j * dim;
            for (int a = 0; a < dim; ++a)
                for (int b = 0; b < dim; ++b) {
                    int col_idx = a + b * dim;
                    int ia = (i == a);
                    int jb = (j == b);
                    dPdF(row_idx, col_idx) = mu * (ia * jb + R(i, b) * R(a, j)) + lambda * R(i, j) * R(a, b);
                }
        }
}
#endif
```

F based Isotropic
$$
\bold F = \bold U\bold \Sigma \bold V^T \\
\bold \Psi(\bold F) = \hat {\bold \Psi}(\bold \Sigma)\\
\bold P = \bold U \frac{\partial \hat{\bold \Psi}}{\partial  \Sigma}\bold V^T\\
\Delta \bold P =  \bold U (\frac{\Delta \bold P}{\Delta \bold F}|_{\Sigma}:(\bold U \Delta \bold F\bold V))\bold V^T
$$

```
template <class T>
template <class TConst>
bool SnowPlasticity<T>::projectStrain(TConst& c, Matrix<T, TConst::dim, TConst::dim>& strain)
{
    static const int dim = TConst::dim;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    TM U, V;
    TV sigma;

    // TODO: this is inefficient because next time step updateState will do the svd again!
    singularValueDecomposition(strain, U, sigma, V);

    T Fe_det = (T)1;
    for (int i = 0; i < dim; i++) {
        sigma(i) = std::max(std::min(sigma(i), (T)1 + theta_s), (T)1 - theta_c);
        Fe_det *= sigma(i);
    }

    Eigen::DiagonalMatrix<T, dim, dim> sigma_m(sigma);
    TM Fe = U * sigma_m * V.transpose();
    // T Jp_new = std::max(std::min(Jp * strain.determinant() / Fe_det, max_Jp), min_Jp);
    T Jp_new = Jp * strain.determinant() / Fe_det;
    if (!(Jp_new <= max_Jp))
        Jp_new = max_Jp;
    if (!(Jp_new >= min_Jp))
        Jp_new = min_Jp;

    strain = Fe;
    c.mu *= std::exp(psi * (Jp - Jp_new));
    c.lambda *= std::exp(psi * (Jp - Jp_new));
    Jp = Jp_new;

    return false;
}
```

