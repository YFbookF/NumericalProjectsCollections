#ifndef SNOW_CONJUGATERESIDUALSOLVER_H
#define SNOW_CONJUGATERESIDUALSOLVER_H
https://github.com/sethlu/renderbox-snow/tree/master/src

#include <vector>

#include <glm/glm.hpp>

#include "logging.h"


// Vector dot product
template<typename V>
inline V operator*(std::vector<V> const &a, std::vector<V> const &b) {
    LOG_ASSERT(a.size() == b.size());

    V result = {};
    for (size_t i = 0, n = a.size(); i < n; i++) {
        result += a[i] * b[i];
    }

    return result;
}

// Vector of vec3 dot product
inline double operator*(std::vector<glm::dvec3> const &a, std::vector<glm::dvec3> const &b) {
    LOG_ASSERT(a.size() == b.size());

    double result = {};
    for (size_t i = 0, n = a.size(); i < n; i++) {
        result += glm::dot(a[i], b[i]);
    }

    return result;
}

// Scalar multiply to vector
template<typename V>
inline std::vector<V> operator*(double a, std::vector<V> const &b) {

    std::vector<V> result(b.size());
    for (size_t i = 0, n = b.size(); i < n; i++) {
        result[i] = a * b[i];
    }

    return result;
}

// Vector addition
template<typename V>
inline std::vector<V> operator+(std::vector<V> const &a, std::vector<V> const &b) {
    LOG_ASSERT(a.size() == b.size());

    std::vector<V> result(a.size());
    for (size_t i = 0, n = a.size(); i < n; i++) {
        result[i] = a[i] + b[i];
    }

    return result;
}

// Vector subtraction
template<typename V>
inline std::vector<V> operator-(std::vector<V> const &a, std::vector<V> const &b) {
    LOG_ASSERT(a.size() == b.size());

    std::vector<V> result(a.size());
    for (size_t i = 0, n = a.size(); i < n; i++) {
        result[i] = a[i] - b[i];
    }

    return result;
}

/**
 * Solves Ax = b
 * The initial guess is passed in as x
 * The result will be written in x
 */
template<typename V>
inline void conjugateResidualSolver(void (*A)(std::vector<V> &Ax, std::vector<V> const &x),
                                    std::vector<V> &x,
                                    std::vector<V> const &b,
                                    int k) {
    std::vector<V> Ax(b.size());

    // Ax_0
    A(Ax, x);

    // r_0
    auto r = b - Ax;

    // p_0
    auto p = r;

    std::vector<V> Ar(b.size());
    A(Ar, r);
    auto dot_r_Ar = r * Ar;

    std::vector<V> Ap(b.size());
    A(Ap, p);

    while (k-- > 0 && r * r >= FLT_EPSILON) {
        LOG(VERBOSE) << "Solving k=" << k << std::endl;

        // r_k^T Ar_k
        auto dot_r_Ar_k = dot_r_Ar;

        // a_k
        auto a = dot_r_Ar_k / (Ap * Ap);

        if (abs(a) < FLT_EPSILON) break; // Non-standard: Break if insignificant increment

        // x_k+1
        x = x + a * p;

        // r_k+1
        r = r - a * Ap;

        // Ar_k+1
        A(Ar, r);

        dot_r_Ar = r * Ar;

        // b_k
        auto beta = dot_r_Ar / dot_r_Ar_k;

        if (abs(beta) < FLT_EPSILON) break; // Non-standard: Break if insignificant increment

        // p_k+1
        p = r + beta * p;

        // Ap_k+1
        Ap = Ar + beta * Ap;

    }

    if (k > 0) {
        LOG(VERBOSE) << "Converged at k=" << k << std::endl;
    } else {
        LOG(VERBOSE) << "Didn't converge" << std::endl;
    }

}

/**
 * Solves Ax = b
 * The initial guess is passed in as x
 * The result will be written in x
 */
template<typename C, typename V>
inline void conjugateResidualSolver(C *instance,
                                    void (C::*A)(std::vector<V> &Ax, std::vector<V> const &x),
                                    std::vector<V> &x,
                                    std::vector<V> const &b,
                                    int k) {
    std::vector<V> Ax(b.size());

    // Ax_0
    (instance->*A)(Ax, x);

    // r_0
    auto r = b - Ax;

    // p_0
    auto p = r;

    std::vector<V> Ar(b.size());
    (instance->*A)(Ar, r);
    auto dot_r_Ar = r * Ar;

    std::vector<V> Ap(b.size());
    (instance->*A)(Ap, p);

    while (k-- > 0 && r * r >= FLT_EPSILON) {
        LOG(VERBOSE) << "Solving k=" << k << std::endl;

        // r_k^T Ar_k
        auto dot_r_Ar_k = dot_r_Ar;

        // a_k
        auto a = dot_r_Ar_k / (Ap * Ap);

        if (abs(a) < FLT_EPSILON) break; // Non-standard: Break if insignificant increment

        // x_k+1
        x = x + a * p;

        // r_k+1
        r = r - a * Ap;

        // Ar_k+1
        (instance->*A)(Ar, r);

        dot_r_Ar = r * Ar;

        // b_k
        auto beta = dot_r_Ar / dot_r_Ar_k;

        if (abs(beta) < FLT_EPSILON) break; // Non-standard: Break if insignificant increment

        // p_k+1
        p = r + beta * p;

        // Ap_k+1
        Ap = Ar + beta * Ap;

    }

    if (k > 0) {
        LOG(VERBOSE) << "Converged at k=" << k << std::endl;
    } else {
        LOG(VERBOSE) << "Didn't converge" << std::endl;
    }

}


#endif //SNOW_CONJUGATERESIDUALSOLVER_H
