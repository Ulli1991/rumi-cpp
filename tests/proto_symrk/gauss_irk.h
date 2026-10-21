#pragma once
// Prototype: s-stage Gauss-Legendre collocation (order 2s, symplectic and symmetric for any Hamiltonian;
// Hairer, Lubich & Wanner, GNI 2006, Thm. VI.4.3, Sect. VIII.4), implemented as in
//   Hairer, McLachlan & Razakarivony 2008, BIT 48, 231 ("Achieving Brouwer's law with implicit Runge-Kutta
//   methods"), hereafter HMR08: fixed-point iteration to round-off with stopping when the increment is zero or
//   stops decreasing, starting values from the collocation polynomial of the previous step, compensated
//   summation of the update;
//   Antonana, Makazaga & Murua 2017, Numer. Algor. 76, 861 (arXiv:1702.03354), hereafter AMM17: stage form
//   Y_i = y_n + sum_j mu_ij L_j, L_j = h b_j f(Y_j), with mu_ij = a_ij / b_j; the symplecticity condition
//   b_i a_ij + b_j a_ji = b_i b_j reads mu_ij + mu_ji = 1, which is imposed exactly on the rounded mu
//   (mu_ii = 1/2, mu_ji = 1 - mu_ij), so the floating-point map stays as close to symplectic as possible.
// Coefficients are computed once in long double (Newton on P_s, Gauss quadrature of the Lagrange basis).
// Header-only, templated on the scalar (double for runs, a forward-mode dual for exact Jacobians).
#include <array>
#include <cmath>
#include <cstdio>

namespace symrk {

constexpr int S_MAX = 8;
constexpr int DIM_MAX = 6 * 8 + 2;   // q, p of <= 8 bodies + t, p_t

struct GaussCoeffs {
    int s = 0;
    long double c[S_MAX], b[S_MAX], a[S_MAX][S_MAX];
    double mu[S_MAX][S_MAX];       // a_ij / b_j rounded, mu_ij + mu_ji = 1 exactly
    double bd[S_MAX];              // b_i rounded
    double ext[S_MAX][S_MAX + 1];  // starting values: Y_k(next) = ext[k][0] y_n + sum_i ext[k][i+1] Y_i(prev)
};

inline long double legendre(int s, long double x, long double& dp) {
    long double p0 = 1.0L, p1 = x;
    if (s == 0) { dp = 0; return 1; }
    for (int k = 2; k <= s; k++) {
        long double p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k;
        p0 = p1; p1 = p2;
    }
    dp = s * (x * p1 - p0) / (x * x - 1.0L);
    return p1;
}

inline GaussCoeffs make_gauss(int s) {
    GaussCoeffs g;
    g.s = s;
    const long double PI = 3.14159265358979323846264338327950288L;
    for (int i = 0; i < s; i++) {
        long double x = std::cos(PI * (i + 0.75L) / (s + 0.5L)), dp;
        for (int it = 0; it < 100; it++) {
            long double p = legendre(s, x, dp);
            long double dx = p / dp;
            x -= dx;
            if (std::fabs(dx) < 1e-21L) break;
        }
        legendre(s, x, dp);
        g.c[s - 1 - i] = (1.0L + x) / 2.0L;
        g.b[s - 1 - i] = 1.0L / ((1.0L - x * x) * dp * dp);   // w/2 with w = 2 / ((1 - x^2) P'^2)
    }
    auto lag = [&](int j, long double t) {
        long double r = 1.0L;
        for (int m = 0; m < s; m++) if (m != j) r *= (t - g.c[m]) / (g.c[j] - g.c[m]);
        return r;
    };
    for (int i = 0; i < s; i++)
        for (int j = 0; j < s; j++) {
            long double sum = 0.0L;
            for (int k = 0; k < s; k++) sum += g.b[k] * lag(j, g.c[i] * g.c[k]);
            g.a[i][j] = g.c[i] * sum;
        }
    for (int i = 0; i < s; i++) {
        g.bd[i] = (double)g.b[i];
        g.mu[i][i] = 0.5;
        for (int j = i + 1; j < s; j++) {
            g.mu[i][j] = (double)(g.a[i][j] / g.b[j]);
            g.mu[j][i] = 1.0 - g.mu[i][j];
        }
    }
    // Extrapolation of the collocation polynomial through (0, y_n), (c_i, Y_i) to 1 + c_k (HMR08 Sect. 3)
    long double nodes[S_MAX + 1];
    nodes[0] = 0.0L;
    for (int i = 0; i < s; i++) nodes[i + 1] = g.c[i];
    for (int k = 0; k < s; k++) {
        const long double t = 1.0L + g.c[k];
        for (int j = 0; j <= s; j++) {
            long double r = 1.0L;
            for (int m = 0; m <= s; m++) if (m != j) r *= (t - nodes[m]) / (nodes[j] - nodes[m]);
            g.ext[k][j] = (double)r;
        }
    }
    return g;
}

inline double val_(double x) { return x; }
inline double val_(long double x) { return (double)x; }

// Kahan-compensated state: y = hi + lo
template <class T>
struct CState {
    int dim = 0;
    std::array<T, DIM_MAX> y{}, lo{}, prev{};   // prev: y of the last step (starting values)
};

struct IrkStats {
    long long steps = 0, rhs_calls = 0, iters = 0, maxit_hits = 0, diverged = 0;
};

// One Gauss step of size h. sys.rhs(const T* y, T* f) evaluates the vector field. Stage values Ysave hold the
// previous step's stages (for the starting values); first = true uses Y_i = y_n.
// fixed_iters > 0: exactly that many iterations (dual-number Jacobians: derivatives converge with the values).
template <class T, class Sys>
void gauss_step(const GaussCoeffs& g, const Sys& sys, CState<T>& st, double h,
                std::array<std::array<T, DIM_MAX>, S_MAX>& Y, bool& have_prev, IrkStats& stats,
                int fixed_iters = 0) {
    const int s = g.s, d = st.dim;
    std::array<std::array<T, DIM_MAX>, S_MAX> L, F, Ynew;
    if (have_prev) {
        // Ynew_k = ext[k][0] y_prev + sum_i ext[k][i+1] Y_i (collocation polynomial of the last step at 1 + c_k)
        for (int k = 0; k < s; k++)
            for (int m = 0; m < d; m++) {
                T v = g.ext[k][0] * st.prev[m];
                for (int i = 0; i < s; i++) v = v + g.ext[k][i + 1] * Y[i][m];
                Ynew[k][m] = v;
            }
    } else {
        for (int k = 0; k < s; k++)
            for (int m = 0; m < d; m++) Ynew[k][m] = st.y[m];
    }
    double dprev = 1e300, dprev2 = 1e300;
    const int maxit = fixed_iters > 0 ? fixed_iters : 100;
    int it = 0;
    for (; it < maxit; it++) {
        for (int k = 0; k < s; k++) {
            sys.rhs(Ynew[k].data(), F[k].data());
            for (int m = 0; m < d; m++) L[k][m] = (h * g.bd[k]) * F[k][m];
        }
        stats.rhs_calls += s;
        double dmax = 0.0;
        for (int k = 0; k < s; k++)
            for (int m = 0; m < d; m++) {
                T v = st.y[m];
                // sum the small mu_kj L_j first, then add to y (AMM17: Y_i - y_n carried separately)
                T inc = T(0.0);
                for (int j = 0; j < s; j++) inc = inc + g.mu[k][j] * L[j][m];
                v = v + (inc + st.lo[m]);
                const double diff = std::fabs(val_(v) - val_(Ynew[k][m]));
                const double scale = std::fabs(val_(st.y[m])) + std::fabs(val_(inc)) + 1e-300;
                dmax = std::max(dmax, diff / scale);
                Ynew[k][m] = v;
            }
        if (fixed_iters > 0) continue;
        // fixed-point iteration diverges for h L > 1 (strong PN near pericentre, coarse steps): flag it
        if (!(dmax < 1.0) && it > 3) { stats.diverged++; it++; break; }
        // HMR08 stopping: increment zero, or not decreasing (twice in a row) once converged to near round-off
        if (dmax == 0.0) { it++; break; }
        if (dmax >= dprev && dprev >= dprev2) { it++; break; }
        if (dmax >= dprev && dmax < 1e-13) { it++; break; }
        dprev2 = dprev;
        dprev = dmax;
    }
    if (it >= maxit && fixed_iters == 0) stats.maxit_hits++;
    stats.iters += it;
    // Final L from the converged stages (one more evaluation would be the classical choice; HMR08 use the last
    // L, which is consistent with the last Y to the iteration tolerance)
    // update y_{n+1} = y_n + sum L_k, compensated
    for (int m = 0; m < d; m++) {
        st.prev[m] = st.y[m];
        T inc = T(0.0);
        for (int k = s - 1; k >= 0; k--) inc = inc + L[k][m];
        T yk = st.y[m];
        T corr = inc + st.lo[m];
        T ynew = yk + corr;
        st.lo[m] = corr - (ynew - yk);
        st.y[m] = ynew;
    }
    Y = Ynew;
    have_prev = true;
    stats.steps++;
}

}  // namespace symrk
