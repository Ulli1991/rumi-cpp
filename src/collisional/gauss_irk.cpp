#include "gauss_irk.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

constexpr int S = GIRK_STAGES;
constexpr int DMAX = 6 * LOGH_MAX_BODIES + 1;   // q, p, t (p_t is a constant of the call and not integrated)

// Gauss-Legendre coefficients, computed once in long double (Newton on P_s; a_ij by Gauss quadrature of the Lagrange
// basis, exact for the polynomial degree), then rounded; mu_ij = a_ij / b_j with mu_ij + mu_ji = 1 imposed on the
// rounded values (AMM17 Sect. 2: the floating-point map keeps the symplecticity condition b_i a_ij + b_j a_ji = b_i b_j)
struct Coeffs {
    long double cl[S];
    double c[S], b[S], a[S][S], mu[S][S];
    double ext[S][S + 1];   // weights() at 1 + c_k: starting values of a step after one of the same size (HMR08)
    Coeffs() {
        const long double PI = 3.14159265358979323846264338327950288L;
        long double bl[S], al[S][S];
        auto legendre = [](long double x, long double& dp) {
            long double p0 = 1.0L, p1 = x;
            for (int k = 2; k <= S; k++) {
                const long double p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k;
                p0 = p1;
                p1 = p2;
            }
            dp = S * (x * p1 - p0) / (x * x - 1.0L);
            return p1;
        };
        for (int i = 0; i < S; i++) {
            long double x = std::cos(PI * (i + 0.75L) / (S + 0.5L)), dp = 0.0L;
            for (int it = 0; it < 100; it++) {
                const long double dx = legendre(x, dp) / dp;
                x -= dx;
                if (std::fabs(dx) < 1e-21L) break;
            }
            legendre(x, dp);
            cl[S - 1 - i] = (1.0L + x) / 2.0L;
            bl[S - 1 - i] = 1.0L / ((1.0L - x * x) * dp * dp);
        }
        auto lag = [&](int j, long double t) {
            long double r = 1.0L;
            for (int m = 0; m < S; m++)
                if (m != j) r *= (t - cl[m]) / (cl[j] - cl[m]);
            return r;
        };
        for (int i = 0; i < S; i++)
            for (int j = 0; j < S; j++) {
                long double sum = 0.0L;   // a_ij = int_0^{c_i} l_j = c_i sum_k b_k l_j(c_i c_k)
                for (int k = 0; k < S; k++) sum += bl[k] * lag(j, cl[i] * cl[k]);
                al[i][j] = cl[i] * sum;
            }
        for (int i = 0; i < S; i++) {
            c[i] = (double)cl[i];
            b[i] = (double)bl[i];
            for (int j = 0; j < S; j++) a[i][j] = (double)al[i][j];
            mu[i][i] = 0.5;
            for (int j = i + 1; j < S; j++) {
                mu[i][j] = (double)(al[i][j] / bl[j]);
                mu[j][i] = 1.0 - mu[i][j];
            }
        }
        for (int i = 0; i < S; i++) weights(1.0 + c[i], ext[i]);
    }
    // Lagrange weights of the collocation polynomial through (0, y_n), (c_i, Y_i) at x (units of its step)
    void weights(double x, double w[S + 1]) const {
        long double nodes[S + 1];
        nodes[0] = 0.0L;
        for (int i = 0; i < S; i++) nodes[i + 1] = cl[i];
        for (int j = 0; j <= S; j++) {
            long double r = 1.0L;
            for (int m = 0; m <= S; m++)
                if (m != j) r *= ((long double)x - nodes[m]) / (nodes[j] - nodes[m]);
            w[j] = (double)r;
        }
    }
};
const Coeffs& coeffs() {
    static const Coeffs k;
    return k;
}

// Poincare-transformed Newtonian N-body vector field, y = (q[3n], p[3n], t)
struct Sys {
    int n = 0, d = 0;
    double m[LOGH_MAX_BODIES] = {};
    double G = 1.0, pt = 0.0;
    void rhs(const double* y, double* f) const {
        const int n3 = 3 * n;
        const double* q = y;
        const double* p = y + n3;
        double gU[3 * LOGH_MAX_BODIES] = {};
        double U = 0.0, T = 0.0;
        for (int a = 0; a < n; a++)
            for (int b = a + 1; b < n; b++) {
                const double x0 = q[3*a] - q[3*b], x1 = q[3*a+1] - q[3*b+1], x2 = q[3*a+2] - q[3*b+2];
                const double r2 = x0 * x0 + x1 * x1 + x2 * x2, r = std::sqrt(r2);
                const double Gmm = G * m[a] * m[b], w = Gmm / (r2 * r);
                U -= Gmm / r;
                gU[3*a] += w * x0; gU[3*a+1] += w * x1; gU[3*a+2] += w * x2;   // dU/dq_a
                gU[3*b] -= w * x0; gU[3*b+1] -= w * x1; gU[3*b+2] -= w * x2;
            }
        for (int i = 0; i < n3; i++) T += 0.5 * p[i] * p[i] / m[i / 3];
        const double g = -1.0 / U, hpt = (T + U) + pt;
        const double cf = g + hpt * g * g;
        for (int i = 0; i < n3; i++) {
            f[i] = g * p[i] / m[i / 3];
            f[n3 + i] = -cf * gU[i];
        }
        f[2 * n3] = g;
    }
};

// Dense LU with partial pivoting (simplified Newton; s d <= 8 x 49)
bool lu_factor(std::vector<double>& M, std::vector<int>& piv, int N) {
    for (int k = 0; k < N; k++) {
        int p = k;
        double best = std::fabs(M[(size_t)k * N + k]);
        for (int i = k + 1; i < N; i++)
            if (std::fabs(M[(size_t)i * N + k]) > best) { best = std::fabs(M[(size_t)i * N + k]); p = i; }
        if (!(best > 0.0)) return false;
        piv[k] = p;
        if (p != k)
            for (int j = 0; j < N; j++) std::swap(M[(size_t)k * N + j], M[(size_t)p * N + j]);
        const double inv = 1.0 / M[(size_t)k * N + k];
        for (int i = k + 1; i < N; i++) {
            double& lik = M[(size_t)i * N + k];
            lik *= inv;
            if (lik == 0.0) continue;
            for (int j = k + 1; j < N; j++) M[(size_t)i * N + j] -= lik * M[(size_t)k * N + j];
        }
    }
    return true;
}
void lu_solve(const std::vector<double>& M, const std::vector<int>& piv, int N, double* x) {
    for (int k = 0; k < N; k++) {
        std::swap(x[k], x[piv[k]]);
        for (int i = k + 1; i < N; i++) x[i] -= M[(size_t)i * N + k] * x[k];
    }
    for (int i = N - 1; i >= 0; i--) {
        double s = x[i];
        for (int j = i + 1; j < N; j++) s -= M[(size_t)i * N + j] * x[j];
        x[i] = s / M[(size_t)i * N + i];
    }
}

using Stages = double[S][DMAX];

// Simplified Newton for the stage increments Z_i = Y_i - y_n (HLW Sect. VIII.6.2): (I - h A x J) dZ = -Z + h (A x I) F,
// J = f' by forward differences (at the mean starting stage), one LU per step. Y holds the starting values on entry, the stages on exit
bool newton_stages(const Coeffs& k, const Sys& sys, const double* y, const double* lo, double h, Stages& Y,
                   GaussIrkStats& st) {
    const int d = sys.d, N = S * d;
    std::vector<double> M((size_t)N * N, 0.0), J((size_t)d * d, 0.0);
    std::vector<int> piv(N);
    double f0[DMAX], f1[DMAX], yp[DMAX], ym[DMAX];
    // J at the mean of the starting stages (~ y(t_n + h/2)): with J(y_n) the iteration diverged on steps into an
    // e = 0.9 pericentre (the field changes over the step there), contraction 0.05-0.1 per iteration elsewhere
    for (int r = 0; r < d; r++) {
        double m = 0.0;
        for (int i = 0; i < S; i++) m += Y[i][r];
        ym[r] = m / S;
    }
    sys.rhs(ym, f0);
    st.rhs++;
    double sq = 0.0, sp = 0.0;   // block scales for the difference quotients
    for (int i = 0; i < 3 * sys.n; i++) {
        sq = std::max(sq, std::fabs(ym[i]));
        sp = std::max(sp, std::fabs(ym[3 * sys.n + i]));
    }
    for (int c = 0; c < d - 1; c++) {   // f does not depend on t (last column zero)
        std::memcpy(yp, ym, d * sizeof(double));
        const double del = 1.5e-8 * (std::fabs(ym[c]) + 1e-3 * (c < 3 * sys.n ? sq : sp) + 1e-300);
        yp[c] += del;
        sys.rhs(yp, f1);
        st.rhs++;
        for (int r = 0; r < d; r++) J[(size_t)r * d + c] = (f1[r] - f0[r]) / del;
    }
    for (int i = 0; i < S; i++)
        for (int j = 0; j < S; j++)
            for (int r = 0; r < d; r++)
                for (int c = 0; c < d; c++)
                    M[(size_t)(i * d + r) * N + j * d + c] = (i == j && r == c ? 1.0 : 0.0) - h * k.a[i][j] * J[(size_t)r * d + c];
    if (!lu_factor(M, piv, N)) return false;
    std::vector<double> Z((size_t)N), R((size_t)N);
    for (int i = 0; i < S; i++)
        for (int r = 0; r < d; r++) Z[i * d + r] = Y[i][r] - y[r];
    double prev = 1e300;
    for (int it = 0; it < 40; it++) {
        double F[S][DMAX];
        for (int i = 0; i < S; i++) {
            for (int r = 0; r < d; r++) yp[r] = y[r] + (Z[i * d + r] + lo[r]);
            sys.rhs(yp, F[i]);
        }
        st.rhs += S;
        for (int i = 0; i < S; i++)
            for (int r = 0; r < d; r++) {
                double s = 0.0;
                for (int j = 0; j < S; j++) s += k.a[i][j] * F[j][r];
                R[i * d + r] = -Z[i * d + r] + h * s;
            }
        lu_solve(M, piv, N, R.data());
        double nrm = 0.0;
        bool nrm_ok = true;
        for (int i = 0; i < S; i++)
            for (int r = 0; r < d; r++) {
                Z[i * d + r] += R[i * d + r];
                if (!std::isfinite(Z[i * d + r])) nrm_ok = false;
                nrm = std::max(nrm, std::fabs(R[i * d + r]) / (std::fabs(y[r]) + std::fabs(Z[i * d + r]) + 1e-300));
            }
        if (!std::isfinite(nrm) || !(nrm_ok)) return false;   // NaN in a stage (std::max ignores NaN)
        const bool done = nrm < 1e-15 || (nrm >= prev && nrm < 1e-12);
        if (done) {
            for (int i = 0; i < S; i++)
                for (int r = 0; r < d; r++) Y[i][r] = y[r] + (Z[i * d + r] + lo[r]);
            return true;
        }
        if (it >= 2 && nrm >= prev && nrm > 1e-10) return false;   // not contracting
        prev = nrm;
    }
    return false;
}

// One Gauss step of size h from y + lo. Y: starting values in, converged stages out; L_j = h b_j f(Y_j)
bool gauss_stages(const Coeffs& k, const Sys& sys, const double* y, const double* lo, double h, Stages& Y,
                  Stages& L, GaussIrkStats& st, bool force_newton) {
    const int d = sys.d;
    double F[DMAX];
    double dprev = 1e300;
    int ngrow = 0;   // successive iterations with a growing increment
    Stages Y0;
    std::memcpy(Y0, Y, sizeof(Stages));
    bool conv = false;
    for (int it = 0; it < 60; it++) {
        for (int i = 0; i < S; i++) {
            sys.rhs(Y[i], F);
            for (int r = 0; r < d; r++) L[i][r] = (h * k.b[i]) * F[r];
        }
        st.rhs += S;
        st.iters++;
        double dmax = 0.0;
        bool finite = true;   // std::max(dmax, NaN) = dmax would hide a NaN stage value: test every one
        for (int i = 0; i < S; i++)
            for (int r = 0; r < d; r++) {
                double inc = 0.0;   // sum the small mu_ij L_j first, then add to y (AMM17)
                for (int j = 0; j < S; j++) inc += k.mu[i][j] * L[j][r];
                const double v = y[r] + (inc + lo[r]);
                dmax = std::max(dmax, std::fabs(v - Y[i][r]) / (std::fabs(y[r]) + std::fabs(inc) + 1e-300));
                Y[i][r] = v;
                if (!std::isfinite(v)) finite = false;
            }
        if (!finite || !std::isfinite(dmax)) break;
        // HMR08: stop when the increment vanishes or stops decreasing near round-off
        if (dmax == 0.0 || (dmax >= dprev && dmax < 1e-13)) {
            conv = true;
            break;
        }
        if (it >= 3 && dmax >= dprev && dmax > 1e-6) break;   // not contracting (h L > 1)
        // W10: an increment growing over successive iterations above round-off is divergence, not the HMR08 plateau
        ngrow = (dmax > dprev) ? ngrow + 1 : 0;
        if (ngrow >= 3 && dmax > 1e-13) break;
        dprev = dmax;
    }
    auto finite_L = [&]() {   // W10: never accept a non-finite stage value
        for (int i = 0; i < S; i++)
            for (int r = 0; r < d; r++)
                if (!std::isfinite(L[i][r])) return false;
        return true;
    };
    if (conv && !force_newton) return finite_L();
    // Simplified Newton from the original starting values
    std::memcpy(Y, Y0, sizeof(Stages));
    if (!newton_stages(k, sys, y, lo, h, Y, st)) return false;
    st.newton++;
    for (int i = 0; i < S; i++) {
        sys.rhs(Y[i], F);
        for (int r = 0; r < d; r++) L[i][r] = (h * k.b[i]) * F[r];
    }
    st.rhs += S;
    return finite_L();
}

// y_{n+1} = y_n + sum L_k, compensated (Kahan): the result is hi + lo
void update(int d, const Stages& L, double* y, double* lo) {
    for (int r = 0; r < d; r++) {
        double inc = 0.0;
        for (int i = S - 1; i >= 0; i--) inc += L[i][r];
        const double corr = inc + lo[r];
        const double yn = y[r] + corr;
        lo[r] = corr - (yn - y[r]);
        y[r] = yn;
    }
}

// Starting values from the collocation polynomial (0, y0), (c_i, Yp_i) of a step: Y_k = u(x0 + c_k x1)
void start_from_poly(const Coeffs& k, int d, const double* y0, const Stages& Yp, double x0, double x1, Stages& Y) {
    for (int i = 0; i < S; i++) {
        double wb[S + 1];
        const double* w = wb;
        if (x0 == 1.0 && x1 == 1.0) w = k.ext[i];
        else k.weights(x0 + k.c[i] * x1, wb);
        for (int r = 0; r < d; r++) {
            double v = w[0] * y0[r];
            for (int j = 0; j < S; j++) v += w[j + 1] * Yp[j][r];
            Y[i][r] = v;
        }
    }
}

}  // namespace

double gauss_irk_ds(const LogHState& s, double G, double spo, bool spo_floor) {
    const int n = s.n_bodies;
    double U = 0.0, U_avg = 0.0, P_min = 1e300, r_min = 1e300, M = 0.0, e_max = 0.0;
    double pe[LOGH_MAX_BODIES * (LOGH_MAX_BODIES - 1) / 2][2];   // bound pairs: period, eccentricity
    int np = 0;
    for (int i = 0; i < n; i++) M += s.mass[i];
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            double r2 = 0.0, v2 = 0.0;
            for (int c = 0; c < 3; c++) {
                const double dx = s.pos[j*3+c] - s.pos[i*3+c], dv = s.vel[j*3+c] - s.vel[i*3+c];
                r2 += dx * dx;
                v2 += dv * dv;
            }
            const double r = std::sqrt(r2), Gmm = G * s.mass[i] * s.mass[j], mu = G * (s.mass[i] + s.mass[j]);
            U -= Gmm / r;
            U_avg -= Gmm / r;
            r_min = std::min(r_min, r);
            const double eps = 0.5 * v2 - mu / r;
            if (!(eps < 0.0)) continue;
            const double a = -mu / (2.0 * eps), P = 2.0 * M_PI * std::sqrt(a * a * a / mu);
            double rv = 0.0;
            for (int c = 0; c < 3; c++) rv += (s.pos[j*3+c] - s.pos[i*3+c]) * (s.vel[j*3+c] - s.vel[i*3+c]);
            pe[np][0] = P;
            pe[np][1] = std::sqrt(std::max(0.0, 1.0 + 2.0 * eps * (r2 * v2 - rv * rv) / (mu * mu)));
            np++;
            U_avg += Gmm / r - Gmm / a;   // a bound pair at its orbit average (time average of 1/r is 1/a)
            P_min = std::min(P_min, P);
        }
    // The fixed-point iteration converges at the pericentre of an eccentric pair only for spo >~ 4 / sqrt(1 - e)
    // (measured e = 0.99 / 0.995 / 0.999: 40 / 60 / 120), and the O(h^16) error stays at round-off only above
    // ~8 / sqrt(1 - e) (e = 0.9 at spo 19: 1e-12 per call ending on a partial step, at 25: 1e-15): floor 8 / sqrt(1 - e)
    // (pairs with P < 10 P_min: a wide near-parabolic pair of members of two binaries is not at its pericentre here)
    for (int q = 0; q < np; q++) if (pe[q][0] < 10.0 * P_min) e_max = std::max(e_max, pe[q][1]);
    if (spo_floor && e_max < 1.0) spo = std::max(spo, 8.0 / std::sqrt(1.0 - e_max));
    double S_orb;
    if (P_min < 1e300) S_orb = -U_avg * P_min;
    else S_orb = -U * 2.0 * M_PI * std::sqrt(r_min * r_min * r_min / (G * M));
    return S_orb / spo;
}

void gauss_irk_integrate(LogHState& state, double t_end, double G, const GaussIrkOpts& opt, long long max_steps,
                         const double* coll, GaussIrkOut& out, GaussIrkStats* stats) {
    out = GaussIrkOut();
    GaussIrkStats st;
    const Coeffs& k = coeffs();
    Sys sys;
    sys.n = state.n_bodies;
    sys.d = 6 * sys.n + 1;
    sys.G = G;
    const int n = sys.n, n3 = 3 * n, d = sys.d;
    for (int i = 0; i < n; i++) sys.m[i] = state.mass[i];
    double y[DMAX], lo[DMAX] = {};
    for (int i = 0; i < n3; i++) {
        y[i] = state.pos[i];
        y[n3 + i] = state.mass[i / 3] * state.vel[i];
    }
    y[2 * n3] = state.t;
    sys.pt = -logh_total_energy(state, G);
    const double h = gauss_irk_ds(state, G, opt.spo, opt.spo_floor);
    out.ds = h;
    auto finish = [&]() {
        if (stats) {
            stats->steps += st.steps; stats->rhs += st.rhs; stats->iters += st.iters;
            stats->newton += st.newton; stats->rejected += st.rejected;
        }
    };
    if (!(h > 0.0) || !std::isfinite(h)) { out.fail = 4; finish(); return; }
    const double t0 = state.t, span = t_end - t0;
    if (!(span > 0.0)) { out.ok = true; finish(); return; }
    const double t_tol = 1e-14 * std::max(std::fabs(t_end), std::fabs(span));

    // Collision watch: separation or osculating pericentre below the pair's collision distance stops the call
    auto collision_near = [&]() {
        if (!coll) return false;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) {
                const double rc = coll[i * LOGH_MAX_BODIES + j];
                if (!(rc > 0.0)) continue;
                double r[3], v[3];
                for (int c = 0; c < 3; c++) {
                    r[c] = (y[3*j+c] + lo[3*j+c]) - (y[3*i+c] + lo[3*i+c]);
                    v[c] = (y[n3+3*j+c] + lo[n3+3*j+c]) / sys.m[j] - (y[n3+3*i+c] + lo[n3+3*i+c]) / sys.m[i];
                }
                const double rr = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
                if (rr < rc) return true;
                const double mu = G * (sys.m[i] + sys.m[j]);
                const double hx = r[1]*v[2] - r[2]*v[1], hy = r[2]*v[0] - r[0]*v[2], hz = r[0]*v[1] - r[1]*v[0];
                const double h2 = hx*hx + hy*hy + hz*hz;
                const double eps = 0.5 * (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) - mu / rr;
                const double e = std::sqrt(std::max(0.0, 1.0 + 2.0 * eps * h2 / (mu * mu)));
                if (h2 / (mu * (1.0 + e)) < rc) return true;
            }
        return false;
    };

    Stages Y, L, Yp;          // current stages, L, stages of the previous accepted step
    double yprev[DMAX];
    bool have_prev = false;
    double dt_prev = 0.0;
    auto t_of = [&](const Stages& Ls) {   // t after a step with these L (compensated sum, as update)
        double inc = 0.0;
        for (int i = S - 1; i >= 0; i--) inc += Ls[i][2 * n3];
        return y[2 * n3] + (inc + lo[2 * n3]);
    };
    auto g_now = [&](const double* yy) {
        double f[DMAX];
        sys.rhs(yy, f);
        st.rhs++;
        return f[2 * n3];
    };
    auto euler_start = [&](double hh) {
        double f[DMAX];
        sys.rhs(y, f);
        st.rhs++;
        for (int i = 0; i < S; i++)
            for (int r = 0; r < d; r++) Y[i][r] = y[r] + k.c[i] * hh * f[r];
    };
    auto fail = [&](int why) { out.fail = why; out.steps = (int)st.steps; finish(); };

    for (;;) {
        if (st.steps >= max_steps) { fail(1); return; }
        const double t_n = y[2 * n3] + lo[2 * n3];
        // Does the next step pass t_end? After a step of the same size its t is predicted by the collocation
        // polynomial of that step, extrapolated to 2 (step-size ratio 1), and so is h* on landing; else t + h g(y)
        double h_land = 0.0;
        Stages Ytr;               // stages of a trial step that passed t_end (landing starts from its polynomial)
        double h_tr = 0.0;
        bool regular;
        double x_land = 0.0;
        if (have_prev) {
            double tv[S + 1];
            tv[0] = yprev[2 * n3];
            for (int i = 0; i < S; i++) tv[i + 1] = Yp[i][2 * n3];
            auto t_pred = [&](double x) {
                double w[S + 1], v = 0.0;
                k.weights(x, w);
                for (int j = 0; j <= S; j++) v += w[j] * tv[j];
                return v;
            };
            const double t2 = t_pred(2.0);
            regular = t2 < t_end - 1e-6 * (t2 - t_n);
            if (!regular) {
                double a = 1.0, b = 2.0;
                for (int it = 0; it < 60; it++) {
                    const double x = 0.5 * (a + b);
                    if (t_pred(x) < t_end) a = x; else b = x;
                }
                x_land = 0.5 * (a + b) - 1.0;
            }
        } else {
            regular = t_n + h * g_now(y) * (1.0 + 1e-3) < t_end;
        }
        (void)dt_prev;
        if (regular) {
            // Regular step of size h
            if (have_prev) start_from_poly(k, d, yprev, Yp, 1.0, 1.0, Y);
            else euler_start(h);
            if (!gauss_stages(k, sys, y, lo, h, Y, L, st, opt.force_newton)) { fail(2); return; }
            const double t_new = t_of(L);
            if (t_new < t_end - t_tol) {
                std::memcpy(yprev, y, sizeof(yprev));
                std::memcpy(Yp, Y, sizeof(Stages));
                update(d, L, y, lo);
                have_prev = true;
                dt_prev = t_new - t_n;
                st.steps++;
                if (!std::isfinite(y[0] + y[n3] + y[2 * n3])) { fail(4); return; }
                if (collision_near()) { fail(3); return; }
                continue;
            }
            // Passed t_end: h* from the collocation polynomial of this step, u_t(theta) = t_end (t increases with s)
            st.rejected++;
            double tn_v[S + 1];
            tn_v[0] = y[2 * n3];
            for (int i = 0; i < S; i++) tn_v[i + 1] = Y[i][2 * n3];
            double lo_th = 0.0, hi_th = 1.0;
            for (int it = 0; it < 60; it++) {
                const double th = 0.5 * (lo_th + hi_th);
                double w[S + 1], v = 0.0;
                k.weights(th, w);
                for (int j = 0; j <= S; j++) v += w[j] * tn_v[j];
                if (v < t_end) lo_th = th; else hi_th = th;
            }
            h_land = 0.5 * (lo_th + hi_th) * h;
            std::memcpy(Ytr, Y, sizeof(Stages));
            h_tr = h;
        } else {
            // starting values: the previous step's polynomial, or Euler
            h_land = have_prev ? x_land * h : (t_end - t_n) / g_now(y);
        }
        // Landing: Newton on t(h*) = t_end, dt/dh = g at the end of the step
        bool landed = false;
        for (int att = 0; att < 8; att++) {
            if (h_tr > 0.0) start_from_poly(k, d, y, Ytr, 0.0, h_land / h_tr, Y);
            else if (have_prev) start_from_poly(k, d, yprev, Yp, 1.0, h_land / h, Y);
            else euler_start(h_land);
            if (!gauss_stages(k, sys, y, lo, h_land, Y, L, st, opt.force_newton)) { fail(2); return; }
            const double t1 = t_of(L);
            double yend[DMAX], loend[DMAX];
            std::memcpy(yend, y, sizeof(yend));
            std::memcpy(loend, lo, sizeof(loend));
            update(d, L, yend, loend);
            const double err = t_end - t1;
            if (std::fabs(err) <= t_tol || att == 7) {
                if (std::fabs(err) > 1e3 * t_tol) { fail(2); return; }
                std::memcpy(y, yend, sizeof(y));
                std::memcpy(lo, loend, sizeof(lo));
                st.steps++;
                landed = true;
                break;
            }
            // next attempt starts from this attempt's polynomial
            std::memcpy(Ytr, Y, sizeof(Stages));
            h_tr = h_land;
            h_land += err / g_now(yend);
        }
        if (!landed || !std::isfinite(y[0] + y[n3])) { fail(4); return; }
        if (collision_near()) { fail(3); return; }
        break;
    }
    for (int i = 0; i < n3; i++) {
        state.pos[i] = y[i] + lo[i];
        state.vel[i] = (y[n3 + i] + lo[n3 + i]) / state.mass[i / 3];
    }
    state.t = t_end;
    out.ok = true;
    out.steps = (int)st.steps;
    finish();
}
