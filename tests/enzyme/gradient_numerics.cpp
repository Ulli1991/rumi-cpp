// Derivative identities of the production kernels against Enzyme (forward, reverse, batched forward, nested):
//   spline softening (constants.h Softening): g = -w'/r, k = g'/r, continuity at u = 0.5 and 1
//   2-body perturbation rates (pert_rates.h): Edot, adot, edot = d/dt of the code's own E, a, |e| along f, 1e-10 <= e
//   FSI gradient kernel (n2_kernels.h GradK) = Enzyme J a of the force field
//   monopole kernel K (gradient_correction_interaction) = Hessian of -1/s; dipole kernel (gradient_dipole_interaction)
//   = third derivative of -1/s contracted with D + P w^T; tree cell term (monopole + dipole) vs J a of the cell
// Exit status: number of failed checks. Build: CMake option RUMI_ENZYME_TESTS (docs/ENZYME.md), or
//   clang++ -std=c++20 -O2 -fopenmp -fplugin=ClangEnzyme-18.so -I<kokkos>/include -Isrc/core -Isrc/gravity
//           -Isrc/collisional tests/enzyme/gradient_numerics.cpp
#include "n2_kernels.h"
#include "pert_rates.h"
#include <algorithm>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

extern int enzyme_dup, enzyme_const, enzyme_dupnoneed, enzyme_width, enzyme_dupv, enzyme_dupnoneedv;
template <typename R, typename... A> R __enzyme_fwddiff(void*, A...);
template <typename R, typename... A> R __enzyme_autodiff(void*, A...);

static std::mt19937_64 rng(20261024);
static double U(double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng); }
static double logU(double a, double b) { return std::pow(10.0, U(std::log10(a), std::log10(b))); }
static void rdir(double* d) { double z = U(-1, 1), p = U(0, 2 * M_PI), s = std::sqrt(1 - z * z); d[0] = s * std::cos(p); d[1] = s * std::sin(p); d[2] = z; }
static double nrm(const double* v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

static int g_fail = 0;
static void check(const char* what, double err, double bound) {
    const bool ok = err <= bound;   // NaN fails
    std::printf("%-4s %-62s %.3e (<= %.1e)\n", ok ? "ok" : "FAIL", what, err, bound);
    g_fail += !ok;
}

// ---------------------------------------------------------------- spline softening
struct SP { Softening s; };
__attribute__((noinline)) static double sp_w(double r2, const SP* p) { return p->s.spline_w(r2); }
__attribute__((noinline)) static double sp_g(double r2, const SP* p) { double g, k; p->s.spline_gk(r2, g, k); return g; }
static void spline() {
    SP p; p.s.spline = 1; p.s.h = 2.8e-4;
    double eg = 0, ek = 0, egr = 0;
    for (int t = 0; t < 4000; t++) {
        const double r = p.s.h * (t % 2 ? U(0.01, 1.3) : logU(1e-3, 1e3)), r2 = r * r;
        double g, k; p.s.spline_gk(r2, g, k);
        const double dw = __enzyme_fwddiff<double>((void*)sp_w, r2, 1.0, enzyme_const, &p);
        const double dg = __enzyme_fwddiff<double>((void*)sp_g, r2, 1.0, enzyme_const, &p);
        const double dwr = __enzyme_autodiff<double>((void*)sp_w, r2, enzyme_const, &p);   // reverse mode of the same
        eg = std::max(eg, std::fabs((g + 2 * dw) / g));
        egr = std::max(egr, std::fabs(dwr - dw) / std::fabs(dw));
        ek = std::max(ek, std::fabs((k - 2 * dg) / (std::fabs(k) + g / r2)));
    }
    check("spline g = -w'(r)/r (Enzyme fwd)", eg, 1e-13);
    check("spline w' fwd = rev", egr, 1e-14);
    check("spline k = g'(r)/r", ek, 1e-13);
    double jg = 0, jw = 0;
    for (double u : {0.5, 1.0}) {
        const double a = (u * p.s.h) * (u * p.s.h) * (1 - 1e-15), b = (u * p.s.h) * (u * p.s.h) * (1 + 1e-15);
        double ga, gb, ka, kb; p.s.spline_gk(a, ga, ka); p.s.spline_gk(b, gb, kb);
        jg = std::max(jg, std::fabs((ga - gb) / ga)); jw = std::max(jw, std::fabs((p.s.spline_w(a) - p.s.spline_w(b)) / p.s.spline_w(a)));
    }
    check("spline jumps |dg|/g at u = 0.5, 1", jg, 5e-14);
    check("spline jumps |dw|/w at u = 0.5, 1", jw, 5e-14);
}

// ---------------------------------------------------------------- 2-body perturbation rates
struct RP { double mu; };
__attribute__((noinline)) static void elems(const double* s, double* o, const RP* p) {   // the code's own E, a, |e|
    const double f0[3] = {0, 0, 0};
    const PertRates2<double> r = pert_rates_2body<double>(s, s + 3, f0, p->mu);
    o[0] = r.E; o[1] = r.a; o[2] = r.e;
}
static void rates() {
    const double es[] = {1e-10, 1e-8, 1e-4, 0.1, 0.5, 0.9, 0.99, 0.999999, 1.5, 10.0, 346.0};
    for (double e : es) {
        double wE = 0, wa = 0, we = 0;
        for (int t = 0; t < 200; t++) {
            RP p{4.3e-3 * logU(2e-3, 2.0)};
            const double q = 5e-3 * logU(1e-2, 1e2), pp = q * (1 + e);
            double P[3], Q[3]; rdir(P); rdir(Q);
            const double pq = P[0] * Q[0] + P[1] * Q[1] + P[2] * Q[2];
            for (int k = 0; k < 3; k++) Q[k] -= pq * P[k];
            const double qn = nrm(Q); for (int k = 0; k < 3; k++) Q[k] /= qn;
            const double numax = e < 1 ? M_PI : std::acos(-1.0 / e) * 0.999, nu = U(-numax, numax);
            const double rn = pp / (1 + e * std::cos(nu)), vr = std::sqrt(p.mu / pp) * e * std::sin(nu), vt = std::sqrt(p.mu / pp) * (1 + e * std::cos(nu));
            double s[6], ds[6], f[3], fd[3], o[3], dO[3];
            rdir(fd);
            const double fm = 1e-3 * p.mu / (rn * rn) * logU(1e-3, 1.0);
            for (int k = 0; k < 3; k++) {
                const double rh = std::cos(nu) * P[k] + std::sin(nu) * Q[k], th = -std::sin(nu) * P[k] + std::cos(nu) * Q[k];
                s[k] = rn * rh; s[3 + k] = vr * rh + vt * th; f[k] = fm * fd[k]; ds[k] = 0; ds[3 + k] = f[k];
            }
            __enzyme_fwddiff<void>((void*)elems, enzyme_dup, s, ds, enzyme_dupnoneed, o, dO, enzyme_const, &p);
            const PertRates2<double> R = pert_rates_2body<double>(s, s + 3, f, p.mu);
            double L[3] = {s[1] * s[5] - s[2] * s[4], s[2] * s[3] - s[0] * s[5], s[0] * s[4] - s[1] * s[3]};
            const double v = nrm(s + 3), sE = v * fm, sg = (fm * nrm(L) + v * rn * fm) / p.mu;
            wE = std::max(wE, std::fabs(R.Edot - dO[0]) / sE);
            wa = std::max(wa, std::fabs(R.adot - dO[1]) / (2 * R.a * R.a * sE / p.mu));
            we = std::max(we, std::fabs(R.edot - dO[2]) / sg);
        }
        char w[96];
        snprintf(w, sizeof w, "pert_rates Edot = dE/dt  (e = %g)", e); check(w, wE, 1e-14);
        snprintf(w, sizeof w, "pert_rates adot = da/dt  (e = %g)", e); check(w, wa, 1e-14);
        snprintf(w, sizeof w, "pert_rates edot = d|e|/dt (e = %g)", e); check(w, we, 1e-14);
    }
}

// ---------------------------------------------------------------- FSI gradient kernel = J a
constexpr int NMAX = 32;
struct NB { int n; double m[NMAX], G, soft2; };
__attribute__((noinline)) static void field(const double* x, double* a, const NB* c) {
    for (int i = 0; i < c->n; i++) {
        double s[3] = {0, 0, 0};
        for (int j = 0; j < c->n; j++) {
            if (j == i) continue;
            const double d[3] = {x[3 * j] - x[3 * i], x[3 * j + 1] - x[3 * i + 1], x[3 * j + 2] - x[3 * i + 2]};
            const double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + c->soft2, ir = 1.0 / std::sqrt(r2);
            for (int q = 0; q < 3; q++) s[q] += c->G * c->m[j] * d[q] * ir * ir * ir;
        }
        for (int q = 0; q < 3; q++) a[3 * i + q] = s[q];
    }
}
static void fsi_grad() {
    double worst = 0, wrev = 0;
    for (int n : {2, 3, 8, 32})
        for (int soft = 0; soft < 2; soft++)
            for (int t = 0; t < 20; t++) {
                NB c; c.n = n; c.G = 4.3e-3; c.soft2 = soft ? 1e-8 : 0.0;
                std::vector<double> x(3 * n), a(3 * n), Ja(3 * n), dum(3 * n);
                for (int k = 0; k < n; k++) { double d[3]; rdir(d); const double r = 1e-2 * logU(1e-2, 1.0); for (int q = 0; q < 3; q++) x[3 * k + q] = r * d[q]; c.m[k] = logU(1e-2, 1.0); }
                field(x.data(), a.data(), &c);
                __enzyme_fwddiff<void>((void*)field, enzyme_dup, x.data(), a.data(), enzyme_dupnoneed, dum.data(), Ja.data(), enzyme_const, &c);
                // reverse: w^T (J a) = (J^T w) . a for a random w
                std::vector<double> wv(3 * n), gx(3 * n, 0.0), out(3 * n);
                for (auto& q : wv) q = U(-1, 1);
                std::vector<double> wcopy(wv);
                __enzyme_autodiff<void>((void*)field, enzyme_dup, x.data(), gx.data(), enzyme_dupnoneed, out.data(), wcopy.data(), enzyme_const, &c);
                double l = 0, r = 0, sc = 0;
                for (int k = 0; k < 3 * n; k++) { l += wv[k] * Ja[k]; r += gx[k] * a[k]; sc += std::fabs(wv[k] * Ja[k]); }
                wrev = std::max(wrev, std::fabs(l - r) / sc);
                // production GradK (SOFT selects n2_r2 with soft2)
                std::vector<double> X(n), Y(n), Z(n), AX(n), AY(n), AZ(n), SX(n), SY(n), SZ(n), M(c.m, c.m + n);
                for (int i = 0; i < n; i++) { X[i] = x[3 * i]; Y[i] = x[3 * i + 1]; Z[i] = x[3 * i + 2]; SX[i] = a[3 * i]; SY[i] = a[3 * i + 1]; SZ[i] = a[3 * i + 2]; }
                auto setup = [&](auto& k) {
                    k.x = {X.data()}; k.y = {Y.data()}; k.z = {Z.data()}; k.m = {M.data()};
                    k.ax = {AX.data()}; k.ay = {AY.data()}; k.az = {AZ.data()};
                    k.isx = {SX.data()}; k.isy = {SY.data()}; k.isz = {SZ.data()}; k.jsx = {SX.data()}; k.jsy = {SY.data()}; k.jsz = {SZ.data()};
                    k.ioff = k.joff = 0; k.gmag = {nullptr}; k.sid = nullptr;
                    k.coeff = c.G; k.soft2 = c.soft2; k.inv_dt2 = 1.0;
                };
                GradK<false, false> k0{}; GradK<false, true> k1{}; setup(k0); setup(k1);
                auto pairsum = [&](int i, double* sum) {
                    double s8[8];
                    if (soft) { auto snk = k1.sink(i); for (int j = 0; j < n; j++) if (j != i) { k1.source(j, s8); k1.pair(snk, i, s8, j, sum, nullptr); } }
                    else { auto snk = k0.sink(i); for (int j = 0; j < n; j++) if (j != i) { k0.source(j, s8); k0.pair(snk, i, s8, j, sum, nullptr); } }
                };
                for (int i = 0; i < n; i++) {
                    double sum[3] = {0, 0, 0}, scale = 0;
                    pairsum(i, sum);
                    for (int j = 0; j < n; j++) {
                        if (j == i) continue;
                        const double d[3] = {x[3 * j] - x[3 * i], x[3 * j + 1] - x[3 * i + 1], x[3 * j + 2] - x[3 * i + 2]};
                        const double da[3] = {a[3 * j] - a[3 * i], a[3 * j + 1] - a[3 * i + 1], a[3 * j + 2] - a[3 * i + 2]};
                        const double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + c.soft2;
                        scale += 4 * c.G * c.m[j] * nrm(da) / (r2 * std::sqrt(r2));
                    }
                    const double e[3] = {sum[0] - Ja[3 * i], sum[1] - Ja[3 * i + 1], sum[2] - Ja[3 * i + 2]};
                    worst = std::max(worst, nrm(e) / scale);
                }
            }
    check("FSI GradK (n2_kernels.h) = Enzyme J a", worst, 1e-14);
    check("force field J: fwd w^T(J a) = rev (J^T w).a", wrev, 1e-14);
}

// ---------------------------------------------------------------- monopole/dipole kernels (nested Enzyme)
struct PsiP { double soft2; };
__attribute__((noinline)) static double psi(const double* d, const PsiP* p) { return -1.0 / std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + p->soft2); }
__attribute__((noinline)) static void grad_psi(const double* d, double* g, const PsiP* p) {
    g[0] = g[1] = g[2] = 0; __enzyme_autodiff<void>((void*)psi, enzyme_dup, d, g, enzyme_const, p);
}
__attribute__((noinline)) static void hvp(const double* d, const double* v, double* hv, const PsiP* p) {
    double g[3], dg[3] = {0, 0, 0};
    __enzyme_fwddiff<void>((void*)grad_psi, enzyme_dup, d, v, enzyme_dupnoneed, g, dg, enzyme_const, p);
    hv[0] = dg[0]; hv[1] = dg[1]; hv[2] = dg[2];
}
__attribute__((noinline)) static void force_unit(const double* d, double* a, const PsiP* p) {
    const double s2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + p->soft2, ir = 1.0 / std::sqrt(s2), f = ir * ir * ir;
    a[0] = f * d[0]; a[1] = f * d[1]; a[2] = f * d[2];
}
static void kernels() {
    double eh = 0, e3 = 0, eb = 0;
    for (int t = 0; t < 2000; t++) {
        PsiP p{(t & 1) ? 0.0 : logU(1e-4, 1.0)};
        double d[3]; rdir(d); const double r = logU(1e-2, 1e2); for (int q = 0; q < 3; q++) d[q] *= r;
        const double s2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + p.soft2, s = std::sqrt(s2);
        double Jf[3][3];
        for (int l = 0; l < 3; l++) {
            double el[3] = {0, 0, 0}; el[l] = 1; double H[3], K[3] = {0, 0, 0}, a[3], da[3];
            hvp(d, el, H, &p);
            gradient_correction_interaction(d[0], d[1], d[2], el[0], el[1], el[2], s2, 1.0, K[0], K[1], K[2]);
            for (int q = 0; q < 3; q++) eh = std::max(eh, std::fabs(H[q] - K[q]) * s2 * s);
            __enzyme_fwddiff<void>((void*)force_unit, enzyme_dup, d, el, enzyme_dupnoneed, a, da, enzyme_const, &p);
            for (int q = 0; q < 3; q++) Jf[q][l] = da[q];
        }
        double a[3], dd[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, da[9];
        __enzyme_fwddiff<void>((void*)force_unit, enzyme_width, 3, enzyme_dupv, sizeof(double) * 3, d, dd,
                               enzyme_dupnoneedv, sizeof(double) * 3, a, da, enzyme_const, &p);
        for (int l = 0; l < 3; l++) for (int q = 0; q < 3; q++) eb = std::max(eb, std::fabs(da[3 * l + q] - Jf[q][l]) * s2 * s);
        double D[12], w[3] = {U(-1, 1), U(-1, 1), U(-1, 1)}, ref[3] = {0, 0, 0}, c[3] = {0, 0, 0};
        for (double& q : D) q = U(-1, 1);
        for (int l = 0; l < 3; l++) {
            double el[3] = {0, 0, 0}; el[l] = 1;
            double v[3] = {D[3 * l] + D[9 + l] * w[0], D[3 * l + 1] + D[9 + l] * w[1], D[3 * l + 2] + D[9 + l] * w[2]}, hv[3], dhv[3] = {0, 0, 0};
            __enzyme_fwddiff<void>((void*)hvp, enzyme_dup, d, el, enzyme_const, v, enzyme_dupnoneed, hv, dhv, enzyme_const, &p);
            for (int q = 0; q < 3; q++) ref[q] += dhv[q];
        }
        gradient_dipole_interaction(d[0], d[1], d[2], w[0], w[1], w[2], s2, 1.0, D, c[0], c[1], c[2]);
        double sc = 0; for (double q : D) sc = std::max(sc, std::fabs(q));
        const double e[3] = {c[0] - ref[0], c[1] - ref[1], c[2] - ref[2]};
        e3 = std::max(e3, nrm(e) / (sc * 3 * (1 + nrm(w)) / (s2 * s2)));
    }
    check("K (gradient_correction_interaction) = Hessian(-1/s), fwd/rev", eh, 1e-14);
    check("force Jacobian batched fwd (width 3) = fwd", eb, 1e-14);
    check("gradient_dipole_interaction = d^3(-1/s) : (D + P w^T)", e3, 1e-14);
}

// ---------------------------------------------------------------- tree cell term vs J a of the cell's direct sum
struct CellP { int n; const double* m; double soft2; };
__attribute__((noinline)) static void cell_force(const double* x, double* f, const CellP* c) {   // sink 0, sources 1..n-1
    double s[3] = {0, 0, 0};
    for (int j = 1; j < c->n; j++) {
        const double d[3] = {x[3 * j] - x[0], x[3 * j + 1] - x[1], x[3 * j + 2] - x[2]};
        const double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + c->soft2, ir = 1.0 / std::sqrt(r2);
        for (int q = 0; q < 3; q++) s[q] += c->m[j] * d[q] * ir * ir * ir;
    }
    f[0] = s[0]; f[1] = s[1]; f[2] = s[2];
}
// median relative error of the cell term, monopole and monopole + dipole (cell dipole as Octree::compute_cell_accelerations)
static void cell_term(int field_kind, double sD, double& med_mono, double& med_dip) {
    std::vector<double> em, ed;
    for (int t = 0; t < 400; t++) {
        const int n = 9; double m[n]; std::vector<double> x(3 * n, 0.0), a(3 * n);
        CellP cp{n, m, (t & 1) ? 0.0 : 1e-6};
        for (int k = 0; k < n; k++) m[k] = logU(1e-3, 1);
        double dc[3]; rdir(dc);
        for (int k = 1; k < n; k++) { double o[3]; rdir(o); const double rr = U(0, 1) * sD; for (int q = 0; q < 3; q++) x[3 * k + q] = dc[q] + rr * o[q]; }
        double M = 0, com[3] = {0, 0, 0};
        for (int k = 1; k < n; k++) { M += m[k]; for (int q = 0; q < 3; q++) com[q] += m[k] * x[3 * k + q]; }
        for (int q = 0; q < 3; q++) com[q] /= M;
        for (int k = 0; k < n; k++) {   // 0: a = A (x - com) on the cell, a_0 = 0 (w = 0); 1: + the cell's own field
            for (int q = 0; q < 3; q++) a[3 * k + q] = k ? (q + 1.0) * (x[3 * k + q] - com[q]) + 0.5 * (x[3 * k + (q + 1) % 3] - com[(q + 1) % 3]) : 0.0;
            if (field_kind == 1)
                for (int j = 1; j < n; j++) {
                    if (j == k) continue;
                    const double d[3] = {x[3 * j] - x[3 * k], x[3 * j + 1] - x[3 * k + 1], x[3 * j + 2] - x[3 * k + 2]};
                    const double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + cp.soft2, ir = 1.0 / std::sqrt(r2);
                    for (int q = 0; q < 3; q++) a[3 * k + q] += m[j] * d[q] * ir * ir * ir;
                }
        }
        double f[3], Ja[3];
        __enzyme_fwddiff<void>((void*)cell_force, enzyme_dup, x.data(), a.data(), enzyme_dupnoneed, f, Ja, enzyme_const, &cp);
        double ab[3] = {0, 0, 0}, S[9] = {}, P[3] = {}, dip[12];
        for (int k = 1; k < n; k++) for (int q = 0; q < 3; q++) ab[q] += m[k] * a[3 * k + q] / M;
        const double* ar = &a[3];
        for (int k = 1; k < n; k++) {
            const double md[3] = {m[k] * (x[3 * k] - com[0]), m[k] * (x[3 * k + 1] - com[1]), m[k] * (x[3 * k + 2] - com[2])};
            for (int u = 0; u < 3; u++) { P[u] += md[u]; for (int v = 0; v < 3; v++) S[3 * u + v] += md[u] * (a[3 * k + v] - ar[v]); }
        }
        for (int u = 0; u < 3; u++) { dip[9 + u] = P[u]; for (int v = 0; v < 3; v++) dip[3 * u + v] = S[3 * u + v] + P[u] * (ar[v] - ab[v]); }
        const double d[3] = {com[0] - x[0], com[1] - x[1], com[2] - x[2]}, w[3] = {ab[0] - a[0], ab[1] - a[1], ab[2] - a[2]};
        const double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + cp.soft2;
        double c1[3] = {0, 0, 0};
        gradient_correction_interaction(d[0], d[1], d[2], w[0], w[1], w[2], r2, M, c1[0], c1[1], c1[2]);
        double c2[3] = {c1[0], c1[1], c1[2]};
        gradient_dipole_interaction(d[0], d[1], d[2], w[0], w[1], w[2], r2, 1.0, dip, c2[0], c2[1], c2[2]);
        const double e1[3] = {c1[0] - Ja[0], c1[1] - Ja[1], c1[2] - Ja[2]}, e2[3] = {c2[0] - Ja[0], c2[1] - Ja[1], c2[2] - Ja[2]};
        em.push_back(nrm(e1) / nrm(Ja)); ed.push_back(nrm(e2) / nrm(Ja));
    }
    std::sort(em.begin(), em.end()); std::sort(ed.begin(), ed.end());
    med_mono = em[em.size() / 2]; med_dip = ed[ed.size() / 2];
}
static void tree_cell() {
    const char* nm[2] = {"a = A (x - com), w = 0", "a = linear + the cell's own field"};
    for (int f = 0; f < 2; f++) {
        double m1, d1, m2, d2;
        cell_term(f, 0.1, m1, d1); cell_term(f, 0.0125, m2, d2);
        std::printf("     tree cell term [%s]: median rel err monopole %.2e / %.2e, + dipole %.2e / %.2e (s/D 0.1 / 0.0125)\n",
                    nm[f], m1, m2, d1, d2);
        char w[128];
        snprintf(w, sizeof w, "tree cell + dipole: first order in s/D [%s]", nm[f]);
        check(w, std::fabs(std::log(d1 / d2) / std::log(8.0) - 1.0), 0.25);
        snprintf(w, sizeof w, "tree cell + dipole at s/D = 0.0125 [%s]", nm[f]);
        check(w, d2, 0.02);
    }
}

int main() {
    spline();
    rates();
    fsi_grad();
    kernels();
    tree_cell();
    std::printf(g_fail ? "FAIL (%d)\n" : "PASS\n", g_fail);
    return g_fail;
}
