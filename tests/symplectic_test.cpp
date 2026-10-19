// ctest logh_symplectic: structural properties of the subsystem maps from their Jacobians (4th-order central
// differences in double; tests/enzyme/symplectic/ has the same with exact Enzyme Jacobians and the parameter maps).
//
// 1. LogH leapfrog (logh_leapfrog_step) in the extended phase space z = (q, t; p, B), p = m v, p_t = B = -E.
//    BIFROST (Rantala et al. 2023, Eqs. 2-3; Mikkola & Tanikawa 1999; Preto & Tremaine 1999): drift with
//    dt/ds = 1/(T + B), kicks with dv/ds = a / Ubar, Ubar = -U(q). Both sub-steps are exact flows of
//    Gamma = ln(T + p_t) - ln(-U), so the leapfrog is symplectic in z: M^T J M = J. The code kicks with
//    1/Omega, Omega = T(v_mid) + B (cubic): v' depends on v and B at fixed q, so the kick (and the step) is not
//    symplectic in z; the exact flow of the code's equations keeps Omega dV, not dV (det = Omega(z) / Omega(z')).
//    Measured (Enzyme, e = 0.5 binary, ds = 1e-2): defect 4.0e-2 (O(ds): 0.32 at ds = 0.1, 4e-3 at 1e-3), on the
//    energy shell 1.9e-5 (O(ds^3)); the BIFROST kick 2.4e-16. The switch to the BIFROST kick was rejected by the
//    200k cluster A/B (docs/CASES.md R2), so the code keeps 1/Omega: the defect is checked as a regression guard
//    (it must stay at its measured O(ds) value, 1.3x bound), not against zero.
// 2. Guards (pass now): time reversibility step(-ds) o step(ds) = id; the derivative commutes with translations and
//    rotations (P, L); kepler_drift symplectic (the exact flow, Enzyme defect <= 4e-13 relative over e in
//    [0, 10] and dt <= 1000 P).
// Pass: LogH defect <= 1.3x measured (4.0e-2 n=2, 4.9e-2 n=3; the BIFROST kick gives < 1e-8), Kepler < 1e-7 (noise up to 5e-9 at e = 3),
// reversibility < 1e-13, equivariance < 1e-10 (noise 6e-12). Measured with g++ 14 -O2, Raven login node.
#include "logh.h"
#include "kepler.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

constexpr int DM = 2 * (3 * LOGH_MAX_BODIES + 1);
static int g_n = 2;
static double g_mass[LOGH_MAX_BODIES], g_ds = 1e-2;
static double g_mu = 1.0, g_dt = 1.0;

using Map = void (*)(const double*, double*);

static void logh_map(const double* z, double* out) {
    LogHState s;
    std::memset(&s, 0, sizeof(s));
    const int n3 = 3 * g_n;
    s.n_bodies = g_n;
    for (int i = 0; i < g_n; i++) s.mass[i] = g_mass[i];
    for (int i = 0; i < n3; i++) { s.pos[i] = z[i]; s.vel[i] = z[n3 + 1 + i] / g_mass[i / 3]; }
    s.t = z[n3];
    s.B = z[2 * n3 + 1];
    logh_leapfrog_step(s, g_ds, 1.0);
    for (int i = 0; i < n3; i++) { out[i] = s.pos[i]; out[n3 + 1 + i] = s.vel[i] * g_mass[i / 3]; }
    out[n3] = s.t;
    out[2 * n3 + 1] = s.B;
}
static void kepler_map(const double* z, double* out) {
    double r[3] = {z[0], z[1], z[2]}, v[3] = {z[3], z[4], z[5]};
    kepler_drift(r, v, g_mu, g_dt);
    for (int k = 0; k < 3; k++) { out[k] = r[k]; out[3 + k] = v[k]; }
}
static void jac(Map F, const double* z, int d, double M[DM][DM]) {
    for (int j = 0; j < d; j++) {
        const double h = 1e-4 * std::max(std::fabs(z[j]), 1.0);
        double zp[DM], o[4][DM];
        const double sh[4] = {h, -h, 2 * h, -2 * h};
        for (int k = 0; k < 4; k++) {
            std::memcpy(zp, z, d * sizeof(double));
            zp[j] += sh[k];
            F(zp, o[k]);
        }
        for (int i = 0; i < d; i++) M[i][j] = (8.0 * (o[0][i] - o[1][i]) - (o[2][i] - o[3][i])) / (12.0 * h);
    }
}
// max |M^T J M - J| / max(1, max|M|^2), z = (q_1..q_m, p_1..p_m)
static double symp_defect(const double M[DM][DM], int d) {
    const int m = d / 2;
    double worst = 0.0, mm = 0.0;
    for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) mm = std::max(mm, std::fabs(M[i][j]));
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++) {
            long double s = 0.0L;
            for (int k = 0; k < m; k++) s += (long double)M[k][i] * M[k + m][j] - (long double)M[k + m][i] * M[k][j];
            const double J = (i < m && j == i + m) ? 1.0 : (i >= m && j == i - m) ? -1.0 : 0.0;
            worst = std::max(worst, (double)std::fabs(s - J));
        }
    return worst / std::max(1.0, mm * mm);
}
static void binary(double m1, double m2, double a, double e, double f, double* x1, double* x2, double* v1, double* v2) {
    const double mu = m1 + m2, p = a * (1 - e * e), r = p / (1 + e * std::cos(f));
    const double rx[3] = {r * std::cos(f), r * std::sin(f), 0.0};
    const double rv[3] = {-std::sin(f) * std::sqrt(mu / p), (e + std::cos(f)) * std::sqrt(mu / p), 0.0};
    for (int k = 0; k < 3; k++) {
        x1[k] = -m2 / mu * rx[k]; x2[k] = m1 / mu * rx[k]; v1[k] = -m2 / mu * rv[k]; v2[k] = m1 / mu * rv[k];
    }
}
// e = 0.5 binary 0.3 rad before pericentre (n = 2), or that binary with an inclined outer body (n = 3); B = -E
static void ics(int n, double* z) {
    g_n = n;
    const int n3 = 3 * n;
    double pos[9] = {}, vel[9] = {};
    if (n == 2) {
        g_mass[0] = 0.6; g_mass[1] = 0.4;
        binary(0.6, 0.4, 1.0, 0.5, -0.3, pos, pos + 3, vel, vel + 3);
        pos[2] += 0.01; vel[5] += 0.02;
    } else {
        g_mass[0] = 0.5; g_mass[1] = 0.3; g_mass[2] = 0.2;
        double xi[3], xo[3], vi[3], vo[3], X1[3], X2[3], V1[3], V2[3];
        binary(0.5, 0.3, 1.0, 0.5, -0.3, X1, X2, V1, V2);
        binary(0.8, 0.2, 5.0, 0.3, 1.0, xi, xo, vi, vo);
        for (int k = 0; k < 3; k++) {
            pos[k] = X1[k] + xi[k]; pos[3 + k] = X2[k] + xi[k]; pos[6 + k] = xo[k];
            vel[k] = V1[k] + vi[k]; vel[3 + k] = V2[k] + vi[k]; vel[6 + k] = vo[k];
        }
        pos[8] += 0.3; vel[2] += 0.05; vel[5] -= 0.03;
    }
    LogHState s{};
    s.n_bodies = n;
    for (int i = 0; i < n; i++) s.mass[i] = g_mass[i];
    for (int i = 0; i < n3; i++) { s.pos[i] = pos[i]; s.vel[i] = vel[i]; z[i] = pos[i]; z[n3 + 1 + i] = vel[i] * g_mass[i / 3]; }
    z[n3] = 0.3;
    z[2 * n3 + 1] = -logh_total_energy(s, 1.0);
}
// max over the 3 translations / rotations of |M xi(z) - xi(z')| (xi: generator acting on q and p, not on t, B)
static double equivariance(const double* z, const double* zo, const double M[DM][DM], int d, bool rot) {
    const int n3 = 3 * g_n;
    double worst = 0.0;
    for (int k = 0; k < 3; k++) {
        double xi[DM] = {}, xo[DM] = {};
        const double w[3] = {k == 0 ? 1.0 : 0.0, k == 1 ? 1.0 : 0.0, k == 2 ? 1.0 : 0.0};
        for (int i = 0; i < g_n; i++) {
            if (!rot) { xi[3 * i + k] = xo[3 * i + k] = 1.0; continue; }
            for (int blk = 0; blk < 2; blk++) {
                const int o = (blk ? n3 + 1 : 0) + 3 * i;
                for (int pass = 0; pass < 2; pass++) {
                    const double* x = (pass ? zo : z) + o;
                    double* g = (pass ? xo : xi) + o;
                    g[0] = w[1] * x[2] - w[2] * x[1]; g[1] = w[2] * x[0] - w[0] * x[2]; g[2] = w[0] * x[1] - w[1] * x[0];
                }
            }
        }
        for (int i = 0; i < d; i++) {
            double s = 0.0;
            for (int j = 0; j < d; j++) s += M[i][j] * xi[j];
            worst = std::max(worst, std::fabs(s - xo[i]));
        }
    }
    return worst;
}

int main() {
    int fails = 0;
    static double M[DM][DM];
    // 1. LogH leapfrog in the extended phase space
    for (int n : {2, 3}) {
        double z[DM], zo[DM], zb[DM];
        ics(n, z);
        const int d = 2 * (3 * n + 1);
        g_ds = 1e-2;
        logh_map(z, zo);
        jac(logh_map, z, d, M);
        const double def = symp_defect(M, d);
        const double tr = equivariance(z, zo, M, d, false), rot = equivariance(z, zo, M, d, true);
        g_ds = -1e-2;
        logh_map(zo, zb);
        double rev = 0.0;
        for (int j = 0; j < d; j++) rev = std::max(rev, std::fabs(zb[j] - z[j]) / std::max(1.0, std::fabs(z[j])));
        std::printf("logh_leapfrog_step n=%d ds=1e-2: |M^T J M - J| = %.2e (BIFROST kick: 2e-16; code kick 1/Omega, CASES R2: 4e-2), "
                    "reversibility %.1e, translation %.1e, rotation %.1e\n", n, def, rev, tr, rot);
        // Regression guard for the 1/Omega kick kept after CASES R2: the O(ds) defect must not grow
        const double def_bound = 1.3 * (n == 2 ? 4.0e-2 : 4.9e-2);
        if (!(def <= def_bound)) { std::printf("FAIL LogH symplectic defect %.2e above %.2e (R2 value)\n", def, def_bound); fails++; }
        if (!(rev < 1e-13)) { std::printf("FAIL LogH leapfrog not time-reversible\n"); fails++; }
        if (!(tr < 1e-10 && rot < 1e-10)) { std::printf("FAIL LogH leapfrog derivative breaks P or L\n"); fails++; }
    }
    // 2. kepler_drift: the exact two-body flow, symplectic for every e (pericentre q = 1, inclined orbit)
    double worst = 0.0;
    for (double e : {0.05, 0.3, 0.5, 0.9, 0.99, 0.99999, 1.0, 1.00005, 1.5, 3.0})
        for (double f0 : {0.0, -0.3, 2.0})
            for (double dtu : {1e-3, 0.1, 0.37}) {
                const double p = 1.0 + e, r = p / (1 + e * std::cos(f0)), c = std::cos(0.7), s = std::sin(0.7);
                const double z[6] = {r * std::cos(f0), r * std::sin(f0) * c, r * std::sin(f0) * s,
                                     -std::sin(f0) / std::sqrt(p), (e + std::cos(f0)) / std::sqrt(p) * c, (e + std::cos(f0)) / std::sqrt(p) * s};
                const double a = e == 1.0 ? 1.0 : 1.0 / std::fabs(1.0 - e);
                const double T = e < 1.0 ? 2.0 * M_PI * std::sqrt(a * a * a) : std::sqrt(a * a * a);
                g_dt = dtu * std::min(T, 100.0);
                jac(kepler_map, z, 6, M);
                worst = std::max(worst, symp_defect(M, 6));
            }
    std::printf("kepler_drift: worst |M^T J M - J| %.2e over e in [0.05, 3], dt <= 0.37 P\n", worst);
    if (!(worst < 1e-7)) { std::printf("FAIL kepler_drift not symplectic\n"); fails++; }
    std::printf(fails ? "FAIL\n" : "PASS\n");
    return fails;
}
