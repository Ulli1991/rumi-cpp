// Target 4: FSI step (fsi_step, non-tree path) and one fixed-partition HHS level (hhs_fsi_recursive_impl: cross kicks
// with the cross gradient), kernels mirrored from n2_kernels.h ForceK / GradK (pair formulas verbatim; Kahan
// compensation of kick_velocities and of the n2 driver omitted: it does not change the map mathematically).
// z = (x, p), p = m v. Also: map jump where a level assignment flips; d(dE)/d(dt) (reverse vs forward, local order).
// Build (from the repository root; logh_ctrl.cpp: python3.11 tests/enzyme/symplectic/make_ctrl.py <dir>):
//   clang++ -std=c++20 -O2 -fplugin=ClangEnzyme-18.so -Isrc/collisional tests/enzyme/symplectic/symp_fsi.cpp -o symp_fsi
//   (Enzyme)
#include "jac.h"
template <typename... T> double __enzyme_fwddiff_scalar(void*, T...);   // Enzyme matches the name prefix

static int N = 5, NS = 3;           // slow set [0, NS), fast set [NS, N)
static double m[8], G = 1.0, soft2 = 0.0, DT = 0.05;
static int mode = 0;                // 0: fsi_step on all; 1: HHS level (S, F)
constexpr double FSI_KICK = 1.0 / 6.0, FSI_DRIFT = 0.5, FSI_MOD_KICK = 2.0 / 3.0, GRAD_COEFF = 1.0 / 24.0;

// ForceK: a_i = sum_j G m_j d / (r2 + soft2)^(3/2), sinks [i0,i1), sources [j0,j1), overwrites a_i
static void accel(const double* x, double* a, int i0, int i1, int j0, int j1) {
    for (int i = i0; i < i1; i++) {
        double s[3] = {0, 0, 0};
        for (int j = j0; j < j1; j++) {
            const bool w = i != j;
            double dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
            double r2 = w ? dx * dx + dy * dy + dz * dz + soft2 : 1.0;
            double ir = 1.0 / std::sqrt(r2);
            double f = (w ? G * m[j] : 0.0) * ir * ir * ir;
            s[0] += f * dx; s[1] += f * dy; s[2] += f * dz;
        }
        for (int k = 0; k < 3; k++) a[3 * i + k] = s[k];
    }
}
// GradK: a_i <- a0_i + coeff sum_j m_j [r2 da - 3 (r.da) d] / r^5, snapshots a0 (sinks) and a0 (sources)
static void grad(const double* x, double* a, const double* snap, int i0, int i1, int j0, int j1, double dt) {
    const double coeff = G * dt * dt * GRAD_COEFF;
    for (int i = i0; i < i1; i++) {
        double s[3] = {0, 0, 0};
        for (int j = j0; j < j1; j++) {
            const bool w = i != j;
            double dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
            double dax = snap[3 * j] - snap[3 * i], day = snap[3 * j + 1] - snap[3 * i + 1], daz = snap[3 * j + 2] - snap[3 * i + 2];
            double r2 = w ? dx * dx + dy * dy + dz * dz + soft2 : 1.0;
            double ir = 1.0 / std::sqrt(r2), ir2 = ir * ir;
            double f = (w ? coeff * m[j] : 0.0) * ir2 * ir2 * ir;
            double rda = dx * dax + dy * day + dz * daz;
            s[0] += f * (r2 * dax - 3.0 * rda * dx); s[1] += f * (r2 * day - 3.0 * rda * dy); s[2] += f * (r2 * daz - 3.0 * rda * dz);
        }
        for (int k = 0; k < 3; k++) a[3 * i + k] = snap[3 * i + k] + s[k];
    }
}
static void kick(double* v, const double* a, int i0, int i1, double h) { for (int i = 3 * i0; i < 3 * i1; i++) v[i] += a[i] * h; }
static void drift(double* x, const double* v, int i0, int i1, double h) { for (int i = 3 * i0; i < 3 * i1; i++) x[i] += v[i] * h; }

static void fsi(double* x, double* v, int i0, int i1, double dt) {
    double a[24], snap[24];
    accel(x, a, i0, i1, i0, i1); kick(v, a, i0, i1, dt * FSI_KICK);
    drift(x, v, i0, i1, dt * FSI_DRIFT);
    accel(x, a, i0, i1, i0, i1);
    std::memcpy(snap, a, sizeof(a));
    grad(x, a, snap, i0, i1, i0, i1, dt);
    kick(v, a, i0, i1, dt * FSI_MOD_KICK);
    drift(x, v, i0, i1, dt * FSI_DRIFT);
    accel(x, a, i0, i1, i0, i1); kick(v, a, i0, i1, dt * FSI_KICK);
}
static void cross(const double* x, double* a) { accel(x, a, 0, NS, NS, N); accel(x, a, NS, N, 0, NS); }
static void hhs(double* x, double* v, double dt) {
    double a[24], snap[24];
    cross(x, a); kick(v, a, 0, N, dt * FSI_KICK);
    fsi(x, v, 0, NS, dt * 0.5);
    fsi(x, v, NS, N, dt * 0.5);                      // recursion of F at pivot dt/2 (all slow there)
    cross(x, a); std::memcpy(snap, a, sizeof(a));
    grad(x, a, snap, 0, NS, NS, N, dt); grad(x, a, snap, NS, N, 0, NS, dt);   // cross_gradient_correction
    kick(v, a, 0, N, dt * FSI_MOD_KICK);
    fsi(x, v, NS, N, dt * 0.5);
    fsi(x, v, 0, NS, dt * 0.5);
    cross(x, a); kick(v, a, 0, N, dt * FSI_KICK);
}
static void fmap(const double* z, double* out) {
    double x[24], v[24];
    for (int i = 0; i < 3 * N; i++) { x[i] = z[i]; v[i] = z[3 * N + i] / m[i / 3]; }
    if (mode == 0) fsi(x, v, 0, N, DT); else hhs(x, v, DT);
    for (int i = 0; i < 3 * N; i++) { out[i] = x[i]; out[3 * N + i] = v[i] * m[i / 3]; }
}
static double energy(const double* x, const double* v) {
    double E = 0;
    for (int i = 0; i < N; i++) {
        E += 0.5 * m[i] * (v[3 * i] * v[3 * i] + v[3 * i + 1] * v[3 * i + 1] + v[3 * i + 2] * v[3 * i + 2]);
        for (int j = i + 1; j < N; j++) {
            const double dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
            E -= G * m[i] * m[j] / std::sqrt(dx * dx + dy * dy + dz * dz + soft2);
        }
    }
    return E;
}
// energy error after K steps of sizes dts[k] (mode 0 FSI, 1 HHS level) from the state Z0
static double Z0[DMAX];
static int KST = 1;
__attribute__((noinline)) static double dE_steps(const double* dts) {
    double x[24], v[24];
    for (int i = 0; i < 3 * N; i++) { x[i] = Z0[i]; v[i] = Z0[3 * N + i] / m[i / 3]; }
    const double E0 = energy(x, v);
    for (int k = 0; k < KST; k++) { if (mode == 0) fsi(x, v, 0, N, dts[k]); else hhs(x, v, dts[k]); }
    return energy(x, v) - E0;
}
template <typename... T> double __enzyme_autodiff(void*, T...);
// kinetic and potential energy after the same steps (scale of the cancellation in d(dE)/d(dt))
__attribute__((noinline)) static void TU_steps(const double* dts, double* tu) {
    double x[24], v[24];
    for (int i = 0; i < 3 * N; i++) { x[i] = Z0[i]; v[i] = Z0[3 * N + i] / m[i / 3]; }
    for (int k = 0; k < KST; k++) { if (mode == 0) fsi(x, v, 0, N, dts[k]); else hhs(x, v, dts[k]); }
    double z[24] = {};
    tu[1] = energy(x, z);
    tu[0] = energy(x, v) - tu[1];
}
static double M[DMAX][DMAX], Mb[DMAX][DMAX], Mfd[DMAX][DMAX];
int main() {
    int fails = 0;
    // 3 slow bodies + a fast binary
    const double m0[5] = {1.0, 0.7, 0.5, 0.3, 0.2};
    const double x0[15] = {0, 0, 0, 3, 0.5, -0.2, -2, 2.5, 0.4, 0.8, -3, 0.1, 0.8, -2.8, 0.15};
    const double v0[15] = {0, 0.1, 0, -0.1, 0.5, 0.05, 0.3, -0.2, 0, 0.4, 0.1, 0, 0.4, 0.1 - 0.9, 0.2};
    for (int i = 0; i < 5; i++) m[i] = m0[i];
    std::printf("# mode soft2 dt | defect(enzyme) defect(fd) det-1 | rev_val rev_jac | P L (equivariance + invariance rows)\n");
    for (int md : {0, 1})
        for (double s2 : {0.0, 1e-2})
            for (double dt : {0.1, 0.03, 0.01}) {
                mode = md; soft2 = s2; DT = dt;
                const int d = 6 * N;
                double z[DMAX], zo[DMAX], zb[DMAX];
                for (int i = 0; i < 15; i++) { z[i] = x0[i]; z[15 + i] = v0[i] * m[i / 3]; }
                fmap(z, zo);
                jac_enzyme<fmap>(z, d, M);
                jac_fd(fmap, z, d, Mfd, 1e-4, 1.0);
                DT = -dt; fmap(zo, zb); jac_enzyme<fmap>(zo, d, Mb); DT = dt;
                double rv = 0; for (int j = 0; j < d; j++) rv = std::max(rv, std::fabs(zb[j] - z[j]) / std::max(1.0, std::fabs(z[j])));
                // translation equivariance and P, L invariance rows
                double tr = 0, dP = 0, dL = 0;
                for (int k = 0; k < 3; k++) {
                    for (int i = 0; i < d; i++) { double s = 0; for (int b = 0; b < N; b++) s += M[i][3 * b + k]; double ref = (i < 3 * N && i % 3 == k) ? 1 : 0; tr = std::max(tr, std::fabs(s - ref)); }
                    for (int j = 0; j < d; j++) { double s = 0; for (int b = 0; b < N; b++) s += M[3 * N + 3 * b + k][j]; double ref = (j >= 3 * N && j % 3 == k) ? 1 : 0; dP = std::max(dP, std::fabs(s - ref)); }
                    const int k1 = (k + 1) % 3, k2 = (k + 2) % 3;
                    for (int j = 0; j < d; j++) {
                        double s = 0, ref = 0;
                        for (int b = 0; b < N; b++) {
                            const double *q = zo + 3 * b, *p = zo + 3 * N + 3 * b, *q0 = z + 3 * b, *p0 = z + 3 * N + 3 * b;
                            s += p[k2] * M[3 * b + k1][j] + q[k1] * M[3 * N + 3 * b + k2][j] - p[k1] * M[3 * b + k2][j] - q[k2] * M[3 * N + 3 * b + k1][j];
                            if (j == 3 * b + k1) ref += p0[k2];
                            if (j == 3 * N + 3 * b + k2) ref += q0[k1];
                            if (j == 3 * b + k2) ref -= p0[k1];
                            if (j == 3 * N + 3 * b + k1) ref -= q0[k2];
                        }
                        dL = std::max(dL, std::fabs(s - ref));
                    }
                }
                std::printf("%d %5.0e %5.2f | %8.1e %8.1e %9.1e | %8.1e %8.1e | tr %8.1e dP %8.1e dL %8.1e\n", md, s2, dt, symp_defect(M, d), symp_defect(Mfd, d),
                            det(M, d) - 1, rv, prod_minus_I(Mb, M, d), tr, dP, dL);
                fails += !(symp_defect(M, d) < 1e-14) + !(prod_minus_I(Mb, M, d) < 1e-12) + !(std::max(tr, std::max(dP, dL)) < 1e-12);
            }
    // level change: the binary in the fast set (HHS level) or with everything in one FSI step; the map jumps by
    // |Phi_HHS - Phi_FSI| where the level assignment flips (each piece symplectic, rows above)
    std::printf("# level flip | dt | max |Phi_HHS(z) - Phi_FSI(z)| / max(1, |z|) | ratio to dt / 2\n");
    double prev = 0, slope = 0;
    for (double dt : {0.08, 0.04, 0.02, 0.01}) {
        soft2 = 0; DT = dt;
        double z[DMAX], za[DMAX], zb[DMAX];
        for (int i = 0; i < 15; i++) { z[i] = x0[i]; z[15 + i] = v0[i] * m[i / 3]; }
        mode = 1; fmap(z, za); mode = 0; fmap(z, zb);
        double j = 0; for (int i = 0; i < 6 * N; i++) j = std::max(j, std::fabs(za[i] - zb[i]) / std::max(1.0, std::fabs(z[i])));
        if (prev > 0) slope = std::log2(prev / j);
        std::printf("  %5.3f  %9.3e  %s%.2f\n", dt, j, prev > 0 ? "order " : "", prev > 0 ? slope : 0.0);
        prev = j;
    }
    fails += !(slope > 4.5 && slope < 5.5);   // both 4th order: the jump is a local error, O(dt^5)
    // sensitivities (d(dE)/d(dt) = grad E . dPhi/d(dt) cancels along the flow: compared on the scale |dT| + |dU|)
    // local order: one step, p = dt d(dE)/d(dt) / dE -> 5 (= d ln|dE| / d ln eta at fixed levels);
    // K = 16 steps: d(dE)/d(dt_k) of every step by one reverse sweep vs forward mode along dt, share of the largest
    std::printf("# sensitivity | mode K dt | dE | p (fwd) | rev-fwd / (|dT|+|dU|) | share of the largest step (k)\n");
    for (int md : {0, 1}) {
        mode = md; soft2 = 0;
        for (int i = 0; i < 15; i++) { Z0[i] = x0[i]; Z0[15 + i] = v0[i] * m[i / 3]; }
        double p1 = 0;
        for (double dt : {0.004, 0.002, 0.001}) {
            KST = 1; double d1[1] = {dt}, o1[1] = {dt};
            const double dE = dE_steps(d1), f = __enzyme_fwddiff_scalar((void*)dE_steps, d1, o1);
            p1 = f / dE;
            std::printf("  %d  1 %5.3f | %10.3e | %6.3f\n", md, dt, dE, p1);
        }
        fails += !(p1 > 4.5 && p1 < 5.5);
        KST = 16;
        double dts[64], g[64] = {}, ones[64], tu[2], dtu[2] = {0, 0};
        for (int k = 0; k < KST; k++) { dts[k] = 0.002; ones[k] = 0.002; }
        const double dE = dE_steps(dts);
        __enzyme_autodiff((void*)dE_steps, enzyme_dup, dts, g);
        const double fwd = __enzyme_fwddiff_scalar((void*)dE_steps, dts, ones);
        __enzyme_fwddiff((void*)TU_steps, enzyme_dup, dts, ones, enzyme_dupnoneed, tu, dtu);
        double sum = 0, big = 0; int kb = 0;
        for (int k = 0; k < KST; k++) { sum += dts[k] * g[k]; if (std::fabs(dts[k] * g[k]) > big) { big = std::fabs(dts[k] * g[k]); kb = k; } }
        const double rel = std::fabs(sum - fwd) / (std::fabs(dtu[0]) + std::fabs(dtu[1]));
        std::printf("  %d %d %5.3f | %10.3e | %6.3f | %8.1e | %.2f (%d)\n", md, KST, 0.002, dE, fwd / dE, rel, big / std::fabs(sum), kb);
        fails += !(rel < 1e-12);
    }
    std::printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails;
}
