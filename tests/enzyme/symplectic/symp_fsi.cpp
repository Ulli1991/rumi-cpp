// Target 4: FSI step (fsi_step, non-tree path) and one fixed-partition HHS level (hhs_fsi_recursive_impl: cross kicks
// with the cross gradient), kernels mirrored from n2_kernels.h ForceK / GradK (pair formulas verbatim; Kahan
// compensation of kick_velocities and of the n2 driver omitted: it does not change the map mathematically).
// z = (x, p), p = m v.
// Build (from the repository root; logh_ctrl.cpp: python3.11 tests/enzyme/symplectic/make_ctrl.py <dir>):
//   clang++ -std=c++20 -O2 -fplugin=ClangEnzyme-18.so -Isrc/collisional tests/enzyme/symplectic/symp_fsi.cpp -o symp_fsi
//   (Enzyme)
#include "jac.h"

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
static double M[DMAX][DMAX], Mb[DMAX][DMAX], Mfd[DMAX][DMAX];
int main() {
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
            }
}
