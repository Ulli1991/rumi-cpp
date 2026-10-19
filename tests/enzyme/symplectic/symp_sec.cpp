// Target 5: secular.cpp rotate_about / reshape_orbit (anonymous namespace: included). det M, symplectic defect,
// rotation equivariance, E and L invariance rows.
// Build (from the repository root; logh_ctrl.cpp: python3.11 tests/enzyme/symplectic/make_ctrl.py <dir>):
//   clang++ -std=c++20 -O2 -fplugin=ClangEnzyme-18.so -Isrc/collisional tests/enzyme/symplectic/symp_sec.cpp -o symp_sec
//   (Enzyme)
#include "../../../src/collisional/secular.cpp"
#include "../../../src/collisional/kepler.cpp"
#include "jac.h"

static int g_mode = 0;   // 0 rotate_about fixed n; 1 rotate_about own L-hat; 2 reshape a1 = a(z), k = 1; 3 reshape a1, k const
static double g_dw = 0.3, g_a1 = 1.0, g_k = 1.0, g_dM = 0.7;
static const double NFIX[3] = {0.48, 0.6, 0.64};
static void smap(const double* z, double* o) {
    double r[3] = {z[0], z[1], z[2]}, v[3] = {z[3], z[4], z[5]};
    if (g_mode == 0) rotate_about(r, v, NFIX, g_dw);
    else if (g_mode == 1) {
        double h[3]; cross3(r, v, h); const double hn = std::sqrt(dot3(h, h));
        for (int c = 0; c < 3; c++) h[c] /= hn;
        rotate_about(r, v, h, g_dw);
    } else if (g_mode == 2) {
        const double a = 1.0 / (2.0 / std::sqrt(dot3(r, r)) - dot3(v, v));
        reshape_orbit(r, v, 1.0, a, 1.0, g_dM);
    } else reshape_orbit(r, v, 1.0, g_a1, g_k, g_dM);
    for (int c = 0; c < 3; c++) { o[c] = r[c]; o[3 + c] = v[c]; }
}
static double M[DMAX][DMAX], Mfd[DMAX][DMAX];
int main() {
    std::printf("# mode e f0 (a1 k) | defect(enz) defect(fd) det-1 | rotEq dE-row dL-row\n");
    for (int mode : {0, 1, 2, 3})
        for (double e : {0.0, 1e-4, 0.3, 0.9, 0.999})
            for (double f0 : {0.0, -0.3, 2.5}) {
                g_mode = mode;
                const double a = 1.0, p = a * (1 - e * e), r = p / (1 + e * std::cos(f0));
                double z[6] = {r * std::cos(f0), r * std::sin(f0) * 0.8, r * std::sin(f0) * 0.6,
                               -std::sin(f0) / std::sqrt(p), (e + std::cos(f0)) / std::sqrt(p) * 0.8, (e + std::cos(f0)) / std::sqrt(p) * 0.6};
                g_a1 = 1.02; g_k = e > 0 ? 0.97 : 1.0;
                double zo[6]; smap(z, zo);
                jac_enzyme<smap>(z, 6, M);
                jac_fd(smap, z, 6, Mfd, 1e-5, 1.0);
                double req = 0, dE = 0, dL = 0;
                for (int k = 0; k < 3; k++) {
                    double w[3] = {0, 0, 0}; w[k] = 1;
                    double xi[6], xo[6];
                    for (int b = 0; b < 2; b++) { cross3(w, z + 3 * b, xi + 3 * b); cross3(w, zo + 3 * b, xo + 3 * b); }
                    for (int i = 0; i < 6; i++) { double s = 0; for (int j = 0; j < 6; j++) s += M[i][j] * xi[j]; req = std::max(req, std::fabs(s - xo[i])); }
                }
                auto gE = [](const double* y, double* g) { const double rr = std::sqrt(dot3(y, y)); for (int k = 0; k < 3; k++) { g[k] = y[k] / (rr * rr * rr); g[3 + k] = y[3 + k]; } };
                double g0[6], g1[6]; gE(z, g0); gE(zo, g1);
                for (int j = 0; j < 6; j++) { double s = 0; for (int i = 0; i < 6; i++) s += g1[i] * M[i][j]; dE = std::max(dE, std::fabs(s - g0[j])); }
                for (int k = 0; k < 3; k++) {
                    const int a1 = (k + 1) % 3, b1 = (k + 2) % 3;
                    for (int j = 0; j < 6; j++) {
                        const double s = zo[3 + b1] * M[a1][j] + zo[a1] * M[3 + b1][j] - zo[3 + a1] * M[b1][j] - zo[b1] * M[3 + a1][j];
                        double ref = 0;
                        if (j == a1) ref += z[3 + b1]; if (j == 3 + b1) ref += z[a1]; if (j == b1) ref -= z[3 + a1]; if (j == 3 + a1) ref -= z[b1];
                        dL = std::max(dL, std::fabs(s - ref));
                    }
                }
                std::printf("%d %6.4g %5.2f | %8.1e %8.1e %9.2e | %8.1e %8.1e %8.1e\n", mode, e, f0, symp_defect(M, 6), symp_defect(Mfd, 6), det(M, 6) - 1, req, dE, dL);
            }
}
