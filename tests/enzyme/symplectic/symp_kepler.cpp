// Target 3: kepler_drift (exact two-body flow) -- symplectic defect, Enzyme vs FD, energy/L/LRL invariance of the
// derivative, rotation equivariance, reversibility, over e, dt/P, phase.
// Build (from the repository root; logh_ctrl.cpp: python3.11 tests/enzyme/symplectic/make_ctrl.py <dir>):
//   clang++ -std=c++20 -O2 -fplugin=ClangEnzyme-18.so -Isrc/collisional tests/enzyme/symplectic/symp_kepler.cpp -o symp_kepler
//   (Enzyme; argument f: fine scan e -> 1)
#include "../../../src/collisional/kepler.cpp"
#include "jac.h"
#include <cstdlib>

static double g_dt = 1.0, g_mu = 1.0;
static int g_it = 0;
static void kmap(const double* z, double* out) {
    double r[3] = {z[0], z[1], z[2]}, v[3] = {z[3], z[4], z[5]};
    g_it = kepler_drift(r, v, g_mu, g_dt);
    for (int k = 0; k < 3; k++) { out[k] = r[k]; out[3 + k] = v[k]; }
}
// state at true anomaly f of an orbit with pericentre q = 1 (so parabolic/hyperbolic are fine), inclined
static void ics(double e, double f, double* z) {
    const double p = 1.0 * (1 + e), r = p / (1 + e * std::cos(f));
    double x[3] = {r * std::cos(f), r * std::sin(f), 0}, v[3] = {-std::sin(f) / std::sqrt(p), (e + std::cos(f)) / std::sqrt(p), 0};
    const double inc = 0.7, Om = 0.3, w = 1.1;
    auto rot = [&](double* a) {
        double b[3];
        b[0] = std::cos(w) * a[0] - std::sin(w) * a[1]; b[1] = std::sin(w) * a[0] + std::cos(w) * a[1]; b[2] = a[2];
        double c[3] = {b[0], std::cos(inc) * b[1] - std::sin(inc) * b[2], std::sin(inc) * b[1] + std::cos(inc) * b[2]};
        a[0] = std::cos(Om) * c[0] - std::sin(Om) * c[1]; a[1] = std::sin(Om) * c[0] + std::cos(Om) * c[1]; a[2] = c[2];
    };
    rot(x); rot(v);
    for (int k = 0; k < 3; k++) { z[k] = x[k]; z[3 + k] = v[k]; }
}
static void grads(const double* z, double gE[6], double gL[3][6]) {
    const double r = std::sqrt(z[0] * z[0] + z[1] * z[1] + z[2] * z[2]);
    for (int k = 0; k < 3; k++) { gE[k] = z[k] / (r * r * r); gE[3 + k] = z[3 + k]; }
    for (int k = 0; k < 3; k++) {
        const int a = (k + 1) % 3, b = (k + 2) % 3;
        for (int j = 0; j < 6; j++) gL[k][j] = 0;
        gL[k][a] = z[3 + b]; gL[k][3 + b] = z[a]; gL[k][b] = -z[3 + a]; gL[k][3 + a] = -z[b];
    }
}
static double M[DMAX][DMAX], Mfd[DMAX][DMAX], Mb[DMAX][DMAX];

int main(int argc, char** argv) {
    const bool fine = argc > 1 && argv[1][0] == 'f';
    std::printf("# e f0 dt/T | defect(enzyme) defect(fd) enz-vs-fd | dE-row dL-row rotEq | rev_val rev_jac | |M| iters branch\n");
    std::vector<double> es = {0.0, 1e-6, 1e-3, 0.05, 0.0999999, 0.1000001, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 0.99998, 0.999991, 0.9999999,
                              1.0, 1.0000001, 1.00005, 1.0001001, 1.001, 1.5, 3.0, 10.0};
    std::vector<double> fs = {0.0, -0.3, 2.0, 3.14159};
    std::vector<double> dts = {1e-3, 0.1, 0.37, 0.49999, 0.50001, 1.0, 10.3, 1000.3};
    if (fine) { es.clear(); for (int i = 0; i <= 60; i++) es.push_back(1.0 - std::pow(10.0, -i * 0.15)); }
    double worst = 0; double worst_at[3] = {};
    for (double e : es)
        for (double f0 : fs)
            for (double dtu : dts) {
                if (e >= 1.0 && std::fabs(f0) > 2.0) continue;   // beyond the asymptote for small e - 1
                double z[6], zo[6], zb[6];
                ics(e, f0, z);
                // time unit: period for bound orbits, sqrt(|a|^3/mu) otherwise (a from pericentre 1)
                const double a = (e == 1.0) ? 1.0 : 1.0 / std::fabs(1.0 - e);
                const double T = (e < 1.0) ? 2 * M_PI * std::sqrt(a * a * a) : std::sqrt(a * a * a);
                g_dt = dtu * std::min(T, 1e6);
                kmap(z, zo);
                const int it = g_it;
                jac_enzyme<kmap>(z, 6, M);
                jac_fd(kmap, z, 6, Mfd, 1e-5, 1.0);
                double ad;
                const double de = symp_defect(M, 6, &ad), dfd = symp_defect(Mfd, 6), efd = max_diff(Mfd, M, 6);
                double gE0[6], gL0[3][6], gE1[6], gL1[3][6];
                grads(z, gE0, gL0); grads(zo, gE1, gL1);
                double dE = 0, dL = 0, sE = 0;
                for (int j = 0; j < 6; j++) {
                    double s = 0; for (int i = 0; i < 6; i++) s += gE1[i] * M[i][j];
                    sE = std::max(sE, std::fabs(gE0[j]));
                    dE = std::max(dE, std::fabs(s - gE0[j]));
                    for (int k = 0; k < 3; k++) { double t = 0; for (int i = 0; i < 6; i++) t += gL1[k][i] * M[i][j]; dL = std::max(dL, std::fabs(t - gL0[k][j])); }
                }
                double mm = 0; for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) mm = std::max(mm, std::fabs(M[i][j]));
                // rotation equivariance M (w x z) = w x z'
                double req = 0;
                for (int k = 0; k < 3; k++) {
                    double w[3] = {0, 0, 0}; w[k] = 1;
                    double xi[6], xo[6];
                    for (int b = 0; b < 2; b++) {
                        const double* x = z + 3 * b; const double* y = zo + 3 * b;
                        xi[3 * b] = w[1] * x[2] - w[2] * x[1]; xi[3 * b + 1] = w[2] * x[0] - w[0] * x[2]; xi[3 * b + 2] = w[0] * x[1] - w[1] * x[0];
                        xo[3 * b] = w[1] * y[2] - w[2] * y[1]; xo[3 * b + 1] = w[2] * y[0] - w[0] * y[2]; xo[3 * b + 2] = w[0] * y[1] - w[1] * y[0];
                    }
                    for (int i = 0; i < 6; i++) { double s = 0; for (int j = 0; j < 6; j++) s += M[i][j] * xi[j]; req = std::max(req, std::fabs(s - xo[i])); }
                }
                const double dtsave = g_dt; g_dt = -dtsave; kmap(zo, zb); jac_enzyme<kmap>(zo, 6, Mb); g_dt = dtsave;
                double rv = 0; for (int j = 0; j < 6; j++) rv = std::max(rv, std::fabs(zb[j] - z[j]) / std::max(1e-300, std::fabs(z[j]) + 1e-3));
                const double rj = prod_minus_I(Mb, M, 6);
                // branch as in kepler_drift
                const char* br = (e < 0.1 || (e < 1.0 && 1.0 - e < 1e-5) || (e >= 1.0 && e - 1.0 < 1e-4)) ? "fg" : (e < 1 ? "ell" : "hyp");
                std::printf("%-10.8g %5.2f %8.5g | %8.1e %8.1e %8.1e | %8.1e %8.1e %8.1e | %8.1e %8.1e | %8.1e %3d %s\n", e, f0, dtu, de, dfd, efd,
                            dE / std::max(1.0, sE * mm), dL / std::max(1.0, mm), req / std::max(1.0, mm), rv, rj / std::max(1.0, mm * mm), mm, it, br);
                if (de > worst) { worst = de; worst_at[0] = e; worst_at[1] = f0; worst_at[2] = dtu; }
            }
    std::printf("# worst defect %.2e at e=%.10g f0=%.2f dt/T=%g\n", worst, worst_at[0], worst_at[1], worst_at[2]);
}
