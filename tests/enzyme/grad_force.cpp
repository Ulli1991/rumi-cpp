// FSI gradient force (FROST Eq. 19, GradK in n2_kernels.h) against the exact gradient of Phi = 1/2 sum_k m_k |a_k|^2
// (Enzyme). Force-gradient splitting (Chin 1997; Omelyan, Mryglod & Folk 2002): the modified force of the middle
// kick is F_i + c dt^2 grad_i (sum_k |F_k|^2 / m_k), so the kernel's correction divided by (1/m_i) dPhi/dx_i must be
// one constant for every particle and component; its value fixes the coefficient (GRAD_COEFF dt^2 times 2).
//   clang++ -std=c++20 -O2 -fplugin=ClangEnzyme-18.so -I<kokkos-clang>/include -Isrc/core -Isrc/gravity ...
#include "n2_kernels.h"
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

extern int enzyme_dup, enzyme_const;
template <typename... T> void __enzyme_autodiff(void*, T...);

static void newton(const double* x, const double* m, int n, double G, double* a) {
    for (int i = 0; i < 3 * n; i++) a[i] = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            const double d[3] = {x[3*j]-x[3*i], x[3*j+1]-x[3*i+1], x[3*j+2]-x[3*i+2]};
            const double r2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2], ir3 = 1.0 / (r2 * std::sqrt(r2));
            for (int k = 0; k < 3; k++) a[3*i+k] += G * m[j] * d[k] * ir3;
        }
}
// Phi = 1/2 sum_k m_k |a_k|^2
static double phi(const double* x, const double* m, int n, double G) {
    double s = 0.0;
    for (int k = 0; k < n; k++) {
        double ak[3] = {0.0, 0.0, 0.0};
        for (int j = 0; j < n; j++) {
            if (j == k) continue;
            const double d[3] = {x[3*j]-x[3*k], x[3*j+1]-x[3*k+1], x[3*j+2]-x[3*k+2]};
            const double r2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2], ir3 = 1.0 / (r2 * std::sqrt(r2));
            for (int q = 0; q < 3; q++) ak[q] += G * m[j] * d[q] * ir3;
        }
        s += 0.5 * m[k] * (ak[0]*ak[0] + ak[1]*ak[1] + ak[2]*ak[2]);
    }
    return s;
}

int main() {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    const double G = 4.302194e-3, dt = 1e-3;
    double worst = 0.0;
    for (int n : {2, 3, 8, 32}) {
        std::vector<double> x(3 * n), m(n), a0(3 * n), gp(3 * n, 0.0);
        for (auto& q : x) q = U(rng);
        for (auto& q : m) q = 0.1 + 10.0 * (U(rng) + 1.0);
        newton(x.data(), m.data(), n, G, a0.data());
        __enzyme_autodiff((void*)phi, enzyme_dup, x.data(), gp.data(), enzyme_const, m.data(), enzyme_const, n,
                          enzyme_const, G);
        // the real kernel: corrected a = a0 + sum over sources (soft2 0, no subsystems)
        std::vector<double> X(n), Y(n), Z(n), AX(n), AY(n), AZ(n), SX(n), SY(n), SZ(n), M(m);
        for (int i = 0; i < n; i++) { X[i] = x[3*i]; Y[i] = x[3*i+1]; Z[i] = x[3*i+2];
                                      SX[i] = a0[3*i]; SY[i] = a0[3*i+1]; SZ[i] = a0[3*i+2]; }
        GradK<false, false> k{};
        k.x = {X.data()}; k.y = {Y.data()}; k.z = {Z.data()}; k.m = {M.data()};
        k.ax = {AX.data()}; k.ay = {AY.data()}; k.az = {AZ.data()};
        k.isx = {SX.data()}; k.isy = {SY.data()}; k.isz = {SZ.data()}; k.jsx = {SX.data()}; k.jsy = {SY.data()}; k.jsz = {SZ.data()};
        k.ioff = k.joff = 0; k.gmag = {nullptr}; k.sid = nullptr;
        k.coeff = G * dt * dt * GRAD_COEFF; k.soft2 = 0.0; k.inv_dt2 = 1.0 / (dt * dt);
        double rmin = 1e300, rmax = -1e300;
        for (int i = 0; i < n; i++) {
            auto snk = k.sink(i);
            double sum[3] = {0.0, 0.0, 0.0}, s[8];
            for (int j = 0; j < n; j++) { if (j == i) continue; k.source(j, s); k.pair(snk, i, s, j, sum, nullptr); }
            for (int q = 0; q < 3; q++) {
                const double ref = gp[3*i+q] / m[i];
                if (std::fabs(ref) < 1e-12 * std::fabs(gp[0] / m[0])) continue;
                const double r = sum[q] / ref;
                rmin = std::min(rmin, r); rmax = std::max(rmax, r);
            }
        }
        std::printf("n=%3d: kernel correction / ((1/m_i) dPhi/dx_i) = %.15e .. %.15e   (in units of GRAD_COEFF dt^2: %.12f .. %.12f)\n",
                    n, rmin, rmax, rmin / (GRAD_COEFF * dt * dt), rmax / (GRAD_COEFF * dt * dt));
        worst = std::max({worst, std::fabs(rmin / (GRAD_COEFF * dt * dt) - 1.0), std::fabs(rmax / (GRAD_COEFF * dt * dt) - 1.0)});
    }
    const bool ok = worst < 1e-12;
    std::printf("%s: max |ratio / (GRAD_COEFF dt^2) - 1| = %.1e\n", ok ? "PASS" : "FAIL", worst);
    return !ok;
}
