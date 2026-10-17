// PN consistency check with Enzyme: under the PN1 equations of motion the energy E = T + U + E_pn1 must be conserved
// to O(1/c^4). dE/dt = grad_x E . v + grad_v E . a with the exact gradients of pn1_energy (Enzyme) and the code's
// acceleration a = a_Newton + sum over pairs of eih_pn_acceleration_pair (PN1 only), as LogH integrates it.
// Consistent energy and acceleration: dE/dt ~ c^-4 (doubling c divides it by 16); a mismatch: ~ c^-2 (by 4).
//   clang++ -std=c++20 -O2 -fplugin=ClangEnzyme-18.so -Isrc/collisional tests/enzyme/pn_consistency.cpp
#include "../../src/collisional/post_newtonian_core.cpp"
#include <cstdio>
#include <random>
#include <vector>

extern int enzyme_dup, enzyme_const;
template <typename... T> void __enzyme_autodiff(void*, T...);

static const double G = 4.302194e-3;   // pc (km/s)^2 / Msun

static int g_pn = 1;   // 0: Newtonian only (harness check)
static int g_full = 0;  // 1: the genuine N-body EIH acceleration (eih1_nbody_acceleration) instead of pair sums
static double g_pp_sign = 1.0;   // experiment: sign of the G^2 potential-potential term of pn1_energy (1: as coded)
// T + U + E_pn1 of n bodies
static double energy(const double* pos, const double* vel, const double* mass, int n, double c) {
    double E = 0.0;
    for (int A = 0; A < n; A++) {
        E += 0.5 * mass[A] * (vel[3*A]*vel[3*A] + vel[3*A+1]*vel[3*A+1] + vel[3*A+2]*vel[3*A+2]);
        for (int B = A + 1; B < n; B++) {
            const double dx = pos[3*A]-pos[3*B], dy = pos[3*A+1]-pos[3*B+1], dz = pos[3*A+2]-pos[3*B+2];
            E -= G * mass[A] * mass[B] / std::sqrt(dx*dx + dy*dy + dz*dz);
        }
    }
    if (!g_pn) return E;
    double pp = 0.0;   // the potential-potential part of pn1_energy, to flip it
    for (int A = 0; A < n; A++)
        for (int B = A + 1; B < n; B++) {
            const double dx = pos[3*A]-pos[3*B], dy = pos[3*A+1]-pos[3*B+1], dz = pos[3*A+2]-pos[3*B+2];
            pp -= G * G * mass[A] * mass[B] * (mass[A] + mass[B]) * 0.5 / (dx*dx + dy*dy + dz*dz);
        }
    return E + pn1_energy(pos, vel, mass, n, G, c) + (g_pp_sign - 1.0) * pp / (c * c);
}

// the code's acceleration: Newton + pairwise EIH (PN1)
static void accel(const double* pos, const double* vel, const double* mass, int n, double c, double* a) {
    for (int i = 0; i < 3 * n; i++) a[i] = 0.0;
    for (int A = 0; A < n; A++)
        for (int B = A + 1; B < n; B++) {
            const double dx = pos[3*B]-pos[3*A], dy = pos[3*B+1]-pos[3*A+1], dz = pos[3*B+2]-pos[3*A+2];
            const double r2 = dx*dx + dy*dy + dz*dz, r3 = r2 * std::sqrt(r2);
            const double d[3] = {dx, dy, dz};
            for (int k = 0; k < 3; k++) { a[3*A+k] += G * mass[B] * d[k] / r3; a[3*B+k] -= G * mass[A] * d[k] / r3; }
            if (!g_pn || g_full) continue;
            double ai[3], aj[3];
            eih_pn_acceleration_pair(&pos[3*A], &vel[3*A], mass[A], &pos[3*B], &vel[3*B], mass[B], G, c, ai, aj, 1);
            for (int k = 0; k < 3; k++) { a[3*A+k] += ai[k]; a[3*B+k] += aj[k]; }
        }
    if (g_pn && g_full) {
        std::vector<double> ap(3 * n);
        eih1_nbody_acceleration(pos, vel, mass, n, G, c, ap.data());
        for (int i = 0; i < 3 * n; i++) a[i] += ap[i];
    }
}

// EIH Lagrangian (Landau & Lifshitz sec. 106)
static double lagrangian(const double* x, const double* v, const double* m, int n, double c) {
    double L = 0.0, ic2 = 1.0 / (c * c);
    for (int A = 0; A < n; A++) {
        const double vA2 = v[3*A]*v[3*A] + v[3*A+1]*v[3*A+1] + v[3*A+2]*v[3*A+2];
        L += 0.5 * m[A] * vA2 + ic2 * 0.125 * m[A] * vA2 * vA2;
        for (int B = A + 1; B < n; B++) {
            const double d[3] = {x[3*A]-x[3*B], x[3*A+1]-x[3*B+1], x[3*A+2]-x[3*B+2]};
            const double r = std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
            const double vB2 = v[3*B]*v[3*B] + v[3*B+1]*v[3*B+1] + v[3*B+2]*v[3*B+2];
            const double vAvB = v[3*A]*v[3*B] + v[3*A+1]*v[3*B+1] + v[3*A+2]*v[3*B+2];
            const double nvA = (d[0]*v[3*A] + d[1]*v[3*A+1] + d[2]*v[3*A+2]) / r;
            const double nvB = (d[0]*v[3*B] + d[1]*v[3*B+1] + d[2]*v[3*B+2]) / r;
            L += G * m[A] * m[B] / r + ic2 * G * m[A] * m[B] / (2.0 * r) * (3.0 * (vA2 + vB2) - 7.0 * vAvB - nvA * nvB);
        }
        for (int B = 0; B < n; B++)
            for (int C = 0; C < n; C++) {
                if (B == A || C == A) continue;
                const double rAB = std::sqrt((x[3*A]-x[3*B])*(x[3*A]-x[3*B]) + (x[3*A+1]-x[3*B+1])*(x[3*A+1]-x[3*B+1]) + (x[3*A+2]-x[3*B+2])*(x[3*A+2]-x[3*B+2]));
                const double rAC = std::sqrt((x[3*A]-x[3*C])*(x[3*A]-x[3*C]) + (x[3*A+1]-x[3*C+1])*(x[3*A+1]-x[3*C+1]) + (x[3*A+2]-x[3*C+2])*(x[3*A+2]-x[3*C+2]));
                L -= ic2 * 0.5 * G * G * m[A] * m[B] * m[C] / (rAB * rAC);
            }
    }
    return L;
}
// dL/dx, dL/dv (Enzyme)
static void lag_grad(const double* x, const double* v, const double* m, int n, double c, double* gx, double* gv) {
    for (int i = 0; i < 3 * n; i++) gx[i] = gv[i] = 0.0;
    __enzyme_autodiff((void*)lagrangian, enzyme_dup, x, gx, enzyme_dup, v, gv, enzyme_const, m, enzyme_const, n,
                      enzyme_const, c);
}
// Euler-Lagrange acceleration: J_v a = dL/dx - J_x v with J_v = d(dL/dv)/dv, J_x = d(dL/dv)/dx (central differences
// of the exact Enzyme gradient), solved by Gaussian elimination
static void el_accel(const double* x, const double* v, const double* m, int n, double c, double* a) {
    const int N = 3 * n;
    std::vector<double> gx(N), gv(N), Jv(N * N), rhs(N), gxp(N), gvp(N), gxm(N), gvm(N), xx(x, x + N), vv(v, v + N);
    lag_grad(x, v, m, n, c, gx.data(), gv.data());
    for (int i = 0; i < N; i++) rhs[i] = gx[i];
    for (int j = 0; j < N; j++) {
        const double hv = 1e-6 * (std::fabs(v[j]) + 1.0), hx = 1e-6 * (std::fabs(x[j]) + 1e-7);
        vv[j] = v[j] + hv; lag_grad(x, vv.data(), m, n, c, gxp.data(), gvp.data());
        vv[j] = v[j] - hv; lag_grad(x, vv.data(), m, n, c, gxm.data(), gvm.data()); vv[j] = v[j];
        for (int i = 0; i < N; i++) Jv[i * N + j] = (gvp[i] - gvm[i]) / (2 * hv);
        xx[j] = x[j] + hx; lag_grad(xx.data(), v, m, n, c, gxp.data(), gvp.data());
        xx[j] = x[j] - hx; lag_grad(xx.data(), v, m, n, c, gxm.data(), gvm.data()); xx[j] = x[j];
        for (int i = 0; i < N; i++) rhs[i] -= (gvp[i] - gvm[i]) / (2 * hx) * v[j];
    }
    for (int k = 0; k < N; k++)   // Gaussian elimination with partial pivoting
        for (int i = k + 1; i < N; i++) {
            int piv = k; for (int r = k; r < N; r++) if (std::fabs(Jv[r*N+k]) > std::fabs(Jv[piv*N+k])) piv = r;
            if (piv != k) { for (int q = 0; q < N; q++) std::swap(Jv[k*N+q], Jv[piv*N+q]); std::swap(rhs[k], rhs[piv]); }
            const double f = Jv[i*N+k] / Jv[k*N+k];
            for (int q = k; q < N; q++) Jv[i*N+q] -= f * Jv[k*N+q];
            rhs[i] -= f * rhs[k];
        }
    for (int i = N - 1; i >= 0; i--) {
        double s2 = rhs[i];
        for (int q = i + 1; q < N; q++) s2 -= Jv[i*N+q] * a[q];
        a[i] = s2 / Jv[i*N+i];
    }
}

// dE/dt and the scale |dE_pn/dt| ~ |E_pn| v / r for normalisation
static double g_scale = 0.0;   // sum |m v . a_PN|: the size of the individual 1PN terms of dE/dt
static double dEdt(const double* pos, const double* vel, const double* mass, int n, double c) {
    std::vector<double> gx(3 * n, 0.0), gv(3 * n, 0.0), a(3 * n);
    __enzyme_autodiff((void*)energy, enzyme_dup, pos, gx.data(), enzyme_dup, vel, gv.data(), enzyme_const, mass,
                      enzyme_const, n, enzyme_const, c);
    accel(pos, vel, mass, n, c, a.data());
    double s = 0.0;
    for (int i = 0; i < 3 * n; i++) s += gx[i] * vel[i] + gv[i] * a[i];
    if (g_pn) {   // scale: m v . a_PN
        std::vector<double> aN(3 * n); g_pn = 0; accel(pos, vel, mass, n, c, aN.data()); g_pn = 1;
        g_scale = 0.0;
        for (int A = 0; A < n; A++)
            for (int k = 0; k < 3; k++) g_scale += std::fabs(mass[A] * vel[3*A+k] * (a[3*A+k] - aN[3*A+k]));
    }
    return s;
}

int main(int argc, char** argv) {
    if (argc > 1) g_pp_sign = std::atof(argv[1]);
    if (argc > 2) g_full = std::atoi(argv[2]);
    if (g_full) {   // acceleration of the code vs the Euler-Lagrange acceleration of the EIH Lagrangian
        std::mt19937_64 r2(777);
        std::uniform_real_distribution<double> U2(-1.0, 1.0);
        for (int n : {2, 3, 4}) {
            double worst = 0.0;
            for (int t = 0; t < 10; t++) {
                std::vector<double> m(n), x(3 * n), v(3 * n), ae(3 * n), ac(3 * n), aN(3 * n);
                double M = 0; for (auto& q : m) { q = 10.0 + 20.0 * (U2(r2) + 1.0); M += q; }
                for (auto& q : x) q = 1e-6 * U2(r2);
                const double vs = std::sqrt(G * M / 1e-6);
                for (auto& q : v) q = 0.5 * vs * U2(r2);
                const double c = 100.0 * vs;
                el_accel(x.data(), v.data(), m.data(), n, c, ae.data());
                accel(x.data(), v.data(), m.data(), n, c, ac.data());
                g_pn = 0; accel(x.data(), v.data(), m.data(), n, c, aN.data()); g_pn = 1;
                double num = 0, den = 0;   // |PN1 part of the code - PN1 part of Euler-Lagrange| / |PN1 part|
                for (int i = 0; i < 3 * n; i++) { num += (ac[i] - ae[i]) * (ac[i] - ae[i]); den += (ac[i] - aN[i]) * (ac[i] - aN[i]); }
                worst = std::max(worst, std::sqrt(num / den));
            }
            std::printf("n=%d: |a_code - a_EulerLagrange| / |a_PN1| max over 10 configurations: %.3e (O(v^2/c^2) = 1e-4 if right)\n",
                        n, worst);
        }
    }
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    const double c0 = 299792.458;
    for (int n : {2, 3, 4}) {
        double worst_ratio = 1e30, best_ratio = 0;
        for (int trial = 0; trial < 20; trial++) {
            std::vector<double> m(n), x(3 * n), v(3 * n);
            for (int A = 0; A < n; A++) {
                m[A] = 10.0 + 20.0 * (U(rng) + 1.0);            // 10-50 Msun
                for (int k = 0; k < 3; k++) x[3*A+k] = 1e-6 * U(rng);   // ~0.2 AU scale
            }
            // velocities of the virial scale sqrt(G M / r)
            double M = 0; for (double mm : m) M += mm;
            const double vs = std::sqrt(G * M / 1e-6);
            for (int i = 0; i < 3 * n; i++) v[i] = 0.5 * vs * U(rng);
            // c scaled so that v/c ~ 1e-2 and 5e-3 (PN expansion valid, residual dominated by its leading power)
            const double c1 = 100.0 * vs, c2 = 2.0 * c1;
            const double r1 = dEdt(x.data(), v.data(), m.data(), n, c1);
            const double r2 = dEdt(x.data(), v.data(), m.data(), n, c2);
            const double ratio = std::fabs(r1 / r2);
            worst_ratio = std::min(worst_ratio, ratio); best_ratio = std::max(best_ratio, ratio);
            const double sc1 = g_scale;
            g_pn = 0; const double rn = dEdt(x.data(), v.data(), m.data(), n, c1); g_pn = 1;
            if (trial < 3) std::printf("n=%d trial %d: dE/dt(c) %.3e dE/dt(2c) %.3e ratio %.2f | residual / 1PN terms %.3e | Newtonian only %.3e\n",
                                       n, trial, r1, r2, ratio, std::fabs(r1) / sc1, rn);
        }
        std::printf("n=%d: ratio dE/dt(c)/dE/dt(2c) over 20 configurations: %.2f - %.2f (16: consistent, c^-4; 4: mismatch, c^-2)\n",
                    n, worst_ratio, best_ratio);
    }
    (void)c0;
}
