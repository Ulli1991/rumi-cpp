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
            if (!g_pn) continue;
            double ai[3], aj[3];
            eih_pn_acceleration_pair(&pos[3*A], &vel[3*A], mass[A], &pos[3*B], &vel[3*B], mass[B], G, c, ai, aj, 1);
            for (int k = 0; k < 3; k++) { a[3*A+k] += ai[k]; a[3*B+k] += aj[k]; }
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
