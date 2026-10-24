// One fixed-signature LogH + GBS step (logh_gbs_step, k_fixed stages, always accepted) differentiated with Enzyme:
// d(energy error)/d(initial positions) of an eccentric binary against central differences of the same function.
// Newtonian, and PN1-2.5 with the auxiliary velocities. Also forward vs reverse mode, and d(dE)/d(dt) (local order).
//   clang++ -std=c++20 -O2 -fplugin=ClangEnzyme-18.so -Isrc/collisional tests/enzyme/logh_step.cpp
#include "../../src/collisional/logh.cpp"
#include "../../src/collisional/logh_step.cpp"
#include "../../src/collisional/post_newtonian_core.cpp"
#include <cstdio>

extern int enzyme_dup, enzyme_const;
template <typename... T> double __enzyme_autodiff(void*, T...);
template <typename... T> double __enzyme_fwddiff(void*, T...);

static const int SEQ[] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
static int g_pn = 0;

// (T + U)(end) - (T + U)(start) after one GBS step of physical time dt (3 stages: 2, 4, 6 leapfrog steps)
static double energy_error(const double* pos0, const double* vel0, const double* mass, double dt) {
    GbsParams par;
    par.seq = SEQ;
    par.k_fixed = 3;
    par.G = 1.0;
    par.c_code = g_pn ? 30.0 : 0.0;
    par.pn_order = g_pn ? 7 : 0;
    double pos[6], vel[6];
    for (int i = 0; i < 6; i++) { pos[i] = pos0[i]; vel[i] = vel0[i]; }
    LoghStepResult res;
    logh_gbs_step(par, pos, vel, mass, 2, dt, 1, nullptr, &res);
    return res.dE;
}

int main() {
    // a = 1, e = 0.9, G M = 1, from 0.3 rad before pericentre over dt = 0.1 (through the pericentre passage)
    const double mass[2] = {0.6, 0.4}, e = 0.9, f0 = -0.3;
    const double p = 1.0 - e * e, r = p / (1.0 + e * std::cos(f0));
    const double rel_x[3] = {r * std::cos(f0), r * std::sin(f0), 0.0};
    const double rel_v[3] = {-std::sin(f0) / std::sqrt(p), (e + std::cos(f0)) / std::sqrt(p), 0.0};
    double pos[6], vel[6];
    for (int k = 0; k < 3; k++) {
        pos[k] = -mass[1] * rel_x[k];  pos[3 + k] = mass[0] * rel_x[k];
        vel[k] = -mass[1] * rel_v[k];  vel[3 + k] = mass[0] * rel_v[k];
    }
    pos[2] += 0.01; vel[5] += 0.02;   // out of the plane: every component active
    const double dt = 0.1;
    int fails = 0;
    for (g_pn = 0; g_pn <= 1; g_pn++) {
        double grad[6] = {};
        __enzyme_autodiff((void*)energy_error, enzyme_dup, pos, grad, enzyme_const, vel, enzyme_const, mass,
                          enzyme_const, dt);
        double worst = 0.0;
        std::printf("%s: dE = %.6e\n", g_pn ? "PN1-2.5 (c = 30)" : "Newtonian", energy_error(pos, vel, mass, dt));
        for (int i = 0; i < 6; i++) {
            const double h = 1e-4 * std::fabs(pos[i]) + 1e-5;
            double xp[6], xm[6];
            for (int j = 0; j < 6; j++) xp[j] = xm[j] = pos[j];
            xp[i] += h; xm[i] -= h;
            const double fd = (energy_error(xp, vel, mass, dt) - energy_error(xm, vel, mass, dt)) / (2.0 * h);
            const double rel = std::fabs(grad[i] - fd) / std::max(std::fabs(fd), 1e-300);
            worst = std::max(worst, rel);
            std::printf("  d/dx[%d]  enzyme % .12e  central diff % .12e  rel %.1e\n", i, grad[i], fd, rel);
        }
        std::printf("  worst relative difference %.1e\n", worst);
        if (!(worst < 1e-5)) fails++;
        // forward mode along a direction vs the reverse gradient
        const double dir[6] = {0.3, -0.7, 0.2, 0.5, 0.1, -0.4};
        const double fwd = __enzyme_fwddiff((void*)energy_error, enzyme_dup, pos, dir, enzyme_const, vel, enzyme_const, mass,
                                            enzyme_const, dt);
        double rev = 0.0, sc = 0.0;
        for (int i = 0; i < 6; i++) { rev += grad[i] * dir[i]; sc += std::fabs(grad[i] * dir[i]); }
        std::printf("  forward %.12e  reverse %.12e  rel %.1e\n", fwd, rev, std::fabs(fwd - rev) / sc);
        if (!(std::fabs(fwd - rev) <= 1e-12 * sc)) fails++;
        // local order (Newtonian; with PN dE holds the radiated energy): p = dt d(dE)/d(dt) / dE, forward mode in dt
        for (double h : {0.05, 0.025, 0.0125}) {
            if (g_pn) break;
            const double dE = energy_error(pos, vel, mass, h);
            const double dd = __enzyme_fwddiff((void*)energy_error, enzyme_const, pos, enzyme_const, vel, enzyme_const, mass, h, 1.0);
            std::printf("  dt = %.4f: dE = % .3e  p = dt d(dE)/d(dt) / dE = %.3f\n", h, dE, h * dd / dE);
        }
    }
    std::printf(fails ? "FAIL\n" : "PASS\n");
    return fails;
}
