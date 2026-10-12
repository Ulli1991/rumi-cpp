// GW recoil formula (src/collisional/gw_recoil.h) against limits from the literature:
//  - equal masses, no spins: 0
//  - q = 0.5, no spins: v_m = 1.2e4 eta^2 (1 - q)/(1 + q) (1 - 0.93 eta) = 156.7 km/s (Gonzalez et al. 2007: maximum
//    175 km/s at q ~ 0.36)
//  - equal masses, equal aligned spins: 0 (symmetry)
//  - equal masses, chi = 0.9 in the plane, antiparallel ("superkick", Campanelli et al. 2007; Lousto et al. 2012):
//    maximum over phi = 16/32 V11 x 1.8 = 3310 km/s
//  - mass symmetry: swapping the bodies leaves |v| unchanged
#include "gw_recoil.h"
#include <cmath>
#include <cstdio>

static double norm(const double* v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

int main() {
    int fail = 0;
    auto check = [&](const char* label, double val, double lo, double hi) {
        const bool ok = val >= lo && val <= hi;
        std::printf("%s %s = %.6g (in [%.6g, %.6g])\n", ok ? "ok  " : "FAIL", label, val, lo, hi);
        if (!ok) fail++;
    };
    const double L[3] = {0, 0, 1}, z0[3] = {0, 0, 0};
    double v[3];
    gw_recoil_kick(10, 10, z0, z0, L, 0.3, 0.7, v);
    check("equal masses, no spins |v| [km/s]", norm(v), 0.0, 1e-9);
    gw_recoil_kick(20, 10, z0, z0, L, 0.3, 0.7, v);
    check("q = 0.5, no spins |v| [km/s]", norm(v), 156.6, 156.8);
    check("q = 0.5, no spins: in the orbital plane |v_z|", std::fabs(v[2]), 0.0, 1e-9);
    const double al[3] = {0, 0, 0.7};
    gw_recoil_kick(10, 10, al, al, L, 0.3, 0.7, v);
    check("equal masses, equal aligned spins |v|", norm(v), 0.0, 1e-9);
    const double s1[3] = {0.9, 0, 0}, s2[3] = {-0.9, 0, 0};
    double vmax = 0.0;
    for (int i = 0; i < 1000; i++) { gw_recoil_kick(10, 10, s1, s2, L, i / 1000.0, 0.2, v); vmax = std::fmax(vmax, norm(v)); }
    check("superkick chi = 0.9 antiparallel in plane, max |v| over phi", vmax, 3300.0, 3320.0);
    double va[3], vb[3];
    const double c1[3] = {0.3, -0.2, 0.5}, c2[3] = {-0.6, 0.1, 0.2};
    gw_recoil_kick(30, 12, c1, c2, L, 0.41, 0.13, va);
    gw_recoil_kick(12, 30, c2, c1, L, 0.41, 0.13, vb);
    check("body swap |v_a| - |v_b|", std::fabs(norm(va) - norm(vb)), 0.0, 1e-9);
    return fail ? 1 : 0;
}
