// PnGeom (post_newtonian_core.h): eih1_nbody_acceleration_geom and eih_pn_acceleration_pair_geom must equal
// eih1_nbody_acceleration and eih_pn_acceleration_pair bit for bit (LogH evaluates the PN accelerations through the
// cached geometry, 5 times per leapfrog step). Random subsystems of 2-8 bodies over many scales, every pn_order
// 0-63, the N-body EIH PN1 on and off, several velocity sets per position set. Prints the number of mismatches.
#include "post_newtonian_core.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

int main(int argc, char** argv) {
    const long n_conf = argc > 1 ? std::atol(argv[1]) : 20000;
    std::mt19937_64 rng(20261017);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::uniform_real_distribution<double> L(0.0, 1.0);
    long mismatches = 0, compared = 0, bad_eih1 = 0, bad_pair[64] = {};
    for (long c = 0; c < n_conf; c++) {
        const int n = 2 + int(c % 7);
        // scales: lengths 1e-9 .. 1e-2, velocities 1 .. 1e4, masses 1e-6 .. 1e3 (log-uniform), G, c as in the code
        const double lx = std::pow(10.0, -9.0 + 7.0 * L(rng)), lv = std::pow(10.0, 4.0 * L(rng));
        const double G = c % 2 ? 4.30091e-3 : 1.0, c_code = c % 3 ? 299792.458 : 30.0 + 1e3 * L(rng);
        double pos[3 * PN_GEOM_MAX_BODIES], vel[3 * PN_GEOM_MAX_BODIES], mass[PN_GEOM_MAX_BODIES];
        for (int i = 0; i < n; i++) {
            mass[i] = std::pow(10.0, -6.0 + 9.0 * L(rng));
            for (int k = 0; k < 3; k++) pos[3*i+k] = lx * U(rng);
        }
        const int pn_order = int(c % 64);
        const bool eih_nbody = n >= 3 && (pn_order & 1) && (c / 64) % 2 == 0;
        const int pair_order = eih_nbody ? (pn_order & ~1) : pn_order;
        PnGeom geom;
        pn_geom_set_masses(mass, n, G, c_code, eih_nbody, pair_order, geom);
        pn_geom_set_positions(pos, geom);
        for (int s = 0; s < 3; s++) {
            for (int i = 0; i < 3 * n; i++) vel[i] = lv * U(rng);
            double ref[3 * PN_GEOM_MAX_BODIES], got[3 * PN_GEOM_MAX_BODIES];
            if (eih_nbody) {
                eih1_nbody_acceleration(pos, vel, mass, n, G, c_code, ref);
                eih1_nbody_acceleration_geom(geom, vel, got);
                compared++;
                if (std::memcmp(ref, got, 3 * n * sizeof(double))) {
                    if (mismatches < 10) std::printf("eih1 mismatch: conf %ld n %d: %a vs %a\n", c, n, ref[0], got[0]);
                    mismatches++;
                    bad_eih1++;
                }
            }
            if (pair_order == 0) continue;
            double a_pairs[PN_GEOM_MAX_DIR * 3];
            eih_pn_pairs_geom(geom, vel, pair_order, a_pairs);
            int q = 0;
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++, q++) {
                    double ri[3], rj[3], gi[3], gj[3];
                    eih_pn_acceleration_pair(&pos[3*i], &vel[3*i], mass[i], &pos[3*j], &vel[3*j], mass[j], G, c_code,
                                             ri, rj, pair_order);
                    if (geom.pair[q].i != i || geom.pair[q].j != j) { std::printf("pair order\n"); return 1; }
                    for (int k = 0; k < 3; k++) { gi[k] = a_pairs[6*q + k]; gj[k] = a_pairs[6*q + 3 + k]; }
                    compared++;
                    if (std::memcmp(ri, gi, sizeof ri) || std::memcmp(rj, gj, sizeof rj)) {
                        if (mismatches < 10)
                            std::printf("pair mismatch: conf %ld n %d order %d pair %d-%d: %a %a vs %a %a\n", c, n,
                                        pair_order, i, j, ri[0], rj[0], gi[0], gj[0]);
                        mismatches++;
                        bad_pair[pair_order]++;
                    }
                }
        }
    }
    std::printf("%ld evaluations compared, %ld mismatches\n", compared, mismatches);
    if (mismatches) {
        std::printf("  N-body PN1: %ld\n", bad_eih1);
        for (int o = 0; o < 64; o++)
            if (bad_pair[o]) std::printf("  pair order %d: %ld\n", o, bad_pair[o]);
    }
    std::printf(mismatches ? "FAIL\n" : "PASS\n");
    return mismatches ? 1 : 0;
}
