// SPDX-License-Identifier: BSD-3-Clause
// RUMI++ core: thin interface to the optional MSTAR module.
//
// MSTAR (minimum-spanning-tree algorithmic regularisation; Rantala et al. 2020, MNRAS 492, 4131; Mannerkoski et al.
// 2023, MNRAS 524, 4062) integrates subsystems larger than LogH's LOGH_MAX_BODIES. Its implementation (the KETJU
// integrator library and its Kokkos port) lives in the separate, GPLv3-licensed module src/mstar, compiled only with
// the CMake option ENABLE_MSTAR=ON (binaries built so are GPLv3). The core never includes module code: it calls the
// functions below, which the module implements (src/mstar/mstar_dispatch.cpp) or, without the module, the BSD stub
// mstar_dispatch_stub.cpp does (mstar_available() = false; every call fails, the caller keeps its LogH/FSI path).

#pragma once

#include <vector>
#include <cmath>
#include <Kokkos_Macros.hpp>

// Collision distance of a pair inside an MSTAR group, evaluated on the device every accepted step. It is the rule of
// collision_distance (mergers.cpp; FROST-CLUSTERS I App. B3), restated here so that the module can evaluate it
// without core code (ctest mstar_coll_rule checks the two agree): BH-BH 20 G (m1 + m2) / c^2 (10 R_S of the total
// mass) or, bh_max_single, 6 G max(m1, m2) / c^2; BH-star max(1.3 (M_bh / m)^(1/3) R, R) or the ISCO 6 G (m1 + m2) /
// c^2 if larger; star-star R1 + R2. type: 1 stellar collision, 2 tidal disruption, 3 ISCO / BH merger.
KOKKOS_INLINE_FUNCTION double mstar_collision_distance(bool bh1, double m1, double R1, bool bh2, double m2, double R2,
                                                       double G, double c, bool bh_max_single, int& type) {
    const double r_isco = c > 0.0 ? 6.0 * G * (m1 + m2) / (c * c) : 0.0;
    if (bh1 && bh2) {
        type = 3;
        if (!(c > 0.0)) return 0.0;
        return bh_max_single ? 6.0 * G * (m1 > m2 ? m1 : m2) / (c * c) : 20.0 * G * (m1 + m2) / (c * c);
    }
    if (bh1 || bh2) {
        const double m_s = bh1 ? m2 : m1, R_s = bh1 ? R2 : R1, M_bh = bh1 ? m1 : m2;
        double r_t = 0.0;
        if (R_s > 0.0 && m_s > 0.0) {
            r_t = 1.3 * cbrt(M_bh / m_s) * R_s;
            if (r_t < R_s) r_t = R_s;
        }
        if (r_isco >= r_t) { type = 3; return r_isco; }
        type = 2;
        return r_t;
    }
    type = 1;
    return R1 + R2;
}

struct MstarMerger {
    double t;                          // time of the merger within the call (from its start)
    double m1, m2, m_remnant;
    double chi1, chi2, chi_remnant;    // dimensionless spin magnitudes
    double v_kick;                     // recoil speed (units of the call)
    int index1, index2;                // caller indices (MstarGroup::index) of the components; index1 keeps the remnant
};

// One group integration in the caller's units (normalised: G and c as given). Arrays are in/out; mergers shrink
// n / n_pn and move the merged-away bodies to the end of index[] (their slots in mass/pos/vel are then unused).
struct MstarGroup {
    // in
    int n = 0, n_pn = 0;               // bodies; the first n_pn are PN bodies (BHs)
    double* mass = nullptr;            // [n]
    double* pos = nullptr;             // [3n]
    double* vel = nullptr;             // [3n]
    double* spin = nullptr;            // [3 n_pn] spin angular momentum vectors (length x mass x velocity)
    int* index = nullptr;              // [n] caller indices (permuted by mergers)
    double dt = 0;                     // time to advance
    double G = 1, c = 0;               // c <= 0: no PN
    unsigned pn_flags = 0;             // KETJU bits: 1 1PN, 2 three-body 1PN, 4 2PN, 8 2.5PN, 16 3PN, 32 3.5PN,
                                       // 256 1.5PN spin-orbit, 512 2PN spin-spin, 1024 spin evolution
    double tol = 1e-10;                // GBS relative tolerance per step
    int kmax = 10;                     // GBS maximum order (4..16)
    bool bh_mergers = false;           // merge PN bodies within bh_merge_rs x 2G(m1+m2)/c^2 (KETJU rule)
    double bh_merge_rs = 12;
    bool bh_kicks = true;              // merger recoil (Zlochower & Lousto 2015 fits)
    long max_steps = -1;               // > 0: give up after this many attempted steps (ok = false)
    int device = -1;                   // -1: automatic (device above the crossover size), 0: host C library, 1: device
    // Collisions (device path only): after every accepted step, a pair closer than mstar_collision_distance, or
    // approaching with osculating pericentre inside it and reaching that pericentre within one step, stops the call
    // (status 1) at that step; the caller merges the pair and calls again for the rest of dt
    const double* coll_R = nullptr;    // [n] radius per body (units of pos; nullptr: no collision check)
    const char* coll_bh = nullptr;     // [n] 1: black hole
    double coll_c = 0;                 // speed of light for the ISCO / BH radii (units of the call; 0: none)
    bool coll_bh_max_single = false;
    // out
    bool ok = false;                   // integrated to dt
    int status = 0;                    // 0 reached dt, 1 collision (coll_*), 2 step limit
    double t_reached = 0;              // time advanced
    int coll_i = -1, coll_j = -1;      // colliding pair (positions in the arrays), coll_type as above
    int coll_type = 0;
    double coll_rsep = 0, coll_rcoll = 0;   // separation at the decision (osculating pericentre if so decided)
    double pn_dE = 0;                  // change of the Newtonian T + U by the PN forces (extrapolated PN work, PN
                                       // groups); the rest of the T + U change is integration error
    long steps = 0, failed_steps = 0;
    double rel_energy_error = 0;       // (B_end - B_start) / B_start, B = U - T (Newtonian; meaningful without PN)
    double wall = 0;                   // seconds
    int used_device = 0;               // 1: the Kokkos port ran it
    std::vector<MstarMerger> mergers;
};

bool mstar_available();
// Group size from which the device path is used when device = -1 (NBODY_MSTAR_CROSSOVER overrides)
int mstar_crossover_n();
// Integrate g (not thread safe for the device path: call from one host thread at a time)
bool mstar_integrate(MstarGroup& g);
