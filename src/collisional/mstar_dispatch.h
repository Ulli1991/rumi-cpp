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
    // out
    bool ok = false;                   // integrated to dt
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
