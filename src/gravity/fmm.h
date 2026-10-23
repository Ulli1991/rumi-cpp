#pragma once

// Fast multipole method for the far field (DEV_NOTES.md item 7; prototype and derivations in tools/fmm).
// Solid-harmonic expansions to order p about cell centres of mass, mutual-orientation M2L by rotation to the
// z axis (O(p^3), real form), exact Newtonian far field. Near field: direct sums with the selected softening
// kernel, which must be exactly Newtonian beyond its support (none or spline), and subsystem exclusion.
// Momentum is conserved to round-off: both directions of a cell pair use the same rotation.

#include <Kokkos_Core.hpp>

struct FmmParams {
    int p = 12;            // expansion order
    double theta = 0.5;    // opening angle: cells interact through expansions when r_A + r_B < theta |z_A - z_B|
    int n_leaf = 64;       // particles per leaf
    int kernel = 0;        // near-field softening: 0 none, 2 cubic spline (h = 2.8 eps)
    double eps = 0.0;
    double G = 1.0;
    double min_far_sep = 0.0;   // far pairs need |z_A - z_B| - r_A - r_B > this: closer particle pairs (subsystem
                                // partners, within r_ngb) always go through the near field, where exclusion applies
};

struct FmmStats {
    int n_cells = 0, n_leaves = 0;
    long long n_m2l = 0, n_p2p_pairs = 0;
    double t_tree = 0, t_up = 0, t_lists = 0, t_m2l = 0, t_p2p = 0, t_down = 0;
};

// Accelerations of particles [i0, i0+n) from each other (overwrites ax, ay, az there). sid: subsystem ids
// (members of one subsystem do not interact) or empty.
void fmm_accelerations(const Kokkos::View<double*>& x, const Kokkos::View<double*>& y, const Kokkos::View<double*>& z,
                       const Kokkos::View<double*>& m, const Kokkos::View<int*>& sid, int i0, int n,
                       const FmmParams& prm, Kokkos::View<double*> ax, Kokkos::View<double*> ay,
                       Kokkos::View<double*> az, FmmStats* stats = nullptr);

void fmm_release();   // before Kokkos::finalize
