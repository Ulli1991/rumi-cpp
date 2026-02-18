#pragma once

#include "particles.h"
#include "subsystem.h"
#include "logh.h"
#include "gbs.h"

// MSTAR: Minimum Spanning Tree Algorithmic Regularization
// For large (8-1000 body) regularized regions around SMBHs

// Maximum bodies for MSTAR (GPU memory limited)
constexpr int MSTAR_MAX_BODIES = 1024;

struct MSTARState {
    int n_bodies;
    double t;                              // physical time
    double B;                              // binding energy

    // Absolute coordinates
    double pos[MSTAR_MAX_BODIES * 3];
    double vel[MSTAR_MAX_BODIES * 3];
    double mass[MSTAR_MAX_BODIES];
    double ext_accel[MSTAR_MAX_BODIES * 3];

    // MST chain coordinates (relative positions along MST edges)
    int chain_order[MSTAR_MAX_BODIES];     // permutation for chain
    int chain_pairs[MSTAR_MAX_BODIES * 2]; // MST edges (i, j pairs)
    int n_edges;                            // = n_bodies - 1
    double chain_pos[MSTAR_MAX_BODIES * 3];// chain-relative positions
    double chain_vel[MSTAR_MAX_BODIES * 3];// chain-relative velocities
};

// Build minimum spanning tree from distance matrix (Prim's algorithm)
void build_mst(const double pos[], const double mass[], int n,
               int chain_order[], int chain_pairs[], int& n_edges);

// Transform absolute -> chain coordinates
void absolute_to_chain(const MSTARState& state);

// Transform chain -> absolute coordinates
void chain_to_absolute(MSTARState& state);

// MSTAR leapfrog step in regularized time
void mstar_leapfrog_step(MSTARState& state, double ds, double G);

// Dispatch: use LogH for n <= threshold, MSTAR for n > threshold
void regularized_integrate(Subsystem& sub, Particles& p,
                           double G, double dt, double eta_gbs,
                           int mstar_threshold);
