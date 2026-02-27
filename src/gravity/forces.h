#pragma once

#include "particles.h"

// Full N-body
void compute_acceleration_and_timestep(Particles& p, double G, double soft2, double eta);

// Self-interaction within [start, start+count)
void self_acceleration(Particles& p, int start, int count,
                       double G, double soft2);

void self_gradient_correction(Particles& p, int start, int count,
                              double G, double soft2, double dt);

// Cross-interaction between two disjoint groups
void cross_acceleration(Particles& p,
                        int a_start, int a_count,
                        int b_start, int b_count,
                        double G, double soft2);

void cross_gradient_correction(Particles& p,
                               int a_start, int a_count,
                               int b_start, int b_count,
                               double G, double soft2, double dt);

// Compute adaptive timesteps for a subset of particles [start, start+count)
// Only considers mutual interactions within the subset
void compute_timestep_subset(Particles& p, int start, int count,
                          double G, double soft2, double eta,
                          bool symmetrize = false);

// Fused self-force + symmetrized timestep (no kick, stores forces for reuse)
void self_acceleration_and_timestep(Particles& p, int start, int count,
                                     double G, double soft2, double eta);

// Standalone neighbor search (BIFROST Section 3.5)
// Lightweight N² tiled kernel: detects neighbor pairs without computing forces
// or timesteps. No subsystem exclusion — all pairs are checked.
// Writes detected pairs to g_subsystem_mgr neighbor buffers as array indices.
void neighbor_search_subset(Particles& p, int start, int count,
                            double G, double r_ngb, double r_ngb_max,
                            double pivot_dt);

