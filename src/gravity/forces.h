#pragma once

#include "particles.h"

// FLYBY_CM_VELOCITY: pair time steps use the subsystem centre-of-mass velocity of members (FROST-CLUSTERS II)
extern bool flyby_cm_velocity;

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
                          double G, double soft2, double eta);

// Range versions: sinks [i0, i0+ni) from sources [j0, j0+nj). The MPI paths pass their slice of the sinks.
void acceleration_and_timestep_range(Particles& p, int i0, int ni, int j0, int nj, double G, double soft2,
                                     double eta);
void acceleration_range(Particles& p, int i0, int ni, int j0, int nj, double G, double soft2);
void timestep_range(Particles& p, int i0, int ni, int j0, int nj, double G, double soft2, double eta);

// Snapshot of uncorrected accelerations for the gradient correction: element i - offset belongs to particle i
struct GradSnapshot { double* x; double* y; double* z; int offset; };
void gradient_correction_range(Particles& p, int i0, int ni, int j0, int nj, const GradSnapshot& si,
                               const GradSnapshot& sj, double G, double soft2, double dt, bool write_gmag);

// Standalone neighbor search (BIFROST Section 3.5)
// Lightweight N² tiled kernel: detects neighbor pairs without computing forces
// or timesteps. No subsystem exclusion — all pairs are checked.
// Writes detected pairs to g_subsystem_mgr neighbor buffers as array indices.
void neighbor_search_subset(Particles& p, int start, int count,
                            double G, double r_ngb, double r_ngb_max,
                            double pivot_dt);

