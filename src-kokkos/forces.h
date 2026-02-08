#pragma once

#include "particles.h"

// Runtime flags (host-side)
extern bool grad_n3l;
extern bool use_kahan;
extern bool use_memory_tiling;

// N3L control: UseN3L=true uses upper-triangle + atomics, UseN3L=false uses full N^2
// Full N-body
template <bool UseN3L>
void compute_acceleration(Particles& p, double G, double soft2);

template <bool UseN3L>
void compute_acceleration_and_timestep(Particles& p, double G, double soft2, double eta);

// Self-interaction within [start, start+count)
template <bool UseN3L>
void self_acceleration(Particles& p, int start, int count,
                       double G, double soft2);

template <bool UseN3L>
void self_gradient_correction(Particles& p, int start, int count,
                              double G, double soft2, double dt);

template <bool UseN3L>
void self_acceleration_and_gradient(Particles& p, int start, int count,
                                    double G, double soft2, double dt);

// Cross-interaction between two disjoint groups
template <bool UseN3L>
void cross_acceleration(Particles& p,
                        int a_start, int a_count,
                        int b_start, int b_count,
                        double G, double soft2);

template <bool UseN3L>
void cross_gradient_correction(Particles& p,
                               int a_start, int a_count,
                               int b_start, int b_count,
                               double G, double soft2, double dt);

template <bool UseN3L>
void cross_acceleration_and_gradient(Particles& p,
                                     int a_start, int a_count,
                                     int b_start, int b_count,
                                     double G, double soft2, double dt);

// Compute adaptive timesteps for a subset of particles [start, start+count)
// Only considers mutual interactions within the subset
template <bool UseN3L>
void compute_timestep_subset(Particles& p, int start, int count,
                          double G, double soft2, double eta);

// ==========================================================================
// Fused force + kick functions (reduce kernel launches)
// ==========================================================================
template <bool UseN3L>
void self_acceleration_kick(Particles& p, int start, int count,
                            double G, double soft2, double kick_dt);

template <bool UseN3L>
void self_acceleration_and_gradient_kick(Particles& p, int start, int count,
                                         double G, double soft2, double grad_dt, double kick_dt);

template <bool UseN3L>
void cross_acceleration_kick(Particles& p,
                             int a_start, int a_count,
                             int b_start, int b_count,
                             double G, double soft2, double kick_dt);

template <bool UseN3L>
void cross_acceleration_and_gradient_kick(Particles& p,
                                          int a_start, int a_count,
                                          int b_start, int b_count,
                                          double G, double soft2, double grad_dt, double kick_dt);

// ==========================================================================
// Cached cross-force functions (for HHS Ahmad-Cohen style optimization)
// ==========================================================================
// Compute cross-force and cache accelerations in scratch arrays
template <bool UseN3L>
void cross_acceleration_kick_and_cache(Particles& p,
                                       int a_start, int a_count,
                                       int b_start, int b_count,
                                       double G, double soft2, double kick_dt);

// Apply kick using cached accelerations (no force recomputation)
void kick_from_cache(Particles& p, int a_start, int a_count,
                     int b_start, int b_count, double kick_dt);

// ==========================================================================
// Distributed (Multi-GPU) force functions
// ==========================================================================

// Compute gradient correction from ghost particles
// Adds correction to local particle accelerations based on ghost accelerations
// Requires: local.ax/ay/az contain total acceleration (local self + ghost forces)
//           ghosts.ax/ay/az contain ghost particle accelerations (from exchange)
void ghost_gradient_correction(Particles& local, const Particles& ghosts,
                               double G, double soft2, double dt);

// ==========================================================================
// Ghost cross-kick functions for distributed HHS
// These compute forces between local and ghost particles based on slow/fast partition
// ==========================================================================

// Compute forces from ghost particles on local particles and apply kick
// Only considers ghost particles with dt_adapt < pivot (fast ghosts) acting on
// local particles in [local_start, local_start+local_count) which are slow (dt >= pivot)
// This is one-way: we kick local particles, not ghosts (ghosts are kicked on their home rank)
void ghost_cross_kick_slow_from_fast(Particles& local, const Particles& ghosts,
                                     int local_start, int local_count,
                                     double pivot, double G, double soft2, double kick_dt);

// Compute forces from slow ghost particles on fast local particles and apply kick
// Only considers ghost particles with dt_adapt >= pivot (slow ghosts) acting on
// local particles in [local_start, local_start+local_count) which are fast (dt < pivot)
void ghost_cross_kick_fast_from_slow(Particles& local, const Particles& ghosts,
                                     int local_start, int local_count,
                                     double pivot, double G, double soft2, double kick_dt);

// Combined: compute ghost cross-kicks for both directions in HHS
// slow_start/count: local slow particles, fast_start/count: local fast particles
void ghost_cross_kicks_hhs(Particles& local, const Particles& ghosts,
                           int slow_start, int slow_count,
                           int fast_start, int fast_count,
                           double pivot, double G, double soft2, double kick_dt);

// Ghost cross-kick with gradient correction for the middle HHS kick
void ghost_cross_kicks_hhs_with_gradient(Particles& local, const Particles& ghosts,
                                         int slow_start, int slow_count,
                                         int fast_start, int fast_count,
                                         double pivot, double G, double soft2,
                                         double grad_dt, double kick_dt);
