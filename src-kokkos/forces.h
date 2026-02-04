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
void compute_acceleration_and_frost(Particles& p, double G, double soft2, double eta);

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

// Compute FROST timesteps for a subset of particles [start, start+count)
// Only considers mutual interactions within the subset
template <bool UseN3L>
void compute_frost_subset(Particles& p, int start, int count,
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
