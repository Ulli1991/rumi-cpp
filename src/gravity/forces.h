#pragma once

#include "particles.h"

// Full N-body
void compute_acceleration_and_timestep(Particles& p, double G, double soft2, double eta);

// Self-interaction within [start, start+count)
void self_acceleration(Particles& p, int start, int count,
                       double G, double soft2);

void self_gradient_correction(Particles& p, int start, int count,
                              double G, double soft2, double dt);

void self_acceleration_and_gradient(Particles& p, int start, int count,
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

void cross_acceleration_and_gradient(Particles& p,
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

// ==========================================================================
// Fused force + kick functions (reduce kernel launches)
// ==========================================================================
void self_acceleration_kick(Particles& p, int start, int count,
                            double G, double soft2, double kick_dt);

void self_acceleration_and_gradient_kick(Particles& p, int start, int count,
                                         double G, double soft2, double grad_dt, double kick_dt);

void cross_acceleration_kick(Particles& p,
                             int a_start, int a_count,
                             int b_start, int b_count,
                             double G, double soft2, double kick_dt);

void cross_acceleration_and_gradient_kick(Particles& p,
                                          int a_start, int a_count,
                                          int b_start, int b_count,
                                          double G, double soft2, double grad_dt, double kick_dt);

// ==========================================================================
// Cached cross-force functions (for HHS Ahmad-Cohen style optimization)
// ==========================================================================
// Compute cross-force and cache accelerations in scratch arrays
void cross_acceleration_kick_and_cache(Particles& p,
                                       int a_start, int a_count,
                                       int b_start, int b_count,
                                       double G, double soft2, double kick_dt);
