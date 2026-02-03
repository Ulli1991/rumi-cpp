#pragma once

#include "particles.h"

// Runtime flags (host-side)
extern bool grad_n3l;
extern bool use_kahan;

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
