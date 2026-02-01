#pragma once

#include "particles.h"

// When true, gradient corrections use N3L (halves work, ~1e-14 momentum).
// When false, uses full N² per-particle sums (original pattern, ~1e-15 momentum).
extern bool grad_n3l;

// ---- Full N-body (used for initial setup, global FSI, diagnostics) ----

// Full N² acceleration with N3L. Zeros and fills ax/ay/az for all particles.
void compute_acceleration(Particles& p, double G, double soft2);

// Full N² acceleration + FROST timestep in single fused pass.
// Zeros and fills ax/ay/az and dt_frost for all particles.
void compute_acceleration_and_frost(Particles& p, double G, double soft2, double eta);

// ---- Self-interaction within a contiguous group [start, start+count) ----
// These are O(count²) with N3L. They zero ax/ay/az for the group first.

void self_acceleration(Particles& p, int start, int count,
                       double G, double soft2);

// Gradient correction among [start, start+count) only.
// Reads existing ax/ay/az as input, ADDS correction in-place.
void self_gradient_correction(Particles& p, int start, int count,
                              double G, double soft2, double dt);

// Fused: self_acceleration + self_gradient_correction in reduced passes.
// Zeros and fills ax/ay/az, then adds gradient correction. Single function.
void self_acceleration_and_gradient(Particles& p, int start, int count,
                                    double G, double soft2, double dt);

// ---- Cross-interaction between two disjoint groups ----
// O(count_a * count_b), computes both directions in one pass.
// Zeros ax/ay/az for BOTH groups, then fills with cross-forces only.

void cross_acceleration(Particles& p,
                        int a_start, int a_count,
                        int b_start, int b_count,
                        double G, double soft2);

// Cross gradient correction between two groups.
// Reads existing ax/ay/az, ADDS correction in-place for both groups.
void cross_gradient_correction(Particles& p,
                               int a_start, int a_count,
                               int b_start, int b_count,
                               double G, double soft2, double dt);

// Fused: cross_acceleration + cross_gradient_correction.
// Zeros and fills ax/ay/az for both groups, then adds gradient correction.
void cross_acceleration_and_gradient(Particles& p,
                                     int a_start, int a_count,
                                     int b_start, int b_count,
                                     double G, double soft2, double dt);
