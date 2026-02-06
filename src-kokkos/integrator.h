#pragma once

#include "particles.h"
#include <vector>
#include <string>

// Forward declaration for distributed particles
struct DistributedParticles;

// Set force function pointers based on UseN3L flag
void set_force_dispatch(bool use_n3l);

// Single-level FSI step on [start, start+count)
void fsi_step(Particles& p, int start, int count,
              double G, double soft2, double dt);

// HHS-FSI recursive step with dynamic pivot-based partitioning (per paper's Algorithm 4)
// Partitions particles [start, start+count) into slow (dt >= pivot) and fast (dt < pivot)
// Recomputes FROST for the current subset at each level
void hhs_fsi_step_dynamic(Particles& p, int start, int count, double pivot_dt,
                          double G, double soft2, double dt_min,
                          double eta, bool use_n3l);

// =============================================================================
// Distributed (Multi-GPU) HHS Integration
// =============================================================================
// NEW DESIGN: Scatter-compute-gather paradigm
//
// All ranks maintain replicated particle data in dp.local. The HHS recursion
// runs IDENTICALLY on all ranks. Only force computation is distributed
// (each rank computes for its slice, then Allgather accelerations).
//
// This achieves O(N) communication per step regardless of HHS depth.
// =============================================================================

// Distributed HHS-FSI using scatter-compute-gather
// dp: distributed particles manager with replicated particle data
// All ranks execute identical HHS recursion; only force computation is distributed
void hhs_fsi_step_distributed(DistributedParticles& dp, double pivot_dt,
                              double G, double soft2, double dt_min,
                              double eta, bool use_n3l, int global_levels = 2);

// Kick velocities using forces (for distributed case where force already computed)
void kick_velocities(Particles& p, int start, int count, double dt);

// Drift positions
void drift(Particles& p, int start, int count, double dt);

// Distributed force + FROST computation for between-step updates
// Each rank computes for its slice, then allgather accelerations and FROST
void compute_acceleration_and_frost_distributed(DistributedParticles& dp,
                                                 double G, double soft2, double eta);
