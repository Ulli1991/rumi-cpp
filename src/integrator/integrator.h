#pragma once

#include "particles.h"
#include <vector>
#include <string>

// Serial N² GPU threshold: below this particle count, skip MPI distribution
// and compute full N² forces locally on each rank (no tree, no MPI)
extern int serial_n2_gpu;              // 0 = disabled (always distributed)

// Serial N² CPU threshold: below this particle count, use CPU serial loops
// for full N² instead of GPU kernels (eliminates kernel launch overhead)
extern int serial_n2_cpu;              // 0 = disabled (always GPU)

extern bool use_integer_positions;      // INTEGER_POSITIONS config toggle
extern double int_pos_scale;            // physical_pos = int_pos * scale
extern double int_pos_inv_scale;        // 1.0 / scale (for drift)

extern bool use_nonblocking_mpi;        // MPI_NON_BLOCKING config toggle

// symmetric_timesteps removed — always enabled (hardcoded true)

// Barnes-Hut tree configuration
extern bool use_tree;                   // USE_TREE config toggle
extern bool distribute_subsystems;      // DISTRIBUTE_SUBSYSTEMS: each MPI rank integrates its share of subsystems
extern bool all_slow_split;             // ALL_SLOW_SPLIT (default on): an all-slow set takes two FSI steps of tau/2 (FROST Alg. 3)
extern int tree_threshold;              // Particle count above which tree is used (default: 10000)
extern double tree_theta;               // Opening angle (default: 0.5)
extern int tree_n_leaf;                 // Max particles per leaf (default: 32)
extern int tree_n_group;                // Particles per traversal group (default: 64)
// Forward declaration for distributed particles
struct DistributedParticles;

// Single-level FSI step on [start, start+count)
// forces_valid: if true, skip first force computation (accelerations already in ax/ay/az)
void fsi_step(Particles& p, int start, int count,
              double G, double soft2, double dt, bool forces_valid = false,
              bool force_direct_n2 = false);

// HHS-FSI recursive step with dynamic pivot-based partitioning (per paper's Algorithm 4)
// Partitions particles [start, start+count) into slow (dt >= pivot) and fast (dt < pivot)
// Recomputes adaptive timesteps for the current subset at each level
void hhs_fsi_step_dynamic(Particles& p, int start, int count, double pivot_dt,
                          double G, double soft2, double dt_min,
                          double eta);

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
                              double eta, int global_levels = 2);

// Kick velocities using forces (for distributed case where force already computed)
void kick_velocities(Particles& p, int start, int count, double dt);

// Drift positions
void drift(Particles& p, int start, int count, double dt);

// Drift field particles only (skip subsystem members with subsystem_id >= 0)
// Falls through to standard drift() when !p.collisional
void drift_field_only(Particles& p, int start, int count, double dt);

// Advance subsystems whose members are all in [start, start+count) via LogH
void advance_subsystems_in_range(Particles& p, int start, int count, double G, double dt);

// Distributed force + timestep computation for between-step updates
// Each rank computes for its slice, then allgather accelerations and timesteps
void compute_acceleration_and_timestep_distributed(DistributedParticles& dp,
                                                 double G, double soft2, double eta);

// Release global Kokkos Views held by the integrator (call before Kokkos::finalize)
void release_integrator_globals();
