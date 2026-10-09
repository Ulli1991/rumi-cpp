#pragma once

#include "particles.h"
#include "subsystem.h"
#include "logh.h"
#include <Kokkos_Core.hpp>
#include <vector>

// GPU-batched LogH integration: each Kokkos team integrates one subsystem
// Pack all subsystem states into contiguous arrays for GPU parallelism

struct SubsystemBatch {
    int n_subsystems;

    // Per-subsystem metadata
    Kokkos::View<int*>    n_bodies;       // body count per subsystem
    Kokkos::View<int*>    offset;         // offset into flat arrays
    Kokkos::View<double*> dt_target;      // target physical time advance

    // Flat packed arrays: [sum(n_bodies)] entries
    Kokkos::View<double*> flat_pos;       // [x,y,z] interleaved
    Kokkos::View<double*> flat_vel;       // [vx,vy,vz] interleaved
    Kokkos::View<double*> flat_mass;

    // Output: updated positions and velocities (same layout)
    Kokkos::View<double*> flat_pos_out;
    Kokkos::View<double*> flat_vel_out;

    // Per-subsystem results
    Kokkos::View<double*> energy_error;   // |dE/E| per subsystem
    Kokkos::View<int*>    converged;      // 1 if GBS converged

    // Persistent auxiliary velocities for PN (code units, including CM).
    // Flat layout: stored_aux_vel[offset[s] * 3 + b*3 + c]
    // stored_aux_vel_offsets[s] = starting body index for subsystem s
    std::vector<double> stored_aux_vel;
    std::vector<int> stored_aux_vel_offsets;
    std::vector<char> has_stored_aux_vel;

    // Persistent spin vectors for BH binaries (2-body only, code units)
    // stored_spins[s] = {chi1[3], chi2[3]} for subsystem s
    std::vector<double> stored_spins;  // flat: [n_sub * 6]
    std::vector<char> has_stored_spins;
    std::vector<int> stored_stellar_type;  // flat: [n_sub * 2] for 2-body (spins)

    // Persistent normalization scales (computed once per subsystem, fixed thereafter)
    std::vector<double> stored_V_scale, stored_R_scale, stored_T_scale, stored_E_scale;
    std::vector<char> stored_scales_set;

    // Cached member indices from pack() — reused by unpack() to avoid recomputation
    std::vector<int> cached_member_indices;
    std::unordered_map<int, int> cached_idx_to_local;

    // Output statistics (filled by logh_integrate_batch_gpu)
    int stat_total_gbs = 0;        // total GBS steps across all subsystems
    double stat_worst_dE = 0;      // worst |dE/E|
    double stat_best_dE = 1e30;    // best |dE/E|
    int stat_cap_hit = 0;          // subsystems that hit GBS step limit

    // Allocate buffers
    void allocate(int n_sub, int total_bodies);

    // Pack subsystems from SubsystemManager into batch
    void pack(const SubsystemManager& mgr, const Particles& p,
              double G, double dt);

    // Unpack results back into Particles
    void unpack(SubsystemManager& mgr, Particles& p);
};

// Integrate all subsystems in batch on GPU
// Each Kokkos team (warp or block) handles one subsystem
void logh_integrate_batch_gpu(SubsystemBatch& batch, double G, double eta_gbs);

// Global batch instance (reused across steps to avoid reallocation)
extern SubsystemBatch g_logh_batch;
