#pragma once

#include "particles.h"
#include "subsystem.h"
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
    Kokkos::View<double*> flat_ext_accel; // external accelerations [ax,ay,az]

    // Output: updated positions and velocities (same layout)
    Kokkos::View<double*> flat_pos_out;
    Kokkos::View<double*> flat_vel_out;

    // Per-subsystem results
    Kokkos::View<double*> energy_error;   // |dE/E| per subsystem
    Kokkos::View<int*>    converged;      // 1 if GBS converged

    // Persistent binding energy B in physical units (one per subsystem).
    // Threaded through pack→integrate→unpack to carry B across LogH calls.
    std::vector<double> stored_B;

    // Persistent auxiliary velocities for PN (code units, including CM).
    // Flat: stored_aux_vel[s][b*3+c] for subsystem s, body b, component c.
    std::vector<std::vector<double>> stored_aux_vel;
    std::vector<bool> has_stored_aux_vel;

    // Allocate buffers
    void allocate(int n_sub, int total_bodies);

    // Pack subsystems from SubsystemManager into batch
    void pack(const SubsystemManager& mgr, const Particles& p,
              double G, double dt);

    // Unpack results back into Particles and write B_phys back to subsystems
    void unpack(SubsystemManager& mgr, Particles& p);
};

// Integrate all subsystems in batch on GPU
// Each Kokkos team (warp or block) handles one subsystem
void logh_integrate_batch_gpu(SubsystemBatch& batch, double G, double eta_gbs);

// Global batch instance (reused across steps to avoid reallocation)
extern SubsystemBatch g_logh_batch;
