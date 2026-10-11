#pragma once

#include "particles.h"
#include "subsystem.h"
#include "logh.h"
#include <Kokkos_Core.hpp>
#include <vector>
#include <unordered_map>

// Batched LogH integration of independent subsystems on the host threads. The packed state lives in host
// memory: pack() fills it from a host copy of the members (HostSource) or by one gather from the device;
// unpack() writes the results back with one scatter to the device (and optionally into the host copy).

// Host copy of particle data [start, start + count) the caller already holds (column pointers, index - start)
struct HostSource {
    int start = 0, count = 0;
    double *x = nullptr, *y = nullptr, *z = nullptr, *vx = nullptr, *vy = nullptr, *vz = nullptr;
    const double* m = nullptr;
};

struct SubsystemBatch {
    int n_subsystems = 0;

    // Per-subsystem metadata
    std::vector<int>    n_bodies;       // body count per subsystem
    std::vector<int>    offset;         // offset into flat arrays
    std::vector<double> dt_target;      // target physical time advance

    // Flat packed arrays: [sum(n_bodies)] entries
    std::vector<double> flat_pos;       // [x,y,z] interleaved
    std::vector<double> flat_vel;       // [vx,vy,vz] interleaved
    std::vector<double> flat_mass;

    // Output: updated positions and velocities (same layout)
    std::vector<double> flat_pos_out;
    std::vector<double> flat_vel_out;

    // Per-subsystem results
    std::vector<double> energy_error;   // |dE/E| per subsystem
    std::vector<int>    converged;      // 1 if GBS converged

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

    // Per-body stellar radius (R_sun) and type (host, same layout as flat_mass): SDAR collision candidates
    std::vector<double> flat_radius;
    std::vector<int> flat_stype;

    // SDAR: energy jumps sum delta H_sd (physical units) of each subsystem's integration (Wang+2020 Eq. 24;
    // the change of the physical energy beyond the integration error, booked as E_sd by the caller)
    std::vector<double> sd_jump;

    // Member slots from pack() (unique, first-appearance order) and each body's row in that list, reused by unpack()
    std::vector<int> cached_member_indices;
    std::vector<int> body_member;       // [sum(n_bodies)]: row in cached_member_indices

    // Output statistics (filled by logh_integrate_batch_gpu)
    int stat_total_gbs = 0;        // total GBS steps across all subsystems
    double stat_worst_dE = 0;      // worst |dE/E|
    double stat_best_dE = 1e30;    // best |dE/E|
    int stat_cap_hit = 0;          // subsystems that hit GBS step limit
    int stat_sd_calls = 0;         // SDAR: integrations with a slowed binary at some step
    double stat_sd_kappa_max = 0;  // SDAR: largest slow-down factor
    std::vector<double> t_reached; // physical time each subsystem's GBS integration reached (code units)

    // Size the host arrays
    void allocate(int n_sub, int total_bodies);

    // Pack subsystems from SubsystemManager into batch. With src (all members inside its range) the state is read
    // from that host copy, and spins are read from the device only when the spin couplings are on (otherwise LogH
    // carries them unchanged); without src everything is gathered from the device.
    void pack(const SubsystemManager& mgr, const Particles& p,
              double G, double dt, const HostSource* src = nullptr);

    // Unpack results back into Particles (one scatter); with dst also into that host copy (positions, velocities)
    void unpack(SubsystemManager& mgr, Particles& p, const HostSource* dst = nullptr);
};

// Wall-time split of the batch calls (J-1 profile): accumulated by pack/run/unpack, read and cleared by the
// caller after each call. *_xfer: allocations, gather/scatter kernels and host<->device copies; integ: wall time
// of the host integration loop; crit: its longest single subsystem; work: sum over subsystems (all host threads)
struct LoghBatchProf {
    double pack_xfer = 0, run_xfer = 0, integ = 0, crit = 0, work = 0, unpack_xfer = 0;
    long long n_sub = 0;
};
extern LoghBatchProf g_logh_prof;

// Integrate all subsystems of the batch (host threads, one subsystem per task)
void logh_integrate_batch_gpu(SubsystemBatch& batch, double G, double eta_gbs);

// Global batch instance (reused across steps to avoid reallocation)
extern SubsystemBatch g_logh_batch;
