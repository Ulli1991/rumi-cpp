#pragma once

#include "particles.h"
#include "subsystem.h"
#include "logh.h"
#include "mergers.h"
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
    double* m = nullptr;   // unpack writes masses back too (collisions change them)
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
    std::vector<double> ext_tidal;   // Subsystem::ext_tidal per subsystem (SDAR, Msun / pc^3)

    // Per-body stellar radius (R_sun) and type (host, same layout as flat_mass): SDAR collision candidates
    std::vector<double> flat_radius;
    std::vector<int> flat_stype;
    std::vector<int> flat_pid;          // particle id per body (same layout)

    // Collisions decided inside each subsystem's integration (FROST-CLUSTERS I App. B3), completed by the caller
    // with finalize_subsystem_collision; the merged-away member is written back massless (a "ghost")
    std::vector<std::vector<SubsystemCollision>> collisions;
    // Internal energy change of each subsystem's collisions (physical, sum of SubsystemCollision::dE_int)
    std::vector<double> coll_dE;

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
    int stat_sec_inner = 0;        // secular inner binaries collapsed inside PN subsystems (CASES.md C20)
    int stat_irk = 0;              // integrations done by Gauss-IRK (NBODY_GAUSS_IRK, CASES.md C21)
    int stat_irk_fb = 0;           // ... chosen for Gauss-IRK but finished by LogH+GBS (no convergence, collision watch)
    std::vector<char> irk_used;    // per subsystem: integrated by Gauss-IRK
    // MSTAR subsystems (g_collisional.mstar_group; docs/MSTAR.md): per subsystem
    std::vector<char> mstar_used;  // integrated by MSTAR
    std::vector<double> pn_dE;     // change of the members' Newtonian T + U by the PN forces (physical; MSTAR's PN
                                   // work ledger); the rest of the T + U change (minus collisions) is integration error
    int stat_mstar = 0, stat_mstar_dev = 0, stat_mstar_nmax = 0;
    long stat_mstar_gbs = 0;
    // Per-body spin vectors (chi, same layout as flat_mass) gathered when the spin couplings are on (has_flat_spin):
    // MSTAR subsystems evolve and write back the spins of their PN members
    std::vector<double> flat_spin;
    bool has_flat_spin = false;
    // NBODY_MSTAR_TIDE: tidal tensor of the outside bodies at each MSTAR subsystem's c.o.m. (9 per subsystem, row-major,
    // code units 1 / time^2; filled by pack) and the work of that tide on the members' T + U in the call (physical)
    std::vector<double> tide_A;
    std::vector<double> ext_dE;
    std::vector<double> t_reached; // physical time each subsystem's GBS integration reached (code units)
    int stat_collisions = 0;       // collisions decided inside the integrations
    int stat_peri_landings = 0;    // candidate pericentres a step was forced onto
    int stat_peri_misses = 0;      // ... whose actual pericentre cleared the collision distance

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
