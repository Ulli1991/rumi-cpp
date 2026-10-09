#pragma once

#include "particles.h"
#include "kepler.h"
#include "logh.h"
#include <Kokkos_Core.hpp>
#include <vector>
#include <unordered_set>
#include <cmath>

// Neighbor pair detected during pairwise force/timestep computation
struct NeighborPair {
    int pid_i, pid_j;       // particle IDs (permanent, not array indices)
    double a;               // semi-major axis (negative if unbound)
    double e;               // eccentricity
    double E_orb;           // specific orbital energy
    bool bound;             // whether the pair is gravitationally bound
};

// Classification of subsystem type
enum class SubsystemType {
    BINARY,
    FLYBY,
    TRIPLE_STABLE,
    TRIPLE_UNSTABLE,
    MULTIPLET
};

struct Subsystem {
    int id;
    std::vector<int> particle_ids;      // permanent IDs of members
    int n_bodies;
    double dt_sub;                       // external timestep

    // Binary orbital elements (for 2-body subsystems)
    OrbitalElements orb;

    SubsystemType type;
    bool is_secular;                     // orbit-averaged evolution
    int n_orb_per_dt;                    // orbits per FSI timestep

    // Persistent auxiliary velocities for PN (code units, including CM).
    // The Hellström & Mikkola extended phase space method requires w ≠ v
    // after the first integration step. Re-initializing w = v each call
    // discards this divergence and causes O(v²/c²) energy errors.
    double aux_vel_phys[LOGH_MAX_BODIES * 3] = {};
    bool has_aux_vel = false;

    // Static normalization scales (computed once on first LogH call, fixed thereafter).
    // V_scale = sqrt(2*T_CM/M_total), R_scale = G*M/V², T_scale = R/V, E_scale = M*V².
    // Fixed scales prevent persistent B from drifting due to E_scale changes between calls.
    double V_scale = 0, R_scale = 0, T_scale = 0, E_scale = 0;
    bool scales_set = false;

    // Hierarchy info (for triples)
    int inner_binary_id;                 // -1 if not hierarchical
    int outer_companion_pid;             // -1 if not hierarchical
};

struct SubsystemManager {
    std::vector<Subsystem> active;
    int next_id = 0;

    // Cached particle_id → array index mapping (rebuilt per detect_from_pairs call)
    std::vector<int> pid_to_idx_;
    int pid_to_idx_max_ = 0;
    void rebuild_pid_to_idx(const Particles& p);
    void check_pid_to_idx(const Particles& p) const;   // full comparison (NBODY_CHECK_PIDMAP)
    const Particles* map_source_ = nullptr;          // particle set the map was built for
    Kokkos::View<int*> flag_buf_;                     // (index, sid) pairs uploaded by update_particle_flags
    const std::vector<int>& pid_to_idx() const { return pid_to_idx_; }
    int pid_to_idx_max() const { return pid_to_idx_max_; }

    // Bounds-checked lookup: returns array index for particle ID, or -1 if invalid
    int safe_pid_to_idx(int pid) const {
        return (pid >= 0 && pid < (int)pid_to_idx_.size()) ? pid_to_idx_[pid] : -1;
    }

    // Particle IDs removed by mergers. IDs are never reused, so the set only grows.
    std::unordered_set<int> removed_pids;

    // Reinstate a subsystem list saved before a recursive call, dropping subsystems
    // with a member that a merger removed in the meantime.
    void restore_active(const std::vector<Subsystem>& saved);

    // GPU output buffer for neighbor pairs (written by pairwise kernels)
    Kokkos::View<int*>    ngb_pair_i;    // particle index i
    Kokkos::View<int*>    ngb_pair_j;    // particle index j
    Kokkos::View<double*> ngb_pair_a;    // semi-major axis
    Kokkos::View<double*> ngb_pair_e;    // eccentricity
    Kokkos::View<double*> ngb_pair_E;    // orbital energy
    Kokkos::View<int*>    ngb_pair_bound;// 1=bound, 0=unbound
    Kokkos::View<int*>    ngb_count;     // atomic counter (single element)
    int max_ngb_pairs = 0;

    // Allocate GPU buffers for neighbor pair output
    void allocate_ngb_buffers(int max_pairs);

    // Reset the GPU neighbor pair counter to zero (call before force kernels)
    void reset_ngb_count();

    // Detect subsystems from neighbor pairs found during pairwise computation
    // Called after force/timestep kernels have written to ngb buffers
    void detect_from_pairs(Particles& p, double G, double dt_step,
                           double r_ngb, double r_ngb_max, int n_orb_secular);

    // Remap internal indices after partition_by_pivot reorders particles
    void remap_indices(const Particles& p);

    // Update subsystem_id field in Particles based on active subsystems
    void update_particle_flags(Particles& p);

    // Enforce shared timestep: all members of each subsystem get min dt_adapt
    // Call after detection and before partition_by_pivot
    void enforce_shared_timestep(Particles& p) const;

    // Perturbation timestep: clamp dt_adapt of subsystem members so the FSI
    // timestep resolves changes in orbital elements caused by external tidal
    // perturbations (BIFROST Section 3.4.3).
    // dt_pert = eta_pert * |v_rel| / |delta_a_tidal|
    // where delta_a_tidal = (a_j - a_i) + G*(m1+m2)/r^3 * r_rel
    // Call after enforce_shared_timestep, before partition_by_pivot.
    void apply_perturbation_timestep(Particles& p, double G, double eta_pert) const;

    // Per-level diagnostics: set by detect_from_pairs/accumulate_diagnostics
    int last_level_pairs = 0;       // neighbor pairs found at most recent detect_from_pairs
    int last_level_bound = 0;       // bound pairs at most recent detect_from_pairs

    // Step-level diagnostics: accumulated across HHS recursion levels
    int step_total_pairs = 0;       // total neighbor pairs found across all levels
    int step_total_bound = 0;       // total bound pairs across all levels
    int step_total_subsystems = 0;  // total subsystems formed
    int step_n_bin = 0, step_n_fly = 0, step_n_trip_s = 0, step_n_trip_u = 0, step_n_multi = 0;
    int step_n_secular = 0;
    int step_total_members = 0;

    // Step-level subsystem integration statistics
    int step_logh_calls = 0;           // subsystems integrated via LogH
    int step_logh_gbs_total = 0;       // total GBS steps
    double step_logh_worst_dE = 0;     // worst |dE/E| across all LogH calls
    double step_logh_best_dE = 1e30;   // best |dE/E|
    int step_logh_cap_hit = 0;         // hit GBS step limit (max_iter)
    int step_logh_restarts = 0;        // per-subsystem restarts triggered
    int step_logh_restart_ok = 0;      // restarts that succeeded
    int step_secular_evolved = 0;      // secular binaries evolved
    int step_secular_triples = 0;      // secular triples evolved

    // Step-level merger and escaper counters
    int step_mergers = 0;              // mergers executed this step
    int step_escapers = 0;             // escapers removed this step

    // Reset step-level counters (call at start of each outer step)
    void reset_step_diagnostics();

    // Accumulate current detection results into step counters
    void accumulate_diagnostics();

    // Print summary for the completed outer step
    void print_step_summary() const;

    // Clear all subsystems
    void clear();

    // Mardling (2001) stability criterion for hierarchical triples
    // Returns true if the triple is stable
    static bool mardling_stability(double a_in, double e_in,
                                   double a_out, double e_out,
                                   double m1, double m2, double m3);
};

// Global subsystem manager
extern SubsystemManager g_subsystem_mgr;
