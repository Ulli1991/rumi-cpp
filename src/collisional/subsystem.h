#pragma once

#include "particles.h"
#include "kepler.h"
#include "logh.h"
#include <Kokkos_Core.hpp>
#include <vector>
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

    // Persistent binding energy B in physical (code) units.
    // NAN = not yet initialized (compute from -(T+U) on first LogH call).
    // Carried across LogH calls to properly track PN energy exchange.
    double B_phys = NAN;

    // Persistent auxiliary velocities for PN (code units, including CM).
    // The Hellström & Mikkola extended phase space method requires w ≠ v
    // after the first integration step. Re-initializing w = v each call
    // discards this divergence and causes O(v²/c²) energy errors.
    double aux_vel_phys[LOGH_MAX_BODIES * 3] = {};
    bool has_aux_vel = false;

    // Hierarchy info (for triples)
    int inner_binary_id;                 // -1 if not hierarchical
    int outer_companion_pid;             // -1 if not hierarchical
};

struct SubsystemManager {
    std::vector<Subsystem> active;
    int next_id = 0;

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

    // Dissolve subsystems: binaries when unbound, flybys when separated beyond r_ngb_max
    void dissolve_subsystems(Particles& p, double G, double r_ngb_max);

    // Update subsystem_id field in Particles based on active subsystems
    void update_particle_flags(Particles& p) const;

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
