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
    Kokkos::View<int*> cm_buf_;                       // member offsets and slots of fill_cm_velocity
    Kokkos::View<int*>::host_mirror_type cm_buf_host_;
    std::vector<int> cm_host_;
    Kokkos::View<int*> flag_buf_;                     // (index, sid) pairs uploaded by update_particle_flags
    std::vector<int> pid_scratch_;                    // pid -> local id inside detect_from_pairs (-1 between calls)

    // FLYBY_CM_KICK (mollified c.o.m. kicks): bound 2-body subsystems whose quadrupole tidal energy against all
    // particles, D = mu r_apo^2 sum_j G m_j / d_j^3 (d_j: distance to the c.o.m.), is at most FLYBY_CM_KICK_EPS x |E0|
    // (update_cm_kick_flags, once per step at HHS depth 0) interact with every other particle through their c.o.m.:
    // the force and gradient kernels see both members at the c.o.m. (cm_kick_swap / cm_kick_restore around each
    // launch). Pairwise central forces, so momentum and angular momentum stay exact; the dynamics conserve
    // H_mod = H - sum_flagged (U_quad + higher), which differs from the measured energy by at most sum D (bounded,
    // not accumulated). Their members need no tidal-sampling pair step (fill_cm_velocity: k = 1e30, floor 0).
    Kokkos::View<int*> cmk_tab_;                      // subsystem id -> flagged binary index (-1: not flagged)
    int cmk_ntab_ = 0, cmk_nb_ = 0;
    Kokkos::View<double*> cmk_sum_;                   // per flagged binary: m, m x, m y, m z of the swapped members
    Kokkos::View<int*> cmk_cnt_;                      // per flagged binary: members found in the swap range
    Kokkos::View<double*> cmk_save_;                  // true x, y, z of swapped particles (3 N)
    Kokkos::View<int*> cmk_swapped_;                  // 1: particle sits at its c.o.m. now
    int cmk_depth_ = 0;                               // nesting of cm_kick_swap (only the outermost swaps)
    int cmk_lo_ = 0, cmk_hi_ = 0;
    double cmk_dsum_ = 0.0, cmk_dmax_ = 0.0;          // sum and max of D over the flagged binaries (last update)
    int cmk_nbound_ = 0;                              // bound 2-body subsystems at the last update
    void update_cm_kick_flags(const Particles& p, double G, double tol);
    void cm_kick_swap(Particles& p, int lo, int hi);
    void cm_kick_restore(Particles& p);
    const SubsystemManager* shared_map_ = nullptr;   // local managers read the global map
    const std::vector<int>& pid_to_idx() const { return shared_map_ ? shared_map_->pid_to_idx_ : pid_to_idx_; }
    int pid_to_idx_max() const { return shared_map_ ? shared_map_->pid_to_idx_max_ : pid_to_idx_max_; }

    // Bounds-checked lookup: returns array index for particle ID, or -1 if invalid
    int safe_pid_to_idx(int pid) const {
        const auto& m = pid_to_idx();
        return (pid >= 0 && pid < (int)m.size()) ? m[pid] : -1;
    }

    // Particle IDs removed by mergers. IDs are never reused, so the set only grows.
    std::unordered_set<int> removed_pids;

    // Members merged away inside a subsystem integration (finalize_subsystem_collision) that are still in the
    // particle arrays as massless placeholders: they stay members of their subsystem (so the FSI never applies the
    // remnant's force to them) until the next compaction of their HHS range moves them to removed_pids
    std::unordered_set<int> ghost_pids;

    // Reinstate a subsystem list saved before a recursive call: removed members are dropped from their
    // subsystems (strip_removed), subsystems left with fewer than two members are dropped.
    void restore_active(const std::vector<Subsystem>& saved);

    // Drop removed_pids members from every active subsystem (persistent LogH state reset where members went),
    // and subsystems left with fewer than two members
    void strip_removed();

    // After compact_remove_zero_mass: the compacted ghosts (p.removed_pid_log from log_from on) become removed
    // pids, subsystems are stripped, the pid map and the flags rebuilt
    void ghosts_compacted(Particles& p, size_t log_from);

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

    // cvx, cvy, cvz <- velocities on [lo, hi), except that subsystem members get their subsystem's
    // centre-of-mass velocity (fly-by time-step criterion with FLYBY_CM_VELOCITY, FROST-CLUSTERS II sec. 2.1)
    // With tol > 0 (FLYBY_CM_EPS x |E0|) also ck <- tidal-sampling constants (PairVelocities::k): 1e30 for
    // non-members, tol / (2 G R mu v) for members, with R the apocentre separation and v the pericentre relative
    // speed of a bound pair (else the separation and relative speed), mu its reduced mass; for larger subsystems R
    // is twice the largest member distance from the c.o.m. and mu v the sum of m |v - v_cm|.
    // cf <- floor of the tidal-sampling step: res x P (1 - e)^(3/2) for members of a bound 2-body subsystem, else 0.
    // cm = false (component velocities, FLYBY_CM_VELOCITY off): only ck, cf are written, cvx, cvy, cvz are not touched
    void fill_cm_velocity(const Particles& p, int lo, int hi, Kokkos::View<double*> cvx, Kokkos::View<double*> cvy,
                          Kokkos::View<double*> cvz, Kokkos::View<double*> ck, Kokkos::View<double*> cf, double tol,
                          double res, bool cm = true);

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

    // enforce_shared_timestep then apply_perturbation_timestep with one member gather and scatter
    void subsystem_timesteps(Particles& p, double G, double eta_pert) const;

    // Per-level diagnostics: set by detect_from_pairs/accumulate_diagnostics
    int last_level_pairs = 0;       // neighbor pairs found at most recent detect_from_pairs
    int last_level_bound = 0;       // bound pairs at most recent detect_from_pairs
    int last_level_flyby_cm_dropped = 0;
    int last_level_flyby_curved = 0;   // distant fly-by pairs whose straight-line extrapolation fails
    int last_level_ngb_mass_kept = 0;      // pairs kept as strong perturbers by NGB_MASS_SCALING  // distant fly-by pairs of a binary member that fail the c.o.m. test (FLYBY_NGB_CM)

    // Step-level diagnostics: accumulated across HHS recursion levels
    int step_total_pairs = 0;       // total neighbor pairs found across all levels
    int step_total_bound = 0;       // total bound pairs across all levels
    int step_total_subsystems = 0;  // total subsystems formed
    int step_n_bin = 0, step_n_fly = 0, step_n_trip_s = 0, step_n_trip_u = 0, step_n_multi = 0;
    int step_flyby_cm_dropped = 0;
    int step_flyby_curved = 0;
    int step_ngb_mass_kept = 0;
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
    int step_sdar_calls = 0;           // SDAR: LogH integrations with a slowed inner binary
    double step_sdar_kappa_max = 0.0;  // SDAR: largest slow-down factor of the step
    int step_secular_evolved = 0;      // secular binaries evolved
    long long step_secular_substeps = 0;  // internal steps of the secular binary integrator
    int step_kepler2 = 0;              // ... of them on the Kepler drift only as non-PN pairs (unbound, or < N_ORB_SECULAR orbits)
    int step_secular_triples = 0;      // secular triples evolved

    // Step-level merger and escaper counters
    int step_mergers = 0;              // mergers executed this step
    int step_subsys_collisions = 0;    // ... of them decided inside subsystem integrations (FROST-CLUSTERS I App. B3)
    int step_peri_landings = 0;        // collision candidates a step was forced onto the pericentre of
    int step_peri_misses = 0;          // ... whose actual pericentre cleared the collision distance
    int step_escapers = 0;             // escapers removed this step

    // Reset step-level counters (call at start of each outer step)
    void reset_step_diagnostics();

    // Accumulate current detection results into step counters
    void accumulate_diagnostics();

    // Print summary for the completed outer step
    void print_step_summary() const;

    // Clear all subsystems
    void clear();

    // Mardling & Aarseth (2001) stability criterion for hierarchical triples (BIFROST Sec. 3.11); i_mut is the
    // mutual inclination of the inner and outer orbits in radians. Returns true if the triple is stable
    static bool mardling_stability(double a_in, double a_out, double e_out,
                                   double m1, double m2, double m3, double i_mut);
    // Angle between the inner and outer orbital angular momenta (relative coordinates), radians in [0, pi]
    static double mutual_inclination(const double r_in[3], const double v_in[3],
                                     const double r_out[3], const double v_out[3]);
};

// Global subsystem manager
extern SubsystemManager g_subsystem_mgr;
