#pragma once
// Restart files (HDF5): everything a run needs to continue bit for bit (HARD_PROBLEMS.md G1).
//
// File layout (format "cpp-nbody-restart", version 1; all doubles IEEE 64-bit, code units pc, M_sun, km/s,
// TU = pc / (km/s) = 0.977792 Myr):
//   /                attrs format, format_version, code_version, time, step, n_ranks, created
//   /run             attrs t, step, dt_global, snapshot_index, integrator, test; dataset diag0 [11];
//                    group blobs/: byte copies of main's trivially copyable trackers (BinaryTracker, ...)
//   /config          attrs config_text (the config file as read), config_json (key -> value)
//   /particles       attrs N, capacity, collisional, n_levels; one dataset per per-particle View, first N rows
//                    in storage order (x y z ix iy iz vx vy vz cvx cvy cvz ax ay az mass dt_adapt level
//                    orig_index pos_packed[N,4] vel_packed[N,4]; collisional: particle_id subsystem_id
//                    stellar_type m_zams stellar_age stellar_radius luminosity core_mass spin_vec[N,3] spin_mag
//                    dt_level0 grad_mag); level_start, level_count
//   /collisional     (ENABLE_COLLISIONAL builds)
//     config         attrs: every CollisionalConfig field (e_ref is restored, the rest compared with the config)
//     subsystems     attrs next_id, max_ngb_pairs; per-subsystem columns (id, n_bodies, dt_sub, orb[10], type,
//                    is_secular, n_orb_per_dt, aux_vel_phys[24], has_aux_vel, scales[4] = V R T E, scales_set,
//                    inner_binary_id, outer_companion_pid), member_offset[n+1], member_pid; removed_pids,
//                    ghost_pids (sorted)
//     energy_books   attrs E_escape E_gw E_merge E_sse E_pn E_sd P_sink[3] L_sink[3]
//     merger_tree    attr next_id; ints [n,5] (id pid1 pid2 pid_remnant type), reals [n,11] (time m1 m2 m_remnant
//                    a e v_kick E_radiated spin_remnant[3])
//     escaper_log    pid [n], reals [n,12] (time mass pos[3] vel[3] r v E_specific)
//     eattr          byte copy of g_eattr (NBODY_EATTR accumulators)
// There is no runtime random number generator (SSE kicks are deterministic, the IC generator's mt19937_64 is
// only used before step 0), and SSE keeps no state outside the particle fields, so neither appears here.

#include "particles.h"
#include "diagnostics.h"
#include <cstring>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

// Run-loop state owned by main.cpp
struct RunState {
    double t = 0.0;
    int step = 0;
    double dt_global = 0.0;      // fixed step of the FSI integrator (0 for the HHS)
    int snapshot_index = 0;      // number of the next snapshot
    DiagnosticState diag0{};     // reference state of |dE/E|, |dP|, |dL/L|
    std::string integrator, test;
    std::map<std::string, std::vector<unsigned char>> blobs;

    template <class T> void put(const std::string& key, const T& v) {
        static_assert(std::is_trivially_copyable_v<T>, "blobs hold trivially copyable objects only");
        auto* b = reinterpret_cast<const unsigned char*>(&v);
        blobs[key].assign(b, b + sizeof(T));
    }
    template <class T> bool get(const std::string& key, T& v) const {
        static_assert(std::is_trivially_copyable_v<T>, "blobs hold trivially copyable objects only");
        auto it = blobs.find(key);
        if (it == blobs.end() || it->second.size() != sizeof(T)) return false;
        std::memcpy(static_cast<void*>(&v), it->second.data(), sizeof(T));
        return true;
    }
};

struct RestartOptions {
    std::string dir = ".";       // RESTART_DIR
    int keep = 2;                // RESTART_KEEP: complete restart files kept (older ones are deleted)
};

// Writes <dir>/restart_<step, 8 digits>.h5 through <name>.partial and a rename, then deletes all but the newest
// `keep` restart files. Call on one rank. Returns the path.
std::string write_restart(const RestartOptions& opt, const Particles& p, const RunState& rs,
                          const std::string& config_text, const std::map<std::string, std::string>& params);

// Reads a restart file: allocates p (alloc = true) or overwrites the saved fields of an allocated p
// (alloc = false, after the MPI distribution), restores the collisional globals (alloc = true only)
// and fills rs. params_out (optional): the config keys the file was written with.
void read_restart(const std::string& path, Particles& p, RunState& rs, bool alloc,
                  std::map<std::string, std::string>* params_out = nullptr);

// Newest complete restart file in dir ("" if none)
std::string latest_restart(const std::string& dir);

// Config keys as a JSON object of strings (LUNA's config_json attribute)
std::string config_json(const std::map<std::string, std::string>& params);

// git describe of the source tree at build time
const char* code_version();

// MPI: true on every rank if it is true on any rank (no MPI: v)
bool any_rank(bool v);
// MPI: the non-particle state that goes into a restart file (time, step, energy books, subsystems, merger
// tree, escaper log) is the same on every rank; prints a message and returns false otherwise
bool ranks_agree_on_run_state(const RunState& rs);

// SIGUSR1 (e.g. sbatch --signal=USR1@600) asks for a restart file and a clean stop after the current step
void install_stop_signals();
bool stop_requested();
