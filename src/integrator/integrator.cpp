#include "small_for.h"
#include "integrator.h"
#include "integrator_cpu.h"
#include "hhs_counters.h"
#include "forces.h"
#include "timestep.h"
#include "constants.h"
#include "mpi_comm.h"
#include "profiling.h"
#include "tree.h"
#ifdef ENABLE_COLLISIONAL
#include "collisional_config.h"
#include "subsystem.h"
#include "logh_batch.h"
#include "post_newtonian.h"
#include "secular.h"
#include "mergers.h"
#include "merger_tree.h"
#include "checkpoint.h"
#include "diagnostics.h"
#endif
#include <Kokkos_Core.hpp>
#include <vector>
#include <set>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <chrono>

// Global profiler instance
StepProfiler g_profiler;
SubsetHistogram g_subset_hist;

// HHS recursion debug counters (shared with integrator_cpu.cpp)
HHSCounters g_hhs_counters;

// Serial N² GPU threshold: skip MPI, compute full N² locally (no tree)
int serial_n2_gpu = 0;

// Serial N² CPU threshold: use CPU serial loops for full N²
int serial_n2_cpu = 0;

// Fused force+gradient: use stale accelerations to compute gradient in same N² pass as force

bool use_integer_positions = false;
double int_pos_scale = 0.0;
double int_pos_inv_scale = 0.0;

bool use_nonblocking_mpi = false;

// symmetric_timesteps removed — always enabled (hardcoded true)

// Barnes-Hut tree configuration
bool use_tree = false;
bool all_slow_split = false;
bool hhs_cc = false;
int hhs_cc_max = -1;   // -1: SERIAL_N2_GPU
bool distribute_subsystems = false;
int distribute_subsystems_min = 256;
int tree_threshold = 10000;
double tree_theta = 0.5;
int tree_n_leaf = 32;
int tree_n_group = 64;
// Global tree instance (rebuilt each top-level step)
static Octree g_octree;

void release_integrator_globals() { g_octree = Octree(); }  // before Kokkos::finalize

// TILE_SIZE defined in constants.h

// =============================================================================
// Distributed force + timestep computation for between-step updates
// Each rank computes for its slice [my_start, my_count) from ALL N particles
// Then allgather both accelerations and timestep values
// Uses tiled GPU kernel with shared memory for efficiency
// =============================================================================
void compute_acceleration_and_timestep_distributed(DistributedParticles& dp,
                                                 double G, double soft2, double eta) {
#ifdef USE_MPI
    if (MPIComm::size() == 1) {
        // Single rank: use optimized single-GPU version
        compute_acceleration_and_timestep(dp.local, G, soft2, eta);
        return;
    }

    const int N = dp.N_total;
    const int my_start = dp.my_start;
    const int my_count = dp.my_count;

    if (my_count == 0) {
        dp.allgather_accelerations();
        dp.allgather_timestep();
        return;
    }



    (void)N;
    acceleration_and_timestep_range(dp.local, my_start, my_count, 0, N, G, soft2, eta);
    Kokkos::fence();

    // Allgather accelerations and timesteps
    dp.allgather_accelerations();
    dp.allgather_timestep();
#else
    compute_acceleration_and_timestep(dp.local, G, soft2, eta);
#endif
}


// Debug: check for duplicate positions in the global tail of the particle array
// Uses FULL array copy (not subview) to match diagnostic's copy method
// Helper: drift positions for [start, start+count)
// Not static - also used by distributed FSI
void drift(Particles& p, int start, int count, double dt) {
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    if (use_integer_positions) {
        auto pix = p.ix; auto piy = p.iy; auto piz = p.iz;
        double scale = int_pos_scale;
        double inv_scale = int_pos_inv_scale;
        small_for("drift_int", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            pix(i) += llrint(pvx(i) * dt * inv_scale);
            piy(i) += llrint(pvy(i) * dt * inv_scale);
            piz(i) += llrint(pvz(i) * dt * inv_scale);
            px(i) = (double)pix(i) * scale;
            py(i) = (double)piy(i) * scale;
            pz(i) = (double)piz(i) * scale;
        });
    } else {
        small_for("drift", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            px(i) += pvx(i) * dt;
            py(i) += pvy(i) * dt;
            pz(i) += pvz(i) * dt;
        });
    }
    // No fence - next operation will handle dependency
}

#ifdef ENABLE_COLLISIONAL
void drift_field_only(Particles& p, int start, int count, double dt) {
    if (!p.collisional) {
        drift(p, start, count, dt);
        return;
    }
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    auto p_sid = p.subsystem_id;
    if (use_integer_positions) {
        auto pix = p.ix; auto piy = p.iy; auto piz = p.iz;
        double scale = int_pos_scale;
        double inv_scale = int_pos_inv_scale;
        small_for("drift_field_int", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            if (p_sid(i) >= 0) return;  // skip subsystem members
            pix(i) += llrint(pvx(i) * dt * inv_scale);
            piy(i) += llrint(pvy(i) * dt * inv_scale);
            piz(i) += llrint(pvz(i) * dt * inv_scale);
            px(i) = (double)pix(i) * scale;
            py(i) = (double)piy(i) * scale;
            pz(i) = (double)piz(i) * scale;
        });
    } else {
        small_for("drift_field", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            if (p_sid(i) >= 0) return;  // skip subsystem members
            px(i) += pvx(i) * dt;
            py(i) += pvy(i) * dt;
            pz(i) += pvz(i) * dt;
        });
    }
}

// =============================================================================

#if defined(ENABLE_COLLISIONAL) && defined(USE_MPI)
// DISTRIBUTE_SUBSYSTEMS: each rank advanced only the eligible subsystems it owns (position % ranks).
// Make the replicas identical again: members' state, persistent LogH state, bookkeeping deltas.
static void exchange_subsystem_results(Particles& p, const std::vector<int>& eligible,
                                       const EnergyBookkeeping& books_before) {
    const int nr = MPIComm::size(), me = MPIComm::rank();
    auto all_gather = [&](const std::vector<double>& mine) {
        int n = (int)mine.size();
        std::vector<int> counts(nr), displs(nr, 0);
        MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        for (int r = 1; r < nr; r++) displs[r] = displs[r - 1] + counts[r - 1];
        std::vector<double> all(displs[nr - 1] + counts[nr - 1]);
        MPI_Allgatherv(mine.data(), n, MPI_DOUBLE, all.data(), counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        return all;
    };
    // 1. Members' state from the owner (position, velocity, spin, kick compensation, and mass: the
    //    LogH unpack writes back m / M * M)
    constexpr int MS = 14;  // idx, x, y, z, vx, vy, vz, sx, sy, sz, cx, cy, cz, m
    std::vector<int> mine_idx;
    for (int k = me; k < (int)eligible.size(); k += nr)
        for (int pid : g_subsystem_mgr.active[eligible[k]].particle_ids) {
            int i = g_subsystem_mgr.safe_pid_to_idx(pid);
            if (i >= 0) mine_idx.push_back(i);
        }
    const int nm = (int)mine_idx.size();
    std::vector<double> mine(nm * (size_t)MS);
    if (nm > 0) {
        Kokkos::View<int*> d_idx("xch_idx", nm);
        Kokkos::View<double*> d_buf("xch_buf", nm * (size_t)MS);
        auto h_idx = Kokkos::create_mirror_view(d_idx);
        for (int k = 0; k < nm; k++) h_idx(k) = mine_idx[k];
        Kokkos::deep_copy(d_idx, h_idx);
        auto px = p.x, py = p.y, pz = p.z, pvx = p.vx, pvy = p.vy, pvz = p.vz;
        auto pcx = p.cvx, pcy = p.cvy, pcz = p.cvz;
        auto ps = p.spin_vec;
        auto pm = p.mass;
        Kokkos::parallel_for("xch_gather", nm, KOKKOS_LAMBDA(int k) {
            int i = d_idx(k); double* b = &d_buf(k * MS);
            b[0] = i; b[1] = px(i); b[2] = py(i); b[3] = pz(i); b[4] = pvx(i); b[5] = pvy(i); b[6] = pvz(i);
            b[7] = ps(i, 0); b[8] = ps(i, 1); b[9] = ps(i, 2); b[10] = pcx(i); b[11] = pcy(i); b[12] = pcz(i);
            b[13] = pm(i);
        });
        auto h_buf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_buf);
        std::copy(h_buf.data(), h_buf.data() + nm * (size_t)MS, mine.begin());
    }
    std::vector<double> all = all_gather(mine);
    const int na = (int)(all.size() / MS);
    if (na > 0) {
        Kokkos::View<double*> d_all("xch_all", all.size());
        auto h_all = Kokkos::create_mirror_view(d_all);
        std::copy(all.begin(), all.end(), h_all.data());
        Kokkos::deep_copy(d_all, h_all);
        auto px = p.x, py = p.y, pz = p.z, pvx = p.vx, pvy = p.vy, pvz = p.vz;
        auto pcx = p.cvx, pcy = p.cvy, pcz = p.cvz;
        auto ps = p.spin_vec;
        auto pm = p.mass;
        Kokkos::parallel_for("xch_scatter", na, KOKKOS_LAMBDA(int k) {
            const double* b = &d_all(k * MS); int i = (int)b[0];
            px(i) = b[1]; py(i) = b[2]; pz(i) = b[3]; pvx(i) = b[4]; pvy(i) = b[5]; pvz(i) = b[6];
            ps(i, 0) = b[7]; ps(i, 1) = b[8]; ps(i, 2) = b[9]; pcx(i) = b[10]; pcy(i) = b[11]; pcz(i) = b[12];
            pm(i) = b[13];
        });
    }
    // 2. Persistent per-subsystem state of the owner (LogH auxiliary velocities, scales)
    constexpr int SS = 7 + LOGH_MAX_BODIES * 3;  // k, has_aux_vel, scales_set, V, R, T, E, aux_vel_phys
    std::vector<double> sub_mine;
    for (int k = me; k < (int)eligible.size(); k += nr) {
        const auto& sub = g_subsystem_mgr.active[eligible[k]];
        sub_mine.insert(sub_mine.end(), {(double)k, sub.has_aux_vel ? 1.0 : 0.0, sub.scales_set ? 1.0 : 0.0,
                                         sub.V_scale, sub.R_scale, sub.T_scale, sub.E_scale});
        sub_mine.insert(sub_mine.end(), sub.aux_vel_phys, sub.aux_vel_phys + LOGH_MAX_BODIES * 3);
    }
    std::vector<double> sub_all = all_gather(sub_mine);
    for (size_t o = 0; o + SS <= sub_all.size(); o += SS) {
        auto& sub = g_subsystem_mgr.active[eligible[(int)sub_all[o]]];
        sub.has_aux_vel = sub_all[o + 1] != 0.0; sub.scales_set = sub_all[o + 2] != 0.0;
        sub.V_scale = sub_all[o + 3]; sub.R_scale = sub_all[o + 4]; sub.T_scale = sub_all[o + 5];
        sub.E_scale = sub_all[o + 6];
        std::copy(&sub_all[o + 7], &sub_all[o + 7] + LOGH_MAX_BODIES * 3, sub.aux_vel_phys);
    }
    // 3. Bookkeeping: every rank adds all ranks' deltas in rank order (bitwise the same everywhere)
    auto pack = [](const EnergyBookkeeping& b) {
        return std::vector<double>{b.E_escape, b.E_gw, b.E_merge, b.E_sse, b.E_pn, b.P_sink[0], b.P_sink[1],
                                   b.P_sink[2], b.L_sink[0], b.L_sink[1], b.L_sink[2]};
    };
    std::vector<double> now = pack(g_energy_books), before = pack(books_before), delta(now.size());
    for (size_t c = 0; c < now.size(); c++) delta[c] = now[c] - before[c];
    std::vector<double> deltas = all_gather(delta);
    std::vector<double> sum = before;
    for (int r = 0; r < nr; r++)
        for (size_t c = 0; c < sum.size(); c++) sum[c] += deltas[r * sum.size() + c];
    auto& b = g_energy_books;
    b.E_escape = sum[0]; b.E_gw = sum[1]; b.E_merge = sum[2]; b.E_sse = sum[3]; b.E_pn = sum[4];
    for (int c = 0; c < 3; c++) { b.P_sink[c] = sum[5 + c]; b.L_sink[c] = sum[8 + c]; }
}
#endif

// Wall time per phase of advance_subsystems_in_range, summed over a step (printed with the step summary)
struct SubsystemPhaseTimes {
    enum { MAP, SECULAR, SEC_TRIPLE, LOGH_PREP, LOGH_RUN, LOGH_UNPACK, LOGH_RETRY, N };
    static constexpr int MAXD = 40;
    double t[N] = {};
    double td[MAXD][N] = {};   // the same per HHS depth
    long long calls[MAXD] = {};
    void add(int phase, double dt) {
        t[phase] += dt;
        int d = std::min(g_profiler.cur_depth, MAXD - 1);
        td[d][phase] += dt;
    }
    bool any() const { for (double v : t) if (v > 0.0) return true; return false; }
    void report_and_reset() {
        static const char* name[N] = {"map+select", "secular", "sec_triple", "logh_prep+pack", "logh_run",
                                      "logh_unpack", "logh_retry"};
        std::printf("  Subsystem phases [s]:");
        for (int i = 0; i < N; i++) std::printf(" %s=%.3f", name[i], t[i]);
        std::printf("\n  %-6s %8s", "Depth", "calls");
        for (int i = 0; i < N; i++) std::printf(" %14s", name[i]);
        std::printf("\n");
        for (int d = 0; d < MAXD; d++) {
            if (calls[d] == 0) continue;
            std::printf("  d%-5d %8lld", d, calls[d]);
            for (int i = 0; i < N; i++) std::printf(" %14.3f", td[d][i]);
            std::printf("\n");
        }
        *this = SubsystemPhaseTimes();
    }
};
static SubsystemPhaseTimes g_subsys_phase;
namespace {
struct PhaseClock {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double lap() {
        auto t1 = std::chrono::steady_clock::now();
        double d = std::chrono::duration<double>(t1 - t0).count();
        t0 = t1;
        return d;
    }
};
}  // namespace

// Host copy of x, y, z, vx, vy, vz, mass of [start, start + count) for the subsystem advance: one packed device
// buffer and one transfer per direction instead of a transfer (and a host allocation) per array. Buffers persist.
namespace {
struct RangeMirror {
    enum { X, Y, Z, VX, VY, VZ, M, NF };
    using HostCol = Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
    int start = 0, count = 0;
    HostCol col[NF];
    static Kokkos::View<double*>& dbuf() {
        static Kokkos::View<double*> b("range_mirror_dev", 0);
        static bool hook = (Kokkos::push_finalize_hook([] { b = Kokkos::View<double*>(); }), true);
        (void)hook;
        return b;
    }
    static Kokkos::View<double*, Kokkos::HostSpace>& hbuf() {
        static Kokkos::View<double*, Kokkos::HostSpace> b("range_mirror_host", 0);
        static bool hook = (Kokkos::push_finalize_hook([] { b = Kokkos::View<double*, Kokkos::HostSpace>(); }), true);
        (void)hook;
        return b;
    }
    // nf = 6: positions and velocities, 7: also masses
    void pull(Particles& p, int nf) {
        auto d = dbuf();
        RawView x{p.x.data()}, y{p.y.data()}, z{p.z.data()}, vx{p.vx.data()}, vy{p.vy.data()}, vz{p.vz.data()},
            m{p.mass.data()};
        RawView o{d.data()};
        const int s0 = start, n = count;
        small_for("range_mirror_pack", n, KOKKOS_LAMBDA(int ii) {
            int i = s0 + ii;
            o(ii) = x(i); o(n + ii) = y(i); o(2 * n + ii) = z(i);
            o(3 * n + ii) = vx(i); o(4 * n + ii) = vy(i); o(5 * n + ii) = vz(i);
            if (nf > 6) o(6 * n + ii) = m(i);
        });
        auto rng = Kokkos::make_pair(0, nf * n);
        Kokkos::deep_copy(Kokkos::subview(hbuf(), rng), Kokkos::subview(d, rng));
    }
    void push(Particles& p) {   // positions and velocities
        auto d = dbuf();
        auto rng = Kokkos::make_pair(0, 6 * count);
        Kokkos::deep_copy(Kokkos::subview(d, rng), Kokkos::subview(hbuf(), rng));
        RawView x{p.x.data()}, y{p.y.data()}, z{p.z.data()}, vx{p.vx.data()}, vy{p.vy.data()}, vz{p.vz.data()};
        RawView o{d.data()};
        const int s0 = start, n = count;
        small_for("range_mirror_unpack", n, KOKKOS_LAMBDA(int ii) {
            int i = s0 + ii;
            x(i) = o(ii); y(i) = o(n + ii); z(i) = o(2 * n + ii);
            vx(i) = o(3 * n + ii); vy(i) = o(4 * n + ii); vz(i) = o(5 * n + ii);
        });
    }
    RangeMirror(Particles& p, int s, int n) : start(s), count(n) {
        if ((int)dbuf().extent(0) < NF * n) Kokkos::realloc(dbuf(), NF * n);
        if ((int)hbuf().extent(0) < NF * n) Kokkos::realloc(hbuf(), NF * n);
        for (int f = 0; f < NF; f++) col[f] = HostCol(hbuf().data() + (size_t)f * n, n);
        pull(p, NF);
    }
};

// NBODY_CHECK_SUBSYS=1: every subsystem this call owns must be advanced exactly once, over the full dt, by
// one of the paths (secular binary, secular triple, LogH). The FSI drifts no subsystem members, so a path that
// skips a subsystem (a `continue`, an early return) or stops short freezes its members with no energy change,
// invisible to the energy check and the restarts (AUDIT_2026-10.md F5). Aborts with the offending subsystem.
struct SubsysAdvanceCheck {
    bool on = false;
    double dt = 0.0;
    std::vector<int> owned;           // indices into g_subsystem_mgr.active this call must advance
    std::vector<int> n_adv;           // per active index: number of advances
    std::vector<double> t_adv;        // per active index: time advanced
    std::vector<const char*> path;    // per active index: last path that advanced it

    static bool enabled() {
        static const bool e = [] { const char* v = std::getenv("NBODY_CHECK_SUBSYS"); return v && v[0] && v[0] != '0'; }();
        return e;
    }
    void begin(int n_active, double dt_) {
        on = enabled();
        if (!on) return;
        dt = dt_;
        owned.clear();
        n_adv.assign(n_active, 0);
        t_adv.assign(n_active, 0.0);
        path.assign(n_active, "none");
    }
    void own(int s) { if (on) owned.push_back(s); }
    void advanced(int s, double t, const char* via) {
        if (!on) return;
        n_adv[s]++;
        t_adv[s] += t;
        path[s] = via;
    }
    void set_time(int s, double t) { if (on) t_adv[s] = t; }   // LogH retry replaces the first attempt
    void verify(int start, int count) const {
        if (!on) return;
        // GBS lands on the end time to a relative 1e-4 (NBODY_GBS_TTOL); anything further off is a partial advance
        const double tol = 1e-3 * std::fabs(dt);
        for (int s : owned) {
            if (n_adv[s] == 1 && std::fabs(t_adv[s] - dt) <= tol) continue;
            const auto& sub = g_subsystem_mgr.active[s];
            std::fprintf(stderr,
                "NBODY_CHECK_SUBSYS: subsystem %d (n_bodies=%d, type=%d, is_secular=%d, pids", s, sub.n_bodies,
                (int)sub.type, (int)sub.is_secular);
            for (int pid : sub.particle_ids) std::fprintf(stderr, " %d", pid);
            std::fprintf(stderr, ") advanced %d times over %.6e of dt=%.6e (last path: %s) in range [%d,%d)\n",
                         n_adv[s], t_adv[s], dt, path[s], start, start + count);
            std::fflush(stderr);
            std::abort();
        }
    }
};
}  // namespace

void advance_subsystems_in_range(Particles& p, int start, int count,
                                  double G, double dt) {
    PhaseClock phase_clock;
    if (!p.collisional || g_subsystem_mgr.active.empty()) return;
    g_subsys_phase.calls[std::min(g_profiler.cur_depth, SubsystemPhaseTimes::MAXD - 1)]++;

    // Rebuild pid→idx mapping: partition_by_pivot() reorders particles,
    // making the mapping from detect_from_pairs() stale.
    g_subsystem_mgr.rebuild_pid_to_idx(p);

    // Find which subsystems have ALL members in [start, start+count)
    const auto& pid_to_idx = g_subsystem_mgr.pid_to_idx();

    // Filter: only subsystems fully contained in [start, start+count)
    std::vector<int> eligible;
    int end = start + count;
    auto rng = Kokkos::pair<int,int>(start, end);
    for (int s = 0; s < (int)g_subsystem_mgr.active.size(); s++) {
        const auto& sub = g_subsystem_mgr.active[s];
        bool all_in = true;
        for (int pid : sub.particle_ids) {
            int idx = (pid < (int)pid_to_idx.size()) ? pid_to_idx[pid] : -1;
            if (idx < start || idx >= end) { all_in = false; break; }
        }
        if (all_in) eligible.push_back(s);
    }

    if (eligible.empty()) return;

    // Host copy of the particle data of [start, start+count); all accessed indices are in this range,
    // access as h_x(idx - start). Changes go back with mirror.push(p).
    RangeMirror mirror(p, start, count);
    auto h_x = mirror.col[RangeMirror::X], h_y = mirror.col[RangeMirror::Y], h_z = mirror.col[RangeMirror::Z];
    auto h_vx = mirror.col[RangeMirror::VX], h_vy = mirror.col[RangeMirror::VY], h_vz = mirror.col[RangeMirror::VZ];
    auto h_m = mirror.col[RangeMirror::M];

    // Re-evaluate secular/LogH classification from current orbital state.
    // Subsystems persist across steps, so the detection-time is_secular may be stale.
    {

        int n_orb_secular = g_collisional.n_orb_secular;

        for (int s : eligible) {
            auto& sub = g_subsystem_mgr.active[s];

            if (sub.n_bodies == 2) {
                int idx_i = g_subsystem_mgr.safe_pid_to_idx(sub.particle_ids[0]);
                int idx_j = g_subsystem_mgr.safe_pid_to_idx(sub.particle_ids[1]);
                if (idx_i < 0 || idx_j < 0) continue;

                double rx = h_x(idx_j - start) - h_x(idx_i - start);
                double ry = h_y(idx_j - start) - h_y(idx_i - start);
                double rz = h_z(idx_j - start) - h_z(idx_i - start);
                double vx = h_vx(idx_j - start) - h_vx(idx_i - start);
                double vy = h_vy(idx_j - start) - h_vy(idx_i - start);
                double vz = h_vz(idx_j - start) - h_vz(idx_i - start);
                double r2 = rx*rx + ry*ry + rz*rz;
                double v2 = vx*vx + vy*vy + vz*vz;
                double r_mag = std::sqrt(r2);
                double mu = G * (h_m(idx_i - start) + h_m(idx_j - start));
                double E_orb = 0.5 * v2 - mu / r_mag;

                // Update orbital elements
                sub.orb.E_orb = E_orb;
                if (E_orb < 0.0) {
                    double a = -mu / (2.0 * E_orb);
                    double L2 = (ry*vz-rz*vy)*(ry*vz-rz*vy) +
                                (rz*vx-rx*vz)*(rz*vx-rx*vz) +
                                (rx*vy-ry*vx)*(rx*vy-ry*vx);
                    double e2 = 1.0 - L2 / (mu * a);
                    sub.orb.a = a;
                    sub.orb.e = (e2 > 0.0) ? std::sqrt(e2) : 0.0;
                    sub.orb.period = 2.0 * M_PI * std::sqrt(a * a * a / mu);
                    sub.n_orb_per_dt = (int)(dt / sub.orb.period);
                    sub.is_secular = (sub.n_orb_per_dt >= n_orb_secular);

                    // Transition: was FLYBY but now bound → BINARY
                    if (sub.type == SubsystemType::FLYBY) sub.type = SubsystemType::BINARY;
                } else {
                    sub.orb.a = -mu / (2.0 * E_orb);  // hyperbolic
                    sub.is_secular = false;

                    // Transition: was BINARY but now unbound → FLYBY
                    if (sub.type == SubsystemType::BINARY) sub.type = SubsystemType::FLYBY;
                }

            } else if (sub.n_bodies == 3 && sub.inner_binary_id >= 0 && sub.outer_companion_pid >= 0) {
                // Recheck stability for triples with known hierarchy
                int inner_i = sub.inner_binary_id;
                int outer_pid = sub.outer_companion_pid;
                int inner_j = -1;
                for (int pid : sub.particle_ids) {
                    if (pid != inner_i && pid != outer_pid) { inner_j = pid; break; }
                }
                if (inner_j < 0) continue;

                int idx_i = (inner_i < (int)pid_to_idx.size()) ? pid_to_idx[inner_i] : -1;
                int idx_j = (inner_j < (int)pid_to_idx.size()) ? pid_to_idx[inner_j] : -1;
                int idx_o = (outer_pid < (int)pid_to_idx.size()) ? pid_to_idx[outer_pid] : -1;
                if (idx_i < 0 || idx_j < 0 || idx_o < 0) continue;

                double m1 = h_m(idx_i - start), m2 = h_m(idx_j - start), m3 = h_m(idx_o - start);
                double mu_in = G * (m1 + m2);

                // Inner binary elements
                double r_in[3] = {h_x(idx_j - start)-h_x(idx_i - start), h_y(idx_j - start)-h_y(idx_i - start), h_z(idx_j - start)-h_z(idx_i - start)};
                double v_in[3] = {h_vx(idx_j - start)-h_vx(idx_i - start), h_vy(idx_j - start)-h_vy(idx_i - start), h_vz(idx_j - start)-h_vz(idx_i - start)};
                auto oe_in = cartesian_to_elements(r_in, v_in, mu_in);

                if (oe_in.a > 0) {
                    sub.orb.a = oe_in.a;
                    sub.orb.e = oe_in.e;

                    // Outer orbit relative to inner COM
                    double M_in = m1 + m2;
                    double cx  = (m1*h_x(idx_i - start)  + m2*h_x(idx_j - start))  / M_in;
                    double cy  = (m1*h_y(idx_i - start)  + m2*h_y(idx_j - start))  / M_in;
                    double cz  = (m1*h_z(idx_i - start)  + m2*h_z(idx_j - start))  / M_in;
                    double cvx = (m1*h_vx(idx_i - start) + m2*h_vx(idx_j - start)) / M_in;
                    double cvy = (m1*h_vy(idx_i - start) + m2*h_vy(idx_j - start)) / M_in;
                    double cvz = (m1*h_vz(idx_i - start) + m2*h_vz(idx_j - start)) / M_in;
                    double r_out[3] = {h_x(idx_o - start)-cx, h_y(idx_o - start)-cy, h_z(idx_o - start)-cz};
                    double v_out[3] = {h_vx(idx_o - start)-cvx, h_vy(idx_o - start)-cvy, h_vz(idx_o - start)-cvz};
                    auto oe_out = cartesian_to_elements(r_out, v_out, G * (M_in + m3));

                    if (oe_out.a > 0) {
                        bool stable = SubsystemManager::mardling_stability(
                            oe_in.a, oe_in.e, oe_out.a, oe_out.e, m1, m2, m3);
                        sub.type = stable ? SubsystemType::TRIPLE_STABLE : SubsystemType::TRIPLE_UNSTABLE;

                        if (stable) {
                            double P_in = 2.0 * M_PI * std::sqrt(oe_in.a*oe_in.a*oe_in.a / mu_in);
                            sub.n_orb_per_dt = (int)(dt / P_in);
                            sub.is_secular = (sub.n_orb_per_dt >= n_orb_secular);
                        } else {
                            sub.is_secular = false;
                        }
                    } else {
                        sub.type = SubsystemType::TRIPLE_UNSTABLE;
                        sub.is_secular = false;
                    }
                } else {
                    sub.type = SubsystemType::TRIPLE_UNSTABLE;
                    sub.is_secular = false;
                }
            }
        }
    }

    // Partition eligible subsystems into secular binaries, secular triples, and LogH. With
    // DISTRIBUTE_SUBSYSTEMS each rank takes the ones at positions rank, rank + ranks, ... when there are
    // enough of them to pay for the exchange (replicas agree on `eligible`, so all ranks decide alike)
    bool distributed_subs = false;
#ifdef USE_MPI
    distributed_subs = distribute_subsystems && MPIComm::size() > 1 &&
                       (int)eligible.size() >= distribute_subsystems_min;
    const EnergyBookkeeping books_before = g_energy_books;
#endif
    std::vector<int> secular_idx, secular_triple_idx, logh_idx;
    SubsysAdvanceCheck adv_check;
    adv_check.begin((int)g_subsystem_mgr.active.size(), dt);
    for (int k = 0; k < (int)eligible.size(); k++) {
        const int s = eligible[k];
#ifdef USE_MPI
        if (distributed_subs && k % MPIComm::size() != MPIComm::rank()) continue;  // another rank advances it
#endif
        adv_check.own(s);
        const auto& sub = g_subsystem_mgr.active[s];
        if (sub.is_secular && sub.n_bodies == 2) {
            secular_idx.push_back(s);
        } else if (sub.is_secular && sub.type == SubsystemType::TRIPLE_STABLE) {
            secular_triple_idx.push_back(s);
        } else {
            logh_idx.push_back(s);
        }
    }

    g_subsys_phase.add(SubsystemPhaseTimes::MAP, phase_clock.lap());

    // --- Secular binaries: evolve on host ---
    if (!secular_idx.empty()) {
        // Binaries are independent (disjoint members): evolve them on all host threads. The GW energy is
        // summed afterwards in a fixed order, so every MPI rank books the same value.
        const int n_sec = (int)secular_idx.size();
        std::vector<double> dE_gw(n_sec, 0.0);
        std::vector<char> evolved(n_sec, 0);
        std::vector<int> substeps(n_sec, 0);
        std::vector<EnergyBookkeeping> sec_books(n_sec);  // P/L the secular evolution moves (PN, GW)
        auto evolve = [&](int k) {
            const auto& sub = g_subsystem_mgr.active[secular_idx[k]];
            int idx0 = g_subsystem_mgr.safe_pid_to_idx(sub.particle_ids[0]);
            int idx1 = g_subsystem_mgr.safe_pid_to_idx(sub.particle_ids[1]);
            if (idx0 < 0 || idx1 < 0) return;

            double p1[3] = {h_x(idx0 - start), h_y(idx0 - start), h_z(idx0 - start)};
            double v1[3] = {h_vx(idx0 - start), h_vy(idx0 - start), h_vz(idx0 - start)};
            double p2[3] = {h_x(idx1 - start), h_y(idx1 - start), h_z(idx1 - start)};
            double v2[3] = {h_vx(idx1 - start), h_vy(idx1 - start), h_vz(idx1 - start)};

            evolved[k] = 1;

            // Compute orbital energy before secular evolution for GW bookkeeping
            double m1_sec = h_m(idx0 - start), m2_sec = h_m(idx1 - start);
            double mu_sec = G * (m1_sec + m2_sec);
            double r_rel_pre[3] = {p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]};
            double v_rel_pre[3] = {v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]};
            auto oe_pre = cartesian_to_elements(r_rel_pre, v_rel_pre, mu_sec);
            double E_orb_pre = -G * m1_sec * m2_sec / (2.0 * oe_pre.a);

            substeps[k] = integrate_secular_binary(p1, v1, h_m(idx0 - start),
                                     p2, v2, h_m(idx1 - start),
                                     G, g_collisional.c_code,
                                     g_collisional.pn_order, dt);

            // GW energy bookkeeping: energy radiated = E_orb_before - E_orb_after
            double r_rel_post[3] = {p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]};
            double v_rel_post[3] = {v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]};
            auto oe_post = cartesian_to_elements(r_rel_post, v_rel_post, mu_sec);
            if (oe_post.a > 0.0 && oe_pre.a > 0.0) {
                double E_orb_post = -G * m1_sec * m2_sec / (2.0 * oe_post.a);
                dE_gw[k] = E_orb_pre - E_orb_post;
            }

            // P and L moved by the evolution (Kepler motion and the CoM drift conserve both)
            {
                double r1o[3] = {h_x(idx0 - start), h_y(idx0 - start), h_z(idx0 - start)};
                double v1o[3] = {h_vx(idx0 - start), h_vy(idx0 - start), h_vz(idx0 - start)};
                double r2o[3] = {h_x(idx1 - start), h_y(idx1 - start), h_z(idx1 - start)};
                double v2o[3] = {h_vx(idx1 - start), h_vy(idx1 - start), h_vz(idx1 - start)};
                sec_books[k].sink_PL(m1_sec, r1o, v1o);
                sec_books[k].sink_PL(m2_sec, r2o, v2o);
                sec_books[k].sink_PL(m1_sec, p1, v1, -1.0);
                sec_books[k].sink_PL(m2_sec, p2, v2, -1.0);
            }

            // Write back to host mirrors
            h_x(idx0 - start) = p1[0]; h_y(idx0 - start) = p1[1]; h_z(idx0 - start) = p1[2];
            h_vx(idx0 - start) = v1[0]; h_vy(idx0 - start) = v1[1]; h_vz(idx0 - start) = v1[2];
            h_x(idx1 - start) = p2[0]; h_y(idx1 - start) = p2[1]; h_z(idx1 - start) = p2[2];
            h_vx(idx1 - start) = v2[0]; h_vy(idx1 - start) = v2[1]; h_vz(idx1 - start) = v2[2];
        };
        // A few binaries (deep levels, thousands of calls per step) cost less than starting the host threads
        if (n_sec < 32) {
            for (int k = 0; k < n_sec; k++) evolve(k);
        } else {
            Kokkos::parallel_for("secular_binaries",
                Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>(0, n_sec),
                evolve);
        }
        for (int k = 0; k < n_sec; k++) {
            g_energy_books.E_gw += dE_gw[k];
            for (int c = 0; c < 3; c++) {
                g_energy_books.P_sink[c] += sec_books[k].P_sink[c];
                g_energy_books.L_sink[c] += sec_books[k].L_sink[c];
            }
            g_subsystem_mgr.step_secular_evolved += evolved[k];
            if (evolved[k]) adv_check.advanced(secular_idx[k], dt, "secular binary");
            g_subsystem_mgr.step_secular_substeps += substeps[k];
        }

        // Copy modified host subviews back to device
        mirror.push(p);

        // Clear aux_vel for secular subsystems (not used by secular path;
        // if subsystem transitions back to LogH, false triggers fresh init)
        for (int s : secular_idx) {
            auto& dst = g_subsystem_mgr.active[s];
            dst.has_aux_vel = false;
        }
    }

    g_subsys_phase.add(SubsystemPhaseTimes::SECULAR, phase_clock.lap());

    // --- Secular triples: Kozai-Lidov + GW dissipation ---
    if (!secular_triple_idx.empty()) {
        for (int s : secular_triple_idx) {
            g_subsystem_mgr.step_secular_triples++;
            const auto& sub = g_subsystem_mgr.active[s];
            // A triple this path cannot evolve goes to LogH: skipping it would leave its members frozen for the
            // whole step (the FSI drifts no subsystem members)
            if (sub.n_bodies != 3 || sub.inner_binary_id < 0 || sub.outer_companion_pid < 0) {
                logh_idx.push_back(s);
                continue;
            }

            // Inner binary and outer companion from the classification (particle_ids are sorted by id, so
            // their order says nothing about the hierarchy)
            int inner_j = -1;
            for (int pid : sub.particle_ids)
                if (pid != sub.inner_binary_id && pid != sub.outer_companion_pid) inner_j = pid;
            int idx0 = g_subsystem_mgr.safe_pid_to_idx(sub.inner_binary_id);
            int idx1 = inner_j >= 0 ? g_subsystem_mgr.safe_pid_to_idx(inner_j) : -1;
            int idx2 = g_subsystem_mgr.safe_pid_to_idx(sub.outer_companion_pid);
            if (idx0 < 0 || idx1 < 0 || idx2 < 0) {
                logh_idx.push_back(s);
                continue;
            }

            double m1 = h_m(idx0 - start), m2 = h_m(idx1 - start), m3 = h_m(idx2 - start);
            double mu_in = G * (m1 + m2);

            // Inner binary relative coords
            double r_in[3] = {h_x(idx1 - start)-h_x(idx0 - start), h_y(idx1 - start)-h_y(idx0 - start), h_z(idx1 - start)-h_z(idx0 - start)};
            double v_in[3] = {h_vx(idx1 - start)-h_vx(idx0 - start), h_vy(idx1 - start)-h_vy(idx0 - start), h_vz(idx1 - start)-h_vz(idx0 - start)};

            // Inner binary -> elements
            OrbitalElements oe_in = cartesian_to_elements(r_in, v_in, mu_in);

            // Outer orbit: CM of inner binary vs outer companion
            double M12 = m1 + m2;
            double cm_x = (m1*h_x(idx0 - start) + m2*h_x(idx1 - start)) / M12;
            double cm_y = (m1*h_y(idx0 - start) + m2*h_y(idx1 - start)) / M12;
            double cm_z = (m1*h_z(idx0 - start) + m2*h_z(idx1 - start)) / M12;
            double cm_vx = (m1*h_vx(idx0 - start) + m2*h_vx(idx1 - start)) / M12;
            double cm_vy = (m1*h_vy(idx0 - start) + m2*h_vy(idx1 - start)) / M12;
            double cm_vz = (m1*h_vz(idx0 - start) + m2*h_vz(idx1 - start)) / M12;

            double r_out[3] = {h_x(idx2 - start)-cm_x, h_y(idx2 - start)-cm_y, h_z(idx2 - start)-cm_z};
            double v_out[3] = {h_vx(idx2 - start)-cm_vx, h_vy(idx2 - start)-cm_vy, h_vz(idx2 - start)-cm_vz};
            double mu_out = G * (M12 + m3);
            OrbitalElements oe_out = cartesian_to_elements(r_out, v_out, mu_out);

            if (oe_in.a <= 0 || oe_out.a <= 0) {  // no longer hierarchical: integrate it directly
                logh_idx.push_back(s);
                continue;
            }

            // Mutual inclination from angular momentum vectors
            double L_in[3] = {r_in[1]*v_in[2]-r_in[2]*v_in[1],
                              r_in[2]*v_in[0]-r_in[0]*v_in[2],
                              r_in[0]*v_in[1]-r_in[1]*v_in[0]};
            double L_out[3] = {r_out[1]*v_out[2]-r_out[2]*v_out[1],
                               r_out[2]*v_out[0]-r_out[0]*v_out[2],
                               r_out[0]*v_out[1]-r_out[1]*v_out[0]};
            double L_in_mag = std::sqrt(L_in[0]*L_in[0]+L_in[1]*L_in[1]+L_in[2]*L_in[2]);
            double L_out_mag = std::sqrt(L_out[0]*L_out[0]+L_out[1]*L_out[1]+L_out[2]*L_out[2]);
            double cos_i_mut = (L_in_mag > 0 && L_out_mag > 0) ?
                (L_in[0]*L_out[0]+L_in[1]*L_out[1]+L_in[2]*L_out[2]) / (L_in_mag*L_out_mag) : 1.0;

            // Build SecularTriple and evolve
            SecularTriple tri{};
            tri.inner.a = oe_in.a; tri.inner.e = oe_in.e; tri.inner.omega = oe_in.omega;
            tri.inner.inc = oe_in.inc; tri.inner.Omega = oe_in.Omega;
            tri.inner.m1 = m1; tri.inner.m2 = m2;
            tri.inner.pid1 = sub.inner_binary_id; tri.inner.pid2 = inner_j;
            tri.a_out = oe_out.a; tri.e_out = oe_out.e;
            tri.omega_out = oe_out.omega; tri.inc_out = oe_out.inc; tri.Omega_out = oe_out.Omega;
            tri.m3 = m3; tri.pid3 = sub.outer_companion_pid;
            tri.cos_i_mut = cos_i_mut;

            printf("  [SEC_TRI] pid=%d,%d,%d  a_in=%.4e  e_in=%.6f  a_out=%.4e  cos_i=%.4f  dt=%.4e\n",
                   sub.inner_binary_id, inner_j, sub.outer_companion_pid,
                   tri.inner.a, tri.inner.e, tri.a_out, cos_i_mut, dt);

            double a_in_pre = tri.inner.a;  // save for GW bookkeeping

            secular_evolve_triple(tri, dt, G, g_collisional.c_code, g_collisional.pn_order);
            adv_check.advanced(s, dt, "secular triple");

            // GW energy bookkeeping for inner binary inspiral
            if (tri.inner.a > 0.0 && a_in_pre > 0.0) {
                double E_orb_pre_tri = -G * m1 * m2 / (2.0 * a_in_pre);
                double E_orb_post_tri = -G * m1 * m2 / (2.0 * tri.inner.a);
                g_energy_books.E_gw += E_orb_pre_tri - E_orb_post_tri;
            }

            // Update inner orbital plane to reflect changed mutual inclination.
            // The secular integrator evolves cos_i_mut but not (inc, Omega).
            // Rotate L_in around the node vector (L_out × L_in) by delta_i.
            {
                double L_in_hat[3] = {L_in[0]/L_in_mag, L_in[1]/L_in_mag, L_in[2]/L_in_mag};
                double L_out_hat[3] = {L_out[0]/L_out_mag, L_out[1]/L_out_mag, L_out[2]/L_out_mag};

                // Node vector = L_out × L_in (rotation axis)
                double nd[3] = {L_out_hat[1]*L_in_hat[2] - L_out_hat[2]*L_in_hat[1],
                                L_out_hat[2]*L_in_hat[0] - L_out_hat[0]*L_in_hat[2],
                                L_out_hat[0]*L_in_hat[1] - L_out_hat[1]*L_in_hat[0]};
                double nd_mag = std::sqrt(nd[0]*nd[0] + nd[1]*nd[1] + nd[2]*nd[2]);

                if (nd_mag > 1e-30) {
                    nd[0] /= nd_mag; nd[1] /= nd_mag; nd[2] /= nd_mag;

                    double i_old = std::acos(std::max(-1.0, std::min(1.0, cos_i_mut)));
                    double i_new = std::acos(std::max(-1.0, std::min(1.0, tri.cos_i_mut)));
                    double delta_i = i_new - i_old;

                    // Rodrigues' rotation: R(v,θ) = v cosθ + (k×v) sinθ + k(k·v)(1-cosθ)
                    double cd = std::cos(delta_i), sd = std::sin(delta_i);
                    double kdv = nd[0]*L_in_hat[0] + nd[1]*L_in_hat[1] + nd[2]*L_in_hat[2];
                    double kcv[3] = {nd[1]*L_in_hat[2] - nd[2]*L_in_hat[1],
                                     nd[2]*L_in_hat[0] - nd[0]*L_in_hat[2],
                                     nd[0]*L_in_hat[1] - nd[1]*L_in_hat[0]};
                    double L_new[3];
                    for (int k = 0; k < 3; k++)
                        L_new[k] = L_in_hat[k]*cd + kcv[k]*sd + nd[k]*kdv*(1.0-cd);

                    // Extract new inc, Omega from rotated L_in
                    double Lz = std::max(-1.0, std::min(1.0, L_new[2]));
                    tri.inner.inc = std::acos(Lz);
                    tri.inner.Omega = std::atan2(L_new[0], -L_new[1]);
                }
            }

            // Convert evolved elements back to Cartesian
            // Inner binary
            double n_in = std::sqrt(mu_in / (tri.inner.a * tri.inner.a * tri.inner.a));
            double M_in_new = oe_in.mean_anom + n_in * dt;
            double r_in_new[3], v_in_new[3];
            elements_to_cartesian(tri.inner.a, tri.inner.e, tri.inner.inc,
                                   tri.inner.Omega, tri.inner.omega, M_in_new, mu_in,
                                   r_in_new, v_in_new);

            // Outer orbit
            double n_out = std::sqrt(mu_out / (tri.a_out * tri.a_out * tri.a_out));
            double M_out_new = oe_out.mean_anom + n_out * dt;
            double r_out_new[3], v_out_new[3];
            elements_to_cartesian(tri.a_out, tri.e_out, tri.inc_out,
                                   tri.Omega_out, tri.omega_out, M_out_new, mu_out,
                                   r_out_new, v_out_new);

            // Reconstruct absolute coordinates
            // Total CM
            double M_tot = M12 + m3;
            double cm_tot[3] = {
                (M12*cm_x + m3*h_x(idx2 - start)) / M_tot,
                (M12*cm_y + m3*h_y(idx2 - start)) / M_tot,
                (M12*cm_z + m3*h_z(idx2 - start)) / M_tot
            };
            double cm_tot_v[3] = {
                (M12*cm_vx + m3*h_vx(idx2 - start)) / M_tot,
                (M12*cm_vy + m3*h_vy(idx2 - start)) / M_tot,
                (M12*cm_vz + m3*h_vz(idx2 - start)) / M_tot
            };

            // Drift total CM
            for (int k = 0; k < 3; k++) cm_tot[k] += cm_tot_v[k] * dt;

            // Outer orbit positions: CM_inner and particle 3
            double cm_in_new[3], v_cm_in_new[3];
            for (int k = 0; k < 3; k++) {
                cm_in_new[k] = cm_tot[k] - (m3/M_tot) * r_out_new[k];
                v_cm_in_new[k] = cm_tot_v[k] - (m3/M_tot) * v_out_new[k];
            }

            // Inner binary: particle 0 and 1
            h_x(idx0 - start) = cm_in_new[0] - (m2/M12)*r_in_new[0];
            h_y(idx0 - start) = cm_in_new[1] - (m2/M12)*r_in_new[1];
            h_z(idx0 - start) = cm_in_new[2] - (m2/M12)*r_in_new[2];
            h_vx(idx0 - start) = v_cm_in_new[0] - (m2/M12)*v_in_new[0];
            h_vy(idx0 - start) = v_cm_in_new[1] - (m2/M12)*v_in_new[1];
            h_vz(idx0 - start) = v_cm_in_new[2] - (m2/M12)*v_in_new[2];

            h_x(idx1 - start) = cm_in_new[0] + (m1/M12)*r_in_new[0];
            h_y(idx1 - start) = cm_in_new[1] + (m1/M12)*r_in_new[1];
            h_z(idx1 - start) = cm_in_new[2] + (m1/M12)*r_in_new[2];
            h_vx(idx1 - start) = v_cm_in_new[0] + (m1/M12)*v_in_new[0];
            h_vy(idx1 - start) = v_cm_in_new[1] + (m1/M12)*v_in_new[1];
            h_vz(idx1 - start) = v_cm_in_new[2] + (m1/M12)*v_in_new[2];

            h_x(idx2 - start) = cm_tot[0] + (M12/M_tot)*r_out_new[0];
            h_y(idx2 - start) = cm_tot[1] + (M12/M_tot)*r_out_new[1];
            h_z(idx2 - start) = cm_tot[2] + (M12/M_tot)*r_out_new[2];
            h_vx(idx2 - start) = cm_tot_v[0] + (M12/M_tot)*v_out_new[0];
            h_vy(idx2 - start) = cm_tot_v[1] + (M12/M_tot)*v_out_new[1];
            h_vz(idx2 - start) = cm_tot_v[2] + (M12/M_tot)*v_out_new[2];
        }

        mirror.push(p);

        for (int s : secular_triple_idx) {
            auto& dst = g_subsystem_mgr.active[s];
            dst.has_aux_vel = false;
        }
    }

    g_subsys_phase.add(SubsystemPhaseTimes::SEC_TRIPLE, phase_clock.lap());

    // --- LogH subsystems: batch integrate with per-subsystem restart (BIFROST §3.19) ---
    if (!logh_idx.empty()) {
        // Most expensive first (longest-processing-time order), so a long integration does not start last
        // while the other host threads idle: cost ~ N^2 x orbits per step
        auto cost = [&](int s) {
            const auto& sub = g_subsystem_mgr.active[s];
            double n_orb = sub.orb.period > 0.0 ? dt / sub.orb.period : 1.0;
            return (double)sub.n_bodies * sub.n_bodies * (1.0 + n_orb);
        };
        std::stable_sort(logh_idx.begin(), logh_idx.end(), [&](int a, int b) { return cost(a) > cost(b); });
        SubsystemManager logh_mgr;
        for (int s : logh_idx) {
            logh_mgr.active.push_back(g_subsystem_mgr.active[s]);
        }
        logh_mgr.rebuild_pid_to_idx(p);

        const int n_sub = (int)logh_mgr.active.size();
        const double eta_restart = g_collisional.eta_restart;
        constexpr int MAX_SUBDIV = 8;  // max 2^8 = 256 sub-steps

        // Save pre-integration state for subsystem members (needed for restart)
        // Reuse host mirrors from top of function (already up-to-date after secular evolution)
        struct SavedBody { double x, y, z, vx, vy, vz; };
        std::vector<std::vector<SavedBody>> saved_bodies(n_sub);
        std::vector<Subsystem> saved_subs(logh_mgr.active.begin(), logh_mgr.active.end());
        for (int si = 0; si < n_sub; si++) {
            const auto& sub = logh_mgr.active[si];
            saved_bodies[si].resize(sub.n_bodies);
            for (int b = 0; b < sub.n_bodies; b++) {
                int idx = g_subsystem_mgr.safe_pid_to_idx(sub.particle_ids[b]);
                if (idx < 0) continue;
                saved_bodies[si][b] = {h_x(idx - start), h_y(idx - start), h_z(idx - start),
                                       h_vx(idx - start), h_vy(idx - start), h_vz(idx - start)};
            }
        }

        // Initial batch integration
        g_logh_batch.pack(logh_mgr, p, G, dt);
        g_subsys_phase.add(SubsystemPhaseTimes::LOGH_PREP, phase_clock.lap());
        logh_integrate_batch_gpu(g_logh_batch, G, g_collisional.eta_gbs);
        g_subsys_phase.add(SubsystemPhaseTimes::LOGH_RUN, phase_clock.lap());

        // Check per-subsystem energy errors
        auto h_eerr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                           g_logh_batch.energy_error);
        std::vector<int> failed;
        for (int si = 0; si < n_sub; si++) {
            double dE = h_eerr(si);
            if (eta_restart > 0 && (dE > eta_restart || std::isnan(dE))) {
                failed.push_back(si);
            }
        }

        // Unpack all results (passed get final state; failed get best-effort temporarily)
        g_logh_batch.unpack(logh_mgr, p);
        g_subsys_phase.add(SubsystemPhaseTimes::LOGH_UNPACK, phase_clock.lap());

        // Accumulate LogH stats into step-level counters
        g_subsystem_mgr.step_logh_calls += n_sub;
        for (int si = 0; si < n_sub; si++) adv_check.advanced(logh_idx[si], g_logh_batch.t_reached[si], "LogH");
        g_subsystem_mgr.step_logh_gbs_total += g_logh_batch.stat_total_gbs;
        if (g_logh_batch.stat_worst_dE > g_subsystem_mgr.step_logh_worst_dE)
            g_subsystem_mgr.step_logh_worst_dE = g_logh_batch.stat_worst_dE;
        if (g_logh_batch.stat_best_dE < g_subsystem_mgr.step_logh_best_dE)
            g_subsystem_mgr.step_logh_best_dE = g_logh_batch.stat_best_dE;
        g_subsystem_mgr.step_logh_cap_hit += g_logh_batch.stat_cap_hit;

        // Retry failed subsystems with subdivided timestep (BIFROST §3.19)
        if (!failed.empty()) {
            g_subsystem_mgr.step_logh_restarts += (int)failed.size();
            std::printf("  LogH restart: %d/%d subs exceeded eta_restart=%.1e\n",
                        (int)failed.size(), n_sub, eta_restart);

            // Helper: restore failed subsystems' particle state from saved copy
            // Refresh host mirrors from device (LogH modified positions), then overwrite failed members
            auto restore_failed = [&]() {
                mirror.pull(p, 6);
                for (int si : failed) {
                    const auto& sub = saved_subs[si];
                    for (int b = 0; b < sub.n_bodies; b++) {
                        int idx = g_subsystem_mgr.safe_pid_to_idx(sub.particle_ids[b]);
                        if (idx < 0) continue;
                        h_x(idx - start)  = saved_bodies[si][b].x;
                        h_y(idx - start)  = saved_bodies[si][b].y;
                        h_z(idx - start)  = saved_bodies[si][b].z;
                        h_vx(idx - start) = saved_bodies[si][b].vx;
                        h_vy(idx - start) = saved_bodies[si][b].vy;
                        h_vz(idx - start) = saved_bodies[si][b].vz;
                    }
                    logh_mgr.active[si] = saved_subs[si];
                }
                mirror.push(p);
            };

            restore_failed();

            // Subdivision retry loop: 2, 4, 8, ... sub-steps
            for (int subdiv = 1; subdiv <= MAX_SUBDIV; subdiv++) {
                int n_steps = 1 << subdiv;
                double dt_sub = dt / n_steps;

                SubsystemManager retry_mgr;
                for (int si : failed) retry_mgr.active.push_back(logh_mgr.active[si]);
                retry_mgr.rebuild_pid_to_idx(p);

                bool all_passed = true;
                double worst_dE = 0;
                const bool last_try = (subdiv == MAX_SUBDIV);
                std::vector<double> t_sum(failed.size(), 0.0);

                for (int step = 0; step < n_steps; step++) {
                    SubsystemBatch retry_batch;
                    retry_batch.pack(retry_mgr, p, G, dt_sub);
                    logh_integrate_batch_gpu(retry_batch, G, g_collisional.eta_gbs);

                    auto h_e = Kokkos::create_mirror_view_and_copy(
                        Kokkos::HostSpace(), retry_batch.energy_error);
                    for (int r = 0; r < (int)retry_mgr.active.size(); r++) {
                        double dE = h_e(r);
                        if (dE > eta_restart || std::isnan(dE)) all_passed = false;
                        worst_dE = std::max(worst_dE, std::isnan(dE) ? 1e30 : dE);
                    }

                    retry_batch.unpack(retry_mgr, p);
                    for (size_t r = 0; r < failed.size(); r++) t_sum[r] += retry_batch.t_reached[r];

                    // No point continuing if a sub-step failed, except on the last try: there the result stands,
                    // and stopping early would leave the members short of dt (frozen for the rest of the step)
                    if (!all_passed && !last_try) break;
                }
                if (all_passed || last_try)
                    for (size_t r = 0; r < failed.size(); r++) adv_check.set_time(logh_idx[failed[r]], t_sum[r]);

                if (all_passed) {
                    std::printf("  LogH restart: OK with %d sub-steps (worst dE=%.2e)\n",
                                n_steps, worst_dE);
                    g_subsystem_mgr.step_logh_restart_ok += (int)failed.size();
                    // Copy final persistent state back to logh_mgr
                    for (size_t r = 0; r < failed.size(); r++)
                        logh_mgr.active[failed[r]] = retry_mgr.active[r];
                    failed.clear();
                    break;
                }

                std::printf("  LogH restart: %d sub-steps failed (worst dE=%.2e)\n",
                            n_steps, worst_dE);

                if (!last_try) {
                    restore_failed();
                } else {   // keep the persistent state (aux velocities, scales) consistent with the members
                    for (size_t r = 0; r < failed.size(); r++) logh_mgr.active[failed[r]] = retry_mgr.active[r];
                }
            }

            if (!failed.empty())
                std::printf("  LogH restart: GAVE UP after %d subdivisions\n", MAX_SUBDIV);
        }

        // PN-active subsystems: the global diagnostics are Newtonian, so book the change of the members'
        // Newtonian energy, momentum and angular momentum over this advance (Newtonian LogH conserves
        // them; what changed is the PN dynamics, plus GBS error at its tolerance)
        if (g_collisional.pn_order > 0 && g_collisional.c_code > 0.0) {
            mirror.pull(p, 6);
            for (int si = 0; si < n_sub; si++) {
                const auto& sub = saved_subs[si];
                int nb = sub.n_bodies;
                int idx[LOGH_MAX_BODIES];
                bool pn_active = false, ok = true;
                for (int b = 0; b < nb; b++) {
                    idx[b] = g_subsystem_mgr.safe_pid_to_idx(sub.particle_ids[b]);
                    if (idx[b] < 0) { ok = false; break; }
                    if (h_m(idx[b] - start) >= g_collisional.m_pn) pn_active = true;
                }
                if (!ok || !pn_active) continue;
                double E_pre = 0.0, E_post = 0.0;
                for (int b = 0; b < nb; b++) {
                    const auto& s0 = saved_bodies[si][b];
                    int i = idx[b] - start;
                    double m = h_m(i);
                    double r0[3] = {s0.x, s0.y, s0.z}, v0[3] = {s0.vx, s0.vy, s0.vz};
                    double r1[3] = {h_x(i), h_y(i), h_z(i)}, v1[3] = {h_vx(i), h_vy(i), h_vz(i)};
                    E_pre  += 0.5 * m * (v0[0]*v0[0] + v0[1]*v0[1] + v0[2]*v0[2]);
                    E_post += 0.5 * m * (v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]);
                    g_energy_books.sink_PL(m, r0, v0);
                    g_energy_books.sink_PL(m, r1, v1, -1.0);
                    for (int c = b + 1; c < nb; c++) {
                        const auto& s1 = saved_bodies[si][c];
                        int j = idx[c] - start;
                        double Gmm = G * m * h_m(j);
                        E_pre  -= Gmm / std::sqrt((s0.x-s1.x)*(s0.x-s1.x) + (s0.y-s1.y)*(s0.y-s1.y) + (s0.z-s1.z)*(s0.z-s1.z));
                        E_post -= Gmm / std::sqrt((h_x(i)-h_x(j))*(h_x(i)-h_x(j)) + (h_y(i)-h_y(j))*(h_y(i)-h_y(j)) +
                                                  (h_z(i)-h_z(j))*(h_z(i)-h_z(j)));
                    }
                }
                g_energy_books.E_pn += E_pre - E_post;
            }
        }

        // Propagate persistent state back from logh_mgr to g_subsystem_mgr
        // (B is always recomputed fresh — only aux_vel and scales need persistence)
        for (int i = 0; i < n_sub; i++) {
            auto& dst = g_subsystem_mgr.active[logh_idx[i]];
            const auto& src = logh_mgr.active[i];
            dst.has_aux_vel = src.has_aux_vel;
            if (src.has_aux_vel) {
                std::memcpy(dst.aux_vel_phys, src.aux_vel_phys,
                            LOGH_MAX_BODIES * 3 * sizeof(double));
            }
            dst.V_scale = src.V_scale;
            dst.R_scale = src.R_scale;
            dst.T_scale = src.T_scale;
            dst.E_scale = src.E_scale;
            dst.scales_set = src.scales_set;
        }
        g_subsys_phase.add(SubsystemPhaseTimes::LOGH_RETRY, phase_clock.lap());
    }
#ifdef USE_MPI
    if (distributed_subs) exchange_subsystem_results(p, eligible, books_before);
#else
    (void)distributed_subs;
#endif
    adv_check.verify(start, count);
}
#endif

void fsi_step(Particles& p, int start, int count,
              double G, double soft2, double dt, bool forces_valid,
              bool force_direct_n2) {
    g_subset_hist.record(SubsetHistogram::FSI_STEP, count);
    bool tree_mode = !force_direct_n2 && (use_tree && count >= tree_threshold);
    bool tree_built = false;

    // U kick (dt/6)
    if (forces_valid) {
        g_profiler.start();
        kick_velocities(p, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else if (tree_mode) {
        g_profiler.start();
        g_octree.n_leaf = tree_n_leaf;
        g_octree.n_group = tree_n_group;
        g_octree.build(p, start, count, tree_theta);
        tree_self_acceleration(p, g_octree, start, count, G, soft2);
        tree_momentum_correction(p, start, count);
        tree_built = true;
        g_profiler.stop(StepProfiler::SELF_FORCE);
        g_profiler.start();
        kick_velocities(p, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else {
        g_profiler.start();
        self_acceleration(p, start, count, G, soft2);
        g_profiler.stop(StepProfiler::SELF_FORCE);
        g_profiler.start();
        kick_velocities(p, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // H_S (dt/2) then T' drift (dt/2) — BIFROST Algorithm 1 ordering
    g_profiler.start();
#ifdef ENABLE_COLLISIONAL
    if (p.collisional && !g_subsystem_mgr.active.empty()) {
        advance_subsystems_in_range(p, start, count, G, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::SUBSYSTEM_INTEG);
        g_profiler.start();
        drift_field_only(p, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else
#endif
    {
        drift(p, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // U-tilde kick (2dt/3)
    if (tree_mode) {
        g_profiler.start();
        if (!tree_built) {
            g_octree.n_leaf = tree_n_leaf;
            g_octree.n_group = tree_n_group;
            g_octree.build(p, start, count, tree_theta);
            tree_built = true;
        } else {
            g_octree.refresh_positions(p);
        }
        tree_self_acceleration(p, g_octree, start, count, G, soft2);
        tree_momentum_correction(p, start, count);
        g_profiler.stop(StepProfiler::SELF_FORCE);
        g_profiler.start();
        g_octree.compute_cell_accelerations(p);   // FSI gradient term, as in fsi_step_distributed
        tree_self_gradient_correction(p, g_octree, G, soft2, dt);
        tree_momentum_correction(p, start, count);
        g_profiler.stop(StepProfiler::SELF_GRAD);
        g_profiler.start();
        kick_velocities(p, start, count, dt * FSI_MOD_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else {
        g_profiler.start();
        self_acceleration(p, start, count, G, soft2);
        g_profiler.stop(StepProfiler::SELF_FORCE);
        g_profiler.start();
        self_gradient_correction(p, start, count, G, soft2, dt);
        g_profiler.stop(StepProfiler::SELF_GRAD);
        g_profiler.start();
        kick_velocities(p, start, count, dt * FSI_MOD_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // H_S (dt/2) then T' drift (dt/2) — BIFROST Algorithm 1 ordering
    g_profiler.start();
#ifdef ENABLE_COLLISIONAL
    if (p.collisional && !g_subsystem_mgr.active.empty()) {
        advance_subsystems_in_range(p, start, count, G, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::SUBSYSTEM_INTEG);
        g_profiler.start();
        drift_field_only(p, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else
#endif
    {
        drift(p, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // U kick (dt/6)
    if (tree_mode) {
        g_profiler.start();
        g_octree.refresh_positions(p);
        tree_self_acceleration(p, g_octree, start, count, G, soft2);
        tree_momentum_correction(p, start, count);
        g_profiler.stop(StepProfiler::SELF_FORCE);
        g_profiler.start();
        kick_velocities(p, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else {
        g_profiler.start();
        self_acceleration(p, start, count, G, soft2);
        g_profiler.stop(StepProfiler::SELF_FORCE);
        g_profiler.start();
        kick_velocities(p, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }
}

// ==========================================================================
// Dynamic pivot-based partitioning for HHS (per paper's Algorithm 4)
// Partitions particles [start, start+count) into:
//   - slow: dt_adapt >= pivot (at beginning)
//   - fast: dt_adapt < pivot (at end)
// Returns the count of slow particles
//
// IMPORTANT: This must be DETERMINISTIC across MPI ranks! We use prefix sums
// instead of atomics to ensure all ranks produce identical permutations.
// ==========================================================================
static int partition_by_pivot(Particles& p, int start, int count, double pivot) {
    if (count <= 1) return count;

    RawView dt_f{p.dt_adapt.data()};
    auto perm = p.scratch_i;
    auto is_slow = p.scratch_i2;  // temporary: 1 if slow, 0 if fast

    // Fused mark + exclusive prefix sum in single kernel
    // is_slow(ii) = 1 if slow, 0 if fast
    // slow_prefix(ii) = number of slow particles with index < ii
    auto slow_prefix = p.scratch_i3;
    Kokkos::parallel_scan("mark_prefix_slow", count,
        KOKKOS_LAMBDA(int ii, int& partial_sum, bool is_final) {
            int val = (dt_f(start + ii) >= pivot) ? 1 : 0;
            if (is_final) {
                is_slow(ii) = val;
                slow_prefix(ii) = partial_sum;
            }
            partial_sum += val;
        });
    Kokkos::fence();

    // Get total slow count: deep_copy last element instead of launching a kernel
    int last_prefix = 0, last_is_slow = 0;
    Kokkos::deep_copy(last_prefix, Kokkos::subview(slow_prefix, count - 1));
    Kokkos::deep_copy(last_is_slow, Kokkos::subview(is_slow, count - 1));
    int slow_count = last_prefix + last_is_slow;

    if (slow_count == 0 || slow_count == count) {
        return slow_count;
    }

    // Step 3: Build permutation deterministically
    // - Slow particle at position ii goes to slow_prefix(ii)
    // - Fast particle at position ii goes to slow_count + (ii - slow_prefix(ii) - (1-is_slow(ii)))
    //   = slow_count + (ii - slow_prefix(ii)) for fast particles
    int sc = slow_count;
    Kokkos::parallel_for("build_perm_deterministic", count, KOKKOS_LAMBDA(int ii) {
        int dest;
        if (is_slow(ii)) {
            dest = slow_prefix(ii);
        } else {
            // Fast particles: position among fast = ii - slow_prefix(ii)
            // (since slow_prefix(ii) counts slow particles before ii)
            dest = sc + (ii - slow_prefix(ii));
        }
        perm(dest) = start + ii;
    });

    p.apply_permutation(start, count);

    return slow_count;
}

// Forward declaration: unified HHS-FSI recursion (defined after all helper functions).
// Returns the number of particles that mergers removed from [start, start+count).
static int hhs_fsi_recursive(Particles& p, DistributedParticles* dp,
    int start, int count, double pivot_dt,
    double G, double soft2, double dt_min, double eta,
    bool dt_valid, bool global_tree_available = false, int depth = 0);

// After compact_remove_zero_mass: the cached pid map points at old slots, and in MPI mode the
// per-rank slices still cover the old particle count.
static void sync_after_removal(Particles& p, DistributedParticles* dp) {
#ifdef ENABLE_COLLISIONAL
    if (p.collisional) g_subsystem_mgr.rebuild_pid_to_idx(p);
#endif
    if (dp) dp->sync_count();
}

void hhs_fsi_step_dynamic(Particles& p, int start, int count, double pivot_dt,
                          double G, double soft2, double dt_min,
                          double eta) {
#ifdef ENABLE_COLLISIONAL
    if (p.collisional) g_subsystem_mgr.reset_step_diagnostics();
    // Global PN1.0 half-kicks around the FSI step (BIFROST Section 3.7), as in hhs_fsi_step_distributed
    bool do_global_pn = p.collisional && (g_collisional.pn_order & 1) && g_collisional.c_code > 0.0;
    if (do_global_pn)
        global_pn1_kick(p, start, count, G, g_collisional.c_code, g_collisional.m_pn, pivot_dt * 0.5);
#endif
    if (hhs_cc) g_hhs_counters.reset();
    int n_removed = hhs_fsi_recursive(p, nullptr, start, count, pivot_dt, G, soft2, dt_min, eta, false);
    if (hhs_cc) {   // component statistics per depth
        std::printf("  HHS_CC components:");
        for (int d = 0; d < HHSCounters::MAX_DEPTH; d++)
            if (g_hhs_counters.cc_calls[d] > 0)
                std::printf(" d%d %lld x %.1f (largest %.0f)", d, g_hhs_counters.cc_calls[d],
                            (double)g_hhs_counters.cc_comps[d] / g_hhs_counters.cc_calls[d],
                            (double)g_hhs_counters.cc_largest[d] / g_hhs_counters.cc_calls[d]);
        std::printf("\n");
    }
#ifdef ENABLE_COLLISIONAL
    if (do_global_pn)
        global_pn1_kick(p, start, count - n_removed, G, g_collisional.c_code, g_collisional.m_pn, pivot_dt * 0.5);
    // Per-step subsystem summary, as hhs_fsi_step_distributed prints it (tests read its counters)
    if (p.collisional) g_subsystem_mgr.print_step_summary();
    if (p.collisional && g_subsys_phase.any()) g_subsys_phase.report_and_reset();
#else
    (void)n_removed;
#endif
}

// =============================================================================
// Distributed (Multi-GPU) HHS Integration
// =============================================================================
// NEW DESIGN: Scatter-compute-gather paradigm (matching FROST paper)
//
// All ranks maintain replicated particle data. The HHS recursion runs
// IDENTICALLY on all ranks (partitioning, drifts, kicks). Only force
// computation is distributed:
//   1. Each rank computes forces for its slice of particles
//   2. Allgather accelerations after each force computation
//
// This achieves O(N) communication per step regardless of HHS depth.
// =============================================================================

// Kick velocities using pre-computed accelerations
void kick_velocities(Particles& p, int start, int count, double dt) {
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView cvx{p.cvx.data()}, cvy{p.cvy.data()}, cvz{p.cvz.data()};

    // Compensated: a plain v += a dt rounds at ~eps|v| per kick and sets the energy floor
    small_for("kick", count, KOKKOS_LAMBDA(int ii) {
        int i = start + ii;
        kahan_add(pvx(i), cvx(i), pax(i) * dt);
        kahan_add(pvy(i), cvy(i), pay(i) * dt);
        kahan_add(pvz(i), cvz(i), paz(i) * dt);
    });
}

// =============================================================================
// Distributed force computation: scatter-compute-gather with tiled GPU kernels
// Each rank computes forces for particles in its slice FROM ALL particles
// in the working set [start, start+count). Then allgather accelerations.
// Uses shared memory tiling per FROST paper.
// =============================================================================
static void self_acceleration_distributed(DistributedParticles& dp, int start, int count,
                                          double G, double soft2) {
    if (count <= 0) return;


    int slice_start, slice_count;
    dp.get_force_slice(start, count, slice_start, slice_count);
    if (slice_count == 0) {
        dp.allgather_accelerations_subset(start, count);
        return;
    }

    acceleration_range(dp.local, slice_start, slice_count, start, count, G, soft2);
    Kokkos::fence();

    if (use_nonblocking_mpi) {
        dp.iallgather_accelerations_subset(start, count, 0);
        dp.wait_and_unpack_accelerations(start, count, 0);
    } else {
        dp.allgather_accelerations_subset(start, count);
    }
}

// =============================================================================
// Distributed gradient correction: computes gradient correction for my slice,
// applies it to accelerations, then allgather corrected accelerations.
// =============================================================================
static void self_gradient_correction_distributed(DistributedParticles& dp, int start, int count,
                                                  double G, double soft2, double grad_dt) {
    if (count <= 0) return;

    // Snapshot pre-correction accelerations for the full range.
    // The gradient kernel reads j-particle accelerations while writing corrected
    // i-particle accelerations to the same array — without a snapshot, j-particles
    // in the local MPI slice can read partially-corrected values (data race),
    // breaking the antisymmetry corr_ij = -corr_ji and violating momentum conservation.
    static Kokkos::View<double*> ax_save("ax_save_grad", 0);
    static Kokkos::View<double*> ay_save("ay_save_grad", 0);
    static Kokkos::View<double*> az_save("az_save_grad", 0);
    static bool released_at_finalize = (Kokkos::push_finalize_hook([] {
        ax_save = {}; ay_save = {}; az_save = {};
    }), true);
    (void)released_at_finalize;
    if ((int)ax_save.extent(0) < count) {
        Kokkos::realloc(ax_save, count);
        Kokkos::realloc(ay_save, count);
        Kokkos::realloc(az_save, count);
    }
    Kokkos::deep_copy(Kokkos::subview(ax_save, Kokkos::make_pair(0, count)),
                      Kokkos::subview(dp.local.ax, Kokkos::make_pair(start, start + count)));
    Kokkos::deep_copy(Kokkos::subview(ay_save, Kokkos::make_pair(0, count)),
                      Kokkos::subview(dp.local.ay, Kokkos::make_pair(start, start + count)));
    Kokkos::deep_copy(Kokkos::subview(az_save, Kokkos::make_pair(0, count)),
                      Kokkos::subview(dp.local.az, Kokkos::make_pair(start, start + count)));

    int slice_start, slice_count;
    dp.get_force_slice(start, count, slice_start, slice_count);

    if (slice_count > 0) {
        GradSnapshot snap{ax_save.data(), ay_save.data(), az_save.data(), start};
        gradient_correction_range(dp.local, slice_start, slice_count, start, count, snap, snap, G, soft2, grad_dt,
                                  false);   // grad_mag below, from the gathered accelerations
        Kokkos::fence();
    }

    // Allgather corrected accelerations
    if (use_nonblocking_mpi) {
        dp.iallgather_accelerations_subset(start, count, 0);
        dp.wait_and_unpack_accelerations(start, count, 0);
    } else {
        dp.allgather_accelerations_subset(start, count);
    }

#ifdef ENABLE_COLLISIONAL
    // Compute grad_mag from acceleration difference — identical on all ranks
    if (dp.local.collisional) {
        auto ax_v = dp.local.ax;
        auto ay_v = dp.local.ay;
        auto az_v = dp.local.az;
        auto gm = dp.local.grad_mag;
        double inv_dt2 = 1.0 / (grad_dt * grad_dt);
        int s = start;
        auto axs = ax_save;
        auto ays = ay_save;
        auto azs = az_save;
        Kokkos::parallel_for("compute_grad_mag_dist", count, KOKKOS_LAMBDA(int ii) {
            double dx = ax_v(s + ii) - axs(ii);
            double dy = ay_v(s + ii) - ays(ii);
            double dz = az_v(s + ii) - azs(ii);
            gm(s + ii) = Kokkos::sqrt(dx*dx + dy*dy + dz*dz) * inv_dt2;
        });
        Kokkos::fence();
    }
#endif
}


// =============================================================================
// Tree MPI distribution helpers: divide tree groups across MPI ranks
// Each rank computes forces for its subset of groups, then allreduce(SUM)
// =============================================================================
static void get_tree_group_range(const Octree& tree, int& g_offset, int& g_count) {
    int rank = MPIComm::rank();
    int size = MPIComm::size();
    int total_groups = tree.n_groups;
    g_offset = rank * total_groups / size;
    int g_end = (rank + 1) * total_groups / size;
    g_count = g_end - g_offset;
}

static void zero_accelerations(Particles& p, int start, int count) {
    auto ax = p.ax; auto ay = p.ay; auto az = p.az;
    small_for("zero_acc", count, KOKKOS_LAMBDA(int ii) {
        int i = start + ii;
        ax(i) = 0.0; ay(i) = 0.0; az(i) = 0.0;
    });
}

static void init_timesteps_max(Particles& p, int start, int count) {
    auto dt_f = p.dt_adapt;
    small_for("init_dt_max", count, KOKKOS_LAMBDA(int ii) {
        dt_f(start + ii) = 1e30;
    });
}

// =============================================================================
// Distributed FSI step using scatter-compute-gather
// All ranks execute identical code; only force computation is distributed
// Uses fused force+kick kernels to reduce kernel launches and memory traffic
//
// tree_prebuilt: if true, g_octree is already built for [start, count) —
//   skip first tree build (reuse from timestep computation or previous call)
// global_tree_subset: -1 = build own tree (default), 0 or 1 = use global tree
//   with subset monopoles (0=slow, 1=fast). No MPI allreduce needed.
// =============================================================================
static void fsi_step_distributed(DistributedParticles& dp, int start, int count,
                                 double G, double soft2, double dt,
                                 bool forces_valid = false,
                                 bool tree_prebuilt = false,
                                 int global_tree_subset = -1) {
    if (count <= 0) return;

    // Global tree mode: use subset monopoles from the once-built global tree
    if (global_tree_subset >= 0) {
        long long self_pairs = (long long)count * count;

        // Step 1: force + kick (dt/6)
        if (forces_valid) {
            g_profiler.start();
            kick_velocities(dp.local, start, count, dt * FSI_KICK);
            g_profiler.stop(StepProfiler::DRIFT_KICK);
        } else {
            g_profiler.start();
            tree_global_self_acceleration(dp.local, g_octree, global_tree_subset, G, soft2);
            tree_momentum_correction(dp.local, start, count);
            g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
            g_profiler.start();
            kick_velocities(dp.local, start, count, dt * FSI_KICK);
            g_profiler.stop(StepProfiler::DRIFT_KICK);
        }

        // H_S (dt/2) then T' drift (dt/2) — BIFROST Algorithm 1 ordering
        g_profiler.start();
#ifdef ENABLE_COLLISIONAL
        if (dp.local.collisional && !g_subsystem_mgr.active.empty()) {
            advance_subsystems_in_range(dp.local, start, count, G, dt * FSI_DRIFT);
            g_profiler.stop(StepProfiler::SUBSYSTEM_INTEG);
            g_profiler.start();
            drift_field_only(dp.local, start, count, dt * FSI_DRIFT);
            g_profiler.stop(StepProfiler::DRIFT_KICK);
        } else
#endif
        {
            drift(dp.local, start, count, dt * FSI_DRIFT);
            g_profiler.stop(StepProfiler::DRIFT_KICK);
        }

        // Refresh positions in global tree after drift
        g_profiler.start();
        g_octree.refresh_positions(dp.local);
        g_profiler.stop(StepProfiler::TREE_MAINT);

        // Step 3: force + gradient + kick (2dt/3)
        g_profiler.start();
        tree_global_self_acceleration(dp.local, g_octree, global_tree_subset, G, soft2);
        tree_momentum_correction(dp.local, start, count);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        g_profiler.start();
        g_octree.compute_dual_cell_accelerations(dp.local);
        tree_global_self_gradient_correction(dp.local, g_octree, global_tree_subset, G, soft2, dt);
        tree_momentum_correction(dp.local, start, count);
        g_profiler.stop(StepProfiler::SELF_GRAD, self_pairs);
        g_profiler.start();
        kick_velocities(dp.local, start, count, dt * FSI_MOD_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);

        // H_S (dt/2) then T' drift (dt/2) — BIFROST Algorithm 1 ordering
        g_profiler.start();
#ifdef ENABLE_COLLISIONAL
        if (dp.local.collisional && !g_subsystem_mgr.active.empty()) {
            advance_subsystems_in_range(dp.local, start, count, G, dt * FSI_DRIFT);
            g_profiler.stop(StepProfiler::SUBSYSTEM_INTEG);
            g_profiler.start();
            drift_field_only(dp.local, start, count, dt * FSI_DRIFT);
            g_profiler.stop(StepProfiler::DRIFT_KICK);
        } else
#endif
        {
            drift(dp.local, start, count, dt * FSI_DRIFT);
            g_profiler.stop(StepProfiler::DRIFT_KICK);
        }

        // Refresh positions in global tree after drift
        g_profiler.start();
        g_octree.refresh_positions(dp.local);
        g_profiler.stop(StepProfiler::TREE_MAINT);

        // Step 5: force + kick (dt/6)
        g_profiler.start();
        tree_global_self_acceleration(dp.local, g_octree, global_tree_subset, G, soft2);
        tree_momentum_correction(dp.local, start, count);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        g_profiler.start();
        kick_velocities(dp.local, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);

        return;
    }

    bool tree_mode = use_tree && count >= tree_threshold;
    long long self_pairs = (long long)count * count;

    // Sync packed positions for vectorized kernel loads
    g_profiler.start();
    dp.local.pack_pos_range(start, count);
    g_profiler.stop(StepProfiler::PACK_DATA);

    // U kick (dt/6) - force then kick (separate, matching FROST)
    if (forces_valid) {
        g_profiler.start();
        kick_velocities(dp.local, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else if (tree_mode) {
        g_profiler.start();
        if (!tree_prebuilt) {
            g_octree.n_leaf = tree_n_leaf;
            g_octree.n_group = tree_n_group;
            g_octree.build(dp.local, start, count, tree_theta);
            tree_prebuilt = true;
        }
        int my_g_off, my_g_cnt;
        get_tree_group_range(g_octree, my_g_off, my_g_cnt);
        zero_accelerations(dp.local, start, count);
        tree_self_acceleration(dp.local, g_octree, start, count, G, soft2, my_g_off, my_g_cnt);
        dp.allreduce_accelerations_subset(start, count);
        tree_momentum_correction(dp.local, start, count);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        g_profiler.start();
        kick_velocities(dp.local, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else {
        g_profiler.start();
        self_acceleration_distributed(dp, start, count, G, soft2);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        g_profiler.start();
        kick_velocities(dp.local, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // H_S (dt/2) then T' drift (dt/2) — BIFROST Algorithm 1 ordering
    g_profiler.start();
#ifdef ENABLE_COLLISIONAL
    if (dp.local.collisional && !g_subsystem_mgr.active.empty()) {
        advance_subsystems_in_range(dp.local, start, count, G, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::SUBSYSTEM_INTEG);
        g_profiler.start();
        drift_field_only(dp.local, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else
#endif
    {
        drift(dp.local, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // Re-sync packed positions after drift
    g_profiler.start();
    dp.local.pack_pos_range(start, count);
    g_profiler.stop(StepProfiler::PACK_DATA);

    // U-tilde kick (2dt/3) with gradient correction
    if (tree_mode) {
        // Tree mode: force + gradient correction + kick
        // Refresh positions (O(N)) instead of rebuilding tree (O(N log N))
        g_profiler.start();
        if (!tree_prebuilt) {
            g_octree.n_leaf = tree_n_leaf;
            g_octree.n_group = tree_n_group;
            g_octree.build(dp.local, start, count, tree_theta);
            tree_prebuilt = true;
        } else {
            g_octree.refresh_positions(dp.local);
        }
        int my_g_off, my_g_cnt;
        get_tree_group_range(g_octree, my_g_off, my_g_cnt);
        zero_accelerations(dp.local, start, count);
        tree_self_acceleration(dp.local, g_octree, start, count, G, soft2, my_g_off, my_g_cnt);
        dp.allreduce_accelerations_subset(start, count);
        tree_momentum_correction(dp.local, start, count);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        // Gradient correction via tree: all ranks compute (not distributed)
        // Forces must be correct (allreduced) before gradient computation
        g_profiler.start();
        g_octree.compute_cell_accelerations(dp.local);
        tree_self_gradient_correction(dp.local, g_octree, G, soft2, dt);
        tree_momentum_correction(dp.local, start, count);
        g_profiler.stop(StepProfiler::SELF_GRAD, self_pairs);
        g_profiler.start();
        kick_velocities(dp.local, start, count, dt * FSI_MOD_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else {
        // Separate: force → allgather → gradient correction → allgather → kick
        g_profiler.start();
        self_acceleration_distributed(dp, start, count, G, soft2);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        g_profiler.start();
        self_gradient_correction_distributed(dp, start, count, G, soft2, dt);
        g_profiler.stop(StepProfiler::SELF_GRAD, self_pairs);
        g_profiler.start();
        kick_velocities(dp.local, start, count, dt * FSI_MOD_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // H_S (dt/2) then T' drift (dt/2) — BIFROST Algorithm 1 ordering
    g_profiler.start();
#ifdef ENABLE_COLLISIONAL
    if (dp.local.collisional && !g_subsystem_mgr.active.empty()) {
        advance_subsystems_in_range(dp.local, start, count, G, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::SUBSYSTEM_INTEG);
        g_profiler.start();
        drift_field_only(dp.local, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else
#endif
    {
        drift(dp.local, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // Re-sync packed positions after drift
    g_profiler.start();
    dp.local.pack_pos_range(start, count);
    g_profiler.stop(StepProfiler::PACK_DATA);

    // U kick (dt/6) - force then kick
    if (tree_mode) {
        g_profiler.start();
        g_octree.refresh_positions(dp.local);
        int my_g_off, my_g_cnt;
        get_tree_group_range(g_octree, my_g_off, my_g_cnt);
        zero_accelerations(dp.local, start, count);
        tree_self_acceleration(dp.local, g_octree, start, count, G, soft2, my_g_off, my_g_cnt);
        dp.allreduce_accelerations_subset(start, count);
        tree_momentum_correction(dp.local, start, count);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        g_profiler.start();
        kick_velocities(dp.local, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else {
        g_profiler.start();
        self_acceleration_distributed(dp, start, count, G, soft2);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        g_profiler.start();
        kick_velocities(dp.local, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }
}

// =============================================================================
// Distributed cross-acceleration between slow and fast particles
// Tiled GPU kernel per FROST paper
// =============================================================================
static void cross_acceleration_distributed(DistributedParticles& dp,
                                           int slow_start, int slow_count,
                                           int fast_start, int fast_count,
                                           double G, double soft2) {
    if (slow_count <= 0 || fast_count <= 0) return;


    // Slow particles from fast, fast from slow: each rank its slice of the sinks
    int slice_start, slice_count;
    dp.get_force_slice(slow_start, slow_count, slice_start, slice_count);
    acceleration_range(dp.local, slice_start, slice_count, fast_start, fast_count, G, soft2);
    int slice_start_f, slice_count_f;
    dp.get_force_slice(fast_start, fast_count, slice_start_f, slice_count_f);
    acceleration_range(dp.local, slice_start_f, slice_count_f, slow_start, slow_count, G, soft2);
    Kokkos::fence();

    // Must use separate allgathers: each rank computed its slice of slow and
    // its slice of fast independently, so the distributions don't match a combined range
    if (use_nonblocking_mpi) {
        dp.iallgather_accelerations_subset(slow_start, slow_count, 0);
        dp.iallgather_accelerations_subset(fast_start, fast_count, 1);
        dp.wait_and_unpack_accelerations(slow_start, slow_count, 0);
        dp.wait_and_unpack_accelerations(fast_start, fast_count, 1);
    } else {
        dp.allgather_accelerations_subset(slow_start, slow_count);
        dp.allgather_accelerations_subset(fast_start, fast_count);
    }
}

// Cross-acceleration with gradient correction - tiled GPU kernel
static void cross_acceleration_and_gradient_distributed(DistributedParticles& dp,
                                                        int slow_start, int slow_count,
                                                        int fast_start, int fast_count,
                                                        double G, double soft2, double dt) {
    // First compute cross-accelerations (already tiled)
    cross_acceleration_distributed(dp, slow_start, slow_count,
                                   fast_start, fast_count, G, soft2);


    RawView pax{dp.local.ax.data()}, pay{dp.local.ay.data()}, paz{dp.local.az.data()};

    // Snapshot accelerations for both slow and fast ranges BEFORE gradient kernels.
    // The slow-from-fast kernel modifies slow accelerations; without a snapshot the
    // fast-from-slow kernel would read corrected (not original) slow values, breaking
    // the antisymmetry corr_ij = -corr_ji and violating momentum conservation.
    static Kokkos::View<double*> ax_snap_s("ax_snap_cross_s", 0);
    static Kokkos::View<double*> ay_snap_s("ay_snap_cross_s", 0);
    static Kokkos::View<double*> az_snap_s("az_snap_cross_s", 0);
    static Kokkos::View<double*> ax_snap_f("ax_snap_cross_f", 0);
    static Kokkos::View<double*> ay_snap_f("ay_snap_cross_f", 0);
    static Kokkos::View<double*> az_snap_f("az_snap_cross_f", 0);
    static bool released_at_finalize = (Kokkos::push_finalize_hook([] {
        ax_snap_s = {}; ay_snap_s = {}; az_snap_s = {}; ax_snap_f = {}; ay_snap_f = {}; az_snap_f = {};
    }), true);
    (void)released_at_finalize;
    if ((int)ax_snap_s.extent(0) < slow_count) {
        Kokkos::realloc(ax_snap_s, slow_count);
        Kokkos::realloc(ay_snap_s, slow_count);
        Kokkos::realloc(az_snap_s, slow_count);
    }
    if ((int)ax_snap_f.extent(0) < fast_count) {
        Kokkos::realloc(ax_snap_f, fast_count);
        Kokkos::realloc(ay_snap_f, fast_count);
        Kokkos::realloc(az_snap_f, fast_count);
    }
    RawView sax_s{ax_snap_s.data()}, say_s{ay_snap_s.data()}, saz_s{az_snap_s.data()};
    RawView sax_f{ax_snap_f.data()}, say_f{ay_snap_f.data()}, saz_f{az_snap_f.data()};
    small_for("snap_cross_slow", slow_count, KOKKOS_LAMBDA(int ii) {
        sax_s(ii) = pax(slow_start + ii);
        say_s(ii) = pay(slow_start + ii);
        saz_s(ii) = paz(slow_start + ii);
    });
    small_for("snap_cross_fast", fast_count, KOKKOS_LAMBDA(int ii) {
        sax_f(ii) = pax(fast_start + ii);
        say_f(ii) = pay(fast_start + ii);
        saz_f(ii) = paz(fast_start + ii);
    });

    GradSnapshot snap_s{ax_snap_s.data(), ay_snap_s.data(), az_snap_s.data(), slow_start};
    GradSnapshot snap_f{ax_snap_f.data(), ay_snap_f.data(), az_snap_f.data(), fast_start};
    int slice_start, slice_count;
    dp.get_force_slice(slow_start, slow_count, slice_start, slice_count);
    gradient_correction_range(dp.local, slice_start, slice_count, fast_start, fast_count, snap_s, snap_f,
                              G, soft2, dt, false);
    int slice_start_f, slice_count_f;
    dp.get_force_slice(fast_start, fast_count, slice_start_f, slice_count_f);
    gradient_correction_range(dp.local, slice_start_f, slice_count_f, slow_start, slow_count, snap_f, snap_s,
                              G, soft2, dt, false);
    Kokkos::fence();

    // Must use separate allgathers: force slices were computed independently
    if (use_nonblocking_mpi) {
        dp.iallgather_accelerations_subset(slow_start, slow_count, 0);
        dp.iallgather_accelerations_subset(fast_start, fast_count, 1);
        dp.wait_and_unpack_accelerations(slow_start, slow_count, 0);
        dp.wait_and_unpack_accelerations(fast_start, fast_count, 1);
    } else {
        dp.allgather_accelerations_subset(slow_start, slow_count);
        dp.allgather_accelerations_subset(fast_start, fast_count);
    }
}

// Distributed adaptive timestep computation - tiled GPU kernel
static void compute_timestep_distributed(DistributedParticles& dp, int start, int count,
                                      double G, double soft2, double eta) {
    (void)eta;
    if (count <= 1) return;


    int slice_start, slice_count;
    dp.get_force_slice(start, count, slice_start, slice_count);

    if (slice_count == 0) {
        dp.allgather_timestep_subset(start, count);
        return;
    }

    timestep_range(dp.local, slice_start, slice_count, start, count, G, soft2, eta);
#ifdef ENABLE_COLLISIONAL
    bool coll = dp.local.collisional;
#endif
    Kokkos::fence();

    // Allgather dt_adapt values (uses GPU-aware MPI when available)
    dp.allgather_timestep_subset(start, count);
#ifdef ENABLE_COLLISIONAL
    if (coll) dp.allgather_accelerations_subset(start, count);
#endif
}

// CPU serial path: CpuSubset + all cpu_* functions in integrator_cpu.cpp
// (hhs_fsi_recursive_cpu declared in integrator_cpu.h)
// hhs_fsi_recursive_local removed — merged into hhs_fsi_recursive

// =============================================================================
// Tree-based cross-force using global tree with dual monopoles
// One global tree (g_octree) built at top of step; dual monopoles recomputed
// at each HHS level after partition. No separate tree builds needed.
// =============================================================================

// Cross-force via global tree: both directions (slow←fast, fast←slow)
static void cross_acceleration_global_tree(Particles& p,
                                            int slow_start, int slow_count,
                                            int fast_start, int fast_count,
                                            double G, double soft2) {
    (void)fast_start;
    if (slow_count <= 0 || fast_count <= 0) return;

    // slow ← fast (source_flag=1 means fast sources, target=slow)
    tree_global_cross_acceleration(p, g_octree, 1, G, soft2);
    // fast ← slow (source_flag=0 means slow sources, target=fast)
    tree_global_cross_acceleration(p, g_octree, 0, G, soft2);

    // Momentum correction on combined range
    tree_momentum_correction(p, slow_start, slow_count + fast_count);
}

// Cross-force + gradient via global tree
static void cross_acceleration_and_gradient_global_tree(Particles& p,
                                                         int slow_start, int slow_count,
                                                         int fast_start, int fast_count,
                                                         double G, double soft2, double dt) {
    (void)fast_start;
    if (slow_count <= 0 || fast_count <= 0) return;

    // Cross-forces (overwrites ax/ay/az for both subsets)
    tree_global_cross_acceleration(p, g_octree, 1, G, soft2);
    tree_global_cross_acceleration(p, g_octree, 0, G, soft2);
    tree_momentum_correction(p, slow_start, slow_count + fast_count);

    // Compute dual cell accelerations for gradient correction
    g_octree.compute_dual_cell_accelerations(p);

    // Gradient correction: slow from fast, fast from slow
    tree_global_cross_gradient_correction(p, g_octree, 1, G, soft2, dt);
    tree_global_cross_gradient_correction(p, g_octree, 0, G, soft2, dt);
    tree_momentum_correction(p, slow_start, slow_count + fast_count);
}

// =============================================================================
// HHS with connected components (HHS_CC, Janes et al. 2014): the fast set F splits into components (pair time
// step below the pivot, or subsystem partners, connect). Interactions S<->F and between components are kicked at
// this level (with the gradient term in the middle kick); each component recurses alone, so the deep levels become
// independent pieces of work. Same kick-drift structure as Algorithm 4.
// =============================================================================
static int hhs_cc_step(Particles& p, DistributedParticles* dp, int slow_start, int slow_count, int fast_start,
                       std::vector<int> comps, double pivot_dt, double G, double soft2, double dt_min, double eta,
                       int depth) {
    static Kokkos::View<int*> cid("cc_cid", 0);
    static Kokkos::View<double*> snx("cc_snx", 0), sny("cc_sny", 0), snz("cc_snz", 0);
    static bool hook = (Kokkos::push_finalize_hook([] { cid = Kokkos::View<int*>(); snx = sny = snz = Kokkos::View<double*>(); }), true);
    (void)hook;
    int n_removed = 0;
    auto fast_count = [&]() { int n = 0; for (int c : comps) n += c; return n; };
    auto set_cid = [&]() {   // component index per particle of F (nested levels reuse the array)
        if ((int)cid.extent(0) < p.N) Kokkos::realloc(cid, p.N);
        std::vector<int> h;
        for (size_t c = 0; c < comps.size(); c++) h.insert(h.end(), comps[c], (int)c);
        if (h.empty()) return;
        Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hv(h.data(), h.size());
        Kokkos::deep_copy(Kokkos::subview(cid, Kokkos::make_pair(fast_start, fast_start + (int)h.size())), hv);
    };
    auto cross = [&](bool with_grad) {
        const int nf = fast_count();
        g_profiler.start();
        cross_acceleration(p, slow_start, slow_count, fast_start, nf, G, soft2);   // nf = 0 or slow = 0: zero
        set_cid();
        cc_inter_acceleration(p, fast_start, nf, cid.data(), G, soft2);
        if (with_grad) {
            if ((int)snx.extent(0) < nf) { Kokkos::realloc(snx, nf); Kokkos::realloc(sny, nf); Kokkos::realloc(snz, nf); }
            RawView ax{p.ax.data()}, ay{p.ay.data()}, az{p.az.data()}, ox{snx.data()}, oy{sny.data()}, oz{snz.data()};
            const int f0 = fast_start;
            small_for("cc_snap", nf, KOKKOS_LAMBDA(int ii) {
                ox(ii) = ax(f0 + ii); oy(ii) = ay(f0 + ii); oz(ii) = az(f0 + ii);
            });
            if (slow_count > 0)
                cross_gradient_correction(p, slow_start, slow_count, fast_start, nf, G, soft2, pivot_dt);
            cc_inter_gradient(p, fast_start, nf, cid.data(), GradSnapshot{snx.data(), sny.data(), snz.data(), fast_start},
                              G, soft2, pivot_dt);
        }
        g_profiler.stop(with_grad ? StepProfiler::CROSS_GRAD : StepProfiler::CROSS_FORCE);
    };
    auto kick = [&](double f) {
        g_profiler.start();
        kick_velocities(p, slow_start, slow_count, pivot_dt * f);
        kick_velocities(p, fast_start, fast_count(), pivot_dt * f);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    };
    auto recurse_all = [&]() {
        int s0 = fast_start;
        for (int& c : comps) {
            if (c == 1) {   // a lone particle has no internal interactions: its own Hamiltonian is a free drift
                g_profiler.start();
                drift(p, s0, 1, pivot_dt * 0.5);
                g_profiler.stop(StepProfiler::DRIFT_KICK);
                s0 += 1;
                continue;
            }
            int k = hhs_fsi_recursive(p, dp, s0, c, pivot_dt * 0.5, G, soft2, dt_min, eta, false, false, depth + 1);
            c -= k;
            n_removed += k;
            s0 += c;
        }
    };
#ifdef ENABLE_COLLISIONAL
    auto saved_active = g_subsystem_mgr.active;
    auto restore = [&]() {
        g_profiler.start();
        g_subsystem_mgr.restore_active(saved_active);
        if (!g_subsystem_mgr.active.empty()) g_subsystem_mgr.rebuild_pid_to_idx(p);
        g_subsystem_mgr.update_particle_flags(p);
        g_profiler.stop(StepProfiler::SUBSYSTEM_BOOKKEEP);
    };
#else
    auto restore = []() {};
#endif
    cross(false); kick(FSI_KICK);
    if (slow_count > 0) fsi_step(p, slow_start, slow_count, G, soft2, pivot_dt * 0.5, false, true);
    recurse_all();
    restore();
    cross(true); kick(FSI_MOD_KICK);
    recurse_all();
    restore();
    if (slow_count > 0) fsi_step(p, slow_start, slow_count, G, soft2, pivot_dt * 0.5, false, true);
    cross(false); kick(FSI_KICK);
    return n_removed;
}

// =============================================================================
// Unified HHS-FSI recursion (Algorithm 4 from FROST paper)
//
// Single function handles all GPU paths: MPI-distributed, local GPU, and tree.
// dp == nullptr → single-rank mode (no MPI, no tree).
// dp != nullptr → dispatch MPI vs local GPU based on count vs serial_n2_gpu.
// Tree operations require dp != nullptr (tree_threshold >> serial_n2_gpu).
// CPU serial path (count <= serial_n2_cpu) delegated to hhs_fsi_recursive_cpu.
// =============================================================================
static int hhs_fsi_recursive(
    Particles& p,
    DistributedParticles* dp,
    int start, int count,
    double pivot_dt,
    double G, double soft2,
    double dt_min, double eta,
    bool dt_valid,
    bool global_tree_available,
    int depth) {

    if (count == 0) return 0;
    int n_removed = 0;  // merged-away particles; [start, start+count) is always a suffix of [0, N)
    struct DepthScope {   // profiler charges work to this depth until the call returns
        int saved;
        explicit DepthScope(int d) : saved(g_profiler.cur_depth) { g_profiler.cur_depth = d; }
        ~DepthScope() { g_profiler.cur_depth = saved; }
    } depth_scope(depth);

    // HHS_CC splits sets of at most cc_max particles; they run locally on every rank (replicated), so that the
    // integration does not depend on the number of ranks
    const int cc_max = hhs_cc_max >= 0 ? hhs_cc_max : std::max(serial_n2_gpu, 16384);
    const bool cc_set = hhs_cc && count <= cc_max;
    // Dispatch flags: MPI for large subsets when distributed
    bool use_mpi = dp && (serial_n2_gpu == 0 || count > serial_n2_gpu) && !cc_set;
    bool tree_eligible = dp && use_tree && count >= tree_threshold;

    // Counters
    if (use_mpi) g_hhs_counters.distributed++;
    else g_hhs_counters.local++;
    if (depth > g_hhs_counters.max_depth) g_hhs_counters.max_depth = depth;
    if (depth < HHSCounters::MAX_DEPTH) {
        if (use_mpi) {
            g_hhs_counters.dist_depth_calls[depth]++;
            g_hhs_counters.dist_depth_count[depth] += count;
        } else {
            g_hhs_counters.local_depth_calls[depth]++;
            g_hhs_counters.local_depth_count[depth] += count;
        }
    }
    g_subset_hist.record(use_mpi ? SubsetHistogram::DISTRIBUTED : SubsetHistogram::LOCAL_GPU, count);

    // CPU serial for very small subsets
    if (serial_n2_cpu > 0 && count <= serial_n2_cpu) {
        g_hhs_counters.cpu++;
        g_profiler.start();
        n_removed = hhs_fsi_recursive_cpu(p, start, count, pivot_dt, G, soft2, dt_min, eta, dt_valid, depth);
        g_profiler.stop(StepProfiler::CPU_PATH);
        if (n_removed > 0) sync_after_removal(p, dp);
        if (dp) p.pack_pos_range(start, count - n_removed);
        return n_removed;
    }

    // Base case: at minimum timestep, use FSI on the subset
    if (pivot_dt * 0.5 < dt_min) {
        if (dp && global_tree_available && tree_eligible) {
            g_profiler.start();
            g_octree.update_subsets(p, start, count);
            g_profiler.stop(StepProfiler::TREE_MAINT);
            fsi_step_distributed(*dp, start, count, G, soft2, pivot_dt, false, false, 0);
        } else if (use_mpi) {
            fsi_step_distributed(*dp, start, count, G, soft2, pivot_dt);
        } else {
            fsi_step(p, start, count, G, soft2, pivot_dt, false, true);
        }
        return 0;
    }

    // BIFROST Section 3.5: fresh neighbor search + detection, then timestep
    bool forces_valid = false;
    bool tree_built_for_subset = false;
    if (!dt_valid && count > 1) {
#ifdef ENABLE_COLLISIONAL
        // Step 1: Standalone neighbor search + detection (before timestep)
        // NBODY_DETECT_TOP_ONLY (diagnostic): detect at depth 0 only and keep those subsystems below
        static const bool detect_top_only = std::getenv("NBODY_DETECT_TOP_ONLY") != nullptr;
        if (p.collisional && !(detect_top_only && depth > 0)) {
            g_profiler.start();
            g_subsystem_mgr.reset_ngb_count();
            neighbor_search_subset(p, start, count, G,
                                   g_collisional.r_ngb, g_collisional.r_ngb_max, pivot_dt);
            g_subsystem_mgr.detect_from_pairs(p, G, pivot_dt,
                                               g_collisional.r_ngb, g_collisional.r_ngb_max,
                                               g_collisional.n_orb_secular);
            g_subsystem_mgr.update_particle_flags(p);
            g_subsystem_mgr.accumulate_diagnostics();
            if (depth < HHSCounters::MAX_DEPTH)
                g_hhs_counters.dist_depth_ngb_pairs[depth] += g_subsystem_mgr.last_level_pairs;
            g_profiler.stop(StepProfiler::SUBSYSTEM_DETECT);
        }
#endif
        // Step 2: Timestep computation (with subsystem exclusion active)
        g_profiler.start();
        if (global_tree_available && tree_eligible) {
            g_octree.update_subsets(p, start, count);
            tree_global_self_acceleration_and_timestep(p, g_octree, 0, G, soft2, eta);
            tree_momentum_correction(p, start, count);
            forces_valid = true;
        } else if (!global_tree_available && tree_eligible) {
            g_octree.n_leaf = tree_n_leaf;
            g_octree.n_group = tree_n_group;
            g_octree.build(p, start, count, tree_theta);
            int my_g_off, my_g_cnt;
            get_tree_group_range(g_octree, my_g_off, my_g_cnt);
            zero_accelerations(p, start, count);
            init_timesteps_max(p, start, count);
            tree_self_acceleration_and_timestep(p, g_octree, start, count, G, soft2, eta, my_g_off, my_g_cnt);
            dp->allreduce_accelerations_subset(start, count);
            dp->allreduce_timestep_subset(start, count);
            tree_momentum_correction(p, start, count);
            forces_valid = true;
            tree_built_for_subset = true;
        } else if (use_mpi) {
            p.pack_pos_range(start, count);
            p.pack_vel_range(start, count);
            compute_timestep_distributed(*dp, start, count, G, soft2, eta);
        } else {
            compute_timestep_subset(p, start, count, G, soft2, eta);
        }
#ifdef ENABLE_COLLISIONAL
        // Gradient + v/a timestep criteria (BIFROST Section 3.4.4, Eq. 24/26/27), each off at eta = 0
        if (p.collisional && (g_collisional.eta_grad > 0.0 || g_collisional.eta_va > 0.0)) {
            auto dt_f = p.dt_adapt;
            auto gm = p.grad_mag;
            RawView agx{p.ax.data()}, agy{p.ay.data()}, agz{p.az.data()};
            RawView vvx{p.vx.data()}, vvy{p.vy.data()}, vvz{p.vz.data()};
            double eta_g = g_collisional.eta_grad;
            double eta_va = g_collisional.eta_va;   // halved by the restart loop like the other etas
            small_for("dt_grad", count, KOKKOS_LAMBDA(int ii) {
                int i = start + ii;
                double a_mag = Kokkos::sqrt(agx(i)*agx(i) + agy(i)*agy(i) + agz(i)*agz(i));
                double v_mag = Kokkos::sqrt(vvx(i)*vvx(i) + vvy(i)*vvy(i) + vvz(i)*vvz(i));
                double g = gm(i);
                if (eta_g > 0.0 && g > EPS) {
                    if (a_mag > EPS) {
                        double dt_g2 = eta_g * Kokkos::sqrt(a_mag / g);
                        if (dt_g2 < dt_f(i)) dt_f(i) = dt_g2;
                    }
                    if (v_mag > EPS) {
                        double dt_g1 = eta_g * Kokkos::cbrt(v_mag / g);
                        if (dt_g1 < dt_f(i)) dt_f(i) = dt_g1;
                    }
                }
                if (eta_va > 0.0 && a_mag > EPS && v_mag > EPS) {
                    double dt_va = eta_va * v_mag / a_mag;
                    if (dt_va < dt_f(i)) dt_f(i) = dt_va;
                }
            });
        }
        // Level-0 clamping (BIFROST Section 3.4.1)
        if (p.collisional && depth > 0) {
            auto dt_f = p.dt_adapt;
            auto dt_l0 = p.dt_level0;
            int n_clamped = 0;
            Kokkos::parallel_reduce("clamp_dt_level0", count,
                KOKKOS_LAMBDA(int ii, int& cnt) {
                    int i = start + ii;
                    if (dt_f(i) > dt_l0(i)) {
                        dt_f(i) = dt_l0(i);
                        cnt++;
                    }
                }, n_clamped);
            if (depth < HHSCounters::MAX_DEPTH) g_hhs_counters.dist_depth_l0_clamped[depth] += n_clamped;
        }
#endif
        g_profiler.stop(StepProfiler::ADAPTIVE_DT, (long long)count * count);
#ifdef ENABLE_COLLISIONAL
        // Step 3: Subsystem timestep adjustments
        if (p.collisional) {
            g_profiler.start();
            g_subsystem_mgr.subsystem_timesteps(p, G, g_collisional.eta_pert);

            // Level-0 store: capture dt AFTER subsystem adjustments
            if (depth == 0) {
                auto dt_f = p.dt_adapt;
                auto dt_l0 = p.dt_level0;
                small_for("copy_dt_level0", count, KOKKOS_LAMBDA(int ii) {
                    int i = start + ii;
                    dt_l0(i) = dt_f(i);
                });
            }

            // Merger detection at every HHS level (BIFROST §3.14)
            if (g_collisional.c_code > 0.0) {
                auto mergers = detect_mergers(g_subsystem_mgr, p, G,
                                               g_collisional.c_code, pivot_dt);
                for (const auto& ev : mergers) {
                    auto* sub = find_subsystem_for_merger(g_subsystem_mgr, ev);
                    if (sub) g_merger_tree.record(ev, *sub, g_sim_time);
                    execute_merger(ev, p, g_subsystem_mgr, g_energy_books,
                                   G, g_collisional.c_code);
                }
                g_subsystem_mgr.step_mergers += (int)mergers.size();
                if (!mergers.empty()) {
                    // Compaction shifts only the tail [idx, N), which callers see as a shorter suffix
                    int k = p.compact_remove_zero_mass(start);
                    sync_after_removal(p, dp);
                    n_removed += k;
                    count -= k;
                    if (count <= 0) return n_removed;
                    g_subsystem_mgr.update_particle_flags(p);
                }
            }
            g_profiler.stop(StepProfiler::SUBSYSTEM_DT);
        }
#endif
    }

#ifdef ENABLE_COLLISIONAL
    // A set that is one subsystem (or one particle) has no interaction left for the recursion to resolve: the
    // members' mutual force is the subsystem's (LogH), and external perturbations reach it only from the parent
    // levels. Recursing would only cut its LogH integration into 2^d pieces; one FSI step over the pivot (drift and
    // one subsystem advance) is the same Hamiltonian, exact to the GBS tolerance.
    if (p.collisional && count <= LOGH_MAX_BODIES && pivot_dt * 0.5 >= dt_min) {
        bool one_subsystem = (count == 1);
        if (!one_subsystem) {
            int s0 = -2, same = 1;
            auto h_sid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                             Kokkos::subview(p.subsystem_id, Kokkos::make_pair(start, start + count)));
            for (int k = 0; k < count; k++) {
                int sk = h_sid(k);
                if (sk < 0 || (s0 != -2 && sk != s0)) { same = 0; break; }
                s0 = sk;
            }
            // Not for a triple: whether a (stable, re-evaluated in the advance) triple goes secular depends on the
            // step it is advanced with (inner orbits per dt), so the longer pivot would hand it to the orbit-averaged
            // evolution where the recursion uses LogH (200k, 10 % binaries: |dE/E| 5e-10 -> 3e-9 within three steps
            // of the first secular triple)
            one_subsystem = same && count != 3;
        }
        if (one_subsystem) {
            if (depth < HHSCounters::MAX_DEPTH) g_hhs_counters.isolated_sets[depth]++;
            fsi_step(p, start, count, G, soft2, pivot_dt, false, true);
            return n_removed;
        }
    }
#endif

    // Partition into slow (dt >= pivot) and fast (dt < pivot)
    g_profiler.start();
    int slow_count = partition_by_pivot(p, start, count, pivot_dt);
    g_profiler.stop(StepProfiler::PARTITION);
    int slow_start = start;
    int fast_start = start + slow_count;
    int fast_count = count - slow_count;

    // Tree availability for sub-calls (either built here or inherited)
    bool tree_for_subs = global_tree_available || tree_built_for_subset;

    // Base case: all particles are slow — reuse fused forces for first FSI kick
    if (fast_count == 0) {
        // ALL_SLOW_SPLIT (FROST Algorithm 3, F empty): fsi(S, tau/2) twice, on every path
        const int n_sub = all_slow_split ? 2 : 1;
        const double h = pivot_dt / n_sub;
        if (dp && global_tree_available && tree_eligible) {
            for (int k = 0; k < n_sub; k++)
                fsi_step_distributed(*dp, start, count, G, soft2, h, k == 0 && forces_valid, false, 0);
        } else if (use_mpi) {
            for (int k = 0; k < n_sub; k++)
                fsi_step_distributed(*dp, start, count, G, soft2, h, k == 0 && forces_valid,
                                     k == 0 && tree_built_for_subset);
        } else if (all_slow_split) {
            // FROST Algorithm 3 with an empty fast set: fsi(S, tau/2) twice
            fsi_step(p, start, count, G, soft2, pivot_dt * 0.5, forces_valid, true);
            fsi_step(p, start, count, G, soft2, pivot_dt * 0.5, false, true);
        } else {
            fsi_step(p, start, count, G, soft2, pivot_dt, forces_valid, true);
        }
        return n_removed;
    }

    // HHS_CC: split the fast set into components when it falls apart (local path only)
    if (cc_set && !tree_eligible && fast_count >= 2) {
        g_profiler.start();
        std::vector<int> comps = cc_components(p, fast_start, fast_count, pivot_dt, G, soft2, eta);
        g_profiler.stop(StepProfiler::PARTITION);
        if (depth < HHSCounters::MAX_DEPTH) {
            g_hhs_counters.cc_calls[depth]++;
            g_hhs_counters.cc_comps[depth] += (long long)comps.size();
            g_hhs_counters.cc_largest[depth] += *std::max_element(comps.begin(), comps.end());
        }
        if (comps.size() > 1)
            return n_removed + hhs_cc_step(p, dp, slow_start, slow_count, fast_start, std::move(comps), pivot_dt,
                                           G, soft2, dt_min, eta, depth);
    }

    // Base case: all particles are fast
    if (slow_count == 0) {
        if (use_mpi) g_hhs_counters.slow0_dist++;
        else g_hhs_counters.slow0_local++;
        double sub_pivot = pivot_dt * 0.5;
        int k = hhs_fsi_recursive(p, dp, start, count, sub_pivot,
                                  G, soft2, dt_min, eta, true, tree_for_subs, depth + 1);
        n_removed += k;
        count -= k;
        n_removed += hhs_fsi_recursive(p, dp, start, count, sub_pivot,
                                       G, soft2, dt_min, eta, false, tree_for_subs, depth + 1);
        return n_removed;
    }

    long long cross_pairs = 2LL * slow_count * fast_count;

    // Re-sync packed positions after partition reorder (only needed for MPI)
    if (use_mpi) {
        g_profiler.start();
        p.pack_pos_range(start, count);
        g_profiler.stop(StepProfiler::PACK_DATA);
    }

    // ==========================================================================
    // ALGORITHM 4: HHS-FSI with slow (S) and fast (F) particles
    // ==========================================================================

    // Tree flags for cross-force and self-force
    bool use_tree_cross = tree_for_subs && use_tree && count >= tree_threshold;
    bool use_tree_self = tree_for_subs && use_tree && slow_count >= tree_threshold;

    // Mark slow/fast/other in global tree
    if (use_tree_cross || use_tree_self) {
        g_profiler.start();
        if (use_tree_cross)
            g_octree.update_subsets_3flag(p, start, count, slow_count);
        else
            g_octree.update_subsets(p, slow_start, slow_count);
        g_profiler.stop(StepProfiler::TREE_MAINT);
    }

#ifdef ENABLE_COLLISIONAL
    // Save subsystem state before recursive calls can overwrite it
    auto saved_active = g_subsystem_mgr.active;
#endif

    // Step 1: Cross-force S<->F then kick (dt/6)
    g_profiler.start();
    if (use_tree_cross) {
        cross_acceleration_global_tree(p, slow_start, slow_count,
                                        fast_start, fast_count, G, soft2);
    } else if (use_mpi) {
        p.pack_pos_range(slow_start, slow_count + fast_count);
        cross_acceleration_distributed(*dp, slow_start, slow_count, fast_start, fast_count, G, soft2);
    } else {
        cross_acceleration(p, slow_start, slow_count, fast_start, fast_count, G, soft2);
    }
    g_profiler.stop(StepProfiler::CROSS_FORCE, cross_pairs);
    g_profiler.start();
    kick_velocities(p, slow_start, slow_count, pivot_dt * FSI_KICK);
    kick_velocities(p, fast_start, fast_count, pivot_dt * FSI_KICK);
    g_profiler.stop(StepProfiler::DRIFT_KICK);

    // Step 2: FSI half-step on slow
    if (use_tree_self || use_tree_cross) {
        fsi_step_distributed(*dp, slow_start, slow_count, G, soft2, pivot_dt * 0.5,
                             false, false, 0);
    } else if (use_mpi) {
        fsi_step_distributed(*dp, slow_start, slow_count, G, soft2, pivot_dt * 0.5);
    } else {
        fsi_step(p, slow_start, slow_count, G, soft2, pivot_dt * 0.5, false, true);
    }

    // Step 3: Recurse on fast (first half) — always compute fresh timesteps
    int k = hhs_fsi_recursive(p, dp, fast_start, fast_count, pivot_dt * 0.5,
                              G, soft2, dt_min, eta, false, tree_for_subs, depth + 1);
    n_removed += k;
    fast_count -= k;
    count -= k;

    // Restore subsystem state after recursion overwrote it
#ifdef ENABLE_COLLISIONAL
    g_profiler.start();
    g_subsystem_mgr.restore_active(saved_active);
    // The flags go by particle id; partitions below moved particles since the map was last built
    if (!g_subsystem_mgr.active.empty()) g_subsystem_mgr.rebuild_pid_to_idx(p);
    g_subsystem_mgr.update_particle_flags(p);
    g_profiler.stop(StepProfiler::SUBSYSTEM_BOOKKEEP);
#endif

    // Restore tree state after steps 2-3
    if (use_tree_cross || use_tree_self) {
        g_profiler.start();
        if (use_tree_cross)
            g_octree.update_subsets_3flag(p, start, count, slow_count);
        else
            g_octree.update_subsets(p, slow_start, slow_count);
        g_profiler.stop(StepProfiler::TREE_MAINT);
    }

    // Step 4: Cross-kick S<->F with gradient (2dt/3)
    // Re-sync packed positions: step 3's recursion may have run locally (use_mpi=false)
    // modifying SoA positions via drifts without updating pos_packed.
    if (use_mpi) {
        g_profiler.start();
        p.pack_pos_range(slow_start, slow_count + fast_count);
        g_profiler.stop(StepProfiler::PACK_DATA);
    }
    g_profiler.start();
    if (use_tree_cross) {
        cross_acceleration_and_gradient_global_tree(p, slow_start, slow_count,
                                                     fast_start, fast_count, G, soft2, pivot_dt);
    } else if (use_mpi) {
        cross_acceleration_and_gradient_distributed(*dp, slow_start, slow_count,
                                                    fast_start, fast_count, G, soft2, pivot_dt);
    } else {
        cross_acceleration(p, slow_start, slow_count, fast_start, fast_count, G, soft2);
        cross_gradient_correction(p, slow_start, slow_count, fast_start, fast_count, G, soft2, pivot_dt);
    }
    g_profiler.stop(StepProfiler::CROSS_GRAD, cross_pairs * 2);
    g_profiler.start();
    kick_velocities(p, slow_start, slow_count, pivot_dt * FSI_MOD_KICK);
    kick_velocities(p, fast_start, fast_count, pivot_dt * FSI_MOD_KICK);
    g_profiler.stop(StepProfiler::DRIFT_KICK);

    // Step 5: Recurse on fast (second half)
    k = hhs_fsi_recursive(p, dp, fast_start, fast_count, pivot_dt * 0.5,
                          G, soft2, dt_min, eta, false, tree_for_subs, depth + 1);
    n_removed += k;
    fast_count -= k;
    count -= k;

    // Restore again for the second half-step on slow
#ifdef ENABLE_COLLISIONAL
    g_profiler.start();
    g_subsystem_mgr.restore_active(saved_active);
    // The flags go by particle id; partitions below moved particles since the map was last built
    if (!g_subsystem_mgr.active.empty()) g_subsystem_mgr.rebuild_pid_to_idx(p);
    g_subsystem_mgr.update_particle_flags(p);
    g_profiler.stop(StepProfiler::SUBSYSTEM_BOOKKEEP);
#endif

    // Restore tree state for step 6 and 7
    if (use_tree_cross || use_tree_self) {
        g_profiler.start();
        if (use_tree_cross)
            g_octree.update_subsets_3flag(p, start, count, slow_count);
        else
            g_octree.update_subsets(p, slow_start, slow_count);
        g_profiler.stop(StepProfiler::TREE_MAINT);
    }

    // Step 6: FSI half-step on slow
    if (use_tree_self || use_tree_cross) {
        fsi_step_distributed(*dp, slow_start, slow_count, G, soft2, pivot_dt * 0.5,
                             false, false, 0);
    } else if (use_mpi) {
        fsi_step_distributed(*dp, slow_start, slow_count, G, soft2, pivot_dt * 0.5);
    } else {
        fsi_step(p, slow_start, slow_count, G, soft2, pivot_dt * 0.5, false, true);
    }

    // Refresh tree for step 7 cross-force (step 6 FSI drifted slow particles)
    if (use_tree_cross) {
        g_profiler.start();
        g_octree.update_subsets_3flag(p, start, count, slow_count);
        g_profiler.stop(StepProfiler::TREE_MAINT);
    }

    // Step 7: Cross-force S<->F then kick (dt/6)
    g_profiler.start();
    if (use_tree_cross) {
        cross_acceleration_global_tree(p, slow_start, slow_count,
                                        fast_start, fast_count, G, soft2);
    } else if (use_mpi) {
        p.pack_pos_range(slow_start, slow_count + fast_count);
        cross_acceleration_distributed(*dp, slow_start, slow_count, fast_start, fast_count, G, soft2);
    } else {
        cross_acceleration(p, slow_start, slow_count, fast_start, fast_count, G, soft2);
    }
    g_profiler.stop(StepProfiler::CROSS_FORCE, cross_pairs);
    g_profiler.start();
    kick_velocities(p, slow_start, slow_count, pivot_dt * FSI_KICK);
    kick_velocities(p, fast_start, fast_count, pivot_dt * FSI_KICK);
    g_profiler.stop(StepProfiler::DRIFT_KICK);
    return n_removed;
}

// Distributed HHS-FSI - main entry point
void hhs_fsi_step_distributed(DistributedParticles& dp, double pivot_dt,
                              double G, double soft2, double dt_min,
                              double eta, int global_levels) {
    if (dp.N_total == 0) return;

    (void)global_levels;  // Not used in new implementation

    // Reset debug counters
    g_hhs_counters.reset();

#ifdef ENABLE_COLLISIONAL
    if (dp.local.collisional) g_subsystem_mgr.reset_step_diagnostics();
#endif

#ifdef USE_MPI
    bool is_distributed = MPIComm::size() > 1;
#else
    bool is_distributed = false;
#endif

    // Global PN1.0 half-kick before FSI (BIFROST Section 3.7)
#ifdef ENABLE_COLLISIONAL
    bool do_global_pn = dp.local.collisional && (g_collisional.pn_order & 1) &&
                        g_collisional.c_code > 0.0;
    if (do_global_pn) {
        global_pn1_kick(dp.local, 0, dp.N_total, G, g_collisional.c_code,
                        g_collisional.m_pn, pivot_dt * 0.5);
    }
#endif

    // Unified recursion: dp != nullptr enables MPI dispatch for large subsets
    // Single-rank mode: MPI allgather is a no-op, all forces use local GPU
    hhs_fsi_recursive(dp.local, is_distributed ? &dp : nullptr,
                      0, dp.N_total, pivot_dt, G, soft2, dt_min, eta, false);

    // Global PN1.0 half-kick after FSI (BIFROST Section 3.7)
#ifdef ENABLE_COLLISIONAL
    if (do_global_pn) {
        global_pn1_kick(dp.local, 0, dp.N_total, G, g_collisional.c_code,
                        g_collisional.m_pn, pivot_dt * 0.5);
    }
    // print_step_summary moved into unified block below
#endif

    // Unified post-step summary: HHS hierarchy + subsystem stats
#ifdef USE_MPI
    if (MPIComm::rank() == 0)
#endif
    {
        // --- HHS Hierarchy ---
        std::printf("\n  --- HHS Hierarchy (mpi=%d gpu=%d cpu=%d) ",
                    g_hhs_counters.distributed, g_hhs_counters.local, g_hhs_counters.cpu);
        for (int i = 0; i < 40; i++) std::printf("-");
        std::printf("\n");

        // Find last populated depth (across all paths: distributed, local GPU, CPU serial)
        int last_depth = -1;
        for (int d = HHSCounters::MAX_DEPTH - 1; d >= 0; d--) {
            if (g_hhs_counters.dist_depth_calls[d] > 0 || g_hhs_counters.local_depth_calls[d] > 0 || g_hhs_counters.cpu_depth_calls[d] > 0) {
                last_depth = d;
                break;
            }
        }
        for (int d = 0; d <= last_depth; d++) {
            int dc = g_hhs_counters.dist_depth_calls[d];
            int lc = g_hhs_counters.local_depth_calls[d];
            int cc = g_hhs_counters.cpu_depth_calls[d];
            int total_calls = dc + lc + cc;
            long long total_count = g_hhs_counters.dist_depth_count[d] + g_hhs_counters.local_depth_count[d] + g_hhs_counters.cpu_depth_count[d];
            long long avg_n = total_calls > 0 ? total_count / total_calls : 0;
            std::printf("    L%d: %3d calls", d, total_calls);
            if (dc > 0 || lc > 0 || cc > 0) {
                std::printf(" [");
                bool first = true;
                if (dc > 0) { std::printf("mpi=%d", dc); first = false; }
                if (lc > 0) { std::printf("%sgpu=%d", first ? "" : " ", lc); first = false; }
                if (cc > 0) { std::printf("%scpu=%d", first ? "" : " ", cc); first = false; }
                std::printf("]");
            }
            std::printf("  avg_n=%5lld", avg_n);
            if (g_hhs_counters.dist_depth_ngb_pairs[d] > 0)
                std::printf("  ngb=%d", g_hhs_counters.dist_depth_ngb_pairs[d]);
            if (g_hhs_counters.dist_depth_l0_clamped[d] > 0)
                std::printf("  l0_clamp=%d", g_hhs_counters.dist_depth_l0_clamped[d]);
            if (g_hhs_counters.isolated_sets[d] > 0) std::printf("  isolated=%lld", g_hhs_counters.isolated_sets[d]);
            if (g_hhs_counters.cc_calls[d] > 0)
                std::printf("  cc: %lld calls, %.1f comps, largest %.0f", g_hhs_counters.cc_calls[d],
                            (double)g_hhs_counters.cc_comps[d] / g_hhs_counters.cc_calls[d],
                            (double)g_hhs_counters.cc_largest[d] / g_hhs_counters.cc_calls[d]);
            if (g_hhs_counters.local_depth_time[d] > 0.001)
                std::printf("  time=%.3fs", g_hhs_counters.local_depth_time[d]);
            std::printf("\n");
        }

#ifdef ENABLE_COLLISIONAL
        if (dp.local.collisional) {
            g_subsystem_mgr.print_step_summary();
            g_subsys_phase.report_and_reset();
        }
#endif
    }
}

