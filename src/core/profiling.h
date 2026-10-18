#pragma once

#include <chrono>
#include <climits>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include "lean_sync.h"

#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

// NBODY_NVTX (gpu-idle prototype): profiler regions become NVTX ranges for nsys idle-time attribution.
// start() pushes "P"; stop(t) pops it and emits a mark "<timer>@<depth>" at the same instant. A Region pushes a
// range named after its bucket.
#if defined(NBODY_NVTX) && defined(KOKKOS_ENABLE_CUDA)
#include <nvtx3/nvToolsExt.h>
#define NBODY_NVTX_PUSH(s) nvtxRangePushA(s)
#define NBODY_NVTX_POP() nvtxRangePop()
#define NBODY_NVTX_MARK(s) nvtxMarkA(s)
#else
#define NBODY_NVTX_PUSH(s) ((void)0)
#define NBODY_NVTX_POP() ((void)0)
#define NBODY_NVTX_MARK(s) ((void)0)
#endif
struct NvtxScope {
    explicit NvtxScope(const char* s) { (void)s; NBODY_NVTX_PUSH(s); }
    ~NvtxScope() { NBODY_NVTX_POP(); }
};

// core/mpi_comm.cpp: element-wise min, max and sum over the ranks (all ranks get the result)
namespace MPIComm {
int rank();
int size();
void reduce_min_max_sum(const double* in, double* mn, double* mx, double* sum, int n);
}

// Step profiler (PROFILING = true). Every second of wall time between two reports belongs to a named bucket:
//   * start() ... stop(T) open and close a frame. Frames nest; a bucket gets the exclusive time of its frames (the
//     time not spent in frames nested inside), so the buckets add up to the wall time of the interval.
//   * StepProfiler::Region r(g_profiler, T) is the same as a scope, with a named NVTX range under NBODY_NVTX.
//   * add_child_time(T, dt) charges a lap of a sequential phase clock (SubsystemPhaseTimes) to T, inside the open
//     frame; frames that ran during that lap are not counted twice.
//   * The interval runs from one report to the next (each report is charged to the following interval as
//     profile_report). What no frame covers is printed as "unaccounted".
// Plugging in: add an enum entry, its name and its parent below, and bracket the work with start()/stop(T) or a
// Region. Slots for other modules: STELLAR_EVOLUTION (runtime_sse.cpp can nest finer frames inside it) and
// SUBSYS_LOGH (LogH/GBS internals: nest frames inside, with parent SUBSYS_LOGH).
// stop() fences the device when fence = true (PROFILING_FENCES), so that each bucket is charged its own kernels.
// Timers never touch the integration: results are bit-identical with PROFILING on or off.
struct StepProfiler {
    enum Timer {
        ADAPTIVE_DT = 0,
        SELF_FORCE,
        SELF_GRAD,
        CROSS_FORCE,
        CROSS_GRAD,
        PARTITION,
        DRIFT_KICK,
        TREE_MAINT,
        SUBSYSTEM_DETECT,
        SUBSYSTEM_INTEG,
        SUBSYSTEM_DT,
        SUBSYSTEM_BOOKKEEP,
        ENERGY_CHECK,
        PACK_DATA,
        CPU_PATH,
        // containers and paths of a step outside the HHS components above
        STEP,               // one main-loop step (step_t0 .. step_t1)
        HHS,                // the integrator call (HHS-FSI recursion, or the FSI step)
        STELLAR_EVOLUTION,  // runtime SSE before the step (with the removal of massless remnants)
        CHECKPOINT,         // adaptive restarts: checkpoint save / restore
        TIDAL,              // external tidal field kicks
        GLOBAL_PN,          // global PN1.0 half kicks around the recursion
        GHOST_COMPACT,      // removal of merged placeholders after the recursion
        STEP_PRINT,         // per-step output: HHS hierarchy, subsystem summary, fence statistics, "Step n"
        REPLICA_CHECK,      // replicated-state agreement check
        HOST_SUBTREE,       // host-resident subtree: copies and pid-map glue (its recursion has its own buckets)
        // phases of advance_subsystems_in_range (laps of SubsystemPhaseTimes)
        SUBSYS_MIRROR,
        SUBSYS_MAP,
        SUBSYS_SECULAR,
        SUBSYS_SEC_TRIPLE,
        SUBSYS_LOGH_PREP,
        SUBSYS_LOGH_PACK,
        SUBSYS_LOGH,        // the LogH/GBS batch run
        SUBSYS_LOGH_UNPACK,
        SUBSYS_LOGH_POST,
        DET_NGB,
        DET_PAIRS,
        DET_FLAGS,
        DT_WAIT,
        DT_SUBSYS,
        DT_MERGE,
        // between steps
        DIAG,               // diagnostics of a print step
        DIAG_FORCES,        // forces + time steps recomputed for the diagnostics
        DIAG_PRINT,         // energy, time bins, binary/triple trackers, per-subsystem orbital elements
        ESCAPERS,
        EVENT_LOGS,         // merger tree / escaper log files
        OUTPUT_IO,          // after-step output: wall-clock stop check, snapshots, restart files
        SNAPSHOT_IO,
        RESTART_IO,
        CLUSTER_DIAG,
        PROFILE_REPORT,     // the previous report (with its MPI reduction)
        // set-up (printed once, before the first step)
        SETUP_IC,
        SETUP_FORCES,
        SETUP_DIAG,
        NUM_TIMERS
    };

    static const char* timer_name(int i) {
        static const char* names[NUM_TIMERS] = {
            "Adaptive timestep",
            "N-body force (self)",
            "Gradient (self)",
            "N-body force (cross)",
            "Gradient (cross)",
            "Partition",
            "Drift/Kick",
            "Tree maintenance",
            "Subsystem detect",
            "Subsystem integ",
            "Subsystem timestep",
            "Subsystem bookkeep",
            "Energy check",
            "Pack data",
            "CPU small-set path",
            "step",
            "hhs",
            "stellar_evolution",
            "checkpoint",
            "tidal",
            "global_pn",
            "ghost_compact",
            "step_print",
            "replica_check",
            "host_subtree",
            "subsys_mirror",
            "subsys_map",
            "subsys_secular",
            "subsys_sec_triple",
            "subsys_logh_prep",
            "subsys_logh_pack",
            "subsys_logh",
            "subsys_logh_unpack",
            "subsys_logh_post",
            "det_ngb",
            "det_pairs",
            "det_flags",
            "dt_wait",
            "dt_subsys",
            "dt_merge",
            "diagnostics",
            "diag_forces",
            "diag_print",
            "escapers",
            "event_logs",
            "output_io",
            "snapshot_io",
            "restart_io",
            "cluster_diag",
            "profile_report",
            "setup_ic",
            "setup_forces",
            "setup_diag",
        };
        return names[i];
    }
    // Parent bucket in the printed tree (-1: top level)
    static int parent(int i) {
        switch (i) {
        case STEP: case DIAG: case OUTPUT_IO: case CLUSTER_DIAG: case PROFILE_REPORT:
        case SETUP_IC: case SETUP_FORCES: case SETUP_DIAG: return -1;
        case HHS: case STELLAR_EVOLUTION: case CHECKPOINT: case TIDAL: case STEP_PRINT: case REPLICA_CHECK:
        case ENERGY_CHECK: return STEP;
        case SUBSYS_MIRROR: case SUBSYS_MAP: case SUBSYS_SECULAR: case SUBSYS_SEC_TRIPLE: case SUBSYS_LOGH_PREP:
        case SUBSYS_LOGH_PACK: case SUBSYS_LOGH: case SUBSYS_LOGH_UNPACK: case SUBSYS_LOGH_POST: return SUBSYSTEM_INTEG;
        case DET_NGB: case DET_PAIRS: case DET_FLAGS: return SUBSYSTEM_DETECT;
        case DT_WAIT: case DT_SUBSYS: case DT_MERGE: return SUBSYSTEM_DT;
        case DIAG_FORCES: case DIAG_PRINT: case ESCAPERS: case EVENT_LOGS: return DIAG;
        case SNAPSHOT_IO: case RESTART_IO: return OUTPUT_IO;
        default: return HHS;   // force, kick, partition, tree and subsystem components; global PN; ghost compaction
        }
    }
    static bool is_setup(int i) { return i == SETUP_IC || i == SETUP_FORCES || i == SETUP_DIAG; }

    double elapsed[NUM_TIMERS];       // exclusive time per bucket
    double elapsed_mpi[NUM_TIMERS];   // MPI time within each component
    int counts[NUM_TIMERS];
    int counts_mpi[NUM_TIMERS];
    long long pairs[NUM_TIMERS];
    double step_wall_total;

    // Per HHS depth: time of each component, and the MPI time inside it, charged to the depth that ran it
    static constexpr int MAX_DEPTH = 40;
    int cur_depth = 0;
    double depth_elapsed[MAX_DEPTH][NUM_TIMERS];
    double depth_mpi[MAX_DEPTH];
    double pending_mpi = 0;   // MPI time since the last stop(), charged to the component stop() closes
    // Per depth, LogH batches: calls, subsystems, GBS steps, and the sum over calls of the largest single
    // subsystem's GBS steps (sum / max_sum = parallelism available to the host threads)
    long long logh_calls[MAX_DEPTH], logh_nsub[MAX_DEPTH], logh_gbs[MAX_DEPTH], logh_gbs_max[MAX_DEPTH];
    void record_logh(int n_sub, int gbs, int gbs_max) {
        int d = cur_depth < MAX_DEPTH ? cur_depth : MAX_DEPTH - 1;
        logh_calls[d]++; logh_nsub[d] += n_sub; logh_gbs[d] += gbs; logh_gbs_max[d] += gbs_max;
    }
    int step_count;
    // Per depth: HHS sets entered, their particles (sum), and the force pairs of the components charged there
    long long depth_sets[MAX_DEPTH], depth_particles[MAX_DEPTH], depth_pairs[MAX_DEPTH];
    void record_set(int depth, int count) {
        if (!detailed) return;
        int d = depth < MAX_DEPTH ? depth : MAX_DEPTH - 1;
        depth_sets[d]++; depth_particles[d] += count;
    }

    // Work counters of the interval (reduced over the ranks in the report: mean, min, max)
    enum Counter {
        C_STEPS, C_H2D_BYTES, C_D2H_BYTES, C_D2D_BYTES, C_COPIES, C_MPI_BYTES, C_MPI_CALLS,
        C_SUBSYS_ACTIVE, C_LOGH_SUBSYS, C_GBS_STEPS, C_SDAR_CALLS, C_SECULAR, C_SEC_SUBSTEPS, C_KEPLER2, C_SEC_TRIPLE,
        C_MERGERS, C_ESCAPERS, C_SSE_STARS, C_RESTART_RETRIES, NUM_COUNTERS
    };
    static const char* counter_name(int i) {
        static const char* names[NUM_COUNTERS] = {
            "steps", "host_to_device_bytes", "device_to_host_bytes", "device_to_device_bytes", "deep_copies",
            "mpi_bytes (force collectives)", "mpi_calls (force collectives)", "subsystems_active (sum over steps)",
            "logh_integrations", "gbs_steps", "sdar_calls", "secular_binary_advances", "secular_substeps",
            "kepler2_advances", "secular_triple_advances", "mergers", "escapers", "sse_stars_evolved", "energy_restart_retries"};
        return names[i];
    }
    double counter[NUM_COUNTERS];
    void count(Counter c, double v) { if (detailed) counter[c] += v; }
    // Kokkos Tools deep-copy callback (main.cpp installs it when PROFILING is on): bytes by direction
    static bool is_host_space(const char* n) { return std::strncmp(n, "Host", 4) == 0 || std::strstr(n, "HostPinned"); }
    void record_copy(const char* dst, const char* src, unsigned long long bytes) {
        const bool dh = is_host_space(dst), sh = is_host_space(src);
        counter[C_COPIES] += 1;
        if (dh && !sh) counter[C_D2H_BYTES] += (double)bytes;
        else if (sh && !dh) counter[C_H2D_BYTES] += (double)bytes;
        else if (!sh && !dh) counter[C_D2D_BYTES] += (double)bytes;
    }

    // Runtime switches (main.cpp): enabled = PROFILING; fence = PROFILING_FENCES (default true). stop() fences the
    // device so that each component is charged its own asynchronous kernels; ~50k fences per 200k step cost 4 % of
    // the step (AUDIT G row 6). Without the fence the timers measure host wall time between start and stop, and
    // queued device work is charged to whichever component next synchronises; the sum is unaffected.
    // PROFILING = off: enabled = false (no timers, no counters, no fences); on: enabled; detailed: also the work
    // counters, per-rank min/max (an MPI reduction per report), calls, pairs and the per-depth table
    bool enabled = true;
    bool detailed = false;
    bool fence = true;

    using clock = std::chrono::steady_clock;
    // Open frames: start, time of the frames nested inside, and the part of it already seen by add_child_time
    struct Frame { clock::time_point t0; double child; double child_lapped; };
    static constexpr int MAX_FRAMES = 64;
    Frame frames[MAX_FRAMES];
    int n_frames = 0;
    long long n_unbalanced = 0;    // stop() without start(), frames left open by an early return, overflow
    clock::time_point t_interval;  // start of the interval
    clock::time_point t_report;    // start of the last report (charged to the next interval)
    bool report_pending = false;
    clock::time_point t_mpi;

    StepProfiler() { reset(); }

    void reset() {
        std::memset(elapsed, 0, sizeof(elapsed));
        std::memset(elapsed_mpi, 0, sizeof(elapsed_mpi));
        std::memset(counts, 0, sizeof(counts));
        std::memset(counts_mpi, 0, sizeof(counts_mpi));
        std::memset(pairs, 0, sizeof(pairs));
        std::memset(depth_elapsed, 0, sizeof(depth_elapsed));
        std::memset(depth_mpi, 0, sizeof(depth_mpi));
        std::memset(logh_calls, 0, sizeof(logh_calls));
        std::memset(logh_nsub, 0, sizeof(logh_nsub));
        std::memset(logh_gbs, 0, sizeof(logh_gbs));
        std::memset(logh_gbs_max, 0, sizeof(logh_gbs_max));
        std::memset(depth_sets, 0, sizeof(depth_sets));
        std::memset(depth_particles, 0, sizeof(depth_particles));
        std::memset(depth_pairs, 0, sizeof(depth_pairs));
        std::memset(counter, 0, sizeof(counter));
        pending_mpi = 0;
        step_wall_total = 0.0;
        step_count = 0;
        n_unbalanced = 0;
        auto now = clock::now();
        // Frames still open (a reset inside a region) restart now, so that no time before the interval is charged
        for (int k = 0; k < n_frames; k++) frames[k] = Frame{now, 0.0, 0.0};
        if (report_pending) {   // the report just printed opens the new interval
            t_interval = t_report;
            elapsed[PROFILE_REPORT] = std::chrono::duration<double>(now - t_report).count();
            counts[PROFILE_REPORT] = 1;
            report_pending = false;
        } else {
            t_interval = now;
        }
    }

    void push_frame() {
        if (n_frames == MAX_FRAMES) { n_unbalanced++; return; }
        frames[n_frames++] = Frame{clock::now(), 0.0, 0.0};
    }

    void start(Timer ctx = NUM_TIMERS) {
        (void)ctx;
        NBODY_NVTX_PUSH("P");
#ifndef DISABLE_PROFILING_FENCES
        if (!enabled) return;
        push_frame();
#endif
    }

    // Pops the frames above `level` that an early return left open: their nested time stays charged, their own
    // time goes to the frame below
    void unwind_to(int level) {
        while (n_frames > level) {
            Frame f = frames[--n_frames];
            if (n_frames > 0) frames[n_frames - 1].child += f.child;
            n_unbalanced++;
        }
    }

    void charge(Timer t, double dt, long long n_pairs) {
        elapsed[t] += dt;
        int d = cur_depth < MAX_DEPTH ? cur_depth : MAX_DEPTH - 1;
        depth_elapsed[d][t] += dt;
        if (pending_mpi > 0) {
            elapsed_mpi[t] += pending_mpi;
            counts_mpi[t]++;
            depth_mpi[d] += pending_mpi;
            pending_mpi = 0;
        }
        counts[t]++;
        pairs[t] += n_pairs;
        depth_pairs[d] += n_pairs;
    }

    void pop_frame(Timer t, long long n_pairs) {
        if (n_frames == 0) { n_unbalanced++; return; }
        if (fence && !lean_sync()) Kokkos::fence("profiling.h:stop");  // GPU work complete before timing
        Frame f = frames[--n_frames];
        double dt = std::chrono::duration<double>(clock::now() - f.t0).count();
        double excl = dt - f.child;
        charge(t, excl > 0.0 ? excl : 0.0, n_pairs);
        if (n_frames > 0) frames[n_frames - 1].child += dt;
    }

    void stop(Timer t, long long n_pairs = 0) {
#ifndef DISABLE_PROFILING_FENCES
        if (enabled) pop_frame(t, n_pairs);
#else
        (void)t; (void)n_pairs;
#endif
#if defined(NBODY_NVTX) && defined(KOKKOS_ENABLE_CUDA)
        NBODY_NVTX_POP();
        char buf[48];
        std::snprintf(buf, sizeof buf, "%d@%d", (int)t, cur_depth);
        NBODY_NVTX_MARK(buf);
#endif
    }

    // A lap dt of a sequential phase clock inside the open frame, charged to bucket t (no fence)
    void add_child_time(Timer t, double dt) {
#ifndef DISABLE_PROFILING_FENCES
        if (!enabled) return;
        if (n_frames == 0) { charge(t, dt, 0); return; }
        Frame& f = frames[n_frames - 1];
        double excl = dt - (f.child - f.child_lapped);   // frames that ran inside this lap are charged already
        if (excl < 0.0) excl = 0.0;
        charge(t, excl, 0);
        f.child += excl;
        f.child_lapped = f.child;
#else
        (void)t; (void)dt;
#endif
    }

    // Scoped frame with a named NVTX range; closes frames an early return left open inside it
    struct Region {
        StepProfiler& p;
        Timer t;
        int level = -1;
        Region(StepProfiler& p_, Timer t_) : p(p_), t(t_) {
            NBODY_NVTX_PUSH(timer_name(t_));
#ifndef DISABLE_PROFILING_FENCES
            if (p.enabled && p.n_frames < MAX_FRAMES) { level = p.n_frames; p.push_frame(); }
#endif
        }
        ~Region() {
            if (level >= 0 && p.enabled) {
                p.unwind_to(level + 1);
                if (p.n_frames == level + 1) p.pop_frame(t, 0);
            }
            NBODY_NVTX_POP();
        }
        Region(const Region&) = delete;
        Region& operator=(const Region&) = delete;
    };

    // MPI sub-timing: called from inside MPI communication functions
    void mpi_start() {
        t_mpi = clock::now();
    }

    void mpi_stop(double bytes = 0.0) {
        auto t_end = clock::now();
        pending_mpi += std::chrono::duration<double>(t_end - t_mpi).count();
        if (detailed) { counter[C_MPI_BYTES] += bytes; counter[C_MPI_CALLS] += 1; }
    }

    // Bucket tree of the interval (setup: of the set-up), every bucket reduced over the ranks: mean, min and max
    // of its inclusive time. Collective (every rank calls it); rank 0 prints. Returns unaccounted / wall.
    double print_tree(bool setup, double wall = -1.0) const {
        if (wall < 0.0) wall = std::chrono::duration<double>(clock::now() - t_interval).count();
        // per rank: inclusive time per bucket, wall, unaccounted, calls and pairs per bucket, counters
        constexpr int O_CALLS = NUM_TIMERS + 2, O_PAIRS = O_CALLS + NUM_TIMERS, O_CNT = O_PAIRS + NUM_TIMERS;
        constexpr int NV = O_CNT + NUM_COUNTERS;
        double v[NV], mn[NV], mx[NV], sm[NV];
        for (int i = 0; i < NV; i++) v[i] = 0.0;
        double sum = 0.0;
        for (int i = 0; i < NUM_TIMERS; i++) {
            if (is_setup(i) != setup) continue;
            sum += elapsed[i];
            for (int a = i; a >= 0; a = parent(a)) v[a] += elapsed[i];   // inclusive = sum over the subtree
            v[O_CALLS + i] = counts[i];
            v[O_PAIRS + i] = (double)pairs[i];
        }
        v[NUM_TIMERS] = wall;
        v[NUM_TIMERS + 1] = wall - sum;
        for (int c = 0; c < NUM_COUNTERS; c++) v[O_CNT + c] = setup ? 0.0 : counter[c];
        const int nr = detailed ? MPIComm::size() : 1;   // on: this rank's times only (no reduction)
        if (nr > 1) MPIComm::reduce_min_max_sum(v, mn, mx, sm, NV);
        else for (int i = 0; i < NV; i++) mn[i] = mx[i] = sm[i] = v[i];
        const double wmean = sm[NUM_TIMERS] / nr;
        const double unacc = sm[NUM_TIMERS + 1] / nr;
        if (MPIComm::rank() != 0) return wmean > 0.0 ? unacc / wmean : 0.0;
        write_jsonl(setup, nr, sm, mn, mx, wmean, unacc, O_CALLS, O_PAIRS, O_CNT);
        if (detailed)
            std::printf("\n--- Profile %s (%d steps, %d ranks): inclusive [s] mean over ranks, %% of wall, min, max; "
                        "calls and Gpairs of the bucket itself (mean) ----\n", setup ? "set-up" : "interval",
                        step_count, nr);
        else
            std::printf("\n--- Profile %s (%d steps, rank 0): inclusive [s], %% of wall (PROFILING = detailed: per-rank "
                        "min/max, calls, counters) ----\n", setup ? "set-up" : "interval", step_count);
        const bool det = detailed;
        auto row = [&](int depth, const char* name, double s, double a, double b, double calls, double gp) {
            if (det)
                std::printf("  PROF %*s%-*s %11.4f %6.1f%% %11.4f %11.4f %10.0f %9.3f\n", 2 * depth, "",
                            34 - 2 * depth, name, s, wmean > 0.0 ? 100.0 * s / wmean : 0.0, a, b, calls, gp);
            else
                std::printf("  PROF %*s%-*s %11.4f %6.1f%%\n", 2 * depth, "", 34 - 2 * depth, name, s,
                            wmean > 0.0 ? 100.0 * s / wmean : 0.0);
        };
        // Depth-first over the tree; a container also shows its own time ("self", rank 0)
        struct Walk {
            const StepProfiler& P; const double *sm, *mn, *mx; int nr; bool setup; double wmean;
            void go(int par, int depth, const decltype(row)& r) const {
                for (int i = 0; i < NUM_TIMERS; i++) {
                    if (parent(i) != par || is_setup(i) != setup || mx[i] <= 0.0) continue;
                    r(depth, timer_name(i), sm[i] / nr, mn[i], mx[i], sm[O_CALLS + i] / nr, sm[O_PAIRS + i] / nr * 1e-9);
                    bool kids = false;
                    for (int c = 0; c < NUM_TIMERS; c++) kids = kids || (parent(c) == i && mx[c] > 0.0);
                    if (!kids) continue;
                    go(i, depth + 1, r);
                    std::printf("  PROF %*s%-*s %11.4f %6.1f%%\n", 2 * (depth + 1), "", 34 - 2 * (depth + 1),
                                "(self)", P.elapsed[i], wmean > 0.0 ? 100.0 * P.elapsed[i] / wmean : 0.0);
                }
            }
        };
        Walk{*this, sm, mn, mx, nr, setup, wmean}.go(-1, 0, row);
        row(0, "unaccounted", unacc, mn[NUM_TIMERS + 1], mx[NUM_TIMERS + 1], 0.0, 0.0);
        row(0, "interval wall", wmean, mn[NUM_TIMERS], mx[NUM_TIMERS], 0.0, 0.0);
        std::printf("  PROF_UNACCOUNTED %.6f s = %.3f %% of %.6f s\n", unacc, wmean > 0.0 ? 100.0 * unacc / wmean : 0.0,
                    wmean);
        if (n_unbalanced) std::printf("  PROF_UNBALANCED %lld frames (rank 0)\n", n_unbalanced);
        if (!setup && detailed) {
            std::printf("  PROF_COUNT %-36s %14s %14s %14s\n", "counter", "mean", "min", "max");
            for (int c = 0; c < NUM_COUNTERS; c++)
                if (mx[O_CNT + c] > 0.0)
                    std::printf("  PROF_COUNT %-36s %14.6g %14.6g %14.6g\n", counter_name(c), sm[O_CNT + c] / nr,
                                mn[O_CNT + c], mx[O_CNT + c]);
            // per HHS depth (rank 0; the recursion is replicated): sets, particles, force pairs, time, LogH work
            std::printf("  PROF_DEPTH %5s %9s %12s %10s %10s %9s %9s %10s\n", "depth", "sets", "particles", "Gpairs",
                        "time[s]", "logh_bat", "logh_sub", "gbs_steps");
            for (int d = 0; d < MAX_DEPTH; d++) {
                double t = 0.0;
                for (int i = 0; i < NUM_TIMERS; i++)
                    for (int a = parent(i); a >= 0; a = parent(a))
                        if (a == HHS) { t += depth_elapsed[d][i]; break; }
                if (depth_sets[d] == 0 && t <= 0.0) continue;
                std::printf("  PROF_DEPTH %5d %9lld %12lld %10.4f %10.4f %9lld %9lld %10lld\n", d, depth_sets[d],
                            depth_particles[d], depth_pairs[d] * 1e-9, t, logh_calls[d], logh_nsub[d], logh_gbs[d]);
            }
        }
        std::fflush(stdout);
        return wmean > 0.0 ? unacc / wmean : 0.0;
    }

    // profile.jsonl in the run directory (rank 0, appended, so a restarted run continues it): one line per report
    // with the step and time it was printed at, the wall time, unaccounted, and per bucket the inclusive time
    // (mean over the ranks; min, max), its own time (rank 0), calls and force pairs; detailed: the counters
    int report_step = -1;        // set by main.cpp before each report
    double report_t_myr = 0.0;
    void write_jsonl(bool setup, int nr, const double* sm, const double* mn, const double* mx, double wmean,
                     double unacc, int o_calls, int o_pairs, int o_cnt) const {
        FILE* f = std::fopen("profile.jsonl", "a");
        if (!f) return;
        std::fprintf(f, "{\"kind\": \"%s\", \"level\": \"%s\", \"step\": %d, \"t_myr\": %.10g, \"steps\": %d, "
                     "\"ranks\": %d, \"reduced_over_ranks\": %s, \"wall\": %.6f, \"unaccounted\": %.6f, "
                     "\"step_wall\": %.6f, \"buckets\": {", setup ? "setup" : "interval", detailed ? "detailed" : "on",
                     report_step, report_t_myr, step_count, MPIComm::size(), nr > 1 ? "true" : "false", wmean, unacc,
                     step_wall_total);
        bool first = true;
        for (int i = 0; i < NUM_TIMERS; i++) {
            if (is_setup(i) != setup || mx[i] <= 0.0) continue;
            const int par = parent(i);
            std::fprintf(f, "%s\"%s\": {\"parent\": \"%s\", \"s\": %.6f, \"min\": %.6f, \"max\": %.6f, "
                         "\"self\": %.6f, \"calls\": %.1f, \"pairs\": %.6g}", first ? "" : ", ", timer_name(i),
                         par >= 0 ? timer_name(par) : "", sm[i] / nr, mn[i], mx[i], elapsed[i], sm[o_calls + i] / nr,
                         sm[o_pairs + i] / nr);
            first = false;
        }
        std::fprintf(f, "}");
        if (detailed && !setup) {
            std::fprintf(f, ", \"counters\": {");
            for (int c = 0; c < NUM_COUNTERS; c++)
                std::fprintf(f, "%s\"%s\": {\"mean\": %.6g, \"min\": %.6g, \"max\": %.6g}", c ? ", " : "",
                             counter_name(c), sm[o_cnt + c] / nr, mn[o_cnt + c], mx[o_cnt + c]);
            std::fprintf(f, "}, \"depth\": [");
            bool f1 = true;
            for (int d = 0; d < MAX_DEPTH; d++) {
                if (depth_sets[d] == 0) continue;
                std::fprintf(f, "%s{\"d\": %d, \"sets\": %lld, \"particles\": %lld, \"pairs\": %lld, \"logh_subsys\": %lld, "
                             "\"gbs\": %lld}", f1 ? "" : ", ", d, depth_sets[d], depth_particles[d], depth_pairs[d],
                             logh_nsub[d], logh_gbs[d]);
                f1 = false;
            }
            std::fprintf(f, "]");
        }
        std::fprintf(f, "}\n");
        std::fclose(f);
    }

    // Collective (every rank): the interval tree, then the per-component table of the steps (rank 0)
    void print_report(int N, int n_ranks) {
        t_report = clock::now();
        report_pending = true;
        const double wall = std::chrono::duration<double>(t_report - t_interval).count();
        print_tree(false, wall);
        if (MPIComm::rank() == 0) print_components(N, n_ranks, wall);
    }
    // Set-up profile (collective); a fresh interval starts after it
    void print_setup() {
        print_tree(true);
        reset();
    }

    // Flat table (rank 0): the exclusive time of every bucket of the interval (a container's row is its own time,
    // outside its named children), in % of the interval wall time (report to report, diagnostics and output
    // included). Other = the wall time no bucket covers (the tree's "unaccounted"). Step wall = the steps alone.
    void print_components(int /*N*/, int n_ranks, double wall) const {
        double accounted = 0.0;
        for (int i = 0; i < NUM_TIMERS; i++)
            if (!is_setup(i)) accounted += elapsed[i];
        double other = wall - accounted;
        if (other < 0.0) other = 0.0;

        std::printf("\n--- Performance (%d steps, %d ranks) ", step_count, n_ranks);
        for (int i = 0; i < 46; i++) std::printf("-");
        std::printf("\n");
        std::printf("  %-24s %10s %8s %10s %8s %10s %10s\n",
                    "Component", "Time [s]", "Calls", "Avg [ms]", "% Total",
                    "Gpairs", "Gp/s/rank");

        for (int i = 0; i < NUM_TIMERS; i++) {
            if (counts[i] == 0 || is_setup(i)) continue;
            double avg_ms = (elapsed[i] / counts[i]) * 1000.0;
            double pct = (wall > 0.0) ? 100.0 * elapsed[i] / wall : 0.0;
            if (pairs[i] > 0) {
                double gpairs = pairs[i] / 1.0e9;
                double gp_per_s_per_rank = (gpairs / n_ranks) / elapsed[i];
                std::printf("  %-24s %10.3f %8d %10.3f %7.1f%% %10.1f %10.1f\n",
                            timer_name(i), elapsed[i], counts[i], avg_ms, pct,
                            gpairs, gp_per_s_per_rank);
            } else {
                std::printf("  %-24s %10.3f %8d %10.3f %7.1f%%\n",
                            timer_name(i), elapsed[i], counts[i], avg_ms, pct);
            }
        }

        // Other (covered by no bucket)
        double pct_other = (wall > 0.0) ? 100.0 * other / wall : 0.0;
        std::printf("  %-24s %10.3f %8s %10s %7.1f%%\n",
                    "Other", other, "-", "-", pct_other);

        std::printf("  ");
        for (int i = 0; i < 68; i++) std::printf("-");
        std::printf("\n");

        double ms_per_step = (step_count > 0) ? (wall / step_count) * 1000.0 : 0.0;
        std::printf("  %-24s %10.3f %8d %10.3f ms/step\n",
                    "Total wall", wall, step_count, ms_per_step);
        std::printf("  %-24s %10.3f %8d %10.3f ms/step (steps only: diagnostics and output excluded)\n", "Step wall",
                    step_wall_total, step_count, step_count > 0 ? step_wall_total / step_count * 1000.0 : 0.0);

        // Throughput summary
        long long total_pairs = 0;
        double compute_time = 0.0;
        for (int i = 0; i < NUM_TIMERS; i++) {
            total_pairs += pairs[i];
            if (i == SELF_FORCE || i == SELF_GRAD || i == CROSS_FORCE ||
                i == CROSS_GRAD || i == ADAPTIVE_DT)
                compute_time += elapsed[i];
        }
        if (total_pairs > 0 && compute_time > 0.0) {
            double gpairs = total_pairs / 1.0e9;
            double gpairs_per_rank = gpairs / n_ranks;
            double gpairs_per_sec_per_rank = gpairs_per_rank / compute_time;
            std::printf("  Throughput: %.1f Gpairs total, %.1f Gpairs/rank, "
                        "%.1f Gpairs/s/rank\n",
                        gpairs, gpairs_per_rank, gpairs_per_sec_per_rank);
        }

        // MPI breakdown per component
        double mpi_total = 0;
        int mpi_total_calls = 0;
        bool has_mpi = false;
        for (int i = 0; i < NUM_TIMERS; i++) {
            if (counts_mpi[i] > 0) { has_mpi = true; mpi_total += elapsed_mpi[i]; mpi_total_calls += counts_mpi[i]; }
        }
        if (has_mpi) {
            std::printf("\n  --- MPI breakdown (included in timers above) ");
            for (int i = 0; i < 23; i++) std::printf("-");
            std::printf("\n");
            for (int i = 0; i < NUM_TIMERS; i++) {
                if (counts_mpi[i] == 0) continue;
                double avg_ms = (elapsed_mpi[i] / counts_mpi[i]) * 1000.0;
                double compute = elapsed[i] - elapsed_mpi[i];
                std::printf("  %-24s %10.3f %8d %10.3f   (compute: %.3f s, MPI: %.3f s)\n",
                            timer_name(i), elapsed_mpi[i], counts_mpi[i], avg_ms,
                            compute, elapsed_mpi[i]);
            }
            double pct_mpi = (wall > 0.0) ? 100.0 * mpi_total / wall : 0.0;
            std::printf("  %-24s %10.3f %8d %10s %7.1f%%\n",
                        "Total MPI", mpi_total, mpi_total_calls, "-", pct_mpi);
        }

        // Per-depth breakdown
        int last = -1;
        for (int d = 0; d < MAX_DEPTH; d++)
            for (int i = 0; i < NUM_TIMERS; i++)
                if (depth_elapsed[d][i] > 0) last = d;
        if (last >= 0) {
            std::printf("\n  --- Time per HHS depth [s] ");
            for (int i = 0; i < 71; i++) std::printf("-");
            std::printf("\n  %-6s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n", "Depth", "total", "forces", "sub_det",
                        "sub_int", "sub_dt", "sub_book", "kick/part", "other", "MPI");
            for (int d = 0; d <= last; d++) {
                const double* e = depth_elapsed[d];
                double f = e[ADAPTIVE_DT] + e[SELF_FORCE] + e[SELF_GRAD] + e[CROSS_FORCE] + e[CROSS_GRAD] + e[TREE_MAINT];
                double sdet = e[SUBSYSTEM_DETECT] + e[DET_NGB] + e[DET_PAIRS] + e[DET_FLAGS];
                double sint = e[SUBSYSTEM_INTEG];
                for (int i = SUBSYS_MIRROR; i <= SUBSYS_LOGH_POST; i++) sint += e[i];
                double sdt = e[SUBSYSTEM_DT] + e[DT_WAIT] + e[DT_SUBSYS] + e[DT_MERGE];
                double sub = sdet + sint + sdt + e[SUBSYSTEM_BOOKKEEP];
                double kp = e[DRIFT_KICK] + e[PARTITION] + e[PACK_DATA];
                double tot = 0;   // the HHS subtree
                for (int i = 0; i < NUM_TIMERS; i++)
                    for (int a = parent(i); a >= 0; a = parent(a))
                        if (a == HHS) { tot += e[i]; break; }
                std::printf("  d%-5d %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f\n", d, tot, f, sdet,
                            sint, sdt, e[SUBSYSTEM_BOOKKEEP], kp, tot - f - sub - kp, depth_mpi[d]);
            }
            std::printf("\n  %-6s %9s %11s %12s %12s %10s\n", "LogH", "batches", "subsys/call", "GBS steps", "max/call sum", "parallel");
            for (int d = 0; d <= last; d++) {
                if (logh_calls[d] == 0) continue;
                std::printf("  d%-5d %9lld %11.1f %12lld %12lld %10.1f\n", d, logh_calls[d],
                            (double)logh_nsub[d] / logh_calls[d], logh_gbs[d], logh_gbs_max[d],
                            logh_gbs_max[d] > 0 ? (double)logh_gbs[d] / logh_gbs_max[d] : 0.0);
            }
        }

        std::fflush(stdout);
    }

    void print_memory_report(int N, int n_ranks) const {
        // Per-particle memory: 11 doubles (x,y,z,vx,vy,vz,ax,ay,az,mass,dt_adapt) = 88 B
        //   + 2 ints (level, orig_index) = 8 B
        //   + 2 packed views [N][4] = 64 B
        //   + 11 double scratch + 3 int scratch = 100 B
        //   Total: ~260 B
        double particle_bytes = 260.0;
        double particle_mb = (N * particle_bytes) / (1024.0 * 1024.0);

        // MPI buffers: send (3 * max_per_rank) + recv (3 * N) doubles
        int max_per_rank = (N + n_ranks - 1) / n_ranks;
        double mpi_bytes = (3.0 * max_per_rank + 3.0 * N) * sizeof(double);
        double mpi_mb = mpi_bytes / (1024.0 * 1024.0);

        std::printf("\n--- Memory ");
        for (int i = 0; i < 61; i++) std::printf("-");
        std::printf("\n");
        std::printf("  Particles: %d x %d B = %.1f MB (device)\n", N, (int)particle_bytes, particle_mb);
        if (n_ranks > 1) {
            std::printf("  MPI buffers: %.1f MB\n", mpi_mb);
        }

#ifdef KOKKOS_ENABLE_CUDA
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        double used_mb = (total_mem - free_mem) / (1024.0 * 1024.0);
        double total_mb = total_mem / (1024.0 * 1024.0);
        double pct = 100.0 * (total_mem - free_mem) / (double)total_mem;
        std::printf("  CUDA device: %.0f MB used / %.0f MB total (%.1f%%)\n",
                    used_mb, total_mb, pct);
#endif

        std::printf("\n");
        std::fflush(stdout);
    }
};

extern StepProfiler g_profiler;

// ==========================================================================
// Subset size histogram: tracks where N² work happens in HHS recursion
// ==========================================================================
struct SubsetHistogram {
    static constexpr int N_BUCKETS = 7;
    static constexpr int boundaries[N_BUCKETS] = {64, 256, 1024, 4096, 10000, 50000, INT_MAX};

    // Separate counters for each code path
    enum Path { DISTRIBUTED = 0, LOCAL_GPU, LOCAL_CPU, FSI_STEP, N_PATHS };
    static const char* path_name(int p) {
        static const char* names[N_PATHS] = {"Distributed", "Local GPU", "Local CPU", "FSI step"};
        return names[p];
    }

    long long calls[N_PATHS][N_BUCKETS];
    long long pairs[N_PATHS][N_BUCKETS];  // N² pairs per bucket

    SubsetHistogram() { reset(); }

    void reset() {
        std::memset(calls, 0, sizeof(calls));
        std::memset(pairs, 0, sizeof(pairs));
    }

    void record(Path path, int count) {
        for (int b = 0; b < N_BUCKETS; b++) {
            if (count < boundaries[b]) {
                calls[path][b]++;
                pairs[path][b] += (long long)count * count;
                return;
            }
        }
    }

    void print_report() const {
        // Check if any data recorded
        long long total = 0;
        for (int p = 0; p < N_PATHS; p++)
            for (int b = 0; b < N_BUCKETS; b++)
                total += calls[p][b];
        if (total == 0) return;

        std::printf("\n--- Subset size histogram ");
        for (int i = 0; i < 47; i++) std::printf("-");
        std::printf("\n");
        std::printf("  %-14s", "Size range");
        int prev = 0;
        for (int b = 0; b < N_BUCKETS; b++) {
            if (boundaries[b] == INT_MAX)
                std::printf(" %7d+", prev);
            else
                std::printf(" %5d-%-5d", prev, boundaries[b]);
            prev = boundaries[b];
        }
        std::printf("\n");

        for (int p = 0; p < N_PATHS; p++) {
            long long row_total = 0;
            for (int b = 0; b < N_BUCKETS; b++) row_total += calls[p][b];
            if (row_total == 0) continue;

            std::printf("  %-14s", path_name(p));
            for (int b = 0; b < N_BUCKETS; b++) {
                if (boundaries[b] == INT_MAX)
                    std::printf(" %8lld", calls[p][b]);
                else
                    std::printf(" %10lld", calls[p][b]);
            }
            std::printf("  (calls)\n");

            std::printf("  %-14s", "");
            for (int b = 0; b < N_BUCKETS; b++) {
                double gp = pairs[p][b] / 1.0e9;
                if (boundaries[b] == INT_MAX)
                    std::printf(" %7.1fG", gp);
                else
                    std::printf(" %9.1fG", gp);
            }
            std::printf("  (pairs)\n");
        }
        std::fflush(stdout);
    }
};

extern SubsetHistogram g_subset_hist;
