#pragma once

#include <chrono>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>

#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

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
            "CPU small-set path"
        };
        return names[i];
    }

    double elapsed[NUM_TIMERS];
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

    // Runtime switches (main.cpp): enabled = PROFILING; fence = PROFILING_FENCES (default true). stop() fences the
    // device so that each component is charged its own asynchronous kernels; ~50k fences per 200k step cost 4 % of
    // the step (AUDIT G row 6). Without the fence the timers measure host wall time between start and stop, and
    // queued device work is charged to whichever component next synchronises; the step total is unaffected.
    bool enabled = true;
    bool fence = true;

    Timer active_component;  // which component is currently being timed
    std::chrono::high_resolution_clock::time_point t_start;
    std::chrono::high_resolution_clock::time_point t_mpi;

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
        pending_mpi = 0;
        step_wall_total = 0.0;
        step_count = 0;
        active_component = NUM_TIMERS;
    }

    void start(Timer ctx = NUM_TIMERS) {
#ifndef DISABLE_PROFILING_FENCES
        if (!enabled) return;
        t_start = std::chrono::high_resolution_clock::now();
        active_component = ctx;
#else
        (void)ctx;
#endif
    }

    void stop(Timer t, long long n_pairs = 0) {
#ifndef DISABLE_PROFILING_FENCES
        if (!enabled) return;
        if (fence) Kokkos::fence();  // ensure GPU work is complete before timing
        auto t_end = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(t_end - t_start).count();
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
        active_component = NUM_TIMERS;
#else
        (void)t; (void)n_pairs;
#endif
    }

    // MPI sub-timing: called from inside MPI communication functions
    // Accumulates into elapsed_mpi[active_component] if a component is active
    void mpi_start() {
        t_mpi = std::chrono::high_resolution_clock::now();
    }

    void mpi_stop() {
        auto t_end = std::chrono::high_resolution_clock::now();
        pending_mpi += std::chrono::duration<double>(t_end - t_mpi).count();
    }

    void print_report(int /*N*/, int n_ranks) const {
        double accounted = 0.0;
        for (int i = 0; i < NUM_TIMERS; i++)
            accounted += elapsed[i];
        double other = step_wall_total - accounted;
        if (other < 0.0) other = 0.0;

        std::printf("\n--- Performance (%d steps, %d ranks) ", step_count, n_ranks);
        for (int i = 0; i < 46; i++) std::printf("-");
        std::printf("\n");
        std::printf("  %-24s %10s %8s %10s %8s %10s %10s\n",
                    "Component", "Time [s]", "Calls", "Avg [ms]", "% Total",
                    "Gpairs", "Gp/s/rank");

        for (int i = 0; i < NUM_TIMERS; i++) {
            if (counts[i] == 0) continue;
            double avg_ms = (elapsed[i] / counts[i]) * 1000.0;
            double pct = (step_wall_total > 0.0) ? 100.0 * elapsed[i] / step_wall_total : 0.0;
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

        // Other
        double pct_other = (step_wall_total > 0.0) ? 100.0 * other / step_wall_total : 0.0;
        std::printf("  %-24s %10.3f %8s %10s %7.1f%%\n",
                    "Other", other, "-", "-", pct_other);

        std::printf("  ");
        for (int i = 0; i < 68; i++) std::printf("-");
        std::printf("\n");

        double ms_per_step = (step_count > 0) ? (step_wall_total / step_count) * 1000.0 : 0.0;
        std::printf("  %-24s %10.3f %8d %10.3f ms/step\n",
                    "Total wall", step_wall_total, step_count, ms_per_step);

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
            double pct_mpi = (step_wall_total > 0.0) ? 100.0 * mpi_total / step_wall_total : 0.0;
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
                double sub = e[SUBSYSTEM_DETECT] + e[SUBSYSTEM_INTEG] + e[SUBSYSTEM_DT] + e[SUBSYSTEM_BOOKKEEP];
                double kp = e[DRIFT_KICK] + e[PARTITION] + e[PACK_DATA];
                double tot = 0;
                for (int i = 0; i < NUM_TIMERS; i++) tot += e[i];
                std::printf("  d%-5d %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f\n", d, tot, f, e[SUBSYSTEM_DETECT],
                            e[SUBSYSTEM_INTEG], e[SUBSYSTEM_DT], e[SUBSYSTEM_BOOKKEEP], kp, tot - f - sub - kp, depth_mpi[d]);
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
