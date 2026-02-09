#pragma once

#include <chrono>
#include <cstdio>
#include <cstring>

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
            "Drift/Kick"
        };
        return names[i];
    }

    double elapsed[NUM_TIMERS];
    int counts[NUM_TIMERS];
    long long pairs[NUM_TIMERS];
    double step_wall_total;
    int step_count;

    std::chrono::high_resolution_clock::time_point t_start;

    StepProfiler() { reset(); }

    void reset() {
        std::memset(elapsed, 0, sizeof(elapsed));
        std::memset(counts, 0, sizeof(counts));
        std::memset(pairs, 0, sizeof(pairs));
        step_wall_total = 0.0;
        step_count = 0;
    }

    void start() {
        t_start = std::chrono::high_resolution_clock::now();
    }

    void stop(Timer t, long long n_pairs = 0) {
        auto t_end = std::chrono::high_resolution_clock::now();
        elapsed[t] += std::chrono::duration<double>(t_end - t_start).count();
        counts[t]++;
        pairs[t] += n_pairs;
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
        std::printf("  %-24s %10s %8s %10s %8s\n",
                    "Component", "Time [s]", "Calls", "Avg [ms]", "% Total");

        for (int i = 0; i < NUM_TIMERS; i++) {
            if (counts[i] == 0) continue;
            double avg_ms = (elapsed[i] / counts[i]) * 1000.0;
            double pct = (step_wall_total > 0.0) ? 100.0 * elapsed[i] / step_wall_total : 0.0;
            std::printf("  %-24s %10.3f %8d %10.3f %7.1f%%\n",
                        timer_name(i), elapsed[i], counts[i], avg_ms, pct);
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
