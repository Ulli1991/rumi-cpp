#pragma once

#include <chrono>
#include <cstring>

// =============================================================================
// HHS Recursion Debug Counters
// Tracks call counts, particle counts, and timing per depth for all 3 paths
// (distributed, local GPU, CPU serial). Reset at the start of each outer step.
// =============================================================================
struct HHSCounters {
    int distributed = 0;
    int local = 0;
    int cpu = 0;
    int slow0_dist = 0;
    int slow0_local = 0;
    int max_depth = 0;

    static constexpr int MAX_DEPTH = 40;

    // Per-depth: local GPU path
    int local_depth_calls[MAX_DEPTH] = {};
    long long local_depth_count[MAX_DEPTH] = {};
    double local_depth_time[MAX_DEPTH] = {};

    // Per-depth: distributed path
    int dist_depth_calls[MAX_DEPTH] = {};
    long long dist_depth_count[MAX_DEPTH] = {};
    int dist_depth_ngb_pairs[MAX_DEPTH] = {};
    int dist_depth_l0_clamped[MAX_DEPTH] = {};

    // Per-depth: CPU serial path
    int cpu_depth_calls[MAX_DEPTH] = {};
    long long cpu_depth_count[MAX_DEPTH] = {};

    void reset() {
        distributed = local = cpu = slow0_dist = slow0_local = max_depth = 0;
        std::memset(local_depth_calls, 0, sizeof(local_depth_calls));
        std::memset(local_depth_count, 0, sizeof(local_depth_count));
        std::memset(local_depth_time, 0, sizeof(local_depth_time));
        std::memset(dist_depth_calls, 0, sizeof(dist_depth_calls));
        std::memset(dist_depth_count, 0, sizeof(dist_depth_count));
        std::memset(dist_depth_ngb_pairs, 0, sizeof(dist_depth_ngb_pairs));
        std::memset(dist_depth_l0_clamped, 0, sizeof(dist_depth_l0_clamped));
        std::memset(cpu_depth_calls, 0, sizeof(cpu_depth_calls));
        std::memset(cpu_depth_count, 0, sizeof(cpu_depth_count));
    }
};

extern HHSCounters g_hhs_counters;

// RAII timer for per-depth inclusive wall time in local GPU path
struct ScopedDepthTimer {
    int depth;
    std::chrono::high_resolution_clock::time_point t0;
    ScopedDepthTimer(int d) : depth(d), t0(std::chrono::high_resolution_clock::now()) {}
    ~ScopedDepthTimer() {
        auto t1 = std::chrono::high_resolution_clock::now();
        if (depth < HHSCounters::MAX_DEPTH)
            g_hhs_counters.local_depth_time[depth] += std::chrono::duration<double>(t1 - t0).count();
    }
};
