#pragma once

#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

struct Particles {
    int N = 0;

    // SoA, 64-byte aligned for AVX-512
    double *__restrict__ x  = nullptr, *__restrict__ y  = nullptr, *__restrict__ z  = nullptr;
    double *__restrict__ vx = nullptr, *__restrict__ vy = nullptr, *__restrict__ vz = nullptr;
    double *__restrict__ ax = nullptr, *__restrict__ ay = nullptr, *__restrict__ az = nullptr;
    double *__restrict__ mass = nullptr;
    double *__restrict__ dt_frost = nullptr;
    int *level = nullptr;
    int *orig_index = nullptr;        // original particle index (survives partition)

    // Level partitioning
    int *level_start = nullptr;
    int *level_count = nullptr;
    int n_levels = 0;

    // Scratch buffers — allocated once, reused in hot path (zero malloc)
    double *scratch1 = nullptr, *scratch2 = nullptr, *scratch3 = nullptr;  // 3×N for acc snapshots
    double *scratch4 = nullptr, *scratch5 = nullptr, *scratch6 = nullptr;  // 3×N for second group snapshots
    double *scratch7 = nullptr, *scratch8 = nullptr, *scratch9 = nullptr;  // 3×N for Kahan compensation
    double *scratch_d = nullptr;   // N doubles for permutation tmp
    int *scratch_i = nullptr;      // N ints for permutation tmp
    int *scratch_i2 = nullptr;     // N ints for cursor tmp
    int *level_cursor = nullptr;   // pre-allocated cursor for partition_by_level
    int level_cursor_cap = 0;      // capacity of level_cursor

    void allocate(int n);
    void deallocate();
    void partition_by_level();

private:
    // Helper for aligned allocation
    static double* alloc_double(int n) {
        void* p = std::aligned_alloc(64, ((size_t)n * sizeof(double) + 63) & ~63UL);
        std::memset(p, 0, ((size_t)n * sizeof(double) + 63) & ~63UL);
        return static_cast<double*>(p);
    }
};
