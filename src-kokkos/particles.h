#pragma once

#include <Kokkos_Core.hpp>
#include <vector>

struct Particles {
    int N = 0;

    // SoA using Kokkos Views (default memory space)
    Kokkos::View<double*> x, y, z;
    Kokkos::View<double*> vx, vy, vz;
    Kokkos::View<double*> ax, ay, az;
    Kokkos::View<double*> mass;
    Kokkos::View<double*> dt_adapt;
    Kokkos::View<int*> level;
    Kokkos::View<int*> orig_index;

    // Packed views for vectorized memory access: [N][4] layout
    // pos_packed[i] = {x, y, z, mass}
    // vel_packed[i] = {vx, vy, vz, 0}
    Kokkos::View<double*[4]> pos_packed;
    Kokkos::View<double*[4]> vel_packed;

    // Level partitioning (host-side)
    int* level_start = nullptr;
    int* level_count = nullptr;
    int n_levels = 0;

    // Cached device View for level timesteps (avoids reallocation every step)
    Kokkos::View<double*> d_level_dts;

    // Scratch buffers (11 doubles + 3 ints for fused partitioning)
    Kokkos::View<double*> scratch1, scratch2, scratch3;
    Kokkos::View<double*> scratch4, scratch5, scratch6;
    Kokkos::View<double*> scratch7, scratch8, scratch9;
    Kokkos::View<double*> scratch10, scratch11;
    Kokkos::View<int*> scratch_i;   // used for permutation
    Kokkos::View<int*> scratch_i2, scratch_i3;

    void allocate(int n);
    void deallocate();
    void partition_by_level();
    void pack();    // Copy SoA to packed format
    void unpack();  // Copy packed back to SoA (for velocities after kick)
};
