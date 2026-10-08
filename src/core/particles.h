#pragma once

#include <Kokkos_Core.hpp>
#include <vector>

struct Particles {
    int N = 0;
    bool collisional = false;  // whether collisional fields are allocated

    // SoA using Kokkos Views (default memory space)
    Kokkos::View<double*> x, y, z;
    Kokkos::View<int64_t*> ix, iy, iz;   // Integer positions (source of truth when enabled)
    Kokkos::View<double*> vx, vy, vz;
    Kokkos::View<double*> cvx, cvy, cvz;  // Kahan compensation of the kicks (0 is always valid)
    Kokkos::View<double*> ax, ay, az;
    Kokkos::View<double*> mass;
    Kokkos::View<double*> dt_adapt;
    Kokkos::View<int*> level;
    Kokkos::View<int*> orig_index;

    // Packed views for vectorized memory access: [N][4] LayoutRight
    // LayoutRight makes pos(i,0..3) contiguous, enabling double4 loads.
    // pos_packed[i] = {x, y, z, mass}
    // vel_packed[i] = {vx, vy, vz, 0}
    Kokkos::View<double*[4], Kokkos::LayoutRight> pos_packed;
    Kokkos::View<double*[4], Kokkos::LayoutRight> vel_packed;

    // Level partitioning (host-side)
    std::vector<int> level_start;
    std::vector<int> level_count;
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
    Kokkos::View<int64_t*> scratch_l1, scratch_l2, scratch_l3;  // int64 scratch for permutation

    // =========================================================================
    // Collisional fields (allocated only when collisional=true)
    // =========================================================================

    // Permanent unique ID (survives mergers, removals — distinct from orig_index)
    Kokkos::View<int*> particle_id;

    // Subsystem membership (-1 = field particle, >= 0 = subsystem index)
    Kokkos::View<int*> subsystem_id;

    // Stellar properties (for runtime SSE)
    Kokkos::View<int*>    stellar_type;    // kw: 0-15 (Hurley classification)
    Kokkos::View<double*> m_zams;          // ZAMS mass (M_sun)
    Kokkos::View<double*> stellar_age;     // age (Myr)
    Kokkos::View<double*> stellar_radius;  // radius (R_sun)
    Kokkos::View<double*> luminosity;      // L_sun
    Kokkos::View<double*> core_mass;       // core mass (M_sun)

    // Post-Newtonian spin (for BHs)
    Kokkos::View<double*[3], Kokkos::LayoutRight> spin_vec;
    Kokkos::View<double*> spin_mag;        // |chi| dimensionless parameter

    // Level-0 clamping
    Kokkos::View<double*> dt_level0;       // timestep from top level, never increases

    // Gradient timestep criterion
    Kokkos::View<double*> grad_mag;        // |gradient correction| from previous step (stale)

    void allocate(int n, bool collisional_mode = false);
    void deallocate();
    void partition_by_level();
    void pack();    // Copy SoA to packed format
    void unpack();  // Copy packed back to SoA (for velocities after kick)
    void pack_pos_range(int start, int count);  // Pack x,y,z,mass into pos_packed for [start, start+count)
    void pack_vel_range(int start, int count);  // Pack vx,vy,vz into vel_packed for [start, start+count)

    // Remove particles with mass==0, compact remaining, update N
    // Returns number of particles removed
    int compact_remove_zero_mass();
    void zero_kick_compensation(int start = 0, int count = -1);  // after velocities are overwritten
};
