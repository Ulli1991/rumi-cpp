#include "particles.h"
#include <cstdlib>
#include <cstring>

void Particles::allocate(int n) {
    N = n;
    x  = Kokkos::View<double*>("x",  n);
    y  = Kokkos::View<double*>("y",  n);
    z  = Kokkos::View<double*>("z",  n);
    vx = Kokkos::View<double*>("vx", n);
    vy = Kokkos::View<double*>("vy", n);
    vz = Kokkos::View<double*>("vz", n);
    ax = Kokkos::View<double*>("ax", n);
    ay = Kokkos::View<double*>("ay", n);
    az = Kokkos::View<double*>("az", n);
    mass     = Kokkos::View<double*>("mass", n);
    dt_frost = Kokkos::View<double*>("dt_frost", n);
    level      = Kokkos::View<int*>("level", n);
    orig_index = Kokkos::View<int*>("orig_index", n);

    // Initialize orig_index on host then copy
    {
        auto h = Kokkos::create_mirror_view(orig_index);
        for (int i = 0; i < n; i++) h(i) = i;
        Kokkos::deep_copy(orig_index, h);
    }

    level_start = nullptr;
    level_count = nullptr;
    n_levels = 0;

    scratch1 = Kokkos::View<double*>("s1", n);
    scratch2 = Kokkos::View<double*>("s2", n);
    scratch3 = Kokkos::View<double*>("s3", n);
    scratch4 = Kokkos::View<double*>("s4", n);
    scratch5 = Kokkos::View<double*>("s5", n);
    scratch6 = Kokkos::View<double*>("s6", n);
    scratch7 = Kokkos::View<double*>("s7", n);
    scratch8 = Kokkos::View<double*>("s8", n);
    scratch9 = Kokkos::View<double*>("s9", n);
    scratch_d  = Kokkos::View<double*>("sd", n);
    scratch_i  = Kokkos::View<int*>("si", n);
    scratch_i2 = Kokkos::View<int*>("si2", n);
}

void Particles::deallocate() {
    // Kokkos Views are reference-counted; reassigning releases memory
    x = y = z = Kokkos::View<double*>();
    vx = vy = vz = Kokkos::View<double*>();
    ax = ay = az = Kokkos::View<double*>();
    mass = dt_frost = Kokkos::View<double*>();
    level = orig_index = Kokkos::View<int*>();
    scratch1 = scratch2 = scratch3 = Kokkos::View<double*>();
    scratch4 = scratch5 = scratch6 = Kokkos::View<double*>();
    scratch7 = scratch8 = scratch9 = Kokkos::View<double*>();
    scratch_d = Kokkos::View<double*>();
    scratch_i = scratch_i2 = Kokkos::View<int*>();
    std::free(level_start); level_start = nullptr;
    std::free(level_count); level_count = nullptr;
    N = 0; n_levels = 0;
}

void Particles::partition_by_level() {
    if (N == 0 || n_levels == 0) return;

    // Mirror all arrays to host for sorting
    auto h_x  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
    auto h_y  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
    auto h_z  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), z);
    auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vx);
    auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vy);
    auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vz);
    auto h_ax = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ax);
    auto h_ay = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ay);
    auto h_az = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), az);
    auto h_mass     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mass);
    auto h_dt_frost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dt_frost);
    auto h_level    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), level);
    auto h_orig     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), orig_index);

    // Realloc level_count / level_start
    std::free(level_count);
    std::free(level_start);
    level_count = static_cast<int*>(std::calloc(n_levels, sizeof(int)));
    level_start = static_cast<int*>(std::calloc(n_levels, sizeof(int)));

    for (int i = 0; i < N; i++) level_count[h_level(i)]++;
    level_start[0] = 0;
    for (int l = 1; l < n_levels; l++)
        level_start[l] = level_start[l-1] + level_count[l-1];

    // Build permutation via counting sort
    std::vector<int> perm(N);
    std::vector<int> cursor(n_levels);
    for (int l = 0; l < n_levels; l++) cursor[l] = level_start[l];
    for (int i = 0; i < N; i++) {
        int l = h_level(i);
        perm[cursor[l]++] = i;
    }

    // Apply permutation using scratch on host
    auto h_tmp = Kokkos::create_mirror_view(scratch_d);

    auto apply_perm = [&](auto& h_arr) {
        for (int i = 0; i < N; i++) h_tmp(i) = h_arr(perm[i]);
        for (int i = 0; i < N; i++) h_arr(i) = h_tmp(i);
    };

    apply_perm(h_x);  apply_perm(h_y);  apply_perm(h_z);
    apply_perm(h_vx); apply_perm(h_vy); apply_perm(h_vz);
    apply_perm(h_ax); apply_perm(h_ay); apply_perm(h_az);
    apply_perm(h_mass);
    apply_perm(h_dt_frost);

    // Permute int arrays
    auto h_tmp_i = Kokkos::create_mirror_view(scratch_i);
    auto apply_perm_int = [&](auto& h_arr) {
        for (int i = 0; i < N; i++) h_tmp_i(i) = h_arr(perm[i]);
        for (int i = 0; i < N; i++) h_arr(i) = h_tmp_i(i);
    };
    apply_perm_int(h_level);
    apply_perm_int(h_orig);

    // Deep copy back to device
    Kokkos::deep_copy(x, h_x);   Kokkos::deep_copy(y, h_y);   Kokkos::deep_copy(z, h_z);
    Kokkos::deep_copy(vx, h_vx); Kokkos::deep_copy(vy, h_vy); Kokkos::deep_copy(vz, h_vz);
    Kokkos::deep_copy(ax, h_ax); Kokkos::deep_copy(ay, h_ay); Kokkos::deep_copy(az, h_az);
    Kokkos::deep_copy(mass, h_mass);
    Kokkos::deep_copy(dt_frost, h_dt_frost);
    Kokkos::deep_copy(level, h_level);
    Kokkos::deep_copy(orig_index, h_orig);
}
