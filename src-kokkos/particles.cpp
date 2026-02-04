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

    scratch1  = Kokkos::View<double*>("s1", n);
    scratch2  = Kokkos::View<double*>("s2", n);
    scratch3  = Kokkos::View<double*>("s3", n);
    scratch4  = Kokkos::View<double*>("s4", n);
    scratch5  = Kokkos::View<double*>("s5", n);
    scratch6  = Kokkos::View<double*>("s6", n);
    scratch7  = Kokkos::View<double*>("s7", n);
    scratch8  = Kokkos::View<double*>("s8", n);
    scratch9  = Kokkos::View<double*>("s9", n);
    scratch10 = Kokkos::View<double*>("s10", n);
    scratch11 = Kokkos::View<double*>("s11", n);
    scratch_i  = Kokkos::View<int*>("si", n);
    scratch_i2 = Kokkos::View<int*>("si2", n);
    scratch_i3 = Kokkos::View<int*>("si3", n);

    // Packed views for vectorized memory access
    pos_packed = Kokkos::View<double*[4]>("pos_packed", n);
    vel_packed = Kokkos::View<double*[4]>("vel_packed", n);
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
    scratch10 = scratch11 = Kokkos::View<double*>();
    scratch_i = scratch_i2 = scratch_i3 = Kokkos::View<int*>();
    d_level_dts = Kokkos::View<double*>();
    pos_packed = Kokkos::View<double*[4]>();
    vel_packed = Kokkos::View<double*[4]>();
    std::free(level_start); level_start = nullptr;
    std::free(level_count); level_count = nullptr;
    N = 0; n_levels = 0;
}

void Particles::pack() {
    // Copy SoA data into packed format for vectorized access
    auto px = x; auto py = y; auto pz = z;
    auto pvx = vx; auto pvy = vy; auto pvz = vz;
    auto pm = mass;
    auto pos = pos_packed;
    auto vel = vel_packed;
    int NN = N;

    Kokkos::parallel_for("pack", NN, KOKKOS_LAMBDA(int i) {
        pos(i, 0) = px(i);
        pos(i, 1) = py(i);
        pos(i, 2) = pz(i);
        pos(i, 3) = pm(i);
        vel(i, 0) = pvx(i);
        vel(i, 1) = pvy(i);
        vel(i, 2) = pvz(i);
        vel(i, 3) = 0.0;
    });
    Kokkos::fence();
}

void Particles::unpack() {
    // Copy packed data back to SoA (mainly for velocities after kicks)
    auto px = x; auto py = y; auto pz = z;
    auto pvx = vx; auto pvy = vy; auto pvz = vz;
    auto pos = pos_packed;
    auto vel = vel_packed;
    int NN = N;

    Kokkos::parallel_for("unpack", NN, KOKKOS_LAMBDA(int i) {
        px(i) = pos(i, 0);
        py(i) = pos(i, 1);
        pz(i) = pos(i, 2);
        // mass doesn't change, no need to unpack
        pvx(i) = vel(i, 0);
        pvy(i) = vel(i, 1);
        pvz(i) = vel(i, 2);
    });
    Kokkos::fence();
}

void Particles::partition_by_level() {
    if (N == 0 || n_levels == 0) return;

    // Realloc level_count / level_start on host
    std::free(level_count);
    std::free(level_start);
    level_count = static_cast<int*>(std::calloc(n_levels, sizeof(int)));
    level_start = static_cast<int*>(std::calloc(n_levels, sizeof(int)));

    // Create device Views for counts and cursors
    Kokkos::View<int*> d_counts("d_counts", n_levels);
    Kokkos::View<int*> d_cursors("d_cursors", n_levels);

    // Step 1: Count particles per level on GPU
    auto p_level = level;
    Kokkos::parallel_for("count_levels", N, KOKKOS_LAMBDA(int i) {
        Kokkos::atomic_fetch_add(&d_counts(p_level(i)), 1);
    });
    Kokkos::fence();

    // Copy counts to host and compute prefix sums
    auto h_counts = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_counts);
    for (int l = 0; l < n_levels; l++) level_count[l] = h_counts(l);
    level_start[0] = 0;
    for (int l = 1; l < n_levels; l++)
        level_start[l] = level_start[l-1] + level_count[l-1];

    // Copy level_start to device cursors
    auto h_cursors = Kokkos::create_mirror_view(d_cursors);
    for (int l = 0; l < n_levels; l++) h_cursors(l) = level_start[l];
    Kokkos::deep_copy(d_cursors, h_cursors);

    // Step 2: Build permutation on GPU using atomics
    auto perm = scratch_i;  // reuse scratch for permutation
    Kokkos::parallel_for("build_perm", N, KOKKOS_LAMBDA(int i) {
        int l = p_level(i);
        int dest = Kokkos::atomic_fetch_add(&d_cursors(l), 1);
        perm(dest) = i;  // perm[dest] = source index i
    });
    Kokkos::fence();

    // Step 3: Apply permutation with fused gather/copy (2 kernels instead of 26)
    auto px = x; auto py = y; auto pz = z;
    auto pvx = vx; auto pvy = vy; auto pvz = vz;
    auto pax = ax; auto pay = ay; auto paz = az;
    auto pmass = mass; auto pdt = dt_frost;
    auto porig = orig_index;
    int NN = N;

    // Scratch arrays for all 13 fields
    auto t1 = scratch1;  auto t2 = scratch2;  auto t3 = scratch3;
    auto t4 = scratch4;  auto t5 = scratch5;  auto t6 = scratch6;
    auto t7 = scratch7;  auto t8 = scratch8;  auto t9 = scratch9;
    auto t10 = scratch10; auto t11 = scratch11;
    auto ti1 = scratch_i2; auto ti2 = scratch_i3;

    // Single fused gather of all fields
    Kokkos::parallel_for("gather_all", NN, KOKKOS_LAMBDA(int i) {
        int src = perm(i);
        t1(i) = px(src);   t2(i) = py(src);   t3(i) = pz(src);
        t4(i) = pvx(src);  t5(i) = pvy(src);  t6(i) = pvz(src);
        t7(i) = pax(src);  t8(i) = pay(src);  t9(i) = paz(src);
        t10(i) = pmass(src); t11(i) = pdt(src);
        ti1(i) = porig(src); ti2(i) = p_level(src);
    });
    Kokkos::fence();

    // Single fused copy back
    Kokkos::parallel_for("copy_all", NN, KOKKOS_LAMBDA(int i) {
        px(i) = t1(i);   py(i) = t2(i);   pz(i) = t3(i);
        pvx(i) = t4(i);  pvy(i) = t5(i);  pvz(i) = t6(i);
        pax(i) = t7(i);  pay(i) = t8(i);  paz(i) = t9(i);
        pmass(i) = t10(i); pdt(i) = t11(i);
        porig(i) = ti1(i); p_level(i) = ti2(i);
    });
    Kokkos::fence();
}
