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
    d_level_dts = Kokkos::View<double*>();
    std::free(level_start); level_start = nullptr;
    std::free(level_count); level_count = nullptr;
    N = 0; n_levels = 0;
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

    // Step 3: Apply permutation entirely on GPU using gather
    // Use scratch arrays as temporary storage
    auto tmp_d = scratch_d;    // double temp
    auto tmp_i = scratch_i2;   // int temp

    // Helper lambdas for gather operations
    auto px = x; auto py = y; auto pz = z;
    auto pvx = vx; auto pvy = vy; auto pvz = vz;
    auto pax = ax; auto pay = ay; auto paz = az;
    auto pmass = mass; auto pdt = dt_frost;
    auto porig = orig_index;
    int NN = N;

    // Gather x
    Kokkos::parallel_for("gather_x", NN, KOKKOS_LAMBDA(int i) { tmp_d(i) = px(perm(i)); });
    Kokkos::fence();
    Kokkos::parallel_for("copy_x", NN, KOKKOS_LAMBDA(int i) { px(i) = tmp_d(i); });

    // Gather y
    Kokkos::parallel_for("gather_y", NN, KOKKOS_LAMBDA(int i) { tmp_d(i) = py(perm(i)); });
    Kokkos::fence();
    Kokkos::parallel_for("copy_y", NN, KOKKOS_LAMBDA(int i) { py(i) = tmp_d(i); });

    // Gather z
    Kokkos::parallel_for("gather_z", NN, KOKKOS_LAMBDA(int i) { tmp_d(i) = pz(perm(i)); });
    Kokkos::fence();
    Kokkos::parallel_for("copy_z", NN, KOKKOS_LAMBDA(int i) { pz(i) = tmp_d(i); });

    // Gather vx
    Kokkos::parallel_for("gather_vx", NN, KOKKOS_LAMBDA(int i) { tmp_d(i) = pvx(perm(i)); });
    Kokkos::fence();
    Kokkos::parallel_for("copy_vx", NN, KOKKOS_LAMBDA(int i) { pvx(i) = tmp_d(i); });

    // Gather vy
    Kokkos::parallel_for("gather_vy", NN, KOKKOS_LAMBDA(int i) { tmp_d(i) = pvy(perm(i)); });
    Kokkos::fence();
    Kokkos::parallel_for("copy_vy", NN, KOKKOS_LAMBDA(int i) { pvy(i) = tmp_d(i); });

    // Gather vz
    Kokkos::parallel_for("gather_vz", NN, KOKKOS_LAMBDA(int i) { tmp_d(i) = pvz(perm(i)); });
    Kokkos::fence();
    Kokkos::parallel_for("copy_vz", NN, KOKKOS_LAMBDA(int i) { pvz(i) = tmp_d(i); });

    // Gather ax
    Kokkos::parallel_for("gather_ax", NN, KOKKOS_LAMBDA(int i) { tmp_d(i) = pax(perm(i)); });
    Kokkos::fence();
    Kokkos::parallel_for("copy_ax", NN, KOKKOS_LAMBDA(int i) { pax(i) = tmp_d(i); });

    // Gather ay
    Kokkos::parallel_for("gather_ay", NN, KOKKOS_LAMBDA(int i) { tmp_d(i) = pay(perm(i)); });
    Kokkos::fence();
    Kokkos::parallel_for("copy_ay", NN, KOKKOS_LAMBDA(int i) { pay(i) = tmp_d(i); });

    // Gather az
    Kokkos::parallel_for("gather_az", NN, KOKKOS_LAMBDA(int i) { tmp_d(i) = paz(perm(i)); });
    Kokkos::fence();
    Kokkos::parallel_for("copy_az", NN, KOKKOS_LAMBDA(int i) { paz(i) = tmp_d(i); });

    // Gather mass
    Kokkos::parallel_for("gather_mass", NN, KOKKOS_LAMBDA(int i) { tmp_d(i) = pmass(perm(i)); });
    Kokkos::fence();
    Kokkos::parallel_for("copy_mass", NN, KOKKOS_LAMBDA(int i) { pmass(i) = tmp_d(i); });

    // Gather dt_frost
    Kokkos::parallel_for("gather_dt", NN, KOKKOS_LAMBDA(int i) { tmp_d(i) = pdt(perm(i)); });
    Kokkos::fence();
    Kokkos::parallel_for("copy_dt", NN, KOKKOS_LAMBDA(int i) { pdt(i) = tmp_d(i); });

    // Gather level (int)
    Kokkos::parallel_for("gather_level", NN, KOKKOS_LAMBDA(int i) { tmp_i(i) = p_level(perm(i)); });
    Kokkos::fence();
    Kokkos::parallel_for("copy_level", NN, KOKKOS_LAMBDA(int i) { p_level(i) = tmp_i(i); });

    // Gather orig_index (int)
    Kokkos::parallel_for("gather_orig", NN, KOKKOS_LAMBDA(int i) { tmp_i(i) = porig(perm(i)); });
    Kokkos::fence();
    Kokkos::parallel_for("copy_orig", NN, KOKKOS_LAMBDA(int i) { porig(i) = tmp_i(i); });
}
