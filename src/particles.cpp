#include "particles.h"

void Particles::allocate(int n) {
    N = n;
    x  = alloc_double(n); y  = alloc_double(n); z  = alloc_double(n);
    vx = alloc_double(n); vy = alloc_double(n); vz = alloc_double(n);
    ax = alloc_double(n); ay = alloc_double(n); az = alloc_double(n);
    mass = alloc_double(n);
    dt_frost = alloc_double(n);
    level = static_cast<int*>(std::calloc(n, sizeof(int)));
    orig_index = static_cast<int*>(std::malloc(n * sizeof(int)));
    for (int i = 0; i < n; i++) orig_index[i] = i;
    level_start = nullptr;
    level_count = nullptr;
    n_levels = 0;

    // Scratch buffers
    scratch1 = alloc_double(n); scratch2 = alloc_double(n); scratch3 = alloc_double(n);
    scratch4 = alloc_double(n); scratch5 = alloc_double(n); scratch6 = alloc_double(n);
    scratch7 = alloc_double(n); scratch8 = alloc_double(n); scratch9 = alloc_double(n);
    scratch_d = alloc_double(n);
    scratch_i = static_cast<int*>(std::malloc(n * sizeof(int)));
    scratch_i2 = static_cast<int*>(std::malloc(n * sizeof(int)));
    level_cursor_cap = 64;
    level_cursor = static_cast<int*>(std::malloc(level_cursor_cap * sizeof(int)));
}

void Particles::deallocate() {
    std::free(x);  std::free(y);  std::free(z);
    std::free(vx); std::free(vy); std::free(vz);
    std::free(ax); std::free(ay); std::free(az);
    std::free(mass); std::free(dt_frost);
    std::free(level); std::free(orig_index);
    std::free(level_start); std::free(level_count);
    std::free(scratch1); std::free(scratch2); std::free(scratch3);
    std::free(scratch4); std::free(scratch5); std::free(scratch6);
    std::free(scratch7); std::free(scratch8); std::free(scratch9);
    std::free(scratch_d); std::free(scratch_i); std::free(scratch_i2);
    std::free(level_cursor);
    N = 0; n_levels = 0;
    x = y = z = vx = vy = vz = ax = ay = az = mass = dt_frost = nullptr;
    level = orig_index = level_start = level_count = nullptr;
    scratch1 = scratch2 = scratch3 = scratch4 = scratch5 = scratch6 = nullptr;
    scratch7 = scratch8 = scratch9 = scratch_d = nullptr;
    scratch_i = scratch_i2 = nullptr;
    level_cursor = nullptr; level_cursor_cap = 0;
}

void Particles::partition_by_level() {
    if (N == 0 || n_levels == 0) return;

    std::free(level_count);
    std::free(level_start);
    level_count = static_cast<int*>(std::calloc(n_levels, sizeof(int)));
    level_start = static_cast<int*>(std::calloc(n_levels, sizeof(int)));

    for (int i = 0; i < N; i++) {
        level_count[level[i]]++;
    }

    level_start[0] = 0;
    for (int l = 1; l < n_levels; l++) {
        level_start[l] = level_start[l-1] + level_count[l-1];
    }

    // Build permutation via counting sort
    int* perm = scratch_i;
    // Use pre-allocated level_cursor, grow if needed
    if (n_levels > level_cursor_cap) {
        std::free(level_cursor);
        level_cursor_cap = n_levels;
        level_cursor = static_cast<int*>(std::malloc(level_cursor_cap * sizeof(int)));
    }
    int* cursor = level_cursor;
    for (int l = 0; l < n_levels; l++) cursor[l] = level_start[l];
    for (int i = 0; i < N; i++) {
        int l = level[i];
        perm[cursor[l]++] = i;
    }

    // Apply permutation to all double arrays (using scratch_d as temp)
    double* tmp = scratch_d;
    auto apply_perm = [&](double* arr) {
        for (int i = 0; i < N; i++) tmp[i] = arr[perm[i]];
        std::memcpy(arr, tmp, N * sizeof(double));
    };

    apply_perm(x);  apply_perm(y);  apply_perm(z);
    apply_perm(vx); apply_perm(vy); apply_perm(vz);
    apply_perm(ax); apply_perm(ay); apply_perm(az);
    apply_perm(mass);
    apply_perm(dt_frost);

    // Permute int arrays (reuse scratch_i2/cursor as temp — done with it)
    int* tmp_i = scratch_i2;
    auto apply_perm_int = [&](int* arr) {
        for (int i = 0; i < N; i++) tmp_i[i] = arr[perm[i]];
        std::memcpy(arr, tmp_i, N * sizeof(int));
    };

    // Must permute level before orig_index since perm uses scratch_i
    // Actually perm is read-only here, so order doesn't matter
    apply_perm_int(level);
    apply_perm_int(orig_index);

    // cursor is pre-allocated, no free needed
}
