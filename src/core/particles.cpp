#include "particles.h"
#include <cstdlib>
#include <cstring>

void Particles::allocate(int n, bool collisional_mode) {
    N = n;
    collisional = collisional_mode;
    x  = Kokkos::View<double*>("x",  n);
    y  = Kokkos::View<double*>("y",  n);
    z  = Kokkos::View<double*>("z",  n);
    vx = Kokkos::View<double*>("vx", n);
    vy = Kokkos::View<double*>("vy", n);
    vz = Kokkos::View<double*>("vz", n);
    cvx = Kokkos::View<double*>("cvx", n);
    cvy = Kokkos::View<double*>("cvy", n);
    cvz = Kokkos::View<double*>("cvz", n);
    ax = Kokkos::View<double*>("ax", n);
    ay = Kokkos::View<double*>("ay", n);
    az = Kokkos::View<double*>("az", n);
    mass     = Kokkos::View<double*>("mass", n);
    dt_adapt = Kokkos::View<double*>("dt_adapt", n);
    level      = Kokkos::View<int*>("level", n);
    orig_index = Kokkos::View<int*>("orig_index", n);
    ix = Kokkos::View<int64_t*>("ix", n);
    iy = Kokkos::View<int64_t*>("iy", n);
    iz = Kokkos::View<int64_t*>("iz", n);

    // Initialize orig_index on host then copy
    {
        auto h = Kokkos::create_mirror_view(orig_index);
        for (int i = 0; i < n; i++) h(i) = i;
        Kokkos::deep_copy(orig_index, h);
    }

    level_start.clear();
    level_count.clear();
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
    scratch_l1 = Kokkos::View<int64_t*>("sl1", n);
    scratch_l2 = Kokkos::View<int64_t*>("sl2", n);
    scratch_l3 = Kokkos::View<int64_t*>("sl3", n);

    // Packed views for vectorized memory access (LayoutRight for double4 loads)
    pos_packed = Kokkos::View<double*[4], Kokkos::LayoutRight>("pos_packed", n);
    vel_packed = Kokkos::View<double*[4], Kokkos::LayoutRight>("vel_packed", n);

    // Collisional fields (only allocated when needed)
    if (collisional_mode) {
        particle_id   = Kokkos::View<int*>("particle_id", n);
        subsystem_id  = Kokkos::View<int*>("subsystem_id", n);
        stellar_type  = Kokkos::View<int*>("stellar_type", n);
        m_zams        = Kokkos::View<double*>("m_zams", n);
        stellar_age   = Kokkos::View<double*>("stellar_age", n);
        stellar_radius = Kokkos::View<double*>("stellar_radius", n);
        luminosity    = Kokkos::View<double*>("luminosity", n);
        core_mass     = Kokkos::View<double*>("core_mass", n);
        spin_vec      = Kokkos::View<double*[3], Kokkos::LayoutRight>("spin_vec", n);
        spin_mag      = Kokkos::View<double*>("spin_mag", n);
        dt_level0     = Kokkos::View<double*>("dt_level0", n);
        grad_mag      = Kokkos::View<double*>("grad_mag", n);

        // Initialize particle_id = 0..N-1
        auto pid = particle_id;
        Kokkos::parallel_for("init_pid", n, KOKKOS_LAMBDA(int i) { pid(i) = i; });

        // Initialize subsystem_id = -1 (all field particles)
        auto sid = subsystem_id;
        Kokkos::parallel_for("init_sid", n, KOKKOS_LAMBDA(int i) { sid(i) = -1; });
    }
}

void Particles::deallocate() {
    // Kokkos Views are reference-counted; reassigning releases memory
    x = y = z = Kokkos::View<double*>();
    vx = vy = vz = Kokkos::View<double*>();
    cvx = cvy = cvz = Kokkos::View<double*>();
    ax = ay = az = Kokkos::View<double*>();
    mass = dt_adapt = Kokkos::View<double*>();
    level = orig_index = Kokkos::View<int*>();
    scratch1 = scratch2 = scratch3 = Kokkos::View<double*>();
    scratch4 = scratch5 = scratch6 = Kokkos::View<double*>();
    scratch7 = scratch8 = scratch9 = Kokkos::View<double*>();
    scratch10 = scratch11 = Kokkos::View<double*>();
    scratch_i = scratch_i2 = scratch_i3 = Kokkos::View<int*>();
    ix = iy = iz = Kokkos::View<int64_t*>();
    scratch_l1 = scratch_l2 = scratch_l3 = Kokkos::View<int64_t*>();
    d_level_dts = Kokkos::View<double*>();
    pos_packed = Kokkos::View<double*[4], Kokkos::LayoutRight>();
    vel_packed = Kokkos::View<double*[4], Kokkos::LayoutRight>();

    // Collisional fields
    particle_id = Kokkos::View<int*>();
    subsystem_id = Kokkos::View<int*>();
    stellar_type = Kokkos::View<int*>();
    m_zams = stellar_age = stellar_radius = luminosity = core_mass = Kokkos::View<double*>();
    spin_vec = Kokkos::View<double*[3], Kokkos::LayoutRight>();
    spin_mag = dt_level0 = grad_mag = Kokkos::View<double*>();

    level_start.clear();
    level_count.clear();
    N = 0; n_levels = 0;
    collisional = false;
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

}

void Particles::pack_pos_range(int start, int count) {
    auto px = x; auto py = y; auto pz = z; auto pm = mass;
    auto pos = pos_packed;
    Kokkos::parallel_for("pack_pos_range", count, KOKKOS_LAMBDA(int ii) {
        int i = start + ii;
        pos(i, 0) = px(i);
        pos(i, 1) = py(i);
        pos(i, 2) = pz(i);
        pos(i, 3) = pm(i);
    });

}

void Particles::pack_vel_range(int start, int count) {
    auto pvx = vx; auto pvy = vy; auto pvz = vz;
    auto vel = vel_packed;
    Kokkos::parallel_for("pack_vel_range", count, KOKKOS_LAMBDA(int ii) {
        int i = start + ii;
        vel(i, 0) = pvx(i);
        vel(i, 1) = pvy(i);
        vel(i, 2) = pvz(i);
        vel(i, 3) = 0.0;
    });

}

int Particles::compact_remove_zero_mass() {
    // Copy mass to host to find which particles to keep
    auto h_m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mass);
    int N_old = N;

    // Build keep list (indices of particles with mass > 0)
    std::vector<int> keep;
    keep.reserve(N);
    for (int i = 0; i < N; i++) {
        if (h_m(i) > 0.0) keep.push_back(i);
    }
    int N_new = (int)keep.size();
    int n_removed = N_old - N_new;
    if (n_removed == 0) return 0;

    // Mirror all fields to host
    auto h_x  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
    auto h_y  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
    auto h_z  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), z);
    auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vx);
    auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vy);
    auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vz);
    auto h_ax = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ax);
    auto h_ay = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ay);
    auto h_az = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), az);
    auto h_dt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dt_adapt);
    auto h_lev = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), level);
    auto h_orig = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), orig_index);
    auto h_ix = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ix);
    auto h_iy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), iy);
    auto h_iz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), iz);
    auto h_cvx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cvx);
    auto h_cvy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cvy);
    auto h_cvz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cvz);

    // Compact in-place on host mirrors
    for (int j = 0; j < N_new; j++) {
        int src = keep[j];
        if (src == j) continue;
        h_x(j) = h_x(src); h_y(j) = h_y(src); h_z(j) = h_z(src);
        h_vx(j) = h_vx(src); h_vy(j) = h_vy(src); h_vz(j) = h_vz(src);
        h_ax(j) = h_ax(src); h_ay(j) = h_ay(src); h_az(j) = h_az(src);
        h_m(j) = h_m(src); h_dt(j) = h_dt(src);
        h_lev(j) = h_lev(src); h_orig(j) = h_orig(src);
        h_ix(j) = h_ix(src); h_iy(j) = h_iy(src); h_iz(j) = h_iz(src);
        h_cvx(j) = h_cvx(src); h_cvy(j) = h_cvy(src); h_cvz(j) = h_cvz(src);
    }
    Kokkos::deep_copy(cvx, h_cvx); Kokkos::deep_copy(cvy, h_cvy); Kokkos::deep_copy(cvz, h_cvz);

    // Copy back to device
    Kokkos::deep_copy(x, h_x); Kokkos::deep_copy(y, h_y); Kokkos::deep_copy(z, h_z);
    Kokkos::deep_copy(vx, h_vx); Kokkos::deep_copy(vy, h_vy); Kokkos::deep_copy(vz, h_vz);
    Kokkos::deep_copy(ax, h_ax); Kokkos::deep_copy(ay, h_ay); Kokkos::deep_copy(az, h_az);
    Kokkos::deep_copy(mass, h_m); Kokkos::deep_copy(dt_adapt, h_dt);
    Kokkos::deep_copy(level, h_lev); Kokkos::deep_copy(orig_index, h_orig);
    Kokkos::deep_copy(ix, h_ix); Kokkos::deep_copy(iy, h_iy); Kokkos::deep_copy(iz, h_iz);

    // Collisional fields
    if (collisional) {
        auto h_pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), particle_id);
        auto h_sid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), subsystem_id);
        auto h_stype = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), stellar_type);
        auto h_mzams = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m_zams);
        auto h_sage = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), stellar_age);
        auto h_srad = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), stellar_radius);
        auto h_lum = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), luminosity);
        auto h_cmass = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), core_mass);
        auto h_spin = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), spin_vec);
        auto h_smag = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), spin_mag);
        auto h_dtl0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dt_level0);
        auto h_gmag = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grad_mag);

        for (int j = 0; j < N_new; j++) {
            int src = keep[j];
            if (src == j) continue;
            h_pid(j) = h_pid(src); h_sid(j) = h_sid(src);
            h_stype(j) = h_stype(src);
            h_mzams(j) = h_mzams(src); h_sage(j) = h_sage(src);
            h_srad(j) = h_srad(src); h_lum(j) = h_lum(src);
            h_cmass(j) = h_cmass(src);
            h_spin(j, 0) = h_spin(src, 0);
            h_spin(j, 1) = h_spin(src, 1);
            h_spin(j, 2) = h_spin(src, 2);
            h_smag(j) = h_smag(src);
            h_dtl0(j) = h_dtl0(src); h_gmag(j) = h_gmag(src);
        }

        Kokkos::deep_copy(particle_id, h_pid);
        Kokkos::deep_copy(subsystem_id, h_sid);
        Kokkos::deep_copy(stellar_type, h_stype);
        Kokkos::deep_copy(m_zams, h_mzams);
        Kokkos::deep_copy(stellar_age, h_sage);
        Kokkos::deep_copy(stellar_radius, h_srad);
        Kokkos::deep_copy(luminosity, h_lum);
        Kokkos::deep_copy(core_mass, h_cmass);
        Kokkos::deep_copy(spin_vec, h_spin);
        Kokkos::deep_copy(spin_mag, h_smag);
        Kokkos::deep_copy(dt_level0, h_dtl0);
        Kokkos::deep_copy(grad_mag, h_gmag);
    }

    N = N_new;
    return n_removed;
}

void Particles::partition_by_level() {
    if (N == 0 || n_levels == 0) return;

    // Resize level_count / level_start on host
    level_count.assign(n_levels, 0);
    level_start.assign(n_levels, 0);

    // Step 1: Count particles per level and compute prefix sums
    // Use prefix scans per level for deterministic counting
    auto p_level = level;

    level_start[0] = 0;
    for (int l = 0; l < n_levels; l++) {
        int count_l = 0;
        Kokkos::parallel_reduce("count_level", N,
            KOKKOS_LAMBDA(int i, int& sum) {
                if (p_level(i) == l) sum += 1;
            }, count_l);
        level_count[l] = count_l;
        if (l > 0) level_start[l] = level_start[l-1] + level_count[l-1];
    }

    // Step 2: Build permutation DETERMINISTICALLY using prefix scans
    // (atomics would be non-deterministic across MPI ranks on different GPUs)
    auto perm = scratch_i;  // reuse scratch for permutation
    for (int l = 0; l < n_levels; l++) {
        int l_start = level_start[l];
        Kokkos::parallel_scan("level_perm_scan", N,
            KOKKOS_LAMBDA(int i, int& partial_sum, bool is_final) {
                int is_level = (p_level(i) == l) ? 1 : 0;
                if (is_final && is_level) {
                    perm(l_start + partial_sum) = i;
                }
                partial_sum += is_level;
            });
    }


    // Step 3: Apply permutation with fused gather/copy (2 kernels instead of 26)
    auto px = x; auto py = y; auto pz = z;
    auto pvx = vx; auto pvy = vy; auto pvz = vz;
    auto pax = ax; auto pay = ay; auto paz = az;
    auto pmass = mass; auto pdt = dt_adapt;
    auto porig = orig_index;
    int NN = N;

    // Scratch arrays for all 13 fields
    auto t1 = scratch1;  auto t2 = scratch2;  auto t3 = scratch3;
    auto t4 = scratch4;  auto t5 = scratch5;  auto t6 = scratch6;
    auto t7 = scratch7;  auto t8 = scratch8;  auto t9 = scratch9;
    auto t10 = scratch10; auto t11 = scratch11;
    auto ti1 = scratch_i2; auto ti2 = scratch_i3;
    auto pix = ix; auto piy = iy; auto piz = iz;
    auto tl1 = scratch_l1; auto tl2 = scratch_l2; auto tl3 = scratch_l3;

    // Single fused gather of all fields
    Kokkos::parallel_for("gather_all", NN, KOKKOS_LAMBDA(int i) {
        int src = perm(i);
        t1(i) = px(src);   t2(i) = py(src);   t3(i) = pz(src);
        t4(i) = pvx(src);  t5(i) = pvy(src);  t6(i) = pvz(src);
        t7(i) = pax(src);  t8(i) = pay(src);  t9(i) = paz(src);
        t10(i) = pmass(src); t11(i) = pdt(src);
        ti1(i) = porig(src); ti2(i) = p_level(src);
        tl1(i) = pix(src); tl2(i) = piy(src); tl3(i) = piz(src);
    });


    // Single fused copy back
    Kokkos::parallel_for("copy_all", NN, KOKKOS_LAMBDA(int i) {
        px(i) = t1(i);   py(i) = t2(i);   pz(i) = t3(i);
        pvx(i) = t4(i);  pvy(i) = t5(i);  pvz(i) = t6(i);
        pax(i) = t7(i);  pay(i) = t8(i);  paz(i) = t9(i);
        pmass(i) = t10(i); pdt(i) = t11(i);
        porig(i) = ti1(i); p_level(i) = ti2(i);
        pix(i) = tl1(i); piy(i) = tl2(i); piz(i) = tl3(i);
    });

    // Kick compensation follows its particle (t1-t3 are free after the copy-back)
    auto pcx = cvx; auto pcy = cvy; auto pcz = cvz;
    Kokkos::parallel_for("gather_comp", NN, KOKKOS_LAMBDA(int i) {
        int src = perm(i);
        t1(i) = pcx(src); t2(i) = pcy(src); t3(i) = pcz(src);
    });
    Kokkos::parallel_for("copy_comp", NN, KOKKOS_LAMBDA(int i) {
        pcx(i) = t1(i); pcy(i) = t2(i); pcz(i) = t3(i);
    });


    // Second pass: permute collisional-only fields (avoids overhead in softened mode)
    if (collisional) {
        auto ppid = particle_id;   auto psid = subsystem_id;
        auto pstype = stellar_type;
        auto pmzams = m_zams;      auto psage = stellar_age;
        auto psrad = stellar_radius; auto plum = luminosity;
        auto pcmass = core_mass;   auto psmag = spin_mag;
        auto pdtl0 = dt_level0;   auto pgmag = grad_mag;
        auto pspin = spin_vec;

        // Gather collisional fields using scratch (reuse t1-t10 for doubles, ti1-ti2 for ints)
        // Pass 1: ints (particle_id, subsystem_id, stellar_type) + doubles (m_zams, stellar_age, stellar_radius, luminosity, core_mass, spin_mag, dt_level0, grad_mag)
        Kokkos::parallel_for("gather_coll", NN, KOKKOS_LAMBDA(int i) {
            int src = perm(i);
            ti1(i) = ppid(src);
            ti2(i) = psid(src);
            tl1(i) = pstype(src);  // reuse int64 scratch for int
            t1(i) = pmzams(src);
            t2(i) = psage(src);
            t3(i) = psrad(src);
            t4(i) = plum(src);
            t5(i) = pcmass(src);
            t6(i) = psmag(src);
            t7(i) = pdtl0(src);
            t8(i) = pgmag(src);
        });

        Kokkos::parallel_for("copy_coll", NN, KOKKOS_LAMBDA(int i) {
            ppid(i) = ti1(i);
            psid(i) = ti2(i);
            pstype(i) = (int)tl1(i);
            pmzams(i) = t1(i);
            psage(i) = t2(i);
            psrad(i) = t3(i);
            plum(i) = t4(i);
            pcmass(i) = t5(i);
            psmag(i) = t6(i);
            pdtl0(i) = t7(i);
            pgmag(i) = t8(i);
        });

        // Spin vector (3 components) — separate pass using t1-t3
        Kokkos::parallel_for("gather_spin", NN, KOKKOS_LAMBDA(int i) {
            int src = perm(i);
            t1(i) = pspin(src, 0);
            t2(i) = pspin(src, 1);
            t3(i) = pspin(src, 2);
        });
        Kokkos::parallel_for("copy_spin", NN, KOKKOS_LAMBDA(int i) {
            pspin(i, 0) = t1(i);
            pspin(i, 1) = t2(i);
            pspin(i, 2) = t3(i);
        });
    }
}

void Particles::zero_kick_compensation(int start, int count) {
    if (count < 0) count = N - start;
    auto c0 = cvx; auto c1 = cvy; auto c2 = cvz;
    Kokkos::parallel_for("zero_kick_comp", count, KOKKOS_LAMBDA(int ii) {
        c0(start + ii) = 0.0; c1(start + ii) = 0.0; c2(start + ii) = 0.0;
    });
}
