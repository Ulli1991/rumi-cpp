#include "forces.h"
#include "constants.h"
#include "integrator.h"
#include <Kokkos_Core.hpp>
#include <cfloat>

// TILE_SIZE defined in constants.h

// ==========================================================================
// Full N^2 acceleration + adaptive timestep (fused)
// ==========================================================================
void compute_acceleration_and_timestep(Particles& p, double G, double soft2, double eta) {
    const int N = p.N;
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView m{p.mass.data()};
    RawView dt_f{p.dt_adapt.data()};

    // Tiled version using shared memory with packed vector loads
    // Each team of TILE_SIZE threads handles TILE_SIZE particles (i)
    // All threads collaborate to load tiles of j particles
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    // Pack SoA data into vectorized format for efficient global memory access
    p.pack();

    size_t scratch_size = TILE_SIZE * 7 * sizeof(double);
    int num_teams = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Use packed views for vectorized global memory access
    auto pos = p.pos_packed;
    auto vel = p.vel_packed;

    Kokkos::parallel_for("acc_dt_tiled",
        team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int i = team.league_rank() * TILE_SIZE + team.team_rank();

            // AoS interleaved tile: [x,y,z,m,vx,vy,vz] per j-particle
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 7);

            double xi = 0, yi = 0, zi = 0;
            double vxi = 0, vyi = 0, vzi = 0;
            double mi = 0, pad = 0;
            if (i < N) {
                load4(&pos(i, 0), xi, yi, zi, mi);
                load4(&vel(i, 0), vxi, vyi, vzi, pad);
            }

            double axi = 0, ayi = 0, azi = 0;
            double t_min_i = DBL_MAX;

            for (int tile_start = 0; tile_start < N; tile_start += TILE_SIZE) {
                int tile_end = (tile_start + TILE_SIZE < N) ? tile_start + TILE_SIZE : N;
                int tile_count = tile_end - tile_start;

                // Collaborative AoS tile load with vectorized reads
                int t = team.team_rank();
                if (t < tile_count) {
                    int idx = tile_start + t;
                    int base = t * 7;
                    double vpad;
                    load4(&pos(idx, 0), tile(base), tile(base+1), tile(base+2), tile(base+3));
                    load4(&vel(idx, 0), tile(base+4), tile(base+5), tile(base+6), vpad);
                }
                team.team_barrier();

                if (i < N) {
                    for (int jt = 0; jt < tile_count; jt++) {
                        int j = tile_start + jt;
                        if (j == i) continue;

                        int jb = jt * 7;
                        double dx = tile(jb    ) - xi;
                        double dy = tile(jb + 1) - yi;
                        double dz = tile(jb + 2) - zi;
                        double mj = tile(jb + 3);
                        double r2 = dx*dx + dy*dy + dz*dz;
                        double r2s = r2 + soft2;
                        double inv_r = fast_rsqrt(r2s);
                        double inv_r3 = inv_r * inv_r * inv_r;

                        axi += G * mj * inv_r3 * dx;
                        ayi += G * mj * inv_r3 * dy;
                        azi += G * mj * inv_r3 * dz;

                        // Adaptive timestep: reuse inv_r, no extra sqrt
                        double dvx = tile(jb + 4) - vxi;
                        double dvy = tile(jb + 5) - vyi;
                        double dvz = tile(jb + 6) - vzi;
                        double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
                        double mass_sum = mi + mj;

                        // r2s^1.5 = r2s * r2s * inv_r (reuses rsqrt)
                        double r3 = r2s * r2s * inv_r;
                        double t_ff_sq = r3 / (G * mass_sum + EPS);
                        double t_fb_sq = r2s / (v_sq + EPS);
                        double t_min_sq_pair = Kokkos::fmin(t_ff_sq, t_fb_sq);
                        t_min_i = Kokkos::fmin(t_min_i, t_min_sq_pair);
                    }
                }
                team.team_barrier();
            }

            if (i < N) {
                pax(i) = axi; pay(i) = ayi; paz(i) = azi;
                dt_f(i) = Kokkos::sqrt(t_min_i);
            }
        });
    Kokkos::fence();

    // Apply eta (tiled path already applied sqrt)
    Kokkos::parallel_for("apply_eta", N, KOKKOS_LAMBDA(int i) {
        dt_f(i) *= eta;
    });
    Kokkos::fence();
}

// ==========================================================================
// Self-interaction: [start, start+count)
// ==========================================================================
void self_acceleration(Particles& p, int start, int count,
                       double G, double soft2) {
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView m{p.mass.data()};
    const int end = start + count;

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    size_t scratch_size = TILE_SIZE * 4 * sizeof(double);
    int num_teams = (count + TILE_SIZE - 1) / TILE_SIZE;

    Kokkos::parallel_for("self_acc_tiled",
        team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * TILE_SIZE + team.team_rank();
            int i = start + ii;
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 4);

            double xi = 0, yi = 0, zi = 0;
            if (ii < count) { xi = px(i); yi = py(i); zi = pz(i); }
            double axi = 0, ayi = 0, azi = 0;

            for (int tile_start = start; tile_start < end; tile_start += TILE_SIZE) {
                int tile_count = ((tile_start + TILE_SIZE) < end) ? TILE_SIZE : (end - tile_start);
                int t = team.team_rank();
                if (t < tile_count) {
                    int j = tile_start + t;
                    tile(t*4+0) = px(j); tile(t*4+1) = py(j);
                    tile(t*4+2) = pz(j); tile(t*4+3) = m(j);
                }
                team.team_barrier();
                if (ii < count) {
                    for (int jt = 0; jt < tile_count; jt++) {
                        int j = tile_start + jt;
                        if (j == i) continue;
                        double dx = tile(jt*4+0) - xi;
                        double dy = tile(jt*4+1) - yi;
                        double dz = tile(jt*4+2) - zi;
                        double r2 = dx*dx + dy*dy + dz*dz + soft2;
                        double inv_r = fast_rsqrt(r2);
                        double inv_r3 = inv_r * inv_r * inv_r;
                        double fj = G * tile(jt*4+3) * inv_r3;
                        axi += fj * dx; ayi += fj * dy; azi += fj * dz;
                    }
                }
                team.team_barrier();
            }
            if (ii < count) { pax(i) = axi; pay(i) = ayi; paz(i) = azi; }
        });
    Kokkos::fence();
}

// ==========================================================================
// Self gradient correction
// ==========================================================================
void self_gradient_correction(Particles& p, int start, int count,
                              double G, double soft2, double dt) {
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView m{p.mass.data()};
    const double coeff = G * dt * dt * GRAD_COEFF;



    // Snapshot accelerations into scratch
    RawView ax0{p.scratch1.data()}, ay0{p.scratch2.data()}, az0{p.scratch3.data()};
    Kokkos::parallel_for("snap_self_grad", count, KOKKOS_LAMBDA(int ii) {
        ax0(ii) = pax(start + ii);
        ay0(ii) = pay(start + ii);
        az0(ii) = paz(start + ii);
    });
    Kokkos::fence();

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    size_t scratch_size = TILE_SIZE * 7 * sizeof(double);  // [x,y,z,m,ax0,ay0,az0]
    int num_teams = (count + TILE_SIZE - 1) / TILE_SIZE;

    Kokkos::parallel_for("self_grad_tiled",
        team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * TILE_SIZE + team.team_rank();
            int i = start + ii;
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 7);

            double xi = 0, yi = 0, zi = 0;
            double axi0 = 0, ayi0 = 0, azi0 = 0;
            if (ii < count) {
                xi = px(i); yi = py(i); zi = pz(i);
                axi0 = ax0(ii); ayi0 = ay0(ii); azi0 = az0(ii);
            }
            double cx = 0, cy = 0, cz = 0;

            for (int tile_idx = 0; tile_idx < count; tile_idx += TILE_SIZE) {
                int tile_count = ((tile_idx + TILE_SIZE) < count) ? TILE_SIZE : (count - tile_idx);
                int t = team.team_rank();
                if (t < tile_count) {
                    int jj = tile_idx + t;
                    int j = start + jj;
                    int base = t * 7;
                    tile(base+0) = px(j); tile(base+1) = py(j);
                    tile(base+2) = pz(j); tile(base+3) = m(j);
                    tile(base+4) = ax0(jj); tile(base+5) = ay0(jj);
                    tile(base+6) = az0(jj);
                }
                team.team_barrier();
                if (ii < count) {
                    for (int jt = 0; jt < tile_count; jt++) {
                        int jj = tile_idx + jt;
                        if (jj == ii) continue;
                        int jb = jt * 7;
                        double dx = tile(jb+0) - xi;
                        double dy = tile(jb+1) - yi;
                        double dz = tile(jb+2) - zi;
                        double dax = tile(jb+4) - axi0;
                        double day = tile(jb+5) - ayi0;
                        double daz = tile(jb+6) - azi0;
                        double r2 = dx*dx + dy*dy + dz*dz + soft2;
                        double inv_r = fast_rsqrt(r2);
                        double inv_r2 = inv_r * inv_r;
                        double inv_r5 = inv_r2 * inv_r2 * inv_r;
                        double r_dot_a = dx*dax + dy*day + dz*daz;
                        double f = coeff * tile(jb+3) * inv_r5;
                        cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                        cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                        cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
                    }
                }
                team.team_barrier();
            }
            if (ii < count) { pax(i) += cx; pay(i) += cy; paz(i) += cz; }
        });
    Kokkos::fence();
}

// ==========================================================================
// Fused self_acceleration + self_gradient_correction
// ==========================================================================
void self_acceleration_and_gradient(Particles& p, int start, int count,
                                    double G, double soft2, double dt) {
    self_acceleration(p, start, count, G, soft2);
    self_gradient_correction(p, start, count, G, soft2, dt);
}

// ==========================================================================
// Cross-interaction: group A x group B
// ==========================================================================
void cross_acceleration(Particles& p,
                        int a_start, int a_count,
                        int b_start, int b_count,
                        double G, double soft2) {
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView m{p.mass.data()};
    const int b_end = b_start + b_count;
    const int a_end = a_start + a_count;

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    size_t scratch_size = TILE_SIZE * 4 * sizeof(double);

    // A from B
    int num_teams_a = (a_count + TILE_SIZE - 1) / TILE_SIZE;
    Kokkos::parallel_for("cross_acc_a_tiled",
        team_policy(num_teams_a, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * TILE_SIZE + team.team_rank();
            int i = a_start + ii;
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 4);

            double xi = 0, yi = 0, zi = 0;
            if (ii < a_count) { xi = px(i); yi = py(i); zi = pz(i); }
            double axi = 0, ayi = 0, azi = 0;

            for (int tile_start = b_start; tile_start < b_end; tile_start += TILE_SIZE) {
                int tile_count = ((tile_start + TILE_SIZE) < b_end) ? TILE_SIZE : (b_end - tile_start);
                int t = team.team_rank();
                if (t < tile_count) {
                    int j = tile_start + t;
                    tile(t*4+0) = px(j); tile(t*4+1) = py(j);
                    tile(t*4+2) = pz(j); tile(t*4+3) = m(j);
                }
                team.team_barrier();
                if (ii < a_count) {
                    for (int jt = 0; jt < tile_count; jt++) {
                        double dx = tile(jt*4+0) - xi;
                        double dy = tile(jt*4+1) - yi;
                        double dz = tile(jt*4+2) - zi;
                        double r2 = dx*dx + dy*dy + dz*dz + soft2;
                        double inv_r = fast_rsqrt(r2);
                        double inv_r3 = inv_r * inv_r * inv_r;
                        double fj = G * tile(jt*4+3) * inv_r3;
                        axi += fj * dx; ayi += fj * dy; azi += fj * dz;
                    }
                }
                team.team_barrier();
            }
            if (ii < a_count) { pax(i) = axi; pay(i) = ayi; paz(i) = azi; }
        });
    // B from A
    int num_teams_b = (b_count + TILE_SIZE - 1) / TILE_SIZE;
    Kokkos::parallel_for("cross_acc_b_tiled",
        team_policy(num_teams_b, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * TILE_SIZE + team.team_rank();
            int j = b_start + ii;
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 4);

            double xj = 0, yj = 0, zj = 0;
            if (ii < b_count) { xj = px(j); yj = py(j); zj = pz(j); }
            double axj = 0, ayj = 0, azj = 0;

            for (int tile_start = a_start; tile_start < a_end; tile_start += TILE_SIZE) {
                int tile_count = ((tile_start + TILE_SIZE) < a_end) ? TILE_SIZE : (a_end - tile_start);
                int t = team.team_rank();
                if (t < tile_count) {
                    int i2 = tile_start + t;
                    tile(t*4+0) = px(i2); tile(t*4+1) = py(i2);
                    tile(t*4+2) = pz(i2); tile(t*4+3) = m(i2);
                }
                team.team_barrier();
                if (ii < b_count) {
                    for (int it = 0; it < tile_count; it++) {
                        double dx = tile(it*4+0) - xj;
                        double dy = tile(it*4+1) - yj;
                        double dz = tile(it*4+2) - zj;
                        double r2 = dx*dx + dy*dy + dz*dz + soft2;
                        double inv_r = fast_rsqrt(r2);
                        double inv_r3 = inv_r * inv_r * inv_r;
                        double fi = G * tile(it*4+3) * inv_r3;
                        axj += fi * dx; ayj += fi * dy; azj += fi * dz;
                    }
                }
                team.team_barrier();
            }
            if (ii < b_count) { pax(j) = axj; pay(j) = ayj; paz(j) = azj; }
        });
    Kokkos::fence();
}

// ==========================================================================
// Cross gradient correction
// ==========================================================================
void cross_gradient_correction(Particles& p,
                               int a_start, int a_count,
                               int b_start, int b_count,
                               double G, double soft2, double dt) {
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView m{p.mass.data()};
    const double coeff = G * dt * dt * GRAD_COEFF;



    // Snapshot accelerations
    RawView ax0_a{p.scratch1.data()}, ay0_a{p.scratch2.data()}, az0_a{p.scratch3.data()};
    RawView ax0_b{p.scratch4.data()}, ay0_b{p.scratch5.data()}, az0_b{p.scratch6.data()};
    Kokkos::parallel_for("snap_cross_a", a_count, KOKKOS_LAMBDA(int ii) {
        ax0_a(ii) = pax(a_start + ii);
        ay0_a(ii) = pay(a_start + ii);
        az0_a(ii) = paz(a_start + ii);
    });
    Kokkos::parallel_for("snap_cross_b", b_count, KOKKOS_LAMBDA(int jj) {
        ax0_b(jj) = pax(b_start + jj);
        ay0_b(jj) = pay(b_start + jj);
        az0_b(jj) = paz(b_start + jj);
    });
    Kokkos::fence();

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    size_t scratch_size = TILE_SIZE * 7 * sizeof(double);  // [x,y,z,m,ax0,ay0,az0]

    // A from B: tile over B particles
    int num_teams_a = (a_count + TILE_SIZE - 1) / TILE_SIZE;
    Kokkos::parallel_for("cross_grad_a_tiled",
        team_policy(num_teams_a, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * TILE_SIZE + team.team_rank();
            int i = a_start + ii;
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 7);

            double xi = 0, yi = 0, zi = 0;
            double axi0 = 0, ayi0 = 0, azi0 = 0;
            if (ii < a_count) {
                xi = px(i); yi = py(i); zi = pz(i);
                axi0 = ax0_a(ii); ayi0 = ay0_a(ii); azi0 = az0_a(ii);
            }
            double cx = 0, cy = 0, cz = 0;

            for (int tile_idx = 0; tile_idx < b_count; tile_idx += TILE_SIZE) {
                int tile_count = ((tile_idx + TILE_SIZE) < b_count) ? TILE_SIZE : (b_count - tile_idx);
                int t = team.team_rank();
                if (t < tile_count) {
                    int jj = tile_idx + t;
                    int j = b_start + jj;
                    int base = t * 7;
                    tile(base+0) = px(j); tile(base+1) = py(j);
                    tile(base+2) = pz(j); tile(base+3) = m(j);
                    tile(base+4) = ax0_b(jj); tile(base+5) = ay0_b(jj);
                    tile(base+6) = az0_b(jj);
                }
                team.team_barrier();
                if (ii < a_count) {
                    for (int jt = 0; jt < tile_count; jt++) {
                        int jb = jt * 7;
                        double dx = tile(jb+0) - xi;
                        double dy = tile(jb+1) - yi;
                        double dz = tile(jb+2) - zi;
                        double dax = tile(jb+4) - axi0;
                        double day = tile(jb+5) - ayi0;
                        double daz = tile(jb+6) - azi0;
                        double r2 = dx*dx + dy*dy + dz*dz + soft2;
                        double inv_r = fast_rsqrt(r2);
                        double inv_r2 = inv_r * inv_r;
                        double inv_r5 = inv_r2 * inv_r2 * inv_r;
                        double r_dot_a = dx*dax + dy*day + dz*daz;
                        double f = coeff * tile(jb+3) * inv_r5;
                        cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                        cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                        cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
                    }
                }
                team.team_barrier();
            }
            if (ii < a_count) { pax(i) += cx; pay(i) += cy; paz(i) += cz; }
        });
    // B from A: tile over A particles
    int num_teams_b = (b_count + TILE_SIZE - 1) / TILE_SIZE;
    Kokkos::parallel_for("cross_grad_b_tiled",
        team_policy(num_teams_b, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * TILE_SIZE + team.team_rank();
            int j = b_start + ii;
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 7);

            double xj = 0, yj = 0, zj = 0;
            double axj0 = 0, ayj0 = 0, azj0 = 0;
            if (ii < b_count) {
                xj = px(j); yj = py(j); zj = pz(j);
                axj0 = ax0_b(ii); ayj0 = ay0_b(ii); azj0 = az0_b(ii);
            }
            double cx = 0, cy = 0, cz = 0;

            for (int tile_idx = 0; tile_idx < a_count; tile_idx += TILE_SIZE) {
                int tile_count = ((tile_idx + TILE_SIZE) < a_count) ? TILE_SIZE : (a_count - tile_idx);
                int t = team.team_rank();
                if (t < tile_count) {
                    int ai = tile_idx + t;
                    int i2 = a_start + ai;
                    int base = t * 7;
                    tile(base+0) = px(i2); tile(base+1) = py(i2);
                    tile(base+2) = pz(i2); tile(base+3) = m(i2);
                    tile(base+4) = ax0_a(ai); tile(base+5) = ay0_a(ai);
                    tile(base+6) = az0_a(ai);
                }
                team.team_barrier();
                if (ii < b_count) {
                    for (int it = 0; it < tile_count; it++) {
                        int ib = it * 7;
                        double dx = tile(ib+0) - xj;
                        double dy = tile(ib+1) - yj;
                        double dz = tile(ib+2) - zj;
                        double dax = tile(ib+4) - axj0;
                        double day = tile(ib+5) - ayj0;
                        double daz = tile(ib+6) - azj0;
                        double r2 = dx*dx + dy*dy + dz*dz + soft2;
                        double inv_r = fast_rsqrt(r2);
                        double inv_r2 = inv_r * inv_r;
                        double inv_r5 = inv_r2 * inv_r2 * inv_r;
                        double r_dot_a = dx*dax + dy*day + dz*daz;
                        double f = coeff * tile(ib+3) * inv_r5;
                        cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                        cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                        cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
                    }
                }
                team.team_barrier();
            }
            if (ii < b_count) { pax(j) += cx; pay(j) += cy; paz(j) += cz; }
        });
    Kokkos::fence();
}

// ==========================================================================
// Fused cross_acceleration + cross_gradient_correction
// ==========================================================================
void cross_acceleration_and_gradient(Particles& p,
                                     int a_start, int a_count,
                                     int b_start, int b_count,
                                     double G, double soft2, double dt) {
    cross_acceleration(p, a_start, a_count, b_start, b_count, G, soft2);
    cross_gradient_correction(p, a_start, a_count, b_start, b_count, G, soft2, dt);
}

// ==========================================================================
// Compute adaptive timesteps for a subset [start, start+count)
// Only considers mutual interactions within the subset (for HHS recursion)
// ==========================================================================
void compute_timestep_subset(Particles& p, int start, int count,
                          double G, double soft2, double eta) {
    if (count <= 1) return;  // No pairs to consider

    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    RawView m{p.mass.data()};
    RawView dt_f{p.dt_adapt.data()};
    const int end = start + count;

    // Initialize dt_adapt to large value for subset
    Kokkos::parallel_for("init_dt_subset", count, KOKKOS_LAMBDA(int ii) {
        dt_f(start + ii) = DBL_MAX;
    });
    Kokkos::fence();

    // Full N^2 within subset - tiled with shared memory, squared-space timestep
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t scratch_size = TILE_SIZE * 7 * sizeof(double);
    int num_teams = (count + TILE_SIZE - 1) / TILE_SIZE;

    Kokkos::parallel_for("dt_subset_tiled",
        team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * TILE_SIZE + team.team_rank();
            int i = start + ii;

            // AoS interleaved tile: [x,y,z,m,vx,vy,vz] per j-particle
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 7);

            double xi = 0, yi = 0, zi = 0;
            double vxi = 0, vyi = 0, vzi = 0;
            double mi = 0;
            if (ii < count) {
                xi = px(i); yi = py(i); zi = pz(i);
                vxi = pvx(i); vyi = pvy(i); vzi = pvz(i);
                mi = m(i);
            }

            double t_min_sq = 1e60;

            for (int tile_st = start; tile_st < end; tile_st += TILE_SIZE) {
                int tile_end_idx = (tile_st + TILE_SIZE < end) ? tile_st + TILE_SIZE : end;
                int tile_cnt = tile_end_idx - tile_st;

                int t = team.team_rank();
                if (t < tile_cnt) {
                    int idx = tile_st + t;
                    int base = t * 7;
                    tile(base    ) = px(idx);
                    tile(base + 1) = py(idx);
                    tile(base + 2) = pz(idx);
                    tile(base + 3) = m(idx);
                    tile(base + 4) = pvx(idx);
                    tile(base + 5) = pvy(idx);
                    tile(base + 6) = pvz(idx);
                }
                team.team_barrier();

                if (ii < count) {
                    for (int jt = 0; jt < tile_cnt; jt++) {
                        int j = tile_st + jt;
                        if (j == i) continue;

                        int jb = jt * 7;
                        double dx = tile(jb    ) - xi;
                        double dy = tile(jb + 1) - yi;
                        double dz = tile(jb + 2) - zi;
                        double mj = tile(jb + 3);
                        double r2 = dx*dx + dy*dy + dz*dz;
                        double r2s = r2 + soft2;

                        double dvx = tile(jb + 4) - vxi;
                        double dvy = tile(jb + 5) - vyi;
                        double dvz = tile(jb + 6) - vzi;
                        double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;

                        double inv_r = fast_rsqrt(r2s);
                        double mass_sum = mi + mj;
                        double r3 = r2s * r2s * inv_r;
                        double t_ff_sq = r3 / (G * mass_sum + EPS);
                        double t_fb_sq = r2s / (v_sq + EPS);

                        double t_min_sq_pair = Kokkos::fmin(t_ff_sq, t_fb_sq);
                        t_min_sq = Kokkos::fmin(t_min_sq, t_min_sq_pair);
                    }
                }
                team.team_barrier();
            }

            if (ii < count) {
                dt_f(i) = Kokkos::sqrt(t_min_sq) * eta;
            }
        });
    Kokkos::fence();
}

// ==========================================================================
// Fused self_acceleration + kick (reduces kernel launches)
// ==========================================================================
void self_acceleration_kick(Particles& p, int start, int count,
                            double G, double soft2, double kick_dt) {
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView m{p.mass.data()};
    const int end = start + count;

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    size_t scratch_size = TILE_SIZE * 4 * sizeof(double);
    int num_teams = (count + TILE_SIZE - 1) / TILE_SIZE;

    Kokkos::parallel_for("self_acc_kick_tiled",
        team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * TILE_SIZE + team.team_rank();
            int i = start + ii;
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 4);

            double xi = 0, yi = 0, zi = 0;
            if (ii < count) { xi = px(i); yi = py(i); zi = pz(i); }
            double axi = 0, ayi = 0, azi = 0;

            for (int tile_start = start; tile_start < end; tile_start += TILE_SIZE) {
                int tile_count = ((tile_start + TILE_SIZE) < end) ? TILE_SIZE : (end - tile_start);
                int t = team.team_rank();
                if (t < tile_count) {
                    int j = tile_start + t;
                    tile(t*4+0) = px(j); tile(t*4+1) = py(j);
                    tile(t*4+2) = pz(j); tile(t*4+3) = m(j);
                }
                team.team_barrier();
                if (ii < count) {
                    for (int jt = 0; jt < tile_count; jt++) {
                        int j = tile_start + jt;
                        if (j == i) continue;
                        double dx = tile(jt*4+0) - xi;
                        double dy = tile(jt*4+1) - yi;
                        double dz = tile(jt*4+2) - zi;
                        double r2 = dx*dx + dy*dy + dz*dz + soft2;
                        double inv_r = fast_rsqrt(r2);
                        double inv_r3 = inv_r * inv_r * inv_r;
                        double fj = G * tile(jt*4+3) * inv_r3;
                        axi += fj * dx; ayi += fj * dy; azi += fj * dz;
                    }
                }
                team.team_barrier();
            }
            if (ii < count) {
                pax(i) = axi; pay(i) = ayi; paz(i) = azi;
                pvx(i) += axi * kick_dt;
                pvy(i) += ayi * kick_dt;
                pvz(i) += azi * kick_dt;
            }
        });
    Kokkos::fence();
}

// ==========================================================================
// Fused self_acceleration + symmetrized timestep (no kick)
// Computes forces AND timesteps with harmonic symmetrization in one N² pass.
// Forces stored in ax/ay/az for reuse by the first FSI kick.
// ==========================================================================
void self_acceleration_and_timestep(Particles& p, int start, int count,
                                     double G, double soft2, double eta) {
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView m{p.mass.data()};
    RawView dt_f{p.dt_adapt.data()};
    const int end = start + count;

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    size_t scratch_size = TILE_SIZE * 7 * sizeof(double);  // [x,y,z,m,vx,vy,vz]
    int num_teams = (count + TILE_SIZE - 1) / TILE_SIZE;

    Kokkos::parallel_for("self_acc_dt_sym_tiled",
        team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * TILE_SIZE + team.team_rank();
            int i = start + ii;
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 7);

            double xi = 0, yi = 0, zi = 0, mi_val = 0;
            double vxi = 0, vyi = 0, vzi = 0;
            if (ii < count) {
                xi = px(i); yi = py(i); zi = pz(i); mi_val = m(i);
                vxi = pvx(i); vyi = pvy(i); vzi = pvz(i);
            }
            double axi = 0, ayi = 0, azi = 0;
            double t_min_i = 1e30;

            for (int tile_start = start; tile_start < end; tile_start += TILE_SIZE) {
                int tile_count = ((tile_start + TILE_SIZE) < end) ? TILE_SIZE : (end - tile_start);
                int t = team.team_rank();
                if (t < tile_count) {
                    int j = tile_start + t;
                    int base = t * 7;
                    tile(base+0) = px(j); tile(base+1) = py(j);
                    tile(base+2) = pz(j); tile(base+3) = m(j);
                    tile(base+4) = pvx(j); tile(base+5) = pvy(j);
                    tile(base+6) = pvz(j);
                }
                team.team_barrier();
                if (ii < count) {
                    for (int jt = 0; jt < tile_count; jt++) {
                        int j = tile_start + jt;
                        if (j == i) continue;
                        int jb = jt * 7;
                        double dx = tile(jb+0) - xi;
                        double dy = tile(jb+1) - yi;
                        double dz = tile(jb+2) - zi;
                        double mj = tile(jb+3);
                        double r2 = dx*dx + dy*dy + dz*dz;
                        double r2s = r2 + soft2;
                        double inv_r = fast_rsqrt(r2s);
                        double inv_r3 = inv_r * inv_r * inv_r;

                        double fj = G * mj * inv_r3;
                        axi += fj * dx; ayi += fj * dy; azi += fj * dz;

                        // Symmetrized timestep
                        double dvx = tile(jb+4) - vxi;
                        double dvy = tile(jb+5) - vyi;
                        double dvz = tile(jb+6) - vzi;
                        double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
                        double mass_sum = mi_val + mj;

                        double r3 = r2s * r2s * inv_r;
                        double t_ff_sq = r3 / (G * mass_sum + EPS);
                        double t_fb_sq = r2s / (v_sq + EPS);

                        double t_min_sq_pair = Kokkos::fmin(t_ff_sq, t_fb_sq);
                        if (t_min_sq_pair <= 100.0 * t_min_i * t_min_i) {
                            double t_ff = t_ff_sq * fast_rsqrt(t_ff_sq + EPS);
                            double t_fb = t_fb_sq * fast_rsqrt(t_fb_sq + EPS);

                            double r_dot_v = dx*dvx + dy*dvy + dz*dvz;
                            double inv_r2 = inv_r * inv_r;

                            double d_ff = 1.5 * r_dot_v * inv_r2 * t_ff;
                            double d_fb = r_dot_v * inv_r2 * (1.0 + G * mass_sum * inv_r / (v_sq + EPS)) * t_fb;

                            double t_sym_ff = harmonic_sym_factor(d_ff) * t_ff;
                            double t_sym_fb = harmonic_sym_factor(d_fb) * t_fb;

                            t_min_i = Kokkos::fmin(t_min_i, Kokkos::fmin(t_sym_ff, t_sym_fb));
                        }
                    }
                }
                team.team_barrier();
            }
            if (ii < count) {
                pax(i) = axi; pay(i) = ayi; paz(i) = azi;
                dt_f(i) = t_min_i * eta;
            }
        });
    Kokkos::fence();
}

// ==========================================================================
// Fused self_acceleration_and_gradient + kick
// ==========================================================================
void self_acceleration_and_gradient_kick(Particles& p, int start, int count,
                                         double G, double soft2, double grad_dt, double kick_dt) {
    // First compute acceleration (stores in ax, ay, az)
    self_acceleration(p, start, count, G, soft2);
    // Then apply gradient correction (modifies ax, ay, az)
    self_gradient_correction(p, start, count, G, soft2, grad_dt);

    // Finally apply kick
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    Kokkos::parallel_for("kick_grad", count, KOKKOS_LAMBDA(int ii) {
        int i = start + ii;
        pvx(i) += pax(i) * kick_dt;
        pvy(i) += pay(i) * kick_dt;
        pvz(i) += paz(i) * kick_dt;
    });
    Kokkos::fence();
}

// ==========================================================================
// Fused cross_acceleration + kick
// ==========================================================================
void cross_acceleration_kick(Particles& p,
                             int a_start, int a_count,
                             int b_start, int b_count,
                             double G, double soft2, double kick_dt) {
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView m{p.mass.data()};
    const int b_end = b_start + b_count;
    const int a_end = a_start + a_count;

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    size_t scratch_size = TILE_SIZE * 4 * sizeof(double);

    // A from B, fused kick
    int num_teams_a = (a_count + TILE_SIZE - 1) / TILE_SIZE;
    Kokkos::parallel_for("cross_acc_kick_a_tiled",
        team_policy(num_teams_a, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * TILE_SIZE + team.team_rank();
            int i = a_start + ii;
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 4);

            double xi = 0, yi = 0, zi = 0;
            if (ii < a_count) { xi = px(i); yi = py(i); zi = pz(i); }
            double axi = 0, ayi = 0, azi = 0;

            for (int tile_start = b_start; tile_start < b_end; tile_start += TILE_SIZE) {
                int tile_count = ((tile_start + TILE_SIZE) < b_end) ? TILE_SIZE : (b_end - tile_start);
                int t = team.team_rank();
                if (t < tile_count) {
                    int j = tile_start + t;
                    tile(t*4+0) = px(j); tile(t*4+1) = py(j);
                    tile(t*4+2) = pz(j); tile(t*4+3) = m(j);
                }
                team.team_barrier();
                if (ii < a_count) {
                    for (int jt = 0; jt < tile_count; jt++) {
                        double dx = tile(jt*4+0) - xi;
                        double dy = tile(jt*4+1) - yi;
                        double dz = tile(jt*4+2) - zi;
                        double r2 = dx*dx + dy*dy + dz*dz + soft2;
                        double inv_r = fast_rsqrt(r2);
                        double inv_r3 = inv_r * inv_r * inv_r;
                        double fj = G * tile(jt*4+3) * inv_r3;
                        axi += fj * dx; ayi += fj * dy; azi += fj * dz;
                    }
                }
                team.team_barrier();
            }
            if (ii < a_count) {
                pax(i) = axi; pay(i) = ayi; paz(i) = azi;
                pvx(i) += axi * kick_dt;
                pvy(i) += ayi * kick_dt;
                pvz(i) += azi * kick_dt;
            }
        });
    // B from A, fused kick
    int num_teams_b = (b_count + TILE_SIZE - 1) / TILE_SIZE;
    Kokkos::parallel_for("cross_acc_kick_b_tiled",
        team_policy(num_teams_b, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * TILE_SIZE + team.team_rank();
            int j = b_start + ii;
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 4);

            double xj = 0, yj = 0, zj = 0;
            if (ii < b_count) { xj = px(j); yj = py(j); zj = pz(j); }
            double axj = 0, ayj = 0, azj = 0;

            for (int tile_start = a_start; tile_start < a_end; tile_start += TILE_SIZE) {
                int tile_count = ((tile_start + TILE_SIZE) < a_end) ? TILE_SIZE : (a_end - tile_start);
                int t = team.team_rank();
                if (t < tile_count) {
                    int i2 = tile_start + t;
                    tile(t*4+0) = px(i2); tile(t*4+1) = py(i2);
                    tile(t*4+2) = pz(i2); tile(t*4+3) = m(i2);
                }
                team.team_barrier();
                if (ii < b_count) {
                    for (int it = 0; it < tile_count; it++) {
                        double dx = tile(it*4+0) - xj;
                        double dy = tile(it*4+1) - yj;
                        double dz = tile(it*4+2) - zj;
                        double r2 = dx*dx + dy*dy + dz*dz + soft2;
                        double inv_r = fast_rsqrt(r2);
                        double inv_r3 = inv_r * inv_r * inv_r;
                        double fi = G * tile(it*4+3) * inv_r3;
                        axj += fi * dx; ayj += fi * dy; azj += fi * dz;
                    }
                }
                team.team_barrier();
            }
            if (ii < b_count) {
                pax(j) = axj; pay(j) = ayj; paz(j) = azj;
                pvx(j) += axj * kick_dt;
                pvy(j) += ayj * kick_dt;
                pvz(j) += azj * kick_dt;
            }
        });
    Kokkos::fence();
}

// ==========================================================================
// Fused cross_acceleration_and_gradient + kick
// ==========================================================================
void cross_acceleration_and_gradient_kick(Particles& p,
                                          int a_start, int a_count,
                                          int b_start, int b_count,
                                          double G, double soft2, double grad_dt, double kick_dt) {
    // Compute cross acceleration
    cross_acceleration(p, a_start, a_count, b_start, b_count, G, soft2);
    // Apply gradient correction
    cross_gradient_correction(p, a_start, a_count, b_start, b_count, G, soft2, grad_dt);

    // Apply kick to both groups
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};

    Kokkos::parallel_for("kick_cross_grad_a", a_count, KOKKOS_LAMBDA(int ii) {
        int i = a_start + ii;
        pvx(i) += pax(i) * kick_dt;
        pvy(i) += pay(i) * kick_dt;
        pvz(i) += paz(i) * kick_dt;
    });
    Kokkos::parallel_for("kick_cross_grad_b", b_count, KOKKOS_LAMBDA(int jj) {
        int j = b_start + jj;
        pvx(j) += pax(j) * kick_dt;
        pvy(j) += pay(j) * kick_dt;
        pvz(j) += paz(j) * kick_dt;
    });
    Kokkos::fence();
}

// ==========================================================================
// Cached cross-acceleration: compute, cache to scratch, and kick
// Uses scratch4/5/6 for group A cache, scratch7/8/9 for group B cache
// ==========================================================================
void cross_acceleration_kick_and_cache(Particles& p,
                                       int a_start, int a_count,
                                       int b_start, int b_count,
                                       double G, double soft2, double kick_dt) {
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView m{p.mass.data()};
    // Cache arrays (indexed by absolute position)
    RawView cax_a{p.scratch4.data()}, cay_a{p.scratch5.data()}, caz_a{p.scratch6.data()};
    RawView cax_b{p.scratch7.data()}, cay_b{p.scratch8.data()}, caz_b{p.scratch9.data()};
    const int b_end = b_start + b_count;
    const int a_end = a_start + a_count;

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    size_t scratch_size = TILE_SIZE * 4 * sizeof(double);

    // A from B, cache + kick
    int num_teams_a = (a_count + TILE_SIZE - 1) / TILE_SIZE;
    Kokkos::parallel_for("cross_acc_cache_a_tiled",
        team_policy(num_teams_a, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * TILE_SIZE + team.team_rank();
            int i = a_start + ii;
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 4);

            double xi = 0, yi = 0, zi = 0;
            if (ii < a_count) { xi = px(i); yi = py(i); zi = pz(i); }
            double axi = 0, ayi = 0, azi = 0;

            for (int tile_start = b_start; tile_start < b_end; tile_start += TILE_SIZE) {
                int tile_count = ((tile_start + TILE_SIZE) < b_end) ? TILE_SIZE : (b_end - tile_start);
                int t = team.team_rank();
                if (t < tile_count) {
                    int j = tile_start + t;
                    tile(t*4+0) = px(j); tile(t*4+1) = py(j);
                    tile(t*4+2) = pz(j); tile(t*4+3) = m(j);
                }
                team.team_barrier();
                if (ii < a_count) {
                    for (int jt = 0; jt < tile_count; jt++) {
                        double dx = tile(jt*4+0) - xi;
                        double dy = tile(jt*4+1) - yi;
                        double dz = tile(jt*4+2) - zi;
                        double r2 = dx*dx + dy*dy + dz*dz + soft2;
                        double inv_r = fast_rsqrt(r2);
                        double inv_r3 = inv_r * inv_r * inv_r;
                        double fj = G * tile(jt*4+3) * inv_r3;
                        axi += fj * dx; ayi += fj * dy; azi += fj * dz;
                    }
                }
                team.team_barrier();
            }
            if (ii < a_count) {
                pax(i) = axi; pay(i) = ayi; paz(i) = azi;
                cax_a(i) = axi; cay_a(i) = ayi; caz_a(i) = azi;
                pvx(i) += axi * kick_dt;
                pvy(i) += ayi * kick_dt;
                pvz(i) += azi * kick_dt;
            }
        });
    // B from A, cache + kick
    int num_teams_b = (b_count + TILE_SIZE - 1) / TILE_SIZE;
    Kokkos::parallel_for("cross_acc_cache_b_tiled",
        team_policy(num_teams_b, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * TILE_SIZE + team.team_rank();
            int j = b_start + ii;
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 4);

            double xj = 0, yj = 0, zj = 0;
            if (ii < b_count) { xj = px(j); yj = py(j); zj = pz(j); }
            double axj = 0, ayj = 0, azj = 0;

            for (int tile_start = a_start; tile_start < a_end; tile_start += TILE_SIZE) {
                int tile_count = ((tile_start + TILE_SIZE) < a_end) ? TILE_SIZE : (a_end - tile_start);
                int t = team.team_rank();
                if (t < tile_count) {
                    int i2 = tile_start + t;
                    tile(t*4+0) = px(i2); tile(t*4+1) = py(i2);
                    tile(t*4+2) = pz(i2); tile(t*4+3) = m(i2);
                }
                team.team_barrier();
                if (ii < b_count) {
                    for (int it = 0; it < tile_count; it++) {
                        double dx = tile(it*4+0) - xj;
                        double dy = tile(it*4+1) - yj;
                        double dz = tile(it*4+2) - zj;
                        double r2 = dx*dx + dy*dy + dz*dz + soft2;
                        double inv_r = fast_rsqrt(r2);
                        double inv_r3 = inv_r * inv_r * inv_r;
                        double fi = G * tile(it*4+3) * inv_r3;
                        axj += fi * dx; ayj += fi * dy; azj += fi * dz;
                    }
                }
                team.team_barrier();
            }
            if (ii < b_count) {
                pax(j) = axj; pay(j) = ayj; paz(j) = azj;
                cax_b(j) = axj; cay_b(j) = ayj; caz_b(j) = azj;
                pvx(j) += axj * kick_dt;
                pvy(j) += ayj * kick_dt;
                pvz(j) += azj * kick_dt;
            }
        });
    Kokkos::fence();
}
