#include "forces.h"
#include "constants.h"
#include <Kokkos_Core.hpp>
#include <cfloat>

bool grad_n3l = false;
bool use_kahan = false;
bool use_memory_tiling = true;  // Toggle shared memory tiling

// TILE_SIZE defined in constants.h

// ==========================================================================
// Full N-body acceleration: O(N(N-1)/2) with N3L or O(N^2) without
// ==========================================================================
template <bool UseN3L>
void compute_acceleration(Particles& p, double G, double soft2) {
    const int N = p.N;
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView m{p.mass.data()};

    // Zero accelerations
    Kokkos::parallel_for("zero_acc", N, KOKKOS_LAMBDA(int i) {
        pax(i) = 0; pay(i) = 0; paz(i) = 0;
    });
    Kokkos::fence();

    if constexpr (UseN3L) {
        // N3L: outer loop over i, inner j > i, atomics on j
        Kokkos::parallel_for("acc_n3l", N, KOKKOS_LAMBDA(int i) {
            double axi = 0, ayi = 0, azi = 0;
            for (int j = i + 1; j < N; j++) {
                double dx = px(j) - px(i);
                double dy = py(j) - py(i);
                double dz = pz(j) - pz(i);
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r = fast_rsqrt(r2);
                double inv_r3 = inv_r * inv_r * inv_r;

                double fj = G * m(j) * inv_r3;
                double fi = G * m(i) * inv_r3;

                axi += fj * dx; ayi += fj * dy; azi += fj * dz;
                Kokkos::atomic_add(&pax(j), -fi * dx);
                Kokkos::atomic_add(&pay(j), -fi * dy);
                Kokkos::atomic_add(&paz(j), -fi * dz);
            }
            Kokkos::atomic_add(&pax(i), axi);
            Kokkos::atomic_add(&pay(i), ayi);
            Kokkos::atomic_add(&paz(i), azi);
        });
    } else {
        // Full N^2: each particle independently
        Kokkos::parallel_for("acc_full", N, KOKKOS_LAMBDA(int i) {
            double axi = 0, ayi = 0, azi = 0;
            for (int j = 0; j < N; j++) {
                if (j == i) continue;
                double dx = px(j) - px(i);
                double dy = py(j) - py(i);
                double dz = pz(j) - pz(i);
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r = fast_rsqrt(r2);
                double inv_r3 = inv_r * inv_r * inv_r;
                double fj = G * m(j) * inv_r3;
                axi += fj * dx; ayi += fj * dy; azi += fj * dz;
            }
            pax(i) = axi; pay(i) = ayi; paz(i) = azi;
        });
    }
    Kokkos::fence();
}

// ==========================================================================
// Full N^2 acceleration + adaptive timestep (fused)
// ==========================================================================
template <bool UseN3L>
void compute_acceleration_and_timestep(Particles& p, double G, double soft2, double eta) {
    const int N = p.N;
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView m{p.mass.data()};
    RawView dt_f{p.dt_adapt.data()};

    Kokkos::parallel_for("zero_acc_dt", N, KOKKOS_LAMBDA(int i) {
        pax(i) = 0; pay(i) = 0; paz(i) = 0; dt_f(i) = DBL_MAX;
    });
    Kokkos::fence();

    if constexpr (UseN3L) {
        Kokkos::parallel_for("acc_dt_n3l", N, KOKKOS_LAMBDA(int i) {
            double axi = 0, ayi = 0, azi = 0;
            double t_min_i = DBL_MAX;

            for (int j = i + 1; j < N; j++) {
                double dx = px(j) - px(i);
                double dy = py(j) - py(i);
                double dz = pz(j) - pz(i);
                double r2 = dx*dx + dy*dy + dz*dz;
                double r2s = r2 + soft2;
                double inv_r = fast_rsqrt(r2s);
                double inv_r3 = inv_r * inv_r * inv_r;

                double fj = G * m(j) * inv_r3;
                double fi = G * m(i) * inv_r3;
                axi += fj * dx; ayi += fj * dy; azi += fj * dz;
                Kokkos::atomic_add(&pax(j), -fi * dx);
                Kokkos::atomic_add(&pay(j), -fi * dy);
                Kokkos::atomic_add(&paz(j), -fi * dz);

                // Adaptive timestep: reuse inv_r, no extra sqrt
                double dvx = pvx(j) - pvx(i);
                double dvy = pvy(j) - pvy(i);
                double dvz = pvz(j) - pvz(i);
                double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
                double mass_sum = m(i) + m(j);

                // r2s^1.5 = r2s * r2s * inv_r (reuses rsqrt from force)
                double r3 = r2s * r2s * inv_r;
                double t_ff_sq = r3 / (G * mass_sum + EPS);
                double t_fb_sq = r2s / (v_sq + EPS);
                double t_min_sq_pair = Kokkos::fmin(t_ff_sq, t_fb_sq);

                t_min_i = Kokkos::fmin(t_min_i, t_min_sq_pair);
                Kokkos::atomic_min(&dt_f(j), t_min_sq_pair);
            }
            Kokkos::atomic_add(&pax(i), axi);
            Kokkos::atomic_add(&pay(i), ayi);
            Kokkos::atomic_add(&paz(i), azi);
            Kokkos::atomic_min(&dt_f(i), t_min_i);
        });
    } else if (use_memory_tiling) {
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
    } else {
        // Original non-tiled version
        Kokkos::parallel_for("acc_dt_full", N, KOKKOS_LAMBDA(int i) {
            double axi = 0, ayi = 0, azi = 0;
            double t_min_i = DBL_MAX;

            for (int j = 0; j < N; j++) {
                if (j == i) continue;
                double dx = px(j) - px(i);
                double dy = py(j) - py(i);
                double dz = pz(j) - pz(i);
                double r2 = dx*dx + dy*dy + dz*dz;
                double r2s = r2 + soft2;
                double inv_r = fast_rsqrt(r2s);
                double inv_r3 = inv_r * inv_r * inv_r;

                double fj = G * m(j) * inv_r3;
                axi += fj * dx; ayi += fj * dy; azi += fj * dz;

                // Adaptive timestep: reuse inv_r, no extra sqrt
                double dvx = pvx(j) - pvx(i);
                double dvy = pvy(j) - pvy(i);
                double dvz = pvz(j) - pvz(i);
                double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
                double mass_sum = m(i) + m(j);

                double r3 = r2s * r2s * inv_r;
                double t_ff_sq = r3 / (G * mass_sum + EPS);
                double t_fb_sq = r2s / (v_sq + EPS);
                double t_min_sq_pair = Kokkos::fmin(t_ff_sq, t_fb_sq);
                t_min_i = Kokkos::fmin(t_min_i, t_min_sq_pair);
            }
            pax(i) = axi; pay(i) = ayi; paz(i) = azi;
            dt_f(i) = Kokkos::sqrt(t_min_i);
        });
    }
    Kokkos::fence();

    // Apply sqrt + eta (N3L path stores t²; tiled/non-tiled paths already applied sqrt)
    if constexpr (UseN3L) {
        Kokkos::parallel_for("sqrt_eta", N, KOKKOS_LAMBDA(int i) {
            dt_f(i) = Kokkos::sqrt(dt_f(i)) * eta;
        });
    } else {
        Kokkos::parallel_for("apply_eta", N, KOKKOS_LAMBDA(int i) {
            dt_f(i) *= eta;
        });
    }
    Kokkos::fence();
}

// ==========================================================================
// Self-interaction: [start, start+count)
// ==========================================================================
template <bool UseN3L>
void self_acceleration(Particles& p, int start, int count,
                       double G, double soft2) {
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView m{p.mass.data()};
    const int end = start + count;

    if constexpr (UseN3L) {
        // N3L needs zeroing first due to atomic adds
        Kokkos::parallel_for("zero_self", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            pax(i) = 0; pay(i) = 0; paz(i) = 0;
        });
        Kokkos::fence();

        Kokkos::parallel_for("self_acc_n3l", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            double axi = 0, ayi = 0, azi = 0;
            for (int j = i + 1; j < end; j++) {
                double dx = px(j) - px(i);
                double dy = py(j) - py(i);
                double dz = pz(j) - pz(i);
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r = fast_rsqrt(r2);
                double inv_r3 = inv_r * inv_r * inv_r;
                double fj = G * m(j) * inv_r3;
                double fi = G * m(i) * inv_r3;
                axi += fj * dx; ayi += fj * dy; azi += fj * dz;
                Kokkos::atomic_add(&pax(j), -fi * dx);
                Kokkos::atomic_add(&pay(j), -fi * dy);
                Kokkos::atomic_add(&paz(j), -fi * dz);
            }
            Kokkos::atomic_add(&pax(i), axi);
            Kokkos::atomic_add(&pay(i), ayi);
            Kokkos::atomic_add(&paz(i), azi);
        });
        Kokkos::fence();
    } else {
        // Non-N3L: direct assignment, no zeroing needed
        Kokkos::parallel_for("self_acc_full", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            double axi = 0, ayi = 0, azi = 0;
            double xi = px(i), yi = py(i), zi = pz(i);
            for (int j = start; j < end; j++) {
                if (i == j) continue;  // Skip self-interaction
                double dx = px(j) - xi;
                double dy = py(j) - yi;
                double dz = pz(j) - zi;
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r = fast_rsqrt(r2);
                double inv_r3 = inv_r * inv_r * inv_r;
                double fj = G * m(j) * inv_r3;
                axi += fj * dx; ayi += fj * dy; azi += fj * dz;
            }
            pax(i) = axi; pay(i) = ayi; paz(i) = azi;
        });
        Kokkos::fence();
    }
}

// ==========================================================================
// Self gradient correction
// ==========================================================================
template <bool UseN3L>
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

    if (grad_n3l && UseN3L) {
        // N3L gradient: upper triangle with atomics
        RawView cx_acc{p.scratch4.data()}, cy_acc{p.scratch5.data()}, cz_acc{p.scratch6.data()};
        Kokkos::parallel_for("zero_grad_n3l", count, KOKKOS_LAMBDA(int i) {
            cx_acc(i) = 0; cy_acc(i) = 0; cz_acc(i) = 0;
        });
        Kokkos::fence();

        Kokkos::parallel_for("self_grad_n3l", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            double axi0 = ax0(ii), ayi0 = ay0(ii), azi0 = az0(ii);
            double lcx = 0, lcy = 0, lcz = 0;
            for (int jj = ii + 1; jj < count; jj++) {
                int j = start + jj;
                double dx = px(j) - px(i), dy = py(j) - py(i), dz = pz(j) - pz(i);
                double dax = ax0(jj) - axi0, day = ay0(jj) - ayi0, daz = az0(jj) - azi0;
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r = fast_rsqrt(r2);
                double inv_r2 = inv_r * inv_r;
                double inv_r5 = inv_r2 * inv_r2 * inv_r;
                double r_dot_a = dx*dax + dy*day + dz*daz;
                double term_x = r2 * dax - 3.0 * r_dot_a * dx;
                double term_y = r2 * day - 3.0 * r_dot_a * dy;
                double term_z = r2 * daz - 3.0 * r_dot_a * dz;

                double fj = coeff * m(j) * inv_r5;
                lcx += fj * term_x; lcy += fj * term_y; lcz += fj * term_z;
                double fi = coeff * m(i) * inv_r5;
                Kokkos::atomic_add(&cx_acc(jj), -fi * term_x);
                Kokkos::atomic_add(&cy_acc(jj), -fi * term_y);
                Kokkos::atomic_add(&cz_acc(jj), -fi * term_z);
            }
            Kokkos::atomic_add(&cx_acc(ii), lcx);
            Kokkos::atomic_add(&cy_acc(ii), lcy);
            Kokkos::atomic_add(&cz_acc(ii), lcz);
        });
        Kokkos::fence();

        Kokkos::parallel_for("apply_grad_n3l", count, KOKKOS_LAMBDA(int ii) {
            pax(start + ii) += cx_acc(ii);
            pay(start + ii) += cy_acc(ii);
            paz(start + ii) += cz_acc(ii);
        });
    } else {
        // Full N^2: each particle independently
        Kokkos::parallel_for("self_grad_full", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            double cx = 0, cy = 0, cz = 0;
            double axi0 = ax0(ii), ayi0 = ay0(ii), azi0 = az0(ii);
            for (int jj = 0; jj < count; jj++) {
                if (jj == ii) continue;
                int j = start + jj;
                double dx = px(j) - px(i);
                double dy = py(j) - py(i);
                double dz = pz(j) - pz(i);
                double dax = ax0(jj) - axi0;
                double day = ay0(jj) - ayi0;
                double daz = az0(jj) - azi0;
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r = fast_rsqrt(r2);
                double inv_r2 = inv_r * inv_r;
                double inv_r5 = inv_r2 * inv_r2 * inv_r;
                double r_dot_a = dx*dax + dy*day + dz*daz;
                double f = coeff * m(j) * inv_r5;
                cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
            }
            pax(i) += cx; pay(i) += cy; paz(i) += cz;
        });
    }
    Kokkos::fence();
}

// ==========================================================================
// Fused self_acceleration + self_gradient_correction
// ==========================================================================
template <bool UseN3L>
void self_acceleration_and_gradient(Particles& p, int start, int count,
                                    double G, double soft2, double dt) {
    self_acceleration<UseN3L>(p, start, count, G, soft2);
    self_gradient_correction<UseN3L>(p, start, count, G, soft2, dt);
}

// ==========================================================================
// Cross-interaction: group A x group B
// ==========================================================================
template <bool UseN3L>
void cross_acceleration(Particles& p,
                        int a_start, int a_count,
                        int b_start, int b_count,
                        double G, double soft2) {
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView m{p.mass.data()};
    const int b_end = b_start + b_count;

    if constexpr (UseN3L) {
        // N3L needs zeroing first due to atomic adds on B
        Kokkos::parallel_for("zero_cross_a", a_count, KOKKOS_LAMBDA(int ii) {
            int i = a_start + ii;
            pax(i) = 0; pay(i) = 0; paz(i) = 0;
        });
        Kokkos::parallel_for("zero_cross_b", b_count, KOKKOS_LAMBDA(int jj) {
            int j = b_start + jj;
            pax(j) = 0; pay(j) = 0; paz(j) = 0;
        });
        Kokkos::fence();

        // A-side loop, atomics on B
        Kokkos::parallel_for("cross_acc_n3l", a_count, KOKKOS_LAMBDA(int ii) {
            int i = a_start + ii;
            double axi = 0, ayi = 0, azi = 0;
            for (int j = b_start; j < b_end; j++) {
                double dx = px(j) - px(i);
                double dy = py(j) - py(i);
                double dz = pz(j) - pz(i);
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r = fast_rsqrt(r2);
                double inv_r3 = inv_r * inv_r * inv_r;
                double fj = G * m(j) * inv_r3;
                double fi = G * m(i) * inv_r3;
                axi += fj * dx; ayi += fj * dy; azi += fj * dz;
                Kokkos::atomic_add(&pax(j), -fi * dx);
                Kokkos::atomic_add(&pay(j), -fi * dy);
                Kokkos::atomic_add(&paz(j), -fi * dz);
            }
            pax(i) += axi; pay(i) += ayi; paz(i) += azi;
        });
        Kokkos::fence();
    } else {
        // Non-N3L: direct assignment, no zeroing needed
        // Full: A from B
        const int a_end = a_start + a_count;
        Kokkos::parallel_for("cross_acc_a", a_count, KOKKOS_LAMBDA(int ii) {
            int i = a_start + ii;
            double axi = 0, ayi = 0, azi = 0;
            double xi = px(i), yi = py(i), zi = pz(i);
            for (int j = b_start; j < b_end; j++) {
                double dx = px(j) - xi;
                double dy = py(j) - yi;
                double dz = pz(j) - zi;
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r = fast_rsqrt(r2);
                double inv_r3 = inv_r * inv_r * inv_r;
                double fj = G * m(j) * inv_r3;
                axi += fj * dx; ayi += fj * dy; azi += fj * dz;
            }
            pax(i) = axi; pay(i) = ayi; paz(i) = azi;
        });
        // Full: B from A
        Kokkos::parallel_for("cross_acc_b", b_count, KOKKOS_LAMBDA(int jj) {
            int j = b_start + jj;
            double axj = 0, ayj = 0, azj = 0;
            double xj = px(j), yj = py(j), zj = pz(j);
            for (int i = a_start; i < a_end; i++) {
                double dx = px(i) - xj;
                double dy = py(i) - yj;
                double dz = pz(i) - zj;
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r = fast_rsqrt(r2);
                double inv_r3 = inv_r * inv_r * inv_r;
                double fi = G * m(i) * inv_r3;
                axj += fi * dx; ayj += fi * dy; azj += fi * dz;
            }
            pax(j) = axj; pay(j) = ayj; paz(j) = azj;
        });
        Kokkos::fence();
    }
}

// ==========================================================================
// Cross gradient correction
// ==========================================================================
template <bool UseN3L>
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

    if (grad_n3l && UseN3L) {
        // N3L: A loop, atomics on B
        Kokkos::parallel_for("cross_grad_n3l", a_count, KOKKOS_LAMBDA(int ii) {
            int i = a_start + ii;
            double cxi = 0, cyi = 0, czi = 0;
            double axi0 = ax0_a(ii), ayi0 = ay0_a(ii), azi0 = az0_a(ii);
            for (int jj = 0; jj < b_count; jj++) {
                int j = b_start + jj;
                double dx = px(j) - px(i), dy = py(j) - py(i), dz = pz(j) - pz(i);
                double dax = ax0_b(jj) - axi0, day = ay0_b(jj) - ayi0, daz = az0_b(jj) - azi0;
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r = fast_rsqrt(r2);
                double inv_r2 = inv_r * inv_r;
                double inv_r5 = inv_r2 * inv_r2 * inv_r;
                double r_dot_a = dx*dax + dy*day + dz*daz;
                double term_x = r2 * dax - 3.0 * r_dot_a * dx;
                double term_y = r2 * day - 3.0 * r_dot_a * dy;
                double term_z = r2 * daz - 3.0 * r_dot_a * dz;

                double fj = coeff * m(j) * inv_r5;
                cxi += fj * term_x; cyi += fj * term_y; czi += fj * term_z;
                double fi = coeff * m(i) * inv_r5;
                Kokkos::atomic_add(&pax(j), -fi * term_x);
                Kokkos::atomic_add(&pay(j), -fi * term_y);
                Kokkos::atomic_add(&paz(j), -fi * term_z);
            }
            pax(i) += cxi; pay(i) += cyi; paz(i) += czi;
        });
    } else {
        // Full N^2: A from B
        Kokkos::parallel_for("cross_grad_a", a_count, KOKKOS_LAMBDA(int ii) {
            int i = a_start + ii;
            double cx = 0, cy = 0, cz = 0;
            double axi0 = ax0_a(ii), ayi0 = ay0_a(ii), azi0 = az0_a(ii);
            for (int jj = 0; jj < b_count; jj++) {
                int j = b_start + jj;
                double dx = px(j) - px(i), dy = py(j) - py(i), dz = pz(j) - pz(i);
                double dax = ax0_b(jj) - axi0, day = ay0_b(jj) - ayi0, daz = az0_b(jj) - azi0;
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r = fast_rsqrt(r2);
                double inv_r2 = inv_r * inv_r;
                double inv_r5 = inv_r2 * inv_r2 * inv_r;
                double r_dot_a = dx*dax + dy*day + dz*daz;
                double f = coeff * m(j) * inv_r5;
                cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
            }
            pax(i) += cx; pay(i) += cy; paz(i) += cz;
        });
        // Full N^2: B from A
        const int a_end = a_start + a_count;
        Kokkos::parallel_for("cross_grad_b", b_count, KOKKOS_LAMBDA(int jj) {
            int j = b_start + jj;
            double cx = 0, cy = 0, cz = 0;
            double axj0 = ax0_b(jj), ayj0 = ay0_b(jj), azj0 = az0_b(jj);
            for (int i = a_start; i < a_end; i++) {
                int ii = i - a_start;
                double dx = px(i) - px(j), dy = py(i) - py(j), dz = pz(i) - pz(j);
                double dax = ax0_a(ii) - axj0, day = ay0_a(ii) - ayj0, daz = az0_a(ii) - azj0;
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r = fast_rsqrt(r2);
                double inv_r2 = inv_r * inv_r;
                double inv_r5 = inv_r2 * inv_r2 * inv_r;
                double r_dot_a = dx*dax + dy*day + dz*daz;
                double f = coeff * m(i) * inv_r5;
                cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
            }
            pax(j) += cx; pay(j) += cy; paz(j) += cz;
        });
    }
    Kokkos::fence();
}

// ==========================================================================
// Fused cross_acceleration + cross_gradient_correction
// ==========================================================================
template <bool UseN3L>
void cross_acceleration_and_gradient(Particles& p,
                                     int a_start, int a_count,
                                     int b_start, int b_count,
                                     double G, double soft2, double dt) {
    cross_acceleration<UseN3L>(p, a_start, a_count, b_start, b_count, G, soft2);
    cross_gradient_correction<UseN3L>(p, a_start, a_count, b_start, b_count, G, soft2, dt);
}

// ==========================================================================
// Compute adaptive timesteps for a subset [start, start+count)
// Only considers mutual interactions within the subset (for HHS recursion)
// ==========================================================================
template <bool UseN3L>
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

    if constexpr (UseN3L) {
        // N3L: upper triangle with atomics, squared-space timestep
        Kokkos::parallel_for("dt_subset_n3l", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            double t_min_sq_i = 1e60;

            for (int j = i + 1; j < end; j++) {
                double dx = px(j) - px(i);
                double dy = py(j) - py(i);
                double dz = pz(j) - pz(i);
                double r2 = dx*dx + dy*dy + dz*dz;

                double dvx = pvx(j) - pvx(i);
                double dvy = pvy(j) - pvy(i);
                double dvz = pvz(j) - pvz(i);
                double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;

                double r2s = r2 + soft2;
                double inv_r_ts = fast_rsqrt(r2s);
                double mass_sum = m(i) + m(j);
                double r3 = r2s * r2s * inv_r_ts;
                double t_ff_sq = r3 / (G * mass_sum + EPS);
                double t_fb_sq = r2s / (v_sq + EPS);

                double t_min_sq_pair = Kokkos::fmin(t_ff_sq, t_fb_sq);
                t_min_sq_i = Kokkos::fmin(t_min_sq_i, t_min_sq_pair);
                Kokkos::atomic_min(&dt_f(j), t_min_sq_pair);
            }
            Kokkos::atomic_min(&dt_f(i), t_min_sq_i);
        });
        Kokkos::fence();

        // Convert from t^2 to t and apply eta
        Kokkos::parallel_for("sqrt_eta_n3l", count, KOKKOS_LAMBDA(int ii) {
            dt_f(start + ii) = Kokkos::sqrt(dt_f(start + ii)) * eta;
        });
    } else {
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
    }
    Kokkos::fence();
}

// ==========================================================================
// Fused self_acceleration + kick (reduces kernel launches)
// ==========================================================================
template <bool UseN3L>
void self_acceleration_kick(Particles& p, int start, int count,
                            double G, double soft2, double kick_dt) {
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView m{p.mass.data()};
    const int end = start + count;

    if constexpr (UseN3L) {
        // N3L needs zeroing first due to atomic adds
        Kokkos::parallel_for("zero_self_kick", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            pax(i) = 0; pay(i) = 0; paz(i) = 0;
        });
        Kokkos::fence();

        Kokkos::parallel_for("self_acc_kick_n3l", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            double axi = 0, ayi = 0, azi = 0;
            for (int j = i + 1; j < end; j++) {
                double dx = px(j) - px(i);
                double dy = py(j) - py(i);
                double dz = pz(j) - pz(i);
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r = fast_rsqrt(r2);
                double inv_r3 = inv_r * inv_r * inv_r;
                double fj = G * m(j) * inv_r3;
                double fi = G * m(i) * inv_r3;
                axi += fj * dx; ayi += fj * dy; azi += fj * dz;
                Kokkos::atomic_add(&pax(j), -fi * dx);
                Kokkos::atomic_add(&pay(j), -fi * dy);
                Kokkos::atomic_add(&paz(j), -fi * dz);
            }
            Kokkos::atomic_add(&pax(i), axi);
            Kokkos::atomic_add(&pay(i), ayi);
            Kokkos::atomic_add(&paz(i), azi);
        });
        Kokkos::fence();

        // Apply kick
        Kokkos::parallel_for("kick_n3l", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            pvx(i) += pax(i) * kick_dt;
            pvy(i) += pay(i) * kick_dt;
            pvz(i) += paz(i) * kick_dt;
        });
        Kokkos::fence();
    } else {
        // Non-N3L: fused force + kick in single kernel
        Kokkos::parallel_for("self_acc_kick_full", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            double axi = 0, ayi = 0, azi = 0;
            double xi = px(i), yi = py(i), zi = pz(i);
            for (int j = start; j < end; j++) {
                if (i == j) continue;
                double dx = px(j) - xi;
                double dy = py(j) - yi;
                double dz = pz(j) - zi;
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r = fast_rsqrt(r2);
                double inv_r3 = inv_r * inv_r * inv_r;
                double fj = G * m(j) * inv_r3;
                axi += fj * dx; ayi += fj * dy; azi += fj * dz;
            }
            pax(i) = axi; pay(i) = ayi; paz(i) = azi;
            // Fused kick
            pvx(i) += axi * kick_dt;
            pvy(i) += ayi * kick_dt;
            pvz(i) += azi * kick_dt;
        });
        Kokkos::fence();
    }
}

// ==========================================================================
// Fused self_acceleration_and_gradient + kick
// ==========================================================================
template <bool UseN3L>
void self_acceleration_and_gradient_kick(Particles& p, int start, int count,
                                         double G, double soft2, double grad_dt, double kick_dt) {
    // First compute acceleration (stores in ax, ay, az)
    self_acceleration<UseN3L>(p, start, count, G, soft2);
    // Then apply gradient correction (modifies ax, ay, az)
    self_gradient_correction<UseN3L>(p, start, count, G, soft2, grad_dt);

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
template <bool UseN3L>
void cross_acceleration_kick(Particles& p,
                             int a_start, int a_count,
                             int b_start, int b_count,
                             double G, double soft2, double kick_dt) {
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView m{p.mass.data()};
    const int b_end = b_start + b_count;

    if constexpr (UseN3L) {
        // N3L needs zeroing first
        Kokkos::parallel_for("zero_cross_a_kick", a_count, KOKKOS_LAMBDA(int ii) {
            int i = a_start + ii;
            pax(i) = 0; pay(i) = 0; paz(i) = 0;
        });
        Kokkos::parallel_for("zero_cross_b_kick", b_count, KOKKOS_LAMBDA(int jj) {
            int j = b_start + jj;
            pax(j) = 0; pay(j) = 0; paz(j) = 0;
        });
        Kokkos::fence();

        Kokkos::parallel_for("cross_acc_kick_n3l", a_count, KOKKOS_LAMBDA(int ii) {
            int i = a_start + ii;
            double axi = 0, ayi = 0, azi = 0;
            for (int j = b_start; j < b_end; j++) {
                double dx = px(j) - px(i);
                double dy = py(j) - py(i);
                double dz = pz(j) - pz(i);
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r = fast_rsqrt(r2);
                double inv_r3 = inv_r * inv_r * inv_r;
                double fj = G * m(j) * inv_r3;
                double fi = G * m(i) * inv_r3;
                axi += fj * dx; ayi += fj * dy; azi += fj * dz;
                Kokkos::atomic_add(&pax(j), -fi * dx);
                Kokkos::atomic_add(&pay(j), -fi * dy);
                Kokkos::atomic_add(&paz(j), -fi * dz);
            }
            Kokkos::atomic_add(&pax(i), axi);
            Kokkos::atomic_add(&pay(i), ayi);
            Kokkos::atomic_add(&paz(i), azi);
        });
        Kokkos::fence();

        // Apply kick to both groups
        Kokkos::parallel_for("kick_cross_a", a_count, KOKKOS_LAMBDA(int ii) {
            int i = a_start + ii;
            pvx(i) += pax(i) * kick_dt;
            pvy(i) += pay(i) * kick_dt;
            pvz(i) += paz(i) * kick_dt;
        });
        Kokkos::parallel_for("kick_cross_b", b_count, KOKKOS_LAMBDA(int jj) {
            int j = b_start + jj;
            pvx(j) += pax(j) * kick_dt;
            pvy(j) += pay(j) * kick_dt;
            pvz(j) += paz(j) * kick_dt;
        });
        Kokkos::fence();
    } else {
        // Non-N3L: compute forces independently for A and B
        Kokkos::parallel_for("cross_acc_kick_a", a_count, KOKKOS_LAMBDA(int ii) {
            int i = a_start + ii;
            double axi = 0, ayi = 0, azi = 0;
            for (int j = b_start; j < b_end; j++) {
                double dx = px(j) - px(i);
                double dy = py(j) - py(i);
                double dz = pz(j) - pz(i);
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r = fast_rsqrt(r2);
                double inv_r3 = inv_r * inv_r * inv_r;
                double fj = G * m(j) * inv_r3;
                axi += fj * dx; ayi += fj * dy; azi += fj * dz;
            }
            pax(i) = axi; pay(i) = ayi; paz(i) = azi;
            pvx(i) += axi * kick_dt;
            pvy(i) += ayi * kick_dt;
            pvz(i) += azi * kick_dt;
        });

        const int a_end = a_start + a_count;
        Kokkos::parallel_for("cross_acc_kick_b", b_count, KOKKOS_LAMBDA(int jj) {
            int j = b_start + jj;
            double axj = 0, ayj = 0, azj = 0;
            for (int i = a_start; i < a_end; i++) {
                double dx = px(i) - px(j);
                double dy = py(i) - py(j);
                double dz = pz(i) - pz(j);
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r = fast_rsqrt(r2);
                double inv_r3 = inv_r * inv_r * inv_r;
                double fi = G * m(i) * inv_r3;
                axj += fi * dx; ayj += fi * dy; azj += fi * dz;
            }
            pax(j) = axj; pay(j) = ayj; paz(j) = azj;
            pvx(j) += axj * kick_dt;
            pvy(j) += ayj * kick_dt;
            pvz(j) += azj * kick_dt;
        });
        Kokkos::fence();
    }
}

// ==========================================================================
// Fused cross_acceleration_and_gradient + kick
// ==========================================================================
template <bool UseN3L>
void cross_acceleration_and_gradient_kick(Particles& p,
                                          int a_start, int a_count,
                                          int b_start, int b_count,
                                          double G, double soft2, double grad_dt, double kick_dt) {
    // Compute cross acceleration
    cross_acceleration<UseN3L>(p, a_start, a_count, b_start, b_count, G, soft2);
    // Apply gradient correction
    cross_gradient_correction<UseN3L>(p, a_start, a_count, b_start, b_count, G, soft2, grad_dt);

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
template <bool UseN3L>
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

    if constexpr (UseN3L) {
        // Zero accelerations
        Kokkos::parallel_for("zero_cross_cache_a", a_count, KOKKOS_LAMBDA(int ii) {
            int i = a_start + ii;
            pax(i) = 0; pay(i) = 0; paz(i) = 0;
        });
        Kokkos::parallel_for("zero_cross_cache_b", b_count, KOKKOS_LAMBDA(int jj) {
            int j = b_start + jj;
            pax(j) = 0; pay(j) = 0; paz(j) = 0;
        });
        Kokkos::fence();

        Kokkos::parallel_for("cross_acc_cache_n3l", a_count, KOKKOS_LAMBDA(int ii) {
            int i = a_start + ii;
            double axi = 0, ayi = 0, azi = 0;
            for (int j = b_start; j < b_end; j++) {
                double dx = px(j) - px(i);
                double dy = py(j) - py(i);
                double dz = pz(j) - pz(i);
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r = fast_rsqrt(r2);
                double inv_r3 = inv_r * inv_r * inv_r;
                double fj = G * m(j) * inv_r3;
                double fi = G * m(i) * inv_r3;
                axi += fj * dx; ayi += fj * dy; azi += fj * dz;
                Kokkos::atomic_add(&pax(j), -fi * dx);
                Kokkos::atomic_add(&pay(j), -fi * dy);
                Kokkos::atomic_add(&paz(j), -fi * dz);
            }
            Kokkos::atomic_add(&pax(i), axi);
            Kokkos::atomic_add(&pay(i), ayi);
            Kokkos::atomic_add(&paz(i), azi);
        });
        Kokkos::fence();

        // Cache and kick group A
        Kokkos::parallel_for("cache_kick_a", a_count, KOKKOS_LAMBDA(int ii) {
            int i = a_start + ii;
            cax_a(i) = pax(i); cay_a(i) = pay(i); caz_a(i) = paz(i);
            pvx(i) += pax(i) * kick_dt;
            pvy(i) += pay(i) * kick_dt;
            pvz(i) += paz(i) * kick_dt;
        });
        // Cache and kick group B
        Kokkos::parallel_for("cache_kick_b", b_count, KOKKOS_LAMBDA(int jj) {
            int j = b_start + jj;
            cax_b(j) = pax(j); cay_b(j) = pay(j); caz_b(j) = paz(j);
            pvx(j) += pax(j) * kick_dt;
            pvy(j) += pay(j) * kick_dt;
            pvz(j) += paz(j) * kick_dt;
        });
        Kokkos::fence();
    } else {
        // Non-N3L: compute independently for A and B
        Kokkos::parallel_for("cross_acc_cache_a", a_count, KOKKOS_LAMBDA(int ii) {
            int i = a_start + ii;
            double axi = 0, ayi = 0, azi = 0;
            for (int j = b_start; j < b_end; j++) {
                double dx = px(j) - px(i);
                double dy = py(j) - py(i);
                double dz = pz(j) - pz(i);
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r = fast_rsqrt(r2);
                double inv_r3 = inv_r * inv_r * inv_r;
                double fj = G * m(j) * inv_r3;
                axi += fj * dx; ayi += fj * dy; azi += fj * dz;
            }
            pax(i) = axi; pay(i) = ayi; paz(i) = azi;
            cax_a(i) = axi; cay_a(i) = ayi; caz_a(i) = azi;
            pvx(i) += axi * kick_dt;
            pvy(i) += ayi * kick_dt;
            pvz(i) += azi * kick_dt;
        });

        const int a_end = a_start + a_count;
        Kokkos::parallel_for("cross_acc_cache_b", b_count, KOKKOS_LAMBDA(int jj) {
            int j = b_start + jj;
            double axj = 0, ayj = 0, azj = 0;
            for (int i = a_start; i < a_end; i++) {
                double dx = px(i) - px(j);
                double dy = py(i) - py(j);
                double dz = pz(i) - pz(j);
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r = fast_rsqrt(r2);
                double inv_r3 = inv_r * inv_r * inv_r;
                double fi = G * m(i) * inv_r3;
                axj += fi * dx; ayj += fi * dy; azj += fi * dz;
            }
            pax(j) = axj; pay(j) = ayj; paz(j) = azj;
            cax_b(j) = axj; cay_b(j) = ayj; caz_b(j) = azj;
            pvx(j) += axj * kick_dt;
            pvy(j) += ayj * kick_dt;
            pvz(j) += azj * kick_dt;
        });
        Kokkos::fence();
    }
}

// ==========================================================================
// Kick using cached accelerations (no force recomputation)
// ==========================================================================
void kick_from_cache(Particles& p, int a_start, int a_count,
                     int b_start, int b_count, double kick_dt) {
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView cax_a{p.scratch4.data()}, cay_a{p.scratch5.data()}, caz_a{p.scratch6.data()};
    RawView cax_b{p.scratch7.data()}, cay_b{p.scratch8.data()}, caz_b{p.scratch9.data()};

    // Restore cached accelerations and kick group A
    Kokkos::parallel_for("kick_cache_a", a_count, KOKKOS_LAMBDA(int ii) {
        int i = a_start + ii;
        pax(i) = cax_a(i); pay(i) = cay_a(i); paz(i) = caz_a(i);
        pvx(i) += cax_a(i) * kick_dt;
        pvy(i) += cay_a(i) * kick_dt;
        pvz(i) += caz_a(i) * kick_dt;
    });
    // Restore cached accelerations and kick group B
    Kokkos::parallel_for("kick_cache_b", b_count, KOKKOS_LAMBDA(int jj) {
        int j = b_start + jj;
        pax(j) = cax_b(j); pay(j) = cay_b(j); paz(j) = caz_b(j);
        pvx(j) += cax_b(j) * kick_dt;
        pvy(j) += cay_b(j) * kick_dt;
        pvz(j) += caz_b(j) * kick_dt;
    });
    Kokkos::fence();
}

// ==========================================================================
// Ghost gradient correction for distributed (multi-GPU) case
// ==========================================================================
void ghost_gradient_correction(Particles& local, const Particles& ghosts,
                               double G, double soft2, double dt) {
    int N_local = local.N;
    int N_ghosts = ghosts.N;

    if (N_ghosts == 0) return;

    RawView lx{local.x.data()}, ly{local.y.data()}, lz{local.z.data()};
    RawView lax{local.ax.data()}, lay{local.ay.data()}, laz{local.az.data()};

    RawView gx{ghosts.x.data()}, gy{ghosts.y.data()}, gz{ghosts.z.data()};
    RawView gax{ghosts.ax.data()}, gay{ghosts.ay.data()}, gaz{ghosts.az.data()};
    RawView gm{ghosts.mass.data()};

    const double coeff = G * dt * dt * GRAD_COEFF;

    // Each local particle gets gradient correction from all ghost particles
    // correction_i = sum_j { coeff * m_j / r^5 * [ r^2 * delta_a - 3 * (r · delta_a) * r ] }
    // where delta_a = a_j - a_i
    Kokkos::parallel_for("ghost_grad_corr", N_local, KOKKOS_LAMBDA(int i) {
        double axi = lax(i), ayi = lay(i), azi = laz(i);
        double cx = 0.0, cy = 0.0, cz = 0.0;

        for (int j = 0; j < N_ghosts; j++) {
            double dx = gx(j) - lx(i);
            double dy = gy(j) - ly(i);
            double dz = gz(j) - lz(i);

            double dax = gax(j) - axi;
            double day = gay(j) - ayi;
            double daz = gaz(j) - azi;

            double r2 = dx*dx + dy*dy + dz*dz + soft2;
            double inv_r = fast_rsqrt(r2);
            double inv_r2 = inv_r * inv_r;
            double inv_r5 = inv_r2 * inv_r2 * inv_r;

            double r_dot_a = dx*dax + dy*day + dz*daz;
            double f = coeff * gm(j) * inv_r5;

            cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
            cy += f * (r2 * day - 3.0 * r_dot_a * dy);
            cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
        }

        lax(i) += cx;
        lay(i) += cy;
        laz(i) += cz;
    });
    Kokkos::fence();
}

// ==========================================================================
// Ghost cross-kick functions for distributed HHS
// ==========================================================================

// Kick local slow particles from ghost fast particles
void ghost_cross_kick_slow_from_fast(Particles& local, const Particles& ghosts,
                                     int local_start, int local_count,
                                     double pivot, double G, double soft2, double kick_dt) {
    if (local_count == 0 || ghosts.N == 0) return;

    RawView lx{local.x.data()}, ly{local.y.data()}, lz{local.z.data()};
    RawView lvx{local.vx.data()}, lvy{local.vy.data()}, lvz{local.vz.data()};
    RawView lax{local.ax.data()}, lay{local.ay.data()}, laz{local.az.data()};

    RawView gx{ghosts.x.data()}, gy{ghosts.y.data()}, gz{ghosts.z.data()};
    RawView gm{ghosts.mass.data()};
    RawView gdt{ghosts.dt_adapt.data()};

    int N_ghosts = ghosts.N;

    // Each local slow particle accumulates force from all ghost fast particles
    Kokkos::parallel_for("ghost_cross_slow_from_fast", local_count, KOKKOS_LAMBDA(int ii) {
        int i = local_start + ii;
        double xi = lx(i), yi = ly(i), zi = lz(i);
        double axi = 0.0, ayi = 0.0, azi = 0.0;

        for (int j = 0; j < N_ghosts; j++) {
            // Only consider fast ghosts (dt < pivot)
            if (gdt(j) >= pivot) continue;

            double dx = gx(j) - xi;
            double dy = gy(j) - yi;
            double dz = gz(j) - zi;

            double r2 = dx*dx + dy*dy + dz*dz + soft2;
            double inv_r = fast_rsqrt(r2);
            double inv_r3 = inv_r * inv_r * inv_r;
            double mj_inv_r3 = G * gm(j) * inv_r3;

            axi += mj_inv_r3 * dx;
            ayi += mj_inv_r3 * dy;
            azi += mj_inv_r3 * dz;
        }

        // Store cross-acceleration and apply kick
        lax(i) += axi;
        lay(i) += ayi;
        laz(i) += azi;
        lvx(i) += axi * kick_dt;
        lvy(i) += ayi * kick_dt;
        lvz(i) += azi * kick_dt;
    });
    Kokkos::fence();
}

// Kick local fast particles from ghost slow particles
void ghost_cross_kick_fast_from_slow(Particles& local, const Particles& ghosts,
                                     int local_start, int local_count,
                                     double pivot, double G, double soft2, double kick_dt) {
    if (local_count == 0 || ghosts.N == 0) return;

    RawView lx{local.x.data()}, ly{local.y.data()}, lz{local.z.data()};
    RawView lvx{local.vx.data()}, lvy{local.vy.data()}, lvz{local.vz.data()};
    RawView lax{local.ax.data()}, lay{local.ay.data()}, laz{local.az.data()};

    RawView gx{ghosts.x.data()}, gy{ghosts.y.data()}, gz{ghosts.z.data()};
    RawView gm{ghosts.mass.data()};
    RawView gdt{ghosts.dt_adapt.data()};

    int N_ghosts = ghosts.N;

    // Each local fast particle accumulates force from all ghost slow particles
    Kokkos::parallel_for("ghost_cross_fast_from_slow", local_count, KOKKOS_LAMBDA(int ii) {
        int i = local_start + ii;
        double xi = lx(i), yi = ly(i), zi = lz(i);
        double axi = 0.0, ayi = 0.0, azi = 0.0;

        for (int j = 0; j < N_ghosts; j++) {
            // Only consider slow ghosts (dt >= pivot)
            if (gdt(j) < pivot) continue;

            double dx = gx(j) - xi;
            double dy = gy(j) - yi;
            double dz = gz(j) - zi;

            double r2 = dx*dx + dy*dy + dz*dz + soft2;
            double inv_r = fast_rsqrt(r2);
            double inv_r3 = inv_r * inv_r * inv_r;
            double mj_inv_r3 = G * gm(j) * inv_r3;

            axi += mj_inv_r3 * dx;
            ayi += mj_inv_r3 * dy;
            azi += mj_inv_r3 * dz;
        }

        // Store cross-acceleration and apply kick
        lax(i) += axi;
        lay(i) += ayi;
        laz(i) += azi;
        lvx(i) += axi * kick_dt;
        lvy(i) += ayi * kick_dt;
        lvz(i) += azi * kick_dt;
    });
    Kokkos::fence();
}

// Combined ghost cross-kicks for HHS (both directions)
void ghost_cross_kicks_hhs(Particles& local, const Particles& ghosts,
                           int slow_start, int slow_count,
                           int fast_start, int fast_count,
                           double pivot, double G, double soft2, double kick_dt) {
    // Slow local particles kicked by fast ghosts
    ghost_cross_kick_slow_from_fast(local, ghosts, slow_start, slow_count,
                                    pivot, G, soft2, kick_dt);

    // Fast local particles kicked by slow ghosts
    ghost_cross_kick_fast_from_slow(local, ghosts, fast_start, fast_count,
                                    pivot, G, soft2, kick_dt);
}

// Ghost cross-kicks with gradient correction for HHS middle kick
// This computes ghost cross-forces, applies gradient correction, and kicks
// The kicks are applied only with the ghost contribution (local cross is handled separately)
void ghost_cross_kicks_hhs_with_gradient(Particles& local, const Particles& ghosts,
                                         int slow_start, int slow_count,
                                         int fast_start, int fast_count,
                                         double pivot, double G, double soft2,
                                         double grad_dt, double kick_dt) {
    if (ghosts.N == 0) return;

    RawView lx{local.x.data()}, ly{local.y.data()}, lz{local.z.data()};
    RawView lvx{local.vx.data()}, lvy{local.vy.data()}, lvz{local.vz.data()};
    RawView lax{local.ax.data()}, lay{local.ay.data()}, laz{local.az.data()};

    RawView gx{ghosts.x.data()}, gy{ghosts.y.data()}, gz{ghosts.z.data()};
    RawView gax{ghosts.ax.data()}, gay{ghosts.ay.data()}, gaz{ghosts.az.data()};
    RawView gm{ghosts.mass.data()};
    RawView gdt{ghosts.dt_adapt.data()};

    const double grad_coeff = G * grad_dt * grad_dt * GRAD_COEFF;
    int N_ghosts = ghosts.N;

    // Combined: compute ghost cross-forces, gradient correction, and kick for slow particles
    if (slow_count > 0) {
        Kokkos::parallel_for("ghost_cross_grad_kick_slow", slow_count, KOKKOS_LAMBDA(int ii) {
            int i = slow_start + ii;
            double xi = lx(i), yi = ly(i), zi = lz(i);
            double axi_local = lax(i), ayi_local = lay(i), azi_local = laz(i);

            // Accumulate ghost force and gradient correction
            double axi = 0.0, ayi = 0.0, azi = 0.0;
            double cx = 0.0, cy = 0.0, cz = 0.0;

            for (int j = 0; j < N_ghosts; j++) {
                if (gdt(j) >= pivot) continue;  // Only fast ghosts

                double dx = gx(j) - xi;
                double dy = gy(j) - yi;
                double dz = gz(j) - zi;

                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r = fast_rsqrt(r2);
                double inv_r2 = inv_r * inv_r;
                double inv_r3 = inv_r2 * inv_r;
                double mj_inv_r3 = G * gm(j) * inv_r3;

                // Force contribution
                axi += mj_inv_r3 * dx;
                ayi += mj_inv_r3 * dy;
                azi += mj_inv_r3 * dz;

                // Gradient correction
                double dax = gax(j) - axi_local;
                double day = gay(j) - ayi_local;
                double daz = gaz(j) - azi_local;
                double inv_r5 = inv_r2 * inv_r2 * inv_r;
                double r_dot_a = dx*dax + dy*day + dz*daz;
                double f = grad_coeff * gm(j) * inv_r5;

                cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
            }

            // Total ghost contribution (force + gradient)
            double total_ax = axi + cx;
            double total_ay = ayi + cy;
            double total_az = azi + cz;

            // Update acceleration and apply kick
            lax(i) += total_ax;
            lay(i) += total_ay;
            laz(i) += total_az;
            lvx(i) += total_ax * kick_dt;
            lvy(i) += total_ay * kick_dt;
            lvz(i) += total_az * kick_dt;
        });
    }

    // Combined: compute ghost cross-forces, gradient correction, and kick for fast particles
    if (fast_count > 0) {
        Kokkos::parallel_for("ghost_cross_grad_kick_fast", fast_count, KOKKOS_LAMBDA(int ii) {
            int i = fast_start + ii;
            double xi = lx(i), yi = ly(i), zi = lz(i);
            double axi_local = lax(i), ayi_local = lay(i), azi_local = laz(i);

            // Accumulate ghost force and gradient correction
            double axi = 0.0, ayi = 0.0, azi = 0.0;
            double cx = 0.0, cy = 0.0, cz = 0.0;

            for (int j = 0; j < N_ghosts; j++) {
                if (gdt(j) < pivot) continue;  // Only slow ghosts

                double dx = gx(j) - xi;
                double dy = gy(j) - yi;
                double dz = gz(j) - zi;

                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r = fast_rsqrt(r2);
                double inv_r2 = inv_r * inv_r;
                double inv_r3 = inv_r2 * inv_r;
                double mj_inv_r3 = G * gm(j) * inv_r3;

                // Force contribution
                axi += mj_inv_r3 * dx;
                ayi += mj_inv_r3 * dy;
                azi += mj_inv_r3 * dz;

                // Gradient correction
                double dax = gax(j) - axi_local;
                double day = gay(j) - ayi_local;
                double daz = gaz(j) - azi_local;
                double inv_r5 = inv_r2 * inv_r2 * inv_r;
                double r_dot_a = dx*dax + dy*day + dz*daz;
                double f = grad_coeff * gm(j) * inv_r5;

                cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
            }

            // Total ghost contribution (force + gradient)
            double total_ax = axi + cx;
            double total_ay = ayi + cy;
            double total_az = azi + cz;

            // Update acceleration and apply kick
            lax(i) += total_ax;
            lay(i) += total_ay;
            laz(i) += total_az;
            lvx(i) += total_ax * kick_dt;
            lvy(i) += total_ay * kick_dt;
            lvz(i) += total_az * kick_dt;
        });
    }
    Kokkos::fence();
}

// ==========================================================================
// Explicit template instantiations
// ==========================================================================
template void compute_acceleration<true>(Particles&, double, double);
template void compute_acceleration<false>(Particles&, double, double);
template void compute_acceleration_and_timestep<true>(Particles&, double, double, double);
template void compute_acceleration_and_timestep<false>(Particles&, double, double, double);
template void self_acceleration<true>(Particles&, int, int, double, double);
template void self_acceleration<false>(Particles&, int, int, double, double);
template void self_gradient_correction<true>(Particles&, int, int, double, double, double);
template void self_gradient_correction<false>(Particles&, int, int, double, double, double);
template void self_acceleration_and_gradient<true>(Particles&, int, int, double, double, double);
template void self_acceleration_and_gradient<false>(Particles&, int, int, double, double, double);
template void cross_acceleration<true>(Particles&, int, int, int, int, double, double);
template void cross_acceleration<false>(Particles&, int, int, int, int, double, double);
template void cross_gradient_correction<true>(Particles&, int, int, int, int, double, double, double);
template void cross_gradient_correction<false>(Particles&, int, int, int, int, double, double, double);
template void cross_acceleration_and_gradient<true>(Particles&, int, int, int, int, double, double, double);
template void cross_acceleration_and_gradient<false>(Particles&, int, int, int, int, double, double, double);
template void compute_timestep_subset<true>(Particles&, int, int, double, double, double);
template void compute_timestep_subset<false>(Particles&, int, int, double, double, double);
template void self_acceleration_kick<true>(Particles&, int, int, double, double, double);
template void self_acceleration_kick<false>(Particles&, int, int, double, double, double);
template void self_acceleration_and_gradient_kick<true>(Particles&, int, int, double, double, double, double);
template void self_acceleration_and_gradient_kick<false>(Particles&, int, int, double, double, double, double);
template void cross_acceleration_kick<true>(Particles&, int, int, int, int, double, double, double);
template void cross_acceleration_kick<false>(Particles&, int, int, int, int, double, double, double);
template void cross_acceleration_and_gradient_kick<true>(Particles&, int, int, int, int, double, double, double, double);
template void cross_acceleration_and_gradient_kick<false>(Particles&, int, int, int, int, double, double, double, double);
template void cross_acceleration_kick_and_cache<true>(Particles&, int, int, int, int, double, double, double);
template void cross_acceleration_kick_and_cache<false>(Particles&, int, int, int, int, double, double, double);
