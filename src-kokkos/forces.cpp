#include "forces.h"
#include "constants.h"
#include <Kokkos_Core.hpp>
#include <cfloat>

bool grad_n3l = false;
bool use_kahan = false;
bool use_memory_tiling = true;  // Toggle shared memory tiling

// Tile size for shared memory tiling (tune for your GPU)
constexpr int TILE_SIZE = 128;

// ==========================================================================
// Full N-body acceleration: O(N(N-1)/2) with N3L or O(N^2) without
// ==========================================================================
template <bool UseN3L>
void compute_acceleration(Particles& p, double G, double soft2) {
    const int N = p.N;
    auto px = p.x; auto py = p.y; auto pz = p.z;
    auto pax = p.ax; auto pay = p.ay; auto paz = p.az;
    auto m = p.mass;

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
                double inv_r = 1.0 / Kokkos::sqrt(r2);
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
                double inv_r = 1.0 / Kokkos::sqrt(r2);
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
// Full N^2 acceleration + FROST timestep (fused)
// ==========================================================================
template <bool UseN3L>
void compute_acceleration_and_frost(Particles& p, double G, double soft2, double eta) {
    const int N = p.N;
    auto px = p.x; auto py = p.y; auto pz = p.z;
    auto pvx = p.vx; auto pvy = p.vy; auto pvz = p.vz;
    auto pax = p.ax; auto pay = p.ay; auto paz = p.az;
    auto m = p.mass;
    auto dt_f = p.dt_frost;

    Kokkos::parallel_for("zero_acc_frost", N, KOKKOS_LAMBDA(int i) {
        pax(i) = 0; pay(i) = 0; paz(i) = 0; dt_f(i) = DBL_MAX;
    });
    Kokkos::fence();

    if constexpr (UseN3L) {
        Kokkos::parallel_for("acc_frost_n3l", N, KOKKOS_LAMBDA(int i) {
            double axi = 0, ayi = 0, azi = 0;
            double t_min_i = DBL_MAX;

            for (int j = i + 1; j < N; j++) {
                double dx = px(j) - px(i);
                double dy = py(j) - py(i);
                double dz = pz(j) - pz(i);
                double r2 = dx*dx + dy*dy + dz*dz;
                double r2s = r2 + soft2;
                double inv_r = 1.0 / Kokkos::sqrt(r2s);
                double inv_r3 = inv_r * inv_r * inv_r;

                double fj = G * m(j) * inv_r3;
                double fi = G * m(i) * inv_r3;
                axi += fj * dx; ayi += fj * dy; azi += fj * dz;
                Kokkos::atomic_add(&pax(j), -fi * dx);
                Kokkos::atomic_add(&pay(j), -fi * dy);
                Kokkos::atomic_add(&paz(j), -fi * dz);

                // FROST
                double dvx = pvx(j) - pvx(i);
                double dvy = pvy(j) - pvy(i);
                double dvz = pvz(j) - pvz(i);
                double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
                double r_norm = Kokkos::sqrt(r2 + EPS);
                double v_norm = Kokkos::sqrt(v_sq + EPS);
                double mass_sum = m(i) + m(j);

                double t_ff = Kokkos::sqrt(r_norm * r_norm * r_norm / (G * mass_sum + EPS));
                double t_fb = r_norm / (v_norm + EPS);

                double r_dot_v = dx*dvx + dy*dvy + dz*dvz;
                double r2_safe = r2 + EPS;

                double dt_ff_dt = 1.5 * (r_dot_v / r2_safe) * t_ff;
                double factor = 1.0 + G * mass_sum / (v_sq * r_norm + EPS);
                double dt_fb_dt = (r_dot_v / r2_safe) * factor * t_fb;

                auto symmetrize = [](double t, double ddt) -> double {
                    double ddt_safe = (Kokkos::fabs(ddt) > EPS) ? ddt : EPS;
                    return t * (ddt + Kokkos::sqrt(1.0 + ddt * ddt) - 1.0) / ddt_safe;
                };

                double t_ff_sym = symmetrize(t_ff, dt_ff_dt);
                double t_fb_sym = symmetrize(t_fb, dt_fb_dt);
                double t_min_pair = Kokkos::fmin(t_ff_sym, t_fb_sym);

                t_min_i = Kokkos::fmin(t_min_i, t_min_pair);
                Kokkos::atomic_min(&dt_f(j), t_min_pair);
            }
            Kokkos::atomic_add(&pax(i), axi);
            Kokkos::atomic_add(&pay(i), ayi);
            Kokkos::atomic_add(&paz(i), azi);
            Kokkos::atomic_min(&dt_f(i), t_min_i);
        });
    } else if (use_memory_tiling) {
        // Tiled version using shared memory
        // Tile holds: x, y, z, vx, vy, vz, mass (7 doubles per particle)
        using team_policy = Kokkos::TeamPolicy<>;
        using member_type = team_policy::member_type;
        using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

        size_t scratch_size = TILE_SIZE * 7 * sizeof(double);
        int league_size = N;  // One team per particle

        Kokkos::parallel_for("acc_frost_tiled",
            team_policy(league_size, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_LAMBDA(const member_type& team) {
                int i = team.league_rank();
                if (i >= N) return;

                // Scratch memory views for tile data
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_x(team.team_scratch(0), TILE_SIZE);
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_y(reinterpret_cast<double*>(tile_x.data() + TILE_SIZE), TILE_SIZE);
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_z(reinterpret_cast<double*>(tile_y.data() + TILE_SIZE), TILE_SIZE);
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_vx(reinterpret_cast<double*>(tile_z.data() + TILE_SIZE), TILE_SIZE);
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_vy(reinterpret_cast<double*>(tile_vx.data() + TILE_SIZE), TILE_SIZE);
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_vz(reinterpret_cast<double*>(tile_vy.data() + TILE_SIZE), TILE_SIZE);
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_m(reinterpret_cast<double*>(tile_vz.data() + TILE_SIZE), TILE_SIZE);

                // Load particle i's data
                double xi = px(i), yi = py(i), zi = pz(i);
                double vxi = pvx(i), vyi = pvy(i), vzi = pvz(i);
                double mi = m(i);

                double axi = 0, ayi = 0, azi = 0;
                double t_min_i = DBL_MAX;

                // Loop over tiles
                for (int tile_start = 0; tile_start < N; tile_start += TILE_SIZE) {
                    int tile_end = (tile_start + TILE_SIZE < N) ? tile_start + TILE_SIZE : N;
                    int tile_count = tile_end - tile_start;

                    // Collaborative load: each thread loads one element
                    int t = team.team_rank();
                    if (t < tile_count) {
                        int idx = tile_start + t;
                        tile_x(t) = px(idx);
                        tile_y(t) = py(idx);
                        tile_z(t) = pz(idx);
                        tile_vx(t) = pvx(idx);
                        tile_vy(t) = pvy(idx);
                        tile_vz(t) = pvz(idx);
                        tile_m(t) = m(idx);
                    }
                    team.team_barrier();

                    // Accumulate forces and FROST from this tile
                    for (int jt = 0; jt < tile_count; jt++) {
                        int j = tile_start + jt;
                        if (j == i) continue;

                        double dx = tile_x(jt) - xi;
                        double dy = tile_y(jt) - yi;
                        double dz = tile_z(jt) - zi;
                        double r2 = dx*dx + dy*dy + dz*dz;
                        double r2s = r2 + soft2;
                        double inv_r = 1.0 / Kokkos::sqrt(r2s);
                        double inv_r3 = inv_r * inv_r * inv_r;

                        double fj = G * tile_m(jt) * inv_r3;
                        axi += fj * dx; ayi += fj * dy; azi += fj * dz;

                        // FROST
                        double dvx = tile_vx(jt) - vxi;
                        double dvy = tile_vy(jt) - vyi;
                        double dvz = tile_vz(jt) - vzi;
                        double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
                        double r_norm = Kokkos::sqrt(r2 + EPS);
                        double v_norm = Kokkos::sqrt(v_sq + EPS);
                        double mass_sum = mi + tile_m(jt);

                        double t_ff = Kokkos::sqrt(r_norm * r_norm * r_norm / (G * mass_sum + EPS));
                        double t_fb = r_norm / (v_norm + EPS);

                        double r_dot_v = dx*dvx + dy*dvy + dz*dvz;
                        double r2_safe = r2 + EPS;

                        double dt_ff_dt = 1.5 * (r_dot_v / r2_safe) * t_ff;
                        double factor = 1.0 + G * mass_sum / (v_sq * r_norm + EPS);
                        double dt_fb_dt = (r_dot_v / r2_safe) * factor * t_fb;

                        auto symmetrize = [](double t, double ddt) -> double {
                            double ddt_safe = (Kokkos::fabs(ddt) > EPS) ? ddt : EPS;
                            return t * (ddt + Kokkos::sqrt(1.0 + ddt * ddt) - 1.0) / ddt_safe;
                        };

                        double t_ff_sym = symmetrize(t_ff, dt_ff_dt);
                        double t_fb_sym = symmetrize(t_fb, dt_fb_dt);
                        double t_min_pair = Kokkos::fmin(t_ff_sym, t_fb_sym);
                        t_min_i = Kokkos::fmin(t_min_i, t_min_pair);
                    }
                    team.team_barrier();
                }

                pax(i) = axi; pay(i) = ayi; paz(i) = azi;
                dt_f(i) = t_min_i;
            });
    } else {
        // Original non-tiled version
        Kokkos::parallel_for("acc_frost_full", N, KOKKOS_LAMBDA(int i) {
            double axi = 0, ayi = 0, azi = 0;
            double t_min_i = DBL_MAX;

            for (int j = 0; j < N; j++) {
                if (j == i) continue;
                double dx = px(j) - px(i);
                double dy = py(j) - py(i);
                double dz = pz(j) - pz(i);
                double r2 = dx*dx + dy*dy + dz*dz;
                double r2s = r2 + soft2;
                double inv_r = 1.0 / Kokkos::sqrt(r2s);
                double inv_r3 = inv_r * inv_r * inv_r;

                double fj = G * m(j) * inv_r3;
                axi += fj * dx; ayi += fj * dy; azi += fj * dz;

                // FROST
                double dvx = pvx(j) - pvx(i);
                double dvy = pvy(j) - pvy(i);
                double dvz = pvz(j) - pvz(i);
                double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
                double r_norm = Kokkos::sqrt(r2 + EPS);
                double v_norm = Kokkos::sqrt(v_sq + EPS);
                double mass_sum = m(i) + m(j);

                double t_ff = Kokkos::sqrt(r_norm * r_norm * r_norm / (G * mass_sum + EPS));
                double t_fb = r_norm / (v_norm + EPS);

                double r_dot_v = dx*dvx + dy*dvy + dz*dvz;
                double r2_safe = r2 + EPS;

                double dt_ff_dt = 1.5 * (r_dot_v / r2_safe) * t_ff;
                double factor = 1.0 + G * mass_sum / (v_sq * r_norm + EPS);
                double dt_fb_dt = (r_dot_v / r2_safe) * factor * t_fb;

                auto symmetrize = [](double t, double ddt) -> double {
                    double ddt_safe = (Kokkos::fabs(ddt) > EPS) ? ddt : EPS;
                    return t * (ddt + Kokkos::sqrt(1.0 + ddt * ddt) - 1.0) / ddt_safe;
                };

                double t_ff_sym = symmetrize(t_ff, dt_ff_dt);
                double t_fb_sym = symmetrize(t_fb, dt_fb_dt);
                double t_min_pair = Kokkos::fmin(t_ff_sym, t_fb_sym);
                t_min_i = Kokkos::fmin(t_min_i, t_min_pair);
            }
            pax(i) = axi; pay(i) = ayi; paz(i) = azi;
            dt_f(i) = t_min_i;
        });
    }
    Kokkos::fence();

    // Apply eta
    Kokkos::parallel_for("apply_eta", N, KOKKOS_LAMBDA(int i) {
        dt_f(i) *= eta;
    });
    Kokkos::fence();
}

// ==========================================================================
// Self-interaction: [start, start+count)
// ==========================================================================
template <bool UseN3L>
void self_acceleration(Particles& p, int start, int count,
                       double G, double soft2) {
    auto px = p.x; auto py = p.y; auto pz = p.z;
    auto pax = p.ax; auto pay = p.ay; auto paz = p.az;
    auto m = p.mass;
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
                double inv_r = 1.0 / Kokkos::sqrt(r2);
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
        // Non-N3L: direct assignment, no zeroing needed, no branch
        Kokkos::parallel_for("self_acc_full", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            double axi = 0, ayi = 0, azi = 0;
            double xi = px(i), yi = py(i), zi = pz(i);
            for (int j = start; j < end; j++) {
                double dx = px(j) - xi;
                double dy = py(j) - yi;
                double dz = pz(j) - zi;
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r = 1.0 / Kokkos::sqrt(r2);
                double inv_r3 = inv_r * inv_r * inv_r;
                double fj = G * m(j) * inv_r3;
                axi += fj * dx; ayi += fj * dy; azi += fj * dz;
            }
            // Self contribution is 0 since dx=dy=dz=0
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
    auto px = p.x; auto py = p.y; auto pz = p.z;
    auto pax = p.ax; auto pay = p.ay; auto paz = p.az;
    auto m = p.mass;
    const double coeff = G * dt * dt * GRAD_COEFF;

    // Snapshot accelerations into scratch
    auto ax0 = p.scratch1; auto ay0 = p.scratch2; auto az0 = p.scratch3;
    Kokkos::parallel_for("snap_self_grad", count, KOKKOS_LAMBDA(int ii) {
        ax0(ii) = pax(start + ii);
        ay0(ii) = pay(start + ii);
        az0(ii) = paz(start + ii);
    });
    Kokkos::fence();

    if (grad_n3l && UseN3L) {
        // N3L gradient: upper triangle with atomics
        auto cx_acc = p.scratch4; auto cy_acc = p.scratch5; auto cz_acc = p.scratch6;
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
                double inv_r2 = 1.0 / r2;
                double inv_r = 1.0 / Kokkos::sqrt(r2);
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
                double inv_r2 = 1.0 / r2;
                double inv_r = 1.0 / Kokkos::sqrt(r2);
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
    auto px = p.x; auto py = p.y; auto pz = p.z;
    auto pax = p.ax; auto pay = p.ay; auto paz = p.az;
    auto m = p.mass;
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
                double inv_r = 1.0 / Kokkos::sqrt(r2);
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
                double inv_r = 1.0 / Kokkos::sqrt(r2);
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
                double inv_r = 1.0 / Kokkos::sqrt(r2);
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
    auto px = p.x; auto py = p.y; auto pz = p.z;
    auto pax = p.ax; auto pay = p.ay; auto paz = p.az;
    auto m = p.mass;
    const double coeff = G * dt * dt * GRAD_COEFF;

    // Snapshot accelerations
    auto ax0_a = p.scratch1; auto ay0_a = p.scratch2; auto az0_a = p.scratch3;
    auto ax0_b = p.scratch4; auto ay0_b = p.scratch5; auto az0_b = p.scratch6;
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
                double inv_r2 = 1.0 / r2;
                double inv_r = 1.0 / Kokkos::sqrt(r2);
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
                double inv_r2 = 1.0 / r2;
                double inv_r = 1.0 / Kokkos::sqrt(r2);
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
                double inv_r2 = 1.0 / r2;
                double inv_r = 1.0 / Kokkos::sqrt(r2);
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
// Compute FROST timesteps for a subset [start, start+count)
// Only considers mutual interactions within the subset (for HHS recursion)
// ==========================================================================
template <bool UseN3L>
void compute_frost_subset(Particles& p, int start, int count,
                          double G, double soft2, double eta) {
    if (count <= 1) return;  // No pairs to consider

    auto px = p.x; auto py = p.y; auto pz = p.z;
    auto pvx = p.vx; auto pvy = p.vy; auto pvz = p.vz;
    auto m = p.mass;
    auto dt_f = p.dt_frost;
    const int end = start + count;

    // Initialize dt_frost to large value for subset
    Kokkos::parallel_for("init_frost_subset", count, KOKKOS_LAMBDA(int ii) {
        dt_f(start + ii) = DBL_MAX;
    });
    Kokkos::fence();

    if constexpr (UseN3L) {
        // N3L: upper triangle with atomics
        Kokkos::parallel_for("frost_subset_n3l", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            double t_min_i = DBL_MAX;

            for (int j = i + 1; j < end; j++) {
                double dx = px(j) - px(i);
                double dy = py(j) - py(i);
                double dz = pz(j) - pz(i);
                double r2 = dx*dx + dy*dy + dz*dz;

                double dvx = pvx(j) - pvx(i);
                double dvy = pvy(j) - pvy(i);
                double dvz = pvz(j) - pvz(i);
                double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
                double r_norm = Kokkos::sqrt(r2 + EPS);
                double v_norm = Kokkos::sqrt(v_sq + EPS);
                double mass_sum = m(i) + m(j);

                double t_ff = Kokkos::sqrt(r_norm * r_norm * r_norm / (G * mass_sum + EPS));
                double t_fb = r_norm / (v_norm + EPS);

                double r_dot_v = dx*dvx + dy*dvy + dz*dvz;
                double r2_safe = r2 + EPS;

                double dt_ff_dt = 1.5 * (r_dot_v / r2_safe) * t_ff;
                double factor = 1.0 + G * mass_sum / (v_sq * r_norm + EPS);
                double dt_fb_dt = (r_dot_v / r2_safe) * factor * t_fb;

                auto symmetrize = [](double t, double ddt) -> double {
                    double ddt_safe = (Kokkos::fabs(ddt) > EPS) ? ddt : EPS;
                    return t * (ddt + Kokkos::sqrt(1.0 + ddt * ddt) - 1.0) / ddt_safe;
                };

                double t_ff_sym = symmetrize(t_ff, dt_ff_dt);
                double t_fb_sym = symmetrize(t_fb, dt_fb_dt);
                double t_min_pair = Kokkos::fmin(t_ff_sym, t_fb_sym);

                t_min_i = Kokkos::fmin(t_min_i, t_min_pair);
                Kokkos::atomic_min(&dt_f(j), t_min_pair);
            }
            Kokkos::atomic_min(&dt_f(i), t_min_i);
        });
    } else {
        // Full N^2 within subset
        Kokkos::parallel_for("frost_subset_full", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            double t_min_i = DBL_MAX;

            for (int j = start; j < end; j++) {
                if (j == i) continue;
                double dx = px(j) - px(i);
                double dy = py(j) - py(i);
                double dz = pz(j) - pz(i);
                double r2 = dx*dx + dy*dy + dz*dz;

                double dvx = pvx(j) - pvx(i);
                double dvy = pvy(j) - pvy(i);
                double dvz = pvz(j) - pvz(i);
                double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
                double r_norm = Kokkos::sqrt(r2 + EPS);
                double v_norm = Kokkos::sqrt(v_sq + EPS);
                double mass_sum = m(i) + m(j);

                double t_ff = Kokkos::sqrt(r_norm * r_norm * r_norm / (G * mass_sum + EPS));
                double t_fb = r_norm / (v_norm + EPS);

                double r_dot_v = dx*dvx + dy*dvy + dz*dvz;
                double r2_safe = r2 + EPS;

                double dt_ff_dt = 1.5 * (r_dot_v / r2_safe) * t_ff;
                double factor = 1.0 + G * mass_sum / (v_sq * r_norm + EPS);
                double dt_fb_dt = (r_dot_v / r2_safe) * factor * t_fb;

                auto symmetrize = [](double t, double ddt) -> double {
                    double ddt_safe = (Kokkos::fabs(ddt) > EPS) ? ddt : EPS;
                    return t * (ddt + Kokkos::sqrt(1.0 + ddt * ddt) - 1.0) / ddt_safe;
                };

                double t_ff_sym = symmetrize(t_ff, dt_ff_dt);
                double t_fb_sym = symmetrize(t_fb, dt_fb_dt);
                double t_min_pair = Kokkos::fmin(t_ff_sym, t_fb_sym);
                t_min_i = Kokkos::fmin(t_min_i, t_min_pair);
            }
            dt_f(i) = t_min_i;
        });
    }
    Kokkos::fence();

    // Apply eta
    Kokkos::parallel_for("apply_eta_subset", count, KOKKOS_LAMBDA(int ii) {
        dt_f(start + ii) *= eta;
    });
    Kokkos::fence();
}

// ==========================================================================
// Explicit template instantiations
// ==========================================================================
template void compute_acceleration<true>(Particles&, double, double);
template void compute_acceleration<false>(Particles&, double, double);
template void compute_acceleration_and_frost<true>(Particles&, double, double, double);
template void compute_acceleration_and_frost<false>(Particles&, double, double, double);
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
template void compute_frost_subset<true>(Particles&, int, int, double, double, double);
template void compute_frost_subset<false>(Particles&, int, int, double, double, double);
