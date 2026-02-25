#include "forces.h"
#include "constants.h"
#include "integrator.h"
#ifdef ENABLE_COLLISIONAL
#include "collisional_config.h"
#include "subsystem.h"
#endif
#include <Kokkos_Core.hpp>
#include <cfloat>

// TILE_SIZE defined in constants.h

// ==========================================================================
// Full N^2 acceleration + adaptive timestep (fused)
// ==========================================================================
template<bool COLL>
static void compute_acceleration_and_timestep_impl(Particles& p, double G, double soft2, double eta) {
    const int N = p.N;
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView m{p.mass.data()};
    RawView dt_f{p.dt_adapt.data()};
    // Collisional: subsystem exclusion + neighbor detection (raw pointers to avoid View capture overhead)
    const int* sid_raw = COLL ? p.subsystem_id.data() : nullptr;
    const int* pid_raw = COLL ? p.particle_id.data() : nullptr;
#ifdef ENABLE_COLLISIONAL
    int* ngb_i_raw = COLL ? g_subsystem_mgr.ngb_pair_i.data() : nullptr;
    int* ngb_j_raw = COLL ? g_subsystem_mgr.ngb_pair_j.data() : nullptr;
    double* ngb_a_raw = COLL ? g_subsystem_mgr.ngb_pair_a.data() : nullptr;
    double* ngb_e_raw = COLL ? g_subsystem_mgr.ngb_pair_e.data() : nullptr;
    double* ngb_E_raw = COLL ? g_subsystem_mgr.ngb_pair_E.data() : nullptr;
    int* ngb_bound_raw = COLL ? g_subsystem_mgr.ngb_pair_bound.data() : nullptr;
    int* ngb_cnt_raw = COLL ? g_subsystem_mgr.ngb_count.data() : nullptr;
    int max_ngb = COLL ? g_subsystem_mgr.max_ngb_pairs : 0;
    double r_ngb_max_sq_val = COLL ? g_collisional.r_ngb_max_sq : 0.0;
    double r_ngb_val = COLL ? g_collisional.r_ngb : 0.0;
    double pivot_dt_val = COLL ? g_collisional.flyby_pivot_dt : 0.0;
#else
    int* ngb_i_raw = nullptr; int* ngb_j_raw = nullptr;
    double* ngb_a_raw = nullptr; double* ngb_e_raw = nullptr;
    double* ngb_E_raw = nullptr;
    int* ngb_bound_raw = nullptr; int* ngb_cnt_raw = nullptr;
    int max_ngb = 0; double r_ngb_max_sq_val = 0.0; double r_ngb_val = 0.0;
    double pivot_dt_val = 0.0;
#endif

    // Capture eta_ff²/eta_fb² for per-criterion timestep scaling
#ifdef ENABLE_COLLISIONAL
    double eta_ff_sq = g_collisional.eta_ff * g_collisional.eta_ff;
    double eta_fb_sq = g_collisional.eta_fb * g_collisional.eta_fb;
#else
    double eta_ff_sq = eta * eta;
    double eta_fb_sq = eta * eta;
#endif

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
            int sid_i = -1;
            if (i < N) {
                load4(&pos(i, 0), xi, yi, zi, mi);
                load4(&vel(i, 0), vxi, vyi, vzi, pad);
                if (COLL) sid_i = sid_raw[i];
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
                        // Subsystem exclusion: skip intra-subsystem pairs
                        if (COLL) { if (sid_i >= 0 && sid_raw[j] == sid_i) continue; }

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
                        double t_min_sq_pair = Kokkos::fmin(eta_ff_sq * t_ff_sq, eta_fb_sq * t_fb_sq);
                        t_min_i = Kokkos::fmin(t_min_i, t_min_sq_pair);

                        // Neighbor detection: identify bound/close pairs and unbound fly-bys
                        if (COLL) {
                            if (r2 < r_ngb_max_sq_val && i < j) {
                                double mu = G * mass_sum;
                                double r_val = Kokkos::sqrt(r2);
                                double E_orb = 0.5 * v_sq - mu / (r_val + EPS);
                                bool is_bound = (E_orb < 0);
                                double a_val = -mu / (2.0 * E_orb + EPS);
                                double Lx = dy*dvz - dz*dvy;
                                double Ly = dz*dvx - dx*dvz;
                                double Lz = dx*dvy - dy*dvx;
                                double L2 = Lx*Lx + Ly*Ly + Lz*Lz;
                                double e_val;
                                if (is_bound) {
                                    e_val = Kokkos::sqrt(Kokkos::fmax(0.0, 1.0 + 2.0*E_orb*L2/(mu*mu + EPS)));
                                } else {
                                    e_val = Kokkos::sqrt(Kokkos::fmax(1.0, 1.0 + 2.0*E_orb*L2/(mu*mu + EPS)));
                                }

                                bool is_neighbor = false;
                                if (is_bound) {
                                    is_neighbor = (a_val < r_ngb_val || r_val < r_ngb_val);
                                } else {
                                    // Unbound fly-by detection (BIFROST Eq. 4)
                                    // Hyperbolic orbit: a < 0, e > 1, r_peri = |a|*(e-1)
                                    double r_peri = Kokkos::fabs(a_val) * (e_val - 1.0);
                                    if (r_val < r_ngb_val) {
                                        is_neighbor = true;  // already close
                                    } else if (r_peri < r_ngb_val && v_sq > EPS) {
                                        // Check approach timescale: C * r / v < pivot_dt
                                        double t_approach = 0.5 * r_val / Kokkos::sqrt(v_sq);
                                        is_neighbor = (t_approach < pivot_dt_val);
                                    }
                                }

                                if (is_neighbor) {
                                    int idx = Kokkos::atomic_fetch_add(ngb_cnt_raw, 1);
                                    if (idx < max_ngb) {
                                        int pi = pid_raw[i], pj = pid_raw[j];
                                        if (pi > pj) { int tmp = pi; pi = pj; pj = tmp; }
                                        ngb_i_raw[idx] = pi; ngb_j_raw[idx] = pj;
                                        ngb_a_raw[idx] = a_val; ngb_e_raw[idx] = e_val;
                                        ngb_E_raw[idx] = E_orb;
                                        ngb_bound_raw[idx] = is_bound ? 1 : 0;
                                    }
                                }
                            }
                        }
                    }
                }
                team.team_barrier();
            }

            if (i < N) {
                pax(i) = axi; pay(i) = ayi; paz(i) = azi;
                dt_f(i) = Kokkos::sqrt(t_min_i);
            }
        });

}

void compute_acceleration_and_timestep(Particles& p, double G, double soft2, double eta) {
#ifdef ENABLE_COLLISIONAL
    if (p.collisional) compute_acceleration_and_timestep_impl<true>(p, G, soft2, eta);
    else
#endif
    compute_acceleration_and_timestep_impl<false>(p, G, soft2, eta);
}

// ==========================================================================
// Self-interaction: [start, start+count)
// ==========================================================================
template<bool COLL>
static void self_acceleration_impl(Particles& p, int start, int count,
                       double G, double soft2) {
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView m{p.mass.data()};
    const int* sid_raw = COLL ? p.subsystem_id.data() : nullptr;
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
            int sid_i = -1;
            if (ii < count) { xi = px(i); yi = py(i); zi = pz(i); if (COLL) sid_i = sid_raw[i]; }
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
                        if (COLL) { if (sid_i >= 0 && sid_raw[j] == sid_i) continue; }
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

}

void self_acceleration(Particles& p, int start, int count, double G, double soft2) {
#ifdef ENABLE_COLLISIONAL
    if (p.collisional) self_acceleration_impl<true>(p, start, count, G, soft2);
    else
#endif
    self_acceleration_impl<false>(p, start, count, G, soft2);
}

// ==========================================================================
// Self gradient correction
// ==========================================================================
template<bool COLL>
static void self_gradient_correction_impl(Particles& p, int start, int count,
                              double G, double soft2, double dt) {
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView m{p.mass.data()};
    RawView pgmag = COLL ? RawView{p.grad_mag.data()} : RawView{nullptr};
    const int* sid_raw = COLL ? p.subsystem_id.data() : nullptr;
    const double coeff = G * dt * dt * GRAD_COEFF;

    // Snapshot accelerations into scratch
    RawView ax0{p.scratch1.data()}, ay0{p.scratch2.data()}, az0{p.scratch3.data()};
    Kokkos::parallel_for("snap_self_grad", count, KOKKOS_LAMBDA(int ii) {
        ax0(ii) = pax(start + ii);
        ay0(ii) = pay(start + ii);
        az0(ii) = paz(start + ii);
    });

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
            int sid_i = -1;
            if (ii < count) {
                xi = px(i); yi = py(i); zi = pz(i);
                axi0 = ax0(ii); ayi0 = ay0(ii); azi0 = az0(ii);
                if (COLL) sid_i = sid_raw[i];
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
                        if (COLL) { if (sid_i >= 0 && sid_raw[start + jj] == sid_i) continue; }
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
            if (ii < count) {
                pax(i) += cx; pay(i) += cy; paz(i) += cz;
                if (COLL) pgmag(i) = Kokkos::sqrt(cx*cx + cy*cy + cz*cz);
            }
        });

}

void self_gradient_correction(Particles& p, int start, int count, double G, double soft2, double dt) {
#ifdef ENABLE_COLLISIONAL
    if (p.collisional) self_gradient_correction_impl<true>(p, start, count, G, soft2, dt);
    else
#endif
    self_gradient_correction_impl<false>(p, start, count, G, soft2, dt);
}

// ==========================================================================
// Cross-interaction: group A x group B
// ==========================================================================
template<bool COLL>
static void cross_acceleration_impl(Particles& p,
                        int a_start, int a_count,
                        int b_start, int b_count,
                        double G, double soft2) {
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView m{p.mass.data()};
    const int* sid_raw = COLL ? p.subsystem_id.data() : nullptr;
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
            int sid_i = -1;
            if (ii < a_count) { xi = px(i); yi = py(i); zi = pz(i); if (COLL) sid_i = sid_raw[i]; }
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
                        int j_cross = tile_start + jt;
                        if (COLL) { if (sid_i >= 0 && sid_raw[j_cross] == sid_i) continue; }
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
            int sid_j = -1;
            if (ii < b_count) { xj = px(j); yj = py(j); zj = pz(j); if (COLL) sid_j = sid_raw[j]; }
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
                        int i2_cross = tile_start + it;
                        if (COLL) { if (sid_j >= 0 && sid_raw[i2_cross] == sid_j) continue; }
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

}

void cross_acceleration(Particles& p, int a_start, int a_count,
                        int b_start, int b_count, double G, double soft2) {
#ifdef ENABLE_COLLISIONAL
    if (p.collisional) cross_acceleration_impl<true>(p, a_start, a_count, b_start, b_count, G, soft2);
    else
#endif
    cross_acceleration_impl<false>(p, a_start, a_count, b_start, b_count, G, soft2);
}

// ==========================================================================
// Cross gradient correction
// ==========================================================================
template<bool COLL>
static void cross_gradient_correction_impl(Particles& p,
                               int a_start, int a_count,
                               int b_start, int b_count,
                               double G, double soft2, double dt) {
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView m{p.mass.data()};
    RawView pgmag = COLL ? RawView{p.grad_mag.data()} : RawView{nullptr};
    const int* sid_raw = COLL ? p.subsystem_id.data() : nullptr;
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
            int sid_i = -1;
            if (ii < a_count) {
                xi = px(i); yi = py(i); zi = pz(i);
                axi0 = ax0_a(ii); ayi0 = ay0_a(ii); azi0 = az0_a(ii);
                if (COLL) sid_i = sid_raw[i];
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
                        int j_cross = b_start + tile_idx + jt;
                        if (COLL) { if (sid_i >= 0 && sid_raw[j_cross] == sid_i) continue; }
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
            if (ii < a_count) {
                pax(i) += cx; pay(i) += cy; paz(i) += cz;
                if (COLL) pgmag(i) = Kokkos::sqrt(cx*cx + cy*cy + cz*cz);
            }
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
            int sid_j = -1;
            if (ii < b_count) {
                xj = px(j); yj = py(j); zj = pz(j);
                axj0 = ax0_b(ii); ayj0 = ay0_b(ii); azj0 = az0_b(ii);
                if (COLL) sid_j = sid_raw[j];
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
                        int i2_cross = a_start + tile_idx + it;
                        if (COLL) { if (sid_j >= 0 && sid_raw[i2_cross] == sid_j) continue; }
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
            if (ii < b_count) {
                pax(j) += cx; pay(j) += cy; paz(j) += cz;
                if (COLL) pgmag(j) = Kokkos::sqrt(cx*cx + cy*cy + cz*cz);
            }
        });

}

void cross_gradient_correction(Particles& p, int a_start, int a_count,
                               int b_start, int b_count, double G, double soft2, double dt) {
#ifdef ENABLE_COLLISIONAL
    if (p.collisional) cross_gradient_correction_impl<true>(p, a_start, a_count, b_start, b_count, G, soft2, dt);
    else
#endif
    cross_gradient_correction_impl<false>(p, a_start, a_count, b_start, b_count, G, soft2, dt);
}

// ==========================================================================
// Compute adaptive timesteps for a subset [start, start+count)
// Only considers mutual interactions within the subset (for HHS recursion)
// ==========================================================================
template<bool COLL>
static void compute_timestep_subset_impl(Particles& p, int start, int count,
                          double G, double soft2, double eta,
                          bool symmetrize) {
    if (count <= 1) return;  // No pairs to consider

    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    RawView m{p.mass.data()};
    RawView dt_f{p.dt_adapt.data()};
    const int* sid_raw = COLL ? p.subsystem_id.data() : nullptr;
    const int* pid_raw = COLL ? p.particle_id.data() : nullptr;
#ifdef ENABLE_COLLISIONAL
    int* ngb_i_raw = COLL ? g_subsystem_mgr.ngb_pair_i.data() : nullptr;
    int* ngb_j_raw = COLL ? g_subsystem_mgr.ngb_pair_j.data() : nullptr;
    double* ngb_a_raw = COLL ? g_subsystem_mgr.ngb_pair_a.data() : nullptr;
    double* ngb_e_raw = COLL ? g_subsystem_mgr.ngb_pair_e.data() : nullptr;
    double* ngb_E_raw = COLL ? g_subsystem_mgr.ngb_pair_E.data() : nullptr;
    int* ngb_bound_raw = COLL ? g_subsystem_mgr.ngb_pair_bound.data() : nullptr;
    int* ngb_cnt_raw = COLL ? g_subsystem_mgr.ngb_count.data() : nullptr;
    int max_ngb = COLL ? g_subsystem_mgr.max_ngb_pairs : 0;
    double r_ngb_max_sq_val = COLL ? g_collisional.r_ngb_max_sq : 0.0;
    double r_ngb_val = COLL ? g_collisional.r_ngb : 0.0;
    double pivot_dt_val = COLL ? g_collisional.flyby_pivot_dt : 0.0;
#else
    int* ngb_i_raw = nullptr; int* ngb_j_raw = nullptr;
    double* ngb_a_raw = nullptr; double* ngb_e_raw = nullptr;
    double* ngb_E_raw = nullptr;
    int* ngb_bound_raw = nullptr; int* ngb_cnt_raw = nullptr;
    int max_ngb = 0; double r_ngb_max_sq_val = 0.0; double r_ngb_val = 0.0;
    double pivot_dt_val = 0.0;
#endif
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
#ifdef ENABLE_COLLISIONAL
    double eta_jerk_val = COLL ? g_collisional.eta_jerk : 0.0;
#else
    double eta_jerk_val = 0.0;
#endif
    const int end = start + count;

    // Initialize dt_adapt to large value for subset
    Kokkos::parallel_for("init_dt_subset", count, KOKKOS_LAMBDA(int ii) {
        dt_f(start + ii) = DBL_MAX;
    });


    // Capture eta_ff/eta_fb for per-criterion timestep scaling
#ifdef ENABLE_COLLISIONAL
    double eta_ff_v = g_collisional.eta_ff;
    double eta_fb_v = g_collisional.eta_fb;
#else
    double eta_ff_v = eta;
    double eta_fb_v = eta;
#endif
    double eta_ff_sq = eta_ff_v * eta_ff_v;
    double eta_fb_sq = eta_fb_v * eta_fb_v;

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
            int sid_i = -1;
            if (ii < count) {
                xi = px(i); yi = py(i); zi = pz(i);
                vxi = pvx(i); vyi = pvy(i); vzi = pvz(i);
                mi = m(i);
                if (COLL) sid_i = sid_raw[i];
            }

            double t_min_sq = 1e60;   // non-symmetrized path
            double t_min_i = 1e30;    // symmetrized path
            double jerk_x = 0, jerk_y = 0, jerk_z = 0;  // jerk accumulators (collisional only)

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
                        if (COLL) { if (sid_i >= 0 && sid_raw[j] == sid_i) continue; }

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

                        if (symmetrize) {
                            double inv_Gm = 1.0 / (G * mass_sum + EPS);
                            double inv_vsq = 1.0 / (v_sq + EPS);
                            double t_ff_sq = r3 * inv_Gm;
                            double t_fb_sq = r2s * inv_vsq;
                            double t_ff = t_ff_sq * fast_rsqrt(t_ff_sq + EPS);
                            double t_fb = t_fb_sq * fast_rsqrt(t_fb_sq + EPS);

                            double inv_r2 = inv_r * inv_r;
                            double r_dot_v = dx*dvx + dy*dvy + dz*dvz;

                            double c_ff = 1.5 * r_dot_v * inv_r2;
                            double c_fb = r_dot_v * inv_r2 * (1.0 + G * mass_sum * inv_r * inv_vsq);

                            double ct_ff = c_ff * t_ff_sq;
                            double s_ff = 1.0 + c_ff * ct_ff;
                            double u_ff = s_ff * fast_rsqrt(s_ff);
                            double t_sym_ff = t_ff + ct_ff / (u_ff + 1.0);

                            double ct_fb = c_fb * t_fb_sq;
                            double s_fb = 1.0 + c_fb * ct_fb;
                            double u_fb = s_fb * fast_rsqrt(s_fb);
                            double t_sym_fb = t_fb + ct_fb / (u_fb + 1.0);

                            t_min_i = Kokkos::fmin(t_min_i, Kokkos::fmin(eta_ff_v * t_sym_ff, eta_fb_v * t_sym_fb));
                        } else {
                            double t_ff_sq = r3 / (G * mass_sum + EPS);
                            double t_fb_sq = r2s / (v_sq + EPS);
                            double t_min_sq_pair = Kokkos::fmin(eta_ff_sq * t_ff_sq, eta_fb_sq * t_fb_sq);
                            t_min_sq = Kokkos::fmin(t_min_sq, t_min_sq_pair);
                        }

                        // Jerk accumulation (collisional only)
                        // j_ij = G*m_j * (v_ij/r³ - 3*(r·v)/(r^5) * r_ij)
                        if (COLL) {
                            double inv_r3 = inv_r * inv_r * inv_r;
                            double rdotv = dx*dvx + dy*dvy + dz*dvz;
                            double fac1 = G * mj * inv_r3;
                            double fac2 = 3.0 * rdotv * inv_r * inv_r;
                            jerk_x += fac1 * (dvx - fac2 * dx);
                            jerk_y += fac1 * (dvy - fac2 * dy);
                            jerk_z += fac1 * (dvz - fac2 * dz);
                        }

                        // Neighbor detection: identify bound/close pairs and unbound fly-bys
                        if (COLL) {
                            if (r2 < r_ngb_max_sq_val && i < j) {
                                double mu = G * mass_sum;
                                double r_val = Kokkos::sqrt(r2);
                                double E_orb = 0.5 * v_sq - mu / (r_val + EPS);
                                bool is_bound = (E_orb < 0);
                                double a_val = -mu / (2.0 * E_orb + EPS);
                                double Lx = dy*dvz - dz*dvy, Ly = dz*dvx - dx*dvz, Lz = dx*dvy - dy*dvx;
                                double L2 = Lx*Lx + Ly*Ly + Lz*Lz;
                                double e_val;
                                if (is_bound) {
                                    e_val = Kokkos::sqrt(Kokkos::fmax(0.0, 1.0 + 2.0*E_orb*L2/(mu*mu + EPS)));
                                } else {
                                    e_val = Kokkos::sqrt(Kokkos::fmax(1.0, 1.0 + 2.0*E_orb*L2/(mu*mu + EPS)));
                                }

                                bool is_neighbor = false;
                                if (is_bound) {
                                    is_neighbor = (a_val < r_ngb_val || r_val < r_ngb_val);
                                } else {
                                    // Unbound fly-by detection (BIFROST Eq. 4)
                                    double r_peri = Kokkos::fabs(a_val) * (e_val - 1.0);
                                    if (r_val < r_ngb_val) {
                                        is_neighbor = true;
                                    } else if (r_peri < r_ngb_val && v_sq > EPS) {
                                        double t_approach = 0.5 * r_val / Kokkos::sqrt(v_sq);
                                        is_neighbor = (t_approach < pivot_dt_val);
                                    }
                                }

                                if (is_neighbor) {
                                    int idx = Kokkos::atomic_fetch_add(ngb_cnt_raw, 1);
                                    if (idx < max_ngb) {
                                        int pi = pid_raw[i], pj = pid_raw[j];
                                        if (pi > pj) { int tmp = pi; pi = pj; pj = tmp; }
                                        ngb_i_raw[idx] = pi; ngb_j_raw[idx] = pj;
                                        ngb_a_raw[idx] = a_val; ngb_e_raw[idx] = e_val;
                                        ngb_E_raw[idx] = E_orb;
                                        ngb_bound_raw[idx] = is_bound ? 1 : 0;
                                    }
                                }
                            }
                        }
                    }
                }
                team.team_barrier();
            }

            if (ii < count) {
                double dt_val = symmetrize ? t_min_i : Kokkos::sqrt(t_min_sq);

                // Jerk timestep: dt_jerk = eta_j * |a| / |j| (collisional only)
                if (COLL && eta_jerk_val > 0.0) {
                    double a_mag = Kokkos::sqrt(pax(i)*pax(i) + pay(i)*pay(i) + paz(i)*paz(i));
                    double j_mag = Kokkos::sqrt(jerk_x*jerk_x + jerk_y*jerk_y + jerk_z*jerk_z);
                    if (j_mag > EPS) {
                        double dt_jerk = eta_jerk_val * a_mag / j_mag;
                        dt_val = Kokkos::fmin(dt_val, dt_jerk);
                    }
                }

                dt_f(i) = dt_val;
            }
        });

}

void compute_timestep_subset(Particles& p, int start, int count,
                          double G, double soft2, double eta, bool symmetrize) {
#ifdef ENABLE_COLLISIONAL
    if (p.collisional) compute_timestep_subset_impl<true>(p, start, count, G, soft2, eta, symmetrize);
    else
#endif
    compute_timestep_subset_impl<false>(p, start, count, G, soft2, eta, symmetrize);
}

// ==========================================================================
// Fused self_acceleration + symmetrized timestep (no kick)
// Computes forces AND timesteps with harmonic symmetrization in one N² pass.
// Forces stored in ax/ay/az for reuse by the first FSI kick.
// ==========================================================================
template<bool COLL>
static void self_acceleration_and_timestep_impl(Particles& p, int start, int count,
                                     double G, double soft2, double eta) {
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView m{p.mass.data()};
    RawView dt_f{p.dt_adapt.data()};
    const int* sid_raw = COLL ? p.subsystem_id.data() : nullptr;
    const int* pid_raw = COLL ? p.particle_id.data() : nullptr;
#ifdef ENABLE_COLLISIONAL
    int* ngb_i_raw = COLL ? g_subsystem_mgr.ngb_pair_i.data() : nullptr;
    int* ngb_j_raw = COLL ? g_subsystem_mgr.ngb_pair_j.data() : nullptr;
    double* ngb_a_raw = COLL ? g_subsystem_mgr.ngb_pair_a.data() : nullptr;
    double* ngb_e_raw = COLL ? g_subsystem_mgr.ngb_pair_e.data() : nullptr;
    double* ngb_E_raw = COLL ? g_subsystem_mgr.ngb_pair_E.data() : nullptr;
    int* ngb_bound_raw = COLL ? g_subsystem_mgr.ngb_pair_bound.data() : nullptr;
    int* ngb_cnt_raw = COLL ? g_subsystem_mgr.ngb_count.data() : nullptr;
    int max_ngb = COLL ? g_subsystem_mgr.max_ngb_pairs : 0;
    double r_ngb_max_sq_val = COLL ? g_collisional.r_ngb_max_sq : 0.0;
    double r_ngb_val = COLL ? g_collisional.r_ngb : 0.0;
    double pivot_dt_val = COLL ? g_collisional.flyby_pivot_dt : 0.0;
#else
    int* ngb_i_raw = nullptr; int* ngb_j_raw = nullptr;
    double* ngb_a_raw = nullptr; double* ngb_e_raw = nullptr;
    double* ngb_E_raw = nullptr;
    int* ngb_bound_raw = nullptr; int* ngb_cnt_raw = nullptr;
    int max_ngb = 0; double r_ngb_max_sq_val = 0.0; double r_ngb_val = 0.0;
    double pivot_dt_val = 0.0;
#endif
    const int end = start + count;

    // Capture eta_ff/eta_fb for per-criterion timestep scaling
#ifdef ENABLE_COLLISIONAL
    double eta_ff_v = g_collisional.eta_ff;
    double eta_fb_v = g_collisional.eta_fb;
#else
    double eta_ff_v = eta;
    double eta_fb_v = eta;
#endif

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
            int sid_i = -1;
            if (ii < count) {
                xi = px(i); yi = py(i); zi = pz(i); mi_val = m(i);
                vxi = pvx(i); vyi = pvy(i); vzi = pvz(i);
                if (COLL) sid_i = sid_raw[i];
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
                        if (COLL) { if (sid_i >= 0 && sid_raw[j] == sid_i) continue; }
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

                        // Symmetrized timestep — branchless, precomputed reciprocals
                        double dvx = tile(jb+4) - vxi;
                        double dvy = tile(jb+5) - vyi;
                        double dvz = tile(jb+6) - vzi;
                        double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
                        double mass_sum = mi_val + mj;
                        double inv_Gm = 1.0 / (G * mass_sum + EPS);
                        double inv_vsq = 1.0 / (v_sq + EPS);

                        double r3 = r2s * r2s * inv_r;
                        double t_ff_sq = r3 * inv_Gm;
                        double t_fb_sq = r2s * inv_vsq;
                        double t_ff = t_ff_sq * fast_rsqrt(t_ff_sq + EPS);
                        double t_fb = t_fb_sq * fast_rsqrt(t_fb_sq + EPS);

                        double inv_r2 = inv_r * inv_r;
                        double r_dot_v = dx*dvx + dy*dvy + dz*dvz;

                        // d*t products via c*t² (avoids computing d explicitly)
                        double c_ff = 1.5 * r_dot_v * inv_r2;
                        double c_fb = r_dot_v * inv_r2 * (1.0 + G * mass_sum * inv_r * inv_vsq);

                        // Rationalized form: t_sym = t + c*t²/(sqrt(1 + c²*t²) + 1)
                        double ct_ff = c_ff * t_ff_sq;
                        double s_ff = 1.0 + c_ff * ct_ff;
                        double u_ff = s_ff * fast_rsqrt(s_ff);
                        double t_sym_ff = t_ff + ct_ff / (u_ff + 1.0);

                        double ct_fb = c_fb * t_fb_sq;
                        double s_fb = 1.0 + c_fb * ct_fb;
                        double u_fb = s_fb * fast_rsqrt(s_fb);
                        double t_sym_fb = t_fb + ct_fb / (u_fb + 1.0);

                        t_min_i = Kokkos::fmin(t_min_i, Kokkos::fmin(eta_ff_v * t_sym_ff, eta_fb_v * t_sym_fb));

                        // Neighbor detection: identify bound/close pairs and unbound fly-bys
                        if (COLL) {
                            if (r2 < r_ngb_max_sq_val && i < j) {
                                double mu = G * (mi_val + mj);
                                double r_val = Kokkos::sqrt(r2);
                                double v_sq_nd = dvx*dvx + dvy*dvy + dvz*dvz;
                                double E_orb = 0.5 * v_sq_nd - mu / (r_val + EPS);
                                bool is_bound = (E_orb < 0);
                                double a_val = -mu / (2.0 * E_orb + EPS);
                                double Lx = dy*dvz - dz*dvy, Ly = dz*dvx - dx*dvz, Lz = dx*dvy - dy*dvx;
                                double L2 = Lx*Lx + Ly*Ly + Lz*Lz;
                                double e_val;
                                if (is_bound) {
                                    e_val = Kokkos::sqrt(Kokkos::fmax(0.0, 1.0 + 2.0*E_orb*L2/(mu*mu + EPS)));
                                } else {
                                    e_val = Kokkos::sqrt(Kokkos::fmax(1.0, 1.0 + 2.0*E_orb*L2/(mu*mu + EPS)));
                                }

                                bool is_neighbor = false;
                                if (is_bound) {
                                    is_neighbor = (a_val < r_ngb_val || r_val < r_ngb_val);
                                } else {
                                    // Unbound fly-by detection (BIFROST Eq. 4)
                                    double r_peri = Kokkos::fabs(a_val) * (e_val - 1.0);
                                    if (r_val < r_ngb_val) {
                                        is_neighbor = true;
                                    } else if (r_peri < r_ngb_val && v_sq_nd > EPS) {
                                        double t_approach = 0.5 * r_val / Kokkos::sqrt(v_sq_nd);
                                        is_neighbor = (t_approach < pivot_dt_val);
                                    }
                                }

                                if (is_neighbor) {
                                    int idx = Kokkos::atomic_fetch_add(ngb_cnt_raw, 1);
                                    if (idx < max_ngb) {
                                        int pi = pid_raw[i], pj = pid_raw[j];
                                        if (pi > pj) { int tmp = pi; pi = pj; pj = tmp; }
                                        ngb_i_raw[idx] = pi; ngb_j_raw[idx] = pj;
                                        ngb_a_raw[idx] = a_val; ngb_e_raw[idx] = e_val;
                                        ngb_E_raw[idx] = E_orb;
                                        ngb_bound_raw[idx] = is_bound ? 1 : 0;
                                    }
                                }
                            }
                        }
                    }
                }
                team.team_barrier();
            }
            if (ii < count) {
                pax(i) = axi; pay(i) = ayi; paz(i) = azi;
                dt_f(i) = t_min_i;
            }
        });

}

void self_acceleration_and_timestep(Particles& p, int start, int count,
                                     double G, double soft2, double eta) {
#ifdef ENABLE_COLLISIONAL
    if (p.collisional) self_acceleration_and_timestep_impl<true>(p, start, count, G, soft2, eta);
    else
#endif
    self_acceleration_and_timestep_impl<false>(p, start, count, G, soft2, eta);
}

