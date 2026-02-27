#include "integrator.h"
#include "forces.h"
#include "timestep.h"
#include "constants.h"
#include "mpi_comm.h"
#include "profiling.h"
#include "tree.h"
#ifdef ENABLE_COLLISIONAL
#include "collisional_config.h"
#include "subsystem.h"
#include "logh_gpu.h"
#include "post_newtonian.h"
#include "secular.h"
#include "mergers.h"
#include "checkpoint.h"
#include "diagnostics.h"
#endif
#include <Kokkos_Core.hpp>
#include <vector>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <chrono>

// Global profiler instance
StepProfiler g_profiler;
SubsetHistogram g_subset_hist;

// Debug: HHS recursion call counters (per outer step)
static int dbg_hhs_distributed = 0;
static int dbg_hhs_local = 0;
static int dbg_hhs_cpu = 0;
static int dbg_hhs_slow0_dist = 0;
static int dbg_hhs_slow0_local = 0;
static int dbg_hhs_max_depth = 0;
// Per-depth histogram for local path (max 40 depths)
static constexpr int DBG_MAX_DEPTH = 40;
static int dbg_local_depth_calls[DBG_MAX_DEPTH] = {};
static long long dbg_local_depth_count[DBG_MAX_DEPTH] = {};  // sum of 'count' at each depth
static double dbg_local_depth_time[DBG_MAX_DEPTH] = {};      // inclusive wall time per local depth
static int dbg_dist_depth_calls[DBG_MAX_DEPTH] = {};         // per-depth call count (distributed)
static long long dbg_dist_depth_count[DBG_MAX_DEPTH] = {};   // sum of 'count' at each depth (distributed)
static int dbg_dist_depth_ngb_pairs[DBG_MAX_DEPTH] = {};     // neighbor pairs found per depth (distributed)
static int dbg_dist_depth_l0_clamped[DBG_MAX_DEPTH] = {};    // particles clamped by level-0 per depth

// RAII timer for per-depth inclusive wall time in local path
struct ScopedDepthTimer {
    int depth;
    std::chrono::high_resolution_clock::time_point t0;
    ScopedDepthTimer(int d) : depth(d), t0(std::chrono::high_resolution_clock::now()) {}
    ~ScopedDepthTimer() {
        auto t1 = std::chrono::high_resolution_clock::now();
        if (depth < DBG_MAX_DEPTH)
            dbg_local_depth_time[depth] += std::chrono::duration<double>(t1 - t0).count();
    }
};

// Serial N² GPU threshold: skip MPI, compute full N² locally (no tree)
int serial_n2_gpu = 0;

// Serial N² CPU threshold: use CPU serial loops for full N²
int serial_n2_cpu = 0;

// Fused force+gradient: use stale accelerations to compute gradient in same N² pass as force

bool use_integer_positions = false;
double int_pos_scale = 0.0;
double int_pos_inv_scale = 0.0;

bool use_nonblocking_mpi = false;

// symmetric_timesteps removed — always enabled (hardcoded true)

// Barnes-Hut tree configuration
bool use_tree = false;
int tree_threshold = 10000;
double tree_theta = 0.5;
int tree_n_leaf = 32;
int tree_n_group = 64;
bool gadget4_cross = false;

// Global tree instance (rebuilt each top-level step)
static Octree g_octree;

// TILE_SIZE defined in constants.h

// =============================================================================
// Distributed force + timestep computation for between-step updates
// Each rank computes for its slice [my_start, my_count) from ALL N particles
// Then allgather both accelerations and timestep values
// Uses tiled GPU kernel with shared memory for efficiency
// =============================================================================
void compute_acceleration_and_timestep_distributed(DistributedParticles& dp,
                                                 double G, double soft2, double eta) {
#ifdef USE_MPI
    if (MPIComm::size() == 1) {
        // Single rank: use optimized single-GPU version
        compute_acceleration_and_timestep(dp.local, G, soft2, eta);
        return;
    }

    const int N = dp.N_total;
    const int my_start = dp.my_start;
    const int my_count = dp.my_count;

    if (my_count == 0) {
        dp.allgather_accelerations();
        dp.allgather_timestep();
        return;
    }



    // Pack data for vectorized memory access
    dp.local.pack();

    auto pos = dp.local.pos_packed;
    auto vel = dp.local.vel_packed;
    RawView pax{dp.local.ax.data()}, pay{dp.local.ay.data()}, paz{dp.local.az.data()};
    RawView dt_f{dp.local.dt_adapt.data()};

    // Capture eta_ff²/eta_fb² for per-criterion timestep scaling
#ifdef ENABLE_COLLISIONAL
    double eta_ff_sq = g_collisional.eta_ff * g_collisional.eta_ff;
    double eta_fb_sq = g_collisional.eta_fb * g_collisional.eta_fb;
#else
    double eta_ff_sq = eta * eta;
    double eta_fb_sq = eta * eta;
#endif

    // Tiled GPU kernel: each team handles TILE_SIZE i-particles from my slice
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t scratch_size = TILE_SIZE * 7 * sizeof(double);
    int num_teams = (my_count + TILE_SIZE - 1) / TILE_SIZE;

    Kokkos::parallel_for("dist_acc_dt_tiled",
        team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * TILE_SIZE + team.team_rank();
            int i = my_start + ii;

            // AoS interleaved tile: [x,y,z,m,vx,vy,vz] per j-particle
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 7);

            double xi = 0, yi = 0, zi = 0, mi = 0;
            double vxi = 0, vyi = 0, vzi = 0;
            if (ii < my_count) {
                double vpad;
                load4(&pos(i, 0), xi, yi, zi, mi);
                load4(&vel(i, 0), vxi, vyi, vzi, vpad);
            }

            double axi = 0, ayi = 0, azi = 0;
            double t_min_i = DBL_MAX;

            for (int tile_start = 0; tile_start < N; tile_start += TILE_SIZE) {
                int tile_end = (tile_start + TILE_SIZE < N) ? tile_start + TILE_SIZE : N;
                int tile_count = tile_end - tile_start;

                int t = team.team_rank();
                if (t < tile_count) {
                    int idx = tile_start + t;
                    int base = t * 7;
                    double vpad;
                    load4(&pos(idx, 0), tile(base), tile(base+1), tile(base+2), tile(base+3));
                    load4(&vel(idx, 0), tile(base+4), tile(base+5), tile(base+6), vpad);
                }
                team.team_barrier();

                if (ii < my_count) {
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

                        double r3 = r2s * r2s * inv_r;
                        double t_ff_sq = r3 / (G * mass_sum + EPS);
                        double t_fb_sq = r2s / (v_sq + EPS);
                        double t_min_sq_pair = Kokkos::fmin(eta_ff_sq * t_ff_sq, eta_fb_sq * t_fb_sq);
                        t_min_i = Kokkos::fmin(t_min_i, t_min_sq_pair);
                    }
                }
                team.team_barrier();
            }

            if (ii < my_count) {
                pax(i) = axi;
                pay(i) = ayi;
                paz(i) = azi;
                dt_f(i) = Kokkos::sqrt(t_min_i);
            }
        });
    Kokkos::fence();

    // Allgather accelerations and timesteps
    dp.allgather_accelerations();
    dp.allgather_timestep();
#else
    compute_acceleration_and_timestep(dp.local, G, soft2, eta);
#endif
}


// Helper: drift positions for [start, start+count)
// Not static - also used by distributed FSI
void drift(Particles& p, int start, int count, double dt) {
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    if (use_integer_positions) {
        auto pix = p.ix; auto piy = p.iy; auto piz = p.iz;
        double scale = int_pos_scale;
        double inv_scale = int_pos_inv_scale;
        Kokkos::parallel_for("drift_int", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            pix(i) += llrint(pvx(i) * dt * inv_scale);
            piy(i) += llrint(pvy(i) * dt * inv_scale);
            piz(i) += llrint(pvz(i) * dt * inv_scale);
            px(i) = (double)pix(i) * scale;
            py(i) = (double)piy(i) * scale;
            pz(i) = (double)piz(i) * scale;
        });
    } else {
        Kokkos::parallel_for("drift", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            px(i) += pvx(i) * dt;
            py(i) += pvy(i) * dt;
            pz(i) += pvz(i) * dt;
        });
    }
    // No fence - next operation will handle dependency
}

#ifdef ENABLE_COLLISIONAL
void drift_field_only(Particles& p, int start, int count, double dt) {
    if (!p.collisional) {
        drift(p, start, count, dt);
        return;
    }
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    auto p_sid = p.subsystem_id;
    if (use_integer_positions) {
        auto pix = p.ix; auto piy = p.iy; auto piz = p.iz;
        double scale = int_pos_scale;
        double inv_scale = int_pos_inv_scale;
        Kokkos::parallel_for("drift_field_int", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            if (p_sid(i) >= 0) return;  // skip subsystem members
            pix(i) += llrint(pvx(i) * dt * inv_scale);
            piy(i) += llrint(pvy(i) * dt * inv_scale);
            piz(i) += llrint(pvz(i) * dt * inv_scale);
            px(i) = (double)pix(i) * scale;
            py(i) = (double)piy(i) * scale;
            pz(i) = (double)piz(i) * scale;
        });
    } else {
        Kokkos::parallel_for("drift_field", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            if (p_sid(i) >= 0) return;  // skip subsystem members
            px(i) += pvx(i) * dt;
            py(i) += pvy(i) * dt;
            pz(i) += pvz(i) * dt;
        });
    }
}

void advance_subsystems_in_range(Particles& p, int start, int count,
                                  double G, double dt) {
    if (!p.collisional || g_subsystem_mgr.active.empty()) return;

    // Find which subsystems have ALL members in [start, start+count)
    // Build pid->idx mapping on host
    auto h_pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);
    int max_pid = 0;
    for (int i = 0; i < p.N; i++) max_pid = std::max(max_pid, h_pid(i));
    std::vector<int> pid_to_idx(max_pid + 1, -1);
    for (int i = 0; i < p.N; i++) pid_to_idx[h_pid(i)] = i;

    // Filter: only subsystems fully contained in [start, start+count)
    std::vector<int> eligible;
    int end = start + count;
    for (int s = 0; s < (int)g_subsystem_mgr.active.size(); s++) {
        const auto& sub = g_subsystem_mgr.active[s];
        bool all_in = true;
        for (int pid : sub.particle_ids) {
            int idx = (pid <= max_pid) ? pid_to_idx[pid] : -1;
            if (idx < start || idx >= end) { all_in = false; break; }
        }
        if (all_in) eligible.push_back(s);
    }

    if (eligible.empty()) return;

    // Re-evaluate secular/LogH classification from current orbital state.
    // Subsystems persist across steps, so the detection-time is_secular may be stale.
    {
        auto h_x  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
        auto h_y  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
        auto h_z  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
        auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
        auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
        auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
        auto h_m  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);

        int n_orb_secular = g_collisional.n_orb_secular;

        for (int s : eligible) {
            auto& sub = g_subsystem_mgr.active[s];

            if (sub.n_bodies == 2) {
                int idx_i = pid_to_idx[sub.particle_ids[0]];
                int idx_j = pid_to_idx[sub.particle_ids[1]];
                if (idx_i < 0 || idx_j < 0) continue;

                double rx = h_x(idx_j) - h_x(idx_i);
                double ry = h_y(idx_j) - h_y(idx_i);
                double rz = h_z(idx_j) - h_z(idx_i);
                double vx = h_vx(idx_j) - h_vx(idx_i);
                double vy = h_vy(idx_j) - h_vy(idx_i);
                double vz = h_vz(idx_j) - h_vz(idx_i);
                double r2 = rx*rx + ry*ry + rz*rz;
                double v2 = vx*vx + vy*vy + vz*vz;
                double r_mag = std::sqrt(r2);
                double mu = G * (h_m(idx_i) + h_m(idx_j));
                double E_orb = 0.5 * v2 - mu / r_mag;

                // Update orbital elements
                sub.orb.E_orb = E_orb;
                if (E_orb < 0.0) {
                    double a = -mu / (2.0 * E_orb);
                    double L2 = (ry*vz-rz*vy)*(ry*vz-rz*vy) +
                                (rz*vx-rx*vz)*(rz*vx-rx*vz) +
                                (rx*vy-ry*vx)*(rx*vy-ry*vx);
                    double e2 = 1.0 - L2 / (mu * a);
                    sub.orb.a = a;
                    sub.orb.e = (e2 > 0.0) ? std::sqrt(e2) : 0.0;
                    sub.orb.period = 2.0 * M_PI * std::sqrt(a * a * a / mu);
                    sub.n_orb_per_dt = (int)(dt / sub.orb.period);
                    sub.is_secular = (sub.n_orb_per_dt >= n_orb_secular);

                    // Transition: was FLYBY but now bound → BINARY
                    if (sub.type == SubsystemType::FLYBY) sub.type = SubsystemType::BINARY;
                } else {
                    sub.orb.a = -mu / (2.0 * E_orb);  // hyperbolic
                    sub.is_secular = false;

                    // Transition: was BINARY but now unbound → FLYBY
                    if (sub.type == SubsystemType::BINARY) sub.type = SubsystemType::FLYBY;
                }

            } else if (sub.n_bodies == 3 && sub.inner_binary_id >= 0 && sub.outer_companion_pid >= 0) {
                // Recheck stability for triples with known hierarchy
                int inner_i = sub.inner_binary_id;
                int outer_pid = sub.outer_companion_pid;
                int inner_j = -1;
                for (int pid : sub.particle_ids) {
                    if (pid != inner_i && pid != outer_pid) { inner_j = pid; break; }
                }
                if (inner_j < 0) continue;

                int idx_i = (inner_i <= max_pid) ? pid_to_idx[inner_i] : -1;
                int idx_j = (inner_j <= max_pid) ? pid_to_idx[inner_j] : -1;
                int idx_o = (outer_pid <= max_pid) ? pid_to_idx[outer_pid] : -1;
                if (idx_i < 0 || idx_j < 0 || idx_o < 0) continue;

                double m1 = h_m(idx_i), m2 = h_m(idx_j), m3 = h_m(idx_o);
                double mu_in = G * (m1 + m2);

                // Inner binary elements
                double r_in[3] = {h_x(idx_j)-h_x(idx_i), h_y(idx_j)-h_y(idx_i), h_z(idx_j)-h_z(idx_i)};
                double v_in[3] = {h_vx(idx_j)-h_vx(idx_i), h_vy(idx_j)-h_vy(idx_i), h_vz(idx_j)-h_vz(idx_i)};
                auto oe_in = cartesian_to_elements(r_in, v_in, mu_in);

                if (oe_in.a > 0) {
                    sub.orb.a = oe_in.a;
                    sub.orb.e = oe_in.e;

                    // Outer orbit relative to inner COM
                    double M_in = m1 + m2;
                    double cx  = (m1*h_x(idx_i)  + m2*h_x(idx_j))  / M_in;
                    double cy  = (m1*h_y(idx_i)  + m2*h_y(idx_j))  / M_in;
                    double cz  = (m1*h_z(idx_i)  + m2*h_z(idx_j))  / M_in;
                    double cvx = (m1*h_vx(idx_i) + m2*h_vx(idx_j)) / M_in;
                    double cvy = (m1*h_vy(idx_i) + m2*h_vy(idx_j)) / M_in;
                    double cvz = (m1*h_vz(idx_i) + m2*h_vz(idx_j)) / M_in;
                    double r_out[3] = {h_x(idx_o)-cx, h_y(idx_o)-cy, h_z(idx_o)-cz};
                    double v_out[3] = {h_vx(idx_o)-cvx, h_vy(idx_o)-cvy, h_vz(idx_o)-cvz};
                    auto oe_out = cartesian_to_elements(r_out, v_out, G * (M_in + m3));

                    if (oe_out.a > 0) {
                        bool stable = SubsystemManager::mardling_stability(
                            oe_in.a, oe_in.e, oe_out.a, oe_out.e, m1, m2, m3);
                        sub.type = stable ? SubsystemType::TRIPLE_STABLE : SubsystemType::TRIPLE_UNSTABLE;

                        if (stable) {
                            double P_in = 2.0 * M_PI * std::sqrt(oe_in.a*oe_in.a*oe_in.a / mu_in);
                            sub.n_orb_per_dt = (int)(dt / P_in);
                            sub.is_secular = (sub.n_orb_per_dt >= n_orb_secular);
                        } else {
                            sub.is_secular = false;
                        }
                    } else {
                        sub.type = SubsystemType::TRIPLE_UNSTABLE;
                        sub.is_secular = false;
                    }
                } else {
                    sub.type = SubsystemType::TRIPLE_UNSTABLE;
                    sub.is_secular = false;
                }
            }
        }
    }

    // Partition eligible subsystems into secular binaries, secular triples, and LogH
    std::vector<int> secular_idx, secular_triple_idx, logh_idx;
    for (int s : eligible) {
        const auto& sub = g_subsystem_mgr.active[s];
        if (sub.is_secular && sub.n_bodies == 2) {
            secular_idx.push_back(s);
        } else if (sub.is_secular && sub.type == SubsystemType::TRIPLE_STABLE) {
            secular_triple_idx.push_back(s);
        } else {
            logh_idx.push_back(s);
        }
    }

    // --- Secular binaries: evolve on host ---
    if (!secular_idx.empty()) {
        // Copy particle data to host
        auto h_x  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
        auto h_y  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
        auto h_z  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
        auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
        auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
        auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
        auto h_m  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);

        for (int s : secular_idx) {
            const auto& sub = g_subsystem_mgr.active[s];
            int idx0 = pid_to_idx[sub.particle_ids[0]];
            int idx1 = pid_to_idx[sub.particle_ids[1]];

            double p1[3] = {h_x(idx0), h_y(idx0), h_z(idx0)};
            double v1[3] = {h_vx(idx0), h_vy(idx0), h_vz(idx0)};
            double p2[3] = {h_x(idx1), h_y(idx1), h_z(idx1)};
            double v2[3] = {h_vx(idx1), h_vy(idx1), h_vz(idx1)};

            g_subsystem_mgr.step_secular_evolved++;

            // Compute orbital energy before secular evolution for GW bookkeeping
            double m1_sec = h_m(idx0), m2_sec = h_m(idx1);
            double mu_sec = G * (m1_sec + m2_sec);
            double r_rel_pre[3] = {p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]};
            double v_rel_pre[3] = {v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]};
            auto oe_pre = cartesian_to_elements(r_rel_pre, v_rel_pre, mu_sec);
            double E_orb_pre = -G * m1_sec * m2_sec / (2.0 * oe_pre.a);

            integrate_secular_binary(p1, v1, h_m(idx0),
                                     p2, v2, h_m(idx1),
                                     G, g_collisional.c_code,
                                     g_collisional.pn_order, dt);

            // GW energy bookkeeping: energy radiated = E_orb_before - E_orb_after
            double r_rel_post[3] = {p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]};
            double v_rel_post[3] = {v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]};
            auto oe_post = cartesian_to_elements(r_rel_post, v_rel_post, mu_sec);
            if (oe_post.a > 0.0 && oe_pre.a > 0.0) {
                double E_orb_post = -G * m1_sec * m2_sec / (2.0 * oe_post.a);
                g_energy_books.E_gw += E_orb_pre - E_orb_post;
            }

            // Write back to host mirrors
            h_x(idx0) = p1[0]; h_y(idx0) = p1[1]; h_z(idx0) = p1[2];
            h_vx(idx0) = v1[0]; h_vy(idx0) = v1[1]; h_vz(idx0) = v1[2];
            h_x(idx1) = p2[0]; h_y(idx1) = p2[1]; h_z(idx1) = p2[2];
            h_vx(idx1) = v2[0]; h_vy(idx1) = v2[1]; h_vz(idx1) = v2[2];
        }

        // Copy modified host arrays back to device
        Kokkos::deep_copy(p.x, h_x);
        Kokkos::deep_copy(p.y, h_y);
        Kokkos::deep_copy(p.z, h_z);
        Kokkos::deep_copy(p.vx, h_vx);
        Kokkos::deep_copy(p.vy, h_vy);
        Kokkos::deep_copy(p.vz, h_vz);

        // Clear B_phys and aux_vel for secular subsystems (not used by secular path;
        // if subsystem transitions back to LogH, NAN/false triggers fresh init)
        for (int s : secular_idx) {
            auto& dst = g_subsystem_mgr.active[s];
            dst.B_phys = NAN;
            dst.has_aux_vel = false;
        }
    }

    // --- Secular triples: Kozai-Lidov + GW dissipation ---
    if (!secular_triple_idx.empty()) {
        auto h_x  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
        auto h_y  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
        auto h_z  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
        auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
        auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
        auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
        auto h_m  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);

        for (int s : secular_triple_idx) {
            g_subsystem_mgr.step_secular_triples++;
            const auto& sub = g_subsystem_mgr.active[s];
            if (sub.n_bodies != 3) continue;

            // Identify inner binary (first two PIDs) and outer companion (third)
            int idx0 = pid_to_idx[sub.particle_ids[0]];
            int idx1 = pid_to_idx[sub.particle_ids[1]];
            int idx2 = pid_to_idx[sub.particle_ids[2]];

            double m1 = h_m(idx0), m2 = h_m(idx1), m3 = h_m(idx2);
            double mu_in = G * (m1 + m2);

            // Inner binary relative coords
            double r_in[3] = {h_x(idx1)-h_x(idx0), h_y(idx1)-h_y(idx0), h_z(idx1)-h_z(idx0)};
            double v_in[3] = {h_vx(idx1)-h_vx(idx0), h_vy(idx1)-h_vy(idx0), h_vz(idx1)-h_vz(idx0)};

            // Inner binary -> elements
            OrbitalElements oe_in = cartesian_to_elements(r_in, v_in, mu_in);

            // Outer orbit: CM of inner binary vs outer companion
            double M12 = m1 + m2;
            double cm_x = (m1*h_x(idx0) + m2*h_x(idx1)) / M12;
            double cm_y = (m1*h_y(idx0) + m2*h_y(idx1)) / M12;
            double cm_z = (m1*h_z(idx0) + m2*h_z(idx1)) / M12;
            double cm_vx = (m1*h_vx(idx0) + m2*h_vx(idx1)) / M12;
            double cm_vy = (m1*h_vy(idx0) + m2*h_vy(idx1)) / M12;
            double cm_vz = (m1*h_vz(idx0) + m2*h_vz(idx1)) / M12;

            double r_out[3] = {h_x(idx2)-cm_x, h_y(idx2)-cm_y, h_z(idx2)-cm_z};
            double v_out[3] = {h_vx(idx2)-cm_vx, h_vy(idx2)-cm_vy, h_vz(idx2)-cm_vz};
            double mu_out = G * (M12 + m3);
            OrbitalElements oe_out = cartesian_to_elements(r_out, v_out, mu_out);

            if (oe_in.a <= 0 || oe_out.a <= 0) continue;  // unbound

            // Mutual inclination from angular momentum vectors
            double L_in[3] = {r_in[1]*v_in[2]-r_in[2]*v_in[1],
                              r_in[2]*v_in[0]-r_in[0]*v_in[2],
                              r_in[0]*v_in[1]-r_in[1]*v_in[0]};
            double L_out[3] = {r_out[1]*v_out[2]-r_out[2]*v_out[1],
                               r_out[2]*v_out[0]-r_out[0]*v_out[2],
                               r_out[0]*v_out[1]-r_out[1]*v_out[0]};
            double L_in_mag = std::sqrt(L_in[0]*L_in[0]+L_in[1]*L_in[1]+L_in[2]*L_in[2]);
            double L_out_mag = std::sqrt(L_out[0]*L_out[0]+L_out[1]*L_out[1]+L_out[2]*L_out[2]);
            double cos_i_mut = (L_in_mag > 0 && L_out_mag > 0) ?
                (L_in[0]*L_out[0]+L_in[1]*L_out[1]+L_in[2]*L_out[2]) / (L_in_mag*L_out_mag) : 1.0;

            // Build SecularTriple and evolve
            SecularTriple tri{};
            tri.inner.a = oe_in.a; tri.inner.e = oe_in.e; tri.inner.omega = oe_in.omega;
            tri.inner.inc = oe_in.inc; tri.inner.Omega = oe_in.Omega;
            tri.inner.m1 = m1; tri.inner.m2 = m2;
            tri.inner.pid1 = sub.particle_ids[0]; tri.inner.pid2 = sub.particle_ids[1];
            tri.a_out = oe_out.a; tri.e_out = oe_out.e;
            tri.omega_out = oe_out.omega; tri.inc_out = oe_out.inc; tri.Omega_out = oe_out.Omega;
            tri.m3 = m3; tri.pid3 = sub.particle_ids[2];
            tri.cos_i_mut = cos_i_mut;

            printf("  [SEC_TRI] pid=%d,%d,%d  a_in=%.4e  e_in=%.6f  a_out=%.4e  cos_i=%.4f  dt=%.4e\n",
                   sub.particle_ids[0], sub.particle_ids[1], sub.particle_ids[2],
                   tri.inner.a, tri.inner.e, tri.a_out, cos_i_mut, dt);

            double a_in_pre = tri.inner.a;  // save for GW bookkeeping

            secular_evolve_triple(tri, dt, G, g_collisional.c_code, g_collisional.pn_order);

            // GW energy bookkeeping for inner binary inspiral
            if (tri.inner.a > 0.0 && a_in_pre > 0.0) {
                double E_orb_pre_tri = -G * m1 * m2 / (2.0 * a_in_pre);
                double E_orb_post_tri = -G * m1 * m2 / (2.0 * tri.inner.a);
                g_energy_books.E_gw += E_orb_pre_tri - E_orb_post_tri;
            }

            // Update inner orbital plane to reflect changed mutual inclination.
            // The secular integrator evolves cos_i_mut but not (inc, Omega).
            // Rotate L_in around the node vector (L_out × L_in) by delta_i.
            {
                double L_in_hat[3] = {L_in[0]/L_in_mag, L_in[1]/L_in_mag, L_in[2]/L_in_mag};
                double L_out_hat[3] = {L_out[0]/L_out_mag, L_out[1]/L_out_mag, L_out[2]/L_out_mag};

                // Node vector = L_out × L_in (rotation axis)
                double nd[3] = {L_out_hat[1]*L_in_hat[2] - L_out_hat[2]*L_in_hat[1],
                                L_out_hat[2]*L_in_hat[0] - L_out_hat[0]*L_in_hat[2],
                                L_out_hat[0]*L_in_hat[1] - L_out_hat[1]*L_in_hat[0]};
                double nd_mag = std::sqrt(nd[0]*nd[0] + nd[1]*nd[1] + nd[2]*nd[2]);

                if (nd_mag > 1e-30) {
                    nd[0] /= nd_mag; nd[1] /= nd_mag; nd[2] /= nd_mag;

                    double i_old = std::acos(std::max(-1.0, std::min(1.0, cos_i_mut)));
                    double i_new = std::acos(std::max(-1.0, std::min(1.0, tri.cos_i_mut)));
                    double delta_i = i_new - i_old;

                    // Rodrigues' rotation: R(v,θ) = v cosθ + (k×v) sinθ + k(k·v)(1-cosθ)
                    double cd = std::cos(delta_i), sd = std::sin(delta_i);
                    double kdv = nd[0]*L_in_hat[0] + nd[1]*L_in_hat[1] + nd[2]*L_in_hat[2];
                    double kcv[3] = {nd[1]*L_in_hat[2] - nd[2]*L_in_hat[1],
                                     nd[2]*L_in_hat[0] - nd[0]*L_in_hat[2],
                                     nd[0]*L_in_hat[1] - nd[1]*L_in_hat[0]};
                    double L_new[3];
                    for (int k = 0; k < 3; k++)
                        L_new[k] = L_in_hat[k]*cd + kcv[k]*sd + nd[k]*kdv*(1.0-cd);

                    // Extract new inc, Omega from rotated L_in
                    double Lz = std::max(-1.0, std::min(1.0, L_new[2]));
                    tri.inner.inc = std::acos(Lz);
                    tri.inner.Omega = std::atan2(L_new[0], -L_new[1]);
                }
            }

            // Convert evolved elements back to Cartesian
            // Inner binary
            double n_in = std::sqrt(mu_in / (tri.inner.a * tri.inner.a * tri.inner.a));
            double M_in_new = oe_in.mean_anom + n_in * dt;
            double r_in_new[3], v_in_new[3];
            elements_to_cartesian(tri.inner.a, tri.inner.e, tri.inner.inc,
                                   tri.inner.Omega, tri.inner.omega, M_in_new, mu_in,
                                   r_in_new, v_in_new);

            // Outer orbit
            double n_out = std::sqrt(mu_out / (tri.a_out * tri.a_out * tri.a_out));
            double M_out_new = oe_out.mean_anom + n_out * dt;
            double r_out_new[3], v_out_new[3];
            elements_to_cartesian(tri.a_out, tri.e_out, tri.inc_out,
                                   tri.Omega_out, tri.omega_out, M_out_new, mu_out,
                                   r_out_new, v_out_new);

            // Reconstruct absolute coordinates
            // Total CM
            double M_tot = M12 + m3;
            double cm_tot_x = (M12*cm_x + m3*h_x(idx2)) / M_tot;
            double cm_tot_y = (M12*cm_y + m3*h_y(idx2)) / M_tot;
            double cm_tot_z = (M12*cm_z + m3*h_z(idx2)) / M_tot;
            double cm_tot_vx = (M12*cm_vx + m3*h_vx(idx2)) / M_tot;
            double cm_tot_vy = (M12*cm_vy + m3*h_vy(idx2)) / M_tot;
            double cm_tot_vz = (M12*cm_vz + m3*h_vz(idx2)) / M_tot;

            // Drift total CM
            cm_tot_x += cm_tot_vx * dt;
            cm_tot_y += cm_tot_vy * dt;
            cm_tot_z += cm_tot_vz * dt;

            // Outer orbit positions: CM_inner and particle 3
            double cm_in_new[3], v_cm_in_new[3];
            for (int k = 0; k < 3; k++) {
                cm_in_new[k] = (&cm_tot_x)[k] - (m3/M_tot) * r_out_new[k];
                v_cm_in_new[k] = (&cm_tot_vx)[k] - (m3/M_tot) * v_out_new[k];
            }

            // Inner binary: particle 0 and 1
            h_x(idx0) = cm_in_new[0] - (m2/M12)*r_in_new[0];
            h_y(idx0) = cm_in_new[1] - (m2/M12)*r_in_new[1];
            h_z(idx0) = cm_in_new[2] - (m2/M12)*r_in_new[2];
            h_vx(idx0) = v_cm_in_new[0] - (m2/M12)*v_in_new[0];
            h_vy(idx0) = v_cm_in_new[1] - (m2/M12)*v_in_new[1];
            h_vz(idx0) = v_cm_in_new[2] - (m2/M12)*v_in_new[2];

            h_x(idx1) = cm_in_new[0] + (m1/M12)*r_in_new[0];
            h_y(idx1) = cm_in_new[1] + (m1/M12)*r_in_new[1];
            h_z(idx1) = cm_in_new[2] + (m1/M12)*r_in_new[2];
            h_vx(idx1) = v_cm_in_new[0] + (m1/M12)*v_in_new[0];
            h_vy(idx1) = v_cm_in_new[1] + (m1/M12)*v_in_new[1];
            h_vz(idx1) = v_cm_in_new[2] + (m1/M12)*v_in_new[2];

            h_x(idx2) = cm_tot_x + (M12/M_tot)*r_out_new[0];
            h_y(idx2) = cm_tot_y + (M12/M_tot)*r_out_new[1];
            h_z(idx2) = cm_tot_z + (M12/M_tot)*r_out_new[2];
            h_vx(idx2) = cm_tot_vx + (M12/M_tot)*v_out_new[0];
            h_vy(idx2) = cm_tot_vy + (M12/M_tot)*v_out_new[1];
            h_vz(idx2) = cm_tot_vz + (M12/M_tot)*v_out_new[2];
        }

        Kokkos::deep_copy(p.x, h_x);
        Kokkos::deep_copy(p.y, h_y);
        Kokkos::deep_copy(p.z, h_z);
        Kokkos::deep_copy(p.vx, h_vx);
        Kokkos::deep_copy(p.vy, h_vy);
        Kokkos::deep_copy(p.vz, h_vz);

        for (int s : secular_triple_idx) {
            auto& dst = g_subsystem_mgr.active[s];
            dst.B_phys = NAN;
            dst.has_aux_vel = false;
        }
    }

    // --- LogH subsystems: batch integrate with per-subsystem restart (BIFROST §3.19) ---
    if (!logh_idx.empty()) {
        SubsystemManager logh_mgr;
        for (int s : logh_idx) {
            logh_mgr.active.push_back(g_subsystem_mgr.active[s]);
        }

        const int n_sub = (int)logh_mgr.active.size();
        const double eta_restart = g_collisional.eta_restart;
        constexpr int MAX_SUBDIV = 8;  // max 2^8 = 256 sub-steps

        // Save pre-integration state for subsystem members (needed for restart)
        struct SavedBody { double x, y, z, vx, vy, vz; };
        std::vector<std::vector<SavedBody>> saved_bodies(n_sub);
        std::vector<Subsystem> saved_subs(logh_mgr.active.begin(), logh_mgr.active.end());
        {
            auto hx  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
            auto hy  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
            auto hz  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
            auto hvx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
            auto hvy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
            auto hvz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
            for (int si = 0; si < n_sub; si++) {
                const auto& sub = logh_mgr.active[si];
                saved_bodies[si].resize(sub.n_bodies);
                for (int b = 0; b < sub.n_bodies; b++) {
                    int idx = pid_to_idx[sub.particle_ids[b]];
                    saved_bodies[si][b] = {hx(idx), hy(idx), hz(idx),
                                           hvx(idx), hvy(idx), hvz(idx)};
                }
            }
        }

        // Initial batch integration
        g_logh_batch.pack(logh_mgr, p, G, dt);
        logh_integrate_batch_gpu(g_logh_batch, G, g_collisional.eta_gbs);

        // Check per-subsystem energy errors
        auto h_eerr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                           g_logh_batch.energy_error);
        std::vector<int> failed;
        for (int si = 0; si < n_sub; si++) {
            double dE = h_eerr(si);
            if (eta_restart > 0 && (dE > eta_restart || std::isnan(dE))) {
                failed.push_back(si);
            }
        }

        // Unpack all results (passed get final state; failed get best-effort temporarily)
        g_logh_batch.unpack(logh_mgr, p);

        // Accumulate LogH stats into step-level counters
        g_subsystem_mgr.step_logh_calls += n_sub;
        g_subsystem_mgr.step_logh_gbs_total += g_logh_batch.stat_total_gbs;
        if (g_logh_batch.stat_worst_dE > g_subsystem_mgr.step_logh_worst_dE)
            g_subsystem_mgr.step_logh_worst_dE = g_logh_batch.stat_worst_dE;
        if (g_logh_batch.stat_best_dE < g_subsystem_mgr.step_logh_best_dE)
            g_subsystem_mgr.step_logh_best_dE = g_logh_batch.stat_best_dE;
        g_subsystem_mgr.step_logh_cap_hit += g_logh_batch.stat_cap_hit;

        // Retry failed subsystems with subdivided timestep (BIFROST §3.19)
        if (!failed.empty()) {
            g_subsystem_mgr.step_logh_restarts += (int)failed.size();
            std::printf("  LogH restart: %d/%d subs exceeded eta_restart=%.1e\n",
                        (int)failed.size(), n_sub, eta_restart);

            // Helper: restore failed subsystems' particle state from saved copy
            auto restore_failed = [&]() {
                auto hx  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
                auto hy  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
                auto hz  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
                auto hvx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
                auto hvy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
                auto hvz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
                for (int si : failed) {
                    const auto& sub = saved_subs[si];
                    for (int b = 0; b < sub.n_bodies; b++) {
                        int idx = pid_to_idx[sub.particle_ids[b]];
                        hx(idx)  = saved_bodies[si][b].x;
                        hy(idx)  = saved_bodies[si][b].y;
                        hz(idx)  = saved_bodies[si][b].z;
                        hvx(idx) = saved_bodies[si][b].vx;
                        hvy(idx) = saved_bodies[si][b].vy;
                        hvz(idx) = saved_bodies[si][b].vz;
                    }
                    logh_mgr.active[si] = saved_subs[si];
                }
                Kokkos::deep_copy(p.x, hx);
                Kokkos::deep_copy(p.y, hy);
                Kokkos::deep_copy(p.z, hz);
                Kokkos::deep_copy(p.vx, hvx);
                Kokkos::deep_copy(p.vy, hvy);
                Kokkos::deep_copy(p.vz, hvz);
            };

            restore_failed();

            // Subdivision retry loop: 2, 4, 8, ... sub-steps
            for (int subdiv = 1; subdiv <= MAX_SUBDIV; subdiv++) {
                int n_steps = 1 << subdiv;
                double dt_sub = dt / n_steps;

                SubsystemManager retry_mgr;
                for (int si : failed) retry_mgr.active.push_back(logh_mgr.active[si]);

                bool all_passed = true;
                double worst_dE = 0;

                for (int step = 0; step < n_steps; step++) {
                    SubsystemBatch retry_batch;
                    retry_batch.pack(retry_mgr, p, G, dt_sub);
                    logh_integrate_batch_gpu(retry_batch, G, g_collisional.eta_gbs);

                    auto h_e = Kokkos::create_mirror_view_and_copy(
                        Kokkos::HostSpace(), retry_batch.energy_error);
                    for (int r = 0; r < (int)retry_mgr.active.size(); r++) {
                        double dE = h_e(r);
                        if (dE > eta_restart || std::isnan(dE)) all_passed = false;
                        worst_dE = std::max(worst_dE, std::isnan(dE) ? 1e30 : dE);
                    }

                    retry_batch.unpack(retry_mgr, p);

                    if (!all_passed) break;  // no point continuing if a sub-step failed
                }

                if (all_passed) {
                    std::printf("  LogH restart: OK with %d sub-steps (worst dE=%.2e)\n",
                                n_steps, worst_dE);
                    g_subsystem_mgr.step_logh_restart_ok += (int)failed.size();
                    // Copy final persistent state back to logh_mgr
                    for (size_t r = 0; r < failed.size(); r++)
                        logh_mgr.active[failed[r]] = retry_mgr.active[r];
                    failed.clear();
                    break;
                }

                std::printf("  LogH restart: %d sub-steps failed (worst dE=%.2e)\n",
                            n_steps, worst_dE);

                if (subdiv < MAX_SUBDIV) restore_failed();
            }

            if (!failed.empty())
                std::printf("  LogH restart: GAVE UP after %d subdivisions\n", MAX_SUBDIV);
        }

        // Propagate persistent state back from logh_mgr to g_subsystem_mgr
        for (int i = 0; i < n_sub; i++) {
            auto& dst = g_subsystem_mgr.active[logh_idx[i]];
            const auto& src = logh_mgr.active[i];
            dst.B_phys = src.B_phys;
            dst.has_aux_vel = src.has_aux_vel;
            if (src.has_aux_vel) {
                std::memcpy(dst.aux_vel_phys, src.aux_vel_phys,
                            LOGH_MAX_BODIES * 3 * sizeof(double));
            }
            dst.V_scale = src.V_scale;
            dst.R_scale = src.R_scale;
            dst.T_scale = src.T_scale;
            dst.E_scale = src.E_scale;
            dst.scales_set = src.scales_set;
        }
    }
}
#endif

void fsi_step(Particles& p, int start, int count,
              double G, double soft2, double dt, bool forces_valid,
              bool force_direct_n2) {
    g_subset_hist.record(SubsetHistogram::FSI_STEP, count);
    bool tree_mode = !force_direct_n2 && (use_tree && count >= tree_threshold);
    bool tree_built = false;

    // U kick (dt/6)
    if (forces_valid) {
        g_profiler.start();
        kick_velocities(p, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else if (tree_mode) {
        g_profiler.start();
        g_octree.n_leaf = tree_n_leaf;
        g_octree.n_group = tree_n_group;
        g_octree.build(p, start, count, tree_theta);
        tree_self_acceleration(p, g_octree, start, count, G, soft2);
        tree_momentum_correction(p, start, count);
        tree_built = true;
        g_profiler.stop(StepProfiler::SELF_FORCE);
        g_profiler.start();
        kick_velocities(p, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else {
        g_profiler.start();
        self_acceleration(p, start, count, G, soft2);
        g_profiler.stop(StepProfiler::SELF_FORCE);
        g_profiler.start();
        kick_velocities(p, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // T drift (dt/2) — field particles only if subsystems active
    g_profiler.start();
#ifdef ENABLE_COLLISIONAL
    if (p.collisional && !g_subsystem_mgr.active.empty()) {
        drift_field_only(p, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
        g_profiler.start();
        advance_subsystems_in_range(p, start, count, G, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::SUBSYSTEM_INTEG);
    } else
#endif
    {
        drift(p, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // U-tilde kick (2dt/3)
    if (tree_mode) {
        g_profiler.start();
        if (!tree_built) {
            g_octree.n_leaf = tree_n_leaf;
            g_octree.n_group = tree_n_group;
            g_octree.build(p, start, count, tree_theta);
            tree_built = true;
        } else {
            g_octree.refresh_positions(p);
        }
        tree_self_acceleration(p, g_octree, start, count, G, soft2);
        tree_momentum_correction(p, start, count);
        g_profiler.stop(StepProfiler::SELF_FORCE);
        g_profiler.start();
        kick_velocities(p, start, count, dt * FSI_MOD_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else {
        g_profiler.start();
        self_acceleration(p, start, count, G, soft2);
        g_profiler.stop(StepProfiler::SELF_FORCE);
        g_profiler.start();
        self_gradient_correction(p, start, count, G, soft2, dt);
        g_profiler.stop(StepProfiler::SELF_GRAD);
        g_profiler.start();
        kick_velocities(p, start, count, dt * FSI_MOD_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // T drift (dt/2) — field particles only if subsystems active
    g_profiler.start();
#ifdef ENABLE_COLLISIONAL
    if (p.collisional && !g_subsystem_mgr.active.empty()) {
        drift_field_only(p, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
        g_profiler.start();
        advance_subsystems_in_range(p, start, count, G, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::SUBSYSTEM_INTEG);
    } else
#endif
    {
        drift(p, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // U kick (dt/6)
    if (tree_mode) {
        g_profiler.start();
        g_octree.refresh_positions(p);
        tree_self_acceleration(p, g_octree, start, count, G, soft2);
        tree_momentum_correction(p, start, count);
        g_profiler.stop(StepProfiler::SELF_FORCE);
        g_profiler.start();
        kick_velocities(p, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else {
        g_profiler.start();
        self_acceleration(p, start, count, G, soft2);
        g_profiler.stop(StepProfiler::SELF_FORCE);
        g_profiler.start();
        kick_velocities(p, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }
}

// ==========================================================================
// Dynamic pivot-based partitioning for HHS (per paper's Algorithm 4)
// Partitions particles [start, start+count) into:
//   - slow: dt_adapt >= pivot (at beginning)
//   - fast: dt_adapt < pivot (at end)
// Returns the count of slow particles
//
// IMPORTANT: This must be DETERMINISTIC across MPI ranks! We use prefix sums
// instead of atomics to ensure all ranks produce identical permutations.
// ==========================================================================
static int partition_by_pivot(Particles& p, int start, int count, double pivot) {
    if (count <= 1) return count;

    RawView dt_f{p.dt_adapt.data()};
    auto perm = p.scratch_i;
    auto is_slow = p.scratch_i2;  // temporary: 1 if slow, 0 if fast

    // Fused mark + exclusive prefix sum in single kernel
    // is_slow(ii) = 1 if slow, 0 if fast
    // slow_prefix(ii) = number of slow particles with index < ii
    auto slow_prefix = p.scratch_i3;
    Kokkos::parallel_scan("mark_prefix_slow", count,
        KOKKOS_LAMBDA(int ii, int& partial_sum, bool is_final) {
            int val = (dt_f(start + ii) >= pivot) ? 1 : 0;
            if (is_final) {
                is_slow(ii) = val;
                slow_prefix(ii) = partial_sum;
            }
            partial_sum += val;
        });
    Kokkos::fence();

    // Get total slow count: deep_copy last element instead of launching a kernel
    int last_prefix = 0, last_is_slow = 0;
    Kokkos::deep_copy(last_prefix, Kokkos::subview(slow_prefix, count - 1));
    Kokkos::deep_copy(last_is_slow, Kokkos::subview(is_slow, count - 1));
    int slow_count = last_prefix + last_is_slow;

    if (slow_count == 0 || slow_count == count) {
        return slow_count;
    }

    // Step 3: Build permutation deterministically
    // - Slow particle at position ii goes to slow_prefix(ii)
    // - Fast particle at position ii goes to slow_count + (ii - slow_prefix(ii) - (1-is_slow(ii)))
    //   = slow_count + (ii - slow_prefix(ii)) for fast particles
    int sc = slow_count;
    Kokkos::parallel_for("build_perm_deterministic", count, KOKKOS_LAMBDA(int ii) {
        int dest;
        if (is_slow(ii)) {
            dest = slow_prefix(ii);
        } else {
            // Fast particles: position among fast = ii - slow_prefix(ii)
            // (since slow_prefix(ii) counts slow particles before ii)
            dest = sc + (ii - slow_prefix(ii));
        }
        perm(dest) = start + ii;
    });

    // Apply permutation: fused gather of all fields
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView pmass{p.mass.data()}, pdt{p.dt_adapt.data()};
    auto porig = p.orig_index;
    auto plevel = p.level;
    auto pix = p.ix; auto piy = p.iy; auto piz = p.iz;

    // Scratch: 11 doubles + 2 ints + 3 int64s
    auto t1 = p.scratch1;  auto t2 = p.scratch2;  auto t3 = p.scratch3;
    auto t4 = p.scratch4;  auto t5 = p.scratch5;  auto t6 = p.scratch6;
    auto t7 = p.scratch7;  auto t8 = p.scratch8;  auto t9 = p.scratch9;
    auto t10 = p.scratch10; auto t11 = p.scratch11;
    auto ti1 = p.scratch_i2; auto ti2 = p.scratch_i3;
    auto tl1 = p.scratch_l1; auto tl2 = p.scratch_l2; auto tl3 = p.scratch_l3;

    // Single fused gather of all fields
    Kokkos::parallel_for("gather_all", count, KOKKOS_LAMBDA(int ii) {
        int src = perm(ii);
        t1(ii) = px(src);   t2(ii) = py(src);   t3(ii) = pz(src);
        t4(ii) = pvx(src);  t5(ii) = pvy(src);  t6(ii) = pvz(src);
        t7(ii) = pax(src);  t8(ii) = pay(src);  t9(ii) = paz(src);
        t10(ii) = pmass(src); t11(ii) = pdt(src);
        ti1(ii) = porig(src); ti2(ii) = plevel(src);
        tl1(ii) = pix(src); tl2(ii) = piy(src); tl3(ii) = piz(src);
    });

    // Single fused copy back of all fields
    Kokkos::parallel_for("copy_all", count, KOKKOS_LAMBDA(int ii) {
        int dst = start + ii;
        px(dst) = t1(ii);   py(dst) = t2(ii);   pz(dst) = t3(ii);
        pvx(dst) = t4(ii);  pvy(dst) = t5(ii);  pvz(dst) = t6(ii);
        pax(dst) = t7(ii);  pay(dst) = t8(ii);  paz(dst) = t9(ii);
        pmass(dst) = t10(ii); pdt(dst) = t11(ii);
        porig(dst) = ti1(ii); plevel(dst) = ti2(ii);
        pix(dst) = tl1(ii); piy(dst) = tl2(ii); piz(dst) = tl3(ii);
    });

#ifdef ENABLE_COLLISIONAL
    // Second pass: permute collisional-only fields
    if (p.collisional) {
        auto ppid = p.particle_id;   auto psid = p.subsystem_id;
        auto pstype = p.stellar_type;
        auto pmzams = p.m_zams;      auto psage = p.stellar_age;
        auto psrad = p.stellar_radius; auto plum = p.luminosity;
        auto pcmass = p.core_mass;   auto psmag = p.spin_mag;
        auto pdtl0 = p.dt_level0;
        auto pgmag = p.grad_mag;
        auto pspin = p.spin_vec;

        Kokkos::parallel_for("gather_coll_pivot", count, KOKKOS_LAMBDA(int ii) {
            int src = perm(ii);
            ti1(ii) = ppid(src);
            ti2(ii) = psid(src);
            tl1(ii) = pstype(src);
            t1(ii) = pmzams(src);
            t2(ii) = psage(src);
            t3(ii) = psrad(src);
            t4(ii) = plum(src);
            t5(ii) = pcmass(src);
            t6(ii) = psmag(src);
            t7(ii) = pdtl0(src);
            t8(ii) = pgmag(src);
        });
        Kokkos::parallel_for("copy_coll_pivot", count, KOKKOS_LAMBDA(int ii) {
            int dst = start + ii;
            ppid(dst) = ti1(ii);
            psid(dst) = ti2(ii);
            pstype(dst) = (int)tl1(ii);
            pmzams(dst) = t1(ii);
            psage(dst) = t2(ii);
            psrad(dst) = t3(ii);
            plum(dst) = t4(ii);
            pcmass(dst) = t5(ii);
            psmag(dst) = t6(ii);
            pdtl0(dst) = t7(ii);
            pgmag(dst) = t8(ii);
        });

        // Spin vector (3 components)
        Kokkos::parallel_for("gather_spin_pivot", count, KOKKOS_LAMBDA(int ii) {
            int src = perm(ii);
            t1(ii) = pspin(src, 0);
            t2(ii) = pspin(src, 1);
            t3(ii) = pspin(src, 2);
        });
        Kokkos::parallel_for("copy_spin_pivot", count, KOKKOS_LAMBDA(int ii) {
            int dst = start + ii;
            pspin(dst, 0) = t1(ii);
            pspin(dst, 1) = t2(ii);
            pspin(dst, 2) = t3(ii);
        });
    }
#endif

    return slow_count;
}

// ==========================================================================
// HHS-FSI with dynamic pivot-based partitioning (per paper's Algorithm 4)
// ==========================================================================
static void hhs_fsi_step_dynamic_impl(Particles& p, int start, int count, double pivot_dt,
                          double G, double soft2, double dt_min,
                          double eta, bool dt_valid, int depth = 0) {
    if (count == 0) return;

    // BIFROST Section 3.5: fresh neighbor search + subsystem detection at each level,
    // then timestep assignment with subsystem exclusion.
    if (!dt_valid && count > 1) {
#ifdef ENABLE_COLLISIONAL
        // Step 1: Standalone neighbor search (no subsystem exclusion, no forces)
        if (p.collisional) {
            g_profiler.start();
            g_subsystem_mgr.reset_ngb_count();
            neighbor_search_subset(p, start, count, G,
                                   g_collisional.r_ngb, g_collisional.r_ngb_max, pivot_dt);
            g_subsystem_mgr.detect_from_pairs(p, G, pivot_dt,
                                               g_collisional.r_ngb, g_collisional.r_ngb_max,
                                               g_collisional.n_orb_secular);
            g_subsystem_mgr.update_particle_flags(p);
            g_subsystem_mgr.accumulate_diagnostics();
            g_profiler.stop(StepProfiler::SUBSYSTEM_DETECT);
        }
#endif
        // Step 2: Timestep computation (with subsystem exclusion active)
        g_profiler.start();
        compute_timestep_subset(p, start, count, G, soft2, eta, true);
#ifdef ENABLE_COLLISIONAL
        // Gradient timestep criteria (BIFROST Section 3.4.4)
        if (p.collisional && g_collisional.eta_grad > 0.0) {
            auto dt_f = p.dt_adapt;
            auto gm = p.grad_mag;
            RawView agx{p.ax.data()}, agy{p.ay.data()}, agz{p.az.data()};
            RawView vvx{p.vx.data()}, vvy{p.vy.data()}, vvz{p.vz.data()};
            double eta_g = g_collisional.eta_grad;
            Kokkos::parallel_for("dt_grad", count, KOKKOS_LAMBDA(int ii) {
                int i = start + ii;
                double g = gm(i);
                if (g > EPS) {
                    double a_mag = Kokkos::sqrt(agx(i)*agx(i) + agy(i)*agy(i) + agz(i)*agz(i));
                    double v_mag = Kokkos::sqrt(vvx(i)*vvx(i) + vvy(i)*vvy(i) + vvz(i)*vvz(i));
                    if (a_mag > EPS) {
                        double dt_g2 = eta_g * Kokkos::sqrt(a_mag / g);
                        if (dt_g2 < dt_f(i)) dt_f(i) = dt_g2;
                    }
                    if (v_mag > EPS) {
                        double dt_g1 = eta_g * Kokkos::cbrt(v_mag / g);
                        if (dt_g1 < dt_f(i)) dt_f(i) = dt_g1;
                    }
                }
            });
        }
        // Level-0 clamping (BIFROST Section 3.4.1)
        if (p.collisional && depth > 0) {
            auto dt_f = p.dt_adapt;
            auto dt_l0 = p.dt_level0;
            Kokkos::parallel_for("clamp_dt_level0", count, KOKKOS_LAMBDA(int ii) {
                int i = start + ii;
                if (dt_f(i) > dt_l0(i)) dt_f(i) = dt_l0(i);
            });
        }
#endif
        g_profiler.stop(StepProfiler::ADAPTIVE_DT);
#ifdef ENABLE_COLLISIONAL
        // Step 3: Subsystem timestep adjustments
        if (p.collisional) {
            g_subsystem_mgr.enforce_shared_timestep(p);
            g_subsystem_mgr.apply_perturbation_timestep(p, G, g_collisional.eta_pert);

            // Level-0 store: capture dt AFTER subsystem adjustments
            if (depth == 0) {
                auto dt_f = p.dt_adapt;
                auto dt_l0 = p.dt_level0;
                Kokkos::parallel_for("copy_dt_level0", count, KOKKOS_LAMBDA(int ii) {
                    int i = start + ii;
                    dt_l0(i) = dt_f(i);
                });
            }

            // Merger detection at every HHS level (BIFROST §3.14)
            if (g_collisional.c_code > 0.0) {
                auto mergers = detect_mergers(g_subsystem_mgr, p, G,
                                               g_collisional.c_code, pivot_dt);
                for (const auto& ev : mergers) {
                    execute_merger(ev, p, g_subsystem_mgr, g_energy_books,
                                   G, g_collisional.c_code);
                }
                if (!mergers.empty()) {
                    int n_removed = p.compact_remove_zero_mass();
                    count -= n_removed;
                    if (count <= 0) return;
                    g_subsystem_mgr.update_particle_flags(p);
                }
            }
        }
#endif
    }

    // Partition into slow (dt >= pivot) and fast (dt < pivot)
    g_profiler.start();
    int slow_count = partition_by_pivot(p, start, count, pivot_dt);
    g_profiler.stop(StepProfiler::PARTITION);
    int slow_start = start;
    int fast_start = start + slow_count;
    int fast_count = count - slow_count;

    // Base case: no fast particles or at minimum timestep
    if (fast_count == 0 || pivot_dt * 0.5 < dt_min) {
        fsi_step(p, start, count, G, soft2, pivot_dt);
        return;
    }

    // Base case: all particles are fast (no slow particles)
    // Partition was a no-op, so dt_adapt values are still valid.
    if (slow_count == 0) {
        double sub_pivot = pivot_dt * 0.5;
        hhs_fsi_step_dynamic_impl(p, start, count, sub_pivot,
                                  G, soft2, dt_min, eta, true, depth + 1);
        hhs_fsi_step_dynamic_impl(p, start, count, sub_pivot,
                                  G, soft2, dt_min, eta, false, depth + 1);
        return;
    }

    // HHS-FSI structure (Algorithm 4 / Eq. 40)
    // 1. Cross-Newton kick (dt/6)
    g_profiler.start();
    cross_acceleration(p, slow_start, slow_count, fast_start, fast_count, G, soft2);
    g_profiler.stop(StepProfiler::CROSS_FORCE);
    g_profiler.start();
    kick_velocities(p, slow_start, slow_count, pivot_dt * FSI_KICK);
    kick_velocities(p, fast_start, fast_count, pivot_dt * FSI_KICK);
    g_profiler.stop(StepProfiler::DRIFT_KICK);

    // 2. FSI half-step on slow
    fsi_step(p, slow_start, slow_count, G, soft2, pivot_dt * 0.5);

    // 3. Recurse on fast (first half) with pivot/2 — always compute fresh timesteps
    hhs_fsi_step_dynamic_impl(p, fast_start, fast_count, pivot_dt * 0.5, G, soft2, dt_min, eta, false, depth + 1);

    // 4. Cross-gradient kick (2dt/3)
    g_profiler.start();
    cross_acceleration(p, slow_start, slow_count, fast_start, fast_count, G, soft2);
    g_profiler.stop(StepProfiler::CROSS_FORCE);
    g_profiler.start();
    cross_gradient_correction(p, slow_start, slow_count, fast_start, fast_count, G, soft2, pivot_dt);
    g_profiler.stop(StepProfiler::CROSS_GRAD);
    g_profiler.start();
    kick_velocities(p, slow_start, slow_count, pivot_dt * FSI_MOD_KICK);
    kick_velocities(p, fast_start, fast_count, pivot_dt * FSI_MOD_KICK);
    g_profiler.stop(StepProfiler::DRIFT_KICK);

    // 5. Recurse on fast (second half)
    {
        double sub_pivot = pivot_dt * 0.5;
        hhs_fsi_step_dynamic_impl(p, fast_start, fast_count, sub_pivot, G, soft2, dt_min, eta, false, depth + 1);
    }

    // 6. FSI half-step on slow
    fsi_step(p, slow_start, slow_count, G, soft2, pivot_dt * 0.5);

    // 7. Cross-Newton kick (dt/6)
    g_profiler.start();
    cross_acceleration(p, slow_start, slow_count, fast_start, fast_count, G, soft2);
    g_profiler.stop(StepProfiler::CROSS_FORCE);
    g_profiler.start();
    kick_velocities(p, slow_start, slow_count, pivot_dt * FSI_KICK);
    kick_velocities(p, fast_start, fast_count, pivot_dt * FSI_KICK);
    g_profiler.stop(StepProfiler::DRIFT_KICK);
}

void hhs_fsi_step_dynamic(Particles& p, int start, int count, double pivot_dt,
                          double G, double soft2, double dt_min,
                          double eta) {
    hhs_fsi_step_dynamic_impl(p, start, count, pivot_dt, G, soft2, dt_min, eta,
                              false);
}

// =============================================================================
// Distributed (Multi-GPU) HHS Integration
// =============================================================================
// NEW DESIGN: Scatter-compute-gather paradigm (matching FROST paper)
//
// All ranks maintain replicated particle data. The HHS recursion runs
// IDENTICALLY on all ranks (partitioning, drifts, kicks). Only force
// computation is distributed:
//   1. Each rank computes forces for its slice of particles
//   2. Allgather accelerations after each force computation
//
// This achieves O(N) communication per step regardless of HHS depth.
// =============================================================================

// Kick velocities using pre-computed accelerations
void kick_velocities(Particles& p, int start, int count, double dt) {
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};

    Kokkos::parallel_for("kick", count, KOKKOS_LAMBDA(int ii) {
        int i = start + ii;
        pvx(i) += pax(i) * dt;
        pvy(i) += pay(i) * dt;
        pvz(i) += paz(i) * dt;
    });
}

// =============================================================================
// Distributed force computation: scatter-compute-gather with tiled GPU kernels
// Each rank computes forces for particles in its slice FROM ALL particles
// in the working set [start, start+count). Then allgather accelerations.
// Uses shared memory tiling per FROST paper.
// =============================================================================
static void self_acceleration_distributed(DistributedParticles& dp, int start, int count,
                                          double G, double soft2) {
    if (count <= 0) return;


    int slice_start, slice_count;
    dp.get_force_slice(start, count, slice_start, slice_count);
    if (slice_count == 0) {
        dp.allgather_accelerations_subset(start, count);
        return;
    }

    auto pos = dp.local.pos_packed;
    RawView pax{dp.local.ax.data()}, pay{dp.local.ay.data()}, paz{dp.local.az.data()};
#ifdef ENABLE_COLLISIONAL
    const int* sid_raw = dp.local.subsystem_id.data();
    bool coll = dp.local.collisional;
#endif

    const int j_start = start;
    const int j_end = start + count;

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t scratch_size = TILE_SIZE * 4 * sizeof(double); // x,y,z,m
    int num_teams = (slice_count + TILE_SIZE - 1) / TILE_SIZE;

    Kokkos::parallel_for("dist_self_acc_tiled",
        team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * TILE_SIZE + team.team_rank();
            int i = slice_start + ii;

            // AoS interleaved tile: [x,y,z,m] per j-particle
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 4);

            double xi = 0, yi = 0, zi = 0;
#ifdef ENABLE_COLLISIONAL
            int sid_i = -1;
#endif

            if (ii < slice_count) {
                double mi_pad;
                load4(&pos(i, 0), xi, yi, zi, mi_pad);
#ifdef ENABLE_COLLISIONAL
                if (coll) sid_i = sid_raw[i];
#endif
            }

            double axi = 0, ayi = 0, azi = 0;

            for (int tile_start = j_start; tile_start < j_end; tile_start += TILE_SIZE) {
                int tile_end = (tile_start + TILE_SIZE < j_end) ? tile_start + TILE_SIZE : j_end;
                int tile_count = tile_end - tile_start;

                int t = team.team_rank();
                if (t < tile_count) {
                    int idx = tile_start + t;
                    int base = t * 4;
                    load4(&pos(idx, 0), tile(base), tile(base+1), tile(base+2), tile(base+3));
                }
                team.team_barrier();

                if (ii < slice_count) {
                    for (int jt = 0; jt < tile_count; jt++) {
                        int j = tile_start + jt;
                        if (j == i) continue;
#ifdef ENABLE_COLLISIONAL
                        if (coll && sid_i >= 0 && sid_raw[j] == sid_i) continue;
#endif


                        int jb = jt * 4;
                        double dx = tile(jb    ) - xi;
                        double dy = tile(jb + 1) - yi;
                        double dz = tile(jb + 2) - zi;

                        double r2 = dx*dx + dy*dy + dz*dz + soft2;
                        double inv_r = fast_rsqrt(r2);
                        double inv_r3 = inv_r * inv_r * inv_r;
                        double mj_inv_r3 = G * tile(jb + 3) * inv_r3;

                        axi += mj_inv_r3 * dx;
                        ayi += mj_inv_r3 * dy;
                        azi += mj_inv_r3 * dz;
                    }
                }
                team.team_barrier();
            }

            if (ii < slice_count) {
                pax(i) = axi;
                pay(i) = ayi;
                paz(i) = azi;
            }
        });
    Kokkos::fence();

    if (use_nonblocking_mpi) {
        dp.iallgather_accelerations_subset(start, count, 0);
        dp.wait_and_unpack_accelerations(start, count, 0);
    } else {
        dp.allgather_accelerations_subset(start, count);
    }
}

// =============================================================================
// Distributed gradient correction: computes gradient correction for my slice,
// applies it to accelerations, then allgather corrected accelerations.
// =============================================================================
static void self_gradient_correction_distributed(DistributedParticles& dp, int start, int count,
                                                  double G, double soft2, double grad_dt) {
    if (count <= 0) return;

    int slice_start, slice_count;
    dp.get_force_slice(start, count, slice_start, slice_count);
    if (slice_count == 0) {
        dp.allgather_accelerations_subset(start, count);
        return;
    }

    auto pos = dp.local.pos_packed;
    RawView pax{dp.local.ax.data()}, pay{dp.local.ay.data()}, paz{dp.local.az.data()};
#ifdef ENABLE_COLLISIONAL
    const int* sid_raw = dp.local.subsystem_id.data();
    bool coll = dp.local.collisional;
    RawView pgmag{coll ? dp.local.grad_mag.data() : nullptr};
#endif

    const double coeff = G * grad_dt * grad_dt * GRAD_COEFF;
    const int j_start = start;
    const int j_end = start + count;

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t scratch_size = TILE_SIZE * 7 * sizeof(double);
    int num_teams = (slice_count + TILE_SIZE - 1) / TILE_SIZE;

    Kokkos::parallel_for("dist_grad_tiled",
        team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * TILE_SIZE + team.team_rank();
            int i = slice_start + ii;

            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 7);

            double xi = 0, yi = 0, zi = 0;
            double axi = 0, ayi = 0, azi = 0;
#ifdef ENABLE_COLLISIONAL
            int sid_i = -1;
#endif

            if (ii < slice_count) {
                double mi_pad;
                load4(&pos(i, 0), xi, yi, zi, mi_pad);
                axi = pax(i); ayi = pay(i); azi = paz(i);
#ifdef ENABLE_COLLISIONAL
                if (coll) sid_i = sid_raw[i];
#endif
            }

            double cx = 0, cy = 0, cz = 0;

            for (int tile_start = j_start; tile_start < j_end; tile_start += TILE_SIZE) {
                int tile_end = (tile_start + TILE_SIZE < j_end) ? tile_start + TILE_SIZE : j_end;
                int tile_count = tile_end - tile_start;

                int t = team.team_rank();
                if (t < tile_count) {
                    int idx = tile_start + t;
                    int base = t * 7;
                    load4(&pos(idx, 0), tile(base), tile(base+1), tile(base+2), tile(base+3));
                    tile(base + 4) = pax(idx);
                    tile(base + 5) = pay(idx);
                    tile(base + 6) = paz(idx);
                }
                team.team_barrier();

                if (ii < slice_count) {
                    for (int jt = 0; jt < tile_count; jt++) {
                        int j = tile_start + jt;
                        if (j == i) continue;
#ifdef ENABLE_COLLISIONAL
                        if (coll && sid_i >= 0 && sid_raw[j] == sid_i) continue;
#endif


                        int jb = jt * 7;
                        double dx = tile(jb    ) - xi;
                        double dy = tile(jb + 1) - yi;
                        double dz = tile(jb + 2) - zi;

                        double dax = tile(jb + 4) - axi;
                        double day = tile(jb + 5) - ayi;
                        double daz = tile(jb + 6) - azi;

                        double r2 = dx*dx + dy*dy + dz*dz + soft2;
                        double inv_r = fast_rsqrt(r2);
                        double inv_r2 = inv_r * inv_r;
                        double inv_r5 = inv_r2 * inv_r2 * inv_r;

                        double r_dot_a = dx*dax + dy*day + dz*daz;
                        double f = coeff * tile(jb + 3) * inv_r5;

                        cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                        cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                        cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
                    }
                }
                team.team_barrier();
            }

            if (ii < slice_count) {
                pax(i) = axi + cx;
                pay(i) = ayi + cy;
                paz(i) = azi + cz;
#ifdef ENABLE_COLLISIONAL
                if (coll) pgmag(i) = Kokkos::sqrt(cx*cx + cy*cy + cz*cz);
#endif
            }
        });
    Kokkos::fence();

    // Allgather corrected accelerations
    if (use_nonblocking_mpi) {
        dp.iallgather_accelerations_subset(start, count, 0);
        dp.wait_and_unpack_accelerations(start, count, 0);
    } else {
        dp.allgather_accelerations_subset(start, count);
    }
}

// =============================================================================
// RING TOPOLOGY (j-partitioned) force functions
// Each rank computes partial forces for ALL i-particles from its j-chunk,
// then MPI_Allreduce(SUM) combines partial contributions across all ranks.
// Replaces allgatherv-based i-partitioned approach.
// =============================================================================
#ifdef MPI_TOPOLOGY_RING

// Self-force: ALL i-particles from this rank's j-chunk, then allreduce
static void self_acceleration_ring(DistributedParticles& dp, int start, int count,
                                    double G, double soft2) {
    if (count <= 0) return;

    // j-chunk for this rank (reuses same decomposition as i-slicing)
    int j_start, j_count;
    dp.get_force_slice(start, count, j_start, j_count);
    const int j_end = j_start + j_count;

    auto pos = dp.local.pos_packed;
    RawView pax{dp.local.ax.data()}, pay{dp.local.ay.data()}, paz{dp.local.az.data()};
#ifdef ENABLE_COLLISIONAL
    const int* sid_raw = dp.local.subsystem_id.data();
    bool coll = dp.local.collisional;
#endif

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t scratch_size = TILE_SIZE * 4 * sizeof(double);
    int num_teams = (count + TILE_SIZE - 1) / TILE_SIZE;
    const int ist = start;
    const int icnt = count;
    const int js = j_start;
    const int je = j_end;

    Kokkos::parallel_for("ring_self_acc",
        team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * TILE_SIZE + team.team_rank();
            int i = ist + ii;

            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 4);

            double xi = 0, yi = 0, zi = 0;
#ifdef ENABLE_COLLISIONAL
            int sid_i = -1;
#endif

            if (ii < icnt) {
                double mi_pad;
                load4(&pos(i, 0), xi, yi, zi, mi_pad);
#ifdef ENABLE_COLLISIONAL
                if (coll) sid_i = sid_raw[i];
#endif
            }

            double axi = 0, ayi = 0, azi = 0;

            for (int tile_st = js; tile_st < je; tile_st += TILE_SIZE) {
                int tile_end = (tile_st + TILE_SIZE < je) ? tile_st + TILE_SIZE : je;
                int tile_cnt = tile_end - tile_st;

                int t = team.team_rank();
                if (t < tile_cnt) {
                    int idx = tile_st + t;
                    int base = t * 4;
                    load4(&pos(idx, 0), tile(base), tile(base+1), tile(base+2), tile(base+3));
                }
                team.team_barrier();

                if (ii < icnt) {
                    for (int jt = 0; jt < tile_cnt; jt++) {
                        int j = tile_st + jt;
                        if (j == i) continue;
#ifdef ENABLE_COLLISIONAL
                        if (coll && sid_i >= 0 && sid_raw[j] == sid_i) continue;
#endif


                        int jb = jt * 4;
                        double dx = tile(jb) - xi;
                        double dy = tile(jb+1) - yi;
                        double dz = tile(jb+2) - zi;

                        double r2 = dx*dx + dy*dy + dz*dz + soft2;
                        double inv_r = fast_rsqrt(r2);
                        double inv_r3 = inv_r * inv_r * inv_r;
                        double mj_inv_r3 = G * tile(jb+3) * inv_r3;

                        axi += mj_inv_r3 * dx;
                        ayi += mj_inv_r3 * dy;
                        azi += mj_inv_r3 * dz;
                    }
                }
                team.team_barrier();
            }

            if (ii < icnt) {
                pax(i) = axi;
                pay(i) = ayi;
                paz(i) = azi;
            }
        });
    Kokkos::fence();

    // Allreduce: sum partial forces across all ranks → complete forces everywhere
    dp.allreduce_accelerations_subset(start, count);
}

// Self gradient correction: j-partitioned gradient after forces are allreduced.
// Saves complete forces to scratch, computes partial gradient, allreduces, adds back.
static void self_gradient_correction_ring(DistributedParticles& dp, int start, int count,
                                           double G, double soft2, double grad_dt) {
    if (count <= 0) return;

    int j_start, j_count;
    dp.get_force_slice(start, count, j_start, j_count);
    const int j_end = j_start + j_count;

    auto pos = dp.local.pos_packed;
    RawView pax{dp.local.ax.data()}, pay{dp.local.ay.data()}, paz{dp.local.az.data()};
#ifdef ENABLE_COLLISIONAL
    const int* sid_raw = dp.local.subsystem_id.data();
    bool coll = dp.local.collisional;
    RawView pgmag{coll ? dp.local.grad_mag.data() : nullptr};
#endif

    // Save complete forces to scratch (needed after gradient allreduce)
    RawView sax{dp.local.scratch1.data()}, say{dp.local.scratch2.data()}, saz{dp.local.scratch3.data()};
    Kokkos::parallel_for("ring_save_forces", count, KOKKOS_LAMBDA(int ii) {
        int i = start + ii;
        sax(i) = pax(i);
        say(i) = pay(i);
        saz(i) = paz(i);
    });

    const double coeff = G * grad_dt * grad_dt * GRAD_COEFF;

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t scratch_size = TILE_SIZE * 7 * sizeof(double);
    int num_teams = (count + TILE_SIZE - 1) / TILE_SIZE;
    const int ist = start;
    const int icnt = count;
    const int js = j_start;
    const int je = j_end;

    // Gradient kernel: compute partial correction from j-chunk, write to pax
    Kokkos::parallel_for("ring_self_grad",
        team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * TILE_SIZE + team.team_rank();
            int i = ist + ii;

            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 7);

            double xi = 0, yi = 0, zi = 0;
            double axi = 0, ayi = 0, azi = 0;
#ifdef ENABLE_COLLISIONAL
            int sid_i = -1;
#endif

            if (ii < icnt) {
                double mi_pad;
                load4(&pos(i, 0), xi, yi, zi, mi_pad);
                // Read complete forces from scratch
                axi = sax(i); ayi = say(i); azi = saz(i);
#ifdef ENABLE_COLLISIONAL
                if (coll) sid_i = sid_raw[i];
#endif
            }

            double cx = 0, cy = 0, cz = 0;

            for (int tile_st = js; tile_st < je; tile_st += TILE_SIZE) {
                int tile_end = (tile_st + TILE_SIZE < je) ? tile_st + TILE_SIZE : je;
                int tile_cnt = tile_end - tile_st;

                int t = team.team_rank();
                if (t < tile_cnt) {
                    int idx = tile_st + t;
                    int tb = t * 7;
                    load4(&pos(idx, 0), tile(tb), tile(tb+1), tile(tb+2), tile(tb+3));
                    // Read complete forces from scratch for j-particles
                    tile(tb+4) = sax(idx);
                    tile(tb+5) = say(idx);
                    tile(tb+6) = saz(idx);
                }
                team.team_barrier();

                if (ii < icnt) {
                    for (int jt = 0; jt < tile_cnt; jt++) {
                        int j = tile_st + jt;
                        if (j == i) continue;
#ifdef ENABLE_COLLISIONAL
                        if (coll && sid_i >= 0 && sid_raw[j] == sid_i) continue;
#endif


                        int jb = jt * 7;
                        double dx = tile(jb) - xi;
                        double dy = tile(jb+1) - yi;
                        double dz = tile(jb+2) - zi;

                        double dax = tile(jb+4) - axi;
                        double day = tile(jb+5) - ayi;
                        double daz = tile(jb+6) - azi;

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

            // Write ONLY the partial gradient correction (not force + gradient)
            if (ii < icnt) {
                pax(i) = cx;
                pay(i) = cy;
                paz(i) = cz;
            }
        });
    Kokkos::fence();

    // Allreduce: sum partial gradient corrections
    dp.allreduce_accelerations_subset(start, count);

    // Add back saved forces: pax = force + gradient_correction
    // At this point pax = full gradient correction (after allreduce), sax = saved forces
    Kokkos::parallel_for("ring_add_forces", count, KOKKOS_LAMBDA(int ii) {
        int i = start + ii;
#ifdef ENABLE_COLLISIONAL
        if (coll) pgmag(i) = Kokkos::sqrt(pax(i)*pax(i) + pay(i)*pay(i) + paz(i)*paz(i));
#endif
        pax(i) += sax(i);
        pay(i) += say(i);
        paz(i) += saz(i);
    });
}

// Cross-force: ALL slow from fast-chunk + ALL fast from slow-chunk, then allreduce both
static void cross_acceleration_ring(DistributedParticles& dp,
                                     int slow_start, int slow_count,
                                     int fast_start, int fast_count,
                                     double G, double soft2) {
    if (slow_count <= 0 || fast_count <= 0) return;

    auto pos = dp.local.pos_packed;
    RawView pax{dp.local.ax.data()}, pay{dp.local.ay.data()}, paz{dp.local.az.data()};
#ifdef ENABLE_COLLISIONAL
    const int* sid_raw = dp.local.subsystem_id.data();
    bool coll = dp.local.collisional;
#endif

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t scratch_size = TILE_SIZE * 4 * sizeof(double);

    // j-chunk of fast particles for slow-from-fast kernel
    int j_fast_start, j_fast_count;
    dp.get_force_slice(fast_start, fast_count, j_fast_start, j_fast_count);

    // Slow particles: partial force from fast j-chunk
    {
        int num_teams = (slow_count + TILE_SIZE - 1) / TILE_SIZE;
        const int ist = slow_start;
        const int icnt = slow_count;
        const int js = j_fast_start;
        const int je = j_fast_start + j_fast_count;

        Kokkos::parallel_for("ring_cross_slow_from_fast",
            team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_LAMBDA(const member_type& team) {
                int ii = team.league_rank() * TILE_SIZE + team.team_rank();
                int i = ist + ii;

                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile(team.team_scratch(0), TILE_SIZE * 4);

                double xi = 0, yi = 0, zi = 0;
#ifdef ENABLE_COLLISIONAL
                int sid_i = -1;
#endif

                if (ii < icnt) {
                    double mi_pad;
                    load4(&pos(i, 0), xi, yi, zi, mi_pad);
#ifdef ENABLE_COLLISIONAL
                    if (coll) sid_i = sid_raw[i];
#endif
                }

                double axi = 0, ayi = 0, azi = 0;

                for (int tile_st = js; tile_st < je; tile_st += TILE_SIZE) {
                    int tile_end = (tile_st + TILE_SIZE < je) ? tile_st + TILE_SIZE : je;
                    int tile_cnt = tile_end - tile_st;

                    int t = team.team_rank();
                    if (t < tile_cnt) {
                        int idx = tile_st + t;
                        int base = t * 4;
                        load4(&pos(idx, 0), tile(base), tile(base+1), tile(base+2), tile(base+3));
                    }
                    team.team_barrier();

                    if (ii < icnt) {
                        for (int jt = 0; jt < tile_cnt; jt++) {
                            int j_cross = tile_st + jt;
#ifdef ENABLE_COLLISIONAL
                            if (coll && sid_i >= 0 && sid_raw[j_cross] == sid_i) continue;
#endif


                            int jb = jt * 4;
                            double dx = tile(jb) - xi;
                            double dy = tile(jb+1) - yi;
                            double dz = tile(jb+2) - zi;

                            double r2 = dx*dx + dy*dy + dz*dz + soft2;
                            double inv_r = fast_rsqrt(r2);
                            double inv_r3 = inv_r * inv_r * inv_r;
                            double mj_inv_r3 = G * tile(jb+3) * inv_r3;

                            axi += mj_inv_r3 * dx;
                            ayi += mj_inv_r3 * dy;
                            azi += mj_inv_r3 * dz;
                        }
                    }
                    team.team_barrier();
                }

                if (ii < icnt) {
                    pax(i) = axi;
                    pay(i) = ayi;
                    paz(i) = azi;
                }
            });
    }

    // j-chunk of slow particles for fast-from-slow kernel
    int j_slow_start, j_slow_count;
    dp.get_force_slice(slow_start, slow_count, j_slow_start, j_slow_count);

    // Fast particles: partial force from slow j-chunk
    {
        int num_teams = (fast_count + TILE_SIZE - 1) / TILE_SIZE;
        const int ist = fast_start;
        const int icnt = fast_count;
        const int js = j_slow_start;
        const int je = j_slow_start + j_slow_count;

        Kokkos::parallel_for("ring_cross_fast_from_slow",
            team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_LAMBDA(const member_type& team) {
                int ii = team.league_rank() * TILE_SIZE + team.team_rank();
                int i = ist + ii;

                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile(team.team_scratch(0), TILE_SIZE * 4);

                double xi = 0, yi = 0, zi = 0;
#ifdef ENABLE_COLLISIONAL
                int sid_i = -1;
#endif

                if (ii < icnt) {
                    double mi_pad;
                    load4(&pos(i, 0), xi, yi, zi, mi_pad);
#ifdef ENABLE_COLLISIONAL
                    if (coll) sid_i = sid_raw[i];
#endif
                }

                double axi = 0, ayi = 0, azi = 0;

                for (int tile_st = js; tile_st < je; tile_st += TILE_SIZE) {
                    int tile_end = (tile_st + TILE_SIZE < je) ? tile_st + TILE_SIZE : je;
                    int tile_cnt = tile_end - tile_st;

                    int t = team.team_rank();
                    if (t < tile_cnt) {
                        int idx = tile_st + t;
                        int tb = t * 4;
                        load4(&pos(idx, 0), tile(tb), tile(tb+1), tile(tb+2), tile(tb+3));
                    }
                    team.team_barrier();

                    if (ii < icnt) {
                        for (int jt = 0; jt < tile_cnt; jt++) {
                            int j_cross = tile_st + jt;
#ifdef ENABLE_COLLISIONAL
                            if (coll && sid_i >= 0 && sid_raw[j_cross] == sid_i) continue;
#endif

                            int jb = jt * 4;
                            double dx = tile(jb) - xi;
                            double dy = tile(jb+1) - yi;
                            double dz = tile(jb+2) - zi;

                            double r2 = dx*dx + dy*dy + dz*dz + soft2;
                            double inv_r = fast_rsqrt(r2);
                            double inv_r3 = inv_r * inv_r * inv_r;
                            double mj_inv_r3 = G * tile(jb+3) * inv_r3;

                            axi += mj_inv_r3 * dx;
                            ayi += mj_inv_r3 * dy;
                            azi += mj_inv_r3 * dz;
                        }
                    }
                    team.team_barrier();
                }

                if (ii < icnt) {
                    pax(i) = axi;
                    pay(i) = ayi;
                    paz(i) = azi;
                }
            });
    }
    Kokkos::fence();

    // Allreduce both slow and fast forces
    dp.allreduce_accelerations_subset(slow_start, slow_count);
    dp.allreduce_accelerations_subset(fast_start, fast_count);
}

// Cross-force + gradient correction (ring/j-partitioned)
// Flow: cross-force → allreduce → save forces → gradient → allreduce → add back
static void cross_acceleration_and_gradient_ring(DistributedParticles& dp,
                                                  int slow_start, int slow_count,
                                                  int fast_start, int fast_count,
                                                  double G, double soft2, double dt) {
    // Step 1: Compute cross-forces (j-partitioned) and allreduce
    cross_acceleration_ring(dp, slow_start, slow_count, fast_start, fast_count, G, soft2);

    // Now all ranks have complete cross-forces for slow and fast particles.
    // Save complete forces to scratch for gradient computation.
    auto pos = dp.local.pos_packed;
    RawView pax{dp.local.ax.data()}, pay{dp.local.ay.data()}, paz{dp.local.az.data()};
    RawView sax{dp.local.scratch1.data()}, say{dp.local.scratch2.data()}, saz{dp.local.scratch3.data()};
#ifdef ENABLE_COLLISIONAL
    const int* sid_raw = dp.local.subsystem_id.data();
    bool coll = dp.local.collisional;
#endif

    int total_start = slow_start;
    int total_count = slow_count + fast_count;
    Kokkos::parallel_for("ring_save_cross_forces", total_count, KOKKOS_LAMBDA(int ii) {
        int i = total_start + ii;
        sax(i) = pax(i);
        say(i) = pay(i);
        saz(i) = paz(i);
    });

    const double coeff = G * dt * dt * GRAD_COEFF;

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t scratch_size = TILE_SIZE * 7 * sizeof(double);

    // Step 2: Gradient for slow particles from fast j-chunk
    {
        int j_fast_start, j_fast_count;
        dp.get_force_slice(fast_start, fast_count, j_fast_start, j_fast_count);

        int num_teams = (slow_count + TILE_SIZE - 1) / TILE_SIZE;
        const int ist = slow_start;
        const int icnt = slow_count;
        const int js = j_fast_start;
        const int je = j_fast_start + j_fast_count;

        Kokkos::parallel_for("ring_cross_grad_slow",
            team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_LAMBDA(const member_type& team) {
                int ii = team.league_rank() * TILE_SIZE + team.team_rank();
                int i = ist + ii;

                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile(team.team_scratch(0), TILE_SIZE * 7);

                double xi = 0, yi = 0, zi = 0;
                double axi = 0, ayi = 0, azi = 0;
#ifdef ENABLE_COLLISIONAL
                int sid_i = -1;
#endif

                if (ii < icnt) {
                    double mi_pad;
                    load4(&pos(i, 0), xi, yi, zi, mi_pad);
                    axi = sax(i); ayi = say(i); azi = saz(i);
#ifdef ENABLE_COLLISIONAL
                    if (coll) sid_i = sid_raw[i];
#endif
                }

                double cx = 0, cy = 0, cz = 0;

                for (int tile_st = js; tile_st < je; tile_st += TILE_SIZE) {
                    int tile_end = (tile_st + TILE_SIZE < je) ? tile_st + TILE_SIZE : je;
                    int tile_cnt = tile_end - tile_st;

                    int t = team.team_rank();
                    if (t < tile_cnt) {
                        int idx = tile_st + t;
                        int tb = t * 7;
                        load4(&pos(idx, 0), tile(tb), tile(tb+1), tile(tb+2), tile(tb+3));
                        tile(tb+4) = sax(idx);
                        tile(tb+5) = say(idx);
                        tile(tb+6) = saz(idx);
                    }
                    team.team_barrier();

                    if (ii < icnt) {
                        for (int jt = 0; jt < tile_cnt; jt++) {
                            int j_cross = tile_st + jt;
#ifdef ENABLE_COLLISIONAL
                            if (coll && sid_i >= 0 && sid_raw[j_cross] == sid_i) continue;
#endif

                            int jb = jt * 7;
                            double dx = tile(jb) - xi;
                            double dy = tile(jb+1) - yi;
                            double dz = tile(jb+2) - zi;

                            double dax = tile(jb+4) - axi;
                            double day = tile(jb+5) - ayi;
                            double daz = tile(jb+6) - azi;

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

                if (ii < icnt) {
                    pax(i) = cx;
                    pay(i) = cy;
                    paz(i) = cz;
                }
            });
    }

    // Step 3: Gradient for fast particles from slow j-chunk
    {
        int j_slow_start, j_slow_count;
        dp.get_force_slice(slow_start, slow_count, j_slow_start, j_slow_count);

        int num_teams = (fast_count + TILE_SIZE - 1) / TILE_SIZE;
        const int ist = fast_start;
        const int icnt = fast_count;
        const int js = j_slow_start;
        const int je = j_slow_start + j_slow_count;

        Kokkos::parallel_for("ring_cross_grad_fast",
            team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_LAMBDA(const member_type& team) {
                int ii = team.league_rank() * TILE_SIZE + team.team_rank();
                int i = ist + ii;

                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile(team.team_scratch(0), TILE_SIZE * 7);

                double xi = 0, yi = 0, zi = 0;
                double axi = 0, ayi = 0, azi = 0;
#ifdef ENABLE_COLLISIONAL
                int sid_i = -1;
#endif

                if (ii < icnt) {
                    double mi_pad;
                    load4(&pos(i, 0), xi, yi, zi, mi_pad);
                    axi = sax(i); ayi = say(i); azi = saz(i);
#ifdef ENABLE_COLLISIONAL
                    if (coll) sid_i = sid_raw[i];
#endif
                }

                double cx = 0, cy = 0, cz = 0;

                for (int tile_st = js; tile_st < je; tile_st += TILE_SIZE) {
                    int tile_end = (tile_st + TILE_SIZE < je) ? tile_st + TILE_SIZE : je;
                    int tile_cnt = tile_end - tile_st;

                    int t = team.team_rank();
                    if (t < tile_cnt) {
                        int idx = tile_st + t;
                        int tb = t * 7;
                        load4(&pos(idx, 0), tile(tb), tile(tb+1), tile(tb+2), tile(tb+3));
                        tile(tb+4) = sax(idx);
                        tile(tb+5) = say(idx);
                        tile(tb+6) = saz(idx);
                    }
                    team.team_barrier();

                    if (ii < icnt) {
                        for (int jt = 0; jt < tile_cnt; jt++) {
                            int j_cross = tile_st + jt;
#ifdef ENABLE_COLLISIONAL
                            if (coll && sid_i >= 0 && sid_raw[j_cross] == sid_i) continue;
#endif


                            int jb = jt * 7;
                            double dx = tile(jb) - xi;
                            double dy = tile(jb+1) - yi;
                            double dz = tile(jb+2) - zi;

                            double dax = tile(jb+4) - axi;
                            double day = tile(jb+5) - ayi;
                            double daz = tile(jb+6) - azi;

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

                if (ii < icnt) {
                    pax(i) = cx;
                    pay(i) = cy;
                    paz(i) = cz;
                }
            });
    }
    Kokkos::fence();

    // Allreduce gradient corrections for both slow and fast
    dp.allreduce_accelerations_subset(slow_start, slow_count);
    dp.allreduce_accelerations_subset(fast_start, fast_count);

    // Add back saved forces: ax = force + gradient_correction
    Kokkos::parallel_for("ring_add_cross_forces", total_count, KOKKOS_LAMBDA(int ii) {
        int i = total_start + ii;
        pax(i) += sax(i);
        pay(i) += say(i);
        paz(i) += saz(i);
    });
}

// Fused force + symmetrized timestep (ring): ALL i from this rank's j-chunk
// Then allreduce(SUM) for forces, allreduce(MIN) for timesteps
static void self_acceleration_and_timestep_ring(DistributedParticles& dp, int start, int count,
                                                  double G, double soft2, double eta) {
    if (count <= 1) return;

    int j_start, j_count;
    dp.get_force_slice(start, count, j_start, j_count);
    const int j_end = j_start + j_count;

    auto pos = dp.local.pos_packed;
    auto vel = dp.local.vel_packed;
    RawView pax{dp.local.ax.data()}, pay{dp.local.ay.data()}, paz{dp.local.az.data()};
    RawView dt_f{dp.local.dt_adapt.data()};
#ifdef ENABLE_COLLISIONAL
    const int* sid_raw = dp.local.subsystem_id.data();
    bool coll = dp.local.collisional;
#endif

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

    size_t scratch_size = TILE_SIZE * 7 * sizeof(double);
    int num_teams = (count + TILE_SIZE - 1) / TILE_SIZE;
    const int ist = start;
    const int icnt = count;
    const int js = j_start;
    const int je = j_end;

    Kokkos::parallel_for("ring_self_acc_dt_sym",
        team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * TILE_SIZE + team.team_rank();
            int i = ist + ii;

            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 7);

            double xi = 0, yi = 0, zi = 0, mi = 0;
            double vxi = 0, vyi = 0, vzi = 0;
#ifdef ENABLE_COLLISIONAL
            int sid_i = -1;
#endif

            if (ii < icnt) {
                double vpad;
                load4(&pos(i, 0), xi, yi, zi, mi);
                load4(&vel(i, 0), vxi, vyi, vzi, vpad);
#ifdef ENABLE_COLLISIONAL
                if (coll) sid_i = sid_raw[i];
#endif
            }

            double axi = 0, ayi = 0, azi = 0;
            double t_min_i = 1e30;

            for (int tile_st = js; tile_st < je; tile_st += TILE_SIZE) {
                int tile_end = (tile_st + TILE_SIZE < je) ? tile_st + TILE_SIZE : je;
                int tile_cnt = tile_end - tile_st;

                int t = team.team_rank();
                if (t < tile_cnt) {
                    int idx = tile_st + t;
                    int tb = t * 7;
                    double vpad;
                    load4(&pos(idx, 0), tile(tb), tile(tb+1), tile(tb+2), tile(tb+3));
                    load4(&vel(idx, 0), tile(tb+4), tile(tb+5), tile(tb+6), vpad);
                }
                team.team_barrier();

                if (ii < icnt) {
                    for (int jt = 0; jt < tile_cnt; jt++) {
                        int j = tile_st + jt;
                        if (j == i) continue;
#ifdef ENABLE_COLLISIONAL
                        if (coll && sid_i >= 0 && sid_raw[j] == sid_i) continue;
#endif


                        int jb = jt * 7;
                        double dx = tile(jb    ) - xi;
                        double dy = tile(jb + 1) - yi;
                        double dz = tile(jb + 2) - zi;
                        double mj = tile(jb + 3);
                        double r2 = dx*dx + dy*dy + dz*dz;
                        double r2s = r2 + soft2;
                        double inv_r = fast_rsqrt(r2s);
                        double inv_r3 = inv_r * inv_r * inv_r;

                        // Force accumulation
                        double mj_inv_r3 = G * mj * inv_r3;
                        axi += mj_inv_r3 * dx;
                        ayi += mj_inv_r3 * dy;
                        azi += mj_inv_r3 * dz;

                        // Symmetrized timestep — branchless, precomputed reciprocals
                        double dvx = tile(jb + 4) - vxi;
                        double dvy = tile(jb + 5) - vyi;
                        double dvz = tile(jb + 6) - vzi;
                        double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
                        double mass_sum = mi + mj;
                        double inv_Gm = 1.0 / (G * mass_sum + EPS);
                        double inv_vsq = 1.0 / (v_sq + EPS);

                        double r3 = r2s * r2s * inv_r;
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
                    }
                }
                team.team_barrier();
            }

            if (ii < icnt) {
                pax(i) = axi;
                pay(i) = ayi;
                paz(i) = azi;
                dt_f(i) = t_min_i;
            }
        });
    Kokkos::fence();

    // Allreduce: sum partial forces, min partial timesteps
    dp.allreduce_accelerations_subset(start, count);
    dp.allreduce_timestep_subset(start, count);
}

// Timestep: ALL i-particles from this rank's j-chunk, then allreduce(MIN)
static void compute_timestep_ring(DistributedParticles& dp, int start, int count,
                                   double G, double soft2, double eta,
                                   bool symmetrize = false) {
    if (count <= 1) return;

    int j_start, j_count;
    dp.get_force_slice(start, count, j_start, j_count);
    const int j_end = j_start + j_count;

    auto pos = dp.local.pos_packed;
    auto vel = dp.local.vel_packed;
    RawView dt_f{dp.local.dt_adapt.data()};
    RawView pax_dt{dp.local.ax.data()}, pay_dt{dp.local.ay.data()}, paz_dt{dp.local.az.data()};
#ifdef ENABLE_COLLISIONAL
    const int* sid_raw = dp.local.subsystem_id.data();
    bool coll = dp.local.collisional;
    double eta_jerk_val = coll ? g_collisional.eta_jerk : 0.0;
#endif

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

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t scratch_size = TILE_SIZE * 7 * sizeof(double);
    int num_teams = (count + TILE_SIZE - 1) / TILE_SIZE;
    const int ist = start;
    const int icnt = count;
    const int js = j_start;
    const int je = j_end;

    Kokkos::parallel_for("ring_dt",
        team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * TILE_SIZE + team.team_rank();
            int i = ist + ii;

            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 7);

            double xi = 0, yi = 0, zi = 0, mi = 0;
            double vxi = 0, vyi = 0, vzi = 0;
#ifdef ENABLE_COLLISIONAL
            int sid_i = -1;
#endif

            if (ii < icnt) {
                double vpad;
                load4(&pos(i, 0), xi, yi, zi, mi);
                load4(&vel(i, 0), vxi, vyi, vzi, vpad);
#ifdef ENABLE_COLLISIONAL
                if (coll) sid_i = sid_raw[i];
#endif
            }

            double t_min_sq = 1e60;   // non-symmetrized path
            double t_min_i = 1e30;    // symmetrized path
#ifdef ENABLE_COLLISIONAL
            double jerk_x = 0, jerk_y = 0, jerk_z = 0;
            double acc_xi = 0, acc_yi = 0, acc_zi = 0;  // fresh subset acceleration for jerk criterion
#endif

            for (int tile_st = js; tile_st < je; tile_st += TILE_SIZE) {
                int tile_end = (tile_st + TILE_SIZE < je) ? tile_st + TILE_SIZE : je;
                int tile_cnt = tile_end - tile_st;

                int t = team.team_rank();
                if (t < tile_cnt) {
                    int idx = tile_st + t;
                    int tb = t * 7;
                    double vpad;
                    load4(&pos(idx, 0), tile(tb), tile(tb+1), tile(tb+2), tile(tb+3));
                    load4(&vel(idx, 0), tile(tb+4), tile(tb+5), tile(tb+6), vpad);
                }
                team.team_barrier();

                if (ii < icnt) {
                    for (int jt = 0; jt < tile_cnt; jt++) {
                        int j = tile_st + jt;
                        if (j == i) continue;
#ifdef ENABLE_COLLISIONAL
                        if (coll && sid_i >= 0 && sid_raw[j] == sid_i) continue;
#endif


                        int jb = jt * 7;
                        double dx = tile(jb) - xi;
                        double dy = tile(jb+1) - yi;
                        double dz = tile(jb+2) - zi;
                        double r2 = dx*dx + dy*dy + dz*dz;

                        double dvx = tile(jb+4) - vxi;
                        double dvy = tile(jb+5) - vyi;
                        double dvz = tile(jb+6) - vzi;
                        double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;

                        double r2s = r2 + soft2;
                        double inv_r = fast_rsqrt(r2s);
                        double r3 = r2s * r2s * inv_r;
                        double mj = tile(jb+3);
                        double mass_sum = mi + mj;

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

#ifdef ENABLE_COLLISIONAL
                        // Jerk + acceleration accumulation (collisional only)
                        if (coll) {
                            double inv_r3 = inv_r * inv_r * inv_r;
                            double rdotv = dx*dvx + dy*dvy + dz*dvz;
                            double fac1 = G * mj * inv_r3;
                            double fac2 = 3.0 * rdotv * inv_r * inv_r;
                            jerk_x += fac1 * (dvx - fac2 * dx);
                            jerk_y += fac1 * (dvy - fac2 * dy);
                            jerk_z += fac1 * (dvz - fac2 * dz);
                            // Fresh subset acceleration for jerk criterion (BIFROST §3.4.5)
                            acc_xi += fac1 * dx;
                            acc_yi += fac1 * dy;
                            acc_zi += fac1 * dz;
                        }
#endif

                    }
                }
                team.team_barrier();
            }

            if (ii < icnt) {
                dt_f(i) = symmetrize ? t_min_i : Kokkos::sqrt(t_min_sq);
#ifdef ENABLE_COLLISIONAL
                // Jerk timestep: dt_jerk = eta_j * |a| / |j| (BIFROST §3.4.5)
                // Uses freshly accumulated subset acceleration, not stale p.ax/ay/az
                if (coll && eta_jerk_val > 0.0) {
                    double a_mag = Kokkos::sqrt(acc_xi*acc_xi + acc_yi*acc_yi + acc_zi*acc_zi);
                    double j_mag = Kokkos::sqrt(jerk_x*jerk_x + jerk_y*jerk_y + jerk_z*jerk_z);
                    if (j_mag > EPS && a_mag > EPS) {
                        double dt_jerk = eta_jerk_val * a_mag / j_mag;
                        dt_f(i) = Kokkos::fmin(dt_f(i), dt_jerk);
                    }
                }
#endif
            }
        });
    Kokkos::fence();

    // Allreduce(MIN): each rank's partial min → global min
    dp.allreduce_timestep_subset(start, count);
}

// Ring FSI step: force → kick → drift → force → gradient → kick → drift → force → kick
static void fsi_step_ring(DistributedParticles& dp, int start, int count,
                           double G, double soft2, double dt,
                           bool forces_valid = false) {
    if (count <= 0) return;

    long long self_pairs = (long long)count * count;

    dp.local.pack_pos_range(start, count);

    // U kick (dt/6): force + allreduce + kick, or just kick if forces already valid
    if (forces_valid) {
        g_profiler.start();
        kick_velocities(dp.local, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else {
        g_profiler.start();
        self_acceleration_ring(dp, start, count, G, soft2);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        g_profiler.start();
        kick_velocities(dp.local, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // T drift (dt/2) — field particles only if subsystems active
    g_profiler.start();
#ifdef ENABLE_COLLISIONAL
    if (dp.local.collisional && !g_subsystem_mgr.active.empty()) {
        drift_field_only(dp.local, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
        g_profiler.start();
        advance_subsystems_in_range(dp.local, start, count, G, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::SUBSYSTEM_INTEG);
    } else
#endif
    {
        drift(dp.local, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    g_profiler.start();
    dp.local.pack_pos_range(start, count);
    g_profiler.stop(StepProfiler::PACK_DATA);

    // U-tilde kick (2dt/3) with gradient correction
    g_profiler.start();
    self_acceleration_ring(dp, start, count, G, soft2);
    g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
    g_profiler.start();
    self_gradient_correction_ring(dp, start, count, G, soft2, dt);
    g_profiler.stop(StepProfiler::SELF_GRAD, self_pairs);
    g_profiler.start();
    kick_velocities(dp.local, start, count, dt * FSI_MOD_KICK);
    g_profiler.stop(StepProfiler::DRIFT_KICK);

    // T drift (dt/2) — field particles only if subsystems active
    g_profiler.start();
#ifdef ENABLE_COLLISIONAL
    if (dp.local.collisional && !g_subsystem_mgr.active.empty()) {
        drift_field_only(dp.local, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
        g_profiler.start();
        advance_subsystems_in_range(dp.local, start, count, G, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::SUBSYSTEM_INTEG);
    } else
#endif
    {
        drift(dp.local, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    g_profiler.start();
    dp.local.pack_pos_range(start, count);
    g_profiler.stop(StepProfiler::PACK_DATA);

    // U kick (dt/6): force + allreduce + kick
    g_profiler.start();
    self_acceleration_ring(dp, start, count, G, soft2);
    g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
    g_profiler.start();
    kick_velocities(dp.local, start, count, dt * FSI_KICK);
    g_profiler.stop(StepProfiler::DRIFT_KICK);
}

#endif // MPI_TOPOLOGY_RING

// =============================================================================
// Tree MPI distribution helpers: divide tree groups across MPI ranks
// Each rank computes forces for its subset of groups, then allreduce(SUM)
// =============================================================================
static void get_tree_group_range(const Octree& tree, int& g_offset, int& g_count) {
    int rank = MPIComm::rank();
    int size = MPIComm::size();
    int total_groups = tree.n_groups;
    g_offset = rank * total_groups / size;
    int g_end = (rank + 1) * total_groups / size;
    g_count = g_end - g_offset;
}

static void zero_accelerations(Particles& p, int start, int count) {
    auto ax = p.ax; auto ay = p.ay; auto az = p.az;
    Kokkos::parallel_for("zero_acc", count, KOKKOS_LAMBDA(int ii) {
        int i = start + ii;
        ax(i) = 0.0; ay(i) = 0.0; az(i) = 0.0;
    });
}

static void init_timesteps_max(Particles& p, int start, int count) {
    auto dt_f = p.dt_adapt;
    Kokkos::parallel_for("init_dt_max", count, KOKKOS_LAMBDA(int ii) {
        dt_f(start + ii) = 1e30;
    });
}

// =============================================================================
// Distributed FSI step using scatter-compute-gather
// All ranks execute identical code; only force computation is distributed
// Uses fused force+kick kernels to reduce kernel launches and memory traffic
//
// tree_prebuilt: if true, g_octree is already built for [start, count) —
//   skip first tree build (reuse from timestep computation or previous call)
// global_tree_subset: -1 = build own tree (default), 0 or 1 = use global tree
//   with subset monopoles (0=slow, 1=fast). No MPI allreduce needed.
// =============================================================================
static void fsi_step_distributed(DistributedParticles& dp, int start, int count,
                                 double G, double soft2, double dt,
                                 bool forces_valid = false,
                                 bool tree_prebuilt = false,
                                 int global_tree_subset = -1) {
    if (count <= 0) return;

    // Global tree mode: use subset monopoles from the once-built global tree
    if (global_tree_subset >= 0) {
        long long self_pairs = (long long)count * count;

        // Step 1: force + kick (dt/6)
        if (forces_valid) {
            g_profiler.start();
            kick_velocities(dp.local, start, count, dt * FSI_KICK);
            g_profiler.stop(StepProfiler::DRIFT_KICK);
        } else {
            g_profiler.start();
            tree_global_self_acceleration(dp.local, g_octree, global_tree_subset, G, soft2);
            tree_momentum_correction(dp.local, start, count);
            g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
            g_profiler.start();
            kick_velocities(dp.local, start, count, dt * FSI_KICK);
            g_profiler.stop(StepProfiler::DRIFT_KICK);
        }

        // T drift (dt/2) — field particles only if subsystems active
        g_profiler.start();
#ifdef ENABLE_COLLISIONAL
        if (dp.local.collisional && !g_subsystem_mgr.active.empty()) {
            drift_field_only(dp.local, start, count, dt * FSI_DRIFT);
            g_profiler.stop(StepProfiler::DRIFT_KICK);
            g_profiler.start();
            advance_subsystems_in_range(dp.local, start, count, G, dt * FSI_DRIFT);
            g_profiler.stop(StepProfiler::SUBSYSTEM_INTEG);
        } else
#endif
        {
            drift(dp.local, start, count, dt * FSI_DRIFT);
            g_profiler.stop(StepProfiler::DRIFT_KICK);
        }

        // Refresh positions in global tree after drift
        g_profiler.start();
        g_octree.refresh_positions(dp.local);
        g_profiler.stop(StepProfiler::TREE_MAINT);

        // Step 3: force + gradient + kick (2dt/3)
        g_profiler.start();
        tree_global_self_acceleration(dp.local, g_octree, global_tree_subset, G, soft2);
        tree_momentum_correction(dp.local, start, count);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        g_profiler.start();
        g_octree.compute_dual_cell_accelerations(dp.local);
        tree_global_self_gradient_correction(dp.local, g_octree, global_tree_subset, G, soft2, dt);
        tree_momentum_correction(dp.local, start, count);
        g_profiler.stop(StepProfiler::SELF_GRAD, self_pairs);
        g_profiler.start();
        kick_velocities(dp.local, start, count, dt * FSI_MOD_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);

        // T drift (dt/2) — field particles only if subsystems active
        g_profiler.start();
#ifdef ENABLE_COLLISIONAL
        if (dp.local.collisional && !g_subsystem_mgr.active.empty()) {
            drift_field_only(dp.local, start, count, dt * FSI_DRIFT);
            g_profiler.stop(StepProfiler::DRIFT_KICK);
            g_profiler.start();
            advance_subsystems_in_range(dp.local, start, count, G, dt * FSI_DRIFT);
            g_profiler.stop(StepProfiler::SUBSYSTEM_INTEG);
        } else
#endif
        {
            drift(dp.local, start, count, dt * FSI_DRIFT);
            g_profiler.stop(StepProfiler::DRIFT_KICK);
        }

        // Refresh positions in global tree after drift
        g_profiler.start();
        g_octree.refresh_positions(dp.local);
        g_profiler.stop(StepProfiler::TREE_MAINT);

        // Step 5: force + kick (dt/6)
        g_profiler.start();
        tree_global_self_acceleration(dp.local, g_octree, global_tree_subset, G, soft2);
        tree_momentum_correction(dp.local, start, count);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        g_profiler.start();
        kick_velocities(dp.local, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);

        return;
    }

    bool tree_mode = use_tree && count >= tree_threshold;
    long long self_pairs = (long long)count * count;

    // Sync packed positions for vectorized kernel loads
    dp.local.pack_pos_range(start, count);

    // U kick (dt/6) - force then kick (separate, matching FROST)
    if (forces_valid) {
        g_profiler.start();
        kick_velocities(dp.local, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else if (tree_mode) {
        g_profiler.start();
        if (!tree_prebuilt) {
            g_octree.n_leaf = tree_n_leaf;
            g_octree.n_group = tree_n_group;
            g_octree.build(dp.local, start, count, tree_theta);
            tree_prebuilt = true;
        }
        int my_g_off, my_g_cnt;
        get_tree_group_range(g_octree, my_g_off, my_g_cnt);
        zero_accelerations(dp.local, start, count);
        tree_self_acceleration(dp.local, g_octree, start, count, G, soft2, my_g_off, my_g_cnt);
        dp.allreduce_accelerations_subset(start, count);
        tree_momentum_correction(dp.local, start, count);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        g_profiler.start();
        kick_velocities(dp.local, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else {
        g_profiler.start();
        self_acceleration_distributed(dp, start, count, G, soft2);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        g_profiler.start();
        kick_velocities(dp.local, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // T drift (dt/2) — field particles only if subsystems active
    g_profiler.start();
#ifdef ENABLE_COLLISIONAL
    if (dp.local.collisional && !g_subsystem_mgr.active.empty()) {
        drift_field_only(dp.local, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
        g_profiler.start();
        advance_subsystems_in_range(dp.local, start, count, G, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::SUBSYSTEM_INTEG);
    } else
#endif
    {
        drift(dp.local, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // Re-sync packed positions after drift
    g_profiler.start();
    dp.local.pack_pos_range(start, count);
    g_profiler.stop(StepProfiler::PACK_DATA);

    // U-tilde kick (2dt/3) with gradient correction
    if (tree_mode) {
        // Tree mode: force + gradient correction + kick
        // Refresh positions (O(N)) instead of rebuilding tree (O(N log N))
        g_profiler.start();
        if (!tree_prebuilt) {
            g_octree.n_leaf = tree_n_leaf;
            g_octree.n_group = tree_n_group;
            g_octree.build(dp.local, start, count, tree_theta);
            tree_prebuilt = true;
        } else {
            g_octree.refresh_positions(dp.local);
        }
        int my_g_off, my_g_cnt;
        get_tree_group_range(g_octree, my_g_off, my_g_cnt);
        zero_accelerations(dp.local, start, count);
        tree_self_acceleration(dp.local, g_octree, start, count, G, soft2, my_g_off, my_g_cnt);
        dp.allreduce_accelerations_subset(start, count);
        tree_momentum_correction(dp.local, start, count);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        // Gradient correction via tree: all ranks compute (not distributed)
        // Forces must be correct (allreduced) before gradient computation
        g_profiler.start();
        g_octree.compute_cell_accelerations(dp.local);
        tree_self_gradient_correction(dp.local, g_octree, G, soft2, dt);
        tree_momentum_correction(dp.local, start, count);
        g_profiler.stop(StepProfiler::SELF_GRAD, self_pairs);
        g_profiler.start();
        kick_velocities(dp.local, start, count, dt * FSI_MOD_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else {
        // Separate: force → allgather → gradient correction → allgather → kick
        g_profiler.start();
        self_acceleration_distributed(dp, start, count, G, soft2);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        g_profiler.start();
        self_gradient_correction_distributed(dp, start, count, G, soft2, dt);
        g_profiler.stop(StepProfiler::SELF_GRAD, self_pairs);
        g_profiler.start();
        kick_velocities(dp.local, start, count, dt * FSI_MOD_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // T drift (dt/2) — field particles only if subsystems active
    g_profiler.start();
#ifdef ENABLE_COLLISIONAL
    if (dp.local.collisional && !g_subsystem_mgr.active.empty()) {
        drift_field_only(dp.local, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
        g_profiler.start();
        advance_subsystems_in_range(dp.local, start, count, G, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::SUBSYSTEM_INTEG);
    } else
#endif
    {
        drift(dp.local, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // Re-sync packed positions after drift
    g_profiler.start();
    dp.local.pack_pos_range(start, count);
    g_profiler.stop(StepProfiler::PACK_DATA);

    // U kick (dt/6) - force then kick
    if (tree_mode) {
        g_profiler.start();
        g_octree.refresh_positions(dp.local);
        int my_g_off, my_g_cnt;
        get_tree_group_range(g_octree, my_g_off, my_g_cnt);
        zero_accelerations(dp.local, start, count);
        tree_self_acceleration(dp.local, g_octree, start, count, G, soft2, my_g_off, my_g_cnt);
        dp.allreduce_accelerations_subset(start, count);
        tree_momentum_correction(dp.local, start, count);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        g_profiler.start();
        kick_velocities(dp.local, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else {
        g_profiler.start();
        self_acceleration_distributed(dp, start, count, G, soft2);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        g_profiler.start();
        kick_velocities(dp.local, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }
}

// =============================================================================
// Distributed cross-acceleration between slow and fast particles
// Tiled GPU kernel per FROST paper
// =============================================================================
static void cross_acceleration_distributed(DistributedParticles& dp,
                                           int slow_start, int slow_count,
                                           int fast_start, int fast_count,
                                           double G, double soft2) {
    if (slow_count <= 0 || fast_count <= 0) return;


    auto pos = dp.local.pos_packed;
    RawView pax{dp.local.ax.data()}, pay{dp.local.ay.data()}, paz{dp.local.az.data()};
#ifdef ENABLE_COLLISIONAL
    const int* sid_raw = dp.local.subsystem_id.data();
    bool coll = dp.local.collisional;
#endif

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t scratch_size = TILE_SIZE * 4 * sizeof(double);

    // Slow particles: compute force from fast particles
    int slice_start, slice_count;
    dp.get_force_slice(slow_start, slow_count, slice_start, slice_count);

    if (slice_count > 0) {
        int num_teams = (slice_count + TILE_SIZE - 1) / TILE_SIZE;
        const int f_start = fast_start;
        const int f_end = fast_start + fast_count;

        Kokkos::parallel_for("cross_slow_from_fast_tiled",
            team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_LAMBDA(const member_type& team) {
                int ii = team.league_rank() * TILE_SIZE + team.team_rank();
                int i = slice_start + ii;

                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile(team.team_scratch(0), TILE_SIZE * 4);

                double xi = 0, yi = 0, zi = 0;
#ifdef ENABLE_COLLISIONAL
                int sid_i = -1;
#endif

                if (ii < slice_count) {
                    double mi_pad;
                    load4(&pos(i, 0), xi, yi, zi, mi_pad);
#ifdef ENABLE_COLLISIONAL
                    if (coll) sid_i = sid_raw[i];
#endif
                }

                double axi = 0, ayi = 0, azi = 0;

                for (int tile_st = f_start; tile_st < f_end; tile_st += TILE_SIZE) {
                    int tile_end = (tile_st + TILE_SIZE < f_end) ? tile_st + TILE_SIZE : f_end;
                    int tile_cnt = tile_end - tile_st;

                    int t = team.team_rank();
                    if (t < tile_cnt) {
                        int idx = tile_st + t;
                        int base = t * 4;
                        load4(&pos(idx, 0), tile(base), tile(base+1), tile(base+2), tile(base+3));
                    }
                    team.team_barrier();

                    if (ii < slice_count) {
                        for (int jt = 0; jt < tile_cnt; jt++) {
                            int j_cross = f_start + (tile_st - f_start) + jt;
#ifdef ENABLE_COLLISIONAL
                            if (coll && sid_i >= 0 && sid_raw[j_cross] == sid_i) continue;
#endif


                            int jb = jt * 4;
                            double dx = tile(jb    ) - xi;
                            double dy = tile(jb + 1) - yi;
                            double dz = tile(jb + 2) - zi;

                            double r2 = dx*dx + dy*dy + dz*dz + soft2;
                            double inv_r = fast_rsqrt(r2);
                            double inv_r3 = inv_r * inv_r * inv_r;
                            double mj_inv_r3 = G * tile(jb + 3) * inv_r3;

                            axi += mj_inv_r3 * dx;
                            ayi += mj_inv_r3 * dy;
                            azi += mj_inv_r3 * dz;
                        }
                    }
                    team.team_barrier();
                }

                if (ii < slice_count) {
                    pax(i) = axi;
                    pay(i) = ayi;
                    paz(i) = azi;
                }
            });
    }

    // Fast particles: compute force from slow particles
    int slice_start_f, slice_count_f;
    dp.get_force_slice(fast_start, fast_count, slice_start_f, slice_count_f);

    if (slice_count_f > 0) {
        int num_teams_f = (slice_count_f + TILE_SIZE - 1) / TILE_SIZE;
        const int s_start = slow_start;
        const int s_end = slow_start + slow_count;

        Kokkos::parallel_for("cross_fast_from_slow_tiled",
            team_policy(num_teams_f, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_LAMBDA(const member_type& team) {
                int ii = team.league_rank() * TILE_SIZE + team.team_rank();
                int i = slice_start_f + ii;

                const int N = TILE_SIZE;
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile(team.team_scratch(0), N * 4);

                double xi = 0, yi = 0, zi = 0;
#ifdef ENABLE_COLLISIONAL
                int sid_i = -1;
#endif

                if (ii < slice_count_f) {
                    double mi_pad;
                    load4(&pos(i, 0), xi, yi, zi, mi_pad);
#ifdef ENABLE_COLLISIONAL
                    if (coll) sid_i = sid_raw[i];
#endif
                }

                double axi = 0, ayi = 0, azi = 0;

                for (int tile_st = s_start; tile_st < s_end; tile_st += TILE_SIZE) {
                    int tile_end = (tile_st + TILE_SIZE < s_end) ? tile_st + TILE_SIZE : s_end;
                    int tile_cnt = tile_end - tile_st;

                    int t = team.team_rank();
                    if (t < tile_cnt) {
                        int idx = tile_st + t;
                        int tb = t * 4;
                        load4(&pos(idx, 0), tile(tb), tile(tb+1), tile(tb+2), tile(tb+3));
                    }
                    team.team_barrier();

                    if (ii < slice_count_f) {
                        for (int jt = 0; jt < tile_cnt; jt++) {
                            int j_cross = s_start + (tile_st - s_start) + jt;
#ifdef ENABLE_COLLISIONAL
                            if (coll && sid_i >= 0 && sid_raw[j_cross] == sid_i) continue;
#endif

                            int jb = jt * 4;
                            double dx = tile(jb)   - xi;
                            double dy = tile(jb+1) - yi;
                            double dz = tile(jb+2) - zi;

                            double r2 = dx*dx + dy*dy + dz*dz + soft2;
                            double inv_r = fast_rsqrt(r2);
                            double inv_r3 = inv_r * inv_r * inv_r;
                            double mj_inv_r3 = G * tile(jb+3) * inv_r3;

                            axi += mj_inv_r3 * dx;
                            ayi += mj_inv_r3 * dy;
                            azi += mj_inv_r3 * dz;
                        }
                    }
                    team.team_barrier();
                }

                if (ii < slice_count_f) {
                    pax(i) = axi;
                    pay(i) = ayi;
                    paz(i) = azi;
                }
            });
    }
    Kokkos::fence();

    // Must use separate allgathers: each rank computed its slice of slow and
    // its slice of fast independently, so the distributions don't match a combined range
    if (use_nonblocking_mpi) {
        dp.iallgather_accelerations_subset(slow_start, slow_count, 0);
        dp.iallgather_accelerations_subset(fast_start, fast_count, 1);
        dp.wait_and_unpack_accelerations(slow_start, slow_count, 0);
        dp.wait_and_unpack_accelerations(fast_start, fast_count, 1);
    } else {
        dp.allgather_accelerations_subset(slow_start, slow_count);
        dp.allgather_accelerations_subset(fast_start, fast_count);
    }
}

// Cross-acceleration with gradient correction - tiled GPU kernel
static void cross_acceleration_and_gradient_distributed(DistributedParticles& dp,
                                                        int slow_start, int slow_count,
                                                        int fast_start, int fast_count,
                                                        double G, double soft2, double dt) {
    // First compute cross-accelerations (already tiled)
    cross_acceleration_distributed(dp, slow_start, slow_count,
                                   fast_start, fast_count, G, soft2);


    auto pos = dp.local.pos_packed;
    RawView pax{dp.local.ax.data()}, pay{dp.local.ay.data()}, paz{dp.local.az.data()};
#ifdef ENABLE_COLLISIONAL
    const int* sid_raw = dp.local.subsystem_id.data();
    bool coll = dp.local.collisional;
#endif

    const double coeff = G * dt * dt * GRAD_COEFF;

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t scratch_size = TILE_SIZE * 7 * sizeof(double);

    // Gradient correction for slow particles from fast
    int slice_start, slice_count;
    dp.get_force_slice(slow_start, slow_count, slice_start, slice_count);

    if (slice_count > 0) {
        int num_teams = (slice_count + TILE_SIZE - 1) / TILE_SIZE;
        const int f_start = fast_start;
        const int f_end = fast_start + fast_count;

        Kokkos::parallel_for("cross_grad_slow_tiled",
            team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_LAMBDA(const member_type& team) {
                int ii = team.league_rank() * TILE_SIZE + team.team_rank();
                int i = slice_start + ii;

                const int N = TILE_SIZE;
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile(team.team_scratch(0), N * 7);

                double xi = 0, yi = 0, zi = 0;
                double axi = 0, ayi = 0, azi = 0;
#ifdef ENABLE_COLLISIONAL
                int sid_i = -1;
#endif

                if (ii < slice_count) {
                    double mi_pad;
                    load4(&pos(i, 0), xi, yi, zi, mi_pad);
                    axi = pax(i); ayi = pay(i); azi = paz(i);
#ifdef ENABLE_COLLISIONAL
                    if (coll) sid_i = sid_raw[i];
#endif
                }

                double cx = 0, cy = 0, cz = 0;

                for (int tile_st = f_start; tile_st < f_end; tile_st += TILE_SIZE) {
                    int tile_end = (tile_st + TILE_SIZE < f_end) ? tile_st + TILE_SIZE : f_end;
                    int tile_cnt = tile_end - tile_st;

                    int t = team.team_rank();
                    if (t < tile_cnt) {
                        int idx = tile_st + t;
                        int tb = t * 7;
                        load4(&pos(idx, 0), tile(tb), tile(tb+1), tile(tb+2), tile(tb+3));
                        tile(tb+4) = pax(idx);
                        tile(tb+5) = pay(idx);
                        tile(tb+6) = paz(idx);
                    }
                    team.team_barrier();

                    if (ii < slice_count) {
                        for (int jt = 0; jt < tile_cnt; jt++) {
                            int j_cross = tile_st + jt;
#ifdef ENABLE_COLLISIONAL
                            if (coll && sid_i >= 0 && sid_raw[j_cross] == sid_i) continue;
#endif

                            int jb = jt * 7;
                            double dx = tile(jb)   - xi;
                            double dy = tile(jb+1) - yi;
                            double dz = tile(jb+2) - zi;

                            double dax = tile(jb+4) - axi;
                            double day = tile(jb+5) - ayi;
                            double daz = tile(jb+6) - azi;

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

                if (ii < slice_count) {
                    pax(i) += cx;
                    pay(i) += cy;
                    paz(i) += cz;
                }
            });
    }

    // Gradient correction for fast particles from slow
    int slice_start_f, slice_count_f;
    dp.get_force_slice(fast_start, fast_count, slice_start_f, slice_count_f);

    if (slice_count_f > 0) {
        int num_teams_f = (slice_count_f + TILE_SIZE - 1) / TILE_SIZE;
        const int s_start = slow_start;
        const int s_end = slow_start + slow_count;

        Kokkos::parallel_for("cross_grad_fast_tiled",
            team_policy(num_teams_f, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_LAMBDA(const member_type& team) {
                int ii = team.league_rank() * TILE_SIZE + team.team_rank();
                int i = slice_start_f + ii;

                const int N = TILE_SIZE;
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile(team.team_scratch(0), N * 7);

                double xi = 0, yi = 0, zi = 0;
                double axi = 0, ayi = 0, azi = 0;
#ifdef ENABLE_COLLISIONAL
                int sid_i = -1;
#endif

                if (ii < slice_count_f) {
                    double mi_pad;
                    load4(&pos(i, 0), xi, yi, zi, mi_pad);
                    axi = pax(i); ayi = pay(i); azi = paz(i);
#ifdef ENABLE_COLLISIONAL
                    if (coll) sid_i = sid_raw[i];
#endif
                }

                double cx = 0, cy = 0, cz = 0;

                for (int tile_st = s_start; tile_st < s_end; tile_st += TILE_SIZE) {
                    int tile_end = (tile_st + TILE_SIZE < s_end) ? tile_st + TILE_SIZE : s_end;
                    int tile_cnt = tile_end - tile_st;

                    int t = team.team_rank();
                    if (t < tile_cnt) {
                        int idx = tile_st + t;
                        int tb = t * 7;
                        load4(&pos(idx, 0), tile(tb), tile(tb+1), tile(tb+2), tile(tb+3));
                        tile(tb+4) = pax(idx);
                        tile(tb+5) = pay(idx);
                        tile(tb+6) = paz(idx);
                    }
                    team.team_barrier();

                    if (ii < slice_count_f) {
                        for (int jt = 0; jt < tile_cnt; jt++) {
                            int j_cross = tile_st + jt;
#ifdef ENABLE_COLLISIONAL
                            if (coll && sid_i >= 0 && sid_raw[j_cross] == sid_i) continue;
#endif


                            int jb = jt * 7;
                            double dx = tile(jb)   - xi;
                            double dy = tile(jb+1) - yi;
                            double dz = tile(jb+2) - zi;

                            double dax = tile(jb+4) - axi;
                            double day = tile(jb+5) - ayi;
                            double daz = tile(jb+6) - azi;

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

                if (ii < slice_count_f) {
                    pax(i) += cx;
                    pay(i) += cy;
                    paz(i) += cz;
                }
            });
    }
    Kokkos::fence();

    // Must use separate allgathers: force slices were computed independently
    if (use_nonblocking_mpi) {
        dp.iallgather_accelerations_subset(slow_start, slow_count, 0);
        dp.iallgather_accelerations_subset(fast_start, fast_count, 1);
        dp.wait_and_unpack_accelerations(slow_start, slow_count, 0);
        dp.wait_and_unpack_accelerations(fast_start, fast_count, 1);
    } else {
        dp.allgather_accelerations_subset(slow_start, slow_count);
        dp.allgather_accelerations_subset(fast_start, fast_count);
    }
}

// Distributed adaptive timestep computation - tiled GPU kernel
static void compute_timestep_distributed(DistributedParticles& dp, int start, int count,
                                      double G, double soft2, double eta,
                                      bool symmetrize = true) {
    if (count <= 1) return;


    int slice_start, slice_count;
    dp.get_force_slice(start, count, slice_start, slice_count);

    if (slice_count == 0) {
        dp.allgather_timestep_subset(start, count);
        return;
    }

    auto pos = dp.local.pos_packed;
    auto vel = dp.local.vel_packed;
    RawView dt_f{dp.local.dt_adapt.data()};
    RawView pax_dt{dp.local.ax.data()}, pay_dt{dp.local.ay.data()}, paz_dt{dp.local.az.data()};
#ifdef ENABLE_COLLISIONAL
    const int* sid_raw = dp.local.subsystem_id.data();
    bool coll = dp.local.collisional;
    double eta_jerk_val = coll ? g_collisional.eta_jerk : 0.0;
#endif

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

    const int j_start = start;
    const int j_end = start + count;

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    // Need x,y,z,m,vx,vy,vz for adaptive timestep
    size_t scratch_size = TILE_SIZE * 7 * sizeof(double);
    int num_teams = (slice_count + TILE_SIZE - 1) / TILE_SIZE;

    Kokkos::parallel_for("dist_dt_tiled",
        team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * TILE_SIZE + team.team_rank();
            int i = slice_start + ii;

            const int N = TILE_SIZE;
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), N * 7);

            double xi = 0, yi = 0, zi = 0, mi = 0;
            double vxi = 0, vyi = 0, vzi = 0;
#ifdef ENABLE_COLLISIONAL
            int sid_i = -1;
#endif
            if (ii < slice_count) {
                double vpad;
                load4(&pos(i, 0), xi, yi, zi, mi);
                load4(&vel(i, 0), vxi, vyi, vzi, vpad);
#ifdef ENABLE_COLLISIONAL
                if (coll) sid_i = sid_raw[i];
#endif
            }

            double t_min_sq = 1e60;   // non-symmetrized path
            double t_min_i = 1e30;    // symmetrized path
#ifdef ENABLE_COLLISIONAL
            double jerk_x = 0, jerk_y = 0, jerk_z = 0;
            double acc_xi = 0, acc_yi = 0, acc_zi = 0;  // fresh subset acceleration for jerk criterion
#endif

            for (int tile_st = j_start; tile_st < j_end; tile_st += TILE_SIZE) {
                int tile_end = (tile_st + TILE_SIZE < j_end) ? tile_st + TILE_SIZE : j_end;
                int tile_cnt = tile_end - tile_st;

                int t = team.team_rank();
                if (t < tile_cnt) {
                    int idx = tile_st + t;
                    int tb = t * 7;
                    double vpad;
                    load4(&pos(idx, 0), tile(tb), tile(tb+1), tile(tb+2), tile(tb+3));
                    load4(&vel(idx, 0), tile(tb+4), tile(tb+5), tile(tb+6), vpad);
                }
                team.team_barrier();

                if (ii < slice_count) {
                    for (int jt = 0; jt < tile_cnt; jt++) {
                        int j = tile_st + jt;
                        if (j == i) continue;
#ifdef ENABLE_COLLISIONAL
                        if (coll && sid_i >= 0 && sid_raw[j] == sid_i) continue;
#endif

                        int jb = jt * 7;
                        double dx = tile(jb)   - xi;
                        double dy = tile(jb+1) - yi;
                        double dz = tile(jb+2) - zi;
                        double r2 = dx*dx + dy*dy + dz*dz;

                        double dvx = tile(jb+4) - vxi;
                        double dvy = tile(jb+5) - vyi;
                        double dvz = tile(jb+6) - vzi;
                        double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;

                        double r2s = r2 + soft2;
                        double inv_r = fast_rsqrt(r2s);
                        double mj = tile(jb+3);
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

#ifdef ENABLE_COLLISIONAL
                        // Jerk + acceleration accumulation (collisional only)
                        if (coll) {
                            double inv_r3 = inv_r * inv_r * inv_r;
                            double rdotv = dx*dvx + dy*dvy + dz*dvz;
                            double fac1 = G * mj * inv_r3;
                            double fac2 = 3.0 * rdotv * inv_r * inv_r;
                            jerk_x += fac1 * (dvx - fac2 * dx);
                            jerk_y += fac1 * (dvy - fac2 * dy);
                            jerk_z += fac1 * (dvz - fac2 * dz);
                            // Fresh subset acceleration for jerk criterion (BIFROST §3.4.5)
                            acc_xi += fac1 * dx;
                            acc_yi += fac1 * dy;
                            acc_zi += fac1 * dz;
                        }
#endif

                    }
                }
                team.team_barrier();
            }

            if (ii < slice_count) {
                dt_f(i) = symmetrize ? t_min_i : Kokkos::sqrt(t_min_sq);
#ifdef ENABLE_COLLISIONAL
                // Jerk timestep: dt_jerk = eta_j * |a| / |j| (BIFROST §3.4.5)
                // Uses freshly accumulated subset acceleration, not stale p.ax/ay/az
                if (coll && eta_jerk_val > 0.0) {
                    double a_mag = Kokkos::sqrt(acc_xi*acc_xi + acc_yi*acc_yi + acc_zi*acc_zi);
                    double j_mag = Kokkos::sqrt(jerk_x*jerk_x + jerk_y*jerk_y + jerk_z*jerk_z);
                    if (j_mag > EPS && a_mag > EPS) {
                        double dt_jerk = eta_jerk_val * a_mag / j_mag;
                        dt_f(i) = Kokkos::fmin(dt_f(i), dt_jerk);
                    }
                }
#endif
            }
        });
    Kokkos::fence();

    // Allgather dt_adapt values (uses GPU-aware MPI when available)
    dp.allgather_timestep_subset(start, count);
}

// =============================================================================
// CPU serial HHS-FSI for very small subsets (avoids GPU kernel launch overhead)
// Copies subset to host, runs plain C++ loops, copies back.
// =============================================================================

// Host-side data for CPU serial computation
struct CpuSubset {
    std::vector<double> x, y, z, vx, vy, vz, ax, ay, az, mass, dt_adapt;
    std::vector<int64_t> ix, iy, iz;
    std::vector<int> sid;  // subsystem_id (collisional mode)
#ifdef ENABLE_COLLISIONAL
    bool collisional = false;
    std::vector<double> grad_mag;     // gradient magnitude (stale from previous step)
    std::vector<double> dt_level0;    // level-0 timestep for clamping
    std::vector<int> particle_id;     // unique particle IDs for neighbor pair output
#else
    static constexpr bool collisional = false;
#endif
    int count;
    int device_start = 0;  // offset into device Particles arrays

    void resize(int n) {
        count = n;
        x.resize(n); y.resize(n); z.resize(n);
        vx.resize(n); vy.resize(n); vz.resize(n);
        ax.resize(n); ay.resize(n); az.resize(n);
        mass.resize(n); dt_adapt.resize(n);
        ix.resize(n); iy.resize(n); iz.resize(n);
#ifdef ENABLE_COLLISIONAL
        sid.resize(n, -1);
        grad_mag.resize(n, 0.0);
        dt_level0.resize(n, 1e60);
        particle_id.resize(n, 0);
#endif
    }

    // Copy subset [start, start+count) from device to host
    void copy_from_device(Particles& p, int start) {
        auto h_x = Kokkos::create_mirror_view(Kokkos::subview(p.x, Kokkos::pair<int,int>(start, start+count)));
        auto h_y = Kokkos::create_mirror_view(Kokkos::subview(p.y, Kokkos::pair<int,int>(start, start+count)));
        auto h_z = Kokkos::create_mirror_view(Kokkos::subview(p.z, Kokkos::pair<int,int>(start, start+count)));
        auto h_vx = Kokkos::create_mirror_view(Kokkos::subview(p.vx, Kokkos::pair<int,int>(start, start+count)));
        auto h_vy = Kokkos::create_mirror_view(Kokkos::subview(p.vy, Kokkos::pair<int,int>(start, start+count)));
        auto h_vz = Kokkos::create_mirror_view(Kokkos::subview(p.vz, Kokkos::pair<int,int>(start, start+count)));
        auto h_m  = Kokkos::create_mirror_view(Kokkos::subview(p.mass, Kokkos::pair<int,int>(start, start+count)));
        auto h_dt = Kokkos::create_mirror_view(Kokkos::subview(p.dt_adapt, Kokkos::pair<int,int>(start, start+count)));
        auto h_ix = Kokkos::create_mirror_view(Kokkos::subview(p.ix, Kokkos::pair<int,int>(start, start+count)));
        auto h_iy = Kokkos::create_mirror_view(Kokkos::subview(p.iy, Kokkos::pair<int,int>(start, start+count)));
        auto h_iz = Kokkos::create_mirror_view(Kokkos::subview(p.iz, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::deep_copy(h_x, Kokkos::subview(p.x, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::deep_copy(h_y, Kokkos::subview(p.y, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::deep_copy(h_z, Kokkos::subview(p.z, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::deep_copy(h_vx, Kokkos::subview(p.vx, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::deep_copy(h_vy, Kokkos::subview(p.vy, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::deep_copy(h_vz, Kokkos::subview(p.vz, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::deep_copy(h_m, Kokkos::subview(p.mass, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::deep_copy(h_dt, Kokkos::subview(p.dt_adapt, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::deep_copy(h_ix, Kokkos::subview(p.ix, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::deep_copy(h_iy, Kokkos::subview(p.iy, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::deep_copy(h_iz, Kokkos::subview(p.iz, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::fence();
        for (int i = 0; i < count; i++) {
            x[i] = h_x(i); y[i] = h_y(i); z[i] = h_z(i);
            vx[i] = h_vx(i); vy[i] = h_vy(i); vz[i] = h_vz(i);
            mass[i] = h_m(i); dt_adapt[i] = h_dt(i);
            ix[i] = h_ix(i); iy[i] = h_iy(i); iz[i] = h_iz(i);
        }
#ifdef ENABLE_COLLISIONAL
        collisional = p.collisional;
        device_start = start;
        if (collisional) {
            auto h_sid = Kokkos::create_mirror_view(Kokkos::subview(p.subsystem_id, Kokkos::pair<int,int>(start, start+count)));
            auto h_gm  = Kokkos::create_mirror_view(Kokkos::subview(p.grad_mag, Kokkos::pair<int,int>(start, start+count)));
            auto h_dl0 = Kokkos::create_mirror_view(Kokkos::subview(p.dt_level0, Kokkos::pair<int,int>(start, start+count)));
            auto h_pid = Kokkos::create_mirror_view(Kokkos::subview(p.particle_id, Kokkos::pair<int,int>(start, start+count)));
            Kokkos::deep_copy(h_sid, Kokkos::subview(p.subsystem_id, Kokkos::pair<int,int>(start, start+count)));
            Kokkos::deep_copy(h_gm, Kokkos::subview(p.grad_mag, Kokkos::pair<int,int>(start, start+count)));
            Kokkos::deep_copy(h_dl0, Kokkos::subview(p.dt_level0, Kokkos::pair<int,int>(start, start+count)));
            Kokkos::deep_copy(h_pid, Kokkos::subview(p.particle_id, Kokkos::pair<int,int>(start, start+count)));
            Kokkos::fence();
            for (int i = 0; i < count; i++) {
                sid[i] = h_sid(i);
                grad_mag[i] = h_gm(i);
                dt_level0[i] = h_dl0(i);
                particle_id[i] = h_pid(i);
            }
        }
#endif
    }

    // Copy results back to device (positions, velocities, mass, dt_adapt, ix/iy/iz)
    // Must copy mass and dt_adapt because cpu_partition_by_pivot reorders all fields
    void copy_to_device(Particles& p, int start) {
        auto h_x = Kokkos::create_mirror_view(Kokkos::subview(p.x, Kokkos::pair<int,int>(start, start+count)));
        auto h_y = Kokkos::create_mirror_view(Kokkos::subview(p.y, Kokkos::pair<int,int>(start, start+count)));
        auto h_z = Kokkos::create_mirror_view(Kokkos::subview(p.z, Kokkos::pair<int,int>(start, start+count)));
        auto h_vx = Kokkos::create_mirror_view(Kokkos::subview(p.vx, Kokkos::pair<int,int>(start, start+count)));
        auto h_vy = Kokkos::create_mirror_view(Kokkos::subview(p.vy, Kokkos::pair<int,int>(start, start+count)));
        auto h_vz = Kokkos::create_mirror_view(Kokkos::subview(p.vz, Kokkos::pair<int,int>(start, start+count)));
        auto h_m  = Kokkos::create_mirror_view(Kokkos::subview(p.mass, Kokkos::pair<int,int>(start, start+count)));
        auto h_dt = Kokkos::create_mirror_view(Kokkos::subview(p.dt_adapt, Kokkos::pair<int,int>(start, start+count)));
        auto h_ix = Kokkos::create_mirror_view(Kokkos::subview(p.ix, Kokkos::pair<int,int>(start, start+count)));
        auto h_iy = Kokkos::create_mirror_view(Kokkos::subview(p.iy, Kokkos::pair<int,int>(start, start+count)));
        auto h_iz = Kokkos::create_mirror_view(Kokkos::subview(p.iz, Kokkos::pair<int,int>(start, start+count)));
        for (int i = 0; i < count; i++) {
            h_x(i) = x[i]; h_y(i) = y[i]; h_z(i) = z[i];
            h_vx(i) = vx[i]; h_vy(i) = vy[i]; h_vz(i) = vz[i];
            h_m(i) = mass[i]; h_dt(i) = dt_adapt[i];
            h_ix(i) = ix[i]; h_iy(i) = iy[i]; h_iz(i) = iz[i];
        }
        Kokkos::deep_copy(Kokkos::subview(p.x, Kokkos::pair<int,int>(start, start+count)), h_x);
        Kokkos::deep_copy(Kokkos::subview(p.y, Kokkos::pair<int,int>(start, start+count)), h_y);
        Kokkos::deep_copy(Kokkos::subview(p.z, Kokkos::pair<int,int>(start, start+count)), h_z);
        Kokkos::deep_copy(Kokkos::subview(p.vx, Kokkos::pair<int,int>(start, start+count)), h_vx);
        Kokkos::deep_copy(Kokkos::subview(p.vy, Kokkos::pair<int,int>(start, start+count)), h_vy);
        Kokkos::deep_copy(Kokkos::subview(p.vz, Kokkos::pair<int,int>(start, start+count)), h_vz);
        Kokkos::deep_copy(Kokkos::subview(p.mass, Kokkos::pair<int,int>(start, start+count)), h_m);
        Kokkos::deep_copy(Kokkos::subview(p.dt_adapt, Kokkos::pair<int,int>(start, start+count)), h_dt);
        Kokkos::deep_copy(Kokkos::subview(p.ix, Kokkos::pair<int,int>(start, start+count)), h_ix);
        Kokkos::deep_copy(Kokkos::subview(p.iy, Kokkos::pair<int,int>(start, start+count)), h_iy);
        Kokkos::deep_copy(Kokkos::subview(p.iz, Kokkos::pair<int,int>(start, start+count)), h_iz);
#ifdef ENABLE_COLLISIONAL
        if (collisional) {
            auto h_sid = Kokkos::create_mirror_view(Kokkos::subview(p.subsystem_id, Kokkos::pair<int,int>(start, start+count)));
            auto h_dl0 = Kokkos::create_mirror_view(Kokkos::subview(p.dt_level0, Kokkos::pair<int,int>(start, start+count)));
            for (int i = 0; i < count; i++) {
                h_sid(i) = sid[i];
                h_dl0(i) = dt_level0[i];
            }
            Kokkos::deep_copy(Kokkos::subview(p.subsystem_id, Kokkos::pair<int,int>(start, start+count)), h_sid);
            Kokkos::deep_copy(Kokkos::subview(p.dt_level0, Kokkos::pair<int,int>(start, start+count)), h_dl0);
        }
#endif
        Kokkos::fence();
    }
};

// CPU self-acceleration (N3L, no kick)
static void cpu_self_acc(CpuSubset& s, int start, int count,
                         double G, double soft2) {
    const int end = start + count;
    for (int i = start; i < end; i++) {
        s.ax[i] = 0; s.ay[i] = 0; s.az[i] = 0;
    }
    for (int i = start; i < end; i++) {
        for (int j = i + 1; j < end; j++) {
            if (s.collisional && s.sid[i] >= 0 && s.sid[j] == s.sid[i]) continue;
            double dx = s.x[j] - s.x[i];
            double dy = s.y[j] - s.y[i];
            double dz = s.z[j] - s.z[i];
            double r2 = dx*dx + dy*dy + dz*dz + soft2;
            double inv_r = 1.0 / std::sqrt(r2);
            double inv_r3 = inv_r * inv_r * inv_r;
            double fj = G * s.mass[j] * inv_r3;
            double fi = G * s.mass[i] * inv_r3;
            s.ax[i] += fj * dx; s.ay[i] += fj * dy; s.az[i] += fj * dz;
            s.ax[j] -= fi * dx; s.ay[j] -= fi * dy; s.az[j] -= fi * dz;
        }
    }
}

// CPU self-gradient correction (N3L: upper triangle, modifies ax/ay/az in-place)
static void cpu_self_grad(CpuSubset& s, int start, int count,
                          double G, double soft2, double dt) {
    const double coeff = G * dt * dt * GRAD_COEFF;
    // Snapshot accelerations
    std::vector<double> ax0(count), ay0(count), az0(count);
    for (int i = 0; i < count; i++) {
        ax0[i] = s.ax[start + i];
        ay0[i] = s.ay[start + i];
        az0[i] = s.az[start + i];
    }
    // N3L: upper triangle
    for (int ii = 0; ii < count; ii++) {
        int i = start + ii;
        for (int jj = ii + 1; jj < count; jj++) {
            int j = start + jj;
            if (s.collisional && s.sid[i] >= 0 && s.sid[j] == s.sid[i]) continue;
            double dx = s.x[j] - s.x[i];
            double dy = s.y[j] - s.y[i];
            double dz = s.z[j] - s.z[i];
            double dax = ax0[jj] - ax0[ii];
            double day = ay0[jj] - ay0[ii];
            double daz = az0[jj] - az0[ii];
            double r2 = dx*dx + dy*dy + dz*dz + soft2;
            double inv_r = 1.0 / std::sqrt(r2);
            double inv_r2 = inv_r * inv_r;
            double inv_r5 = inv_r2 * inv_r2 * inv_r;
            double r_dot_a = dx*dax + dy*day + dz*daz;
            double term_x = r2 * dax - 3.0 * r_dot_a * dx;
            double term_y = r2 * day - 3.0 * r_dot_a * dy;
            double term_z = r2 * daz - 3.0 * r_dot_a * dz;
            double fj = coeff * s.mass[j] * inv_r5;
            double fi = coeff * s.mass[i] * inv_r5;
            s.ax[i] += fj * term_x; s.ay[i] += fj * term_y; s.az[i] += fj * term_z;
            s.ax[j] -= fi * term_x; s.ay[j] -= fi * term_y; s.az[j] -= fi * term_z;
        }
    }
}

// CPU cross-acceleration (N3L, no kick)
static void cpu_cross_acc(CpuSubset& s,
                           int a_start, int a_count,
                           int b_start, int b_count,
                           double G, double soft2) {
    const int a_end = a_start + a_count;
    const int b_end = b_start + b_count;
    for (int i = a_start; i < a_end; i++) { s.ax[i] = 0; s.ay[i] = 0; s.az[i] = 0; }
    for (int j = b_start; j < b_end; j++) { s.ax[j] = 0; s.ay[j] = 0; s.az[j] = 0; }
    for (int i = a_start; i < a_end; i++) {
        for (int j = b_start; j < b_end; j++) {
            if (s.collisional && s.sid[i] >= 0 && s.sid[j] == s.sid[i]) continue;
            double dx = s.x[j] - s.x[i];
            double dy = s.y[j] - s.y[i];
            double dz = s.z[j] - s.z[i];
            double r2 = dx*dx + dy*dy + dz*dz + soft2;
            double inv_r = 1.0 / std::sqrt(r2);
            double inv_r3 = inv_r * inv_r * inv_r;
            double fj = G * s.mass[j] * inv_r3;
            double fi = G * s.mass[i] * inv_r3;
            s.ax[i] += fj * dx; s.ay[i] += fj * dy; s.az[i] += fj * dz;
            s.ax[j] -= fi * dx; s.ay[j] -= fi * dy; s.az[j] -= fi * dz;
        }
    }
}

// CPU cross-gradient correction (N3L, modifies ax/ay/az in-place)
static void cpu_cross_grad(CpuSubset& s,
                            int a_start, int a_count,
                            int b_start, int b_count,
                            double G, double soft2, double grad_dt) {
    const double coeff = G * grad_dt * grad_dt * GRAD_COEFF;
    std::vector<double> ax0_a(a_count), ay0_a(a_count), az0_a(a_count);
    std::vector<double> ax0_b(b_count), ay0_b(b_count), az0_b(b_count);
    for (int i = 0; i < a_count; i++) {
        ax0_a[i] = s.ax[a_start+i]; ay0_a[i] = s.ay[a_start+i]; az0_a[i] = s.az[a_start+i];
    }
    for (int j = 0; j < b_count; j++) {
        ax0_b[j] = s.ax[b_start+j]; ay0_b[j] = s.ay[b_start+j]; az0_b[j] = s.az[b_start+j];
    }
    for (int ii = 0; ii < a_count; ii++) {
        int i = a_start + ii;
        for (int jj = 0; jj < b_count; jj++) {
            int j = b_start + jj;
            if (s.collisional && s.sid[i] >= 0 && s.sid[j] == s.sid[i]) continue;
            double dx = s.x[j] - s.x[i], dy = s.y[j] - s.y[i], dz = s.z[j] - s.z[i];
            double dax = ax0_b[jj] - ax0_a[ii], day = ay0_b[jj] - ay0_a[ii], daz = az0_b[jj] - az0_a[ii];
            double r2 = dx*dx + dy*dy + dz*dz + soft2;
            double inv_r = 1.0 / std::sqrt(r2);
            double inv_r2 = inv_r * inv_r;
            double inv_r5 = inv_r2 * inv_r2 * inv_r;
            double r_dot_a = dx*dax + dy*day + dz*daz;
            double term_x = r2 * dax - 3.0 * r_dot_a * dx;
            double term_y = r2 * day - 3.0 * r_dot_a * dy;
            double term_z = r2 * daz - 3.0 * r_dot_a * dz;
            double fj = coeff * s.mass[j] * inv_r5;
            double fi = coeff * s.mass[i] * inv_r5;
            s.ax[i] += fj * term_x; s.ay[i] += fj * term_y; s.az[i] += fj * term_z;
            s.ax[j] -= fi * term_x; s.ay[j] -= fi * term_y; s.az[j] -= fi * term_z;
        }
    }
}

// CPU drift
static void cpu_drift(CpuSubset& s, int start, int count, double dt) {
    if (use_integer_positions) {
        double scale = int_pos_scale;
        double inv_scale = int_pos_inv_scale;
        for (int i = start; i < start + count; i++) {
            s.ix[i] += llrint(s.vx[i] * dt * inv_scale);
            s.iy[i] += llrint(s.vy[i] * dt * inv_scale);
            s.iz[i] += llrint(s.vz[i] * dt * inv_scale);
            s.x[i] = (double)s.ix[i] * scale;
            s.y[i] = (double)s.iy[i] * scale;
            s.z[i] = (double)s.iz[i] * scale;
        }
    } else {
        for (int i = start; i < start + count; i++) {
            s.x[i] += s.vx[i] * dt;
            s.y[i] += s.vy[i] * dt;
            s.z[i] += s.vz[i] * dt;
        }
    }
}

#ifdef ENABLE_COLLISIONAL
// CPU drift for field particles only (skip subsystem members)
static void cpu_drift_field_only(CpuSubset& s, int start, int count, double dt) {
    if (!s.collisional) { cpu_drift(s, start, count, dt); return; }
    if (use_integer_positions) {
        double scale = int_pos_scale;
        double inv_scale = int_pos_inv_scale;
        for (int i = start; i < start + count; i++) {
            if (s.sid[i] >= 0) continue;
            s.ix[i] += llrint(s.vx[i] * dt * inv_scale);
            s.iy[i] += llrint(s.vy[i] * dt * inv_scale);
            s.iz[i] += llrint(s.vz[i] * dt * inv_scale);
            s.x[i] = (double)s.ix[i] * scale;
            s.y[i] = (double)s.iy[i] * scale;
            s.z[i] = (double)s.iz[i] * scale;
        }
    } else {
        for (int i = start; i < start + count; i++) {
            if (s.sid[i] >= 0) continue;
            s.x[i] += s.vx[i] * dt;
            s.y[i] += s.vy[i] * dt;
            s.z[i] += s.vz[i] * dt;
        }
    }
}
#endif

// CPU kick using pre-computed accelerations
static void cpu_kick(CpuSubset& s, int start, int count, double kick_dt) {
    const int end = start + count;
    for (int i = start; i < end; i++) {
        s.vx[i] += s.ax[i] * kick_dt;
        s.vy[i] += s.ay[i] * kick_dt;
        s.vz[i] += s.az[i] * kick_dt;
    }
}

// CPU timestep computation (N3L: upper triangle, squared-space)
static void cpu_compute_timestep(CpuSubset& s, int start, int count,
                                 double G, double soft2, double eta) {
#ifdef ENABLE_COLLISIONAL
    double eta_ff_v = g_collisional.eta_ff;
    double eta_fb_v = g_collisional.eta_fb;
    double eta_jerk_val = s.collisional ? g_collisional.eta_jerk : 0.0;
#else
    double eta_ff_v = eta;
    double eta_fb_v = eta;
    double eta_jerk_val = 0.0;
#endif
    const int end = start + count;

    // Initialize timesteps, jerk and acceleration accumulators
    for (int i = start; i < end; i++) s.dt_adapt[i] = 1e30;
    std::vector<double> jerk_x(count, 0), jerk_y(count, 0), jerk_z(count, 0);
    std::vector<double> acc_x(count, 0), acc_y(count, 0), acc_z(count, 0);

    for (int i = start; i < end; i++) {
        for (int j = i + 1; j < end; j++) {
            if (s.collisional && s.sid[i] >= 0 && s.sid[j] == s.sid[i]) continue;
            double dx = s.x[j] - s.x[i];
            double dy = s.y[j] - s.y[i];
            double dz = s.z[j] - s.z[i];
            double dvx = s.vx[j] - s.vx[i];
            double dvy = s.vy[j] - s.vy[i];
            double dvz = s.vz[j] - s.vz[i];
            double r2 = dx*dx + dy*dy + dz*dz;
            double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
            double r2s = r2 + EPS;
            double inv_r = 1.0 / std::sqrt(r2s);
            double mass_sum = s.mass[i] + s.mass[j];
            double r3 = r2s * r2s * inv_r;

            // Symmetric timestep (always on): compute actual timescales per pair
            double inv_Gm = 1.0 / (G * mass_sum + EPS);
            double inv_vsq = 1.0 / (v_sq + EPS);
            double t_ff_sq = r3 * inv_Gm;
            double t_fb_sq = r2s * inv_vsq;
            double t_ff = t_ff_sq * (1.0 / std::sqrt(t_ff_sq + EPS));
            double t_fb = t_fb_sq * (1.0 / std::sqrt(t_fb_sq + EPS));

            double inv_r2 = inv_r * inv_r;
            double r_dot_v = dx*dvx + dy*dvy + dz*dvz;

            double c_ff = 1.5 * r_dot_v * inv_r2;
            double c_fb = r_dot_v * inv_r2 * (1.0 + G * mass_sum * inv_r * inv_vsq);

            double ct_ff = c_ff * t_ff_sq;
            double s_ff = 1.0 + c_ff * ct_ff;
            double u_ff = s_ff * (1.0 / std::sqrt(s_ff));
            double t_sym_ff = t_ff + ct_ff / (u_ff + 1.0);

            double ct_fb = c_fb * t_fb_sq;
            double s_fb = 1.0 + c_fb * ct_fb;
            double u_fb = s_fb * (1.0 / std::sqrt(s_fb));
            double t_sym_fb = t_fb + ct_fb / (u_fb + 1.0);

            double t_pair = std::min(eta_ff_v * t_sym_ff, eta_fb_v * t_sym_fb);
            s.dt_adapt[i] = std::min(s.dt_adapt[i], t_pair);
            s.dt_adapt[j] = std::min(s.dt_adapt[j], t_pair);

#ifdef ENABLE_COLLISIONAL
            // Jerk and acceleration accumulation (collisional only)
            if (s.collisional) {
                double inv_r3 = inv_r * inv_r * inv_r;
                double rdotv = dx*dvx + dy*dvy + dz*dvz;
                double fac1_j = G * s.mass[j] * inv_r3;
                double fac1_i = G * s.mass[i] * inv_r3;
                double fac2 = 3.0 * rdotv * inv_r * inv_r;
                jerk_x[i - start] += fac1_j * (dvx - fac2 * dx);
                jerk_y[i - start] += fac1_j * (dvy - fac2 * dy);
                jerk_z[i - start] += fac1_j * (dvz - fac2 * dz);
                jerk_x[j - start] -= fac1_i * (dvx - fac2 * dx);
                jerk_y[j - start] -= fac1_i * (dvy - fac2 * dy);
                jerk_z[j - start] -= fac1_i * (dvz - fac2 * dz);
                // Fresh subset acceleration for jerk criterion (BIFROST §3.4.5)
                acc_x[i - start] += fac1_j * dx;
                acc_y[i - start] += fac1_j * dy;
                acc_z[i - start] += fac1_j * dz;
                acc_x[j - start] -= fac1_i * dx;
                acc_y[j - start] -= fac1_i * dy;
                acc_z[j - start] -= fac1_i * dz;
            }

#endif
        }
    }

    // Jerk timestep criterion (collisional only)
    // Uses freshly accumulated subset acceleration, not stale p.ax/ay/az
    // (BIFROST §3.4.5: ε_jerk = η_j * |a_i| / |j_i|)
#ifdef ENABLE_COLLISIONAL
    if (s.collisional && eta_jerk_val > 0.0) {
        for (int i = start; i < end; i++) {
            int ii = i - start;
            double a_mag = std::sqrt(acc_x[ii]*acc_x[ii] + acc_y[ii]*acc_y[ii] + acc_z[ii]*acc_z[ii]);
            double j_mag = std::sqrt(jerk_x[ii]*jerk_x[ii] + jerk_y[ii]*jerk_y[ii] + jerk_z[ii]*jerk_z[ii]);
            if (j_mag > EPS && a_mag > EPS) {
                double dt_jerk = eta_jerk_val * a_mag / j_mag;
                s.dt_adapt[i] = std::min(s.dt_adapt[i], dt_jerk);
            }
        }
    }
#endif
}

// CPU partition by pivot (returns slow count, reorders data in-place)
static int cpu_partition_by_pivot(CpuSubset& s, int start, int count, double pivot) {
    if (count <= 1) return count;
    // Count slow particles
    int slow_count = 0;
    for (int i = start; i < start + count; i++) {
        if (s.dt_adapt[i] >= pivot) slow_count++;
    }
    if (slow_count == 0 || slow_count == count) return slow_count;

    // Build reordered copies
    std::vector<double> tx(count), ty(count), tz(count);
    std::vector<double> tvx(count), tvy(count), tvz(count);
    std::vector<double> tax(count), tay(count), taz(count);
    std::vector<double> tm(count), tdt(count);
    std::vector<int64_t> tix(count), tiy(count), tiz(count);
#ifdef ENABLE_COLLISIONAL
    std::vector<int> tsid(count);
    std::vector<double> tgm(count), tdl0(count);
    std::vector<int> tpid(count);
#endif
    int si = 0, fi = slow_count;
    for (int i = 0; i < count; i++) {
        int src = start + i;
        int dst = (s.dt_adapt[src] >= pivot) ? si++ : fi++;
        tx[dst] = s.x[src]; ty[dst] = s.y[src]; tz[dst] = s.z[src];
        tvx[dst] = s.vx[src]; tvy[dst] = s.vy[src]; tvz[dst] = s.vz[src];
        tax[dst] = s.ax[src]; tay[dst] = s.ay[src]; taz[dst] = s.az[src];
        tm[dst] = s.mass[src]; tdt[dst] = s.dt_adapt[src];
        tix[dst] = s.ix[src]; tiy[dst] = s.iy[src]; tiz[dst] = s.iz[src];
#ifdef ENABLE_COLLISIONAL
        tsid[dst] = s.sid[src];
        tgm[dst] = s.grad_mag[src]; tdl0[dst] = s.dt_level0[src];
        tpid[dst] = s.particle_id[src];
#endif
    }
    for (int i = 0; i < count; i++) {
        int dst = start + i;
        s.x[dst] = tx[i]; s.y[dst] = ty[i]; s.z[dst] = tz[i];
        s.vx[dst] = tvx[i]; s.vy[dst] = tvy[i]; s.vz[dst] = tvz[i];
        s.ax[dst] = tax[i]; s.ay[dst] = tay[i]; s.az[dst] = taz[i];
        s.mass[dst] = tm[i]; s.dt_adapt[dst] = tdt[i];
        s.ix[dst] = tix[i]; s.iy[dst] = tiy[i]; s.iz[dst] = tiz[i];
#ifdef ENABLE_COLLISIONAL
        s.sid[dst] = tsid[i];
        s.grad_mag[dst] = tgm[i]; s.dt_level0[dst] = tdl0[i];
        s.particle_id[dst] = tpid[i];
#endif
    }
    return slow_count;
}

// Sync CpuSubset state to device Particles for subsystem detection/integration
// cpu_start: offset within CpuSubset arrays; dev_start: offset within device Particles
#ifdef ENABLE_COLLISIONAL
static void cpu_sync_to_device(CpuSubset& s, Particles& p, int cpu_start, int dev_start, int count) {
    auto h_x = Kokkos::create_mirror_view(Kokkos::subview(p.x, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_y = Kokkos::create_mirror_view(Kokkos::subview(p.y, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_z = Kokkos::create_mirror_view(Kokkos::subview(p.z, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_vx = Kokkos::create_mirror_view(Kokkos::subview(p.vx, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_vy = Kokkos::create_mirror_view(Kokkos::subview(p.vy, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_vz = Kokkos::create_mirror_view(Kokkos::subview(p.vz, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_m  = Kokkos::create_mirror_view(Kokkos::subview(p.mass, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_dt = Kokkos::create_mirror_view(Kokkos::subview(p.dt_adapt, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_sid = Kokkos::create_mirror_view(Kokkos::subview(p.subsystem_id, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    for (int i = 0; i < count; i++) {
        h_x(i) = s.x[cpu_start + i]; h_y(i) = s.y[cpu_start + i]; h_z(i) = s.z[cpu_start + i];
        h_vx(i) = s.vx[cpu_start + i]; h_vy(i) = s.vy[cpu_start + i]; h_vz(i) = s.vz[cpu_start + i];
        h_m(i) = s.mass[cpu_start + i]; h_dt(i) = s.dt_adapt[cpu_start + i];
        h_sid(i) = s.sid[cpu_start + i];
    }
    Kokkos::deep_copy(Kokkos::subview(p.x, Kokkos::pair<int,int>(dev_start, dev_start+count)), h_x);
    Kokkos::deep_copy(Kokkos::subview(p.y, Kokkos::pair<int,int>(dev_start, dev_start+count)), h_y);
    Kokkos::deep_copy(Kokkos::subview(p.z, Kokkos::pair<int,int>(dev_start, dev_start+count)), h_z);
    Kokkos::deep_copy(Kokkos::subview(p.vx, Kokkos::pair<int,int>(dev_start, dev_start+count)), h_vx);
    Kokkos::deep_copy(Kokkos::subview(p.vy, Kokkos::pair<int,int>(dev_start, dev_start+count)), h_vy);
    Kokkos::deep_copy(Kokkos::subview(p.vz, Kokkos::pair<int,int>(dev_start, dev_start+count)), h_vz);
    Kokkos::deep_copy(Kokkos::subview(p.mass, Kokkos::pair<int,int>(dev_start, dev_start+count)), h_m);
    Kokkos::deep_copy(Kokkos::subview(p.dt_adapt, Kokkos::pair<int,int>(dev_start, dev_start+count)), h_dt);
    Kokkos::deep_copy(Kokkos::subview(p.subsystem_id, Kokkos::pair<int,int>(dev_start, dev_start+count)), h_sid);
    Kokkos::fence();
}

// Sync subsystem_id back from device to CpuSubset after detection
static void cpu_sync_sid_from_device(CpuSubset& s, Particles& p, int cpu_start, int dev_start, int count) {
    auto h_sid = Kokkos::create_mirror_view(Kokkos::subview(p.subsystem_id, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    Kokkos::deep_copy(h_sid, Kokkos::subview(p.subsystem_id, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    Kokkos::fence();
    for (int i = 0; i < count; i++) s.sid[cpu_start + i] = h_sid(i);
}

// Sync positions and velocities from device back to CpuSubset
// (after advance_subsystems_in_range modifies subsystem member positions on device)
static void cpu_sync_pos_vel_from_device(CpuSubset& s, Particles& p, int cpu_start, int dev_start, int count) {
    auto h_x  = Kokkos::create_mirror_view(Kokkos::subview(p.x, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_y  = Kokkos::create_mirror_view(Kokkos::subview(p.y, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_z  = Kokkos::create_mirror_view(Kokkos::subview(p.z, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_vx = Kokkos::create_mirror_view(Kokkos::subview(p.vx, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_vy = Kokkos::create_mirror_view(Kokkos::subview(p.vy, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_vz = Kokkos::create_mirror_view(Kokkos::subview(p.vz, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    Kokkos::deep_copy(h_x, Kokkos::subview(p.x, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    Kokkos::deep_copy(h_y, Kokkos::subview(p.y, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    Kokkos::deep_copy(h_z, Kokkos::subview(p.z, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    Kokkos::deep_copy(h_vx, Kokkos::subview(p.vx, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    Kokkos::deep_copy(h_vy, Kokkos::subview(p.vy, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    Kokkos::deep_copy(h_vz, Kokkos::subview(p.vz, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    Kokkos::fence();
    for (int i = 0; i < count; i++) {
        s.x[cpu_start + i] = h_x(i); s.y[cpu_start + i] = h_y(i); s.z[cpu_start + i] = h_z(i);
        s.vx[cpu_start + i] = h_vx(i); s.vy[cpu_start + i] = h_vy(i); s.vz[cpu_start + i] = h_vz(i);
    }
}
#endif

// CPU FSI step
static void cpu_fsi_step(CpuSubset& s, Particles& p, int start, int count,
                         double G, double soft2, double dt,
                         bool forces_valid = false) {
    if (!forces_valid) {
        cpu_self_acc(s, start, count, G, soft2);
    }
    cpu_kick(s, start, count, dt * FSI_KICK);
#ifdef ENABLE_COLLISIONAL
    cpu_drift_field_only(s, start, count, dt * FSI_DRIFT);
    // Advance subsystems (LogH/GBS/secular) for members in this range
    if (s.collisional && !g_subsystem_mgr.active.empty()) {
        int dev_start = s.device_start + start;
        cpu_sync_to_device(s, p, start, dev_start, count);
        advance_subsystems_in_range(p, dev_start, count, G, dt * FSI_DRIFT);
        cpu_sync_pos_vel_from_device(s, p, start, dev_start, count);
    }
#else
    cpu_drift(s, start, count, dt * FSI_DRIFT);
#endif
    cpu_self_acc(s, start, count, G, soft2);
    cpu_self_grad(s, start, count, G, soft2, dt);
    cpu_kick(s, start, count, dt * FSI_MOD_KICK);
#ifdef ENABLE_COLLISIONAL
    cpu_drift_field_only(s, start, count, dt * FSI_DRIFT);
    // Advance subsystems (LogH/GBS/secular) for members in this range
    if (s.collisional && !g_subsystem_mgr.active.empty()) {
        int dev_start = s.device_start + start;
        cpu_sync_to_device(s, p, start, dev_start, count);
        advance_subsystems_in_range(p, dev_start, count, G, dt * FSI_DRIFT);
        cpu_sync_pos_vel_from_device(s, p, start, dev_start, count);
    }
#else
    cpu_drift(s, start, count, dt * FSI_DRIFT);
#endif
    cpu_self_acc(s, start, count, G, soft2);
    cpu_kick(s, start, count, dt * FSI_KICK);
}

// CPU HHS-FSI recursion
static void cpu_hhs_recursive(CpuSubset& s, Particles& p, int start, int count,
                              double pivot_dt, double G, double soft2,
                              double dt_min, double eta, bool dt_valid, int depth = 0) {
    if (count == 0) return;

    if (pivot_dt * 0.5 < dt_min) {
        cpu_fsi_step(s, p, start, count, G, soft2, pivot_dt);
        return;
    }

    // Global device offset for this subset
    int dev_start = s.device_start + start;

    // BIFROST Section 3.5: fresh neighbor search + detection, then timestep
    if (!dt_valid && count > 1) {
#ifdef ENABLE_COLLISIONAL
        // Step 1: Neighbor search + detection (sync to device, run GPU kernel, sync back)
        if (s.collisional) {
            cpu_sync_to_device(s, p, start, dev_start, count);
            g_subsystem_mgr.reset_ngb_count();
            neighbor_search_subset(p, dev_start, count, G,
                                   g_collisional.r_ngb, g_collisional.r_ngb_max, pivot_dt);
            g_subsystem_mgr.detect_from_pairs(p, G, pivot_dt,
                                               g_collisional.r_ngb, g_collisional.r_ngb_max,
                                               g_collisional.n_orb_secular);
            g_subsystem_mgr.update_particle_flags(p);
            g_subsystem_mgr.accumulate_diagnostics();
            cpu_sync_sid_from_device(s, p, start, dev_start, count);
        }
#endif
        // Step 2: CPU timestep (with subsystem exclusion via s.sid)
        cpu_compute_timestep(s, start, count, G, soft2, eta);
#ifdef ENABLE_COLLISIONAL
        // Gradient timestep criteria (BIFROST Section 3.4.4)
        if (s.collisional && g_collisional.eta_grad > 0.0) {
            double eta_g = g_collisional.eta_grad;
            for (int i = start; i < start + count; i++) {
                double g = s.grad_mag[i];
                if (g > EPS) {
                    double a_mag = std::sqrt(s.ax[i]*s.ax[i] + s.ay[i]*s.ay[i] + s.az[i]*s.az[i]);
                    double v_mag = std::sqrt(s.vx[i]*s.vx[i] + s.vy[i]*s.vy[i] + s.vz[i]*s.vz[i]);
                    if (a_mag > EPS) {
                        double dt_g2 = eta_g * std::sqrt(a_mag / g);
                        if (dt_g2 < s.dt_adapt[i]) s.dt_adapt[i] = dt_g2;
                    }
                    if (v_mag > EPS) {
                        double dt_g1 = eta_g * std::cbrt(v_mag / g);
                        if (dt_g1 < s.dt_adapt[i]) s.dt_adapt[i] = dt_g1;
                    }
                }
            }
        }
        // Level-0 clamping (BIFROST Section 3.4.1)
        if (s.collisional && depth > 0) {
            for (int i = start; i < start + count; i++)
                if (s.dt_adapt[i] > s.dt_level0[i]) s.dt_adapt[i] = s.dt_level0[i];
        }

        // Step 3: Subsystem timestep adjustments (sync dt to device, run, sync back)
        if (s.collisional) {
            // Sync dt_adapt to device for enforce/perturbation
            {
                auto h_dt = Kokkos::create_mirror_view(Kokkos::subview(p.dt_adapt, Kokkos::pair<int,int>(dev_start, dev_start+count)));
                for (int i = 0; i < count; i++) h_dt(i) = s.dt_adapt[start + i];
                Kokkos::deep_copy(Kokkos::subview(p.dt_adapt, Kokkos::pair<int,int>(dev_start, dev_start+count)), h_dt);
            }
            g_subsystem_mgr.enforce_shared_timestep(p);
            g_subsystem_mgr.apply_perturbation_timestep(p, G, g_collisional.eta_pert);

            // Merger detection (BIFROST §3.14)
            if (g_collisional.c_code > 0.0) {
                auto mergers = detect_mergers(g_subsystem_mgr, p, G,
                                               g_collisional.c_code, pivot_dt);
                for (const auto& ev : mergers) {
                    execute_merger(ev, p, g_subsystem_mgr, g_energy_books,
                                   G, g_collisional.c_code);
                }
                if (!mergers.empty()) {
                    int n_removed = p.compact_remove_zero_mass();
                    count -= n_removed;
                    if (count <= 0) return;
                }
            }

            // Sync dt_adapt back from device
            {
                auto h_dt = Kokkos::create_mirror_view(Kokkos::subview(p.dt_adapt, Kokkos::pair<int,int>(dev_start, dev_start+count)));
                Kokkos::deep_copy(h_dt, Kokkos::subview(p.dt_adapt, Kokkos::pair<int,int>(dev_start, dev_start+count)));
                Kokkos::fence();
                for (int i = 0; i < count; i++) s.dt_adapt[start + i] = h_dt(i);
            }

            // Level-0 store: capture dt AFTER subsystem adjustments
            if (depth == 0) {
                for (int i = start; i < start + count; i++)
                    s.dt_level0[i] = s.dt_adapt[i];
            }
        }
#endif
    }

    int slow_count = cpu_partition_by_pivot(s, start, count, pivot_dt);
    int slow_start = start;
    int fast_start = start + slow_count;
    int fast_count = count - slow_count;


    if (fast_count == 0) {
        cpu_fsi_step(s, p, start, count, G, soft2, pivot_dt);
        return;
    }

    // Partition was a no-op when slow_count==0, so dt_adapt still valid
    if (slow_count == 0) {
        double sub_pivot = pivot_dt * 0.5;
        cpu_hhs_recursive(s, p, start, count, sub_pivot, G, soft2, dt_min, eta, true, depth + 1);
        cpu_hhs_recursive(s, p, start, count, sub_pivot, G, soft2, dt_min, eta, false, depth + 1);
        return;
    }

    // Step 1: Cross-kick (dt/6)
    cpu_cross_acc(s, slow_start, slow_count, fast_start, fast_count, G, soft2);
    cpu_kick(s, slow_start, slow_count, pivot_dt * FSI_KICK);
    cpu_kick(s, fast_start, fast_count, pivot_dt * FSI_KICK);

    // Step 2: FSI half-step on slow
    cpu_fsi_step(s, p, slow_start, slow_count, G, soft2, pivot_dt * 0.5);

    // Step 3: Recurse on fast (first half)
    cpu_hhs_recursive(s, p, fast_start, fast_count, pivot_dt * 0.5, G, soft2, dt_min, eta, false, depth + 1);

    // Step 4: Cross-gradient kick (2dt/3)
    cpu_cross_acc(s, slow_start, slow_count, fast_start, fast_count, G, soft2);
    cpu_cross_grad(s, slow_start, slow_count, fast_start, fast_count, G, soft2, pivot_dt);
    cpu_kick(s, slow_start, slow_count, pivot_dt * FSI_MOD_KICK);
    cpu_kick(s, fast_start, fast_count, pivot_dt * FSI_MOD_KICK);

    // Step 5: Recurse on fast (second half)
    cpu_hhs_recursive(s, p, fast_start, fast_count, pivot_dt * 0.5, G, soft2, dt_min, eta, false, depth + 1);

    // Step 6: FSI half-step on slow
    cpu_fsi_step(s, p, slow_start, slow_count, G, soft2, pivot_dt * 0.5);

    // Step 7: Cross-kick (dt/6)
    cpu_cross_acc(s, slow_start, slow_count, fast_start, fast_count, G, soft2);
    cpu_kick(s, slow_start, slow_count, pivot_dt * FSI_KICK);
    cpu_kick(s, fast_start, fast_count, pivot_dt * FSI_KICK);
}

// Entry point: copy subset to host, run CPU HHS, copy back
static void hhs_fsi_recursive_cpu(Particles& p, int start, int count,
                                  double pivot_dt, double G, double soft2,
                                  double dt_min, double eta, bool dt_valid) {
    g_subset_hist.record(SubsetHistogram::LOCAL_CPU, count);
    CpuSubset s;
    s.resize(count);
    s.copy_from_device(p, start);
    cpu_hhs_recursive(s, p, 0, count, pivot_dt, G, soft2, dt_min, eta, dt_valid);
    s.copy_to_device(p, start);
}

// =============================================================================
// Local (non-distributed) HHS-FSI recursion for small subsets
// Same algorithm as the distributed version, but all ranks compute full N²
// locally without MPI allgather. Uses non-caching cross-kicks.
// =============================================================================
static void hhs_fsi_recursive_local(
    Particles& p,
    int start, int count,
    double pivot_dt,
    double G, double soft2,
    double dt_min, double eta,
    bool dt_valid,
    int local_depth = 0) {

    if (count == 0) return;
    ScopedDepthTimer _depth_timer(local_depth);
    dbg_hhs_local++;
    if (local_depth < DBG_MAX_DEPTH) {
        dbg_local_depth_calls[local_depth]++;
        dbg_local_depth_count[local_depth] += count;
    }
    g_subset_hist.record(SubsetHistogram::LOCAL_GPU, count);


    // CPU serial path for very small subsets
    if (serial_n2_cpu > 0 && count <= serial_n2_cpu) {
        dbg_hhs_cpu++;
        hhs_fsi_recursive_cpu(p, start, count, pivot_dt, G, soft2, dt_min, eta, dt_valid);
        return;
    }

    // Base case: at minimum timestep, use FSI on the subset (force N² — no tree in serial path)
    if (pivot_dt * 0.5 < dt_min) {
        fsi_step(p, start, count, G, soft2, pivot_dt, false, /*force_direct_n2=*/true);
        return;
    }

    // BIFROST Section 3.5: fresh neighbor search + subsystem detection, then timestep
    if (!dt_valid && count > 1) {
#ifdef ENABLE_COLLISIONAL
        // Step 1: Standalone neighbor search (no subsystem exclusion)
        if (p.collisional) {
            g_profiler.start();
            g_subsystem_mgr.reset_ngb_count();
            neighbor_search_subset(p, start, count, G,
                                   g_collisional.r_ngb, g_collisional.r_ngb_max, pivot_dt);
            g_subsystem_mgr.detect_from_pairs(p, G, pivot_dt,
                                               g_collisional.r_ngb, g_collisional.r_ngb_max,
                                               g_collisional.n_orb_secular);
            g_subsystem_mgr.update_particle_flags(p);
            g_subsystem_mgr.accumulate_diagnostics();
            g_profiler.stop(StepProfiler::SUBSYSTEM_DETECT);
        }
#endif
        // Step 2: Timestep computation (with subsystem exclusion active)
        g_profiler.start();
        compute_timestep_subset(p, start, count, G, soft2, eta, true);
#ifdef ENABLE_COLLISIONAL
        // Gradient timestep criteria (BIFROST Section 3.4.4)
        if (p.collisional && g_collisional.eta_grad > 0.0) {
            auto dt_f = p.dt_adapt;
            auto gm = p.grad_mag;
            RawView agx{p.ax.data()}, agy{p.ay.data()}, agz{p.az.data()};
            RawView vvx{p.vx.data()}, vvy{p.vy.data()}, vvz{p.vz.data()};
            double eta_g = g_collisional.eta_grad;
            Kokkos::parallel_for("dt_grad_local", count, KOKKOS_LAMBDA(int ii) {
                int i = start + ii;
                double g = gm(i);
                if (g > EPS) {
                    double a_mag = Kokkos::sqrt(agx(i)*agx(i) + agy(i)*agy(i) + agz(i)*agz(i));
                    double v_mag = Kokkos::sqrt(vvx(i)*vvx(i) + vvy(i)*vvy(i) + vvz(i)*vvz(i));
                    if (a_mag > EPS) {
                        double dt_g2 = eta_g * Kokkos::sqrt(a_mag / g);
                        if (dt_g2 < dt_f(i)) dt_f(i) = dt_g2;
                    }
                    if (v_mag > EPS) {
                        double dt_g1 = eta_g * Kokkos::cbrt(v_mag / g);
                        if (dt_g1 < dt_f(i)) dt_f(i) = dt_g1;
                    }
                }
            });
        }
        // Level-0 clamping (local path is always at depth > 0)
        if (p.collisional) {
            auto dt_f = p.dt_adapt;
            auto dt_l0 = p.dt_level0;
            Kokkos::parallel_for("clamp_dt_level0_local", count, KOKKOS_LAMBDA(int ii) {
                int i = start + ii;
                if (dt_f(i) > dt_l0(i)) dt_f(i) = dt_l0(i);
            });
        }
#endif
        g_profiler.stop(StepProfiler::ADAPTIVE_DT);
#ifdef ENABLE_COLLISIONAL
        // Step 3: Subsystem timestep adjustments
        if (p.collisional) {
            g_subsystem_mgr.enforce_shared_timestep(p);
            g_subsystem_mgr.apply_perturbation_timestep(p, G, g_collisional.eta_pert);

            // Merger detection (BIFROST §3.14)
            if (g_collisional.c_code > 0.0) {
                auto mergers = detect_mergers(g_subsystem_mgr, p, G,
                                               g_collisional.c_code, pivot_dt);
                for (const auto& ev : mergers) {
                    execute_merger(ev, p, g_subsystem_mgr, g_energy_books,
                                   G, g_collisional.c_code);
                }
                if (!mergers.empty()) {
                    int n_removed = p.compact_remove_zero_mass();
                    count -= n_removed;
                    if (count <= 0) return;
                    g_subsystem_mgr.update_particle_flags(p);
                }
            }
        }
#endif
    }

    // Partition into slow (dt >= pivot) and fast (dt < pivot)
    g_profiler.start();
    int slow_count = partition_by_pivot(p, start, count, pivot_dt);
    g_profiler.stop(StepProfiler::PARTITION);
    int slow_start = start;
    int fast_start = start + slow_count;
    int fast_count = count - slow_count;

    // Base case: all particles are slow
    if (fast_count == 0) {
        fsi_step(p, start, count, G, soft2, pivot_dt, false, /*force_direct_n2=*/true);
        return;
    }

    // Base case: all particles are fast
    // Partition was a no-op (line 754 early return), so dt_adapt values are still valid.
    // First call reuses parent timesteps; second call recomputes (particles evolved).
    if (slow_count == 0) {
        dbg_hhs_slow0_local++;
        double sub_pivot = pivot_dt * 0.5;
        hhs_fsi_recursive_local(p, start, count, sub_pivot,
                                G, soft2, dt_min, eta, true, local_depth + 1);
        hhs_fsi_recursive_local(p, start, count, sub_pivot,
                                G, soft2, dt_min, eta, false, local_depth + 1);
        return;
    }

    // Step 1: Cross-kick S<->F (dt/6)
    g_profiler.start();
    cross_acceleration(p, slow_start, slow_count, fast_start, fast_count, G, soft2);
    g_profiler.stop(StepProfiler::CROSS_FORCE);
    g_profiler.start();
    kick_velocities(p, slow_start, slow_count, pivot_dt * FSI_KICK);
    kick_velocities(p, fast_start, fast_count, pivot_dt * FSI_KICK);
    g_profiler.stop(StepProfiler::DRIFT_KICK);

    // Step 2: FSI half-step on slow (force N² — no tree in serial path)
    fsi_step(p, slow_start, slow_count, G, soft2, pivot_dt * 0.5, false, /*force_direct_n2=*/true);

    // Step 3: Recurse on fast (first half)
    hhs_fsi_recursive_local(p, fast_start, fast_count, pivot_dt * 0.5,
                            G, soft2, dt_min, eta, false, local_depth + 1);

    // Step 4: Cross-gradient kick (2dt/3)
    g_profiler.start();
    cross_acceleration(p, slow_start, slow_count, fast_start, fast_count, G, soft2);
    g_profiler.stop(StepProfiler::CROSS_FORCE);
    g_profiler.start();
    cross_gradient_correction(p, slow_start, slow_count, fast_start, fast_count, G, soft2, pivot_dt);
    g_profiler.stop(StepProfiler::CROSS_GRAD);
    g_profiler.start();
    kick_velocities(p, slow_start, slow_count, pivot_dt * FSI_MOD_KICK);
    kick_velocities(p, fast_start, fast_count, pivot_dt * FSI_MOD_KICK);
    g_profiler.stop(StepProfiler::DRIFT_KICK);

    // Step 5: Recurse on fast (second half)
    hhs_fsi_recursive_local(p, fast_start, fast_count, pivot_dt * 0.5,
                            G, soft2, dt_min, eta, false, local_depth + 1);

    // Step 6: FSI half-step on slow (force N² — no tree in serial path)
    fsi_step(p, slow_start, slow_count, G, soft2, pivot_dt * 0.5, false, /*force_direct_n2=*/true);

    // Step 7: Cross-kick S<->F (dt/6)
    g_profiler.start();
    cross_acceleration(p, slow_start, slow_count, fast_start, fast_count, G, soft2);
    g_profiler.stop(StepProfiler::CROSS_FORCE);
    g_profiler.start();
    kick_velocities(p, slow_start, slow_count, pivot_dt * FSI_KICK);
    kick_velocities(p, fast_start, fast_count, pivot_dt * FSI_KICK);
    g_profiler.stop(StepProfiler::DRIFT_KICK);
}

// =============================================================================
// Tree-based cross-force using global tree with dual monopoles
// One global tree (g_octree) built at top of step; dual monopoles recomputed
// at each HHS level after partition. No separate tree builds needed.
// =============================================================================

// Cross-force via global tree: both directions (slow←fast, fast←slow)
static void cross_acceleration_global_tree(Particles& p,
                                            int slow_start, int slow_count,
                                            int fast_start, int fast_count,
                                            double G, double soft2) {
    if (slow_count <= 0 || fast_count <= 0) return;

    // slow ← fast (source_flag=1 means fast sources, target=slow)
    tree_global_cross_acceleration(p, g_octree, 1, G, soft2);
    // fast ← slow (source_flag=0 means slow sources, target=fast)
    tree_global_cross_acceleration(p, g_octree, 0, G, soft2);

    // Momentum correction on combined range
    tree_momentum_correction(p, slow_start, slow_count + fast_count);
}

// Cross-force + gradient via global tree
static void cross_acceleration_and_gradient_global_tree(Particles& p,
                                                         int slow_start, int slow_count,
                                                         int fast_start, int fast_count,
                                                         double G, double soft2, double dt) {
    if (slow_count <= 0 || fast_count <= 0) return;

    // Cross-forces (overwrites ax/ay/az for both subsets)
    tree_global_cross_acceleration(p, g_octree, 1, G, soft2);
    tree_global_cross_acceleration(p, g_octree, 0, G, soft2);
    tree_momentum_correction(p, slow_start, slow_count + fast_count);

    // Compute dual cell accelerations for gradient correction
    g_octree.compute_dual_cell_accelerations(p);

    // Gradient correction: slow from fast, fast from slow
    tree_global_cross_gradient_correction(p, g_octree, 1, G, soft2, dt);
    tree_global_cross_gradient_correction(p, g_octree, 0, G, soft2, dt);
    tree_momentum_correction(p, slow_start, slow_count + fast_count);
}

// =============================================================================
// Recursive distributed HHS-FSI implementation (Algorithm 4 from FROST paper)
// Scatter-compute-gather paradigm with one global tree.
//
// All ranks run IDENTICAL code for partitioning, drifts, kicks.
// Only force computation is distributed, with Allgather after each.
// When global_tree_available=true, self-forces and timesteps use the global
// tree with subset monopoles, eliminating per-level tree rebuilds.
// =============================================================================
static void hhs_fsi_recursive_distributed(
    DistributedParticles& dp,
    int start, int count,           // Subset of particles to integrate
    double pivot_dt,                // Current pivot timestep
    double G, double soft2,
    double dt_min, double eta,
    bool dt_valid,
    bool global_tree_available = false,
    int depth = 0) {

    if (count == 0) return;
    dbg_hhs_distributed++;
    if (depth > dbg_hhs_max_depth) dbg_hhs_max_depth = depth;
    if (depth < DBG_MAX_DEPTH) {
        dbg_dist_depth_calls[depth]++;
        dbg_dist_depth_count[depth] += count;
    }
    g_subset_hist.record(SubsetHistogram::DISTRIBUTED, count);


    // Below threshold: skip MPI, all ranks compute full N² locally
    if (serial_n2_gpu > 0 && count <= serial_n2_gpu) {
        hhs_fsi_recursive_local(dp.local, start, count, pivot_dt,
                                G, soft2, dt_min, eta, dt_valid);
        // Re-sync packed views: local path modifies SoA directly without updating pos_packed
        dp.local.pack_pos_range(start, count);
        return;
    }

    // Whether this subset is large enough for tree operations
    bool tree_eligible = use_tree && count >= tree_threshold;

    // Base case: at minimum timestep, use FSI on the subset
    if (pivot_dt * 0.5 < dt_min) {
#ifdef MPI_TOPOLOGY_RING
        fsi_step_ring(dp, start, count, G, soft2, pivot_dt);
#else
        if (global_tree_available && tree_eligible) {
            // Mark active range and use global tree for self-force
            g_profiler.start();
            g_octree.update_subsets(dp.local, start, count);
            g_profiler.stop(StepProfiler::TREE_MAINT);
            fsi_step_distributed(dp, start, count, G, soft2, pivot_dt,
                                 false, false, 0);
        } else {
            fsi_step_distributed(dp, start, count, G, soft2, pivot_dt);
        }
#endif
        return;
    }

    // BIFROST Section 3.5: fresh neighbor search + detection, then timestep
    bool forces_valid = false;
    bool tree_built_for_subset = false;
    if (!dt_valid && count > 1) {
#ifdef ENABLE_COLLISIONAL
        // Step 1: Standalone neighbor search + detection (before timestep)
        if (dp.local.collisional) {
            g_profiler.start();
            g_subsystem_mgr.reset_ngb_count();
            neighbor_search_subset(dp.local, start, count, G,
                                   g_collisional.r_ngb, g_collisional.r_ngb_max, pivot_dt);
            g_subsystem_mgr.detect_from_pairs(dp.local, G, pivot_dt,
                                               g_collisional.r_ngb, g_collisional.r_ngb_max,
                                               g_collisional.n_orb_secular);
            g_subsystem_mgr.update_particle_flags(dp.local);
            g_subsystem_mgr.accumulate_diagnostics();
            if (depth < DBG_MAX_DEPTH)
                dbg_dist_depth_ngb_pairs[depth] += g_subsystem_mgr.last_level_pairs;
            g_profiler.stop(StepProfiler::SUBSYSTEM_DETECT);
        }
#endif
        // Step 2: Timestep computation (with subsystem exclusion active)
        g_profiler.start();
        if (global_tree_available && tree_eligible) {
            g_octree.update_subsets(dp.local, start, count);
            tree_global_self_acceleration_and_timestep(dp.local, g_octree, 0, G, soft2, eta);
            tree_momentum_correction(dp.local, start, count);
            forces_valid = true;
        } else if (!global_tree_available && tree_eligible) {
            g_octree.n_leaf = tree_n_leaf;
            g_octree.n_group = tree_n_group;
            g_octree.build(dp.local, start, count, tree_theta);
            int my_g_off, my_g_cnt;
            get_tree_group_range(g_octree, my_g_off, my_g_cnt);
            zero_accelerations(dp.local, start, count);
            init_timesteps_max(dp.local, start, count);
            tree_self_acceleration_and_timestep(dp.local, g_octree, start, count, G, soft2, eta, my_g_off, my_g_cnt);
            dp.allreduce_accelerations_subset(start, count);
            dp.allreduce_timestep_subset(start, count);
            tree_momentum_correction(dp.local, start, count);
            forces_valid = true;
            tree_built_for_subset = true;
        } else {
            dp.local.pack_pos_range(start, count);
            dp.local.pack_vel_range(start, count);
#ifdef MPI_TOPOLOGY_RING
                compute_timestep_ring(dp, start, count, G, soft2, eta, true);
#else
                compute_timestep_distributed(dp, start, count, G, soft2, eta, true);
#endif
        }
#ifdef ENABLE_COLLISIONAL
        // Gradient timestep criterion (BIFROST Section 3.4.4)
        if (dp.local.collisional && g_collisional.eta_grad > 0.0) {
            auto dt_f = dp.local.dt_adapt;
            auto gm = dp.local.grad_mag;
            RawView agx{dp.local.ax.data()}, agy{dp.local.ay.data()}, agz{dp.local.az.data()};
            RawView vvx{dp.local.vx.data()}, vvy{dp.local.vy.data()}, vvz{dp.local.vz.data()};
            double eta_g = g_collisional.eta_grad;
            Kokkos::parallel_for("dt_grad_dist", count, KOKKOS_LAMBDA(int ii) {
                int i = start + ii;
                double g = gm(i);
                if (g > EPS) {
                    double a_mag = Kokkos::sqrt(agx(i)*agx(i) + agy(i)*agy(i) + agz(i)*agz(i));
                    double v_mag = Kokkos::sqrt(vvx(i)*vvx(i) + vvy(i)*vvy(i) + vvz(i)*vvz(i));
                    if (a_mag > EPS) {
                        double dt_g2 = eta_g * Kokkos::sqrt(a_mag / g);
                        if (dt_g2 < dt_f(i)) dt_f(i) = dt_g2;
                    }
                    if (v_mag > EPS) {
                        double dt_g1 = eta_g * Kokkos::cbrt(v_mag / g);
                        if (dt_g1 < dt_f(i)) dt_f(i) = dt_g1;
                    }
                }
            });
        }
        // Level-0 clamping (BIFROST Section 3.4.1)
        if (dp.local.collisional && depth > 0) {
            auto dt_f = dp.local.dt_adapt;
            auto dt_l0 = dp.local.dt_level0;
            int n_clamped = 0;
            Kokkos::parallel_reduce("clamp_dt_level0_dist", count,
                KOKKOS_LAMBDA(int ii, int& cnt) {
                    int i = start + ii;
                    if (dt_f(i) > dt_l0(i)) {
                        dt_f(i) = dt_l0(i);
                        cnt++;
                    }
                }, n_clamped);
            if (depth < DBG_MAX_DEPTH) dbg_dist_depth_l0_clamped[depth] += n_clamped;
        }
#endif
        g_profiler.stop(StepProfiler::ADAPTIVE_DT, (long long)count * count);
#ifdef ENABLE_COLLISIONAL
        // Step 3: Subsystem timestep adjustments
        if (dp.local.collisional) {
            g_subsystem_mgr.enforce_shared_timestep(dp.local);
            g_subsystem_mgr.apply_perturbation_timestep(dp.local, G, g_collisional.eta_pert);

            // Level-0 store: capture dt AFTER subsystem adjustments
            if (depth == 0) {
                auto dt_f = dp.local.dt_adapt;
                auto dt_l0 = dp.local.dt_level0;
                Kokkos::parallel_for("copy_dt_level0_dist", count, KOKKOS_LAMBDA(int ii) {
                    int i = start + ii;
                    dt_l0(i) = dt_f(i);
                });
            }

            // Merger detection at every HHS level (BIFROST §3.14)
            if (g_collisional.c_code > 0.0) {
                auto mergers = detect_mergers(g_subsystem_mgr, dp.local, G,
                                               g_collisional.c_code, pivot_dt);
                for (const auto& ev : mergers) {
                    execute_merger(ev, dp.local, g_subsystem_mgr, g_energy_books,
                                   G, g_collisional.c_code);
                }
                if (!mergers.empty()) {
                    int n_removed = dp.local.compact_remove_zero_mass();
                    count -= n_removed;
                    if (count <= 0) return;
                    g_subsystem_mgr.update_particle_flags(dp.local);
                }
            }
        }
#endif
    }

    // Partition into slow (dt >= pivot) and fast (dt < pivot)
    // All ranks do identical partitioning (deterministic)
    g_profiler.start();
    int slow_count = partition_by_pivot(dp.local, start, count, pivot_dt);
    g_profiler.stop(StepProfiler::PARTITION);
    int slow_start = start;
    int fast_start = start + slow_count;
    int fast_count = count - slow_count;


    // Global tree available for sub-calls (either built here or inherited)
    bool tree_for_subs = global_tree_available || tree_built_for_subset;

    // Base case: all particles are slow (no fast particles)
    // Reuse fused forces for first FSI kick (saves one N² pass)
    if (fast_count == 0) {
#ifdef MPI_TOPOLOGY_RING
        fsi_step_ring(dp, start, count, G, soft2, pivot_dt, forces_valid);
#else
        if (global_tree_available && tree_eligible) {
            // Flags set by timestep's update_subsets — [start, count) is flag=0
            fsi_step_distributed(dp, start, count, G, soft2, pivot_dt,
                                 forces_valid, false, 0);
        } else {
            // tree_built_for_subset: tree is valid for subset, use old tree_prebuilt path
            fsi_step_distributed(dp, start, count, G, soft2, pivot_dt,
                                 forces_valid, tree_built_for_subset);
        }
#endif
        return;
    }

    // Base case: all particles are fast (no slow particles)
    // Partition was a no-op (line 754 early return), so dt_adapt values are still valid.
    // First call reuses parent timesteps; second call recomputes (particles evolved).
    if (slow_count == 0) {
        dbg_hhs_slow0_dist++;
        double sub_pivot = pivot_dt * 0.5;
        hhs_fsi_recursive_distributed(dp, start, count, sub_pivot,
                                      G, soft2, dt_min, eta, true,
                                      tree_for_subs, depth + 1);
        hhs_fsi_recursive_distributed(dp, start, count, sub_pivot,
                                      G, soft2, dt_min, eta, false,
                                      tree_for_subs, depth + 1);
        return;
    }

    long long cross_pairs = 2LL * slow_count * fast_count;

    // Re-sync packed positions after partition reorder
    dp.local.pack_pos_range(start, count);

    // ==========================================================================
    // ALGORITHM 4: HHS-FSI with slow (S) and fast (F) particles
    // ==========================================================================

#ifdef MPI_TOPOLOGY_RING
    // --- Ring topology: j-partitioned force → allreduce → separate kick ---

    // Step 1: Cross-kick S<->F (dt/6)
    g_profiler.start();
    cross_acceleration_ring(dp, slow_start, slow_count, fast_start, fast_count, G, soft2);
    g_profiler.stop(StepProfiler::CROSS_FORCE, cross_pairs);
    g_profiler.start();
    kick_velocities(dp.local, slow_start, slow_count, pivot_dt * FSI_KICK);
    kick_velocities(dp.local, fast_start, fast_count, pivot_dt * FSI_KICK);
    g_profiler.stop(StepProfiler::DRIFT_KICK);

    // Step 2: FSI half-step on slow
    fsi_step_ring(dp, slow_start, slow_count, G, soft2, pivot_dt * 0.5);

    // Step 3: Recurse on fast (first half)
    hhs_fsi_recursive_distributed(dp, fast_start, fast_count, pivot_dt * 0.5,
                                  G, soft2, dt_min, eta, false,
                                  false, depth + 1);

    // Step 4: Cross-force + gradient (ring) + kick (2dt/3)
    g_profiler.start();
    cross_acceleration_and_gradient_ring(dp, slow_start, slow_count,
                                          fast_start, fast_count,
                                          G, soft2, pivot_dt);
    g_profiler.stop(StepProfiler::CROSS_GRAD, cross_pairs * 2);
    g_profiler.start();
    kick_velocities(dp.local, slow_start, slow_count, pivot_dt * FSI_MOD_KICK);
    kick_velocities(dp.local, fast_start, fast_count, pivot_dt * FSI_MOD_KICK);
    g_profiler.stop(StepProfiler::DRIFT_KICK);

    // Step 5: Recurse on fast (second half)
    {
        double sub_pivot = pivot_dt * 0.5;
        hhs_fsi_recursive_distributed(dp, fast_start, fast_count, sub_pivot,
                                      G, soft2, dt_min, eta, false,
                                      false, depth + 1);
    }

    // Step 6: FSI half-step on slow
    fsi_step_ring(dp, slow_start, slow_count, G, soft2, pivot_dt * 0.5);

    // Step 7: Cross-kick S<->F (dt/6)
    g_profiler.start();
    cross_acceleration_ring(dp, slow_start, slow_count, fast_start, fast_count, G, soft2);
    g_profiler.stop(StepProfiler::CROSS_FORCE, cross_pairs);
    g_profiler.start();
    kick_velocities(dp.local, slow_start, slow_count, pivot_dt * FSI_KICK);
    kick_velocities(dp.local, fast_start, fast_count, pivot_dt * FSI_KICK);
    g_profiler.stop(StepProfiler::DRIFT_KICK);

#else
    // --- Default: i-partitioned force → allgatherv → separate kick ---

    // use_tree_cross: use global tree for cross-forces via dual monopoles.
    // With 3-flag system (0=slow, 1=fast, 2=other), this works at ALL recursion depths.
    // Only worth it above tree_threshold — below that, direct N² is faster than tree overhead.
    bool use_tree_cross = tree_for_subs && use_tree && count >= tree_threshold;

    // use_tree_self: use global tree for self-forces on slow subset
    bool use_tree_self = tree_for_subs && use_tree && slow_count >= tree_threshold;

    // Mark slow/fast/other in global tree (3-flag for cross, 2-flag fallback otherwise)
    if (use_tree_cross || use_tree_self) {
        g_profiler.start();
        if (use_tree_cross) {
            g_octree.update_subsets_3flag(dp.local, start, count, slow_count);
        } else {
            g_octree.update_subsets(dp.local, slow_start, slow_count);
        }
        g_profiler.stop(StepProfiler::TREE_MAINT);
    }

    // Step 1: Cross-force S<->F then kick (dt/6)
#ifdef USE_MPI
#endif
    g_profiler.start();
    if (use_tree_cross) {
        cross_acceleration_global_tree(dp.local, slow_start, slow_count,
                                        fast_start, fast_count, G, soft2);
    } else {
        dp.local.pack_pos_range(slow_start, slow_count + fast_count);
        cross_acceleration_distributed(dp, slow_start, slow_count, fast_start, fast_count, G, soft2);
    }
    g_profiler.stop(StepProfiler::CROSS_FORCE, cross_pairs);
    g_profiler.start();
    kick_velocities(dp.local, slow_start, slow_count, pivot_dt * FSI_KICK);
    kick_velocities(dp.local, fast_start, fast_count, pivot_dt * FSI_KICK);
    g_profiler.stop(StepProfiler::DRIFT_KICK);

    // Step 2: FSI half-step on slow
#ifdef USE_MPI
#endif
    // flag=0 = slow (from update_subsets above), so global_tree_subset=0 is correct
    if (use_tree_self || use_tree_cross) {
        fsi_step_distributed(dp, slow_start, slow_count, G, soft2, pivot_dt * 0.5,
                             false, false, 0);
    } else {
        fsi_step_distributed(dp, slow_start, slow_count, G, soft2, pivot_dt * 0.5);
    }

    // Step 3: Recurse on fast (first half) — always compute fresh timesteps
#ifdef USE_MPI
#endif
    hhs_fsi_recursive_distributed(dp, fast_start, fast_count, pivot_dt * 0.5,
                                  G, soft2, dt_min, eta, false,
                                  tree_for_subs, depth + 1);

    // Restore tree state after steps 2-3 (positions and flags changed by drifts/recursion)
    if (use_tree_cross || use_tree_self) {
        g_profiler.start();
        if (use_tree_cross) {
            g_octree.update_subsets_3flag(dp.local, start, count, slow_count);
        } else {
            g_octree.update_subsets(dp.local, slow_start, slow_count);
        }
        g_profiler.stop(StepProfiler::TREE_MAINT);
    }

    // Step 4: Cross-kick S<->F with gradient (2dt/3)
#ifdef USE_MPI
#endif
    g_profiler.start();
    if (use_tree_cross) {
        cross_acceleration_and_gradient_global_tree(dp.local, slow_start, slow_count,
                                                     fast_start, fast_count, G, soft2, pivot_dt);
    } else {
        cross_acceleration_and_gradient_distributed(dp, slow_start, slow_count,
                                                    fast_start, fast_count,
                                                    G, soft2, pivot_dt);
    }
    g_profiler.stop(StepProfiler::CROSS_GRAD, cross_pairs * 2);
    g_profiler.start();
    kick_velocities(dp.local, slow_start, slow_count, pivot_dt * FSI_MOD_KICK);
    kick_velocities(dp.local, fast_start, fast_count, pivot_dt * FSI_MOD_KICK);
    g_profiler.stop(StepProfiler::DRIFT_KICK);

    // Step 5: Recurse on fast (second half)
#ifdef USE_MPI
#endif
    {
        double sub_pivot = pivot_dt * 0.5;
        hhs_fsi_recursive_distributed(dp, fast_start, fast_count, sub_pivot,
                                      G, soft2, dt_min, eta, false,
                                      tree_for_subs, depth + 1);
    }

    // Restore tree state for step 6 (self-force on slow) and step 7 (cross-force)
    if (use_tree_cross || use_tree_self) {
        g_profiler.start();
        if (use_tree_cross) {
            g_octree.update_subsets_3flag(dp.local, start, count, slow_count);
        } else {
            g_octree.update_subsets(dp.local, slow_start, slow_count);
        }
        g_profiler.stop(StepProfiler::TREE_MAINT);
    }

    // Step 6: FSI half-step on slow
#ifdef USE_MPI
#endif
    if (use_tree_self || use_tree_cross) {
        fsi_step_distributed(dp, slow_start, slow_count, G, soft2, pivot_dt * 0.5,
                             false, false, 0);
    } else {
        fsi_step_distributed(dp, slow_start, slow_count, G, soft2, pivot_dt * 0.5);
    }

    // Refresh tree for step 7 cross-force (step 6 FSI drifted slow particles)
    if (use_tree_cross) {
        g_profiler.start();
        g_octree.update_subsets_3flag(dp.local, start, count, slow_count);
        g_profiler.stop(StepProfiler::TREE_MAINT);
    }

    // Step 7: Cross-force S<->F then kick (dt/6)
#ifdef USE_MPI
#endif
    g_profiler.start();
    if (use_tree_cross) {
        cross_acceleration_global_tree(dp.local, slow_start, slow_count,
                                        fast_start, fast_count, G, soft2);
    } else {
        dp.local.pack_pos_range(slow_start, slow_count + fast_count);
        cross_acceleration_distributed(dp, slow_start, slow_count, fast_start, fast_count, G, soft2);
    }
    g_profiler.stop(StepProfiler::CROSS_FORCE, cross_pairs);
    g_profiler.start();
    kick_velocities(dp.local, slow_start, slow_count, pivot_dt * FSI_KICK);
    kick_velocities(dp.local, fast_start, fast_count, pivot_dt * FSI_KICK);
    g_profiler.stop(StepProfiler::DRIFT_KICK);
#ifdef USE_MPI
#endif
#endif // MPI_TOPOLOGY_RING
}

// =============================================================================
// GADGET-4 Cross-Force Decomposition: F_cross = F_all - F_self
//
// Instead of computing explicit cross-forces (N²_slow × N²_fast), we compute:
//   F_cross(i) = F_all(i) - F_self(i)
// where F_all = force from ALL particles (one tree walk using total cell_com),
// and F_self = force within own subset (using dual cell_com_slow/fast).
//
// Uses 3-flag system: 0=slow, 1=fast, 2=other (not at this level).
// Total monopoles (cell_com) always valid; dual monopoles recomputed per level.
// =============================================================================

// Save ax/ay/az → scratch1/2/3 for particles [start, start+count)
static void save_accelerations(Particles& p, int start, int count) {
    auto ax = p.ax, ay = p.ay, az = p.az;
    auto s1 = p.scratch1, s2 = p.scratch2, s3 = p.scratch3;
    Kokkos::parallel_for("save_accel", count, KOKKOS_LAMBDA(int ii) {
        int i = start + ii;
        s1(i) = ax(i);
        s2(i) = ay(i);
        s3(i) = az(i);
    });
}

// Cross-kick: v += (scratch - ax) * dt for particles [start, start+count)
// scratch1/2/3 holds F_all, ax/ay/az holds F_self → difference = F_cross
static void kick_cross_from_scratch(Particles& p, int start, int count, double dt) {
    auto vx = p.vx, vy = p.vy, vz = p.vz;
    auto ax = p.ax, ay = p.ay, az = p.az;
    auto s1 = p.scratch1, s2 = p.scratch2, s3 = p.scratch3;
    Kokkos::parallel_for("kick_cross", count, KOKKOS_LAMBDA(int ii) {
        int i = start + ii;
        vx(i) += (s1(i) - ax(i)) * dt;
        vy(i) += (s2(i) - ay(i)) * dt;
        vz(i) += (s3(i) - az(i)) * dt;
    });
}

// Compute cell_accel for total monopoles from active particles only (sflag != 2)
// Used by tree_all_gradient_on_subset for GADGET-4 G_all gradient correction
static void compute_active_total_cell_accelerations(Particles& p) {
    g_octree.compute_active_cell_accelerations(p);
}

// GADGET-4 HHS recursive implementation (distributed, i-partitioned topology)
// Uses F_cross = F_all - F_self decomposition with global tree.
// forces_from_all_valid: if true, p.ax/ay/az already contain F_all for [start, count)
static void gadget4_hhs_recursive_distributed(
    DistributedParticles& dp,
    int start, int count,
    double pivot_dt,
    double G, double soft2,
    double dt_min, double eta,
    bool forces_from_all_valid) {

    if (count == 0) return;
    g_subset_hist.record(SubsetHistogram::DISTRIBUTED, count);

    // Below serial_n2_gpu threshold: skip MPI, compute full N² locally
    if (serial_n2_gpu > 0 && count <= serial_n2_gpu) {
        hhs_fsi_recursive_local(dp.local, start, count, pivot_dt,
                                G, soft2, dt_min, eta, forces_from_all_valid);
        dp.local.pack_pos_range(start, count);
        return;
    }

    // Base case: at minimum timestep, use FSI on the subset with global tree
    if (pivot_dt * 0.5 < dt_min) {
        g_profiler.start();
        g_octree.update_subsets(dp.local, start, count);
        g_profiler.stop(StepProfiler::TREE_MAINT);
        // All particles at this level are "slow" (flag=0), use global tree self-force
        fsi_step_distributed(dp, start, count, G, soft2, pivot_dt,
                             false, false, 0);
        return;
    }

    // ─── Timestep + Partition ────────────────────────────
    if (!forces_from_all_valid && count > 1) {
        g_profiler.start();
        // Use 3-flag system: all particles at this level are active (flag=0 or 1)
        // For timestep, we need F_all. Use tree_all_forces_and_timestep_on_subset.
        // First, set 3-flag: everything in [start, start+count) = flag 0, rest = flag 2
        g_octree.update_subsets_3flag(dp.local, start, count, count);  // all "slow" for now
        g_octree.recompute_total_monopoles();
        tree_all_forces_and_timestep_on_subset(dp.local, g_octree, G, soft2, eta);
        tree_momentum_correction(dp.local, start, count);
        forces_from_all_valid = true;
        g_profiler.stop(StepProfiler::ADAPTIVE_DT, (long long)count * count);
    }

    // Partition into slow (dt >= pivot) and fast (dt < pivot)
    g_profiler.start();
    int slow_count = partition_by_pivot(dp.local, start, count, pivot_dt);
    g_profiler.stop(StepProfiler::PARTITION);
    int slow_start = start;
    int fast_start = start + slow_count;
    int fast_count = count - slow_count;

    // Base case: all particles are slow
    if (fast_count == 0) {
        g_profiler.start();
        g_octree.update_subsets(dp.local, start, count);
        g_profiler.stop(StepProfiler::TREE_MAINT);
        fsi_step_distributed(dp, start, count, G, soft2, pivot_dt,
                             false, false, 0);
        return;
    }

    // Base case: all particles are fast — partition was a no-op, dt_adapt still valid
    if (slow_count == 0) {
        double sub_pivot = pivot_dt * 0.5;
        gadget4_hhs_recursive_distributed(dp, start, count, sub_pivot,
                                          G, soft2, dt_min, eta, true);
        gadget4_hhs_recursive_distributed(dp, start, count, sub_pivot,
                                          G, soft2, dt_min, eta, false);
        return;
    }

    // ─── Set up 3-flag system for this level ───────────
    g_profiler.start();
    g_octree.update_subsets_3flag(dp.local, start, count, slow_count);
    g_octree.recompute_total_monopoles();
    g_profiler.stop(StepProfiler::TREE_MAINT);

    long long self_pairs_s = (long long)slow_count * slow_count;
    long long self_pairs_f = (long long)fast_count * fast_count;

    // ─── Step 1: Cross-kick(dt/6) ───────────────────────
    // F_all is already in ax (from timestep, permuted with partition)
    // Save F_all → scratch
    g_profiler.start();
    if (forces_from_all_valid) {
        save_accelerations(dp.local, start, count);
    } else {
        // Recompute F_all if not valid
        tree_all_forces_on_subset(dp.local, g_octree, G, soft2);
        tree_momentum_correction(dp.local, start, count);
        save_accelerations(dp.local, start, count);
    }
    // Compute F_self for slow (flag=0) and fast (flag=1)
    tree_global_self_acceleration(dp.local, g_octree, 0, G, soft2);
    tree_momentum_correction(dp.local, slow_start, slow_count);
    tree_global_self_acceleration(dp.local, g_octree, 1, G, soft2);
    tree_momentum_correction(dp.local, fast_start, fast_count);
    g_profiler.stop(StepProfiler::CROSS_FORCE, self_pairs_s + self_pairs_f);

    // kick_cross: v += (F_all - F_self) * dt/6
    g_profiler.start();
    kick_cross_from_scratch(dp.local, start, count, pivot_dt * FSI_KICK);
    g_profiler.stop(StepProfiler::DRIFT_KICK);

    // ─── Step 2: E_S(dt/2) — FSI half-step on slow ──────
    // F_self_slow is in ax for slow particles — reuse as forces_valid
    fsi_step_distributed(dp, slow_start, slow_count, G, soft2, pivot_dt * 0.5,
                         true, false, 0);

    // ─── Step 3: Recurse on fast (first half) ────────────
    gadget4_hhs_recursive_distributed(dp, fast_start, fast_count, pivot_dt * 0.5,
                                      G, soft2, dt_min, eta, false);

    // ─── Step 4: Cross-kick+gradient(2dt/3) ──────────────
    // Restore 3-flag system (recursion and FSI drifts changed particle positions/ordering)
    g_profiler.start();
    g_octree.update_subsets_3flag(dp.local, start, count, slow_count);
    g_octree.recompute_total_monopoles();
    g_profiler.stop(StepProfiler::TREE_MAINT);

    // F_all + G_all
    g_profiler.start();
    tree_all_forces_on_subset(dp.local, g_octree, G, soft2);
    tree_momentum_correction(dp.local, start, count);
    compute_active_total_cell_accelerations(dp.local);
    tree_all_gradient_on_subset(dp.local, g_octree, G, soft2, pivot_dt);
    tree_momentum_correction(dp.local, start, count);
    save_accelerations(dp.local, start, count);  // scratch = F_all + G_all
    g_profiler.stop(StepProfiler::CROSS_FORCE, self_pairs_s + self_pairs_f);

    // F_self for both subsets (must both be in ax before computing cell accelerations)
    g_profiler.start();
    tree_global_self_acceleration(dp.local, g_octree, 0, G, soft2);
    tree_momentum_correction(dp.local, slow_start, slow_count);
    tree_global_self_acceleration(dp.local, g_octree, 1, G, soft2);
    tree_momentum_correction(dp.local, fast_start, fast_count);

    // Dual cell accelerations from F_self (needed for G_self gradient correction)
    g_octree.compute_dual_cell_accelerations(dp.local);

    // G_self for both subsets (ADDED to ax which already has F_self)
    tree_global_self_gradient_correction(dp.local, g_octree, 0, G, soft2, pivot_dt);
    tree_momentum_correction(dp.local, slow_start, slow_count);
    tree_global_self_gradient_correction(dp.local, g_octree, 1, G, soft2, pivot_dt);
    tree_momentum_correction(dp.local, fast_start, fast_count);
    g_profiler.stop(StepProfiler::CROSS_GRAD, self_pairs_s + self_pairs_f);

    // kick_cross: v += (F_all+G_all - F_self-G_self) * 2dt/3
    g_profiler.start();
    kick_cross_from_scratch(dp.local, start, count, pivot_dt * FSI_MOD_KICK);
    g_profiler.stop(StepProfiler::DRIFT_KICK);

    // ─── Step 5: Recurse on fast (second half) ──────────
    gadget4_hhs_recursive_distributed(dp, fast_start, fast_count, pivot_dt * 0.5,
                                      G, soft2, dt_min, eta, false);

    // ─── Step 6: E_S(dt/2) — FSI half-step on slow ──────
    // Restore flags for slow self-force
    g_profiler.start();
    g_octree.update_subsets_3flag(dp.local, start, count, slow_count);
    g_profiler.stop(StepProfiler::TREE_MAINT);
    fsi_step_distributed(dp, slow_start, slow_count, G, soft2, pivot_dt * 0.5,
                         false, false, 0);

    // ─── Step 7: Cross-kick(dt/6) ───────────────────────
    // Restore flags (step 6 FSI drifted slow particles)
    g_profiler.start();
    g_octree.update_subsets_3flag(dp.local, start, count, slow_count);
    g_octree.recompute_total_monopoles();
    g_profiler.stop(StepProfiler::TREE_MAINT);

    g_profiler.start();
    tree_all_forces_on_subset(dp.local, g_octree, G, soft2);
    tree_momentum_correction(dp.local, start, count);
    save_accelerations(dp.local, start, count);  // scratch = F_all
    tree_global_self_acceleration(dp.local, g_octree, 0, G, soft2);
    tree_momentum_correction(dp.local, slow_start, slow_count);
    tree_global_self_acceleration(dp.local, g_octree, 1, G, soft2);
    tree_momentum_correction(dp.local, fast_start, fast_count);
    g_profiler.stop(StepProfiler::CROSS_FORCE, self_pairs_s + self_pairs_f);

    g_profiler.start();
    kick_cross_from_scratch(dp.local, start, count, pivot_dt * FSI_KICK);
    g_profiler.stop(StepProfiler::DRIFT_KICK);
}

// Distributed HHS-FSI - main entry point
void hhs_fsi_step_distributed(DistributedParticles& dp, double pivot_dt,
                              double G, double soft2, double dt_min,
                              double eta, int global_levels) {
    if (dp.N_total == 0) return;

    (void)global_levels;  // Not used in new implementation

    // Reset debug counters
    dbg_hhs_distributed = 0;
    dbg_hhs_local = 0;
    dbg_hhs_cpu = 0;
    dbg_hhs_slow0_dist = 0;
    dbg_hhs_slow0_local = 0;
    dbg_hhs_max_depth = 0;
    std::memset(dbg_local_depth_calls, 0, sizeof(dbg_local_depth_calls));
    std::memset(dbg_local_depth_count, 0, sizeof(dbg_local_depth_count));
    std::memset(dbg_local_depth_time, 0, sizeof(dbg_local_depth_time));
    std::memset(dbg_dist_depth_calls, 0, sizeof(dbg_dist_depth_calls));
    std::memset(dbg_dist_depth_count, 0, sizeof(dbg_dist_depth_count));
    std::memset(dbg_dist_depth_ngb_pairs, 0, sizeof(dbg_dist_depth_ngb_pairs));
    std::memset(dbg_dist_depth_l0_clamped, 0, sizeof(dbg_dist_depth_l0_clamped));

#ifdef ENABLE_COLLISIONAL
    if (dp.local.collisional) g_subsystem_mgr.reset_step_diagnostics();
#endif

#ifdef USE_MPI
    bool is_distributed = MPIComm::size() > 1;
#else
    bool is_distributed = false;
#endif

    // Global PN1.0 half-kick before FSI (BIFROST Section 3.7)
#ifdef ENABLE_COLLISIONAL
    bool do_global_pn = dp.local.collisional && (g_collisional.pn_order & 1) &&
                        g_collisional.c_code > 0.0;
    if (do_global_pn) {
        global_pn1_kick(dp.local, 0, dp.N_total, G, g_collisional.c_code,
                        g_collisional.m_pn, pivot_dt * 0.5);
    }
#endif

    if (!is_distributed) {
        // Single rank: use standard HHS (uses efficient fused kernels)
        hhs_fsi_step_dynamic(dp.local, 0, dp.N_total, pivot_dt, G, soft2, dt_min, eta);
    }
#ifdef USE_MPI
    else if (gadget4_cross && use_tree) {
        // GADGET-4 decomposition: F_cross = F_all - F_self
        // Build the global tree once at the top level
        g_octree.n_leaf = tree_n_leaf;
        g_octree.n_group = tree_n_group;
        g_octree.build(dp.local, 0, dp.N_total, tree_theta);

        gadget4_hhs_recursive_distributed(dp, 0, dp.N_total, pivot_dt,
                                          G, soft2, dt_min, eta, false);
    } else {
        // Standard HHS-FSI with explicit cross-forces
        hhs_fsi_recursive_distributed(dp, 0, dp.N_total, pivot_dt,
                                      G, soft2, dt_min, eta, false);
    }
#endif

    // Global PN1.0 half-kick after FSI (BIFROST Section 3.7)
#ifdef ENABLE_COLLISIONAL
    if (do_global_pn) {
        global_pn1_kick(dp.local, 0, dp.N_total, G, g_collisional.c_code,
                        g_collisional.m_pn, pivot_dt * 0.5);
    }
    // print_step_summary moved into unified block below
#endif

    // Unified post-step summary: HHS hierarchy + subsystem stats
#ifdef USE_MPI
    if (MPIComm::rank() == 0)
#endif
    {
        std::printf("  HHS: dist=%d gpu=%d cpu=%d max_depth=%d\n",
                    dbg_hhs_distributed, dbg_hhs_local, dbg_hhs_cpu, dbg_hhs_max_depth);
        for (int d = 0; d < DBG_MAX_DEPTH; d++) {
            int dc = dbg_dist_depth_calls[d];
            int lc = dbg_local_depth_calls[d];
            if (dc == 0 && lc == 0) break;
            int total_calls = dc + lc;
            long long avg_n = (dbg_dist_depth_count[d] + dbg_local_depth_count[d]) / total_calls;
            std::printf("    L%d: %d calls, avg_n=%lld", d, total_calls, avg_n);
            if (dbg_dist_depth_ngb_pairs[d] > 0)
                std::printf(", ngb=%d", dbg_dist_depth_ngb_pairs[d]);
            if (dbg_dist_depth_l0_clamped[d] > 0)
                std::printf(", l0_clamp=%d", dbg_dist_depth_l0_clamped[d]);
            if (dbg_local_depth_time[d] > 0.001)
                std::printf(", time=%.3fs", dbg_local_depth_time[d]);
            std::printf("\n");
        }

#ifdef ENABLE_COLLISIONAL
        if (dp.local.collisional) g_subsystem_mgr.print_step_summary();
#endif
    }
}

