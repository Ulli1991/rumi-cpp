#include "integrator.h"
#include "forces.h"
#include "timestep.h"
#include "constants.h"
#include "mpi_comm.h"
#include "profiling.h"
#include <Kokkos_Core.hpp>
#include <vector>
#include <cfloat>

// Global profiler instance
StepProfiler g_profiler;

// HHS adaptive timestep skip option
bool skip_second_timestep = false;

// Direct N² threshold for skipping MPI distribution on small subsets
int direct_n2_threshold = 0;

// CPU serial threshold for avoiding GPU kernel launch overhead on tiny subsets
int cpu_serial_threshold = 0;

// Fused force+gradient: use stale accelerations to compute gradient in same N² pass as force
bool fuse_gradient = false;

// Mixed precision: FP32 rsqrt seed + FP64 Newton-Raphson (Level 1)
bool mixed_precision = false;

// TILE_SIZE defined in constants.h

// Fused function pointer types (force + kick combined)
using AccKickFn = void(*)(Particles&, int, int, double, double, double);
using AccGradKickFn = void(*)(Particles&, int, int, double, double, double, double);
using CrossAccGradKickFn = void(*)(Particles&, int, int, int, int, double, double, double, double);
using CrossAccKickCacheFn = void(*)(Particles&, int, int, int, int, double, double, double);

static AccKickFn fn_self_acc_kick = nullptr;
static AccGradKickFn fn_self_acc_grad_kick = nullptr;
static CrossAccGradKickFn fn_cross_acc_grad_kick = nullptr;
static CrossAccKickCacheFn fn_cross_acc_kick_cache = nullptr;
static CrossAccKickCacheFn fn_cross_acc_kick = nullptr;  // non-caching version

// =============================================================================
// Distributed force + timestep computation for between-step updates
// Each rank computes for its slice [my_start, my_count) from ALL N particles
// Then allgather both accelerations and timestep values
// Uses tiled GPU kernel with shared memory for efficiency
// =============================================================================
void compute_acceleration_and_timestep_distributed(DistributedParticles& dp,
                                                 double G, double soft2, double eta,
                                                 bool use_n3l) {
#ifdef USE_MPI
    if (MPIComm::size() == 1) {
        // Single rank: use optimized single-GPU version
        if (use_n3l)
            compute_acceleration_and_timestep<true>(dp.local, G, soft2, eta);
        else
            compute_acceleration_and_timestep<false>(dp.local, G, soft2, eta);
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

    const bool mp = mixed_precision;

    // Pack data for vectorized memory access
    dp.local.pack();

    auto pos = dp.local.pos_packed;
    auto vel = dp.local.vel_packed;
    RawView pax{dp.local.ax.data()}, pay{dp.local.ay.data()}, paz{dp.local.az.data()};
    RawView dt_f{dp.local.dt_adapt.data()};

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
                        double inv_r = fast_rsqrt(r2s, mp);
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
                        double t_min_sq_pair = Kokkos::fmin(t_ff_sq, t_fb_sq);
                        t_min_i = Kokkos::fmin(t_min_i, t_min_sq_pair);
                    }
                }
                team.team_barrier();
            }

            if (ii < my_count) {
                pax(i) = axi;
                pay(i) = ayi;
                paz(i) = azi;
                dt_f(i) = Kokkos::sqrt(t_min_i) * eta;
            }
        });
    Kokkos::fence();

    // Allgather accelerations and timesteps
    dp.allgather_accelerations();
    dp.allgather_timestep();
#else
    if (use_n3l)
        compute_acceleration_and_timestep<true>(dp.local, G, soft2, eta);
    else
        compute_acceleration_and_timestep<false>(dp.local, G, soft2, eta);
#endif
}

void set_force_dispatch(bool use_n3l) {
    if (use_n3l) {
        fn_self_acc_kick = &self_acceleration_kick<true>;
        fn_self_acc_grad_kick = &self_acceleration_and_gradient_kick<true>;
        fn_cross_acc_grad_kick = &cross_acceleration_and_gradient_kick<true>;
        fn_cross_acc_kick_cache = &cross_acceleration_kick_and_cache<true>;
        fn_cross_acc_kick = &cross_acceleration_kick<true>;
    } else {
        fn_self_acc_kick = &self_acceleration_kick<false>;
        fn_self_acc_grad_kick = &self_acceleration_and_gradient_kick<false>;
        fn_cross_acc_grad_kick = &cross_acceleration_and_gradient_kick<false>;
        fn_cross_acc_kick_cache = &cross_acceleration_kick_and_cache<false>;
        fn_cross_acc_kick = &cross_acceleration_kick<false>;
    }
}

// Helper: drift positions for [start, start+count)
// Not static - also used by distributed FSI
void drift(Particles& p, int start, int count, double dt) {
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    Kokkos::parallel_for("drift", count, KOKKOS_LAMBDA(int ii) {
        int i = start + ii;
        px(i) += pvx(i) * dt;
        py(i) += pvy(i) * dt;
        pz(i) += pvz(i) * dt;
    });
    // No fence - next operation will handle dependency
}

void fsi_step(Particles& p, int start, int count,
              double G, double soft2, double dt) {
    // U kick (dt/6) - fused force + kick
    g_profiler.start();
    fn_self_acc_kick(p, start, count, G, soft2, dt * FSI_KICK);
    g_profiler.stop(StepProfiler::SELF_FORCE);

    // T drift (dt/2)
    g_profiler.start();
    drift(p, start, count, dt * FSI_DRIFT);
    g_profiler.stop(StepProfiler::DRIFT_KICK);

    // U-tilde kick (2dt/3) - fused force + gradient + kick
    g_profiler.start();
    fn_self_acc_grad_kick(p, start, count, G, soft2, dt, dt * FSI_MOD_KICK);
    g_profiler.stop(StepProfiler::SELF_GRAD);

    // T drift (dt/2)
    g_profiler.start();
    drift(p, start, count, dt * FSI_DRIFT);
    g_profiler.stop(StepProfiler::DRIFT_KICK);

    // U kick (dt/6) - fused force + kick
    g_profiler.start();
    fn_self_acc_kick(p, start, count, G, soft2, dt * FSI_KICK);
    g_profiler.stop(StepProfiler::SELF_FORCE);
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
    Kokkos::fence();

    // Apply permutation: fused gather of all 13 fields (11 doubles + 2 ints)
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView pmass{p.mass.data()}, pdt{p.dt_adapt.data()};
    auto porig = p.orig_index;
    auto plevel = p.level;

    // Scratch: 11 doubles + 2 ints
    auto t1 = p.scratch1;  auto t2 = p.scratch2;  auto t3 = p.scratch3;
    auto t4 = p.scratch4;  auto t5 = p.scratch5;  auto t6 = p.scratch6;
    auto t7 = p.scratch7;  auto t8 = p.scratch8;  auto t9 = p.scratch9;
    auto t10 = p.scratch10; auto t11 = p.scratch11;
    auto ti1 = p.scratch_i2; auto ti2 = p.scratch_i3;

    // Single fused gather of all fields
    Kokkos::parallel_for("gather_all", count, KOKKOS_LAMBDA(int ii) {
        int src = perm(ii);
        t1(ii) = px(src);   t2(ii) = py(src);   t3(ii) = pz(src);
        t4(ii) = pvx(src);  t5(ii) = pvy(src);  t6(ii) = pvz(src);
        t7(ii) = pax(src);  t8(ii) = pay(src);  t9(ii) = paz(src);
        t10(ii) = pmass(src); t11(ii) = pdt(src);
        ti1(ii) = porig(src); ti2(ii) = plevel(src);
    });
    Kokkos::fence();

    // Single fused copy back of all fields
    Kokkos::parallel_for("copy_all", count, KOKKOS_LAMBDA(int ii) {
        int dst = start + ii;
        px(dst) = t1(ii);   py(dst) = t2(ii);   pz(dst) = t3(ii);
        pvx(dst) = t4(ii);  pvy(dst) = t5(ii);  pvz(dst) = t6(ii);
        pax(dst) = t7(ii);  pay(dst) = t8(ii);  paz(dst) = t9(ii);
        pmass(dst) = t10(ii); pdt(dst) = t11(ii);
        porig(dst) = ti1(ii); plevel(dst) = ti2(ii);
    });
    Kokkos::fence();

    return slow_count;
}

// ==========================================================================
// HHS-FSI with dynamic pivot-based partitioning (per paper's Algorithm 4)
// ==========================================================================
static void hhs_fsi_step_dynamic_impl(Particles& p, int start, int count, double pivot_dt,
                          double G, double soft2, double dt_min,
                          double eta, bool use_n3l, bool dt_valid) {
    if (count == 0) return;

    // Compute adaptive timesteps for this subset (skip if reusing from first half-step)
    if (!dt_valid && count > 1) {
        g_profiler.start();
        if (use_n3l)
            compute_timestep_subset<true>(p, start, count, G, soft2, eta);
        else
            compute_timestep_subset<false>(p, start, count, G, soft2, eta);
        g_profiler.stop(StepProfiler::ADAPTIVE_DT);
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

    // HHS-FSI structure (Algorithm 4 / Eq. 40)
    // 1. Cross-Newton kick (dt/6)
    g_profiler.start();
    fn_cross_acc_kick_cache(p, slow_start, slow_count, fast_start, fast_count, G, soft2, pivot_dt * FSI_KICK);
    g_profiler.stop(StepProfiler::CROSS_FORCE);

    // 2. FSI half-step on slow
    fsi_step(p, slow_start, slow_count, G, soft2, pivot_dt * 0.5);

    // 3. Recurse on fast (first half) with pivot/2 — always compute fresh timesteps
    hhs_fsi_step_dynamic_impl(p, fast_start, fast_count, pivot_dt * 0.5, G, soft2, dt_min, eta, use_n3l, false);

    // 4. Cross-gradient kick (2dt/3)
    g_profiler.start();
    fn_cross_acc_grad_kick(p, slow_start, slow_count, fast_start, fast_count, G, soft2, pivot_dt, pivot_dt * FSI_MOD_KICK);
    g_profiler.stop(StepProfiler::CROSS_GRAD);

    // 5. Recurse on fast (second half)
    {
        double sub_pivot = pivot_dt * 0.5;
        hhs_fsi_step_dynamic_impl(p, fast_start, fast_count, sub_pivot, G, soft2, dt_min, eta, use_n3l, skip_second_timestep);
    }

    // 6. FSI half-step on slow
    fsi_step(p, slow_start, slow_count, G, soft2, pivot_dt * 0.5);

    // 7. Cross-Newton kick (dt/6)
    g_profiler.start();
    fn_cross_acc_kick_cache(p, slow_start, slow_count, fast_start, fast_count, G, soft2, pivot_dt * FSI_KICK);
    g_profiler.stop(StepProfiler::CROSS_FORCE);
}

void hhs_fsi_step_dynamic(Particles& p, int start, int count, double pivot_dt,
                          double G, double soft2, double dt_min,
                          double eta, bool use_n3l) {
    hhs_fsi_step_dynamic_impl(p, start, count, pivot_dt, G, soft2, dt_min, eta, use_n3l, false);
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
    const bool mp = mixed_precision;

    int slice_start, slice_count;
    dp.get_force_slice(start, count, slice_start, slice_count);
    if (slice_count == 0) {
        dp.allgather_accelerations_subset(start, count);
        return;
    }

    auto pos = dp.local.pos_packed;
    RawView pax{dp.local.ax.data()}, pay{dp.local.ay.data()}, paz{dp.local.az.data()};

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
            if (ii < slice_count) {
                double mi_pad;
                load4(&pos(i, 0), xi, yi, zi, mi_pad);
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

                        int jb = jt * 4;
                        double dx = tile(jb    ) - xi;
                        double dy = tile(jb + 1) - yi;
                        double dz = tile(jb + 2) - zi;

                        double r2 = dx*dx + dy*dy + dz*dz + soft2;
                        double inv_r = fast_rsqrt(r2, mp);
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

    dp.allgather_accelerations_subset(start, count);
}

// =============================================================================
// Fused distributed force + kick: computes forces for my slice AND kicks from
// registers, then allgather forces, then kick non-local particles.
// Saves one kernel launch + memory round-trip for accelerations on local slice.
// =============================================================================
static void self_acceleration_kick_distributed(DistributedParticles& dp, int start, int count,
                                                double G, double soft2, double kick_dt) {
    if (count <= 0) return;
    const bool mp = mixed_precision;

    int slice_start, slice_count;
    dp.get_force_slice(start, count, slice_start, slice_count);
    if (slice_count == 0) {
        dp.allgather_accelerations_subset(start, count);
        kick_velocities(dp.local, start, count, kick_dt);
        return;
    }

    auto pos = dp.local.pos_packed;
    RawView pvx{dp.local.vx.data()}, pvy{dp.local.vy.data()}, pvz{dp.local.vz.data()};
    RawView pax{dp.local.ax.data()}, pay{dp.local.ay.data()}, paz{dp.local.az.data()};

    const int j_start = start;
    const int j_end = start + count;
    const double kdt = kick_dt;

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t scratch_size = TILE_SIZE * 4 * sizeof(double);
    int num_teams = (slice_count + TILE_SIZE - 1) / TILE_SIZE;

    Kokkos::parallel_for("dist_self_acc_kick_tiled",
        team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * TILE_SIZE + team.team_rank();
            int i = slice_start + ii;

            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 4);

            double xi = 0, yi = 0, zi = 0;
            if (ii < slice_count) {
                double mi_pad;
                load4(&pos(i, 0), xi, yi, zi, mi_pad);
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

                        int jb = jt * 4;
                        double dx = tile(jb    ) - xi;
                        double dy = tile(jb + 1) - yi;
                        double dz = tile(jb + 2) - zi;

                        double r2 = dx*dx + dy*dy + dz*dz + soft2;
                        double inv_r = fast_rsqrt(r2, mp);
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
                pvx(i) += axi * kdt;
                pvy(i) += ayi * kdt;
                pvz(i) += azi * kdt;
            }
        });
    Kokkos::fence();

    // Allgather forces so all ranks have consistent accelerations
    dp.allgather_accelerations_subset(start, count);

    // Kick non-local particles using allgathered accelerations
    int ss = slice_start, sc = slice_count;
    Kokkos::parallel_for("kick_non_local", count, KOKKOS_LAMBDA(int ii) {
        int i = start + ii;
        if (i >= ss && i < ss + sc) return;  // already kicked in fused kernel
        pvx(i) += pax(i) * kdt;
        pvy(i) += pay(i) * kdt;
        pvz(i) += paz(i) * kdt;
    });
    Kokkos::fence();
}

// =============================================================================
// Fused distributed gradient correction + kick: computes gradient correction
// for my slice AND kicks from corrected accelerations, then allgather, then
// kick non-local particles.
// =============================================================================
static void self_gradient_correction_kick_distributed(DistributedParticles& dp, int start, int count,
                                                       double G, double soft2, double grad_dt, double kick_dt) {
    if (count <= 0) return;
    const bool mp = mixed_precision;

    int slice_start, slice_count;
    dp.get_force_slice(start, count, slice_start, slice_count);
    if (slice_count == 0) {
        dp.allgather_accelerations_subset(start, count);
        kick_velocities(dp.local, start, count, kick_dt);
        return;
    }

    auto pos = dp.local.pos_packed;
    RawView pvx{dp.local.vx.data()}, pvy{dp.local.vy.data()}, pvz{dp.local.vz.data()};
    RawView pax{dp.local.ax.data()}, pay{dp.local.ay.data()}, paz{dp.local.az.data()};

    const double coeff = G * grad_dt * grad_dt * GRAD_COEFF;
    const int j_start = start;
    const int j_end = start + count;
    const double kdt = kick_dt;

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t scratch_size = TILE_SIZE * 7 * sizeof(double);
    int num_teams = (slice_count + TILE_SIZE - 1) / TILE_SIZE;

    Kokkos::parallel_for("dist_grad_kick_tiled",
        team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * TILE_SIZE + team.team_rank();
            int i = slice_start + ii;

            // AoS interleaved tile: [x,y,z,m,ax,ay,az] per j-particle
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 7);

            double xi = 0, yi = 0, zi = 0;
            double axi = 0, ayi = 0, azi = 0;
            if (ii < slice_count) {
                double mi_pad;
                load4(&pos(i, 0), xi, yi, zi, mi_pad);
                axi = pax(i); ayi = pay(i); azi = paz(i);
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

                        int jb = jt * 7;
                        double dx = tile(jb    ) - xi;
                        double dy = tile(jb + 1) - yi;
                        double dz = tile(jb + 2) - zi;

                        double dax = tile(jb + 4) - axi;
                        double day = tile(jb + 5) - ayi;
                        double daz = tile(jb + 6) - azi;

                        double r2 = dx*dx + dy*dy + dz*dz + soft2;
                        double inv_r = fast_rsqrt(r2, mp);
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
                // Apply gradient correction
                double corrected_ax = axi + cx;
                double corrected_ay = ayi + cy;
                double corrected_az = azi + cz;
                pax(i) = corrected_ax;
                pay(i) = corrected_ay;
                paz(i) = corrected_az;
                // Fused kick from corrected accelerations in registers
                pvx(i) += corrected_ax * kdt;
                pvy(i) += corrected_ay * kdt;
                pvz(i) += corrected_az * kdt;
            }
        });
    Kokkos::fence();

    // Allgather corrected accelerations
    dp.allgather_accelerations_subset(start, count);

    // Kick non-local particles using allgathered corrected accelerations
    int ss = slice_start, sc = slice_count;
    Kokkos::parallel_for("kick_grad_non_local", count, KOKKOS_LAMBDA(int ii) {
        int i = start + ii;
        if (i >= ss && i < ss + sc) return;  // already kicked in fused kernel
        pvx(i) += pax(i) * kdt;
        pvy(i) += pay(i) * kdt;
        pvz(i) += paz(i) * kdt;
    });
    Kokkos::fence();
}

// =============================================================================
// Fused distributed force + gradient + kick using STALE accelerations.
// Single tiled N² kernel computes force AND gradient simultaneously.
// The gradient uses stale accelerations (from the previous force evaluation)
// which are snapshotted before the kernel overwrites pax/pay/paz.
// This trades O(dt³) accuracy for ~30% fewer FLOPS and field loads per pair.
// =============================================================================
static void self_acceleration_gradient_kick_fused_distributed(
    DistributedParticles& dp, int start, int count,
    double G, double soft2, double grad_dt, double kick_dt) {
    if (count <= 0) return;
    const bool mp = mixed_precision;

    int slice_start, slice_count;
    dp.get_force_slice(start, count, slice_start, slice_count);
    if (slice_count == 0) {
        dp.allgather_accelerations_subset(start, count);
        kick_velocities(dp.local, start, count, kick_dt);
        return;
    }

    auto pos = dp.local.pos_packed;
    RawView pvx{dp.local.vx.data()}, pvy{dp.local.vy.data()}, pvz{dp.local.vz.data()};
    RawView pax{dp.local.ax.data()}, pay{dp.local.ay.data()}, paz{dp.local.az.data()};

    // Snapshot stale accelerations into scratch before the kernel overwrites them.
    // pax/pay/paz currently hold accelerations from the previous force evaluation
    // (step 1 of FSI, i.e. a(x₀)), which serve as the stale-a for the gradient.
    RawView ax_stale{dp.local.scratch1.data()};
    RawView ay_stale{dp.local.scratch2.data()};
    RawView az_stale{dp.local.scratch3.data()};
    Kokkos::parallel_for("snapshot_stale_a", count, KOKKOS_LAMBDA(int ii) {
        int i = start + ii;
        ax_stale(ii) = pax(i);
        ay_stale(ii) = pay(i);
        az_stale(ii) = paz(i);
    });
    Kokkos::fence();

    const double coeff = G * grad_dt * grad_dt * GRAD_COEFF;
    const int j_start = start;
    const int j_end = start + count;
    const double kdt = kick_dt;
    const int stale_offset = start;  // ax_stale(ii) corresponds to particle start+ii

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    // Tile: [x, y, z, m, ax_stale, ay_stale, az_stale] = 7 fields per j-particle
    size_t scratch_size = TILE_SIZE * 7 * sizeof(double);
    int num_teams = (slice_count + TILE_SIZE - 1) / TILE_SIZE;

    Kokkos::parallel_for("dist_fused_force_grad_kick",
        team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * TILE_SIZE + team.team_rank();
            int i = slice_start + ii;

            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 7);

            double xi = 0, yi = 0, zi = 0;
            double axi_stale = 0, ayi_stale = 0, azi_stale = 0;
            if (ii < slice_count) {
                double mi_pad;
                load4(&pos(i, 0), xi, yi, zi, mi_pad);
                axi_stale = ax_stale(i - stale_offset);
                ayi_stale = ay_stale(i - stale_offset);
                azi_stale = az_stale(i - stale_offset);
            }

            // Accumulators for force and gradient correction
            double axi = 0, ayi = 0, azi = 0;
            double cx = 0, cy = 0, cz = 0;

            for (int tile_start = j_start; tile_start < j_end; tile_start += TILE_SIZE) {
                int tile_end = (tile_start + TILE_SIZE < j_end) ? tile_start + TILE_SIZE : j_end;
                int tile_count = tile_end - tile_start;

                // Collaborative tile load: x,y,z,m from packed view, ax/ay/az from stale snapshot
                int t = team.team_rank();
                if (t < tile_count) {
                    int idx = tile_start + t;
                    int base = t * 7;
                    load4(&pos(idx, 0), tile(base), tile(base+1), tile(base+2), tile(base+3));
                    tile(base + 4) = ax_stale(idx - stale_offset);
                    tile(base + 5) = ay_stale(idx - stale_offset);
                    tile(base + 6) = az_stale(idx - stale_offset);
                }
                team.team_barrier();

                if (ii < slice_count) {
                    for (int jt = 0; jt < tile_count; jt++) {
                        int j = tile_start + jt;
                        if (j == i) continue;

                        int jb = jt * 7;
                        double dx = tile(jb    ) - xi;
                        double dy = tile(jb + 1) - yi;
                        double dz = tile(jb + 2) - zi;
                        double mj = tile(jb + 3);

                        double r2 = dx*dx + dy*dy + dz*dz + soft2;
                        double inv_r = fast_rsqrt(r2, mp);

                        // --- Force accumulation (shared geometry) ---
                        double inv_r3 = inv_r * inv_r * inv_r;
                        double mj_inv_r3 = G * mj * inv_r3;
                        axi += mj_inv_r3 * dx;
                        ayi += mj_inv_r3 * dy;
                        azi += mj_inv_r3 * dz;

                        // --- Gradient accumulation (using stale accelerations) ---
                        double dax = tile(jb + 4) - axi_stale;
                        double day = tile(jb + 5) - ayi_stale;
                        double daz = tile(jb + 6) - azi_stale;

                        double inv_r2 = inv_r * inv_r;
                        double inv_r5 = inv_r2 * inv_r2 * inv_r;
                        double r_dot_a = dx*dax + dy*day + dz*daz;
                        double f = coeff * mj * inv_r5;

                        cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                        cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                        cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
                    }
                }
                team.team_barrier();
            }

            if (ii < slice_count) {
                // Combined acceleration = force + gradient correction
                double total_ax = axi + cx;
                double total_ay = ayi + cy;
                double total_az = azi + cz;
                pax(i) = total_ax;
                pay(i) = total_ay;
                paz(i) = total_az;
                // Fused kick from registers
                pvx(i) += total_ax * kdt;
                pvy(i) += total_ay * kdt;
                pvz(i) += total_az * kdt;
            }
        });
    Kokkos::fence();

    // Allgather combined accelerations
    dp.allgather_accelerations_subset(start, count);

    // Kick non-local particles using allgathered accelerations
    int ss = slice_start, sc = slice_count;
    Kokkos::parallel_for("kick_fused_non_local", count, KOKKOS_LAMBDA(int ii) {
        int i = start + ii;
        if (i >= ss && i < ss + sc) return;  // already kicked in fused kernel
        pvx(i) += pax(i) * kdt;
        pvy(i) += pay(i) * kdt;
        pvz(i) += paz(i) * kdt;
    });
    Kokkos::fence();
}

// =============================================================================
// Distributed FSI step using scatter-compute-gather
// All ranks execute identical code; only force computation is distributed
// Uses fused force+kick kernels to reduce kernel launches and memory traffic
// =============================================================================
static void fsi_step_distributed(DistributedParticles& dp, int start, int count,
                                 double G, double soft2, double dt) {
    if (count <= 0) return;

    long long self_pairs = (long long)count * count;

    // Sync packed positions for vectorized kernel loads
    dp.local.pack_pos_range(start, count);

    // U kick (dt/6) - fused distributed force + kick
    g_profiler.start();
    self_acceleration_kick_distributed(dp, start, count, G, soft2, dt * FSI_KICK);
    g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);

    // T drift (dt/2) - all ranks drift identically
    g_profiler.start();
    drift(dp.local, start, count, dt * FSI_DRIFT);
    g_profiler.stop(StepProfiler::DRIFT_KICK);

    // Re-sync packed positions after drift
    dp.local.pack_pos_range(start, count);

    // U-tilde kick (2dt/3) with gradient correction
    if (fuse_gradient) {
        // Fused: single N² pass computes force + gradient (using stale a) + kick
        g_profiler.start();
        self_acceleration_gradient_kick_fused_distributed(dp, start, count, G, soft2, dt, dt * FSI_MOD_KICK);
        g_profiler.stop(StepProfiler::SELF_GRAD, self_pairs);
    } else {
        // Separate: force → allgather → gradient + kick
        g_profiler.start();
        self_acceleration_distributed(dp, start, count, G, soft2);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        g_profiler.start();
        self_gradient_correction_kick_distributed(dp, start, count, G, soft2, dt, dt * FSI_MOD_KICK);
        g_profiler.stop(StepProfiler::SELF_GRAD, self_pairs);
    }

    // T drift (dt/2)
    g_profiler.start();
    drift(dp.local, start, count, dt * FSI_DRIFT);
    g_profiler.stop(StepProfiler::DRIFT_KICK);

    // Re-sync packed positions after drift
    dp.local.pack_pos_range(start, count);

    // U kick (dt/6) - fused distributed force + kick
    g_profiler.start();
    self_acceleration_kick_distributed(dp, start, count, G, soft2, dt * FSI_KICK);
    g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
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
    const bool mp = mixed_precision;

    auto pos = dp.local.pos_packed;
    RawView pax{dp.local.ax.data()}, pay{dp.local.ay.data()}, paz{dp.local.az.data()};

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
                if (ii < slice_count) {
                    double mi_pad;
                    load4(&pos(i, 0), xi, yi, zi, mi_pad);
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
                            int jb = jt * 4;
                            double dx = tile(jb    ) - xi;
                            double dy = tile(jb + 1) - yi;
                            double dz = tile(jb + 2) - zi;

                            double r2 = dx*dx + dy*dy + dz*dz + soft2;
                            double inv_r = fast_rsqrt(r2, mp);
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
                if (ii < slice_count_f) {
                    double mi_pad;
                    load4(&pos(i, 0), xi, yi, zi, mi_pad);
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
                            int jb = jt * 4;
                            double dx = tile(jb)   - xi;
                            double dy = tile(jb+1) - yi;
                            double dz = tile(jb+2) - zi;

                            double r2 = dx*dx + dy*dy + dz*dz + soft2;
                            double inv_r = fast_rsqrt(r2, mp);
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
    dp.allgather_accelerations_subset(slow_start, slow_count);
    dp.allgather_accelerations_subset(fast_start, fast_count);
}

// =============================================================================
// Fused distributed cross-acceleration + kick
// Computes cross-forces for my slice, kicks from registers, allgathers, then
// kicks non-local particles. Saves 2 separate kick kernel launches.
// =============================================================================
static void cross_acceleration_kick_distributed(DistributedParticles& dp,
                                                 int slow_start, int slow_count,
                                                 int fast_start, int fast_count,
                                                 double G, double soft2, double kick_dt) {
    if (slow_count <= 0 || fast_count <= 0) return;
    const bool mp = mixed_precision;

    auto pos = dp.local.pos_packed;
    RawView pvx{dp.local.vx.data()}, pvy{dp.local.vy.data()}, pvz{dp.local.vz.data()};
    RawView pax{dp.local.ax.data()}, pay{dp.local.ay.data()}, paz{dp.local.az.data()};

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t scratch_size = TILE_SIZE * 4 * sizeof(double);
    const double kdt = kick_dt;

    // Slow particles: force from fast + fused kick
    int slice_start, slice_count;
    dp.get_force_slice(slow_start, slow_count, slice_start, slice_count);

    if (slice_count > 0) {
        int num_teams = (slice_count + TILE_SIZE - 1) / TILE_SIZE;
        const int f_start = fast_start;
        const int f_end = fast_start + fast_count;

        Kokkos::parallel_for("cross_kick_slow_from_fast",
            team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_LAMBDA(const member_type& team) {
                int ii = team.league_rank() * TILE_SIZE + team.team_rank();
                int i = slice_start + ii;

                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile(team.team_scratch(0), TILE_SIZE * 4);

                double xi = 0, yi = 0, zi = 0;
                if (ii < slice_count) {
                    double mi_pad;
                    load4(&pos(i, 0), xi, yi, zi, mi_pad);
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
                            int jb = jt * 4;
                            double dx = tile(jb    ) - xi;
                            double dy = tile(jb + 1) - yi;
                            double dz = tile(jb + 2) - zi;

                            double r2 = dx*dx + dy*dy + dz*dz + soft2;
                            double inv_r = fast_rsqrt(r2, mp);
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
                    pvx(i) += axi * kdt;
                    pvy(i) += ayi * kdt;
                    pvz(i) += azi * kdt;
                }
            });
    }

    // Fast particles: force from slow + fused kick
    int slice_start_f, slice_count_f;
    dp.get_force_slice(fast_start, fast_count, slice_start_f, slice_count_f);

    if (slice_count_f > 0) {
        int num_teams_f = (slice_count_f + TILE_SIZE - 1) / TILE_SIZE;
        const int s_start = slow_start;
        const int s_end = slow_start + slow_count;

        Kokkos::parallel_for("cross_kick_fast_from_slow",
            team_policy(num_teams_f, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_LAMBDA(const member_type& team) {
                int ii = team.league_rank() * TILE_SIZE + team.team_rank();
                int i = slice_start_f + ii;

                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile(team.team_scratch(0), TILE_SIZE * 4);

                double xi = 0, yi = 0, zi = 0;
                if (ii < slice_count_f) {
                    double mi_pad;
                    load4(&pos(i, 0), xi, yi, zi, mi_pad);
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
                            int jb = jt * 4;
                            double dx = tile(jb)   - xi;
                            double dy = tile(jb+1) - yi;
                            double dz = tile(jb+2) - zi;

                            double r2 = dx*dx + dy*dy + dz*dz + soft2;
                            double inv_r = fast_rsqrt(r2, mp);
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
                    pvx(i) += axi * kdt;
                    pvy(i) += ayi * kdt;
                    pvz(i) += azi * kdt;
                }
            });
    }
    Kokkos::fence();

    // Allgather then kick non-local particles
    dp.allgather_accelerations_subset(slow_start, slow_count);
    dp.allgather_accelerations_subset(fast_start, fast_count);

    // Kick non-local slow particles
    int ss1 = slice_start, sc1 = slice_count;
    Kokkos::parallel_for("kick_cross_nl_slow", slow_count, KOKKOS_LAMBDA(int ii) {
        int i = slow_start + ii;
        if (i >= ss1 && i < ss1 + sc1) return;
        pvx(i) += pax(i) * kdt;
        pvy(i) += pay(i) * kdt;
        pvz(i) += paz(i) * kdt;
    });
    // Kick non-local fast particles
    int ss2 = slice_start_f, sc2 = slice_count_f;
    Kokkos::parallel_for("kick_cross_nl_fast", fast_count, KOKKOS_LAMBDA(int ii) {
        int i = fast_start + ii;
        if (i >= ss2 && i < ss2 + sc2) return;
        pvx(i) += pax(i) * kdt;
        pvy(i) += pay(i) * kdt;
        pvz(i) += paz(i) * kdt;
    });
    Kokkos::fence();
}

// =============================================================================
// Fused cross force + gradient using STALE accelerations.
// Single tiled N² pass per direction (slow←fast, fast←slow) computes both
// cross-force and gradient correction simultaneously.
// Stale accelerations are snapshotted from pax/pay/paz before the kernel runs.
// =============================================================================
static void cross_acceleration_and_gradient_fused_distributed(
    DistributedParticles& dp,
    int slow_start, int slow_count,
    int fast_start, int fast_count,
    double G, double soft2, double grad_dt) {
    if (slow_count <= 0 || fast_count <= 0) return;
    const bool mp = mixed_precision;

    auto pos = dp.local.pos_packed;
    RawView pax{dp.local.ax.data()}, pay{dp.local.ay.data()}, paz{dp.local.az.data()};

    // Snapshot stale accelerations for both groups into scratch buffers.
    // These are the accelerations from the PREVIOUS force evaluation in this HHS level.
    // scratch1/2/3 for slow particles (indexed by ii = i - slow_start)
    // scratch4/5/6 for fast particles (indexed by jj = j - fast_start)
    RawView ax_stale_s{dp.local.scratch1.data()};
    RawView ay_stale_s{dp.local.scratch2.data()};
    RawView az_stale_s{dp.local.scratch3.data()};
    RawView ax_stale_f{dp.local.scratch4.data()};
    RawView ay_stale_f{dp.local.scratch5.data()};
    RawView az_stale_f{dp.local.scratch6.data()};
    Kokkos::parallel_for("snap_stale_slow", slow_count, KOKKOS_LAMBDA(int ii) {
        int i = slow_start + ii;
        ax_stale_s(ii) = pax(i);
        ay_stale_s(ii) = pay(i);
        az_stale_s(ii) = paz(i);
    });
    Kokkos::parallel_for("snap_stale_fast", fast_count, KOKKOS_LAMBDA(int jj) {
        int j = fast_start + jj;
        ax_stale_f(jj) = pax(j);
        ay_stale_f(jj) = pay(j);
        az_stale_f(jj) = paz(j);
    });
    Kokkos::fence();

    const double coeff = G * grad_dt * grad_dt * GRAD_COEFF;

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    // Tile: [x, y, z, m, ax_stale, ay_stale, az_stale] = 7 fields
    size_t scratch_size = TILE_SIZE * 7 * sizeof(double);

    // --- Slow particles: force + gradient from fast particles ---
    int slice_start, slice_count;
    dp.get_force_slice(slow_start, slow_count, slice_start, slice_count);

    if (slice_count > 0) {
        int num_teams = (slice_count + TILE_SIZE - 1) / TILE_SIZE;
        const int f_start = fast_start;
        const int f_end = fast_start + fast_count;
        const int f_off = fast_start;  // offset for stale fast array
        const int s_off = slow_start;  // offset for stale slow array

        Kokkos::parallel_for("cross_fused_slow_from_fast",
            team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_LAMBDA(const member_type& team) {
                int ii = team.league_rank() * TILE_SIZE + team.team_rank();
                int i = slice_start + ii;

                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile(team.team_scratch(0), TILE_SIZE * 7);

                double xi = 0, yi = 0, zi = 0;
                double axi_stale = 0, ayi_stale = 0, azi_stale = 0;
                if (ii < slice_count) {
                    double mi_pad;
                    load4(&pos(i, 0), xi, yi, zi, mi_pad);
                    axi_stale = ax_stale_s(i - s_off);
                    ayi_stale = ay_stale_s(i - s_off);
                    azi_stale = az_stale_s(i - s_off);
                }

                double axi = 0, ayi = 0, azi = 0;
                double cx = 0, cy = 0, cz = 0;

                for (int tile_st = f_start; tile_st < f_end; tile_st += TILE_SIZE) {
                    int tile_end = (tile_st + TILE_SIZE < f_end) ? tile_st + TILE_SIZE : f_end;
                    int tile_cnt = tile_end - tile_st;

                    int t = team.team_rank();
                    if (t < tile_cnt) {
                        int idx = tile_st + t;
                        int tb = t * 7;
                        load4(&pos(idx, 0), tile(tb), tile(tb+1), tile(tb+2), tile(tb+3));
                        tile(tb + 4) = ax_stale_f(idx - f_off);
                        tile(tb + 5) = ay_stale_f(idx - f_off);
                        tile(tb + 6) = az_stale_f(idx - f_off);
                    }
                    team.team_barrier();

                    if (ii < slice_count) {
                        for (int jt = 0; jt < tile_cnt; jt++) {
                            int jb = jt * 7;
                            double dx = tile(jb    ) - xi;
                            double dy = tile(jb + 1) - yi;
                            double dz = tile(jb + 2) - zi;
                            double mj = tile(jb + 3);

                            double r2 = dx*dx + dy*dy + dz*dz + soft2;
                            double inv_r = fast_rsqrt(r2, mp);

                            // Force
                            double inv_r3 = inv_r * inv_r * inv_r;
                            double mj_inv_r3 = G * mj * inv_r3;
                            axi += mj_inv_r3 * dx;
                            ayi += mj_inv_r3 * dy;
                            azi += mj_inv_r3 * dz;

                            // Gradient (stale)
                            double dax = tile(jb + 4) - axi_stale;
                            double day = tile(jb + 5) - ayi_stale;
                            double daz = tile(jb + 6) - azi_stale;
                            double inv_r2 = inv_r * inv_r;
                            double inv_r5 = inv_r2 * inv_r2 * inv_r;
                            double r_dot_a = dx*dax + dy*day + dz*daz;
                            double f = coeff * mj * inv_r5;
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
                }
            });
    }

    // --- Fast particles: force + gradient from slow particles ---
    int slice_start_f, slice_count_f;
    dp.get_force_slice(fast_start, fast_count, slice_start_f, slice_count_f);

    if (slice_count_f > 0) {
        int num_teams_f = (slice_count_f + TILE_SIZE - 1) / TILE_SIZE;
        const int s_start = slow_start;
        const int s_end = slow_start + slow_count;
        const int s_off = slow_start;
        const int f_off = fast_start;

        Kokkos::parallel_for("cross_fused_fast_from_slow",
            team_policy(num_teams_f, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_LAMBDA(const member_type& team) {
                int ii = team.league_rank() * TILE_SIZE + team.team_rank();
                int i = slice_start_f + ii;

                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile(team.team_scratch(0), TILE_SIZE * 7);

                double xi = 0, yi = 0, zi = 0;
                double axi_stale = 0, ayi_stale = 0, azi_stale = 0;
                if (ii < slice_count_f) {
                    double mi_pad;
                    load4(&pos(i, 0), xi, yi, zi, mi_pad);
                    axi_stale = ax_stale_f(i - f_off);
                    ayi_stale = ay_stale_f(i - f_off);
                    azi_stale = az_stale_f(i - f_off);
                }

                double axi = 0, ayi = 0, azi = 0;
                double cx = 0, cy = 0, cz = 0;

                for (int tile_st = s_start; tile_st < s_end; tile_st += TILE_SIZE) {
                    int tile_end = (tile_st + TILE_SIZE < s_end) ? tile_st + TILE_SIZE : s_end;
                    int tile_cnt = tile_end - tile_st;

                    int t = team.team_rank();
                    if (t < tile_cnt) {
                        int idx = tile_st + t;
                        int tb = t * 7;
                        load4(&pos(idx, 0), tile(tb), tile(tb+1), tile(tb+2), tile(tb+3));
                        tile(tb + 4) = ax_stale_s(idx - s_off);
                        tile(tb + 5) = ay_stale_s(idx - s_off);
                        tile(tb + 6) = az_stale_s(idx - s_off);
                    }
                    team.team_barrier();

                    if (ii < slice_count_f) {
                        for (int jt = 0; jt < tile_cnt; jt++) {
                            int jb = jt * 7;
                            double dx = tile(jb    ) - xi;
                            double dy = tile(jb + 1) - yi;
                            double dz = tile(jb + 2) - zi;
                            double mj = tile(jb + 3);

                            double r2 = dx*dx + dy*dy + dz*dz + soft2;
                            double inv_r = fast_rsqrt(r2, mp);

                            // Force
                            double inv_r3 = inv_r * inv_r * inv_r;
                            double mj_inv_r3 = G * mj * inv_r3;
                            axi += mj_inv_r3 * dx;
                            ayi += mj_inv_r3 * dy;
                            azi += mj_inv_r3 * dz;

                            // Gradient (stale)
                            double dax = tile(jb + 4) - axi_stale;
                            double day = tile(jb + 5) - ayi_stale;
                            double daz = tile(jb + 6) - azi_stale;
                            double inv_r2 = inv_r * inv_r;
                            double inv_r5 = inv_r2 * inv_r2 * inv_r;
                            double r_dot_a = dx*dax + dy*day + dz*daz;
                            double f = coeff * mj * inv_r5;
                            cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                            cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                            cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
                        }
                    }
                    team.team_barrier();
                }

                if (ii < slice_count_f) {
                    pax(i) = axi + cx;
                    pay(i) = ayi + cy;
                    paz(i) = azi + cz;
                }
            });
    }
    Kokkos::fence();

    dp.allgather_accelerations_subset(slow_start, slow_count);
    dp.allgather_accelerations_subset(fast_start, fast_count);
}

// Cross-acceleration with gradient correction - tiled GPU kernel
static void cross_acceleration_and_gradient_distributed(DistributedParticles& dp,
                                                        int slow_start, int slow_count,
                                                        int fast_start, int fast_count,
                                                        double G, double soft2, double dt) {
    // First compute cross-accelerations (already tiled)
    cross_acceleration_distributed(dp, slow_start, slow_count,
                                   fast_start, fast_count, G, soft2);

    const bool mp = mixed_precision;
    auto pos = dp.local.pos_packed;
    RawView pax{dp.local.ax.data()}, pay{dp.local.ay.data()}, paz{dp.local.az.data()};

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
                if (ii < slice_count) {
                    double mi_pad;
                    load4(&pos(i, 0), xi, yi, zi, mi_pad);
                    axi = pax(i); ayi = pay(i); azi = paz(i);
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
                            int jb = jt * 7;
                            double dx = tile(jb)   - xi;
                            double dy = tile(jb+1) - yi;
                            double dz = tile(jb+2) - zi;

                            double dax = tile(jb+4) - axi;
                            double day = tile(jb+5) - ayi;
                            double daz = tile(jb+6) - azi;

                            double r2 = dx*dx + dy*dy + dz*dz + soft2;
                            double inv_r = fast_rsqrt(r2, mp);
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
                if (ii < slice_count_f) {
                    double mi_pad;
                    load4(&pos(i, 0), xi, yi, zi, mi_pad);
                    axi = pax(i); ayi = pay(i); azi = paz(i);
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
                            int jb = jt * 7;
                            double dx = tile(jb)   - xi;
                            double dy = tile(jb+1) - yi;
                            double dz = tile(jb+2) - zi;

                            double dax = tile(jb+4) - axi;
                            double day = tile(jb+5) - ayi;
                            double daz = tile(jb+6) - azi;

                            double r2 = dx*dx + dy*dy + dz*dz + soft2;
                            double inv_r = fast_rsqrt(r2, mp);
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
    dp.allgather_accelerations_subset(slow_start, slow_count);
    dp.allgather_accelerations_subset(fast_start, fast_count);
}

// Distributed adaptive timestep computation - tiled GPU kernel
static void compute_timestep_distributed(DistributedParticles& dp, int start, int count,
                                      double G, double soft2, double eta) {
    if (count <= 1) return;
    const bool mp = mixed_precision;

    int slice_start, slice_count;
    dp.get_force_slice(start, count, slice_start, slice_count);

    if (slice_count == 0) {
        dp.allgather_timestep_subset(start, count);
        return;
    }

    auto pos = dp.local.pos_packed;
    auto vel = dp.local.vel_packed;
    RawView dt_f{dp.local.dt_adapt.data()};

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
            if (ii < slice_count) {
                double vpad;
                load4(&pos(i, 0), xi, yi, zi, mi);
                load4(&vel(i, 0), vxi, vyi, vzi, vpad);
            }

            double t_min_sq = 1e60;

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

                        int jb = jt * 7;
                        double dx = tile(jb)   - xi;
                        double dy = tile(jb+1) - yi;
                        double dz = tile(jb+2) - zi;
                        double r2 = dx*dx + dy*dy + dz*dz;

                        double dvx = tile(jb+4) - vxi;
                        double dvy = tile(jb+5) - vyi;
                        double dvz = tile(jb+6) - vzi;
                        double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;

                        // Timestep using rsqrt: r^3 = r2s * r2s * inv_r
                        double r2s = r2 + soft2;
                        double inv_r = fast_rsqrt(r2s, mp);
                        double r3 = r2s * r2s * inv_r;
                        double mass_sum = mi + tile(jb+3);
                        double t_ff_sq = r3 / (G * mass_sum + EPS);
                        double t_fb_sq = r2s / (v_sq + EPS);

                        double t_min_sq_pair = Kokkos::fmin(t_ff_sq, t_fb_sq);
                        t_min_sq = Kokkos::fmin(t_min_sq, t_min_sq_pair);
                    }
                }
                team.team_barrier();
            }

            if (ii < slice_count) {
                dt_f(i) = Kokkos::sqrt(t_min_sq) * eta;
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
    int count;

    void resize(int n) {
        count = n;
        x.resize(n); y.resize(n); z.resize(n);
        vx.resize(n); vy.resize(n); vz.resize(n);
        ax.resize(n); ay.resize(n); az.resize(n);
        mass.resize(n); dt_adapt.resize(n);
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
        Kokkos::deep_copy(h_x, Kokkos::subview(p.x, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::deep_copy(h_y, Kokkos::subview(p.y, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::deep_copy(h_z, Kokkos::subview(p.z, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::deep_copy(h_vx, Kokkos::subview(p.vx, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::deep_copy(h_vy, Kokkos::subview(p.vy, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::deep_copy(h_vz, Kokkos::subview(p.vz, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::deep_copy(h_m, Kokkos::subview(p.mass, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::deep_copy(h_dt, Kokkos::subview(p.dt_adapt, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::fence();
        for (int i = 0; i < count; i++) {
            x[i] = h_x(i); y[i] = h_y(i); z[i] = h_z(i);
            vx[i] = h_vx(i); vy[i] = h_vy(i); vz[i] = h_vz(i);
            mass[i] = h_m(i); dt_adapt[i] = h_dt(i);
        }
    }

    // Copy results back to device (positions, velocities, mass, dt_adapt)
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
        for (int i = 0; i < count; i++) {
            h_x(i) = x[i]; h_y(i) = y[i]; h_z(i) = z[i];
            h_vx(i) = vx[i]; h_vy(i) = vy[i]; h_vz(i) = vz[i];
            h_m(i) = mass[i]; h_dt(i) = dt_adapt[i];
        }
        Kokkos::deep_copy(Kokkos::subview(p.x, Kokkos::pair<int,int>(start, start+count)), h_x);
        Kokkos::deep_copy(Kokkos::subview(p.y, Kokkos::pair<int,int>(start, start+count)), h_y);
        Kokkos::deep_copy(Kokkos::subview(p.z, Kokkos::pair<int,int>(start, start+count)), h_z);
        Kokkos::deep_copy(Kokkos::subview(p.vx, Kokkos::pair<int,int>(start, start+count)), h_vx);
        Kokkos::deep_copy(Kokkos::subview(p.vy, Kokkos::pair<int,int>(start, start+count)), h_vy);
        Kokkos::deep_copy(Kokkos::subview(p.vz, Kokkos::pair<int,int>(start, start+count)), h_vz);
        Kokkos::deep_copy(Kokkos::subview(p.mass, Kokkos::pair<int,int>(start, start+count)), h_m);
        Kokkos::deep_copy(Kokkos::subview(p.dt_adapt, Kokkos::pair<int,int>(start, start+count)), h_dt);
        Kokkos::fence();
    }
};

// CPU self-acceleration + kick (N3L: upper triangle, apply to both i and j)
static void cpu_self_acc_kick(CpuSubset& s, int start, int count,
                              double G, double soft2, double kick_dt) {
    const int end = start + count;
    for (int i = start; i < end; i++) {
        s.ax[i] = 0; s.ay[i] = 0; s.az[i] = 0;
    }
    for (int i = start; i < end; i++) {
        for (int j = i + 1; j < end; j++) {
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
    for (int i = start; i < end; i++) {
        s.vx[i] += s.ax[i] * kick_dt;
        s.vy[i] += s.ay[i] * kick_dt;
        s.vz[i] += s.az[i] * kick_dt;
    }
}

// CPU self-acceleration (N3L, no kick)
static void cpu_self_acc(CpuSubset& s, int start, int count,
                         double G, double soft2) {
    const int end = start + count;
    for (int i = start; i < end; i++) {
        s.ax[i] = 0; s.ay[i] = 0; s.az[i] = 0;
    }
    for (int i = start; i < end; i++) {
        for (int j = i + 1; j < end; j++) {
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

// CPU self-acceleration + gradient + kick (FSI middle step)
static void cpu_self_acc_grad_kick(CpuSubset& s, int start, int count,
                                   double G, double soft2, double grad_dt, double kick_dt) {
    cpu_self_acc(s, start, count, G, soft2);
    cpu_self_grad(s, start, count, G, soft2, grad_dt);
    const int end = start + count;
    for (int i = start; i < end; i++) {
        s.vx[i] += s.ax[i] * kick_dt;
        s.vy[i] += s.ay[i] * kick_dt;
        s.vz[i] += s.az[i] * kick_dt;
    }
}

// CPU cross-acceleration + kick (N3L: each pair computed once, applied to both)
static void cpu_cross_acc_kick(CpuSubset& s,
                               int a_start, int a_count,
                               int b_start, int b_count,
                               double G, double soft2, double kick_dt) {
    const int a_end = a_start + a_count;
    const int b_end = b_start + b_count;
    // Zero accelerations
    for (int i = a_start; i < a_end; i++) { s.ax[i] = 0; s.ay[i] = 0; s.az[i] = 0; }
    for (int j = b_start; j < b_end; j++) { s.ax[j] = 0; s.ay[j] = 0; s.az[j] = 0; }
    // N3L: compute each (i,j) pair once
    for (int i = a_start; i < a_end; i++) {
        for (int j = b_start; j < b_end; j++) {
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
    // Kick both groups
    for (int i = a_start; i < a_end; i++) {
        s.vx[i] += s.ax[i] * kick_dt;
        s.vy[i] += s.ay[i] * kick_dt;
        s.vz[i] += s.az[i] * kick_dt;
    }
    for (int j = b_start; j < b_end; j++) {
        s.vx[j] += s.ax[j] * kick_dt;
        s.vy[j] += s.ay[j] * kick_dt;
        s.vz[j] += s.az[j] * kick_dt;
    }
}

// CPU cross-acceleration + gradient + kick (N3L: each pair once)
static void cpu_cross_acc_grad_kick(CpuSubset& s,
                                    int a_start, int a_count,
                                    int b_start, int b_count,
                                    double G, double soft2, double grad_dt, double kick_dt) {
    const int a_end = a_start + a_count;
    const int b_end = b_start + b_count;
    const double coeff = G * grad_dt * grad_dt * GRAD_COEFF;

    // Cross-acceleration (N3L: each pair once)
    for (int i = a_start; i < a_end; i++) { s.ax[i] = 0; s.ay[i] = 0; s.az[i] = 0; }
    for (int j = b_start; j < b_end; j++) { s.ax[j] = 0; s.ay[j] = 0; s.az[j] = 0; }
    for (int i = a_start; i < a_end; i++) {
        for (int j = b_start; j < b_end; j++) {
            double dx = s.x[j] - s.x[i], dy = s.y[j] - s.y[i], dz = s.z[j] - s.z[i];
            double r2 = dx*dx + dy*dy + dz*dz + soft2;
            double inv_r = 1.0 / std::sqrt(r2);
            double inv_r3 = inv_r * inv_r * inv_r;
            double fj = G * s.mass[j] * inv_r3;
            double fi = G * s.mass[i] * inv_r3;
            s.ax[i] += fj * dx; s.ay[i] += fj * dy; s.az[i] += fj * dz;
            s.ax[j] -= fi * dx; s.ay[j] -= fi * dy; s.az[j] -= fi * dz;
        }
    }

    // Snapshot accelerations for gradient
    std::vector<double> ax0_a(a_count), ay0_a(a_count), az0_a(a_count);
    std::vector<double> ax0_b(b_count), ay0_b(b_count), az0_b(b_count);
    for (int i = 0; i < a_count; i++) {
        ax0_a[i] = s.ax[a_start+i]; ay0_a[i] = s.ay[a_start+i]; az0_a[i] = s.az[a_start+i];
    }
    for (int j = 0; j < b_count; j++) {
        ax0_b[j] = s.ax[b_start+j]; ay0_b[j] = s.ay[b_start+j]; az0_b[j] = s.az[b_start+j];
    }

    // Gradient correction (N3L: each pair once)
    for (int ii = 0; ii < a_count; ii++) {
        int i = a_start + ii;
        for (int jj = 0; jj < b_count; jj++) {
            int j = b_start + jj;
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

    // Kick both groups
    for (int i = a_start; i < a_end; i++) {
        s.vx[i] += s.ax[i] * kick_dt;
        s.vy[i] += s.ay[i] * kick_dt;
        s.vz[i] += s.az[i] * kick_dt;
    }
    for (int j = b_start; j < b_end; j++) {
        s.vx[j] += s.ax[j] * kick_dt;
        s.vy[j] += s.ay[j] * kick_dt;
        s.vz[j] += s.az[j] * kick_dt;
    }
}

// CPU drift
static void cpu_drift(CpuSubset& s, int start, int count, double dt) {
    for (int i = start; i < start + count; i++) {
        s.x[i] += s.vx[i] * dt;
        s.y[i] += s.vy[i] * dt;
        s.z[i] += s.vz[i] * dt;
    }
}

// CPU timestep computation (N3L: upper triangle, squared-space)
static void cpu_compute_timestep(CpuSubset& s, int start, int count,
                                 double G, double soft2, double eta) {
    const int end = start + count;
    for (int i = start; i < end; i++) s.dt_adapt[i] = 1e60;
    for (int i = start; i < end; i++) {
        for (int j = i + 1; j < end; j++) {
            double dx = s.x[j] - s.x[i];
            double dy = s.y[j] - s.y[i];
            double dz = s.z[j] - s.z[i];
            double dvx = s.vx[j] - s.vx[i];
            double dvy = s.vy[j] - s.vy[i];
            double dvz = s.vz[j] - s.vz[i];
            double r2 = dx*dx + dy*dy + dz*dz;
            double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
            double r2_eps = r2 + EPS;
            double r_norm = std::sqrt(r2_eps);
            double mass_sum = s.mass[i] + s.mass[j];
            double t_ff_sq = r2_eps * r_norm / (G * mass_sum + EPS);
            double t_fb_sq = r2_eps / (v_sq + EPS);
            double t_min_sq_pair = std::min(t_ff_sq, t_fb_sq);
            s.dt_adapt[i] = std::min(s.dt_adapt[i], t_min_sq_pair);
            s.dt_adapt[j] = std::min(s.dt_adapt[j], t_min_sq_pair);
        }
    }
    for (int i = start; i < end; i++) {
        s.dt_adapt[i] = std::sqrt(s.dt_adapt[i]) * eta;
    }
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
    int si = 0, fi = slow_count;
    for (int i = 0; i < count; i++) {
        int src = start + i;
        int dst = (s.dt_adapt[src] >= pivot) ? si++ : fi++;
        tx[dst] = s.x[src]; ty[dst] = s.y[src]; tz[dst] = s.z[src];
        tvx[dst] = s.vx[src]; tvy[dst] = s.vy[src]; tvz[dst] = s.vz[src];
        tax[dst] = s.ax[src]; tay[dst] = s.ay[src]; taz[dst] = s.az[src];
        tm[dst] = s.mass[src]; tdt[dst] = s.dt_adapt[src];
    }
    for (int i = 0; i < count; i++) {
        int dst = start + i;
        s.x[dst] = tx[i]; s.y[dst] = ty[i]; s.z[dst] = tz[i];
        s.vx[dst] = tvx[i]; s.vy[dst] = tvy[i]; s.vz[dst] = tvz[i];
        s.ax[dst] = tax[i]; s.ay[dst] = tay[i]; s.az[dst] = taz[i];
        s.mass[dst] = tm[i]; s.dt_adapt[dst] = tdt[i];
    }
    return slow_count;
}

// CPU FSI step
static void cpu_fsi_step(CpuSubset& s, int start, int count,
                         double G, double soft2, double dt) {
    cpu_self_acc_kick(s, start, count, G, soft2, dt * FSI_KICK);
    cpu_drift(s, start, count, dt * FSI_DRIFT);
    cpu_self_acc_grad_kick(s, start, count, G, soft2, dt, dt * FSI_MOD_KICK);
    cpu_drift(s, start, count, dt * FSI_DRIFT);
    cpu_self_acc_kick(s, start, count, G, soft2, dt * FSI_KICK);
}

// CPU HHS-FSI recursion
static void cpu_hhs_recursive(CpuSubset& s, int start, int count,
                              double pivot_dt, double G, double soft2,
                              double dt_min, double eta, bool dt_valid) {
    if (count == 0) return;

    if (pivot_dt * 0.5 < dt_min) {
        cpu_fsi_step(s, start, count, G, soft2, pivot_dt);
        return;
    }

    if (!dt_valid && count > 1) {
        cpu_compute_timestep(s, start, count, G, soft2, eta);
    }

    int slow_count = cpu_partition_by_pivot(s, start, count, pivot_dt);
    int slow_start = start;
    int fast_start = start + slow_count;
    int fast_count = count - slow_count;

    if (fast_count == 0) {
        cpu_fsi_step(s, start, count, G, soft2, pivot_dt);
        return;
    }

    if (slow_count == 0) {
        double sub_pivot = pivot_dt * 0.5;
        cpu_hhs_recursive(s, start, count, sub_pivot, G, soft2, dt_min, eta, false);
        cpu_hhs_recursive(s, start, count, sub_pivot, G, soft2, dt_min, eta, skip_second_timestep);
        return;
    }

    // Step 1: Cross-kick (dt/6)
    cpu_cross_acc_kick(s, slow_start, slow_count, fast_start, fast_count, G, soft2, pivot_dt * FSI_KICK);

    // Step 2: FSI half-step on slow
    cpu_fsi_step(s, slow_start, slow_count, G, soft2, pivot_dt * 0.5);

    // Step 3: Recurse on fast (first half)
    cpu_hhs_recursive(s, fast_start, fast_count, pivot_dt * 0.5, G, soft2, dt_min, eta, false);

    // Step 4: Cross-gradient kick (2dt/3)
    cpu_cross_acc_grad_kick(s, slow_start, slow_count, fast_start, fast_count, G, soft2, pivot_dt, pivot_dt * FSI_MOD_KICK);

    // Step 5: Recurse on fast (second half)
    cpu_hhs_recursive(s, fast_start, fast_count, pivot_dt * 0.5, G, soft2, dt_min, eta, skip_second_timestep);

    // Step 6: FSI half-step on slow
    cpu_fsi_step(s, slow_start, slow_count, G, soft2, pivot_dt * 0.5);

    // Step 7: Cross-kick (dt/6)
    cpu_cross_acc_kick(s, slow_start, slow_count, fast_start, fast_count, G, soft2, pivot_dt * FSI_KICK);
}

// Entry point: copy subset to host, run CPU HHS, copy back
static void hhs_fsi_recursive_cpu(Particles& p, int start, int count,
                                  double pivot_dt, double G, double soft2,
                                  double dt_min, double eta, bool dt_valid) {
    CpuSubset s;
    s.resize(count);
    s.copy_from_device(p, start);
    cpu_hhs_recursive(s, 0, count, pivot_dt, G, soft2, dt_min, eta, dt_valid);
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
    bool use_n3l, bool dt_valid) {

    if (count == 0) return;

    // CPU serial path for very small subsets
    if (cpu_serial_threshold > 0 && count <= cpu_serial_threshold) {
        hhs_fsi_recursive_cpu(p, start, count, pivot_dt, G, soft2, dt_min, eta, dt_valid);
        return;
    }

    // Base case: at minimum timestep, use FSI on the subset
    if (pivot_dt * 0.5 < dt_min) {
        fsi_step(p, start, count, G, soft2, pivot_dt);
        return;
    }

    // Compute adaptive timesteps for this subset
    if (!dt_valid && count > 1) {
        g_profiler.start();
        if (use_n3l)
            compute_timestep_subset<true>(p, start, count, G, soft2, eta);
        else
            compute_timestep_subset<false>(p, start, count, G, soft2, eta);
        g_profiler.stop(StepProfiler::ADAPTIVE_DT);
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
        fsi_step(p, start, count, G, soft2, pivot_dt);
        return;
    }

    // Base case: all particles are fast
    if (slow_count == 0) {
        double sub_pivot = pivot_dt * 0.5;
        hhs_fsi_recursive_local(p, start, count, sub_pivot,
                                G, soft2, dt_min, eta, use_n3l, false);
        hhs_fsi_recursive_local(p, start, count, sub_pivot,
                                G, soft2, dt_min, eta, use_n3l, skip_second_timestep);
        return;
    }

    // Step 1: Cross-kick S<->F (dt/6) — no caching
    g_profiler.start();
    fn_cross_acc_kick(p, slow_start, slow_count, fast_start, fast_count, G, soft2, pivot_dt * FSI_KICK);
    g_profiler.stop(StepProfiler::CROSS_FORCE);

    // Step 2: FSI half-step on slow
    fsi_step(p, slow_start, slow_count, G, soft2, pivot_dt * 0.5);

    // Step 3: Recurse on fast (first half)
    hhs_fsi_recursive_local(p, fast_start, fast_count, pivot_dt * 0.5,
                            G, soft2, dt_min, eta, use_n3l, false);

    // Step 4: Cross-gradient kick (2dt/3)
    g_profiler.start();
    fn_cross_acc_grad_kick(p, slow_start, slow_count, fast_start, fast_count, G, soft2, pivot_dt, pivot_dt * FSI_MOD_KICK);
    g_profiler.stop(StepProfiler::CROSS_GRAD);

    // Step 5: Recurse on fast (second half)
    hhs_fsi_recursive_local(p, fast_start, fast_count, pivot_dt * 0.5,
                            G, soft2, dt_min, eta, use_n3l, skip_second_timestep);

    // Step 6: FSI half-step on slow
    fsi_step(p, slow_start, slow_count, G, soft2, pivot_dt * 0.5);

    // Step 7: Cross-kick S<->F (dt/6) — no caching
    g_profiler.start();
    fn_cross_acc_kick(p, slow_start, slow_count, fast_start, fast_count, G, soft2, pivot_dt * FSI_KICK);
    g_profiler.stop(StepProfiler::CROSS_FORCE);
}

// =============================================================================
// Recursive distributed HHS-FSI implementation (Algorithm 4 from FROST paper)
// NEW: Scatter-compute-gather paradigm
//
// All ranks run IDENTICAL code for partitioning, drifts, kicks.
// Only force computation is distributed, with Allgather after each.
// This achieves O(N) communication per step regardless of HHS depth.
// =============================================================================
static void hhs_fsi_recursive_distributed(
    DistributedParticles& dp,
    int start, int count,           // Subset of particles to integrate
    double pivot_dt,                // Current pivot timestep
    double G, double soft2,
    double dt_min, double eta,
    bool use_n3l, bool dt_valid) {

    if (count == 0) return;

    // Below threshold: skip MPI, all ranks compute full N² locally
    if (direct_n2_threshold > 0 && count <= direct_n2_threshold) {
        hhs_fsi_recursive_local(dp.local, start, count, pivot_dt,
                                G, soft2, dt_min, eta, use_n3l, dt_valid);
        // Re-sync packed views: local path modifies SoA directly without updating pos_packed
        dp.local.pack_pos_range(start, count);
        return;
    }

    (void)use_n3l;  // All ranks run identical code, N3L handled in single-GPU path

    // Base case: at minimum timestep, use FSI on the subset
    if (pivot_dt * 0.5 < dt_min) {
        fsi_step_distributed(dp, start, count, G, soft2, pivot_dt);
        return;
    }

    // Compute adaptive timesteps for this subset (skip if reusing from first half-step)
    if (!dt_valid && count > 1) {
        // Sync packed pos+vel for vectorized timestep kernel
        dp.local.pack_pos_range(start, count);
        dp.local.pack_vel_range(start, count);
        g_profiler.start();
        compute_timestep_distributed(dp, start, count, G, soft2, eta);
        g_profiler.stop(StepProfiler::ADAPTIVE_DT, (long long)count * count);
    }

    // Partition into slow (dt >= pivot) and fast (dt < pivot)
    // All ranks do identical partitioning (deterministic)
    g_profiler.start();
    int slow_count = partition_by_pivot(dp.local, start, count, pivot_dt);
    g_profiler.stop(StepProfiler::PARTITION);
    int slow_start = start;
    int fast_start = start + slow_count;
    int fast_count = count - slow_count;

    // Base case: all particles are slow (no fast particles)
    if (fast_count == 0) {
        fsi_step_distributed(dp, start, count, G, soft2, pivot_dt);
        return;
    }

    // Base case: all particles are fast (no slow particles)
    if (slow_count == 0) {
        double sub_pivot = pivot_dt * 0.5;
        hhs_fsi_recursive_distributed(dp, start, count, sub_pivot,
                                      G, soft2, dt_min, eta, false, false);
        hhs_fsi_recursive_distributed(dp, start, count, sub_pivot,
                                      G, soft2, dt_min, eta, false, skip_second_timestep);
        return;
    }

    long long cross_pairs = 2LL * slow_count * fast_count;

    // Re-sync packed positions after partition reorder
    dp.local.pack_pos_range(start, count);

    // ==========================================================================
    // ALGORITHM 4: HHS-FSI with slow (S) and fast (F) particles
    // ==========================================================================

    // Step 1: Cross-kick S<->F (dt/6) - fused distributed force + allgather + kick
    g_profiler.start();
    cross_acceleration_kick_distributed(dp, slow_start, slow_count, fast_start, fast_count, G, soft2, pivot_dt * FSI_KICK);
    g_profiler.stop(StepProfiler::CROSS_FORCE, cross_pairs);

    // Step 2: FSI half-step on slow
    fsi_step_distributed(dp, slow_start, slow_count, G, soft2, pivot_dt * 0.5);

    // Step 3: Recurse on fast (first half) — always compute fresh timesteps
    hhs_fsi_recursive_distributed(dp, fast_start, fast_count, pivot_dt * 0.5,
                                  G, soft2, dt_min, eta, false, false);

    // Step 4: Cross-kick S<->F with gradient (2dt/3)
    if (fuse_gradient) {
        // Fused: single N² pass per direction computes force + gradient (stale a)
        g_profiler.start();
        cross_acceleration_and_gradient_fused_distributed(dp, slow_start, slow_count,
                                                          fast_start, fast_count,
                                                          G, soft2, pivot_dt);
        g_profiler.stop(StepProfiler::CROSS_GRAD, cross_pairs);
    } else {
        // Separate: force → allgather → gradient
        g_profiler.start();
        cross_acceleration_and_gradient_distributed(dp, slow_start, slow_count,
                                                    fast_start, fast_count,
                                                    G, soft2, pivot_dt);
        g_profiler.stop(StepProfiler::CROSS_GRAD, cross_pairs * 2);
    }
    g_profiler.start();
    kick_velocities(dp.local, slow_start, slow_count, pivot_dt * FSI_MOD_KICK);
    kick_velocities(dp.local, fast_start, fast_count, pivot_dt * FSI_MOD_KICK);
    g_profiler.stop(StepProfiler::DRIFT_KICK);

    // Step 5: Recurse on fast (second half)
    {
        double sub_pivot = pivot_dt * 0.5;
        hhs_fsi_recursive_distributed(dp, fast_start, fast_count, sub_pivot,
                                      G, soft2, dt_min, eta, false, skip_second_timestep);
    }

    // Step 6: FSI half-step on slow
    fsi_step_distributed(dp, slow_start, slow_count, G, soft2, pivot_dt * 0.5);

    // Step 7: Cross-kick S<->F (dt/6) - fused distributed force + allgather + kick
    g_profiler.start();
    cross_acceleration_kick_distributed(dp, slow_start, slow_count, fast_start, fast_count, G, soft2, pivot_dt * FSI_KICK);
    g_profiler.stop(StepProfiler::CROSS_FORCE, cross_pairs);
}

// Distributed HHS-FSI - main entry point
void hhs_fsi_step_distributed(DistributedParticles& dp, double pivot_dt,
                              double G, double soft2, double dt_min,
                              double eta, bool use_n3l, int global_levels) {
    if (dp.N_total == 0) return;

    (void)global_levels;  // Not used in new implementation

#ifdef USE_MPI
    bool is_distributed = MPIComm::size() > 1;
#else
    bool is_distributed = false;
#endif

    if (!is_distributed) {
        // Single rank: use standard HHS (uses efficient fused kernels)
        hhs_fsi_step_dynamic(dp.local, 0, dp.N_total, pivot_dt, G, soft2, dt_min, eta, use_n3l);
        return;
    }

#ifdef USE_MPI
    // In replicated mode, all ranks have all particles
    // Just run the recursive implementation - all ranks execute identical code
    // Only force computation is distributed (via Allgather after each)
    hhs_fsi_recursive_distributed(dp, 0, dp.N_total, pivot_dt,
                                  G, soft2, dt_min, eta, use_n3l, false);
#endif
}

