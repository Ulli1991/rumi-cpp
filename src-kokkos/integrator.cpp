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

// Tile size for shared memory tiling (Rantala paper uses p=32)
constexpr int DIST_TILE_SIZE = 32;

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

    // Pack data for vectorized memory access
    dp.local.pack();

    auto pos = dp.local.pos_packed;
    auto vel = dp.local.vel_packed;
    auto pax = dp.local.ax;
    auto pay = dp.local.ay;
    auto paz = dp.local.az;
    auto dt_f = dp.local.dt_adapt;

    // Tiled GPU kernel: each team handles DIST_TILE_SIZE i-particles from my slice
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t scratch_size = DIST_TILE_SIZE * 7 * sizeof(double);
    int num_teams = (my_count + DIST_TILE_SIZE - 1) / DIST_TILE_SIZE;

    Kokkos::parallel_for("dist_acc_dt_tiled",
        team_policy(num_teams, DIST_TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            // Each thread handles one i-particle from my slice
            int ii = team.league_rank() * DIST_TILE_SIZE + team.team_rank();
            int i = my_start + ii;

            // Scratch memory for j-particle tiles
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_x(team.team_scratch(0), DIST_TILE_SIZE);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_y(reinterpret_cast<double*>(tile_x.data() + DIST_TILE_SIZE), DIST_TILE_SIZE);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_z(reinterpret_cast<double*>(tile_y.data() + DIST_TILE_SIZE), DIST_TILE_SIZE);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_vx(reinterpret_cast<double*>(tile_z.data() + DIST_TILE_SIZE), DIST_TILE_SIZE);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_vy(reinterpret_cast<double*>(tile_vx.data() + DIST_TILE_SIZE), DIST_TILE_SIZE);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_vz(reinterpret_cast<double*>(tile_vy.data() + DIST_TILE_SIZE), DIST_TILE_SIZE);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_m(reinterpret_cast<double*>(tile_vz.data() + DIST_TILE_SIZE), DIST_TILE_SIZE);

            // Load i-particle data
            double xi = 0, yi = 0, zi = 0, mi = 0;
            double vxi = 0, vyi = 0, vzi = 0;
            if (ii < my_count) {
                xi = pos(i, 0); yi = pos(i, 1); zi = pos(i, 2); mi = pos(i, 3);
                vxi = vel(i, 0); vyi = vel(i, 1); vzi = vel(i, 2);
            }

            double axi = 0, ayi = 0, azi = 0;
            double t_min_i = DBL_MAX;

            // Loop over ALL j-particles in tiles
            for (int tile_start = 0; tile_start < N; tile_start += DIST_TILE_SIZE) {
                int tile_end = (tile_start + DIST_TILE_SIZE < N) ? tile_start + DIST_TILE_SIZE : N;
                int tile_count = tile_end - tile_start;

                // Collaborative tile load
                int t = team.team_rank();
                if (t < tile_count) {
                    int idx = tile_start + t;
                    tile_x(t) = pos(idx, 0);
                    tile_y(t) = pos(idx, 1);
                    tile_z(t) = pos(idx, 2);
                    tile_m(t) = pos(idx, 3);
                    tile_vx(t) = vel(idx, 0);
                    tile_vy(t) = vel(idx, 1);
                    tile_vz(t) = vel(idx, 2);
                }
                team.team_barrier();

                // Compute forces and adaptive timestep from this tile
                if (ii < my_count) {
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
                        axi += fj * dx;
                        ayi += fj * dy;
                        azi += fj * dz;

                        // Adaptive timestep (squared-space: 1 sqrt instead of 5)
                        double dvx = tile_vx(jt) - vxi;
                        double dvy = tile_vy(jt) - vyi;
                        double dvz = tile_vz(jt) - vzi;
                        double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
                        double r2_eps = r2 + EPS;
                        double r_norm = Kokkos::sqrt(r2_eps);
                        double mass_sum = mi + tile_m(jt);

                        double t_ff_sq = r2_eps * r_norm / (G * mass_sum + EPS);
                        double t_fb_sq = r2_eps / (v_sq + EPS);
                        double t_min_sq_pair = Kokkos::fmin(t_ff_sq, t_fb_sq);
                        t_min_i = Kokkos::fmin(t_min_i, t_min_sq_pair);
                    }
                }
                team.team_barrier();
            }

            // Write results (sqrt from squared-space + eta)
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
    auto px = p.x; auto py = p.y; auto pz = p.z;
    auto pvx = p.vx; auto pvy = p.vy; auto pvz = p.vz;
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

    auto dt_f = p.dt_adapt;
    auto perm = p.scratch_i;
    auto is_slow = p.scratch_i2;  // temporary: 1 if slow, 0 if fast

    // Step 1: Mark each particle as slow (1) or fast (0)
    Kokkos::parallel_for("mark_slow", count, KOKKOS_LAMBDA(int ii) {
        is_slow(ii) = (dt_f(start + ii) >= pivot) ? 1 : 0;
    });
    Kokkos::fence();

    // Step 2: Exclusive prefix sum to get destination indices for slow particles
    // slow_prefix[i] = number of slow particles with index < i
    auto slow_prefix = p.scratch_i3;
    Kokkos::parallel_scan("prefix_slow", count,
        KOKKOS_LAMBDA(int ii, int& partial_sum, bool is_final) {
            if (is_final) slow_prefix(ii) = partial_sum;
            partial_sum += is_slow(ii);
        });
    Kokkos::fence();

    // Get total slow count (last prefix + last is_slow value)
    int slow_count = 0;
    Kokkos::parallel_reduce("total_slow", 1,
        KOKKOS_LAMBDA(int, int& sum) {
            sum = slow_prefix(count - 1) + is_slow(count - 1);
        }, slow_count);
    Kokkos::fence();

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
    auto px = p.x; auto py = p.y; auto pz = p.z;
    auto pvx = p.vx; auto pvy = p.vy; auto pvz = p.vz;
    auto pax = p.ax; auto pay = p.ay; auto paz = p.az;
    auto pmass = p.mass; auto pdt = p.dt_adapt;
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
    auto pvx = p.vx; auto pvy = p.vy; auto pvz = p.vz;
    auto pax = p.ax; auto pay = p.ay; auto paz = p.az;

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

    auto px = dp.local.x; auto py = dp.local.y; auto pz = dp.local.z;
    auto pax = dp.local.ax; auto pay = dp.local.ay; auto paz = dp.local.az;
    auto pm = dp.local.mass;

    const int j_start = start;
    const int j_end = start + count;

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t scratch_size = DIST_TILE_SIZE * 4 * sizeof(double); // x,y,z,m
    int num_teams = (slice_count + DIST_TILE_SIZE - 1) / DIST_TILE_SIZE;

    Kokkos::parallel_for("dist_self_acc_tiled",
        team_policy(num_teams, DIST_TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * DIST_TILE_SIZE + team.team_rank();
            int i = slice_start + ii;

            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_x(team.team_scratch(0), DIST_TILE_SIZE);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_y(tile_x.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_z(tile_y.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_m(tile_z.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);

            double xi = 0, yi = 0, zi = 0;
            if (ii < slice_count) {
                xi = px(i); yi = py(i); zi = pz(i);
            }

            double axi = 0, ayi = 0, azi = 0;

            for (int tile_start = j_start; tile_start < j_end; tile_start += DIST_TILE_SIZE) {
                int tile_end = (tile_start + DIST_TILE_SIZE < j_end) ? tile_start + DIST_TILE_SIZE : j_end;
                int tile_count = tile_end - tile_start;

                int t = team.team_rank();
                if (t < tile_count) {
                    int idx = tile_start + t;
                    tile_x(t) = px(idx);
                    tile_y(t) = py(idx);
                    tile_z(t) = pz(idx);
                    tile_m(t) = pm(idx);
                }
                team.team_barrier();

                if (ii < slice_count) {
                    for (int jt = 0; jt < tile_count; jt++) {
                        int j = tile_start + jt;
                        if (j == i) continue;

                        double dx = tile_x(jt) - xi;
                        double dy = tile_y(jt) - yi;
                        double dz = tile_z(jt) - zi;

                        double r2 = dx*dx + dy*dy + dz*dz + soft2;
                        double inv_r = 1.0 / Kokkos::sqrt(r2);
                        double inv_r3 = inv_r * inv_r * inv_r;
                        double mj_inv_r3 = G * tile_m(jt) * inv_r3;

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

    int slice_start, slice_count;
    dp.get_force_slice(start, count, slice_start, slice_count);
    if (slice_count == 0) {
        dp.allgather_accelerations_subset(start, count);
        kick_velocities(dp.local, start, count, kick_dt);
        return;
    }

    auto px = dp.local.x; auto py = dp.local.y; auto pz = dp.local.z;
    auto pvx = dp.local.vx; auto pvy = dp.local.vy; auto pvz = dp.local.vz;
    auto pax = dp.local.ax; auto pay = dp.local.ay; auto paz = dp.local.az;
    auto pm = dp.local.mass;

    const int j_start = start;
    const int j_end = start + count;
    const double kdt = kick_dt;

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t scratch_size = DIST_TILE_SIZE * 4 * sizeof(double);
    int num_teams = (slice_count + DIST_TILE_SIZE - 1) / DIST_TILE_SIZE;

    Kokkos::parallel_for("dist_self_acc_kick_tiled",
        team_policy(num_teams, DIST_TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * DIST_TILE_SIZE + team.team_rank();
            int i = slice_start + ii;

            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_x(team.team_scratch(0), DIST_TILE_SIZE);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_y(tile_x.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_z(tile_y.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_m(tile_z.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);

            double xi = 0, yi = 0, zi = 0;
            if (ii < slice_count) {
                xi = px(i); yi = py(i); zi = pz(i);
            }

            double axi = 0, ayi = 0, azi = 0;

            for (int tile_start = j_start; tile_start < j_end; tile_start += DIST_TILE_SIZE) {
                int tile_end = (tile_start + DIST_TILE_SIZE < j_end) ? tile_start + DIST_TILE_SIZE : j_end;
                int tile_count = tile_end - tile_start;

                int t = team.team_rank();
                if (t < tile_count) {
                    int idx = tile_start + t;
                    tile_x(t) = px(idx);
                    tile_y(t) = py(idx);
                    tile_z(t) = pz(idx);
                    tile_m(t) = pm(idx);
                }
                team.team_barrier();

                if (ii < slice_count) {
                    for (int jt = 0; jt < tile_count; jt++) {
                        int j = tile_start + jt;
                        if (j == i) continue;

                        double dx = tile_x(jt) - xi;
                        double dy = tile_y(jt) - yi;
                        double dz = tile_z(jt) - zi;

                        double r2 = dx*dx + dy*dy + dz*dz + soft2;
                        double inv_r = 1.0 / Kokkos::sqrt(r2);
                        double inv_r3 = inv_r * inv_r * inv_r;
                        double mj_inv_r3 = G * tile_m(jt) * inv_r3;

                        axi += mj_inv_r3 * dx;
                        ayi += mj_inv_r3 * dy;
                        azi += mj_inv_r3 * dz;
                    }
                }
                team.team_barrier();
            }

            if (ii < slice_count) {
                // Store forces (needed for allgather)
                pax(i) = axi;
                pay(i) = ayi;
                paz(i) = azi;
                // Fused kick directly from registers
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

    int slice_start, slice_count;
    dp.get_force_slice(start, count, slice_start, slice_count);
    if (slice_count == 0) {
        dp.allgather_accelerations_subset(start, count);
        kick_velocities(dp.local, start, count, kick_dt);
        return;
    }

    auto px = dp.local.x; auto py = dp.local.y; auto pz = dp.local.z;
    auto pvx = dp.local.vx; auto pvy = dp.local.vy; auto pvz = dp.local.vz;
    auto pax = dp.local.ax; auto pay = dp.local.ay; auto paz = dp.local.az;
    auto pm = dp.local.mass;

    const double coeff = G * grad_dt * grad_dt * GRAD_COEFF;
    const int j_start = start;
    const int j_end = start + count;
    const double kdt = kick_dt;

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t scratch_size = DIST_TILE_SIZE * 7 * sizeof(double);
    int num_teams = (slice_count + DIST_TILE_SIZE - 1) / DIST_TILE_SIZE;

    Kokkos::parallel_for("dist_grad_kick_tiled",
        team_policy(num_teams, DIST_TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * DIST_TILE_SIZE + team.team_rank();
            int i = slice_start + ii;

            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_x(team.team_scratch(0), DIST_TILE_SIZE);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_y(tile_x.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_z(tile_y.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_m(tile_z.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_ax(tile_m.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_ay(tile_ax.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_az(tile_ay.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);

            double xi = 0, yi = 0, zi = 0;
            double axi = 0, ayi = 0, azi = 0;
            if (ii < slice_count) {
                xi = px(i); yi = py(i); zi = pz(i);
                axi = pax(i); ayi = pay(i); azi = paz(i);
            }

            double cx = 0, cy = 0, cz = 0;

            for (int tile_start = j_start; tile_start < j_end; tile_start += DIST_TILE_SIZE) {
                int tile_end = (tile_start + DIST_TILE_SIZE < j_end) ? tile_start + DIST_TILE_SIZE : j_end;
                int tile_count = tile_end - tile_start;

                int t = team.team_rank();
                if (t < tile_count) {
                    int idx = tile_start + t;
                    tile_x(t) = px(idx);
                    tile_y(t) = py(idx);
                    tile_z(t) = pz(idx);
                    tile_m(t) = pm(idx);
                    tile_ax(t) = pax(idx);
                    tile_ay(t) = pay(idx);
                    tile_az(t) = paz(idx);
                }
                team.team_barrier();

                if (ii < slice_count) {
                    for (int jt = 0; jt < tile_count; jt++) {
                        int j = tile_start + jt;
                        if (j == i) continue;

                        double dx = tile_x(jt) - xi;
                        double dy = tile_y(jt) - yi;
                        double dz = tile_z(jt) - zi;

                        double dax = tile_ax(jt) - axi;
                        double day = tile_ay(jt) - ayi;
                        double daz = tile_az(jt) - azi;

                        double r2 = dx*dx + dy*dy + dz*dz + soft2;
                        double inv_r2 = 1.0 / r2;
                        double inv_r = 1.0 / Kokkos::sqrt(r2);
                        double inv_r5 = inv_r2 * inv_r2 * inv_r;

                        double r_dot_a = dx*dax + dy*day + dz*daz;
                        double f = coeff * tile_m(jt) * inv_r5;

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
// Distributed FSI step using scatter-compute-gather
// All ranks execute identical code; only force computation is distributed
// Uses fused force+kick kernels to reduce kernel launches and memory traffic
// =============================================================================
static void fsi_step_distributed(DistributedParticles& dp, int start, int count,
                                 double G, double soft2, double dt) {
    if (count <= 0) return;

    // U kick (dt/6) - fused distributed force + kick
    g_profiler.start();
    self_acceleration_kick_distributed(dp, start, count, G, soft2, dt * FSI_KICK);
    g_profiler.stop(StepProfiler::SELF_FORCE);

    // T drift (dt/2) - all ranks drift identically
    g_profiler.start();
    drift(dp.local, start, count, dt * FSI_DRIFT);
    g_profiler.stop(StepProfiler::DRIFT_KICK);

    // U-tilde kick (2dt/3) with gradient correction - force then fused gradient + kick
    g_profiler.start();
    self_acceleration_distributed(dp, start, count, G, soft2);
    g_profiler.stop(StepProfiler::SELF_FORCE);
    g_profiler.start();
    self_gradient_correction_kick_distributed(dp, start, count, G, soft2, dt, dt * FSI_MOD_KICK);
    g_profiler.stop(StepProfiler::SELF_GRAD);

    // T drift (dt/2)
    g_profiler.start();
    drift(dp.local, start, count, dt * FSI_DRIFT);
    g_profiler.stop(StepProfiler::DRIFT_KICK);

    // U kick (dt/6) - fused distributed force + kick
    g_profiler.start();
    self_acceleration_kick_distributed(dp, start, count, G, soft2, dt * FSI_KICK);
    g_profiler.stop(StepProfiler::SELF_FORCE);
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

    auto px = dp.local.x; auto py = dp.local.y; auto pz = dp.local.z;
    auto pax = dp.local.ax; auto pay = dp.local.ay; auto paz = dp.local.az;
    auto pm = dp.local.mass;

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t scratch_size = DIST_TILE_SIZE * 4 * sizeof(double);

    // Slow particles: compute force from fast particles
    int slice_start, slice_count;
    dp.get_force_slice(slow_start, slow_count, slice_start, slice_count);

    if (slice_count > 0) {
        int num_teams = (slice_count + DIST_TILE_SIZE - 1) / DIST_TILE_SIZE;
        const int f_start = fast_start;
        const int f_end = fast_start + fast_count;

        Kokkos::parallel_for("cross_slow_from_fast_tiled",
            team_policy(num_teams, DIST_TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_LAMBDA(const member_type& team) {
                int ii = team.league_rank() * DIST_TILE_SIZE + team.team_rank();
                int i = slice_start + ii;

                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_x(team.team_scratch(0), DIST_TILE_SIZE);
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_y(tile_x.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_z(tile_y.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_m(tile_z.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);

                double xi = 0, yi = 0, zi = 0;
                if (ii < slice_count) {
                    xi = px(i); yi = py(i); zi = pz(i);
                }

                double axi = 0, ayi = 0, azi = 0;

                for (int tile_st = f_start; tile_st < f_end; tile_st += DIST_TILE_SIZE) {
                    int tile_end = (tile_st + DIST_TILE_SIZE < f_end) ? tile_st + DIST_TILE_SIZE : f_end;
                    int tile_cnt = tile_end - tile_st;

                    int t = team.team_rank();
                    if (t < tile_cnt) {
                        int idx = tile_st + t;
                        tile_x(t) = px(idx);
                        tile_y(t) = py(idx);
                        tile_z(t) = pz(idx);
                        tile_m(t) = pm(idx);
                    }
                    team.team_barrier();

                    if (ii < slice_count) {
                        for (int jt = 0; jt < tile_cnt; jt++) {
                            double dx = tile_x(jt) - xi;
                            double dy = tile_y(jt) - yi;
                            double dz = tile_z(jt) - zi;

                            double r2 = dx*dx + dy*dy + dz*dz + soft2;
                            double inv_r = 1.0 / Kokkos::sqrt(r2);
                            double inv_r3 = inv_r * inv_r * inv_r;
                            double mj_inv_r3 = G * tile_m(jt) * inv_r3;

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
        int num_teams_f = (slice_count_f + DIST_TILE_SIZE - 1) / DIST_TILE_SIZE;
        const int s_start = slow_start;
        const int s_end = slow_start + slow_count;

        Kokkos::parallel_for("cross_fast_from_slow_tiled",
            team_policy(num_teams_f, DIST_TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_LAMBDA(const member_type& team) {
                int ii = team.league_rank() * DIST_TILE_SIZE + team.team_rank();
                int i = slice_start_f + ii;

                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_x(team.team_scratch(0), DIST_TILE_SIZE);
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_y(tile_x.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_z(tile_y.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_m(tile_z.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);

                double xi = 0, yi = 0, zi = 0;
                if (ii < slice_count_f) {
                    xi = px(i); yi = py(i); zi = pz(i);
                }

                double axi = 0, ayi = 0, azi = 0;

                for (int tile_st = s_start; tile_st < s_end; tile_st += DIST_TILE_SIZE) {
                    int tile_end = (tile_st + DIST_TILE_SIZE < s_end) ? tile_st + DIST_TILE_SIZE : s_end;
                    int tile_cnt = tile_end - tile_st;

                    int t = team.team_rank();
                    if (t < tile_cnt) {
                        int idx = tile_st + t;
                        tile_x(t) = px(idx);
                        tile_y(t) = py(idx);
                        tile_z(t) = pz(idx);
                        tile_m(t) = pm(idx);
                    }
                    team.team_barrier();

                    if (ii < slice_count_f) {
                        for (int jt = 0; jt < tile_cnt; jt++) {
                            double dx = tile_x(jt) - xi;
                            double dy = tile_y(jt) - yi;
                            double dz = tile_z(jt) - zi;

                            double r2 = dx*dx + dy*dy + dz*dz + soft2;
                            double inv_r = 1.0 / Kokkos::sqrt(r2);
                            double inv_r3 = inv_r * inv_r * inv_r;
                            double mj_inv_r3 = G * tile_m(jt) * inv_r3;

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

// Cross-acceleration with gradient correction - tiled GPU kernel
static void cross_acceleration_and_gradient_distributed(DistributedParticles& dp,
                                                        int slow_start, int slow_count,
                                                        int fast_start, int fast_count,
                                                        double G, double soft2, double dt) {
    // First compute cross-accelerations (already tiled)
    cross_acceleration_distributed(dp, slow_start, slow_count,
                                   fast_start, fast_count, G, soft2);

    auto px = dp.local.x; auto py = dp.local.y; auto pz = dp.local.z;
    auto pax = dp.local.ax; auto pay = dp.local.ay; auto paz = dp.local.az;
    auto pm = dp.local.mass;

    const double coeff = G * dt * dt * GRAD_COEFF;

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t scratch_size = DIST_TILE_SIZE * 7 * sizeof(double);

    // Gradient correction for slow particles from fast
    int slice_start, slice_count;
    dp.get_force_slice(slow_start, slow_count, slice_start, slice_count);

    if (slice_count > 0) {
        int num_teams = (slice_count + DIST_TILE_SIZE - 1) / DIST_TILE_SIZE;
        const int f_start = fast_start;
        const int f_end = fast_start + fast_count;

        Kokkos::parallel_for("cross_grad_slow_tiled",
            team_policy(num_teams, DIST_TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_LAMBDA(const member_type& team) {
                int ii = team.league_rank() * DIST_TILE_SIZE + team.team_rank();
                int i = slice_start + ii;

                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_x(team.team_scratch(0), DIST_TILE_SIZE);
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_y(tile_x.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_z(tile_y.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_m(tile_z.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_ax(tile_m.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_ay(tile_ax.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_az(tile_ay.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);

                double xi = 0, yi = 0, zi = 0;
                double axi = 0, ayi = 0, azi = 0;
                if (ii < slice_count) {
                    xi = px(i); yi = py(i); zi = pz(i);
                    axi = pax(i); ayi = pay(i); azi = paz(i);
                }

                double cx = 0, cy = 0, cz = 0;

                for (int tile_st = f_start; tile_st < f_end; tile_st += DIST_TILE_SIZE) {
                    int tile_end = (tile_st + DIST_TILE_SIZE < f_end) ? tile_st + DIST_TILE_SIZE : f_end;
                    int tile_cnt = tile_end - tile_st;

                    int t = team.team_rank();
                    if (t < tile_cnt) {
                        int idx = tile_st + t;
                        tile_x(t) = px(idx);
                        tile_y(t) = py(idx);
                        tile_z(t) = pz(idx);
                        tile_m(t) = pm(idx);
                        tile_ax(t) = pax(idx);
                        tile_ay(t) = pay(idx);
                        tile_az(t) = paz(idx);
                    }
                    team.team_barrier();

                    if (ii < slice_count) {
                        for (int jt = 0; jt < tile_cnt; jt++) {
                            double dx = tile_x(jt) - xi;
                            double dy = tile_y(jt) - yi;
                            double dz = tile_z(jt) - zi;

                            double dax = tile_ax(jt) - axi;
                            double day = tile_ay(jt) - ayi;
                            double daz = tile_az(jt) - azi;

                            double r2 = dx*dx + dy*dy + dz*dz + soft2;
                            double inv_r2 = 1.0 / r2;
                            double inv_r = 1.0 / Kokkos::sqrt(r2);
                            double inv_r5 = inv_r2 * inv_r2 * inv_r;

                            double r_dot_a = dx*dax + dy*day + dz*daz;
                            double f = coeff * tile_m(jt) * inv_r5;

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
        int num_teams_f = (slice_count_f + DIST_TILE_SIZE - 1) / DIST_TILE_SIZE;
        const int s_start = slow_start;
        const int s_end = slow_start + slow_count;

        Kokkos::parallel_for("cross_grad_fast_tiled",
            team_policy(num_teams_f, DIST_TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_LAMBDA(const member_type& team) {
                int ii = team.league_rank() * DIST_TILE_SIZE + team.team_rank();
                int i = slice_start_f + ii;

                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_x(team.team_scratch(0), DIST_TILE_SIZE);
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_y(tile_x.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_z(tile_y.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_m(tile_z.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_ax(tile_m.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_ay(tile_ax.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
                Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                    tile_az(tile_ay.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);

                double xi = 0, yi = 0, zi = 0;
                double axi = 0, ayi = 0, azi = 0;
                if (ii < slice_count_f) {
                    xi = px(i); yi = py(i); zi = pz(i);
                    axi = pax(i); ayi = pay(i); azi = paz(i);
                }

                double cx = 0, cy = 0, cz = 0;

                for (int tile_st = s_start; tile_st < s_end; tile_st += DIST_TILE_SIZE) {
                    int tile_end = (tile_st + DIST_TILE_SIZE < s_end) ? tile_st + DIST_TILE_SIZE : s_end;
                    int tile_cnt = tile_end - tile_st;

                    int t = team.team_rank();
                    if (t < tile_cnt) {
                        int idx = tile_st + t;
                        tile_x(t) = px(idx);
                        tile_y(t) = py(idx);
                        tile_z(t) = pz(idx);
                        tile_m(t) = pm(idx);
                        tile_ax(t) = pax(idx);
                        tile_ay(t) = pay(idx);
                        tile_az(t) = paz(idx);
                    }
                    team.team_barrier();

                    if (ii < slice_count_f) {
                        for (int jt = 0; jt < tile_cnt; jt++) {
                            double dx = tile_x(jt) - xi;
                            double dy = tile_y(jt) - yi;
                            double dz = tile_z(jt) - zi;

                            double dax = tile_ax(jt) - axi;
                            double day = tile_ay(jt) - ayi;
                            double daz = tile_az(jt) - azi;

                            double r2 = dx*dx + dy*dy + dz*dz + soft2;
                            double inv_r2 = 1.0 / r2;
                            double inv_r = 1.0 / Kokkos::sqrt(r2);
                            double inv_r5 = inv_r2 * inv_r2 * inv_r;

                            double r_dot_a = dx*dax + dy*day + dz*daz;
                            double f = coeff * tile_m(jt) * inv_r5;

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

    int slice_start, slice_count;
    dp.get_force_slice(start, count, slice_start, slice_count);

    if (slice_count == 0) {
        dp.allgather_timestep_subset(start, count);
        return;
    }

    auto px = dp.local.x; auto py = dp.local.y; auto pz = dp.local.z;
    auto pvx = dp.local.vx; auto pvy = dp.local.vy; auto pvz = dp.local.vz;
    auto pm = dp.local.mass;
    auto dt_f = dp.local.dt_adapt;

    const int j_start = start;
    const int j_end = start + count;

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    // Need x,y,z,vx,vy,vz,m for adaptive timestep
    size_t scratch_size = DIST_TILE_SIZE * 7 * sizeof(double);
    int num_teams = (slice_count + DIST_TILE_SIZE - 1) / DIST_TILE_SIZE;

    Kokkos::parallel_for("dist_dt_tiled",
        team_policy(num_teams, DIST_TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team) {
            int ii = team.league_rank() * DIST_TILE_SIZE + team.team_rank();
            int i = slice_start + ii;

            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_x(team.team_scratch(0), DIST_TILE_SIZE);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_y(tile_x.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_z(tile_y.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_vx(tile_z.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_vy(tile_vx.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_vz(tile_vy.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile_m(tile_vz.data() + DIST_TILE_SIZE, DIST_TILE_SIZE);

            double xi = 0, yi = 0, zi = 0;
            double vxi = 0, vyi = 0, vzi = 0;
            double mi = 0;
            if (ii < slice_count) {
                xi = px(i); yi = py(i); zi = pz(i);
                vxi = pvx(i); vyi = pvy(i); vzi = pvz(i);
                mi = pm(i);
            }

            double t_min_sq = 1e60;

            for (int tile_st = j_start; tile_st < j_end; tile_st += DIST_TILE_SIZE) {
                int tile_end = (tile_st + DIST_TILE_SIZE < j_end) ? tile_st + DIST_TILE_SIZE : j_end;
                int tile_cnt = tile_end - tile_st;

                int t = team.team_rank();
                if (t < tile_cnt) {
                    int idx = tile_st + t;
                    tile_x(t) = px(idx);
                    tile_y(t) = py(idx);
                    tile_z(t) = pz(idx);
                    tile_vx(t) = pvx(idx);
                    tile_vy(t) = pvy(idx);
                    tile_vz(t) = pvz(idx);
                    tile_m(t) = pm(idx);
                }
                team.team_barrier();

                if (ii < slice_count) {
                    for (int jt = 0; jt < tile_cnt; jt++) {
                        int j = tile_st + jt;
                        if (j == i) continue;

                        double dx = tile_x(jt) - xi;
                        double dy = tile_y(jt) - yi;
                        double dz = tile_z(jt) - zi;
                        double r2 = dx*dx + dy*dy + dz*dz;

                        double dvx = tile_vx(jt) - vxi;
                        double dvy = tile_vy(jt) - vyi;
                        double dvz = tile_vz(jt) - vzi;
                        double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;

                        // Squared-space timestep: 1 sqrt/pair instead of 5
                        // t_ff^2 = r^3 / (G*M), t_fb^2 = r^2 / v^2
                        double r2_eps = r2 + EPS;
                        double r_norm = Kokkos::sqrt(r2_eps);
                        double mass_sum = mi + tile_m(jt);
                        double t_ff_sq = r2_eps * r_norm / (G * mass_sum + EPS);
                        double t_fb_sq = r2_eps / (v_sq + EPS);

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
        g_profiler.start();
        compute_timestep_distributed(dp, start, count, G, soft2, eta);
        g_profiler.stop(StepProfiler::ADAPTIVE_DT);
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

    // ==========================================================================
    // ALGORITHM 4: HHS-FSI with slow (S) and fast (F) particles
    // ==========================================================================

    // Step 1: Cross-kick S<->F (dt/6) - distributed force + allgather + kick
    g_profiler.start();
    cross_acceleration_distributed(dp, slow_start, slow_count, fast_start, fast_count, G, soft2);
    g_profiler.stop(StepProfiler::CROSS_FORCE);
    g_profiler.start();
    kick_velocities(dp.local, slow_start, slow_count, pivot_dt * FSI_KICK);
    kick_velocities(dp.local, fast_start, fast_count, pivot_dt * FSI_KICK);
    g_profiler.stop(StepProfiler::DRIFT_KICK);

    // Step 2: FSI half-step on slow
    fsi_step_distributed(dp, slow_start, slow_count, G, soft2, pivot_dt * 0.5);

    // Step 3: Recurse on fast (first half) — always compute fresh timesteps
    hhs_fsi_recursive_distributed(dp, fast_start, fast_count, pivot_dt * 0.5,
                                  G, soft2, dt_min, eta, false, false);

    // Step 4: Cross-kick S<->F with gradient (2dt/3)
    g_profiler.start();
    cross_acceleration_and_gradient_distributed(dp, slow_start, slow_count,
                                                fast_start, fast_count,
                                                G, soft2, pivot_dt);
    g_profiler.stop(StepProfiler::CROSS_GRAD);
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

    // Step 7: Cross-kick S<->F (dt/6)
    g_profiler.start();
    cross_acceleration_distributed(dp, slow_start, slow_count, fast_start, fast_count, G, soft2);
    g_profiler.stop(StepProfiler::CROSS_FORCE);
    g_profiler.start();
    kick_velocities(dp.local, slow_start, slow_count, pivot_dt * FSI_KICK);
    kick_velocities(dp.local, fast_start, fast_count, pivot_dt * FSI_KICK);
    g_profiler.stop(StepProfiler::DRIFT_KICK);
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

