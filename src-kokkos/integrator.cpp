#include "integrator.h"
#include "forces.h"
#include "timestep.h"
#include "constants.h"
#include <Kokkos_Core.hpp>
#include <vector>

// Function pointers for runtime N3L dispatch
static AccFn fn_self_acc = nullptr;
static AccGradFn fn_self_acc_grad = nullptr;
static CrossAccFn fn_cross_acc = nullptr;
static CrossAccGradFn fn_cross_acc_grad = nullptr;

void set_force_dispatch(bool use_n3l) {
    if (use_n3l) {
        fn_self_acc      = &self_acceleration<true>;
        fn_self_acc_grad = &self_acceleration_and_gradient<true>;
        fn_cross_acc     = &cross_acceleration<true>;
        fn_cross_acc_grad = &cross_acceleration_and_gradient<true>;
    } else {
        fn_self_acc      = &self_acceleration<false>;
        fn_self_acc_grad = &self_acceleration_and_gradient<false>;
        fn_cross_acc     = &cross_acceleration<false>;
        fn_cross_acc_grad = &cross_acceleration_and_gradient<false>;
    }
}

// Helper: drift positions for [start, start+count)
static void drift(Particles& p, int start, int count, double dt) {
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

// Helper: kick velocities for [start, start+count)
static void kick(Particles& p, int start, int count, double dt) {
    auto pvx = p.vx; auto pvy = p.vy; auto pvz = p.vz;
    auto pax = p.ax; auto pay = p.ay; auto paz = p.az;
    Kokkos::parallel_for("kick", count, KOKKOS_LAMBDA(int ii) {
        int i = start + ii;
        pvx(i) += pax(i) * dt;
        pvy(i) += pay(i) * dt;
        pvz(i) += paz(i) * dt;
    });
    // No fence - next operation will handle dependency
}

void fsi_step(Particles& p, int start, int count,
              double G, double soft2, double dt) {
    // U kick (dt/6)
    fn_self_acc(p, start, count, G, soft2);
    kick(p, start, count, dt * FSI_KICK);

    // T drift (dt/2)
    drift(p, start, count, dt * FSI_DRIFT);

    // U-tilde kick (2dt/3)
    fn_self_acc_grad(p, start, count, G, soft2, dt);
    kick(p, start, count, dt * FSI_MOD_KICK);

    // T drift (dt/2)
    drift(p, start, count, dt * FSI_DRIFT);

    // U kick (dt/6)
    fn_self_acc(p, start, count, G, soft2);
    kick(p, start, count, dt * FSI_KICK);
}

static void get_slow_fast(const Particles& p, int level,
                          int& slow_start, int& slow_count,
                          int& fast_start, int& fast_count) {
    slow_start = p.level_start[level];
    slow_count = p.level_count[level];
    fast_start = -1;
    fast_count = 0;
    if (level + 1 < p.n_levels) {
        fast_start = p.level_start[level + 1];
        for (int l = level + 1; l < p.n_levels; l++)
            fast_count += p.level_count[l];
    }
}

// ==========================================================================
// Dynamic pivot-based partitioning for HHS (per paper's Algorithm 4)
// Partitions particles [start, start+count) into:
//   - slow: dt_frost >= pivot (at beginning)
//   - fast: dt_frost < pivot (at end)
// Returns the count of slow particles
// ==========================================================================
static int partition_by_pivot(Particles& p, int start, int count, double pivot) {
    if (count <= 1) return count;

    auto dt_f = p.dt_frost;
    auto perm = p.scratch_i;
    auto tmp_d = p.scratch_d;
    auto tmp_i = p.scratch_i2;

    // Count slow particles using scalar reduction
    int slow_count = 0;
    Kokkos::parallel_reduce("count_slow", count,
        KOKKOS_LAMBDA(int ii, int& lsum) {
            if (dt_f(start + ii) >= pivot) lsum++;
        }, slow_count);
    Kokkos::fence();

    if (slow_count == 0 || slow_count == count) {
        // No partitioning needed - all slow or all fast
        return slow_count;
    }

    // Initialize cursors on host and copy to device
    Kokkos::View<int*> d_cursors("cursors", 2);
    auto h_cursors = Kokkos::create_mirror_view(d_cursors);
    h_cursors(0) = 0;           // slow cursor
    h_cursors(1) = slow_count;  // fast cursor
    Kokkos::deep_copy(d_cursors, h_cursors);

    // Build permutation using atomic increments
    Kokkos::parallel_for("build_perm", count, KOKKOS_LAMBDA(int ii) {
        int i = start + ii;
        int dest;
        if (dt_f(i) >= pivot) {
            dest = Kokkos::atomic_fetch_add(&d_cursors(0), 1);
        } else {
            dest = Kokkos::atomic_fetch_add(&d_cursors(1), 1);
        }
        perm(dest) = i;  // perm[dest] = source index
    });
    Kokkos::fence();

    // Apply permutation using gather (same as partition_by_level)
    auto px = p.x; auto py = p.y; auto pz = p.z;
    auto pvx = p.vx; auto pvy = p.vy; auto pvz = p.vz;
    auto pax = p.ax; auto pay = p.ay; auto paz = p.az;
    auto pmass = p.mass; auto pdt = p.dt_frost;
    auto porig = p.orig_index;
    auto plevel = p.level;

    // Gather and copy for each array
    #define GATHER_COPY_D(arr) \
        Kokkos::parallel_for("gather", count, KOKKOS_LAMBDA(int ii) { tmp_d(ii) = arr(perm(ii)); }); \
        Kokkos::fence(); \
        Kokkos::parallel_for("copy", count, KOKKOS_LAMBDA(int ii) { arr(start + ii) = tmp_d(ii); });

    #define GATHER_COPY_I(arr) \
        Kokkos::parallel_for("gather", count, KOKKOS_LAMBDA(int ii) { tmp_i(ii) = arr(perm(ii)); }); \
        Kokkos::fence(); \
        Kokkos::parallel_for("copy", count, KOKKOS_LAMBDA(int ii) { arr(start + ii) = tmp_i(ii); });

    GATHER_COPY_D(px); GATHER_COPY_D(py); GATHER_COPY_D(pz);
    GATHER_COPY_D(pvx); GATHER_COPY_D(pvy); GATHER_COPY_D(pvz);
    GATHER_COPY_D(pax); GATHER_COPY_D(pay); GATHER_COPY_D(paz);
    GATHER_COPY_D(pmass); GATHER_COPY_D(pdt);
    GATHER_COPY_I(porig); GATHER_COPY_I(plevel);

    #undef GATHER_COPY_D
    #undef GATHER_COPY_I

    Kokkos::fence();
    return slow_count;
}

// ==========================================================================
// HHS-FSI with dynamic pivot-based partitioning (per paper's Algorithm 4)
// ==========================================================================
void hhs_fsi_step_dynamic(Particles& p, int start, int count, double pivot_dt,
                          double G, double soft2, double dt_min,
                          double eta, bool use_n3l) {
    if (count == 0) return;

    // Compute FROST timesteps for this subset (considering only internal interactions)
    if (count > 1) {
        if (use_n3l)
            compute_frost_subset<true>(p, start, count, G, soft2, eta);
        else
            compute_frost_subset<false>(p, start, count, G, soft2, eta);
    }

    // Partition into slow (dt >= pivot) and fast (dt < pivot)
    int slow_count = partition_by_pivot(p, start, count, pivot_dt);
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
    fn_cross_acc(p, slow_start, slow_count, fast_start, fast_count, G, soft2);
    kick(p, slow_start, slow_count, pivot_dt * FSI_KICK);
    kick(p, fast_start, fast_count, pivot_dt * FSI_KICK);

    // 2. FSI half-step on slow
    fsi_step(p, slow_start, slow_count, G, soft2, pivot_dt * 0.5);

    // 3. Recurse on fast (first half) with pivot/2
    hhs_fsi_step_dynamic(p, fast_start, fast_count, pivot_dt * 0.5, G, soft2, dt_min, eta, use_n3l);

    // 4. Cross-gradient kick (2dt/3)
    fn_cross_acc_grad(p, slow_start, slow_count, fast_start, fast_count, G, soft2, pivot_dt);
    kick(p, slow_start, slow_count, pivot_dt * FSI_MOD_KICK);
    kick(p, fast_start, fast_count, pivot_dt * FSI_MOD_KICK);

    // 5. Recurse on fast (second half) with pivot/2
    hhs_fsi_step_dynamic(p, fast_start, fast_count, pivot_dt * 0.5, G, soft2, dt_min, eta, use_n3l);

    // 6. FSI half-step on slow
    fsi_step(p, slow_start, slow_count, G, soft2, pivot_dt * 0.5);

    // 7. Cross-Newton kick (dt/6)
    fn_cross_acc(p, slow_start, slow_count, fast_start, fast_count, G, soft2);
    kick(p, slow_start, slow_count, pivot_dt * FSI_KICK);
    kick(p, fast_start, fast_count, pivot_dt * FSI_KICK);
}

// ==========================================================================
// Legacy HHS-FSI with pre-computed global levels
// ==========================================================================
void hhs_fsi_step(Particles& p, int level, double dt,
                  double G, double soft2,
                  const std::vector<double>& level_dts,
                  double eta, bool use_n3l,
                  bool reassign_timesteps) {
    int slow_start, slow_count, fast_start, fast_count;
    get_slow_fast(p, level, slow_start, slow_count, fast_start, fast_count);

    if (fast_count == 0) {
        fsi_step(p, slow_start, slow_count, G, soft2, dt);
        return;
    }

    // Optionally recompute FROST timesteps for fast particles (per paper's Algorithm 4)
    // This considers only mutual interactions within the fast set
    if (reassign_timesteps && fast_count > 1) {
        if (use_n3l)
            compute_frost_subset<true>(p, fast_start, fast_count, G, soft2, eta);
        else
            compute_frost_subset<false>(p, fast_start, fast_count, G, soft2, eta);
        assign_levels_subset(p, fast_start, fast_count, level_dts);
    }

    // 1. Cross-Newton kick (dt/6)
    fn_cross_acc(p, slow_start, slow_count, fast_start, fast_count, G, soft2);
    kick(p, slow_start, slow_count, dt * FSI_KICK);
    kick(p, fast_start, fast_count, dt * FSI_KICK);

    // 2. FSI half-step on slow
    fsi_step(p, slow_start, slow_count, G, soft2, dt * 0.5);

    // 3. Recurse (first half)
    hhs_fsi_step(p, level + 1, dt * 0.5, G, soft2, level_dts, eta, use_n3l, reassign_timesteps);

    // 4. Cross-gradient kick (2dt/3)
    fn_cross_acc_grad(p, slow_start, slow_count, fast_start, fast_count, G, soft2, dt);
    kick(p, slow_start, slow_count, dt * FSI_MOD_KICK);
    kick(p, fast_start, fast_count, dt * FSI_MOD_KICK);

    // 5. Recurse (second half)
    hhs_fsi_step(p, level + 1, dt * 0.5, G, soft2, level_dts, eta, use_n3l, reassign_timesteps);

    // 6. FSI half-step on slow
    fsi_step(p, slow_start, slow_count, G, soft2, dt * 0.5);

    // 7. Cross-Newton kick (dt/6)
    fn_cross_acc(p, slow_start, slow_count, fast_start, fast_count, G, soft2);
    kick(p, slow_start, slow_count, dt * FSI_KICK);
    kick(p, fast_start, fast_count, dt * FSI_KICK);
}

void hhs_fsi_step_stack(Particles& p, int start_level, double dt_outer,
                        double G, double soft2,
                        std::vector<StackEntry>& stack) {
    int finest = p.n_levels - 1;
    stack.clear();
    stack.push_back({(int)StackOp::RECURSE, start_level, dt_outer});

    while (!stack.empty()) {
        StackEntry e = stack.back();
        stack.pop_back();

        int level = e.level;
        double dt = e.dt;

        int slow_start, slow_count, fast_start, fast_count;
        get_slow_fast(p, level, slow_start, slow_count, fast_start, fast_count);

        switch (e.op) {
        case (int)StackOp::RECURSE: {
            bool is_base = (level >= finest) || (fast_count == 0);
            if (is_base) {
                stack.push_back({(int)StackOp::FSI_BASE, level, dt});
            } else {
                stack.push_back({(int)StackOp::CROSS_NEWTON,   level,     dt / 6.0});
                stack.push_back({(int)StackOp::FSI_HALF,       level,     dt});
                stack.push_back({(int)StackOp::RECURSE,        level + 1, dt / 2.0});
                stack.push_back({(int)StackOp::CROSS_GRADIENT, level,     dt});
                stack.push_back({(int)StackOp::RECURSE,        level + 1, dt / 2.0});
                stack.push_back({(int)StackOp::FSI_HALF,       level,     dt});
                stack.push_back({(int)StackOp::CROSS_NEWTON,   level,     dt / 6.0});
            }
            break;
        }
        case (int)StackOp::FSI_BASE:
            fsi_step(p, slow_start, slow_count, G, soft2, dt);
            break;
        case (int)StackOp::FSI_HALF:
            fsi_step(p, slow_start, slow_count, G, soft2, dt * 0.5);
            break;
        case (int)StackOp::CROSS_NEWTON:
            if (fast_count > 0 && slow_count > 0) {
                fn_cross_acc(p, slow_start, slow_count, fast_start, fast_count, G, soft2);
                kick(p, slow_start, slow_count, dt);
                kick(p, fast_start, fast_count, dt);
            }
            break;
        case (int)StackOp::CROSS_GRADIENT:
            if (fast_count > 0 && slow_count > 0) {
                fn_cross_acc_grad(p, slow_start, slow_count, fast_start, fast_count, G, soft2, dt);
                kick(p, slow_start, slow_count, dt * FSI_MOD_KICK);
                kick(p, fast_start, fast_count, dt * FSI_MOD_KICK);
            }
            break;
        }
    }
}
