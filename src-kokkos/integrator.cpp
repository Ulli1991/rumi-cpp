#include "integrator.h"
#include "forces.h"
#include "timestep.h"
#include "constants.h"
#include <Kokkos_Core.hpp>
#include <vector>

// Fused function pointer types (force + kick combined)
using AccKickFn = void(*)(Particles&, int, int, double, double, double);
using AccGradKickFn = void(*)(Particles&, int, int, double, double, double, double);
using CrossAccGradKickFn = void(*)(Particles&, int, int, int, int, double, double, double, double);
using CrossAccKickCacheFn = void(*)(Particles&, int, int, int, int, double, double, double);

static AccKickFn fn_self_acc_kick = nullptr;
static AccGradKickFn fn_self_acc_grad_kick = nullptr;
static CrossAccGradKickFn fn_cross_acc_grad_kick = nullptr;
static CrossAccKickCacheFn fn_cross_acc_kick_cache = nullptr;

void set_force_dispatch(bool use_n3l) {
    if (use_n3l) {
        fn_self_acc_kick = &self_acceleration_kick<true>;
        fn_self_acc_grad_kick = &self_acceleration_and_gradient_kick<true>;
        fn_cross_acc_grad_kick = &cross_acceleration_and_gradient_kick<true>;
        fn_cross_acc_kick_cache = &cross_acceleration_kick_and_cache<true>;
    } else {
        fn_self_acc_kick = &self_acceleration_kick<false>;
        fn_self_acc_grad_kick = &self_acceleration_and_gradient_kick<false>;
        fn_cross_acc_grad_kick = &cross_acceleration_and_gradient_kick<false>;
        fn_cross_acc_kick_cache = &cross_acceleration_kick_and_cache<false>;
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

void fsi_step(Particles& p, int start, int count,
              double G, double soft2, double dt) {
    // U kick (dt/6) - fused force + kick
    fn_self_acc_kick(p, start, count, G, soft2, dt * FSI_KICK);

    // T drift (dt/2)
    drift(p, start, count, dt * FSI_DRIFT);

    // U-tilde kick (2dt/3) - fused force + gradient + kick
    fn_self_acc_grad_kick(p, start, count, G, soft2, dt, dt * FSI_MOD_KICK);

    // T drift (dt/2)
    drift(p, start, count, dt * FSI_DRIFT);

    // U kick (dt/6) - fused force + kick
    fn_self_acc_kick(p, start, count, G, soft2, dt * FSI_KICK);
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

    // Apply permutation: fused gather of all 13 fields (11 doubles + 2 ints)
    auto px = p.x; auto py = p.y; auto pz = p.z;
    auto pvx = p.vx; auto pvy = p.vy; auto pvz = p.vz;
    auto pax = p.ax; auto pay = p.ay; auto paz = p.az;
    auto pmass = p.mass; auto pdt = p.dt_frost;
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
void hhs_fsi_step_dynamic(Particles& p, int start, int count, double pivot_dt,
                          double G, double soft2, double dt_min,
                          double eta, bool use_n3l) {
    if (count == 0) return;

    // Compute FROST timesteps for this subset (always N3L - single O(N²) computation)
    if (count > 1) {
        compute_frost_subset<true>(p, start, count, G, soft2, eta);
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
    fn_cross_acc_kick_cache(p, slow_start, slow_count, fast_start, fast_count, G, soft2, pivot_dt * FSI_KICK);

    // 2. FSI half-step on slow
    fsi_step(p, slow_start, slow_count, G, soft2, pivot_dt * 0.5);

    // 3. Recurse on fast (first half) with pivot/2
    hhs_fsi_step_dynamic(p, fast_start, fast_count, pivot_dt * 0.5, G, soft2, dt_min, eta, use_n3l);

    // 4. Cross-gradient kick (2dt/3)
    fn_cross_acc_grad_kick(p, slow_start, slow_count, fast_start, fast_count, G, soft2, pivot_dt, pivot_dt * FSI_MOD_KICK);

    // 5. Recurse on fast (second half) with pivot/2
    hhs_fsi_step_dynamic(p, fast_start, fast_count, pivot_dt * 0.5, G, soft2, dt_min, eta, use_n3l);

    // 6. FSI half-step on slow
    fsi_step(p, slow_start, slow_count, G, soft2, pivot_dt * 0.5);

    // 7. Cross-Newton kick (dt/6)
    fn_cross_acc_kick_cache(p, slow_start, slow_count, fast_start, fast_count, G, soft2, pivot_dt * FSI_KICK);
}

