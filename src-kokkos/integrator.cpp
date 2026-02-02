#include "integrator.h"
#include "forces.h"
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
    Kokkos::fence();
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
    Kokkos::fence();
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

void hhs_fsi_step(Particles& p, int level, double dt,
                  double G, double soft2,
                  const std::vector<double>& level_dts) {
    int slow_start, slow_count, fast_start, fast_count;
    get_slow_fast(p, level, slow_start, slow_count, fast_start, fast_count);

    if (fast_count == 0) {
        fsi_step(p, slow_start, slow_count, G, soft2, dt);
        return;
    }

    // 1. Cross-Newton kick (dt/6)
    fn_cross_acc(p, slow_start, slow_count, fast_start, fast_count, G, soft2);
    kick(p, slow_start, slow_count, dt * FSI_KICK);
    kick(p, fast_start, fast_count, dt * FSI_KICK);

    // 2. FSI half-step on slow
    fsi_step(p, slow_start, slow_count, G, soft2, dt * 0.5);

    // 3. Recurse (first half)
    hhs_fsi_step(p, level + 1, dt * 0.5, G, soft2, level_dts);

    // 4. Cross-gradient kick (2dt/3)
    fn_cross_acc_grad(p, slow_start, slow_count, fast_start, fast_count, G, soft2, dt);
    kick(p, slow_start, slow_count, dt * FSI_MOD_KICK);
    kick(p, fast_start, fast_count, dt * FSI_MOD_KICK);

    // 5. Recurse (second half)
    hhs_fsi_step(p, level + 1, dt * 0.5, G, soft2, level_dts);

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
