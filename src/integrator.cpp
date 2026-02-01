#include "integrator.h"
#include "forces.h"
#include "constants.h"
#include <vector>

// Helper: drift positions for [start, start+count)
static void drift(Particles& p, int start, int count, double dt) {
    const int end = start + count;
    for (int i = start; i < end; i++) {
        p.x[i] += p.vx[i] * dt;
        p.y[i] += p.vy[i] * dt;
        p.z[i] += p.vz[i] * dt;
    }
}

// Helper: kick velocities for [start, start+count) using current acc
static void kick(Particles& p, int start, int count, double dt) {
    const int end = start + count;
    for (int i = start; i < end; i++) {
        p.vx[i] += p.ax[i] * dt;
        p.vy[i] += p.ay[i] * dt;
        p.vz[i] += p.az[i] * dt;
    }
}

// FSI step on [start, start+count): self-interaction only, O(count²)
// U(dt/6) -> T(dt/2) -> Ũ(2dt/3) -> T(dt/2) -> U(dt/6)
void fsi_step(Particles& p, int start, int count,
              double G, double soft2, double dt) {
    // U kick (dt/6)
    self_acceleration(p, start, count, G, soft2);
    kick(p, start, count, dt * FSI_KICK);

    // T drift (dt/2)
    drift(p, start, count, dt * FSI_DRIFT);

    // Ũ kick (2dt/3): fused acc + gradient correction
    self_acceleration_and_gradient(p, start, count, G, soft2, dt);
    kick(p, start, count, dt * FSI_MOD_KICK);

    // T drift (dt/2)
    drift(p, start, count, dt * FSI_DRIFT);

    // U kick (dt/6)
    self_acceleration(p, start, count, G, soft2);
    kick(p, start, count, dt * FSI_KICK);
}

// Helper to get slow/fast ranges for a given level
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

// HHS-FSI recursive step
void hhs_fsi_step(Particles& p, int level, double dt,
                  double G, double soft2,
                  const std::vector<double>& level_dts) {
    int slow_start, slow_count, fast_start, fast_count;
    get_slow_fast(p, level, slow_start, slow_count, fast_start, fast_count);

    // Base case: no faster particles
    if (fast_count == 0) {
        fsi_step(p, slow_start, slow_count, G, soft2, dt);
        return;
    }

    // 1. Cross-Newton kick (dt/6)
    cross_acceleration(p, slow_start, slow_count, fast_start, fast_count, G, soft2);
    kick(p, slow_start, slow_count, dt * FSI_KICK);
    kick(p, fast_start, fast_count, dt * FSI_KICK);

    // 2. FSI half-step on slow particles (self-only, O(slow²))
    fsi_step(p, slow_start, slow_count, G, soft2, dt * 0.5);

    // 3. Recurse on next level (first half)
    hhs_fsi_step(p, level + 1, dt * 0.5, G, soft2, level_dts);

    // 4. Cross-gradient kick (2dt/3): fused acc + gradient correction
    cross_acceleration_and_gradient(p, slow_start, slow_count, fast_start, fast_count,
                                    G, soft2, dt);
    kick(p, slow_start, slow_count, dt * FSI_MOD_KICK);
    kick(p, fast_start, fast_count, dt * FSI_MOD_KICK);

    // 5. Recurse on next level (second half)
    hhs_fsi_step(p, level + 1, dt * 0.5, G, soft2, level_dts);

    // 6. FSI half-step on slow particles
    fsi_step(p, slow_start, slow_count, G, soft2, dt * 0.5);

    // 7. Cross-Newton kick (dt/6)
    cross_acceleration(p, slow_start, slow_count, fast_start, fast_count, G, soft2);
    kick(p, slow_start, slow_count, dt * FSI_KICK);
    kick(p, fast_start, fast_count, dt * FSI_KICK);
}

// ==========================================================================
// Stack-based HHS-FSI: explicit stack, no recursion
// ==========================================================================

void hhs_fsi_step_stack(Particles& p, int start_level, double dt_outer,
                        double G, double soft2,
                        std::vector<StackEntry>& stack) {
    int finest = p.n_levels - 1;

    // Reuse pre-allocated stack
    stack.clear();

    // Seed with initial RECURSE
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
                // Push single FSI base step
                stack.push_back({(int)StackOp::FSI_BASE, level, dt});
            } else {
                // Push 7 ops in REVERSE order (LIFO)
                stack.push_back({(int)StackOp::CROSS_NEWTON,   level,     dt / 6.0});  // 7
                stack.push_back({(int)StackOp::FSI_HALF,       level,     dt});         // 6
                stack.push_back({(int)StackOp::RECURSE,        level + 1, dt / 2.0});  // 5
                stack.push_back({(int)StackOp::CROSS_GRADIENT, level,     dt});         // 4
                stack.push_back({(int)StackOp::RECURSE,        level + 1, dt / 2.0});  // 3
                stack.push_back({(int)StackOp::FSI_HALF,       level,     dt});         // 2
                stack.push_back({(int)StackOp::CROSS_NEWTON,   level,     dt / 6.0});  // 1
            }
            break;
        }

        case (int)StackOp::FSI_BASE: {
            fsi_step(p, slow_start, slow_count, G, soft2, dt);
            break;
        }

        case (int)StackOp::FSI_HALF: {
            double dt_half = dt * 0.5;
            fsi_step(p, slow_start, slow_count, G, soft2, dt_half);
            break;
        }

        case (int)StackOp::CROSS_NEWTON: {
            if (fast_count > 0 && slow_count > 0) {
                cross_acceleration(p, slow_start, slow_count,
                                   fast_start, fast_count, G, soft2);
                kick(p, slow_start, slow_count, dt);  // dt is already dt_outer/6
                kick(p, fast_start, fast_count, dt);
            }
            break;
        }

        case (int)StackOp::CROSS_GRADIENT: {
            if (fast_count > 0 && slow_count > 0) {
                cross_acceleration_and_gradient(p, slow_start, slow_count,
                                                fast_start, fast_count, G, soft2, dt);
                kick(p, slow_start, slow_count, dt * FSI_MOD_KICK);
                kick(p, fast_start, fast_count, dt * FSI_MOD_KICK);
            }
            break;
        }

        } // switch
    } // while
}
