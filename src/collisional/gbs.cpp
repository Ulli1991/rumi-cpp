#include "gbs.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdio>

constexpr int GBSIntegrator::STEP_SEQ[];

// ============================================================================
// Pack/unpack LogHState to flat array for extrapolation
// ============================================================================

void GBSIntegrator::pack_state(const LogHState& state, double* y, int& dim) {
    dim = 0;
    int n = state.n_bodies;
    for (int i = 0; i < n * 3; i++) y[dim++] = state.pos[i];
    for (int i = 0; i < n * 3; i++) y[dim++] = state.vel[i];
    if (state.use_aux_vel) {
        for (int i = 0; i < n * 3; i++) y[dim++] = state.aux_vel[i];
    }
    y[dim++] = state.t;
    y[dim++] = state.B;
}

void GBSIntegrator::unpack_state(const double* y, int dim, LogHState& state) {
    int idx = 0;
    int n = state.n_bodies;
    for (int i = 0; i < n * 3; i++) state.pos[i] = y[idx++];
    for (int i = 0; i < n * 3; i++) state.vel[i] = y[idx++];
    if (state.use_aux_vel) {
        for (int i = 0; i < n * 3; i++) state.aux_vel[i] = y[idx++];
    }
    state.t = y[idx++];
    state.B = y[idx++];
}

// ============================================================================
// Take n_substeps leapfrog steps over interval ds_total
// Leapfrog is symmetric (time-reversible) so error expands in even powers
// of h — GBS polynomial extrapolation in h² works directly, no smoothing needed
// ============================================================================

void GBSIntegrator::midpoint(const LogHState& in, LogHState& out,
                              int n_substeps, double ds_total, double G,
                              double c_code, int pn_order) {
    double h = ds_total / n_substeps;
    out = in;
    if (pn_order > 0 && out.use_aux_vel) {
        for (int s = 0; s < n_substeps; s++) {
            logh_leapfrog_step_pn(out, h, G, c_code, pn_order);
        }
    } else {
        for (int s = 0; s < n_substeps; s++) {
            logh_leapfrog_step(out, h, G);
        }
    }
}

// ============================================================================
// Polynomial extrapolation (Neville-Aitken) in h²
//
// table[0..stage] hold RAW midpoint results M(n_i) — never modified.
// work[][] is used for the Neville table: T[i][j].
//   T[i][0] = M(n_i)
//   T[i][j] = T[i][j-1] + (T[i][j-1] - T[i-1][j-1]) / ((n_i/n_{i-j})² - 1)
// Best estimate: T[stage][stage] stored in work[stage].
// Error: |T[stage][stage] - T[stage-1][stage-1]|.
//
// IMPORTANT: The old algorithm overwrote table[] in-place, which corrupted
// intermediate values (table[j] = T[k,k] after stage k, not M(n_j)).
// Stages > 1 then used T[k-1,k-1] instead of the raw midpoint, producing
// wrong extrapolation that could falsely report convergence.
// ============================================================================

double GBSIntegrator::extrapolate(int stage, int dim) {
    if (stage == 0) {
        // Save T[0][0] = raw midpoint as the "previous best" for stage 1
        std::memcpy(work[0], table[0], dim * sizeof(double));
        std::memcpy(diag_prev, table[0], dim * sizeof(double));
        return 1.0;  // need at least 2 stages
    }

    // Initialize work[i] = table[i] = raw midpoint result T[i][0]
    for (int i = 0; i <= stage; i++) {
        std::memcpy(work[i], table[i], dim * sizeof(double));
    }

    // Build Neville table column by column (j = extrapolation order)
    // CRITICAL: process i in REVERSE order (stage down to j) so that
    // work[i-1] still holds T[i-1][j-1] when computing T[i][j].
    // Forward order overwrites work[i-1] to T[i-1][j] before it's needed.
    for (int j = 1; j <= stage; j++) {
        for (int i = stage; i >= j; i--) {
            // T[i][j] = T[i][j-1] + (T[i][j-1] - T[i-1][j-1]) / ((n_i/n_{i-j})^2 - 1)
            double ratio = (double)(STEP_SEQ[i] * STEP_SEQ[i]) /
                           (double)(STEP_SEQ[i-j] * STEP_SEQ[i-j]);
            double factor = 1.0 / (ratio - 1.0);

            for (int d = 0; d < dim; d++) {
                double delta = work[i][d] - work[i-1][d];
                work[i][d] += factor * delta;
            }
        }
    }

    // Error: |T[stage][stage] - T[stage-1][stage-1]|
    // diag_prev was saved from the previous call to extrapolate()
    double max_err = 0.0;
    for (int d = 0; d < dim; d++) {
        double scale = std::max(1.0, std::fabs(work[stage][d]));
        double err = std::fabs(work[stage][d] - diag_prev[d]) / scale;
        if (err > max_err) max_err = err;
    }

    // Save T[stage][stage] as "previous best" for the next stage
    std::memcpy(diag_prev, work[stage], dim * sizeof(double));

    return max_err;
}

// ============================================================================
// Full GBS step with adaptive order
// ============================================================================

bool GBSIntegrator::step(LogHState& state, double ds, double G, double& ds_next,
                          double c_code, int pn_order) {
    int dim = 0;
    double y0[LOGH_MAX_BODIES * 9 + 2];
    pack_state(state, y0, dim);

    for (int stage = 0; stage < MAX_STAGES; stage++) {
        LogHState result;
        result = state;  // copy masses, B, n_bodies, etc.

        midpoint(state, result, STEP_SEQ[stage], ds, G, c_code, pn_order);
        pack_state(result, table[stage], dim);

        // Always call extrapolate — stage 0 initializes diag_prev (returns 1.0).
        // Without this, diag_prev is uninitialized garbage and stage 1 can
        // falsely report convergence.
        double err = extrapolate(stage, dim);

        if (stage > 0) {
            if (err < eta_gbs) {
                // Converged: unpack best estimate from work[stage]
                unpack_state(work[stage], dim, state);

                // Suggest next step size
                double safety = 0.8;
                double order = 2.0 * (stage + 1);
                ds_next = ds * safety * std::pow(eta_gbs / (err + 1e-30), 1.0 / order);
                ds_next = std::min(ds_next, 4.0 * ds);  // don't grow too fast
                return true;
            }

            // Early exit: if error is enormous after several stages,
            // the step is clearly too large for the dynamics. Bail out
            // early to save computing higher stages with no hope.
            if (stage >= 3 && err > 1.0) {
                ds_next = 0.5 * ds;
                return false;
            }
        }
    }

    // Did not converge — halve step size and try again
    ds_next = 0.5 * ds;
    return false;
}

// ============================================================================
// Integrate to target physical time using GBS
// ============================================================================

int GBSIntegrator::integrate_to_time(LogHState& state, double t_target, double G,
                                      double ds_initial, double c_code, int pn_order,
                                      int max_iter) {
    // Compute dynamical timescale from closest pair separation.
    // This limits ds so that even GBS stage 0 (2 substeps) can resolve
    // the fastest dynamics (e.g. periapsis passage at high eccentricity).
    double r_min_sq = 1e60;
    double M_total = 0;
    for (int i = 0; i < state.n_bodies; i++) {
        M_total += state.mass[i];
        for (int j = i + 1; j < state.n_bodies; j++) {
            double dx = state.pos[j*3] - state.pos[i*3];
            double dy = state.pos[j*3+1] - state.pos[i*3+1];
            double dz = state.pos[j*3+2] - state.pos[i*3+2];
            r_min_sq = std::min(r_min_sq, dx*dx + dy*dy + dz*dz);
        }
    }
    double r_min = std::sqrt(r_min_sq);
    double t_dyn = std::sqrt(r_min * r_min * r_min / (G * M_total));

    double T0 = logh_kinetic_energy(state);
    double Omega0 = T0 + state.B;
    if (Omega0 < 1e-30) Omega0 = 1e-30;

    // Scale ds so each GBS step covers at most ~1/10 dynamical time.
    // This ensures stage 0 (2 substeps) produces a qualitatively correct
    // solution, which is required for GBS polynomial extrapolation.
    double ds_dyn = t_dyn / (Omega0 * 10.0);
    double ds = std::min(ds_initial, ds_dyn);

    int n_steps = 0;
    int n_failed = 0;
    int n_overshoot = 0;
    double ds_min_used = ds;
    double ds_max_used = ds;

    for (int iter = 0; iter < max_iter; iter++) {
        // Check if we've reached the target time
        double dt_remain = t_target - state.t;
        if (std::fabs(dt_remain) < 1e-4 * std::fabs(t_target)) break;

        // Recompute dynamical time constraint at current state
        // Prevents ds from growing too large when orbit approaches periapsis
        double r_min_now_sq = 1e60;
        for (int i = 0; i < state.n_bodies; i++) {
            for (int j = i + 1; j < state.n_bodies; j++) {
                double dx = state.pos[j*3] - state.pos[i*3];
                double dy = state.pos[j*3+1] - state.pos[i*3+1];
                double dz = state.pos[j*3+2] - state.pos[i*3+2];
                r_min_now_sq = std::min(r_min_now_sq, dx*dx + dy*dy + dz*dz);
            }
        }
        double r_min_now = std::sqrt(r_min_now_sq);
        double t_dyn_now = std::sqrt(r_min_now * r_min_now * r_min_now / (G * M_total));

        // Estimate ds from dt_remain / Omega (caps ds to avoid overshooting)
        double T = logh_kinetic_energy(state);
        double Omega = T + state.B;
        if (Omega < 1e-30) Omega = 1e-30;
        double ds_est = dt_remain / Omega;

        // Also cap ds by dynamical time (prevents too-large steps near periapsis)
        double ds_dyn_now = t_dyn_now / (Omega * 10.0);

        // Use the smallest of current ds, time-based estimate, and dynamical limit
        double ds_try = ds;
        if (std::fabs(ds_est) < std::fabs(ds_try)) ds_try = ds_est;
        if (std::fabs(ds_dyn_now) < std::fabs(ds_try)) ds_try = ds_dyn_now;

        if (std::fabs(ds_try) < ds_min_used) ds_min_used = std::fabs(ds_try);
        if (std::fabs(ds_try) > ds_max_used) ds_max_used = std::fabs(ds_try);

        // Save state for bisection if we overshoot
        LogHState saved = state;
        double ds_next;

        bool ok = step(state, ds_try, G, ds_next, c_code, pn_order);
        n_steps++;

        if (!ok) {
            // Step failed — halve and retry (ds persists to next iteration)
            state = saved;
            ds = ds_try * 0.5;
            n_failed++;
            continue;
        }

        // Check if we overshot the target time
        if ((dt_remain > 0 && state.t > t_target + 1e-10 * std::fabs(t_target)) ||
            (dt_remain < 0 && state.t < t_target - 1e-10 * std::fabs(t_target))) {
            // Overshoot — bisect
            state = saved;
            ds = ds_try * 0.5;
            n_overshoot++;
            continue;
        }

        // Success: use GBS-suggested step size (will be capped by ds_est next iter)
        ds = ds_next;
    }

    double dt_remain = t_target - state.t;
    bool reached = std::fabs(dt_remain) < 1e-4 * std::fabs(t_target);
    if (!reached) {
        std::printf("      GBS: %d steps, t_final=%.6f/%.6f *** DID NOT REACH TARGET ***\n",
                    n_steps, state.t, t_target);
    }

    return n_steps;
}
