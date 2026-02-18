#include "gbs.h"
#include <cstring>
#include <cmath>
#include <algorithm>

constexpr int GBSIntegrator::STEP_SEQ[];

// ============================================================================
// Pack/unpack LogHState to flat array for extrapolation
// ============================================================================

void GBSIntegrator::pack_state(const LogHState& state, double* y, int& dim) {
    dim = 0;
    int n = state.n_bodies;
    for (int i = 0; i < n * 3; i++) y[dim++] = state.pos[i];
    for (int i = 0; i < n * 3; i++) y[dim++] = state.vel[i];
    y[dim++] = state.t;
    y[dim++] = state.B;
}

void GBSIntegrator::unpack_state(const double* y, int dim, LogHState& state) {
    int idx = 0;
    int n = state.n_bodies;
    for (int i = 0; i < n * 3; i++) state.pos[i] = y[idx++];
    for (int i = 0; i < n * 3; i++) state.vel[i] = y[idx++];
    state.t = y[idx++];
    state.B = y[idx++];
}

// ============================================================================
// Take n_substeps leapfrog steps over interval ds_total
// Leapfrog is symmetric (time-reversible) so error expands in even powers
// of h — GBS polynomial extrapolation in h² works directly, no smoothing needed
// ============================================================================

void GBSIntegrator::midpoint(const LogHState& in, LogHState& out,
                              int n_substeps, double ds_total, double G) {
    double h = ds_total / n_substeps;
    out = in;
    for (int step = 0; step < n_substeps; step++) {
        logh_leapfrog_step(out, h, G);
    }
}

// ============================================================================
// Rational polynomial extrapolation (Stoer-Bulirsch / Neville-Aitken)
// Uses the standard Richardson extrapolation table
// Returns max relative error
// ============================================================================

double GBSIntegrator::extrapolate(int stage, int dim) {
    // table[stage] already contains the midpoint result for STEP_SEQ[stage]
    // Compute extrapolated values using polynomial extrapolation

    if (stage == 0) return 1.0;  // need at least 2 stages

    double max_err = 0.0;

    for (int j = stage - 1; j >= 0; j--) {
        double ratio = (double)(STEP_SEQ[stage] * STEP_SEQ[stage]) /
                       (double)(STEP_SEQ[j] * STEP_SEQ[j]);
        double factor = 1.0 / (ratio - 1.0);

        for (int d = 0; d < dim; d++) {
            double delta = table[stage][d] - table[j][d];
            table[j][d] = table[stage][d] + factor * delta;
            table[stage][d] = table[j][d];
        }
    }

    // Error estimate: difference between last two extrapolated values
    for (int d = 0; d < dim; d++) {
        double scale = std::max(1.0, std::fabs(table[0][d]));
        double err = std::fabs(table[1][d] - table[0][d]) / scale;
        if (err > max_err) max_err = err;
    }

    return max_err;
}

// ============================================================================
// Full GBS step with adaptive order
// ============================================================================

bool GBSIntegrator::step(LogHState& state, double ds, double G, double& ds_next) {
    int dim = 0;
    double y0[LOGH_MAX_BODIES * 6 + 2];
    pack_state(state, y0, dim);

    for (int stage = 0; stage < MAX_STAGES; stage++) {
        LogHState result;
        result = state;  // copy masses, B, n_bodies, etc.

        midpoint(state, result, STEP_SEQ[stage], ds, G);
        pack_state(result, table[stage], dim);

        if (stage > 0) {
            double err = extrapolate(stage, dim);

            if (err < eta_gbs) {
                // Converged: unpack best estimate
                unpack_state(table[0], dim, state);

                // Suggest next step size
                double safety = 0.8;
                double order = 2.0 * (stage + 1);
                ds_next = ds * safety * std::pow(eta_gbs / (err + 1e-30), 1.0 / order);
                ds_next = std::min(ds_next, 4.0 * ds);  // don't grow too fast
                return true;
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
                                      double ds_initial, int max_iter) {
    double ds = ds_initial;
    int n_steps = 0;

    for (int iter = 0; iter < max_iter; iter++) {
        // Check if we've reached the target time
        double dt_remain = t_target - state.t;
        if (std::fabs(dt_remain) < 1e-15 * std::fabs(t_target)) break;

        // Estimate ds from dt_remain / Omega
        double T = logh_kinetic_energy(state);
        double Omega = T + state.B;
        if (Omega < 1e-30) Omega = 1e-30;

        // First iteration: use provided ds_initial
        // Subsequent: estimate from remaining physical time
        if (iter > 0) {
            ds = dt_remain / Omega;
        }

        // Save state for bisection if we overshoot
        LogHState saved = state;
        double ds_next;

        bool ok = step(state, ds, G, ds_next);
        n_steps++;

        if (!ok) {
            // Step failed — try with smaller ds
            state = saved;
            ds *= 0.5;
            continue;
        }

        // Check if we overshot the target time
        if ((dt_remain > 0 && state.t > t_target + 1e-10 * std::fabs(t_target)) ||
            (dt_remain < 0 && state.t < t_target - 1e-10 * std::fabs(t_target))) {
            // Overshoot — bisect
            state = saved;
            ds *= 0.5;
            continue;
        }

        ds = ds_next;
    }

    return n_steps;
}
