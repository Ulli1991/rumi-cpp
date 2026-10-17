#include "logh_step.h"
#include <algorithm>
#include <cmath>
#include <cstring>

void gbs_pack_state(const LogHState& state, double* y, int& dim) {
    dim = 0;
    int n = state.n_bodies;
    for (int i = 0; i < n * 3; i++) y[dim++] = state.pos[i];
    for (int i = 0; i < n * 3; i++) y[dim++] = state.vel[i];
    if (state.use_aux_vel) {
        for (int i = 0; i < n * 3; i++) y[dim++] = state.aux_vel[i];
    }
    if (state.has_spins) {
        for (int k = 0; k < 3; k++) y[dim++] = state.spin1[k];
        for (int k = 0; k < 3; k++) y[dim++] = state.spin2[k];
        for (int k = 0; k < 3; k++) y[dim++] = state.aux_spin1[k];
        for (int k = 0; k < 3; k++) y[dim++] = state.aux_spin2[k];
    }
    y[dim++] = state.t;
    y[dim++] = state.B;
}

void gbs_unpack_state(const double* y, LogHState& state) {
    int idx = 0;
    int n = state.n_bodies;
    for (int i = 0; i < n * 3; i++) state.pos[i] = y[idx++];
    for (int i = 0; i < n * 3; i++) state.vel[i] = y[idx++];
    if (state.use_aux_vel) {
        for (int i = 0; i < n * 3; i++) state.aux_vel[i] = y[idx++];
    }
    if (state.has_spins) {
        for (int k = 0; k < 3; k++) state.spin1[k] = y[idx++];
        for (int k = 0; k < 3; k++) state.spin2[k] = y[idx++];
        for (int k = 0; k < 3; k++) state.aux_spin1[k] = y[idx++];
        for (int k = 0; k < 3; k++) state.aux_spin2[k] = y[idx++];
    }
    state.t = y[idx++];
    state.B = y[idx++];
}

static double* spin_vec(LogHState& s, int i) {
    return i == 0 ? s.spin1 : i == 1 ? s.spin2 : i == 2 ? s.aux_spin1 : s.aux_spin2;
}

static void spin_norms(LogHState& state, double norm[4]) {
    for (int i = 0; i < 4; i++) {
        const double* c = spin_vec(state, i);
        norm[i] = std::sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
    }
}

static void restore_spin_norms(LogHState& state, const double norm[4]) {
    double now[4];
    spin_norms(state, now);
    for (int i = 0; i < 4; i++) {
        if (now[i] == 0.0) continue;
        double* c = spin_vec(state, i);
        for (int k = 0; k < 3; k++) c[k] *= norm[i] / now[i];
    }
}

// ============================================================================
// Take n_substeps leapfrog steps over interval ds_total
// Leapfrog is symmetric (time-reversible) so error expands in even powers
// of h — GBS polynomial extrapolation in h² works directly, no smoothing needed
// ============================================================================

void gbs_midpoint(const LogHState& in, LogHState& out, int n_substeps, double ds_total, double G, double c_code,
                  int pn_order, const LogHStepStart* start) {
    double h = ds_total / n_substeps;
    out = in;
    if (pn_order > 0 && out.use_aux_vel) {
        logh_leapfrog_steps_pn(out, h, G, c_code, pn_order, n_substeps, start);
    } else {
        logh_leapfrog_steps(out, h, G, n_substeps, start);
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
// Incremental: row i of the table depends only on M(n_0..n_i), so each call adds one row. On entry work[j] holds
// T[stage-1][j] (j < stage), on exit T[stage][j] (j <= stage): O(stage) row operations per call instead of
// rebuilding the table from table[0..stage] (O(stage^2)). Each T[i][j] is computed with the same operation as in
// the full rebuild, so the result is bit-identical. Calls must come in stage order 0, 1, 2, ... (GBS step).
//
// IMPORTANT: The old algorithm overwrote table[] in-place, which corrupted
// intermediate values (table[j] = T[k,k] after stage k, not M(n_j)).
// Stages > 1 then used T[k-1,k-1] instead of the raw midpoint, producing
// wrong extrapolation that could falsely report convergence.
// ============================================================================

double gbs_extrapolate(const int* seq, GbsTable& tab, int stage, int dim) {
    double* cur = tab.work[stage];   // T[stage][j-1] before column j, T[stage][j] after
    std::memcpy(cur, tab.table[stage], dim * sizeof(double));   // T[stage][0] = raw midpoint result
    if (stage == 0) return 1.0;  // need at least 2 stages

    double max_err = 0.0;
    for (int j = 1; j <= stage; j++) {
        // T[i][j] = T[i][j-1] + (T[i][j-1] - T[i-1][j-1]) / ((n_i/n_{i-j})^2 - 1),  i = stage
        const int i = stage;
        double ratio = (double)(seq[i] * seq[i]) /
                       (double)(seq[i-j] * seq[i-j]);
        double factor = 1.0 / (ratio - 1.0);

        double* prev = tab.work[j-1];   // T[stage-1][j-1], replaced by T[stage][j-1] for the next call
        if (j < stage) {
            for (int d = 0; d < dim; d++) {
                double delta = cur[d] - prev[d];
                double t = cur[d] + factor * delta;
                prev[d] = cur[d];
                cur[d] = t;
            }
        } else {
            // Last column: also the error |T[stage][stage] - T[stage-1][stage-1]| (prev = T[stage-1][stage-1])
            for (int d = 0; d < dim; d++) {
                double delta = cur[d] - prev[d];
                double t = cur[d] + factor * delta;
                double scale = std::max(1.0, std::fabs(t));
                double err = std::fabs(t - prev[d]) / scale;
                if (err > max_err) max_err = err;
                prev[d] = cur[d];
                cur[d] = t;
            }
        }
    }
    return max_err;
}

// ============================================================================
// Full GBS step with adaptive order (k_fixed = 0) or a fixed number of stages
// ============================================================================

// Sum of |w_k| of the extrapolation to h -> 0 from stages 0..stage, w_k = prod_{j != k} n_k^2 / (n_k^2 - n_j^2)
static double neville_weight_sum(const int* seq, int stage) {
    double lam = 0.0;
    for (int k = 0; k <= stage; k++) {
        double w = 1.0;
        for (int j = 0; j <= stage; j++)
            if (j != k) w *= double(seq[k]) * seq[k] / (double(seq[k]) * seq[k] - double(seq[j]) * seq[j]);
        lam += std::fabs(w);
    }
    return lam;
}

void gbs_step(const GbsParams& par, LogHState& state, double ds, GbsTable& tab, int computed,
              const LogHStepStart* start, GbsStepOut& out) {
    int dim = state.n_bodies * (state.use_aux_vel ? 9 : 6) + (state.has_spins ? 12 : 0) + 2;   // gbs_pack_state
    const bool fixed = par.k_fixed > 0;
    const int k_max = fixed ? std::min(par.k_fixed, GBS_MAX_STAGES) : par.k_max;
    out.accepted = false;
    for (int stage = 0; stage < k_max; stage++) {
        if (stage > computed) {
            LogHState result;   // midpoint copies state into it (masses, B, n_bodies, etc.)
            gbs_midpoint(state, result, par.seq[stage], ds, par.G, par.c_code, par.pn_order, start);
            gbs_pack_state(result, tab.table[stage], dim);
            computed = stage;
        }

        // Always call extrapolate, in stage order — stage 0 starts the Neville table (returns 1.0).
        // Without this, the table row is uninitialized garbage and stage 1 can
        // falsely report convergence.
        const double err = gbs_extrapolate(par.seq, tab, stage, dim);
        out.stage = stage;
        out.err = err;

        if (stage > 0) {
            // Tolerance at least the round-off floor of the error estimate: rounding of ~eps in each stage is
            // amplified by the sum of the absolute Neville weights Lambda (Bulirsch ~9, Deuflhard 119-553 at 8-10
            // stages). Below it the step size collapsed (ETA_GBS 1e-14: 1.2e6 steps per orbit, test gbs_eta_floor)
            const double eta = std::max(par.eta, 1e3 * neville_weight_sum(par.seq, stage) * 2.220446049250313e-16);
            if (fixed ? stage == k_max - 1 : err < eta) {
                // Converged: unpack best estimate from work[stage]
                double chi_norm[4];
                if (state.has_spins) spin_norms(state, chi_norm);
                gbs_unpack_state(tab.work[stage], state);
                // |chi| is an exact invariant of dS/dt = Omega × S (the leapfrog rotations keep it); the
                // polynomial extrapolation does not, to within eta_gbs: restore the norms
                if (state.has_spins) restore_spin_norms(state, chi_norm);

                // Suggest next step size
                double safety = 0.8;
                double order = 2.0 * (stage + 1);
                double ds_next = ds * safety * std::pow(eta / (err + 1e-30), 1.0 / order);
                out.ds_next = std::min(ds_next, 4.0 * ds);  // don't grow too fast
                out.accepted = true;
                return;
            }

            // Early exit: if error is enormous after several stages,
            // the step is clearly too large for the dynamics. Bail out
            // early to save computing higher stages with no hope.
            if (!fixed && stage >= 3 && err > 1.0) {
                out.ds_next = 0.5 * ds;
                return;
            }
        }
    }

    // Did not converge — halve step size and try again
    out.ds_next = 0.5 * ds;
}

void logh_gbs_step(const GbsParams& par, double* pos, double* vel, const double* mass, int n, double dt,
                   int n_steps, double* aux_vel, LoghStepResult* res) {
    LogHState state{};
    state.n_bodies = n;
    for (int i = 0; i < 3 * n; i++) {
        state.pos[i] = pos[i];
        state.vel[i] = vel[i];
    }
    for (int i = 0; i < n; i++) state.mass[i] = mass[i];
    state.c_code = par.c_code;
    state.use_aux_vel = par.pn_order > 0 && par.c_code > 0.0;
    if (state.use_aux_vel)
        for (int i = 0; i < 3 * n; i++) state.aux_vel[i] = aux_vel ? aux_vel[i] : vel[i];
    const double E0 = logh_total_energy(state, par.G);
    state.B = -E0;
    const double ds = dt * logh_omega(state) / n_steps;

    GbsTable tab;
    const int pn_order = state.use_aux_vel ? par.pn_order : 0;
    GbsParams p = par;
    p.pn_order = pn_order;
    int rejected = 0;
    for (int s = 0; s < n_steps; s++) {
        LogHStepStart start;
        logh_step_start(state, p.G, p.c_code, pn_order, start);
        GbsStepOut o;
        gbs_step(p, state, ds, tab, -1, &start, o);
        if (!o.accepted) rejected++;
    }

    for (int i = 0; i < 3 * n; i++) {
        pos[i] = state.pos[i];
        vel[i] = state.vel[i];
    }
    if (aux_vel && state.use_aux_vel)
        for (int i = 0; i < 3 * n; i++) aux_vel[i] = state.aux_vel[i];
    res->t = state.t;
    res->dE = logh_total_energy(state, par.G) - E0;
    res->B = state.B;
    res->rejected = rejected;
}
