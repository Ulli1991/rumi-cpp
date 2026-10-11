#include "gbs.h"
#include <atomic>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

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
    if (state.has_spins) {
        for (int k = 0; k < 3; k++) y[dim++] = state.spin1[k];
        for (int k = 0; k < 3; k++) y[dim++] = state.spin2[k];
        for (int k = 0; k < 3; k++) y[dim++] = state.aux_spin1[k];
        for (int k = 0; k < 3; k++) y[dim++] = state.aux_spin2[k];
    }
    y[dim++] = state.t;
    y[dim++] = state.B;
}

void GBSIntegrator::unpack_state(const double* y, int dim, LogHState& state) {
    (void)dim;
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

void GBSIntegrator::spin_norms(LogHState& state, double norm[4]) {
    for (int i = 0; i < 4; i++) {
        const double* c = spin_vec(state, i);
        norm[i] = std::sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
    }
}

void GBSIntegrator::restore_spin_norms(LogHState& state, const double norm[4]) {
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

// DEBUG: compute omega from packed state for 2-body
static double omega_from_packed(const double* y, int nb) {
    if (nb != 2) return 0.0;
    double rx = y[3] - y[0];   // pos[1] - pos[0]  (x)
    double ry = y[4] - y[1];   // pos[1] - pos[0]  (y)
    double rz = y[5] - y[2];
    double vx = y[9] - y[6];   // vel[1] - vel[0]  (x)
    double vy = y[10] - y[7];  // vel[1] - vel[0]  (y)
    double rmag = std::sqrt(rx*rx + ry*ry + rz*rz);
    double Lz = rx*vy - ry*vx;
    // mu = G*(m0+m1) = 1.0 in normalized units
    double ex = vy*Lz - rx/rmag;
    double ey = -vx*Lz - ry/rmag;
    return std::atan2(ey, ex);
}

static std::atomic<int> gbs_debug_counter{0};  // host batch integrates subsystems in parallel

// NBODY_GBS_STATS=1: totals over the run, printed at exit (calls, GBS steps, failed and overshooting steps, the
// stage at which converged steps stopped)
namespace {
struct GbsStats {
    std::atomic<long long> calls{0}, steps{0}, failed{0}, overshoot{0}, stage[GBSIntegrator::MAX_STAGES] = {};
    ~GbsStats() {
        if (!std::getenv("NBODY_GBS_STATS") || calls == 0) return;
        std::printf("GBS stats: %lld calls, %lld steps (%.1f / call), failed %lld, overshoot %lld; converged at stage:",
                    calls.load(), steps.load(), (double)steps / calls, failed.load(), overshoot.load());
        for (int k = 0; k < GBSIntegrator::MAX_STAGES; k++) std::printf(" %lld", stage[k].load());
        std::printf("\n");
    }
} g_gbs_stats;
}  // namespace

bool GBSIntegrator::step(LogHState& state, double ds, double G, double& ds_next,
                          double c_code, int pn_order) {
    int dim = 0;
    double y0[GBS_DIM_MAX];
    pack_state(state, y0, dim);

    bool do_debug = (state.n_bodies == 2 && gbs_debug_counter < 5);

    if (do_debug) {
        double w0 = omega_from_packed(y0, 2);
        std::printf("      GBS step #%d: ds=%.6e  omega_in=%.6f\n",
                    gbs_debug_counter.load(), ds, w0);
    }

    for (int stage = 0; stage < MAX_STAGES; stage++) {
        LogHState result;
        result = state;  // copy masses, B, n_bodies, etc.

        midpoint(state, result, STEP_SEQ[stage], ds, G, c_code, pn_order);
        pack_state(result, table[stage], dim);

        if (do_debug) {
            double w_raw = omega_from_packed(table[stage], 2);
            std::printf("        stage %d (n=%d): omega_raw=%.10f  t=%.6f\n",
                        stage, STEP_SEQ[stage], w_raw, table[stage][dim-2]);
        }

        // Always call extrapolate — stage 0 initializes diag_prev (returns 1.0).
        // Without this, diag_prev is uninitialized garbage and stage 1 can
        // falsely report convergence.
        double err = extrapolate(stage, dim);

        if (stage > 0) {
            if (do_debug) {
                double w_ext = omega_from_packed(work[stage], 2);
                std::printf("        extrap[%d]: omega_ext=%.10f  err=%.2e\n",
                            stage, w_ext, err);
            }

            if (err < eta_gbs) {
                g_gbs_stats.stage[stage]++;
                // Converged: unpack best estimate from work[stage]
                double chi_norm[4];
                if (state.has_spins) spin_norms(state, chi_norm);
                unpack_state(work[stage], dim, state);
                // |chi| is an exact invariant of dS/dt = Omega × S (the leapfrog rotations keep it); the
                // polynomial extrapolation does not, to within eta_gbs: restore the norms
                if (state.has_spins) restore_spin_norms(state, chi_norm);

                if (do_debug) {
                    double w_final = omega_from_packed(work[stage], 2);
                    std::printf("        CONVERGED at stage %d: omega_final=%.10f  err=%.2e\n",
                                stage, w_final, err);
                    gbs_debug_counter++;
                }

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
                if (do_debug) gbs_debug_counter++;
                return false;
            }
        }
    }

    // Did not converge — halve step size and try again
    ds_next = 0.5 * ds;
    if (do_debug) gbs_debug_counter++;
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
    // SDAR: a slowed binary's orbit takes kappa P, its dynamical time kappa times longer
    if (state.n_sd > 0) t_dyn = logh_sd_dyn_time(state, G, M_total);

    double Omega0 = logh_omega(state);   // T + B (T_sd + B with slowed binaries)
    if (Omega0 < 1e-30) Omega0 = 1e-30;

    // Scale ds so each GBS step covers at most ~1/dyn_cap dynamical times (default 3; was 10, which cost 4x the
    // GBS steps at unchanged accuracy on the binary, PN, merger and dense-core tests; NBODY_GBS_DYNCAP overrides).
    // BIFROST convention: dt = ds / Omega, so ds = dt * Omega.
    // This ensures stage 0 (2 substeps) produces a qualitatively correct
    // solution, which is required for GBS polynomial extrapolation.
    static const double dyn_cap = std::getenv("NBODY_GBS_DYNCAP") ? std::atof(std::getenv("NBODY_GBS_DYNCAP")) : 3.0;
    double ds_dyn = t_dyn * Omega0 / dyn_cap;
    double ds = std::min(ds_initial, ds_dyn);

    // Relative tolerance on the end time. 1e-12 lands subsystems exactly on the step time, but on the dense-core
    // case it samples the oscillating tidal-coupling floor at another binary phase: |dE/E| ~1 % higher on 4 of 4
    // seeds. Kept at 1e-4 (energy first); NBODY_GBS_TTOL overrides.
    static const double time_tol = std::getenv("NBODY_GBS_TTOL") ? std::atof(std::getenv("NBODY_GBS_TTOL")) : 1e-4;
    int n_steps = 0;
    int n_failed = 0;
    int n_overshoot = 0;
    double ds_min_used = ds;
    double ds_max_used = ds;

    // SDAR: the step limit follows the slowed orbits actually integrated (kappa changes during the call): at
    // most steps_per_orbit x (slowed orbits done + remaining time / current shortest slowed orbit), at least
    // max_iter (the caller's limit at the initial kappa), at most 4e6
    double sd_orbits_done = 0.0, sd_P_eff = 0.0;
    if (sdar && sdar->steps_per_orbit > 0.0) {
        LogHState probe = state;   // current shortest slowed orbit; no jump (same kappa as the caller set)
        sd_P_eff = logh_sd_update(probe, G, *sdar);
    }
    const int iter_cap = (sdar && sdar->steps_per_orbit > 0.0) ? std::max(max_iter, 4000000) : max_iter;
    for (int iter = 0; iter < iter_cap; iter++) {
        // Check if we've reached the target time
        double dt_remain = t_target - state.t;
        if (std::fabs(dt_remain) < time_tol * std::fabs(t_target)) break;
        if (iter_cap != max_iter && iter >= max_iter &&
            iter >= sdar->steps_per_orbit * (sd_orbits_done + std::fabs(dt_remain) / sd_P_eff))
            break;

        // Recompute dynamical time constraint at current state
        // Prevents ds from growing too large when orbit approaches periapsis
        double r_min_now_sq = 1e60;
        for (int i = 0; i < state.n_bodies; i++) {
            for (int j = i + 1; j < state.n_bodies; j++) {
                double dx = state.pos[j*3] - state.pos[i*3];
                double dy = state.pos[j*3+1] - state.pos[i*3+1];
                double dz = state.pos[j*3+2] - state.pos[i*3+2];
                double r2 = dx*dx + dy*dy + dz*dz;
                if (r2 < r_min_now_sq) r_min_now_sq = r2;
            }
        }
        double r_min_now = std::sqrt(r_min_now_sq);
        double t_dyn_now = std::sqrt(r_min_now * r_min_now * r_min_now / (G * M_total));
        if (state.n_sd > 0) t_dyn_now = logh_sd_dyn_time(state, G, M_total);

        // Estimate ds from dt_remain * Omega (BIFROST: ds = dt * Omega)
        double Omega = logh_omega(state);
        if (Omega < 1e-30) Omega = 1e-30;
        double ds_est = dt_remain * Omega;

        // Also cap ds by dynamical time (BIFROST: ds = dt * Omega)
        double ds_dyn_now = t_dyn_now * Omega / dyn_cap;

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
            g_gbs_stats.failed++;
            continue;
        }

        // Check if we overshot the target time
        if ((dt_remain > 0 && state.t > t_target + 1e-10 * std::fabs(t_target)) ||
            (dt_remain < 0 && state.t < t_target - 1e-10 * std::fabs(t_target))) {
            // Overshoot: aim at the target with the secant through this attempt (t(s) is smooth over a step),
            // slightly short so that the next step lands inside the tolerance from below
            const double dt_attempt = state.t - saved.t;
            state = saved;
            ds = (dt_attempt != 0.0) ? ds_try * (dt_remain / dt_attempt) * (1.0 - 1e-6) : ds_try * 0.5;
            n_overshoot++;
            g_gbs_stats.overshoot++;
            continue;
        }

        // Success: use GBS-suggested step size (will be capped by ds_est next iter)
        ds = ds_next;
        // Jumping kappa: new slow-down factors for the next step, energy jump booked into B (Wang+2020 Eq. 24)
        if (sdar) {
            if (sd_P_eff > 0.0) sd_orbits_done += std::fabs(state.t - saved.t) / sd_P_eff;
            const double P_eff = logh_sd_update(state, G, *sdar);
            if (sd_P_eff > 0.0) sd_P_eff = P_eff;
        }
    }

    g_gbs_stats.calls++;
    g_gbs_stats.steps += n_steps;
    double dt_remain = t_target - state.t;
    reached = std::fabs(dt_remain) < time_tol * std::fabs(t_target);
    if (!reached) {
        double Omega_final = logh_omega(state);
        std::printf("      GBS: %d steps, t_final=%.6f/%.6f  Omega=%.3e  ds_range=[%.2e,%.2e]  fail=%d overshoot=%d"
                    " *** DID NOT REACH TARGET ***\n",
                    n_steps, state.t, t_target, Omega_final,
                    ds_min_used, ds_max_used, n_failed, n_overshoot);
    }

    return n_steps;
}
