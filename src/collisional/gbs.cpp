#include "gbs.h"
#include "kepler.h"
#include "collisional_config.h"
#include <limits>
#include <Kokkos_Core.hpp>
#include <atomic>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

constexpr int GBSIntegrator::STEP_SEQ[];
constexpr int GBSIntegrator::STEP_SEQ_DEUFLHARD[];
const int* GBSIntegrator::seq() { return g_collisional.gbs_deuflhard ? STEP_SEQ_DEUFLHARD : STEP_SEQ; }
int GBSIntegrator::kmax() { return std::max(2, std::min(MAX_STAGES, g_collisional.gbs_kmax)); }

// NBODY_GBS_STATS=1: totals over the run, printed at exit (calls, GBS steps, failed and overshooting steps, the
// stage at which converged steps stopped). Counted only then: the host batch integrates subsystems on all
// threads, and atomic updates of shared counters on every step cost cache-line traffic
namespace {
const bool gbs_stats_on = std::getenv("NBODY_GBS_STATS") != nullptr;
struct GbsStats {
    std::atomic<long long> calls{0}, steps{0}, failed{0}, overshoot{0}, stage[GBSIntegrator::MAX_STAGES] = {};
    ~GbsStats() {
        if (!gbs_stats_on || calls == 0) return;
        std::printf("GBS stats: %lld calls, %lld steps (%.1f / call), failed %lld, overshoot %lld; converged at stage:",
                    calls.load(), steps.load(), (double)steps / calls, failed.load(), overshoot.load());
        for (int k = 0; k < GBSIntegrator::MAX_STAGES; k++) std::printf(" %lld", stage[k].load());
        std::printf("\n");
    }
} g_gbs_stats;
}  // namespace

// Midpoint results of stages k0..k1 into tab.table[] on the host threads, the largest stage handed out first
void GBSIntegrator::compute_stages(const LogHState& state, int k0, int k1, double ds, double G,
                                   double c_code, int pn_order, const LogHStepStart* start) {
    using Policy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>;
    Kokkos::parallel_for("gbs_stages", Policy(0, k1 - k0 + 1).set_chunk_size(1), [&](int i) {
        const int k = k1 - i;
        LogHState result;   // midpoint copies state into it
        gbs_midpoint(state, result, seq()[k], ds, G, c_code, pn_order, start);
        int d = 0;
        gbs_pack_state(result, tab.table[k], d);
    });
}

bool GBSIntegrator::step(LogHState& state, double ds, double G, double& ds_next,
                          double c_code, int pn_order) {
    int dim = 0;
    gbs_pack_state(state, y_start, dim);   // the integrated fields of the state before the step (restore_step)

    // Values at the start of the first substep, the same for every stage (not used by the SDAR steps)
    LogHStepStart start_buf;
    const LogHStepStart* start = nullptr;
    if (state.n_sd == 0) {
        logh_step_start(state, G, c_code, pn_order, start_buf);
        start = &start_buf;
    }

    GbsParams par;
    par.seq = seq();
    par.k_max = kmax();
    par.eta = eta_gbs;
    par.G = G;
    par.c_code = c_code;
    par.pn_order = pn_order;
    // Stage-parallel: the substep divisions are independent (Rantala et al. 2020, MSTAR, Sec. 4); stages 0..k1,
    // k1 = the stage the last step converged at, at once, the later ones (if needed) serially. The extrapolation
    // runs in the same order on the same table, so the result is the serial one; unused stages are discarded
    int computed = -1;
    if (par_stages) {
        computed = std::min(par.k_max - 1, std::max(0, par_hint));
        compute_stages(state, 0, computed, ds, G, c_code, pn_order, start);
    }
    GbsStepOut out;
    gbs_step(par, state, ds, tab, computed, start, out);
    ds_next = out.ds_next;
    if (out.accepted) {
        if (gbs_stats_on) g_gbs_stats.stage[out.stage]++;
        par_hint = out.stage;
    }
    return out.accepted;
}

// ============================================================================
// Collision scan (FROST-CLUSTERS I App. B3)
// ============================================================================

namespace {
struct PairState { double r[3], v[3], rr, rv, mu; };
PairState pair_state(const LogHState& s, int i, int j, double G) {
    PairState q;
    for (int k = 0; k < 3; k++) { q.r[k] = s.pos[j*3+k] - s.pos[i*3+k]; q.v[k] = s.vel[j*3+k] - s.vel[i*3+k]; }
    q.rr = std::sqrt(q.r[0]*q.r[0] + q.r[1]*q.r[1] + q.r[2]*q.r[2]);
    q.rv = q.r[0]*q.v[0] + q.r[1]*q.v[1] + q.r[2]*q.v[2];
    q.mu = G * (s.mass[i] + s.mass[j]);
    return q;
}
// The slowed relative motion of an SDAR binary runs 1/kappa times slower: its pericentre is kappa tau away
double sd_factor(const LogHState& s, int i, int j) {
    for (int k = 0; k < s.n_sd; k++)
        if (s.sd_a[k] == i && s.sd_b[k] == j) return s.sd_kappa[k];
    return 1.0;
}
struct CollScan {
    int ci = -1, cj = -1;   // collision now (smallest r / r_coll)
    double r = 0.0, best = 1e300;
    int pi = -1, pj = -1;   // candidate with the earliest pericentre
    double tau = std::numeric_limits<double>::infinity(), tscale = 0.0;
    bool skip_receding = false;   // the skipped pair has passed its pericentre
};
void collision_scan(const LogHState& s, double G, const double* coll, int skip_i, int skip_j, CollScan& out) {
    for (int i = 0; i < s.n_bodies; i++)
        for (int j = i + 1; j < s.n_bodies; j++) {
            const double rc = coll[i * LOGH_MAX_BODIES + j];
            if (!(rc > 0.0)) continue;
            const PairState q = pair_state(s, i, j, G);
            if (q.rr < rc) {
                if (q.rr / rc < out.best) { out.best = q.rr / rc; out.ci = i; out.cj = j; out.r = q.rr; }
                continue;
            }
            if (i == skip_i && j == skip_j) {
                if (q.rv > 0.0) out.skip_receding = true;
                continue;
            }
            double rp = 0.0;
            const double tau = kepler_time_to_pericentre(q.r, q.v, q.mu, &rp) * sd_factor(s, i, j);
            if (rp < rc && tau < out.tau) {
                out.tau = tau; out.pi = i; out.pj = j;
                out.tscale = std::sqrt(rp * rp * rp / q.mu);   // duration of the pericentre passage
            }
        }
}
}  // namespace

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

    // Collision candidates (FROST-CLUSTERS I App. B3): t_goal is the next end time of the step-size control, the
    // earliest candidate pericentre inside the interval or t_target. The same control (ds = Omega (t_goal - t),
    // secant on overshoot) then lands a step on the pericentre; the osculating elements re-aim it after every
    // step, so the landing iterates onto the actual pericentre (as NBODY6's chain, Aarseth 2003, DERQP / TPERI).
    // peri_tol: landing tolerance, 1e-3 of the pericentre passage time sqrt(r_p^3 / (G m)).
    coll_i = coll_j = -1;
    double t_goal = t_target;
    int peri_i = -1, peri_j = -1, skip_i = -1, skip_j = -1, peri_aims = 0;
    double peri_tol = 0.0;
    const double dir = t_target >= state.t ? 1.0 : -1.0;
    auto scan = [&]() -> bool {
        if (!coll) return false;
        CollScan cs;
        collision_scan(state, G, coll, skip_i, skip_j, cs);        if (cs.skip_receding) skip_i = skip_j = -1;
        if (cs.ci >= 0) { coll_i = cs.ci; coll_j = cs.cj; coll_rsep = cs.r; return true; }
        const int prev_i = peri_i, prev_j = peri_j;
        t_goal = t_target;
        peri_i = peri_j = -1;
        if (cs.pi >= 0 && dir * (t_target - state.t) > cs.tau) {
            t_goal = state.t + dir * cs.tau;
            peri_i = cs.pi; peri_j = cs.pj;
            peri_tol = 1e-3 * cs.tscale;
            peri_aims = (peri_i == prev_i && peri_j == prev_j) ? peri_aims + 1 : 0;
        }
        return false;
    };
    // At a candidate's pericentre (landed within peri_tol, or just past it, or re-aimed 8 times): decide on the
    // osculating pericentre distance of this passage; true: collision (stop)
    auto decide_at_peri = [&]() -> bool {
        if (peri_i < 0) return false;
        const PairState q = pair_state(state, peri_i, peri_j, G);
        double rp = 0.0;
        const double tau = kepler_time_to_pericentre(q.r, q.v, q.mu, &rp) * sd_factor(state, peri_i, peri_j);
        const bool at_peri = q.rv >= 0.0 || tau <= peri_tol || peri_aims >= 8;
        if (!at_peri) return false;
        peri_landings++;
        const double rc = coll[peri_i * LOGH_MAX_BODIES + peri_j];
        if (std::min(rp, q.rr) < rc) {
            coll_i = peri_i; coll_j = peri_j; coll_rsep = std::min(rp, q.rr);
            return true;
        }
        peri_misses++;                       // perturbed: the actual pericentre clears r_coll
        skip_i = peri_i; skip_j = peri_j;    // not this passage again
        peri_i = peri_j = -1;
        return false;
    };
    scan();   // a collision already at the start stops before the first step
    for (int iter = 0; iter < iter_cap && coll_i < 0; iter++) {
        // Candidate pericentre reached: decide there, then aim at the next goal
        if (peri_i >= 0 && dir * (t_goal - state.t) <= peri_tol) {
            if (decide_at_peri()) break;
            if (scan()) break;
        }
        // Check if we've reached the target time
        double dt_remain = t_goal - state.t;
        const double dt_left = t_target - state.t;
        if (peri_i < 0 && std::fabs(dt_left) < time_tol * std::fabs(t_target)) break;
        if (iter_cap != max_iter && iter >= max_iter &&
            iter >= sdar->steps_per_orbit * (sd_orbits_done + std::fabs(dt_left) / sd_P_eff))
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

        // For the bisection on overshoot: step() leaves the state unchanged when it fails and keeps the fields it
        // changes for restore_step (no full copy of the state per step)
        const double t_saved = state.t;
        double ds_next;

        bool ok = step(state, ds_try, G, ds_next, c_code, pn_order);
        n_steps++;

        if (!ok) {
            // Step failed (state unchanged) — halve and retry (ds persists to next iteration)
            ds = ds_try * 0.5;
            n_failed++;
            if (gbs_stats_on) g_gbs_stats.failed++;
            continue;
        }

        // Check if we overshot the goal (target time, or a candidate's pericentre by more than peri_tol)
        const double t_over = peri_i >= 0 ? peri_tol : 1e-10 * std::fabs(t_target);
        if ((dt_remain > 0 && state.t > t_goal + t_over) ||
            (dt_remain < 0 && state.t < t_goal - t_over)) {
            // Overshoot: aim at the target with the secant through this attempt (t(s) is smooth over a step),
            // slightly short so that the next step lands inside the tolerance from below
            const double dt_attempt = state.t - t_saved;
            restore_step(state);
            ds = (dt_attempt != 0.0) ? ds_try * (dt_remain / dt_attempt) * (1.0 - 1e-6) : ds_try * 0.5;
            n_overshoot++;
            if (gbs_stats_on) g_gbs_stats.overshoot++;
            continue;
        }

        // Success: use GBS-suggested step size (will be capped by ds_est next iter)
        ds = ds_next;
        // Jumping kappa: new slow-down factors for the next step, energy jump booked into B (Wang+2020 Eq. 24)
        if (sdar) {
            if (sd_P_eff > 0.0) sd_orbits_done += std::fabs(state.t - t_saved) / sd_P_eff;
            const double P_eff = logh_sd_update(state, G, *sdar);
            if (sd_P_eff > 0.0) sd_P_eff = P_eff;
        }
        // Collisions: a step that ended on the aimed pericentre, or past it (the aim moved), decides it; then
        // all pairs are scanned again
        if (coll) {
            if (peri_i >= 0) {
                const PairState q = pair_state(state, peri_i, peri_j, G);
                if ((dir * (t_goal - state.t) <= peri_tol || q.rv >= 0.0) && decide_at_peri()) break;
            }
            if (scan()) break;
        }
    }

    if (gbs_stats_on) {
        g_gbs_stats.calls++;
        g_gbs_stats.steps += n_steps;
    }
    double dt_remain = t_target - state.t;
    reached = std::fabs(dt_remain) < time_tol * std::fabs(t_target);
    if (!reached && coll_i < 0) {
        double Omega_final = logh_omega(state);
        std::printf("      GBS: %d steps, t_final=%.6f/%.6f  Omega=%.3e  ds_range=[%.2e,%.2e]  fail=%d overshoot=%d"
                    " *** DID NOT REACH TARGET ***\n",
                    n_steps, state.t, t_target, Omega_final,
                    ds_min_used, ds_max_used, n_failed, n_overshoot);
    }

    return n_steps;
}
