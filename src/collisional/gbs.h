#pragma once

#include "logh.h"
#include "logh_step.h"

// GBS (Gragg-Bulirsch-Stoer) extrapolation integrator
// for high-accuracy integration of LogH equations of motion

// Secular inner binaries of a PN subsystem (logh_batch.cpp, docs/CASES.md C20): body com[q] stands for a binary whose
// internal orbit is advanced secularly; the binary is expanded again when the tide on it grows: hit when
// qf[q] x sum_k m_com m_k / d_k^3 > 1, with qf = r_apo^3 n_orb / (m_a m_b k_ref) (the collapse criterion
// gamma n_orb <= k_ref at the current distances)
struct SecWatch {
    int n = 0;
    int com[LOGH_MAX_BODIES / 2];
    double qf[LOGH_MAX_BODIES / 2];
};
bool sec_watch_hit(const LogHState& state, const SecWatch& w);

struct GBSIntegrator {
    // Bulirsch-Stoer step sequence (even numbers for symmetric methods)
    static constexpr int STEP_SEQ[] = {2, 4, 6, 8, 12, 16, 24, 32, 48, 64};
    // Deuflhard (1983) sequence n_k = 2 (k + 1), as in MSTAR (Rantala et al. 2020, sec. 2.3) and FROST-CLUSTERS
    static constexpr int STEP_SEQ_DEUFLHARD[] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
    static constexpr int MAX_STAGES = GBS_MAX_STAGES;
    // GBS_SEQUENCE (bulirsch | deuflhard) and GBS_KMAX (<= MAX_STAGES; FROST-CLUSTERS I: 8) from g_collisional
    static const int* seq();
    static int kmax();
    double eta_gbs;  // accuracy parameter (default 1e-10)
    // SDAR (nullptr: plain LogH): kappa re-evaluated after every accepted GBS step (jumping kappa, Wang+2020
    // sec. 2.4); the caller sets the initial kappa and removes the slow-down at the end (logh_sd_update)
    const SdarParams* sdar = nullptr;
    // Secular inner binaries (nullptr: none): checked after every accepted step; a hit stops integrate_to_time with
    // sec_hit set (the caller expands the binaries and continues)
    const SecWatch* sec_watch = nullptr;
    bool sec_hit = false;
    bool reached = false;   // integrate_to_time reached t_target (within its tolerance)

    // Collisions inside the integration (FROST-CLUSTERS I, Rantala et al. 2024, App. B3; nullptr: off): pair
    // collision distances coll[i * LOGH_MAX_BODIES + j] (i < j, normalised units, 0: none). After every accepted
    // step each pair is checked: separation < r_coll is a collision; osculating pericentre < r_coll makes the
    // pair a candidate, and the step size is forced so that a step ends on its pericentre (time from the
    // osculating elements, re-aimed after every step: the end-time control with t_peri in place of t_target).
    // There the collision is decided on the osculating pericentre of that passage. integrate_to_time stops at a
    // collision with coll_i < coll_j set (the caller merges the pair and continues).
    const double* coll = nullptr;
    int coll_i = -1, coll_j = -1;
    double coll_rsep = 0.0;     // separation at the decision (pericentre: the osculating pericentre distance)
    int peri_landings = 0;      // statistics: pericentres a step was forced onto
    int peri_misses = 0;        // statistics: candidates whose actual pericentre stayed outside r_coll

    GbsTable tab;   // extrapolation table (logh_step.h)
    // Packed state at the start of the last step() call: the only fields a step changes (pos, vel, aux_vel,
    // spins, t, B), so that restore_step undoes an accepted step without a copy of the whole LogHState
    double y_start[GBS_DIM_MAX];

    // Stage-parallel steps (host threads; only outside other host-parallel regions): the midpoint integrations
    // of stages k..par_hint run concurrently, par_hint = the stage the previous step converged at. Same result
    // as serial
    bool par_stages = false;
    int par_hint = 4;

    GBSIntegrator(double eta = 1e-10) : eta_gbs(eta) {}

    // Full GBS step (gbs_step, logh_step.h): advance LogHState by ds
    // Returns true if converged within eta_gbs tolerance
    // out: converged state, ds_next: suggested next step size
    bool step(LogHState& state, double ds, double G, double& ds_next,
              double c_code = 0.0, int pn_order = 0);

    // Integrate from current time to t_target
    // Iterates on ds until physical time matches t_target
    // Returns actual number of GBS steps taken
    int integrate_to_time(LogHState& state, double t_target, double G,
                          double ds_initial, double c_code = 0.0, int pn_order = 0,
                          int max_iter = 100000);

private:
    void compute_stages(const LogHState& state, int k0, int k1, double ds, double G, double c_code, int pn_order,
                        const LogHStepStart* start);
    // Back to the state before the last step() (overshoot of the end time)
    void restore_step(LogHState& state) { gbs_unpack_state(y_start, state); }
};
