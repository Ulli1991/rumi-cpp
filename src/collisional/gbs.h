#pragma once

#include "logh.h"

// pos + vel + aux_vel per body, 2 spins + 2 auxiliary spins, t, B
constexpr int GBS_DIM_MAX = LOGH_MAX_BODIES * 9 + 12 + 2;

// GBS (Gragg-Bulirsch-Stoer) extrapolation integrator
// for high-accuracy integration of LogH equations of motion

struct GBSIntegrator {
    // Bulirsch-Stoer step sequence (even numbers for symmetric methods)
    static constexpr int STEP_SEQ[] = {2, 4, 6, 8, 12, 16, 24, 32, 48, 64};
    static constexpr int MAX_STAGES = 10;
    double eta_gbs;  // accuracy parameter (default 1e-10)
    // SDAR (nullptr: plain LogH): kappa re-evaluated after every accepted GBS step (jumping kappa, Wang+2020
    // sec. 2.4); the caller sets the initial kappa and removes the slow-down at the end (logh_sd_update)
    const SdarParams* sdar = nullptr;
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

    // Extrapolation table (Neville-Aitken)
    // table[i] stores RAW midpoint result M(n_i) — never modified by extrapolation
    // work[j] stores the last row T[k][j] of the Neville table (extrapolate adds row k)
    double table[MAX_STAGES][GBS_DIM_MAX];  // pos+vel+aux_vel(+spins)+t+B per stage
    double work[MAX_STAGES][GBS_DIM_MAX];

    // Stage-parallel steps (host threads; only outside other host-parallel regions): the midpoint integrations
    // of stages k..par_hint run concurrently, par_hint = the stage the previous step converged at. Same result
    // as serial
    bool par_stages = false;
    int par_hint = 4;

    GBSIntegrator(double eta = 1e-10) : eta_gbs(eta) {}

    // Modified midpoint method: integrate LogHState by n_substeps of ds_total/n
    // start: the values at the start of the first substep (logh_step_start of in; nullptr: computed here)
    void midpoint(const LogHState& in, LogHState& out,
                  int n_substeps, double ds_total, double G,
                  double c_code = 0.0, int pn_order = 0, const LogHStepStart* start = nullptr);

    // Rational polynomial extrapolation (Stoer & Bulirsch)
    // Returns max relative error across all extrapolated components
    double extrapolate(int stage, int dim);

    // Full GBS step: advance LogHState by ds
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
    // Pack LogHState into flat array for extrapolation
    void pack_state(const LogHState& state, double* y, int& dim);
    // Unpack flat array back into LogHState
    void unpack_state(const double* y, int dim, LogHState& state);
    // |chi| of spin1, spin2, aux_spin1, aux_spin2, and their restoration after extrapolation
    static void spin_norms(LogHState& state, double norm[4]);
    static void restore_spin_norms(LogHState& state, const double norm[4]);
};
