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

    // Extrapolation table (Neville-Aitken)
    // table[i] stores RAW midpoint result M(n_i) — never modified by extrapolation
    // work[i] stores the Neville table entries during extrapolation
    double table[MAX_STAGES][GBS_DIM_MAX];  // pos+vel+aux_vel(+spins)+t+B per stage
    double work[MAX_STAGES][GBS_DIM_MAX];
    double diag_prev[GBS_DIM_MAX];  // T[k-1][k-1] saved between extrapolate() calls

    GBSIntegrator(double eta = 1e-10) : eta_gbs(eta) {}

    // Modified midpoint method: integrate LogHState by n_substeps of ds_total/n
    void midpoint(const LogHState& in, LogHState& out,
                  int n_substeps, double ds_total, double G,
                  double c_code = 0.0, int pn_order = 0);

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
    // Pack LogHState into flat array for extrapolation
    void pack_state(const LogHState& state, double* y, int& dim);
    // Unpack flat array back into LogHState
    void unpack_state(const double* y, int dim, LogHState& state);
    // |chi| of spin1, spin2, aux_spin1, aux_spin2, and their restoration after extrapolation
    static void spin_norms(LogHState& state, double norm[4]);
    static void restore_spin_norms(LogHState& state, const double norm[4]);
};
