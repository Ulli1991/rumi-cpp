#pragma once

#include "logh.h"

// GBS (Gragg-Bulirsch-Stoer) extrapolation integrator
// for high-accuracy integration of LogH equations of motion

struct GBSIntegrator {
    // Bulirsch-Stoer step sequence (even numbers for symmetric methods)
    static constexpr int STEP_SEQ[] = {2, 4, 6, 8, 12, 16, 24, 32, 48, 64};
    static constexpr int MAX_STAGES = 10;
    double eta_gbs;  // accuracy parameter (default 1e-10)

    // Extrapolation table (Neville-Aitken)
    // table[i] stores RAW midpoint result M(n_i) — never modified by extrapolation
    // work[i] stores the Neville table entries during extrapolation
    double table[MAX_STAGES][LOGH_MAX_BODIES * 9 + 2];  // pos+vel+aux_vel+t+B per stage
    double work[MAX_STAGES][LOGH_MAX_BODIES * 9 + 2];
    double diag_prev[LOGH_MAX_BODIES * 9 + 2];  // T[k-1][k-1] saved between extrapolate() calls

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
};
