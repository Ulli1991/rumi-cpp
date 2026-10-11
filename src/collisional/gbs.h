#pragma once

#include "logh.h"
#include <atomic>

// pos + vel + aux_vel per body, 2 spins + 2 auxiliary spins, t, B
constexpr int GBS_DIM_MAX = LOGH_MAX_BODIES * 9 + 12 + 2;

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
    double table[MAX_STAGES][GBS_DIM_MAX];  // pos+vel+aux_vel(+spins)+t+B per stage
    double work[MAX_STAGES][GBS_DIM_MAX];
    double diag_prev[GBS_DIM_MAX];  // T[k-1][k-1] saved between extrapolate() calls

    // Stage-parallel steps: the midpoint integrations of stages k..par_hint run concurrently, par_hint = the
    // stage the previous step converged at. Same result as serial. PAR_HOST: Kokkos host parallel_for (only
    // outside other host-parallel regions); PAR_TASKS: OpenMP tasks inside a task-parallel batch, once *pending
    // (subsystems of the batch not yet started) is 0, i.e. when other threads would otherwise idle
    enum { PAR_OFF, PAR_HOST, PAR_TASKS };
    int par_mode = PAR_OFF;
    const std::atomic<int>* pending = nullptr;
    int par_hint = 4;

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
    void compute_stages(const LogHState& state, int k0, int k1, double ds, double G, double c_code, int pn_order);
    // Pack LogHState into flat array for extrapolation
    void pack_state(const LogHState& state, double* y, int& dim);
    // Unpack flat array back into LogHState
    void unpack_state(const double* y, int dim, LogHState& state);
    // |chi| of spin1, spin2, aux_spin1, aux_spin2, and their restoration after extrapolation
    static void spin_norms(LogHState& state, double norm[4]);
    static void restore_spin_norms(LogHState& state, const double norm[4]);
};
