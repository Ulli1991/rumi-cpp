#pragma once
// One GBS step of a LogH subsystem (leapfrog stages + polynomial extrapolation), plain C++ without Kokkos
// (tests/enzyme differentiates it). GBSIntegrator::step (adaptive: error test, rejection, step-size suggestion) and
// the fixed-path variant (k_fixed stages, always accepted: a smooth function of the state for a fixed signature)
// run the same code; integrate_to_time (step-size control, SDAR kappa jumps, collisions) stays in gbs.cpp.
#include "logh.h"

constexpr int GBS_MAX_STAGES = 10;
// pos + vel + aux_vel per body, 2 spins + 2 auxiliary spins, t, B
constexpr int GBS_DIM_MAX = LOGH_MAX_BODIES * 9 + 12 + 2;

struct GbsParams {
    const int* seq = nullptr;   // leapfrog substeps of stage k (seq[0 .. k_max-1])
    int k_max = 8;              // stages at most
    double eta = 1e-10;         // convergence tolerance on the extrapolation error (adaptive path)
    int k_fixed = 0;            // > 0: fixed path, exactly k_fixed stages (>= 2), accepted without the error test
    double G = 1.0, c_code = 0.0;
    int pn_order = 0;
};

// Extrapolation table (Neville-Aitken): table[k] the raw stage result M(n_k), work[j] the last row T[k][j]
struct GbsTable {
    double table[GBS_MAX_STAGES][GBS_DIM_MAX];
    double work[GBS_MAX_STAGES][GBS_DIM_MAX];
};

struct GbsStepOut {
    bool accepted = false;
    int stage = -1;          // stage of acceptance (or the last one tried)
    double err = 0.0;        // extrapolation error estimate at that stage
    double ds_next = 0.0;    // suggested next step (adaptive path)
};

// Fields a GBS step changes (pos, vel, aux_vel, spins, t, B) to and from a flat array
void gbs_pack_state(const LogHState& state, double* y, int& dim);
void gbs_unpack_state(const double* y, LogHState& state);
// n_substeps leapfrog steps of ds_total / n_substeps (start: logh_step_start of in, or nullptr)
void gbs_midpoint(const LogHState& in, LogHState& out, int n_substeps, double ds_total, double G, double c_code,
                  int pn_order, const LogHStepStart* start);
// Adds row `stage` to the Neville table (calls in stage order 0, 1, ...); returns the error estimate
double gbs_extrapolate(const int* seq, GbsTable& tab, int stage, int dim);

// One GBS step of ds. table[0 .. computed] already hold the stage results of this state (-1: none; stage-parallel
// callers); start: logh_step_start of the state, or nullptr (SDAR states). Accepted: state advanced, else unchanged.
void gbs_step(const GbsParams& par, LogHState& state, double ds, GbsTable& tab, int computed,
              const LogHStepStart* start, GbsStepOut& out);

// Plain-array front end: n bodies (pos, vel in and out, mass), Omega0 = T + B with B = -(T + U) at the start,
// n_steps GBS steps of ds = dt Omega0 / n_steps each; aux_vel (PN, nullptr: w = v at the start and discarded).
// No SDAR, spins, collisions. For par.k_fixed > 0 the result is smooth in pos, vel, mass, dt.
struct LoghStepResult {
    double t = 0.0;          // physical time reached (dt up to the integration error of t(s))
    double dE = 0.0;         // (T + U)(end) - (T + U)(start); Newtonian: the energy error
    double B = 0.0;          // B at the end (PN: B - B0 is the energy radiated / exchanged)
    int rejected = 0;        // adaptive path: steps not converged (state left at the last accepted step)
};
void logh_gbs_step(const GbsParams& par, double* pos, double* vel, const double* mass, int n, double dt,
                   int n_steps, double* aux_vel, LoghStepResult* res);
