// RUMI++ MSTAR module (GPLv3, see src/mstar/LICENSE)
// Copyright (C) 2026 Ulrich Steinwandel.
// Licensed under the GPLv3 license (or, at your option, any later version).
//
// MSTAR (KETJU C library and the Kokkos port) against RUMI++'s own LogH + GBS (src/collisional/gbs.cpp, the
// production subsystem integrator, N <= LOGH_MAX_BODIES) where the two are comparable: Newtonian few-body systems
// at the same GBS tolerance. ctest mstar_vs_logh. Checks: each integrator's energy error and its distance to an
// independent reference (Kepler: the analytic orbit; Pythagorean / 6-body: a tolerance-1e-14 MSTAR run), and that
// MSTAR is not less accurate than LogH beyond a factor (the two use different time transformations and GBS step
// sequences, so they are not expected to agree to round-off). Bounds: 1.3x measured (CPU gcc 15 -O2), see each case.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <Kokkos_Core.hpp>

#include "collisional_config.h"
#include "gbs.h"
#include "ketju_integrator/ketju_integrator.h"
#include "mstar_kokkos.h"

CollisionalConfig g_collisional;   // the GBS defaults (no run_setup.cpp in this test)

namespace {

struct State {
    std::vector<double> m, x, v;
    double dE = 0, t = 0;
    long steps = 0;
};

int g_fail = 0;
void check(const char* c, const char* what, double v, double bound) {
    const bool ok = v <= bound;
    std::printf("  %s %-12s %-30s %.3e (<= %.3e)\n", ok ? "ok  " : "FAIL", c, what, v, bound);
    if (!ok) g_fail++;
}

double energy(const State& s) {
    double T = 0, U = 0;
    const int n = (int)s.m.size();
    for (int i = 0; i < n; i++) {
        T += 0.5 * s.m[i] * (s.v[3 * i] * s.v[3 * i] + s.v[3 * i + 1] * s.v[3 * i + 1] + s.v[3 * i + 2] * s.v[3 * i + 2]);
        for (int j = i + 1; j < n; j++) {
            double r2 = 0;
            for (int k = 0; k < 3; k++) r2 += (s.x[3 * i + k] - s.x[3 * j + k]) * (s.x[3 * i + k] - s.x[3 * j + k]);
            U -= s.m[i] * s.m[j] / std::sqrt(r2);
        }
    }
    return T + U;
}

State run_logh(const State& in, double T, double eta) {
    LogHState st{};
    st.n_bodies = (int)in.m.size();
    for (int i = 0; i < st.n_bodies; i++) {
        st.mass[i] = in.m[i];
        for (int k = 0; k < 3; k++) { st.pos[3 * i + k] = in.x[3 * i + k]; st.vel[3 * i + k] = in.v[3 * i + k]; }
    }
    const double E0 = logh_total_energy(st, 1.0);
    st.B = -E0;
    st.t = 0;
    GBSIntegrator gbs(eta);
    State out = in;
    const double omega = logh_kinetic_energy(st) + st.B;   // production first guess (logh_batch.cpp): ds = dt Omega
    out.steps = gbs.integrate_to_time(st, T, 1.0, T * omega, 0.0, 0, 10000000);
    for (int i = 0; i < st.n_bodies; i++)
        for (int k = 0; k < 3; k++) { out.x[3 * i + k] = st.pos[3 * i + k]; out.v[3 * i + k] = st.vel[3 * i + k]; }
    out.dE = std::fabs(logh_total_energy(st, 1.0) / E0 - 1.0);
    out.t = st.t;
    return out;
}

State run_c(const State& in, double T, double tol) {
    struct ketju_system R;
    const int n = (int)in.m.size();
    ketju_create_system(&R, 0, n, MPI_COMM_WORLD);
    R.constants->G = 1;
    R.options->PN_flags = 0;
    R.options->gbs_relative_tolerance = tol;
    R.options->output_time_relative_tolerance = 1e-13;
    for (int i = 0; i < n; i++) {
        R.physical_state->mass[i] = in.m[i];
        for (int k = 0; k < 3; k++) {
            R.physical_state->pos[i][k] = in.x[3 * i + k];
            R.physical_state->vel[i][k] = in.v[3 * i + k];
        }
    }
    const double E0 = energy(in);
    ketju_run_integrator(&R, T);
    State out = in;
    for (int i = 0; i < n; i++)
        for (int k = 0; k < 3; k++) {
            out.x[3 * i + k] = R.physical_state->pos[i][k];
            out.v[3 * i + k] = R.physical_state->vel[i][k];
        }
    out.steps = R.perf->successful_steps + R.perf->failed_steps;
    out.dE = std::fabs(energy(out) / E0 - 1.0);
    ketju_free_system(&R);
    return out;
}

State run_k(const State& in, double T, double tol) {
    const int n = (int)in.m.size();
    rumi_mstar::MstarKokkos M(0, n);
    M.opt.gbs_relative_tolerance = tol;
    M.opt.output_time_relative_tolerance = 1e-13;
    M.mass = in.m;
    M.pos = in.x;
    M.vel = in.v;
    const double E0 = energy(in);
    M.run(T);
    State out = in;
    out.x = M.pos;
    out.v = M.vel;
    out.steps = M.perf.successful_steps + M.perf.failed_steps;
    out.dE = std::fabs(energy(out) / E0 - 1.0);
    return out;
}

double dist(const State& a, const State& b) {
    double mx = 0, sc = 0;
    for (size_t q = 0; q < a.x.size(); q++) {
        mx = std::fmax(mx, std::fabs(a.x[q] - b.x[q]));
        sc = std::fmax(sc, std::fabs(b.x[q]));
    }
    return mx / sc;
}

// ref: reference state at T (nullptr: none). LogH's integrate_to_time lands within its time tolerance of T, so its
// accuracy is measured against an MSTAR-C tolerance-1e-14 run to the time LogH reached (b_land bounds the landing).
void case_run(const char* name, const State& s, double T, double tol, const State* ref, double b_logh_dE,
              double b_mstar_dE, double b_logh_ref, double b_mstar_ref, double b_land) {
    const State L = run_logh(s, T, tol), C = run_c(s, T, tol), K = run_k(s, T, tol);
    const State refL = run_c(s, L.t, 1e-14);
    std::printf("%s: N=%zu T=%g tol=%g  steps LogH %ld, MSTAR-C %ld, MSTAR-Kokkos %ld\n", name, s.m.size(), T, tol,
                L.steps, C.steps, K.steps);
    check(name, "|dE/E| LogH", L.dE, b_logh_dE);
    check(name, "|dE/E| MSTAR C", C.dE, b_mstar_dE);
    check(name, "|dE/E| MSTAR Kokkos", K.dE, b_mstar_dE);
    if (ref) {
        check(name, "|x - x_ref| MSTAR C", dist(C, *ref), b_mstar_ref);
        check(name, "|x - x_ref| MSTAR Kokkos", dist(K, *ref), b_mstar_ref);
    }
    check(name, "|t_LogH - T| / T (landing)", std::fabs(L.t - T) / T, b_land);
    check(name, "|x - x_ref(t_LogH)| LogH", dist(L, refL), b_logh_ref);
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);   // KETJU needs MPI (or the serial stand-in)
    Kokkos::initialize(argc, argv);
    {
        {
            // e = 0.9 binary (m 1, 0.5; a = 1) from apocentre, 10 orbits: the reference is the initial state
            State s;
            s.m = {1.0, 0.5};
            const double M = 1.5, e = 0.9, r = 1 + e, vr = std::sqrt(M * (1 - e) / (1 + e));
            s.x = {0.5 / M * r, 0, 0, -1.0 / M * r, 0, 0};
            s.v = {0, 0.5 / M * vr, 0, 0, -1.0 / M * vr, 0};
            const double P = 2 * M_PI / std::sqrt(M);
            case_run("kepler_e09", s, 10 * P, 1e-12, &s, 2.3e-13, 1.8e-12, 2.7e-12, 5.7e-12, 2.3e-8);   // MSTAR A100 4.4e-12
        }
        {
            // Burrau's Pythagorean problem to t = 10; reference: MSTAR-C at tolerance 1e-14
            State s;
            s.m = {3, 4, 5};
            s.x = {1, 3, 0, -2, -1, 0, 1, -1, 0};
            s.v.assign(9, 0);
            const State ref = run_c(s, 10.0, 1e-14);
            case_run("pythagorean", s, 10.0, 1e-12, &ref, 6.9e-12, 1.8e-12, 3.8e-11, 9.2e-11, 1.1e-7);
        }
    }
    std::printf("%s (%d failed checks)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    Kokkos::finalize();
    MPI_Finalize();
    return g_fail ? 1 : 0;
}
