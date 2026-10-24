// RUMI++ MSTAR module (GPLv3, see src/mstar/LICENSE)
// Copyright (C) 2026 Ulrich Steinwandel.
// Licensed under the GPLv3 license (or, at your option, any later version).
//
// Scaling benchmark of the Kokkos MSTAR port against the KETJU C library (one host core): Plummer spheres of
// N = 16 .. 4096 (G = M = 1, virial units) integrated for a fixed number of GBS steps (attempted, tolerance 1e-10);
// wall time per step, launches per step, MST and host-sync shares; plus the launch / launch+fence / reduce+readback
// latency of the device. Usage: mstar-bench [N_max] [steps] [C_N_max] [team_N_max]

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <Kokkos_Core.hpp>

#include "ketju_integrator/ketju_integrator.h"
#include "mstar_kokkos.h"

namespace {
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

void plummer(int n, unsigned seed, std::vector<double>& m, std::vector<double>& x, std::vector<double>& v) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    m.assign(n, 1.0 / n);
    x.assign(3 * n, 0);
    v.assign(3 * n, 0);
    const double rs = 3 * M_PI / 16;
    for (int i = 0; i < n; i++) {
        double r;
        do { r = 1.0 / std::sqrt(std::pow(U(g), -2.0 / 3.0) - 1.0); } while (r > 10);
        const double ct = 2 * U(g) - 1, ph = 2 * M_PI * U(g), st = std::sqrt(1 - ct * ct);
        x[3 * i] = rs * r * st * std::cos(ph); x[3 * i + 1] = rs * r * st * std::sin(ph); x[3 * i + 2] = rs * r * ct;
        double q, y;
        do { q = U(g); y = 0.1 * U(g); } while (y > q * q * std::pow(1 - q * q, 3.5));
        const double ve = q * std::sqrt(2.0) * std::pow(1 + r * r, -0.25) / std::sqrt(rs);
        const double c2 = 2 * U(g) - 1, p2 = 2 * M_PI * U(g), s2 = std::sqrt(1 - c2 * c2);
        v[3 * i] = ve * s2 * std::cos(p2); v[3 * i + 1] = ve * s2 * std::sin(p2); v[3 * i + 2] = ve * c2;
    }
}
}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);   // KETJU needs MPI (or the serial stand-in)
    Kokkos::initialize(argc, argv);
    {
        const int nmax = argc > 1 ? std::atoi(argv[1]) : 4096;
        const int steps = argc > 2 ? std::atoi(argv[2]) : 20;
        const int cmax = argc > 3 ? std::atoi(argv[3]) : 512;
        const int teamn = argc > 4 ? std::atoi(argv[4]) : 96;
        double lf, rr, la;
        rumi_mstar::measure_launch_latency(2000, lf, rr, la);
        std::printf("# exec space %s\n", rumi_mstar::ExecSpace::name());
        std::printf("# latency [us]: launch+fence %.2f, reduce+readback %.2f, async launch %.2f\n", lf, rr, la);
        std::printf("# %6s %6s %10s %10s %10s %8s %8s %8s %10s %8s\n", "N", "steps", "dev_ms/st", "C_ms/st",
                    "speedup", "launch/st", "mst_%", "sync_%", "dev_Gint/s", "k_end");
        for (int n = 16; n <= nmax; n *= 2) {
            std::vector<double> m, x, v;
            plummer(n, 1234 + n, m, x, v);
            rumi_mstar::MstarKokkos M(0, n);
            M.opt.gbs_relative_tolerance = 1e-10;
            M.opt.max_step_count = steps;
            M.opt.team_rows_max_n = teamn;
            M.mass = m; M.pos = x; M.vel = v;
            // warm-up (allocations) with 2 steps
            M.opt.max_step_count = 2;
            M.run(1.0);
            M.reset(0, n);
            M.mass = m; M.pos = x; M.vel = v;
            M.opt.max_step_count = steps;
            Kokkos::fence();
            const double t0 = now();
            M.run(1.0);
            Kokkos::fence();
            const double td = now() - t0;
            const long st = M.perf.successful_steps + M.perf.failed_steps;
            double tc = -1;
            if (n <= cmax) {
                struct ketju_system R;
                ketju_create_system(&R, 0, n, MPI_COMM_WORLD);
                R.options->PN_flags = 0;
                R.constants->G = 1;
                R.options->gbs_relative_tolerance = 1e-10;
                for (int i = 0; i < n; i++) {
                    R.physical_state->mass[i] = m[i];
                    for (int k = 0; k < 3; k++) { R.physical_state->pos[i][k] = x[3 * i + k]; R.physical_state->vel[i][k] = v[3 * i + k]; }
                }
                // the same number of attempted steps: integrate to the time the port reached
                const double t1 = now();
                ketju_run_integrator(&R, M.time > 0 ? M.time : 1e-3);
                const long cst = R.perf->successful_steps + R.perf->failed_steps;
                tc = (now() - t1) / std::max(cst, 1L);
                ketju_free_system(&R);
            }
            const double work = M.perf.total_work;   // pair interactions x substeps
            std::printf("  %6d %6ld %10.3f %10.3f %10.2f %8.1f %8.1f %8.1f %10.3f %8s\n", n, st, 1e3 * td / st,
                        tc > 0 ? 1e3 * tc : -1.0, tc > 0 ? tc / (td / st) : -1.0, (double)M.perf.launches / st,
                        100 * M.perf.t_mst / td, 100 * M.perf.t_sync / td, work / td * 1e-9, "-");
            std::fflush(stdout);
        }
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
