// RUMI++ MSTAR module (GPLv3, see src/mstar/LICENSE)
// Copyright (C) 2026 RUMI++ authors.
// Licensed under the GPLv3 license (or, at your option, any later version).
//
// Implementation of the core's dispatch interface (src/collisional/mstar_dispatch.h, BSD) on top of the vendored
// KETJU C library (host path, small groups) and its Kokkos port (mstar_kokkos.cpp, device path).

#include "mstar_dispatch.h"

#include <chrono>
#include <cstdlib>
#include <mutex>
#include <type_traits>

#include "ketju_integrator/ketju_integrator.h"
#include "mstar_kokkos.h"

namespace {
double wall_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

bool host_path(MstarGroup& g) {
    struct ketju_system R;
    ketju_create_system(&R, g.n_pn, g.n - g.n_pn, MPI_COMM_SELF);
    R.constants->G = g.G;
    R.constants->c = g.c > 0 ? g.c : 1.0;
    R.options->PN_flags = g.c > 0 ? g.pn_flags : 0;
    R.options->gbs_relative_tolerance = g.tol;
    R.options->gbs_kmax = g.kmax;
    R.options->enable_bh_mergers = g.bh_mergers && g.c > 0;
    R.options->bh_merger_distance_in_Rs = g.bh_merge_rs;
    R.options->enable_bh_merger_kicks = g.bh_kicks;
    R.options->max_step_count = -1;   // KETJU terminates the program past the limit: checked below instead
    for (int i = 0; i < g.n; i++) {
        R.physical_state->mass[i] = g.mass[i];
        for (int k = 0; k < 3; k++) {
            R.physical_state->pos[i][k] = g.pos[3 * i + k];
            R.physical_state->vel[i][k] = g.vel[3 * i + k];
        }
    }
    for (int i = 0; i < g.n_pn; i++)
        for (int k = 0; k < 3; k++) R.physical_state->spin[i][k] = g.spin ? g.spin[3 * i + k] : 0.0;
    R.particle_extra_data = g.index;
    R.particle_extra_data_elem_size = g.index ? sizeof(int) : 0;
    ketju_run_integrator(&R, g.dt);
    for (int m = 0; m < R.num_mergers; m++) {
        const auto& mi = R.merger_infos[m];
        g.mergers.push_back({mi.t_merger, mi.m1, mi.m2, mi.m_remnant, mi.chi1, mi.chi2, mi.chi_remnant, mi.v_kick,
                             g.index ? g.index[mi.extra_index1] : mi.extra_index1,
                             g.index ? g.index[mi.extra_index2] : mi.extra_index2});
    }
    g.n = R.num_particles;
    g.n_pn = R.num_pn_particles;
    for (int i = 0; i < g.n; i++) {
        g.mass[i] = R.physical_state->mass[i];
        for (int k = 0; k < 3; k++) {
            g.pos[3 * i + k] = R.physical_state->pos[i][k];
            g.vel[3 * i + k] = R.physical_state->vel[i][k];
        }
    }
    for (int i = 0; i < g.n_pn; i++)
        for (int k = 0; k < 3; k++) if (g.spin) g.spin[3 * i + k] = R.physical_state->spin[i][k];
    g.steps = R.perf->successful_steps;
    g.failed_steps = R.perf->failed_steps;
    g.rel_energy_error = R.perf->relative_energy_error;
    ketju_free_system(&R);
    return true;
}

bool device_path(MstarGroup& g) {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    static rumi_mstar::MstarKokkos* M = nullptr;   // kept between calls (device buffers reused when sizes match)
    if (!M) M = new rumi_mstar::MstarKokkos(g.n_pn, g.n - g.n_pn);
    else M->reset(g.n_pn, g.n - g.n_pn);
    M->opt = rumi_mstar::Options();
    M->opt.G = g.G;
    M->opt.c = g.c > 0 ? g.c : 1.0;
    M->opt.PN_flags = g.c > 0 ? g.pn_flags : 0;
    M->opt.gbs_relative_tolerance = g.tol;
    M->opt.gbs_kmax = g.kmax;
    M->opt.enable_bh_mergers = g.bh_mergers && g.c > 0;
    M->opt.bh_merger_distance_in_Rs = g.bh_merge_rs;
    M->opt.enable_bh_merger_kicks = g.bh_kicks;
    M->opt.max_step_count = g.max_steps;
    if (const char* e = std::getenv("NBODY_MSTAR_TEAM_N")) M->opt.team_rows_max_n = std::atoi(e);
    for (int i = 0; i < g.n; i++) {
        M->mass[i] = g.mass[i];
        for (int k = 0; k < 3; k++) { M->pos[3 * i + k] = g.pos[3 * i + k]; M->vel[3 * i + k] = g.vel[3 * i + k]; }
    }
    for (int i = 0; i < 3 * g.n_pn; i++) M->spin[i] = g.spin ? g.spin[i] : 0.0;
    if (g.index)
        for (int i = 0; i < g.n; i++) M->extra_index[i] = g.index[i];
    const bool ok = M->run(g.dt);
    for (const auto& mi : M->mergers)
        g.mergers.push_back({mi.t_merger, mi.m1, mi.m2, mi.m_remnant, mi.chi1, mi.chi2, mi.chi_remnant, mi.v_kick,
                             mi.extra_index1, mi.extra_index2});
    g.n = M->n;
    g.n_pn = M->n_pn;
    for (int i = 0; i < g.n; i++) {
        g.mass[i] = M->mass[i];
        for (int k = 0; k < 3; k++) { g.pos[3 * i + k] = M->pos[3 * i + k]; g.vel[3 * i + k] = M->vel[3 * i + k]; }
    }
    for (int i = 0; i < 3 * g.n_pn; i++) if (g.spin) g.spin[i] = M->spin[i];
    if (g.index)
        for (size_t i = 0; i < M->extra_index.size(); i++) g.index[i] = M->extra_index[i];
    g.steps = M->perf.successful_steps;
    g.failed_steps = M->perf.failed_steps;
    g.rel_energy_error = M->perf.relative_energy_error;
    return ok;
}
}  // namespace

bool mstar_available() { return true; }

int mstar_crossover_n() {
    static const int n = [] {
        if (const char* e = std::getenv("NBODY_MSTAR_CROSSOVER")) return std::atoi(e);
        // host-only Kokkos builds: the serial C library (the port's team kernels gain nothing on the host)
        if (std::is_same<rumi_mstar::ExecSpace, Kokkos::DefaultHostExecutionSpace>::value) return 1 << 30;
        return 64;   // A100 measurement: docs/MSTAR.md (mstar-bench)
    }();
    return n;
}

bool mstar_integrate(MstarGroup& g) {
    const double t0 = wall_s();
    g.mergers.clear();
    const bool dev = g.device == 1 || (g.device < 0 && g.n >= mstar_crossover_n());
    g.used_device = dev ? 1 : 0;
    g.ok = dev ? device_path(g) : host_path(g);
    g.wall = wall_s() - t0;
    return g.ok;
}
