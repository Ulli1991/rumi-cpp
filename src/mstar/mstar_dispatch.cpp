// RUMI++ MSTAR module (GPLv3, see src/mstar/LICENSE)
// Copyright (C) 2026 Ulrich Steinwandel.
// Licensed under the GPLv3 license (or, at your option, any later version).
//
// Implementation of the core's dispatch interface (src/collisional/mstar_dispatch.h, BSD) on the C++/Kokkos MSTAR
// (mstar_kokkos.cpp): on the device for large groups, on the host execution space below the crossover size.

#include "mstar_dispatch.h"

#include <chrono>
#include <cstdlib>
#include <mutex>
#include <type_traits>

#include "mstar_kokkos.h"

namespace {
double wall_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

bool port_path(MstarGroup& g, bool on_host) {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    static rumi_mstar::MstarKokkos* M = nullptr;   // kept between calls (buffers reused when sizes match)
    if (!M) M = new rumi_mstar::MstarKokkos(g.n_pn, g.n - g.n_pn);
    else M->reset(g.n_pn, g.n - g.n_pn);
    M->opt = rumi_mstar::Options();
    M->opt.host_engine = on_host;
    M->opt.G = g.G;
    M->opt.c = g.c > 0 ? g.c : 1.0;
    M->opt.PN_flags = g.c > 0 ? g.pn_flags : 0;
    M->opt.gbs_relative_tolerance = g.tol;
    M->opt.gbs_kmax = g.kmax;
    M->opt.enable_bh_mergers = g.bh_mergers && g.c > 0;
    M->opt.bh_merger_distance_in_Rs = g.bh_merge_rs;
    M->opt.enable_bh_merger_kicks = g.bh_kicks;
    M->opt.max_step_count = g.max_steps;
    if (const char* e = std::getenv("NBODY_MSTAR_TEAM_N")) M->opt.team_rows_max_n = M->opt.team_rows_max_n_device = std::atoi(e);
    for (int i = 0; i < g.n; i++) {
        M->mass[i] = g.mass[i];
        for (int k = 0; k < 3; k++) { M->pos[3 * i + k] = g.pos[3 * i + k]; M->vel[3 * i + k] = g.vel[3 * i + k]; }
    }
    for (int i = 0; i < 3 * g.n_pn; i++) M->spin[i] = g.spin ? g.spin[i] : 0.0;
    if (g.index)
        for (int i = 0; i < g.n; i++) M->extra_index[i] = g.index[i];
    M->coll_R.clear();
    M->coll_bh.clear();
    if (g.coll_R) {
        M->coll_R.assign(g.coll_R, g.coll_R + g.n);
        M->coll_bh.assign(g.n, 0);
        if (g.coll_bh) for (int i = 0; i < g.n; i++) M->coll_bh[i] = g.coll_bh[i];
        M->coll_c = g.coll_c;
        M->coll_bh_max_single = g.coll_bh_max_single;
    }
    if (g.tide) {
        M->opt.extra.vindep = true;
        M->opt.extra.track_work = true;
        M->opt.extra.remove_mean = true;
        if (g.pert && g.n_pert > 0) M->perturbers.assign(g.pert, g.pert + 10 * g.n_pert);
        for (int q = 0; q < 9; q++) M->opt.extra.A[q] = g.tide_A[q];
    }
    const double t0 = M->time;
    const bool ok = M->run(g.dt);
    g.t_reached = M->time - t0;
    g.coll_i = M->coll_i;
    g.coll_j = M->coll_j;
    g.coll_type = M->coll_type;
    g.coll_rsep = M->coll_rsep;
    g.coll_rcoll = M->coll_rcoll;
    g.status = !ok ? 2 : (M->coll_i >= 0 ? 1 : 0);
    g.pn_dE = -M->pn_dB - M->ext_work;   // B also carries the tide's work
    g.ext_dE = M->ext_work;
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
        // host-only Kokkos builds: the host engine for every group
        if (std::is_same<rumi_mstar::ExecSpace, Kokkos::DefaultHostExecutionSpace>::value) return 1 << 30;
        // A100 vs the best full Raven node (Kokkos-OpenMP 1-72 threads, cpujobs/mbench-omp, 2026-10-26): CPU faster at
        // N = 32 (0.23 vs 0.41 ms per GBS step), A100 faster from N = 64 (0.42 vs 0.66 ms): crossover ~48
        return 48;
    }();
    return n;
}

bool mstar_integrate(MstarGroup& g) {
    const double t0 = wall_s();
    g.mergers.clear();
    const bool dev = g.device == 1 || (g.device < 0 && g.n >= mstar_crossover_n());
    g.used_device = dev ? 1 : 0;
    g.coll_i = g.coll_j = -1;
    g.pn_dE = 0;
    g.ext_dE = 0;
    g.ok = port_path(g, !dev);
    g.wall = wall_s() - t0;
    return g.ok;
}
