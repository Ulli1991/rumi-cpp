// RUMI++ library interface (rumi_lib.h): the standalone program's set-up and HHS step (main.cpp, INTEGRATOR = hhs,
// ETA_RESTART = 0) behind host-array calls, so a host code can integrate its bodies with RUMI++.
#include "rumi_lib.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
#include <Kokkos_Core.hpp>

#include "run_setup.h"
#include "params.h"
#include "particles.h"
#include "forces.h"
#include "integrator.h"
#include "timestep.h"
#include "diagnostics.h"
#include "constants.h"
#include "profiling.h"
#include "fence_stats.h"
#include "tidal_field.h"
#include "collisional_config.h"
#include "subsystem.h"
#include "logh_batch.h"
#include "gw_output.h"
#include "stellar_cache.h"
#include "merger_tree.h"

#ifndef ENABLE_COLLISIONAL
#error "the RUMI++ library needs ENABLE_COLLISIONAL"
#endif
#ifdef USE_MPI
#error "the RUMI++ library runs on one host process: build it with USE_MPI=OFF"
#endif

namespace rumi_lib {
namespace {
using clk = std::chrono::steady_clock;
double secs(clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); }

bool own_kokkos = false;     // this library initialised Kokkos (and finalises it)
bool configured = false;
Particles P;                 // the bodies, Kokkos default memory space
bool have_p = false;
int nslots = 0;
std::vector<long long> slot_id;
std::vector<double> last_m, last_x, last_v;   // state by slot as last returned (merged bodies keep theirs)
std::vector<int> slot_kw;
std::vector<double> slot_r;
double eta = 0.2, dt_min = 0.0, soft2 = 0.0, t_lib = 0.0;
RunSettings rs;
Timers T;
int merged = 0;

// slot of each present body (index i -> particle_id)
std::vector<int> present_slots(int& n) {
    n = P.N;
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(P.particle_id, Kokkos::make_pair(0, n)));
    std::vector<int> s(n);
    for (int i = 0; i < n; i++) s[i] = h(i);
    return s;
}
}  // namespace

int kokkos_initialize(int num_threads, int device_id) {
    if (Kokkos::is_initialized()) { own_kokkos = false; return 0; }
    Kokkos::InitializationSettings s;
    if (num_threads > 0) s.set_num_threads(num_threads);
    if (device_id >= 0) s.set_device_id(device_id);
    Kokkos::initialize(s);
    own_kokkos = true;
    fence_stats::init();
    return 0;
}

void kokkos_finalize() {
    if (have_p) { P.deallocate(); have_p = false; }
    g_logh_batch = SubsystemBatch();          // global Views go before Kokkos::finalize (as main.cpp)
    g_subsystem_mgr = SubsystemManager();
    release_integrator_globals();
    if (own_kokkos && Kokkos::is_initialized() && !Kokkos::is_finalized()) Kokkos::finalize();
    own_kokkos = false;
}

const char* backend_name() { return Kokkos::DefaultExecutionSpace::name(); }

int configure(const char* config_file, int verbose) {
    Config cfg;
    cfg.load(config_file, verbose != 0);
    g_profiler.enabled = false;
    g_profiler.detailed = false;
    g_profiler.fence = false;
    rs = RunSettings();
    rs.soft_pc = cfg.get_double("SOFTENING_PC");
    if (apply_run_settings(cfg, rs) != 0) return 1;
    auto fail = [](const char* m) { std::fprintf(stderr, "RUMI++ library: %s\n", m); return 1; };
    if (!g_collisional.enabled) return fail("needs COLLISIONAL = true");
    if (use_tree) return fail("USE_TREE is not supported");
    if (cfg.get_str("INTEGRATOR") != "hhs") return fail("needs INTEGRATOR = hhs");
    if (g_collisional.eta_restart > 0.0) return fail("ETA_RESTART > 0 (step restarts) is not supported: use PRESET = cluster or ETA_RESTART = 0");
    if (g_collisional.sse_enabled) return fail("SSE_ENABLED: the host evolves the stars");
    if (cfg.get_str("TIDAL_FIELD") != "none") return fail("TIDAL_FIELD: the host supplies the external field");
    if (use_integer_positions) return fail("INTEGER_POSITIONS is not supported");
    eta = cfg.get_double("ETA");
    dt_min = cfg.get_double("DT_MIN_MYR") / TIME_UNIT_MYR;
    soft2 = rs.soft_pc * rs.soft_pc;
    g_gw_log.on = false;                      // the host writes the outputs
    g_tidal.type = TidalField::NONE;
    configured = true;
    T = Timers();
    if (verbose)
        std::printf("RUMI++ library: %s, %s, ETA_FF %.3g ETA_GBS %.3g FLYBY_CM_EPS %.3g\n", config_file, backend_name(),
                    g_collisional.eta_ff, g_collisional.eta_gbs, g_collisional.flyby_cm_eps);
    return 0;
}

int set_particles(int n, const long long* id, const double* m, const double* x, const double* v, const int* kw,
                  const double* radius_rsun, double G, double dt) {
    if (!configured) return 1;
    auto t0 = clk::now();
    if (have_p) { P.deallocate(); have_p = false; }
    nslots = n; merged = 0;
    slot_id.assign(id, id + n);
    last_m.assign(m, m + n); last_x.assign(x, x + 3 * n); last_v.assign(v, v + 3 * n);
    slot_kw.assign(n, 1); slot_r.assign(n, 0.0);
    for (int i = 0; i < n; i++) { if (kw) slot_kw[i] = kw[i]; if (radius_rsun) slot_r[i] = radius_rsun[i]; }
    if (n == 0) { T.rebuild += secs(t0); return 0; }
    P = Particles();
    P.allocate(n, true);
    have_p = true;
    {
        auto hx = Kokkos::create_mirror_view(P.x), hy = Kokkos::create_mirror_view(P.y), hz = Kokkos::create_mirror_view(P.z);
        auto hvx = Kokkos::create_mirror_view(P.vx), hvy = Kokkos::create_mirror_view(P.vy), hvz = Kokkos::create_mirror_view(P.vz);
        auto hm = Kokkos::create_mirror_view(P.mass);
        auto hst = Kokkos::create_mirror_view(P.stellar_type);
        auto hr = Kokkos::create_mirror_view(P.stellar_radius);
        for (int i = 0; i < n; i++) {
            hm(i) = m[i]; hx(i) = x[3*i]; hy(i) = x[3*i+1]; hz(i) = x[3*i+2];
            hvx(i) = v[3*i]; hvy(i) = v[3*i+1]; hvz(i) = v[3*i+2];
            hst(i) = kw ? kw[i] : 1; hr(i) = radius_rsun ? radius_rsun[i] : 0.0;
        }
        Kokkos::deep_copy(P.x, hx); Kokkos::deep_copy(P.y, hy); Kokkos::deep_copy(P.z, hz);
        Kokkos::deep_copy(P.vx, hvx); Kokkos::deep_copy(P.vy, hvy); Kokkos::deep_copy(P.vz, hvz);
        Kokkos::deep_copy(P.mass, hm); Kokkos::deep_copy(P.stellar_type, hst); Kokkos::deep_copy(P.stellar_radius, hr);
    }
    StellarPropsCache::get().invalidate();
    // fresh subsystem state for the new set (main.cpp: initial forces, subsystems and levels)
    g_logh_batch = SubsystemBatch();
    g_subsystem_mgr = SubsystemManager();
    g_collisional.G = G;
    g_subsystem_mgr.allocate_ngb_buffers(std::max(1000, n / 10));
    g_subsystem_mgr.reset_ngb_count();
    // |E0| of the FLYBY_CM_EPS tolerance: the first set's energy (as main.cpp), raised when a later set is more bound
    if (g_collisional.flyby_cm_eps > 0.0 || g_collisional.subsys_err_tol > 0.0) {
        const double e = rs.e_ref_set > 0.0 ? rs.e_ref_set : std::fabs(compute_diagnostics(P, G, soft2).E_total);
        if (e > g_collisional.e_ref) g_collisional.e_ref = e;
    }
    if (rs.ngb_mass_scaling && g_collisional.ngb_inv_mref == 0.0) {   // <m> of the first set, fixed afterwards
        double s = 0; for (int i = 0; i < n; i++) s += m[i];
        g_collisional.ngb_inv_mref = rs.ngb_mref > 0.0 ? 1.0 / rs.ngb_mref : n / s;
    }
    compute_acceleration_and_timestep(P, G, soft2, eta);
    g_subsystem_mgr.reset_ngb_count();
    neighbor_search_subset(P, 0, P.N, G, g_collisional.r_ngb, g_collisional.r_ngb_max, dt);
    g_subsystem_mgr.detect_from_pairs(P, G, dt, g_collisional.r_ngb, g_collisional.r_ngb_max, g_collisional.n_orb_secular);
    g_subsystem_mgr.update_particle_flags(P);
    g_subsystem_mgr.reset_step_diagnostics();
    g_subsystem_mgr.accumulate_diagnostics();
    auto levels = build_levels(dt, dt_min);
    assign_levels(P, levels);
    P.partition_by_level();
    Kokkos::fence();
    T.rebuild += secs(t0); T.n_rebuild++;
    return 0;
}

int n_slots() { return nslots; }

int get_state(double* m, double* x, double* v) {
    auto t0 = clk::now();
    int n = 0;
    if (have_p) {
        std::vector<int> s = present_slots(n);
        auto hm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.mass);
        auto hx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.x);
        auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.y);
        auto hz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.z);
        auto hvx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.vx);
        auto hvy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.vy);
        auto hvz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.vz);
        std::vector<char> seen(nslots, 0);
        for (int i = 0; i < n; i++) {
            const int k = s[i];
            if (k < 0 || k >= nslots) continue;
            seen[k] = 1;
            last_m[k] = hm(i);
            last_x[3*k] = hx(i); last_x[3*k+1] = hy(i); last_x[3*k+2] = hz(i);
            last_v[3*k] = hvx(i); last_v[3*k+1] = hvy(i); last_v[3*k+2] = hvz(i);
        }
        int nm = 0;
        for (int k = 0; k < nslots; k++) if (!seen[k]) { last_m[k] = 0.0; nm++; }
        merged = nm;
    }
    for (int k = 0; k < nslots; k++) {
        m[k] = last_m[k];
        for (int j = 0; j < 3; j++) { x[3*k+j] = last_x[3*k+j]; v[3*k+j] = last_v[3*k+j]; }
    }
    T.d2h += secs(t0);
    return n;
}

namespace {
void add_by_slot(const double* d, Kokkos::View<double*> a, Kokkos::View<double*> b, Kokkos::View<double*> c) {
    int n = 0;
    std::vector<int> s = present_slots(n);
    auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a);
    auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
    auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c);
    bool any = false;
    for (int i = 0; i < n; i++) {
        const int k = s[i];
        if (k < 0 || k >= nslots) continue;
        if (d[3*k] != 0.0 || d[3*k+1] != 0.0 || d[3*k+2] != 0.0) any = true;
        ha(i) += d[3*k]; hb(i) += d[3*k+1]; hc(i) += d[3*k+2];
    }
    if (!any) return;   // exact no-op: the state keeps its bits
    Kokkos::deep_copy(a, ha); Kokkos::deep_copy(b, hb); Kokkos::deep_copy(c, hc);
}
}  // namespace

void kick(const double* dv) {
    if (!have_p || !dv) return;
    auto t0 = clk::now();
    add_by_slot(dv, P.vx, P.vy, P.vz);
    T.h2d += secs(t0);
}

void shift(const double* dx) {
    if (!have_p || !dx) return;
    auto t0 = clk::now();
    add_by_slot(dx, P.x, P.y, P.z);
    T.h2d += secs(t0);
}

int advance(double dt, double G) {
    if (!configured) return 1;
    if (!have_p || P.N == 0) { t_lib += dt; return 0; }
    auto t0 = clk::now();
    g_collisional.G = G;
    g_sim_time = t_lib;
    g_step_in_progress++;
    hhs_fsi_step_dynamic(P, 0, P.N, dt, G, soft2, dt_min, eta);
    Kokkos::fence();
    t_lib += dt;
    T.advance += secs(t0); T.n_advance++;
    return 0;
}

void energy(double G, double* E_bodies, double* E_books) {
    *E_bodies = have_p && P.N > 0 ? compute_diagnostics(P, G, soft2).E_total : 0.0;
    *E_books = g_energy_books.total_sink();
}

int dump_ic(const char* path) {
    std::FILE* f = std::fopen(path, "w");
    if (!f) return 1;
    std::vector<double> m(nslots), x(3 * nslots), v(3 * nslots);
    get_state(m.data(), x.data(), v.data());
    std::fprintf(f, "# RUMI++ library state: m x y z vx vy vz kw R (pc, Msun, km/s), slot order\n");
    for (int k = 0; k < nslots; k++)
        if (m[k] > 0.0)
            std::fprintf(f, "%.17g %.17g %.17g %.17g %.17g %.17g %.17g %d %.17g\n", m[k], x[3*k], x[3*k+1], x[3*k+2],
                         v[3*k], v[3*k+1], v[3*k+2], slot_kw[k], slot_r[k]);
    std::fclose(f);
    return 0;
}

const Timers& timers() { return T; }
int n_mergers() { return merged; }

}  // namespace rumi_lib
