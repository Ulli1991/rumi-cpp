// RUMI++ library interface (rumi_lib.h): the standalone program's set-up and HHS step (main.cpp, INTEGRATOR = hhs,
// ETA_RESTART = 0) behind host-array calls, so a host code can integrate its bodies with RUMI++.
#include "rumi_lib.h"
#include "git_version.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
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
#include "mpi_comm.h"
#include "collisional_config.h"
#include "subsystem.h"
#include "logh_batch.h"
#include "gw_output.h"
#include "stellar_cache.h"
#include "merger_tree.h"
#include "runtime_sse.h"
#ifdef USE_HDF5
#include "restart.h"
#endif

#ifndef ENABLE_COLLISIONAL
#error "the RUMI++ library needs ENABLE_COLLISIONAL"
#endif

#define RUMI_LIB_STR2(x) #x
#define RUMI_LIB_STR(x) RUMI_LIB_STR2(x)
// found by `strings librumi.a` (host Makefiles) and returned by build_id()
extern "C" __attribute__((used)) const char rumi_lib_build_id[] =
    "RUMI_LIB_ID api=" RUMI_LIB_STR(RUMI_LIB_API_VERSION) " commit=" NBODY_GIT_VERSION;

namespace rumi_lib {
namespace {
using clk = std::chrono::steady_clock;
double secs(clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); }

bool own_kokkos = false;     // this library initialised Kokkos (and finalises it)
bool configured = false;
bool distributed = false;    // attach_comm with more than one rank
Particles P;                 // the bodies (one process), Kokkos default memory space
DistributedParticles DP;     // the bodies (distributed: DP.local, replicated on every rank)
bool have_p = false;
int nslots = 0;
std::vector<long long> slot_id;
std::vector<double> last_m, last_x, last_v;   // state by slot as last returned (merged bodies keep theirs)
std::vector<int> slot_kw;
std::vector<double> slot_r;
double eta = 0.2, dt_min = 0.0, soft2 = 0.0, t_lib = 0.0;
RunSettings rs;
Config cfg_lib;
Timers T;
Ledger L;
int merged = 0;
size_t merger_seen = 0;      // g_merger_tree events handed to the host

Particles& W() { return distributed ? DP.local : P; }

// Rank-0 host data to every rank of the communicator (no-op on one process)
template <class V> void bcast(V* data, size_t n) {
#ifdef USE_MPI
    if (distributed && n > 0) MPI_Bcast(data, (int)(n * sizeof(V)), MPI_BYTE, 0, MPIComm::comm());
#else
    (void)data; (void)n;
#endif
}
template <class V> std::vector<V> from_root(const V* p, size_t n) {
    std::vector<V> v(n);
    if (MPIComm::rank() == 0 && p) std::memcpy(v.data(), p, n * sizeof(V));
    bcast(v.data(), n);
    return v;
}
int bcast_flag(bool present) {   // whether the root passed an array
    int f = present ? 1 : 0;
    bcast(&f, 1);
    return f;
}

// slot of each present body (index i -> particle_id)
std::vector<int> present_slots(int& n) {
    n = W().N;
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(W().particle_id, Kokkos::make_pair(0, n)));
    std::vector<int> s(n);
    for (int i = 0; i < n; i++) s[i] = h(i);
    return s;
}

// energy scale of the set, centre-of-mass frame: max(|E_int|, |PE| / 2) (= |E| for a virialised cluster; does not
// pass through zero while a forming set goes from unbound to bound). Collective under MPI.
double energy_scale(double G) {
    Particles& p = W();
    const DiagnosticState d = compute_diagnostics(p, G, soft2);
    double M = 0;
    auto pm = p.mass;
    Kokkos::parallel_reduce("lib_mtot", p.N, KOKKOS_LAMBDA(int i, double& s) { s += pm(i); }, M);
    const double P2 = d.Px * d.Px + d.Py * d.Py + d.Pz * d.Pz;
    const double E_int = d.E_total - (M > 0 ? 0.5 * P2 / M : 0.0);
    return std::max(std::fabs(E_int), 0.5 * std::fabs(d.PE));
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
    if (have_p) { if (distributed) DP.local.deallocate(); else P.deallocate(); have_p = false; }
    DP = DistributedParticles();              // its staging Views too: globals are destroyed after Kokkos::finalize
    P = Particles();
    g_logh_batch = SubsystemBatch();          // global Views go before Kokkos::finalize (as main.cpp)
    g_subsystem_mgr = SubsystemManager();
    release_integrator_globals();
    if (own_kokkos && Kokkos::is_initialized() && !Kokkos::is_finalized()) Kokkos::finalize();
    own_kokkos = false;
}

const char* backend_name() { return Kokkos::DefaultExecutionSpace::name(); }

int attach_comm(int fortran_comm) {
#ifdef USE_MPI
    MPIComm::attach(MPI_Comm_f2c(fortran_comm));
    distributed = MPIComm::size() > 1;
    return 0;
#else
    (void)fortran_comm;
    std::fprintf(stderr, "RUMI++ library: attach_comm needs a build with USE_MPI\n");
    return 1;
#endif
}

int comm_rank() { return MPIComm::rank(); }

int configure(const char* config_file, int verbose) {
    Config& cfg = cfg_lib;
    cfg.load(config_file, verbose != 0 && MPIComm::rank() == 0);
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
    L = Ledger();
    if (verbose && MPIComm::rank() == 0)
        std::printf("RUMI++ library: %s, %s, %d rank(s), ETA_FF %.3g ETA_GBS %.3g FLYBY_CM_EPS %.3g\n", config_file,
                    backend_name(), MPIComm::size(), g_collisional.eta_ff, g_collisional.eta_gbs, g_collisional.flyby_cm_eps);
    return 0;
}

int set_particles(int n_in, const long long* id_in, const double* m_in, const double* x_in, const double* v_in,
                  const int* kw_in, const double* r_in, double G, double dt) {
    if (!configured) return 1;
    auto t0 = clk::now();
    int n = n_in; bcast(&n, 1);
    const std::vector<long long> id = from_root(id_in, n);
    const std::vector<double> m = from_root(m_in, n), x = from_root(x_in, 3 * (size_t)n), v = from_root(v_in, 3 * (size_t)n);
    const int has_kw = bcast_flag(kw_in != nullptr), has_r = bcast_flag(r_in != nullptr);
    const std::vector<int> kw = has_kw ? from_root(kw_in, n) : std::vector<int>(n, 1);
    const std::vector<double> rr = has_r ? from_root(r_in, n) : std::vector<double>(n, 0.0);
    double E_old = 0;
    if (have_p && W().N > 0) E_old = compute_diagnostics(W(), G, soft2).E_total;
    if (have_p) { if (distributed) DP.local.deallocate(); else P.deallocate(); have_p = false; }
    nslots = n; merged = 0;
    slot_id = id; last_m = m; last_x = x; last_v = v; slot_kw = kw; slot_r = rr;
    if (n == 0) { L.E_set += -E_old; T.rebuild += secs(t0); return 0; }
    // every rank builds the same set (bit-identical inputs); distributed mode distributes rank 0's copy as main.cpp
    Particles fresh;
    fresh.allocate(n, true);
    {
        auto hx = Kokkos::create_mirror_view(fresh.x), hy = Kokkos::create_mirror_view(fresh.y), hz = Kokkos::create_mirror_view(fresh.z);
        auto hvx = Kokkos::create_mirror_view(fresh.vx), hvy = Kokkos::create_mirror_view(fresh.vy), hvz = Kokkos::create_mirror_view(fresh.vz);
        auto hm = Kokkos::create_mirror_view(fresh.mass);
        auto hst = Kokkos::create_mirror_view(fresh.stellar_type);
        auto hr = Kokkos::create_mirror_view(fresh.stellar_radius);
        for (int i = 0; i < n; i++) {
            hm(i) = m[i]; hx(i) = x[3*i]; hy(i) = x[3*i+1]; hz(i) = x[3*i+2];
            hvx(i) = v[3*i]; hvy(i) = v[3*i+1]; hvz(i) = v[3*i+2];
            hst(i) = kw[i]; hr(i) = rr[i];
        }
        Kokkos::deep_copy(fresh.x, hx); Kokkos::deep_copy(fresh.y, hy); Kokkos::deep_copy(fresh.z, hz);
        Kokkos::deep_copy(fresh.vx, hvx); Kokkos::deep_copy(fresh.vy, hvy); Kokkos::deep_copy(fresh.vz, hvz);
        Kokkos::deep_copy(fresh.mass, hm); Kokkos::deep_copy(fresh.stellar_type, hst); Kokkos::deep_copy(fresh.stellar_radius, hr);
    }
    if (distributed) { DP = DistributedParticles(); DP.distribute_initial(fresh); fresh.deallocate(); }
    else P = fresh;
    have_p = true;
    Particles& p = W();
    StellarPropsCache::get().invalidate();
    // fresh subsystem state for the new set (main.cpp: initial forces, subsystems and levels)
    g_logh_batch = SubsystemBatch();
    g_subsystem_mgr = SubsystemManager();
    g_collisional.G = G;
    g_subsystem_mgr.allocate_ngb_buffers(std::max(1000, n / 10));
    g_subsystem_mgr.reset_ngb_count();
    // |E_ref| of FLYBY_CM_EPS: E_REF, else the set's energy scale in its centre-of-mass frame (frame-independent; it
    // follows the integrated set, so the budget stays the same fraction of the system's own binding energy)
    if (g_collisional.flyby_cm_eps > 0.0 || g_collisional.subsys_err_tol > 0.0)
        g_collisional.e_ref = rs.e_ref_set > 0.0 ? rs.e_ref_set : energy_scale(G);
    if (rs.ngb_mass_scaling && g_collisional.ngb_inv_mref == 0.0) {   // <m> of the first set, fixed afterwards
        double s = 0; for (int i = 0; i < n; i++) s += m[i];
        g_collisional.ngb_inv_mref = rs.ngb_mref > 0.0 ? 1.0 / rs.ngb_mref : n / s;
    }
    if (distributed) compute_acceleration_and_timestep_distributed(DP, G, soft2, eta);
    else compute_acceleration_and_timestep(p, G, soft2, eta);
    g_subsystem_mgr.reset_ngb_count();
    neighbor_search_subset(p, 0, p.N, G, g_collisional.r_ngb, g_collisional.r_ngb_max, dt);
    g_subsystem_mgr.detect_from_pairs(p, G, dt, g_collisional.r_ngb, g_collisional.r_ngb_max, g_collisional.n_orb_secular);
    g_subsystem_mgr.update_particle_flags(p);
    g_subsystem_mgr.reset_step_diagnostics();
    g_subsystem_mgr.accumulate_diagnostics();
    auto levels = build_levels(dt, dt_min);
    assign_levels(p, levels);
    p.partition_by_level();
    L.E_set += compute_diagnostics(p, G, soft2).E_total - E_old;
    Kokkos::fence();
    T.rebuild += secs(t0); T.n_rebuild++;
    return 0;
}

int n_slots() { return nslots; }

int get_state(double* m, double* x, double* v) {
    auto t0 = clk::now();
    int n = 0;
    if (have_p) {
        Particles& p = W();
        std::vector<int> s = present_slots(n);
        auto hm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
        auto hx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
        auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
        auto hz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
        auto hvx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
        auto hvy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
        auto hvz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
        std::vector<char> seen(nslots, 0);
        for (int i = 0; i < n; i++) {
            const int k = s[i];
            if (k < 0 || k >= nslots || hm(i) <= 0.0) continue;
            seen[k] = 1;
            last_m[k] = hm(i);
            last_x[3*k] = hx(i); last_x[3*k+1] = hy(i); last_x[3*k+2] = hz(i);
            last_v[3*k] = hvx(i); last_v[3*k+1] = hvy(i); last_v[3*k+2] = hvz(i);
        }
        int nm = 0;
        for (int k = 0; k < nslots; k++) if (!seen[k]) { last_m[k] = 0.0; nm++; }
        merged = nm;
    }
    if (m && x && v)
        for (int k = 0; k < nslots; k++) {
            m[k] = last_m[k];
            for (int j = 0; j < 3; j++) { x[3*k+j] = last_x[3*k+j]; v[3*k+j] = last_v[3*k+j]; }
        }
    T.d2h += secs(t0);
    return n;
}

namespace {
// adds d (by slot) to (a, b, c); with mass m != null returns the kinetic-energy change 1/2 m (|u + d|^2 - |u|^2)
double add_by_slot(const std::vector<double>& d, Kokkos::View<double*> a, Kokkos::View<double*> b, Kokkos::View<double*> c,
                   bool work) {
    bool any = false;
    for (double e : d) if (e != 0.0) { any = true; break; }
    if (!any) return 0.0;   // exact no-op: the state keeps its bits
    int n = 0;
    std::vector<int> s = present_slots(n);
    auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a);
    auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
    auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c);
    auto hm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), W().mass);
    double dK = 0.0;
    for (int i = 0; i < n; i++) {
        const int k = s[i];
        if (k < 0 || k >= nslots || hm(i) <= 0.0) continue;
        const double u0 = ha(i), u1 = hb(i), u2 = hc(i);
        ha(i) += d[3*k]; hb(i) += d[3*k+1]; hc(i) += d[3*k+2];
        if (work) dK += 0.5 * hm(i) * (ha(i)*ha(i) + hb(i)*hb(i) + hc(i)*hc(i) - u0*u0 - u1*u1 - u2*u2);
    }
    Kokkos::deep_copy(a, ha); Kokkos::deep_copy(b, hb); Kokkos::deep_copy(c, hc);
    W().zero_kick_compensation();
    return dK;
}
}  // namespace

void kick(const double* dv) {
    if (!have_p) return;
    auto t0 = clk::now();
    if (!bcast_flag(dv != nullptr)) return;
    const std::vector<double> d = from_root(dv, 3 * (size_t)nslots);
    const double dK = add_by_slot(d, W().vx, W().vy, W().vz, true);
    L.W_kick += dK; L.W_kick_abs += std::fabs(dK);
    T.h2d += secs(t0);
}

void shift(const double* dx) {
    if (!have_p) return;
    auto t0 = clk::now();
    if (!bcast_flag(dx != nullptr)) return;
    const std::vector<double> d = from_root(dx, 3 * (size_t)nslots);
    add_by_slot(d, W().x, W().y, W().z, false);
    T.h2d += secs(t0);
}

void set_mass(const double* m_in, const double* r_in, double G) {
    if (!have_p) return;
    auto t0 = clk::now();
    const std::vector<double> mnew = from_root(m_in, nslots);
    const int has_r = bcast_flag(r_in != nullptr);
    const std::vector<double> rnew = has_r ? from_root(r_in, nslots) : std::vector<double>();
    Particles& p = W();
    int n = 0;
    std::vector<int> s = present_slots(n);
    auto hm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
    auto hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_radius);
    auto hvx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
    auto hvy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
    auto hvz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
    auto hx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
    auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
    auto hz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
    std::vector<int> touched; std::vector<double> dmass, mbar;
    double KE_change = 0.0;
    bool radius_changed = false;
    for (int i = 0; i < n; i++) {
        const int k = s[i];
        if (k < 0 || k >= nslots || hm(i) <= 0.0) continue;
        if (has_r && rnew[k] != hr(i)) { hr(i) = rnew[k]; slot_r[k] = rnew[k]; radius_changed = true; }
        if (mnew[k] <= 0.0 || mnew[k] == hm(i)) continue;
        const double v2 = hvx(i)*hvx(i) + hvy(i)*hvy(i) + hvz(i)*hvz(i);
        KE_change += 0.5 * (mnew[k] - hm(i)) * v2;
        const double r[3] = {hx(i), hy(i), hz(i)}, vv[3] = {hvx(i), hvy(i), hvz(i)};
        g_energy_books.sink_PL(hm(i), r, vv);
        g_energy_books.sink_PL(mnew[k], r, vv, -1.0);
        touched.push_back(i); dmass.push_back(mnew[k] - hm(i)); mbar.push_back(0.5 * (mnew[k] + hm(i)));
    }
    if (!touched.empty()) {
        for (size_t q = 0; q < touched.size(); q++) hm(touched[q]) += dmass[q];
        Kokkos::deep_copy(p.mass, hm);
        const double dU = potential_change(p, touched, dmass, mbar, G);   // with the new masses on the device
        g_energy_books.E_sse += -KE_change - dU;
        T.n_set_mass++;
    }
    if (radius_changed) Kokkos::deep_copy(p.stellar_radius, hr);
    if (!touched.empty() || radius_changed) StellarPropsCache::get().invalidate();
    T.h2d += secs(t0);
}

int advance(double dt, double G) {
    if (!configured) return 1;
    if (!have_p || W().N == 0) { t_lib += dt; return 0; }
    auto t0 = clk::now();
    g_collisional.G = G;
    g_sim_time = t_lib;
    g_step_in_progress++;
    if (distributed) hhs_fsi_step_distributed(DP, dt, G, soft2, dt_min, eta, 2);
    else hhs_fsi_step_dynamic(P, 0, P.N, dt, G, soft2, dt_min, eta);
    Kokkos::fence();
    t_lib += dt;
    T.advance += secs(t0); T.n_advance++;
    return 0;
}

void energy(double G, double* E_bodies, double* E_books) {   // collective under MPI; null outputs allowed
    const double E = have_p && W().N > 0 ? compute_diagnostics(W(), G, soft2).E_total : 0.0;
    if (E_bodies) *E_bodies = E;
    if (E_books) *E_books = g_energy_books.total_sink();
}

void books(double out[7]) {
    const EnergyBookkeeping& b = g_energy_books;
    out[0] = b.E_escape; out[1] = b.E_gw; out[2] = b.E_merge; out[3] = b.E_sse; out[4] = b.E_pn; out[5] = b.E_sd; out[6] = b.E_tidal;
}

const Ledger& ledger() { return L; }

int merger_events(std::vector<int>& kept, std::vector<int>& gone, std::vector<double>& mprod) {
    kept.clear(); gone.clear(); mprod.clear();
    const auto& ev = g_merger_tree.events;
    if (merger_seen > ev.size()) merger_seen = ev.size();
    for (size_t k = merger_seen; k < ev.size(); k++) {
        kept.push_back(ev[k].pid_remnant);
        gone.push_back(ev[k].pid_remnant == ev[k].pid1 ? ev[k].pid2 : ev[k].pid1);
        mprod.push_back(ev[k].m_remnant);
    }
    merger_seen = ev.size();
    return (int)kept.size();
}
double e_ref() { return g_collisional.e_ref; }

int dump_ic(const char* path) {
    std::vector<double> m(nslots), x(3 * nslots), v(3 * nslots);
    get_state(m.data(), x.data(), v.data());
    if (MPIComm::rank() != 0) return 0;
    std::FILE* f = std::fopen(path, "w");
    if (!f) return 1;
    std::fprintf(f, "# RUMI++ library state: m x y z vx vy vz kw R (pc, Msun, km/s), slot order\n");
    for (int k = 0; k < nslots; k++)
        if (m[k] > 0.0)
            std::fprintf(f, "%.17g %.17g %.17g %.17g %.17g %.17g %.17g %d %.17g\n", m[k], x[3*k], x[3*k+1], x[3*k+2],
                         v[3*k], v[3*k+1], v[3*k+2], slot_kw[k], slot_r[k]);
    std::fclose(f);
    return 0;
}

#ifdef USE_HDF5
namespace {
template <class V> void put_vec(RunState& r, const std::string& key, const std::vector<V>& v) {
    auto& b = r.blobs[key];
    b.resize(v.size() * sizeof(V));
    if (!v.empty()) std::memcpy(b.data(), v.data(), b.size());
}
template <class V> void get_vec(const RunState& r, const std::string& key, std::vector<V>& v) {
    auto it = r.blobs.find(key);
    v.clear();
    if (it == r.blobs.end()) return;
    v.resize(it->second.size() / sizeof(V));
    if (!v.empty()) std::memcpy(v.data(), it->second.data(), v.size() * sizeof(V));
}
}  // namespace
#endif

int save_state(const char* dir, const std::vector<char>& host_blob) {
#ifdef USE_HDF5
    if (!have_p) return 1;
    RunState r;
    r.t = t_lib; r.step = (int)T.n_advance; r.integrator = "hhs"; r.test = "library";
    put_vec(r, "lib_slot_id", slot_id); put_vec(r, "lib_last_m", last_m); put_vec(r, "lib_last_x", last_x);
    put_vec(r, "lib_last_v", last_v); put_vec(r, "lib_slot_kw", slot_kw); put_vec(r, "lib_slot_r", slot_r);
    put_vec(r, "lib_host", host_blob);
    r.put("lib_ledger", L); r.put("lib_nslots", nslots); r.put("lib_merged", merged);
    r.put("lib_step_in_progress", g_step_in_progress); r.put("lib_merger_seen", merger_seen);
    if (MPIComm::rank() == 0) {
        std::filesystem::create_directories(dir);
        RestartOptions opt; opt.dir = dir; opt.keep = 1;
        write_restart(opt, W(), r, cfg_lib.text, cfg_lib.params);
    }
    MPIComm::barrier();
    return 0;
#else
    (void)dir; (void)host_blob;
    std::fprintf(stderr, "RUMI++ library: save_state needs a build with HDF5\n");
    return 1;
#endif
}

int load_state(const char* dir, std::vector<char>& host_blob) {
#ifdef USE_HDF5
    const std::string path = latest_restart(dir);
    if (path.empty()) return 1;
    RunState r;
    if (have_p) { if (distributed) DP.local.deallocate(); else P.deallocate(); have_p = false; }
    g_logh_batch = SubsystemBatch();
    g_subsystem_mgr = SubsystemManager();
    Particles fresh;
    read_restart(path, fresh, r, true);   // every rank reads (as main.cpp)
    if (distributed) {
        DP = DistributedParticles(); DP.distribute_initial(fresh); fresh.deallocate();
        read_restart(path, DP.local, r, false);
    } else P = fresh;
    have_p = true;
    g_subsystem_mgr.rebuild_pid_to_idx(W());
    StellarPropsCache::get().invalidate();
    t_lib = r.t; T.n_advance = r.step;
    get_vec(r, "lib_slot_id", slot_id); get_vec(r, "lib_last_m", last_m); get_vec(r, "lib_last_x", last_x);
    get_vec(r, "lib_last_v", last_v); get_vec(r, "lib_slot_kw", slot_kw); get_vec(r, "lib_slot_r", slot_r);
    std::vector<char> hb; get_vec(r, "lib_host", hb); host_blob = hb;
    r.get("lib_ledger", L); r.get("lib_nslots", nslots); r.get("lib_merged", merged);
    r.get("lib_step_in_progress", g_step_in_progress); r.get("lib_merger_seen", merger_seen);
    return 0;
#else
    (void)dir; (void)host_blob;
    std::fprintf(stderr, "RUMI++ library: load_state needs a build with HDF5\n");
    return 1;
#endif
}

const char* build_id() { return rumi_lib_build_id; }
const Timers& timers() { return T; }
int n_mergers() { return merged; }

}  // namespace rumi_lib
