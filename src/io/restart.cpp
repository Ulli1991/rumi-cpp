#include "restart.h"
#include "h5_util.h"
#include "git_version.h"
#include "tidal_field.h"
#ifdef ENABLE_COLLISIONAL
#include "collisional_config.h"
#include "subsystem.h"
#include "merger_tree.h"
#include "stellar_cache.h"
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fcntl.h>
#include <unistd.h>

namespace fs = std::filesystem;

const char* code_version() { return NBODY_GIT_VERSION; }

// ---------------------------------------------------------------------------------------------------------------
// Struct layouts the serialisation below knows field by field. A new field changes the size and stops the build
// here: add it to write/read below (and bump format_version if old files can no longer be read).
// Sizes for x86-64 with libstdc++ (host and nvcc builds).
#ifdef ENABLE_COLLISIONAL
static_assert(sizeof(OrbitalElements) == 10 * sizeof(double), "OrbitalElements changed: update io/restart.cpp");
static_assert(sizeof(Subsystem) == 392, "Subsystem changed: update io/restart.cpp (write_subsystems/read_subsystems)");
static_assert(sizeof(EnergyBookkeeping) == 13 * sizeof(double), "EnergyBookkeeping changed: update io/restart.cpp");
static_assert(sizeof(MergerTreeNode) == 112, "MergerTreeNode changed: update io/restart.cpp");
static_assert(sizeof(EscaperEvent) == 96, "EscaperEvent changed: update io/restart.cpp");
static_assert(std::is_trivially_copyable_v<EnergyAttribution>, "g_eattr is stored as bytes");
#endif
static_assert(sizeof(DiagnosticState) == 11 * sizeof(double), "DiagnosticState changed: update io/restart.cpp");

namespace {

constexpr int FORMAT_VERSION = 1;

// ---- particle Views ---------------------------------------------------------------------------------------------
template <class V>
void put_view(hid_t g, const char* name, const V& v, int n) {
    using T = typename V::non_const_value_type;
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    const size_t ncol = V::rank == 2 ? v.extent(1) : 0;
    h5::write<T>(g, name, h.data(), (size_t)n, ncol);
}

template <class V>
void get_view(hid_t g, const char* name, const V& v, int n) {
    using T = typename V::non_const_value_type;
    std::vector<T> buf;
    const size_t rows = h5::read<T>(g, name, buf);
    const size_t ncol = V::rank == 2 ? v.extent(1) : 1;
    if (rows != (size_t)n || buf.size() != (size_t)n * ncol || v.extent(0) < (size_t)n)
        h5::fail(std::string("restart particle field ") + name + ": size mismatch");
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);   // keeps rows >= n as they are
    std::copy(buf.begin(), buf.end(), h.data());
    Kokkos::deep_copy(v, h);
}

// Every per-particle View that carries state from one step to the next. Scratch views are rewritten before they
// are read. A new per-particle field of Particles must be added here.
template <class F>
void visit_particle_views(Particles& p, F&& f) {
    f("x", p.x); f("y", p.y); f("z", p.z);
    f("ix", p.ix); f("iy", p.iy); f("iz", p.iz);
    f("vx", p.vx); f("vy", p.vy); f("vz", p.vz);
    f("cvx", p.cvx); f("cvy", p.cvy); f("cvz", p.cvz);     // Kahan compensation of the kicks
    f("ax", p.ax); f("ay", p.ay); f("az", p.az);
    f("mass", p.mass); f("dt_adapt", p.dt_adapt); f("level", p.level); f("orig_index", p.orig_index);
    f("pos_packed", p.pos_packed); f("vel_packed", p.vel_packed);
    if (p.collisional) {
        f("particle_id", p.particle_id); f("subsystem_id", p.subsystem_id); f("stellar_type", p.stellar_type);
        f("m_zams", p.m_zams); f("stellar_age", p.stellar_age); f("stellar_radius", p.stellar_radius);
        f("luminosity", p.luminosity); f("core_mass", p.core_mass);
        f("spin_vec", p.spin_vec); f("spin_mag", p.spin_mag);
        f("dt_level0", p.dt_level0); f("grad_mag", p.grad_mag);
    }
}

void write_particles(hid_t file, const Particles& p_const) {
    Particles& p = const_cast<Particles&>(p_const);   // visit_particle_views only reads here
    auto g = h5::create_group(file, "particles");
    h5::attr(g, "N", p.N);
    h5::attr(g, "capacity", (long long)p.x.extent(0));
    h5::attr(g, "collisional", (int)p.collisional);
    h5::attr(g, "n_levels", p.n_levels);
    visit_particle_views(p, [&](const char* name, const auto& v) { put_view(g, name, v, p.N); });
    h5::write(g, "level_start", p.level_start);
    h5::write(g, "level_count", p.level_count);
}

void read_particles(hid_t file, Particles& p, bool alloc) {
    auto g = h5::open_group(file, "particles");
    const int N = (int)h5::get_int(g, "N");
    const bool coll = h5::get_int(g, "collisional") != 0;
    if (alloc) {
        p.allocate((int)h5::get_int(g, "capacity"), coll);
    } else if (p.collisional != coll || (int)p.x.extent(0) < N) {
        h5::fail("restart particles do not fit the distributed particle arrays");
    }
    p.N = N;
    visit_particle_views(p, [&](const char* name, const auto& v) { get_view(g, name, v, N); });
    p.n_levels = (int)h5::get_int(g, "n_levels");
    h5::read(g, "level_start", p.level_start);
    h5::read(g, "level_count", p.level_count);
    p.layout_dirty_lo = 0; p.layout_dirty_hi = INT_MAX;   // every slot changed: full pid-map rebuild
    p.removed_pid_log.clear();
}

// ---- run state ------------------------------------------------------------------------------------------------
std::vector<double> diag_vec(const DiagnosticState& d) {
    return {d.KE, d.PE, d.E_total, d.Px, d.Py, d.Pz, d.Lx, d.Ly, d.Lz, d.E_pn1, d.E_pn2};
}
DiagnosticState diag_from(const std::vector<double>& v) {
    if (v.size() != 11) h5::fail("restart diag0 has the wrong size");
    DiagnosticState d;
    d.KE = v[0]; d.PE = v[1]; d.E_total = v[2]; d.Px = v[3]; d.Py = v[4]; d.Pz = v[5];
    d.Lx = v[6]; d.Ly = v[7]; d.Lz = v[8]; d.E_pn1 = v[9]; d.E_pn2 = v[10];
    return d;
}

void write_run(hid_t file, const RunState& rs) {
    auto g = h5::create_group(file, "run");
    h5::attr(g, "t", rs.t);
    h5::attr(g, "step", rs.step);
    h5::attr(g, "dt_global", rs.dt_global);
    h5::attr(g, "snapshot_index", rs.snapshot_index);
    h5::attr(g, "integrator", rs.integrator);
    h5::attr(g, "test", rs.test);
    h5::write(g, "diag0", diag_vec(rs.diag0));
    // tidal field reference point (gravity/tidal_field.h): run state
    h5::attr(g, "tidal_X", std::vector<double>(g_tidal.X, g_tidal.X + 3));
    h5::attr(g, "tidal_V", std::vector<double>(g_tidal.V, g_tidal.V + 3));
    auto b = h5::create_group(g, "blobs");
    for (const auto& kv : rs.blobs) h5::write<uint8_t>(b, kv.first, kv.second.data(), kv.second.size());
}

void read_run(hid_t file, RunState& rs) {
    auto g = h5::open_group(file, "run");
    rs.t = h5::get_double(g, "t");
    rs.step = (int)h5::get_int(g, "step");
    rs.dt_global = h5::get_double(g, "dt_global");
    rs.snapshot_index = (int)h5::get_int(g, "snapshot_index");
    rs.integrator = h5::get_string(g, "integrator");
    rs.test = h5::get_string(g, "test");
    std::vector<double> d;
    h5::read(g, "diag0", d);
    rs.diag0 = diag_from(d);
    if (h5::has_attr(g, "tidal_X")) {
        auto X = h5::get_doubles(g, "tidal_X"), V = h5::get_doubles(g, "tidal_V");
        for (int k = 0; k < 3 && k < (int)X.size(); k++) { g_tidal.X[k] = X[k]; g_tidal.V[k] = V[k]; }
    }
    rs.blobs.clear();
    auto b = h5::open_group(g, "blobs");
    H5G_info_t info;
    H5Gget_info(b, &info);
    for (hsize_t k = 0; k < info.nlinks; k++) {
        char name[256];
        if (H5Lget_name_by_idx(b, ".", H5_INDEX_NAME, H5_ITER_INC, k, name, sizeof name, H5P_DEFAULT) < 0) continue;
        std::vector<uint8_t> v;
        h5::read(b, name, v);
        rs.blobs[name].assign(v.begin(), v.end());
    }
}

#ifdef ENABLE_COLLISIONAL
// ---- collisional state -----------------------------------------------------------------------------------------
template <class F>
void visit_config(CollisionalConfig& c, F&& f) {
    f("enabled", c.enabled); f("r_ngb", c.r_ngb); f("r_ngb_max", c.r_ngb_max);
    f("eta_ff", c.eta_ff); f("eta_fb", c.eta_fb); f("eta_grad", c.eta_grad); f("eta_jerk", c.eta_jerk);
    f("eta_va", c.eta_va); f("eta_gbs", c.eta_gbs); f("eta_pert", c.eta_pert);
    f("secular_triples", c.secular_triples);
    f("tidal_gamma", c.tidal_gamma); f("flyby_cm_eps", c.flyby_cm_eps); f("e_ref", c.e_ref);
    f("tidal_step_res", c.tidal_step_res); f("flyby_ngb_cm", c.flyby_ngb_cm); f("tidal_res", c.tidal_res);
    f("eta_restart", c.eta_restart); f("sdar", c.sdar); f("sdar_kref", c.sdar_kref); f("sdar_c", c.sdar_c);
    f("n_orb_secular", c.n_orb_secular); f("m_pn", c.m_pn);
    f("pn_order", c.pn_order); f("r_escape_rh", c.r_escape_rh); f("r_escape_rj", c.r_escape_rj); f("mstar_threshold", c.mstar_threshold);
    f("sse_enabled", c.sse_enabled); f("sse_metallicity", c.sse_metallicity); f("sse_wind", c.sse_wind);
    f("sse_remnant", c.sse_remnant); f("sse_ppisn", c.sse_ppisn); f("sse_ecsn", c.sse_ecsn);
    f("sse_mxns", c.sse_mxns); f("sse_flbv", c.sse_flbv); f("r_ngb_sq", c.r_ngb_sq);
    f("r_ngb_max_sq", c.r_ngb_max_sq); f("c_code", c.c_code); f("ngb_inv_mref", c.ngb_inv_mref);
    f("tde_accretion", c.tde_accretion); f("r_ngb_wide", c.r_ngb_wide);
    f("gbs_deuflhard", c.gbs_deuflhard); f("gbs_kmax", c.gbs_kmax);
    f("grad_fresh", c.grad_fresh); f("bh_merge_max_single", c.bh_merge_max_single);
    f("coll_mass_loss_c", c.coll_mass_loss_c);
    f("sse_natal_kicks", c.sse_natal_kicks); f("kick_seed", c.kick_seed);
}

void write_config(hid_t parent) {
    auto g = h5::create_group(parent, "config");
    visit_config(g_collisional, [&](const char* name, auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) h5::attr(g, name, v);
        else if constexpr (std::is_same_v<T, bool>) h5::attr(g, name, (int)v);
        else if constexpr (std::is_same_v<T, int>) h5::attr(g, name, v);
        else h5::attr(g, name, (double)v);
    });
}

// e_ref (|E0| of the run) is run state: restored. Everything else comes from the config file of the restarted
// run, so a run can be continued with other parameters; differences are reported.
void read_config(hid_t parent) {
    auto g = h5::open_group(parent, "config");
    g_collisional.e_ref = h5::get_double(g, "e_ref");
    // NGB_MASS_SCALING <m>: a run constant like e_ref (from the initial particles), restored
    if (h5::has_attr(g, "ngb_inv_mref")) g_collisional.ngb_inv_mref = h5::get_double(g, "ngb_inv_mref");
    int n_diff = 0;
    visit_config(g_collisional, [&](const char* name, auto& v) {
        using T = std::decay_t<decltype(v)>;
        if (!h5::has_attr(g, name)) return;
        bool same;
        std::string saved;
        if constexpr (std::is_same_v<T, std::string>) { saved = h5::get_string(g, name); same = saved == v; }
        else if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, int>) {
            long long s = h5::get_int(g, name); same = s == (long long)v; saved = std::to_string(s);
        } else {
            double s = h5::get_double(g, name); same = s == v;
            char b[64]; std::snprintf(b, sizeof b, "%.17g", s); saved = b;
        }
        if (!same) {
            n_diff++;
            std::fprintf(stderr, "WARNING: restart: collisional parameter %s differs from the restart file (file: %s); "
                         "the continuation is not the uninterrupted run\n", name, saved.c_str());
        }
    });
    (void)n_diff;
}

void write_subsystems(hid_t parent, const SubsystemManager& m) {
    auto g = h5::create_group(parent, "subsystems");
    h5::attr(g, "next_id", m.next_id);
    h5::attr(g, "max_ngb_pairs", m.max_ngb_pairs);
    const size_t n = m.active.size();
    std::vector<int> id(n), nb(n), type(n), sec(n), norb(n), has_aux(n), sc_set(n), inner(n), outer(n), off(n + 1, 0);
    std::vector<int> member;
    std::vector<double> dt(n), orb(n * 10), aux(n * LOGH_MAX_BODIES * 3), scales(n * 4);
    for (size_t s = 0; s < n; s++) {
        const Subsystem& q = m.active[s];
        id[s] = q.id; nb[s] = q.n_bodies; type[s] = (int)q.type; sec[s] = q.is_secular; norb[s] = q.n_orb_per_dt;
        has_aux[s] = q.has_aux_vel; sc_set[s] = q.scales_set; inner[s] = q.inner_binary_id;
        outer[s] = q.outer_companion_pid; dt[s] = q.dt_sub;
        const OrbitalElements& o = q.orb;
        const double ov[10] = {o.a, o.e, o.inc, o.Omega, o.omega, o.true_anom, o.mean_anom, o.E_orb, o.L_mag, o.period};
        std::copy(ov, ov + 10, &orb[s * 10]);
        std::copy(q.aux_vel_phys, q.aux_vel_phys + LOGH_MAX_BODIES * 3, &aux[s * LOGH_MAX_BODIES * 3]);
        const double sv[4] = {q.V_scale, q.R_scale, q.T_scale, q.E_scale};
        std::copy(sv, sv + 4, &scales[s * 4]);
        member.insert(member.end(), q.particle_ids.begin(), q.particle_ids.end());
        off[s + 1] = (int)member.size();
    }
    h5::write(g, "id", id); h5::write(g, "n_bodies", nb); h5::write(g, "dt_sub", dt);
    h5::write(g, "orb", orb, 10); h5::write(g, "type", type); h5::write(g, "is_secular", sec);
    h5::write(g, "n_orb_per_dt", norb); h5::write(g, "aux_vel_phys", aux, LOGH_MAX_BODIES * 3);
    h5::write(g, "has_aux_vel", has_aux); h5::write(g, "scales", scales, 4); h5::write(g, "scales_set", sc_set);
    h5::write(g, "inner_binary_id", inner); h5::write(g, "outer_companion_pid", outer);
    h5::write(g, "member_offset", off); h5::write(g, "member_pid", member);
    std::vector<int> removed(m.removed_pids.begin(), m.removed_pids.end()), ghosts(m.ghost_pids.begin(), m.ghost_pids.end());
    std::sort(removed.begin(), removed.end());
    std::sort(ghosts.begin(), ghosts.end());
    h5::write(g, "removed_pids", removed);
    h5::write(g, "ghost_pids", ghosts);
}

void read_subsystems(hid_t parent, SubsystemManager& m) {
    auto g = h5::open_group(parent, "subsystems");
    std::vector<int> id, nb, type, sec, norb, has_aux, sc_set, inner, outer, off, member;
    std::vector<double> dt, orb, aux, scales;
    h5::read(g, "id", id); h5::read(g, "n_bodies", nb); h5::read(g, "dt_sub", dt); h5::read(g, "orb", orb);
    h5::read(g, "type", type); h5::read(g, "is_secular", sec); h5::read(g, "n_orb_per_dt", norb);
    h5::read(g, "aux_vel_phys", aux); h5::read(g, "has_aux_vel", has_aux); h5::read(g, "scales", scales);
    h5::read(g, "scales_set", sc_set); h5::read(g, "inner_binary_id", inner); h5::read(g, "outer_companion_pid", outer);
    h5::read(g, "member_offset", off); h5::read(g, "member_pid", member);
    const size_t n = id.size();
    if (off.size() != n + 1 || orb.size() != n * 10 || aux.size() != n * LOGH_MAX_BODIES * 3 || scales.size() != n * 4)
        h5::fail("restart subsystem table is inconsistent");
    m.active.clear();
    m.active.resize(n);
    for (size_t s = 0; s < n; s++) {
        Subsystem& q = m.active[s];
        q.id = id[s]; q.n_bodies = nb[s]; q.type = (SubsystemType)type[s]; q.is_secular = sec[s] != 0;
        q.n_orb_per_dt = norb[s]; q.has_aux_vel = has_aux[s] != 0; q.scales_set = sc_set[s] != 0;
        q.inner_binary_id = inner[s]; q.outer_companion_pid = outer[s]; q.dt_sub = dt[s];
        const double* o = &orb[s * 10];
        q.orb = OrbitalElements{o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], o[9]};
        std::copy(&aux[s * LOGH_MAX_BODIES * 3], &aux[(s + 1) * LOGH_MAX_BODIES * 3], q.aux_vel_phys);
        q.V_scale = scales[s * 4 + 0]; q.R_scale = scales[s * 4 + 1];
        q.T_scale = scales[s * 4 + 2]; q.E_scale = scales[s * 4 + 3];
        q.particle_ids.assign(member.begin() + off[s], member.begin() + off[s + 1]);
    }
    m.next_id = (int)h5::get_int(g, "next_id");
    m.allocate_ngb_buffers((int)h5::get_int(g, "max_ngb_pairs"));
    m.reset_ngb_count();
    std::vector<int> removed, ghosts;
    h5::read(g, "removed_pids", removed);
    h5::read(g, "ghost_pids", ghosts);
    m.removed_pids = std::unordered_set<int>(removed.begin(), removed.end());
    m.ghost_pids = std::unordered_set<int>(ghosts.begin(), ghosts.end());
}

void write_books(hid_t parent, const EnergyBookkeeping& b) {
    auto g = h5::create_group(parent, "energy_books");
    h5::attr(g, "E_escape", b.E_escape); h5::attr(g, "E_gw", b.E_gw); h5::attr(g, "E_merge", b.E_merge);
    h5::attr(g, "E_sse", b.E_sse); h5::attr(g, "E_pn", b.E_pn); h5::attr(g, "E_sd", b.E_sd);
    h5::attr(g, "E_tidal", b.E_tidal);
    h5::attr(g, "P_sink", std::vector<double>(b.P_sink, b.P_sink + 3));
    h5::attr(g, "L_sink", std::vector<double>(b.L_sink, b.L_sink + 3));
}

void read_books(hid_t parent, EnergyBookkeeping& b) {
    auto g = h5::open_group(parent, "energy_books");
    b.E_escape = h5::get_double(g, "E_escape"); b.E_gw = h5::get_double(g, "E_gw");
    b.E_merge = h5::get_double(g, "E_merge"); b.E_sse = h5::get_double(g, "E_sse");
    b.E_pn = h5::get_double(g, "E_pn"); b.E_sd = h5::get_double(g, "E_sd");
    b.E_tidal = h5::has_attr(g, "E_tidal") ? h5::get_double(g, "E_tidal") : 0.0;
    auto P = h5::get_doubles(g, "P_sink"), L = h5::get_doubles(g, "L_sink");
    if (P.size() != 3 || L.size() != 3) h5::fail("restart energy books: bad sink vectors");
    for (int k = 0; k < 3; k++) { b.P_sink[k] = P[k]; b.L_sink[k] = L[k]; }
}

void write_merger_tree(hid_t parent, const MergerTree& t) {
    auto g = h5::create_group(parent, "merger_tree");
    h5::attr(g, "next_id", t.next_id);
    std::vector<int> ints;
    std::vector<double> reals;
    for (const auto& e : t.events) {
        ints.insert(ints.end(), {e.id, e.pid1, e.pid2, e.pid_remnant, (int)e.type});
        reals.insert(reals.end(), {e.time, e.m1, e.m2, e.m_remnant, e.a, e.e, e.v_kick, e.E_radiated,
                                   e.spin_remnant[0], e.spin_remnant[1], e.spin_remnant[2]});
    }
    h5::write(g, "ints", ints.data(), t.events.size(), 5);
    h5::write(g, "reals", reals.data(), t.events.size(), 11);
}

void read_merger_tree(hid_t parent, MergerTree& t) {
    auto g = h5::open_group(parent, "merger_tree");
    std::vector<int> ints;
    std::vector<double> reals;
    const size_t n = h5::read(g, "ints", ints);
    h5::read(g, "reals", reals);
    if (ints.size() != n * 5 || reals.size() != n * 11) h5::fail("restart merger tree is inconsistent");
    t.clear();
    for (size_t k = 0; k < n; k++) {
        MergerTreeNode e{};
        const int* I = &ints[k * 5];
        const double* R = &reals[k * 11];
        e.id = I[0]; e.pid1 = I[1]; e.pid2 = I[2]; e.pid_remnant = I[3]; e.type = (MergerType)I[4];
        e.time = R[0]; e.m1 = R[1]; e.m2 = R[2]; e.m_remnant = R[3]; e.a = R[4]; e.e = R[5]; e.v_kick = R[6];
        e.E_radiated = R[7]; e.spin_remnant[0] = R[8]; e.spin_remnant[1] = R[9]; e.spin_remnant[2] = R[10];
        const int idx = (int)t.events.size();
        t.events.push_back(e);
        // history as MergerTree::record builds it
        t.history[e.pid1].push_back(idx);
        if (e.pid2 != e.pid1) t.history[e.pid2].push_back(idx);
        if (e.pid_remnant != e.pid1 && e.pid_remnant != e.pid2) t.history[e.pid_remnant].push_back(idx);
    }
    t.next_id = (int)h5::get_int(g, "next_id");
}

void write_escapers(hid_t parent, const EscaperLog& log) {
    auto g = h5::create_group(parent, "escaper_log");
    std::vector<int> pid;
    std::vector<double> reals;
    for (const auto& e : log.events) {
        pid.push_back(e.pid);
        reals.insert(reals.end(), {e.time, e.mass, e.pos[0], e.pos[1], e.pos[2], e.vel[0], e.vel[1], e.vel[2],
                                   e.r, e.v, e.E_specific, 0.0});
    }
    h5::write(g, "pid", pid);
    h5::write(g, "reals", reals.data(), log.events.size(), 12);
}

void read_escapers(hid_t parent, EscaperLog& log) {
    auto g = h5::open_group(parent, "escaper_log");
    std::vector<int> pid;
    std::vector<double> reals;
    const size_t n = h5::read(g, "pid", pid);
    h5::read(g, "reals", reals);
    if (reals.size() != n * 12) h5::fail("restart escaper log is inconsistent");
    log.clear();
    for (size_t k = 0; k < n; k++) {
        const double* R = &reals[k * 12];
        EscaperEvent e{};
        e.pid = pid[k]; e.time = R[0]; e.mass = R[1];
        for (int c = 0; c < 3; c++) { e.pos[c] = R[2 + c]; e.vel[c] = R[5 + c]; }
        e.r = R[8]; e.v = R[9]; e.E_specific = R[10];
        log.events.push_back(e);
    }
}

void write_collisional(hid_t file) {
    auto g = h5::create_group(file, "collisional");
    write_config(g);
    write_subsystems(g, g_subsystem_mgr);
    write_books(g, g_energy_books);
    write_merger_tree(g, g_merger_tree);
    write_escapers(g, g_escaper_log);
    h5::write<uint8_t>(g, "eattr", reinterpret_cast<const uint8_t*>(&g_eattr), sizeof(EnergyAttribution));
}

void read_collisional(hid_t file) {
    auto g = h5::open_group(file, "collisional");
    read_config(g);
    read_subsystems(g, g_subsystem_mgr);
    read_books(g, g_energy_books);
    read_merger_tree(g, g_merger_tree);
    read_escapers(g, g_escaper_log);
    std::vector<uint8_t> ea;
    h5::read(g, "eattr", ea);
    if (ea.size() == sizeof(EnergyAttribution)) {
        const bool on = g_eattr.on;   // NBODY_EATTR of this run decides whether the lines are printed
        std::memcpy(static_cast<void*>(&g_eattr), ea.data(), ea.size());
        g_eattr.on = on;
    }
}
#endif  // ENABLE_COLLISIONAL

std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\t': o += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += c;
        }
    }
    return o;
}

void fsync_path(const std::string& path, bool dir) {
    int fd = ::open(path.c_str(), dir ? O_RDONLY | O_DIRECTORY : O_RDONLY);
    if (fd >= 0) { ::fsync(fd); ::close(fd); }
}

// restart_<digits>.h5 -> step, else -1
long long restart_step(const fs::path& f) {
    const std::string n = f.filename().string();
    if (n.size() < 12 || n.rfind("restart_", 0) != 0 || n.substr(n.size() - 3) != ".h5") return -1;
    const std::string d = n.substr(8, n.size() - 11);
    if (d.empty() || !std::all_of(d.begin(), d.end(), ::isdigit)) return -1;
    return std::stoll(d);
}

std::vector<std::pair<long long, fs::path>> restart_files(const std::string& dir) {
    std::vector<std::pair<long long, fs::path>> out;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir.empty() ? "." : dir, ec)) {
        long long s = restart_step(e.path());
        if (s >= 0 && e.is_regular_file()) out.push_back({s, e.path()});
    }
    std::sort(out.begin(), out.end());
    return out;
}

volatile std::sig_atomic_t g_stop_flag = 0;
extern "C" void on_stop_signal(int) { g_stop_flag = 1; }

}  // namespace

std::string config_json(const std::map<std::string, std::string>& params) {
    std::string s = "{";
    bool first = true;
    for (const auto& kv : params) {
        s += (first ? "\"" : ", \"") + json_escape(kv.first) + "\": \"" + json_escape(kv.second) + "\"";
        first = false;
    }
    return s + "}";
}

std::string write_restart(const RestartOptions& opt, const Particles& p, const RunState& rs,
                          const std::string& config_text, const std::map<std::string, std::string>& params) {
    const std::string dir = opt.dir.empty() ? "." : opt.dir;
    std::error_code ec;
    fs::create_directories(dir, ec);
    char name[64];
    std::snprintf(name, sizeof name, "restart_%08d.h5", rs.step);
    const std::string path = (fs::path(dir) / name).string(), tmp = path + ".partial";
    {
        auto f = h5::create_file(tmp);
        h5::attr(f, "format", "cpp-nbody-restart");
        h5::attr(f, "format_version", FORMAT_VERSION);
        h5::attr(f, "code_version", code_version());
        h5::attr(f, "time", rs.t);
        h5::attr(f, "step", rs.step);
#ifdef USE_MPI
        int nr = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &nr);
        h5::attr(f, "n_ranks", nr);
#else
        h5::attr(f, "n_ranks", 1);
#endif
        std::time_t now = std::time(nullptr);
        char tb[64];
        std::strftime(tb, sizeof tb, "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
        h5::attr(f, "created", tb);
        write_run(f, rs);
        auto c = h5::create_group(f, "config");
        h5::attr(c, "config_text", config_text);
        h5::attr(c, "config_json", config_json(params));
        write_particles(f, p);
#ifdef ENABLE_COLLISIONAL
        write_collisional(f);
#endif
        H5Fflush(f, H5F_SCOPE_GLOBAL);
    }
    fsync_path(tmp, false);
    fs::rename(tmp, path);   // atomic on one file system: a killed job leaves either the old or the new set
    fsync_path(dir, true);
    auto files = restart_files(dir);
    const int keep = std::max(1, opt.keep);
    for (size_t k = 0; k + keep < files.size(); k++) fs::remove(files[k].second, ec);
    return path;
}

void read_restart(const std::string& path, Particles& p, RunState& rs, bool alloc,
                  std::map<std::string, std::string>* params_out) {
    auto f = h5::open_file(path);
    if (!h5::has_attr(f, "format") || h5::get_string(f, "format") != "cpp-nbody-restart")
        h5::fail(path + " is not a cpp-nbody restart file");
    const int version = (int)h5::get_int(f, "format_version");
    if (version != FORMAT_VERSION)
        h5::fail(path + ": restart format version " + std::to_string(version) + ", this build reads " +
                 std::to_string(FORMAT_VERSION));
    read_run(f, rs);
    read_particles(f, p, alloc);
#ifdef ENABLE_COLLISIONAL
    if (alloc) {
        if (!h5::exists(f, "collisional")) h5::fail(path + " has no collisional state");
        read_collisional(f);
    }
#else
    if (h5::exists(f, "collisional"))
        std::fprintf(stderr, "WARNING: restart: %s holds collisional state this build ignores\n", path.c_str());
#endif
    if (params_out) {
        params_out->clear();
        auto c = h5::open_group(f, "config");
        // config_json is flat {"KEY": "value", ...}; parse it back
        const std::string s = h5::get_string(c, "config_json");
        size_t i = 0;
        auto str = [&](std::string& out) {
            out.clear();
            i = s.find('"', i);
            if (i == std::string::npos) return false;
            for (++i; i < s.size() && s[i] != '"'; i++) {
                if (s[i] == '\\' && i + 1 < s.size()) {
                    char e = s[++i];
                    out += e == 'n' ? '\n' : e == 't' ? '\t' : e;
                } else out += s[i];
            }
            i++;
            return true;
        };
        std::string k, v;
        while (str(k) && str(v)) (*params_out)[k] = v;
    }
    const std::string ver = h5::has_attr(f, "code_version") ? h5::get_string(f, "code_version") : "?";
    if (ver != code_version())
        std::fprintf(stderr, "WARNING: restart: %s was written by %s, this is %s\n", path.c_str(), ver.c_str(),
                     code_version());
#ifdef ENABLE_COLLISIONAL
    StellarPropsCache::get().invalidate();   // stellar types and radii of the restored particles
#endif
}

std::string latest_restart(const std::string& dir) {
    auto files = restart_files(dir);
    return files.empty() ? std::string() : files.back().second.string();
}

bool any_rank(bool v) {
#ifdef USE_MPI
    int in = v ? 1 : 0, out = 0;
    MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return out != 0;
#else
    return v;
#endif
}

bool ranks_agree_on_run_state(const RunState& rs) {
#ifdef USE_MPI
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size == 1) return true;
    uint64_t h = 1469598103934665603ULL;
    auto add = [&](const void* d, size_t n) {
        auto* b = static_cast<const unsigned char*>(d);
        for (size_t k = 0; k < n; k++) { h ^= b[k]; h *= 1099511628211ULL; }
    };
    add(&rs.t, sizeof rs.t); add(&rs.step, sizeof rs.step); add(&rs.snapshot_index, sizeof rs.snapshot_index);
#ifdef ENABLE_COLLISIONAL
    add(&g_energy_books, sizeof g_energy_books);
    add(&g_subsystem_mgr.next_id, sizeof(int));
    for (const auto& s : g_subsystem_mgr.active) {
        add(&s.id, sizeof s.id); add(&s.dt_sub, sizeof s.dt_sub); add(&s.orb, sizeof s.orb);
        add(s.aux_vel_phys, sizeof s.aux_vel_phys); add(&s.V_scale, 4 * sizeof(double));
        add(s.particle_ids.data(), s.particle_ids.size() * sizeof(int));
    }
    size_t nm = g_merger_tree.events.size(), ne = g_escaper_log.events.size(), nr = g_subsystem_mgr.removed_pids.size();
    add(&nm, sizeof nm); add(&ne, sizeof ne); add(&nr, sizeof nr);
#endif
    unsigned long long mine = h, lo = 0, hi = 0;
    MPI_Allreduce(&mine, &lo, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&mine, &hi, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    if (lo == hi) return true;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) std::printf("RESTART: ranks hold different run states at step %d; no restart file written\n", rs.step);
    return false;
#else
    (void)rs;
    return true;
#endif
}

void install_stop_signals() {
    std::signal(SIGUSR1, on_stop_signal);
}

bool stop_requested() { return g_stop_flag != 0; }
