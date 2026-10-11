#include "forces.h"
#include "constants.h"
#include "integrator.h"
#include "n2_kernels.h"
#include "dt_tree.h"
#ifdef ENABLE_COLLISIONAL
#include "collisional_config.h"
#include "subsystem.h"
#endif
#include <Kokkos_Core.hpp>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

// All pairwise kernels run through run_n2 (n2_driver.h) with the interactions of n2_kernels.h.

namespace {
struct Etas { double ff, fb, jerk, va; };
Etas etas(double eta) {
#ifdef ENABLE_COLLISIONAL
    (void)eta;
    return {g_collisional.eta_ff, g_collisional.eta_fb, g_collisional.eta_jerk, g_collisional.eta_ff};
#else
    return {eta, eta, 0.0, eta};
#endif
}
const int* sid_ptr(const Particles& p, bool coll) { return coll ? p.subsystem_id.data() : nullptr; }

// Velocities for the pair time step: vx, vy, vz, or with FLYBY_CM_VELOCITY the subsystem CoM velocity for members
struct PairVel { RawView x, y, z; };
PairVel pair_velocities(Particles& p, bool coll) {
#ifdef ENABLE_COLLISIONAL
    if (coll && flyby_cm_velocity) {
        static Kokkos::View<double*> cx("cm_vx", 0), cy("cm_vy", 0), cz("cm_vz", 0);
        static bool hook = (Kokkos::push_finalize_hook([] { cx = cy = cz = Kokkos::View<double*>(); }), true);
        (void)hook;
        if ((int)cx.extent(0) < p.N) {
            Kokkos::realloc(cx, p.N); Kokkos::realloc(cy, p.N); Kokkos::realloc(cz, p.N);
        }
        g_subsystem_mgr.fill_cm_velocity(p, cx, cy, cz);
        return {{cx.data()}, {cy.data()}, {cz.data()}};
    }
#else
    (void)coll;
#endif
    return {{p.vx.data()}, {p.vy.data()}, {p.vz.data()}};
}
}  // namespace

bool flyby_cm_velocity = false;
int tree_dt_min_n = 0;
int tree_dt_cpu_max = 0;
double tree_dt_vc_fac = 30.0;

// ==========================================================================
// Exact tree searches for time steps and neighbours (dt_tree.h). TREE_DT_MIN_N: device tree for source sets of at
// least this many particles; TREE_DT_CPU_MAX: host-thread tree for source sets up to this size, where the host can
// read device memory (host builds, APUs). 0 = off (default). NBODY_CHECK_TREE_DT=1 runs the O(N^2) kernel after
// every tree search and aborts on any bitwise difference in dt, the partner-free accelerations or the neighbour pairs.
// ==========================================================================
namespace {
using DevExec = Kokkos::DefaultExecutionSpace;
using HostExec = Kokkos::DefaultHostExecutionSpace;
enum class TreeMode { None, Device, Host };

TreeMode tree_mode(int nj) {
    if (nj < 2) return TreeMode::None;
    if constexpr (kHostAccessibleDevice) {
        if (tree_dt_cpu_max > 0 && nj <= tree_dt_cpu_max) return TreeMode::Host;
    }
    if (tree_dt_min_n > 0 && nj >= tree_dt_min_n) return TreeMode::Device;
    return TreeMode::None;
}

bool check_tree() {
    static const bool v = std::getenv("NBODY_CHECK_TREE_DT") && std::atoi(std::getenv("NBODY_CHECK_TREE_DT")) != 0;
    return v;
}

struct TreeStats {
    long long dt_calls = 0, dt_sinks = 0, dt_pairs = 0, dt_n2_pairs = 0, ngb_calls = 0, ngb_pairs = 0, checked = 0;
    ~TreeStats() {
        if (dt_calls + ngb_calls == 0) return;
        std::printf("TREE_DT: dt calls=%lld sinks=%lld pairs=%lld (N^2 %lld) ngb calls=%lld pairs=%lld checked=%lld\n",
                    dt_calls, dt_sinks, dt_pairs, dt_n2_pairs, ngb_calls, ngb_pairs, checked);
        std::fflush(stdout);
    }
} g_tree_stats;

template <class Exec>
struct TreePair {
    dtt::PointTree<Exec> pos, vel;
};
template <class Exec>
TreePair<Exec>& trees() {
    static TreePair<Exec>* t = nullptr;
    if (!t) {
        t = new TreePair<Exec>();
        Kokkos::push_finalize_hook([] { delete t; t = nullptr; });
    }
    return *t;
}

// Minimum pair step of the sinks over the sources into dt_adapt (sink i), on Exec
template <class Exec>
void tree_min_dt_on(Particles& p, int i0, int ni, int j0, int nj, const PairVel& cv, const int* sid, double G,
                    double soft2, double eta_ff, double eta_fb) {
    auto& T = trees<Exec>();
    const double *X = p.x.data(), *Y = p.y.data(), *Z = p.z.data(), *M = p.mass.data();
    T.pos.build(j0, nj, X, Y, Z, cv.x.ptr, cv.y.ptr, cv.z.ptr, M, false, true);
    T.vel.build(j0, nj, X, Y, Z, cv.x.ptr, cv.y.ptr, cv.z.ptr, M, true);
    dtt::DtParams prm{G, soft2, eta_ff, eta_fb, tree_dt_vc_fac};
    long long np = 0;
    dtt::min_pair_dt(T.pos, T.vel, i0, ni, X, Y, Z, cv.x.ptr, cv.y.ptr, cv.z.ptr, M, sid, p.dt_adapt.data(), prm, &np);
    g_tree_stats.dt_pairs += np;
}

void tree_min_dt(TreeMode mode, Particles& p, int i0, int ni, int j0, int nj, const PairVel& cv, const int* sid,
                 double G, double soft2, double eta_ff, double eta_fb) {
    g_tree_stats.dt_calls++;
    g_tree_stats.dt_sinks += ni;
    g_tree_stats.dt_n2_pairs += (long long)ni * nj;
    if (mode == TreeMode::Host) {
        Kokkos::fence();
        tree_min_dt_on<HostExec>(p, i0, ni, j0, nj, cv, sid, G, soft2, eta_ff, eta_fb);
    } else {
        tree_min_dt_on<DevExec>(p, i0, ni, j0, nj, cv, sid, G, soft2, eta_ff, eta_fb);
    }
}

std::vector<double> host_range(const Kokkos::View<double*>& v, int i0, int n) {
    std::vector<double> out(n);
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(v, Kokkos::make_pair(i0, i0 + n)));
    std::memcpy(out.data(), h.data(), n * sizeof(double));
    return out;
}

void compare_or_abort(const char* what, const std::vector<double>& a, const std::vector<double>& b, int i0) {
    int bad = 0;
    for (size_t k = 0; k < a.size(); k++) {
        if (std::memcmp(&a[k], &b[k], sizeof(double)) != 0) {
            if (bad < 10) std::fprintf(stderr, "TREE_DT CHECK: %s differs at particle %d: tree %.17g N^2 %.17g\n", what,
                                       i0 + (int)k, a[k], b[k]);
            bad++;
        }
    }
    if (bad) {
        std::fprintf(stderr, "TREE_DT CHECK FAILED: %d of %zu values of %s differ\n", bad, a.size(), what);
        std::fflush(stderr);
        std::abort();
    }
}
}  // namespace

// ==========================================================================
// Accelerations and adaptive time steps of all particles
// ==========================================================================
template <bool COLL>
static void acc_dt_range(Particles& p, int i0, int ni, int j0, int nj, double G, double soft2, double eta) {
    Etas e = etas(eta);
    PairVel cv = pair_velocities(p, COLL);
    AccDtK<COLL> k{{p.x.data()}, {p.y.data()}, {p.z.data()}, {p.mass.data()}, {p.vx.data()}, {p.vy.data()},
                   {p.vz.data()}, {p.ax.data()}, {p.ay.data()}, {p.az.data()}, {p.dt_adapt.data()},
                   cv.x, cv.y, cv.z, sid_ptr(p, COLL), G, soft2, e.ff, e.fb, e.jerk, e.va};
    run_n2("acc_dt", i0, ni, j0, nj, k);
}

void acceleration_and_timestep_range(Particles& p, int i0, int ni, int j0, int nj, double G, double soft2,
                                     double eta) {
#ifdef ENABLE_COLLISIONAL
    if (p.collisional) { acc_dt_range<true>(p, i0, ni, j0, nj, G, soft2, eta); return; }
#endif
    acc_dt_range<false>(p, i0, ni, j0, nj, G, soft2, eta);
}

void compute_acceleration_and_timestep(Particles& p, double G, double soft2, double eta) {
    acceleration_and_timestep_range(p, 0, p.N, 0, p.N, G, soft2, eta);
}

// ==========================================================================
// Accelerations of sinks [i0, i0+ni) from sources [j0, j0+nj)
// ==========================================================================
template <bool COLL>
static void force_range(Particles& p, int i0, int ni, int j0, int nj, double G, double soft2) {
    ForceK<COLL> k{{p.x.data()}, {p.y.data()}, {p.z.data()}, {p.mass.data()}, {p.ax.data()}, {p.ay.data()},
                   {p.az.data()}, sid_ptr(p, COLL), G, soft2};
    run_n2("force", i0, ni, j0, nj, k);
}

void acceleration_range(Particles& p, int i0, int ni, int j0, int nj, double G, double soft2) {
#ifdef ENABLE_COLLISIONAL
    if (p.collisional) { force_range<true>(p, i0, ni, j0, nj, G, soft2); return; }
#endif
    force_range<false>(p, i0, ni, j0, nj, G, soft2);
}

void self_acceleration(Particles& p, int start, int count, double G, double soft2) {
    acceleration_range(p, start, count, start, count, G, soft2);
}

// An empty group gives the other one zero acceleration, which callers kick with
void cross_acceleration(Particles& p, int a_start, int a_count, int b_start, int b_count, double G, double soft2) {
    acceleration_range(p, a_start, a_count, b_start, b_count, G, soft2);
    acceleration_range(p, b_start, b_count, a_start, a_count, G, soft2);
}

// ==========================================================================
// Gradient correction of sinks [i0, i0+ni) from sources [j0, j0+nj), from acceleration snapshots
// ==========================================================================
template <bool COLL>
static void grad_range(Particles& p, int i0, int ni, int j0, int nj, const GradSnapshot& si,
                       const GradSnapshot& sj, double G, double soft2, double dt, bool write_gmag) {
    RawView gm{nullptr};
#ifdef ENABLE_COLLISIONAL
    if (COLL && write_gmag) gm = RawView{p.grad_mag.data()};
#else
    (void)write_gmag;
#endif
    GradK<COLL> k{{p.x.data()}, {p.y.data()}, {p.z.data()}, {p.mass.data()}, {p.ax.data()}, {p.ay.data()},
                  {p.az.data()}, {si.x}, {si.y}, {si.z}, {sj.x}, {sj.y}, {sj.z}, si.offset, sj.offset, gm,
                  sid_ptr(p, COLL), G * dt * dt * GRAD_COEFF, soft2, 1.0 / (dt * dt)};
    run_n2("grad", i0, ni, j0, nj, k);
}

void gradient_correction_range(Particles& p, int i0, int ni, int j0, int nj, const GradSnapshot& si,
                               const GradSnapshot& sj, double G, double soft2, double dt, bool write_gmag) {
#ifdef ENABLE_COLLISIONAL
    if (p.collisional) { grad_range<true>(p, i0, ni, j0, nj, si, sj, G, soft2, dt, write_gmag); return; }
#endif
    grad_range<false>(p, i0, ni, j0, nj, si, sj, G, soft2, dt, write_gmag);
}

// Copy ax, ay, az of [start, start+count) into three scratch arrays (index ii = i - start)
static GradSnapshot snapshot_acc(Particles& p, int start, int count, Kokkos::View<double*> sx,
                                 Kokkos::View<double*> sy, Kokkos::View<double*> sz) {
    RawView ax{p.ax.data()}, ay{p.ay.data()}, az{p.az.data()};
    RawView ox{sx.data()}, oy{sy.data()}, oz{sz.data()};
    Kokkos::parallel_for("snap_acc", count, KOKKOS_LAMBDA(int ii) {
        ox(ii) = ax(start + ii); oy(ii) = ay(start + ii); oz(ii) = az(start + ii);
    });
    return {sx.data(), sy.data(), sz.data(), start};
}

void self_gradient_correction(Particles& p, int start, int count, double G, double soft2, double dt) {
    if (count <= 0) return;
    GradSnapshot s = snapshot_acc(p, start, count, p.scratch1, p.scratch2, p.scratch3);
    gradient_correction_range(p, start, count, start, count, s, s, G, soft2, dt, true);
}

void cross_gradient_correction(Particles& p, int a_start, int a_count, int b_start, int b_count,
                               double G, double soft2, double dt) {
    GradSnapshot sa = snapshot_acc(p, a_start, a_count, p.scratch1, p.scratch2, p.scratch3);
    GradSnapshot sb = snapshot_acc(p, b_start, b_count, p.scratch4, p.scratch5, p.scratch6);
    gradient_correction_range(p, a_start, a_count, b_start, b_count, sa, sb, G, soft2, dt, true);
    gradient_correction_range(p, b_start, b_count, a_start, a_count, sb, sa, G, soft2, dt, true);
}

// ==========================================================================
// Adaptive time steps of sinks [i0, i0+ni) from the subset [j0, j0+nj) (HHS recursion)
// ==========================================================================
template <bool COLL>
static void dt_range(Particles& p, int i0, int ni, int j0, int nj, double G, double soft2, double eta) {
    Etas e = etas(eta);
    PairVel cv = pair_velocities(p, COLL);
    const TreeMode mode = tree_mode(nj);
    if (mode != TreeMode::None) {
        // minimum over the pair steps from the tree into dt_adapt; collisional mode then accumulates the partner-free
        // acceleration and jerk over all sources (O(N^2), identical sums) and applies the jerk criterion as DtK does
        tree_min_dt(mode, p, i0, ni, j0, nj, cv, sid_ptr(p, COLL), G, soft2, e.ff, e.fb);
        if (COLL) {
            DtK<COLL, true> ks{{p.x.data()}, {p.y.data()}, {p.z.data()}, {p.mass.data()}, {p.vx.data()}, {p.vy.data()},
                               {p.vz.data()}, {p.ax.data()}, {p.ay.data()}, {p.az.data()}, {p.dt_adapt.data()},
                               cv.x, cv.y, cv.z, sid_ptr(p, COLL), G, soft2, e.ff, e.fb, COLL ? e.jerk : 0.0};
            run_n2("dt_sums", i0, ni, j0, nj, ks);
        }
        if (!check_tree()) return;
        Kokkos::fence();
        std::vector<double> tdt = host_range(p.dt_adapt, i0, ni), tax, tay, taz;
        if (COLL) { tax = host_range(p.ax, i0, ni); tay = host_range(p.ay, i0, ni); taz = host_range(p.az, i0, ni); }
        DtK<COLL> k{{p.x.data()}, {p.y.data()}, {p.z.data()}, {p.mass.data()}, {p.vx.data()}, {p.vy.data()},
                    {p.vz.data()}, {p.ax.data()}, {p.ay.data()}, {p.az.data()}, {p.dt_adapt.data()},
                    cv.x, cv.y, cv.z, sid_ptr(p, COLL), G, soft2, e.ff, e.fb, COLL ? e.jerk : 0.0};
        run_n2("dt_subset", i0, ni, j0, nj, k);
        Kokkos::fence();
        compare_or_abort("dt", tdt, host_range(p.dt_adapt, i0, ni), i0);
        if (COLL) {
            compare_or_abort("ax", tax, host_range(p.ax, i0, ni), i0);
            compare_or_abort("ay", tay, host_range(p.ay, i0, ni), i0);
            compare_or_abort("az", taz, host_range(p.az, i0, ni), i0);
        }
        g_tree_stats.checked++;
        return;
    }
    DtK<COLL> k{{p.x.data()}, {p.y.data()}, {p.z.data()}, {p.mass.data()}, {p.vx.data()}, {p.vy.data()},
                {p.vz.data()}, {p.ax.data()}, {p.ay.data()}, {p.az.data()}, {p.dt_adapt.data()},
                cv.x, cv.y, cv.z, sid_ptr(p, COLL), G, soft2, e.ff, e.fb, COLL ? e.jerk : 0.0};
    run_n2("dt_subset", i0, ni, j0, nj, k);
}

void timestep_range(Particles& p, int i0, int ni, int j0, int nj, double G, double soft2, double eta) {
#ifdef ENABLE_COLLISIONAL
    if (p.collisional) { dt_range<true>(p, i0, ni, j0, nj, G, soft2, eta); return; }
#endif
    dt_range<false>(p, i0, ni, j0, nj, G, soft2, eta);
}

void compute_timestep_subset(Particles& p, int start, int count, double G, double soft2, double eta) {
    if (count <= 1) return;  // no pairs
    timestep_range(p, start, count, start, count, G, soft2, eta);
}

// ==========================================================================
// Standalone neighbor search (BIFROST Section 3.5)
// Lightweight N² tiled kernel that detects neighbor pairs without computing
// forces or timesteps. Called once per HHS level BEFORE subsystem detection
// and timestep computation.
// No subsystem exclusion — every pair is checked fresh.
// ==========================================================================
#ifdef ENABLE_COLLISIONAL
static void neighbor_search_kernel(Particles& p, int start, int count, double G, double r_ngb, double r_ngb_max,
                                   double pivot_dt, bool allow_tree = true) {
    auto& mgr = g_subsystem_mgr;
    const TreeMode mode = allow_tree ? tree_mode(count) : TreeMode::None;
    if (mode != TreeMode::None) {
        dtt::NgbOut o{mgr.ngb_pair_i.data(), mgr.ngb_pair_j.data(), mgr.ngb_pair_bound.data(), mgr.ngb_count.data(),
                      mgr.ngb_pair_a.data(), mgr.ngb_pair_e.data(), mgr.ngb_pair_E.data(), mgr.max_ngb_pairs};
        const double *X = p.x.data(), *Y = p.y.data(), *Z = p.z.data(), *VX = p.vx.data(), *VY = p.vy.data(),
                     *VZ = p.vz.data(), *M = p.mass.data();
        if (mode == TreeMode::Host) {
            Kokkos::fence();
            auto& T = trees<HostExec>();
            T.pos.build(start, count, X, Y, Z, VX, VY, VZ, M, false);
            dtt::ngb_pairs(T.pos, start, count, X, Y, Z, VX, VY, VZ, M, G, r_ngb, r_ngb_max * r_ngb_max, pivot_dt, o);
        } else {
            auto& T = trees<DevExec>();
            T.pos.build(start, count, X, Y, Z, VX, VY, VZ, M, false);
            dtt::ngb_pairs(T.pos, start, count, X, Y, Z, VX, VY, VZ, M, G, r_ngb, r_ngb_max * r_ngb_max, pivot_dt, o);
        }
        Kokkos::fence();
        return;
    }
    NgbK k{{p.x.data()}, {p.y.data()}, {p.z.data()}, {p.mass.data()}, {p.vx.data()}, {p.vy.data()}, {p.vz.data()},
           mgr.ngb_pair_i.data(), mgr.ngb_pair_j.data(), mgr.ngb_pair_bound.data(), mgr.ngb_count.data(),
           mgr.ngb_pair_a.data(), mgr.ngb_pair_e.data(), mgr.ngb_pair_E.data(), mgr.max_ngb_pairs,
           G, r_ngb, r_ngb_max * r_ngb_max, pivot_dt};
    run_n2("ngb_search", start, count, start, count, k);
    Kokkos::fence();
}

// NBODY_CHECK_TREE_DT: the pairs found by the tree (sorted, with their a, e, E and bound flag) must equal the O(N^2)
// kernel's bitwise; the N^2 result stays in the buffers
static void check_ngb_against_n2(Particles& p, int start, int count, double G, double r_ngb, double r_ngb_max,
                                 double pivot_dt) {
    auto& mgr = g_subsystem_mgr;
    struct Row { int i, j, b; double a, e, E; };
    auto rows = [&]() {
        int n = 0;
        Kokkos::deep_copy(n, Kokkos::subview(mgr.ngb_count, 0));
        n = std::min(n, mgr.max_ngb_pairs);
        auto cp = [&](auto v) {
            return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(v, Kokkos::make_pair(0, n)));
        };
        auto hi = cp(mgr.ngb_pair_i), hj = cp(mgr.ngb_pair_j), hb = cp(mgr.ngb_pair_bound);
        auto ha = cp(mgr.ngb_pair_a), he = cp(mgr.ngb_pair_e), hE = cp(mgr.ngb_pair_E);
        std::vector<Row> r(n);
        for (int k = 0; k < n; k++) r[k] = {hi(k), hj(k), hb(k), ha(k), he(k), hE(k)};
        std::sort(r.begin(), r.end(), [](const Row& x, const Row& y) { return x.i != y.i ? x.i < y.i : x.j < y.j; });
        return r;
    };
    int n_before = 0;
    Kokkos::deep_copy(n_before, Kokkos::subview(mgr.ngb_count, 0));
    std::vector<Row> tree_rows = rows();
    // the HHS call sites reset the counter before the search, so the buffer holds this search's pairs only
    mgr.reset_ngb_count();
    neighbor_search_kernel(p, start, count, G, r_ngb, r_ngb_max, pivot_dt, false);
    int n_after = 0;
    Kokkos::deep_copy(n_after, Kokkos::subview(mgr.ngb_count, 0));
    std::vector<Row> n2_rows = rows();
    bool ok = n_before == n_after && tree_rows.size() == n2_rows.size();
    for (size_t k = 0; ok && k < tree_rows.size(); k++) {
        const Row &a = tree_rows[k], &b = n2_rows[k];
        ok = a.i == b.i && a.j == b.j && a.b == b.b && std::memcmp(&a.a, &b.a, sizeof(double)) == 0 &&
             std::memcmp(&a.e, &b.e, sizeof(double)) == 0 && std::memcmp(&a.E, &b.E, sizeof(double)) == 0;
        if (!ok) std::fprintf(stderr, "TREE_DT CHECK: neighbour pair %zu differs: tree (%d,%d) N^2 (%d,%d)\n", k, a.i, a.j,
                              b.i, b.j);
    }
    if (!ok) {
        std::fprintf(stderr, "TREE_DT CHECK FAILED: neighbour pairs tree %d (%zu stored) vs N^2 %d (%zu stored)\n",
                     n_before, tree_rows.size(), n_after, n2_rows.size());
        std::fflush(stderr);
        std::abort();
    }
    g_tree_stats.ngb_pairs += n_after;
    g_tree_stats.checked++;
}

void neighbor_search_subset(Particles& p, int start, int count,
                            double G, double r_ngb, double r_ngb_max,
                            double pivot_dt) {
    if (count <= 1) return;
    if (tree_mode(count) != TreeMode::None) g_tree_stats.ngb_calls++;   // once per search (a buffer overflow repeats it)
    neighbor_search_kernel(p, start, count, G, r_ngb, r_ngb_max, pivot_dt);
    // The kernel counts every pair but stores only max_ngb_pairs: an overflow would silently drop pairs
    // (binaries left undetected are then integrated orbit by orbit by the FSI). Grow and search again.
    int n_found = 0;
    Kokkos::deep_copy(n_found, Kokkos::subview(g_subsystem_mgr.ngb_count, 0));
    if (n_found > g_subsystem_mgr.max_ngb_pairs) {
        g_subsystem_mgr.allocate_ngb_buffers(2 * n_found);
        g_subsystem_mgr.reset_ngb_count();
        neighbor_search_kernel(p, start, count, G, r_ngb, r_ngb_max, pivot_dt);
    }
    if (check_tree() && tree_mode(count) != TreeMode::None) check_ngb_against_n2(p, start, count, G, r_ngb, r_ngb_max, pivot_dt);
}
#else
void neighbor_search_subset(Particles&, int, int, double, double, double, double) {}
#endif


// ==========================================================================
// HHS with connected components (HHS_CC)
// ==========================================================================
namespace {
Kokkos::View<int*>& cc_buf(int k) {   // 0: edge i, 1: edge j, 2: counter
    static Kokkos::View<int*> b[3] = {Kokkos::View<int*>("cc_ei", 0), Kokkos::View<int*>("cc_ej", 0), Kokkos::View<int*>("cc_cnt", 1)};
    static bool hook = (Kokkos::push_finalize_hook([] { for (auto& v : b) v = Kokkos::View<int*>(); }), true);
    (void)hook;
    return b[k];
}
}  // namespace

std::vector<int> cc_components(Particles& p, int start, int count, double pivot, double G, double soft2, double eta) {
    std::vector<int> sizes{count};
    if (count < 2) return sizes;
    const int max_edges = 16 * count + 1024;
    if ((int)cc_buf(0).extent(0) < max_edges) { Kokkos::realloc(cc_buf(0), max_edges); Kokkos::realloc(cc_buf(1), max_edges); }
    if (cc_buf(2).extent(0) < 1) Kokkos::realloc(cc_buf(2), 1);
    Kokkos::deep_copy(cc_buf(2), 0);
    Etas e = etas(eta);
    static Kokkos::View<double*> dmin("cc_dmin", 0);
    static bool hook = (Kokkos::push_finalize_hook([] { dmin = Kokkos::View<double*>(); }), true);
    (void)hook;
    if ((int)dmin.extent(0) < count) Kokkos::realloc(dmin, count);
    PairMinK pm{{p.x.data()}, {p.y.data()}, {p.z.data()}, {p.mass.data()}, {p.vx.data()}, {p.vy.data()}, {p.vz.data()},
                {dmin.data()}, start, sid_ptr(p, p.collisional), G, soft2, e.ff, e.fb};
    run_n2("cc_pair_min", start, count, start, count, pm);
    auto launch = [&](auto coll_tag) {
        constexpr bool COLL = decltype(coll_tag)::value;
        EdgeK<COLL> k{{p.x.data()}, {p.y.data()}, {p.z.data()}, {p.mass.data()}, {p.vx.data()}, {p.vy.data()}, {p.vz.data()},
                      sid_ptr(p, COLL), cc_buf(0).data(), cc_buf(1).data(), cc_buf(2).data(), max_edges,
                      G, soft2, e.ff, e.fb, pivot, {dmin.data()}, start};
        run_n2("cc_edges", start, count, start, count, k);
    };
#ifdef ENABLE_COLLISIONAL
    if (p.collisional) launch(std::true_type{}); else
#endif
    launch(std::false_type{});
    int n_edges = 0;
    Kokkos::deep_copy(n_edges, Kokkos::subview(cc_buf(2), 0));
    if (n_edges > max_edges) return sizes;   // dense: one component
    auto hi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(cc_buf(0), Kokkos::make_pair(0, n_edges)));
    auto hj = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(cc_buf(1), Kokkos::make_pair(0, n_edges)));
    std::vector<int> parent(count);
    for (int k = 0; k < count; k++) parent[k] = k;
    auto find = [&](int a) { while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
    for (int k = 0; k < n_edges; k++) {
        int a = find(hi(k) - start), b = find(hj(k) - start);
        if (a != b) parent[std::max(a, b)] = std::min(a, b);
    }
    // components ordered by their first slot; slots in their original order inside a component
    std::vector<int> root(count), comp_of_root(count, -1), order;
    sizes.clear();
    for (int k = 0; k < count; k++) {
        root[k] = find(k);
        if (comp_of_root[root[k]] < 0) { comp_of_root[root[k]] = (int)sizes.size(); sizes.push_back(0); }
        sizes[comp_of_root[root[k]]]++;
    }
    if (sizes.size() == 1) return sizes;
    std::vector<int> offs(sizes.size(), 0);
    for (size_t c = 1; c < sizes.size(); c++) offs[c] = offs[c - 1] + sizes[c - 1];
    order.resize(count);
    for (int k = 0; k < count; k++) order[offs[comp_of_root[root[k]]]++] = start + k;
    auto hs = Kokkos::create_mirror_view(p.scratch_i);
    Kokkos::deep_copy(hs, p.scratch_i);
    for (int k = 0; k < count; k++) hs(k) = order[k];
    Kokkos::deep_copy(p.scratch_i, hs);
    p.apply_permutation(start, count);
    return sizes;
}

void cc_inter_acceleration(Particles& p, int start, int count, const int* cid, double G, double soft2) {
    auto launch = [&](auto coll_tag) {
        constexpr bool COLL = decltype(coll_tag)::value;
        ForceCidK<COLL> k{{p.x.data()}, {p.y.data()}, {p.z.data()}, {p.mass.data()}, {p.ax.data()}, {p.ay.data()},
                          {p.az.data()}, sid_ptr(p, COLL), cid, G, soft2};
        run_n2("cc_inter_force", start, count, start, count, k);
    };
#ifdef ENABLE_COLLISIONAL
    if (p.collisional) { launch(std::true_type{}); return; }
#endif
    launch(std::false_type{});
}

void cc_inter_gradient(Particles& p, int start, int count, const int* cid, const GradSnapshot& s, double G, double soft2,
                       double dt) {
    auto launch = [&](auto coll_tag) {
        constexpr bool COLL = decltype(coll_tag)::value;
        GradCidK<COLL> k{{p.x.data()}, {p.y.data()}, {p.z.data()}, {p.mass.data()}, {p.ax.data()}, {p.ay.data()},
                         {p.az.data()}, {s.x}, {s.y}, {s.z}, s.offset, sid_ptr(p, COLL), cid,
                         G * dt * dt * GRAD_COEFF, soft2};
        run_n2("cc_inter_grad", start, count, start, count, k);
    };
#ifdef ENABLE_COLLISIONAL
    if (p.collisional) { launch(std::true_type{}); return; }
#endif
    launch(std::false_type{});
}
