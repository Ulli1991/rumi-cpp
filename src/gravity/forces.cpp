#include "forces.h"
#include "constants.h"
#include "integrator.h"
#include "n2_kernels.h"
#ifdef ENABLE_COLLISIONAL
#include "collisional_config.h"
#include "subsystem.h"
#endif
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

// All pairwise kernels run through run_n2 (n2_driver.h) with the interactions of n2_kernels.h.

namespace {
struct Etas { double ff, fb, jerk, va; };
Etas etas(double eta) {
#ifdef ENABLE_COLLISIONAL
    (void)eta;
    return {g_collisional.eta_ff, g_collisional.eta_fb, g_collisional.eta_jerk, g_collisional.eta_va};
#else
    return {eta, eta, 0.0, eta};
#endif
}
const int* sid_ptr(const Particles& p, bool coll) { return coll ? p.subsystem_id.data() : nullptr; }

struct PairVel { RawView x, y, z; };
PairVel pair_vel(Particles& p, bool coll, int lo, int hi) {
    PairVelocities v = coll ? pair_velocities(p, lo, hi) : PairVelocities{p.vx.data(), p.vy.data(), p.vz.data()};
    return {{v.x}, {v.y}, {v.z}};
}
// Union of the sink and source ranges of a pairwise kernel
int range_lo(int i0, int ni, int j0, int nj) { return std::min(ni > 0 ? i0 : j0, nj > 0 ? j0 : i0); }
int range_hi(int i0, int ni, int j0, int nj) { return std::max(i0 + ni, j0 + nj); }
}  // namespace

bool flyby_cm_velocity = false;

PairVelocities pair_velocities(Particles& p, int lo, int hi) {
#ifdef ENABLE_COLLISIONAL
    if (p.collisional && flyby_cm_velocity) {
        static Kokkos::View<double*> cx("cm_vx", 0), cy("cm_vy", 0), cz("cm_vz", 0);
        static bool hook = (Kokkos::push_finalize_hook([] { cx = cy = cz = Kokkos::View<double*>(); }), true);
        (void)hook;
        if ((int)cx.extent(0) < p.N) {
            Kokkos::realloc(cx, p.N); Kokkos::realloc(cy, p.N); Kokkos::realloc(cz, p.N);
        }
        g_subsystem_mgr.fill_cm_velocity(p, std::max(lo, 0), std::min(hi, p.N), cx, cy, cz);
        return {cx.data(), cy.data(), cz.data()};
    }
#else
    (void)lo; (void)hi;
#endif
    return {p.vx.data(), p.vy.data(), p.vz.data()};
}

// ==========================================================================
// Accelerations and adaptive time steps of all particles
// ==========================================================================
template <bool COLL>
static void acc_dt_range(Particles& p, int i0, int ni, int j0, int nj, double G, double soft2, double eta) {
    Etas e = etas(eta);
    PairVel cv = pair_vel(p, COLL, range_lo(i0, ni, j0, nj), range_hi(i0, ni, j0, nj));
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
    PairVel cv = pair_vel(p, COLL, range_lo(i0, ni, j0, nj), range_hi(i0, ni, j0, nj));
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
// Pairs i < j of the subset that BIFROST sec. 3.5 counts as neighbours: bound with a or r below r_ngb, or a
// fly-by reaching pericentre below r_ngb within the step (Eq. 4). Appended to the neighbour buffers through an
// atomic counter (order is irrelevant: detect_from_pairs sorts); NS = NM = 0, so run_n2 only drives the tiling.
struct NgbK {
    static constexpr int NSRC = 7, NS = 0, NM = 0;
    RawView x, y, z, m, vx, vy, vz;
    int *pi, *pj, *pb, *cnt;
    double *pa, *pe, *pE;
    int max_ngb;
    double G, r_ngb, r_ngb_max_sq, pivot_dt;
    struct Sink { double x, y, z, m, vx, vy, vz; };
    KOKKOS_INLINE_FUNCTION Sink sink(int i) const { return {x(i), y(i), z(i), m(i), vx(i), vy(i), vz(i)}; }
    KOKKOS_INLINE_FUNCTION void source(int j, double* s) const {
        s[0] = x(j); s[1] = y(j); s[2] = z(j); s[3] = m(j); s[4] = vx(j); s[5] = vy(j); s[6] = vz(j);
    }
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, double*, double*) const {
        if (j <= i) return;  // upper triangle
        double dx = s[0] - a.x, dy = s[1] - a.y, dz = s[2] - a.z;
        double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= r_ngb_max_sq) return;
        double dvx = s[4] - a.vx, dvy = s[5] - a.vy, dvz = s[6] - a.vz;
        double v_sq = dvx * dvx + dvy * dvy + dvz * dvz;
        double mu = G * (a.m + s[3]);
        double r_val = Kokkos::sqrt(r2);
        double E_orb = 0.5 * v_sq - mu / (r_val + EPS);
        bool is_bound = (E_orb < 0);
        double a_val = -mu / (2.0 * E_orb + EPS);
        double Lx = dy * dvz - dz * dvy, Ly = dz * dvx - dx * dvz, Lz = dx * dvy - dy * dvx;
        double L2 = Lx * Lx + Ly * Ly + Lz * Lz;
        double e_arg = 1.0 + 2.0 * E_orb * L2 / (mu * mu + EPS);
        double e_val = Kokkos::sqrt(Kokkos::fmax(is_bound ? 0.0 : 1.0, e_arg));
        bool is_neighbor = false;
        if (is_bound) {
            is_neighbor = (a_val < r_ngb || r_val < r_ngb);
        } else if (r_val < r_ngb) {
            is_neighbor = true;
        } else if (Kokkos::fabs(a_val) * (e_val - 1.0) < r_ngb && v_sq > EPS) {   // fly-by, BIFROST Eq. 4
            is_neighbor = 0.5 * r_val / Kokkos::sqrt(v_sq) < pivot_dt;
        }
        if (is_neighbor) {
            int k = Kokkos::atomic_fetch_add(cnt, 1);
            if (k < max_ngb) {   // array indices; detect_from_pairs converts to particle ids
                pi[k] = i; pj[k] = j; pa[k] = a_val; pe[k] = e_val; pE[k] = E_orb; pb[k] = is_bound ? 1 : 0;
            }
        }
    }
    KOKKOS_INLINE_FUNCTION void finish(int, const Sink&, const double*, const double*) const {}
};

static void neighbor_search_kernel(Particles& p, int start, int count, double G, double r_ngb, double r_ngb_max,
                                   double pivot_dt) {
    auto& mgr = g_subsystem_mgr;
    NgbK k{{p.x.data()}, {p.y.data()}, {p.z.data()}, {p.mass.data()}, {p.vx.data()}, {p.vy.data()}, {p.vz.data()},
           mgr.ngb_pair_i.data(), mgr.ngb_pair_j.data(), mgr.ngb_pair_bound.data(), mgr.ngb_count.data(),
           mgr.ngb_pair_a.data(), mgr.ngb_pair_e.data(), mgr.ngb_pair_E.data(), mgr.max_ngb_pairs,
           G, r_ngb, r_ngb_max * r_ngb_max, pivot_dt};
    run_n2("ngb_search", start, count, start, count, k);
    Kokkos::fence();
}

void neighbor_search_subset(Particles& p, int start, int count,
                            double G, double r_ngb, double r_ngb_max,
                            double pivot_dt) {
    if (count <= 1) return;
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
    // the pair time steps of DtK/AccDtK: with FLYBY_CM_VELOCITY the subsystem c.o.m. velocity of members
    // (FROST-CLUSTERS II sec. 2.1)
    PairVel cv = pair_vel(p, p.collisional, start, start + count);
    PairMinK pm{{p.x.data()}, {p.y.data()}, {p.z.data()}, {p.mass.data()}, cv.x, cv.y, cv.z,
                {dmin.data()}, start, sid_ptr(p, p.collisional), G, soft2, e.ff, e.fb};
    run_n2("cc_pair_min", start, count, start, count, pm);
    // NBODY_CHECK_CC: the split must see the pair time steps of the time-step kernel (DtK, over a superset of
    // this set, only lowered afterwards by other criteria), so no particle's pair minimum here may lie below
    // its dt_adapt. Reports the particles that violate it (relative tolerance 1e-12 for FMA contraction).
    static const bool check_cc = std::getenv("NBODY_CHECK_CC") != nullptr;
    if (check_cc) {
        RawView dm{dmin.data()}, dta{p.dt_adapt.data()};
        int nbad = 0;
        Kokkos::parallel_reduce("cc_check", count, KOKKOS_LAMBDA(int k, int& n) {
            if (dm(k) < dta(start + k) * (1.0 - 1e-12)) n++;
        }, nbad);
        if (nbad > 0) std::printf("CC_CHECK: %d of %d particles with a pair step below dt_adapt\n", nbad, count);
    }
    auto launch = [&](auto coll_tag) {
        constexpr bool COLL = decltype(coll_tag)::value;
        EdgeK<COLL> k{{p.x.data()}, {p.y.data()}, {p.z.data()}, {p.mass.data()}, cv.x, cv.y, cv.z,
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
