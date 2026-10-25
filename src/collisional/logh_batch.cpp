#include "small_for.h"
#include "lean_sync.h"
#include "device_staging.h"
#include "logh_batch.h"
#include "constants.h"
#include "stellar_cache.h"
#include "logh.h"
#include "gbs.h"
#include "gauss_irk.h"
#include "collisional_config.h"
#include "post_newtonian.h"
#include "secular.h"
#include "kepler.h"
#include "profiling.h"
#include "stall.h"
#include "mstar_dispatch.h"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <cstdio>
#include <string>
#include <chrono>
#include <atomic>

SubsystemBatch g_logh_batch;
LoghBatchProf g_logh_prof;

namespace {
inline double wall_now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

constexpr int GBS_MAX_STEPS = 1000;

// GBS step limit of one LogH call: 1000, or 64 steps per orbit of the tightest bound pair over the interval
// (a tight binary inside a fly-by or an unstable triple is integrated orbit by orbit: thousands of orbits per
// interval at coarse levels hit the fixed limit after a few per cent of the interval, and the retry split it
// into up to 256 pieces). Hard ceiling 4e6 steps; NBODY_GBS_STEPS_PER_ORBIT overrides 64 (0: fixed 1000).
// The cost stays ~ orbits per interval: the remedy for weakly perturbed inner binaries is the slow-down of their
// orbital motion (Mikkola & Aarseth 1996; SDAR, Wang, Nitadori & Makino 2020, MNRAS 493, 3398), which replaced
// plain LogH in FROST-CLUSTERS I (Rantala et al. 2024, App. A2; kappa = 1 for PN subsystems there).
// With SDAR the limit counts slowed orbits: a slowed pair's period is kappa P (kappa at the start of the call).
static double gbs_steps_per_orbit() {
    static const double per_orbit = std::getenv("NBODY_GBS_STEPS_PER_ORBIT")
                                        ? std::atof(std::getenv("NBODY_GBS_STEPS_PER_ORBIT")) : 64.0;
    return per_orbit;
}

static int gbs_step_limit(const LogHState& st, double dt, double G) {
    const double per_orbit = gbs_steps_per_orbit();
    double P_min = std::numeric_limits<double>::max();
    for (int i = 0; i < st.n_bodies; i++)
        for (int j = i + 1; j < st.n_bodies; j++) {
            double r2 = 0.0, v2 = 0.0;
            for (int k = 0; k < 3; k++) {
                const double dx = st.pos[j*3+k] - st.pos[i*3+k], dv = st.vel[j*3+k] - st.vel[i*3+k];
                r2 += dx * dx;
                v2 += dv * dv;
            }
            const double mu = G * (st.mass[i] + st.mass[j]);
            const double eps = 0.5 * v2 - mu / std::sqrt(r2);   // specific two-body energy
            if (eps >= 0.0) continue;
            const double a = -mu / (2.0 * eps);
            double P = 2.0 * M_PI * std::sqrt(a * a * a / mu);
            for (int k = 0; k < st.n_sd; k++)
                if (st.sd_a[k] == i && st.sd_b[k] == j) P *= st.sd_kappa[k];
            P_min = std::min(P_min, P);
        }
    if (per_orbit <= 0.0 || P_min == std::numeric_limits<double>::max()) return GBS_MAX_STEPS;
    const double n = per_orbit * std::fabs(dt) / P_min;
    return n > (double)GBS_MAX_STEPS ? (int)std::min(n, 4.0e6) : GBS_MAX_STEPS;
}

// ============================================================================
// Gauss-IRK for compact Newtonian subsystems (docs/CASES.md C22; NBODY_GAUSS_IRK=1, default off)
// ============================================================================

static bool gauss_irk_on() {
    static const bool on = [] { const char* e = std::getenv("NBODY_GAUSS_IRK"); return e && std::atoi(e) != 0; }();
    return on;
}

// Totals over the run, printed at exit when NBODY_GAUSS_IRK is on
namespace {
struct IrkTotals {
    std::atomic<long long> calls{0}, fallbacks{0}, steps{0}, rhs{0}, newton{0}, fail[5] = {}, not_chosen[8] = {};
    ~IrkTotals() {
        if (!gauss_irk_on()) return;
        std::printf("Gauss-IRK stats: %lld calls, %lld fell back to GBS (step limit %lld, no convergence %lld, "
                    "collision watch %lld, other %lld), %lld steps, %lld force evaluations, %lld Newton steps\n",
                    calls.load(), fallbacks.load(), fail[1].load(), fail[2].load(), fail[3].load(),
                    fail[4].load() + fail[0].load(), steps.load(), rhs.load(), newton.load());
        std::printf("Gauss-IRK not chosen: size %lld, collision candidate %lld, e >= 0.99 %lld, hard encounter %lld, "
                    "stable hierarchy %lld, PN member %lld, SDAR slow-down %lld\n", not_chosen[1].load(),
                    not_chosen[2].load(), not_chosen[3].load(), not_chosen[4].load(), not_chosen[5].load(),
                    not_chosen[6].load(), not_chosen[7].load());
    }
} g_irk_totals;
}  // namespace

// Classification, local criteria only (normalised state, G = 1); returns 0 (Gauss-IRK) or the reason against it:
// 1 size, 2 collision candidate, 3 hard pericentre, 4 hard encounter, 5 stable hierarchy (6 PN member, 7 SDAR: caller): 2-4 bodies, no collision candidate (osculating
// pericentre inside the collision distance), no hard pericentre (bound pair with e >= 0.99) or hard encounter (an
// approaching unbound pair reaching r_p < 0.01 r within the call), and no stable hierarchy by the Mardling & Aarseth
// test of the triple classification (n = 3: tightest pair + third body; n = 4 with an unbound second pair: tightest
// pair + the body nearest to it; a 2+2 quadruple has no MA test in the code and qualifies). Gauss-IRK loses on
// strongly hierarchical systems (prototype: Kozai triple 4x the evaluations of LogH+GBS), and those are the SDAR /
// secular cases; PN members and SDAR slow-downs are excluded by the caller.
static int gauss_irk_choose(const LogHState& st, double G, double dt, const double* coll) {
    const int n = st.n_bodies;
    if (n < 2 || n > 4) return 1;
    int bi = -1, bj = -1;
    double a_min = 1e300;
    bool bound[LOGH_MAX_BODIES][LOGH_MAX_BODIES] = {};
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            double r[3], v[3];
            for (int c = 0; c < 3; c++) { r[c] = st.pos[j*3+c] - st.pos[i*3+c]; v[c] = st.vel[j*3+c] - st.vel[i*3+c]; }
            const double rr = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
            const double v2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2], rv = r[0]*v[0] + r[1]*v[1] + r[2]*v[2];
            const double mu = G * (st.mass[i] + st.mass[j]);
            const double eps = 0.5 * v2 - mu / rr;
            const double h2 = rr * rr * v2 - rv * rv;
            const double e = std::sqrt(std::max(0.0, 1.0 + 2.0 * eps * h2 / (mu * mu)));
            const double rp = h2 / (mu * (1.0 + e));
            const double rc = coll ? coll[i * LOGH_MAX_BODIES + j] : 0.0;
            if (rc > 0.0 && rp < rc) return 2;                        // collision candidate
            if (eps < 0.0) {
                if (e >= 0.99) return 3;                              // hard pericentre
                bound[i][j] = true;
                const double a = -mu / (2.0 * eps);
                if (a < a_min) { a_min = a; bi = i; bj = j; }
            } else if (rv < 0.0 && rp < 1e-2 * rr) {                  // hard encounter within the call
                double d[3] = {r[0], r[1], r[2]}, u[3] = {v[0], v[1], v[2]}, rpk = 0.0;
                if (kepler_time_to_pericentre(d, u, mu, &rpk) < dt) return 4;
            }
        }
    if (n == 2 || bi < 0) return 0;
    int rest[2], nr = 0;
    for (int k = 0; k < n; k++) if (k != bi && k != bj) rest[nr++] = k;
    if (n == 4 && bound[std::min(rest[0], rest[1])][std::max(rest[0], rest[1])]) return 0;   // 2+2
    const double m1 = st.mass[bi], m2 = st.mass[bj], Mi = m1 + m2;
    double xc[3], vc[3];
    for (int c = 0; c < 3; c++) {
        xc[c] = (m1 * st.pos[bi*3+c] + m2 * st.pos[bj*3+c]) / Mi;
        vc[c] = (m1 * st.vel[bi*3+c] + m2 * st.vel[bj*3+c]) / Mi;
    }
    int ko = rest[0];
    if (nr == 2) {
        double d0 = 0.0, d1 = 0.0;
        for (int c = 0; c < 3; c++) {
            d0 += (st.pos[rest[0]*3+c] - xc[c]) * (st.pos[rest[0]*3+c] - xc[c]);
            d1 += (st.pos[rest[1]*3+c] - xc[c]) * (st.pos[rest[1]*3+c] - xc[c]);
        }
        if (d1 < d0) ko = rest[1];
    }
    double r_in[3], v_in[3], r_out[3], v_out[3];
    for (int c = 0; c < 3; c++) {
        r_in[c] = st.pos[bj*3+c] - st.pos[bi*3+c]; v_in[c] = st.vel[bj*3+c] - st.vel[bi*3+c];
        r_out[c] = st.pos[ko*3+c] - xc[c]; v_out[c] = st.vel[ko*3+c] - vc[c];
    }
    const double mu_o = G * (Mi + st.mass[ko]);
    const double ro = std::sqrt(r_out[0]*r_out[0] + r_out[1]*r_out[1] + r_out[2]*r_out[2]);
    const double vo2 = v_out[0]*v_out[0] + v_out[1]*v_out[1] + v_out[2]*v_out[2];
    const double eps_o = 0.5 * vo2 - mu_o / ro;
    if (!(eps_o < 0.0)) return 0;   // unbound outer body: no hierarchy
    const double a_o = -mu_o / (2.0 * eps_o);
    const double rvo = r_out[0]*v_out[0] + r_out[1]*v_out[1] + r_out[2]*v_out[2];
    const double e_o = std::sqrt(std::max(0.0, 1.0 + 2.0 * eps_o * (ro * ro * vo2 - rvo * rvo) / (mu_o * mu_o)));
    return SubsystemManager::mardling_stability(a_min, a_o, e_o, m1, m2, st.mass[ko],
                                                 SubsystemManager::mutual_inclination(r_in, v_in, r_out, v_out)) ? 5 : 0;
}

// ============================================================================
// Host arrays and device staging
// ============================================================================

namespace {
// v(i) access to a host vector (the integration loop below indexes the packed arrays that way)
template <class T>
struct VecAcc {
    T* p;
    T& operator()(size_t i) const { return p[i]; }
};
template <class T>
VecAcc<T> acc(std::vector<T>& v) { return VecAcc<T>{v.data()}; }

// Persistent device buffers for the member gather and scatter (no View or host allocation per call)
using Staging = DeviceStaging;
Staging& staging() { return DeviceStaging::get<0>(); }

}  // namespace

// 3x3 tensor sum (NBODY_MSTAR_TIDE tidal tensors)
struct Tide9 {
    double a[9];
    KOKKOS_INLINE_FUNCTION Tide9() { for (int q = 0; q < 9; q++) a[q] = 0; }
    KOKKOS_INLINE_FUNCTION Tide9& operator+=(const Tide9& o) { for (int q = 0; q < 9; q++) a[q] += o.a[q]; return *this; }
};
namespace Kokkos {
template <> struct reduction_identity<Tide9> {
    KOKKOS_FORCEINLINE_FUNCTION static Tide9 sum() { return Tide9(); }
};
}  // namespace Kokkos

void SubsystemBatch::allocate(int n_sub, int total_bodies) {
    n_subsystems = n_sub;
    n_bodies.resize(n_sub);
    offset.resize(n_sub);
    dt_target.resize(n_sub);
    flat_pos.resize((size_t)total_bodies * 3);
    flat_vel.resize((size_t)total_bodies * 3);
    flat_mass.resize(total_bodies);
    flat_pos_out.resize((size_t)total_bodies * 3);
    flat_vel_out.resize((size_t)total_bodies * 3);
    energy_error.resize(n_sub);
    converged.resize(n_sub);
}

// ============================================================================
// Pack subsystems into the host arrays
// ============================================================================

void SubsystemBatch::pack(const SubsystemManager& mgr, const Particles& p,
                           double G, double dt, const HostSource* src) {
    (void)G;
    int n_sub = (int)mgr.active.size();
    if (n_sub == 0) return;

    // Count total bodies
    int total = 0;
    for (const auto& sub : mgr.active) total += sub.n_bodies;

    allocate(n_sub, total);

    // Use cached pid->idx from SubsystemManager
    const auto& pid_to_idx = mgr.pid_to_idx();

    // Unique member slots (first appearance) and each body's row among them; cached for unpack()
    cached_member_indices.clear();
    body_member.resize(total);
    if (src) {   // every member must lie in the host copy's range, else gather from the device
        for (const auto& sub : mgr.active)
            for (int b = 0; b < sub.n_bodies && src; b++) {
                int idx = pid_to_idx[sub.particle_ids[b]];
                if (idx < src->start || idx >= src->start + src->count) src = nullptr;
            }
    }
    if (src) {
        std::vector<int> row_of(src->count, -1);
        int k = 0;
        for (const auto& sub : mgr.active)
            for (int b = 0; b < sub.n_bodies; b++, k++) {
                int idx = pid_to_idx[sub.particle_ids[b]];
                int& r = row_of[idx - src->start];
                if (r < 0) { r = (int)cached_member_indices.size(); cached_member_indices.push_back(idx); }
                body_member[k] = r;
            }
    } else {
        std::unordered_map<int, int> row_of;
        int k = 0;
        for (const auto& sub : mgr.active)
            for (int b = 0; b < sub.n_bodies; b++, k++) {
                int idx = pid_to_idx[sub.particle_ids[b]];
                auto it = row_of.emplace(idx, (int)cached_member_indices.size());
                if (it.second) cached_member_indices.push_back(idx);
                body_member[k] = it.first->second;
            }
    }
    const auto& member_indices = cached_member_indices;
    int n_mem = (int)member_indices.size();

    // Device gather of the members: everything without a host copy; with one, only spins and stellar types, and
    // those only when the spin couplings evolve the spins (otherwise LogH carries them unchanged and they are
    // not written back)
    // Buffer layout per member: x,y,z,vx,vy,vz,mass,spin_x,spin_y,spin_z,stype_as_double,radius (12 doubles)
    constexpr int PACK_STRIDE = 12;
    const bool want_spins = (g_collisional.pn_order & PN_SPIN) != 0;
    // In-integration collisions (and SDAR) need the radii and types of all members: from the host cache by particle
    // id (stellar_cache.h; a device gather per call cost ~40 us), from the gathered buffer when there is one
    const bool gather = !src || want_spins;
    StellarPropsCache& props = StellarPropsCache::get();
    if (!gather) {
        props.ensure(p);
        bool all = true;
        for (const auto& sub : mgr.active)
            for (int b = 0; b < sub.n_bodies; b++) all = all && props.has(sub.particle_ids[b]);
        if (!all) { props.invalidate(); props.ensure(p); }
    }
    has_stored_spins.assign(n_sub, 0);
    stored_spins.resize(n_sub * 6, 0.0);
    stored_stellar_type.assign(n_sub * 2, 0);
    Staging& st = staging();
    if (gather) {
        double t_x0 = wall_now();
        st.reserve(n_mem, PACK_STRIDE);
        for (int i = 0; i < n_mem; i++) st.h_idx(i) = member_indices[i];
        dcopy(Kokkos::subview(st.d_idx, Kokkos::make_pair(0, n_mem)),
                          Kokkos::subview(st.h_idx, Kokkos::make_pair(0, n_mem)));
        auto d_idx = raw(st.d_idx);
        auto d_buf = raw(st.d_buf);
        auto px = raw(p.x); auto py = raw(p.y); auto pz = raw(p.z);
        auto pvx = raw(p.vx); auto pvy = raw(p.vy); auto pvz = raw(p.vz);
        auto pm = raw(p.mass); auto pspin = raw(p.spin_vec); auto pstype = raw(p.stellar_type);
        auto prad = raw(p.stellar_radius);
        small_for("pack_gather", n_mem, KOKKOS_LAMBDA(int i) {
            int k = d_idx(i);
            int base = i * PACK_STRIDE;
            d_buf(base + 0) = px(k);  d_buf(base + 1) = py(k);  d_buf(base + 2) = pz(k);
            d_buf(base + 3) = pvx(k); d_buf(base + 4) = pvy(k); d_buf(base + 5) = pvz(k);
            d_buf(base + 6) = pm(k);
            d_buf(base + 7) = pspin(k, 0); d_buf(base + 8) = pspin(k, 1); d_buf(base + 9) = pspin(k, 2);
            d_buf(base + 10) = static_cast<double>(pstype(k));
            d_buf(base + 11) = prad(k);
        });
        auto rng = Kokkos::make_pair((size_t)0, (size_t)n_mem * PACK_STRIDE);
        dcopy(Kokkos::subview(st.h_buf, rng), Kokkos::subview(st.d_buf, rng));
        g_logh_prof.pack_xfer += wall_now() - t_x0;
    }
    auto h_buf = st.h_buf;

    // Copy persistent auxiliary velocities and normalization scales from subsystems
    has_stored_aux_vel.resize(n_sub);
    // Compute flat offsets for aux_vel
    stored_aux_vel_offsets.resize(n_sub);
    {
        int aux_off = 0;
        for (int s = 0; s < n_sub; s++) {
            stored_aux_vel_offsets[s] = aux_off;
            aux_off += mgr.active[s].n_bodies;
        }
        stored_aux_vel.resize(aux_off * 3);
    }
    stored_V_scale.resize(n_sub);
    stored_R_scale.resize(n_sub);
    stored_T_scale.resize(n_sub);
    stored_E_scale.resize(n_sub);
    stored_scales_set.resize(n_sub);
    ext_tidal.resize(n_sub);
    for (int s = 0; s < n_sub; s++) {
        ext_tidal[s] = mgr.active[s].ext_tidal;
        has_stored_aux_vel[s] = mgr.active[s].has_aux_vel;
        if (mgr.active[s].has_aux_vel) {
            int av_off = stored_aux_vel_offsets[s] * 3;
            std::memcpy(&stored_aux_vel[av_off], mgr.active[s].aux_vel_phys,
                        mgr.active[s].n_bodies * 3 * sizeof(double));
        }
        stored_V_scale[s] = mgr.active[s].V_scale;
        stored_R_scale[s] = mgr.active[s].R_scale;
        stored_T_scale[s] = mgr.active[s].T_scale;
        stored_E_scale[s] = mgr.active[s].E_scale;
        stored_scales_set[s] = mgr.active[s].scales_set;
    }

    // Spin vectors and stellar types of 2-body subsystems from the gathered buffer
    int off = 0;
    for (int s = 0; s < n_sub; s++) {
        const auto& sub = mgr.active[s];
        if (!gather && sub.n_bodies == 2) {
            stored_stellar_type[s*2+0] = props.type_of(sub.particle_ids[0]);
            stored_stellar_type[s*2+1] = props.type_of(sub.particle_ids[1]);
        } else if (gather && sub.n_bodies == 2) {
            int li0 = body_member[off], li1 = body_member[off + 1];
            stored_stellar_type[s*2+0] = static_cast<int>(h_buf(li0 * PACK_STRIDE + 10));
            stored_stellar_type[s*2+1] = static_cast<int>(h_buf(li1 * PACK_STRIDE + 10));
            bool bh0 = (stored_stellar_type[s*2+0] == 14);
            bool bh1 = (stored_stellar_type[s*2+1] == 14);
            if (bh0 || bh1) {
                has_stored_spins[s] = true;
                for (int c = 0; c < 3; c++) {
                    stored_spins[s*6+c]     = h_buf(li0 * PACK_STRIDE + 7 + c);
                    stored_spins[s*6+3+c]   = h_buf(li1 * PACK_STRIDE + 7 + c);
                }
            }
        }
        off += sub.n_bodies;
    }

    // Spin vectors of all members (MSTAR subsystems evolve the spins of their PN members)
    has_flat_spin = gather && want_spins;
    flat_spin.assign((size_t)total * 3, 0.0);
    if (has_flat_spin)
        for (int k = 0; k < total; k++)
            for (int c = 0; c < 3; c++) flat_spin[k * 3 + c] = h_buf(body_member[k] * PACK_STRIDE + 7 + c);

    // Stellar radii and types of all members (SDAR collision candidates)
    flat_radius.assign(total, 0.0);
    flat_stype.assign(total, 0);
    flat_pid.assign(total, -1);
    {
        int k = 0;
        for (const auto& sub : mgr.active)
            for (int b = 0; b < sub.n_bodies; b++) flat_pid[k++] = sub.particle_ids[b];
    }
    if (gather) {
        for (int k = 0; k < total; k++) {
            flat_stype[k] = static_cast<int>(h_buf(body_member[k] * PACK_STRIDE + 10));
            flat_radius[k] = h_buf(body_member[k] * PACK_STRIDE + 11);
        }
    } else {
        for (int k = 0; k < total; k++) {
            flat_stype[k] = props.type_of(flat_pid[k]);
            flat_radius[k] = props.radius_of(flat_pid[k]);
        }
        if (StellarPropsCache::check_enabled()) {
            std::vector<int> idx(total);
            for (int k = 0; k < total; k++) idx[k] = member_indices[body_member[k]];
            props.verify(p, idx, flat_pid, "LogH pack");
        }
    }

    off = 0;
    for (int s = 0; s < n_sub; s++) {
        const auto& sub = mgr.active[s];
        n_bodies[s] = sub.n_bodies;
        offset[s] = off;
        dt_target[s] = dt;

        // Pack pos/vel/mass from the host copy or the gathered buffer
        for (int b = 0; b < sub.n_bodies; b++) {
            const int k = off + b;
            if (src) {
                const int i = member_indices[body_member[k]] - src->start;
                flat_pos[k * 3 + 0] = src->x[i];
                flat_pos[k * 3 + 1] = src->y[i];
                flat_pos[k * 3 + 2] = src->z[i];
                flat_vel[k * 3 + 0] = src->vx[i];
                flat_vel[k * 3 + 1] = src->vy[i];
                flat_vel[k * 3 + 2] = src->vz[i];
                flat_mass[k] = src->m[i];
            } else {
                const int base = body_member[k] * PACK_STRIDE;
                for (int c = 0; c < 3; c++) {
                    flat_pos[k * 3 + c] = h_buf(base + c);
                    flat_vel[k * 3 + c] = h_buf(base + 3 + c);
                }
                flat_mass[k] = h_buf(base + 6);
            }
        }
        off += sub.n_bodies;
    }

    // NBODY_MSTAR_TIDE: tidal tensor of every body outside an MSTAR subsystem at its c.o.m. (one device reduction per
    // subsystem over all particles; positions replicated on every rank)
    tide_A.assign((size_t)n_sub * 9, 0.0);
    if (g_collisional.mstar && g_collisional.mstar_tide) {
        static Kokkos::View<char*> mask;
        static bool hook = (Kokkos::push_finalize_hook([] { mask = Kokkos::View<char*>(); }), true);
        (void)hook;
        if ((int)mask.extent(0) < p.N) mask = Kokkos::View<char*>("mstar_tide_mask", p.N);
        auto px = p.x, py = p.y, pz = p.z, pm = p.mass;
        for (int s = 0; s < n_sub; s++) {
            if (!g_collisional.mstar_group(n_bodies[s], LOGH_MAX_BODIES)) continue;
            double M = 0, X[3] = {0, 0, 0};
            std::vector<int> idx(n_bodies[s]);
            for (int b = 0; b < n_bodies[s]; b++) {
                const int k = offset[s] + b;
                idx[b] = member_indices[body_member[k]];
                M += flat_mass[k];
                for (int c = 0; c < 3; c++) X[c] += flat_mass[k] * flat_pos[k * 3 + c];
            }
            if (!(M > 0)) continue;
            for (int c = 0; c < 3; c++) X[c] /= M;
            Kokkos::View<int*, Kokkos::HostSpace> hidx(idx.data(), idx.size());
            auto didx = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace::memory_space(), hidx);
            const int nm = (int)idx.size();
            auto mk = mask;
            Kokkos::parallel_for("mstar_tide_mark", nm, KOKKOS_LAMBDA(int q) { mk(didx(q)) = 1; });
            Tide9 T;
            const double X0 = X[0], X1 = X[1], X2 = X[2];
            Kokkos::parallel_reduce("mstar_tide", p.N, KOKKOS_LAMBDA(int k, Tide9& t) {
                if (mk(k) || !(pm(k) > 0.0)) return;
                const double d0 = px(k) - X0, d1 = py(k) - X1, d2 = pz(k) - X2;
                const double r2 = d0 * d0 + d1 * d1 + d2 * d2;
                const double ir2 = 1.0 / r2, ir3 = ir2 / sqrt(r2), gm = G * pm(k) * ir3;
                const double d[3] = {d0, d1, d2};
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++) t.a[3 * i + j] += gm * (3.0 * d[i] * d[j] * ir2 - (i == j ? 1.0 : 0.0));
            }, Kokkos::Sum<Tide9>(T));
            Kokkos::parallel_for("mstar_tide_unmark", nm, KOKKOS_LAMBDA(int q) { mk(didx(q)) = 0; });
            for (int q = 0; q < 9; q++) tide_A[(size_t)s * 9 + q] = T.a[q];
        }
    }
}

// ============================================================================
// Unpack results back into Particles
// ============================================================================

void SubsystemBatch::unpack(SubsystemManager& mgr, Particles& p, const HostSource* dst) {
    int n_sub = (int)mgr.active.size();
    if (n_sub == 0) return;

    const auto& pid_to_idx = mgr.pid_to_idx();

    // Scatter rows (one per member): x, y, z, vx, vy, vz, mass from the LogH output
    constexpr int UNPACK_STRIDE = 7;
    const auto& member_indices = cached_member_indices;
    int n_mem = (int)member_indices.size();
    double t_x0 = wall_now();
    Staging& st = staging();
    st.reserve(n_mem, UNPACK_STRIDE);
    g_logh_prof.unpack_xfer += wall_now() - t_x0;
    for (int i = 0; i < n_mem; i++) st.h_idx(i) = member_indices[i];
    for (int s = 0; s < n_sub; s++) {
        const auto& sub = mgr.active[s];
        int off = offset[s];
        for (int b = 0; b < sub.n_bodies; b++) {
            const int k = off + b;
            int base = body_member[k] * UNPACK_STRIDE;
            for (int c = 0; c < 3; c++) {
                st.h_buf(base + c) = flat_pos_out[k * 3 + c];
                st.h_buf(base + 3 + c) = flat_vel_out[k * 3 + c];
            }
            st.h_buf(base + 6) = flat_mass[k];
            if (dst) {   // the caller's host copy (positions and velocities)
                const int i = member_indices[body_member[k]] - dst->start;
                dst->x[i] = flat_pos_out[k * 3 + 0];
                dst->y[i] = flat_pos_out[k * 3 + 1];
                dst->z[i] = flat_pos_out[k * 3 + 2];
                dst->vx[i] = flat_vel_out[k * 3 + 0];
                dst->vy[i] = flat_vel_out[k * 3 + 1];
                dst->vz[i] = flat_vel_out[k * 3 + 2];
                if (dst->m) dst->m[i] = flat_mass[k];
            }
        }
    }

    // Upload and scatter: write only modified particles back to device arrays
    t_x0 = wall_now();
    st.upload_idx(n_mem);
    st.upload_buf((size_t)n_mem * UNPACK_STRIDE);
    auto d_idx = raw(st.d_idx);
    auto d_buf = raw(st.d_buf);
    auto px = raw(p.x); auto py = raw(p.y); auto pz = raw(p.z);
    auto pvx = raw(p.vx); auto pvy = raw(p.vy); auto pvz = raw(p.vz);
    auto pcx = raw(p.cvx); auto pcy = raw(p.cvy); auto pcz = raw(p.cvz);
    auto pm = raw(p.mass);
    small_for("unpack_scatter", n_mem, KOKKOS_LAMBDA(int i) {
        int k = d_idx(i);
        int base = i * UNPACK_STRIDE;
        px(k) = d_buf(base + 0); py(k) = d_buf(base + 1); pz(k) = d_buf(base + 2);
        pvx(k) = d_buf(base + 3); pvy(k) = d_buf(base + 4); pvz(k) = d_buf(base + 5);
        pcx(k) = 0.0; pcy(k) = 0.0; pcz(k) = 0.0;  // velocity replaced by the subsystem integrator
        pm(k) = d_buf(base + 6);
    });
    g_logh_prof.unpack_xfer += wall_now() - t_x0;

    // Write persistent auxiliary velocities and scales back to subsystems
    for (int s = 0; s < n_sub; s++) {
        mgr.active[s].has_aux_vel = has_stored_aux_vel[s];
        if (has_stored_aux_vel[s]) {
            int av_off = stored_aux_vel_offsets[s] * 3;
            std::memcpy(mgr.active[s].aux_vel_phys, &stored_aux_vel[av_off],
                        mgr.active[s].n_bodies * 3 * sizeof(double));
        }
        mgr.active[s].V_scale = stored_V_scale[s];
        mgr.active[s].R_scale = stored_R_scale[s];
        mgr.active[s].T_scale = stored_T_scale[s];
        mgr.active[s].E_scale = stored_E_scale[s];
        mgr.active[s].scales_set = stored_scales_set[s];
    }

    // Write evolved spins back to particles via small scatter (only spins gathered by this pack)
    std::vector<int> spin_indices;
    std::vector<double> spin_values;
    for (int s = 0; s < n_sub; s++) {
        if (!has_stored_spins[s] || mgr.active[s].n_bodies != 2) continue;
        const auto& sub = mgr.active[s];
        int idx0 = pid_to_idx[sub.particle_ids[0]];
        int idx1 = pid_to_idx[sub.particle_ids[1]];
        spin_indices.push_back(idx0);
        spin_values.insert(spin_values.end(), {stored_spins[s*6+0], stored_spins[s*6+1], stored_spins[s*6+2]});
        spin_indices.push_back(idx1);
        spin_values.insert(spin_values.end(), {stored_spins[s*6+3], stored_spins[s*6+4], stored_spins[s*6+5]});
    }
    for (int s = 0; s < n_sub && has_flat_spin; s++) {   // MSTAR subsystems: their PN members' spins
        if (s >= (int)mstar_used.size() || !mstar_used[s]) continue;
        const auto& sub = mgr.active[s];
        for (int b = 0; b < sub.n_bodies; b++) {
            const int k = offset[s] + b;
            if (flat_stype[k] != 14) continue;
            spin_indices.push_back(pid_to_idx[sub.particle_ids[b]]);
            spin_values.insert(spin_values.end(), {flat_spin[k * 3], flat_spin[k * 3 + 1], flat_spin[k * 3 + 2]});
        }
    }
    if (!spin_indices.empty()) {
        t_x0 = wall_now();
        int n_spin = (int)spin_indices.size();
        st.reserve(n_spin, 3);
        for (int i = 0; i < n_spin; i++) st.h_idx(i) = spin_indices[i];
        std::memcpy(st.h_buf.data(), spin_values.data(), n_spin * 3 * sizeof(double));
        st.upload_idx(n_spin);
        st.upload_buf((size_t)n_spin * 3);
        auto d_si = raw(st.d_idx);
        auto d_sv = raw(st.d_buf);
        auto pspin = raw(p.spin_vec);
        small_for("spin_scatter", n_spin, KOKKOS_LAMBDA(int i) {
            int k = d_si(i);
            pspin(k, 0) = d_sv(i * 3 + 0);
            pspin(k, 1) = d_sv(i * 3 + 1);
            pspin(k, 2) = d_sv(i * 3 + 2);
        });
        g_logh_prof.unpack_xfer += wall_now() - t_x0;
    }
}

// ============================================================================
// GPU-batched integration
// For now: host-side per-subsystem GBS (GPU kernel version requires
// GBS entirely in device code, which we'll implement after basic correctness)
//
// IMPORTANT: The TTL leapfrog equations (Rantala et al. 2023 convention:
// dv/ds = a/Ω, dx/ds = v/Ω, dt/ds = 1/Ω) are only dimensionally
// consistent when the energy unit ≈ 1. In Hénon units (G=M=1) this
// holds naturally. In code units (pc, M_sun, km/s), Ω ~ 10^7.
//
// Fix: normalize each subsystem to internal N-body units (G'=1, M'=1)
// before integration, then denormalize afterward.
// CRITICAL: normalization scales are computed ONCE on the first LogH
// call and REUSED on all subsequent calls (stored in Subsystem struct).
// This prevents persistent B from drifting due to E_scale changes
// when FSI kicks modify subsystem member velocities between calls.
// ============================================================================

// ============================================================================
// MSTAR subsystems (g_collisional.mstar_group; docs/MSTAR.md): one group through the MSTAR dispatch interface
// (mstar_dispatch.h; the GPLv3 module src/mstar implements it). Same normalisation as LogH (c.o.m. frame, fixed
// scales, G = M = 1). Collisions are decided inside the call (the module checks every accepted step: separation or
// osculating pericentre inside collision_distance's rule); the pair merges here as in the LogH path (c.o.m., momentum
// exact, the change of T + U booked as dE_int) and the call continues for the rest of dt. Energy: the module's PN
// work ledger gives the change of the Newtonian T + U by the PN forces (pn_dE); everything else the members' T + U
// does besides collisions is integration error, left unbooked (visible in |dE/E|) and reported as the energy check.
// ============================================================================
static void mstar_integrate_one(SubsystemBatch& batch, int s, double G, double eta_gbs, double& dE_out, long& steps_out,
                                bool& converged_out) {
    const int nb = batch.n_bodies[s], off = batch.offset[s];
    const double dt = batch.dt_target[s];
    double* hpos = batch.flat_pos.data();
    double* hvel = batch.flat_vel.data();
    double* hmass = batch.flat_mass.data();
    double* hpos_out = batch.flat_pos_out.data();
    double* hvel_out = batch.flat_vel_out.data();
    double M_total = 0, cm[3] = {0, 0, 0}, cmv[3] = {0, 0, 0};
    for (int b = 0; b < nb; b++) {
        const double mb = hmass[off + b];
        M_total += mb;
        for (int c = 0; c < 3; c++) { cm[c] += mb * hpos[(off + b) * 3 + c]; cmv[c] += mb * hvel[(off + b) * 3 + c]; }
    }
    for (int c = 0; c < 3; c++) { cm[c] /= M_total; cmv[c] /= M_total; }
    if (!batch.stored_scales_set[s]) {   // first call: scales from the c.o.m.-frame kinetic energy (as LogH)
        double T_cm = 0;
        for (int b = 0; b < nb; b++) {
            double v2 = 0;
            for (int c = 0; c < 3; c++) { const double u = hvel[(off + b) * 3 + c] - cmv[c]; v2 += u * u; }
            T_cm += 0.5 * hmass[off + b] * v2;
        }
        double V = std::sqrt(2.0 * T_cm / M_total);
        if (V < 1e-30) V = 1.0;
        batch.stored_V_scale[s] = V;
        batch.stored_R_scale[s] = G * M_total / (V * V);
        batch.stored_T_scale[s] = batch.stored_R_scale[s] / V;
        batch.stored_E_scale[s] = M_total * V * V;
        batch.stored_scales_set[s] = true;
    }
    const double V_scale = batch.stored_V_scale[s], R_scale = batch.stored_R_scale[s];
    const double T_scale = batch.stored_T_scale[s], E_scale = batch.stored_E_scale[s];
    batch.has_stored_aux_vel[s] = false;   // MSTAR restarts its auxiliary velocities every step (KETJU)

    const double c_code = g_collisional.c_code;
    const double c_norm = c_code > 0.0 ? c_code / V_scale : 0.0;
    const bool pn_on = g_collisional.pn_order > 0 && c_code > 0.0;
    constexpr double R_sun_to_pc = 2.254e-8;   // as mergers.cpp

    // Live members (massless: placeholders of earlier collisions), PN bodies first (KETJU's layout)
    std::vector<int> order;
    order.reserve(nb);
    int n_pn = 0;
    for (int pass = 0; pass < 2; pass++)
        for (int b = 0; b < nb; b++) {
            const double mb = hmass[off + b];
            if (!(mb > 0.0)) continue;
            const bool pn = pn_on && g_collisional.pn_body(mb, batch.flat_stype[off + b]);
            if (pn == (pass == 0)) { order.push_back(b); if (pn) n_pn++; }
        }
    int n = (int)order.size();
    std::vector<double> m(n), x(3 * n), v(3 * n), S(3 * std::max(n_pn, 1), 0.0), R(n);
    std::vector<char> bh(n);
    std::vector<int> kw(n), idx(n);
    for (int k = 0; k < n; k++) {
        const int b = order[k];
        m[k] = hmass[off + b] / M_total;
        for (int c = 0; c < 3; c++) {
            x[3 * k + c] = (hpos[(off + b) * 3 + c] - cm[c]) / R_scale;
            v[3 * k + c] = (hvel[(off + b) * 3 + c] - cmv[c]) / V_scale;
        }
        kw[k] = batch.flat_stype[off + b];
        bh[k] = kw[k] == 14;
        R[k] = (kw[k] == 14 ? 0.0 : batch.flat_radius[off + b] * R_sun_to_pc) / R_scale;
        idx[k] = k;
        // spin angular momentum S = chi G m^2 / c (normalised units) of the PN bodies
        if (k < n_pn && batch.has_flat_spin && c_norm > 0.0)
            for (int c = 0; c < 3; c++) S[3 * k + c] = batch.flat_spin[(off + b) * 3 + c] * m[k] * m[k] / c_norm;
    }
    unsigned flags = 0;
    if (pn_on && n_pn >= 2) {
        const int po = g_collisional.pn_order;
        if (po & 1) flags |= 1u | 2u;      // 1PN with the N-body (three-body) terms
        if (po & 2) flags |= 4u;           // 2PN
        if (po & 4) flags |= 8u;           // 2.5PN
        if (po & 8) flags |= 16u;          // 3PN
        if (po & 16) flags |= 32u;         // 3.5PN
        if (po & PN_SPIN) flags |= 256u | 512u | 1024u;
    }
    auto energy = [&]() {   // Newtonian T + U of the state (normalised)
        double T = 0, U = 0;
        for (int i = 0; i < n; i++) {
            T += 0.5 * m[i] * (v[3*i] * v[3*i] + v[3*i+1] * v[3*i+1] + v[3*i+2] * v[3*i+2]);
            for (int j = i + 1; j < n; j++) {
                double r2 = 0;
                for (int c = 0; c < 3; c++) { const double d = x[3*i+c] - x[3*j+c]; r2 += d * d; }
                U -= m[i] * m[j] / std::sqrt(r2);
            }
        }
        return T + U;
    };
    const double E0 = energy();
    double E_ref = E0, pn_dE = 0.0, ext_dE = 0.0, t = 0.0;
    const bool tide = g_collisional.mstar_tide && (size_t)s * 9 + 8 < batch.tide_A.size();
    const double dt_norm = dt / T_scale;
    static const long max_steps = std::getenv("NBODY_MSTAR_MAX_STEPS") ? std::atol(std::getenv("NBODY_MSTAR_MAX_STEPS"))
                                                                        : 2000000;
    std::vector<int> ghost_of(nb, -1);
    std::vector<double> ghost_d(3 * nb, 0.0);
    auto& colls = batch.collisions[s];
    long steps = 0;
    bool reached = false, used_dev = false;
    while (n >= 2) {
        MstarGroup g;
        g.n = n; g.n_pn = n_pn;
        g.mass = m.data(); g.pos = x.data(); g.vel = v.data(); g.spin = S.data(); g.index = idx.data();
        g.dt = dt_norm - t;
        g.G = 1.0;
        g.c = flags ? c_norm : 0.0;
        g.pn_flags = flags;
        g.tol = eta_gbs;
        g.max_steps = max_steps;
        g.device = -1;   // device from mstar_crossover_n() bodies (NBODY_MSTAR_CROSSOVER), host engine below
        if (tide) {      // a = A x about the c.o.m.: normalised A' = A T_scale^2
            g.tide = true;
            for (int q = 0; q < 9; q++) g.tide_A[q] = batch.tide_A[(size_t)s * 9 + q] * T_scale * T_scale;
        }
        if (c_code > 0.0) {   // collisions as in the LogH path (coll_on: c_code > 0)
            g.coll_R = R.data();
            g.coll_bh = bh.data();
            g.coll_c = c_norm;
            g.coll_bh_max_single = g_collisional.bh_merge_max_single;
        }
        mstar_integrate(g);
        used_dev = used_dev || g.used_device;
        steps += g.steps + g.failed_steps;
        t += g.t_reached;
        pn_dE += g.pn_dE;
        ext_dE += g.ext_dE;
        if (g.status == 0 && g.ok) { reached = true; break; }
        if (g.status != 1) break;   // step limit
        // --- collision of state bodies ci < cj: merge into the compact member (BH) or the heavier, at the c.o.m.
        const int ci = g.coll_i, cj = g.coll_j;
        const bool ci_keeps = (kw[ci] >= 10) != (kw[cj] >= 10) ? kw[ci] >= 10 : m[ci] >= m[cj];
        const int kk = ci_keeps ? ci : cj, kg = ci_keeps ? cj : ci;
        const double mk = m[kk], mg = m[kg], mm = mk + mg;
        double xc[3], vc[3];
        for (int c = 0; c < 3; c++) {
            xc[c] = (mk * x[3*kk+c] + mg * x[3*kg+c]) / mm;
            vc[c] = (mk * v[3*kk+c] + mg * v[3*kg+c]) / mm;
        }
        SubsystemCollision rec;
        rec.type = g.coll_type == 1 ? MergerType::STELLAR_COLLISION
                 : g.coll_type == 2 ? MergerType::TIDAL_DISRUPTION : MergerType::ISCO;
        rec.pid_keep = batch.flat_pid[off + order[kk]];
        rec.pid_gone = batch.flat_pid[off + order[kg]];
        rec.kw_keep = kw[kk]; rec.kw_gone = kw[kg];
        rec.m_keep = mk * M_total; rec.m_gone = mg * M_total;
        for (int c = 0; c < 3; c++) {
            rec.d_keep[c] = (x[3*kk+c] - xc[c]) * R_scale;
            rec.d_gone[c] = (x[3*kg+c] - xc[c]) * R_scale;
            rec.u_keep[c] = (v[3*kk+c] - vc[c]) * V_scale;
            rec.u_gone[c] = (v[3*kg+c] - vc[c]) * V_scale;
        }
        rec.r_sep = g.coll_rsep * R_scale;
        rec.r_coll = g.coll_rcoll * R_scale;
        rec.t_rel = t * T_scale;
        rec.n_members = nb;
        rec.member_pids.assign(&batch.flat_pid[off], &batch.flat_pid[off] + nb);
        const double E_before = energy();
        ghost_of[order[kg]] = order[kk];
        for (int c = 0; c < 3; c++) ghost_d[order[kg] * 3 + c] = x[3*kg+c] - xc[c];
        for (int c = 0; c < 3; c++) { x[3*kk+c] = xc[c]; v[3*kk+c] = vc[c]; }
        m[kk] = mm;
        if (kw[kg] >= 10 || kw[kk] >= 10) kw[kk] = std::max(kw[kk], kw[kg]);
        bh[kk] = kw[kk] == 14;
        R[kk] = bh[kk] ? 0.0 : std::max(R[kk], R[kg]);
        // two PN bodies: the remnant carries the sum of the spins until the core's remnant physics sets it
        if (kk < n_pn && kg < n_pn)
            for (int c = 0; c < 3; c++) S[3*kk+c] += S[3*kg+c];
        const bool kg_pn = kg < n_pn;   // remove kg (the PN bodies stay first)
        auto erase3 = [&](std::vector<double>& a) { a.erase(a.begin() + 3 * kg, a.begin() + 3 * kg + 3); };
        erase3(x); erase3(v);
        m.erase(m.begin() + kg); R.erase(R.begin() + kg); bh.erase(bh.begin() + kg); kw.erase(kw.begin() + kg);
        order.erase(order.begin() + kg); idx.erase(idx.begin() + kg);
        if (kg_pn) { erase3(S); n_pn--; }
        n--;
        if (n_pn < 2) flags = 0;
        const double dE = energy() - E_before;
        rec.dE_int = dE * E_scale;
        batch.coll_dE[s] += rec.dE_int;
        colls.push_back(rec);
        E_ref += dE;
    }
    if (n < 2 && !reached) {   // one body left (from the start or after a collision): free drift to the end
        for (int c = 0; c < 3 * n; c++) x[c] += v[c] * (dt_norm - t);
        t = dt_norm;
        reached = true;
    }
    // Write back (c.o.m. drifts uniformly), ghosts at their collision offsets from the remnant, placeholders co-move
    const double cmf[3] = {cm[0] + cmv[0] * dt, cm[1] + cmv[1] * dt, cm[2] + cmv[2] * dt};
    std::vector<int> slot(nb, -1);
    for (int k = 0; k < n; k++) slot[order[k]] = k;
    for (int b = 0; b < nb; b++) {
        if (slot[b] >= 0) {
            const int k = slot[b];
            for (int c = 0; c < 3; c++) {
                hpos_out[(off + b) * 3 + c] = x[3*k+c] * R_scale + cmf[c];
                hvel_out[(off + b) * 3 + c] = v[3*k+c] * V_scale + cmv[c];
            }
            hmass[off + b] = m[k] * M_total;
            if (k < n_pn && batch.has_flat_spin && c_norm > 0.0 && m[k] > 0.0)
                for (int c = 0; c < 3; c++) batch.flat_spin[(off + b) * 3 + c] = S[3*k+c] * c_norm / (m[k] * m[k]);
        } else if (ghost_of[b] >= 0) {
            int r = ghost_of[b];
            while (slot[r] < 0 && ghost_of[r] >= 0) r = ghost_of[r];
            const int k = slot[r];
            for (int c = 0; c < 3; c++) {
                hpos_out[(off + b) * 3 + c] = (x[3*k+c] + ghost_d[b * 3 + c]) * R_scale + cmf[c];
                hvel_out[(off + b) * 3 + c] = v[3*k+c] * V_scale + cmv[c];
            }
            hmass[off + b] = 0.0;
        } else {
            for (int c = 0; c < 3; c++) {
                hpos_out[(off + b) * 3 + c] = hpos[(off + b) * 3 + c] + hvel[(off + b) * 3 + c] * dt;
                hvel_out[(off + b) * 3 + c] = hvel[(off + b) * 3 + c];
            }
            hmass[off + b] = 0.0;
        }
    }
    // Energy check: the integration error (T + U change minus collisions minus PN work) over |E0|
    const double err = std::fabs(energy() - E_ref - pn_dE - ext_dE) / (std::fabs(E0) + 1e-300);
    dE_out = reached ? err : 1e30;
    converged_out = reached;
    steps_out = steps;
    batch.pn_dE[s] = pn_dE * E_scale;
    batch.ext_dE[s] = ext_dE * E_scale;
    batch.t_reached[s] = t * T_scale;
    batch.mstar_used[s] = 1;
    batch.stat_mstar++;
    batch.stat_mstar_dev += used_dev ? 1 : 0;
    batch.stat_mstar_gbs += steps;
    batch.stat_mstar_nmax = std::max(batch.stat_mstar_nmax, nb);
    if (!reached)
        std::printf("      MSTAR step limit %ld: n=%d t=%.3e of %.3e (normalised)\n", steps, nb, t, dt_norm);
}

void logh_integrate_batch_gpu(SubsystemBatch& batch, double G, double eta_gbs) {
    int n_sub = batch.n_subsystems;
    if (n_sub == 0) return;

    // The packed state is host memory (pack() and unpack() move it from and to the particles)
    auto h_nbod = acc(batch.n_bodies);
    auto h_off  = acc(batch.offset);
    auto h_dt   = acc(batch.dt_target);
    auto h_pos  = acc(batch.flat_pos);
    auto h_vel  = acc(batch.flat_vel);
    auto h_mass = acc(batch.flat_mass);
    auto h_pos_out = acc(batch.flat_pos_out);
    auto h_vel_out = acc(batch.flat_vel_out);
    auto h_eerr = acc(batch.energy_error);
    auto h_conv = acc(batch.converged);
    std::vector<double> t_sub(n_sub, 0.0);   // wall time per subsystem (profile)

    int total_gbs_steps = 0;
    std::vector<int> gbs_per_sub(n_sub, 0);   // per-depth load-balance statistics
    batch.t_reached.assign(n_sub, 0.0);   // read by the subsystem-advance check
    double max_dE = 0.0;
    double min_dE = 1e30;
    int cap_hit = 0;
    batch.sd_jump.assign(n_sub, 0.0);
    std::vector<double> sd_kappa_sub(n_sub, 0.0);
    std::vector<int> sec_used_sub(n_sub, 0);   // secular inner binaries collapsed per integration
    batch.collisions.assign(n_sub, {});
    batch.coll_dE.assign(n_sub, 0.0);
    std::vector<int> peri_land(n_sub, 0), peri_miss(n_sub, 0);
    std::vector<int> irk_sub(n_sub, 0);   // 1: Gauss-IRK, 2: Gauss-IRK fell back to LogH+GBS

    // Subsystems are independent: integrate them on all host threads (one GBS integration is long
    // serial work; dynamic scheduling balances binaries against multiplets)
    using HostPolicy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>;
    // Up to NBODY_GBS_STAGE_PAR subsystems (default 2; 0: never) are integrated one after the other with the GBS
    // stages of each step on the host threads instead (Rantala et al. 2020, Sec. 4): at deep levels a batch is
    // one or two subsystems and the other threads would idle. Results are the same either way
    static const int stage_par_max = std::getenv("NBODY_GBS_STAGE_PAR") ? std::atoi(std::getenv("NBODY_GBS_STAGE_PAR"))
                                                                         : 2;
    const int host_threads = Kokkos::DefaultHostExecutionSpace().concurrency();
    const bool serial_subs = n_sub == 1 || n_sub <= stage_par_max;
    // MSTAR subsystems first, one after the other on this thread (device kernels are not launched from inside the
    // host-parallel loop below)
    batch.mstar_used.assign(n_sub, 0);
    batch.pn_dE.assign(n_sub, 0.0);
    batch.ext_dE.assign(n_sub, 0.0);
    batch.stat_mstar = batch.stat_mstar_dev = batch.stat_mstar_nmax = 0;
    batch.stat_mstar_gbs = 0;
    std::vector<char> is_mstar(n_sub, 0);
    double mstar_max_dE = 0.0, mstar_min_dE = 1e30;
    int mstar_caps = 0;
    for (int s = 0; s < n_sub; s++) {
        if (!g_collisional.mstar_group(h_nbod(s), LOGH_MAX_BODIES)) continue;
        is_mstar[s] = 1;
        const double t_s0 = wall_now();
        double dE = 0;
        long steps = 0;
        bool conv = false;
        mstar_integrate_one(batch, s, G, eta_gbs, dE, steps, conv);
        h_eerr(s) = dE;
        h_conv(s) = conv ? 1 : 0;
        gbs_per_sub[s] = (int)std::min<long>(steps, std::numeric_limits<int>::max());
        mstar_max_dE = std::max(mstar_max_dE, dE);
        mstar_min_dE = std::min(mstar_min_dE, dE);
        if (!conv) mstar_caps++;
        t_sub[s] = wall_now() - t_s0;
    }
    auto integrate_one = [&](int s, int& gbs_sum, double& dE_max, double& dE_min, int& caps) {
        if (is_mstar[s]) return;
        const double t_s0 = wall_now();
        int nb = h_nbod(s);
        int off = h_off(s);
        double dt = h_dt(s);

        // ==================================================================
        // Step 1: Compute CM and use fixed normalization scales.
        // Scales are computed once on first LogH call and reused thereafter.
        // This prevents persistent B from drifting due to E_scale changes.
        // ==================================================================
        double M_total = 0;
        double cm_x = 0, cm_y = 0, cm_z = 0;
        double cm_vx = 0, cm_vy = 0, cm_vz = 0;
        for (int b = 0; b < nb; b++) {
            double m = h_mass(off + b);
            M_total += m;
            cm_x += m * h_pos((off+b)*3+0);
            cm_y += m * h_pos((off+b)*3+1);
            cm_z += m * h_pos((off+b)*3+2);
            cm_vx += m * h_vel((off+b)*3+0);
            cm_vy += m * h_vel((off+b)*3+1);
            cm_vz += m * h_vel((off+b)*3+2);
        }
        cm_x /= M_total; cm_y /= M_total; cm_z /= M_total;
        cm_vx /= M_total; cm_vy /= M_total; cm_vz /= M_total;

        double V_scale, R_scale, T_scale, E_scale;
        if (batch.stored_scales_set[s]) {
            // Reuse fixed scales from first call
            V_scale = batch.stored_V_scale[s];
            R_scale = batch.stored_R_scale[s];
            T_scale = batch.stored_T_scale[s];
            E_scale = batch.stored_E_scale[s];
        } else {
            // First call: compute normalization scales from current CM-frame KE
            double T_cm = 0;
            for (int b = 0; b < nb; b++) {
                double dvx = h_vel((off+b)*3+0) - cm_vx;
                double dvy = h_vel((off+b)*3+1) - cm_vy;
                double dvz = h_vel((off+b)*3+2) - cm_vz;
                T_cm += 0.5 * h_mass(off+b) * (dvx*dvx + dvy*dvy + dvz*dvz);
            }
            double V_scale_sq = 2.0 * T_cm / M_total;
            V_scale = std::sqrt(V_scale_sq);
            if (V_scale < 1e-30) V_scale = 1.0;
            V_scale_sq = V_scale * V_scale;
            R_scale = G * M_total / V_scale_sq;
            T_scale = R_scale / V_scale;
            E_scale = M_total * V_scale_sq;
            // Store for all future calls
            batch.stored_V_scale[s] = V_scale;
            batch.stored_R_scale[s] = R_scale;
            batch.stored_T_scale[s] = T_scale;
            batch.stored_E_scale[s] = E_scale;
            batch.stored_scales_set[s] = true;
        }
        double G_norm = 1.0;  // by construction: G*M/(R*V²) = 1

        // ==================================================================
        // Step 2: Build LogHState in normalized units (CM-centered)
        // ==================================================================
        LogHState state{};
        state.t = 0.0;

        double c_code = g_collisional.c_code;
        int pn_order = g_collisional.pn_order;
        bool any_compact = false;

        // Bodies in the integration: live[k] is the member of state body k. A massless member is the placeholder of
        // a body merged inside an earlier integration (finalize_subsystem_collision); it only co-moves
        int live[LOGH_MAX_BODIES], n_live = 0;
        for (int b = 0; b < nb; b++)
            if (h_mass(off + b) > 0.0) live[n_live++] = b;
        state.n_bodies = n_live;
        for (int k = 0; k < n_live; k++) {
            const int b = live[k];
            // Positions: subtract CM, normalize by R_scale
            state.pos[k*3+0] = (h_pos((off+b)*3+0) - cm_x) / R_scale;
            state.pos[k*3+1] = (h_pos((off+b)*3+1) - cm_y) / R_scale;
            state.pos[k*3+2] = (h_pos((off+b)*3+2) - cm_z) / R_scale;
            // Velocities: subtract CM, normalize by V_scale
            state.vel[k*3+0] = (h_vel((off+b)*3+0) - cm_vx) / V_scale;
            state.vel[k*3+1] = (h_vel((off+b)*3+1) - cm_vy) / V_scale;
            state.vel[k*3+2] = (h_vel((off+b)*3+2) - cm_vz) / V_scale;
            // Masses: normalize to sum = 1
            state.mass[k] = h_mass(off+b) / M_total;
            if (g_collisional.pn_body(h_mass(off + b), batch.flat_stype[off + b])) any_compact = true;
        }

        // Normalized PN speed of light
        double c_norm = c_code / V_scale;

        // (omega_before debug removed — stats tracked via batch.stat_* fields)

        // Enable auxiliary velocity for PN when compact objects present.
        // Restore persistent aux_vel if available (H&M extended phase space
        // requires w ≠ v after first step; re-initializing w = v each call
        // discards this and causes O(v²/c²) energy errors).
        if (any_compact && pn_order > 0 && c_code > 0.0) {
            state.use_aux_vel = true;
            if (batch.has_stored_aux_vel[s]) {
                // Restore from previous call: same normalization as vel
                int av_off = batch.stored_aux_vel_offsets[s] * 3;
                for (int k = 0; k < n_live; k++) {
                    const int b = live[k];
                    state.aux_vel[k*3+0] = (batch.stored_aux_vel[av_off + b*3+0] - cm_vx) / V_scale;
                    state.aux_vel[k*3+1] = (batch.stored_aux_vel[av_off + b*3+1] - cm_vy) / V_scale;
                    state.aux_vel[k*3+2] = (batch.stored_aux_vel[av_off + b*3+2] - cm_vz) / V_scale;
                }
            } else {
                // First call: initialize w = v
                std::memcpy(state.aux_vel, state.vel, n_live * 3 * sizeof(double));
            }
        } else {
            state.use_aux_vel = false;
            pn_order = 0;
        }

        // Load spin vectors for 2-body BH binaries; spin couplings (PN_ORDER bit 32, Rantala et al. 2023, sec. 3.7) need the
        // PN path (state.use_aux_vel), otherwise the spins are carried unchanged
        state.has_spins = false;
        state.c_code = c_norm;
        if (!g_collisional.pn_frame_com) {   // PN_FRAME = lab
            state.v_frame[0] = cm_vx / V_scale;
            state.v_frame[1] = cm_vy / V_scale;
            state.v_frame[2] = cm_vz / V_scale;
        }
        if (nb == 2 && n_live == 2 && batch.has_stored_spins[s] && state.use_aux_vel && (pn_order & PN_SPIN)) {
            // Spins are dimensionless chi vectors — no normalization needed
            state.has_spins = true;
            state.spin1[0] = batch.stored_spins[s*6+0];
            state.spin1[1] = batch.stored_spins[s*6+1];
            state.spin1[2] = batch.stored_spins[s*6+2];
            state.spin2[0] = batch.stored_spins[s*6+3];
            state.spin2[1] = batch.stored_spins[s*6+4];
            state.spin2[2] = batch.stored_spins[s*6+5];
            std::memcpy(state.aux_spin1, state.spin1, sizeof(state.spin1));   // auxiliary spins Z = S
            std::memcpy(state.aux_spin2, state.spin2, sizeof(state.spin2));
        }

        // Always recompute B = -(T+U) from current state.
        // Between LogH calls, FSI kicks change particle velocities (T changes)
        // while a persisted B would reflect the old state, driving Omega = T+B
        // negative.  Recomputing B fresh gives Omega = -U > 0 always.
        // All information about previous PN evolution is already encoded in the
        // current positions and velocities, so nothing is lost.
        double T_norm = logh_kinetic_energy(state);
        double U_norm = logh_potential_energy(state, G_norm);
        state.B = -U_norm - T_norm;

        double E0_norm = T_norm + U_norm;

        // Save initial state for PN energy comparison after integration and for the diagnostics
        double state_init_pos[LOGH_MAX_BODIES * 3];
        double state_init_vel[LOGH_MAX_BODIES * 3];
        std::memcpy(state_init_pos, state.pos, n_live * 3 * sizeof(double));
        std::memcpy(state_init_vel, state.vel, n_live * 3 * sizeof(double));
        double mass_init[LOGH_MAX_BODIES];
        std::memcpy(mass_init, state.mass, n_live * sizeof(double));
        const int n0 = n_live;

        // Normalized time target
        double dt_norm = dt / T_scale;

        // ==================================================================
        // Step 3: Integrate in normalized units via GBS
        // ==================================================================
        // Collision distances of the live pairs (FROST-CLUSTERS I App. B3; collision_distance: R1 + R2, the tidal
        // radius 1.3 (M / m)^{1/3} R, ISCO), normalised, and the event each gives; kw / radius per state body (a
        // remnant of a collision in this call keeps the compact type and the larger radius)
        constexpr double R_sun_to_pc = 2.254e-8;   // as mergers.cpp
        const bool coll_on = g_collisional.c_code > 0.0;
        int kw[LOGH_MAX_BODIES];
        double rad_pc[LOGH_MAX_BODIES];
        for (int k = 0; k < n_live; k++) {
            kw[k] = batch.flat_stype[off + live[k]];
            rad_pc[k] = batch.flat_radius[off + live[k]] * R_sun_to_pc;
        }
        double coll[LOGH_MAX_BODIES * LOGH_MAX_BODIES] = {};
        MergerType ctype[LOGH_MAX_BODIES * LOGH_MAX_BODIES];
        auto build_coll = [&]() {
            for (int i = 0; i < state.n_bodies; i++)
                for (int j = i + 1; j < state.n_bodies; j++) {
                    const double r = collision_distance(kw[i], state.mass[i] * M_total, rad_pc[i], kw[j],
                                                        state.mass[j] * M_total, rad_pc[j], G, c_code,
                                                        ctype[i * LOGH_MAX_BODIES + j]);
                    coll[i * LOGH_MAX_BODIES + j] = r / R_scale;
                }
        };
        build_coll();
        double coll0[LOGH_MAX_BODIES * LOGH_MAX_BODIES];
        std::memcpy(coll0, coll, sizeof(coll));

        // Secular inner binaries of PN subsystems (docs/CASES.md C20). A bound pair, the tightest of both members,
        // with n_orb >= N_ORB_SECULAR orbits in the call whose omitted tide stays small over the whole call,
        //   gamma n_orb <= k_ref,  gamma = (r_apo^3 / (m_a m_b)) sum_k m_ab m_k / r_close,k^3          (W20 Eq. 22)
        // (r_close: the closest approach of perturber k in its current two-body orbit about the pair's c.o.m.), is
        // replaced by one body at its c.o.m. for the call: the limit kappa -> n_orb of SDAR (W20: kappa_org = k_ref /
        // gamma >= n_orb, the call is shorter than one slowed orbit), as NBODY6 treats unperturbed KS binaries and
        // stable hierarchies (Aarseth 2003 sec. 8, 11). The internal orbit is advanced at the end by the 2-body secular
        // path (integrate_secular_binary: orbit-averaged PN of PN pairs, Kepler drift of the others), which also
        // checks the PN write-back (C19) and the merger in a dry run before the collapse. FROST-CLUSTERS I (App. A2)
        // keeps kappa = 1 in PN subsystems, so LogH follows every orbit: 2.5-4.5 M GBS steps for one interval of a
        // 3-6 body set with a 0.02-0.05 AU BH binary (harvest-2026-10-19). The tide is watched after every GBS step
        // (SecWatch); above the same bound at the current distances the binary is expanded and LogH continues.
        // Energy: the collapse and the expansion change T + U of the state at a fixed phase-space point; those jumps
        // (minus the binary's internal energy, which is carried) go to B, as the SDAR kappa jumps (W20 Eq. 24), and
        // to sd_jump (booked as E_sd); the internal energy change by the secular PN is E_pn as all PN work.
        // NBODY_SEC_INNER=0 switches it off (A/B).
        static const bool sec_inner_on = [] { const char* e = std::getenv("NBODY_SEC_INNER"); return !e || std::atoi(e) != 0; }();
        struct SecRec { int com, live_b, kw_b, pn; double m_a, m_b, rad_b, r[3], v[3], t0, E_int0, qf; };
        SecRec sec[LOGH_MAX_BODIES / 2];
        int n_secr = 0, n_sec_used = 0;
        SecWatch sec_w;
        auto sec_watch_build = [&]() {
            sec_w.n = n_secr;
            for (int q = 0; q < n_secr; q++) { sec_w.com[q] = sec[q].com; sec_w.qf[q] = sec[q].qf; }
        };
        auto zero_com_coll = [&]() {   // a collapsed binary never collides (the watch expands it long before)
            for (int q = 0; q < n_secr; q++)
                for (int k = 0; k < state.n_bodies; k++) {
                    const int lo = std::min(k, sec[q].com), hi = std::max(k, sec[q].com);
                    if (lo != hi) coll[lo * LOGH_MAX_BODIES + hi] = 0.0;
                }
        };
        auto e_int = [&](double ma, double mb, const double* r, const double* v) {
            const double M = ma + mb;
            return ma * mb / M * (0.5 * (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) -
                                  G_norm * M / std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]));
        };
        // Body j joins body i at their c.o.m. (i keeps the slot, j is removed)
        auto sec_collapse = [&](int i, int j, int pn_b, double qf) {
            const double E0s = logh_total_energy(state, G_norm);
            SecRec q;
            q.m_a = state.mass[i]; q.m_b = state.mass[j];
            const double M = q.m_a + q.m_b;
            for (int c = 0; c < 3; c++) {
                q.r[c] = state.pos[j*3+c] - state.pos[i*3+c];
                q.v[c] = state.vel[j*3+c] - state.vel[i*3+c];
            }
            q.E_int0 = e_int(q.m_a, q.m_b, q.r, q.v);
            q.t0 = state.t; q.live_b = live[j]; q.kw_b = kw[j]; q.rad_b = rad_pc[j]; q.pn = pn_b; q.qf = qf;
            for (int c = 0; c < 3; c++) {
                state.pos[i*3+c] = (q.m_a * state.pos[i*3+c] + q.m_b * state.pos[j*3+c]) / M;
                state.vel[i*3+c] = (q.m_a * state.vel[i*3+c] + q.m_b * state.vel[j*3+c]) / M;
                state.aux_vel[i*3+c] = (q.m_a * state.aux_vel[i*3+c] + q.m_b * state.aux_vel[j*3+c]) / M;
            }
            state.mass[i] = M;
            for (int k = j; k + 1 < state.n_bodies; k++) {
                for (int c = 0; c < 3; c++) {
                    state.pos[k*3+c] = state.pos[(k+1)*3+c];
                    state.vel[k*3+c] = state.vel[(k+1)*3+c];
                    state.aux_vel[k*3+c] = state.aux_vel[(k+1)*3+c];
                }
                state.mass[k] = state.mass[k+1];
                kw[k] = kw[k+1]; rad_pc[k] = rad_pc[k+1]; live[k] = live[k+1];
            }
            state.n_bodies--;
            n_live = state.n_bodies;
            for (int r = 0; r < n_secr; r++) if (sec[r].com > j) sec[r].com--;
            q.com = i > j ? i - 1 : i;
            sec[n_secr++] = q;
            const double dE = logh_total_energy(state, G_norm) - E0s;
            state.B -= dE;
            state.sd_jump += dE + q.E_int0;
            n_sec_used++;
        };
        // Record qi: its internal orbit advanced secularly over the time since the collapse, the members restored
        // (member a in the c.o.m. slot, member b appended)
        auto sec_expand = [&](int qi) {
            const SecRec q = sec[qi];
            for (int r = qi; r + 1 < n_secr; r++) sec[r] = sec[r + 1];
            n_secr--;
            const int c0 = q.com;
            const double M = q.m_a + q.m_b;
            double p1[3], v1[3], p2[3], v2[3];
            auto split = [&]() {
                for (int c = 0; c < 3; c++) {
                    p1[c] = -q.m_b / M * q.r[c]; p2[c] = q.m_a / M * q.r[c];
                    v1[c] = -q.m_b / M * q.v[c]; v2[c] = q.m_a / M * q.v[c];
                }
            };
            split();
            const double dts = state.t - q.t0;
            if (dts != 0.0 &&
                integrate_secular_binary(p1, v1, q.m_a, p2, v2, q.m_b, G_norm, q.pn ? c_norm : 0.0, q.pn, dts) < 0) {
                split();   // PN write-back refused although the dry run passed: Kepler drift (booked as E_pn all the same)
                integrate_secular_binary(p1, v1, q.m_a, p2, v2, q.m_b, G_norm, 0.0, 0, dts);
            }
            double rr[3], vv[3];
            for (int c = 0; c < 3; c++) { rr[c] = p2[c] - p1[c]; vv[c] = v2[c] - v1[c]; }
            const double E_int1 = e_int(q.m_a, q.m_b, rr, vv);
            const double E0s = logh_total_energy(state, G_norm);
            const int nn = state.n_bodies;
            for (int c = 0; c < 3; c++) {
                const double xc = state.pos[c0*3+c], vc = state.vel[c0*3+c], wc = state.aux_vel[c0*3+c];
                state.pos[c0*3+c] = xc + p1[c]; state.vel[c0*3+c] = vc + v1[c]; state.aux_vel[c0*3+c] = wc + v1[c];
                state.pos[nn*3+c] = xc + p2[c]; state.vel[nn*3+c] = vc + v2[c]; state.aux_vel[nn*3+c] = wc + v2[c];
                state.ext_accel[nn*3+c] = 0.0;
            }
            state.mass[c0] = q.m_a;
            state.mass[nn] = q.m_b;
            kw[nn] = q.kw_b; rad_pc[nn] = q.rad_b; live[nn] = q.live_b;
            state.n_bodies++;
            n_live = state.n_bodies;
            const double dE = logh_total_energy(state, G_norm) - E0s;
            state.B -= dE;
            state.sd_jump += dE - E_int1;
        };
        if (sec_inner_on && state.use_aux_vel && n_live >= 3 && dt_norm > 0.0) {
            struct Cand { double a, e; int i, j; };
            Cand cand[LOGH_MAX_BODIES * (LOGH_MAX_BODIES - 1) / 2];
            int nc = 0;
            for (int i = 0; i < n_live; i++)
                for (int j = i + 1; j < n_live; j++) {
                    double r[3], v[3];
                    for (int c = 0; c < 3; c++) {
                        r[c] = state.pos[j*3+c] - state.pos[i*3+c];
                        v[c] = state.vel[j*3+c] - state.vel[i*3+c];
                    }
                    const double mu = G_norm * (state.mass[i] + state.mass[j]);
                    const double r2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
                    const double eps = 0.5 * (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) - mu / std::sqrt(r2);
                    if (!(eps < 0.0)) continue;
                    const double hx = r[1]*v[2] - r[2]*v[1], hy = r[2]*v[0] - r[0]*v[2], hz = r[0]*v[1] - r[1]*v[0];
                    const double e2 = 1.0 + 2.0 * eps * (hx*hx + hy*hy + hz*hz) / (mu * mu);
                    cand[nc++] = {-mu / (2.0 * eps), std::sqrt(std::max(0.0, e2)), i, j};
                }
            std::sort(cand, cand + nc, [](const Cand& x, const Cand& y) { return x.a < y.a; });
            bool used[LOGH_MAX_BODIES] = {};
            int sel_a[LOGH_MAX_BODIES / 2], sel_b[LOGH_MAX_BODIES / 2], sel_pn[LOGH_MAX_BODIES / 2], n_sel = 0;
            double sel_qf[LOGH_MAX_BODIES / 2];
            for (int q = 0; q < nc; q++) {
                const int i = cand[q].i, j = cand[q].j;
                if (used[i] || used[j]) continue;
                used[i] = used[j] = true;   // the tightest pair of a body is its inner orbit
                if (n_live - n_sel - 1 < 2) break;
                const double a = cand[q].a, e = cand[q].e;
                const double mi = state.mass[i], mj = state.mass[j], mb = mi + mj;
                const double P = 2.0 * M_PI * std::sqrt(a * a * a / (G_norm * mb));
                const double n_orb = dt_norm / P;
                if (n_orb < g_collisional.n_orb_secular) continue;
                const double r_coll = coll[i * LOGH_MAX_BODIES + j];
                if (r_coll > 0.0 && a * (1.0 - e) < r_coll) continue;   // collision candidate: LogH (FC-I App. B3)
                double xcm[3], vcm[3];
                for (int c = 0; c < 3; c++) {
                    xcm[c] = (mi * state.pos[i*3+c] + mj * state.pos[j*3+c]) / mb;
                    vcm[c] = (mi * state.vel[i*3+c] + mj * state.vel[j*3+c]) / mb;
                }
                double pert = 0.0;
                for (int k = 0; k < n_live; k++) {
                    if (k == i || k == j) continue;
                    double d[3], u[3];
                    for (int c = 0; c < 3; c++) { d[c] = state.pos[k*3+c] - xcm[c]; u[c] = state.vel[k*3+c] - vcm[c]; }
                    const double rn = std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
                    // Closest approach within the call: the pericentre if it comes before the end, else the nearer
                    // end point (the distance falls monotonically up to the pericentre)
                    const double mu_k = G_norm * (mb + state.mass[k]);
                    double rp_k = 0.0;
                    const double tau = kepler_time_to_pericentre(d, u, mu_k, &rp_k);
                    double r_close = std::min(rn, rp_k);
                    if (!(tau <= dt_norm)) {
                        kepler_drift(d, u, mu_k, dt_norm);
                        r_close = std::min(rn, std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]));
                    }
                    pert += mb * state.mass[k] / (r_close * r_close * r_close);
                }
                const double r_apo = a * (1.0 + e);
                const double gamma = pert * r_apo * r_apo * r_apo / (mi * mj);
                const double k_ref = g_collisional.sdar_kref;
                if (!(gamma * n_orb <= k_ref)) continue;
                const bool pn_pair = g_collisional.pn_body(state.mass[i] * M_total, kw[i]) ||
                                     g_collisional.pn_body(state.mass[j] * M_total, kw[j]);
                const int pn_b = pn_pair ? pn_order : 0;
                // Dry run of the secular advance over the whole call: a refused write-back (C19), a merger (members
                // left unchanged) or a non-finite result keep the pair in LogH
                double p1[3], v1[3], p2[3], v2[3];
                for (int c = 0; c < 3; c++) {
                    p1[c] = state.pos[i*3+c] - xcm[c]; v1[c] = state.vel[i*3+c] - vcm[c];
                    p2[c] = state.pos[j*3+c] - xcm[c]; v2[c] = state.vel[j*3+c] - vcm[c];
                }
                const double p1_0 = p1[0];
                if (integrate_secular_binary(p1, v1, mi, p2, v2, mj, G_norm, pn_b ? c_norm : 0.0, pn_b, dt_norm) < 0) continue;
                if (!std::isfinite(p1[0] + p1[1] + p1[2] + v1[0] + v1[1] + v1[2]) || p1[0] == p1_0) continue;
                sel_a[n_sel] = live[i]; sel_b[n_sel] = live[j]; sel_pn[n_sel] = pn_b;
                sel_qf[n_sel] = r_apo * r_apo * r_apo * n_orb / (mi * mj * k_ref);
                n_sel++;
            }
            for (int q = 0; q < n_sel; q++) {
                int i = -1, j = -1;
                for (int k = 0; k < n_live; k++) { if (live[k] == sel_a[q]) i = k; if (live[k] == sel_b[q]) j = k; }
                if (i >= 0 && j >= 0) sec_collapse(i, j, sel_pn[q], sel_qf[q]);
            }
            if (n_secr > 0) {
                build_coll();
                zero_com_coll();
                sec_watch_build();
            }
        }

        // Gauss-IRK (CASES.md C22, NBODY_GAUSS_IRK=1): compact Newtonian subsystems (gauss_irk_choose) that SDAR would
        // not slow are integrated by the symplectic Gauss-Legendre method; if it does not finish (no convergence,
        // collision watch, step limit) the call starts again from the same state with LogH+GBS below
        bool irk_done = false;
        if (gauss_irk_on() && state.use_aux_vel) g_irk_totals.not_chosen[6]++;
        if (gauss_irk_on() && !state.use_aux_vel && n_secr == 0 && n_live >= 2 && dt_norm > 0.0) {
            bool slowed = false;
            if (g_collisional.sdar && n_live >= 3 && n_live <= 4) {
                LogHState probe = state;
                SdarParams sp;
                sp.k_ref = g_collisional.sdar_kref;
                sp.c = g_collisional.sdar_c;
                sp.coll_pair = coll;
                logh_sd_update(probe, G_norm, sp);
                for (int q = 0; q < probe.n_sd; q++) if (probe.sd_kappa[q] > 1.0) slowed = true;
            }
            const int why = slowed ? 7 : gauss_irk_choose(state, G_norm, dt_norm, coll_on ? coll : nullptr);
            if (why > 0) g_irk_totals.not_chosen[why]++;
            if (why == 0) {
                LogHState trial = state;
                GaussIrkOut io;
                GaussIrkStats ist;
                gauss_irk_integrate(trial, dt_norm, G_norm, GaussIrkOpts(), 2LL * gbs_step_limit(state, dt_norm, G_norm) + 16,
                                    coll_on ? coll : nullptr, io, &ist);
                g_irk_totals.calls++;
                g_irk_totals.steps += ist.steps;
                g_irk_totals.rhs += ist.rhs;
                g_irk_totals.newton += ist.newton;
                if (io.ok) {
                    state = trial;
                    irk_done = true;
                    irk_sub[s] = 1;
                } else {
                    g_irk_totals.fallbacks++;
                    g_irk_totals.fail[std::min(std::max(io.fail, 0), 4)]++;
                    irk_sub[s] = 2;
                }
            }
        }

        // SDAR (FROST-CLUSTERS I App. A2): slow-down of weakly perturbed inner binaries; kappa = 1 for 2-body
        // subsystems (no perturber inside the subsystem: the external tide comes from the FSI kicks, which need the
        // true phase) and for collision candidates (osculating pericentre inside the collision distance, FC-I App.
        // B3). FC-I sets kappa = 1 for whole PN subsystems; since C20 (PLAN item 4c; NBODY_SDAR_PN=0: off) the code keeps kappa = 1 only for
        // pairs with a member above M_PN and slows the other inner binaries of PN subsystems
        static const bool sdar_pn = [] { const char* e = std::getenv("NBODY_SDAR_PN"); return !e || std::atoi(e) != 0; }();
        const bool sd_on = g_collisional.sdar && (sdar_pn || !state.use_aux_vel) && n_live >= 3 && !irk_done;
        SdarParams sd_par;
        auto sd_pn_build = [&]() {   // PN bodies by state slot (a collapsed binary: either member)
            for (int k = 0; k < n_live; k++) {
                sd_par.pn[k] = g_collisional.pn_body(h_mass(off + live[k]), batch.flat_stype[off + live[k]]);
                for (int q = 0; q < n_secr; q++)
                    if (sec[q].com == k &&
                        g_collisional.pn_body(h_mass(off + sec[q].live_b), batch.flat_stype[off + sec[q].live_b]))
                        sd_par.pn[k] = true;
            }
        };
        if (sd_on && state.use_aux_vel) sd_pn_build();
        if (sd_on) {
            sd_par.k_ref = g_collisional.sdar_kref;
            sd_par.c = g_collisional.sdar_c;
            sd_par.steps_per_orbit = gbs_steps_per_orbit();   // the limit follows kappa during the call
            sd_par.coll_pair = coll;
            // perturbers outside the subsystem (C21): their tide reaches the members through the FSI kicks, which
            // need the true phase; normalised sum m_k / d_k^3
            if (s < (int)batch.ext_tidal.size())
                sd_par.ext_tidal = batch.ext_tidal[s] * R_scale * R_scale * R_scale / M_total;
            static const bool sdar_ext = [] { const char* e = std::getenv("NBODY_SDAR_EXT"); return !e || std::atoi(e) != 0; }();
            if (!sdar_ext) sd_par.ext_tidal = 0.0;   // A/B: outside perturbers ignored (before C21)
            else sd_par.ext_share = g_collisional.flyby_cm_eps * g_collisional.e_ref / E_scale;
            logh_sd_update(state, G_norm, sd_par);   // initial kappa: B = -H_sd
        }

        // Integrate; a collision stops the integration (GBSIntegrator::coll): the pair merges into one body of mass
        // m1 + m2 at its centre of mass with its velocity (momentum exact, SDAR / PeTar merge the same way inside
        // the integrator: Wang et al. 2020), the change of T + U is booked (dE_int, into B so that the integration
        // error stays T + U + B, and into the reference energy of the check) and the others continue
        int n_gbs_steps = 0, max_steps = 0;
        bool reached = false, limit_hit = false;
        double E_ref = E0_norm;              // E0 plus the collisions' internal energy changes
        int ghost_of[LOGH_MAX_BODIES];       // member merged in this call -> member that kept the remnant (-1: none)
        double ghost_d[LOGH_MAX_BODIES * 3]; // its offset from the remnant at the collision (normalised)
        for (int b = 0; b < nb; b++) ghost_of[b] = -1;
        auto& colls = batch.collisions[s];
        const double t_stop = dt_norm;
        if (irk_done) reached = true;
        if (n_live <= 1) {   // one body left after a collision in an earlier call: a free drift
            for (int c = 0; c < 3 * n_live; c++) state.pos[c] += state.vel[c] * t_stop;
            state.t = t_stop;
            reached = true;
        }
        while (n_live > 1 && !irk_done) {
            const int seg_limit = gbs_step_limit(state, t_stop - state.t, G_norm);
            if (max_steps == 0) max_steps = seg_limit;
            GBSIntegrator gbs(eta_gbs);
            gbs.par_stages = serial_subs && stage_par_max > 0 && host_threads > 1;
            if (sd_on) gbs.sdar = &sd_par;
            if (coll_on) gbs.coll = coll;
            if (n_secr > 0) gbs.sec_watch = &sec_w;
            double Omega_cur = logh_omega(state);
            double ds_est = (Omega_cur > 1e-30) ? (t_stop - state.t) * Omega_cur : (t_stop - state.t);
            const int n_seg = gbs.integrate_to_time(state, t_stop, G_norm, ds_est, c_norm, pn_order, seg_limit);
            n_gbs_steps += n_seg;
            peri_land[s] += gbs.peri_landings;
            peri_miss[s] += gbs.peri_misses;
            if (gbs.coll_i < 0 && gbs.sec_hit) {   // the tide on a collapsed binary grew: expand all, continue
                if (sd_on) logh_sd_update(state, G_norm, sd_par, true);
                while (n_secr > 0) sec_expand(n_secr - 1);
                sec_w.n = 0;
                build_coll();
                if (sd_on && state.use_aux_vel) sd_pn_build();
                if (sd_on && state.n_bodies >= 3) logh_sd_update(state, G_norm, sd_par);
                continue;
            }
            if (gbs.coll_i < 0) {
                reached = gbs.reached;
                limit_hit = n_seg >= seg_limit;
                break;
            }
            // --- Collision of state bodies ci < cj
            const int ci = gbs.coll_i, cj = gbs.coll_j;
            if (sd_on) logh_sd_update(state, G_norm, sd_par, true);   // physical H at the collision (jump booked)
            if (state.has_spins) {   // keep what the spins evolved to
                std::memcpy(&batch.stored_spins[s*6+0], state.spin1, 3 * sizeof(double));
                std::memcpy(&batch.stored_spins[s*6+3], state.spin2, 3 * sizeof(double));
                state.has_spins = false;
            }
            // The remnant keeps the compact member (BH: the TDE / ISCO convention of execute_merger), else the heavier
            const bool ci_keeps = (kw[ci] >= 10) != (kw[cj] >= 10) ? kw[ci] >= 10 : state.mass[ci] >= state.mass[cj];
            const int kk = ci_keeps ? ci : cj, kg = ci_keeps ? cj : ci;
            const double mk = state.mass[kk], mg = state.mass[kg], mm = mk + mg;
            double xc[3], vc[3], wc[3];
            for (int c = 0; c < 3; c++) {
                xc[c] = (mk * state.pos[kk*3+c] + mg * state.pos[kg*3+c]) / mm;
                vc[c] = (mk * state.vel[kk*3+c] + mg * state.vel[kg*3+c]) / mm;
                wc[c] = (mk * state.aux_vel[kk*3+c] + mg * state.aux_vel[kg*3+c]) / mm;
            }
            SubsystemCollision rec;
            rec.type = ctype[ci * LOGH_MAX_BODIES + cj];
            rec.pid_keep = batch.flat_pid[off + live[kk]];
            rec.pid_gone = batch.flat_pid[off + live[kg]];
            rec.kw_keep = kw[kk]; rec.kw_gone = kw[kg];
            rec.m_keep = mk * M_total; rec.m_gone = mg * M_total;
            for (int c = 0; c < 3; c++) {
                rec.d_keep[c] = (state.pos[kk*3+c] - xc[c]) * R_scale;
                rec.d_gone[c] = (state.pos[kg*3+c] - xc[c]) * R_scale;
                rec.u_keep[c] = (state.vel[kk*3+c] - vc[c]) * V_scale;
                rec.u_gone[c] = (state.vel[kg*3+c] - vc[c]) * V_scale;
            }
            rec.r_sep = gbs.coll_rsep * R_scale;
            rec.r_coll = coll[ci * LOGH_MAX_BODIES + cj] * R_scale;
            rec.t_rel = state.t * T_scale;
            rec.n_members = nb;
            rec.member_pids.assign(&batch.flat_pid[off], &batch.flat_pid[off] + nb);
            const double E_before = logh_total_energy(state, G_norm);
            ghost_of[live[kg]] = live[kk];
            for (int c = 0; c < 3; c++) ghost_d[live[kg]*3+c] = state.pos[kg*3+c] - xc[c];
            for (int c = 0; c < 3; c++) {
                state.pos[kk*3+c] = xc[c]; state.vel[kk*3+c] = vc[c]; state.aux_vel[kk*3+c] = wc[c];
            }
            state.mass[kk] = mm;
            if (kw[kg] >= 10 || kw[kk] >= 10) kw[kk] = std::max(kw[kk], kw[kg]);
            rad_pc[kk] = (kw[kk] == 14) ? 0.0 : std::max(rad_pc[kk], rad_pc[kg]);
            for (int k = kg; k + 1 < state.n_bodies; k++) {   // remove kg from the state
                for (int c = 0; c < 3; c++) {
                    state.pos[k*3+c] = state.pos[(k+1)*3+c];
                    state.vel[k*3+c] = state.vel[(k+1)*3+c];
                    state.aux_vel[k*3+c] = state.aux_vel[(k+1)*3+c];
                    state.ext_accel[k*3+c] = state.ext_accel[(k+1)*3+c];
                }
                state.mass[k] = state.mass[k+1];
                kw[k] = kw[k+1];
                rad_pc[k] = rad_pc[k+1];
                live[k] = live[k+1];
            }
            state.n_bodies--;
            n_live = state.n_bodies;
            for (int q = 0; q < n_secr; q++) if (sec[q].com > kg) sec[q].com--;
            const double dE = logh_total_energy(state, G_norm) - E_before;
            rec.dE_int = dE * E_scale;
            batch.coll_dE[s] += rec.dE_int;
            colls.push_back(rec);
            state.B -= dE;
            E_ref += dE;
            if (state.n_bodies == 1) {   // nothing left to integrate: free drift to the end time
                for (int c = 0; c < 3; c++) state.pos[c] += state.vel[c] * (t_stop - state.t);
                state.t = t_stop;
                reached = true;
                break;
            }
            build_coll();
            zero_com_coll();
            sec_watch_build();
            if (sd_on && state.use_aux_vel) sd_pn_build();
            if (sd_on && state.n_bodies >= 3) logh_sd_update(state, G_norm, sd_par);
        }
        // Remove the slow-down: the last jump back to H (W20 Eq. 24); then T + U + B is the integration error of
        // the slowed Hamiltonian over the call (W20 Eq. 25), and sd_jump the booked physical energy change
        if (sd_on) logh_sd_update(state, G_norm, sd_par, true);
        while (n_secr > 0) sec_expand(n_secr - 1);   // secular inner binaries: internal orbits advanced to the end
        if (sd_on || n_sec_used > 0) batch.sd_jump[s] = state.sd_jump * E_scale;
        if (sd_on) sd_kappa_sub[s] = state.sd_kappa_peak;
        sec_used_sub[s] = n_sec_used;

        // ==================================================================
        // Step 4: Denormalize and store results
        // CM drifts uniformly: x_CM(t) = x_CM(0) + v_CM * dt
        // ==================================================================
        double dt_actual = dt;
        double cm_x_final = cm_x + cm_vx * dt_actual;
        double cm_y_final = cm_y + cm_vy * dt_actual;
        double cm_z_final = cm_z + cm_vz * dt_actual;

        int slot[LOGH_MAX_BODIES];   // member -> state body (-1: massless)
        for (int b = 0; b < nb; b++) slot[b] = -1;
        for (int k = 0; k < n_live; k++) slot[live[k]] = k;
        for (int b = 0; b < nb; b++) {
            if (slot[b] >= 0) {
                const int k = slot[b];
                h_pos_out((off+b)*3+0) = state.pos[k*3+0] * R_scale + cm_x_final;
                h_pos_out((off+b)*3+1) = state.pos[k*3+1] * R_scale + cm_y_final;
                h_pos_out((off+b)*3+2) = state.pos[k*3+2] * R_scale + cm_z_final;
                h_vel_out((off+b)*3+0) = state.vel[k*3+0] * V_scale + cm_vx;
                h_vel_out((off+b)*3+1) = state.vel[k*3+1] * V_scale + cm_vy;
                h_vel_out((off+b)*3+2) = state.vel[k*3+2] * V_scale + cm_vz;
                h_mass(off + b) = state.mass[k] * M_total;
            } else if (ghost_of[b] >= 0) {   // merged in this call: at its collision offset from the remnant
                int r = ghost_of[b];
                while (slot[r] < 0 && ghost_of[r] >= 0) r = ghost_of[r];   // the remnant merged again
                const int k = slot[r];
                for (int c = 0; c < 3; c++) {
                    h_pos_out((off+b)*3+c) = (state.pos[k*3+c] + ghost_d[b*3+c]) * R_scale + (c == 0 ? cm_x_final : c == 1 ? cm_y_final : cm_z_final);
                    h_vel_out((off+b)*3+c) = state.vel[k*3+c] * V_scale + (c == 0 ? cm_vx : c == 1 ? cm_vy : cm_vz);
                }
                h_mass(off + b) = 0.0;
            } else {   // massless placeholder from an earlier call: co-moves at its velocity
                for (int c = 0; c < 3; c++) {
                    h_pos_out((off+b)*3+c) = h_pos((off+b)*3+c) + h_vel((off+b)*3+c) * dt_actual;
                    h_vel_out((off+b)*3+c) = h_vel((off+b)*3+c);
                }
                h_mass(off + b) = 0.0;
            }
        }

        // Store auxiliary velocities and scales for persistence across calls.
        // (B is always recomputed fresh from current T+U — no persistence needed.)
        if (state.use_aux_vel) {
            batch.has_stored_aux_vel[s] = true;
            int av_off = batch.stored_aux_vel_offsets[s] * 3;
            for (int b = 0; b < nb; b++) {
                const int k = slot[b];
                for (int c = 0; c < 3; c++)   // massless members: their velocity
                    batch.stored_aux_vel[av_off + b*3+c] = k >= 0 ? state.aux_vel[k*3+c] * V_scale + (c == 0 ? cm_vx : c == 1 ? cm_vy : cm_vz)
                                                                  : h_vel_out((off+b)*3+c);
            }
        }

        // Store evolved spins back (dimensionless, no normalization needed)
        if (state.has_spins) {
            batch.stored_spins[s*6+0] = state.spin1[0];
            batch.stored_spins[s*6+1] = state.spin1[1];
            batch.stored_spins[s*6+2] = state.spin1[2];
            batch.stored_spins[s*6+3] = state.spin2[0];
            batch.stored_spins[s*6+4] = state.spin2[1];
            batch.stored_spins[s*6+5] = state.spin2[2];
        }

        // Energy error in normalized units (dimensionless)
        double Ef_norm = logh_total_energy(state, G_norm);
        double dE_rel;
        if (any_compact && pn_order > 0 && c_norm > 0.0 && (pn_order & 1) && colls.empty()) {
            double v0_lab[LOGH_MAX_BODIES * 3], vf_lab[LOGH_MAX_BODIES * 3];   // 1PN energy in the lab frame
            for (int i = 0; i < 3 * nb; i++) {
                v0_lab[i] = state_init_vel[i] + state.v_frame[i % 3];
                vf_lab[i] = state.vel[i] + state.v_frame[i % 3];
            }
            double E_pn_0 = pn1_energy(state_init_pos, v0_lab, state.mass,
                                        nb, G_norm, c_norm);
            double E_pn_f = pn1_energy(state.pos, vf_lab, state.mass,
                                        nb, G_norm, c_norm);
            double E0_total = E0_norm + E_pn_0;
            double Ef_total = Ef_norm + E_pn_f;
            dE_rel = std::fabs((Ef_total - E0_total) / (std::fabs(E0_total) + 1e-30));
        } else if (sd_on) {
            dE_rel = std::fabs((Ef_norm + state.B) / (std::fabs(E0_norm) + 1e-30));
        } else {
            dE_rel = std::fabs((Ef_norm - E_ref) / (std::fabs(E0_norm) + 1e-30));
        }
        // With PN terms there is no energy this check can hold to eta_restart: E_N + E_1PN is conserved
        // only to O(v^4/c^4), and 2PN/3PN/radiation reaction change it further. PN subsystems restart on
        // GBS non-convergence instead; their Newtonian energy change is booked as E_pn by the caller.
        // A step-limited integration stopped short of the end time: never accept it on its energy (the
        // Newtonian check passed it and the members silently lagged behind the rest of the system)
        bool converged = (sd_on || !colls.empty()) ? reached : !limit_hit;
        if (any_compact && pn_order > 0 && c_norm > 0.0) dE_rel = 0.0;
        // The members and the bound pairs with their orbits per interval (initial state)
        auto describe = [&](std::string msg) {
            char buf[160];
            for (int b = 0; b < nb; b++) {
                std::snprintf(buf, sizeof(buf), "%s%.3g", b ? "," : "", h_mass(off + b));
                msg += buf;
            }
            for (int i = 0; i < n0; i++)
                for (int j = i + 1; j < n0; j++) {
                    double r2 = 0.0, v2 = 0.0, dx[3], dv[3];
                    for (int k = 0; k < 3; k++) {
                        dx[k] = state_init_pos[j*3+k] - state_init_pos[i*3+k];
                        dv[k] = state_init_vel[j*3+k] - state_init_vel[i*3+k];
                        r2 += dx[k] * dx[k];
                        v2 += dv[k] * dv[k];
                    }
                    const double mu = mass_init[i] + mass_init[j];   // G_norm = 1
                    const double eps = 0.5 * v2 - mu / std::sqrt(r2);
                    const double a = -mu / (2.0 * eps);
                    std::snprintf(buf, sizeof(buf), " | %d-%d r=%.3g a=%.3g pc%s", i, j, std::sqrt(r2) * R_scale,
                                  a * R_scale, eps < 0.0 ? "" : " (unbound)");
                    msg += buf;
                    if (eps < 0.0) {
                        std::snprintf(buf, sizeof(buf), " orbits/interval=%.3g",
                                      dt_norm / (2.0 * M_PI * std::sqrt(a * a * a / mu)));
                        msg += buf;
                    }
                    if (eps < 0.0) {   // pericentre over the collision distance (< 1: candidate, kappa = 1)
                        const double hx = dx[1]*dv[2] - dx[2]*dv[1], hy = dx[2]*dv[0] - dx[0]*dv[2];
                        const double hz = dx[0]*dv[1] - dx[1]*dv[0];
                        const double e = std::sqrt(std::max(0.0, 1.0 + 2.0 * eps * (hx*hx + hy*hy + hz*hz) / (mu * mu)));
                        const double rc = coll0[i * LOGH_MAX_BODIES + j];
                        std::snprintf(buf, sizeof(buf), " e=%.4f r_p/r_coll=%.3g", e, rc > 0.0 ? a * (1.0 - e) / rc : -1.0);
                        msg += buf;
                    }
                }
            return msg;
        };
        if (!converged) {
            dE_rel = 1e30;
            std::printf("%s\n", describe("      LogH step limit " + std::to_string(sd_on ? n_gbs_steps : max_steps) + ": n=" +
                                         std::to_string(nb) + " m[Msun]=").c_str());
        }
        // NBODY_LOGH_SLOW=k: name every integration that took at least k GBS steps (the stragglers of a batch)
        static const int slow_log = std::getenv("NBODY_LOGH_SLOW") ? std::atoi(std::getenv("NBODY_LOGH_SLOW")) : 0;
        if (slow_log > 0 && n_gbs_steps >= slow_log) {
            char head[200];
            std::snprintf(head, sizeof(head), "      LogH slow: gbs=%d dt=%.3e Myr sd=%d kappa_peak=%.3g secinner=%d colls=%zu n=%d m[Msun]=",
                          n_gbs_steps, dt_norm * T_scale * TIME_UNIT_MYR, (int)sd_on, state.sd_kappa_peak, n_sec_used,
                          colls.size(), nb);
            std::printf("%s\n", describe(head).c_str());
            std::fflush(stdout);
        }
        // STALL_FACTOR / STALL_DE_FACTOR: the costliest integrations and those with the largest energy error
        {
            const double err = converged && g_collisional.e_ref > 0.0
                                   ? dE_rel * std::fabs(E0_norm) * E_scale / g_collisional.e_ref : 0.0;
            if (n_gbs_steps >= 1000 || err > 1e-15) {
                double ms[LOGH_MAX_BODIES];
                for (int b = 0; b < nb; b++) ms[b] = h_mass(off + b);
                if (n_gbs_steps >= 1000)
                    g_stall.note_logh(n_gbs_steps, dt_norm * T_scale * TIME_UNIT_MYR, &batch.flat_pid[off], ms, nb);
                if (err > 1e-15) g_stall.note_logh_err(err, &batch.flat_pid[off], ms, nb);
            }
        }
        // SDAR: name the integrations that fail the energy check (members, slowed binaries, steps)
        if (sd_on && converged && g_collisional.eta_restart > 0.0 && dE_rel > g_collisional.eta_restart) {
            char buf[256];
            std::snprintf(buf, sizeof(buf), "      SDAR energy check: dE=%.3e n=%d gbs=%d kappa_peak=%.3g jump/E0=%.3e m[Msun]=",
                          dE_rel, nb, n_gbs_steps, state.sd_kappa_peak, state.sd_jump / (std::fabs(E0_norm) + 1e-300));
            std::string msg = buf;
            for (int b = 0; b < nb; b++) {
                std::snprintf(buf, sizeof(buf), "%s%.3g", b ? "," : "", h_mass(off + b));
                msg += buf;
            }
            // Bound pairs at the start: a, e, pericentre over the collision distance (< 1: collision candidate,
            // kept at kappa = 1)
            for (int i = 0; i < n0; i++)
                for (int j = i + 1; j < n0; j++) {
                    double r[3], v[3], r2 = 0.0, v2 = 0.0;
                    for (int k = 0; k < 3; k++) {
                        r[k] = state_init_pos[j*3+k] - state_init_pos[i*3+k];
                        v[k] = state_init_vel[j*3+k] - state_init_vel[i*3+k];
                        r2 += r[k] * r[k];
                        v2 += v[k] * v[k];
                    }
                    const double mu = mass_init[i] + mass_init[j];
                    const double eps = 0.5 * v2 - mu / std::sqrt(r2);
                    if (eps >= 0.0) continue;
                    const double hx = r[1]*v[2] - r[2]*v[1], hy = r[2]*v[0] - r[0]*v[2], hz = r[0]*v[1] - r[1]*v[0];
                    const double a = -mu / (2.0 * eps);
                    const double e = std::sqrt(std::max(0.0, 1.0 + 2.0 * eps * (hx*hx + hy*hy + hz*hz) / (mu * mu)));
                    const double rc = coll0[i * LOGH_MAX_BODIES + j];
                    std::snprintf(buf, sizeof(buf), " | %d-%d a=%.3g pc e=%.4f r_p/r_coll=%.3g", i, j, a * R_scale, e,
                                  rc > 0.0 ? a * (1.0 - e) / rc : -1.0);
                    msg += buf;
                }
            std::printf("%s\n", msg.c_str());
        }
        h_eerr(s) = dE_rel;
        h_conv(s) = converged ? 1 : 0;

        gbs_sum += n_gbs_steps;
        gbs_per_sub[s] = n_gbs_steps;
        batch.t_reached[s] = state.t * T_scale;
        if (dE_rel > dE_max) dE_max = dE_rel;
        if (dE_rel < dE_min) dE_min = dE_rel;
        if (!converged) caps++;
        t_sub[s] = wall_now() - t_s0;
    };
    const double t_i0 = wall_now();
    if (serial_subs) {   // most deep-level batches: one subsystem after the other, GBS stages in parallel
        max_dE = std::numeric_limits<double>::lowest();
        min_dE = std::numeric_limits<double>::max();
        for (int s = 0; s < n_sub; s++) integrate_one(s, total_gbs_steps, max_dE, min_dE, cap_hit);
    } else {
        // Longest first (LPT): with dynamic scheduling a long integration started last leaves the other threads
        // idle at the end of the batch. Cost estimate: n^2 x orbits per interval of the tightest bound pair (the
        // GBS steps follow the orbits). Each subsystem's result does not depend on the order
        std::vector<int> order(n_sub);
        std::vector<double> cost(n_sub, 0.0);
        for (int s = 0; s < n_sub; s++) {
            order[s] = s;
            const int nb = h_nbod(s), off = h_off(s);
            double orbits = 1.0;
            for (int i = 0; i < nb; i++)
                for (int j = i + 1; j < nb; j++) {
                    double r2 = 0.0, v2 = 0.0;
                    for (int c = 0; c < 3; c++) {
                        const double d = h_pos((off + j) * 3 + c) - h_pos((off + i) * 3 + c);
                        const double u = h_vel((off + j) * 3 + c) - h_vel((off + i) * 3 + c);
                        r2 += d * d; v2 += u * u;
                    }
                    const double mu = G * (h_mass(off + i) + h_mass(off + j));
                    if (!(r2 > 0.0) || !(mu > 0.0)) continue;
                    const double en = 0.5 * v2 - mu / std::sqrt(r2);
                    if (en >= 0.0) continue;
                    const double a = -mu / (2.0 * en);
                    orbits = std::max(orbits, h_dt(s) / (2.0 * M_PI * std::sqrt(a * a * a / mu)));
                }
            cost[s] = (double)nb * nb * orbits;
        }
        std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return cost[x] > cost[y]; });
        Kokkos::parallel_reduce("logh_batch_host", HostPolicy(0, n_sub).set_chunk_size(1),
                                [&](int k, int& gbs, double& mx, double& mn, int& caps) {
                                    integrate_one(order[k], gbs, mx, mn, caps);
                                },
                                Kokkos::Sum<int>(total_gbs_steps), Kokkos::Max<double>(max_dE),
                                Kokkos::Min<double>(min_dE), Kokkos::Sum<int>(cap_hit));
    }
    g_logh_prof.integ += wall_now() - t_i0;
    for (int s = 0; s < n_sub; s++) total_gbs_steps += is_mstar[s] ? gbs_per_sub[s] : 0;
    if (batch.stat_mstar > 0) {
        max_dE = std::max(max_dE, mstar_max_dE);
        min_dE = std::min(min_dE, mstar_min_dE);
        cap_hit += mstar_caps;
    }
    {
        double mx = 0.0, sum = 0.0;
        for (double t : t_sub) { mx = std::max(mx, t); sum += t; }
        g_logh_prof.crit += mx;
        g_logh_prof.work += sum;
        g_logh_prof.n_sub += n_sub;
    }

    // Store stats for caller (printed in step summary, not here)
    batch.stat_total_gbs = total_gbs_steps;
    {
        int mx = 0;
        for (int g : gbs_per_sub) mx = std::max(mx, g);
        g_profiler.record_logh(n_sub, total_gbs_steps, mx);
    }
    batch.stat_worst_dE  = max_dE;
    batch.stat_best_dE   = min_dE;
    batch.stat_cap_hit   = cap_hit;
    batch.stat_sd_calls = 0;
    batch.stat_sd_kappa_max = 0.0;
    batch.stat_collisions = 0;
    batch.stat_peri_landings = 0;
    batch.stat_peri_misses = 0;
    for (int s2 = 0; s2 < n_sub; s2++) {
        batch.stat_collisions += (int)batch.collisions[s2].size();
        batch.stat_peri_landings += peri_land[s2];
        batch.stat_peri_misses += peri_miss[s2];
    }
    batch.stat_sec_inner = 0;
    for (int u : sec_used_sub) batch.stat_sec_inner += u;
    batch.stat_irk = batch.stat_irk_fb = 0;
    batch.irk_used.assign(n_sub, 0);
    for (int s2 = 0; s2 < n_sub; s2++) {
        batch.stat_irk += irk_sub[s2] == 1;
        batch.stat_irk_fb += irk_sub[s2] == 2;
        batch.irk_used[s2] = irk_sub[s2] == 1;
    }
    for (double k : sd_kappa_sub) {
        if (k > 1.0) batch.stat_sd_calls++;
        batch.stat_sd_kappa_max = std::max(batch.stat_sd_kappa_max, k);
    }
}
