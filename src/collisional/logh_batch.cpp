#include "small_for.h"
#include "device_staging.h"
#include "logh_batch.h"
#include "logh.h"
#include "gbs.h"
#include "collisional_config.h"
#include "post_newtonian.h"
#include "profiling.h"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <cstdio>
#include <string>
#include <chrono>

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
// LogH in BIFROST (FROST-CLUSTERS I, Rantala et al. 2024, App. A2; kappa = 1 for PN subsystems there).
static int gbs_step_limit(const LogHState& st, double dt, double G) {
    static const double per_orbit = std::getenv("NBODY_GBS_STEPS_PER_ORBIT")
                                        ? std::atof(std::getenv("NBODY_GBS_STEPS_PER_ORBIT")) : 64.0;
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
            P_min = std::min(P_min, 2.0 * M_PI * std::sqrt(a * a * a / mu));
        }
    if (per_orbit <= 0.0 || P_min == std::numeric_limits<double>::max()) return GBS_MAX_STEPS;
    const double n = per_orbit * std::fabs(dt) / P_min;
    return n > (double)GBS_MAX_STEPS ? (int)std::min(n, 4.0e6) : GBS_MAX_STEPS;
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
    // Buffer layout per member: x,y,z,vx,vy,vz,mass,spin_x,spin_y,spin_z,stype_as_double (11 doubles)
    constexpr int PACK_STRIDE = 11;
    const bool want_spins = (g_collisional.pn_order & PN_SPIN) != 0;
    const bool gather = !src || want_spins;
    has_stored_spins.assign(n_sub, 0);
    stored_spins.resize(n_sub * 6, 0.0);
    stored_stellar_type.assign(n_sub * 2, 0);
    Staging& st = staging();
    if (gather) {
        double t_x0 = wall_now();
        st.reserve(n_mem, PACK_STRIDE);
        for (int i = 0; i < n_mem; i++) st.h_idx(i) = member_indices[i];
        Kokkos::deep_copy(Kokkos::subview(st.d_idx, Kokkos::make_pair(0, n_mem)),
                          Kokkos::subview(st.h_idx, Kokkos::make_pair(0, n_mem)));
        auto d_idx = st.d_idx;
        auto d_buf = st.d_buf;
        auto px = p.x; auto py = p.y; auto pz = p.z;
        auto pvx = p.vx; auto pvy = p.vy; auto pvz = p.vz;
        auto pm = p.mass; auto pspin = p.spin_vec; auto pstype = p.stellar_type;
        small_for("pack_gather", n_mem, KOKKOS_LAMBDA(int i) {
            int k = d_idx(i);
            int base = i * PACK_STRIDE;
            d_buf(base + 0) = px(k);  d_buf(base + 1) = py(k);  d_buf(base + 2) = pz(k);
            d_buf(base + 3) = pvx(k); d_buf(base + 4) = pvy(k); d_buf(base + 5) = pvz(k);
            d_buf(base + 6) = pm(k);
            d_buf(base + 7) = pspin(k, 0); d_buf(base + 8) = pspin(k, 1); d_buf(base + 9) = pspin(k, 2);
            d_buf(base + 10) = static_cast<double>(pstype(k));
        });
        auto rng = Kokkos::make_pair((size_t)0, (size_t)n_mem * PACK_STRIDE);
        Kokkos::deep_copy(Kokkos::subview(st.h_buf, rng), Kokkos::subview(st.d_buf, rng));
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
    for (int s = 0; s < n_sub; s++) {
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
        if (gather && sub.n_bodies == 2) {
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
            }
        }
    }

    // Upload and scatter: write only modified particles back to device arrays
    t_x0 = wall_now();
    st.upload_idx(n_mem);
    st.upload_buf((size_t)n_mem * UNPACK_STRIDE);
    auto d_idx = st.d_idx;
    auto d_buf = st.d_buf;
    auto px = p.x; auto py = p.y; auto pz = p.z;
    auto pvx = p.vx; auto pvy = p.vy; auto pvz = p.vz;
    auto pcx = p.cvx; auto pcy = p.cvy; auto pcz = p.cvz;
    auto pm = p.mass;
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
    if (!spin_indices.empty()) {
        t_x0 = wall_now();
        int n_spin = (int)spin_indices.size();
        st.reserve(n_spin, 3);
        for (int i = 0; i < n_spin; i++) st.h_idx(i) = spin_indices[i];
        std::memcpy(st.h_buf.data(), spin_values.data(), n_spin * 3 * sizeof(double));
        st.upload_idx(n_spin);
        st.upload_buf((size_t)n_spin * 3);
        auto d_si = st.d_idx;
        auto d_sv = st.d_buf;
        auto pspin = p.spin_vec;
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
// IMPORTANT: The TTL leapfrog equations (BIFROST convention:
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

    // Subsystems are independent: integrate them on all host threads (one GBS integration is long
    // serial work; dynamic scheduling balances binaries against multiplets)
    using HostPolicy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>;
    auto integrate_one = [&](int s, int& gbs_sum, double& dE_max, double& dE_min, int& caps) {
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
        state.n_bodies = nb;
        state.t = 0.0;

        double c_code = g_collisional.c_code;
        int pn_order = g_collisional.pn_order;
        bool any_compact = false;

        for (int b = 0; b < nb; b++) {
            // Positions: subtract CM, normalize by R_scale
            state.pos[b*3+0] = (h_pos((off+b)*3+0) - cm_x) / R_scale;
            state.pos[b*3+1] = (h_pos((off+b)*3+1) - cm_y) / R_scale;
            state.pos[b*3+2] = (h_pos((off+b)*3+2) - cm_z) / R_scale;
            // Velocities: subtract CM, normalize by V_scale
            state.vel[b*3+0] = (h_vel((off+b)*3+0) - cm_vx) / V_scale;
            state.vel[b*3+1] = (h_vel((off+b)*3+1) - cm_vy) / V_scale;
            state.vel[b*3+2] = (h_vel((off+b)*3+2) - cm_vz) / V_scale;
            // Masses: normalize to sum = 1
            state.mass[b] = h_mass(off+b) / M_total;
            if (h_mass(off+b) >= g_collisional.m_pn) any_compact = true;
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
                for (int b = 0; b < nb; b++) {
                    state.aux_vel[b*3+0] = (batch.stored_aux_vel[av_off + b*3+0] - cm_vx) / V_scale;
                    state.aux_vel[b*3+1] = (batch.stored_aux_vel[av_off + b*3+1] - cm_vy) / V_scale;
                    state.aux_vel[b*3+2] = (batch.stored_aux_vel[av_off + b*3+2] - cm_vz) / V_scale;
                }
            } else {
                // First call: initialize w = v
                std::memcpy(state.aux_vel, state.vel, nb * 3 * sizeof(double));
            }
        } else {
            state.use_aux_vel = false;
            pn_order = 0;
        }

        // Load spin vectors for 2-body BH binaries; spin couplings (PN_ORDER bit 32, BIFROST sec. 3.7) need the
        // PN path (state.use_aux_vel), otherwise the spins are carried unchanged
        state.has_spins = false;
        state.c_code = c_norm;
        if (nb == 2 && batch.has_stored_spins[s] && state.use_aux_vel && (pn_order & PN_SPIN)) {
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
        double Omega_norm = T_norm + state.B;  // == -U_norm, always > 0

        // Save initial state for PN energy comparison after integration
        double state_init_pos[LOGH_MAX_BODIES * 3];
        double state_init_vel[LOGH_MAX_BODIES * 3];
        std::memcpy(state_init_pos, state.pos, nb * 3 * sizeof(double));
        std::memcpy(state_init_vel, state.vel, nb * 3 * sizeof(double));

        // Normalized time target
        double dt_norm = dt / T_scale;

        // ==================================================================
        // Step 3: Integrate in normalized units via GBS
        // ==================================================================
        int n_gbs_steps = 0;
        const int max_steps = gbs_step_limit(state, dt_norm, G_norm);
        {
            GBSIntegrator gbs(eta_gbs);
            double Omega_cur = logh_kinetic_energy(state) + state.B;
            double ds_est = (Omega_cur > 1e-30) ? dt_norm * Omega_cur : dt_norm;
            n_gbs_steps = gbs.integrate_to_time(state, dt_norm, G_norm, ds_est,
                                                 c_norm, pn_order, max_steps);
        }

        // ==================================================================
        // Step 4: Denormalize and store results
        // CM drifts uniformly: x_CM(t) = x_CM(0) + v_CM * dt
        // ==================================================================
        double dt_actual = dt;
        double cm_x_final = cm_x + cm_vx * dt_actual;
        double cm_y_final = cm_y + cm_vy * dt_actual;
        double cm_z_final = cm_z + cm_vz * dt_actual;

        for (int b = 0; b < nb; b++) {
            h_pos_out((off+b)*3+0) = state.pos[b*3+0] * R_scale + cm_x_final;
            h_pos_out((off+b)*3+1) = state.pos[b*3+1] * R_scale + cm_y_final;
            h_pos_out((off+b)*3+2) = state.pos[b*3+2] * R_scale + cm_z_final;
            h_vel_out((off+b)*3+0) = state.vel[b*3+0] * V_scale + cm_vx;
            h_vel_out((off+b)*3+1) = state.vel[b*3+1] * V_scale + cm_vy;
            h_vel_out((off+b)*3+2) = state.vel[b*3+2] * V_scale + cm_vz;
            h_mass(off + b) = state.mass[b] * M_total;
        }

        // Store auxiliary velocities and scales for persistence across calls.
        // (B is always recomputed fresh from current T+U — no persistence needed.)
        if (state.use_aux_vel) {
            batch.has_stored_aux_vel[s] = true;
            int av_off = batch.stored_aux_vel_offsets[s] * 3;
            for (int b = 0; b < nb; b++) {
                batch.stored_aux_vel[av_off + b*3+0] = state.aux_vel[b*3+0] * V_scale + cm_vx;
                batch.stored_aux_vel[av_off + b*3+1] = state.aux_vel[b*3+1] * V_scale + cm_vy;
                batch.stored_aux_vel[av_off + b*3+2] = state.aux_vel[b*3+2] * V_scale + cm_vz;
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
        if (any_compact && pn_order > 0 && c_norm > 0.0 && (pn_order & 1)) {
            double E_pn_0 = pn1_energy(state_init_pos, state_init_vel, state.mass,
                                        nb, G_norm, c_norm);
            double E_pn_f = pn1_energy(state.pos, state.vel, state.mass,
                                        nb, G_norm, c_norm);
            double E0_total = E0_norm + E_pn_0;
            double Ef_total = Ef_norm + E_pn_f;
            dE_rel = std::fabs((Ef_total - E0_total) / (std::fabs(E0_total) + 1e-30));
        } else {
            dE_rel = std::fabs((Ef_norm - E0_norm) / (std::fabs(E0_norm) + 1e-30));
        }
        // With PN terms there is no energy this check can hold to eta_restart: E_N + E_1PN is conserved
        // only to O(v^4/c^4), and 2PN/3PN/radiation reaction change it further. PN subsystems restart on
        // GBS non-convergence instead; their Newtonian energy change is booked as E_pn by the caller.
        // A step-limited integration stopped short of the end time: never accept it on its energy (the
        // Newtonian check passed it and the members silently lagged behind the rest of the system)
        bool converged = n_gbs_steps < max_steps;
        if (any_compact && pn_order > 0 && c_norm > 0.0) dE_rel = 0.0;
        if (!converged) {
            dE_rel = 1e30;
            // What stopped: the members, and the bound pairs with their orbits per interval (initial state)
            std::string msg = "      LogH step limit " + std::to_string(max_steps) + ": n=" + std::to_string(nb) +
                              " m[Msun]=";
            char buf[160];
            for (int b = 0; b < nb; b++) {
                std::snprintf(buf, sizeof(buf), "%s%.3g", b ? "," : "", h_mass(off + b));
                msg += buf;
            }
            for (int i = 0; i < nb; i++)
                for (int j = i + 1; j < nb; j++) {
                    double r2 = 0.0, v2 = 0.0;
                    for (int k = 0; k < 3; k++) {
                        const double dx = state_init_pos[j*3+k] - state_init_pos[i*3+k];
                        const double dv = state_init_vel[j*3+k] - state_init_vel[i*3+k];
                        r2 += dx * dx;
                        v2 += dv * dv;
                    }
                    const double mu = state.mass[i] + state.mass[j];   // G_norm = 1
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
    if (n_sub == 1) {   // most deep-level batches: no host threads to start
        max_dE = std::numeric_limits<double>::lowest();
        min_dE = std::numeric_limits<double>::max();
        integrate_one(0, total_gbs_steps, max_dE, min_dE, cap_hit);
    } else {
        Kokkos::parallel_reduce("logh_batch_host", HostPolicy(0, n_sub).set_chunk_size(1), integrate_one,
                                Kokkos::Sum<int>(total_gbs_steps), Kokkos::Max<double>(max_dE),
                                Kokkos::Min<double>(min_dE), Kokkos::Sum<int>(cap_hit));
    }
    g_logh_prof.integ += wall_now() - t_i0;
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
}
