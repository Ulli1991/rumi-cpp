#include "small_for.h"
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

SubsystemBatch g_logh_batch;

constexpr int GBS_MAX_STEPS = 1000;

// GBS step limit of one LogH call: 1000, or 64 steps per orbit of the tightest bound pair over the interval
// (a tight binary inside a fly-by or an unstable triple is integrated orbit by orbit: thousands of orbits per
// interval at coarse levels hit the fixed limit after a few per cent of the interval, and the retry split it
// into up to 256 pieces). Hard ceiling 4e6 steps; NBODY_GBS_STEPS_PER_ORBIT overrides 64 (0: fixed 1000).
// The cost stays ~ orbits per interval: the remedy for weakly perturbed inner binaries is the slow-down of their
// orbital motion (Mikkola & Aarseth 1996; SDAR, Wang, Nitadori & Makino 2020, MNRAS 493, 3398), which replaced
// LogH in BIFROST (FROST-CLUSTERS I, Rantala et al. 2024, App. A2; kappa = 1 for PN subsystems there).
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
// Allocate batch buffers
// ============================================================================

void SubsystemBatch::allocate(int n_sub, int total_bodies) {
    // Skip reallocation if existing views are large enough
    if (flat_mass.extent(0) >= (size_t)total_bodies &&
        n_bodies.extent(0) >= (size_t)n_sub) {
        n_subsystems = n_sub;
        return;
    }
    n_subsystems = n_sub;
    n_bodies      = Kokkos::View<int*>("batch_nbod", n_sub);
    offset        = Kokkos::View<int*>("batch_off", n_sub);
    dt_target     = Kokkos::View<double*>("batch_dt", n_sub);
    flat_pos      = Kokkos::View<double*>("batch_pos", total_bodies * 3);
    flat_vel      = Kokkos::View<double*>("batch_vel", total_bodies * 3);
    flat_mass     = Kokkos::View<double*>("batch_mass", total_bodies);
    flat_pos_out  = Kokkos::View<double*>("batch_pos_out", total_bodies * 3);
    flat_vel_out  = Kokkos::View<double*>("batch_vel_out", total_bodies * 3);
    energy_error  = Kokkos::View<double*>("batch_eerr", n_sub);
    converged     = Kokkos::View<int*>("batch_conv", n_sub);
}

// ============================================================================
// Pack subsystems into contiguous GPU arrays
// ============================================================================

void SubsystemBatch::pack(const SubsystemManager& mgr, const Particles& p,
                           double G, double dt) {
    (void)G;
    int n_sub = (int)mgr.active.size();
    if (n_sub == 0) return;

    // Count total bodies
    int total = 0;
    for (const auto& sub : mgr.active) total += sub.n_bodies;

    allocate(n_sub, total);

    // Use cached pid->idx from SubsystemManager
    const auto& pid_to_idx = mgr.pid_to_idx();

    // Collect unique particle indices for selective gather (avoid full-array D2H)
    // Cache for reuse in unpack()
    cached_member_indices.clear();
    cached_idx_to_local.clear();
    for (const auto& sub : mgr.active) {
        for (int b = 0; b < sub.n_bodies; b++) {
            int idx = pid_to_idx[sub.particle_ids[b]];
            if (cached_idx_to_local.find(idx) == cached_idx_to_local.end()) {
                cached_idx_to_local[idx] = (int)cached_member_indices.size();
                cached_member_indices.push_back(idx);
            }
        }
    }
    auto& member_indices = cached_member_indices;
    auto& idx_to_local = cached_idx_to_local;
    int n_mem = (int)member_indices.size();

    // Upload indices to device, gather only needed particle data
    // Buffer layout per member: x,y,z,vx,vy,vz,mass,spin_x,spin_y,spin_z,stype_as_double,radius (12 doubles)
    constexpr int PACK_STRIDE = 12;
    Kokkos::View<int*> d_idx("pack_idx", n_mem);
    Kokkos::View<double*> d_buf("pack_gather", n_mem * PACK_STRIDE);
    {
        auto h_idx = Kokkos::create_mirror_view(d_idx);
        for (int i = 0; i < n_mem; i++) h_idx(i) = member_indices[i];
        Kokkos::deep_copy(d_idx, h_idx);
    }
    auto px = p.x; auto py = p.y; auto pz = p.z;
    auto pvx = p.vx; auto pvy = p.vy; auto pvz = p.vz;
    auto pm = p.mass; auto pspin = p.spin_vec; auto pstype = p.stellar_type; auto prad = p.stellar_radius;
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
    auto h_buf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_buf);

    // Pack on host using gathered data
    auto h_nbod = Kokkos::create_mirror_view(n_bodies);
    auto h_off  = Kokkos::create_mirror_view(offset);
    auto h_dt   = Kokkos::create_mirror_view(dt_target);
    auto h_pos  = Kokkos::create_mirror_view(flat_pos);
    auto h_vel  = Kokkos::create_mirror_view(flat_vel);
    auto h_mass = Kokkos::create_mirror_view(flat_mass);

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
    stored_spins.resize(n_sub * 6, 0.0);
    has_stored_spins.resize(n_sub, false);
    stored_stellar_type.resize(n_sub * 2, 0);
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

    // Read spin vectors and stellar types from gathered buffer
    for (int s = 0; s < n_sub; s++) {
        const auto& sub = mgr.active[s];
        if (sub.n_bodies == 2) {
            int li0 = idx_to_local[pid_to_idx[sub.particle_ids[0]]];
            int li1 = idx_to_local[pid_to_idx[sub.particle_ids[1]]];
            stored_stellar_type[s*2+0] = static_cast<int>(h_buf(li0 * PACK_STRIDE + 10));
            stored_stellar_type[s*2+1] = static_cast<int>(h_buf(li1 * PACK_STRIDE + 10));
            bool bh0 = (stored_stellar_type[s*2+0] == 14);
            bool bh1 = (stored_stellar_type[s*2+1] == 14);
            if (bh0 || bh1) {
                has_stored_spins[s] = true;
                stored_spins[s*6+0] = h_buf(li0 * PACK_STRIDE + 7);
                stored_spins[s*6+1] = h_buf(li0 * PACK_STRIDE + 8);
                stored_spins[s*6+2] = h_buf(li0 * PACK_STRIDE + 9);
                stored_spins[s*6+3] = h_buf(li1 * PACK_STRIDE + 7);
                stored_spins[s*6+4] = h_buf(li1 * PACK_STRIDE + 8);
                stored_spins[s*6+5] = h_buf(li1 * PACK_STRIDE + 9);
            }
        }
    }

    flat_radius.assign(total, 0.0);
    flat_stype.assign(total, 0);
    int off = 0;
    for (int s = 0; s < n_sub; s++) {
        const auto& sub = mgr.active[s];
        h_nbod(s) = sub.n_bodies;
        h_off(s) = off;
        h_dt(s) = dt;

        // Pack pos/vel/mass from gathered buffer
        for (int b = 0; b < sub.n_bodies; b++) {
            int li = idx_to_local[pid_to_idx[sub.particle_ids[b]]];
            int base = li * PACK_STRIDE;
            h_pos((off + b) * 3 + 0) = h_buf(base + 0);
            h_pos((off + b) * 3 + 1) = h_buf(base + 1);
            h_pos((off + b) * 3 + 2) = h_buf(base + 2);
            h_vel((off + b) * 3 + 0) = h_buf(base + 3);
            h_vel((off + b) * 3 + 1) = h_buf(base + 4);
            h_vel((off + b) * 3 + 2) = h_buf(base + 5);
            h_mass(off + b) = h_buf(base + 6);
            flat_stype[off + b] = static_cast<int>(h_buf(base + 10));
            flat_radius[off + b] = h_buf(base + 11);
        }
        off += sub.n_bodies;
    }

    // Copy to device
    Kokkos::deep_copy(n_bodies, h_nbod);
    Kokkos::deep_copy(offset, h_off);
    Kokkos::deep_copy(dt_target, h_dt);
    Kokkos::deep_copy(flat_pos, h_pos);
    Kokkos::deep_copy(flat_vel, h_vel);
    Kokkos::deep_copy(flat_mass, h_mass);
}

// ============================================================================
// Unpack results back into Particles
// ============================================================================

void SubsystemBatch::unpack(SubsystemManager& mgr, Particles& p) {
    int n_sub = (int)mgr.active.size();
    if (n_sub == 0) return;

    const auto& pid_to_idx = mgr.pid_to_idx();

    // D2H the small LogH output arrays (subsystem-sized, not N-sized)
    auto h_pos_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), flat_pos_out);
    auto h_vel_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), flat_vel_out);
    auto h_mass_flat = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), flat_mass);
    auto h_off = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), offset);

    // Reuse cached member indices from pack() (avoids recomputation and rehashing)
    constexpr int UNPACK_STRIDE = 7;
    auto& member_indices = cached_member_indices;
    auto& idx_to_local = cached_idx_to_local;
    int n_mem = (int)member_indices.size();

    // Build scatter buffer from LogH output
    std::vector<double> scatter_buf(n_mem * UNPACK_STRIDE);
    for (int s = 0; s < n_sub; s++) {
        const auto& sub = mgr.active[s];
        int off = h_off(s);
        for (int b = 0; b < sub.n_bodies; b++) {
            int li = idx_to_local[pid_to_idx[sub.particle_ids[b]]];
            int base = li * UNPACK_STRIDE;
            scatter_buf[base + 0] = h_pos_out((off + b) * 3 + 0);
            scatter_buf[base + 1] = h_pos_out((off + b) * 3 + 1);
            scatter_buf[base + 2] = h_pos_out((off + b) * 3 + 2);
            scatter_buf[base + 3] = h_vel_out((off + b) * 3 + 0);
            scatter_buf[base + 4] = h_vel_out((off + b) * 3 + 1);
            scatter_buf[base + 5] = h_vel_out((off + b) * 3 + 2);
            scatter_buf[base + 6] = h_mass_flat(off + b);
        }
    }

    // Upload scatter buffer + indices to device
    Kokkos::View<int*> d_idx("unpack_idx", n_mem);
    Kokkos::View<double*> d_buf("unpack_scatter", n_mem * UNPACK_STRIDE);
    {
        auto h_idx = Kokkos::create_mirror_view(d_idx);
        auto h_buf = Kokkos::create_mirror_view(d_buf);
        for (int i = 0; i < n_mem; i++) h_idx(i) = member_indices[i];
        std::memcpy(h_buf.data(), scatter_buf.data(), n_mem * UNPACK_STRIDE * sizeof(double));
        Kokkos::deep_copy(d_idx, h_idx);
        Kokkos::deep_copy(d_buf, h_buf);
    }

    // Scatter kernel: write only modified particles back to device arrays
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

    // Write evolved spins back to particles via small scatter
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
        int n_spin = (int)spin_indices.size();
        Kokkos::View<int*> d_si("spin_idx", n_spin);
        Kokkos::View<double*> d_sv("spin_val", n_spin * 3);
        auto h_si = Kokkos::create_mirror_view(d_si);
        auto h_sv = Kokkos::create_mirror_view(d_sv);
        for (int i = 0; i < n_spin; i++) h_si(i) = spin_indices[i];
        std::memcpy(h_sv.data(), spin_values.data(), n_spin * 3 * sizeof(double));
        Kokkos::deep_copy(d_si, h_si);
        Kokkos::deep_copy(d_sv, h_sv);
        auto pspin = p.spin_vec;
        small_for("spin_scatter", n_spin, KOKKOS_LAMBDA(int i) {
            int k = d_si(i);
            pspin(k, 0) = d_sv(i * 3 + 0);
            pspin(k, 1) = d_sv(i * 3 + 1);
            pspin(k, 2) = d_sv(i * 3 + 2);
        });
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

    // Copy to host for CPU-side GBS (GPU kernel version in future)
    auto h_nbod = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), batch.n_bodies);
    auto h_off  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), batch.offset);
    auto h_dt   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), batch.dt_target);
    auto h_pos  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), batch.flat_pos);
    auto h_vel  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), batch.flat_vel);
    auto h_mass = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), batch.flat_mass);
    auto h_pos_out = Kokkos::create_mirror_view(batch.flat_pos_out);
    auto h_vel_out = Kokkos::create_mirror_view(batch.flat_vel_out);
    auto h_eerr = Kokkos::create_mirror_view(batch.energy_error);
    auto h_conv = Kokkos::create_mirror_view(batch.converged);

    int total_gbs_steps = 0;
    std::vector<int> gbs_per_sub(n_sub, 0);   // per-depth load-balance statistics
    batch.t_reached.assign(n_sub, 0.0);   // read by the subsystem-advance check
    double max_dE = 0.0;
    double min_dE = 1e30;
    int cap_hit = 0;
    batch.sd_jump.assign(n_sub, 0.0);
    std::vector<double> sd_kappa_sub(n_sub, 0.0);

    // Subsystems are independent: integrate them on all host threads (one GBS integration is long
    // serial work; dynamic scheduling balances binaries against multiplets)
    using HostPolicy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>;
    auto integrate_one = [&](int s, int& gbs_sum, double& dE_max, double& dE_min, int& caps) {
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
        // SDAR (FROST-CLUSTERS I App. A2): slow-down of weakly perturbed inner binaries; kappa = 1 throughout
        // for PN subsystems (any member above M_PN with PN on; FC-I sec. A2) and for 2-body subsystems (no
        // perturber inside the subsystem: the external tide comes from the FSI kicks, which need the true phase)
        const bool sd_on = g_collisional.sdar && !state.use_aux_vel && nb >= 3;
        SdarParams sd_par;
        if (sd_on) {
            sd_par.k_ref = g_collisional.sdar_kref;
            sd_par.c = g_collisional.sdar_c;
            sd_par.steps_per_orbit = gbs_steps_per_orbit();   // the limit follows kappa during the call
            constexpr double R_sun_to_pc = 2.254e-8;   // as mergers.cpp
            for (int b = 0; b < nb; b++) {
                // Collision distance r_a + r_b (FC-I App. B3): the stellar radius; a BH its ISCO 6 G m / c^2
                // (BH-BH: sum of the ISCOs, >= FC-I's larger ISCO)
                const double r_pc = batch.flat_stype[off + b] == 14
                                        ? 6.0 * G * h_mass(off + b) / (c_code * c_code)
                                        : batch.flat_radius[off + b] * R_sun_to_pc;
                sd_par.coll_radius[b] = r_pc / R_scale;
            }
            logh_sd_update(state, G_norm, sd_par);   // initial kappa: B = -H_sd
        }

        int n_gbs_steps = 0;
        const int max_steps = gbs_step_limit(state, dt_norm, G_norm);
        bool reached = false;
        {
            GBSIntegrator gbs(eta_gbs);
            if (sd_on) gbs.sdar = &sd_par;
            double Omega_cur = logh_omega(state);
            double ds_est = (Omega_cur > 1e-30) ? dt_norm * Omega_cur : dt_norm;
            n_gbs_steps = gbs.integrate_to_time(state, dt_norm, G_norm, ds_est,
                                                 c_norm, pn_order, max_steps);
            reached = gbs.reached;
        }
        // Remove the slow-down: the last jump back to H (W20 Eq. 24); then T + U + B is the integration error of
        // the slowed Hamiltonian over the call (W20 Eq. 25), and sd_jump the booked physical energy change
        if (sd_on) {
            logh_sd_update(state, G_norm, sd_par, true);
            batch.sd_jump[s] = state.sd_jump * E_scale;
            sd_kappa_sub[s] = state.sd_kappa_peak;
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
        } else if (sd_on) {
            dE_rel = std::fabs((Ef_norm + state.B) / (std::fabs(E0_norm) + 1e-30));
        } else {
            dE_rel = std::fabs((Ef_norm - E0_norm) / (std::fabs(E0_norm) + 1e-30));
        }
        // With PN terms there is no energy this check can hold to eta_restart: E_N + E_1PN is conserved
        // only to O(v^4/c^4), and 2PN/3PN/radiation reaction change it further. PN subsystems restart on
        // GBS non-convergence instead; their Newtonian energy change is booked as E_pn by the caller.
        // A step-limited integration stopped short of the end time: never accept it on its energy (the
        // Newtonian check passed it and the members silently lagged behind the rest of the system)
        bool converged = sd_on ? reached : n_gbs_steps < max_steps;
        if (any_compact && pn_order > 0 && c_norm > 0.0) dE_rel = 0.0;
        if (!converged) {
            dE_rel = 1e30;
            // What stopped: the members, and the bound pairs with their orbits per interval (initial state)
            std::string msg = "      LogH step limit " + std::to_string(sd_on ? n_gbs_steps : max_steps) + ": n=" + std::to_string(nb) +
                              " m[Msun]=";
            char buf[160];
            for (int b = 0; b < nb; b++) {
                std::snprintf(buf, sizeof(buf), "%s%.3g", b ? "," : "", h_mass(off + b));
                msg += buf;
            }
            for (int i = 0; i < nb; i++)
                for (int j = i + 1; j < nb; j++) {
                    double r2 = 0.0, v2 = 0.0, dx[3], dv[3];
                    for (int k = 0; k < 3; k++) {
                        dx[k] = state_init_pos[j*3+k] - state_init_pos[i*3+k];
                        dv[k] = state_init_vel[j*3+k] - state_init_vel[i*3+k];
                        r2 += dx[k] * dx[k];
                        v2 += dv[k] * dv[k];
                    }
                    const double mu = state.mass[i] + state.mass[j];   // G_norm = 1
                    const double eps = 0.5 * v2 - mu / std::sqrt(r2);
                    const double a = -mu / (2.0 * eps);
                    if (sd_on && eps < 0.0) {   // SDAR: pericentre over the collision distance (< 1: kappa = 1)
                        const double hx = dx[1]*dv[2] - dx[2]*dv[1], hy = dx[2]*dv[0] - dx[0]*dv[2];
                        const double hz = dx[0]*dv[1] - dx[1]*dv[0];
                        const double e = std::sqrt(std::max(0.0, 1.0 + 2.0 * eps * (hx*hx + hy*hy + hz*hz) / (mu * mu)));
                        const double rc = sd_par.coll_radius[i] + sd_par.coll_radius[j];
                        std::snprintf(buf, sizeof(buf), " e=%.4f r_p/r_coll=%.3g", e, rc > 0.0 ? a * (1.0 - e) / rc : -1.0);
                        msg += buf;
                    }
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
            for (int i = 0; i < nb; i++)
                for (int j = i + 1; j < nb; j++) {
                    double r[3], v[3], r2 = 0.0, v2 = 0.0;
                    for (int k = 0; k < 3; k++) {
                        r[k] = state_init_pos[j*3+k] - state_init_pos[i*3+k];
                        v[k] = state_init_vel[j*3+k] - state_init_vel[i*3+k];
                        r2 += r[k] * r[k];
                        v2 += v[k] * v[k];
                    }
                    const double mu = state.mass[i] + state.mass[j];
                    const double eps = 0.5 * v2 - mu / std::sqrt(r2);
                    if (eps >= 0.0) continue;
                    const double hx = r[1]*v[2] - r[2]*v[1], hy = r[2]*v[0] - r[0]*v[2], hz = r[0]*v[1] - r[1]*v[0];
                    const double a = -mu / (2.0 * eps);
                    const double e = std::sqrt(std::max(0.0, 1.0 + 2.0 * eps * (hx*hx + hy*hy + hz*hz) / (mu * mu)));
                    const double rc = sd_par.coll_radius[i] + sd_par.coll_radius[j];
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
    };
    if (n_sub == 1) {   // most deep-level batches: no host threads to start
        max_dE = std::numeric_limits<double>::lowest();
        min_dE = std::numeric_limits<double>::max();
        integrate_one(0, total_gbs_steps, max_dE, min_dE, cap_hit);
    } else {
        Kokkos::parallel_reduce("logh_batch_host", HostPolicy(0, n_sub).set_chunk_size(1), integrate_one,
                                Kokkos::Sum<int>(total_gbs_steps), Kokkos::Max<double>(max_dE),
                                Kokkos::Min<double>(min_dE), Kokkos::Sum<int>(cap_hit));
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
    for (double k : sd_kappa_sub) {
        if (k > 1.0) batch.stat_sd_calls++;
        batch.stat_sd_kappa_max = std::max(batch.stat_sd_kappa_max, k);
    }

    Kokkos::deep_copy(batch.flat_pos_out, h_pos_out);
    Kokkos::deep_copy(batch.flat_vel_out, h_vel_out);
    Kokkos::deep_copy(batch.flat_mass, h_mass);
    Kokkos::deep_copy(batch.energy_error, h_eerr);
    Kokkos::deep_copy(batch.converged, h_conv);
}
