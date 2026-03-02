#include "logh_batch.h"
#include "logh.h"
#include "gbs.h"
#include "collisional_config.h"
#include "post_newtonian.h"
#include <Kokkos_Core.hpp>
#include <cstring>

SubsystemBatch g_logh_batch;

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
    flat_ext_accel = Kokkos::View<double*>("batch_ext", total_bodies * 3);
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
    // Buffer layout per member: x,y,z,vx,vy,vz,mass,spin_x,spin_y,spin_z,stype_as_double (11 doubles)
    constexpr int PACK_STRIDE = 11;
    Kokkos::View<int*> d_idx("pack_idx", n_mem);
    Kokkos::View<double*> d_buf("pack_gather", n_mem * PACK_STRIDE);
    {
        auto h_idx = Kokkos::create_mirror_view(d_idx);
        for (int i = 0; i < n_mem; i++) h_idx(i) = member_indices[i];
        Kokkos::deep_copy(d_idx, h_idx);
    }
    auto px = p.x; auto py = p.y; auto pz = p.z;
    auto pvx = p.vx; auto pvy = p.vy; auto pvz = p.vz;
    auto pm = p.mass; auto pspin = p.spin_vec; auto pstype = p.stellar_type;
    Kokkos::parallel_for("pack_gather", n_mem, KOKKOS_LAMBDA(int i) {
        int k = d_idx(i);
        int base = i * PACK_STRIDE;
        d_buf(base + 0) = px(k);  d_buf(base + 1) = py(k);  d_buf(base + 2) = pz(k);
        d_buf(base + 3) = pvx(k); d_buf(base + 4) = pvy(k); d_buf(base + 5) = pvz(k);
        d_buf(base + 6) = pm(k);
        d_buf(base + 7) = pspin(k, 0); d_buf(base + 8) = pspin(k, 1); d_buf(base + 9) = pspin(k, 2);
        d_buf(base + 10) = static_cast<double>(pstype(k));
    });
    auto h_buf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_buf);

    // Pack on host using gathered data
    auto h_nbod = Kokkos::create_mirror_view(n_bodies);
    auto h_off  = Kokkos::create_mirror_view(offset);
    auto h_dt   = Kokkos::create_mirror_view(dt_target);
    auto h_pos  = Kokkos::create_mirror_view(flat_pos);
    auto h_vel  = Kokkos::create_mirror_view(flat_vel);
    auto h_mass = Kokkos::create_mirror_view(flat_mass);
    auto h_ext  = Kokkos::create_mirror_view(flat_ext_accel);

    // Copy persistent B values, auxiliary velocities, and normalization scales from subsystems
    stored_B.resize(n_sub);
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
        stored_B[s] = mgr.active[s].B_phys;
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

    int off = 0;
    for (int s = 0; s < n_sub; s++) {
        const auto& sub = mgr.active[s];
        h_nbod(s) = sub.n_bodies;
        h_off(s) = off;
        h_dt(s) = dt;

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
            h_ext((off + b) * 3 + 0) = 0.0;
            h_ext((off + b) * 3 + 1) = 0.0;
            h_ext((off + b) * 3 + 2) = 0.0;
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
    Kokkos::deep_copy(flat_ext_accel, h_ext);
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
    auto pm = p.mass;
    Kokkos::parallel_for("unpack_scatter", n_mem, KOKKOS_LAMBDA(int i) {
        int k = d_idx(i);
        int base = i * UNPACK_STRIDE;
        px(k) = d_buf(base + 0); py(k) = d_buf(base + 1); pz(k) = d_buf(base + 2);
        pvx(k) = d_buf(base + 3); pvy(k) = d_buf(base + 4); pvz(k) = d_buf(base + 5);
        pm(k) = d_buf(base + 6);
    });

    // Write persistent B, auxiliary velocities, and scales back to subsystems
    for (int s = 0; s < n_sub; s++) {
        mgr.active[s].B_phys = stored_B[s];
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
        Kokkos::parallel_for("spin_scatter", n_spin, KOKKOS_LAMBDA(int i) {
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
    double max_dE = 0.0;
    double min_dE = 1e30;
    int cap_hit = 0;

    for (int s = 0; s < n_sub; s++) {
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

        // Load spin vectors for 2-body BH binaries
        state.has_spins = false;
        state.c_code = c_norm;
        if (nb == 2 && batch.has_stored_spins[s]) {
            // Spins are dimensionless chi vectors — no normalization needed
            state.has_spins = true;
            state.spin1[0] = batch.stored_spins[s*6+0];
            state.spin1[1] = batch.stored_spins[s*6+1];
            state.spin1[2] = batch.stored_spins[s*6+2];
            state.spin2[0] = batch.stored_spins[s*6+3];
            state.spin2[1] = batch.stored_spins[s*6+4];
            state.spin2[2] = batch.stored_spins[s*6+5];
        }

        // Initialize B in normalized units.
        // Persistent B is stored directly in these fixed normalized units — no
        // conversion needed since scales are fixed. This eliminates the drift
        // that occurred when E_scale changed between calls.
        double T_norm = logh_kinetic_energy(state);
        double U_norm = logh_potential_energy(state, G_norm);

        if (std::isnan(batch.stored_B[s])) {
            // First call: B = -(T+U). PN forces adjust B naturally via dB/ds.
            state.B = -U_norm - T_norm;
        } else {
            // Persistent B — already in these fixed normalized units, no conversion
            state.B = batch.stored_B[s];
        }

        double E0_norm = T_norm + U_norm;
        double Omega_norm = T_norm + state.B;

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
        {
            GBSIntegrator gbs(eta_gbs);
            double Omega_cur = logh_kinetic_energy(state) + state.B;
            double ds_est = (Omega_cur > 1e-30) ? dt_norm * Omega_cur : dt_norm;
            n_gbs_steps = gbs.integrate_to_time(state, dt_norm, G_norm, ds_est,
                                                 c_norm, pn_order, 1000);
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

        // Store updated B and auxiliary velocities for persistence across calls
        // B stays in fixed normalized units — no E_scale conversion needed
        batch.stored_B[s] = state.B;
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
        h_eerr(s) = dE_rel;
        h_conv(s) = (n_gbs_steps < 1000) ? 1 : 0;

        total_gbs_steps += n_gbs_steps;
        if (dE_rel > max_dE) max_dE = dE_rel;
        if (dE_rel < min_dE) min_dE = dE_rel;
        if (n_gbs_steps >= 1000) cap_hit++;
    }

    // Store stats for caller (printed in step summary, not here)
    batch.stat_total_gbs = total_gbs_steps;
    batch.stat_worst_dE  = max_dE;
    batch.stat_best_dE   = min_dE;
    batch.stat_cap_hit   = cap_hit;

    Kokkos::deep_copy(batch.flat_pos_out, h_pos_out);
    Kokkos::deep_copy(batch.flat_vel_out, h_vel_out);
    Kokkos::deep_copy(batch.flat_mass, h_mass);
    Kokkos::deep_copy(batch.energy_error, h_eerr);
    Kokkos::deep_copy(batch.converged, h_conv);
}
