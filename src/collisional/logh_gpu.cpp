#include "logh_gpu.h"
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
    int n_sub = (int)mgr.active.size();
    if (n_sub == 0) return;

    // Count total bodies
    int total = 0;
    for (const auto& sub : mgr.active) total += sub.n_bodies;

    allocate(n_sub, total);

    // Copy particle data to host
    auto h_x  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
    auto h_y  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
    auto h_z  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
    auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
    auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
    auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
    auto h_m  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
    auto h_pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);

    // Build pid->idx
    int max_pid = 0;
    for (int i = 0; i < p.N; i++) max_pid = std::max(max_pid, h_pid(i));
    std::vector<int> pid_to_idx(max_pid + 1, -1);
    for (int i = 0; i < p.N; i++) pid_to_idx[h_pid(i)] = i;

    // Pack on host
    auto h_nbod = Kokkos::create_mirror_view(n_bodies);
    auto h_off  = Kokkos::create_mirror_view(offset);
    auto h_dt   = Kokkos::create_mirror_view(dt_target);
    auto h_pos  = Kokkos::create_mirror_view(flat_pos);
    auto h_vel  = Kokkos::create_mirror_view(flat_vel);
    auto h_mass = Kokkos::create_mirror_view(flat_mass);
    auto h_ext  = Kokkos::create_mirror_view(flat_ext_accel);

    // Copy persistent B values from subsystems
    stored_B.resize(n_sub);
    for (int s = 0; s < n_sub; s++) {
        stored_B[s] = mgr.active[s].B_phys;
    }

    int off = 0;
    for (int s = 0; s < n_sub; s++) {
        const auto& sub = mgr.active[s];
        h_nbod(s) = sub.n_bodies;
        h_off(s) = off;
        h_dt(s) = dt;

        for (int b = 0; b < sub.n_bodies; b++) {
            int idx = pid_to_idx[sub.particle_ids[b]];
            h_pos((off + b) * 3 + 0) = h_x(idx);
            h_pos((off + b) * 3 + 1) = h_y(idx);
            h_pos((off + b) * 3 + 2) = h_z(idx);
            h_vel((off + b) * 3 + 0) = h_vx(idx);
            h_vel((off + b) * 3 + 1) = h_vy(idx);
            h_vel((off + b) * 3 + 2) = h_vz(idx);
            h_mass(off + b) = h_m(idx);
            // No external perturbation inside LogH — handled by FSI kicks
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

    auto h_pos_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), flat_pos_out);
    auto h_vel_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), flat_vel_out);

    auto h_x  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
    auto h_y  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
    auto h_z  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
    auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
    auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
    auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
    auto h_pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);

    int max_pid = 0;
    for (int i = 0; i < p.N; i++) max_pid = std::max(max_pid, h_pid(i));
    std::vector<int> pid_to_idx(max_pid + 1, -1);
    for (int i = 0; i < p.N; i++) pid_to_idx[h_pid(i)] = i;

    auto h_off = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), offset);

    for (int s = 0; s < n_sub; s++) {
        const auto& sub = mgr.active[s];
        int off = h_off(s);

        for (int b = 0; b < sub.n_bodies; b++) {
            int idx = pid_to_idx[sub.particle_ids[b]];
            h_x(idx)  = h_pos_out((off + b) * 3 + 0);
            h_y(idx)  = h_pos_out((off + b) * 3 + 1);
            h_z(idx)  = h_pos_out((off + b) * 3 + 2);
            h_vx(idx) = h_vel_out((off + b) * 3 + 0);
            h_vy(idx) = h_vel_out((off + b) * 3 + 1);
            h_vz(idx) = h_vel_out((off + b) * 3 + 2);
        }
    }

    Kokkos::deep_copy(p.x, h_x);
    Kokkos::deep_copy(p.y, h_y);
    Kokkos::deep_copy(p.z, h_z);
    Kokkos::deep_copy(p.vx, h_vx);
    Kokkos::deep_copy(p.vy, h_vy);
    Kokkos::deep_copy(p.vz, h_vz);

    // Write persistent B back to subsystems
    for (int s = 0; s < n_sub; s++) {
        mgr.active[s].B_phys = stored_B[s];
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

    for (int s = 0; s < n_sub; s++) {
        int nb = h_nbod(s);
        int off = h_off(s);
        double dt = h_dt(s);

        // ==================================================================
        // Step 1: Compute normalization scales
        // V_scale = sqrt(2*T_CM / M_total), R_scale = G*M/V^2, T_scale = R/V
        // In normalized units: G'=1, M'=1, Ω'~O(1)
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

        // Kinetic energy in CM frame
        double T_cm = 0;
        for (int b = 0; b < nb; b++) {
            double dvx = h_vel((off+b)*3+0) - cm_vx;
            double dvy = h_vel((off+b)*3+1) - cm_vy;
            double dvz = h_vel((off+b)*3+2) - cm_vz;
            T_cm += 0.5 * h_mass(off+b) * (dvx*dvx + dvy*dvy + dvz*dvz);
        }

        // Normalization scales
        double V_scale_sq = 2.0 * T_cm / M_total;
        double V_scale = std::sqrt(V_scale_sq);
        if (V_scale < 1e-30) V_scale = 1.0;  // degenerate case
        V_scale_sq = V_scale * V_scale;

        double R_scale = G * M_total / V_scale_sq;
        double T_scale = R_scale / V_scale;
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

        // === DEBUG: compute omega (eccentricity vector angle) before integration ===
        double omega_before = 0.0;
        if (nb == 2) {
            double rx = state.pos[1*3+0] - state.pos[0*3+0];
            double ry = state.pos[1*3+1] - state.pos[0*3+1];
            double rz = state.pos[1*3+2] - state.pos[0*3+2];
            double vx = state.vel[1*3+0] - state.vel[0*3+0];
            double vy = state.vel[1*3+1] - state.vel[0*3+1];
            double vz = state.vel[1*3+2] - state.vel[0*3+2];
            double rmag = std::sqrt(rx*rx + ry*ry + rz*rz);
            double mu_bin = G_norm * (state.mass[0] + state.mass[1]);
            double Lz = rx*vy - ry*vx;
            double ex = vy*Lz/mu_bin - rx/rmag;
            double ey = -vx*Lz/mu_bin - ry/rmag;
            omega_before = std::atan2(ey, ex);
            double emag = std::sqrt(ex*ex + ey*ey);
            double L_mag = std::sqrt(Lz*Lz + (rz*vx - rx*vz)*(rz*vx - rx*vz)
                                     + (rx*vy - ry*vx)*(rx*vy - ry*vx));
            // Extra: check angular momentum in normalized units
            // Not using L_mag for now, just omega_before
        }

        // Enable auxiliary velocity for PN when compact objects present
        if (any_compact && pn_order > 0 && c_code > 0.0) {
            state.use_aux_vel = true;
            std::memcpy(state.aux_vel, state.vel, nb * 3 * sizeof(double));
        } else {
            state.use_aux_vel = false;
            pn_order = 0;
        }

        // Initialize B in normalized units.
        // If we have a stored B from a previous LogH call, convert from physical
        // to normalized units. Otherwise compute from scratch: B = -(T + U).
        // Persistent B is critical for PN: re-initializing discards PN energy
        // tracking and causes monotonic secular drift.
        double T_norm = logh_kinetic_energy(state);
        double U_norm = logh_potential_energy(state, G_norm);
        double E_scale = M_total * V_scale_sq;  // energy normalization factor

        if (std::isnan(batch.stored_B[s])) {
            // First call for this subsystem: B = -(T+U) in normalized units
            state.B = -U_norm - T_norm;
        } else {
            // Use persistent B from previous call, converted to current normalization
            state.B = batch.stored_B[s] / E_scale;
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

        // ds estimate in normalized units (BIFROST: ds = dt * Omega)
        double ds_init = (Omega_norm > 1e-30) ? dt_norm * Omega_norm : dt_norm;

        // ==================================================================
        // Step 3: Integrate in normalized units
        // ==================================================================
        GBSIntegrator gbs(eta_gbs);
        int n_gbs_steps = gbs.integrate_to_time(state, dt_norm, G_norm, ds_init,
                                                 c_norm, pn_order);

        // === DEBUG: compute omega after integration ===
        double omega_after = 0.0;
        if (nb == 2) {
            double rx = state.pos[1*3+0] - state.pos[0*3+0];
            double ry = state.pos[1*3+1] - state.pos[0*3+1];
            double rz = state.pos[1*3+2] - state.pos[0*3+2];
            double vx = state.vel[1*3+0] - state.vel[0*3+0];
            double vy = state.vel[1*3+1] - state.vel[0*3+1];
            double vz = state.vel[1*3+2] - state.vel[0*3+2];
            double rmag = std::sqrt(rx*rx + ry*ry + rz*rz);
            double mu_bin = G_norm * (state.mass[0] + state.mass[1]);
            double Lz = rx*vy - ry*vx;
            double ex = vy*Lz/mu_bin - rx/rmag;
            double ey = -vx*Lz/mu_bin - ry/rmag;
            omega_after = std::atan2(ey, ex);
            double emag = std::sqrt(ex*ex + ey*ey);

            // Compute angular momentum before and after
            double L_z_phys_before = 0.0;  // computed from input state
            for (int b = 0; b < nb; b++) {
                double px = (h_pos((off+b)*3+0) - cm_x);
                double py = (h_pos((off+b)*3+1) - cm_y);
                double pvx = (h_vel((off+b)*3+0) - cm_vx);
                double pvy = (h_vel((off+b)*3+1) - cm_vy);
                L_z_phys_before += h_mass(off+b) * (px*pvy - py*pvx);
            }

            double L_z_norm_after = 0.0;
            for (int b = 0; b < nb; b++) {
                L_z_norm_after += state.mass[b] * (state.pos[b*3+0]*state.vel[b*3+1]
                                                   - state.pos[b*3+1]*state.vel[b*3+0]);
            }

            double dw = omega_after - omega_before;
            if (dw < -M_PI) dw += 2.0 * M_PI;
            if (dw > M_PI) dw -= 2.0 * M_PI;
            std::printf("    LogH_debug: omega_before=%.6f  omega_after=%.6f  dw=%.6e"
                        "  e=%.9f  dt_norm=%.4f  Lz_norm=%.10e  n_gbs=%d\n",
                        omega_before, omega_after, dw, emag, dt_norm,
                        L_z_norm_after, n_gbs_steps);
        }

        // ==================================================================
        // Step 4: Denormalize and store results
        // CM drifts uniformly: x_CM(t) = x_CM(0) + v_CM * dt
        // ==================================================================
        double cm_x_final = cm_x + cm_vx * dt;
        double cm_y_final = cm_y + cm_vy * dt;
        double cm_z_final = cm_z + cm_vz * dt;

        for (int b = 0; b < nb; b++) {
            h_pos_out((off+b)*3+0) = state.pos[b*3+0] * R_scale + cm_x_final;
            h_pos_out((off+b)*3+1) = state.pos[b*3+1] * R_scale + cm_y_final;
            h_pos_out((off+b)*3+2) = state.pos[b*3+2] * R_scale + cm_z_final;
            h_vel_out((off+b)*3+0) = state.vel[b*3+0] * V_scale + cm_vx;
            h_vel_out((off+b)*3+1) = state.vel[b*3+1] * V_scale + cm_vy;
            h_vel_out((off+b)*3+2) = state.vel[b*3+2] * V_scale + cm_vz;
        }

        // Store updated B back to physical units for persistence across calls
        batch.stored_B[s] = state.B * E_scale;

        // Energy error in normalized units (dimensionless)
        // Include PN1 energy correction when PN is active for meaningful dE
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
        h_conv(s) = 1;

        total_gbs_steps += n_gbs_steps;
        if (dE_rel > max_dE) max_dE = dE_rel;
        if (dE_rel < min_dE) min_dE = dE_rel;
    }

    // Print compact summary (one line per batch)
    if (n_sub > 0) {
        std::printf("  LogH: %d subs, %d gbs_steps, dE=[%.2e, %.2e]\n",
                    n_sub, total_gbs_steps, min_dE, max_dE);
    }

    Kokkos::deep_copy(batch.flat_pos_out, h_pos_out);
    Kokkos::deep_copy(batch.flat_vel_out, h_vel_out);
    Kokkos::deep_copy(batch.energy_error, h_eerr);
    Kokkos::deep_copy(batch.converged, h_conv);
}
