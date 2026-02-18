#include "logh_gpu.h"
#include "logh.h"
#include "gbs.h"
#include "collisional_config.h"
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

void SubsystemBatch::unpack(const SubsystemManager& mgr, Particles& p) const {
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
}

// ============================================================================
// GPU-batched integration
// For now: host-side per-subsystem GBS (GPU kernel version requires
// GBS entirely in device code, which we'll implement after basic correctness)
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

    for (int s = 0; s < n_sub; s++) {
        int nb = h_nbod(s);
        int off = h_off(s);
        double dt = h_dt(s);

        // Build LogHState
        LogHState state{};
        state.n_bodies = nb;
        state.t = 0.0;
        state.use_aux_vel = false;

        for (int b = 0; b < nb; b++) {
            state.pos[b*3+0] = h_pos((off+b)*3+0);
            state.pos[b*3+1] = h_pos((off+b)*3+1);
            state.pos[b*3+2] = h_pos((off+b)*3+2);
            state.vel[b*3+0] = h_vel((off+b)*3+0);
            state.vel[b*3+1] = h_vel((off+b)*3+1);
            state.vel[b*3+2] = h_vel((off+b)*3+2);
            state.mass[b] = h_mass(off+b);
            // ext_accel stays zero (LogHState{} zero-initializes)
            // External perturbation handled by FSI kicks, not inside LogH
        }

        // Initialize binding energy B = -E₀ = -(T + U) (positive for bound systems)
        double T = logh_kinetic_energy(state);
        double W = logh_potential_energy(state, G);
        state.B = -W - T;  // B = |U| - T = -E₀

        double E0 = T + W;

        // Estimate initial ds from dt / Omega
        double Omega = T + state.B;
        double ds_init = (Omega > 1e-30) ? dt / Omega : dt;

        // Integrate to target time
        GBSIntegrator gbs(eta_gbs);
        gbs.integrate_to_time(state, dt, G, ds_init);

        // Store results
        for (int b = 0; b < nb; b++) {
            h_pos_out((off+b)*3+0) = state.pos[b*3+0];
            h_pos_out((off+b)*3+1) = state.pos[b*3+1];
            h_pos_out((off+b)*3+2) = state.pos[b*3+2];
            h_vel_out((off+b)*3+0) = state.vel[b*3+0];
            h_vel_out((off+b)*3+1) = state.vel[b*3+1];
            h_vel_out((off+b)*3+2) = state.vel[b*3+2];
        }

        double Ef = logh_total_energy(state, G);
        h_eerr(s) = std::fabs((Ef - E0) / (std::fabs(E0) + 1e-30));
        h_conv(s) = 1;
    }

    Kokkos::deep_copy(batch.flat_pos_out, h_pos_out);
    Kokkos::deep_copy(batch.flat_vel_out, h_vel_out);
    Kokkos::deep_copy(batch.energy_error, h_eerr);
    Kokkos::deep_copy(batch.converged, h_conv);
}
