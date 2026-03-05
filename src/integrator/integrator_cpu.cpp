#include "integrator_cpu.h"
#include "integrator.h"
#include "hhs_counters.h"
#include "constants.h"
#include "profiling.h"
#include "forces.h"
#ifdef ENABLE_COLLISIONAL
#include "collisional_config.h"
#include "subsystem.h"
#include "mergers.h"
#include "merger_tree.h"
#include "diagnostics.h"
#endif
#include <Kokkos_Core.hpp>
#include <vector>
#include <cmath>

// =============================================================================
// CpuSubset methods
// =============================================================================

void CpuSubset::resize(int n) {
    count = n;
    x.resize(n); y.resize(n); z.resize(n);
    vx.resize(n); vy.resize(n); vz.resize(n);
    ax.resize(n); ay.resize(n); az.resize(n);
    mass.resize(n); dt_adapt.resize(n);
    ix.resize(n); iy.resize(n); iz.resize(n);
#ifdef ENABLE_COLLISIONAL
    sid.resize(n, -1);
    grad_mag.resize(n, 0.0);
    dt_level0.resize(n, 1e60);
    particle_id.resize(n, 0);
#endif
}

void CpuSubset::copy_from_device(Particles& p, int start) {
    auto h_x = Kokkos::create_mirror_view(Kokkos::subview(p.x, Kokkos::pair<int,int>(start, start+count)));
    auto h_y = Kokkos::create_mirror_view(Kokkos::subview(p.y, Kokkos::pair<int,int>(start, start+count)));
    auto h_z = Kokkos::create_mirror_view(Kokkos::subview(p.z, Kokkos::pair<int,int>(start, start+count)));
    auto h_vx = Kokkos::create_mirror_view(Kokkos::subview(p.vx, Kokkos::pair<int,int>(start, start+count)));
    auto h_vy = Kokkos::create_mirror_view(Kokkos::subview(p.vy, Kokkos::pair<int,int>(start, start+count)));
    auto h_vz = Kokkos::create_mirror_view(Kokkos::subview(p.vz, Kokkos::pair<int,int>(start, start+count)));
    auto h_m  = Kokkos::create_mirror_view(Kokkos::subview(p.mass, Kokkos::pair<int,int>(start, start+count)));
    auto h_dt = Kokkos::create_mirror_view(Kokkos::subview(p.dt_adapt, Kokkos::pair<int,int>(start, start+count)));
    auto h_ix = Kokkos::create_mirror_view(Kokkos::subview(p.ix, Kokkos::pair<int,int>(start, start+count)));
    auto h_iy = Kokkos::create_mirror_view(Kokkos::subview(p.iy, Kokkos::pair<int,int>(start, start+count)));
    auto h_iz = Kokkos::create_mirror_view(Kokkos::subview(p.iz, Kokkos::pair<int,int>(start, start+count)));
    Kokkos::deep_copy(h_x, Kokkos::subview(p.x, Kokkos::pair<int,int>(start, start+count)));
    Kokkos::deep_copy(h_y, Kokkos::subview(p.y, Kokkos::pair<int,int>(start, start+count)));
    Kokkos::deep_copy(h_z, Kokkos::subview(p.z, Kokkos::pair<int,int>(start, start+count)));
    Kokkos::deep_copy(h_vx, Kokkos::subview(p.vx, Kokkos::pair<int,int>(start, start+count)));
    Kokkos::deep_copy(h_vy, Kokkos::subview(p.vy, Kokkos::pair<int,int>(start, start+count)));
    Kokkos::deep_copy(h_vz, Kokkos::subview(p.vz, Kokkos::pair<int,int>(start, start+count)));
    Kokkos::deep_copy(h_m, Kokkos::subview(p.mass, Kokkos::pair<int,int>(start, start+count)));
    Kokkos::deep_copy(h_dt, Kokkos::subview(p.dt_adapt, Kokkos::pair<int,int>(start, start+count)));
    Kokkos::deep_copy(h_ix, Kokkos::subview(p.ix, Kokkos::pair<int,int>(start, start+count)));
    Kokkos::deep_copy(h_iy, Kokkos::subview(p.iy, Kokkos::pair<int,int>(start, start+count)));
    Kokkos::deep_copy(h_iz, Kokkos::subview(p.iz, Kokkos::pair<int,int>(start, start+count)));
    Kokkos::fence();
    for (int i = 0; i < count; i++) {
        x[i] = h_x(i); y[i] = h_y(i); z[i] = h_z(i);
        vx[i] = h_vx(i); vy[i] = h_vy(i); vz[i] = h_vz(i);
        mass[i] = h_m(i); dt_adapt[i] = h_dt(i);
        ix[i] = h_ix(i); iy[i] = h_iy(i); iz[i] = h_iz(i);
    }
#ifdef ENABLE_COLLISIONAL
    collisional = p.collisional;
    device_start = start;
    if (collisional) {
        auto h_sid = Kokkos::create_mirror_view(Kokkos::subview(p.subsystem_id, Kokkos::pair<int,int>(start, start+count)));
        auto h_gm  = Kokkos::create_mirror_view(Kokkos::subview(p.grad_mag, Kokkos::pair<int,int>(start, start+count)));
        auto h_dl0 = Kokkos::create_mirror_view(Kokkos::subview(p.dt_level0, Kokkos::pair<int,int>(start, start+count)));
        auto h_pid = Kokkos::create_mirror_view(Kokkos::subview(p.particle_id, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::deep_copy(h_sid, Kokkos::subview(p.subsystem_id, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::deep_copy(h_gm, Kokkos::subview(p.grad_mag, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::deep_copy(h_dl0, Kokkos::subview(p.dt_level0, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::deep_copy(h_pid, Kokkos::subview(p.particle_id, Kokkos::pair<int,int>(start, start+count)));
        Kokkos::fence();
        for (int i = 0; i < count; i++) {
            sid[i] = h_sid(i);
            grad_mag[i] = h_gm(i);
            dt_level0[i] = h_dl0(i);
            particle_id[i] = h_pid(i);
        }
    }
#endif
}

void CpuSubset::copy_to_device(Particles& p, int start) {
    auto h_x = Kokkos::create_mirror_view(Kokkos::subview(p.x, Kokkos::pair<int,int>(start, start+count)));
    auto h_y = Kokkos::create_mirror_view(Kokkos::subview(p.y, Kokkos::pair<int,int>(start, start+count)));
    auto h_z = Kokkos::create_mirror_view(Kokkos::subview(p.z, Kokkos::pair<int,int>(start, start+count)));
    auto h_vx = Kokkos::create_mirror_view(Kokkos::subview(p.vx, Kokkos::pair<int,int>(start, start+count)));
    auto h_vy = Kokkos::create_mirror_view(Kokkos::subview(p.vy, Kokkos::pair<int,int>(start, start+count)));
    auto h_vz = Kokkos::create_mirror_view(Kokkos::subview(p.vz, Kokkos::pair<int,int>(start, start+count)));
    auto h_m  = Kokkos::create_mirror_view(Kokkos::subview(p.mass, Kokkos::pair<int,int>(start, start+count)));
    auto h_dt = Kokkos::create_mirror_view(Kokkos::subview(p.dt_adapt, Kokkos::pair<int,int>(start, start+count)));
    auto h_ix = Kokkos::create_mirror_view(Kokkos::subview(p.ix, Kokkos::pair<int,int>(start, start+count)));
    auto h_iy = Kokkos::create_mirror_view(Kokkos::subview(p.iy, Kokkos::pair<int,int>(start, start+count)));
    auto h_iz = Kokkos::create_mirror_view(Kokkos::subview(p.iz, Kokkos::pair<int,int>(start, start+count)));
    for (int i = 0; i < count; i++) {
        h_x(i) = x[i]; h_y(i) = y[i]; h_z(i) = z[i];
        h_vx(i) = vx[i]; h_vy(i) = vy[i]; h_vz(i) = vz[i];
        h_m(i) = mass[i]; h_dt(i) = dt_adapt[i];
        h_ix(i) = ix[i]; h_iy(i) = iy[i]; h_iz(i) = iz[i];
    }
    Kokkos::deep_copy(Kokkos::subview(p.x, Kokkos::pair<int,int>(start, start+count)), h_x);
    Kokkos::deep_copy(Kokkos::subview(p.y, Kokkos::pair<int,int>(start, start+count)), h_y);
    Kokkos::deep_copy(Kokkos::subview(p.z, Kokkos::pair<int,int>(start, start+count)), h_z);
    Kokkos::deep_copy(Kokkos::subview(p.vx, Kokkos::pair<int,int>(start, start+count)), h_vx);
    Kokkos::deep_copy(Kokkos::subview(p.vy, Kokkos::pair<int,int>(start, start+count)), h_vy);
    Kokkos::deep_copy(Kokkos::subview(p.vz, Kokkos::pair<int,int>(start, start+count)), h_vz);
    Kokkos::deep_copy(Kokkos::subview(p.mass, Kokkos::pair<int,int>(start, start+count)), h_m);
    Kokkos::deep_copy(Kokkos::subview(p.dt_adapt, Kokkos::pair<int,int>(start, start+count)), h_dt);
    Kokkos::deep_copy(Kokkos::subview(p.ix, Kokkos::pair<int,int>(start, start+count)), h_ix);
    Kokkos::deep_copy(Kokkos::subview(p.iy, Kokkos::pair<int,int>(start, start+count)), h_iy);
    Kokkos::deep_copy(Kokkos::subview(p.iz, Kokkos::pair<int,int>(start, start+count)), h_iz);
#ifdef ENABLE_COLLISIONAL
    if (collisional) {
        auto h_sid = Kokkos::create_mirror_view(Kokkos::subview(p.subsystem_id, Kokkos::pair<int,int>(start, start+count)));
        auto h_dl0 = Kokkos::create_mirror_view(Kokkos::subview(p.dt_level0, Kokkos::pair<int,int>(start, start+count)));
        for (int i = 0; i < count; i++) {
            h_sid(i) = sid[i];
            h_dl0(i) = dt_level0[i];
        }
        Kokkos::deep_copy(Kokkos::subview(p.subsystem_id, Kokkos::pair<int,int>(start, start+count)), h_sid);
        Kokkos::deep_copy(Kokkos::subview(p.dt_level0, Kokkos::pair<int,int>(start, start+count)), h_dl0);
    }
#endif
    Kokkos::fence();
}

// =============================================================================
// CPU N² force computation functions (all static — internal to this TU)
// =============================================================================

// CPU self-acceleration (N3L, no kick)
static void cpu_self_acc(CpuSubset& s, int start, int count,
                         double G, double soft2) {
    const int end = start + count;
    for (int i = start; i < end; i++) {
        s.ax[i] = 0; s.ay[i] = 0; s.az[i] = 0;
    }
    for (int i = start; i < end; i++) {
        for (int j = i + 1; j < end; j++) {
            if (s.collisional && s.sid[i] >= 0 && s.sid[j] == s.sid[i]) continue;
            double dx = s.x[j] - s.x[i];
            double dy = s.y[j] - s.y[i];
            double dz = s.z[j] - s.z[i];
            double r2 = dx*dx + dy*dy + dz*dz + soft2;
            double inv_r = 1.0 / std::sqrt(r2);
            double inv_r3 = inv_r * inv_r * inv_r;
            double fj = G * s.mass[j] * inv_r3;
            double fi = G * s.mass[i] * inv_r3;
            s.ax[i] += fj * dx; s.ay[i] += fj * dy; s.az[i] += fj * dz;
            s.ax[j] -= fi * dx; s.ay[j] -= fi * dy; s.az[j] -= fi * dz;
        }
    }
}

// CPU self-gradient correction (N3L: upper triangle, modifies ax/ay/az in-place)
static void cpu_self_grad(CpuSubset& s, int start, int count,
                          double G, double soft2, double dt) {
    const double coeff = G * dt * dt * GRAD_COEFF;
    std::vector<double> ax0(count), ay0(count), az0(count);
    for (int i = 0; i < count; i++) {
        ax0[i] = s.ax[start + i];
        ay0[i] = s.ay[start + i];
        az0[i] = s.az[start + i];
    }
    for (int ii = 0; ii < count; ii++) {
        int i = start + ii;
        for (int jj = ii + 1; jj < count; jj++) {
            int j = start + jj;
            if (s.collisional && s.sid[i] >= 0 && s.sid[j] == s.sid[i]) continue;
            double dx = s.x[j] - s.x[i];
            double dy = s.y[j] - s.y[i];
            double dz = s.z[j] - s.z[i];
            double dax = ax0[jj] - ax0[ii];
            double day = ay0[jj] - ay0[ii];
            double daz = az0[jj] - az0[ii];
            double r2 = dx*dx + dy*dy + dz*dz + soft2;
            double inv_r = 1.0 / std::sqrt(r2);
            double inv_r2 = inv_r * inv_r;
            double inv_r5 = inv_r2 * inv_r2 * inv_r;
            double r_dot_a = dx*dax + dy*day + dz*daz;
            double term_x = r2 * dax - 3.0 * r_dot_a * dx;
            double term_y = r2 * day - 3.0 * r_dot_a * dy;
            double term_z = r2 * daz - 3.0 * r_dot_a * dz;
            double fj = coeff * s.mass[j] * inv_r5;
            double fi = coeff * s.mass[i] * inv_r5;
            s.ax[i] += fj * term_x; s.ay[i] += fj * term_y; s.az[i] += fj * term_z;
            s.ax[j] -= fi * term_x; s.ay[j] -= fi * term_y; s.az[j] -= fi * term_z;
        }
    }
#ifdef ENABLE_COLLISIONAL
    // Store bare |g'| = |correction| / dt² for gradient timestep criteria
    if (s.collisional) {
        double inv_dt2 = 1.0 / (dt * dt);
        for (int ii = 0; ii < count; ii++) {
            int i = start + ii;
            double cx = s.ax[i] - ax0[ii];
            double cy = s.ay[i] - ay0[ii];
            double cz = s.az[i] - az0[ii];
            s.grad_mag[i] = std::sqrt(cx*cx + cy*cy + cz*cz) * inv_dt2;
        }
    }
#endif
}

// CPU cross-acceleration (N3L, no kick)
static void cpu_cross_acc(CpuSubset& s,
                           int a_start, int a_count,
                           int b_start, int b_count,
                           double G, double soft2) {
    const int a_end = a_start + a_count;
    const int b_end = b_start + b_count;
    for (int i = a_start; i < a_end; i++) { s.ax[i] = 0; s.ay[i] = 0; s.az[i] = 0; }
    for (int j = b_start; j < b_end; j++) { s.ax[j] = 0; s.ay[j] = 0; s.az[j] = 0; }
    for (int i = a_start; i < a_end; i++) {
        for (int j = b_start; j < b_end; j++) {
            if (s.collisional && s.sid[i] >= 0 && s.sid[j] == s.sid[i]) continue;
            double dx = s.x[j] - s.x[i];
            double dy = s.y[j] - s.y[i];
            double dz = s.z[j] - s.z[i];
            double r2 = dx*dx + dy*dy + dz*dz + soft2;
            double inv_r = 1.0 / std::sqrt(r2);
            double inv_r3 = inv_r * inv_r * inv_r;
            double fj = G * s.mass[j] * inv_r3;
            double fi = G * s.mass[i] * inv_r3;
            s.ax[i] += fj * dx; s.ay[i] += fj * dy; s.az[i] += fj * dz;
            s.ax[j] -= fi * dx; s.ay[j] -= fi * dy; s.az[j] -= fi * dz;
        }
    }
}

// CPU cross-gradient correction (N3L, modifies ax/ay/az in-place)
static void cpu_cross_grad(CpuSubset& s,
                            int a_start, int a_count,
                            int b_start, int b_count,
                            double G, double soft2, double grad_dt) {
    const double coeff = G * grad_dt * grad_dt * GRAD_COEFF;
    std::vector<double> ax0_a(a_count), ay0_a(a_count), az0_a(a_count);
    std::vector<double> ax0_b(b_count), ay0_b(b_count), az0_b(b_count);
    for (int i = 0; i < a_count; i++) {
        ax0_a[i] = s.ax[a_start+i]; ay0_a[i] = s.ay[a_start+i]; az0_a[i] = s.az[a_start+i];
    }
    for (int j = 0; j < b_count; j++) {
        ax0_b[j] = s.ax[b_start+j]; ay0_b[j] = s.ay[b_start+j]; az0_b[j] = s.az[b_start+j];
    }
    for (int ii = 0; ii < a_count; ii++) {
        int i = a_start + ii;
        for (int jj = 0; jj < b_count; jj++) {
            int j = b_start + jj;
            if (s.collisional && s.sid[i] >= 0 && s.sid[j] == s.sid[i]) continue;
            double dx = s.x[j] - s.x[i], dy = s.y[j] - s.y[i], dz = s.z[j] - s.z[i];
            double dax = ax0_b[jj] - ax0_a[ii], day = ay0_b[jj] - ay0_a[ii], daz = az0_b[jj] - az0_a[ii];
            double r2 = dx*dx + dy*dy + dz*dz + soft2;
            double inv_r = 1.0 / std::sqrt(r2);
            double inv_r2 = inv_r * inv_r;
            double inv_r5 = inv_r2 * inv_r2 * inv_r;
            double r_dot_a = dx*dax + dy*day + dz*daz;
            double term_x = r2 * dax - 3.0 * r_dot_a * dx;
            double term_y = r2 * day - 3.0 * r_dot_a * dy;
            double term_z = r2 * daz - 3.0 * r_dot_a * dz;
            double fj = coeff * s.mass[j] * inv_r5;
            double fi = coeff * s.mass[i] * inv_r5;
            s.ax[i] += fj * term_x; s.ay[i] += fj * term_y; s.az[i] += fj * term_z;
            s.ax[j] -= fi * term_x; s.ay[j] -= fi * term_y; s.az[j] -= fi * term_z;
        }
    }
#ifdef ENABLE_COLLISIONAL
    // Store bare |g'| = |correction| / dt² for gradient timestep criteria
    if (s.collisional) {
        double inv_dt2 = 1.0 / (grad_dt * grad_dt);
        for (int ii = 0; ii < a_count; ii++) {
            int i = a_start + ii;
            double cx = s.ax[i] - ax0_a[ii];
            double cy = s.ay[i] - ay0_a[ii];
            double cz = s.az[i] - az0_a[ii];
            s.grad_mag[i] = std::sqrt(cx*cx + cy*cy + cz*cz) * inv_dt2;
        }
        for (int jj = 0; jj < b_count; jj++) {
            int j = b_start + jj;
            double cx = s.ax[j] - ax0_b[jj];
            double cy = s.ay[j] - ay0_b[jj];
            double cz = s.az[j] - az0_b[jj];
            s.grad_mag[j] = std::sqrt(cx*cx + cy*cy + cz*cz) * inv_dt2;
        }
    }
#endif
}

// CPU drift
static void cpu_drift(CpuSubset& s, int start, int count, double dt) {
    if (use_integer_positions) {
        double scale = int_pos_scale;
        double inv_scale = int_pos_inv_scale;
        for (int i = start; i < start + count; i++) {
            s.ix[i] += llrint(s.vx[i] * dt * inv_scale);
            s.iy[i] += llrint(s.vy[i] * dt * inv_scale);
            s.iz[i] += llrint(s.vz[i] * dt * inv_scale);
            s.x[i] = (double)s.ix[i] * scale;
            s.y[i] = (double)s.iy[i] * scale;
            s.z[i] = (double)s.iz[i] * scale;
        }
    } else {
        for (int i = start; i < start + count; i++) {
            s.x[i] += s.vx[i] * dt;
            s.y[i] += s.vy[i] * dt;
            s.z[i] += s.vz[i] * dt;
        }
    }
}

#ifdef ENABLE_COLLISIONAL
// CPU drift for field particles only (skip subsystem members)
static void cpu_drift_field_only(CpuSubset& s, int start, int count, double dt) {
    if (!s.collisional) { cpu_drift(s, start, count, dt); return; }
    if (use_integer_positions) {
        double scale = int_pos_scale;
        double inv_scale = int_pos_inv_scale;
        for (int i = start; i < start + count; i++) {
            if (s.sid[i] >= 0) continue;
            s.ix[i] += llrint(s.vx[i] * dt * inv_scale);
            s.iy[i] += llrint(s.vy[i] * dt * inv_scale);
            s.iz[i] += llrint(s.vz[i] * dt * inv_scale);
            s.x[i] = (double)s.ix[i] * scale;
            s.y[i] = (double)s.iy[i] * scale;
            s.z[i] = (double)s.iz[i] * scale;
        }
    } else {
        for (int i = start; i < start + count; i++) {
            if (s.sid[i] >= 0) continue;
            s.x[i] += s.vx[i] * dt;
            s.y[i] += s.vy[i] * dt;
            s.z[i] += s.vz[i] * dt;
        }
    }
}
#endif

// CPU kick using pre-computed accelerations
static void cpu_kick(CpuSubset& s, int start, int count, double kick_dt) {
    const int end = start + count;
    for (int i = start; i < end; i++) {
        s.vx[i] += s.ax[i] * kick_dt;
        s.vy[i] += s.ay[i] * kick_dt;
        s.vz[i] += s.az[i] * kick_dt;
    }
}

// CPU timestep computation (N3L: upper triangle, squared-space)
static void cpu_compute_timestep(CpuSubset& s, int start, int count,
                                 double G, double soft2, double eta) {
    (void)soft2;
    (void)eta;
#ifdef ENABLE_COLLISIONAL
    double eta_ff_v = g_collisional.eta_ff;
    double eta_fb_v = g_collisional.eta_fb;
    double eta_jerk_val = s.collisional ? g_collisional.eta_jerk : 0.0;
#else
    double eta_ff_v = eta;
    double eta_fb_v = eta;
    double eta_jerk_val = 0.0;
#endif
    const int end = start + count;

    // Initialize timesteps, jerk and acceleration accumulators
    for (int i = start; i < end; i++) s.dt_adapt[i] = 1e30;
    std::vector<double> jerk_x(count, 0), jerk_y(count, 0), jerk_z(count, 0);
    std::vector<double> acc_x(count, 0), acc_y(count, 0), acc_z(count, 0);

    for (int i = start; i < end; i++) {
        for (int j = i + 1; j < end; j++) {
            if (s.collisional && s.sid[i] >= 0 && s.sid[j] == s.sid[i]) continue;
            double dx = s.x[j] - s.x[i];
            double dy = s.y[j] - s.y[i];
            double dz = s.z[j] - s.z[i];
            double dvx = s.vx[j] - s.vx[i];
            double dvy = s.vy[j] - s.vy[i];
            double dvz = s.vz[j] - s.vz[i];
            double r2 = dx*dx + dy*dy + dz*dz;
            double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
            double r2s = r2 + EPS;
            double inv_r = 1.0 / std::sqrt(r2s);
            double mass_sum = s.mass[i] + s.mass[j];
            double r3 = r2s * r2s * inv_r;

            // Symmetric timestep (always on): compute actual timescales per pair
            double inv_Gm = 1.0 / (G * mass_sum + EPS);
            double inv_vsq = 1.0 / (v_sq + EPS);
            double t_ff_sq = r3 * inv_Gm;
            double t_fb_sq = r2s * inv_vsq;
            double t_ff = t_ff_sq * (1.0 / std::sqrt(t_ff_sq + EPS));
            double t_fb = t_fb_sq * (1.0 / std::sqrt(t_fb_sq + EPS));

            double inv_r2 = inv_r * inv_r;
            double r_dot_v = dx*dvx + dy*dvy + dz*dvz;

            double c_ff = 1.5 * r_dot_v * inv_r2;
            double c_fb = r_dot_v * inv_r2 * (1.0 + G * mass_sum * inv_r * inv_vsq);

            double ct_ff = c_ff * t_ff_sq;
            double s_ff = 1.0 + c_ff * ct_ff;
            double u_ff = s_ff * (1.0 / std::sqrt(s_ff));
            double t_sym_ff = t_ff + ct_ff / (u_ff + 1.0);

            double ct_fb = c_fb * t_fb_sq;
            double s_fb = 1.0 + c_fb * ct_fb;
            double u_fb = s_fb * (1.0 / std::sqrt(s_fb));
            double t_sym_fb = t_fb + ct_fb / (u_fb + 1.0);

            double t_pair = std::min(eta_ff_v * t_sym_ff, eta_fb_v * t_sym_fb);
            s.dt_adapt[i] = std::min(s.dt_adapt[i], t_pair);
            s.dt_adapt[j] = std::min(s.dt_adapt[j], t_pair);

#ifdef ENABLE_COLLISIONAL
            // Jerk and acceleration accumulation (collisional only)
            if (s.collisional) {
                double inv_r3 = inv_r * inv_r * inv_r;
                double rdotv = dx*dvx + dy*dvy + dz*dvz;
                double fac1_j = G * s.mass[j] * inv_r3;
                double fac1_i = G * s.mass[i] * inv_r3;
                double fac2 = 3.0 * rdotv * inv_r * inv_r;
                jerk_x[i - start] += fac1_j * (dvx - fac2 * dx);
                jerk_y[i - start] += fac1_j * (dvy - fac2 * dy);
                jerk_z[i - start] += fac1_j * (dvz - fac2 * dz);
                jerk_x[j - start] -= fac1_i * (dvx - fac2 * dx);
                jerk_y[j - start] -= fac1_i * (dvy - fac2 * dy);
                jerk_z[j - start] -= fac1_i * (dvz - fac2 * dz);
                // Fresh subset acceleration for jerk criterion (BIFROST §3.4.5)
                acc_x[i - start] += fac1_j * dx;
                acc_y[i - start] += fac1_j * dy;
                acc_z[i - start] += fac1_j * dz;
                acc_x[j - start] -= fac1_i * dx;
                acc_y[j - start] -= fac1_i * dy;
                acc_z[j - start] -= fac1_i * dz;
            }

#endif
        }
    }

    // Jerk timestep criterion (collisional only)
#ifdef ENABLE_COLLISIONAL
    if (s.collisional && eta_jerk_val > 0.0) {
        for (int i = start; i < end; i++) {
            int ii = i - start;
            double a_mag = std::sqrt(acc_x[ii]*acc_x[ii] + acc_y[ii]*acc_y[ii] + acc_z[ii]*acc_z[ii]);
            double j_mag = std::sqrt(jerk_x[ii]*jerk_x[ii] + jerk_y[ii]*jerk_y[ii] + jerk_z[ii]*jerk_z[ii]);
            if (j_mag > EPS && a_mag > EPS) {
                double dt_jerk = eta_jerk_val * a_mag / j_mag;
                s.dt_adapt[i] = std::min(s.dt_adapt[i], dt_jerk);
            }
        }
    }
#endif
}

// CPU partition by pivot (returns slow count, reorders data in-place)
static int cpu_partition_by_pivot(CpuSubset& s, int start, int count, double pivot) {
    if (count <= 1) return count;
    int slow_count = 0;
    for (int i = start; i < start + count; i++) {
        if (s.dt_adapt[i] >= pivot) slow_count++;
    }
    if (slow_count == 0 || slow_count == count) return slow_count;

    // Build reordered copies
    std::vector<double> tx(count), ty(count), tz(count);
    std::vector<double> tvx(count), tvy(count), tvz(count);
    std::vector<double> tax(count), tay(count), taz(count);
    std::vector<double> tm(count), tdt(count);
    std::vector<int64_t> tix(count), tiy(count), tiz(count);
#ifdef ENABLE_COLLISIONAL
    std::vector<int> tsid(count);
    std::vector<double> tgm(count), tdl0(count);
    std::vector<int> tpid(count);
#endif
    int si = 0, fi = slow_count;
    for (int i = 0; i < count; i++) {
        int src = start + i;
        int dst = (s.dt_adapt[src] >= pivot) ? si++ : fi++;
        tx[dst] = s.x[src]; ty[dst] = s.y[src]; tz[dst] = s.z[src];
        tvx[dst] = s.vx[src]; tvy[dst] = s.vy[src]; tvz[dst] = s.vz[src];
        tax[dst] = s.ax[src]; tay[dst] = s.ay[src]; taz[dst] = s.az[src];
        tm[dst] = s.mass[src]; tdt[dst] = s.dt_adapt[src];
        tix[dst] = s.ix[src]; tiy[dst] = s.iy[src]; tiz[dst] = s.iz[src];
#ifdef ENABLE_COLLISIONAL
        tsid[dst] = s.sid[src];
        tgm[dst] = s.grad_mag[src]; tdl0[dst] = s.dt_level0[src];
        tpid[dst] = s.particle_id[src];
#endif
    }
    for (int i = 0; i < count; i++) {
        int dst = start + i;
        s.x[dst] = tx[i]; s.y[dst] = ty[i]; s.z[dst] = tz[i];
        s.vx[dst] = tvx[i]; s.vy[dst] = tvy[i]; s.vz[dst] = tvz[i];
        s.ax[dst] = tax[i]; s.ay[dst] = tay[i]; s.az[dst] = taz[i];
        s.mass[dst] = tm[i]; s.dt_adapt[dst] = tdt[i];
        s.ix[dst] = tix[i]; s.iy[dst] = tiy[i]; s.iz[dst] = tiz[i];
#ifdef ENABLE_COLLISIONAL
        s.sid[dst] = tsid[i];
        s.grad_mag[dst] = tgm[i]; s.dt_level0[dst] = tdl0[i];
        s.particle_id[dst] = tpid[i];
#endif
    }
    return slow_count;
}

// =============================================================================
// Device sync helpers (collisional only)
// =============================================================================

#ifdef ENABLE_COLLISIONAL
// Sync CpuSubset state to device Particles for subsystem detection/integration
static void cpu_sync_to_device(CpuSubset& s, Particles& p, int cpu_start, int dev_start, int count) {
    auto h_x = Kokkos::create_mirror_view(Kokkos::subview(p.x, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_y = Kokkos::create_mirror_view(Kokkos::subview(p.y, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_z = Kokkos::create_mirror_view(Kokkos::subview(p.z, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_vx = Kokkos::create_mirror_view(Kokkos::subview(p.vx, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_vy = Kokkos::create_mirror_view(Kokkos::subview(p.vy, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_vz = Kokkos::create_mirror_view(Kokkos::subview(p.vz, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_m  = Kokkos::create_mirror_view(Kokkos::subview(p.mass, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_dt = Kokkos::create_mirror_view(Kokkos::subview(p.dt_adapt, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_sid = Kokkos::create_mirror_view(Kokkos::subview(p.subsystem_id, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    for (int i = 0; i < count; i++) {
        h_x(i) = s.x[cpu_start + i]; h_y(i) = s.y[cpu_start + i]; h_z(i) = s.z[cpu_start + i];
        h_vx(i) = s.vx[cpu_start + i]; h_vy(i) = s.vy[cpu_start + i]; h_vz(i) = s.vz[cpu_start + i];
        h_m(i) = s.mass[cpu_start + i]; h_dt(i) = s.dt_adapt[cpu_start + i];
        h_sid(i) = s.sid[cpu_start + i];
    }
    Kokkos::deep_copy(Kokkos::subview(p.x, Kokkos::pair<int,int>(dev_start, dev_start+count)), h_x);
    Kokkos::deep_copy(Kokkos::subview(p.y, Kokkos::pair<int,int>(dev_start, dev_start+count)), h_y);
    Kokkos::deep_copy(Kokkos::subview(p.z, Kokkos::pair<int,int>(dev_start, dev_start+count)), h_z);
    Kokkos::deep_copy(Kokkos::subview(p.vx, Kokkos::pair<int,int>(dev_start, dev_start+count)), h_vx);
    Kokkos::deep_copy(Kokkos::subview(p.vy, Kokkos::pair<int,int>(dev_start, dev_start+count)), h_vy);
    Kokkos::deep_copy(Kokkos::subview(p.vz, Kokkos::pair<int,int>(dev_start, dev_start+count)), h_vz);
    Kokkos::deep_copy(Kokkos::subview(p.mass, Kokkos::pair<int,int>(dev_start, dev_start+count)), h_m);
    Kokkos::deep_copy(Kokkos::subview(p.dt_adapt, Kokkos::pair<int,int>(dev_start, dev_start+count)), h_dt);
    Kokkos::deep_copy(Kokkos::subview(p.subsystem_id, Kokkos::pair<int,int>(dev_start, dev_start+count)), h_sid);
    Kokkos::fence();
}

// Sync subsystem_id back from device to CpuSubset after detection
static void cpu_sync_sid_from_device(CpuSubset& s, Particles& p, int cpu_start, int dev_start, int count) {
    auto h_sid = Kokkos::create_mirror_view(Kokkos::subview(p.subsystem_id, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    Kokkos::deep_copy(h_sid, Kokkos::subview(p.subsystem_id, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    Kokkos::fence();
    for (int i = 0; i < count; i++) s.sid[cpu_start + i] = h_sid(i);
}

// Sync positions and velocities from device back to CpuSubset
static void cpu_sync_pos_vel_from_device(CpuSubset& s, Particles& p, int cpu_start, int dev_start, int count) {
    auto h_x  = Kokkos::create_mirror_view(Kokkos::subview(p.x, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_y  = Kokkos::create_mirror_view(Kokkos::subview(p.y, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_z  = Kokkos::create_mirror_view(Kokkos::subview(p.z, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_vx = Kokkos::create_mirror_view(Kokkos::subview(p.vx, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_vy = Kokkos::create_mirror_view(Kokkos::subview(p.vy, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    auto h_vz = Kokkos::create_mirror_view(Kokkos::subview(p.vz, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    Kokkos::deep_copy(h_x, Kokkos::subview(p.x, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    Kokkos::deep_copy(h_y, Kokkos::subview(p.y, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    Kokkos::deep_copy(h_z, Kokkos::subview(p.z, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    Kokkos::deep_copy(h_vx, Kokkos::subview(p.vx, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    Kokkos::deep_copy(h_vy, Kokkos::subview(p.vy, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    Kokkos::deep_copy(h_vz, Kokkos::subview(p.vz, Kokkos::pair<int,int>(dev_start, dev_start+count)));
    Kokkos::fence();
    for (int i = 0; i < count; i++) {
        s.x[cpu_start + i] = h_x(i); s.y[cpu_start + i] = h_y(i); s.z[cpu_start + i] = h_z(i);
        s.vx[cpu_start + i] = h_vx(i); s.vy[cpu_start + i] = h_vy(i); s.vz[cpu_start + i] = h_vz(i);
    }
}
#endif

// =============================================================================
// CPU FSI step and HHS recursion
// =============================================================================

// CPU FSI step
static void cpu_fsi_step(CpuSubset& s, Particles& p, int start, int count,
                         double G, double soft2, double dt,
                         bool forces_valid = false) {
    if (!forces_valid) {
        cpu_self_acc(s, start, count, G, soft2);
    }
    cpu_kick(s, start, count, dt * FSI_KICK);
#ifdef ENABLE_COLLISIONAL
    // H_S (dt/2) then T' drift (dt/2) — BIFROST Algorithm 1 ordering
    if (s.collisional && !g_subsystem_mgr.active.empty()) {
        int dev_start = s.device_start + start;
        cpu_sync_to_device(s, p, start, dev_start, count);
        advance_subsystems_in_range(p, dev_start, count, G, dt * FSI_DRIFT);
        cpu_sync_pos_vel_from_device(s, p, start, dev_start, count);
    }
    cpu_drift_field_only(s, start, count, dt * FSI_DRIFT);
#else
    cpu_drift(s, start, count, dt * FSI_DRIFT);
#endif
    cpu_self_acc(s, start, count, G, soft2);
    cpu_self_grad(s, start, count, G, soft2, dt);
    cpu_kick(s, start, count, dt * FSI_MOD_KICK);
#ifdef ENABLE_COLLISIONAL
    // H_S (dt/2) then T' drift (dt/2) — BIFROST Algorithm 1 ordering
    if (s.collisional && !g_subsystem_mgr.active.empty()) {
        int dev_start = s.device_start + start;
        cpu_sync_to_device(s, p, start, dev_start, count);
        advance_subsystems_in_range(p, dev_start, count, G, dt * FSI_DRIFT);
        cpu_sync_pos_vel_from_device(s, p, start, dev_start, count);
    }
    cpu_drift_field_only(s, start, count, dt * FSI_DRIFT);
#else
    cpu_drift(s, start, count, dt * FSI_DRIFT);
#endif
    cpu_self_acc(s, start, count, G, soft2);
    cpu_kick(s, start, count, dt * FSI_KICK);
}

// CPU HHS-FSI recursion
static void cpu_hhs_recursive(CpuSubset& s, Particles& p, int start, int count,
                              double pivot_dt, double G, double soft2,
                              double dt_min, double eta, bool dt_valid, int depth = 0) {
    if (count == 0) return;
    if (depth < HHSCounters::MAX_DEPTH) {
        g_hhs_counters.cpu_depth_calls[depth]++;
        g_hhs_counters.cpu_depth_count[depth] += count;
    }

    if (pivot_dt * 0.5 < dt_min) {
        cpu_fsi_step(s, p, start, count, G, soft2, pivot_dt);
        return;
    }

    // Global device offset for this subset
    int dev_start = s.device_start + start;

    // BIFROST Section 3.5: fresh neighbor search + detection, then timestep
    if (!dt_valid && count > 1) {
#ifdef ENABLE_COLLISIONAL
        // Step 1: Neighbor search + detection (sync to device, run GPU kernel, sync back)
        if (s.collisional) {
            cpu_sync_to_device(s, p, start, dev_start, count);
            g_subsystem_mgr.reset_ngb_count();
            neighbor_search_subset(p, dev_start, count, G,
                                   g_collisional.r_ngb, g_collisional.r_ngb_max, pivot_dt);
            g_subsystem_mgr.detect_from_pairs(p, G, pivot_dt,
                                               g_collisional.r_ngb, g_collisional.r_ngb_max,
                                               g_collisional.n_orb_secular);
            g_subsystem_mgr.update_particle_flags(p);
            g_subsystem_mgr.accumulate_diagnostics();
            cpu_sync_sid_from_device(s, p, start, dev_start, count);
        }
#endif
        // Step 2: CPU timestep (with subsystem exclusion via s.sid)
        cpu_compute_timestep(s, start, count, G, soft2, eta);
#ifdef ENABLE_COLLISIONAL
        // Gradient + v/a timestep criteria (BIFROST Section 3.4.4, Eq. 24/26/27)
        if (s.collisional && g_collisional.eta_grad > 0.0) {
            double eta_g = g_collisional.eta_grad;
            double eta_va = eta;  // v/a criterion uses general eta (0.2)
            for (int i = start; i < start + count; i++) {
                double a_mag = std::sqrt(s.ax[i]*s.ax[i] + s.ay[i]*s.ay[i] + s.az[i]*s.az[i]);
                double v_mag = std::sqrt(s.vx[i]*s.vx[i] + s.vy[i]*s.vy[i] + s.vz[i]*s.vz[i]);
                double g = s.grad_mag[i];
                if (g > EPS) {
                    if (a_mag > EPS) {
                        double dt_g2 = eta_g * std::sqrt(a_mag / g);
                        if (dt_g2 < s.dt_adapt[i]) s.dt_adapt[i] = dt_g2;
                    }
                    if (v_mag > EPS) {
                        double dt_g1 = eta_g * std::cbrt(v_mag / g);
                        if (dt_g1 < s.dt_adapt[i]) s.dt_adapt[i] = dt_g1;
                    }
                }
                // v/a criterion (BIFROST Eq. 24)
                if (a_mag > EPS && v_mag > EPS) {
                    double dt_va = eta_va * v_mag / a_mag;
                    if (dt_va < s.dt_adapt[i]) s.dt_adapt[i] = dt_va;
                }
            }
        }
        // Level-0 clamping (BIFROST Section 3.4.1)
        if (s.collisional && depth > 0) {
            for (int i = start; i < start + count; i++)
                if (s.dt_adapt[i] > s.dt_level0[i]) s.dt_adapt[i] = s.dt_level0[i];
        }

        // Step 3: Subsystem timestep adjustments (sync dt to device, run, sync back)
        if (s.collisional) {
            {
                auto h_dt = Kokkos::create_mirror_view(Kokkos::subview(p.dt_adapt, Kokkos::pair<int,int>(dev_start, dev_start+count)));
                for (int i = 0; i < count; i++) h_dt(i) = s.dt_adapt[start + i];
                Kokkos::deep_copy(Kokkos::subview(p.dt_adapt, Kokkos::pair<int,int>(dev_start, dev_start+count)), h_dt);
            }
            g_subsystem_mgr.enforce_shared_timestep(p);
            g_subsystem_mgr.apply_perturbation_timestep(p, G, g_collisional.eta_pert);

            // Merger detection (BIFROST §3.14)
            if (g_collisional.c_code > 0.0) {
                auto mergers = detect_mergers(g_subsystem_mgr, p, G,
                                               g_collisional.c_code, pivot_dt);
                for (const auto& ev : mergers) {
                    auto* sub = find_subsystem_for_merger(g_subsystem_mgr, ev);
                    if (sub) g_merger_tree.record(ev, *sub, g_sim_time);
                    execute_merger(ev, p, g_subsystem_mgr, g_energy_books,
                                   G, g_collisional.c_code);
                }
                g_subsystem_mgr.step_mergers += (int)mergers.size();
                if (!mergers.empty()) {
                    int n_removed = p.compact_remove_zero_mass();
                    count -= n_removed;
                    if (count <= 0) return;
                }
            }

            // Sync dt_adapt back from device
            {
                auto h_dt = Kokkos::create_mirror_view(Kokkos::subview(p.dt_adapt, Kokkos::pair<int,int>(dev_start, dev_start+count)));
                Kokkos::deep_copy(h_dt, Kokkos::subview(p.dt_adapt, Kokkos::pair<int,int>(dev_start, dev_start+count)));
                Kokkos::fence();
                for (int i = 0; i < count; i++) s.dt_adapt[start + i] = h_dt(i);
            }

            // Level-0 store: capture dt AFTER subsystem adjustments
            if (depth == 0) {
                for (int i = start; i < start + count; i++)
                    s.dt_level0[i] = s.dt_adapt[i];
            }
        }
#endif
    }

    int slow_count = cpu_partition_by_pivot(s, start, count, pivot_dt);
    int slow_start = start;
    int fast_start = start + slow_count;
    int fast_count = count - slow_count;


    if (fast_count == 0) {
        cpu_fsi_step(s, p, start, count, G, soft2, pivot_dt);
        return;
    }

    // Partition was a no-op when slow_count==0, so dt_adapt still valid
    if (slow_count == 0) {
        double sub_pivot = pivot_dt * 0.5;
        cpu_hhs_recursive(s, p, start, count, sub_pivot, G, soft2, dt_min, eta, true, depth + 1);
        cpu_hhs_recursive(s, p, start, count, sub_pivot, G, soft2, dt_min, eta, false, depth + 1);
        return;
    }

    // Save this level's subsystem state before recursive calls can overwrite it.
    auto saved_active_cpu = g_subsystem_mgr.active;

    // Step 1: Cross-kick (dt/6)
    cpu_cross_acc(s, slow_start, slow_count, fast_start, fast_count, G, soft2);
    cpu_kick(s, slow_start, slow_count, pivot_dt * FSI_KICK);
    cpu_kick(s, fast_start, fast_count, pivot_dt * FSI_KICK);

    // Step 2: FSI half-step on slow
    cpu_fsi_step(s, p, slow_start, slow_count, G, soft2, pivot_dt * 0.5);

    // Step 3: Recurse on fast (first half)
    cpu_hhs_recursive(s, p, fast_start, fast_count, pivot_dt * 0.5, G, soft2, dt_min, eta, false, depth + 1);

    // Restore this level's subsystem state after recursion overwrote it
    g_subsystem_mgr.active = saved_active_cpu;
    g_subsystem_mgr.update_particle_flags(p);

    // Step 4: Cross-gradient kick (2dt/3)
    cpu_cross_acc(s, slow_start, slow_count, fast_start, fast_count, G, soft2);
    cpu_cross_grad(s, slow_start, slow_count, fast_start, fast_count, G, soft2, pivot_dt);
    cpu_kick(s, slow_start, slow_count, pivot_dt * FSI_MOD_KICK);
    cpu_kick(s, fast_start, fast_count, pivot_dt * FSI_MOD_KICK);

    // Step 5: Recurse on fast (second half)
    cpu_hhs_recursive(s, p, fast_start, fast_count, pivot_dt * 0.5, G, soft2, dt_min, eta, false, depth + 1);

    // Restore again for the second half-step on slow
    g_subsystem_mgr.active = saved_active_cpu;
    g_subsystem_mgr.update_particle_flags(p);

    // Step 6: FSI half-step on slow
    cpu_fsi_step(s, p, slow_start, slow_count, G, soft2, pivot_dt * 0.5);

    // Step 7: Cross-kick (dt/6)
    cpu_cross_acc(s, slow_start, slow_count, fast_start, fast_count, G, soft2);
    cpu_kick(s, slow_start, slow_count, pivot_dt * FSI_KICK);
    cpu_kick(s, fast_start, fast_count, pivot_dt * FSI_KICK);
}

// Entry point: copy subset to host, run CPU HHS, copy back
void hhs_fsi_recursive_cpu(Particles& p, int start, int count,
                           double pivot_dt, double G, double soft2,
                           double dt_min, double eta, bool dt_valid,
                           int depth) {
    g_subset_hist.record(SubsetHistogram::LOCAL_CPU, count);
    CpuSubset s;
    s.resize(count);
    s.copy_from_device(p, start);
    cpu_hhs_recursive(s, p, 0, count, pivot_dt, G, soft2, dt_min, eta, dt_valid, depth);
    s.copy_to_device(p, start);
}
