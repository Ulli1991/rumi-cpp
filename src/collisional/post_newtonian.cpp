#include "post_newtonian.h"
#include "constants.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstring>
#include <vector>

// ============================================================================
// PN acceleration terms (Blanchet 2014 / Will 1996 conventions)
// All in code units (pc, M_sun, km/s, G_CODE)
// ============================================================================

void pn_acceleration(const double r[3], const double v[3],
                     double m1, double m2, double G, double c_code,
                     double a_pn[3], int pn_order) {
    a_pn[0] = a_pn[1] = a_pn[2] = 0.0;

    double r2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
    double r_mag = std::sqrt(r2);
    double inv_r = 1.0 / r_mag;
    double inv_r2 = inv_r * inv_r;
    double inv_r3 = inv_r2 * inv_r;

    double v2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    double rdotv = r[0]*v[0] + r[1]*v[1] + r[2]*v[2];
    double n_dot_v = rdotv * inv_r;

    double mu = G * (m1 + m2);
    double eta = m1 * m2 / ((m1 + m2) * (m1 + m2));  // symmetric mass ratio

    double c2 = c_code * c_code;
    double c4 = c2 * c2;
    double inv_c2 = 1.0 / c2;
    double inv_c4 = 1.0 / c4;

    double Gm_r = mu * inv_r;  // G*M/r

    // PN1.0 (conservative)
    if (pn_order & 1) {
        double A1 = (1.0 + 3.0*eta) * v2 - 2.0*(2.0 + eta) * Gm_r
                    - 1.5 * eta * n_dot_v * n_dot_v;
        double B1 = 2.0 * (2.0 - eta) * n_dot_v;

        for (int k = 0; k < 3; k++) {
            a_pn[k] += inv_c2 * mu * inv_r2 * (A1 * r[k] * inv_r + B1 * v[k]);
        }
    }

    // PN2.0 (conservative)
    if (pn_order & 2) {
        double Gm_r2 = Gm_r * Gm_r * inv_r;
        double A2 = -eta * (3.0 - 4.0*eta) * v2 * v2 / 4.0
                    + eta * (13.0/4.0 - eta) * v2 * Gm_r
                    - eta * (3.0/2.0 - 3.0*eta) * v2 * n_dot_v * n_dot_v
                    + (9.0 + 87.0/4.0 * eta) * Gm_r * Gm_r * inv_r
                    + 15.0/8.0 * eta * (1.0 - 3.0*eta) * n_dot_v * n_dot_v * n_dot_v * n_dot_v;
        double B2 = -eta * (15.0 + 4.0*eta) * n_dot_v * Gm_r / 2.0
                    + eta * (3.0 + 4.0*eta) * v2 * n_dot_v / 2.0
                    - 1.5 * eta * (3.0 - 4.0*eta) * n_dot_v * n_dot_v * n_dot_v;

        for (int k = 0; k < 3; k++) {
            a_pn[k] += inv_c4 * mu * inv_r2 * (A2 * r[k] * inv_r + B2 * v[k]);
        }
    }

    // PN2.5 (dissipative — gravitational radiation reaction)
    if (pn_order & 4) {
        double inv_c5 = inv_c4 / c_code;
        double A25 = 8.0/5.0 * eta * Gm_r * (17.0/3.0 * Gm_r + 3.0 * v2) * n_dot_v;
        double B25 = -8.0/5.0 * eta * Gm_r * (3.0 * Gm_r + v2);

        for (int k = 0; k < 3; k++) {
            a_pn[k] += inv_c5 * mu * inv_r2 * (A25 * r[k] * inv_r + B25 * v[k]);
        }
    }

    // PN3.0 and PN3.5: higher-order corrections (binary only, optional)
    // Implementing the leading terms from Blanchet (2014)
    // These are computationally expensive and usually only needed for BH mergers
    // Will add on demand
}

// ============================================================================
// Spin precession
// ============================================================================

void spin_precession(const double r[3], const double v[3],
                     double m1, double m2, const double S1[3], const double S2[3],
                     double G, double c_code,
                     double Omega_prec_1[3], double Omega_prec_2[3]) {
    double r2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
    double r_mag = std::sqrt(r2);
    double inv_r = 1.0 / r_mag;
    double inv_r2 = inv_r * inv_r;
    double inv_r3 = inv_r2 * inv_r;
    double c2 = c_code * c_code;

    // Angular momentum L = m * r x v (relative)
    double Lx = r[1]*v[2] - r[2]*v[1];
    double Ly = r[2]*v[0] - r[0]*v[2];
    double Lz = r[0]*v[1] - r[1]*v[0];

    double mu_red = m1 * m2 / (m1 + m2);  // reduced mass
    double M = m1 + m2;

    // Spin-orbit precession (de Sitter / geodetic precession)
    // Omega_SO_1 = G / (2 c^2 r^3) * (4 + 3*m2/m1) * L
    double coeff_SO_1 = G / (2.0 * c2 * r2 * r_mag) * (4.0 + 3.0 * m2 / m1);
    double coeff_SO_2 = G / (2.0 * c2 * r2 * r_mag) * (4.0 + 3.0 * m1 / m2);

    Omega_prec_1[0] = coeff_SO_1 * Lx;
    Omega_prec_1[1] = coeff_SO_1 * Ly;
    Omega_prec_1[2] = coeff_SO_1 * Lz;

    Omega_prec_2[0] = coeff_SO_2 * (-Lx);  // L is opposite for particle 2
    Omega_prec_2[1] = coeff_SO_2 * (-Ly);
    Omega_prec_2[2] = coeff_SO_2 * (-Lz);

    // Spin-spin precession: Omega_SS_1 = G / (c^2 r^3) * [S2 - 3(S2.n)n]
    double n[3] = {r[0]*inv_r, r[1]*inv_r, r[2]*inv_r};
    double S2_dot_n = S2[0]*n[0] + S2[1]*n[1] + S2[2]*n[2];
    double S1_dot_n = S1[0]*n[0] + S1[1]*n[1] + S1[2]*n[2];

    double coeff_SS = G / (c2 * r2 * r_mag);
    Omega_prec_1[0] += coeff_SS * (S2[0] - 3.0 * S2_dot_n * n[0]);
    Omega_prec_1[1] += coeff_SS * (S2[1] - 3.0 * S2_dot_n * n[1]);
    Omega_prec_1[2] += coeff_SS * (S2[2] - 3.0 * S2_dot_n * n[2]);

    Omega_prec_2[0] += coeff_SS * (S1[0] - 3.0 * S1_dot_n * n[0]);
    Omega_prec_2[1] += coeff_SS * (S1[1] - 3.0 * S1_dot_n * n[1]);
    Omega_prec_2[2] += coeff_SS * (S1[2] - 3.0 * S1_dot_n * n[2]);
}

void evolve_spins(double S1[3], double S2[3],
                  const double Omega_prec_1[3], const double Omega_prec_2[3],
                  double dt) {
    // dS/dt = Omega x S  — rotate spin vectors
    // Simple Euler for now (PN spin precession is slow compared to orbital motion)
    double dS1[3], dS2[3];
    dS1[0] = (Omega_prec_1[1]*S1[2] - Omega_prec_1[2]*S1[1]) * dt;
    dS1[1] = (Omega_prec_1[2]*S1[0] - Omega_prec_1[0]*S1[2]) * dt;
    dS1[2] = (Omega_prec_1[0]*S1[1] - Omega_prec_1[1]*S1[0]) * dt;

    dS2[0] = (Omega_prec_2[1]*S2[2] - Omega_prec_2[2]*S2[1]) * dt;
    dS2[1] = (Omega_prec_2[2]*S2[0] - Omega_prec_2[0]*S2[2]) * dt;
    dS2[2] = (Omega_prec_2[0]*S2[1] - Omega_prec_2[1]*S2[0]) * dt;

    S1[0] += dS1[0]; S1[1] += dS1[1]; S1[2] += dS1[2];
    S2[0] += dS2[0]; S2[1] += dS2[1]; S2[2] += dS2[2];
}

// ============================================================================
// Global PN1.0 kick for compact objects outside subsystems
// (BIFROST Section 3.7)
//
// Applied to all pairs where at least one particle has mass >= m_pn
// and both are NOT in the same subsystem (subsystem_id < 0, or different IDs).
// Only PN1.0 (conservative) is applied globally — higher PN orders are
// handled inside LogH subsystem integration.
// ============================================================================

void global_pn1_kick(Particles& p, int start, int count,
                     double G, double c_code, double m_pn, double dt) {
    if (count == 0 || c_code <= 0.0 || dt == 0.0) return;

    // Copy particle data to host
    auto h_x  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
    auto h_y  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
    auto h_z  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
    auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
    auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
    auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
    auto h_m  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
    auto h_sid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.subsystem_id);

    // Find compact objects (mass >= m_pn) in [start, start+count)
    int end = start + count;
    std::vector<int> compact;
    compact.reserve(256);
    for (int i = start; i < end; i++) {
        if (h_m(i) >= m_pn) compact.push_back(i);
    }

    int n_compact = (int)compact.size();
    if (n_compact < 2) return;  // need at least 2 compact objects

    // Accumulate PN1.0 velocity kicks on host
    // dvx[i], dvy[i], dvz[i] for each compact object
    std::vector<double> dvx(n_compact, 0.0), dvy(n_compact, 0.0), dvz(n_compact, 0.0);

    for (int ii = 0; ii < n_compact; ii++) {
        int i = compact[ii];
        int sid_i = h_sid(i);

        for (int jj = ii + 1; jj < n_compact; jj++) {
            int j = compact[jj];
            int sid_j = h_sid(j);

            // Skip pairs in the same subsystem (they get full PN inside LogH)
            if (sid_i >= 0 && sid_j == sid_i) continue;

            double r[3] = { h_x(j) - h_x(i), h_y(j) - h_y(i), h_z(j) - h_z(i) };
            double v[3] = { h_vx(j) - h_vx(i), h_vy(j) - h_vy(i), h_vz(j) - h_vz(i) };

            double a_rel[3];
            pn_acceleration(r, v, h_m(i), h_m(j), G, c_code, a_rel, 1);  // PN1.0 only

            // Convert relative PN acceleration to individual:
            // a_i = -(m_j/M) * a_rel,  a_j = +(m_i/M) * a_rel
            // (pn_acceleration returns a_rel for the relative coord r=r_j-r_i)
            double M_ij = h_m(i) + h_m(j);
            double fi = h_m(j) / M_ij;
            double fj = h_m(i) / M_ij;

            dvx[ii] -= dt * fi * a_rel[0];
            dvy[ii] -= dt * fi * a_rel[1];
            dvz[ii] -= dt * fi * a_rel[2];
            dvx[jj] += dt * fj * a_rel[0];
            dvy[jj] += dt * fj * a_rel[1];
            dvz[jj] += dt * fj * a_rel[2];
        }
    }

    // Apply velocity kicks
    for (int ii = 0; ii < n_compact; ii++) {
        int i = compact[ii];
        h_vx(i) += dvx[ii];
        h_vy(i) += dvy[ii];
        h_vz(i) += dvz[ii];
    }

    // Copy velocities back to device
    Kokkos::deep_copy(p.vx, h_vx);
    Kokkos::deep_copy(p.vy, h_vy);
    Kokkos::deep_copy(p.vz, h_vz);
}
