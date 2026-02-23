#include "post_newtonian.h"
#include "constants.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstring>
#include <vector>

// ============================================================================
// EIH (Einstein-Infeld-Hoffmann) PN acceleration
// Computes acceleration of body A due to body B directly.
// (Poisson & Will 2014, Eq. 9.139 / BIFROST Eq. 5)
//
// Convention: n_AB = (x_A - x_B) / r_AB  (points from B toward A)
// ============================================================================

void eih_pn_acceleration(const double x_A[3], const double v_A[3], double m_A,
                         const double x_B[3], const double v_B[3], double m_B,
                         double G, double c_code, double a_pn_A[3], int pn_order) {
    a_pn_A[0] = a_pn_A[1] = a_pn_A[2] = 0.0;

    // Separation vector: x_A - x_B (points from B to A)
    double r_AB[3] = { x_A[0]-x_B[0], x_A[1]-x_B[1], x_A[2]-x_B[2] };
    double r2 = r_AB[0]*r_AB[0] + r_AB[1]*r_AB[1] + r_AB[2]*r_AB[2];
    double r_mag = std::sqrt(r2);
    double inv_r = 1.0 / r_mag;
    double inv_r2 = inv_r * inv_r;

    // Unit vector n_AB = r_AB / |r_AB|
    double n[3] = { r_AB[0]*inv_r, r_AB[1]*inv_r, r_AB[2]*inv_r };

    // Velocity difference
    double dv[3] = { v_A[0]-v_B[0], v_A[1]-v_B[1], v_A[2]-v_B[2] };

    // Dot products
    double vA2 = v_A[0]*v_A[0] + v_A[1]*v_A[1] + v_A[2]*v_A[2];
    double vB2 = v_B[0]*v_B[0] + v_B[1]*v_B[1] + v_B[2]*v_B[2];
    double vA_dot_vB = v_A[0]*v_B[0] + v_A[1]*v_B[1] + v_A[2]*v_B[2];
    double n_dot_vA = n[0]*v_A[0] + n[1]*v_A[1] + n[2]*v_A[2];
    double n_dot_vB = n[0]*v_B[0] + n[1]*v_B[1] + n[2]*v_B[2];

    double c2 = c_code * c_code;
    double inv_c2 = 1.0 / c2;
    double Gm_A = G * m_A;
    double Gm_B = G * m_B;

    // PN1.0 (conservative) — Poisson & Will 2014 Eq. 9.139 / BIFROST Eq. 5
    if (pn_order & 1) {
        // Coefficient of n_AB:
        double coeff_n = -vA2 - 2.0*vB2 + 4.0*vA_dot_vB
                         + 1.5 * n_dot_vB * n_dot_vB
                         + 5.0 * Gm_A * inv_r + 4.0 * Gm_B * inv_r;

        // Coefficient of (v_A - v_B):
        double coeff_v = 4.0 * n_dot_vA - 3.0 * n_dot_vB;

        double prefactor = Gm_B * inv_r2 * inv_c2;
        for (int k = 0; k < 3; k++) {
            a_pn_A[k] += prefactor * (coeff_n * n[k] + coeff_v * dv[k]);
        }
    }

    // PN2.0 (conservative) — Poisson & Will 2014 Eq. 9.140
    if (pn_order & 2) {
        double inv_c4 = inv_c2 * inv_c2;
        double vA4 = vA2 * vA2;
        double vB4 = vB2 * vB2;
        double Gm_A_r = Gm_A * inv_r;
        double Gm_B_r = Gm_B * inv_r;

        double coeff_n =
            -2.0 * vB2 * vB2
            + 4.0 * vB2 * vA_dot_vB - 2.0 * vA_dot_vB * vA_dot_vB
            + 1.5 * vA2 * n_dot_vB * n_dot_vB
            + 4.5 * vB2 * n_dot_vB * n_dot_vB
            - 6.0 * vA_dot_vB * n_dot_vB * n_dot_vB
            - (15.0/8.0) * n_dot_vB * n_dot_vB * n_dot_vB * n_dot_vB
            + Gm_A_r * (-15.0/4.0 * vA2 + 5.0/4.0 * vB2 - 2.5 * vA_dot_vB
                         + 39.0/2.0 * n_dot_vA * n_dot_vA
                         - 39.0 * n_dot_vA * n_dot_vB
                         + 17.0/2.0 * n_dot_vB * n_dot_vB)
            + Gm_B_r * (4.0 * vB2 - 8.0 * vA_dot_vB
                         + 2.0 * n_dot_vA * n_dot_vA
                         - 4.0 * n_dot_vA * n_dot_vB
                         - 6.0 * n_dot_vB * n_dot_vB)
            + Gm_A_r * Gm_A_r * inv_r * (-57.0/4.0)
            + Gm_B_r * Gm_B_r * inv_r * (-9.0)
            + Gm_A_r * Gm_B_r * inv_r * (-69.0/2.0);

        double coeff_v =
            vA2 * n_dot_vB + 4.0 * vA2 * n_dot_vA
            - 5.0 * vB2 * n_dot_vA - 4.0 * vA_dot_vB * n_dot_vA
            + 4.0 * vA_dot_vB * n_dot_vB
            - 6.0 * n_dot_vA * n_dot_vB * n_dot_vB
            + (9.0/2.0) * n_dot_vB * n_dot_vB * n_dot_vB
            + Gm_A_r * (-63.0/4.0 * n_dot_vA + 55.0/4.0 * n_dot_vB)
            + Gm_B_r * (-2.0 * n_dot_vA - 2.0 * n_dot_vB);

        double prefactor = Gm_B * inv_r2 * inv_c4;
        for (int k = 0; k < 3; k++) {
            a_pn_A[k] += prefactor * (coeff_n * n[k] + coeff_v * dv[k]);
        }
    }

    // PN2.5 (dissipative — radiation reaction) — Will 1996 / Blanchet 2014
    if (pn_order & 4) {
        double inv_c5 = inv_c2 * inv_c2 / c_code;
        double n_dot_dv = n[0]*dv[0] + n[1]*dv[1] + n[2]*dv[2];
        double dv2 = dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2];
        double Gm_A_r = Gm_A * inv_r;
        double Gm_B_r = Gm_B * inv_r;

        double coeff_n =
            (4.0/5.0) * Gm_B_r * n_dot_dv * (
                -6.0 * Gm_A_r + (52.0/3.0) * Gm_B_r + 3.0 * dv2
            )
            + (8.0/5.0) * Gm_A_r * Gm_B_r * n_dot_dv * (
                (17.0/3.0) * Gm_A_r + Gm_B_r
            );

        double coeff_v =
            (4.0/5.0) * Gm_B_r * (
                2.0 * Gm_A_r - 8.0 * Gm_B_r - dv2
            )
            + (8.0/5.0) * Gm_A_r * Gm_B_r * (
                -3.0 * Gm_A_r + Gm_B_r
            );

        // PN2.5 uses a simpler two-body form for the dissipative terms
        // Revert to relative formulation for PN2.5 since the EIH N-body
        // dissipative terms require careful treatment.
        // For the two-body case this is exact.
        double M = m_A + m_B;
        double mu_rel = G * M;
        double eta = m_A * m_B / (M * M);
        double r_dot_v = r_AB[0]*dv[0] + r_AB[1]*dv[1] + r_AB[2]*dv[2];
        double n_dot_v_rel = r_dot_v * inv_r;

        double A25 = 8.0/5.0 * eta * mu_rel * inv_r * (17.0/3.0 * mu_rel * inv_r + 3.0 * dv2) * n_dot_v_rel;
        double B25 = -8.0/5.0 * eta * mu_rel * inv_r * (3.0 * mu_rel * inv_r + dv2);

        double fA = m_B / M;  // fraction for body A from relative acceleration
        double prefactor = mu_rel * inv_r2 * inv_c5;
        for (int k = 0; k < 3; k++) {
            // Relative accel = prefactor * (A25*n + B25*dv), body A gets -fA fraction
            // Note: n here points from B to A (same direction as r_AB), and our relative
            // convention is r = x_B - x_A for the Blanchet formula, so we negate.
            a_pn_A[k] += prefactor * (-A25 * n[k] * fA + B25 * (-dv[k]) * fA);
        }
    }

    // PN3.0 (conservative, c^-6) — Blanchet 2014 / Blanchet & Iyer 2003
    // Two-body only, using relative-coordinate formulation mapped to body A.
    if (pn_order & 8) {
        double M = m_A + m_B;
        double mu_rel = G * M;
        double eta = m_A * m_B / (M * M);
        double inv_c6 = inv_c2 * inv_c2 * inv_c2;
        double Gm_r = mu_rel * inv_r;
        double dv2 = dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2];
        double r_dot_v = r_AB[0]*dv[0] + r_AB[1]*dv[1] + r_AB[2]*dv[2];
        double n_dot_v = r_dot_v * inv_r;

        double Gm_r2 = Gm_r * Gm_r;
        double Gm_r3 = Gm_r2 * Gm_r * inv_r;
        double v4 = dv2 * dv2;
        double v6 = v4 * dv2;
        double ndv2 = n_dot_v * n_dot_v;
        double ndv4 = ndv2 * ndv2;

        // Coefficient of n (radial): from Blanchet 2014 Eq. (203)
        double A3 =
            + eta * (3.0 - 6.0*eta + 3.0*eta*eta) * v6 / 16.0
            + eta * Gm_r * (
                - (13.5 - 5.0*eta + 21.75*eta*eta) * v4
                + eta * (6.0 - 22.5*eta) * dv2 * ndv2
                + (7.5*eta*eta) * ndv4
            )
            + Gm_r2 * inv_r * (
                + (16.0 + 35.0/6.0*eta - 203.0/12.0*eta*eta) * dv2
                + eta * (-31.5 + 22.5*eta) * ndv2
            )
            + Gm_r3 * (
                + (227.0/12.0 - 53.0/4.0*eta + 41.0/64.0*M_PI*M_PI*eta)
            );

        // Coefficient of v (tangential)
        double B3 =
            + eta * Gm_r * n_dot_v * (
                (13.5 - 15.75*eta + 2.0*eta*eta) * dv2
                - eta * (3.0 - 7.5*eta) * ndv2
            )
            + Gm_r2 * inv_r * n_dot_v * (
                + (-16.0 + 13.0*eta + 4.5*eta*eta)
            );

        double fA = m_B / M;
        double prefactor = mu_rel * inv_r2 * inv_c6;
        for (int k = 0; k < 3; k++) {
            // Map relative acceleration to body A: a_A = (m_B/M) * a_rel
            a_pn_A[k] += prefactor * (A3 * n[k] + B3 * dv[k]) * fA;
        }
    }

    // PN3.5 (dissipative, c^-7) — Blanchet 2014 / Iyer & Will 1995
    // Higher-order radiation reaction beyond PN2.5
    if (pn_order & 16) {
        double M = m_A + m_B;
        double mu_rel = G * M;
        double eta = m_A * m_B / (M * M);
        double inv_c7 = inv_c2 * inv_c2 * inv_c2 / c_code;
        double Gm_r = mu_rel * inv_r;
        double dv2 = dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2];
        double r_dot_v = r_AB[0]*dv[0] + r_AB[1]*dv[1] + r_AB[2]*dv[2];
        double n_dot_v = r_dot_v * inv_r;

        double Gm_r2 = Gm_r * Gm_r;
        double ndv2 = n_dot_v * n_dot_v;

        // Coefficient of n (radial) — Iyer & Will 1995 / Blanchet 2014 Eq. (205)
        double A35 = n_dot_v * eta * (
            + (8.0/5.0) * Gm_r * (
                + (23.0/14.0 - 295.0/14.0*eta) * dv2 * dv2
                + Gm_r * (-(1690.0/21.0 + 15.0*eta) * dv2 + (326.0/21.0 + 536.0/7.0*eta) * ndv2)
                + Gm_r2 * inv_r * (12404.0/105.0 + 1816.0/21.0*eta)
            )
        );

        // Coefficient of v (tangential)
        double B35 = eta * (
            + (8.0/5.0) * Gm_r * (
                + (-3.0/14.0 + 55.0/14.0*eta) * dv2 * dv2
                + Gm_r * ((454.0/21.0 - 3.0*eta) * dv2 + (-74.0/21.0 - 148.0/7.0*eta) * ndv2)
                + Gm_r2 * inv_r * (-4538.0/105.0 - 344.0/21.0*eta)
            )
        );

        double fA = m_B / M;
        double prefactor = mu_rel * inv_r2 * inv_c7;
        for (int k = 0; k < 3; k++) {
            a_pn_A[k] += prefactor * (A35 * n[k] + B35 * dv[k]) * fA;
        }
    }
}

// ============================================================================
// PN1 conserved energy correction (Poisson & Will 2014, Section 9.3)
// E_total = T + U + E_pn1 is conserved at PN1 order.
//
// E_pn1 = (1/c^2) * { (3/8) Sigma_A m_A v_A^4
//          + Sigma_{A<B} G*m_A*m_B / (2*r_AB) * [3(v_A^2+v_B^2) - 7(v_A.v_B) - (n.v_A)(n.v_B)]
//          - Sigma_{A<B} G^2*m_A*m_B*(m_A+m_B) / (2*r_AB^2) }
// ============================================================================

double pn1_energy(const double* pos, const double* vel, const double* mass,
                  int n, double G, double c_code) {
    double c2 = c_code * c_code;
    double inv_c2 = 1.0 / c2;
    double E_kin = 0.0;
    double E_pot_kin = 0.0;
    double E_pot_pot = 0.0;

    // Kinetic correction: (3/8) * Sigma_A m_A * v_A^4
    for (int A = 0; A < n; A++) {
        double vA2 = vel[A*3]*vel[A*3] + vel[A*3+1]*vel[A*3+1] + vel[A*3+2]*vel[A*3+2];
        E_kin += mass[A] * vA2 * vA2;
    }
    E_kin *= 3.0 / 8.0;

    // Pair terms
    for (int A = 0; A < n; A++) {
        for (int B = A + 1; B < n; B++) {
            double dx = pos[A*3] - pos[B*3];
            double dy = pos[A*3+1] - pos[B*3+1];
            double dz = pos[A*3+2] - pos[B*3+2];
            double r2 = dx*dx + dy*dy + dz*dz;
            double r = std::sqrt(r2);
            double inv_r = 1.0 / r;

            double n_AB[3] = { dx*inv_r, dy*inv_r, dz*inv_r };

            double vA2 = vel[A*3]*vel[A*3] + vel[A*3+1]*vel[A*3+1] + vel[A*3+2]*vel[A*3+2];
            double vB2 = vel[B*3]*vel[B*3] + vel[B*3+1]*vel[B*3+1] + vel[B*3+2]*vel[B*3+2];
            double vA_dot_vB = vel[A*3]*vel[B*3] + vel[A*3+1]*vel[B*3+1] + vel[A*3+2]*vel[B*3+2];
            double n_dot_vA = n_AB[0]*vel[A*3] + n_AB[1]*vel[A*3+1] + n_AB[2]*vel[A*3+2];
            double n_dot_vB = n_AB[0]*vel[B*3] + n_AB[1]*vel[B*3+1] + n_AB[2]*vel[B*3+2];

            double Gm_AB = G * mass[A] * mass[B];

            // Potential-kinetic term
            E_pot_kin += Gm_AB * inv_r * 0.5 *
                (3.0 * (vA2 + vB2) - 7.0 * vA_dot_vB - n_dot_vA * n_dot_vB);

            // Potential-potential term
            E_pot_pot -= G * G * mass[A] * mass[B] * (mass[A] + mass[B]) * 0.5 * inv_r * inv_r;
        }
    }

    return inv_c2 * (E_kin + E_pot_kin + E_pot_pot);
}

// ============================================================================
// PN2.5 instantaneous GW power (Peters 1964 quadrupole formula)
// For a two-body system: P_GW = (32/5) * G^4 * m1^2 * m2^2 * (m1+m2) / (c^5 * r^5)
// Generalized to N-body via the quadrupole moment.
//
// For practical use (two-body subsystems), we use the Peters formula directly.
// ============================================================================

double pn25_power(const double* pos, const double* vel, const double* mass,
                  int n, double G, double c_code) {
    if (n != 2) return 0.0;  // only implemented for two-body

    double dx = pos[3] - pos[0];
    double dy = pos[4] - pos[1];
    double dz = pos[5] - pos[2];
    double r2 = dx*dx + dy*dy + dz*dz;
    double r = std::sqrt(r2);

    double dvx = vel[3] - vel[0];
    double dvy = vel[4] - vel[1];
    double dvz = vel[5] - vel[2];
    double v2 = dvx*dvx + dvy*dvy + dvz*dvz;
    double rdotv = dx*dvx + dy*dvy + dz*dvz;
    double rdot = rdotv / r;

    double c5 = c_code * c_code * c_code * c_code * c_code;
    double G3 = G * G * G;
    double m1 = mass[0], m2 = mass[1];
    double mu = m1 * m2 / (m1 + m2);
    double M = m1 + m2;

    // Instantaneous GW luminosity (Blanchet 2014 Eq. 4.8)
    // L_GW = (8/15) * G^4/(c^5) * mu^2 * M^3 / r^4 * (12*v^2 - 11*rdot^2)
    double inv_r4 = 1.0 / (r2 * r2);
    double L_gw = (8.0/15.0) * G3 * G * mu * mu * M * M * M * inv_r4 / c5
                  * (12.0 * v2 - 11.0 * rdot * rdot);

    return L_gw;
}

// ============================================================================
// Legacy relative PN acceleration (renamed from pn_acceleration)
// Blanchet 2014 / Will 1996 conventions, relative coordinates.
// Kept for regression testing.
// ============================================================================

void pn_acceleration_relative(const double r[3], const double v[3],
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
        double A1 = 2.0*(2.0 + eta) * Gm_r - (1.0 + 3.0*eta) * v2
                    + 1.5 * eta * n_dot_v * n_dot_v;
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

    // PN3.0 (conservative, c^-6) — Blanchet 2014
    if (pn_order & 8) {
        double inv_c6 = inv_c4 * inv_c2;
        double Gm_r2 = Gm_r * Gm_r;
        double Gm_r3 = Gm_r2 * Gm_r * inv_r;
        double v4 = v2 * v2;
        double v6 = v4 * v2;
        double ndv2 = n_dot_v * n_dot_v;
        double ndv4 = ndv2 * ndv2;

        double A3 =
            + eta * (3.0 - 6.0*eta + 3.0*eta*eta) * v6 / 16.0
            + eta * Gm_r * (
                - (13.5 - 5.0*eta + 21.75*eta*eta) * v4
                + eta * (6.0 - 22.5*eta) * v2 * ndv2
                + (7.5*eta*eta) * ndv4
            )
            + Gm_r2 * inv_r * (
                + (16.0 + 35.0/6.0*eta - 203.0/12.0*eta*eta) * v2
                + eta * (-31.5 + 22.5*eta) * ndv2
            )
            + Gm_r3 * (
                + (227.0/12.0 - 53.0/4.0*eta + 41.0/64.0*M_PI*M_PI*eta)
            );

        double B3 =
            + eta * Gm_r * n_dot_v * (
                (13.5 - 15.75*eta + 2.0*eta*eta) * v2
                - eta * (3.0 - 7.5*eta) * ndv2
            )
            + Gm_r2 * inv_r * n_dot_v * (
                + (-16.0 + 13.0*eta + 4.5*eta*eta)
            );

        for (int k = 0; k < 3; k++) {
            a_pn[k] += inv_c6 * mu * inv_r2 * (A3 * r[k] * inv_r + B3 * v[k]);
        }
    }

    // PN3.5 (dissipative, c^-7) — Iyer & Will 1995 / Blanchet 2014
    if (pn_order & 16) {
        double inv_c7 = inv_c4 * inv_c2 / c_code;
        double Gm_r2 = Gm_r * Gm_r;
        double ndv2 = n_dot_v * n_dot_v;

        double A35 = n_dot_v * eta * (
            (8.0/5.0) * Gm_r * (
                (23.0/14.0 - 295.0/14.0*eta) * v2 * v2
                + Gm_r * (-(1690.0/21.0 + 15.0*eta) * v2 + (326.0/21.0 + 536.0/7.0*eta) * ndv2)
                + Gm_r2 * inv_r * (12404.0/105.0 + 1816.0/21.0*eta)
            )
        );

        double B35 = eta * (
            (8.0/5.0) * Gm_r * (
                (-3.0/14.0 + 55.0/14.0*eta) * v2 * v2
                + Gm_r * ((454.0/21.0 - 3.0*eta) * v2 + (-74.0/21.0 - 148.0/7.0*eta) * ndv2)
                + Gm_r2 * inv_r * (-4538.0/105.0 - 344.0/21.0*eta)
            )
        );

        for (int k = 0; k < 3; k++) {
            a_pn[k] += inv_c7 * mu * inv_r2 * (A35 * r[k] * inv_r + B35 * v[k]);
        }
    }
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
    double coeff_SO_1 = G / (2.0 * c2 * r2 * r_mag) * (4.0 + 3.0 * m2 / m1);
    double coeff_SO_2 = G / (2.0 * c2 * r2 * r_mag) * (4.0 + 3.0 * m1 / m2);

    Omega_prec_1[0] = coeff_SO_1 * Lx;
    Omega_prec_1[1] = coeff_SO_1 * Ly;
    Omega_prec_1[2] = coeff_SO_1 * Lz;

    Omega_prec_2[0] = coeff_SO_2 * (-Lx);
    Omega_prec_2[1] = coeff_SO_2 * (-Ly);
    Omega_prec_2[2] = coeff_SO_2 * (-Lz);

    // Spin-spin precession
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
// (BIFROST Section 3.7) — now using EIH formulation directly
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
    if (n_compact < 2) return;

    // Accumulate PN1.0 velocity kicks on host
    std::vector<double> dvx(n_compact, 0.0), dvy(n_compact, 0.0), dvz(n_compact, 0.0);

    for (int ii = 0; ii < n_compact; ii++) {
        int i = compact[ii];
        int sid_i = h_sid(i);

        for (int jj = ii + 1; jj < n_compact; jj++) {
            int j = compact[jj];
            int sid_j = h_sid(j);

            // Skip pairs in the same subsystem (they get full PN inside LogH)
            if (sid_i >= 0 && sid_j == sid_i) continue;

            double x_i[3] = { h_x(i), h_y(i), h_z(i) };
            double v_i[3] = { h_vx(i), h_vy(i), h_vz(i) };
            double x_j[3] = { h_x(j), h_y(j), h_z(j) };
            double v_j[3] = { h_vx(j), h_vy(j), h_vz(j) };

            // EIH: get acceleration of i due to j, and j due to i, directly
            double a_pn_i[3], a_pn_j[3];
            eih_pn_acceleration(x_i, v_i, h_m(i), x_j, v_j, h_m(j),
                                G, c_code, a_pn_i, 1);
            eih_pn_acceleration(x_j, v_j, h_m(j), x_i, v_i, h_m(i),
                                G, c_code, a_pn_j, 1);

            dvx[ii] += dt * a_pn_i[0];
            dvy[ii] += dt * a_pn_i[1];
            dvz[ii] += dt * a_pn_i[2];
            dvx[jj] += dt * a_pn_j[0];
            dvy[jj] += dt * a_pn_j[1];
            dvz[jj] += dt * a_pn_j[2];
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
