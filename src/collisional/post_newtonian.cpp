#include "post_newtonian.h"
#include "constants.h"
#include "diagnostics.h"
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
// Spin-dependent PN dynamics of a binary (BIFROST sec. 3.7; Thorne & Hartle 1985; explicit forms of
// Kidder 1995, PRD 52, 821, eqs. 2.2c-d, 2.4, 2.9, covariant spin supplementary condition) plus the
// spin-induced quadrupole of a Kerr black hole (Poisson 1998, PRD 57, 5287; Racine 2008, PRD 78, 044021).
// Relative coordinates x = x1 - x2, v = v1 - v2, n = x / r; S_i are physical spins (|S| = chi G m^2 / c).
// ============================================================================

namespace {
inline double dot3(const double a[3], const double b[3]) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
inline void cross3(const double a[3], const double b[3], double o[3]) {
    o[0] = a[1]*b[2] - a[2]*b[1];
    o[1] = a[2]*b[0] - a[0]*b[2];
    o[2] = a[0]*b[1] - a[1]*b[0];
}
}  // namespace

void spin_precession(const double x[3], const double v[3],
                     double m1, double m2, const double S1[3], const double S2[3],
                     double G, double c_code,
                     double Omega1[3], double Omega2[3]) {
    const double r = std::sqrt(dot3(x, x));
    const double n[3] = {x[0] / r, x[1] / r, x[2] / r};
    const double mu = m1 * m2 / (m1 + m2);
    double LN[3];
    cross3(x, v, LN);
    for (int k = 0; k < 3; k++) LN[k] *= mu;   // Newtonian orbital angular momentum mu x × v
    const double pref = G / (c_code * c_code * r * r * r);
    const double nS1 = dot3(n, S1), nS2 = dot3(n, S2);
    // Spin-orbit (geodetic, 1.5PN): (2 + 3 m_j / (2 m_i)) L_N; spin-spin (2PN): -S_j + 3 (n·S_j) n;
    // Kerr quadrupole (2PN): 3 (m_j / m_i) (n·S_i) n
    const double so1 = 2.0 + 1.5 * m2 / m1, so2 = 2.0 + 1.5 * m1 / m2;
    for (int k = 0; k < 3; k++) {
        Omega1[k] = pref * (so1 * LN[k] - S2[k] + 3.0 * nS2 * n[k] + 3.0 * (m2 / m1) * nS1 * n[k]);
        Omega2[k] = pref * (so2 * LN[k] - S1[k] + 3.0 * nS1 * n[k] + 3.0 * (m1 / m2) * nS2 * n[k]);
    }
}

void spin_pn_acceleration(const double x[3], const double v[3],
                          double m1, double m2, const double S1[3], const double S2[3],
                          double G, double c_code, double a_rel[3]) {
    const double M = m1 + m2, mu = m1 * m2 / M;
    const double r = std::sqrt(dot3(x, x));
    const double n[3] = {x[0] / r, x[1] / r, x[2] / r};
    const double rdot = dot3(n, v);
    const double c2 = c_code * c_code;
    // Kidder's combinations with S = S1 + S2, Delta = M (S2/m2 - S1/m1), dm = m1 - m2:
    //   A = 2S + (dm/M) Delta,  B = 7S + 3 (dm/M) Delta,  C = 3S + (dm/M) Delta
    double A[3], B[3], C[3];
    for (int k = 0; k < 3; k++) {
        A[k] = (1.0 + m2 / m1) * S1[k] + (1.0 + m1 / m2) * S2[k];
        B[k] = (4.0 + 3.0 * m2 / m1) * S1[k] + (4.0 + 3.0 * m1 / m2) * S2[k];
        C[k] = (2.0 + m2 / m1) * S1[k] + (2.0 + m1 / m2) * S2[k];
    }
    double nxv[3], vxB[3], nxC[3];
    cross3(n, v, nxv);
    cross3(v, B, vxB);
    cross3(n, C, nxC);
    const double pso = G / (c2 * r * r * r);
    const double nvA = dot3(nxv, A);
    // Spin-orbit (1.5PN), Kidder eq. 2.2c
    for (int k = 0; k < 3; k++) a_rel[k] = pso * (6.0 * n[k] * nvA - vxB[k] + 3.0 * rdot * nxC[k]);
    // Spin-spin (2PN), Kidder eq. 2.2d
    const double nS1 = dot3(n, S1), nS2 = dot3(n, S2), S1S2 = dot3(S1, S2);
    const double pss = -3.0 * G / (c2 * mu * r * r * r * r);
    for (int k = 0; k < 3; k++)
        a_rel[k] += pss * (n[k] * S1S2 + S1[k] * nS2 + S2[k] * nS1 - 5.0 * n[k] * nS1 * nS2);
    // Kerr spin-induced quadrupole (2PN, Q_i = -S_i^2 / m_i; Racine 2008 with kappa = 1), from the potential
    // (G m_j / (2 c^2 m_i r^3)) [3 (n·S_i)^2 - S_i^2]
    const double q1 = 0.5 * pss * m2 / m1, q2 = 0.5 * pss * m1 / m2;
    const double S1S1 = dot3(S1, S1), S2S2 = dot3(S2, S2);
    for (int k = 0; k < 3; k++) {
        a_rel[k] += q1 * (n[k] * S1S1 + 2.0 * S1[k] * nS1 - 5.0 * n[k] * nS1 * nS1);
        a_rel[k] += q2 * (n[k] * S2S2 + 2.0 * S2[k] * nS2 - 5.0 * n[k] * nS2 * nS2);
    }
}

void binary_total_angular_momentum(const double x[3], const double v[3],
                                   double m1, double m2, const double S1[3], const double S2[3],
                                   double G, double c_code, bool pn1, double J[3]) {
    const double M = m1 + m2, mu = m1 * m2 / M, eta = mu / M;
    const double r = std::sqrt(dot3(x, x));
    const double n[3] = {x[0] / r, x[1] / r, x[2] / r};
    const double c2 = c_code * c_code;
    double LN[3];
    cross3(x, v, LN);
    // L_N (1 + L_PN / L_N), Kidder eq. 2.9b
    const double fpn = pn1 ? (0.5 * dot3(v, v) * (1.0 - 3.0 * eta) + (3.0 + eta) * G * M / r) / c2 : 0.0;
    double A[3], C[3];
    for (int k = 0; k < 3; k++) {
        A[k] = (m2 / m1) * S1[k] + (m1 / m2) * S2[k];
        C[k] = (2.0 + m2 / m1) * S1[k] + (2.0 + m1 / m2) * S2[k];
    }
    // L_SO = (mu/M) [ (G M / r) n × (n × (3S + (dm/M) Delta)) - (1/2) v × (v × (S + (dm/M) Delta)) ] / c^2
    // (Kidder eq. 2.9c; the combination makes dJ/dt = 0 at 1.5PN with the a_SO and Omega_SO above)
    double nxC[3], nnC[3], vxA[3], vvA[3];
    cross3(n, C, nxC);
    cross3(n, nxC, nnC);
    cross3(v, A, vxA);
    cross3(v, vxA, vvA);
    for (int k = 0; k < 3; k++)
        J[k] = mu * LN[k] * (1.0 + fpn) + eta / c2 * (G * M / r * nnC[k] - 0.5 * vvA[k]) + S1[k] + S2[k];
}

void rotate_about(double S[3], const double Omega[3], double dt) {
    const double w = std::sqrt(dot3(Omega, Omega));
    if (w * std::fabs(dt) == 0.0) return;
    const double th = w * dt, k[3] = {Omega[0] / w, Omega[1] / w, Omega[2] / w};
    const double ct = std::cos(th), st = std::sin(th), kS = dot3(k, S);
    double kxS[3];
    cross3(k, S, kxS);
    for (int i = 0; i < 3; i++) S[i] = S[i] * ct + kxS[i] * st + k[i] * kS * (1.0 - ct);   // Rodrigues
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

    // Apply velocity kicks; the Newtonian E, P, L they change go to the PN sink (diagnostics are Newtonian)
    for (int ii = 0; ii < n_compact; ii++) {
        int i = compact[ii];
        double r[3] = {h_x(i), h_y(i), h_z(i)}, v0[3] = {h_vx(i), h_vy(i), h_vz(i)};
        h_vx(i) += dvx[ii];
        h_vy(i) += dvy[ii];
        h_vz(i) += dvz[ii];
        double v1[3] = {h_vx(i), h_vy(i), h_vz(i)};
        g_energy_books.E_pn += 0.5 * h_m(i) * ((v0[0]*v0[0] + v0[1]*v0[1] + v0[2]*v0[2]) -
                                               (v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]));
        g_energy_books.sink_PL(h_m(i), r, v0);
        g_energy_books.sink_PL(h_m(i), r, v1, -1.0);
    }

    // Copy velocities back to device
    Kokkos::deep_copy(p.vx, h_vx);
    Kokkos::deep_copy(p.vy, h_vy);
    Kokkos::deep_copy(p.vz, h_vz);
    p.zero_kick_compensation();   // velocities changed outside the compensated kicks
}
