#include "post_newtonian_core.h"
#include <cmath>
#include <cstring>

// The pure post-Newtonian functions (no Kokkos; tests/enzyme differentiates them): EIH accelerations, PN1 energy,
// spin dynamics. global_pn1_kick (Particles) is in post_newtonian.cpp.

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
// eih_pn_acceleration of both bodies of a pair at once, bit-identical to the two calls (A = i, B = j) and
// (A = j, B = i), with what both share computed once: 1/r (|x_j - x_i| = |x_i - x_j| exactly: IEEE subtraction
// is sign-symmetric) and the coefficients A, B and prefactors of PN2.5 / 3.0 / 3.5 (functions of 1/r, |dv|^2,
// r.dv, M and eta, all exactly symmetric in i, j). The per-body part (eih_body) repeats the operations of
// eih_pn_acceleration in its own, not inlined function, so that no expression is shared between the two bodies
// and every floating-point contraction (FMA, -march=native) comes out as there. Checked bitwise on 5e6 random
// pairs over all 31 orders, host and nvcc_wrapper builds; NBODY_CHECK_EIH=1 checks every pair of a run.
// ============================================================================

namespace {
struct EihPairCoeffs {   // the pair-symmetric prefactors and coefficients of PN2.5, PN3.0, PN3.5
    double M, p25, A25, B25, p3, A3, B3, p35, A35, B35;
};
}  // namespace

// PN terms of body A from B (eih_pn_acceleration) with 1/r and the pair coefficients h given
[[gnu::noinline]] static void eih_body(const double x_A[3], const double v_A[3], double m_A,
                                       const double x_B[3], const double v_B[3], double m_B,
                                       double G, double inv_c2, double inv_r, const EihPairCoeffs& h,
                                       int pn_order, double a_pn_A[3]) {
    a_pn_A[0] = a_pn_A[1] = a_pn_A[2] = 0.0;

    double r_AB[3] = { x_A[0]-x_B[0], x_A[1]-x_B[1], x_A[2]-x_B[2] };
    double inv_r2 = inv_r * inv_r;
    double n[3] = { r_AB[0]*inv_r, r_AB[1]*inv_r, r_AB[2]*inv_r };
    double dv[3] = { v_A[0]-v_B[0], v_A[1]-v_B[1], v_A[2]-v_B[2] };
    double vA2 = v_A[0]*v_A[0] + v_A[1]*v_A[1] + v_A[2]*v_A[2];
    double vB2 = v_B[0]*v_B[0] + v_B[1]*v_B[1] + v_B[2]*v_B[2];
    double vA_dot_vB = v_A[0]*v_B[0] + v_A[1]*v_B[1] + v_A[2]*v_B[2];
    double n_dot_vA = n[0]*v_A[0] + n[1]*v_A[1] + n[2]*v_A[2];
    double n_dot_vB = n[0]*v_B[0] + n[1]*v_B[1] + n[2]*v_B[2];
    double Gm_A = G * m_A;
    double Gm_B = G * m_B;

    if (pn_order & 1) {
        double coeff_n = -vA2 - 2.0*vB2 + 4.0*vA_dot_vB
                         + 1.5 * n_dot_vB * n_dot_vB
                         + 5.0 * Gm_A * inv_r + 4.0 * Gm_B * inv_r;
        double coeff_v = 4.0 * n_dot_vA - 3.0 * n_dot_vB;
        double prefactor = Gm_B * inv_r2 * inv_c2;
        for (int k = 0; k < 3; k++) {
            a_pn_A[k] += prefactor * (coeff_n * n[k] + coeff_v * dv[k]);
        }
    }

    if (pn_order & 2) {
        double inv_c4 = inv_c2 * inv_c2;
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

    const double fA = m_B / h.M;
    if (pn_order & 4) {
        for (int k = 0; k < 3; k++) {
            a_pn_A[k] += h.p25 * (-h.A25 * n[k] * fA + h.B25 * (-dv[k]) * fA);
        }
    }
    if (pn_order & 8) {
        for (int k = 0; k < 3; k++) {
            a_pn_A[k] += h.p3 * (h.A3 * n[k] + h.B3 * dv[k]) * fA;
        }
    }
    if (pn_order & 16) {
        for (int k = 0; k < 3; k++) {
            a_pn_A[k] += h.p35 * (h.A35 * n[k] + h.B35 * dv[k]) * fA;
        }
    }
}

void eih_pn_acceleration_pair(const double x_i[3], const double v_i[3], double m_i,
                              const double x_j[3], const double v_j[3], double m_j,
                              double G, double c_code, double a_i[3], double a_j[3], int pn_order) {
    // Geometry seen from i (A = i, B = j), as in eih_pn_acceleration
    double r_AB[3] = { x_i[0]-x_j[0], x_i[1]-x_j[1], x_i[2]-x_j[2] };
    double r2 = r_AB[0]*r_AB[0] + r_AB[1]*r_AB[1] + r_AB[2]*r_AB[2];
    double r_mag = std::sqrt(r2);
    double inv_r = 1.0 / r_mag;
    double inv_r2 = inv_r * inv_r;
    double dv[3] = { v_i[0]-v_j[0], v_i[1]-v_j[1], v_i[2]-v_j[2] };
    double c2 = c_code * c_code;
    double inv_c2 = 1.0 / c2;

    EihPairCoeffs h{};
    h.M = m_i + m_j;
    if (pn_order & 4) {
        double inv_c5 = inv_c2 * inv_c2 / c_code;
        double dv2 = dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2];
        // 3 |dv|^2 before A25: in eih_pn_acceleration the (unused) PN2.5 coeff_n computes it first, which makes
        // the FMA of A25's sum take 3 |dv|^2 as its product (GCC 13; otherwise ~20 % of results differ by 1 ulp)
        double dv2_3 = 3.0 * dv2;
        double M = m_i + m_j;
        double mu_rel = G * M;
        double eta = m_i * m_j / (M * M);
        double r_dot_v = r_AB[0]*dv[0] + r_AB[1]*dv[1] + r_AB[2]*dv[2];
        double n_dot_v_rel = r_dot_v * inv_r;
        h.A25 = 8.0/5.0 * eta * mu_rel * inv_r * (17.0/3.0 * mu_rel * inv_r + dv2_3) * n_dot_v_rel;
        h.B25 = -8.0/5.0 * eta * mu_rel * inv_r * (3.0 * mu_rel * inv_r + dv2);
        h.p25 = mu_rel * inv_r2 * inv_c5;
    }
    if (pn_order & 8) {
        double M = m_i + m_j;
        double mu_rel = G * M;
        double eta = m_i * m_j / (M * M);
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

        h.A3 =
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
        h.B3 =
            + eta * Gm_r * n_dot_v * (
                (13.5 - 15.75*eta + 2.0*eta*eta) * dv2
                - eta * (3.0 - 7.5*eta) * ndv2
            )
            + Gm_r2 * inv_r * n_dot_v * (
                + (-16.0 + 13.0*eta + 4.5*eta*eta)
            );
        h.p3 = mu_rel * inv_r2 * inv_c6;
    }
    if (pn_order & 16) {
        double M = m_i + m_j;
        double mu_rel = G * M;
        double eta = m_i * m_j / (M * M);
        double inv_c7 = inv_c2 * inv_c2 * inv_c2 / c_code;
        double Gm_r = mu_rel * inv_r;
        double dv2 = dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2];
        double r_dot_v = r_AB[0]*dv[0] + r_AB[1]*dv[1] + r_AB[2]*dv[2];
        double n_dot_v = r_dot_v * inv_r;

        double Gm_r2 = Gm_r * Gm_r;
        double ndv2 = n_dot_v * n_dot_v;

        h.A35 = n_dot_v * eta * (
            + (8.0/5.0) * Gm_r * (
                + (23.0/14.0 - 295.0/14.0*eta) * dv2 * dv2
                + Gm_r * (-(1690.0/21.0 + 15.0*eta) * dv2 + (326.0/21.0 + 536.0/7.0*eta) * ndv2)
                + Gm_r2 * inv_r * (12404.0/105.0 + 1816.0/21.0*eta)
            )
        );
        h.B35 = eta * (
            + (8.0/5.0) * Gm_r * (
                + (-3.0/14.0 + 55.0/14.0*eta) * dv2 * dv2
                + Gm_r * ((454.0/21.0 - 3.0*eta) * dv2 + (-74.0/21.0 - 148.0/7.0*eta) * ndv2)
                + Gm_r2 * inv_r * (-4538.0/105.0 - 344.0/21.0*eta)
            )
        );
        h.p35 = mu_rel * inv_r2 * inv_c7;
    }

    eih_body(x_i, v_i, m_i, x_j, v_j, m_j, G, inv_c2, inv_r, h, pn_order, a_i);   // A = i, B = j
    eih_body(x_j, v_j, m_j, x_i, v_i, m_i, G, inv_c2, inv_r, h, pn_order, a_j);   // A = j, B = i
}

// ============================================================================
// PN1 conserved energy correction (Poisson & Will 2014, Section 9.3)
// E_total = T + U + E_pn1 is conserved at PN1 order.
//
// E_pn1 = (1/c^2) * { (3/8) Sigma_A m_A v_A^4
//          + Sigma_{A<B} G*m_A*m_B / (2*r_AB) * [3(v_A^2+v_B^2) - 7(v_A.v_B) - (n.v_A)(n.v_B)]
//          + Sigma_{A<B} G^2*m_A*m_B*(m_A+m_B) / (2*r_AB^2) }
// The G^2 term is positive (two bodies: E/mu = ... + (GM/2r)[(3 + nu) v^2 + nu rdot^2 + GM/r], Blanchet 2014,
// Living Rev. Relativ. 17, 2, eq. 233 at 1PN). It had the opposite sign until 2026-10-17: found with Enzyme
// (tests/enzyme/pn_consistency.cpp: dE/dt under the PN1 equations of motion scaled as c^-2 instead of c^-4);
// test pn1_energy_consistency. Pair terms only: the genuine three-body terms of the N-body EIH energy (and
// acceleration) are missing, so for n >= 3 there is no conserved 1PN energy (docs/CASES.md C7)
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
            E_pot_pot += G * G * mass[A] * mass[B] * (mass[A] + mass[B]) * 0.5 * inv_r * inv_r;
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
