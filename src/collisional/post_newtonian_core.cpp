#include "post_newtonian_core.h"
#include <cmath>
#include <cstring>

// The pure post-Newtonian functions (no Kokkos; tests/enzyme differentiates them): EIH accelerations, PN1 energy,
// spin dynamics. global_pn1_kick (Particles) is in post_newtonian.cpp.

// ============================================================================
// PN2.5, PN3.0 and PN3.5 of a pair: the centre-of-mass relative acceleration of Blanchet, Living Rev. Rel. 17, 2
// (2014, v3 2024), eq. (dvdt): a = a_1 - a_2 = -(G M / r^2) [A n + B v], n = (x_1 - x_2)/r, v = v_1 - v_2,
// r' = n.v, with A, B from eqs. (calAcoeff), (calBcoeff); 3PN in modified harmonic coordinates (eq. ABcoeffMH,
// no ln(r/r'_0)). Body 1 gets (m_2/M) a and body 2 -(m_1/M) a (momentum exact). Returned as C_n, C_v with
// a = C_n n + C_v v, summed over the orders in pn_order (bits 4, 8, 16). The inputs are symmetric in the two bodies
// (r' = (-n).(-v)), so both single-body calls and the pair call get the same bits. Checked: circular-orbit energy
// loss = GW flux to 1PN (32/5 nu^2 x^5 [1 - (1247/336 + 35/12 nu) x]), tests/pn_energy_test.cpp. (Up to 2026-10
// these orders had the wrong sign at 2.5PN, wrong 3.5PN coefficients (+11 x the 2.5PN loss, as a gain) and
// incomplete 3PN coefficients: CASES C8.)
// ============================================================================
[[gnu::noinline]] static void pn_rel_highorder(double inv_r, double v2, double rd, double Gm, double nu,
                                               double c_code, int pn_order, double& Cn, double& Cv) {
    const double inv_c2 = 1.0 / (c_code * c_code);
    const double u = Gm * inv_r;
    const double pi2 = M_PI * M_PI;
    double A = 0.0, B = 0.0;
    if (pn_order & 4) {
        const double inv_c5 = inv_c2 * inv_c2 / c_code;
        const double f = 8.0/5.0 * nu * u;
        A += inv_c5 * f * rd * (-17.0/3.0 * u - 3.0 * v2);
        B += inv_c5 * f * (3.0 * u + v2);
    }
    if (pn_order & 8) {
        const double inv_c6 = inv_c2 * inv_c2 * inv_c2;
        const double nu2 = nu * nu, nu3 = nu2 * nu;
        const double rd2 = rd * rd, rd3 = rd2 * rd, rd4 = rd2 * rd2, rd5 = rd4 * rd, rd6 = rd4 * rd2;
        const double v4 = v2 * v2, v6 = v4 * v2;
        const double A3 =
              rd6 * (-35.0/16.0 * nu + 175.0/16.0 * nu2 - 175.0/16.0 * nu3)
            + rd4 * v2 * (15.0/2.0 * nu - 135.0/4.0 * nu2 + 255.0/8.0 * nu3)
            + rd2 * v4 * (-15.0/2.0 * nu + 237.0/8.0 * nu2 - 45.0/2.0 * nu3)
            + v6 * (11.0/4.0 * nu - 49.0/4.0 * nu2 + 13.0 * nu3)
            + u * (rd4 * (79.0 * nu - 69.0/2.0 * nu2 - 30.0 * nu3)
                   + rd2 * v2 * (-121.0 * nu + 16.0 * nu2 + 20.0 * nu3)
                   + v4 * (75.0/4.0 * nu + 8.0 * nu2 - 10.0 * nu3))
            + u * u * (rd2 * (1.0 + 22717.0/168.0 * nu + 11.0/8.0 * nu2 - 7.0 * nu3 + 615.0/64.0 * pi2 * nu)
                       + v2 * (-20827.0/840.0 * nu + nu3 - 123.0/64.0 * pi2 * nu))
            + u * u * u * (-16.0 - 1399.0/12.0 * nu - 71.0/2.0 * nu2 + 41.0/16.0 * pi2 * nu);
        const double B3 =
              rd5 * (-45.0/8.0 * nu + 15.0 * nu2 + 15.0/4.0 * nu3)
            + rd3 * v2 * (12.0 * nu - 111.0/4.0 * nu2 - 12.0 * nu3)
            + rd * v4 * (-65.0/8.0 * nu + 19.0 * nu2 + 6.0 * nu3)
            + u * (rd3 * (329.0/6.0 * nu + 59.0/2.0 * nu2 + 18.0 * nu3)
                   + rd * v2 * (-15.0 * nu - 27.0 * nu2 - 10.0 * nu3))
            + u * u * rd * (-4.0 - 5849.0/840.0 * nu + 25.0 * nu2 + 8.0 * nu3 - 123.0/32.0 * pi2 * nu);
        A += inv_c6 * A3;
        B += inv_c6 * B3;
    }
    if (pn_order & 16) {
        const double inv_c7 = inv_c2 * inv_c2 * inv_c2 / c_code;
        const double rd2 = rd * rd, v4 = v2 * v2;
        const double f = nu * u;
        A += inv_c7 * f * rd * (u * u * (3956.0/35.0 + 184.0/5.0 * nu) + u * v2 * (692.0/35.0 - 724.0/15.0 * nu)
                                + v4 * (366.0/35.0 + 12.0 * nu) + u * rd2 * (294.0/5.0 + 376.0/5.0 * nu)
                                - v2 * rd2 * (114.0 + 12.0 * nu) + 112.0 * rd2 * rd2);
        B += inv_c7 * f * (u * u * (-1060.0/21.0 - 104.0/5.0 * nu) + u * v2 * (164.0/21.0 + 148.0/5.0 * nu)
                           + v4 * (-626.0/35.0 - 12.0/5.0 * nu) + u * rd2 * (-82.0/3.0 - 848.0/15.0 * nu)
                           + v2 * rd2 * (678.0/5.0 + 12.0/5.0 * nu) - 120.0 * rd2 * rd2);
    }
    const double g = -Gm * inv_r * inv_r;
    Cn = g * A;
    Cv = g * B;
}

// ============================================================================
// EIH (Einstein-Infeld-Hoffmann) PN acceleration
// Computes acceleration of body A due to body B directly.
// (Poisson & Will 2014, Eq. 9.139 / Rantala et al. 2023, Eq. 5)
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

    // PN1.0 (conservative) — Poisson & Will 2014 Eq. 9.139 / Rantala et al. 2023, Eq. 5
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

    // PN2.0 (conservative): Blanchet LRR 2014 eq. (acc3PN), the 1/c^4 terms (general frame). Up to 2026-10 the
    // G^2 terms had an extra 1/r and two v-terms wrong (docs/CASES.md C8)
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
            + Gm_A_r * Gm_A_r * (-57.0/4.0)
            + Gm_B_r * Gm_B_r * (-9.0)
            + Gm_A_r * Gm_B_r * (-69.0/2.0);

        double coeff_v =
            vA2 * n_dot_vB + 4.0 * vB2 * n_dot_vA
            - 5.0 * vB2 * n_dot_vB - 4.0 * vA_dot_vB * n_dot_vA
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

    // PN2.5, PN3.0, PN3.5: Blanchet's two-body c.o.m. relative acceleration, body A's share m_B / M
    if (pn_order & 28) {
        double M = m_A + m_B;
        double dv2 = dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2];
        double rd = (r_AB[0]*dv[0] + r_AB[1]*dv[1] + r_AB[2]*dv[2]) * inv_r;
        double Cn, Cv;
        pn_rel_highorder(inv_r, dv2, rd, G * M, m_A * m_B / (M * M), c_code, pn_order, Cn, Cv);
        const double fA = m_B / M;
        for (int k = 0; k < 3; k++) {
            a_pn_A[k] += fA * (Cn * n[k] + Cv * dv[k]);
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
struct EihPairCoeffs {   // the pair-symmetric total mass and PN2.5-3.5 coefficients (pn_rel_highorder)
    double M, Cn, Cv;
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
            + Gm_A_r * Gm_A_r * (-57.0/4.0)
            + Gm_B_r * Gm_B_r * (-9.0)
            + Gm_A_r * Gm_B_r * (-69.0/2.0);

        double coeff_v =
            vA2 * n_dot_vB + 4.0 * vB2 * n_dot_vA
            - 5.0 * vB2 * n_dot_vB - 4.0 * vA_dot_vB * n_dot_vA
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

    if (pn_order & 28) {
        const double fA = m_B / h.M;
        for (int k = 0; k < 3; k++) {
            a_pn_A[k] += fA * (h.Cn * n[k] + h.Cv * dv[k]);
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
    double dv[3] = { v_i[0]-v_j[0], v_i[1]-v_j[1], v_i[2]-v_j[2] };
    double c2 = c_code * c_code;
    double inv_c2 = 1.0 / c2;

    EihPairCoeffs h{};
    h.M = m_i + m_j;
    if (pn_order & 28) {
        double dv2 = dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2];
        double rd = (r_AB[0]*dv[0] + r_AB[1]*dv[1] + r_AB[2]*dv[2]) * inv_r;
        pn_rel_highorder(inv_r, dv2, rd, G * h.M, m_i * m_j / (h.M * h.M), c_code, pn_order, h.Cn, h.Cv);
    }

    eih_body(x_i, v_i, m_i, x_j, v_j, m_j, G, inv_c2, inv_r, h, pn_order, a_i);   // A = i, B = j
    eih_body(x_j, v_j, m_j, x_i, v_i, m_i, G, inv_c2, inv_r, h, pn_order, a_j);   // A = j, B = i
}

// ============================================================================
// Genuine N-body EIH 1PN acceleration (PN1 part). n_AB = (x_A - x_B) / r_AB.
// a_A = (1/c^2) sum_{B != A} { -(G m_B / r_AB^2) n_AB [ v_A^2 + 2 v_B^2 - 4 v_A.v_B - 3/2 (n_AB.v_B)^2
//          - 4 sum_{C != A} G m_C / r_AC - sum_{C != B} G m_C / r_BC + 1/2 sum_{C != B} G m_C (x_B - x_A).(x_C - x_B) / r_BC^3 ]
//        + (G m_B / r_AB^2) [n_AB.(4 v_A - 3 v_B)] (v_A - v_B)
//        - 7/2 (G m_B / r_AB) sum_{C != B} G m_C (x_B - x_C) / r_BC^3 }
// (Poisson & Will 2014 eq. 9.127; checked with Enzyme against the Euler-Lagrange acceleration of the EIH Lagrangian
// and against pn1_energy, tests/enzyme/pn_consistency.cpp). For two bodies it is the pair formula: C = B in the
// first sum gives 4 G m_B / r, C = A in the second G m_A / r, in the third G m_A / (2 r) with the sign that makes
// 5 G m_A / r together with the 7/2 term.
// ============================================================================
void eih1_nbody_acceleration(const double* pos, const double* vel, const double* mass, int n, double G,
                             double c_code, double* a) {
    const double inv_c2 = 1.0 / (c_code * c_code);
    // Newtonian potentials phi_A = sum_{C != A} G m_C / r_AC and fields g_B = sum_{C != B} G m_C (x_C - x_B) / r_BC^3
    double phi[8], g[8][3];
    const int nn = n < 8 ? n : 8;
    for (int A = 0; A < nn; A++) {
        phi[A] = 0.0; g[A][0] = g[A][1] = g[A][2] = 0.0;
        for (int C = 0; C < nn; C++) {
            if (C == A) continue;
            const double d[3] = {pos[3*C] - pos[3*A], pos[3*C+1] - pos[3*A+1], pos[3*C+2] - pos[3*A+2]};
            const double r = std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
            phi[A] += G * mass[C] / r;
            for (int k = 0; k < 3; k++) g[A][k] += G * mass[C] * d[k] / (r * r * r);
        }
    }
    for (int A = 0; A < nn; A++) {
        double acc[3] = {0.0, 0.0, 0.0};
        const double* vA = &vel[3*A];
        const double vA2 = vA[0]*vA[0] + vA[1]*vA[1] + vA[2]*vA[2];
        for (int B = 0; B < nn; B++) {
            if (B == A) continue;
            const double* vB = &vel[3*B];
            const double dAB[3] = {pos[3*A] - pos[3*B], pos[3*A+1] - pos[3*B+1], pos[3*A+2] - pos[3*B+2]};
            const double r = std::sqrt(dAB[0]*dAB[0] + dAB[1]*dAB[1] + dAB[2]*dAB[2]);
            const double nAB[3] = {dAB[0] / r, dAB[1] / r, dAB[2] / r};
            const double vB2 = vB[0]*vB[0] + vB[1]*vB[1] + vB[2]*vB[2];
            const double vAvB = vA[0]*vB[0] + vA[1]*vB[1] + vA[2]*vB[2];
            const double nvA = nAB[0]*vA[0] + nAB[1]*vA[1] + nAB[2]*vA[2];
            const double nvB = nAB[0]*vB[0] + nAB[1]*vB[1] + nAB[2]*vB[2];
            // (x_B - x_A).(x_C - x_B) / r_BC^3 summed with G m_C = (x_B - x_A).g_B
            const double xg = -(dAB[0]*g[B][0] + dAB[1]*g[B][1] + dAB[2]*g[B][2]);
            const double bracket = vA2 + 2.0*vB2 - 4.0*vAvB - 1.5*nvB*nvB - 4.0*phi[A] - phi[B] + 0.5*xg;
            const double Gm_r2 = G * mass[B] / (r * r);
            const double nv43 = 4.0*nvA - 3.0*nvB;
            for (int k = 0; k < 3; k++)
                acc[k] += -Gm_r2 * nAB[k] * bracket + Gm_r2 * nv43 * (vA[k] - vB[k])
                          + 3.5 * (G * mass[B] / r) * g[B][k];
        }
        for (int k = 0; k < 3; k++) a[3*A+k] = inv_c2 * acc[k];
    }
}

// ============================================================================
// eih1_nbody_acceleration and eih_pn_acceleration_pair with the mass- and position-only quantities from PnGeom
// (post_newtonian_core.h). Every cached value below repeats the expression of the original function; the
// velocity-dependent expressions are copied unchanged, with the cached values in place of their computation.
// ============================================================================
void pn_geom_set_masses(const double* mass, int n, double G, double c_code, bool eih_nbody, int pair_order,
                        PnGeom& geom) {
    geom.n = n < PN_GEOM_MAX_BODIES ? n : PN_GEOM_MAX_BODIES;
    geom.eih_nbody = eih_nbody;
    geom.pair_order = pair_order;
    geom.G = G;
    const double c2 = c_code * c_code;
    geom.inv_c2 = 1.0 / c2;
    geom.inv_c4 = geom.inv_c2 * geom.inv_c2;
    geom.inv_c5 = geom.inv_c2 * geom.inv_c2 / c_code;
    geom.inv_c6 = geom.inv_c2 * geom.inv_c2 * geom.inv_c2;
    geom.inv_c7 = geom.inv_c2 * geom.inv_c2 * geom.inv_c2 / c_code;
    for (int A = 0; A < geom.n; A++) geom.Gm[A] = G * mass[A];
    geom.n_pairs = 0;
    if (pair_order == 0) return;
    const double pi2 = M_PI * M_PI;
    for (int i = 0; i < geom.n; i++)
        for (int j = i + 1; j < geom.n; j++) {
            PnPairGeom& p = geom.pair[geom.n_pairs++];
            p.i = i;
            p.j = j;
            p.M = mass[i] + mass[j];
            p.Gm = G * p.M;
            const double nu = mass[i] * mass[j] / (p.M * p.M);
            p.nu = nu;
            p.f25 = 8.0/5.0 * nu;
            const double nu2 = nu * nu, nu3 = nu2 * nu;
            p.P3A[0] = (-35.0/16.0 * nu + 175.0/16.0 * nu2 - 175.0/16.0 * nu3);
            p.P3A[1] = (15.0/2.0 * nu - 135.0/4.0 * nu2 + 255.0/8.0 * nu3);
            p.P3A[2] = (-15.0/2.0 * nu + 237.0/8.0 * nu2 - 45.0/2.0 * nu3);
            p.P3A[3] = (11.0/4.0 * nu - 49.0/4.0 * nu2 + 13.0 * nu3);
            p.P3A[4] = (79.0 * nu - 69.0/2.0 * nu2 - 30.0 * nu3);
            p.P3A[5] = (-121.0 * nu + 16.0 * nu2 + 20.0 * nu3);
            p.P3A[6] = (75.0/4.0 * nu + 8.0 * nu2 - 10.0 * nu3);
            p.P3A[7] = (1.0 + 22717.0/168.0 * nu + 11.0/8.0 * nu2 - 7.0 * nu3 + 615.0/64.0 * pi2 * nu);
            p.P3A[8] = (-20827.0/840.0 * nu + nu3 - 123.0/64.0 * pi2 * nu);
            p.P3A[9] = (-16.0 - 1399.0/12.0 * nu - 71.0/2.0 * nu2 + 41.0/16.0 * pi2 * nu);
            p.P3B[0] = (-45.0/8.0 * nu + 15.0 * nu2 + 15.0/4.0 * nu3);
            p.P3B[1] = (12.0 * nu - 111.0/4.0 * nu2 - 12.0 * nu3);
            p.P3B[2] = (-65.0/8.0 * nu + 19.0 * nu2 + 6.0 * nu3);
            p.P3B[3] = (329.0/6.0 * nu + 59.0/2.0 * nu2 + 18.0 * nu3);
            p.P3B[4] = (-15.0 * nu - 27.0 * nu2 - 10.0 * nu3);
            p.P3B[5] = (-4.0 - 5849.0/840.0 * nu + 25.0 * nu2 + 8.0 * nu3 - 123.0/32.0 * pi2 * nu);
            p.P35A[0] = (3956.0/35.0 + 184.0/5.0 * nu);
            p.P35A[1] = (692.0/35.0 - 724.0/15.0 * nu);
            p.P35A[2] = (366.0/35.0 + 12.0 * nu);
            p.P35A[3] = (294.0/5.0 + 376.0/5.0 * nu);
            p.P35A[4] = (114.0 + 12.0 * nu);
            p.P35B[0] = (-1060.0/21.0 - 104.0/5.0 * nu);
            p.P35B[1] = (164.0/21.0 + 148.0/5.0 * nu);
            p.P35B[2] = (-626.0/35.0 - 12.0/5.0 * nu);
            p.P35B[3] = (-82.0/3.0 - 848.0/15.0 * nu);
            p.P35B[4] = (678.0/5.0 + 12.0/5.0 * nu);
            for (int s = 0; s < 2; s++) {
                const int A = s ? j : i, B = s ? i : j;
                PnBodyGeom& b = p.body[s];
                b.Gm_A = G * mass[A];
                b.Gm_B = G * mass[B];
                b.fA = mass[B] / p.M;
            }
        }
}

void pn_geom_set_positions(const double* pos_in, PnGeom& geom) {
    const int nn = geom.n;
    double pos[3 * PN_GEOM_MAX_BODIES], Gm[PN_GEOM_MAX_BODIES];   // local copies: no aliasing with the stores to geom
    for (int i = 0; i < 3 * nn; i++) pos[i] = pos_in[i];
    for (int A = 0; A < nn; A++) Gm[A] = geom.Gm[A];
    if (geom.eih_nbody) {
        // Newtonian potentials phi_A and fields g_A, as eih1_nbody_acceleration. Its two loops compute every r
        // separately (r_AC, r_CA and the r of the second loop are mathematically the same number, but the compiler may
        // order the sums of squares differently and so round them differently): so do they here
        double phi[PN_GEOM_MAX_BODIES], g[PN_GEOM_MAX_BODIES][3];
        for (int A = 0; A < nn; A++) {
            phi[A] = 0.0; g[A][0] = g[A][1] = g[A][2] = 0.0;
            for (int C = 0; C < nn; C++) {
                if (C == A) continue;
                const double d[3] = {pos[3*C] - pos[3*A], pos[3*C+1] - pos[3*A+1], pos[3*C+2] - pos[3*A+2]};
                const double r = std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
                phi[A] += Gm[C] / r;
                for (int k = 0; k < 3; k++) g[A][k] += Gm[C] * d[k] / (r * r * r);
            }
        }
        for (int A = 0; A < nn; A++) {
            geom.phi[A] = phi[A];
            for (int k = 0; k < 3; k++) geom.g[A][k] = g[A][k];
        }
        for (int A = 0; A < nn; A++)
            for (int B = 0; B < nn; B++) {
                if (B == A) continue;
                const double dAB[3] = {pos[3*A] - pos[3*B], pos[3*A+1] - pos[3*B+1], pos[3*A+2] - pos[3*B+2]};
                const double r = std::sqrt(dAB[0]*dAB[0] + dAB[1]*dAB[1] + dAB[2]*dAB[2]);
                for (int k = 0; k < 3; k++) geom.nAB[A][B][k] = dAB[k] / r;
                geom.xg[A][B] = -(dAB[0]*g[B][0] + dAB[1]*g[B][1] + dAB[2]*g[B][2]);
                geom.Gm_r2[A][B] = Gm[B] / (r * r);
                geom.c35_r[A][B] = 3.5 * (Gm[B] / r);
            }
    }
    for (int q = 0; q < geom.n_pairs; q++) {
        PnPairGeom& p = geom.pair[q];
        const double* x_i = &pos[3*p.i];
        const double* x_j = &pos[3*p.j];
        // as eih_pn_acceleration_pair
        double r_AB[3] = { x_i[0]-x_j[0], x_i[1]-x_j[1], x_i[2]-x_j[2] };
        double r2 = r_AB[0]*r_AB[0] + r_AB[1]*r_AB[1] + r_AB[2]*r_AB[2];
        double r_mag = std::sqrt(r2);
        double inv_r = 1.0 / r_mag;
        for (int k = 0; k < 3; k++) p.rij[k] = r_AB[k];
        p.inv_r = inv_r;
        // as eih_body (A = i, B = j and A = j, B = i)
        double inv_r2 = inv_r * inv_r;
        for (int s = 0; s < 2; s++) {
            const double* x_A = s ? x_j : x_i;
            const double* x_B = s ? x_i : x_j;
            PnBodyGeom& b = p.body[s];
            double rA[3] = { x_A[0]-x_B[0], x_A[1]-x_B[1], x_A[2]-x_B[2] };
            for (int k = 0; k < 3; k++) b.n[k] = rA[k] * inv_r;
            b.Gm_A_r = b.Gm_A * inv_r;
            b.Gm_B_r = b.Gm_B * inv_r;
            b.pf1 = b.Gm_B * inv_r2 * geom.inv_c2;
            b.pf2 = b.Gm_B * inv_r2 * geom.inv_c4;
        }
        // as pn_rel_highorder
        const double u = p.Gm * inv_r;
        p.u = u;
        p.uu = u * u;
        p.uuu = u * u * u;
        p.c25 = geom.inv_c5 * (p.f25 * u);
        p.c35 = geom.inv_c7 * (p.nu * u);
        p.g = -p.Gm * inv_r * inv_r;
    }
}

void eih1_nbody_acceleration_geom(const PnGeom& geom, const double* vel, double* a) {
    const int nn = geom.n;
    for (int A = 0; A < nn; A++) {
        double acc[3] = {0.0, 0.0, 0.0};
        const double* vA = &vel[3*A];
        const double vA2 = vA[0]*vA[0] + vA[1]*vA[1] + vA[2]*vA[2];
        for (int B = 0; B < nn; B++) {
            if (B == A) continue;
            const double* vB = &vel[3*B];
            const double* nAB = geom.nAB[A][B];
            const double vB2 = vB[0]*vB[0] + vB[1]*vB[1] + vB[2]*vB[2];
            const double vAvB = vA[0]*vB[0] + vA[1]*vB[1] + vA[2]*vB[2];
            const double nvA = nAB[0]*vA[0] + nAB[1]*vA[1] + nAB[2]*vA[2];
            const double nvB = nAB[0]*vB[0] + nAB[1]*vB[1] + nAB[2]*vB[2];
            const double xg = geom.xg[A][B];
            const double bracket = vA2 + 2.0*vB2 - 4.0*vAvB - 1.5*nvB*nvB - 4.0*geom.phi[A] - geom.phi[B] + 0.5*xg;
            const double Gm_r2 = geom.Gm_r2[A][B];
            const double nv43 = 4.0*nvA - 3.0*nvB;
            const double c35 = geom.c35_r[A][B];
            for (int k = 0; k < 3; k++)
                acc[k] += -Gm_r2 * nAB[k] * bracket + Gm_r2 * nv43 * (vA[k] - vB[k])
                          + c35 * geom.g[B][k];
        }
        for (int k = 0; k < 3; k++) a[3*A+k] = geom.inv_c2 * acc[k];
    }
}

// pn_rel_highorder with the mass- and position-only factors of pair p
[[gnu::noinline]] static void pn_rel_highorder_geom(const PnGeom& geom, const PnPairGeom& p, double v2, double rd,
                                                    int pn_order, double& Cn, double& Cv) {
    const double u = p.u;
    double A = 0.0, B = 0.0;
    if (pn_order & 4) {
        A += p.c25 * rd * (-17.0/3.0 * u - 3.0 * v2);
        B += p.c25 * (3.0 * u + v2);
    }
    if (pn_order & 8) {
        const double* PA = p.P3A;
        const double* PB = p.P3B;
        const double rd2 = rd * rd, rd3 = rd2 * rd, rd4 = rd2 * rd2, rd5 = rd4 * rd, rd6 = rd4 * rd2;
        const double v4 = v2 * v2, v6 = v4 * v2;
        const double A3 =
              rd6 * PA[0]
            + rd4 * v2 * PA[1]
            + rd2 * v4 * PA[2]
            + v6 * PA[3]
            + u * (rd4 * PA[4]
                   + rd2 * v2 * PA[5]
                   + v4 * PA[6])
            + p.uu * (rd2 * PA[7]
                      + v2 * PA[8])
            + p.uuu * PA[9];
        const double B3 =
              rd5 * PB[0]
            + rd3 * v2 * PB[1]
            + rd * v4 * PB[2]
            + u * (rd3 * PB[3]
                   + rd * v2 * PB[4])
            + p.uu * rd * PB[5];
        A += geom.inv_c6 * A3;
        B += geom.inv_c6 * B3;
    }
    if (pn_order & 16) {
        const double* PA = p.P35A;
        const double* PB = p.P35B;
        const double rd2 = rd * rd, v4 = v2 * v2;
        A += p.c35 * rd * (p.uu * PA[0] + u * v2 * PA[1]
                           + v4 * PA[2] + u * rd2 * PA[3]
                           - v2 * rd2 * PA[4] + 112.0 * rd2 * rd2);
        B += p.c35 * (p.uu * PB[0] + u * v2 * PB[1]
                      + v4 * PB[2] + u * rd2 * PB[3]
                      + v2 * rd2 * PB[4] - 120.0 * rd2 * rd2);
    }
    Cn = p.g * A;
    Cv = p.g * B;
}

// eih_body with the mass- and position-only factors of body A from B
[[gnu::noinline]] static void eih_body_geom(const double v_A[3], const double v_B[3], const PnBodyGeom& b,
                                            double inv_r, const EihPairCoeffs& h, int pn_order, double a_pn_A[3]) {
    a_pn_A[0] = a_pn_A[1] = a_pn_A[2] = 0.0;

    const double* n = b.n;
    double dv[3] = { v_A[0]-v_B[0], v_A[1]-v_B[1], v_A[2]-v_B[2] };
    double vA2 = v_A[0]*v_A[0] + v_A[1]*v_A[1] + v_A[2]*v_A[2];
    double vB2 = v_B[0]*v_B[0] + v_B[1]*v_B[1] + v_B[2]*v_B[2];
    double vA_dot_vB = v_A[0]*v_B[0] + v_A[1]*v_B[1] + v_A[2]*v_B[2];
    double n_dot_vA = n[0]*v_A[0] + n[1]*v_A[1] + n[2]*v_A[2];
    double n_dot_vB = n[0]*v_B[0] + n[1]*v_B[1] + n[2]*v_B[2];
    double Gm_A = b.Gm_A;
    double Gm_B = b.Gm_B;

    if (pn_order & 1) {
        double coeff_n = -vA2 - 2.0*vB2 + 4.0*vA_dot_vB
                         + 1.5 * n_dot_vB * n_dot_vB
                         + 5.0 * Gm_A * inv_r + 4.0 * Gm_B * inv_r;
        double coeff_v = 4.0 * n_dot_vA - 3.0 * n_dot_vB;
        double prefactor = b.pf1;
        for (int k = 0; k < 3; k++) {
            a_pn_A[k] += prefactor * (coeff_n * n[k] + coeff_v * dv[k]);
        }
    }

    if (pn_order & 2) {
        double Gm_A_r = b.Gm_A_r;
        double Gm_B_r = b.Gm_B_r;

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
            + Gm_A_r * Gm_A_r * (-57.0/4.0)
            + Gm_B_r * Gm_B_r * (-9.0)
            + Gm_A_r * Gm_B_r * (-69.0/2.0);

        double coeff_v =
            vA2 * n_dot_vB + 4.0 * vB2 * n_dot_vA
            - 5.0 * vB2 * n_dot_vB - 4.0 * vA_dot_vB * n_dot_vA
            + 4.0 * vA_dot_vB * n_dot_vB
            - 6.0 * n_dot_vA * n_dot_vB * n_dot_vB
            + (9.0/2.0) * n_dot_vB * n_dot_vB * n_dot_vB
            + Gm_A_r * (-63.0/4.0 * n_dot_vA + 55.0/4.0 * n_dot_vB)
            + Gm_B_r * (-2.0 * n_dot_vA - 2.0 * n_dot_vB);

        double prefactor = b.pf2;
        for (int k = 0; k < 3; k++) {
            a_pn_A[k] += prefactor * (coeff_n * n[k] + coeff_v * dv[k]);
        }
    }

    if (pn_order & 28) {
        const double fA = b.fA;
        for (int k = 0; k < 3; k++) {
            a_pn_A[k] += fA * (h.Cn * n[k] + h.Cv * dv[k]);
        }
    }
}

void eih_pn_pairs_geom(const PnGeom& geom, const double* vel, int pn_order, double* a) {
    for (int q = 0; q < geom.n_pairs; q++) {
        // as eih_pn_acceleration_pair
        const PnPairGeom& p = geom.pair[q];
        const double* v_i = &vel[3*p.i];
        const double* v_j = &vel[3*p.j];
        const double* r_AB = p.rij;
        double dv[3] = { v_i[0]-v_j[0], v_i[1]-v_j[1], v_i[2]-v_j[2] };
        EihPairCoeffs h{};
        h.M = p.M;
        if (pn_order & 28) {
            double dv2 = dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2];
            double rd = (r_AB[0]*dv[0] + r_AB[1]*dv[1] + r_AB[2]*dv[2]) * p.inv_r;
            pn_rel_highorder_geom(geom, p, dv2, rd, pn_order, h.Cn, h.Cv);
        }
        eih_body_geom(v_i, v_j, p.body[0], p.inv_r, h, pn_order, &a[6*q]);       // A = i, B = j
        eih_body_geom(v_j, v_i, p.body[1], p.inv_r, h, pn_order, &a[6*q + 3]);   // A = j, B = i
    }
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

    // Genuine three-body terms of the EIH energy, +1/2 sum_A sum_{B != A} sum_{C != A, C != B} G^2 m_A m_B m_C /
    // (r_AB r_AC) (the C = B terms are the pair term above); none for n = 2 (bit-identical there)
    double E_three = 0.0;
    for (int A = 0; A < n && n > 2; A++)
        for (int B = 0; B < n; B++) {
            if (B == A) continue;
            const double rAB = std::sqrt((pos[A*3]-pos[B*3])*(pos[A*3]-pos[B*3]) + (pos[A*3+1]-pos[B*3+1])*(pos[A*3+1]-pos[B*3+1]) +
                                         (pos[A*3+2]-pos[B*3+2])*(pos[A*3+2]-pos[B*3+2]));
            for (int C = 0; C < n; C++) {
                if (C == A || C == B) continue;
                const double rAC = std::sqrt((pos[A*3]-pos[C*3])*(pos[A*3]-pos[C*3]) + (pos[A*3+1]-pos[C*3+1])*(pos[A*3+1]-pos[C*3+1]) +
                                             (pos[A*3+2]-pos[C*3+2])*(pos[A*3+2]-pos[C*3+2]));
                E_three += 0.5 * G * G * mass[A] * mass[B] * mass[C] / (rAB * rAC);
            }
        }
    return inv_c2 * (E_kin + E_pot_kin + E_pot_pot + E_three);
}

// ============================================================================
// Spin-dependent PN dynamics of a binary (Rantala et al. 2023, sec. 3.7; Thorne & Hartle 1985; explicit forms of
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
