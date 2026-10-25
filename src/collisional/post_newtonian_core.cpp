#include "post_newtonian_core.h"
#include <cmath>
#include <cstring>

// The pure post-Newtonian functions (no Kokkos; tests/enzyme differentiates them): EIH accelerations, PN1 energy,
// spin dynamics. global_pn1_kick (Particles) is in post_newtonian.cpp.

// ============================================================================
// PN2.5, PN3.0 and PN3.5 of body A from body B in a general frame: Blanchet, Living Rev. Rel. 17, 2 (2014, v3 2024),
// eq. (acc3PN), body 1 = A, n_12 = n_AB = (x_A - x_B) / r, v_12 = v_A - v_B. 3PN in modified harmonic coordinates:
// the ln(r/r'_0) terms dropped and the gauge vector of the footnote to eq. (contacttransf) applied, y_MH = y_SH -
// delta y_A, delta y_A = (22/3) G^3 m_A^2 m_B n ln(r/r'_0) / (c^6 r^2), which adds
//   (22/3) G^3 m_A^2 m_B [(8 r'^2 - v^2) / r^4 + G M / r^5] n - (44/3) G^3 m_A^2 m_B r' / r^4 v
// (its ln terms cancel those of eq. acc3PN; the relative acceleration is eq. ABcoeffMH). 3.5PN: the three 1/r_12^6
// of the printed G^4 terms are 1/r_12^5 (dimensions). The Newtonian centre of mass moves with these terms (Blanchet
// sec. 6.3); before 2026-10 the c.o.m. relative acceleration was split by m_B / M, which off the PN c.o.m. frame
// changed the inspiral (0.3 rad over 300 orbits against KETJU, q = 1/3, v/c = 0.085).
// ============================================================================
namespace {
struct InvC {   // 1/c^5, 1/c^6, 1/c^7 (as PnGeom: from 1/c^2 and c)
    double c5, c6, c7;
};
InvC inv_c_powers(double c_code) {
    const double inv_c2 = 1.0 / (c_code * c_code);
    return {inv_c2 * inv_c2 / c_code, inv_c2 * inv_c2 * inv_c2, inv_c2 * inv_c2 * inv_c2 / c_code};
}
}  // namespace

[[gnu::noinline]] static void pn_body_highorder(const double n[3], const double v_A[3], const double v_B[3],
                                                const double dv[3], double GmA, double GmB, double inv_r,
                                                const InvC& ic, int pn_order, double a[3]) {
    const double nvA = n[0]*v_A[0] + n[1]*v_A[1] + n[2]*v_A[2];
    const double nvB = n[0]*v_B[0] + n[1]*v_B[1] + n[2]*v_B[2];
    const double nv = n[0]*dv[0] + n[1]*dv[1] + n[2]*dv[2];
    const double vA2 = v_A[0]*v_A[0] + v_A[1]*v_A[1] + v_A[2]*v_A[2];
    const double vB2 = v_B[0]*v_B[0] + v_B[1]*v_B[1] + v_B[2]*v_B[2];
    const double vAvB = v_A[0]*v_B[0] + v_A[1]*v_B[1] + v_A[2]*v_B[2];
    const double v2 = dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2];
    const double r1 = inv_r, r2 = r1 * r1, r3 = r2 * r1, r4 = r2 * r2, r5 = r4 * r1;
    const double A1B1 = GmA * GmB, A2B1 = GmA * A1B1, A1B2 = A1B1 * GmB;
    double Cn = 0.0, Cv = 0.0;
    if (pn_order & 4) {
        Cn += ic.c5 * nv * (208.0/15.0 * A1B2 * r4 - 24.0/5.0 * A2B1 * r4 + 12.0/5.0 * A1B1 * r3 * v2);
        Cv += ic.c5 * (8.0/5.0 * A2B1 * r4 - 32.0/5.0 * A1B2 * r4 - 4.0/5.0 * A1B1 * r3 * v2);
    }
    if (pn_order & 8) {
        const double pi2 = M_PI * M_PI;
        const double nvA2 = nvA * nvA, nvB2 = nvB * nvB, nvA3 = nvA2 * nvA, nvB3 = nvB2 * nvB;
        const double nvA4 = nvA2 * nvA2, nvB4 = nvB2 * nvB2, nvB5 = nvB4 * nvB, nvB6 = nvB4 * nvB2;
        const double vB4 = vB2 * vB2, vB6 = vB4 * vB2, vA4 = vA2 * vA2, vAvB2 = vAvB * vAvB;
        const double B1 = GmB, B2 = GmB * GmB, B3 = B2 * GmB, B4 = B2 * B2;
        const double A2B2 = GmA * A1B2, A3B1 = GmA * A2B1, A1B3 = A1B2 * GmB;
        const double n3 =
              B1 * r2 * (35.0/16.0 * nvB6 - 15.0/8.0 * nvB4 * vA2 + 15.0/2.0 * nvB4 * vAvB + 3.0 * nvB2 * vAvB2
                         - 15.0/2.0 * nvB4 * vB2 + 3.0/2.0 * nvB2 * vA2 * vB2 - 12.0 * nvB2 * vAvB * vB2
                         - 2.0 * vAvB2 * vB2 + 15.0/2.0 * nvB2 * vB4 + 4.0 * vAvB * vB4 - 2.0 * vB6)
            + A1B1 * r3 * (-171.0/8.0 * nvA4 + 171.0/2.0 * nvA3 * nvB - 723.0/4.0 * nvA2 * nvB2
                           + 383.0/2.0 * nvA * nvB3 - 455.0/8.0 * nvB4 + 229.0/4.0 * nvA2 * vA2
                           - 205.0/2.0 * nvA * nvB * vA2 + 191.0/4.0 * nvB2 * vA2 - 91.0/8.0 * vA4
                           - 229.0/2.0 * nvA2 * vAvB + 244.0 * nvA * nvB * vAvB - 225.0/2.0 * nvB2 * vAvB
                           + 91.0/2.0 * vA2 * vAvB - 177.0/4.0 * vAvB2 + 229.0/4.0 * nvA2 * vB2
                           - 283.0/2.0 * nvA * nvB * vB2 + 259.0/4.0 * nvB2 * vB2 - 91.0/4.0 * vA2 * vB2
                           + 43.0 * vAvB * vB2 - 81.0/8.0 * vB4)
            + B2 * r3 * (-6.0 * nvA2 * nvB2 + 12.0 * nvA * nvB3 + 6.0 * nvB4 + 4.0 * nvA * nvB * vAvB
                         + 12.0 * nvB2 * vAvB + 4.0 * vAvB2 - 4.0 * nvA * nvB * vB2 - 12.0 * nvB2 * vB2
                         - 8.0 * vAvB * vB2 + 4.0 * vB4)
            + B3 * r4 * (-nvA2 + 2.0 * nvA * nvB + 43.0/2.0 * nvB2 + 18.0 * vAvB - 9.0 * vB2)
            + A1B2 * r4 * (415.0/8.0 * nvA2 - 375.0/4.0 * nvA * nvB + 1113.0/8.0 * nvB2 - 615.0/64.0 * nv * nv * pi2
                           + 18.0 * vA2 + 123.0/64.0 * pi2 * v2 + 33.0 * vAvB - 33.0/2.0 * vB2)
            + A2B1 * r4 * (-45887.0/168.0 * nvA2 + 24025.0/42.0 * nvA * nvB - 10469.0/42.0 * nvB2
                           + 48197.0/840.0 * vA2 - 36227.0/420.0 * vAvB + 36227.0/840.0 * vB2)
            + r5 * (16.0 * B4 + A2B2 * (175.0 - 41.0/16.0 * pi2) - 3187.0/1260.0 * A3B1
                    + A1B3 * (110741.0/630.0 - 41.0/16.0 * pi2))
            + 22.0/3.0 * A2B1 * ((8.0 * nv * nv - v2) * r4 + (GmA + GmB) * r5);   // modified harmonic
        const double v3 =
              B1 * r2 * (15.0/2.0 * nvA * nvB4 - 45.0/8.0 * nvB5 - 3.0/2.0 * nvB3 * vA2 + 6.0 * nvA * nvB2 * vAvB
                         - 6.0 * nvB3 * vAvB - 2.0 * nvB * vAvB2 - 12.0 * nvA * nvB2 * vB2 + 12.0 * nvB3 * vB2
                         + nvB * vA2 * vB2 - 4.0 * nvA * vAvB * vB2 + 8.0 * nvB * vAvB * vB2 + 4.0 * nvA * vB4
                         - 7.0 * nvB * vB4)
            + B2 * r3 * (-2.0 * nvA2 * nvB + 8.0 * nvA * nvB2 + 2.0 * nvB3 + 2.0 * nvA * vAvB + 4.0 * nvB * vAvB
                         - 2.0 * nvA * vB2 - 4.0 * nvB * vB2)
            + A1B1 * r3 * (-243.0/4.0 * nvA3 + 565.0/4.0 * nvA2 * nvB - 269.0/4.0 * nvA * nvB2 - 95.0/12.0 * nvB3
                           + 207.0/8.0 * nvA * vA2 - 137.0/8.0 * nvB * vA2 - 36.0 * nvA * vAvB + 27.0/4.0 * nvB * vAvB
                           + 81.0/8.0 * nvA * vB2 + 83.0/8.0 * nvB * vB2)
            + B3 * r4 * (4.0 * nvA + 5.0 * nvB)
            + A1B2 * r4 * (-307.0/8.0 * nvA + 479.0/8.0 * nvB + 123.0/32.0 * nv * pi2)
            + A2B1 * r4 * (31397.0/420.0 * nvA - 36227.0/420.0 * nvB)
            - 44.0/3.0 * A2B1 * nv * r4;   // modified harmonic
        Cn += ic.c6 * n3;
        Cv += ic.c6 * v3;
    }
    if (pn_order & 16) {
        const double nvA2 = nvA * nvA, nvB2 = nvB * nvB, nvA3 = nvA2 * nvA, nvB3 = nvB2 * nvB;
        const double nv2 = nv * nv, nv4 = nv2 * nv2, nv5 = nv4 * nv;
        const double vA4 = vA2 * vA2, vB4 = vB2 * vB2, vAvB2 = vAvB * vAvB;
        const double A2B2 = GmA * A1B2, A3B1 = GmA * A2B1, A1B3 = A1B2 * GmB;
        const double n35 =
              A3B1 * r5 * (3992.0/105.0 * nvA - 4328.0/105.0 * nvB)
            + A2B2 * r5 * (-13576.0/105.0 * nvA + 2872.0/21.0 * nvB)
            - 3172.0/21.0 * A1B3 * r5 * nv
            + A2B1 * r4 * (48.0 * nvA3 - 696.0/5.0 * nvA2 * nvB + 744.0/5.0 * nvA * nvB2 - 288.0/5.0 * nvB3
                           - 4888.0/105.0 * nvA * vA2 + 5056.0/105.0 * nvB * vA2 + 2056.0/21.0 * nvA * vAvB
                           - 2224.0/21.0 * nvB * vAvB - 1028.0/21.0 * nvA * vB2 + 5812.0/105.0 * nvB * vB2)
            + A1B2 * r4 * (-582.0/5.0 * nvA3 + 1746.0/5.0 * nvA2 * nvB - 1954.0/5.0 * nvA * nvB2 + 158.0 * nvB3
                           + 3568.0/105.0 * nv * vA2 - 2864.0/35.0 * nvA * vAvB + 10048.0/105.0 * nvB * vAvB
                           + 1432.0/35.0 * nvA * vB2 - 5752.0/105.0 * nvB * vB2)
            + A1B1 * r3 * (-56.0 * nv5 + 60.0 * nvA3 * v2 - 180.0 * nvA2 * nvB * v2 + 174.0 * nvA * nvB2 * v2
                           - 54.0 * nvB3 * v2 - 246.0/35.0 * nv * vA4 + 1068.0/35.0 * nvA * vA2 * vAvB
                           - 984.0/35.0 * nvB * vA2 * vAvB - 1068.0/35.0 * nvA * vAvB2 + 180.0/7.0 * nvB * vAvB2
                           - 534.0/35.0 * nvA * vA2 * vB2 + 90.0/7.0 * nvB * vA2 * vB2 + 984.0/35.0 * nvA * vAvB * vB2
                           - 732.0/35.0 * nvB * vAvB * vB2 - 204.0/35.0 * nvA * vB4 + 24.0/7.0 * nvB * vB4);
        const double v35 =
              r5 * (-184.0/21.0 * A3B1 + 6224.0/105.0 * A2B2 + 6388.0/105.0 * A1B3)
            + A2B1 * r4 * (52.0/15.0 * nvA2 - 56.0/15.0 * nvA * nvB - 44.0/15.0 * nvB2 - 132.0/35.0 * vA2
                           + 152.0/35.0 * vAvB - 48.0/35.0 * vB2)
            + A1B2 * r4 * (454.0/15.0 * nvA2 - 372.0/5.0 * nvA * nvB + 854.0/15.0 * nvB2 - 152.0/21.0 * vA2
                           + 2864.0/105.0 * vAvB - 1768.0/105.0 * vB2)
            + A1B1 * r3 * (60.0 * nv4 - 348.0/5.0 * nvA2 * v2 + 684.0/5.0 * nvA * nvB * v2 - 66.0 * nvB2 * v2
                           + 334.0/35.0 * vA4 - 1336.0/35.0 * vA2 * vAvB + 1308.0/35.0 * vAvB2
                           + 654.0/35.0 * vA2 * vB2 - 1252.0/35.0 * vAvB * vB2 + 292.0/35.0 * vB4);
        Cn += ic.c7 * n35;
        Cv += ic.c7 * v35;
    }
    for (int k = 0; k < 3; k++) a[k] += Cn * n[k] + Cv * dv[k];
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

    // PN2.5, PN3.0, PN3.5 (general frame)
    if (pn_order & 28) pn_body_highorder(n, v_A, v_B, dv, Gm_A, Gm_B, inv_r, inv_c_powers(c_code), pn_order, a_pn_A);
}

// ============================================================================
// eih_pn_acceleration of both bodies of a pair at once, bit-identical to the two calls (A = i, B = j) and
// (A = j, B = i), with what both share computed once: 1/r (|x_j - x_i| = |x_i - x_j| exactly: IEEE subtraction
// is sign-symmetric) and 1/c^5..7. The per-body part (eih_body) repeats the operations of
// eih_pn_acceleration in its own, not inlined function, so that no expression is shared between the two bodies
// and every floating-point contraction (FMA, -march=native) comes out as there. Checked bitwise on 5e6 random
// pairs over all 31 orders, host and nvcc_wrapper builds; NBODY_CHECK_EIH=1 checks every pair of a run.
// ============================================================================


// PN terms of body A from B (eih_pn_acceleration) with 1/r and the pair coefficients h given
[[gnu::noinline]] static void eih_body(const double x_A[3], const double v_A[3], double m_A,
                                       const double x_B[3], const double v_B[3], double m_B,
                                       double G, double inv_c2, double inv_r, const InvC& ic,
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

    if (pn_order & 28) pn_body_highorder(n, v_A, v_B, dv, Gm_A, Gm_B, inv_r, ic, pn_order, a_pn_A);
}

void eih_pn_acceleration_pair(const double x_i[3], const double v_i[3], double m_i,
                              const double x_j[3], const double v_j[3], double m_j,
                              double G, double c_code, double a_i[3], double a_j[3], int pn_order) {
    // Geometry seen from i (A = i, B = j), as in eih_pn_acceleration
    double r_AB[3] = { x_i[0]-x_j[0], x_i[1]-x_j[1], x_i[2]-x_j[2] };
    double r2 = r_AB[0]*r_AB[0] + r_AB[1]*r_AB[1] + r_AB[2]*r_AB[2];
    double r_mag = std::sqrt(r2);
    double inv_r = 1.0 / r_mag;
    double c2 = c_code * c_code;
    double inv_c2 = 1.0 / c2;

    const InvC ic = inv_c_powers(c_code);
    eih_body(x_i, v_i, m_i, x_j, v_j, m_j, G, inv_c2, inv_r, ic, pn_order, a_i);   // A = i, B = j
    eih_body(x_j, v_j, m_j, x_i, v_i, m_i, G, inv_c2, inv_r, ic, pn_order, a_j);   // A = j, B = i
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
    for (int i = 0; i < geom.n; i++)
        for (int j = i + 1; j < geom.n; j++) {
            PnPairGeom& p = geom.pair[geom.n_pairs++];
            p.i = i;
            p.j = j;
            for (int s = 0; s < 2; s++) {
                const int A = s ? j : i, B = s ? i : j;
                PnBodyGeom& b = p.body[s];
                b.Gm_A = G * mass[A];
                b.Gm_B = G * mass[B];
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

// eih_body with the mass- and position-only factors of body A from B
[[gnu::noinline]] static void eih_body_geom(const double v_A[3], const double v_B[3], const PnBodyGeom& b,
                                            double inv_r, const InvC& ic, int pn_order, double a_pn_A[3]) {
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

    if (pn_order & 28) pn_body_highorder(n, v_A, v_B, dv, b.Gm_A, b.Gm_B, inv_r, ic, pn_order, a_pn_A);
}

void eih_pn_pairs_geom(const PnGeom& geom, const double* vel, int pn_order, double* a) {
    for (int q = 0; q < geom.n_pairs; q++) {
        // as eih_pn_acceleration_pair
        const PnPairGeom& p = geom.pair[q];
        const double* v_i = &vel[3*p.i];
        const double* v_j = &vel[3*p.j];
        const InvC ic = {geom.inv_c5, geom.inv_c6, geom.inv_c7};
        eih_body_geom(v_i, v_j, p.body[0], p.inv_r, ic, pn_order, &a[6*q]);       // A = i, B = j
        eih_body_geom(v_j, v_i, p.body[1], p.inv_r, ic, pn_order, &a[6*q + 3]);   // A = j, B = i
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

// Spin terms of body A from body B in a general frame (Thorne & Hartle 1985, PRD 31, 1815, eqs. 4.11b-d and 4.17,
// covariant spin condition, harmonic gauge; KETJU uses the same): n = (x_A - x_B) / r, v = v_A - v_B.
//   a_SO = G / (c^2 m_A r^3) [6 n ((m_A S_B + m_B S_A).(n x v)) + 4 m_A S_B x v + 3 m_B S_A x v
//                             + 3 (n.v) n x (2 m_A S_B + m_B S_A)]                                  (4.11c)
//   a_SS = G / (c^2 m_A r^4) [(15 (n.S_A)(n.S_B) - 3 S_A.S_B) n - 3 (n.S_B) S_A - 3 (n.S_A) S_B]      (4.11d)
//   a_Q  = G / (c^2 m_A r^4) sum over K = A (L = B), B (L = A) of (m_L / m_K) [(15/2 (n.S_K)^2 - 3/2 S_K^2) n
//          - 3 (n.S_K) S_K]                                     (4.11b, Kerr quadrupole Q = -S^2 / m, their 1.7)
//   Omega_A = G / (c^2 r^3) [3 (n.S_B) n - S_B] + G m_B / (c^2 r^2) (2 v_B - 3/2 v_A) x n
//             + 3 G m_B / (c^2 m_A r^3) (n.S_A) n                                                    (4.17a-c)
// In the c.o.m. frame a_A - a_B and Omega are Kidder's (spin_pn_acceleration, spin_precession); the geodetic term
// depends on the frame, the a_A are not split by mass (the Newtonian c.o.m. moves).
static void spin_body(const double x[3], const double vA[3], const double vB[3], double mA, double mB,
                      const double SA[3], const double SB[3], double G, double c_code, double a[3], double Om[3]) {
    const double r = std::sqrt(dot3(x, x));
    const double n[3] = {x[0] / r, x[1] / r, x[2] / r};
    const double v[3] = {vA[0] - vB[0], vA[1] - vB[1], vA[2] - vB[2]};
    const double c2 = c_code * c_code, nv = dot3(n, v);
    double nxv[3], SBxv[3], SAxv[3], nxSB[3], nxSA[3];
    cross3(n, v, nxv);
    cross3(SB, v, SBxv);
    cross3(SA, v, SAxv);
    cross3(n, SB, nxSB);
    cross3(n, SA, nxSA);
    const double nSA = dot3(n, SA), nSB = dot3(n, SB);
    const double so = G / (c2 * mA * r * r * r);
    const double proj = mA * dot3(SB, nxv) + mB * dot3(SA, nxv);
    const double ss = G / (c2 * mA * r * r * r * r);
    const double fQA = mB / mA, fQB = mA / mB;
    for (int k = 0; k < 3; k++) {
        a[k] = so * (6.0 * n[k] * proj + 4.0 * mA * SBxv[k] + 3.0 * mB * SAxv[k]
                     + 3.0 * nv * (2.0 * mA * nxSB[k] + mB * nxSA[k]));
        a[k] += ss * ((15.0 * nSA * nSB - 3.0 * dot3(SA, SB)) * n[k] - 3.0 * nSB * SA[k] - 3.0 * nSA * SB[k]);
        a[k] += ss * (fQB * ((7.5 * nSB * nSB - 1.5 * dot3(SB, SB)) * n[k] - 3.0 * nSB * SB[k])
                      + fQA * ((7.5 * nSA * nSA - 1.5 * dot3(SA, SA)) * n[k] - 3.0 * nSA * SA[k]));
    }
    double w[3], wxn[3];
    for (int k = 0; k < 3; k++) w[k] = 2.0 * vB[k] - 1.5 * vA[k];
    cross3(w, n, wxn);
    const double pss = G / (c2 * r * r * r), pg = G * mB / (c2 * r * r), pq = 3.0 * G * mB / (c2 * mA * r * r * r) * nSA;
    for (int k = 0; k < 3; k++) Om[k] = pss * (3.0 * nSB * n[k] - SB[k]) + pg * wxn[k] + pq * n[k];
}

void spin_pn_bodies(const double x[3], const double v1[3], const double v2[3], double m1, double m2,
                    const double S1[3], const double S2[3], double G, double c_code, double a1[3], double a2[3],
                    double Omega1[3], double Omega2[3]) {
    const double xm[3] = {-x[0], -x[1], -x[2]};
    spin_body(x, v1, v2, m1, m2, S1, S2, G, c_code, a1, Omega1);
    spin_body(xm, v2, v1, m2, m1, S2, S1, G, c_code, a2, Omega2);
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
