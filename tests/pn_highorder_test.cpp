// PN2.0-3.5 of eih_pn_acceleration_pair against Blanchet, Living Rev. Rel. 17, 2 (2014, v3 2024), circular orbits:
//  (1) radiation reaction: the work of PN2.5 + PN3.5 (plus the 1PN energy change under PN2.5) must equal minus the GW
//      flux F = 32/5 c^5/G nu^2 x^5 [1 + (-1247/336 - 35/12 nu) x], x = (G M Omega / c^3)^(2/3) (eq. Fluxx; conservative
//      orders do no work on a circular orbit: dE/dt is odd in the velocities, i.e. proportional to r').
//      PN2.5 alone: ratio 1 to round-off; the 1PN coefficient from two x (linear extrapolation).
//  (2) Kepler's third law to 3PN (eq. keplerlaw, modified harmonic coordinates: no ln(r/r'_0)): the radial relative
//      acceleration on a circular orbit at the formula's Omega must equal r Omega^2 up to the first missing order:
//      residual(gamma) / residual(gamma/2) = 8 with PN1 + PN2, 16 with PN3 added.
//  (3) PN2 alone with r' != 0 against the c.o.m. coefficients A_2PN, B_2PN (eqs. calAcoeff, calBcoeff), q = 1 and
//      q = 1e-12 (no c.o.m. correction at 1PN), relative difference < 1e-10.
// Masses q = 1 (positions and velocities exactly +-x/2, +-v/2) and q = 0.2 with the 1PN c.o.m. relation (eq. CMrel)
// for (1); q = 1 and q = 1e-6 for (2). Up to 2026-10 PN2.5 had the opposite sign (ratio -1) and PN3.5 gave +11 x the
// PN2.5 loss as a gain (docs/CASES.md C8).
#include "post_newtonian_core.h"
#include <cmath>
#include <cstdio>
#include <initializer_list>

static const double G = 4.302194e-3;
static int fails = 0;
static void check(bool ok, const char* what) { if (!ok) { std::printf("FAIL %s\n", what); fails++; } }

// two bodies on a circle of radius r (relative), relative velocity v along y, 1PN c.o.m. relation (P only; r' = 0)
static void setup(double m1, double m2, double r, double v, double c, double x1[3], double x2[3], double v1[3],
                  double v2[3]) {
    const double M = m1 + m2, X1 = m1 / M, X2 = m2 / M, nu = X1 * X2, D = X1 - X2;
    const double P = (0.5 * v * v - 0.5 * G * M / r) / (c * c);
    const double s1 = X2 + nu * D * P, s2 = -X1 + nu * D * P;
    x1[0] = s1 * r; x1[1] = 0; x1[2] = 0; x2[0] = s2 * r; x2[1] = 0; x2[2] = 0;
    v1[0] = 0; v1[1] = s1 * v; v1[2] = 0; v2[0] = 0; v2[1] = s2 * v; v2[2] = 0;
}

static double omega2_formula(double M, double nu, double r, double c, int order) {   // eq. keplerlaw (MH), to order 1-3
    const double g = G * M / (r * c * c);
    double s = 1.0 + (-3.0 + nu) * g;
    if (order >= 2) s += (6.0 + 41.0/4.0 * nu + nu * nu) * g * g;
    if (order >= 3) s += (-10.0 + (-75707.0/840.0 + 41.0/64.0 * M_PI * M_PI) * nu + 19.0/2.0 * nu * nu + nu * nu * nu) * g * g * g;
    return G * M / (r * r * r) * s;
}

// radiated power: work of the dissipative orders on both bodies + d/ds E_pn1(v + s a_2.5) (1PN energy change)
static double power(double m1, double m2, double r, double c) {
    const double M = m1 + m2;
    const double v = r * std::sqrt(omega2_formula(M, m1 * m2 / (M * M), r, c, 3));
    double x1[3], x2[3], v1[3], v2[3], a1[3], a2[3], b1[3], b2[3];
    setup(m1, m2, r, v, c, x1, x2, v1, v2);
    eih_pn_acceleration_pair(x1, v1, m1, x2, v2, m2, G, c, a1, a2, 4 | 16);
    double W = 0.0;
    for (int k = 0; k < 3; k++) W += m1 * v1[k] * a1[k] + m2 * v2[k] * a2[k];
    eih_pn_acceleration_pair(x1, v1, m1, x2, v2, m2, G, c, b1, b2, 4);
    auto E1 = [&](double s) {
        double pos[6], vel[6], m[2] = {m1, m2};
        for (int k = 0; k < 3; k++) {
            pos[k] = x1[k]; pos[3+k] = x2[k]; vel[k] = v1[k] + s * b1[k]; vel[3+k] = v2[k] + s * b2[k];
        }
        return pn1_energy(pos, vel, m, 2, G, c);
    };
    const double h = 1e-3 * v / std::sqrt(b1[0]*b1[0] + b1[1]*b1[1] + b1[2]*b1[2]);
    W += (8.0 * (E1(h) - E1(-h)) - (E1(2 * h) - E1(-2 * h))) / (12.0 * h);
    return W;
}

int main() {
    const double c = 1.0;   // the velocity scale is set by r through gamma = G M / (r c^2)
    // (1) flux balance
    for (double q : {1.0, 0.2}) {
        const double m1 = 10.0, m2 = q * m1, M = m1 + m2, nu = m1 * m2 / (M * M);
        double est[2];
        const double xs[2] = {1e-3, 5e-4};
        for (int t = 0; t < 2; t++) {
            // r from x: x = (G M Omega / c^3)^(2/3); Omega(r) to 3PN, solved by fixed point
            double r = G * M / (xs[t] * c * c);
            for (int it = 0; it < 50; it++) {
                const double Om = std::sqrt(omega2_formula(M, nu, r, c, 3));
                const double x = std::pow(G * M * Om / (c * c * c), 2.0 / 3.0);
                r *= x / xs[t];
            }
            const double F0 = 32.0 / 5.0 * std::pow(c, 5) / G * nu * nu * std::pow(xs[t], 5);
            const double R = -power(m1, m2, r, c) / F0;
            est[t] = (R - 1.0) / xs[t];
            if (t == 0) {   // PN2.5 alone: Peters at the Newtonian circular orbit, exactly
                const double r0 = 1e4 * G * M;
                const double v0 = std::sqrt(G * M / r0);
                double x1[3], x2[3], v1[3], v2[3], a1[3], a2[3];
                setup(m1, m2, r0, v0, 1e30, x1, x2, v1, v2);
                eih_pn_acceleration_pair(x1, v1, m1, x2, v2, m2, G, c, a1, a2, 4);
                double W = 0.0;
                for (int k = 0; k < 3; k++) W += m1 * v1[k] * a1[k] + m2 * v2[k] * a2[k];
                const double peters = -32.0 / 5.0 * std::pow(G, 4) * m1 * m1 * m2 * m2 * M / (std::pow(c, 5) * std::pow(r0, 5));
                std::printf("q = %.1f: PN2.5 work / Peters = %.15f\n", q, W / peters);
                check(std::fabs(W / peters - 1.0) < 1e-12, "PN2.5 energy loss differs from Peters");
            }
        }
        const double c1 = 2.0 * est[1] - est[0];
        const double want = -1247.0 / 336.0 - 35.0 / 12.0 * nu;
        std::printf("q = %.1f: 1PN flux coefficient %.5f (Blanchet %.5f)\n", q, c1, want);
        check(std::fabs(c1 - want) < 1e-2, "PN3.5 does not give the 1PN flux correction");
    }
    // (2) Kepler's third law
    for (double q : {1.0, 1e-6}) {
        const double m1 = 10.0, m2 = q * m1, M = m1 + m2, nu = m1 * m2 / (M * M);
        for (int order : {2, 3}) {
            const int bits = order == 2 ? (1 | 2) : (1 | 2 | 8);
            double res[2];
            for (int t = 0; t < 2; t++) {
                const double gam = t == 0 ? 1e-2 : 5e-3;
                const double r = G * M / (gam * c * c);
                const double Om2 = omega2_formula(M, nu, r, c, order);
                double x1[3], x2[3], v1[3], v2[3], a1[3], a2[3];
                setup(m1, m2, r, r * std::sqrt(Om2), c, x1, x2, v1, v2);
                eih_pn_acceleration_pair(x1, v1, m1, x2, v2, m2, G, c, a1, a2, bits);
                const double arad = G * M / (r * r) - (a1[0] - a2[0]);   // inward
                res[t] = std::fabs(arad / (r * Om2) - 1.0);
            }
            const double ratio = res[0] / res[1];
            const double want = order == 2 ? 8.0 : 16.0;
            std::printf("q = %g, PN up to %d: residual %.2e, ratio %.2f (%g)\n", q, order == 2 ? 2 : 3, res[0], ratio, want);
            check(std::fabs(ratio / want - 1.0) < 0.1, "circular-orbit frequency differs from Blanchet's Kepler law");
        }
    }
    // (3) PN2 with radial velocity
    for (double q : {1e-12, 1.0}) {
        const double m1 = 10.0, m2 = q * m1, M = m1 + m2, X1 = m1 / M, X2 = m2 / M, nu = X1 * X2;
        const double r = 1e-5, u = G * M / r, vt = 0.9 * std::sqrt(u), rd = 0.3 * std::sqrt(u), v2 = vt * vt + rd * rd;
        const double c = 30.0 * std::sqrt(u);
        double x1[3] = {X2 * r, 0, 0}, x2[3] = {-X1 * r, 0, 0}, v1[3] = {X2 * rd, X2 * vt, 0}, v2v[3] = {-X1 * rd, -X1 * vt, 0};
        double a1[3], a2[3];
        eih_pn_acceleration_pair(x1, v1, m1, x2, v2v, m2, G, c, a1, a2, 2);
        const double s4 = std::pow(c, 4) * r * r / (G * M);
        const double rad = -(a1[0] - a2[0]) * s4, tang = -(a1[1] - a2[1]) * s4 / vt;
        const double A = 15.0/8.0 * std::pow(rd, 4) * nu - 45.0/8.0 * std::pow(rd, 4) * nu * nu - 4.5 * rd * rd * nu * v2
                       + 6.0 * rd * rd * nu * nu * v2 + 3.0 * nu * v2 * v2 - 4.0 * nu * nu * v2 * v2
                       + u * (-2.0 * rd * rd - 25.0 * rd * rd * nu - 2.0 * rd * rd * nu * nu - 6.5 * nu * v2 + 2.0 * nu * nu * v2)
                       + u * u * (9.0 + 87.0/4.0 * nu);
        const double B = 4.5 * std::pow(rd, 3) * nu + 3.0 * std::pow(rd, 3) * nu * nu - 7.5 * rd * nu * v2
                       - 2.0 * rd * nu * nu * v2 + u * (2.0 * rd + 20.5 * rd * nu + 4.0 * rd * nu * nu);
        const double e1 = std::fabs(rad / (A + B * rd) - 1.0), e2 = std::fabs(tang / B - 1.0);
        std::printf("q = %g, PN2 with r': radial %.3e, tangential %.3e relative to Blanchet\n", q, e1, e2);
        check(e1 < 1e-10 && e2 < 1e-10, "PN2 differs from Blanchet's A_2PN, B_2PN");
    }
    return fails;
}
