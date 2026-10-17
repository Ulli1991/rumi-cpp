// Spin PN terms of post_newtonian_core (spin_pn_acceleration, spin_precession, binary_total_angular_momentum) against
// the conserved quantities of Kidder 1995 (PRD 52, 821, covariant SSC, harmonic coordinates; eqs. 2.2c-d, 2.4, 2.7,
// 2.9) plus the Kerr spin-induced quadrupole E_Q = G/(2 c^2 r^3) sum_i (m_j/m_i) [3 (n.S_i)^2 - S_i^2]:
//  (1) dE/dt = 0: mu v.a_spin + d/dt (E_SO + E_SS + E_Q) along the code's flow (v, a, Omega_i x S_i), the second
//      term a 4th-order central difference along the flow. Split by powers of the spin scale lambda: order lambda
//      (spin-orbit) vanishes identically, order lambda^2 (spin-spin + quadrupole) up to spin-orbit squared terms,
//      O(v^2/c^2) of the individual terms.
//  (2) dJ/dt = 0 with J = mu x × v + [L_SO + S1 + S2] (binary_total_angular_momentum without PN1): the orders
//      lambda and lambda^2 cancel at c^-3 and c^-4; left are dL_SO/dS.(Omega_SO × S) ~ c^-5 and spin-orbit squared
//      terms ~ c^-6 (beyond the leading order), checked by doubling c.
//  (3) Omega_i × S_i = dH/dS_i × S_i (Poisson bracket) for the canonical leading-order spin Hamiltonian
//      H = 2G/(c^2 r^3) sum_i (1 + 3 m_j/(4 m_i)) L.S_i + E_SS + E_Q, L = mu x × v.
// Masses q = 1, 0.1, 1e-3, random eccentric/unbound states, |chi| <= 1, c = 1000 sqrt(GM/r) for (1), (3) and c = 10,
// 20 for (2) (J - mu x × v loses digits to the orbital part; the residual must stay above that).
// The Enzyme version with exact gradients and c scans: tests/enzyme/spin_consistency.cpp.
#include "post_newtonian_core.h"
#include <cmath>
#include <cstdio>
#include <random>

static const double G = 1.0;
static int fails = 0;
static inline double d3(const double* a, const double* b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
static inline void x3(const double* a, const double* b, double* o) {
    o[0] = a[1]*b[2] - a[2]*b[1]; o[1] = a[2]*b[0] - a[0]*b[2]; o[2] = a[0]*b[1] - a[1]*b[0];
}
struct Par { double m1, m2, c; };

// y = x[3], v[3], S1[3], S2[3]
static double e_spin(const double* y, const Par& p) {
    const double *x = y, *v = y + 3, *S1 = y + 6, *S2 = y + 9;
    const double m1 = p.m1, m2 = p.m2, mu = m1 * m2 / (m1 + m2), r = std::sqrt(d3(x, x));
    double LN[3], SE[3];
    x3(x, v, LN);
    for (int k = 0; k < 3; k++) SE[k] = (m2 / m1) * S1[k] + (m1 / m2) * S2[k];
    const double pre = G / (p.c * p.c * r * r * r), n1 = d3(x, S1) / r, n2 = d3(x, S2) / r;
    return pre * (mu * d3(LN, SE) + 3 * n1 * n2 - d3(S1, S2) +
                  0.5 * ((m2 / m1) * (3 * n1 * n1 - d3(S1, S1)) + (m1 / m2) * (3 * n2 * n2 - d3(S2, S2))));
}
static double h_spin(const double* y, const Par& p) {
    const double *x = y, *v = y + 3, *S1 = y + 6, *S2 = y + 9;
    const double m1 = p.m1, m2 = p.m2, mu = m1 * m2 / (m1 + m2), r = std::sqrt(d3(x, x));
    double L[3];
    x3(x, v, L);
    const double pre = G / (p.c * p.c * r * r * r), n1 = d3(x, S1) / r, n2 = d3(x, S2) / r;
    return pre * (2 * mu * ((1 + 0.75 * m2 / m1) * d3(L, S1) + (1 + 0.75 * m1 / m2) * d3(L, S2)) + 3 * n1 * n2 -
                  d3(S1, S2) + 0.5 * ((m2 / m1) * (3 * n1 * n1 - d3(S1, S1)) + (m1 / m2) * (3 * n2 * n2 - d3(S2, S2))));
}
// L_SO + S1 + S2 component k
static int g_k = 0;
static double j_spin(const double* y, const Par& p) {
    const double mu = p.m1 * p.m2 / (p.m1 + p.m2);
    double J[3], LN[3];
    binary_total_angular_momentum(y, y + 3, p.m1, p.m2, y + 6, y + 9, G, p.c, false, J);
    x3(y, y + 3, LN);
    return J[g_k] - mu * LN[g_k];
}
// flow of the code: (v, a_N + a_spin, Omega1 × S1, Omega2 × S2); returns a_spin
static void flow(const double* y, const Par& p, double* f, double* as) {
    const double M = p.m1 + p.m2, r = std::sqrt(d3(y, y));
    double O1[3], O2[3];
    spin_pn_acceleration(y, y + 3, p.m1, p.m2, y + 6, y + 9, G, p.c, as);
    spin_precession(y, y + 3, p.m1, p.m2, y + 6, y + 9, G, p.c, O1, O2);
    for (int k = 0; k < 3; k++) { f[k] = y[3 + k]; f[3 + k] = -G * M * y[k] / (r * r * r) + as[k]; }
    x3(O1, y + 6, f + 6); x3(O2, y + 9, f + 9);
}
// d/dt F along f, 4th-order central difference
static double ddt(double (*F)(const double*, const Par&), const double* y, const double* f, const Par& p) {
    const double h = 1e-3;
    double w[12], s = 0;
    const double cf[4] = {1.0 / 12, -8.0 / 12, 8.0 / 12, -1.0 / 12}, st[4] = {-2, -1, 1, 2};
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 12; i++) w[i] = y[i] + st[j] * h * f[i];
        s += cf[j] * F(w, p);
    }
    return s / h;
}
// coefficient of lambda^k of g(lambda) sampled at lambda = -2..2 (degree <= 4)
static double lam_coef(const double g[5], int k) {
    static const double C[5][5] = {   // rows: k; columns: lambda = -2, -1, 0, 1, 2
        {0, 0, 1, 0, 0}, {1.0 / 12, -8.0 / 12, 0, 8.0 / 12, -1.0 / 12}, {-1.0 / 24, 16.0 / 24, -30.0 / 24, 16.0 / 24, -1.0 / 24},
        {-1.0 / 12, 2.0 / 12, 0, -2.0 / 12, 1.0 / 12}, {1.0 / 24, -4.0 / 24, 6.0 / 24, -4.0 / 24, 1.0 / 24}};
    double s = 0;
    for (int i = 0; i < 5; i++) s += C[k][i] * g[i];
    return s;
}

int main() {
    std::mt19937_64 rng(17);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (double q : {1.0, 0.1, 1e-3}) {
        const Par pE{1.0 / (1 + q), q / (1 + q), 1000.0}, pJ{pE.m1, pE.m2, 10.0}, pJ2{pE.m1, pE.m2, 20.0};
        const double mu = pE.m1 * pE.m2 / (pE.m1 + pE.m2);
        double we1 = 0, we2 = 0, rj1 = 1e30, rj2 = 1e30, wpb = 0;
        for (int t = 0; t < 20; t++) {
            double y0[12], chi[6], nn = 0;
            for (int k = 0; k < 3; k++) { y0[k] = U(rng); nn += y0[k] * y0[k]; }
            for (int k = 0; k < 3; k++) { y0[k] /= std::sqrt(nn); y0[3 + k] = 0.75 * U(rng); }
            for (int i = 0; i < 6; i++) chi[i] = 0.57 * U(rng);
            double gE[5], sE[5], gJ[2][3][5];
            for (int il = 0; il < 5; il++) {
                const double lam = il - 2.0;
                double y[12], f[12], as[3];
                for (const Par* pp : {&pJ, &pJ2, &pE}) {
                    const Par& p = *pp;
                    for (int k = 0; k < 6; k++) y[k] = y0[k];
                    for (int i = 0; i < 6; i++) y[6 + i] = lam * chi[i] * G * (i < 3 ? p.m1 * p.m1 : p.m2 * p.m2) / p.c;
                    flow(y, p, f, as);
                    if (pp != &pE) {
                        const int ic = pp == &pJ ? 0 : 1;
                        double tq[3];
                        x3(y, as, tq);
                        for (g_k = 0; g_k < 3; g_k++) {
                            const double dJs = ddt(j_spin, y, f, p);
                            gJ[ic][g_k][il] = mu * tq[g_k] + dJs;
                        }
                    } else {
                        const double dEs = ddt(e_spin, y, f, p);
                        gE[il] = mu * d3(y + 3, as) + dEs;
                        sE[il] = std::fabs(mu * d3(y + 3, as)) + std::fabs(dEs);
                    }
                }
                const Par& p = pE;
                if (il == 4) {   // (3) at lambda = 2
                    double gS[6], w[12], o1[3], o2[3], h1[3], h2[3], num = 0, den = 0;
                    for (int i = 0; i < 6; i++) {
                        const double hs = 1e-3 * std::fabs(y[6 + i]) + 1e-30;
                        for (int k = 0; k < 12; k++) w[k] = y[k];
                        w[6 + i] = y[6 + i] + hs; const double hp = h_spin(w, p);
                        w[6 + i] = y[6 + i] - hs; const double hm = h_spin(w, p);
                        gS[i] = (hp - hm) / (2 * hs);   // H is quadratic in S: exact up to round-off
                    }
                    x3(gS, y + 6, h1); x3(gS + 3, y + 9, h2);
                    for (int k = 0; k < 3; k++) { o1[k] = f[6 + k]; o2[k] = f[9 + k]; }
                    for (int k = 0; k < 3; k++) {
                        num += (o1[k] - h1[k]) * (o1[k] - h1[k]) + (o2[k] - h2[k]) * (o2[k] - h2[k]);
                        den += h1[k] * h1[k] + h2[k] * h2[k];
                    }
                    wpb = std::max(wpb, std::sqrt(num / den));
                }
            }
            // scale of order lambda^k: the order-lambda^k size of the individual terms (|.| sampled at lambda = 1, 2)
            const double s1 = sE[3], s2 = sE[4] / 4;
            we1 = std::max(we1, std::fabs(lam_coef(gE, 1)) / s1);
            we2 = std::max(we2, std::fabs(lam_coef(gE, 2)) / s2);
            // J: the leading orders (c^-3 for lambda, c^-4 for lambda^2) cancel; the residual is the next order (c^-5,
            // c^-6): ratio c=10 / c=20 of 32 and 64 (8 and 16 for a leading-order error)
            for (int o = 1; o <= 2; o++) {
                double n0 = 0, n1 = 0;
                for (int k = 0; k < 3; k++) {
                    n0 += lam_coef(gJ[0][k], o) * lam_coef(gJ[0][k], o);
                    n1 += lam_coef(gJ[1][k], o) * lam_coef(gJ[1][k], o);
                }
                double& rr = o == 1 ? rj1 : rj2;
                rr = std::min(rr, std::sqrt(n0 / n1));
            }
        }
        std::printf("q=%g: dE/dt spin-orbit %.2e spin-spin %.2e | dJ/dt residual ratio c/2c spin-orbit %.1f spin-spin %.1f | "
                    "precession vs Poisson bracket %.2e\n", q, we1, we2, rj1, rj2, wpb);
        // measured: dE/dt <= 1.8e-11 (spin-orbit), 8.6e-8 (spin-spin, the c^-6 rest at c = 1000), bracket <= 5.1e-11;
        // a wrong coefficient gives O(0.01-1) (mutations of each coefficient fail)
        if (we1 > 1e-9) { std::printf("FAIL q=%g: spin-orbit dE/dt\n", q); fails++; }
        if (we2 > 1.2e-7) { std::printf("FAIL q=%g: spin-spin dE/dt\n", q); fails++; }
        // q = 1e-3: J is dominated by S1 and the c^-5 residual sinks below its round-off (Enzyme harness: 32, 64)
        if (q >= 0.1 && rj1 < 25) { std::printf("FAIL q=%g: spin-orbit dJ/dt\n", q); fails++; }
        if (q >= 0.1 && rj2 < 50) { std::printf("FAIL q=%g: spin-spin dJ/dt\n", q); fails++; }
        if (wpb > 1e-10) { std::printf("FAIL q=%g: precession vs Poisson bracket\n", q); fails++; }
    }
    std::printf(fails ? "FAILED %d\n" : "PASS\n", fails);
    return fails != 0;
}
