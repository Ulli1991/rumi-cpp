// Spin PN terms (spin_pn_acceleration, spin_precession, binary_total_angular_momentum) against the conserved energy
// and angular momentum of Kidder 1995 (PRD 52, 821; covariant SSC), eqs. 2.2c-d, 2.4, 2.7, 2.9, plus the Kerr
// spin-induced quadrupole (energy G/(2c^2 r^3) sum_i (m_j/m_i) [3 (n.S_i)^2 - S_i^2]; Poisson 1998, Racine 2008).
// Relative coordinates x = x1 - x2, v = v1 - v2 (two-body c.o.m. frame), physical spins S_i.
//  (a) dE/dt = dE/dx.v + dE/dv.a + sum_i dE/dS_i.(Omega_i x S_i), with the exact gradients (Enzyme) of
//      E = E_N [+ E_1PN] + E_SO + E_SS + E_Q and the code's a (Newton [+ EIH 1PN pair] + spin_pn_acceleration) and
//      Omega_i (spin_precession). Split by powers of the spin scale lambda (S -> lambda S): without 1PN the order
//      lambda (spin-orbit) is an identity at c^-3 (round-off, checked at c = 2); order lambda^2 (spin-spin +
//      quadrupole) cancels at c^-4 and leaves spin-orbit squared terms ~ c^-6 (ratio 64 per doubling of c; 16 would
//      be a c^-4 error). With 1PN the lambda residual is the missing next-to-leading spin-orbit order, c^-5 (32; 8
//      would be a leading-order error).
//  (b) dJ/dt with J from binary_total_angular_momentum (L_N [+ L_PN1] + L_SO + S1 + S2): lambda cancels at c^-3 and
//      leaves dL_SO/dS.(Omega_SO x S) ~ c^-5 (32), lambda^2 cancels at c^-4 (64).
//  (c) Omega_i x S_i against the Poisson bracket {S_i, H} = dH/dS_i x S_i of the canonical (Newton-Wigner SSC)
//      leading-order spin Hamiltonian H_SO = 2G/(c^2 r^3) sum_i (1 + 3 m_j / (4 m_i)) L.S_i (L = mu x x v),
//      H_SS = G/(c^2 r^3) [3 (n.S1)(n.S2) - S1.S2], H_Q as E_Q (spin precession is SSC independent at this order).
// Results 2026-10-17: (a) 1e-13 - 1.2e-10, ratios 64.0 and 32.0; (b) 32.0, 64.0; (c) <= 5.6e-16; all q.
// Masses q = m2/m1 = 1, 0.1, 1e-3; random eccentric states (r = 1, |v| up to 1.3 sqrt(GM/r)), |chi| <= 1.
//   /mpcdf/soft/SLE_15/packages/x86_64/clang/18.1.8/bin/clang++ -std=c++20 -O2 \
//     -fplugin=/u/uli/sw-build/enzyme/build/Enzyme/ClangEnzyme-18.so -Isrc/collisional tests/enzyme/spin_consistency.cpp
#include "../../src/collisional/post_newtonian_core.cpp"
#include <cstdio>
#include <random>

extern int enzyme_dup, enzyme_const;
template <typename... T> void __enzyme_autodiff(void*, T...);

static const double G = 1.0;
struct Par { double m1, m2, c; int pn1; };

static inline double d3(const double* a, const double* b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
static inline void x3(const double* a, const double* b, double* o) {
    o[0] = a[1]*b[2] - a[2]*b[1]; o[1] = a[2]*b[0] - a[0]*b[2]; o[2] = a[0]*b[1] - a[1]*b[0];
}

// Kidder eq. 2.7 (G, c restored) + Kerr quadrupole; S[0..2] = S1, S[3..5] = S2
static double energy(const double* x, const double* v, const double* S, const Par* p) {
    const double m1 = p->m1, m2 = p->m2, M = m1 + m2, mu = m1 * m2 / M, eta = mu / M, c2 = p->c * p->c;
    const double r = std::sqrt(d3(x, x)), v2 = d3(v, v), rd = d3(x, v) / r;
    double E = mu * (0.5 * v2 - G * M / r);
    if (p->pn1)
        E += mu / c2 * (0.375 * (1 - 3 * eta) * v2 * v2 + 0.5 * (3 + eta) * v2 * G * M / r +
                        0.5 * eta * G * M / r * rd * rd + 0.5 * (G * M / r) * (G * M / r));
    double LN[3]; x3(x, v, LN);
    const double* S1 = S; const double* S2 = S + 3;
    double SE[3];
    for (int k = 0; k < 3; k++) SE[k] = (m2 / m1) * S1[k] + (m1 / m2) * S2[k];   // S + (dm/M) Delta
    const double pre = G / (c2 * r * r * r);
    E += pre * mu * d3(LN, SE);                                                   // E_SO
    const double n1 = d3(x, S1) / r, n2 = d3(x, S2) / r;
    E += pre * (3 * n1 * n2 - d3(S1, S2));                                         // E_SS
    E += 0.5 * pre * ((m2 / m1) * (3 * n1 * n1 - d3(S1, S1)) + (m1 / m2) * (3 * n2 * n2 - d3(S2, S2)));   // E_Q
    return E;
}

// canonical leading-order spin Hamiltonian (Newton-Wigner SSC), p = mu v
static double hspin(const double* x, const double* v, const double* S, const Par* p) {
    const double m1 = p->m1, m2 = p->m2, M = m1 + m2, mu = m1 * m2 / M, c2 = p->c * p->c;
    const double r = std::sqrt(d3(x, x));
    double L[3]; x3(x, v, L);
    const double* S1 = S; const double* S2 = S + 3;
    const double pre = G / (c2 * r * r * r);
    double H = 2 * pre * mu * ((1 + 0.75 * m2 / m1) * d3(L, S1) + (1 + 0.75 * m1 / m2) * d3(L, S2));
    const double n1 = d3(x, S1) / r, n2 = d3(x, S2) / r;
    H += pre * (3 * n1 * n2 - d3(S1, S2));
    H += 0.5 * pre * ((m2 / m1) * (3 * n1 * n1 - d3(S1, S1)) + (m1 / m2) * (3 * n2 * n2 - d3(S2, S2)));
    return H;
}

static int g_jk = 0;
static double jcomp(const double* x, const double* v, const double* S, const Par* p) {
    double J[3];
    binary_total_angular_momentum(x, v, p->m1, p->m2, S, S + 3, G, p->c, p->pn1, J);
    return J[g_jk];
}

// the code's relative acceleration and precession vectors
static void code_rhs(const double* x, const double* v, const double* S, const Par* p, double* a, double* O1, double* O2) {
    const double m1 = p->m1, m2 = p->m2, M = m1 + m2, r = std::sqrt(d3(x, x));
    for (int k = 0; k < 3; k++) a[k] = -G * M * x[k] / (r * r * r);
    if (p->pn1) {
        double x1[3], x2[3], v1[3], v2[3], a1[3], a2[3];
        for (int k = 0; k < 3; k++) {
            x1[k] = m2 / M * x[k]; x2[k] = -m1 / M * x[k]; v1[k] = m2 / M * v[k]; v2[k] = -m1 / M * v[k];
        }
        eih_pn_acceleration_pair(x1, v1, m1, x2, v2, m2, G, p->c, a1, a2, 1);
        for (int k = 0; k < 3; k++) a[k] += a1[k] - a2[k];
    }
    double as[3];
    spin_pn_acceleration(x, v, m1, m2, S, S + 3, G, p->c, as);
    for (int k = 0; k < 3; k++) a[k] += as[k];
    spin_precession(x, v, m1, m2, S, S + 3, G, p->c, O1, O2);
}

// pieces of d(F)/dt along the code's equations: [x part, v part, S part]
template <double (*f)(const double*, const double*, const double*, const Par*)>
static void ddt( const double* x, const double* v, const double* S, const Par* p, double out[3]) {
    double gx[3] = {0, 0, 0}, gv[3] = {0, 0, 0}, gS[6] = {0, 0, 0, 0, 0, 0}, a[3], O1[3], O2[3], Sd[6];
    __enzyme_autodiff((void*)f, enzyme_dup, x, gx, enzyme_dup, v, gv, enzyme_dup, S, gS, enzyme_const, p);
    code_rhs(x, v, S, p, a, O1, O2);
    x3(O1, S, Sd); x3(O2, S + 3, Sd + 3);
    out[0] = d3(gx, v); out[1] = d3(gv, a); out[2] = d3(gS, Sd) + d3(gS + 3, Sd + 3);
}

// coefficients of lambda^k (k = 0..4) of g(lambda) from lambda = -2..2 (g is a cubic in lambda here)
static void lam_fit(const double g[5], double co[5]) {
    const double L[5] = {-2, -1, 0, 1, 2};
    double A[5][6];
    for (int i = 0; i < 5; i++) { double t = 1; for (int k = 0; k < 5; k++) { A[i][k] = t; t *= L[i]; } A[i][5] = g[i]; }
    for (int k = 0; k < 5; k++)
        for (int i = 0; i < 5; i++) if (i != k) {
            const double f = A[i][k] / A[k][k];
            for (int q = 0; q < 6; q++) A[i][q] -= f * A[k][q];
        }
    for (int k = 0; k < 5; k++) co[k] = A[k][5] / A[k][k];
}

struct State { double x[3], v[3], chi[6]; };

// residual and scale of order lambda^k of dF/dt; jk >= 0: J component
template <double (*f)(const double*, const double*, const double*, const Par*)>
static void order_resid(const State& s, const Par& p, int k, double& res, double& scale) {
    double g[4][5];
    for (int il = 0; il < 5; il++) {
        const double lam = il - 2.0;
        double S[6];
        for (int i = 0; i < 6; i++) S[i] = lam * s.chi[i] * G * (i < 3 ? p.m1 * p.m1 : p.m2 * p.m2) / p.c;
        double out[3];
        ddt<f>(s.x, s.v, S, &p, out);
        for (int q = 0; q < 3; q++) g[q][il] = out[q];
        g[3][il] = out[0] + out[1] + out[2];
    }
    double co[5];
    scale = 0;
    for (int q = 0; q < 3; q++) { lam_fit(g[q], co); scale += std::fabs(co[k]); }
    lam_fit(g[3], co);
    res = co[k];
}

int main() {
    std::mt19937_64 rng(2026);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    int bad = 0;
    for (double q : {1.0, 0.1, 1e-3}) {
        const double m1 = 1.0 / (1.0 + q), m2 = q / (1.0 + q);
        double wa1 = 0, wa2 = 0, wj1 = 0, wj2 = 0, wpb = 0, ra2 = 1e30, rj2 = 1e30, rj1 = 1e30, rmin1 = 1e30, rmax1 = 0, rjmin = 1e30, rjmax = 0;
        for (int t = 0; t < 30; t++) {
            State s;
            double nx[3] = {U(rng), U(rng), U(rng)}, nn = std::sqrt(d3(nx, nx));
            for (int k = 0; k < 3; k++) s.x[k] = nx[k] / nn;
            for (int k = 0; k < 3; k++) s.v[k] = 0.75 * U(rng);   // |v| up to 1.3 sqrt(GM/r): bound and unbound
            for (int i = 0; i < 6; i++) s.chi[i] = 0.57 * U(rng);
            Par p{m1, m2, 2.0, 0};   // c = 2 sqrt(GM/r): the leading-order identities hold for any c, less round-off
            double r, sc, r1, r2, sc2;
            // (a) without 1PN: lambda^1 is an identity at c^-3
            order_resid<energy>(s, p, 1, r, sc); wa1 = std::max(wa1, std::fabs(r) / sc);
            // (b) J, each component, without 1PN: lambda^1 cancels at c^-3; dL_SO/dS . (Omega_SO x S) leaves c^-5
            //     (the next-to-leading spin-orbit order): ratio 32 per doubling of c; lambda^2 cancels at c^-4 (no L_SS)
            for (g_jk = 0; g_jk < 3; g_jk++) {
                p.c = 10.0; order_resid<jcomp>(s, p, 1, r1, sc); p.c = 20.0; order_resid<jcomp>(s, p, 1, r2, sc2);
                rj1 = std::min(rj1, std::fabs(r1 / r2)); wj1 = std::max(wj1, std::fabs(r1) / sc);
            }
            p.c = 2.0;
            // lambda^2 (spin-spin + quadrupole): the c^-4 terms cancel exactly; what is left are spin-orbit squared
            // terms, c^-6 (ratio 64 per doubling of c) if right, c^-4 (16) if not
            p.c = 10.0;
            order_resid<energy>(s, p, 2, r1, sc); p.c = 20.0; order_resid<energy>(s, p, 2, r2, sc2);
            ra2 = std::min(ra2, std::fabs(r1 / r2)); wa2 = std::max(wa2, std::fabs(r1) / sc);
            for (g_jk = 0; g_jk < 3; g_jk++) {
                p.c = 10.0; order_resid<jcomp>(s, p, 2, r1, sc); p.c = 20.0; order_resid<jcomp>(s, p, 2, r2, sc2);
                rj2 = std::min(rj2, std::fabs(r1 / r2)); wj2 = std::max(wj2, std::fabs(r1) / sc);
            }
            // with 1PN: lambda^1 residual ~ c^-5 (ratio 32 per doubling of c) if leading order right, ~ c^-3 (8) if not
            p.pn1 = 1; p.c = 10.0;
            order_resid<energy>(s, p, 1, r1, sc); p.c = 20.0; order_resid<energy>(s, p, 1, r2, sc);
            rmin1 = std::min(rmin1, std::fabs(r1 / r2)); rmax1 = std::max(rmax1, std::fabs(r1 / r2));
            for (g_jk = 0; g_jk < 3; g_jk++) {
                p.c = 10.0; order_resid<jcomp>(s, p, 1, r1, sc); p.c = 20.0; order_resid<jcomp>(s, p, 1, r2, sc);
                rjmin = std::min(rjmin, std::fabs(r1 / r2)); rjmax = std::max(rjmax, std::fabs(r1 / r2));
            }
            p.pn1 = 0; p.c = 2.0;
            // (c) precession against the Poisson bracket of the canonical spin Hamiltonian
            double S[6];
            for (int i = 0; i < 6; i++) S[i] = s.chi[i] * G * (i < 3 ? m1 * m1 : m2 * m2) / p.c;
            double gx[3] = {0, 0, 0}, gv[3] = {0, 0, 0}, gS[6] = {0, 0, 0, 0, 0, 0}, a[3], O1[3], O2[3];
            __enzyme_autodiff((void*)hspin, enzyme_dup, s.x, gx, enzyme_dup, s.v, gv, enzyme_dup, S, gS, enzyme_const, &p);
            code_rhs(s.x, s.v, S, &p, a, O1, O2);
            double c1[3], c2[3], h1[3], h2[3];
            x3(O1, S, c1); x3(O2, S + 3, c2); x3(gS, S, h1); x3(gS + 3, S + 3, h2);
            double num = 0, den = 0;
            for (int k = 0; k < 3; k++) { num += (c1[k]-h1[k])*(c1[k]-h1[k]) + (c2[k]-h2[k])*(c2[k]-h2[k]); den += h1[k]*h1[k] + h2[k]*h2[k]; }
            wpb = std::max(wpb, std::sqrt(num / den));
        }
        std::printf("q=%g (a) dE/dt no 1PN: lambda^1 |resid| / terms %.2e (round-off if consistent); lambda^2 at c=10: "
                    "|resid| / terms %.2e, min ratio c/2c %.1f (64: c^-6 left, 16: c^-4 error)\n", q, wa1, wa2, ra2);
        std::printf("q=%g (b) dJ/dt no 1PN: lambda^1 at c=10 |resid| / terms %.2e, min ratio c/2c %.1f (32: c^-5 left, 8: c^-3 error); lambda^2 at c=10: |resid| / terms %.2e, "
                    "min ratio c/2c %.1f\n", q, wj1, rj1, wj2, rj2);
        std::printf("q=%g (a') with 1PN, lambda^1 of dE/dt, ratio c/2c: %.1f - %.1f (32: next order; 8: leading-order error)\n", q, rmin1, rmax1);
        std::printf("q=%g (b') with 1PN, lambda^1 of dJ/dt, ratio c/2c: %.1f - %.1f\n", q, rjmin, rjmax);
        std::printf("q=%g (c) |Omega_i x S_i - {S_i, H}| / |{S_i, H}| max %.2e\n", q, wpb);
        if (wa1 > 1e-9 || rj1 < 25 || ra2 < 50 || rj2 < 50 || wpb > 1e-12 || rmin1 < 25 || rjmin < 25) bad++;
    }
    std::printf(bad ? "FAIL\n" : "PASS\n");
    return bad != 0;
}
