#pragma once
// Prototype (not production): ADM-Hamiltonian post-Newtonian dynamics for Gauss-Legendre IRK (tests/proto_symrk).
//
// Conservative part, two-body c.o.m. reduced Hamiltonian H/mu (x = x_1 - x_2, phat = p_rel / mu, u = G M / r,
// pr = n.phat), Damour, Jaranowski & Schaefer 2000, PRD 62, 044024, Eqs. (3.3)-(3.6), with the regularization
// parameters fixed by dimensional regularization (DJS 2001, PLB 513, 147): omega_static = 0, omega_kinetic = 41/24.
// Same as Schaefer & Jaranowski 2018 LRR 21:7 Eq. (4.13)-(4.16) and Blanchet LRR Eq. (HADMcm) truncated at 3PN.
//   H_N   = p^2/2 - u
//   H_1PN = (3 nu - 1)/8 p^4 - [(3 + nu) p^2 + nu pr^2] u/2 + u^2/2
//   H_2PN, H_3PN: monomials c u^a p^(2b) pr^c below.
// N-body use (Schaefer 1985; LRR 2018 Eq. 4.4): Newtonian + N-body ADM 1PN (symrk::NBody) + the 2PN and 3PN two-body
// terms of each PN pair in the pair's relative variables x_ij, p_rel = (m_j p_i - m_i p_j) / M (exact in the pair's
// c.o.m. frame; the c.o.m.-motion corrections at 2PN are dropped, the same approximation as the production code,
// which applies the c.o.m. relative 2.5PN-3.5PN accelerations of Blanchet LRR in any frame).
//
// Radiation reaction (dissipative 2.5PN, 3.5PN) in ADM gauge: relative acceleration of Koenigsdoerffer, Faye &
// Schaefer 2003, PRD 68, 044004, Eq. (5.3) (= Iyer & Will 1995 with alpha = 5/3, beta = 3 and the delta_i of KFS
// Sect. V), from the 2.5PN Hamiltonian of Schaefer 1985 and the 3.5PN one of Jaranowski & Schaefer 1997. It is
// applied as a non-Hamiltonian force F on phat with xdot = dH/dphat unchanged, F = (d^2 H/dphat^2)^-1 a_RR(x, v), so
// the relative (x, v) dynamics is exactly KFS's (a = a_cons(x, v) + H_pp F).
//
// ADM -> modified-harmonic (MH) contact transformation (c.o.m.): x_h = x + d2 / c^4 + d3 / c^6, coefficients from
// tests/proto_admpn/derive_contact.py (solved against Blanchet LRR's MH acceleration; equivalent to Andrade, Blanchet
// & Faye 2001, CQG 18, 753, and DJS 2001 CQG 18, 3165, in MH instead of standard-harmonic coordinates); v_h = dx_h/dt
// along the ADM conservative flow. At 1PN the two coordinate systems coincide (the EIH Lagrangian's Legendre
// transform is H_1PN). The 2.5PN gauge differs (harmonic: alpha = -1, beta = 0): an O(c^-5) oscillatory shift that
// the map does not include.
#include "../proto_symrk/nbody_ham.h"
#include <cmath>
#include <vector>

namespace admpn {

using symrk::Dual;

struct Mono { double c; int a, b, pc, k; };   // c * u^a * p2^b * pr^pc / c^(2k)

struct PairHam {
    std::vector<Mono> t;
    double nu = 0.25;
    // orders: bit 1 = 1PN, 2 = 2PN, 8 = 3PN (as PN_ORDER), conservative only
    void build(double nu_, int orders) {
        nu = nu_;
        const double n = nu, n2 = n * n, n3 = n2 * n, pi2 = M_PI * M_PI;
        t.clear();
        if (orders & 1) {
            t.push_back({(3 * n - 1) / 8.0, 0, 2, 0, 1});
            t.push_back({-(3 + n) / 2.0, 1, 1, 0, 1});
            t.push_back({-n / 2.0, 1, 0, 2, 1});
            t.push_back({0.5, 2, 0, 0, 1});
        }
        if (orders & 2) {
            t.push_back({(1 - 5 * n + 5 * n2) / 16.0, 0, 3, 0, 2});
            t.push_back({(5 - 20 * n - 3 * n2) / 8.0, 1, 2, 0, 2});
            t.push_back({-2 * n2 / 8.0, 1, 1, 2, 2});
            t.push_back({-3 * n2 / 8.0, 1, 0, 4, 2});
            t.push_back({(5 + 8 * n) / 2.0, 2, 1, 0, 2});
            t.push_back({1.5 * n, 2, 0, 2, 2});
            t.push_back({-(1 + 3 * n) / 4.0, 3, 0, 0, 2});
        }
        if (orders & 8) {
            t.push_back({(-5 + 35 * n - 70 * n2 + 35 * n3) / 128.0, 0, 4, 0, 3});
            t.push_back({(-7 + 42 * n - 53 * n2 - 5 * n3) / 16.0, 1, 3, 0, 3});
            t.push_back({(2 - 3 * n) * n2 / 16.0, 1, 2, 2, 3});
            t.push_back({3 * (1 - n) * n2 / 16.0, 1, 1, 4, 3});
            t.push_back({-5 * n3 / 16.0, 1, 0, 6, 3});
            t.push_back({(-27 + 136 * n + 109 * n2) / 16.0, 2, 2, 0, 3});
            t.push_back({(17 + 30 * n) * n / 16.0, 2, 1, 2, 3});
            t.push_back({(5 + 43 * n) * n / 12.0, 2, 0, 4, 3});
            // omega_kinetic = 41/24 folded in: -55/12 + 41/24 = -23/8, 27/8 - 3 * 41/24 = -7/4
            t.push_back({-25.0 / 8 + (pi2 / 64 - 335.0 / 48) * n - 23.0 / 8 * n2, 3, 1, 0, 3});
            t.push_back({(-85.0 / 16 - 3 * pi2 / 64 - 7.0 / 4 * n) * n, 3, 0, 2, 3});
            t.push_back({1.0 / 8 + (109.0 / 12 - 21.0 / 32 * pi2) * n, 4, 0, 0, 3});
        }
    }
    // f = H_PN/mu (no Newtonian part) and its partial derivatives in (u, p2, pr); second p-derivatives if want2
    template <class T>
    void eval(T u, T p2, T pr, double ic2, T& f, T& fu, T& fp2, T& fpr, T* f2 = nullptr) const {
        f = fu = fp2 = fpr = T(0.0);
        T fp2p2 = T(0.0), fp2pr = T(0.0), fprpr = T(0.0);
        // power tables (index i holds x^i; index -1 -> 0 via offset 1)
        T U[7], P2[7], PR[9];
        U[0] = P2[0] = PR[0] = T(0.0);
        U[1] = P2[1] = PR[1] = T(1.0);
        for (int i = 2; i < 7; i++) { U[i] = U[i - 1] * u; P2[i] = P2[i - 1] * p2; }
        for (int i = 2; i < 9; i++) PR[i] = PR[i - 1] * pr;
        const double ck_[4] = {1.0, ic2, ic2 * ic2, ic2 * ic2 * ic2};
        for (const Mono& m : t) {
            const double ck = m.c * ck_[m.k];
            const T ua = U[m.a + 1], pb = P2[m.b + 1], pcp = PR[m.pc + 1];
            f = f + ck * ua * pb * pcp;
            if (m.a) fu = fu + (ck * m.a) * U[m.a] * pb * pcp;
            if (m.b) fp2 = fp2 + (ck * m.b) * ua * P2[m.b] * pcp;
            if (m.pc) fpr = fpr + (ck * m.pc) * ua * pb * PR[m.pc];
            if (f2) {
                if (m.b >= 2) fp2p2 = fp2p2 + (ck * m.b * (m.b - 1)) * ua * P2[m.b - 1] * pcp;
                if (m.b && m.pc) fp2pr = fp2pr + (ck * m.b * m.pc) * ua * P2[m.b] * PR[m.pc];
                if (m.pc >= 2) fprpr = fprpr + (ck * m.pc * (m.pc - 1)) * ua * pb * PR[m.pc - 1];
            }
        }
        if (f2) { f2[0] = fp2p2; f2[1] = fp2pr; f2[2] = fprpr; }
    }
    // H_PN/mu at relative x, phat; gradients dx (d/dx), dp (d/dphat); Gm = G M
    template <class T>
    T value_grad(const T* x, const T* p, double Gm, double ic2, T* dx, T* dp, T* Hpp = nullptr) const {
        T r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        T r = sqrt(r2), ir = T(1.0) / r;
        T n[3] = {x[0] * ir, x[1] * ir, x[2] * ir};
        T u = Gm * ir;
        T p2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
        T pr = n[0] * p[0] + n[1] * p[1] + n[2] * p[2];
        T f, fu, fp2, fpr, f2[3];
        eval(u, p2, pr, ic2, f, fu, fp2, fpr, Hpp ? f2 : nullptr);
        for (int k = 0; k < 3; k++) {
            dx[k] = fu * (-u * n[k] * ir) + fpr * (p[k] - pr * n[k]) * ir;
            dp[k] = 2.0 * fp2 * p[k] + fpr * n[k];
        }
        if (Hpp)
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    Hpp[3 * i + j] = (i == j ? 2.0 * fp2 : T(0.0)) + 4.0 * f2[0] * p[i] * p[j]
                                     + 2.0 * f2[1] * (p[i] * n[j] + n[i] * p[j]) + f2[2] * n[i] * n[j];
        return f;
    }
};

// ADM-gauge radiation-reaction relative acceleration (KFS 2003 Eq. 5.3); orders bit 4 = 2.5PN, 16 = 3.5PN
inline void rr_accel_adm(const double* x, const double* v, double Gm, double nu, double c, int orders, double* a) {
    const double r = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]), ir = 1.0 / r;
    const double n[3] = {x[0] * ir, x[1] * ir, x[2] * ir};
    const double v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    const double rd = n[0] * v[0] + n[1] * v[1] + n[2] * v[2];
    const double m = Gm * ir, nu2 = nu * nu;
    double cn = 0.0, cv = 0.0;
    const double f2 = m * m * ir;   // (G M)^2 / r^3
    if (orders & 4) {
        const double ic5 = 1.0 / (c * c * c * c * c);
        cn += ic5 * (f2 * (-24.0 * rd * rd * rd * nu + 96.0 / 5 * rd * v2 * nu) + f2 * m * (16.0 / 5 * rd * nu));
        cv += ic5 * (f2 * (64.0 / 5 * rd * rd * nu - 88.0 / 15 * v2 * nu) + f2 * m * (-8.0 / 15 * nu));
    }
    if (orders & 16) {
        const double ic7 = 1.0 / (c * c * c * c * c * c * c);
        const double rd2 = rd * rd, rd3 = rd2 * rd, rd4 = rd2 * rd2, rd5 = rd4 * rd, v4 = v2 * v2;
        cn += ic7 * (f2 * (-46.0 * rd5 * nu + 24.0 * rd5 * nu2 + v4 * (-138.0 / 35 * rd * nu - 516.0 / 35 * rd * nu2)
                           + v2 * (56.0 * rd3 * nu - 4.0 / 7 * rd3 * nu2))
                     + f2 * m * (10188.0 / 35 * rd3 * nu + 324.0 / 7 * rd3 * nu2
                                 + v2 * (-18656.0 / 105 * rd * nu - 1116.0 / 35 * rd * nu2))
                     + f2 * m * m * (-152.0 / 15 * rd * nu - 632.0 / 105 * rd * nu2));
        cv += ic7 * (f2 * (334.0 / 7 * rd4 * nu - 268.0 / 7 * rd4 * nu2 + v4 * (1006.0 / 105 * nu - 64.0 / 105 * nu2)
                           + v2 * (-2356.0 / 35 * rd2 * nu + 148.0 / 5 * rd2 * nu2))
                     + f2 * m * (-17308.0 / 105 * rd2 * nu - 244.0 / 21 * rd2 * nu2 + v2 * (4394.0 / 105 * nu - 16.0 / 35 * nu2))
                     + f2 * m * m * (-(386.0 / 105 * nu + 16.0 / 15 * nu2)));
    }
    for (int k = 0; k < 3; k++) a[k] = cn * n[k] + cv * v[k];
}

inline void solve3(const double* A, const double* b, double* x) {
    const double d = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
    double inv[9] = {A[4] * A[8] - A[5] * A[7], A[2] * A[7] - A[1] * A[8], A[1] * A[5] - A[2] * A[4],
                     A[5] * A[6] - A[3] * A[8], A[0] * A[8] - A[2] * A[6], A[2] * A[3] - A[0] * A[5],
                     A[3] * A[7] - A[4] * A[6], A[1] * A[6] - A[0] * A[7], A[0] * A[4] - A[1] * A[3]};
    for (int i = 0; i < 3; i++) x[i] = (inv[3 * i] * b[0] + inv[3 * i + 1] * b[1] + inv[3 * i + 2] * b[2]) / d;
}

// ----------------------------------------------------------------------------------------------------------------
// N-body system in the extended phase space y = (q[3n], p[3n], t, p_t), Poincare time transformation g = 1/(-U)
// (as symrk::NBody POINC): K = g (H + p_t); with radiation reaction p_t follows -W (radiated work) so that
// H + p_t = 0 is kept: dp_t/ds = -g sum_a v_a . F_a.
// ----------------------------------------------------------------------------------------------------------------
struct PnPair { int i, j; };

struct AdmSystem {
    symrk::NBody nb;              // Newtonian + (nb.pn) N-body ADM 1PN; nb.form unused
    std::vector<PnPair> pairs;    // PN pairs: 2PN/3PN two-body terms (+ 1PN if !nb.pn) and RR
    int cons_orders = 1 | 2 | 8;  // of the pair terms (bit 1 ignored when nb.pn)
    int rr_orders = 0;            // 4 = 2.5PN, 16 = 3.5PN
    bool rr_in_rhs = true;        // variant A (RR force inside the collocation); false: conservative rhs (split, B)
    std::vector<PairHam> ph;      // per pair
    std::vector<PairHam> ph1;     // per pair, 1PN..3PN (for the Hessian of the RR force)
    double c = 0.0;
    int dim() const { return 6 * nb.n + 2; }

    void setup() {
        ph.resize(pairs.size()); ph1.resize(pairs.size());
        for (size_t k = 0; k < pairs.size(); k++) {
            const double mi = nb.m[pairs[k].i], mj = nb.m[pairs[k].j], M = mi + mj;
            int o = cons_orders & ~(nb.pn ? 1 : 0);
            ph[k].build(mi * mj / (M * M), o);
            ph1[k].build(mi * mj / (M * M), (nb.pn ? 1 : 0) | cons_orders);   // the conservative orders actually present
        }
        nb.c = c;
    }

    // H(q, p) and gradients (Newton + PN)
    double hamiltonian(const double* q, const double* p, double& U, double* gU, double* Hq, double* Hp) const {
        U = nb.potential(q, gU);
        double H = nb.hamiltonian(q, p, U, gU, Hq, Hp);
        const double ic2 = c > 0 ? 1.0 / (c * c) : 0.0;
        if (!(c > 0)) return H;
        for (size_t k = 0; k < pairs.size(); k++) {
            const int i = pairs[k].i, j = pairs[k].j;
            const double mi = nb.m[i], mj = nb.m[j], M = mi + mj, mu = mi * mj / M;
            double x[3], ph_[3], dx[3], dp[3];
            for (int d = 0; d < 3; d++) {
                x[d] = q[3 * i + d] - q[3 * j + d];
                ph_[d] = (mj * p[3 * i + d] - mi * p[3 * j + d]) / (M * mu);
            }
            H += mu * ph[k].value_grad(x, ph_, nb.G * M, ic2, dx, dp);
            for (int d = 0; d < 3; d++) {
                Hq[3 * i + d] += mu * dx[d];
                Hq[3 * j + d] -= mu * dx[d];
                // d/dp_i = (m_j / (M mu)) d/dphat * mu
                Hp[3 * i + d] += dp[d] * mj / M;
                Hp[3 * j + d] -= dp[d] * mi / M;
            }
        }
        return H;
    }
    double energy(const double* y) const {
        double U, gU[24], Hq[24], Hp[24];
        return hamiltonian(y, y + 3 * nb.n, U, gU, Hq, Hp);
    }

    // RR forces on the momenta (F[3n]) given q, p and the velocities v = dH/dp; returns sum v . F
    double rr_force(const double* q, const double* p, const double* v, double* F) const {
        const int n3 = 3 * nb.n;
        for (int i = 0; i < n3; i++) F[i] = 0.0;
        if (!rr_orders || !(c > 0)) return 0.0;
        const double ic2 = 1.0 / (c * c);
        double P = 0.0;
        for (size_t k = 0; k < pairs.size(); k++) {
            const int i = pairs[k].i, j = pairs[k].j;
            const double mi = nb.m[i], mj = nb.m[j], M = mi + mj, mu = mi * mj / M, nu = mu / M;
            double x[3], vr[3], ph_[3], a[3], dx[3], dp[3], Hpp[9], Fh[3];
            for (int d = 0; d < 3; d++) {
                x[d] = q[3 * i + d] - q[3 * j + d];
                vr[d] = v[3 * i + d] - v[3 * j + d];
                ph_[d] = (mj * p[3 * i + d] - mi * p[3 * j + d]) / (M * mu);
            }
            rr_accel_adm(x, vr, nb.G * M, nu, c, rr_orders, a);
            ph1[k].value_grad(x, ph_, nb.G * M, ic2, dx, dp, Hpp);
            for (int d = 0; d < 3; d++) Hpp[4 * d] += 1.0;   // Newtonian p^2/2
            solve3(Hpp, a, Fh);
            for (int d = 0; d < 3; d++) {
                F[3 * i + d] += mu * Fh[d];
                F[3 * j + d] -= mu * Fh[d];
                P += mu * Fh[d] * vr[d];
            }
        }
        return P;
    }

    template <class T>
    void rhs(const T* y, T* f) const {
        const int n3 = 3 * nb.n;
        const double* q = y; const double* p = y + n3; const double pt = y[2 * n3 + 1];
        double U, gU[24], Hq[24], Hp[24];
        const double H = hamiltonian(q, p, U, gU, Hq, Hp);
        const double g = -1.0 / U, hpt = H + pt, g2 = g * g;
        for (int i = 0; i < n3; i++) {
            f[i] = g * Hp[i];
            f[n3 + i] = -g * Hq[i] - hpt * gU[i] * g2;
        }
        f[2 * n3] = g;
        f[2 * n3 + 1] = 0.0;
        if (rr_in_rhs && rr_orders) {
            double F[24];
            const double P = rr_force(q, p, Hp, F);
            for (int i = 0; i < n3; i++) f[n3 + i] += g * F[i];
            f[2 * n3 + 1] = -g * P;
        }
    }
};

// ----------------------------------------------------------------------------------------------------------------
// Contact transformation ADM -> MH, c.o.m. reduced variables (G M = Gm, x physical, phat per reduced mass).
// d = (A2 + A3) n + (B2 + B3) pr phat with A2 = [a1 p^2 + a2 pr^2 + a3 u] (G M) / c^4 etc. (derive_contact.py)
// ----------------------------------------------------------------------------------------------------------------
struct ContactCoeffs { double a[4], b[9]; };
ContactCoeffs contact_coeffs(double nu);   // defined in contact_coeffs.h (generated)

template <class T>
void contact_shift(const T* x, const T* p, double Gm, double c, const ContactCoeffs& K, T* d) {
    T r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    T r = sqrt(r2), ir = T(1.0) / r;
    T n[3] = {x[0] * ir, x[1] * ir, x[2] * ir};
    T u = Gm * ir;
    T p2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    T pr = n[0] * p[0] + n[1] * p[1] + n[2] * p[2];
    const double ic2 = 1.0 / (c * c), ic4 = ic2 * ic2, ic6 = ic4 * ic2;
    // units: in G M = 1 the shift is (a1 p^2 + a2 pr^2 + a3 u) n; physically times G M / c^4 (velocities^2 / c^2 ...)
    T A = (Gm * ic4) * (K.a[0] * p2 + K.a[1] * pr * pr + K.a[2] * u)
        + (Gm * ic6) * (K.b[0] * p2 * p2 + K.b[1] * p2 * pr * pr + K.b[2] * pr * pr * pr * pr + K.b[3] * p2 * u
                        + K.b[4] * pr * pr * u + K.b[5] * u * u);
    T B = (Gm * ic4) * (K.a[3] * pr) + (Gm * ic6) * ((K.b[6] * p2 + K.b[7] * pr * pr + K.b[8] * u) * pr);
    for (int k = 0; k < 3; k++) d[k] = A * n[k] + B * p[k];
}

// Reduced c.o.m. two-body ADM conservative Hamiltonian (Newton + orders) gradient, for the map
struct Reduced {
    PairHam h;
    double Gm = 1.0, c = 1.0, nu = 0.25;
    void build(double nu_, double Gm_, double c_, int orders) { nu = nu_; Gm = Gm_; c = c_; h.build(nu_, orders); }
    double H(const double* x, const double* p, double* Hx, double* Hp) const {
        double dx[3], dp[3];
        double f = h.value_grad(x, p, Gm, 1.0 / (c * c), dx, dp);
        const double r = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        double p2 = 0;
        for (int k = 0; k < 3; k++) {
            p2 += p[k] * p[k];
            Hx[k] = dx[k] + Gm * x[k] / (r * r * r);
            Hp[k] = dp[k] + p[k];
        }
        return f + 0.5 * p2 - Gm / r;
    }
};

// (x, phat) ADM -> (x_h, v_h) MH; v_h = d x_h / dt along the conservative ADM flow (exact chain rule, dual numbers)
inline void adm_to_harm(const Reduced& R, const ContactCoeffs& K, const double* x, const double* p, double* xh, double* vh) {
    using D6 = Dual<6>;
    D6 X[3], P[3], d[3];
    for (int k = 0; k < 3; k++) { X[k] = D6(x[k]); X[k].d[k] = 1.0; P[k] = D6(p[k]); P[k].d[3 + k] = 1.0; }
    contact_shift(X, P, R.Gm, R.c, K, d);
    double Hx[3], Hp[3];
    R.H(x, p, Hx, Hp);
    for (int k = 0; k < 3; k++) {
        xh[k] = x[k] + d[k].v;
        double dd = 0.0;
        for (int j = 0; j < 3; j++) dd += d[k].d[j] * Hp[j] - d[k].d[3 + j] * Hx[j];
        vh[k] = Hp[k] + dd;
    }
}

// inverse: fixed-point / Newton iteration (the map is identity + O(1/c^2) in p, O(1/c^4) in x)
inline int harm_to_adm(const Reduced& R, const ContactCoeffs& K, const double* xh, const double* vh, double* x, double* p) {
    for (int k = 0; k < 3; k++) { x[k] = xh[k]; p[k] = vh[k]; }
    int it = 0;
    for (; it < 60; it++) {
        double xh1[3], vh1[3];
        adm_to_harm(R, K, x, p, xh1, vh1);
        // Jacobian dv/dp ~ Hpp: use the 1PN Hessian for a Newton step on p, plain correction on x
        double Hpp[9], dx[3], dp[3];
        R.h.value_grad(x, p, R.Gm, 1.0 / (R.c * R.c), dx, dp, Hpp);
        for (int d = 0; d < 3; d++) Hpp[4 * d] += 1.0;
        double rv[3], corr[3], err = 0.0;
        for (int k = 0; k < 3; k++) rv[k] = vh[k] - vh1[k];
        solve3(Hpp, rv, corr);
        for (int k = 0; k < 3; k++) {
            const double ex = xh[k] - xh1[k];
            x[k] += ex; p[k] += corr[k];
            err = std::max(err, std::fabs(ex) / (std::fabs(xh[k]) + 1e-300));
            err = std::max(err, std::fabs(rv[k]) / (std::fabs(vh[k]) + 1e-300));
        }
        if (err < 1e-16) break;
    }
    return it;
}

}  // namespace admpn
