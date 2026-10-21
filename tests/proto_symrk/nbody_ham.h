#pragma once
// Time-transformed N-body Hamiltonians in the extended phase space y = (q[3n], p[3n], t, p_t), p_t = -E0
// (HLW GNI Sect. VIII.2; Mikkola & Tanikawa 1999, MNRAS 310, 745; Preto & Tremaine 1999, AJ 118, 2532):
//  LOG  : Gamma = ln(T(p) + p_t) - ln(-U(q))           (LogH, Newtonian only; separable)
//         dq/ds = (p/m) / (T + p_t), dp/ds = grad U / U, dt/ds = 1 / (T + p_t)
//  POINC: K = g(q) (H(q, p) + p_t), g = 1 / (-U(q))    (Poincare/Sundman; any H, also the non-separable PN1)
//         dq/ds = g dH/dp, dp/ds = -g dH/dq - (H + p_t) grad g, dt/ds = g
// On the energy shell H + p_t = 0 both give dt/ds = 1/(-U) (LogH time transformation).
// PN1: conservative N-body ADM Hamiltonian (Schaefer 1985; Schaefer & Jaranowski 2018, LRR 21:7, Eq. 4.?; two-body
// reduction checked against H_1PN/mu = (3 nu - 1) p^4/8 - [(3 + nu) p^2 + nu (n.p)^2]/(2r) + 1/(2 r^2)):
//   H1 = -sum_a p_a^4 / (8 m_a^3 c^2)
//        - G/(4c^2) sum_a sum_{b!=a} (m_a m_b / r_ab) [6 p_a^2/m_a^2 - 7 p_a.p_b/(m_a m_b) - (n.p_a)(n.p_b)/(m_a m_b)]
//        + G^2/(2c^2) sum_a m_a Phi_a^2,  Phi_a = sum_{b!=a} m_b / r_ab
// ADM coordinates differ from the harmonic (EIH) ones of the production LogH PN at O(1/c^2); each scheme is
// judged on the conservation of its own 1PN energy.
#include "gauss_irk.h"
#include <cmath>

namespace symrk {

// Forward-mode dual number with N directional derivatives (exact Jacobians of a whole step)
template <int N>
struct Dual {
    double v = 0.0;
    double d[N] = {};
    Dual() = default;
    Dual(double x) : v(x) {}
};
template <int N> inline double val_(const Dual<N>& x) { return x.v; }
#define DUAL_BIN(op, vexpr, dexpr)                                                                  \
    template <int N> inline Dual<N> operator op(const Dual<N>& a, const Dual<N>& b) {              \
        Dual<N> r; r.v = vexpr; for (int i = 0; i < N; i++) r.d[i] = dexpr; return r; }            \
    template <int N> inline Dual<N> operator op(double a_, const Dual<N>& b) { Dual<N> a(a_); return a op b; } \
    template <int N> inline Dual<N> operator op(const Dual<N>& a, double b_) { Dual<N> b(b_); return a op b; }
DUAL_BIN(+, a.v + b.v, a.d[i] + b.d[i])
DUAL_BIN(-, a.v - b.v, a.d[i] - b.d[i])
DUAL_BIN(*, a.v * b.v, a.d[i] * b.v + a.v * b.d[i])
DUAL_BIN(/, a.v / b.v, (a.d[i] * b.v - a.v * b.d[i]) / (b.v * b.v))
#undef DUAL_BIN
template <int N> inline Dual<N> operator-(const Dual<N>& a) { Dual<N> r; r.v = -a.v; for (int i = 0; i < N; i++) r.d[i] = -a.d[i]; return r; }
template <int N> inline Dual<N> sqrt(const Dual<N>& a) {
    Dual<N> r; r.v = std::sqrt(a.v); for (int i = 0; i < N; i++) r.d[i] = a.d[i] / (2.0 * r.v); return r; }
template <int N> inline Dual<N> fabs(const Dual<N>& a) { return a.v < 0 ? -a : a; }
using std::sqrt;

enum Form { LOG = 0, POINC = 1 };

struct NBody {
    int n = 2;
    double m[8] = {};
    double G = 1.0, c = 0.0;   // c > 0 and pn: PN1 Hamiltonian (POINC only)
    bool pn = false;
    Form form = POINC;
    int dim() const { return 6 * n + 2; }

    // U, grad U (Newtonian)
    template <class T>
    T potential(const T* q, T* gU) const {
        T U = T(0.0);
        for (int i = 0; i < 3 * n; i++) gU[i] = T(0.0);
        for (int a = 0; a < n; a++)
            for (int b = a + 1; b < n; b++) {
                T x[3]; T r2 = T(0.0);
                for (int k = 0; k < 3; k++) { x[k] = q[3 * a + k] - q[3 * b + k]; r2 = r2 + x[k] * x[k]; }
                T r = sqrt(r2), ir3 = T(1.0) / (r2 * r);
                const double Gmm = G * m[a] * m[b];
                U = U - Gmm / r;
                for (int k = 0; k < 3; k++) {   // dU/dq_a = G m_a m_b x / r^3
                    T f = Gmm * x[k] * ir3;
                    gU[3 * a + k] = gU[3 * a + k] + f;
                    gU[3 * b + k] = gU[3 * b + k] - f;
                }
            }
        return U;
    }

    // H (Newtonian [+ PN1]) and its gradients; U, gU are the Newtonian potential and gradient
    template <class T>
    T hamiltonian(const T* q, const T* p, T U, const T* gU, T* Hq, T* Hp) const {
        T H = U;
        for (int i = 0; i < 3 * n; i++) { Hq[i] = gU[i]; Hp[i] = p[i] / m[i / 3]; }
        for (int a = 0; a < n; a++)
            for (int k = 0; k < 3; k++) H = H + 0.5 * p[3 * a + k] * p[3 * a + k] / m[a];
        if (!(pn && c > 0.0)) return H;
        const double ic2 = 1.0 / (c * c);
        // kinetic: -p^4 / (8 m^3 c^2)
        for (int a = 0; a < n; a++) {
            const T* pa = p + 3 * a;
            T p2 = pa[0] * pa[0] + pa[1] * pa[1] + pa[2] * pa[2];
            const double m3 = m[a] * m[a] * m[a];
            H = H - 0.125 * ic2 * p2 * p2 / m3;
            for (int k = 0; k < 3; k++) Hp[3 * a + k] = Hp[3 * a + k] - 0.5 * ic2 * p2 * pa[k] / m3;
        }
        T Phi[8];
        for (int a = 0; a < n; a++) Phi[a] = T(0.0);
        const double kk = -G * 0.25 * ic2;
        for (int a = 0; a < n; a++)
            for (int b = a + 1; b < n; b++) {
                const T* pa = p + 3 * a; const T* pb = p + 3 * b;
                T x[3]; T r2 = T(0.0);
                for (int k = 0; k < 3; k++) { x[k] = q[3 * a + k] - q[3 * b + k]; r2 = r2 + x[k] * x[k]; }
                T r = sqrt(r2), ir = T(1.0) / r, ir3 = ir * ir * ir, ir5 = ir3 * ir * ir;
                Phi[a] = Phi[a] + m[b] * ir;
                Phi[b] = Phi[b] + m[a] * ir;
                T pa2 = T(0.0), pb2 = T(0.0), papb = T(0.0), xpa = T(0.0), xpb = T(0.0);
                for (int k = 0; k < 3; k++) {
                    pa2 = pa2 + pa[k] * pa[k]; pb2 = pb2 + pb[k] * pb[k]; papb = papb + pa[k] * pb[k];
                    xpa = xpa + x[k] * pa[k]; xpb = xpb + x[k] * pb[k];
                }
                const double rba = m[b] / m[a], rab = m[a] / m[b];
                // W = kk [A / r - 2 (x.pa)(x.pb) / r^3], A = 6 (mb/ma) pa^2 + 6 (ma/mb) pb^2 - 14 pa.pb
                T A = 6.0 * rba * pa2 + 6.0 * rab * pb2 - 14.0 * papb;
                H = H + kk * (A * ir - 2.0 * xpa * xpb * ir3);
                for (int k = 0; k < 3; k++) {
                    Hp[3 * a + k] = Hp[3 * a + k] + kk * ((12.0 * rba * pa[k] - 14.0 * pb[k]) * ir - 2.0 * x[k] * xpb * ir3);
                    Hp[3 * b + k] = Hp[3 * b + k] + kk * ((12.0 * rab * pb[k] - 14.0 * pa[k]) * ir - 2.0 * x[k] * xpa * ir3);
                    T dx = kk * (-A * x[k] * ir3 - 2.0 * (pa[k] * xpb + pb[k] * xpa) * ir3 + 6.0 * xpa * xpb * x[k] * ir5);
                    Hq[3 * a + k] = Hq[3 * a + k] + dx;
                    Hq[3 * b + k] = Hq[3 * b + k] - dx;
                }
            }
        // G^2/(2c^2) sum_a m_a Phi_a^2
        const double k3 = 0.5 * G * G * ic2;
        for (int a = 0; a < n; a++) H = H + k3 * m[a] * Phi[a] * Phi[a];
        for (int a = 0; a < n; a++)
            for (int b = a + 1; b < n; b++) {
                T x[3]; T r2 = T(0.0);
                for (int k = 0; k < 3; k++) { x[k] = q[3 * a + k] - q[3 * b + k]; r2 = r2 + x[k] * x[k]; }
                T r = sqrt(r2), ir3 = T(1.0) / (r2 * r);
                T f = (2.0 * k3) * m[a] * m[b] * (Phi[a] + Phi[b]) * ir3;
                for (int k = 0; k < 3; k++) {
                    Hq[3 * a + k] = Hq[3 * a + k] - f * x[k];
                    Hq[3 * b + k] = Hq[3 * b + k] + f * x[k];
                }
            }
        return H;
    }

    template <class T>
    void rhs(const T* y, T* f) const {
        const int n3 = 3 * n;
        const T* q = y; const T* p = y + n3; const T pt = y[2 * n3 + 1];
        T gU[24];
        T U = potential(q, gU);
        if (form == LOG) {
            T Tk = T(0.0);
            for (int i = 0; i < n3; i++) Tk = Tk + 0.5 * p[i] * p[i] / m[i / 3];
            T iw = T(1.0) / (Tk + pt), iU = T(1.0) / U;
            for (int i = 0; i < n3; i++) { f[i] = p[i] / m[i / 3] * iw; f[n3 + i] = gU[i] * iU; }
            f[2 * n3] = iw;
        } else {
            T Hq[24], Hp[24];
            T H = hamiltonian(q, p, U, gU, Hq, Hp);
            T g = -T(1.0) / U, hpt = H + pt;
            T g2 = g * g;   // grad g = grad U / U^2
            for (int i = 0; i < n3; i++) {
                f[i] = g * Hp[i];
                f[n3 + i] = -g * Hq[i] - hpt * gU[i] * g2;
            }
            f[2 * n3] = g;
        }
        f[2 * n3 + 1] = T(0.0);
    }

    double energy(const double* y) const {
        double gU[24], Hq[24], Hp[24];
        double U = potential(y, gU);
        bool save = pn;
        double H = hamiltonian(y, y + 3 * n, U, gU, Hq, Hp);
        (void)save;
        return H;
    }
};

}  // namespace symrk
