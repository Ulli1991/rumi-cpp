// Spherical-harmonic fast multipole method (far-field plan phase 1b). Tree, dual walk and tests as in
// fmm_ref.cpp; solid-harmonic expansions (harmonics.h, identities checked in harm_test.cpp), O(p^4) M2L.
// The far field is exactly Newtonian (1/r). Softening acts only in the direct near field (P2P):
//   kernel 0 = none, 1 = Plummer (inconsistent with the far field: for comparison only),
//   2 = cubic spline (Monaghan & Lattanzio 1985), exactly 1/r beyond h = 2.8 eps, so the far field is exact
//   provided every cell pair closer than h is opened (enforced by the MAC below).
// With kernel 0 the forces agree with fmm_ref (eps = 0) at the same p and theta.
//
// Expansions (Phi = sum q / |x - x_i|, phi = -G Phi, a = G grad Phi):
//   P2M  M_n^m = sum q conj(Y_n^m(x_i - z))
//   M2M  M_n^m(parent) += sum_{k,l} conj(Y_k^l(d)) M_{n-k}^{m-l}(child),  d = z_child - z_parent
//   M2L  L_k^l(A) += (-1)^k sum_{n,m} M_n^m(B) T_{n+k}^{m+l}(R),  R = z_A - z_B,  n + k <= p
//        L_k^l(B) += sum_{n,m} (-1)^n M_n^m(A) T_{n+k}^{m+l}(R)          (T(-R) = (-1)^n T(R))
//   L2L  L_q^r(child) += sum_{j,i} conj(Y_j^i(s)) L_{q+j}^{r+i},  s = z_child - z_parent
//   L2P  Phi(z + y) = sum L_k^l conj(Y_k^l(y)); gradient from d_z Y_n^m = Y_{n-1}^m,
//        d_x Y_n^m = (Y_{n-1}^{m-1} - Y_{n-1}^{m+1}) / 2, d_y Y_n^m = i (Y_{n-1}^{m-1} + Y_{n-1}^{m+1}) / 2
//
//   g++ -O3 -march=native -std=c++17 -o fmm_sph fmm_sph.cpp
//   ./fmm_sph N p theta [n_leaf=16] [eps=1e-3] [kernel=2] [n_sample=2000] [seed=1]
#include "harmonics.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// Pair kernel: g(r) with acceleration G m g(r) (x_j - x_i), potential -G m w(r)
struct Kernel {
    int type;      // 0 none, 1 Plummer, 2 spline
    double eps, h; // h: spline support
    void eval(double r2, double& g, double& w) const {
        if (type == 1) { double ir = 1.0 / std::sqrt(r2 + eps * eps); g = ir * ir * ir; w = ir; return; }
        double r = std::sqrt(r2);
        if (type == 0 || r >= h) { double ir = 1.0 / r; g = ir * ir * ir; w = ir; return; }
        // Monaghan & Lattanzio cubic spline potential and force (as in Gadget-2, h = 2.8 eps)
        double hinv = 1.0 / h, u = r * hinv;
        if (u < 0.5) {
            g = hinv * hinv * hinv * (10.666666666667 + u * u * (32.0 * u - 38.4));
            w = -hinv * (-2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6)));
        } else {
            g = hinv * hinv * hinv * (21.333333333333 - 48.0 * u + 38.4 * u * u - 10.666666666667 * u * u * u - 0.066666666667 / (u * u * u));
            w = -hinv * (-3.2 + 0.066666666667 / u + u * u * (10.666666666667 + u * (-16.0 + u * (9.6 - 2.133333333333 * u))));
        }
    }
};

struct Cell {
    double z[3] = {0, 0, 0};
    double rmax = 0, mass = 0;
    int beg = 0, end = 0;
    int child[8];
    int n_child = 0;
    std::vector<cplx> M, L;
};

struct FMM {
    int p;
    double G, theta;
    Kernel ker;
    int n_leaf;
    std::vector<double> x, y, z, m, ax, ay, az, pot;
    std::vector<int> id;   // original index of each tree-ordered particle
    std::vector<Cell> cells;
    long long n_m2l = 0, n_p2p = 0;
    double t_m2l = 0, t_p2p = 0;
    std::vector<cplx> Yb, Tb;   // scratch
    HarmonicRotator rot;
    bool use_rot = getenv("FMM_NOROT") == nullptr;
    std::vector<double> fact;   // n!

    RealMap fx90, fxm90, bx90, bxm90;   // conj(S) and conj(S)^T of the +-90 degree x rotations, real form

    FMM(int p_, double G_, double th, Kernel k, int nl) : p(p_), G(G_), theta(th), ker(k), n_leaf(nl),
                                                       Yb(hsize(p)), Tb(hsize(p)), rot(p_), fact(2 * p_ + 2, 1.0) {
        for (int n = 1; n < (int)fact.size(); n++) fact[n] = fact[n - 1] * n;
        auto mk = [&](const std::vector<std::vector<cplx>>& S, bool tr) {
            return [&S, tr](int n, const cplx* v, cplx* out) {
                const int d = 2 * n + 1;
                for (int a = 0; a < d; a++) {
                    cplx acc = 0;
                    for (int b = 0; b < d; b++) acc += std::conj(tr ? S[n][b * d + a] : S[n][a * d + b]) * v[b];
                    out[a] = acc;
                }
            };
        };
        fx90.build(p, mk(rot.Sx90, false)); fxm90.build(p, mk(rot.Sxm90, false));
        bx90.build(p, mk(rot.Sx90, true));  bxm90.build(p, mk(rot.Sxm90, true));
    }

    static constexpr int RMAX = 64;
    // real form of all degrees: offset of degree n is n^2 (2n+1 entries)
    void pack(const std::vector<cplx>& v, double* r) const {
        for (int n = 0; n <= p; n++) {
            double* rn = r + n * n;
            rn[0] = v[hidx(n, 0)].real();
            for (int m = 1; m <= n; m++) { rn[2 * m - 1] = v[hidx(n, m)].real(); rn[2 * m] = v[hidx(n, m)].imag(); }
        }
    }
    void unpack_add(const double* r, std::vector<cplx>& v) const {
        for (int n = 0; n <= p; n++) {
            const double* rn = r + n * n;
            v[hidx(n, 0)] += rn[0];
            for (int m = 1; m <= n; m++) {
                cplx c(rn[2 * m - 1], rn[2 * m]);
                v[hidx(n, m)] += c;
                v[hidx(n, -m)] += ((m & 1) ? -1.0 : 1.0) * std::conj(c);
            }
        }
    }
    void rz(double alpha, double* r) const {   // v_n^m *= e^{i m alpha}
        double c1 = std::cos(alpha), s1 = std::sin(alpha), cm = 1, sm = 0;
        for (int m = 1; m <= p; m++) {
            double c = cm * c1 - sm * s1, s = sm * c1 + cm * s1;
            cm = c; sm = s;
            for (int n = m; n <= p; n++) {
                double* rn = r + n * n;
                double a = rn[2 * m - 1], b = rn[2 * m];
                rn[2 * m - 1] = a * c - b * s; rn[2 * m] = a * s + b * c;
            }
        }
    }
    void rx(const RealMap& Mp, double* r) const {
        double tmp[2 * RMAX + 1];
        for (int n = 1; n <= p; n++) {
            Mp.apply(n, r + n * n, tmp);
            std::copy(tmp, tmp + 2 * n + 1, r + n * n);
        }
    }
    // m2l_rot in real form
    void m2l_real(Cell& A, Cell& B) {
        n_m2l++;
        double R[3] = {A.z[0] - B.z[0], A.z[1] - B.z[1], A.z[2] - B.z[2]};
        double r = std::sqrt(R[0] * R[0] + R[1] * R[1] + R[2] * R[2]);
        double th = std::acos(std::max(-1.0, std::min(1.0, R[2] / r))), ph = std::atan2(R[1], R[0]);
        double MA[(RMAX + 1) * (RMAX + 1)], MB[(RMAX + 1) * (RMAX + 1)], LA[(RMAX + 1) * (RMAX + 1)], LB[(RMAX + 1) * (RMAX + 1)];
        pack(A.M, MA); pack(B.M, MB);
        for (double* M : {MA, MB}) { rz(ph, M); rx(fx90, M); rz(th, M); rx(fxm90, M); }
        double ir = 1.0 / r, Tz[2 * RMAX + 2], pw = ir;
        for (int j = 0; j <= p; j++) { Tz[j] = fact[j] * pw; pw *= ir; }
        // L'_k^l (l >= 0) = (-1)^k sum_n M'_n^{-l} T_{n+k} with M'_n^{-l} = (-1)^l conj(M'_n^l); similarly for B
        for (int k = 0; k <= p; k++)
            for (int l = 0; l <= k; l++) {
                double sa_re = 0, sa_im = 0, sb_re = 0, sb_im = 0;
                for (int n = l; n + k <= p; n++) {
                    double t = Tz[n + k];
                    const double* mb = MB + n * n; const double* ma = MA + n * n;
                    double bre = l ? mb[2 * l - 1] : mb[0], bim = l ? mb[2 * l] : 0.0;
                    double are = l ? ma[2 * l - 1] : ma[0], aim = l ? ma[2 * l] : 0.0;
                    double sg = (n & 1) ? -1.0 : 1.0;
                    sa_re += bre * t; sa_im += bim * t;
                    sb_re += sg * are * t; sb_im += sg * aim * t;
                }
                double fa = (((k + l) & 1) ? -1.0 : 1.0), fb = ((l & 1) ? -1.0 : 1.0);
                double* la = LA + k * k; double* lb = LB + k * k;
                if (l == 0) { la[0] = fa * sa_re; lb[0] = fb * sb_re; }
                else { la[2 * l - 1] = fa * sa_re; la[2 * l] = -fa * sa_im; lb[2 * l - 1] = fb * sb_re; lb[2 * l] = -fb * sb_im; }
            }
        for (double* L : {LA, LB}) { rx(bxm90, L); rz(th, L); rx(bx90, L); rz(ph, L); }
        unpack_add(LA, A.L); unpack_add(LB, B.L);
    }

    // v <- conj(S) v (forward) or conj(S)^T v (transpose) per degree, S one of the fitted x rotations
    void apply_x(const std::vector<std::vector<cplx>>& S, cplx* v, bool transpose) const {
        cplx tmp[2 * 64 + 1];
        for (int n = 1; n <= p; n++) {
            const int d = 2 * n + 1;
            const cplx* Sn = S[n].data();
            cplx* vn = v + hidx(n, -n);
            for (int a = 0; a < d; a++) {
                cplx acc = 0;
                if (!transpose) for (int b = 0; b < d; b++) acc += std::conj(Sn[a * d + b]) * vn[b];
                else            for (int b = 0; b < d; b++) acc += std::conj(Sn[b * d + a]) * vn[b];
                tmp[a] = acc;
            }
            for (int a = 0; a < d; a++) vn[a] = tmp[a];
        }
    }
    void apply_z(double alpha, cplx* v) const {   // v_n^m *= e^{i m alpha}
        for (int m = 1; m <= p; m++) {
            cplx e = std::polar(1.0, m * alpha), ec = std::conj(e);
            for (int n = m; n <= p; n++) { v[hidx(n, m)] *= e; v[hidx(n, -m)] *= ec; }
        }
    }
    // Mutual M2L by rotation: Q = Rx(-90) Rz(-theta) Rx(90) Rz(-phi) maps R to r z^; M' = conj(S(Q)) M;
    // along z only T_j^0(r z^) = j!/r^{j+1} survives; L = conj(S(Q))^T L'
    void m2l_rot(Cell& A, Cell& B) {
        n_m2l++;
        double R[3] = {A.z[0] - B.z[0], A.z[1] - B.z[1], A.z[2] - B.z[2]};
        double r = std::sqrt(R[0] * R[0] + R[1] * R[1] + R[2] * R[2]);
        double th = std::acos(std::max(-1.0, std::min(1.0, R[2] / r))), ph = std::atan2(R[1], R[0]);
        const int H = hsize(p);
        std::vector<cplx> MA(A.M), MB(B.M), LA(H, 0.0), LB(H, 0.0);
        for (cplx* M : {MA.data(), MB.data()}) {
            apply_z(ph, M); apply_x(rot.Sx90, M, false); apply_z(th, M); apply_x(rot.Sxm90, M, false);
        }
        double ir = 1.0 / r, Tz[2 * 64 + 2];
        double pw = ir;
        for (int j = 0; j <= p; j++) { Tz[j] = fact[j] * pw; pw *= ir; }
        for (int k = 0; k <= p; k++)
            for (int l = -k; l <= k; l++) {
                cplx sa = 0, sb = 0;
                for (int n = std::abs(l); n + k <= p; n++) {
                    double t = Tz[n + k];
                    sa += MB[hidx(n, -l)] * t;
                    sb += ((n & 1) ? -1.0 : 1.0) * MA[hidx(n, -l)] * t;
                }
                LA[hidx(k, l)] = ((k & 1) ? -1.0 : 1.0) * sa;
                LB[hidx(k, l)] = sb;
            }
        for (cplx* L : {LA.data(), LB.data()}) {
            apply_x(rot.Sxm90, L, true); apply_z(th, L); apply_x(rot.Sx90, L, true); apply_z(ph, L);
        }
        for (int t = 0; t < H; t++) { A.L[t] += LA[t]; B.L[t] += LB[t]; }
    }

    int build(int beg, int end, const double lo[3], double size) {
        int c = (int)cells.size();
        cells.emplace_back();
        cells[c].beg = beg; cells[c].end = end;
        if (end - beg > n_leaf && size > 1e-12) {
            double half = 0.5 * size, mid[3] = {lo[0] + half, lo[1] + half, lo[2] + half};
            std::vector<int> oct(end - beg);
            std::array<int, 9> cnt{};
            for (int k = beg; k < end; k++) {
                int o = (x[k] >= mid[0]) | ((y[k] >= mid[1]) << 1) | ((z[k] >= mid[2]) << 2);
                oct[k - beg] = o; cnt[o + 1]++;
            }
            for (int o = 0; o < 8; o++) cnt[o + 1] += cnt[o];
            std::vector<double> tx(end - beg), ty(end - beg), tz(end - beg), tm(end - beg);
            std::vector<int> ti(end - beg);
            std::array<int, 9> pos = cnt;
            for (int k = beg; k < end; k++) { int d = pos[oct[k - beg]]++; tx[d] = x[k]; ty[d] = y[k]; tz[d] = z[k]; tm[d] = m[k]; ti[d] = id[k]; }
            std::copy(tx.begin(), tx.end(), x.begin() + beg); std::copy(ty.begin(), ty.end(), y.begin() + beg);
            std::copy(tz.begin(), tz.end(), z.begin() + beg); std::copy(tm.begin(), tm.end(), m.begin() + beg);
            std::copy(ti.begin(), ti.end(), id.begin() + beg);
            for (int o = 0; o < 8; o++) {
                if (cnt[o + 1] == cnt[o]) continue;
                double clo[3] = {o & 1 ? mid[0] : lo[0], o & 2 ? mid[1] : lo[1], o & 4 ? mid[2] : lo[2]};
                int ch = build(beg + cnt[o], beg + cnt[o + 1], clo, half);
                cells[c].child[cells[c].n_child++] = ch;
            }
        }
        Cell& C = cells[c];
        double mz[3] = {0, 0, 0}, M = 0;
        for (int k = beg; k < end; k++) { M += m[k]; mz[0] += m[k] * x[k]; mz[1] += m[k] * y[k]; mz[2] += m[k] * z[k]; }
        C.mass = M;
        for (int d = 0; d < 3; d++) C.z[d] = mz[d] / M;
        double r2 = 0;
        for (int k = beg; k < end; k++) {
            double dx = x[k] - C.z[0], dy = y[k] - C.z[1], dz = z[k] - C.z[2];
            r2 = std::max(r2, dx * dx + dy * dy + dz * dz);
        }
        C.rmax = std::sqrt(r2);
        C.M.assign(hsize(p), 0.0);
        C.L.assign(hsize(p), 0.0);
        if (C.n_child == 0) {
            for (int k = beg; k < end; k++) {
                double d[3] = {x[k] - C.z[0], y[k] - C.z[1], z[k] - C.z[2]};
                regular(d, p, Yb.data());
                for (int t = 0; t < hsize(p); t++) C.M[t] += m[k] * std::conj(Yb[t]);
            }
        } else {
            for (int q = 0; q < C.n_child; q++) {
                const Cell& ch = cells[C.child[q]];
                double d[3] = {ch.z[0] - C.z[0], ch.z[1] - C.z[1], ch.z[2] - C.z[2]};
                regular(d, p, Yb.data());
                for (int n = 0; n <= p; n++)
                    for (int mm = -n; mm <= n; mm++) {
                        cplx v = 0;
                        for (int k = 0; k <= n; k++)
                            for (int l = -k; l <= k; l++) {
                                int nk = n - k, ml = mm - l;
                                if (std::abs(ml) > nk) continue;
                                v += std::conj(Yb[hidx(k, l)]) * ch.M[hidx(nk, ml)];
                            }
                        C.M[hidx(n, mm)] += v;
                    }
            }
        }
        return c;
    }

    void m2l(Cell& A, Cell& B) {
        n_m2l++;
        double R[3] = {A.z[0] - B.z[0], A.z[1] - B.z[1], A.z[2] - B.z[2]};
        irregular(R, p, Tb.data());
        for (int k = 0; k <= p; k++)
            for (int l = -k; l <= k; l++) {
                cplx sa = 0, sb = 0;
                for (int n = 0; n + k <= p; n++)
                    for (int mm = -n; mm <= n; mm++) {
                        int ml = mm + l;
                        if (std::abs(ml) > n + k) continue;
                        cplx t = Tb[hidx(n + k, ml)];
                        sa += B.M[hidx(n, mm)] * t;
                        sb += ((n & 1) ? -1.0 : 1.0) * A.M[hidx(n, mm)] * t;
                    }
                A.L[hidx(k, l)] += ((k & 1) ? -1.0 : 1.0) * sa;
                B.L[hidx(k, l)] += sb;
            }
    }

    // Directed M2L (target t <- source s) in real form. The pair is oriented by cell index (lo = min, hi = max,
    // R = z_lo - z_hi) for both directions, so t <- s and s <- t use the same rotation: the two local updates
    // are those of the mutual m2l_real and momentum stays conserved to round-off.
    void m2l_dir(int t, int s) {
        Cell& T = cells[t];
        const Cell& S = cells[s];
        const int lo = std::min(t, s);
        const Cell& Lc = cells[lo];
        const Cell& Hc = cells[std::max(t, s)];
        double R[3] = {Lc.z[0] - Hc.z[0], Lc.z[1] - Hc.z[1], Lc.z[2] - Hc.z[2]};
        double r = std::sqrt(R[0] * R[0] + R[1] * R[1] + R[2] * R[2]);
        double th = std::acos(std::max(-1.0, std::min(1.0, R[2] / r))), ph = std::atan2(R[1], R[0]);
        double Ms[(RMAX + 1) * (RMAX + 1)], Lt[(RMAX + 1) * (RMAX + 1)];
        pack(S.M, Ms);
        rz(ph, Ms); rx(fx90, Ms); rz(th, Ms); rx(fxm90, Ms);
        double ir = 1.0 / r, Tz[2 * RMAX + 2], pw = ir;
        for (int j = 0; j <= p; j++) { Tz[j] = fact[j] * pw; pw *= ir; }
        const bool t_is_lo = (t == lo);   // lo plays A in the mutual formulas: L_A uses (-1)^k, L_B uses (-1)^n
        for (int k = 0; k <= p; k++)
            for (int l = 0; l <= k; l++) {
                double sre = 0, sim = 0;
                for (int n = l; n + k <= p; n++) {
                    double tt = Tz[n + k];
                    const double* ms = Ms + n * n;
                    double re = l ? ms[2 * l - 1] : ms[0], im = l ? ms[2 * l] : 0.0;
                    double sg = (!t_is_lo && (n & 1)) ? -1.0 : 1.0;
                    sre += sg * re * tt; sim += sg * im * tt;
                }
                double f = t_is_lo ? (((k + l) & 1) ? -1.0 : 1.0) : ((l & 1) ? -1.0 : 1.0);
                double* lt = Lt + k * k;
                if (l == 0) lt[0] = f * sre;
                else { lt[2 * l - 1] = f * sre; lt[2 * l] = -f * sim; }
            }
        rx(bxm90, Lt); rz(th, Lt); rx(bx90, Lt); rz(ph, Lt);
        unpack_add(Lt, T.L);
    }
    // Directed P2P: accelerations of target leaf particles from source leaf particles
    void p2p_dir(const Cell& T, const Cell& S) {
        for (int i = T.beg; i < T.end; i++) {
            double sx = 0, sy = 0, sz = 0, sp = 0;
            for (int j = S.beg; j < S.end; j++) {
                if (j == i) continue;
                double dx = x[j] - x[i], dy = y[j] - y[i], dz = z[j] - z[i], g, w;
                ker.eval(dx * dx + dy * dy + dz * dz, g, w);
                sx += m[j] * g * dx; sy += m[j] * g * dy; sz += m[j] * g * dz; sp += m[j] * w;
            }
            ax[i] += G * sx; ay[i] += G * sy; az[i] += G * sz; pot[i] -= G * sp;
        }
    }

    // Interaction lists from the dual walk, executed in parallel over target cells
    std::vector<std::vector<int>> far_src, near_src;
    void collect(int a, int b) {
        Cell& A = cells[a];
        Cell& B = cells[b];
        if (a == b) {
            if (A.n_child == 0) { near_src[a].push_back(a); return; }
            for (int i = 0; i < A.n_child; i++)
                for (int j = i; j < A.n_child; j++) collect(A.child[i], A.child[j]);
            return;
        }
        double dx = A.z[0] - B.z[0], dy = A.z[1] - B.z[1], dz = A.z[2] - B.z[2];
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        bool far = A.rmax + B.rmax < theta * dist && (ker.type != 2 || dist - A.rmax - B.rmax > ker.h);
        if (far) { far_src[a].push_back(b); far_src[b].push_back(a); n_m2l++; return; }
        bool a_leaf = A.n_child == 0, b_leaf = B.n_child == 0;
        if (a_leaf && b_leaf) {
            near_src[a].push_back(b); near_src[b].push_back(a);
            n_p2p += (long long)(A.end - A.beg) * (B.end - B.beg);
            return;
        }
        bool split_a = !a_leaf && (b_leaf || A.rmax >= B.rmax);
        if (split_a) { for (int i = 0; i < A.n_child; i++) collect(A.child[i], b); }
        else { for (int j = 0; j < B.n_child; j++) collect(a, B.child[j]); }
    }
    void run_lists() {
        const int nc = (int)cells.size();
        far_src.assign(nc, {}); near_src.assign(nc, {});
        auto t0 = std::chrono::steady_clock::now();
        collect(0, 0);
        auto t1 = std::chrono::steady_clock::now();
        #pragma omp parallel for schedule(dynamic, 4)
        for (int c = 0; c < nc; c++)
            for (int s : far_src[c]) m2l_dir(c, s);
        auto t2 = std::chrono::steady_clock::now();
        #pragma omp parallel for schedule(dynamic, 4)
        for (int c = 0; c < nc; c++)
            for (int s : near_src[c]) p2p_dir(cells[c], cells[s]);
        auto t3 = std::chrono::steady_clock::now();
        t_collect = std::chrono::duration<double>(t1 - t0).count();
        t_m2l = std::chrono::duration<double>(t2 - t1).count();
        t_p2p = std::chrono::duration<double>(t3 - t2).count();
    }
    double t_collect = 0;

    void pair(int i, int j) {   // mutual direct
        double dx = x[j] - x[i], dy = y[j] - y[i], dz = z[j] - z[i];
        double g, w;
        ker.eval(dx * dx + dy * dy + dz * dz, g, w);
        ax[i] += G * m[j] * g * dx; ay[i] += G * m[j] * g * dy; az[i] += G * m[j] * g * dz;
        ax[j] -= G * m[i] * g * dx; ay[j] -= G * m[i] * g * dy; az[j] -= G * m[i] * g * dz;
        pot[i] -= G * m[j] * w; pot[j] -= G * m[i] * w;
    }
    void p2p(const Cell& A, const Cell& B) {
        n_p2p += (long long)(A.end - A.beg) * (B.end - B.beg);
        for (int i = A.beg; i < A.end; i++)
            for (int j = B.beg; j < B.end; j++) pair(i, j);
    }
    void p2p_self(const Cell& A) {
        n_p2p += (long long)(A.end - A.beg) * (A.end - A.beg - 1) / 2;
        for (int i = A.beg; i < A.end; i++)
            for (int j = i + 1; j < A.end; j++) pair(i, j);
    }

    void interact(int a, int b) {
        Cell& A = cells[a];
        Cell& B = cells[b];
        if (a == b) {
            if (A.n_child == 0) { auto t0 = std::chrono::steady_clock::now(); p2p_self(A); t_p2p += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); return; }
            for (int i = 0; i < A.n_child; i++)
                for (int j = i; j < A.n_child; j++) interact(A.child[i], A.child[j]);
            return;
        }
        double dx = A.z[0] - B.z[0], dy = A.z[1] - B.z[1], dz = A.z[2] - B.z[2];
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        // well separated, and (spline) every particle pair beyond the softening support
        bool far = A.rmax + B.rmax < theta * dist && (ker.type != 2 || dist - A.rmax - B.rmax > ker.h);
        if (far) { auto t0 = std::chrono::steady_clock::now(); if (use_rot) m2l_real(A, B); else m2l(A, B); t_m2l += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); return; }
        bool a_leaf = A.n_child == 0, b_leaf = B.n_child == 0;
        if (a_leaf && b_leaf) { auto t0 = std::chrono::steady_clock::now(); p2p(A, B); t_p2p += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); return; }
        bool split_a = !a_leaf && (b_leaf || A.rmax >= B.rmax);
        if (split_a) { for (int i = 0; i < A.n_child; i++) interact(A.child[i], b); }
        else { for (int j = 0; j < B.n_child; j++) interact(a, B.child[j]); }
    }

    // Accelerations of particles (px, py, pz, pm) in their given order
    void solve(const std::vector<double>& px, const std::vector<double>& py, const std::vector<double>& pz,
               const std::vector<double>& pm, std::vector<double>& oax, std::vector<double>& oay, std::vector<double>& oaz) {
        const int N = (int)px.size();
        x = px; y = py; z = pz; m = pm;
        id.resize(N);
        for (int i = 0; i < N; i++) id[i] = i;
        double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
        for (int i = 0; i < N; i++) {
            lo[0] = std::min(lo[0], x[i]); hi[0] = std::max(hi[0], x[i]);
            lo[1] = std::min(lo[1], y[i]); hi[1] = std::max(hi[1], y[i]);
            lo[2] = std::min(lo[2], z[i]); hi[2] = std::max(hi[2], z[i]);
        }
        double size = std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]}) * 1.0000001;
        cells.clear();
        build(0, N, lo, size);
        ax.assign(N, 0); ay.assign(N, 0); az.assign(N, 0); pot.assign(N, 0);
        if (getenv("FMM_RECURSIVE")) interact(0, 0); else run_lists();
        downward(0);
        oax.resize(N); oay.resize(N); oaz.resize(N);
        for (int k = 0; k < N; k++) { oax[id[k]] = ax[k]; oay[id[k]] = ay[k]; oaz[id[k]] = az[k]; }
    }

    void downward(int c) {
        Cell& C = cells[c];
        if (C.n_child == 0) {
            for (int i = C.beg; i < C.end; i++) {
                double yv[3] = {x[i] - C.z[0], y[i] - C.z[1], z[i] - C.z[2]};
                regular(yv, p, Yb.data());
                auto Y = [&](int n, int mm) { return (n < 0 || std::abs(mm) > n) ? cplx(0) : std::conj(Yb[hidx(n, mm)]); };
                cplx ph = 0, gx = 0, gy = 0, gz = 0;
                for (int k = 0; k <= p; k++)
                    for (int l = -k; l <= k; l++) {
                        cplx Lk = C.L[hidx(k, l)];
                        ph += Lk * Y(k, l);
                        if (k == 0) continue;
                        // gradient of conj(Y_k^l): conj of the gradient of Y_k^l
                        gz += Lk * Y(k - 1, l);
                        gx += Lk * 0.5 * (Y(k - 1, l - 1) - Y(k - 1, l + 1));
                        gy += Lk * cplx(0, -0.5) * (Y(k - 1, l - 1) + Y(k - 1, l + 1));
                    }
                ax[i] += G * gx.real(); ay[i] += G * gy.real(); az[i] += G * gz.real();
                pot[i] -= G * ph.real();
            }
            return;
        }
        for (int q = 0; q < C.n_child; q++) {
            Cell& ch = cells[C.child[q]];
            double s[3] = {ch.z[0] - C.z[0], ch.z[1] - C.z[1], ch.z[2] - C.z[2]};
            regular(s, p, Yb.data());
            for (int qn = 0; qn <= p; qn++)
                for (int r = -qn; r <= qn; r++) {
                    cplx v = 0;
                    for (int j = 0; j + qn <= p; j++)
                        for (int i2 = -j; i2 <= j; i2++) {
                            int ri = r + i2;
                            if (std::abs(ri) > qn + j) continue;
                            v += std::conj(Yb[hidx(j, i2)]) * C.L[hidx(qn + j, ri)];
                        }
                    ch.L[hidx(qn, r)] += v;
                }
            downward(C.child[q]);
        }
    }
};

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s N p theta [n_leaf] [eps] [kernel] [n_sample] [seed]\n", argv[0]); return 1; }
    int N = std::atoi(argv[1]), p = std::atoi(argv[2]);
    double theta = std::atof(argv[3]);
    int n_leaf = argc > 4 ? std::atoi(argv[4]) : 16;
    double eps = argc > 5 ? std::atof(argv[5]) : 1e-3;
    int ktype = argc > 6 ? std::atoi(argv[6]) : 2;
    int n_sample = argc > 7 ? std::atoi(argv[7]) : 2000;
    int seed = argc > 8 ? std::atoi(argv[8]) : 1;
    Kernel ker{ktype, eps, 2.8 * eps};
    if (getenv("FMM_INTEG")) {   // leapfrog with direct or FMM forces from the same initial conditions
        int steps = std::atoi(getenv("FMM_INTEG"));
        double dt = getenv("FMM_DT") ? std::atof(getenv("FMM_DT")) : 1e-3;
        bool direct = theta <= 0;
        std::mt19937_64 rg(seed);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::vector<double> X(N), Yv(N), Z(N), Mv(N, 1.0 / N), VX(N), VY(N), VZ(N), AX, AY, AZ;
        for (int i = 0; i < N; i++) {   // Plummer positions and velocities (Aarseth, Henon & Wielen 1974), G = M = a = 1
            double r;
            do { r = 1.0 / std::sqrt(std::pow(U(rg), -2.0 / 3.0) - 1.0); } while (r > 50.0);
            double ct = 2 * U(rg) - 1, ph = 2 * M_PI * U(rg), st = std::sqrt(1 - ct * ct);
            X[i] = r * st * std::cos(ph); Yv[i] = r * st * std::sin(ph); Z[i] = r * ct;
            double q, g;
            do { q = U(rg); g = 0.1 * U(rg); } while (g > q * q * std::pow(1 - q * q, 3.5));
            double v = q * std::sqrt(2.0) * std::pow(1 + r * r, -0.25);
            double cv = 2 * U(rg) - 1, pv = 2 * M_PI * U(rg), sv = std::sqrt(1 - cv * cv);
            VX[i] = v * sv * std::cos(pv); VY[i] = v * sv * std::sin(pv); VZ[i] = v * cv;
        }
        FMM f(p, 1.0, theta, ker, n_leaf);
        auto forces = [&]() {
            if (!direct) { f.solve(X, Yv, Z, Mv, AX, AY, AZ); return; }
            AX.assign(N, 0); AY.assign(N, 0); AZ.assign(N, 0);
            for (int i = 0; i < N; i++)
                for (int j = i + 1; j < N; j++) {
                    double dx = X[j] - X[i], dy = Yv[j] - Yv[i], dz = Z[j] - Z[i], g, w;
                    ker.eval(dx * dx + dy * dy + dz * dz, g, w);
                    AX[i] += Mv[j] * g * dx; AY[i] += Mv[j] * g * dy; AZ[i] += Mv[j] * g * dz;
                    AX[j] -= Mv[i] * g * dx; AY[j] -= Mv[i] * g * dy; AZ[j] -= Mv[i] * g * dz;
                }
        };
        auto energy = [&]() {   // exact (direct) energy with the same kernel
            long double T = 0, W = 0;
            for (int i = 0; i < N; i++) T += 0.5L * Mv[i] * (VX[i] * VX[i] + VY[i] * VY[i] + VZ[i] * VZ[i]);
            for (int i = 0; i < N; i++)
                for (int j = i + 1; j < N; j++) {
                    double dx = X[j] - X[i], dy = Yv[j] - Yv[i], dz = Z[j] - Z[i], g, w;
                    ker.eval(dx * dx + dy * dy + dz * dz, g, w);
                    W -= (long double)Mv[i] * Mv[j] * w;
                }
            return (double)(T + W);
        };
        double E0 = energy();
        forces();
        double t_force = 0, maxdE = 0;
        int every = std::max(1, steps / 10);
        for (int s = 1; s <= steps; s++) {
            for (int i = 0; i < N; i++) { VX[i] += 0.5 * dt * AX[i]; VY[i] += 0.5 * dt * AY[i]; VZ[i] += 0.5 * dt * AZ[i]; }
            for (int i = 0; i < N; i++) { X[i] += dt * VX[i]; Yv[i] += dt * VY[i]; Z[i] += dt * VZ[i]; }
            auto t0 = std::chrono::steady_clock::now();
            forces();
            t_force += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            for (int i = 0; i < N; i++) { VX[i] += 0.5 * dt * AX[i]; VY[i] += 0.5 * dt * AY[i]; VZ[i] += 0.5 * dt * AZ[i]; }
            if (s % every == 0) {
                double dE = std::abs((energy() - E0) / E0);
                maxdE = std::max(maxdE, dE);
                std::printf("  step %5d  |dE/E| = %.3e  E = %.17e\n", s, dE, energy());
            }
        }
        std::printf("INTEG %s N=%d p=%d theta=%.2f leaf=%d eps=%.1e dt=%.1e steps=%d: max|dE/E| = %.3e, force time %.2f s (%.4f s/eval)\n",
                    direct ? "direct" : "fmm", N, p, theta, n_leaf, eps, dt, steps, maxdE, t_force, t_force / steps);
        return 0;
    }
    if (getenv("FMM_ROT_TEST")) {   // m2l vs m2l_rot on random cell pairs
        FMM f(p, 1.0, theta, ker, n_leaf);
        std::mt19937_64 rg(5);
        std::uniform_real_distribution<double> V(-1, 1);
        double worst = 0;
        for (int trial = 0; trial < 20; trial++) {
            Cell A, B;
            for (int d = 0; d < 3; d++) { A.z[d] = V(rg); B.z[d] = V(rg) + 3.0; }
            A.M.resize(hsize(p)); B.M.resize(hsize(p));
            for (int n = 0; n <= p; n++) for (int mm = 0; mm <= n; mm++) {   // multipoles of a real distribution
                cplx a(V(rg), mm ? V(rg) : 0.0), b(V(rg), mm ? V(rg) : 0.0);
                A.M[hidx(n, mm)] = a; B.M[hidx(n, mm)] = b;
                if (mm) { A.M[hidx(n, -mm)] = ((mm & 1) ? -1.0 : 1.0) * std::conj(a); B.M[hidx(n, -mm)] = ((mm & 1) ? -1.0 : 1.0) * std::conj(b); }
            }
            Cell A2 = A, B2 = B;
            A.L.assign(hsize(p), 0.0); B.L.assign(hsize(p), 0.0); A2.L = A.L; B2.L = B.L;
            f.m2l(A, B); f.m2l_real(A2, B2);
            for (int t = 0; t < hsize(p); t++) {
                worst = std::max(worst, std::abs(A.L[t] - A2.L[t]) / (std::abs(A.L[t]) + 1e-300));
                worst = std::max(worst, std::abs(B.L[t] - B2.L[t]) / (std::abs(B.L[t]) + 1e-300));
            }
        }
        std::printf("m2l vs m2l_real, p=%d: max rel diff %.2e\n", p, worst);
        return 0;
    }

    std::mt19937_64 rng(seed);   // same Plummer sphere as fmm_ref
    std::uniform_real_distribution<double> U(0.0, 1.0);
    FMM f(p, 1.0, theta, ker, n_leaf);
    f.x.resize(N); f.y.resize(N); f.z.resize(N); f.m.assign(N, 1.0 / N);
    f.id.resize(N);
    for (int i = 0; i < N; i++) f.id[i] = i;
    for (int i = 0; i < N; i++) {
        double r;
        do { r = 1.0 / std::sqrt(std::pow(U(rng), -2.0 / 3.0) - 1.0); } while (r > 50.0);
        double ct = 2 * U(rng) - 1, ph = 2 * M_PI * U(rng), st = std::sqrt(1 - ct * ct);
        f.x[i] = r * st * std::cos(ph); f.y[i] = r * st * std::sin(ph); f.z[i] = r * ct;
    }
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (int i = 0; i < N; i++) {
        lo[0] = std::min(lo[0], f.x[i]); hi[0] = std::max(hi[0], f.x[i]);
        lo[1] = std::min(lo[1], f.y[i]); hi[1] = std::max(hi[1], f.y[i]);
        lo[2] = std::min(lo[2], f.z[i]); hi[2] = std::max(hi[2], f.z[i]);
    }
    double size = std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]}) * 1.0000001;

    auto t0 = std::chrono::steady_clock::now();
    f.build(0, N, lo, size);
    auto t1 = std::chrono::steady_clock::now();
    f.ax.assign(N, 0); f.ay.assign(N, 0); f.az.assign(N, 0); f.pot.assign(N, 0);
    if (getenv("FMM_RECURSIVE")) f.interact(0, 0); else f.run_lists();
    auto t2 = std::chrono::steady_clock::now();
    f.downward(0);
    auto t3 = std::chrono::steady_clock::now();
    double tb = std::chrono::duration<double>(t1 - t0).count(), tw = std::chrono::duration<double>(t2 - t1).count(),
           td = std::chrono::duration<double>(t3 - t2).count();

    long double P[3] = {0, 0, 0}, Pa = 0;
    for (int i = 0; i < N; i++) {
        P[0] += f.m[i] * f.ax[i]; P[1] += f.m[i] * f.ay[i]; P[2] += f.m[i] * f.az[i];
        Pa += f.m[i] * std::sqrt(f.ax[i] * f.ax[i] + f.ay[i] * f.ay[i] + f.az[i] * f.az[i]);
    }
    double dP = (double)(std::sqrt(P[0] * P[0] + P[1] * P[1] + P[2] * P[2]) / Pa);

    std::vector<double> err;
    std::uniform_int_distribution<int> pick(0, N - 1);
    auto td0 = std::chrono::steady_clock::now();
    for (int s = 0; s < n_sample; s++) {
        int i = pick(rng);
        double a[3] = {0, 0, 0}, c[3] = {0, 0, 0};
        for (int j = 0; j < N; j++) {
            if (j == i) continue;
            double dx = f.x[j] - f.x[i], dy = f.y[j] - f.y[i], dz = f.z[j] - f.z[i];
            double g, w;
            ker.eval(dx * dx + dy * dy + dz * dz, g, w);
            double v[3] = {f.m[j] * g * dx, f.m[j] * g * dy, f.m[j] * g * dz};
            for (int d = 0; d < 3; d++) { double yv = v[d] - c[d], t = a[d] + yv; c[d] = (t - a[d]) - yv; a[d] = t; }
        }
        err.push_back(std::sqrt(std::pow(f.ax[i] - a[0], 2) + std::pow(f.ay[i] - a[1], 2) + std::pow(f.az[i] - a[2], 2)) /
                      std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]));
    }
    double t_dir = std::chrono::duration<double>(std::chrono::steady_clock::now() - td0).count() / n_sample * N;
    std::sort(err.begin(), err.end());
    double rms = 0;
    for (double e : err) rms += e * e;
    rms = std::sqrt(rms / err.size());
    std::printf("SPH N=%d p=%d theta=%.2f n_leaf=%d eps=%.1e kernel=%d | err rms=%.2e median=%.2e max=%.2e | dP=%.1e | "
                "build %.3f s walk %.3f s [lists %.3f s M2L %.3f s P2P %.3f s] down %.3f s (M2L %lld, P2P %.2e) | direct est %.1f s\n",
                N, p, theta, n_leaf, eps, ktype, rms, err[err.size() / 2], err.back(), dP, tb, tw, f.t_collect, f.t_m2l, f.t_p2p, td, f.n_m2l,
                (double)f.n_p2p, t_dir);
    return 0;
}
