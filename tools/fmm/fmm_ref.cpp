// Reference Cartesian fast multipole method (phase 1 of the far-field plan, DEV_NOTES.md item 7).
//
// Purpose: force error and cost against expansion order p and opening angle theta for a Plummer sphere,
// before anything is built into the integrator. Serial and written for clarity, not speed.
//
// Method (Dehnen 2000, 2002; Cartesian Taylor expansions):
//   - octree, leaves of at most n_leaf particles; multipoles M_n = sum m (x - z)^n / n! about each cell's
//     centre of mass z, for multi-indices |n| <= p
//   - dual tree walk with mutual interactions: a well-separated pair (r_A + r_B < theta |z_A - z_B|) gets
//     both local expansions from one derivative tensor D_n(z_A - z_B) of the softened kernel; close leaf
//     pairs interact directly and mutually. Every interaction is antisymmetric, so total momentum is
//     conserved to round-off (Newton's third law by construction)
//   - downward pass: local expansions shifted to the children, evaluated at the particles
// Kernel: phi = -G m / sqrt(r^2 + eps^2) (Plummer softening, as in RUMI++). Its derivatives follow from
//   (r^2 + eps^2) d_i f = -R_i f differentiated with Leibniz' rule:
//   s D_n = -R_i D_{n-e_i} - q_i D_{n-2e_i} - sum_j [2 q_j R_j D_{n-e_j} + q_j (q_j - 1) D_{n-2e_j}],
//   s = r^2 + eps^2, q = n - e_i for any i with n_i > 0.
//
//   g++ -O3 -march=native -std=c++17 -o fmm_ref fmm_ref.cpp
//   ./fmm_ref N p theta [n_leaf=16] [eps=1e-3] [n_sample=2000] [seed=1]
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

constexpr int PMAX = 12;

// Multi-indices with |n| <= p, ordered by total degree
struct Multi {
    int p = 0, n_terms = 0;
    std::vector<std::array<int, 3>> idx;
    std::vector<int> deg;
    std::vector<double> inv_fact;            // 1 / (nx! ny! nz!)
    int lut[PMAX + 1][PMAX + 1][PMAX + 1];   // (nx, ny, nz) -> term, -1 if |n| > p
    explicit Multi(int p_) : p(p_) {
        for (auto& a : lut) for (auto& b : a) for (auto& c : b) c = -1;
        for (int d = 0; d <= p; d++)
            for (int x = d; x >= 0; x--)
                for (int y = d - x; y >= 0; y--) {
                    int z = d - x - y;
                    lut[x][y][z] = (int)idx.size();
                    idx.push_back({x, y, z});
                    deg.push_back(d);
                    inv_fact.push_back(1.0 / (std::tgamma(x + 1.0) * std::tgamma(y + 1.0) * std::tgamma(z + 1.0)));
                }
        n_terms = (int)idx.size();
    }
    int at(int x, int y, int z) const { return (x < 0 || y < 0 || z < 0 || x + y + z > p) ? -1 : lut[x][y][z]; }
    // powers r^n / n! for all terms
    void powers(const double r[3], double* out) const {
        double px[PMAX + 1], py[PMAX + 1], pz[PMAX + 1];
        px[0] = py[0] = pz[0] = 1.0;
        for (int k = 1; k <= p; k++) { px[k] = px[k - 1] * r[0]; py[k] = py[k - 1] * r[1]; pz[k] = pz[k - 1] * r[2]; }
        for (int t = 0; t < n_terms; t++) out[t] = px[idx[t][0]] * py[idx[t][1]] * pz[idx[t][2]] * inv_fact[t];
    }
    // D_n(R) of f = 1/sqrt(r^2 + eps^2) for all |n| <= p
    void derivs(const double R[3], double eps2, double* D) const {
        double s = R[0] * R[0] + R[1] * R[1] + R[2] * R[2] + eps2;
        double inv_s = 1.0 / s;
        D[0] = std::sqrt(inv_s);
        for (int t = 1; t < n_terms; t++) {
            const auto& n = idx[t];
            int i = n[0] > 0 ? 0 : (n[1] > 0 ? 1 : 2);
            int q[3] = {n[0], n[1], n[2]};
            q[i]--;
            auto get = [&](int dx, int dy, int dz) {   // D_{n - (dx,dy,dz)}
                int k = at(n[0] - dx, n[1] - dy, n[2] - dz);
                return k < 0 ? 0.0 : D[k];
            };
            int ei[3] = {0, 0, 0};
            ei[i] = 1;
            double v = -R[i] * get(ei[0], ei[1], ei[2]) - q[i] * get(2 * ei[0], 2 * ei[1], 2 * ei[2]);
            for (int j = 0; j < 3; j++) {
                if (q[j] == 0) continue;
                int ej[3] = {0, 0, 0};
                ej[j] = 1;
                v -= 2.0 * q[j] * R[j] * get(ej[0], ej[1], ej[2]);
                if (q[j] > 1) v -= q[j] * (q[j] - 1.0) * get(2 * ej[0], 2 * ej[1], 2 * ej[2]);
            }
            D[t] = v * inv_s;
        }
    }
};

struct Cell {
    double z[3] = {0, 0, 0};   // centre of mass
    double rmax = 0;           // max distance of a particle from z
    double mass = 0;
    int beg = 0, end = 0;      // particles [beg, end) in tree order
    int child[8];
    int n_child = 0;
    std::vector<double> M, L;  // multipoles, local coefficients (L_k = d^k phi / dx^k at z)
};

struct FMM {
    const Multi& mi;
    double G, eps2, theta;
    int n_leaf;
    std::vector<double> x, y, z, m;        // tree order
    std::vector<double> ax, ay, az, pot;
    std::vector<Cell> cells;
    long long n_m2l = 0, n_p2p = 0;

    FMM(const Multi& mi_, double G_, double eps, double th, int nl) : mi(mi_), G(G_), eps2(eps * eps), theta(th), n_leaf(nl) {}

    int build(int beg, int end, const double lo[3], double size) {
        int c = (int)cells.size();
        cells.emplace_back();
        cells[c].beg = beg; cells[c].end = end;
        if (end - beg > n_leaf && size > 1e-12) {
            double half = 0.5 * size;
            double mid[3] = {lo[0] + half, lo[1] + half, lo[2] + half};
            // partition into 8 octants (stable via buckets)
            std::vector<int> oct(end - beg);
            std::array<int, 9> cnt{};
            for (int k = beg; k < end; k++) {
                int o = (x[k] >= mid[0]) | ((y[k] >= mid[1]) << 1) | ((z[k] >= mid[2]) << 2);
                oct[k - beg] = o;
                cnt[o + 1]++;
            }
            for (int o = 0; o < 8; o++) cnt[o + 1] += cnt[o];
            std::vector<double> tx(end - beg), ty(end - beg), tz(end - beg), tm(end - beg);
            std::array<int, 9> pos = cnt;
            for (int k = beg; k < end; k++) {
                int d = pos[oct[k - beg]]++;
                tx[d] = x[k]; ty[d] = y[k]; tz[d] = z[k]; tm[d] = m[k];
            }
            std::copy(tx.begin(), tx.end(), x.begin() + beg);
            std::copy(ty.begin(), ty.end(), y.begin() + beg);
            std::copy(tz.begin(), tz.end(), z.begin() + beg);
            std::copy(tm.begin(), tm.end(), m.begin() + beg);
            for (int o = 0; o < 8; o++) {
                if (cnt[o + 1] == cnt[o]) continue;
                double clo[3] = {o & 1 ? mid[0] : lo[0], o & 2 ? mid[1] : lo[1], o & 4 ? mid[2] : lo[2]};
                int ch = build(beg + cnt[o], beg + cnt[o + 1], clo, half);
                cells[c].child[cells[c].n_child++] = ch;
            }
        }
        // centre of mass, rmax, multipoles (P2M for leaves, M2M otherwise)
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
        C.M.assign(mi.n_terms, 0.0);
        C.L.assign(mi.n_terms, 0.0);
        std::vector<double> pw(mi.n_terms);
        if (C.n_child == 0) {
            for (int k = beg; k < end; k++) {
                double d[3] = {x[k] - C.z[0], y[k] - C.z[1], z[k] - C.z[2]};
                mi.powers(d, pw.data());
                for (int t = 0; t < mi.n_terms; t++) C.M[t] += m[k] * pw[t];
            }
        } else {
            for (int q = 0; q < C.n_child; q++) {
                const Cell& ch = cells[C.child[q]];
                double d[3] = {ch.z[0] - C.z[0], ch.z[1] - C.z[1], ch.z[2] - C.z[2]};
                mi.powers(d, pw.data());   // d^j / j!
                for (int t = 0; t < mi.n_terms; t++) {   // M_n += sum_{j <= n} d^{n-j}/(n-j)! M^c_j
                    const auto& n = mi.idx[t];
                    double s = 0;
                    for (int jx = 0; jx <= n[0]; jx++)
                        for (int jy = 0; jy <= n[1]; jy++)
                            for (int jz = 0; jz <= n[2]; jz++)
                                s += pw[mi.at(n[0] - jx, n[1] - jy, n[2] - jz)] * ch.M[mi.at(jx, jy, jz)];
                    C.M[t] += s;
                }
            }
        }
        return c;
    }

    // Mutual M2L: L_A,k += -G sum_n (-1)^|n| M_B,n D_{n+k}(R),  L_B,k += -G (-1)^|k| sum_n M_A,n D_{n+k}(R)
    void m2l(Cell& A, Cell& B) {
        n_m2l++;
        double R[3] = {A.z[0] - B.z[0], A.z[1] - B.z[1], A.z[2] - B.z[2]};
        double D[1024];
        mi.derivs(R, eps2, D);
        for (int k = 0; k < mi.n_terms; k++) {
            const auto& kk = mi.idx[k];
            double sa = 0, sb = 0;
            for (int n = 0; n < mi.n_terms; n++) {
                if (mi.deg[n] + mi.deg[k] > mi.p) break;   // ordered by degree
                const auto& nn = mi.idx[n];
                double d = D[mi.at(nn[0] + kk[0], nn[1] + kk[1], nn[2] + kk[2])];
                sa += ((mi.deg[n] & 1) ? -B.M[n] : B.M[n]) * d;
                sb += A.M[n] * d;
            }
            A.L[k] += -G * sa;
            B.L[k] += -G * ((mi.deg[k] & 1) ? -sb : sb);
        }
    }

    void p2p(const Cell& A, const Cell& B) {   // A != B, mutual
        n_p2p += (long long)(A.end - A.beg) * (B.end - B.beg);
        for (int i = A.beg; i < A.end; i++)
            for (int j = B.beg; j < B.end; j++) {
                double dx = x[j] - x[i], dy = y[j] - y[i], dz = z[j] - z[i];
                double s = dx * dx + dy * dy + dz * dz + eps2;
                double ir = 1.0 / std::sqrt(s), ir3 = ir * ir * ir;
                ax[i] += G * m[j] * ir3 * dx; ay[i] += G * m[j] * ir3 * dy; az[i] += G * m[j] * ir3 * dz;
                ax[j] -= G * m[i] * ir3 * dx; ay[j] -= G * m[i] * ir3 * dy; az[j] -= G * m[i] * ir3 * dz;
                pot[i] -= G * m[j] * ir; pot[j] -= G * m[i] * ir;
            }
    }
    void p2p_self(const Cell& A) {
        n_p2p += (long long)(A.end - A.beg) * (A.end - A.beg - 1) / 2;
        for (int i = A.beg; i < A.end; i++)
            for (int j = i + 1; j < A.end; j++) {
                double dx = x[j] - x[i], dy = y[j] - y[i], dz = z[j] - z[i];
                double s = dx * dx + dy * dy + dz * dz + eps2;
                double ir = 1.0 / std::sqrt(s), ir3 = ir * ir * ir;
                ax[i] += G * m[j] * ir3 * dx; ay[i] += G * m[j] * ir3 * dy; az[i] += G * m[j] * ir3 * dz;
                ax[j] -= G * m[i] * ir3 * dx; ay[j] -= G * m[i] * ir3 * dy; az[j] -= G * m[i] * ir3 * dz;
                pot[i] -= G * m[j] * ir; pot[j] -= G * m[i] * ir;
            }
    }

    void interact(int a, int b) {
        Cell& A = cells[a];
        Cell& B = cells[b];
        if (a == b) {
            if (A.n_child == 0) { p2p_self(A); return; }
            for (int i = 0; i < A.n_child; i++)
                for (int j = i; j < A.n_child; j++) interact(A.child[i], A.child[j]);
            return;
        }
        double dx = A.z[0] - B.z[0], dy = A.z[1] - B.z[1], dz = A.z[2] - B.z[2];
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (A.rmax + B.rmax < theta * dist) { m2l(A, B); return; }
        bool a_leaf = A.n_child == 0, b_leaf = B.n_child == 0;
        if (a_leaf && b_leaf) { p2p(A, B); return; }
        bool split_a = !a_leaf && (b_leaf || A.rmax >= B.rmax);
        if (split_a) { for (int i = 0; i < A.n_child; i++) interact(A.child[i], b); }
        else { for (int j = 0; j < B.n_child; j++) interact(a, B.child[j]); }
    }

    // L2L to children, L2P at leaves: a = -grad phi = -sum_k y^k/k! L_{k+e_d}
    void downward(int c) {
        Cell& C = cells[c];
        std::vector<double> pw(mi.n_terms);
        if (C.n_child == 0) {
            for (int i = C.beg; i < C.end; i++) {
                double yv[3] = {x[i] - C.z[0], y[i] - C.z[1], z[i] - C.z[2]};
                mi.powers(yv, pw.data());
                double g[3] = {0, 0, 0}, ph = 0;
                for (int k = 0; k < mi.n_terms; k++) {
                    const auto& kk = mi.idx[k];
                    ph += pw[k] * C.L[k];
                    if (mi.deg[k] == mi.p) continue;
                    g[0] += pw[k] * C.L[mi.at(kk[0] + 1, kk[1], kk[2])];
                    g[1] += pw[k] * C.L[mi.at(kk[0], kk[1] + 1, kk[2])];
                    g[2] += pw[k] * C.L[mi.at(kk[0], kk[1], kk[2] + 1)];
                }
                ax[i] -= g[0]; ay[i] -= g[1]; az[i] -= g[2]; pot[i] += ph;
            }
            return;
        }
        for (int q = 0; q < C.n_child; q++) {
            Cell& ch = cells[C.child[q]];
            double s[3] = {ch.z[0] - C.z[0], ch.z[1] - C.z[1], ch.z[2] - C.z[2]};
            mi.powers(s, pw.data());
            for (int k = 0; k < mi.n_terms; k++) {   // L^c_k += sum_j s^j/j! L_{k+j}
                const auto& kk = mi.idx[k];
                double v = 0;
                for (int j = 0; j < mi.n_terms; j++) {
                    if (mi.deg[j] + mi.deg[k] > mi.p) break;
                    const auto& jj = mi.idx[j];
                    v += pw[j] * C.L[mi.at(kk[0] + jj[0], kk[1] + jj[1], kk[2] + jj[2])];
                }
                ch.L[k] += v;
            }
            downward(C.child[q]);
        }
    }
};

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s N p theta [n_leaf] [eps] [n_sample] [seed]\n", argv[0]); return 1; }
    int N = std::atoi(argv[1]), p = std::atoi(argv[2]);
    double theta = std::atof(argv[3]);
    int n_leaf = argc > 4 ? std::atoi(argv[4]) : 16;
    double eps = argc > 5 ? std::atof(argv[5]) : 1e-3;
    int n_sample = argc > 6 ? std::atoi(argv[6]) : 2000;
    int seed = argc > 7 ? std::atoi(argv[7]) : 1;
    if (p > PMAX) { std::fprintf(stderr, "p <= %d\n", PMAX); return 1; }
    Multi mi(p);

    // Plummer sphere, a = 1, M = 1, G = 1 (eps in units of a)
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    FMM f(mi, 1.0, eps, theta, n_leaf);
    f.x.resize(N); f.y.resize(N); f.z.resize(N); f.m.assign(N, 1.0 / N);
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
    f.interact(0, 0);
    auto t2 = std::chrono::steady_clock::now();
    f.downward(0);
    auto t3 = std::chrono::steady_clock::now();
    double tb = std::chrono::duration<double>(t1 - t0).count(), tw = std::chrono::duration<double>(t2 - t1).count(),
           td = std::chrono::duration<double>(t3 - t2).count();

    // momentum: sum m a relative to sum m |a|
    long double P[3] = {0, 0, 0}, Pa = 0;
    for (int i = 0; i < N; i++) {
        P[0] += f.m[i] * f.ax[i]; P[1] += f.m[i] * f.ay[i]; P[2] += f.m[i] * f.az[i];
        Pa += f.m[i] * std::sqrt(f.ax[i] * f.ax[i] + f.ay[i] * f.ay[i] + f.az[i] * f.az[i]);
    }
    double dP = (double)(std::sqrt(P[0] * P[0] + P[1] * P[1] + P[2] * P[2]) / Pa);

    // direct-summation reference on a sample (compensated sums)
    std::vector<double> err;
    std::uniform_int_distribution<int> pick(0, N - 1);
    double t_dir0 = 0;
    auto td0 = std::chrono::steady_clock::now();
    for (int s = 0; s < n_sample; s++) {
        int i = pick(rng);
        double a[3] = {0, 0, 0}, c[3] = {0, 0, 0};
        for (int j = 0; j < N; j++) {
            if (j == i) continue;
            double dx = f.x[j] - f.x[i], dy = f.y[j] - f.y[i], dz = f.z[j] - f.z[i];
            double sq = dx * dx + dy * dy + dz * dz + f.eps2;
            double ir = 1.0 / std::sqrt(sq), w = f.m[j] * ir * ir * ir;
            double v[3] = {w * dx, w * dy, w * dz};
            for (int d = 0; d < 3; d++) { double yv = v[d] - c[d], t = a[d] + yv; c[d] = (t - a[d]) - yv; a[d] = t; }
        }
        double e = std::sqrt(std::pow(f.ax[i] - a[0], 2) + std::pow(f.ay[i] - a[1], 2) + std::pow(f.az[i] - a[2], 2)) /
                   std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        err.push_back(e);
    }
    t_dir0 = std::chrono::duration<double>(std::chrono::steady_clock::now() - td0).count();
    std::sort(err.begin(), err.end());
    auto q = [&](double f_) { return err[std::min((size_t)(f_ * err.size()), err.size() - 1)]; };
    double rms = 0;
    for (double e : err) rms += e * e;
    rms = std::sqrt(rms / err.size());
    std::printf("N=%d p=%d theta=%.2f n_leaf=%d eps=%.1e | err rms=%.2e median=%.2e 99%%=%.2e max=%.2e | |sum m a|/sum m|a|=%.1e | "
                "build %.3f s walk %.3f s down %.3f s (M2L %lld, P2P %.2e pairs) | direct est %.1f s\n",
                N, p, theta, n_leaf, eps, rms, q(0.5), q(0.99), err.back(), dP, tb, tw, td, f.n_m2l, (double)f.n_p2p,
                t_dir0 / n_sample * N);
    return 0;
}
