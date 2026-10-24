// GPU fast multipole method (Kokkos). See fmm.h; derivations, identity checks and the CPU reference in tools/fmm
// (harmonics.h, harm_test.cpp, fmm_sph.cpp). Formulas, with Phi = sum q / |x - x_i|, phi = -G Phi:
//   regular   Y_n^m(x) = r^n / (n+m)! P_n^m(cos t) e^{i m phi}   (no Condon-Shortley phase)
//   P2M  M_n^m = sum q conj(Y_n^m(x_i - z))
//   M2M  M_n^m(parent) += sum_{k,l} conj(Y_k^l(d)) M_{n-k}^{m-l}(child),  d = z_child - z_parent
//   M2L  along z by rotation: Q = Rx(-90) Rz(-theta) Rx(90) Rz(-phi) maps R = z_lo - z_hi to r z^,
//        T_j^0(r z^) = j!/r^{j+1};  L_lo <- (-1)^k sum_n M'_n^{-l} T_{n+k},  L_hi <- sum_n (-1)^n M'_n^{-l} T_{n+k}
//   L2L  L_q^r(child) += sum_{j,i} conj(Y_j^i(s)) L_{q+j}^{r+i},  s = z_child - z_parent
//   L2P  Phi(z + y) = sum L_k^l conj(Y_k^l(y)), a = G grad Phi
// Expansions of real distributions are stored in real form: degree n at offset n^2 as (a_0, a_1, b_1, ..., a_n, b_n),
// v_n^m = a_m + i b_m, v_n^{-m} = (-1)^m conj(v_n^m).
#include "fmm.h"
#include "constants.h"
#include <Kokkos_Complex.hpp>
#include <Kokkos_Sort.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

namespace {

using kc = Kokkos::complex<double>;
constexpr int PMAX = 20;
constexpr int TRI_MAX = (PMAX + 1) * (PMAX + 2) / 2;
constexpr int NR_MAX = (PMAX + 1) * (PMAX + 1);
constexpr int MORTON_BITS = 21;

KOKKOS_INLINE_FUNCTION int tri(int n, int m) { return n * (n + 1) / 2 + m; }

// Regular solid harmonics, m >= 0 only (recurrences of tools/fmm/harmonics.h)
KOKKOS_INLINE_FUNCTION void reg_pos(double x, double y, double z, int p, kc* Y) {
    const double r2 = x * x + y * y + z * z;
    const kc w(x, y);
    Y[0] = 1.0;
    for (int m = 1; m <= p; m++) Y[tri(m, m)] = Y[tri(m - 1, m - 1)] * w / (2.0 * m);
    for (int m = 0; m < p; m++) Y[tri(m + 1, m)] = z * Y[tri(m, m)];
    for (int m = 0; m <= p; m++)
        for (int n = m + 2; n <= p; n++)
            Y[tri(n, m)] = ((2.0 * n - 1.0) * z * Y[tri(n - 1, m)] - r2 * Y[tri(n - 2, m)]) / ((double)(n + m) * (n - m));
}
KOKKOS_INLINE_FUNCTION kc getY(const kc* Y, int n, int m) {   // any m, 0 outside |m| <= n
    if (n < 0 || m > n || -m > n) return kc(0.0);
    if (m >= 0) return Y[tri(n, m)];
    kc c = Kokkos::conj(Y[tri(n, -m)]);
    return ((-m) & 1) ? -c : c;
}
KOKKOS_INLINE_FUNCTION kc getR(const double* v, int n, int m) {   // real form, any m
    if (n < 0 || m > n || -m > n) return kc(0.0);
    const double* vn = v + n * n;
    int am = m < 0 ? -m : m;
    kc c = am ? kc(vn[2 * am - 1], vn[2 * am]) : kc(vn[0], 0.0);
    if (m < 0) { c = Kokkos::conj(c); if (am & 1) c = -c; }
    return c;
}
KOKKOS_INLINE_FUNCTION void addR(double* v, int n, int m, kc c) {   // m >= 0
    double* vn = v + n * n;
    if (m == 0) vn[0] += c.real();
    else { vn[2 * m - 1] += c.real(); vn[2 * m] += c.imag(); }
}

// Near-field kernel: acceleration factor g(r) (a = G m g r_vec); exactly Newtonian beyond the spline support
KOKKOS_INLINE_FUNCTION double kernel_g(double r2, int type, double h) {
    double r = Kokkos::sqrt(r2);
    if (type != 2 || r >= h) { double ir = 1.0 / r; return ir * ir * ir; }
    double hinv = 1.0 / h, u = r * hinv;
    return hinv * hinv * hinv * spline_g_inner(u);
}

KOKKOS_INLINE_FUNCTION uint64_t spread3(uint64_t v) {   // 21 bits -> every third bit
    v &= 0x1fffff;
    v = (v | v << 32) & 0x1f00000000ffffULL;
    v = (v | v << 16) & 0x1f0000ff0000ffULL;
    v = (v | v << 8) & 0x100f00f00f00f00fULL;
    v = (v | v << 4) & 0x10c30c30c30c30c3ULL;
    v = (v | v << 2) & 0x1249249249249249ULL;
    return v;
}

// ---------------------------------------------------------------------------------------------------------------
// Host: rotation matrices of the +-90 degree x rotations, in real form, sparse (tools/fmm/harmonics.h)
using sc = std::complex<double>;
inline int hidx(int n, int m) { return n * n + n + m; }
void regular_host(const double x[3], int p, std::vector<sc>& Y) {
    Y.assign((p + 1) * (p + 1), 0.0);
    const double r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    const sc w(x[0], x[1]);
    Y[hidx(0, 0)] = 1.0;
    for (int m = 1; m <= p; m++) Y[hidx(m, m)] = Y[hidx(m - 1, m - 1)] * w / (2.0 * m);
    for (int m = 0; m < p; m++) Y[hidx(m + 1, m)] = x[2] * Y[hidx(m, m)];
    for (int m = 0; m <= p; m++)
        for (int n = m + 2; n <= p; n++)
            Y[hidx(n, m)] = ((2.0 * n - 1.0) * x[2] * Y[hidx(n - 1, m)] - r2 * Y[hidx(n - 2, m)]) / ((double)(n + m) * (n - m));
    for (int n = 1; n <= p; n++)
        for (int m = 1; m <= n; m++) Y[hidx(n, -m)] = ((m & 1) ? -1.0 : 1.0) * std::conj(Y[hidx(n, m)]);
}
// S_n of Y_n(Rx(ang) x) = S_n Y_n(x) by an exact Gauss-Legendre x uniform quadrature over the sphere
std::vector<sc> rotation_x(int n, double ang) {
    const int d = 2 * n + 1, nmu = n + 2, nphi = 2 * n + 3;
    std::vector<double> mu(nmu), w(nmu);
    for (int i = 0; i < nmu; i++) {
        double x = std::cos(M_PI * (i + 0.75) / (nmu + 0.5)), dp = 0;
        for (int it = 0; it < 100; it++) {
            double p0 = 1, p1 = x;
            for (int k = 2; k <= nmu; k++) { double p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
            dp = nmu * (x * p1 - p0) / (x * x - 1);
            double dx = p1 / dp;
            x -= dx;
            if (std::abs(dx) < 1e-16) break;
        }
        mu[i] = x; w[i] = 2.0 / ((1 - x * x) * dp * dp);
    }
    std::vector<sc> S(d * d, 0.0), Yx, Yq;
    std::vector<double> norm(d, 0.0);
    const double c = std::cos(ang), s = std::sin(ang);
    for (int i = 0; i < nmu; i++)
        for (int j = 0; j < nphi; j++) {
            double st = std::sqrt(std::max(0.0, 1 - mu[i] * mu[i])), ph = 2 * M_PI * j / nphi;
            double x[3] = {st * std::cos(ph), st * std::sin(ph), mu[i]};
            double q[3] = {x[0], c * x[1] - s * x[2], s * x[1] + c * x[2]};
            regular_host(x, n, Yx);
            regular_host(q, n, Yq);
            double wt = w[i] * 2 * M_PI / nphi;
            for (int mp = -n; mp <= n; mp++) {
                sc cy = std::conj(Yx[hidx(n, mp)]) * wt;
                norm[mp + n] += std::norm(Yx[hidx(n, mp)]) * wt;
                for (int m = -n; m <= n; m++) S[(m + n) * d + mp + n] += Yq[hidx(n, m)] * cy;
            }
        }
    for (int m = 0; m < d; m++) for (int mp = 0; mp < d; mp++) S[m * d + mp] /= norm[mp];
    return S;
}
// Sparse real matrix of v -> conj(S) v (or conj(S)^T v) over all degrees: rows r (global index n^2 + local)
struct HostCSR { std::vector<int> ptr, col; std::vector<double> val; };
HostCSR real_map(int p, const std::vector<std::vector<sc>>& S, bool transpose) {
    HostCSR out;
    out.ptr.push_back(0);
    for (int n = 0; n <= p; n++) {
        const int d = 2 * n + 1;
        std::vector<double> Mr(d * d, 0.0);
        for (int c = 0; c < d; c++) {
            std::vector<sc> v(d, 0.0), o(d, 0.0);
            int m = (c + 1) / 2;
            sc unit = (c == 0) ? sc(1, 0) : ((c & 1) ? sc(1, 0) : sc(0, 1));
            v[m + n] = unit;
            if (m > 0) v[-m + n] = ((m & 1) ? -1.0 : 1.0) * std::conj(unit);
            for (int a = 0; a < d; a++) {
                sc acc = 0;
                for (int b = 0; b < d; b++) acc += std::conj(transpose ? S[n][b * d + a] : S[n][a * d + b]) * v[b];
                o[a] = acc;
            }
            for (int r = 0; r < d; r++) {
                int mo = (r + 1) / 2;
                Mr[r * d + c] = (r == 0) ? o[mo + n].real() : ((r & 1) ? o[mo + n].real() : o[mo + n].imag());
            }
        }
        double mx = 0;
        for (double v : Mr) mx = std::max(mx, std::abs(v));
        for (int r = 0; r < d; r++) {
            for (int c = 0; c < d; c++)
                if (std::abs(Mr[r * d + c]) > 1e-14 * mx) { out.col.push_back(n * n + c); out.val.push_back(Mr[r * d + c]); }
            out.ptr.push_back((int)out.col.size());
        }
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------
struct DevCSR {
    Kokkos::View<int*> ptr, col;
    Kokkos::View<double*> val;
};
DevCSR to_dev(const HostCSR& h, const char* name) {
    DevCSR d;
    d.ptr = Kokkos::View<int*>(std::string(name) + "_ptr", h.ptr.size());
    d.col = Kokkos::View<int*>(std::string(name) + "_col", std::max<size_t>(1, h.col.size()));
    d.val = Kokkos::View<double*>(std::string(name) + "_val", std::max<size_t>(1, h.val.size()));
    auto hp = Kokkos::create_mirror_view(d.ptr); auto hc = Kokkos::create_mirror_view(d.col);
    auto hv = Kokkos::create_mirror_view(d.val);
    for (size_t i = 0; i < h.ptr.size(); i++) hp(i) = h.ptr[i];
    for (size_t i = 0; i < h.col.size(); i++) { hc(i) = h.col[i]; hv(i) = h.val[i]; }
    Kokkos::deep_copy(d.ptr, hp); Kokkos::deep_copy(d.col, hc); Kokkos::deep_copy(d.val, hv);
    return d;
}

struct State {
    int p = -1;
    DevCSR fx90, fxm90, bx90, bxm90;
    Kokkos::View<double*> fact;                     // j!
    Kokkos::View<int*> pair_n, pair_m;              // (n, m >= 1) pairs for z rotations
    Kokkos::View<int*> tri_k, tri_l;                // (k, l >= 0) for the z translation
    // per call
    Kokkos::View<uint64_t*> keys;
    Kokkos::View<int*> perm, leaf_of;
    Kokkos::View<double*> sx, sy, sz, sm, ax, ay, az;
    Kokkos::View<int*> ssid;
    Kokkos::View<int*> cbeg, cend, cparent, cfirst, cnchild;
    Kokkos::View<double*[4]> cz;                    // centre of mass, rmax
    Kokkos::View<double**, Kokkos::LayoutRight> M, L;
    Kokkos::View<int*> far_ptr, far_idx, near_ptr, near_idx, leaves;
};
State* g_state = nullptr;

void setup(State& S, int p) {
    if (S.p == p) return;
    S.p = p;
    std::vector<std::vector<sc>> Sx(p + 1), Sxm(p + 1);
    for (int n = 0; n <= p; n++) { Sx[n] = rotation_x(n, M_PI / 2); Sxm[n] = rotation_x(n, -M_PI / 2); }
    S.fx90 = to_dev(real_map(p, Sx, false), "fx90");
    S.fxm90 = to_dev(real_map(p, Sxm, false), "fxm90");
    S.bx90 = to_dev(real_map(p, Sx, true), "bx90");
    S.bxm90 = to_dev(real_map(p, Sxm, true), "bxm90");
    S.fact = Kokkos::View<double*>("fmm_fact", 2 * p + 2);
    auto hf = Kokkos::create_mirror_view(S.fact);
    hf(0) = 1;
    for (int j = 1; j < 2 * p + 2; j++) hf(j) = hf(j - 1) * j;
    Kokkos::deep_copy(S.fact, hf);
    std::vector<int> pn, pm, tk, tl;
    for (int n = 1; n <= p; n++) for (int m = 1; m <= n; m++) { pn.push_back(n); pm.push_back(m); }
    for (int k = 0; k <= p; k++) for (int l = 0; l <= k; l++) { tk.push_back(k); tl.push_back(l); }
    auto up = [](const std::vector<int>& v, const char* nm) {
        Kokkos::View<int*> d(nm, v.size());
        auto h = Kokkos::create_mirror_view(d);
        for (size_t i = 0; i < v.size(); i++) h(i) = v[i];
        Kokkos::deep_copy(d, h);
        return d;
    };
    S.pair_n = up(pn, "fmm_pn"); S.pair_m = up(pm, "fmm_pm"); S.tri_k = up(tk, "fmm_tk"); S.tri_l = up(tl, "fmm_tl");
}

template <class V>
void ensure(V& v, size_t n, const char* name) { if (v.extent(0) < n) v = V(name, n); }

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

}  // namespace

void fmm_release() { delete g_state; g_state = nullptr; }

void fmm_accelerations(const Kokkos::View<double*>& x, const Kokkos::View<double*>& y, const Kokkos::View<double*>& z,
                       const Kokkos::View<double*>& m, const Kokkos::View<int*>& sid, int i0, int n,
                       const FmmParams& prm, Kokkos::View<double*> oax, Kokkos::View<double*> oay,
                       Kokkos::View<double*> oaz, FmmStats* stats) {
    if (n <= 0) return;
    if (prm.p > PMAX) Kokkos::abort("fmm: p > PMAX");
    if (!g_state) g_state = new State;
    State& S = *g_state;
    setup(S, prm.p);
    const int p = prm.p, NR = (p + 1) * (p + 1), NT = (p + 1) * (p + 2) / 2;
    FmmStats st;
    double t0 = now();

    // ---- Morton keys, sort, sorted copies -------------------------------------------------------------------
    ensure(S.keys, n, "fmm_keys"); ensure(S.perm, n, "fmm_perm");
    ensure(S.sx, n, "fmm_sx"); ensure(S.sy, n, "fmm_sy"); ensure(S.sz, n, "fmm_sz"); ensure(S.sm, n, "fmm_sm");
    ensure(S.ax, n, "fmm_ax"); ensure(S.ay, n, "fmm_ay"); ensure(S.az, n, "fmm_az"); ensure(S.ssid, n, "fmm_ssid");
    ensure(S.leaf_of, n, "fmm_leaf_of");
    double lo[3], hi[3];
    {
        Kokkos::MinMaxScalar<double> rx, ry, rz;
        Kokkos::parallel_reduce("fmm_bbox_x", n, KOKKOS_LAMBDA(int i, Kokkos::MinMaxScalar<double>& v) {
            double q = x(i0 + i); if (q < v.min_val) v.min_val = q; if (q > v.max_val) v.max_val = q; }, Kokkos::MinMax<double>(rx));
        Kokkos::parallel_reduce("fmm_bbox_y", n, KOKKOS_LAMBDA(int i, Kokkos::MinMaxScalar<double>& v) {
            double q = y(i0 + i); if (q < v.min_val) v.min_val = q; if (q > v.max_val) v.max_val = q; }, Kokkos::MinMax<double>(ry));
        Kokkos::parallel_reduce("fmm_bbox_z", n, KOKKOS_LAMBDA(int i, Kokkos::MinMaxScalar<double>& v) {
            double q = z(i0 + i); if (q < v.min_val) v.min_val = q; if (q > v.max_val) v.max_val = q; }, Kokkos::MinMax<double>(rz));
        lo[0] = rx.min_val; lo[1] = ry.min_val; lo[2] = rz.min_val; hi[0] = rx.max_val; hi[1] = ry.max_val; hi[2] = rz.max_val;
    }
    const double size = std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]}) * (1.0 + 1e-9) + 1e-300;
    const double scale = (double)(1u << MORTON_BITS) / size, lx = lo[0], ly = lo[1], lz = lo[2];
    auto keys = S.keys;
    auto perm = S.perm;
    Kokkos::parallel_for("fmm_keys", n, KOKKOS_LAMBDA(int i) {
        auto q = [&](double v, double l) {
            int64_t k = (int64_t)((v - l) * scale);
            return (uint64_t)(k < 0 ? 0 : (k > (1 << MORTON_BITS) - 1 ? (1 << MORTON_BITS) - 1 : k));
        };
        keys(i) = spread3(q(x(i0 + i), lx)) | (spread3(q(y(i0 + i), ly)) << 1) | (spread3(q(z(i0 + i), lz)) << 2);
        perm(i) = i;
    });
    auto keys_n = Kokkos::subview(keys, Kokkos::make_pair(0, n));
    auto perm_n = Kokkos::subview(perm, Kokkos::make_pair(0, n));
    Kokkos::Experimental::sort_by_key(Kokkos::DefaultExecutionSpace(), keys_n, perm_n);
    auto sx = S.sx, sy = S.sy, sz = S.sz, sm = S.sm;
    auto ssid = S.ssid;
    const bool has_sid = sid.extent(0) > 0;
    Kokkos::parallel_for("fmm_gather", n, KOKKOS_LAMBDA(int k) {
        int i = i0 + perm(k);
        sx(k) = x(i); sy(k) = y(i); sz(k) = z(i); sm(k) = m(i);
        ssid(k) = has_sid ? sid(i) : -1;
    });

    // ---- tree topology on the host from the sorted keys (BFS order: levels contiguous) ------------------------
    auto hkeys = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), keys_n);
    std::vector<int> hbeg, hend, hpar, hfirst, hnch, hlevel;
    std::vector<int> level_start;
    {
        struct Q { int beg, end, level, parent; };
        std::vector<Q> cur{{0, n, 0, -1}};
        while (!cur.empty()) {
            level_start.push_back((int)hbeg.size());
            std::vector<Q> next;
            for (const Q& q : cur) {
                int c = (int)hbeg.size();
                hbeg.push_back(q.beg); hend.push_back(q.end); hpar.push_back(q.parent); hlevel.push_back(q.level);
                hfirst.push_back(-1); hnch.push_back(0);
                if (q.end - q.beg > prm.n_leaf && q.level < MORTON_BITS) {
                    const int shift = 3 * (MORTON_BITS - 1 - q.level);
                    int b = q.beg;
                    for (int o = 0; o < 8; o++) {
                        // first index in [b, end) whose octant digit at this level exceeds o
                        int e = (int)(std::upper_bound(hkeys.data() + b, hkeys.data() + q.end, (uint64_t)o,
                                                       [shift](uint64_t v, uint64_t k) { return v < ((k >> shift) & 7); }) - hkeys.data());
                        if (e > b) next.push_back({b, e, q.level + 1, c});
                        b = e;
                    }
                }
            }
            // children of the cells just added start at the current end of the cell arrays, in order
            int first = (int)hbeg.size();
            for (const Q& q : next) {
                if (hfirst[q.parent] < 0) hfirst[q.parent] = first;
                hnch[q.parent]++;
                first++;
            }
            cur.swap(next);
        }
        level_start.push_back((int)hbeg.size());
    }
    const int nc = (int)hbeg.size();
    std::vector<int> hleaves;
    for (int c = 0; c < nc; c++) if (hnch[c] == 0) hleaves.push_back(c);
    st.n_cells = nc; st.n_leaves = (int)hleaves.size();
    auto upload = [](Kokkos::View<int*>& d, const std::vector<int>& h, const char* name) {
        ensure(d, std::max<size_t>(1, h.size()), name);
        auto hm = Kokkos::create_mirror_view(d);
        for (size_t i = 0; i < h.size(); i++) hm(i) = h[i];
        Kokkos::deep_copy(d, hm);
    };
    upload(S.cbeg, hbeg, "fmm_cbeg"); upload(S.cend, hend, "fmm_cend"); upload(S.cparent, hpar, "fmm_cpar");
    upload(S.cfirst, hfirst, "fmm_cfirst"); upload(S.cnchild, hnch, "fmm_cnch"); upload(S.leaves, hleaves, "fmm_leaves");
    if (S.cz.extent(0) < (size_t)nc) S.cz = Kokkos::View<double*[4]>("fmm_cz", nc);
    if (S.M.extent(0) < (size_t)nc || (int)S.M.extent(1) != NR) {
        S.M = Kokkos::View<double**, Kokkos::LayoutRight>("fmm_M", nc, NR);
        S.L = Kokkos::View<double**, Kokkos::LayoutRight>("fmm_L", nc, NR);
    }
    auto cbeg = S.cbeg, cend = S.cend, cpar = S.cparent, cfirst = S.cfirst, cnch = S.cnchild, leaves = S.leaves;
    auto cz = S.cz;
    auto M = S.M, L = S.L;
    auto leaf_of = S.leaf_of;

    // ---- geometry: centre of mass and rmax per cell -------------------------------------------------------------
    using team_policy = Kokkos::TeamPolicy<>;
    using member = team_policy::member_type;
    Kokkos::parallel_for("fmm_geometry", team_policy(nc, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t) {
        const int c = t.league_rank(), b = cbeg(c), e = cend(c);
        double mt = 0, mx = 0, my = 0, mz = 0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, b, e), [&](int k, double& a, double& bx, double& by, double& bz) {
            a += sm(k); bx += sm(k) * sx(k); by += sm(k) * sy(k); bz += sm(k) * sz(k);
        }, mt, mx, my, mz);
        const double zx = mx / mt, zy = my / mt, zz = mz / mt;
        double r2 = 0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, b, e), [&](int k, double& v) {
            double dx = sx(k) - zx, dy = sy(k) - zy, dz = sz(k) - zz, q = dx * dx + dy * dy + dz * dz;
            if (q > v) v = q;
        }, Kokkos::Max<double>(r2));
        Kokkos::single(Kokkos::PerTeam(t), [&]() { cz(c, 0) = zx; cz(c, 1) = zy; cz(c, 2) = zz; cz(c, 3) = Kokkos::sqrt(r2); });
    });
    const int nl = (int)hleaves.size();
    Kokkos::parallel_for("fmm_leaf_of", nl, KOKKOS_LAMBDA(int q) {
        int c = leaves(q);
        for (int k = cbeg(c); k < cend(c); k++) leaf_of(k) = c;
    });
    Kokkos::fence();
    double t1 = now();

    // ---- upward pass: P2M at leaves, M2M level by level ---------------------------------------------------------
    Kokkos::deep_copy(Kokkos::subview(M, Kokkos::make_pair(0, nc), Kokkos::ALL), 0.0);
    Kokkos::parallel_for("fmm_p2m", nl, KOKKOS_LAMBDA(int q) {
        const int c = leaves(q);
        kc Y[TRI_MAX];
        double* Mc = &M(c, 0);
        for (int k = cbeg(c); k < cend(c); k++) {
            reg_pos(sx(k) - cz(c, 0), sy(k) - cz(c, 1), sz(k) - cz(c, 2), p, Y);
            for (int nn = 0; nn <= p; nn++)
                for (int mm = 0; mm <= nn; mm++) addR(Mc, nn, mm, sm(k) * Kokkos::conj(Y[tri(nn, mm)]));
        }
    });
    const int n_levels = (int)level_start.size() - 1;
    for (int lev = n_levels - 2; lev >= 0; lev--) {
        const int c0 = level_start[lev], c1 = level_start[lev + 1];
        Kokkos::parallel_for("fmm_m2m", Kokkos::RangePolicy<>(c0, c1), KOKKOS_LAMBDA(int c) {
            if (cnch(c) == 0) return;
            double* Mp = &M(c, 0);
            kc Y[TRI_MAX];
            for (int q = 0; q < cnch(c); q++) {
                const int ch = cfirst(c) + q;
                reg_pos(cz(ch, 0) - cz(c, 0), cz(ch, 1) - cz(c, 1), cz(ch, 2) - cz(c, 2), p, Y);
                const double* Mc = &M(ch, 0);
                for (int nn = 0; nn <= p; nn++)
                    for (int mm = 0; mm <= nn; mm++) {
                        kc v(0.0);
                        for (int k = 0; k <= nn; k++)
                            for (int l = -k; l <= k; l++) {
                                int ml = mm - l;
                                if (ml > nn - k || -ml > nn - k) continue;
                                v += Kokkos::conj(getY(Y, k, l)) * getR(Mc, nn - k, ml);
                            }
                        addR(Mp, nn, mm, v);
                    }
            }
        });
    }
    Kokkos::fence();
    double t2 = now();

    // ---- interaction lists: dual walk on the host --------------------------------------------------------------
    auto hcz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cz);
    std::vector<std::vector<int>> far_src(nc), near_src(nc);
    const double h_soft = prm.kernel == 2 ? 2.8 * prm.eps : 0.0;
    const double min_sep = std::max(h_soft, prm.min_far_sep);
    long long n_m2l = 0, n_pp = 0;
    std::vector<std::pair<int, int>> stack{{0, 0}};
    while (!stack.empty()) {
        auto [a, b] = stack.back();
        stack.pop_back();
        if (a == b) {
            if (hnch[a] == 0) { near_src[a].push_back(a); n_pp += (long long)(hend[a] - hbeg[a]) * (hend[a] - hbeg[a] - 1); continue; }
            for (int i = 0; i < hnch[a]; i++)
                for (int j = i; j < hnch[a]; j++) stack.push_back({hfirst[a] + i, hfirst[a] + j});
            continue;
        }
        double dx = hcz(a, 0) - hcz(b, 0), dy = hcz(a, 1) - hcz(b, 1), dz = hcz(a, 2) - hcz(b, 2);
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz), ra = hcz(a, 3), rb = hcz(b, 3);
        if (ra + rb < prm.theta * dist && dist - ra - rb > min_sep) {
            far_src[a].push_back(b); far_src[b].push_back(a); n_m2l++;
            continue;
        }
        bool al = hnch[a] == 0, bl = hnch[b] == 0;
        if (al && bl) {
            near_src[a].push_back(b); near_src[b].push_back(a);
            n_pp += 2LL * (hend[a] - hbeg[a]) * (hend[b] - hbeg[b]);
            continue;
        }
        if (!al && (bl || ra >= rb)) { for (int i = 0; i < hnch[a]; i++) stack.push_back({hfirst[a] + i, b}); }
        else { for (int j = 0; j < hnch[b]; j++) stack.push_back({a, hfirst[b] + j}); }
    }
    std::vector<int> fp(nc + 1, 0), fi, np(nc + 1, 0), ni;
    for (int c = 0; c < nc; c++) {
        fp[c + 1] = fp[c] + (int)far_src[c].size();
        np[c + 1] = np[c] + (int)near_src[c].size();
    }
    fi.reserve(fp[nc]); ni.reserve(np[nc]);
    for (int c = 0; c < nc; c++) {
        fi.insert(fi.end(), far_src[c].begin(), far_src[c].end());
        ni.insert(ni.end(), near_src[c].begin(), near_src[c].end());
    }
    upload(S.far_ptr, fp, "fmm_fp"); upload(S.far_idx, fi, "fmm_fi");
    upload(S.near_ptr, np, "fmm_np"); upload(S.near_idx, ni, "fmm_ni");
    st.n_m2l = n_m2l; st.n_p2p_pairs = n_pp;
    auto far_ptr = S.far_ptr, far_idx = S.far_idx, near_ptr = S.near_ptr, near_idx = S.near_idx;
    double t3 = now();

    // ---- M2L: one team per target cell, cooperating on each interaction -----------------------------------------
    Kokkos::deep_copy(Kokkos::subview(L, Kokkos::make_pair(0, nc), Kokkos::ALL), 0.0);
    {
        auto fx90 = S.fx90, fxm90 = S.fxm90, bx90 = S.bx90, bxm90 = S.bxm90;
        auto fact = S.fact;
        auto pair_n = S.pair_n, pair_m = S.pair_m, tri_k = S.tri_k, tri_l = S.tri_l;
        const int NP = (int)pair_n.extent(0);
        using scratch = Kokkos::DefaultExecutionSpace::scratch_memory_space;
        using SV = Kokkos::View<double*, scratch, Kokkos::MemoryUnmanaged>;
        const size_t bytes = 4 * SV::shmem_size(NR) + SV::shmem_size(2 * p + 2);
        Kokkos::parallel_for("fmm_m2l", team_policy(nc, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(bytes)),
            KOKKOS_LAMBDA(const member& t) {
            const int c = t.league_rank();
            const int f0 = far_ptr(c), f1 = far_ptr(c + 1);
            if (f0 == f1) return;
            SV A(t.team_scratch(0), NR), B(t.team_scratch(0), NR), Lacc(t.team_scratch(0), NR), Tz(t.team_scratch(0), 2 * p + 2);
            SV Cw(t.team_scratch(0), NR);
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, NR), [&](int i) { Lacc(i) = 0.0; });
            for (int f = f0; f < f1; f++) {
                const int s = far_idx(f);
                const int lo = c < s ? c : s, hi = c < s ? s : c;
                const double Rx = cz(lo, 0) - cz(hi, 0), Ry = cz(lo, 1) - cz(hi, 1), Rz = cz(lo, 2) - cz(hi, 2);
                const double r = Kokkos::sqrt(Rx * Rx + Ry * Ry + Rz * Rz);
                double ct = Rz / r;
                ct = ct > 1 ? 1 : (ct < -1 ? -1 : ct);
                const double th = Kokkos::acos(ct), ph = Kokkos::atan2(Ry, Rx);
                const bool c_is_lo = (c == lo);
                Kokkos::parallel_for(Kokkos::TeamThreadRange(t, NR), [&](int i) { A(i) = M(s, i); });
                Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 2 * p + 2), [&](int j) {
                    Tz(j) = j <= p ? fact(j) * Kokkos::pow(1.0 / r, j + 1) : 0.0;
                });
                t.team_barrier();
                // forward rotation of the source multipole: rz(ph), rx(+90), rz(th), rx(-90)
                auto rz = [&](SV v, double alpha) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, NP), [&](int q) {
                        const int nn = pair_n(q), mm = pair_m(q);
                        const double cs = Kokkos::cos(mm * alpha), sn = Kokkos::sin(mm * alpha);
                        double* vn = &v(nn * nn);
                        const double a = vn[2 * mm - 1], b = vn[2 * mm];
                        vn[2 * mm - 1] = a * cs - b * sn; vn[2 * mm] = a * sn + b * cs;
                    });
                    t.team_barrier();
                };
                auto rx = [&](SV in, SV out, const DevCSR& R) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, NR), [&](int row) {
                        double acc = 0;
                        for (int k = R.ptr(row); k < R.ptr(row + 1); k++) acc += R.val(k) * in(R.col(k));
                        out(row) = acc;
                    });
                    t.team_barrier();
                };
                rz(A, ph); rx(A, B, fx90); rz(B, th); rx(B, A, fxm90);
                // translation along z into B
                Kokkos::parallel_for(Kokkos::TeamThreadRange(t, NT), [&](int q) {
                    const int k = tri_k(q), l = tri_l(q);
                    double sre = 0, sim = 0;
                    for (int nn = l; nn + k <= p; nn++) {
                        const double tt = Tz(nn + k);
                        const double re = l ? A(nn * nn + 2 * l - 1) : A(nn * nn), im = l ? A(nn * nn + 2 * l) : 0.0;
                        const double sg = (!c_is_lo && (nn & 1)) ? -1.0 : 1.0;
                        sre += sg * re * tt; sim += sg * im * tt;
                    }
                    const double fs = c_is_lo ? (((k + l) & 1) ? -1.0 : 1.0) : ((l & 1) ? -1.0 : 1.0);
                    if (l == 0) B(k * k) = fs * sre;
                    else { B(k * k + 2 * l - 1) = fs * sre; B(k * k + 2 * l) = -fs * sim; }
                });
                t.team_barrier();
                // back rotation of the local expansion: rx^T(-90), rz(th), rx^T(+90), rz(ph)
                rx(B, Cw, bxm90); rz(Cw, th); rx(Cw, B, bx90); rz(B, ph);
                Kokkos::parallel_for(Kokkos::TeamThreadRange(t, NR), [&](int i) { Lacc(i) += B(i); });
                t.team_barrier();
            }
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, NR), [&](int i) { L(c, i) = Lacc(i); });
        });
    }
    Kokkos::fence();
    double t4 = now();

    // ---- P2P: one team per target leaf, directed --------------------------------------------------------------
    auto ax = S.ax, ay = S.ay, az = S.az;
    {
        const double G = prm.G, h = h_soft;
        const int ktype = prm.kernel;
        Kokkos::parallel_for("fmm_p2p", team_policy(nl, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t) {
            const int c = leaves(t.league_rank());
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, cbeg(c), cend(c)), [&](int i) {
                double sxa = 0, sya = 0, sza = 0, cx = 0, cy = 0, cz2 = 0;
                const double xi = sx(i), yi = sy(i), zi = sz(i);
                const int si = ssid(i);
                for (int q = near_ptr(c); q < near_ptr(c + 1); q++) {
                    const int sc = near_idx(q);
                    double bx = 0, by = 0, bz = 0;
                    for (int j = cbeg(sc); j < cend(sc); j++) {
                        const bool w = (j != i) & !((si >= 0) & (ssid(j) == si));
                        double dx = sx(j) - xi, dy = sy(j) - yi, dz = sz(j) - zi;
                        double r2 = w ? dx * dx + dy * dy + dz * dz : 1.0;
                        double g = (w ? sm(j) : 0.0) * kernel_g(r2, ktype, h);
                        bx += g * dx; by += g * dy; bz += g * dz;
                    }
                    // compensated sum over source leaves
                    double yv = bx - cx, tv = sxa + yv; cx = (tv - sxa) - yv; sxa = tv;
                    yv = by - cy; tv = sya + yv; cy = (tv - sya) - yv; sya = tv;
                    yv = bz - cz2; tv = sza + yv; cz2 = (tv - sza) - yv; sza = tv;
                }
                ax(i) = G * (sxa - cx); ay(i) = G * (sya - cy); az(i) = G * (sza - cz2);
            });
        });
    }
    Kokkos::fence();
    double t5 = now();

    // ---- downward pass: L2L level by level, L2P per particle, scatter -------------------------------------------
    for (int lev = 1; lev < n_levels; lev++) {
        const int c0 = level_start[lev], c1 = level_start[lev + 1];
        Kokkos::parallel_for("fmm_l2l", Kokkos::RangePolicy<>(c0, c1), KOKKOS_LAMBDA(int c) {
            const int pa = cpar(c);
            kc Y[TRI_MAX];
            reg_pos(cz(c, 0) - cz(pa, 0), cz(c, 1) - cz(pa, 1), cz(c, 2) - cz(pa, 2), p, Y);
            const double* Lp = &L(pa, 0);
            double* Lc = &L(c, 0);
            for (int qn = 0; qn <= p; qn++)
                for (int r = 0; r <= qn; r++) {
                    kc v(0.0);
                    for (int j = 0; j + qn <= p; j++)
                        for (int i = -j; i <= j; i++) {
                            int ri = r + i;
                            if (ri > qn + j || -ri > qn + j) continue;
                            v += Kokkos::conj(getY(Y, j, i)) * getR(Lp, qn + j, ri);
                        }
                    addR(Lc, qn, r, v);
                }
        });
    }
    {
        const double G = prm.G;
        Kokkos::parallel_for("fmm_l2p", n, KOKKOS_LAMBDA(int k) {
            const int c = leaf_of(k);
            kc Y[TRI_MAX];
            reg_pos(sx(k) - cz(c, 0), sy(k) - cz(c, 1), sz(k) - cz(c, 2), p, Y);
            const double* Lc = &L(c, 0);
            kc gx(0.0), gy(0.0), gz(0.0);
            for (int kk = 1; kk <= p; kk++)
                for (int l = -kk; l <= kk; l++) {
                    const kc Lk = getR(Lc, kk, l);
                    gz += Lk * Kokkos::conj(getY(Y, kk - 1, l));
                    const kc ym = Kokkos::conj(getY(Y, kk - 1, l - 1)), yp = Kokkos::conj(getY(Y, kk - 1, l + 1));
                    gx += Lk * 0.5 * (ym - yp);
                    gy += Lk * kc(0.0, -0.5) * (ym + yp);
                }
            const int i = i0 + perm(k);
            oax(i) = ax(k) + G * gx.real(); oay(i) = ay(k) + G * gy.real(); oaz(i) = az(k) + G * gz.real();
        });
    }
    Kokkos::fence();
    double t6 = now();
    st.t_tree = t1 - t0; st.t_up = t2 - t1; st.t_lists = t3 - t2; st.t_m2l = t4 - t3; st.t_p2p = t5 - t4; st.t_down = t6 - t5;
    if (stats) *stats = st;
}
