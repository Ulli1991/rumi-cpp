#pragma once

// Exact tree searches for the HHS time steps and the neighbour pairs (plan M7.4). Forces stay direct.
//
// FROST-CLUSTERS II (Rantala et al. 2025, arXiv:2506.04330, sec. 2.1) added an octree to BIFROST for the time-step and
// neighbour searches. PeTar (Wang et al. 2020, arXiv:2006.16560, sec. 2.3) finds neighbour candidates with a
// particle tree inside a radius and then applies the Kepler pericentre test pair by pair; GADGET-4 (Springel et al.
// 2021, arXiv:2010.03567, sec. 5) keeps min/max coordinates and velocities per node for conservative search bounds.
// Here the same per-node bounds drive a branch-and-bound minimum search (Friedman, Bentley & Finkel 1977; Gray &
// Moore 2000, "N-body problems in statistical learning"): a node is skipped only when a rigorous lower bound of the
// pair step over all its members is >= the running minimum, and leaves evaluate exactly the expression of DtK
// (n2_kernels.h). The minimum is therefore bit-identical to the O(N^2) kernel (min is order independent).
//
// Tree: the source points are sorted by a Morton key; leaves are runs of LEAF consecutive points; an implicit
// binary tree (heap order, node 1 the root, leaves P..2P-1) is built bottom-up over the leaves. Each node keeps the
// box of the positions, the box of the pair-step velocities and the min/max mass of its members.
//
// Lower bound of the pair step (core/constants.h symmetrized_timestep; pair_dt = min(eta_ff t_ff', eta_fb t_fb')).
// With d = c t eta, eta t' = eta t f(d), f(d) = 1 + d / (1 + s), s = sqrt(1 + d^2). f is increasing (f' = 1/(s(1+s))),
// so |d| <= D gives f(d) >= f(-D) = (1 + 1/(s + D)) / (1 + s) (stable form). r is the softened distance, v = |dv|,
// v_e = sqrt(v^2 + EPS), m = m_i + m_j, and |r.dv| <= r v:
//   ff: t = r^1.5 / sqrt(G m), |d| <= x = 1.5 eta v t / r.  eta t f(-x) = (r / (1.5 v)) x f(-x), and x f(-x) increases
//       with x ((x f(-x))' = [(1 + 1/(s+x))^2 - x^2/(s(s+x))] / (1+s)^2 > 0), so the bound increases with r and decreases
//       with v and m: take r_min, v_max, m_max.
//   fb: t = r / v_e, |d| <= D = eta (1 + G m / (r v_e^2)). H(v_e) = eta (r / v_e) f(-D) increases with r, decreases with
//       m, and is unimodal in v_e: with u = 1/v_e^2, d ln H / du = 1/(2u) - eta G m / r * f'(-D)/f(-D) vanishes where
//       g(w) = 2(w - eta) - s - s/(s + w) = 0 (w = D), and g' = 2 - w/s + 1/(s (s+w)^2) > 0, a single root (the maximum,
//       H -> 0 at both ends). Over [v_min, v_max] the minimum is at an end: min(H(v_min), H(v_max)).
// The computed t' = t + c t^2 eta / (u + 1) carries an absolute round-off of ~20 eps t (cancellation for d << 0), so
// 1e-13 eta t_max is subtracted, and the result is scaled by (1 - 1e-6) for the relative round-off of t and the bound.
//
// Tidal-sampling pair step (FLYBY_CM_EPS, n2_kernels.h tidal_sampling_dt): t = max(min(k_i / m_j, k_j / m_i) r^3,
// max(f_i, f_j)) enters the minimum next to pair_dt. Over a node (k_j >= k_min, m_j <= m_max, r >= r_min, f_j >= 0):
// t >= max(min(k_i / m_max, k_min / m_i) r_min^3, f_i). fmin, fmax and the f floor are exact; the product and r^3
// carry a few ulp, covered by the factor (1 - 1e-6). The node bound is the smaller of the two bounds, so a pruned
// node holds no pair whose min(pair_dt, t) is below the running minimum, and leaves evaluate DtK's expression: the
// minimum stays bit-identical to the O(N^2) kernel with the tidal step too.
//
// Low relative velocities: H -> r^2 v / (G m) -> 0 for v -> 0 (an approaching pair at rest has a vanishing fly-by step),
// and a spatial node's velocity box almost always contains the sink velocity, so v_min = 0 would block all pruning.
// Each sink therefore picks a speed V_c: members with |dv| < V_c are found in a second tree over the velocities (its
// nodes' velocity boxes are small), and the position tree assumes v >= V_c. Every pair is covered by one of the two
// searches, so the minimum stays exact for any V_c; V_c only moves work between them.

#include "n2_kernels.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <cmath>
#include <cstdlib>
#include <cstdint>

namespace dtt {

#ifndef DTT_LEAF
#define DTT_LEAF 8
#endif
constexpr int LEAF = DTT_LEAF;    // points per leaf
constexpr int NB = 15;            // node record: key box lo[3] hi[3], position box lo[3] hi[3] ... see below
constexpr int STACK = 64;
// node record layout: [0..2] position lo, [3..5] position hi, [6..8] velocity lo, [9..11] velocity hi, [12] m max,
// [13] m min, [14] k min of the tidal-sampling step (1e300 without it) (empty node: lo = +inf, hi = -inf)

KOKKOS_INLINE_FUNCTION uint64_t spread21(uint64_t v) {
    v &= 0x1fffffULL;
    v = (v | v << 32) & 0x1f00000000ffffULL;
    v = (v | v << 16) & 0x1f0000ff0000ffULL;
    v = (v | v << 8) & 0x100f00f00f00f00fULL;
    v = (v | v << 4) & 0x10c30c30c30c30c3ULL;
    v = (v | v << 2) & 0x1249249249249249ULL;
    return v;
}

// Squared distance from a point to a box, and to the box's farthest corner
KOKKOS_INLINE_FUNCTION void box_dist2(const double* lo, const double* hi, double px, double py, double pz, double& dmin2,
                                      double& dmax2) {
    const double p[3] = {px, py, pz};
    dmin2 = 0.0; dmax2 = 0.0;
    for (int d = 0; d < 3; d++) {
        const double a = lo[d] - p[d], b = p[d] - hi[d];
        const double g = Kokkos::fmax(Kokkos::fmax(a, b), 0.0);
        const double f = Kokkos::fmax(Kokkos::fabs(a), Kokkos::fabs(b));
        dmin2 += g * g; dmax2 += f * f;
    }
}

// f(-x) = 1 - x / (1 + sqrt(1 + x^2)) for x >= 0, without cancellation
KOKKOS_INLINE_FUNCTION double fneg(double x) {
    const double s = Kokkos::sqrt(1.0 + x * x);
    return (1.0 + 1.0 / (s + x)) / (1.0 + s);
}

// Lower bound of pair_dt over pairs with softened distance in [sqrt(r2lo + soft2), sqrt(r2hi + soft2)], mass sum in
// [mlo, mhi] and relative speed in [vlo, vhi] (derivation at the top of this file); 0 when nothing can be said
KOKKOS_INLINE_FUNCTION double dt_lower_bound(double r2lo, double r2hi, double soft2, double mlo, double mhi, double vlo,
                                             double vhi, double G, double eta_ff, double eta_fb) {
    const double rs = Kokkos::sqrt(r2lo + soft2);
    if (!(rs > 0.0) || !(vhi >= vlo)) return 0.0;
    const double rh = Kokkos::sqrt(r2hi + soft2);
    const double gmh = G * mhi + EPS, gml = G * mlo + EPS;
    const double vle = Kokkos::sqrt(vlo * vlo + EPS), vhe = Kokkos::sqrt(vhi * vhi + EPS);
    const double T = rs * Kokkos::sqrt(rs) / Kokkos::sqrt(gmh);
    double lb_ff = eta_ff * T * fneg(1.5 * eta_ff * vhi * T / rs);
    lb_ff -= 1e-13 * eta_ff * rh * Kokkos::sqrt(rh) / Kokkos::sqrt(gml);
    const double c = eta_fb * gmh / rs;
    const double h_lo = eta_fb * rs / vle * fneg(eta_fb + c / (vle * vle));
    const double h_hi = eta_fb * rs / vhe * fneg(eta_fb + c / (vhe * vhe));
    const double lb_fb = Kokkos::fmin(h_lo, h_hi) - 1e-13 * eta_fb * rh / vle;
    return (1.0 - 1e-6) * Kokkos::fmin(lb_ff, lb_fb);
}

// Lower bound of the tidal-sampling step of sink (k_i, m_i, f_i) over sources with k >= kmin, m <= mmax at softened
// distance >= sqrt(r2lo + soft2) (derivation at the top of this file)
KOKKOS_INLINE_FUNCTION double tidal_lower_bound(double r2lo, double soft2, double ki, double mi, double fi, double kmin,
                                                double mmax) {
    const double r2 = r2lo + soft2;
    return Kokkos::fmax((1.0 - 1e-6) * Kokkos::fmin(ki / mmax, kmin / mi) * r2 * Kokkos::sqrt(r2), fi);
}

// Source set of one tree: points sorted by the Morton key of (kx, ky, kz) = positions (position tree) or pair
// velocities (velocity tree). Positions, velocities and masses are copied in sorted order (bitwise copies).
template <class Exec>
struct PointTree {
    using Mem = typename Exec::memory_space;
    int n = 0, P = 1, nlev = 0, j0 = 0;   // j0: first particle index of the source range
    double vmean[3] = {0, 0, 0}, vsig = 0;   // mean and rms deviation of the velocities (vx, vy, vz)
    Kokkos::View<double*, Mem> px, py, pz, vx, vy, vz, m, k, kf;   // k, kf: tidal-sampling step (has_k)
    bool has_k = false;
    Kokkos::View<int*, Mem> idx;          // sorted position -> particle index
    Kokkos::View<double**, Kokkos::LayoutRight, Mem> node;
    Kokkos::View<uint64_t*, Mem> key;
    Kokkos::View<int*, Mem> perm;

    void reserve(int nn) {
        if ((int)px.extent(0) < nn) {
            const int c = nn + nn / 4;
            for (auto* v : {&px, &py, &pz, &vx, &vy, &vz, &m, &k, &kf}) *v = Kokkos::View<double*, Mem>("dtt_pt", c);
            idx = Kokkos::View<int*, Mem>("dtt_idx", c);
            perm = Kokkos::View<int*, Mem>("dtt_perm", c);
            key = Kokkos::View<uint64_t*, Mem>("dtt_key", c);
        }
        if ((int)node.extent(0) < 2 * P) node = Kokkos::View<double**, Kokkos::LayoutRight, Mem>("dtt_node", 2 * P, NB);
    }

    // Points j0 .. j0 + nj - 1 of the particle arrays (pointers accessible from Exec); by_velocity: key on (vx, vy, vz).
    // vclass (position tree of the time-step search): the top 3 key bits are a speed class of u = |v - <v>|, class 0
    // for u < 3 sigma_core (sigma_core: rms of the particles within 3 rms), class k for u in [3 sigma_core 2^(k-1),
    // 3 sigma_core 2^k), above a 60-bit position Morton key. Fast particles (binary components with their orbital
    // velocities) then do not widen the velocity boxes of the nodes of field stars (8192-body cluster with 10 %
    // binaries, leaf 8: 2.6x fewer pair evaluations than without classes; finer classes or velocity octants lose
    // spatial resolution: the sink falls inside the boxes of many sparse leaves, where no bound can prune).
    // K, KF (optional): the tidal-sampling arrays (PairVelocities::k, kf), copied in sorted order with a per-node k min.
    void build(int j0, int nj, const double* X, const double* Y, const double* Z, const double* VX, const double* VY,
               const double* VZ, const double* M, bool by_velocity, bool vclass = false, const double* K = nullptr,
               const double* KF = nullptr) {
        n = nj; this->j0 = j0;
        has_k = K != nullptr;
        const int nleaf = (n + LEAF - 1) / LEAF;
        P = 1; nlev = 0;
        while (P < nleaf) { P *= 2; nlev++; }
        reserve(n);
        const double* KX = by_velocity ? VX : X;
        const double* KY = by_velocity ? VY : Y;
        const double* KZ = by_velocity ? VZ : Z;
        double lo[3], hi[3];
        Kokkos::parallel_reduce("dtt_bbox", Kokkos::RangePolicy<Exec>(0, n),
            KOKKOS_LAMBDA(int k, double& a0, double& a1, double& a2, double& b0, double& b1, double& b2) {
                const int j = j0 + k;
                a0 = Kokkos::fmin(a0, KX[j]); a1 = Kokkos::fmin(a1, KY[j]); a2 = Kokkos::fmin(a2, KZ[j]);
                b0 = Kokkos::fmax(b0, KX[j]); b1 = Kokkos::fmax(b1, KY[j]); b2 = Kokkos::fmax(b2, KZ[j]);
            }, Kokkos::Min<double>(lo[0]), Kokkos::Min<double>(lo[1]), Kokkos::Min<double>(lo[2]),
               Kokkos::Max<double>(hi[0]), Kokkos::Max<double>(hi[1]), Kokkos::Max<double>(hi[2]));
        double mv[3] = {0, 0, 0}, s2 = 0, s2c = 0, ncore = 0;
        if (vclass) {
            Kokkos::parallel_reduce("dtt_vmean", Kokkos::RangePolicy<Exec>(0, n),
                KOKKOS_LAMBDA(int k, double& a, double& b, double& c, double& q) {
                    const int j = j0 + k;
                    a += VX[j]; b += VY[j]; c += VZ[j]; q += VX[j] * VX[j] + VY[j] * VY[j] + VZ[j] * VZ[j];
                }, mv[0], mv[1], mv[2], s2);
            for (double& v : mv) v /= n;
            s2 = std::fmax(0.0, s2 / n - (mv[0] * mv[0] + mv[1] * mv[1] + mv[2] * mv[2]));
            const double cut2 = 9.0 * s2, c0 = mv[0], c1 = mv[1], c2 = mv[2];
            Kokkos::parallel_reduce("dtt_vcore", Kokkos::RangePolicy<Exec>(0, n),
                KOKKOS_LAMBDA(int k, double& q, double& c) {
                    const int j = j0 + k;
                    const double ux = VX[j] - c0, uy = VY[j] - c1, uz = VZ[j] - c2, u2 = ux * ux + uy * uy + uz * uz;
                    if (u2 <= cut2) { q += u2; c += 1.0; }
                }, s2c, ncore);
        }
        for (int d = 0; d < 3; d++) vmean[d] = mv[d];
        vsig = std::sqrt(s2);
        const double m0 = mv[0], m1 = mv[1], m2 = mv[2];
        const double inv_core = s2c > 0 ? 1.0 / (3.0 * std::sqrt(s2c / ncore)) : 0.0;
        const double kmax = vclass ? 1048575.0 : 2097151.0;
        double sc[3];
        for (int d = 0; d < 3; d++) sc[d] = hi[d] > lo[d] ? kmax / (hi[d] - lo[d]) : 0.0;
        const double l0 = lo[0], l1 = lo[1], l2 = lo[2], s0 = sc[0], s1 = sc[1], sz = sc[2];
        auto key_ = key; auto perm_ = perm;
        Kokkos::parallel_for("dtt_keys", Kokkos::RangePolicy<Exec>(0, n), KOKKOS_LAMBDA(int k) {
            const int j = j0 + k;
            const uint64_t a = (uint64_t)Kokkos::fmin(kmax, (KX[j] - l0) * s0);
            const uint64_t b = (uint64_t)Kokkos::fmin(kmax, (KY[j] - l1) * s1);
            const uint64_t c = (uint64_t)Kokkos::fmin(kmax, (KZ[j] - l2) * sz);
            uint64_t kk = spread21(a) | spread21(b) << 1 | spread21(c) << 2;
            if (vclass) {
                const double ux = VX[j] - m0, uy = VY[j] - m1, uz = VZ[j] - m2;
                const double q = Kokkos::sqrt(ux * ux + uy * uy + uz * uz) * inv_core;
                uint64_t cls = 0;
                for (double t = 1.0; cls < 7 && q >= t; t *= 2.0) cls++;
                kk |= cls << 60;
            }
            key_(k) = kk;
            perm_(k) = k;
        });
        auto ks = Kokkos::subview(key, Kokkos::make_pair(0, n));
        auto ps = Kokkos::subview(perm, Kokkos::make_pair(0, n));
        Kokkos::Experimental::sort_by_key(Exec(), ks, ps);
        auto px_ = px, py_ = py, pz_ = pz, vx_ = vx, vy_ = vy, vz_ = vz, m_ = m, k_ = k, kf_ = kf;
        auto idx_ = idx;
        Kokkos::parallel_for("dtt_gather", Kokkos::RangePolicy<Exec>(0, n), KOKKOS_LAMBDA(int q) {
            const int j = j0 + perm_(q);
            idx_(q) = j;
            px_(q) = X[j]; py_(q) = Y[j]; pz_(q) = Z[j]; vx_(q) = VX[j]; vy_(q) = VY[j]; vz_(q) = VZ[j]; m_(q) = M[j];
            if (K) { k_(q) = K[j]; kf_(q) = KF[j]; }
        });
        auto nd = node;
        const int nn = n, PP = P;
        Kokkos::parallel_for("dtt_leaves", Kokkos::RangePolicy<Exec>(0, P), KOKKOS_LAMBDA(int l) {
            double r[NB];
            for (int d = 0; d < 3; d++) { r[d] = r[6 + d] = 1e300; r[3 + d] = r[9 + d] = -1e300; }
            r[12] = -1e300; r[13] = 1e300; r[14] = 1e300;
            const int k1 = Kokkos::min(nn, (l + 1) * LEAF);
            for (int k = l * LEAF; k < k1; k++) {
                if (K) r[14] = Kokkos::fmin(r[14], k_(k));
                const double q[6] = {px_(k), py_(k), pz_(k), vx_(k), vy_(k), vz_(k)};
                for (int d = 0; d < 3; d++) {
                    r[d] = Kokkos::fmin(r[d], q[d]); r[3 + d] = Kokkos::fmax(r[3 + d], q[d]);
                    r[6 + d] = Kokkos::fmin(r[6 + d], q[3 + d]); r[9 + d] = Kokkos::fmax(r[9 + d], q[3 + d]);
                }
                r[12] = Kokkos::fmax(r[12], m_(k)); r[13] = Kokkos::fmin(r[13], m_(k));
            }
            for (int a = 0; a < NB; a++) nd(PP + l, a) = r[a];
        });
        for (int lev = nlev - 1; lev >= 0; lev--) {
            const int b = 1 << lev;
            Kokkos::parallel_for("dtt_up", Kokkos::RangePolicy<Exec>(b, 2 * b), KOKKOS_LAMBDA(int c) {
                for (int d = 0; d < 3; d++) {
                    nd(c, d) = Kokkos::fmin(nd(2 * c, d), nd(2 * c + 1, d));
                    nd(c, 3 + d) = Kokkos::fmax(nd(2 * c, 3 + d), nd(2 * c + 1, 3 + d));
                    nd(c, 6 + d) = Kokkos::fmin(nd(2 * c, 6 + d), nd(2 * c + 1, 6 + d));
                    nd(c, 9 + d) = Kokkos::fmax(nd(2 * c, 9 + d), nd(2 * c + 1, 9 + d));
                }
                nd(c, 12) = Kokkos::fmax(nd(2 * c, 12), nd(2 * c + 1, 12));
                nd(c, 13) = Kokkos::fmin(nd(2 * c, 13), nd(2 * c + 1, 13));
                nd(c, 14) = Kokkos::fmin(nd(2 * c, 14), nd(2 * c + 1, 14));
            });
        }
    }
};

// Parameters of the time-step search (the DtK constants)
struct DtParams {
    double G, soft2, eta_ff, eta_fb;
    double vc_fac;   // V_c = min(sigma, vc_fac * 2 G m mn / R_c^2): split between the velocity and position searches
    const double* k = nullptr;    // tidal-sampling step of the sinks (PairVelocities::k, kf; the trees must carry it)
    const double* kf = nullptr;
};

// Minimum pair step of each sink i in [i0, i0 + ni) over the sources of the trees, written to dt(i). sid: subsystem
// ids (nullptr: no exclusion). X..CVZ, M: particle arrays accessible from Exec. Counts visited pairs into *npairs if set.
template <class Exec>
void min_pair_dt(const PointTree<Exec>& T, const PointTree<Exec>& V, int i0, int ni, const double* X, const double* Y,
                 const double* Z, const double* CVX, const double* CVY, const double* CVZ, const double* M, const int* sid,
                 double* dt, const DtParams& prm, long long* npairs = nullptr, long long* npairsB = nullptr) {
    auto tn = T.node; auto tpx = T.px, tpy = T.py, tpz = T.pz, tvx = T.vx, tvy = T.vy, tvz = T.vz, tm = T.m;
    auto tidx = T.idx;
    auto tk = T.k, tkf = T.kf, vk = V.k, vkf = V.kf;
    const double* K = prm.k;
    const double* KF = prm.kf;
    if (K && !(T.has_k && V.has_k)) Kokkos::abort("dt_tree: tidal-sampling step without the trees' k arrays");
    auto vn = V.node; auto vpx = V.px, vpy = V.py, vpz = V.pz, vvx = V.vx, vvy = V.vy, vvz = V.vz, vm = V.m;
    auto vidx = V.idx;
    const int TP = T.P, VP = V.P, Tn = T.n, Vn = V.n;
    const bool same = i0 == T.j0 && ni == T.n;
    const double G = prm.G, soft2 = prm.soft2, eff = prm.eta_ff, efb = prm.eta_fb, vcf = prm.vc_fac;
    const double vmx = T.vmean[0], vmy = T.vmean[1], vmz = T.vmean[2], vsig = T.vsig;
    long long cnt = 0, cntB = 0;
    Kokkos::parallel_reduce("dtt_min_dt", Kokkos::RangePolicy<Exec, Kokkos::Schedule<Kokkos::Dynamic>>(0, ni),
        KOKKOS_LAMBDA(int ii, long long& npair, long long& npairB) {
        const int i = same ? tidx(ii) : i0 + ii;   // sinks in Morton order when they are the sources (coherent walks)
        const double xi = X[i], yi = Y[i], zi = Z[i], ui = CVX[i], vi = CVY[i], wi = CVZ[i], mi = M[i];
        const int si = sid ? sid[i] : -1;
        const double ki = K ? K[i] : 1e30, fi = K ? KF[i] : 0.0;
        double mn = 1e30;
        // leaf members k (sorted arrays of one tree): the pair expression of DtK
        auto eval = [&](const double* qx, const double* qy, const double* qz, const double* qu, const double* qv,
                        const double* qw, const double* qm, const double* qk, const double* qkf, const int* qi, int k0,
                        int k1) {
            for (int k = k0; k < k1; k++) {
                const int j = qi[k];
                const bool w = pair_weight(i, j, si, sid ? sid[j] : -1);
                if (!w) continue;
                double dx = qx[k] - xi, dy = qy[k] - yi, dz = qz[k] - zi;
                double r2s = dx * dx + dy * dy + dz * dz + soft2;
                double ir = fast_rsqrt(r2s);
                double t = pair_dt_below(mn, r2s, ir, dx, dy, dz, qu[k] - ui, qv[k] - vi, qw[k] - wi, G, mi + qm[k], eff, efb);
                if (K) t = Kokkos::fmin(t, tidal_sampling_dt(ki, qk[k], mi, qm[k], r2s, fi, qkf[k]));
                mn = Kokkos::fmin(mn, t);
            }
            npair += k1 - k0;
        };
        // 1. the leaf of the position tree nearest to the sink: a first estimate of the minimum
        int c = 1;
        while (c < TP) {
            double a2, b2, f;
            box_dist2(&tn(2 * c, 0), &tn(2 * c, 3), xi, yi, zi, a2, f);
            box_dist2(&tn(2 * c + 1, 0), &tn(2 * c + 1, 3), xi, yi, zi, b2, f);
            c = (b2 < a2) ? 2 * c + 1 : 2 * c;
        }
        const int leaf0 = c;
        {
            const int l = leaf0 - TP;
            eval(tpx.data(), tpy.data(), tpz.data(), tvx.data(), tvy.data(), tvz.data(), tm.data(), tk.data(), tkf.data(),
                 tidx.data(), l * LEAF, Kokkos::min(Tn, (l + 1) * LEAF));
        }
        // 2. V_c: a node of slow particles at distance r is pruned by the fly-by bound at its largest relative speed
        // (~ |v_i - <v>| + 2 sigma) beyond R_c = mn (2 + eta) v / eta, and by the low-speed end only if
        // H(R_c, V_c) ~ R_c^2 V_c / (G m) >= mn, i.e. V_c ~ 2 G m mn / R_c^2, times vcf (30: fewest pair evaluations on the
        // 8192-body cluster with binaries, 0.03-1000 tried); capped at the rms speed. Any V_c keeps the result exact.
        const double ux = ui - vmx, uy = vi - vmy, uz = wi - vmz;
        const double vtyp = Kokkos::sqrt(ux * ux + uy * uy + uz * uz) + 2.0 * vsig;
        double Vc = vsig;
        if (mn < 1e29 && vtyp > 0.0) {
            const double rc = mn * (2.0 + efb) * vtyp / efb;
            Vc = Kokkos::fmin(Vc, vcf * 2.0 * G * (mi + tn(1, 12)) * mn / (rc * rc));
        }
        const double VcB = Vc * (1.0 + 1e-9);
        int stack[STACK];
        // 3. velocity tree: members with |dv| < V_c
        if (Vc > 0.0 && Vn > 0) {
            int sp = 0;
            stack[sp++] = 1;
            while (sp > 0) {
                const int c2 = stack[--sp];
                if (vn(c2, 12) < 0.0) continue;   // empty
                double v2lo, v2hi, r2lo, r2hi;
                box_dist2(&vn(c2, 6), &vn(c2, 9), ui, vi, wi, v2lo, v2hi);
                const double vlo = Kokkos::sqrt(v2lo);
                if (!(vlo < VcB)) continue;
                box_dist2(&vn(c2, 0), &vn(c2, 3), xi, yi, zi, r2lo, r2hi);
                double lb = dt_lower_bound(r2lo, r2hi, soft2, mi + vn(c2, 13), mi + vn(c2, 12), vlo,
                                           Kokkos::fmin(Kokkos::sqrt(v2hi), VcB), G, eff, efb);
                if (K) lb = Kokkos::fmin(lb, tidal_lower_bound(r2lo, soft2, ki, mi, fi, vn(c2, 14), vn(c2, 12)));
                if (lb >= mn) continue;
                if (c2 >= VP) {
                    const int l = c2 - VP;
                    eval(vpx.data(), vpy.data(), vpz.data(), vvx.data(), vvy.data(), vvz.data(), vm.data(), vk.data(),
                         vkf.data(), vidx.data(), l * LEAF, Kokkos::min(Vn, (l + 1) * LEAF));
                    npairB += Kokkos::min(Vn, (l + 1) * LEAF) - l * LEAF;
                } else if (sp + 2 <= STACK) {
                    stack[sp++] = 2 * c2; stack[sp++] = 2 * c2 + 1;
                } else {
                    Kokkos::abort("dt_tree: stack overflow");
                }
            }
        }
        // 4. position tree with |dv| >= V_c, nearer child first
        {
            int sp = 0;
            stack[sp++] = 1;
            while (sp > 0) {
                const int c2 = stack[--sp];
                if (c2 == leaf0 || tn(c2, 12) < 0.0) continue;
                double v2lo, v2hi, r2lo, r2hi;
                box_dist2(&tn(c2, 0), &tn(c2, 3), xi, yi, zi, r2lo, r2hi);
                box_dist2(&tn(c2, 6), &tn(c2, 9), ui, vi, wi, v2lo, v2hi);
                const double vlo = Kokkos::fmax(Kokkos::sqrt(v2lo), Vc), vhi = Kokkos::sqrt(v2hi);
                if (vlo > vhi) continue;   // all members slower than V_c: found by the velocity tree
                double lb = dt_lower_bound(r2lo, r2hi, soft2, mi + tn(c2, 13), mi + tn(c2, 12), vlo, vhi, G, eff, efb);
                if (K) lb = Kokkos::fmin(lb, tidal_lower_bound(r2lo, soft2, ki, mi, fi, tn(c2, 14), tn(c2, 12)));
                if (lb >= mn) continue;
                if (c2 >= TP) {
                    const int l = c2 - TP;
                    eval(tpx.data(), tpy.data(), tpz.data(), tvx.data(), tvy.data(), tvz.data(), tm.data(), tk.data(),
                         tkf.data(), tidx.data(), l * LEAF, Kokkos::min(Tn, (l + 1) * LEAF));
                } else if (sp + 2 <= STACK) {
                    double a2, b2, f;
                    box_dist2(&tn(2 * c2, 0), &tn(2 * c2, 3), xi, yi, zi, a2, f);
                    box_dist2(&tn(2 * c2 + 1, 0), &tn(2 * c2 + 1, 3), xi, yi, zi, b2, f);
                    if (a2 <= b2) { stack[sp++] = 2 * c2 + 1; stack[sp++] = 2 * c2; }
                    else { stack[sp++] = 2 * c2; stack[sp++] = 2 * c2 + 1; }
                } else {
                    Kokkos::abort("dt_tree: stack overflow");
                }
            }
        }
        dt[i] = mn;
    }, cnt, cntB);
    if (npairs) *npairs += cnt;
    if (npairsB) *npairsB += cntB;
}

// Neighbour pairs (i < j, array indices) of the sinks [i0, i0 + ni) within the sources of T with r2 < r_ngb_max^2 and
// the criteria of NgbK (ngb_pair_test), appended through the atomic counter cnt as NgbK does
struct NgbOut {
    int *pi, *pj, *pb, *cnt;
    double *pa, *pe, *pE;
    int max_ngb;
};

template <class Exec>
void ngb_pairs(const PointTree<Exec>& T, int i0, int ni, const double* X, const double* Y, const double* Z,
               const double* VX, const double* VY, const double* VZ, const double* M, double G, double r_ngb,
               double r_ngb_max_sq, double pivot_dt, const NgbOut& o) {
    auto tn = T.node; auto tpx = T.px, tpy = T.py, tpz = T.pz, tvx = T.vx, tvy = T.vy, tvz = T.vz, tm = T.m;
    auto tidx = T.idx;
    const int TP = T.P, Tn = T.n;
    const bool same = i0 == T.j0 && ni == T.n;
    const double prune2 = r_ngb_max_sq * (1.0 + 1e-12);
    Kokkos::parallel_for("dtt_ngb", Kokkos::RangePolicy<Exec, Kokkos::Schedule<Kokkos::Dynamic>>(0, ni),
        KOKKOS_LAMBDA(int ii) {
        const int i = same ? tidx(ii) : i0 + ii;
        const double xi = X[i], yi = Y[i], zi = Z[i], ui = VX[i], vi = VY[i], wi = VZ[i], mi = M[i];
        int stack[STACK];
        int sp = 0;
        stack[sp++] = 1;
        while (sp > 0) {
            const int c = stack[--sp];
            if (tn(c, 12) < 0.0) continue;   // empty
            double r2lo, r2hi;
            box_dist2(&tn(c, 0), &tn(c, 3), xi, yi, zi, r2lo, r2hi);
            if (r2lo >= prune2) continue;
            if (c < TP) {
                if (sp + 2 > STACK) Kokkos::abort("dt_tree: stack overflow");
                stack[sp++] = 2 * c; stack[sp++] = 2 * c + 1;
                continue;
            }
            const int l = c - TP, k1 = Kokkos::min(Tn, (l + 1) * LEAF);
            for (int k = l * LEAF; k < k1; k++) {
                const int j = tidx(k);
                if (j <= i) continue;   // upper triangle
                double dx = tpx(k) - xi, dy = tpy(k) - yi, dz = tpz(k) - zi;
                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 >= r_ngb_max_sq) continue;
                double a_val, e_val, E_orb;
                bool is_bound;
                if (ngb_pair_test(dx, dy, dz, r2, tvx(k) - ui, tvy(k) - vi, tvz(k) - wi, G * (mi + tm(k)), r_ngb, pivot_dt,
                                  a_val, e_val, E_orb, is_bound)) {
                    int q = Kokkos::atomic_fetch_add(o.cnt, 1);
                    if (q < o.max_ngb) {
                        o.pi[q] = i; o.pj[q] = j; o.pa[q] = a_val; o.pe[q] = e_val; o.pE[q] = E_orb;
                        o.pb[q] = is_bound ? 1 : 0;
                    }
                }
            }
        }
    });
}

}  // namespace dtt
