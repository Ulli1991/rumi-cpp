#pragma once

// Pairwise interactions run by run_n2 (n2_driver.h). Each is used by the single-rank and the MPI paths;
// they differ only in the sink range (all of a subset, or this rank's slice of it).
// COLL: collisional mode, members of one subsystem do not interact here (their LogH integration does).

#include "n2_driver.h"
#include "constants.h"

// r^2 + soft2 of a pair; SOFT = false (soft2 == 0, collisional runs; run_n2 picks it): r^2 + 0 == r^2 exactly for
// r^2 >= 0, so the sum is bit-identical at one FP64 addition less per pair
template <bool SOFT>
KOKKOS_INLINE_FUNCTION double n2_r2(double dx, double dy, double dz, double soft2) {
    return SOFT ? dx * dx + dy * dy + dz * dz + soft2 : dx * dx + dy * dy + dz * dz;
}

// Acceleration on the sinks: a_i = sum_j G m_j r_ij / |r_ij|^3 (softened). Overwrites ax, ay, az.
template <bool COLL, bool SOFT = true>
struct ForceK {
    static constexpr int NSRC = COLL ? 5 : 4, NS = 3, NM = 0;
    RawView x, y, z, m, ax, ay, az;
    const int* sid;
    double G, soft2;
    struct Sink { double x, y, z; int sid; };
    KOKKOS_INLINE_FUNCTION Sink sink(int i) const { return {x(i), y(i), z(i), COLL ? sid[i] : -1}; }
    KOKKOS_INLINE_FUNCTION void source(int j, double* s) const {
        s[0] = x(j); s[1] = y(j); s[2] = z(j); s[3] = G * m(j);
        if (COLL) s[4] = n2_pack(sid[j]);
    }
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, double* sum, double*) const {
        const bool w = pair_weight(i, j, a.sid, COLL ? n2_lo(s[4]) : -1);
        double dx = s[0] - a.x, dy = s[1] - a.y, dz = s[2] - a.z;
        double r2 = w ? n2_r2<SOFT>(dx, dy, dz, soft2) : 1.0;
        double ir = fast_rsqrt(r2);
        double f = (w ? s[3] : 0.0) * ir * ir * ir;
        sum[0] += f * dx; sum[1] += f * dy; sum[2] += f * dz;
    }
    KOKKOS_INLINE_FUNCTION void finish(int i, const Sink&, const double* sum, const double*) const {
        ax(i) = sum[0]; ay(i) = sum[1]; az(i) = sum[2];
    }
};

// FSI gradient correction (FROST Eq. 19): a_i <- a0_i + coeff * sum_j m_j [r^2 da - 3 (r.da) r] / r^5,
// da = a0_j - a0_i, from snapshots of the uncorrected accelerations (sink snapshot index i - ioff, source
// snapshot index j - joff). Optionally stores |correction| / dt^2 in grad_mag for the gradient criterion.
template <bool COLL, bool SOFT = true>
struct GradK {
    static constexpr int NSRC = COLL ? 8 : 7, NS = 3, NM = 0;
    RawView x, y, z, m, ax, ay, az;
    RawView isx, isy, isz, jsx, jsy, jsz;   // snapshots
    int ioff, joff;
    RawView gmag;                           // null: not written
    const int* sid;
    double coeff, soft2, inv_dt2;
    struct Sink { double x, y, z, ax, ay, az; int sid; };
    KOKKOS_INLINE_FUNCTION Sink sink(int i) const {
        return {x(i), y(i), z(i), isx(i - ioff), isy(i - ioff), isz(i - ioff), COLL ? sid[i] : -1};
    }
    KOKKOS_INLINE_FUNCTION void source(int j, double* s) const {
        s[0] = x(j); s[1] = y(j); s[2] = z(j); s[3] = coeff * m(j);
        s[4] = jsx(j - joff); s[5] = jsy(j - joff); s[6] = jsz(j - joff);
        if (COLL) s[7] = n2_pack(sid[j]);
    }
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, double* sum, double*) const {
        const bool w = pair_weight(i, j, a.sid, COLL ? n2_lo(s[7]) : -1);
        double dx = s[0] - a.x, dy = s[1] - a.y, dz = s[2] - a.z;
        double dax = s[4] - a.ax, day = s[5] - a.ay, daz = s[6] - a.az;
        double r2 = w ? n2_r2<SOFT>(dx, dy, dz, soft2) : 1.0;
        double ir = fast_rsqrt(r2);
        double ir2 = ir * ir;
        double f = (w ? s[3] : 0.0) * ir2 * ir2 * ir;
        double rda = dx * dax + dy * day + dz * daz;
        sum[0] += f * (r2 * dax - 3.0 * rda * dx);
        sum[1] += f * (r2 * day - 3.0 * rda * dy);
        sum[2] += f * (r2 * daz - 3.0 * rda * dz);
    }
    KOKKOS_INLINE_FUNCTION void finish(int i, const Sink& a, const double* sum, const double*) const {
        ax(i) = a.ax + sum[0]; ay(i) = a.ay + sum[1]; az(i) = a.az + sum[2];
        if (gmag.ptr) gmag(i) = Kokkos::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]) * inv_dt2;
    }
};

// GRAD_FRESH (fix 3): |g'_i| of the gradient time-step criteria (Rantala et al. 2023, sec. 3.4.4) as the GradK sum with
// coeff = G GRAD_COEFF from the current accelerations (snapshots = ax, ay, az; nothing corrected), stored in gmag
// (inv_dt2 = 1). Evaluated with the time steps from the current state, as the tidal-tensor step of Grudic & Hopkins
// 2020 (arXiv:1910.06349), instead of read from whichever gradient kick last wrote it
template <bool COLL>
struct GradMagK : GradK<COLL> {
    KOKKOS_INLINE_FUNCTION void finish(int i, const typename GradK<COLL>::Sink&, const double* sum, const double*) const {
        this->gmag(i) = Kokkos::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
    }
};

// Symmetrized pair time step (FROST Eqs. 23, 49) folded over the sources
KOKKOS_INLINE_FUNCTION double pair_dt(double r2s, double ir, double dx, double dy, double dz,
                                      double dvx, double dvy, double dvz, double G, double msum,
                                      double eta_ff, double eta_fb) {
    double tff, tfb;
    symmetrized_timestep(r2s, ir, dx, dy, dz, dvx, dvy, dvz, G, msum, eta_ff, eta_fb, tff, tfb);
    return n2_fmin(eta_ff * tff, eta_fb * tfb);
}

// Single-precision pre-rejects of the pair time step (AUDIT G row 2a) and of the tidal-sampling step (device code with
// DT_FP32_PREREJECT, default on; DtK and AccDtK). The time-step kernels are FP64-throughput bound and the FP32 pipe
// idles, while most pairs are distant and rejected by the fp64 test of pair_dt_below. The fp32 tests decide the same
// question and only ever say "reject" when the pair cannot lower the minimum, so skipping its fp64 evaluation leaves
// the minimum bit-identical:
//  * Exact inequality (pair_dt_below below): pair_dt >= min over ff, fb of eta t / (1 + d-), d- = max(0, -d),
//    d = c t eta, taken on the fp64 inputs (r2s, dx.., dv.., m) as exact numbers.
//    dv is the velocity difference the pair step uses (component velocities, or the subsystem c.o.m. velocities of
//    FLYBY_CM_VELOCITY); the fp32 test and the fp64 step take the same dv, so nothing below depends on that choice.
//  * fp32 inputs: r2s, |dv|^2 and r.dv are formed in fp64 (|dv|^2: rel. error < 4e-16; r.dv: abs. error < 4e-16 r v
//    by Cauchy-Schwarz, r >= |dx| since r2s = |dx|^2 + soft2) and rounded to fp32 (rel. u = 2^-24); G m_sum is
//    (float)G ((float)m_i + (float)m_j) (rel. < 4u); cur is rounded to fp32 (rel. u). rsqrtf <= 2 ulp; t_ff, t_fb, r,
//    v and the factor (1 + G m / (r v^2)) are products/quotients of positive terms, so their relative error is below
//    ~20 u = 1.2e-6. max(0, -r.dv) <= max(0, -(r.dv)_32 + 1e-6 r v) is an upper bound (1e-6 r v > u |r.dv| +
//    4e-16 r v, with room for the fp32 r and v), and c has the sign of r.dv for both criteria.
//  * The test eta t_32 (1 - 4e-5) >= M cur_32 (1 + 4e-5) (1 + d-_up32 (1 + 4e-5)) for both criteria therefore
//    implies the exact eta t >= M (1 - 1e-5) cur (1 + d-), i.e. pair_dt >= 0.9999 M cur, M = N2_DT_REJECT_MARGIN:
//    the premise of the fp64 test below, whose fp64 evaluation error (~1e-15) the margin covers many times.
//  * Range: r2s outside [1e-30, 1e30] is not tested (no reject). Overflow on the right side gives inf (< 3e38 fails:
//    no reject), NaN fails every comparison (no reject); an infinite left side means a true value beyond FLT_MAX,
//    above any finite right side.
// Cost: 64-bit conversions run at half the FP64 rate on A100 (16/clk/SM) and the old version converted ~15 values per
// pair (positions, velocities, masses, k, cur), as much FP64-pipe time as the fp64 arithmetic of the pair. Now a pair
// converts three (r2s, |dv|^2, r.dv): m, k and 1/m come as fp32 from the source tile (n2_pack) and the sink, cur from
// DtSink (refreshed only when the minimum changes).
// NBODY_CHECK_DT_FP32=1 runs the fp64-only DtK after every time-step kernel and aborts on any bitwise difference
// (forces.cpp); n2-bench checksums dt against dt64.
//  * Instruction count: rsqrt.approx.ftz (rel. error < 2^-22.9, as rsqrtf, without its subnormal fix-up) on arguments
//    that are normal: r2 in [1e-30, 1e30] (other r2 are not tested; a flushed r2 gives ir = inf and fails the range
//    tests), v2 >= 1e-30, r max(G m, 1e-20) >= 1e-35 (replacing G m by max(G m, 1e-20) only lowers t_ff and raises
//    d_fb: conservative). The cur-dependent factors 3 (1 + 4e-5) cur and (1 + 4e-5) cur are kept in the sink (DtSink),
//    the eta factors are launch constants; each regrouping adds at most a rounding (u) inside the 20 u budget.
#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && defined(DT_FP32_PREREJECT)
#define N2_DT_F32 1
#endif
KOKKOS_INLINE_FUNCTION float rsqrt_f32_normal(float x) {   // x normal, or the result is not used
#if defined(__CUDA_ARCH__)
    float y;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
#elif defined(__HIP_DEVICE_COMPILE__)
    return rsqrtf(x);
#else
    return 1.0f / std::sqrt(x);
#endif
}
// Margin of the pair-step rejects over the exact lower bound eta t / (1 + d-) (DtK, AccDtK, EdgeK-free tree search);
// any value > 1 keeps the minimum bit-identical, the margin only covers the evaluation errors (fp32 ~1e-6, fp64 ~1e-15)
constexpr double N2_DT_REJECT_MARGIN = 1.25;
constexpr float N2_DT_REJECT_MARGIN_F = (float)N2_DT_REJECT_MARGIN * (1.0f + 4e-5f);   // fp32 test: M (1 + 4e-5)
// cur3 = M (1 + 4e-5) cur_32
KOKKOS_INLINE_FUNCTION bool pair_dt_reject_f32(float cur3, float r2, float ir, double v2s, double rdvs, float gm,
                                               float eff, float efb) {
#ifdef N2_DT_F32
    if (!(r2 >= 1e-30f && r2 <= 1e30f)) return false;
    gm = n2_fmax(gm, 1e-20f);
    const float v2 = (float)v2s + 1e-30f;
    const float r = r2 * ir;
    const float tff = r2 * rsqrt_f32_normal(r * gm);
    const float iv = rsqrt_f32_normal(v2), tfb = r * iv;
    const float rdvn = n2_fmax(0.0f, 1e-6f * r * v2 * iv - (float)rdvs);   // >= max(0, -r.dv)
    const float c = rdvn * ir * ir;                                       // >= max(0, -c_ff / 1.5), max(0, -c_fb / ...)
    const float kff = 1.5f * (1.0f + 4e-5f) * eff, kfb = (1.0f + 4e-5f) * efb;   // launch constants
    const float rff = cur3 * (1.0f + (c * tff) * kff), rfb = cur3 * (1.0f + (c * (1.0f + gm * ir * iv * iv) * tfb) * kfb);
    return rff < 3e38f && rfb < 3e38f && tff * ((1.0f - 4e-5f) * eff) >= rff && tfb * ((1.0f - 4e-5f) * efb) >= rfb;
#else
    (void)cur3; (void)r2; (void)ir; (void)v2s; (void)rdvs; (void)gm; (void)eff; (void)efb;
    return false;
#endif
}

// pair_dt, skipped (1e30) when it cannot go below `cur`: with d = c t eta (FROST Eq. 49) the symmetrized step is
// eta t f(d), f(d) = 1 + d / (1 + sqrt(1 + d^2)), for either criterion. f >= 1 for d >= 0; for d = -a < 0, with
// s = sqrt(1 + a^2) and s - a = 1 / (s + a), f = (1 + 1 / (s + a)) / (1 + s) >= 1 / (1 + a) (equivalent to
// 1 + a >= 1). So pair_dt >= eta t / (1 + max(0, -d)) (before 2026-10: the looser eta t / (2 (1 + |d|)), i.e. twice
// or more as many pairs evaluated in full). Pairs that pass the test with the margin N2_DT_REJECT_MARGIN get the
// unchanged pair_dt, so the minimum over pairs is bit-identical; most distant pairs skip two reciprocal roots and both
// divisions.
KOKKOS_INLINE_FUNCTION double pair_dt_below(double cur, double r2s, double ir, double dx, double dy, double dz,
                                            double dvx, double dvy, double dvz, double G, double msum,
                                            double eta_ff, double eta_fb) {
    const double Gm = G * msum;
    const double v_sq = dvx * dvx + dvy * dvy + dvz * dvz;
    const double r = r2s * ir;
    const double t_ff = r2s * fast_rsqrt(r * (Gm + EPS));
    const double inv_v = fast_rsqrt(v_sq + EPS);
    const double t_fb = r * inv_v;
    const double rdv_ir2 = (dx * dvx + dy * dvy + dz * dvz) * ir * ir;
    const double d_ff = 1.5 * rdv_ir2 * t_ff * eta_ff;
    const double d_fb = rdv_ir2 * (1.0 + Gm * ir * inv_v * inv_v) * t_fb * eta_fb;
    const double mc = N2_DT_REJECT_MARGIN * cur;
    if (eta_ff * t_ff >= mc * (1.0 + n2_fmax(0.0, -d_ff)) && eta_fb * t_fb >= mc * (1.0 + n2_fmax(0.0, -d_fb)))
        return 1e30;
    return pair_dt(r2s, ir, dx, dy, dz, dvx, dvy, dvz, G, msum, eta_ff, eta_fb);
}

// Neighbour radius of a pair (NGB_MASS_SCALING): r_ngb max(1, m_max / <m>)^(1/3) with m_max the heavier particle,
// inv_mref = 1 / <m> (0: off, r_ngb). PeTar's changeover radius (Wang et al. 2020, 2006.16560, sec. 6.1, Eq. 12):
// a perturber's tide on a binary scales as m / r^3, so equal tidal strength needs r ~ m^(1/3). With a fixed r_ngb a
// 23+23 Msun BH binary passed a 46 AU stellar binary 1.6 mpc away (outside r_ngb = 1 mpc) at tidal ratio 0.11; the
// FSI kicks then carried that strong interaction (200k, 10 % binaries, seed 42, step 19: |dE/E| +4.7e-10)
KOKKOS_INLINE_FUNCTION double ngb_radius(double r_ngb, double mi, double mj, double inv_mref) {
    return r_ngb * n2_fmax(1.0, Kokkos::cbrt(n2_fmax(mi, mj) * inv_mref));
}

// The neighbour criteria of NgbK (Rantala et al. 2023, sec. 3.5) for a pair with r2 < r_ngb_max^2 (dx = x_j - x_i, dv = v_j - v_i,
// mu = G (m_i + m_j)): bound with a or r below r_ngb, or a fly-by reaching pericentre below r_ngb within the step
// (Eq. 4). Shared by the O(N^2) kernel and the tree range search (dt_tree.h), so both give bit-identical pairs.
KOKKOS_INLINE_FUNCTION bool ngb_pair_test(double dx, double dy, double dz, double r2, double dvx, double dvy, double dvz,
                                          double mu, double r_ngb, double pivot_dt, double& a_val, double& e_val,
                                          double& E_orb, bool& is_bound, double r_wide = 0.0) {
    double v_sq = dvx * dvx + dvy * dvy + dvz * dvz;
    double r_val = Kokkos::sqrt(r2);
    E_orb = 0.5 * v_sq - mu / (r_val + EPS);
    is_bound = (E_orb < 0);
    a_val = -mu / (2.0 * E_orb + EPS);
    double Lx = dy * dvz - dz * dvy, Ly = dz * dvx - dx * dvz, Lz = dx * dvy - dy * dvx;
    double L2 = Lx * Lx + Ly * Ly + Lz * Lz;
    double e_arg = 1.0 + 2.0 * E_orb * L2 / (mu * mu + EPS);
    e_val = Kokkos::sqrt(n2_fmax(is_bound ? 0.0 : 1.0, e_arg));
    bool is_neighbor = false;
    if (is_bound) {
        // r_wide (NGB_WIDE_FAC): bound pairs up to a < r_wide too, so wide binaries just above r_ngb become
        // (Kepler-drift) subsystems instead of FSI pairs stepped at a fraction of their orbit (AUDIT G row 1a:
        // 39 pairs of 206-500 AU caused 92 % of the deep all-fast calls at 1e5, f_bin = 1); fly-bys keep r_ngb
        is_neighbor = (a_val < r_ngb || r_val < r_ngb || a_val < r_wide);
    } else if (r_val < r_ngb) {
        is_neighbor = true;
    } else if (Kokkos::fabs(a_val) * (e_val - 1.0) < r_ngb && v_sq > EPS) {   // fly-by, Rantala et al. 2023, Eq. 4
        is_neighbor = 0.5 * r_val / Kokkos::sqrt(v_sq) < pivot_dt;
    }
    return is_neighbor;
}

// Pairs i < j of the subset that Rantala et al. (2023, sec. 3.5) count as neighbours: bound with a or r below r_ngb, or a
// fly-by reaching pericentre below r_ngb within the step (Eq. 4). Appended to the neighbour buffers through an
// atomic counter (order is irrelevant: detect_from_pairs sorts); NS = NM = 0, so run_n2 only drives the tiling.
struct NgbK {
    static constexpr int NSRC = 7, NS = 0, NM = 0;
    RawView x, y, z, m, vx, vy, vz;
    int *pi, *pj, *pb, *cnt;
    double *pa, *pe, *pE;
    int max_ngb;
    double G, r_ngb, r_ngb_max_sq, pivot_dt, inv_mref, r_wide = 0.0;
    struct Sink { double x, y, z, m, vx, vy, vz; };
    KOKKOS_INLINE_FUNCTION Sink sink(int i) const { return {x(i), y(i), z(i), m(i), vx(i), vy(i), vz(i)}; }
    KOKKOS_INLINE_FUNCTION void source(int j, double* s) const {
        s[0] = x(j); s[1] = y(j); s[2] = z(j); s[3] = m(j); s[4] = vx(j); s[5] = vy(j); s[6] = vz(j);
    }
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, double*, double*) const {
        if (j <= i) return;  // upper triangle
        double dx = s[0] - a.x, dy = s[1] - a.y, dz = s[2] - a.z;
        double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= r_ngb_max_sq) return;
        double a_val, e_val, E_orb;
        bool is_bound;
        bool is_neighbor = ngb_pair_test(dx, dy, dz, r2, s[4] - a.vx, s[5] - a.vy, s[6] - a.vz, G * (a.m + s[3]),
                                         ngb_radius(r_ngb, a.m, s[3], inv_mref), pivot_dt, a_val, e_val, E_orb, is_bound,
                                         r_wide);
        if (is_neighbor) {
            int k = Kokkos::atomic_fetch_add(cnt, 1);
            if (k < max_ngb) {   // array indices; detect_from_pairs converts to particle ids
                pi[k] = i; pj[k] = j; pa[k] = a_val; pe[k] = e_val; pE[k] = E_orb; pb[k] = is_bound ? 1 : 0;
            }
        }
    }
    KOKKOS_INLINE_FUNCTION void finish(int, const Sink&, const double*, const double*) const {}
};

// FLYBY_CM_EPS tidal-sampling step min(k_i / m_j, k_j / m_i) r^3 (PairVelocities::k, 1e30 for non-members), not
// below max(f_i, f_j) (PairVelocities::kf: TIDAL_STEP_RES x P (1 - e)^(3/2) of a member's binary, 0 otherwise):
// kicks that resolve the pericentre passage no longer sample the orbit at random phases, so the bound mu |f| v dt
// does not apply below. Without the floor a perturber drove a 29+29 Msun, 54 yr binary and itself to steps of 4e-9
// Myr and an interval cost 150-270 s instead of ~15 s (200k, 10 % binaries, old seed-7 ICs); on the CM_TIDE test
// (e = 0.97) RES = 1 gives |dE/E| 3.7e-10 in 4 s, no floor 5.7e-10 in 544 s, RES >= 5 degrades (5.5e-9)
KOKKOS_INLINE_FUNCTION double tidal_sampling_dt(double ki, double kj, double mi, double mj, double r2s, double fi,
                                                double fj) {
    return n2_fmax(n2_fmin(ki / mj, kj / mi) * r2s * Kokkos::sqrt(r2s), n2_fmax(fi, fj));
}

// fp32 pre-reject of tidal_sampling_dt (device code with DT_FP32_PREREJECT): the step is >= min(k_i / m_j, k_j / m_i)
// r^3, evaluated as min(k_i (1/m_j), k_j (1/m_i)) r2 r with fp32 k and 1/m (k rel. u, 1/m = 1 / (float)m rel. 2u) and
// r = r2 rsqrtf(r2) (rsqrtf <= 2 ulp): rel. error < 1e-6, covered by the factors 1 -+ 4e-5; an overflow to inf means a
// true value beyond FLT_MAX > cur, NaN fails. Non-members carry k = 1e30, so almost every pair skips the two fp64
// divisions and the square root.
// curt = (1 + 4e-5) cur_32
KOKKOS_INLINE_FUNCTION bool tidal_dt_reject_f32(float curt, float r2, float r, float ki, float rmj, float kj, float rmi) {
#ifdef N2_DT_F32
    if (!(r2 >= 1e-30f)) return false;
    return n2_fmin(ki * rmj, kj * rmi) * r2 * r * (1.0f - 4e-5f) >= curt;
#else
    (void)curt; (void)r2; (void)r; (void)ki; (void)rmj; (void)kj; (void)rmi;
    return false;
#endif
}

// Sink of the time-step kernels. m32, k32, rm32: fp32 m, k, 1/m for the pre-rejects; cur3, curt: M (1 + 4e-5) and
// (1 + 4e-5) times the running minimum in fp32 (updated with it, so a pair does not convert it)
struct DtSink {
    double x, y, z, m, vx, vy, vz, k, kf;
    int sid;
    float m32, k32, rm32;
    mutable float cur3, curt;
};
KOKKOS_INLINE_FUNCTION DtSink dt_sink(double x, double y, double z, double m, double vx, double vy, double vz,
                                      double k, double kf, int sid) {
    const float m32 = (float)m;
    return {x, y, z, m, vx, vy, vz, k, kf, sid, m32, (float)k, 1.0f / m32, 1e30f * N2_DT_REJECT_MARGIN_F,
            1e30f * (1.0f + 4e-5f)};
}
// fmin(mn[0], t), keeping cur3 and curt
KOKKOS_INLINE_FUNCTION void dt_min(const DtSink& a, double* mn, double t) {
#ifdef N2_DT_F32
    if (t < mn[0]) {
        mn[0] = t;
        const float c = (float)t;
        a.cur3 = c * N2_DT_REJECT_MARGIN_F;
        a.curt = c * (1.0f + 4e-5f);
    }
#else
    (void)a;
    mn[0] = n2_fmin(mn[0], t);
#endif
}

// Running-minimum update of the time-step kernels by one pair: the pair step (pair-step velocity difference cd) and,
// in collisional mode, the tidal-sampling step, each skipped when its fp32 pre-reject (F32) shows that it cannot go
// below the minimum. Source data: m (fp64), m32, and k, kf, k32, rm32 (collisional)
template <bool COLL, bool F32>
KOKKOS_INLINE_FUNCTION void dt_pair_min(const DtSink& a, double* mn, bool w, double r2s, double ir, double dx, double dy,
                                        double dz, double cdx, double cdy, double cdz, double G, double eta_ff,
                                        double eta_fb, double mj, float m32j, double kj, double kfj, float k32j,
                                        float rm32j) {
    bool rp = false, rt = false;
#ifdef N2_DT_F32
    if (F32) {
        const float r2f = (float)r2s, irf = rsqrt_f32_normal(r2f);
        rp = pair_dt_reject_f32(a.cur3, r2f, irf, cdx * cdx + cdy * cdy + cdz * cdz, dx * cdx + dy * cdy + dz * cdz,
                                (float)G * (a.m32 + m32j), (float)eta_ff, (float)eta_fb);
        if (COLL) rt = tidal_dt_reject_f32(a.curt, r2f, r2f * irf, a.k32, rm32j, k32j, a.rm32);
    }
#else
    (void)m32j; (void)k32j; (void)rm32j;
#endif
    double t = rp ? 1e30 : pair_dt_below(mn[0], r2s, ir, dx, dy, dz, cdx, cdy, cdz, G, a.m + mj, eta_ff, eta_fb);
    if (COLL) t = n2_fmin(t, rt ? 1e30 : tidal_sampling_dt(a.k, kj, a.m, mj, r2s, a.kf, kfj));
    dt_min(a, mn, w ? t : 1e30);
}

// Sink of the time-step kernels with FLYBY_CM_VELOCITY (CV): the pair-step velocity cvx..cvz (the subsystem c.o.m.
// velocity of members, the particle velocity otherwise)
struct DtSinkCV : DtSink {
    double cvx, cvy, cvz;
};

// Time steps of the sinks from the sources (HHS subset). Collisional mode also accumulates the
// partner-free acceleration and jerk: jerk criterion (Rantala et al. 2023, sec. 3.4.5), and ax, ay, az are overwritten
// with that acceleration for the v/a, gradient and perturbation criteria that follow.
// MN_IN (tree path, dt_tree.h): the minimum over the pair steps is already in dt (exact tree search); only the sums
// are accumulated here, with the same per-pair expressions and order as without MN_IN, so they are bit-identical.
// CV (collisional, FLYBY_CM_VELOCITY, FROST-CLUSTERS II sec. 2.1): the pair time steps take cvx..cvz, the subsystem
// c.o.m. velocity of members, from a second staged copy (source slots 8-10). CV = false: the pair steps take the
// component velocities, the staged vx..vz (cvx..cvz unused; 3 source doubles fewer, -5 % of DtK, AUDIT G row 2b).
// The jerk sums always take vx..vz.
// F32: single-precision pre-rejects of the pair and tidal-sampling steps (bit-identical minimum, see above). The
// bounds hold for any velocity difference; the fp32 test and pair_dt_below both take the pair-step one (cd).
// JERK = false: no jerk sums (ETA_JERK = 0, the default; run_n2 below picks it): the acceleration sums and the minimum
// are the same expressions in the same order, so bit-identical, at ~10 fewer FP64 operations per pair.
// Source tile: x y z m vx vy vz | (sid, fp32 m) | [cvx cvy cvz (CV)] k kf (fp32 k, fp32 1/m) (collisional)
template <bool COLL, bool MN_IN = false, bool CV = false, bool F32 = true, bool JERK = true, bool SOFT = true>
struct DtK {
    static constexpr bool C = COLL && CV;
    static constexpr int KI = C ? 11 : 8;   // source slots of k, kf and the packed (fp32 k, fp32 1/m) (collisional)
    static constexpr int NSRC = COLL ? KI + 3 : 8, NS = COLL ? (JERK ? 6 : 3) : 0, NM = 1;
    static constexpr bool FILL_ONLY = true;   // P > 1 threads per sink only for nearly empty launches (n2_driver.h)
    RawView x, y, z, m, vx, vy, vz, ax, ay, az, dt;
    RawView cvx, cvy, cvz;   // velocities for the pair time step (CV: subsystem c.o.m. velocity or vx, vy, vz)
    RawView ktid, kfl;       // PairVelocities::k, kf (null: no tidal-sampling step)
    const int* sid;
    double G, soft2, eta_ff, eta_fb, eta_jerk;
    using Sink = std::conditional_t<C, DtSinkCV, DtSink>;
    template <bool J, bool S>
    DtK<COLL, MN_IN, CV, F32, J, S> with() const {
        return {x, y, z, m, vx, vy, vz, ax, ay, az, dt, cvx, cvy, cvz, ktid, kfl, sid, G, soft2, eta_ff, eta_fb, eta_jerk};
    }
    KOKKOS_INLINE_FUNCTION Sink sink(int i) const {
        const DtSink b = dt_sink(x(i), y(i), z(i), m(i), vx(i), vy(i), vz(i), (COLL && ktid.ptr) ? ktid(i) : 1e30,
                                 (COLL && ktid.ptr) ? kfl(i) : 0.0, COLL ? sid[i] : -1);
        if constexpr (C) return Sink{b, cvx(i), cvy(i), cvz(i)};
        else return b;
    }
    KOKKOS_INLINE_FUNCTION void source(int j, double* s) const {
        s[0] = x(j); s[1] = y(j); s[2] = z(j); s[3] = m(j); s[4] = vx(j); s[5] = vy(j); s[6] = vz(j);
        const float m32 = (float)s[3];
        s[7] = n2_pack(COLL ? sid[j] : -1, m32);
        if (COLL) {
            if constexpr (C) { s[8] = cvx(j); s[9] = cvy(j); s[10] = cvz(j); }
            s[COLL ? KI : 0] = ktid.ptr ? ktid(j) : 1e30;
            s[COLL ? KI + 1 : 0] = ktid.ptr ? kfl(j) : 0.0;
            s[COLL ? KI + 2 : 0] = n2_pack((float)s[COLL ? KI : 0], 1.0f / m32);
        }
    }
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, double* sum, double* mn) const {
        const bool w = pair_weight(i, j, a.sid, COLL ? n2_lo(s[7]) : -1);
        double dx = s[0] - a.x, dy = s[1] - a.y, dz = s[2] - a.z;
        double dvx = s[4] - a.vx, dvy = s[5] - a.vy, dvz = s[6] - a.vz;
        double r2s = w ? n2_r2<SOFT>(dx, dy, dz, soft2) : 1.0;
        double ir = fast_rsqrt(r2s);
        if constexpr (!MN_IN) {
            // pair-step velocity difference: the c.o.m. velocities (CV), else the same dv as the jerk sum
            constexpr int K = COLL ? KI : 0;
            double cdx = dvx, cdy = dvy, cdz = dvz;
            if constexpr (C) { cdx = s[8] - a.cvx; cdy = s[9] - a.cvy; cdz = s[10] - a.cvz; }
            dt_pair_min<COLL, F32>(a, mn, w, r2s, ir, dx, dy, dz, cdx, cdy, cdz, G, eta_ff, eta_fb, s[3], n2_hi_f(s[7]),
                                   COLL ? s[K] : 0.0, COLL ? s[K + 1] : 0.0, COLL ? n2_lo_f(s[K + 2]) : 0.0f,
                                   COLL ? n2_hi_f(s[K + 2]) : 0.0f);
        }
        if (COLL) {
            double fac1 = G * (w ? s[3] : 0.0) * ir * ir * ir;
            sum[0] += fac1 * dx; sum[1] += fac1 * dy; sum[2] += fac1 * dz;
            if (JERK) {
                double fac2 = 3.0 * (dx * dvx + dy * dvy + dz * dvz) * ir * ir;
                sum[JERK ? 3 : 0] += fac1 * (dvx - fac2 * dx); sum[JERK ? 4 : 0] += fac1 * (dvy - fac2 * dy);
                sum[JERK ? 5 : 0] += fac1 * (dvz - fac2 * dz);
            }
        }
    }
    KOKKOS_INLINE_FUNCTION void finish(int i, const Sink&, const double* sum, const double* mn) const {
        double d = MN_IN ? dt(i) : mn[0];
        if (COLL) {
            if (JERK && eta_jerk > 0.0) {
                double a_mag = Kokkos::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
                double j_mag = Kokkos::sqrt(sum[JERK ? 3 : 0] * sum[JERK ? 3 : 0] + sum[JERK ? 4 : 0] * sum[JERK ? 4 : 0] +
                                            sum[JERK ? 5 : 0] * sum[JERK ? 5 : 0]);
                if (j_mag > EPS && a_mag > EPS) d = n2_fmin(d, eta_jerk * a_mag / j_mag);
            }
            ax(i) = sum[0]; ay(i) = sum[1]; az(i) = sum[2];
        }
        dt(i) = d;
    }
};

// DtK (collisional, component velocities, no jerk, unsoftened: the production variant) in mixed precision
// (default since 2026-10-16; 200k, 5 seeds: |dE/E| equal (2 seeds bit-identical, 2 better, 1 at the noise floor),
// 6-8 % per step): positions and velocities as fp32 hi + lo pairs, the pair step (FROST Eqs. 23, 49) with
// the lower-bound skip of pair_dt_below, the tidal-sampling step and the partner-free acceleration in fp32; minimum in
// fp32, accelerations summed over 32 sources in fp32 (SUM_F32). Both only feed time-step criteria (no kick takes them).
// MN_IN (tree path): only the accelerations, from positions and G m. n2-bench 200k: dt rel. error median 5e-8, max 4e-7,
// no level changed; acceleration rel. error median 5e-8, max 2e-6; 109 vs 78 Gpairs/s
template <bool MN_IN>
struct DtMixK {
    static constexpr int NSRC = MN_IN ? 4 : 9, NS = 3, NM = 1;
    static constexpr bool SUM_F32 = true, FILL_ONLY = true;
    RawView x, y, z, m, vx, vy, vz, ax, ay, az, dt, ktid, kfl;   // ktid null: no tidal-sampling step
    const int* sid;
    float G, eta_ff, eta_fb;
    struct Sink { float xh, xl, yh, yl, zh, zl, uh, ul, vh, vl, wh, wl, m, k, kf, rm; int sid; };
    static KOKKOS_INLINE_FUNCTION float lo(double v, float h) { return (float)(v - (double)h); }
    KOKKOS_INLINE_FUNCTION Sink sink(int i) const {
        const float xh = (float)x(i), yh = (float)y(i), zh = (float)z(i);
        if constexpr (MN_IN) {
            return {xh, lo(x(i), xh), yh, lo(y(i), yh), zh, lo(z(i), zh), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sid[i]};
        } else {
            const float uh = (float)vx(i), vh = (float)vy(i), wh = (float)vz(i), mi = (float)m(i);
            return {xh, lo(x(i), xh), yh, lo(y(i), yh), zh, lo(z(i), zh), uh, lo(vx(i), uh), vh, lo(vy(i), vh), wh,
                    lo(vz(i), wh), mi, ktid.ptr ? (float)ktid(i) : 1e30f, ktid.ptr ? (float)kfl(i) : 0.0f, 1.0f / mi,
                    sid[i]};
        }
    }
    KOKKOS_INLINE_FUNCTION void source(int j, double* s) const {
        const float xh = (float)x(j), yh = (float)y(j), zh = (float)z(j), mj = (float)m(j);
        s[0] = n2_pack(xh, lo(x(j), xh)); s[1] = n2_pack(yh, lo(y(j), yh)); s[2] = n2_pack(zh, lo(z(j), zh));
        if constexpr (MN_IN) {
            s[3] = n2_pack(sid[j], G * mj);
        } else {
            const float uh = (float)vx(j), vh = (float)vy(j), wh = (float)vz(j);
            s[3] = n2_pack(uh, lo(vx(j), uh)); s[4] = n2_pack(vh, lo(vy(j), vh)); s[5] = n2_pack(wh, lo(vz(j), wh));
            s[6] = n2_pack(sid[j], mj);
            s[7] = n2_pack(ktid.ptr ? (float)ktid(j) : 1e30f, ktid.ptr ? (float)kfl(j) : 0.0f);
            s[8] = n2_pack(1.0f / mj, 0.0f);
        }
    }
    static KOKKOS_INLINE_FUNCTION float sym(float t, float c, float eta) {   // eta t_sym as symmetrized_timestep
        const float ct = c * t * t * eta, sq = 1.0f + c * ct * eta;
        return eta * (t + ct / (sq * rsqrt_f32_normal(sq) + 1.0f));
    }
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, float* sum, float* mn) const {
        const bool w = pair_weight(i, j, a.sid, n2_lo(s[MN_IN ? 3 : 6]));
        const float dx = (n2_lo_f(s[0]) - a.xh) + (n2_hi_f(s[0]) - a.xl);
        const float dy = (n2_lo_f(s[1]) - a.yh) + (n2_hi_f(s[1]) - a.yl);
        const float dz = (n2_lo_f(s[2]) - a.zh) + (n2_hi_f(s[2]) - a.zl);
        const float r2 = w ? dx * dx + dy * dy + dz * dz : 1.0f;
        const float ir = rsqrt_f32_normal(r2), ir2 = ir * ir;
        float gmj;
        if constexpr (MN_IN) {
            gmj = n2_hi_f(s[3]);
        } else {
            const float du = (n2_lo_f(s[3]) - a.uh) + (n2_hi_f(s[3]) - a.ul);
            const float dv = (n2_lo_f(s[4]) - a.vh) + (n2_hi_f(s[4]) - a.vl);
            const float dw = (n2_lo_f(s[5]) - a.wh) + (n2_hi_f(s[5]) - a.wl);
            const float mj = n2_hi_f(s[6]), r = r2 * ir;
            gmj = G * mj;
            const float v2 = du * du + dv * dv + dw * dw + 1e-30f, iv = rsqrt_f32_normal(v2);
            const float gm = G * (a.m + mj);
            const float tff = r2 * rsqrt_f32_normal(r * (gm + 1e-30f)), tfb = r * iv;
            const float rdv = (dx * du + dy * dv + dz * dw) * ir2;
            const float cff = 1.5f * rdv, cfb = rdv * (1.0f + gm * ir * iv * iv);
            // lower bound eta t / (1 + max(0, -c t eta)) of either symmetrized step (pair_dt_below): skip both roots and
            // divisions when it cannot go below the running minimum
            const float lff = eta_ff * tff, lfb = eta_fb * tfb, cur = mn[0] * 1.0001f;
            float t = 1e30f;
            if (lff < cur * (1.0f + n2_fmax(0.0f, -cff * lff)) || lfb < cur * (1.0f + n2_fmax(0.0f, -cfb * lfb)))
                t = n2_fmin(sym(tff, cff, eta_ff), sym(tfb, cfb, eta_fb));
            const float kj = n2_lo_f(s[7]), kfj = n2_hi_f(s[7]), rmj = n2_lo_f(s[8]);
            t = n2_fmin(t, n2_fmax(n2_fmin(a.k * rmj, kj * a.rm) * r2 * r, n2_fmax(a.kf, kfj)));
            mn[0] = w ? n2_fmin(mn[0], t) : mn[0];
        }
        const float f = (w ? gmj : 0.0f) * ir2 * ir;
        sum[0] += f * dx; sum[1] += f * dy; sum[2] += f * dz;
    }
    KOKKOS_INLINE_FUNCTION void finish(int i, const Sink&, const double* sum, const double* mn) const {
        ax(i) = sum[0]; ay(i) = sum[1]; az(i) = sum[2];
        if (!MN_IN) dt(i) = mn[0];
    }
};

// Accelerations and time steps of all particles (start of a step, diagnostics). Collisional mode adds the
// jerk (Eq. 29) and v/a (Eq. 24) criteria. Pre-rejects, CV and JERK as DtK (bit-identical).
// Source tile: x y z m vx vy vz | (sid, fp32 m) | [cvx cvy cvz (CV)] k kf (fp32 k, fp32 1/m) (collisional)
template <bool COLL, bool JERK = true, bool CV = false>
struct AccDtK {
    static constexpr bool C = COLL && CV;
    static constexpr int KI = C ? 11 : 8;
    static constexpr int NSRC = COLL ? KI + 3 : 8, NS = COLL ? (JERK ? 6 : 3) : 3, NM = 1;
    static constexpr bool FILL_ONLY = true;
    RawView x, y, z, m, vx, vy, vz, ax, ay, az, dt;
    RawView cvx, cvy, cvz;   // velocities for the pair time step (CV: subsystem c.o.m. velocity or vx, vy, vz)
    RawView ktid, kfl;       // PairVelocities::k, kf (null: no tidal-sampling step)
    const int* sid;
    double G, soft2, eta_ff, eta_fb, eta_jerk, eta_va;
    using Sink = std::conditional_t<C, DtSinkCV, DtSink>;
    template <bool J>
    AccDtK<COLL, J, CV> with_jerk() const {
        return {x, y, z, m, vx, vy, vz, ax, ay, az, dt, cvx, cvy, cvz, ktid, kfl, sid, G, soft2, eta_ff, eta_fb,
                eta_jerk, eta_va};
    }
    KOKKOS_INLINE_FUNCTION Sink sink(int i) const {
        const DtSink b = dt_sink(x(i), y(i), z(i), m(i), vx(i), vy(i), vz(i), (COLL && ktid.ptr) ? ktid(i) : 1e30,
                                 (COLL && ktid.ptr) ? kfl(i) : 0.0, COLL ? sid[i] : -1);
        if constexpr (C) return Sink{b, cvx(i), cvy(i), cvz(i)};
        else return b;
    }
    KOKKOS_INLINE_FUNCTION void source(int j, double* s) const {
        s[0] = x(j); s[1] = y(j); s[2] = z(j); s[3] = m(j); s[4] = vx(j); s[5] = vy(j); s[6] = vz(j);
        const float m32 = (float)s[3];
        s[7] = n2_pack(COLL ? sid[j] : -1, m32);
        if (COLL) {
            if constexpr (C) { s[8] = cvx(j); s[9] = cvy(j); s[10] = cvz(j); }
            s[COLL ? KI : 0] = ktid.ptr ? ktid(j) : 1e30;
            s[COLL ? KI + 1 : 0] = ktid.ptr ? kfl(j) : 0.0;
            s[COLL ? KI + 2 : 0] = n2_pack((float)s[COLL ? KI : 0], 1.0f / m32);
        }
    }
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, double* sum, double* mn) const {
        const bool w = pair_weight(i, j, a.sid, COLL ? n2_lo(s[7]) : -1);
        double dx = s[0] - a.x, dy = s[1] - a.y, dz = s[2] - a.z;
        double dvx = s[4] - a.vx, dvy = s[5] - a.vy, dvz = s[6] - a.vz;
        double r2s = w ? dx * dx + dy * dy + dz * dz + soft2 : 1.0;
        double ir = fast_rsqrt(r2s);
        double fac1 = G * (w ? s[3] : 0.0) * ir * ir * ir;
        sum[0] += fac1 * dx; sum[1] += fac1 * dy; sum[2] += fac1 * dz;
        if (COLL) {
            constexpr int K = COLL ? KI : 0;
            double cdx = dvx, cdy = dvy, cdz = dvz;
            if constexpr (C) { cdx = s[8] - a.cvx; cdy = s[9] - a.cvy; cdz = s[10] - a.cvz; }
            dt_pair_min<true, true>(a, mn, w, r2s, ir, dx, dy, dz, cdx, cdy, cdz, G, eta_ff, eta_fb, s[3],
                                    n2_hi_f(s[7]), s[K], s[K + 1], n2_lo_f(s[K + 2]), n2_hi_f(s[K + 2]));
        } else
            dt_pair_min<false, true>(a, mn, w, r2s, ir, dx, dy, dz, dvx, dvy, dvz, G, eta_ff, eta_fb, s[3],
                                     n2_hi_f(s[7]), 0.0, 0.0, 0.0f, 0.0f);
        if (COLL && JERK) {
            double fac2 = 3.0 * (dx * dvx + dy * dvy + dz * dvz) * ir * ir;
            sum[JERK ? 3 : 0] += fac1 * (dvx - fac2 * dx); sum[JERK ? 4 : 0] += fac1 * (dvy - fac2 * dy);
            sum[JERK ? 5 : 0] += fac1 * (dvz - fac2 * dz);
        }
    }
    KOKKOS_INLINE_FUNCTION void finish(int i, const Sink& a, const double* sum, const double* mn) const {
        double d = mn[0];
        ax(i) = sum[0]; ay(i) = sum[1]; az(i) = sum[2];
        if (COLL) {
            double a_mag = Kokkos::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
            if (JERK) {
                double j_mag = Kokkos::sqrt(sum[JERK ? 3 : 0] * sum[JERK ? 3 : 0] + sum[JERK ? 4 : 0] * sum[JERK ? 4 : 0] +
                                            sum[JERK ? 5 : 0] * sum[JERK ? 5 : 0]);
                if (eta_jerk > 0.0 && j_mag > EPS && a_mag > EPS) d = n2_fmin(d, eta_jerk * a_mag / j_mag);
            }
            double v_mag = Kokkos::sqrt(a.vx * a.vx + a.vy * a.vy + a.vz * a.vz);
            if (eta_va > 0.0 && a_mag > EPS && v_mag > EPS) d = n2_fmin(d, eta_va * v_mag / a_mag);
        }
        dt(i) = d;
    }
};

// ETA_JERK = 0: the time-step kernels without the jerk sums; soft2 = 0: without the softening addition (both
// bit-identical, see DtK and n2_r2)
template <bool COLL, bool MN_IN, bool CV, bool F32>
void run_n2(const char* name, int i0, int ni, int j0, int nj, const DtK<COLL, MN_IN, CV, F32, true, true>& k) {
    const bool jerk = !COLL || k.eta_jerk > 0.0, soft = k.soft2 != 0.0;
    if (jerk && soft) run_n2<DtK<COLL, MN_IN, CV, F32, true, true>>(name, i0, ni, j0, nj, k);
    else if (jerk) run_n2<DtK<COLL, MN_IN, CV, F32, true, false>>(name, i0, ni, j0, nj, k.template with<true, false>());
    else if (soft) run_n2<DtK<COLL, MN_IN, CV, F32, false, true>>(name, i0, ni, j0, nj, k.template with<false, true>());
    else run_n2<DtK<COLL, MN_IN, CV, F32, false, false>>(name, i0, ni, j0, nj, k.template with<false, false>());
}
template <bool COLL>
void run_n2(const char* name, int i0, int ni, int j0, int nj, const ForceK<COLL, true>& k) {
    if (k.soft2 != 0.0) run_n2<ForceK<COLL, true>>(name, i0, ni, j0, nj, k);
    else run_n2<ForceK<COLL, false>>(name, i0, ni, j0, nj, ForceK<COLL, false>{k.x, k.y, k.z, k.m, k.ax, k.ay, k.az, k.sid, k.G, k.soft2});
}
template <bool COLL>
void run_n2(const char* name, int i0, int ni, int j0, int nj, const GradK<COLL, true>& k) {
    if (k.soft2 != 0.0) run_n2<GradK<COLL, true>>(name, i0, ni, j0, nj, k);
    else run_n2<GradK<COLL, false>>(name, i0, ni, j0, nj, GradK<COLL, false>{k.x, k.y, k.z, k.m, k.ax, k.ay, k.az, k.isx,
        k.isy, k.isz, k.jsx, k.jsy, k.jsz, k.ioff, k.joff, k.gmag, k.sid, k.coeff, k.soft2, k.inv_dt2});
}
template <bool COLL, bool CV>
void run_n2(const char* name, int i0, int ni, int j0, int nj, const AccDtK<COLL, true, CV>& k) {
    if (COLL && !(k.eta_jerk > 0.0)) run_n2<AccDtK<COLL, false, CV>>(name, i0, ni, j0, nj, k.template with_jerk<false>());
    else run_n2<AccDtK<COLL, true, CV>>(name, i0, ni, j0, nj, k);
}

// ---------------------------------------------------------------------------------------------------------------
// HHS with connected components (HHS_CC): the fast set of an HHS call is split into components (particles with a
// pair time step below the pivot, and subsystem partners, are connected). Interactions between components are kicked
// at the current level like the slow-fast ones; each component recurses alone. cid: component id per particle.

// Acceleration of sinks from sources in other components (and other subsystems), ADDED to ax, ay, az
template <bool COLL>
struct ForceCidK {
    static constexpr int NSRC = 5, NS = 3, NM = 0;
    RawView x, y, z, m, ax, ay, az;
    const int* sid;
    const int* cid;
    double G, soft2;
    struct Sink { double x, y, z; int sid, cid; };
    KOKKOS_INLINE_FUNCTION Sink sink(int i) const { return {x(i), y(i), z(i), COLL ? sid[i] : -1, cid[i]}; }
    KOKKOS_INLINE_FUNCTION void source(int j, double* s) const {
        s[0] = x(j); s[1] = y(j); s[2] = z(j); s[3] = G * m(j);
        s[4] = n2_pack(COLL ? sid[j] : -1, cid[j]);
    }
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, double* sum, double*) const {
        const bool w = pair_weight(i, j, a.sid, n2_lo(s[4])) & (n2_hi(s[4]) != a.cid);
        double dx = s[0] - a.x, dy = s[1] - a.y, dz = s[2] - a.z;
        double r2 = w ? dx * dx + dy * dy + dz * dz + soft2 : 1.0;
        double ir = fast_rsqrt(r2);
        double f = (w ? s[3] : 0.0) * ir * ir * ir;
        sum[0] += f * dx; sum[1] += f * dy; sum[2] += f * dz;
    }
    KOKKOS_INLINE_FUNCTION void finish(int i, const Sink&, const double* sum, const double*) const {
        ax(i) += sum[0]; ay(i) += sum[1]; az(i) += sum[2];
    }
};

// Gradient correction from sources in other components, from snapshot accelerations (index i - off), ADDED to ax
template <bool COLL>
struct GradCidK {
    static constexpr int NSRC = 8, NS = 3, NM = 0;
    RawView x, y, z, m, ax, ay, az, snx, sny, snz;
    int off;
    const int* sid;
    const int* cid;
    double coeff, soft2;
    struct Sink { double x, y, z, ax, ay, az; int sid, cid; };
    KOKKOS_INLINE_FUNCTION Sink sink(int i) const {
        return {x(i), y(i), z(i), snx(i - off), sny(i - off), snz(i - off), COLL ? sid[i] : -1, cid[i]};
    }
    KOKKOS_INLINE_FUNCTION void source(int j, double* s) const {
        s[0] = x(j); s[1] = y(j); s[2] = z(j); s[3] = coeff * m(j);
        s[4] = snx(j - off); s[5] = sny(j - off); s[6] = snz(j - off);
        s[7] = n2_pack(COLL ? sid[j] : -1, cid[j]);
    }
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, double* sum, double*) const {
        const bool w = pair_weight(i, j, a.sid, n2_lo(s[7])) & (n2_hi(s[7]) != a.cid);
        double dx = s[0] - a.x, dy = s[1] - a.y, dz = s[2] - a.z;
        double dax = s[4] - a.ax, day = s[5] - a.ay, daz = s[6] - a.az;
        double r2 = w ? dx * dx + dy * dy + dz * dz + soft2 : 1.0;
        double ir = fast_rsqrt(r2);
        double ir2 = ir * ir;
        double f = (w ? s[3] : 0.0) * ir2 * ir2 * ir;
        double rda = dx * dax + dy * day + dz * daz;
        sum[0] += f * (r2 * dax - 3.0 * rda * dx);
        sum[1] += f * (r2 * day - 3.0 * rda * dy);
        sum[2] += f * (r2 * daz - 3.0 * rda * dz);
    }
    KOKKOS_INLINE_FUNCTION void finish(int i, const Sink&, const double* sum, const double*) const {
        ax(i) += sum[0]; ay(i) += sum[1]; az(i) += sum[2];
    }
};

// Edges i < j of the fast set: pair time step below the pivot, or members of one subsystem. Appended through an
// atomic counter (overflow is detected by the caller from the count).
template <bool COLL>
struct EdgeK {
    static constexpr int NSRC = COLL ? 11 : 8, NS = 0, NM = 0, DM = NSRC - 1;   // dmin in the last slot
    RawView x, y, z, m, vx, vy, vz;
    RawView ktid, kfl;   // PairVelocities::k, kf (null: no tidal-sampling step)
    const int* sid;
    int *ei, *ej, *cnt;
    int max_edges;
    double G, soft2, eta_ff, eta_fb, pivot;
    RawView dmin;    // per particle: smallest pair time step within the set (index i - off), from PairMinK
    int off;
    struct Sink { double x, y, z, m, vx, vy, vz, dmin, k, kf; int sid; };
    KOKKOS_INLINE_FUNCTION Sink sink(int i) const {
        return {x(i), y(i), z(i), m(i), vx(i), vy(i), vz(i), dmin(i - off), (COLL && ktid.ptr) ? ktid(i) : 1e30,
                (COLL && ktid.ptr) ? kfl(i) : 0.0, COLL ? sid[i] : -1};
    }
    KOKKOS_INLINE_FUNCTION void source(int j, double* s) const {
        s[0] = x(j); s[1] = y(j); s[2] = z(j); s[3] = m(j); s[4] = vx(j); s[5] = vy(j); s[6] = vz(j);
        s[DM] = dmin(j - off);
        if (COLL) {
            s[7] = n2_pack(sid[j]); s[COLL ? 8 : 0] = ktid.ptr ? ktid(j) : 1e30;
            s[COLL ? 9 : 0] = ktid.ptr ? kfl(j) : 0.0;
        }
    }
    // a pair is an edge if its time step is below the pivot, if it is either particle's dominant partner (smallest
    // pair time step: a particle made fast by its own orbit, e.g. through the v/a criterion, keeps its partner), or
    // if both belong to one subsystem
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, double*, double*) const {
        if (j <= i) return;
        const int sj = COLL ? n2_lo(s[7]) : -1;
        bool edge = (a.sid >= 0) & (sj == a.sid);
        if (!edge) {
            double dx = s[0] - a.x, dy = s[1] - a.y, dz = s[2] - a.z;
            double r2s = dx * dx + dy * dy + dz * dz + soft2;
            double ir = fast_rsqrt(r2s);
            double t = pair_dt(r2s, ir, dx, dy, dz, s[4] - a.vx, s[5] - a.vy, s[6] - a.vz, G, a.m + s[3], eta_ff, eta_fb);
            if (COLL) t = n2_fmin(t, tidal_sampling_dt(a.k, s[COLL ? 8 : 0], a.m, s[3], r2s, a.kf, s[COLL ? 9 : 0]));
            edge = (t < pivot) | (t <= a.dmin) | (t <= s[DM]);
        }
        if (edge) {
            int k = Kokkos::atomic_fetch_add(cnt, 1);
            if (k < max_edges) { ei[k] = i; ej[k] = j; }
        }
    }
    KOKKOS_INLINE_FUNCTION void finish(int, const Sink&, const double*, const double*) const {}
};

// Smallest pair time step of each particle within the set (written to dmin(i - off))
// Smallest pair time step of each particle within the set, skipping its own subsystem (written to dmin(i - off)):
// for a subsystem member the dominant partner is its strongest perturber, which then joins its component
struct PairMinK {
    static constexpr int NSRC = 10, NS = 0, NM = 1;
    RawView x, y, z, m, vx, vy, vz;
    RawView ktid, kfl;   // PairVelocities::k, kf (null: no tidal-sampling step)
    RawView dmin;
    int off;
    const int* sid;   // null: no subsystems
    double G, soft2, eta_ff, eta_fb;
    struct Sink { double x, y, z, m, vx, vy, vz, k, kf; int sid; };
    KOKKOS_INLINE_FUNCTION Sink sink(int i) const {
        return {x(i), y(i), z(i), m(i), vx(i), vy(i), vz(i), ktid.ptr ? ktid(i) : 1e30, ktid.ptr ? kfl(i) : 0.0,
                sid ? sid[i] : -1};
    }
    KOKKOS_INLINE_FUNCTION void source(int j, double* s) const {
        s[0] = x(j); s[1] = y(j); s[2] = z(j); s[3] = m(j); s[4] = vx(j); s[5] = vy(j); s[6] = vz(j);
        s[7] = n2_pack(sid ? sid[j] : -1);
        s[8] = ktid.ptr ? ktid(j) : 1e30;
        s[9] = ktid.ptr ? kfl(j) : 0.0;
    }
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, double*, double* mn) const {
        if (!pair_weight(i, j, a.sid, n2_lo(s[7]))) return;
        double dx = s[0] - a.x, dy = s[1] - a.y, dz = s[2] - a.z;
        double r2s = dx * dx + dy * dy + dz * dz + soft2;
        double ir = fast_rsqrt(r2s);
        mn[0] = n2_fmin(mn[0], n2_fmin(pair_dt(r2s, ir, dx, dy, dz, s[4] - a.vx, s[5] - a.vy, s[6] - a.vz, G, a.m + s[3], eta_ff, eta_fb),
                                                 tidal_sampling_dt(a.k, s[8], a.m, s[3], r2s, a.kf, s[9])));
    }
    KOKKOS_INLINE_FUNCTION void finish(int i, const Sink&, const double*, const double* mn) const { dmin(i - off) = mn[0]; }
};
