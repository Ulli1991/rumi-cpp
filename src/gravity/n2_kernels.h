#pragma once

// Pairwise interactions run by run_n2 (n2_driver.h). Each is used by the single-rank and the MPI paths;
// they differ only in the sink range (all of a subset, or this rank's slice of it).
// COLL: collisional mode, members of one subsystem do not interact here (their LogH integration does).

#include "n2_driver.h"
#include "constants.h"

// Acceleration on the sinks: a_i = sum_j G m_j r_ij / |r_ij|^3 (softened). Overwrites ax, ay, az.
template <bool COLL>
struct ForceK {
    static constexpr int NSRC = COLL ? 5 : 4, NS = 3, NM = 0;
    RawView x, y, z, m, ax, ay, az;
    const int* sid;
    double G, soft2;
    struct Sink { double x, y, z; int sid; };
    KOKKOS_INLINE_FUNCTION Sink sink(int i) const { return {x(i), y(i), z(i), COLL ? sid[i] : -1}; }
    KOKKOS_INLINE_FUNCTION void source(int j, double* s) const {
        s[0] = x(j); s[1] = y(j); s[2] = z(j); s[3] = G * m(j);
        if (COLL) s[4] = static_cast<double>(sid[j]);
    }
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, double* sum, double*) const {
        const bool w = pair_weight(i, j, a.sid, COLL ? static_cast<int>(s[4]) : -1);
        double dx = s[0] - a.x, dy = s[1] - a.y, dz = s[2] - a.z;
        double r2 = w ? dx * dx + dy * dy + dz * dz + soft2 : 1.0;
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
template <bool COLL>
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
        if (COLL) s[7] = static_cast<double>(sid[j]);
    }
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, double* sum, double*) const {
        const bool w = pair_weight(i, j, a.sid, COLL ? static_cast<int>(s[7]) : -1);
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
    KOKKOS_INLINE_FUNCTION void finish(int i, const Sink& a, const double* sum, const double*) const {
        ax(i) = a.ax + sum[0]; ay(i) = a.ay + sum[1]; az(i) = a.az + sum[2];
        if (gmag.ptr) gmag(i) = Kokkos::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]) * inv_dt2;
    }
};

// Symmetrized pair time step (FROST Eqs. 23, 49) folded over the sources
KOKKOS_INLINE_FUNCTION double pair_dt(double r2s, double ir, double dx, double dy, double dz,
                                      double dvx, double dvy, double dvz, double G, double msum,
                                      double eta_ff, double eta_fb) {
    double tff, tfb;
    symmetrized_timestep(r2s, ir, dx, dy, dz, dvx, dvy, dvz, G, msum, eta_ff, eta_fb, tff, tfb);
    return Kokkos::fmin(eta_ff * tff, eta_fb * tfb);
}

// Single-precision pre-reject of the pair time step (AUDIT G row 2a; device code with DT_FP32_PREREJECT, default on).
// DtK is FP64-throughput bound and the FP32 pipe idles, while most pairs are distant and rejected by the fp64 test of
// pair_dt_below. pair_dt_reject_f32 decides the same question in fp32 and only ever says "reject" when the pair cannot
// lower the minimum, so skipping its fp64 evaluation leaves the minimum bit-identical:
//  * Exact inequality (pair_dt_below below): pair_dt >= min over ff, fb of eta t / (2 (1 + |d|)), d = c t eta, taken
//    on the fp64 inputs (r2s, dx.., dv.., m) as exact numbers.
//  * fp32 errors: inputs rounded to fp32 (rel. u = 2^-24); rsqrtf <= 2 ulp; t_ff, t_fb, r, v and the factor
//    (1 + G m / (r v^2)) are products/quotients of positive terms, so their relative error is below ~20 u = 1.2e-6.
//    r.dv can cancel: its absolute error is <= 5 u sum |x_k| |v_k| <= 5 u r v (Cauchy-Schwarz, r >= |dx| since
//    r2s = |dx|^2 + soft2), so |r.dv| <= |r.dv|_32 + 1e-6 r v is an upper bound (1e-6 > 5 u with room for the fp32
//    r and v). cur is rounded to fp32 (rel. u).
//  * The test eta t_32 (1 - 4e-5) >= 3 cur_32 (1 + 4e-5) (1 + |d|_up32 (1 + 4e-5)) for both criteria therefore
//    implies the exact eta t >= 3 (1 - 1e-5) cur (1 + |d|), i.e. pair_dt >= 1.49 cur: the same premise as the fp64
//    test below (factor 3, i.e. margin 1.5 over the exact factor 2), whose fp64 evaluation error it already covers.
//  * Range: r2s outside [1e-30, 1e30] is not tested (no reject). Overflow on the right side gives inf (< 3e38 fails:
//    no reject), NaN fails every comparison (no reject); an infinite left side means a true value beyond FLT_MAX,
//    above any finite right side.
// NBODY_CHECK_DT_FP32=1 runs the fp64-only kernel after every time-step kernel and aborts on any bitwise difference
// (forces.cpp). Measured (one A100, 200k sources, 20 % binary members): DtK x1.27 for 50k sinks, x1.31 at 1e6, x1.15
// for 1024 sinks, bit-identical, and no rejected pair below the running minimum in a per-pair check.
KOKKOS_INLINE_FUNCTION bool pair_dt_reject_f32(double cur, double r2s, double dx, double dy, double dz, double dvx,
                                               double dvy, double dvz, double G, double msum, double eta_ff,
                                               double eta_fb) {
#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && defined(DT_FP32_PREREJECT)
    const float r2 = (float)r2s;
    if (!(r2 >= 1e-30f && r2 <= 1e30f)) return false;
    const float x = (float)dx, y = (float)dy, z = (float)dz, u = (float)dvx, v = (float)dvy, w = (float)dvz;
    const float v2 = u * u + v * v + w * w + 1e-30f;
    const float ir = rsqrtf(r2), r = r2 * ir;
    const float gm = (float)G * (float)msum;
    const float tff = r2 * rsqrtf(r * (gm + 1e-30f));
    const float iv = rsqrtf(v2), tfb = r * iv;
    const float rdv = fabsf(x * u + y * v + z * w) + 1e-6f * r * v2 * iv;   // >= |r.dv|
    const float c = rdv * ir * ir;
    const float eff = (float)eta_ff, efb = (float)eta_fb;
    const float dff = 1.5f * c * tff * eff;
    const float dfb = c * (1.0f + gm * ir * iv * iv) * tfb * efb;
    const float c3 = 3.0f * (float)cur * (1.0f + 4e-5f);
    const float rff = c3 * (1.0f + dff * (1.0f + 4e-5f)), rfb = c3 * (1.0f + dfb * (1.0f + 4e-5f));
    return rff < 3e38f && rfb < 3e38f && eff * tff * (1.0f - 4e-5f) >= rff && efb * tfb * (1.0f - 4e-5f) >= rfb;
#else
    (void)cur; (void)r2s; (void)dx; (void)dy; (void)dz; (void)dvx; (void)dvy; (void)dvz; (void)G; (void)msum;
    (void)eta_ff; (void)eta_fb;
    return false;
#endif
}

// pair_dt, skipped (1e30) when it cannot go below `cur`: with d = c t eta (FROST Eq. 49) the symmetrized step is
// eta t (1 + d / (1 + sqrt(1 + d^2))) >= eta t / (2 (1 + |d|)) for either criterion. Pairs that pass the test (with a
// further factor 1.5 margin) get the unchanged pair_dt, so the minimum over pairs is bit-identical; most distant
// pairs skip two reciprocal roots and both divisions.
// F32: try pair_dt_reject_f32 first (DtK).
template <bool F32 = false>
KOKKOS_INLINE_FUNCTION double pair_dt_below(double cur, double r2s, double ir, double dx, double dy, double dz,
                                            double dvx, double dvy, double dvz, double G, double msum,
                                            double eta_ff, double eta_fb) {
    if (F32 && pair_dt_reject_f32(cur, r2s, dx, dy, dz, dvx, dvy, dvz, G, msum, eta_ff, eta_fb)) return 1e30;
    const double Gm = G * msum;
    const double v_sq = dvx * dvx + dvy * dvy + dvz * dvz;
    const double r = r2s * ir;
    const double t_ff = r2s * fast_rsqrt(r * (Gm + EPS));
    const double inv_v = fast_rsqrt(v_sq + EPS);
    const double t_fb = r * inv_v;
    const double rdv_ir2 = (dx * dvx + dy * dvy + dz * dvz) * ir * ir;
    const double d_ff = 1.5 * rdv_ir2 * t_ff * eta_ff;
    const double d_fb = rdv_ir2 * (1.0 + Gm * ir * inv_v * inv_v) * t_fb * eta_fb;
    if (eta_ff * t_ff >= 3.0 * cur * (1.0 + Kokkos::fabs(d_ff)) && eta_fb * t_fb >= 3.0 * cur * (1.0 + Kokkos::fabs(d_fb)))
        return 1e30;
    return pair_dt(r2s, ir, dx, dy, dz, dvx, dvy, dvz, G, msum, eta_ff, eta_fb);
}

// The neighbour criteria of NgbK (BIFROST sec. 3.5) for a pair with r2 < r_ngb_max^2 (dx = x_j - x_i, dv = v_j - v_i,
// mu = G (m_i + m_j)): bound with a or r below r_ngb, or a fly-by reaching pericentre below r_ngb within the step
// (Eq. 4). Shared by the O(N^2) kernel and the tree range search (dt_tree.h), so both give bit-identical pairs.
KOKKOS_INLINE_FUNCTION bool ngb_pair_test(double dx, double dy, double dz, double r2, double dvx, double dvy, double dvz,
                                          double mu, double r_ngb, double pivot_dt, double& a_val, double& e_val,
                                          double& E_orb, bool& is_bound) {
    double v_sq = dvx * dvx + dvy * dvy + dvz * dvz;
    double r_val = Kokkos::sqrt(r2);
    E_orb = 0.5 * v_sq - mu / (r_val + EPS);
    is_bound = (E_orb < 0);
    a_val = -mu / (2.0 * E_orb + EPS);
    double Lx = dy * dvz - dz * dvy, Ly = dz * dvx - dx * dvz, Lz = dx * dvy - dy * dvx;
    double L2 = Lx * Lx + Ly * Ly + Lz * Lz;
    double e_arg = 1.0 + 2.0 * E_orb * L2 / (mu * mu + EPS);
    e_val = Kokkos::sqrt(Kokkos::fmax(is_bound ? 0.0 : 1.0, e_arg));
    bool is_neighbor = false;
    if (is_bound) {
        is_neighbor = (a_val < r_ngb || r_val < r_ngb);
    } else if (r_val < r_ngb) {
        is_neighbor = true;
    } else if (Kokkos::fabs(a_val) * (e_val - 1.0) < r_ngb && v_sq > EPS) {   // fly-by, BIFROST Eq. 4
        is_neighbor = 0.5 * r_val / Kokkos::sqrt(v_sq) < pivot_dt;
    }
    return is_neighbor;
}

// Pairs i < j of the subset that BIFROST sec. 3.5 counts as neighbours: bound with a or r below r_ngb, or a
// fly-by reaching pericentre below r_ngb within the step (Eq. 4). Appended to the neighbour buffers through an
// atomic counter (order is irrelevant: detect_from_pairs sorts); NS = NM = 0, so run_n2 only drives the tiling.
struct NgbK {
    static constexpr int NSRC = 7, NS = 0, NM = 0;
    RawView x, y, z, m, vx, vy, vz;
    int *pi, *pj, *pb, *cnt;
    double *pa, *pe, *pE;
    int max_ngb;
    double G, r_ngb, r_ngb_max_sq, pivot_dt;
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
        bool is_neighbor = ngb_pair_test(dx, dy, dz, r2, s[4] - a.vx, s[5] - a.vy, s[6] - a.vz, G * (a.m + s[3]), r_ngb,
                                         pivot_dt, a_val, e_val, E_orb, is_bound);
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
    return Kokkos::fmax(Kokkos::fmin(ki / mj, kj / mi) * r2s * Kokkos::sqrt(r2s), Kokkos::fmax(fi, fj));
}

// tidal_sampling_dt, or 1e30 when it cannot go below `cur` (F32, device code with DT_FP32_PREREJECT): the step is
// >= min(k_i / m_j, k_j / m_i) r^3, evaluated in fp32 (inputs rel. u, IEEE division, sqrtf <= 3 ulp: rel. error
// < 1e-6, covered by the factors 1 -+ 4e-5); an overflow to inf means a true value beyond FLT_MAX > cur, NaN fails.
// Non-members carry k = 1e30, so almost every pair skips the two fp64 divisions and the square root.
template <bool F32>
KOKKOS_INLINE_FUNCTION double tidal_sampling_dt_below(double cur, double ki, double kj, double mi, double mj, double r2s,
                                                      double fi, double fj) {
#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && defined(DT_FP32_PREREJECT)
    if (F32) {
        const float r2 = (float)r2s;
        if (r2 >= 1e-30f) {
            const float k1 = (float)ki, k2 = (float)kj, m1 = (float)mi, m2 = (float)mj;
            const float lb = fminf(k1 / m2, k2 / m1) * r2 * sqrtf(r2) * (1.0f - 4e-5f);
            if (lb >= (float)cur * (1.0f + 4e-5f)) return 1e30;
        }
    }
#endif
    return tidal_sampling_dt(ki, kj, mi, mj, r2s, fi, fj);
}

// Time steps of the sinks from the sources (HHS subset). Collisional mode also accumulates the
// partner-free acceleration and jerk: jerk criterion (BIFROST sec. 3.4.5), and ax, ay, az are overwritten
// with that acceleration for the v/a, gradient and perturbation criteria that follow.
// MN_IN (tree path, dt_tree.h): the minimum over the pair steps is already in dt (exact tree search); only the sums
// are accumulated here, with the same per-pair expressions and order as without MN_IN, so they are bit-identical.
// CV = false (collisional, cvx..cvz are vx..vz, i.e. component fly-by velocities): the pair-step velocities are taken
// from the staged vx..vz instead of a second staged copy (13 -> 10 source doubles; same values, bit-identical; -5 %
// of DtK, AUDIT G row 2b).
// F32: single-precision pre-rejects of the pair and tidal-sampling steps (pair_dt_reject_f32; bit-identical minimum).
template <bool COLL, bool MN_IN = false, bool CV = true, bool F32 = true>
struct DtK {
    static constexpr int KI = CV ? 11 : 8;   // source slots of k and kf (collisional)
    static constexpr int NSRC = COLL ? KI + 2 : 7, NS = COLL ? 6 : 0, NM = 1;
    RawView x, y, z, m, vx, vy, vz, ax, ay, az, dt;
    RawView cvx, cvy, cvz;   // velocities for the pair time step (subsystem CoM velocity or vx, vy, vz)
    RawView ktid, kfl;       // PairVelocities::k, kf (null: no tidal-sampling step)
    const int* sid;
    double G, soft2, eta_ff, eta_fb, eta_jerk;
    struct Sink { double x, y, z, m, vx, vy, vz, cvx, cvy, cvz, k, kf; int sid; };
    KOKKOS_INLINE_FUNCTION Sink sink(int i) const {
        return {x(i), y(i), z(i), m(i), vx(i), vy(i), vz(i), (COLL && CV) ? cvx(i) : 0.0, (COLL && CV) ? cvy(i) : 0.0,
                (COLL && CV) ? cvz(i) : 0.0, (COLL && ktid.ptr) ? ktid(i) : 1e30, (COLL && ktid.ptr) ? kfl(i) : 0.0,
                COLL ? sid[i] : -1};
    }
    KOKKOS_INLINE_FUNCTION void source(int j, double* s) const {
        s[0] = x(j); s[1] = y(j); s[2] = z(j); s[3] = m(j); s[4] = vx(j); s[5] = vy(j); s[6] = vz(j);
        if (COLL) {
            s[7] = static_cast<double>(sid[j]);
            if (CV) { s[CV ? 8 : 0] = cvx(j); s[CV ? 9 : 0] = cvy(j); s[CV ? 10 : 0] = cvz(j); }
            s[COLL ? KI : 0] = ktid.ptr ? ktid(j) : 1e30;
            s[COLL ? KI + 1 : 0] = ktid.ptr ? kfl(j) : 0.0;
        }
    }
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, double* sum, double* mn) const {
        const bool w = pair_weight(i, j, a.sid, COLL ? static_cast<int>(s[7]) : -1);
        double dx = s[0] - a.x, dy = s[1] - a.y, dz = s[2] - a.z;
        double dvx = s[4] - a.vx, dvy = s[5] - a.vy, dvz = s[6] - a.vz;
        double r2s = w ? dx * dx + dy * dy + dz * dz + soft2 : 1.0;
        double ir = fast_rsqrt(r2s);
        if constexpr (!MN_IN) {
            // pair-step velocity difference: the c.o.m. velocities (CV), else the same dv as the jerk sum
            const double cdx = CV ? s[CV ? 8 : 0] - a.cvx : dvx, cdy = CV ? s[CV ? 9 : 0] - a.cvy : dvy,
                         cdz = CV ? s[CV ? 10 : 0] - a.cvz : dvz;
            double t = COLL ? Kokkos::fmin(pair_dt_below<F32>(mn[0], r2s, ir, dx, dy, dz, cdx, cdy, cdz, G, a.m + s[3], eta_ff, eta_fb),
                                           tidal_sampling_dt_below<F32>(mn[0], a.k, s[COLL ? KI : 0], a.m, s[3], r2s, a.kf, s[COLL ? KI + 1 : 0]))
                            : pair_dt_below<F32>(mn[0], r2s, ir, dx, dy, dz, dvx, dvy, dvz, G, a.m + s[3], eta_ff, eta_fb);
            mn[0] = Kokkos::fmin(mn[0], w ? t : 1e30);
        }
        if (COLL) {
            double fac1 = G * (w ? s[3] : 0.0) * ir * ir * ir;
            double fac2 = 3.0 * (dx * dvx + dy * dvy + dz * dvz) * ir * ir;
            sum[0] += fac1 * dx; sum[1] += fac1 * dy; sum[2] += fac1 * dz;
            sum[3] += fac1 * (dvx - fac2 * dx); sum[4] += fac1 * (dvy - fac2 * dy); sum[5] += fac1 * (dvz - fac2 * dz);
        }
    }
    KOKKOS_INLINE_FUNCTION void finish(int i, const Sink&, const double* sum, const double* mn) const {
        double d = MN_IN ? dt(i) : mn[0];
        if (COLL) {
            if (eta_jerk > 0.0) {
                double a_mag = Kokkos::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
                double j_mag = Kokkos::sqrt(sum[3] * sum[3] + sum[4] * sum[4] + sum[5] * sum[5]);
                if (j_mag > EPS && a_mag > EPS) d = Kokkos::fmin(d, eta_jerk * a_mag / j_mag);
            }
            ax(i) = sum[0]; ay(i) = sum[1]; az(i) = sum[2];
        }
        dt(i) = d;
    }
};

// Accelerations and time steps of all particles (start of a step, diagnostics). Collisional mode adds the
// jerk (Eq. 29) and v/a (Eq. 24) criteria.
template <bool COLL>
struct AccDtK {
    static constexpr int NSRC = COLL ? 13 : 7, NS = COLL ? 6 : 3, NM = 1;
    RawView x, y, z, m, vx, vy, vz, ax, ay, az, dt;
    RawView cvx, cvy, cvz;   // velocities for the pair time step (subsystem CoM velocity or vx, vy, vz)
    RawView ktid, kfl;       // PairVelocities::k, kf (null: no tidal-sampling step)
    const int* sid;
    double G, soft2, eta_ff, eta_fb, eta_jerk, eta_va;
    struct Sink { double x, y, z, m, vx, vy, vz, cvx, cvy, cvz, k, kf; int sid; };
    KOKKOS_INLINE_FUNCTION Sink sink(int i) const {
        return {x(i), y(i), z(i), m(i), vx(i), vy(i), vz(i), COLL ? cvx(i) : 0.0, COLL ? cvy(i) : 0.0,
                COLL ? cvz(i) : 0.0, (COLL && ktid.ptr) ? ktid(i) : 1e30, (COLL && ktid.ptr) ? kfl(i) : 0.0,
                COLL ? sid[i] : -1};
    }
    KOKKOS_INLINE_FUNCTION void source(int j, double* s) const {
        s[0] = x(j); s[1] = y(j); s[2] = z(j); s[3] = m(j); s[4] = vx(j); s[5] = vy(j); s[6] = vz(j);
        if (COLL) {
            s[7] = static_cast<double>(sid[j]); s[8] = cvx(j); s[9] = cvy(j); s[10] = cvz(j);
            s[COLL ? 11 : 0] = ktid.ptr ? ktid(j) : 1e30;
            s[COLL ? 12 : 0] = ktid.ptr ? kfl(j) : 0.0;
        }
    }
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, double* sum, double* mn) const {
        const bool w = pair_weight(i, j, a.sid, COLL ? static_cast<int>(s[7]) : -1);
        double dx = s[0] - a.x, dy = s[1] - a.y, dz = s[2] - a.z;
        double dvx = s[4] - a.vx, dvy = s[5] - a.vy, dvz = s[6] - a.vz;
        double r2s = w ? dx * dx + dy * dy + dz * dz + soft2 : 1.0;
        double ir = fast_rsqrt(r2s);
        double fac1 = G * (w ? s[3] : 0.0) * ir * ir * ir;
        sum[0] += fac1 * dx; sum[1] += fac1 * dy; sum[2] += fac1 * dz;
            double t = COLL ? Kokkos::fmin(pair_dt_below(mn[0], r2s, ir, dx, dy, dz, s[8] - a.cvx, s[9] - a.cvy, s[10] - a.cvz, G, a.m + s[3], eta_ff, eta_fb),
                                       tidal_sampling_dt(a.k, s[COLL ? 11 : 0], a.m, s[3], r2s, a.kf, s[COLL ? 12 : 0]))
                        : pair_dt_below(mn[0], r2s, ir, dx, dy, dz, dvx, dvy, dvz, G, a.m + s[3], eta_ff, eta_fb);
        mn[0] = Kokkos::fmin(mn[0], w ? t : 1e30);
        if (COLL) {
            double fac2 = 3.0 * (dx * dvx + dy * dvy + dz * dvz) * ir * ir;
            sum[3] += fac1 * (dvx - fac2 * dx); sum[4] += fac1 * (dvy - fac2 * dy); sum[5] += fac1 * (dvz - fac2 * dz);
        }
    }
    KOKKOS_INLINE_FUNCTION void finish(int i, const Sink& a, const double* sum, const double* mn) const {
        double d = mn[0];
        ax(i) = sum[0]; ay(i) = sum[1]; az(i) = sum[2];
        if (COLL) {
            double a_mag = Kokkos::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
            double j_mag = Kokkos::sqrt(sum[3] * sum[3] + sum[4] * sum[4] + sum[5] * sum[5]);
            if (eta_jerk > 0.0 && j_mag > EPS && a_mag > EPS) d = Kokkos::fmin(d, eta_jerk * a_mag / j_mag);
            double v_mag = Kokkos::sqrt(a.vx * a.vx + a.vy * a.vy + a.vz * a.vz);
            if (eta_va > 0.0 && a_mag > EPS && v_mag > EPS) d = Kokkos::fmin(d, eta_va * v_mag / a_mag);
        }
        dt(i) = d;
    }
};

// ---------------------------------------------------------------------------------------------------------------
// HHS with connected components (HHS_CC): the fast set of an HHS call is split into components (particles with a
// pair time step below the pivot, and subsystem partners, are connected). Interactions between components are kicked
// at the current level like the slow-fast ones; each component recurses alone. cid: component id per particle.

// Acceleration of sinks from sources in other components (and other subsystems), ADDED to ax, ay, az
template <bool COLL>
struct ForceCidK {
    static constexpr int NSRC = 6, NS = 3, NM = 0;
    RawView x, y, z, m, ax, ay, az;
    const int* sid;
    const int* cid;
    double G, soft2;
    struct Sink { double x, y, z; int sid, cid; };
    KOKKOS_INLINE_FUNCTION Sink sink(int i) const { return {x(i), y(i), z(i), COLL ? sid[i] : -1, cid[i]}; }
    KOKKOS_INLINE_FUNCTION void source(int j, double* s) const {
        s[0] = x(j); s[1] = y(j); s[2] = z(j); s[3] = G * m(j);
        s[4] = COLL ? static_cast<double>(sid[j]) : -1.0; s[5] = static_cast<double>(cid[j]);
    }
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, double* sum, double*) const {
        const bool w = pair_weight(i, j, a.sid, static_cast<int>(s[4])) & (static_cast<int>(s[5]) != a.cid);
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
    static constexpr int NSRC = 9, NS = 3, NM = 0;
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
        s[7] = COLL ? static_cast<double>(sid[j]) : -1.0; s[8] = static_cast<double>(cid[j]);
    }
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, double* sum, double*) const {
        const bool w = pair_weight(i, j, a.sid, static_cast<int>(s[7])) & (static_cast<int>(s[8]) != a.cid);
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
            s[7] = static_cast<double>(sid[j]); s[COLL ? 8 : 0] = ktid.ptr ? ktid(j) : 1e30;
            s[COLL ? 9 : 0] = ktid.ptr ? kfl(j) : 0.0;
        }
    }
    // a pair is an edge if its time step is below the pivot, if it is either particle's dominant partner (smallest
    // pair time step: a particle made fast by its own orbit, e.g. through the v/a criterion, keeps its partner), or
    // if both belong to one subsystem
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, double*, double*) const {
        if (j <= i) return;
        const int sj = COLL ? static_cast<int>(s[7]) : -1;
        bool edge = (a.sid >= 0) & (sj == a.sid);
        if (!edge) {
            double dx = s[0] - a.x, dy = s[1] - a.y, dz = s[2] - a.z;
            double r2s = dx * dx + dy * dy + dz * dz + soft2;
            double ir = fast_rsqrt(r2s);
            double t = pair_dt(r2s, ir, dx, dy, dz, s[4] - a.vx, s[5] - a.vy, s[6] - a.vz, G, a.m + s[3], eta_ff, eta_fb);
            if (COLL) t = Kokkos::fmin(t, tidal_sampling_dt(a.k, s[COLL ? 8 : 0], a.m, s[3], r2s, a.kf, s[COLL ? 9 : 0]));
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
        s[7] = sid ? static_cast<double>(sid[j]) : -1.0;
        s[8] = ktid.ptr ? ktid(j) : 1e30;
        s[9] = ktid.ptr ? kfl(j) : 0.0;
    }
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, double*, double* mn) const {
        if (!pair_weight(i, j, a.sid, static_cast<int>(s[7]))) return;
        double dx = s[0] - a.x, dy = s[1] - a.y, dz = s[2] - a.z;
        double r2s = dx * dx + dy * dy + dz * dz + soft2;
        double ir = fast_rsqrt(r2s);
        mn[0] = Kokkos::fmin(mn[0], Kokkos::fmin(pair_dt(r2s, ir, dx, dy, dz, s[4] - a.vx, s[5] - a.vy, s[6] - a.vz, G, a.m + s[3], eta_ff, eta_fb),
                                                 tidal_sampling_dt(a.k, s[8], a.m, s[3], r2s, a.kf, s[9])));
    }
    KOKKOS_INLINE_FUNCTION void finish(int i, const Sink&, const double*, const double* mn) const { dmin(i - off) = mn[0]; }
};
