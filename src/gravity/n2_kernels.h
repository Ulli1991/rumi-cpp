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

// Time steps of the sinks from the sources (HHS subset). Collisional mode also accumulates the
// partner-free acceleration and jerk: jerk criterion (BIFROST sec. 3.4.5), and ax, ay, az are overwritten
// with that acceleration for the v/a, gradient and perturbation criteria that follow.
template <bool COLL>
struct DtK {
    static constexpr int NSRC = COLL ? 11 : 7, NS = COLL ? 6 : 0, NM = 1;
    RawView x, y, z, m, vx, vy, vz, ax, ay, az, dt;
    RawView cvx, cvy, cvz;   // velocities for the pair time step (subsystem CoM velocity or vx, vy, vz)
    const int* sid;
    double G, soft2, eta_ff, eta_fb, eta_jerk;
    struct Sink { double x, y, z, m, vx, vy, vz, cvx, cvy, cvz; int sid; };
    KOKKOS_INLINE_FUNCTION Sink sink(int i) const {
        return {x(i), y(i), z(i), m(i), vx(i), vy(i), vz(i), COLL ? cvx(i) : 0.0, COLL ? cvy(i) : 0.0,
                COLL ? cvz(i) : 0.0, COLL ? sid[i] : -1};
    }
    KOKKOS_INLINE_FUNCTION void source(int j, double* s) const {
        s[0] = x(j); s[1] = y(j); s[2] = z(j); s[3] = m(j); s[4] = vx(j); s[5] = vy(j); s[6] = vz(j);
        if (COLL) { s[7] = static_cast<double>(sid[j]); s[8] = cvx(j); s[9] = cvy(j); s[10] = cvz(j); }
    }
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, double* sum, double* mn) const {
        const bool w = pair_weight(i, j, a.sid, COLL ? static_cast<int>(s[7]) : -1);
        double dx = s[0] - a.x, dy = s[1] - a.y, dz = s[2] - a.z;
        double dvx = s[4] - a.vx, dvy = s[5] - a.vy, dvz = s[6] - a.vz;
        double r2s = w ? dx * dx + dy * dy + dz * dz + soft2 : 1.0;
        double ir = fast_rsqrt(r2s);
        double t = COLL ? pair_dt(r2s, ir, dx, dy, dz, s[8] - a.cvx, s[9] - a.cvy, s[10] - a.cvz, G, a.m + s[3], eta_ff, eta_fb)
                        : pair_dt(r2s, ir, dx, dy, dz, dvx, dvy, dvz, G, a.m + s[3], eta_ff, eta_fb);
        mn[0] = Kokkos::fmin(mn[0], w ? t : 1e30);
        if (COLL) {
            double fac1 = G * (w ? s[3] : 0.0) * ir * ir * ir;
            double fac2 = 3.0 * (dx * dvx + dy * dvy + dz * dvz) * ir * ir;
            sum[0] += fac1 * dx; sum[1] += fac1 * dy; sum[2] += fac1 * dz;
            sum[3] += fac1 * (dvx - fac2 * dx); sum[4] += fac1 * (dvy - fac2 * dy); sum[5] += fac1 * (dvz - fac2 * dz);
        }
    }
    KOKKOS_INLINE_FUNCTION void finish(int i, const Sink&, const double* sum, const double* mn) const {
        double d = mn[0];
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
    static constexpr int NSRC = COLL ? 11 : 7, NS = COLL ? 6 : 3, NM = 1;
    RawView x, y, z, m, vx, vy, vz, ax, ay, az, dt;
    RawView cvx, cvy, cvz;   // velocities for the pair time step (subsystem CoM velocity or vx, vy, vz)
    const int* sid;
    double G, soft2, eta_ff, eta_fb, eta_jerk, eta_va;
    struct Sink { double x, y, z, m, vx, vy, vz, cvx, cvy, cvz; int sid; };
    KOKKOS_INLINE_FUNCTION Sink sink(int i) const {
        return {x(i), y(i), z(i), m(i), vx(i), vy(i), vz(i), COLL ? cvx(i) : 0.0, COLL ? cvy(i) : 0.0,
                COLL ? cvz(i) : 0.0, COLL ? sid[i] : -1};
    }
    KOKKOS_INLINE_FUNCTION void source(int j, double* s) const {
        s[0] = x(j); s[1] = y(j); s[2] = z(j); s[3] = m(j); s[4] = vx(j); s[5] = vy(j); s[6] = vz(j);
        if (COLL) { s[7] = static_cast<double>(sid[j]); s[8] = cvx(j); s[9] = cvy(j); s[10] = cvz(j); }
    }
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, double* sum, double* mn) const {
        const bool w = pair_weight(i, j, a.sid, COLL ? static_cast<int>(s[7]) : -1);
        double dx = s[0] - a.x, dy = s[1] - a.y, dz = s[2] - a.z;
        double dvx = s[4] - a.vx, dvy = s[5] - a.vy, dvz = s[6] - a.vz;
        double r2s = w ? dx * dx + dy * dy + dz * dz + soft2 : 1.0;
        double ir = fast_rsqrt(r2s);
        double fac1 = G * (w ? s[3] : 0.0) * ir * ir * ir;
        sum[0] += fac1 * dx; sum[1] += fac1 * dy; sum[2] += fac1 * dz;
        double t = COLL ? pair_dt(r2s, ir, dx, dy, dz, s[8] - a.cvx, s[9] - a.cvy, s[10] - a.cvz, G, a.m + s[3], eta_ff, eta_fb)
                        : pair_dt(r2s, ir, dx, dy, dz, dvx, dvy, dvz, G, a.m + s[3], eta_ff, eta_fb);
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
            if (j_mag > EPS && a_mag > EPS) d = Kokkos::fmin(d, eta_jerk * a_mag / j_mag);
            double v_mag = Kokkos::sqrt(a.vx * a.vx + a.vy * a.vy + a.vz * a.vz);
            if (a_mag > EPS && v_mag > EPS) d = Kokkos::fmin(d, eta_va * v_mag / a_mag);
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
    static constexpr int NSRC = COLL ? 9 : 8, NS = 0, NM = 0, DM = NSRC - 1;   // dmin in the last slot
    RawView x, y, z, m, vx, vy, vz;
    const int* sid;
    int *ei, *ej, *cnt;
    int max_edges;
    double G, soft2, eta_ff, eta_fb, pivot;
    RawView dmin;    // per particle: smallest pair time step within the set (index i - off), from PairMinK
    int off;
    struct Sink { double x, y, z, m, vx, vy, vz, dmin; int sid; };
    KOKKOS_INLINE_FUNCTION Sink sink(int i) const {
        return {x(i), y(i), z(i), m(i), vx(i), vy(i), vz(i), dmin(i - off), COLL ? sid[i] : -1};
    }
    KOKKOS_INLINE_FUNCTION void source(int j, double* s) const {
        s[0] = x(j); s[1] = y(j); s[2] = z(j); s[3] = m(j); s[4] = vx(j); s[5] = vy(j); s[6] = vz(j);
        s[DM] = dmin(j - off);
        if (COLL) s[7] = static_cast<double>(sid[j]);
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
    static constexpr int NSRC = 8, NS = 0, NM = 1;
    RawView x, y, z, m, vx, vy, vz, dmin;
    int off;
    const int* sid;   // null: no subsystems
    double G, soft2, eta_ff, eta_fb;
    struct Sink { double x, y, z, m, vx, vy, vz; int sid; };
    KOKKOS_INLINE_FUNCTION Sink sink(int i) const { return {x(i), y(i), z(i), m(i), vx(i), vy(i), vz(i), sid ? sid[i] : -1}; }
    KOKKOS_INLINE_FUNCTION void source(int j, double* s) const {
        s[0] = x(j); s[1] = y(j); s[2] = z(j); s[3] = m(j); s[4] = vx(j); s[5] = vy(j); s[6] = vz(j);
        s[7] = sid ? static_cast<double>(sid[j]) : -1.0;
    }
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, double*, double* mn) const {
        if (!pair_weight(i, j, a.sid, static_cast<int>(s[7]))) return;
        double dx = s[0] - a.x, dy = s[1] - a.y, dz = s[2] - a.z;
        double r2s = dx * dx + dy * dy + dz * dz + soft2;
        double ir = fast_rsqrt(r2s);
        mn[0] = Kokkos::fmin(mn[0], pair_dt(r2s, ir, dx, dy, dz, s[4] - a.vx, s[5] - a.vy, s[6] - a.vz, G, a.m + s[3], eta_ff, eta_fb));
    }
    KOKKOS_INLINE_FUNCTION void finish(int i, const Sink&, const double*, const double* mn) const { dmin(i - off) = mn[0]; }
};
