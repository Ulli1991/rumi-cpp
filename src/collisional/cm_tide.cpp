// NBODY_CM_TIDE: c.o.m. kicks plus the orbit-averaged tide for binaries the kicks do not resolve (cm_tide.h,
// CASES.md C25)
#include "cm_tide.h"
#include "subsystem.h"
#include "kepler.h"
#include "host_exec.h"
#include "constants.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

namespace cm_tide {
namespace {

struct Corr {
    int k;
    double F[3];
};
struct Bin {
    int i, j;
    double mi, mj;
    double ai[3], aj[3];      // the members' accelerations of this kick (restored after it)
    double tau[3];            // torque the c.o.m. kick removes: mu r x (a_j - a_i)
    double Phi[6];            // tidal tensor of the sources at the c.o.m. (xx yy zz xy xz yz)
    double Fsum[3];           // sum of the pair corrections on the c.o.m.
    double tc[3];             // their torque, sum_k d_k x F_k
    double a_pre;             // semi-major axis before the kick (NBODY_CM_TIDE_CHECK)
    std::vector<Corr> corr;
};
std::vector<Bin> g_bins;
std::unordered_map<long long, char> g_mode;   // hysteresis: binary (pid pair) in c.o.m. mode at its last kick

struct Stats {
    long long kicks = 0, on = 0, off = 0, refused_gamma = 0, refused_slow = 0, j_clamp = 0, resolved = 0;
    double res_min = 1e30, res_max = 0.0;   // kick spacing / P of the resolved (normal) kicks
    ~Stats() {
        if (kicks || refused_gamma || refused_slow)
            std::printf("CM_TIDE: %lld binary kicks in c.o.m. mode, %lld switches on, %lld off, refused: %lld by the "
                        "tide, %lld by the perturber speed, %lld |j| clamps; %lld normal kicks at spacing/P %.3g-%.3g\n",
                        kicks, on, off, refused_gamma, refused_slow, j_clamp, resolved, resolved ? res_min : 0.0,
                        res_max);
        std::fflush(stdout);
    }
} g_stats;

double n_res() {   // kicks per orbit that count as resolving it (NBODY_CM_TIDE_NRES, default 16: spacing P/16)
    static const double n = [] { const char* e = std::getenv("NBODY_CM_TIDE_NRES"); return e ? std::atof(e) : 16.0; }();
    return n;
}

double gamma_on() {
    static const double g = [] { const char* e = std::getenv("NBODY_CM_TIDE_GAMMA"); return e ? std::atof(e) : 1e-3; }();
    return g;
}

inline void cross(const double* a, const double* b, double* c) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}
inline double dot(const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline void sym_mul(const double* S, const double* x, double* y) {   // S: xx yy zz xy xz yz
    y[0] = S[0] * x[0] + S[3] * x[1] + S[4] * x[2];
    y[1] = S[3] * x[0] + S[1] * x[1] + S[5] * x[2];
    y[2] = S[4] * x[0] + S[5] * x[1] + S[2] * x[2];
}
inline void unit(double* x) {
    const double n = std::sqrt(dot(x, x));
    if (n > 0.0) for (int c = 0; c < 3; c++) x[c] /= n;
}
// v rotated by the minimal rotation taking unit vector n0 to unit vector n1 (Rodrigues)
inline void rotate_min(const double* n0, const double* n1, const double* v, double* out) {
    double k[3];
    cross(n0, n1, k);
    const double s = std::sqrt(dot(k, k)), c = dot(n0, n1);
    if (s < 1e-300) { for (int q = 0; q < 3; q++) out[q] = v[q]; return; }
    for (int q = 0; q < 3; q++) k[q] /= s;
    double kxv[3];
    cross(k, v, kxv);
    const double kv = dot(k, v);
    for (int q = 0; q < 3; q++) out[q] = v[q] * c + kxv[q] * s + k[q] * kv * (1.0 - c);
}

}  // namespace

bool enabled() {
    static const bool on = [] { const char* e = std::getenv("NBODY_CM_TIDE"); return e && std::atoi(e) != 0; }();
    return on;
}

KickSrc& kick_src() {
    static KickSrc s;
    return s;
}

bool pre(Particles& p, int start, int count, double h) {
    g_bins.clear();
    const KickSrc src = kick_src();
    if (!enabled() || !p.collisional || src.step <= 0.0 || src.count <= 0 || count < 2) return false;
    const bool same = (src.start == start && src.count == count);
    if (count + (same ? 0 : src.count) > NMAX) return false;
    constexpr bool host_mem =
        Kokkos::SpaceAccessibility<Kokkos::HostSpace, Kokkos::DefaultExecutionSpace::memory_space>::accessible;
    if (!host_mem && !host_exec()) return false;
    SubsystemManager& mgr = g_subsystem_mgr;
    if (mgr.active.empty()) return false;
    Kokkos::fence();

    const double* m = p.mass.data();
    const double *x = p.x.data(), *y = p.y.data(), *z = p.z.data();
    const double *vx = p.vx.data(), *vy = p.vy.data(), *vz = p.vz.data();
    double *ax = p.ax.data(), *ay = p.ay.data(), *az = p.az.data();
    const int* pid = p.particle_id.data();
    const double G = src.G;
    const double spacing = 0.5 * src.step;   // FSI kicks at 0, s/2, s

    for (const auto& sub : mgr.active) {
        if (sub.n_bodies != 2 || (int)sub.particle_ids.size() != 2) continue;
        const int i = mgr.safe_pid_to_idx(sub.particle_ids[0]), j = mgr.safe_pid_to_idx(sub.particle_ids[1]);
        if (i < start || i >= start + count || j < start || j >= start + count) continue;
        if (pid[i] != sub.particle_ids[0] || pid[j] != sub.particle_ids[1]) continue;
        const double mi = m[i], mj = m[j], M = mi + mj;
        if (mi <= 0.0 || mj <= 0.0) continue;
        const double r[3] = {x[j] - x[i], y[j] - y[i], z[j] - z[i]};
        const double v[3] = {vx[j] - vx[i], vy[j] - vy[i], vz[j] - vz[i]};
        const double rr = std::sqrt(dot(r, r)), GM = G * M;
        const double eps = 0.5 * dot(v, v) - GM / rr;
        const long long key = SubsystemManager::pair_key(sub.particle_ids[0], sub.particle_ids[1]);
        auto it = g_mode.find(key);
        const bool was_on = it != g_mode.end() && it->second;
        auto set_mode = [&](bool on) {
            if (on != was_on) { if (on) g_stats.on++; else g_stats.off++; }
            g_mode[key] = on ? 1 : 0;
        };
        if (eps >= 0.0) { set_mode(false); continue; }
        const double a = -GM / (2.0 * eps);
        const double P = 2.0 * M_PI * std::sqrt(a * a * a / GM);
        if (spacing <= (was_on ? 0.5 : 1.0) * P / n_res()) {
            g_stats.resolved++;
            g_stats.res_min = std::min(g_stats.res_min, spacing / P);
            g_stats.res_max = std::max(g_stats.res_max, spacing / P);
            continue;
        }   // resolved by this kick: normal kick, the mode stays (it follows the tide)
        double hv[3], ev[3];
        cross(r, v, hv);
        cross(v, hv, ev);
        for (int c = 0; c < 3; c++) ev[c] = ev[c] / GM - r[c] / rr;
        const double e = std::sqrt(dot(ev, ev));
        const double R[3] = {(mi * x[i] + mj * x[j]) / M, (mi * y[i] + mj * y[j]) / M, (mi * z[i] + mj * z[j]) / M};
        const double V[3] = {(mi * vx[i] + mj * vx[j]) / M, (mi * vy[i] + mj * vy[j]) / M, (mi * vz[i] + mj * vz[j]) / M};
        // sources: tidal strength and how fast they move
        double tid = 0.0, t_min = 1e300;
        const int s0 = src.start, sn = src.count;
        for (int k = s0; k < s0 + sn; k++) {
            if (k == i || k == j || m[k] <= 0.0) continue;
            const double d[3] = {R[0] - x[k], R[1] - y[k], R[2] - z[k]};
            const double u[3] = {V[0] - vx[k], V[1] - vy[k], V[2] - vz[k]};
            const double dd = std::sqrt(dot(d, d)), uu = std::sqrt(dot(u, u));
            tid += m[k] / (dd * dd * dd);
            if (uu > 0.0) t_min = std::min(t_min, dd / uu);
        }
        const double apo = a * (1.0 + e);
        const double gam = 2.0 * apo * apo * apo * tid / M;
        if (gam >= (was_on ? 4.0 : 1.0) * gamma_on()) { g_stats.refused_gamma++; set_mode(false); continue; }
        if (t_min <= 2.0 * P) { g_stats.refused_slow++; set_mode(false); continue; }
        set_mode(true);

        Bin b{};
        b.i = i; b.j = j; b.mi = mi; b.mj = mj; b.a_pre = a;
        b.ai[0] = ax[i]; b.ai[1] = ay[i]; b.ai[2] = az[i];
        b.aj[0] = ax[j]; b.aj[1] = ay[j]; b.aj[2] = az[j];
        const double mu = mi * mj / M;
        const double da[3] = {b.aj[0] - b.ai[0], b.aj[1] - b.ai[1], b.aj[2] - b.ai[2]};
        cross(r, da, b.tau);
        for (int c = 0; c < 3; c++) b.tau[c] *= mu;
        // averaged quadrupole: <r r^T> = (a^2/2) [(1 - e^2) I - j j^T + 5 e e^T], j = r x v / sqrt(GM a); the
        // isotropic part does not couple to a traceless tidal tensor, so Q = (a^2/2) (5 e e^T - j j^T) is used
        double jv[3];
        const double sq = std::sqrt(GM * a);
        for (int c = 0; c < 3; c++) jv[c] = hv[c] / sq;
        const double qa = 0.5 * a * a;
        const double Q[6] = {qa * (5 * ev[0] * ev[0] - jv[0] * jv[0]), qa * (5 * ev[1] * ev[1] - jv[1] * jv[1]),
                             qa * (5 * ev[2] * ev[2] - jv[2] * jv[2]), qa * (5 * ev[0] * ev[1] - jv[0] * jv[1]),
                             qa * (5 * ev[0] * ev[2] - jv[0] * jv[2]), qa * (5 * ev[1] * ev[2] - jv[1] * jv[2])};
        const double trQ = Q[0] + Q[1] + Q[2];
        for (int k = s0; k < s0 + sn; k++) {
            if (k == i || k == j || m[k] <= 0.0) continue;
            const double d[3] = {R[0] - x[k], R[1] - y[k], R[2] - z[k]};
            const double d2 = dot(d, d), dd = std::sqrt(d2), id3 = 1.0 / (d2 * dd), id5 = id3 / d2, id7 = id5 / d2;
            const double Gmk = G * m[k];
            // tidal tensor Phi_ab = -G m_k (3 d_a d_b - d^2 delta_ab) / d^5
            b.Phi[0] += -Gmk * (3 * d[0] * d[0] - d2) * id5;
            b.Phi[1] += -Gmk * (3 * d[1] * d[1] - d2) * id5;
            b.Phi[2] += -Gmk * (3 * d[2] * d[2] - d2) * id5;
            b.Phi[3] += -Gmk * 3 * d[0] * d[1] * id5;
            b.Phi[4] += -Gmk * 3 * d[0] * d[2] * id5;
            b.Phi[5] += -Gmk * 3 * d[1] * d[2] * id5;
            // the point-mass pair forces of k on the members as the kick has them
            const double di[3] = {x[i] - x[k], y[i] - y[k], z[i] - z[k]};
            const double dj[3] = {x[j] - x[k], y[j] - y[k], z[j] - z[k]};
            const double ri2 = dot(di, di), rj2 = dot(dj, dj);
            const double fi = -Gmk * mi / (ri2 * std::sqrt(ri2)), fj = -Gmk * mj / (rj2 * std::sqrt(rj2));
            // averaged: monopole plus -grad_d of U_q = (mu/2) Phi_ab Q_ab
            //   = -(mu G m_k / 2) [3 d.Q d / d^5 - tr Q / d^3]
            double Qd[3];
            sym_mul(Q, d, Qd);
            const double dQd = dot(d, Qd);
            Corr cr;
            cr.k = k;
            for (int c = 0; c < 3; c++) {
                const double gradUq = -0.5 * mu * Gmk * (6.0 * Qd[c] * id5 - 15.0 * dQd * d[c] * id7 + 3.0 * trQ * d[c] * id5);
                const double F_avg = -Gmk * M * d[c] * id3 - gradUq;
                const double F_inst = fi * di[c] + fj * dj[c];
                cr.F[c] = F_avg - F_inst;
                b.Fsum[c] += cr.F[c];
            }
            double t[3];
            cross(d, cr.F, t);
            for (int c = 0; c < 3; c++) b.tc[c] += t[c];
            b.corr.push_back(cr);
        }
        g_bins.push_back(std::move(b));
    }
    if (g_bins.empty()) return false;
    for (const Bin& b : g_bins) {   // c.o.m. kick: both members get the mass-weighted mean acceleration
        const double M = b.mi + b.mj;
        const double acm[3] = {(b.mi * b.ai[0] + b.mj * b.aj[0]) / M, (b.mi * b.ai[1] + b.mj * b.aj[1]) / M,
                               (b.mi * b.ai[2] + b.mj * b.aj[2]) / M};
        ax[b.i] = ax[b.j] = acm[0];
        ay[b.i] = ay[b.j] = acm[1];
        az[b.i] = az[b.j] = acm[2];
    }
    g_stats.kicks += (long long)g_bins.size();
    (void)h;
    return true;
}

void post(Particles& p, double h) {
    if (g_bins.empty()) return;
    Kokkos::fence();
    const double G = kick_src().G;
    double* m = p.mass.data();
    double *x = p.x.data(), *y = p.y.data(), *z = p.z.data();
    double *vx = p.vx.data(), *vy = p.vy.data(), *vz = p.vz.data();
    double *cvx = p.cvx.data(), *cvy = p.cvy.data(), *cvz = p.cvz.data();
    double *ax = p.ax.data(), *ay = p.ay.data(), *az = p.az.data();
    // 1. restore the accelerations, 2. pair corrections (c.o.m. and source, equal and opposite)
    for (const Bin& b : g_bins) {
        ax[b.i] = b.ai[0]; ay[b.i] = b.ai[1]; az[b.i] = b.ai[2];
        ax[b.j] = b.aj[0]; ay[b.j] = b.aj[1]; az[b.j] = b.aj[2];
        const double M = b.mi + b.mj;
        for (int q : {b.i, b.j}) {
            kahan_add(vx[q], cvx[q], b.Fsum[0] * h / M);
            kahan_add(vy[q], cvy[q], b.Fsum[1] * h / M);
            kahan_add(vz[q], cvz[q], b.Fsum[2] * h / M);
        }
        for (const Corr& cr : b.corr) {
            const int k = cr.k;
            kahan_add(vx[k], cvx[k], -cr.F[0] * h / m[k]);
            kahan_add(vy[k], cvy[k], -cr.F[1] * h / m[k]);
            kahan_add(vz[k], cvz[k], -cr.F[2] * h / m[k]);
        }
    }
    // 3. internal orbits: dL = h (tau - sum_k d_k x F_k), de from the Milankovitch rate, at fixed a
    for (const Bin& b : g_bins) {
        const int i = b.i, j = b.j;
        const double mi = b.mi, mj = b.mj, M = mi + mj, mu = mi * mj / M, GM = G * M;
        const double R[3] = {(mi * x[i] + mj * x[j]) / M, (mi * y[i] + mj * y[j]) / M, (mi * z[i] + mj * z[j]) / M};
        const double V[3] = {(mi * vx[i] + mj * vx[j]) / M, (mi * vy[i] + mj * vy[j]) / M, (mi * vz[i] + mj * vz[j]) / M};
        const double r[3] = {x[j] - x[i], y[j] - y[i], z[j] - z[i]};
        const double v[3] = {vx[j] - vx[i], vy[j] - vy[i], vz[j] - vz[i]};
        const double rr = std::sqrt(dot(r, r));
        const double eps = 0.5 * dot(v, v) - GM / rr;
        if (eps >= 0.0) continue;   // cannot happen (c.o.m. kicks leave the relative motion alone)
        double a = -GM / (2.0 * eps);
        static const bool chk = std::getenv("NBODY_CM_TIDE_CHECK") != nullptr;
        if (chk && std::fabs(a / b.a_pre - 1.0) > 1e-12)
            std::printf("CM_TIDE_CHECK kick changed a by %.3e\n", a / b.a_pre - 1.0);
        double hv[3], ev[3];
        cross(r, v, hv);
        cross(v, hv, ev);
        for (int c = 0; c < 3; c++) ev[c] = ev[c] / GM - r[c] / rr;
        const double e0 = std::sqrt(dot(ev, ev));
        const double Lam = mu * std::sqrt(GM * a);
        double jv[3], n0[3] = {hv[0], hv[1], hv[2]};
        for (int c = 0; c < 3; c++) jv[c] = mu * hv[c] / Lam;
        unit(n0);
        // in-plane reference: pericentre direction (r-hat for a circular orbit), true and mean anomaly
        double P0[3];
        if (e0 > 1e-12) { for (int c = 0; c < 3; c++) P0[c] = ev[c] / e0; }
        else { for (int c = 0; c < 3; c++) P0[c] = r[c] / rr; }
        double Q0[3];
        cross(n0, P0, Q0);
        const double f0 = std::atan2(dot(r, Q0), dot(r, P0));
        const double ec = e0 > 1e-12 ? e0 : 0.0;
        const double E0 = 2.0 * std::atan2(std::sqrt(1.0 - ec) * std::sin(0.5 * f0), std::sqrt(1.0 + ec) * std::cos(0.5 * f0));
        const double M0 = E0 - ec * std::sin(E0);
        // new angular momentum (exact balance) and eccentricity vector
        double Lnew[3];
        for (int c = 0; c < 3; c++) Lnew[c] = mu * hv[c] + h * (b.tau[c] - b.tc[c]);
        double Pe[3], Pj[3], t1[3], t2[3];
        sym_mul(b.Phi, ev, Pe);
        sym_mul(b.Phi, jv, Pj);
        cross(jv, Pe, t1);
        cross(ev, Pj, t2);
        double enew[3];
        const double fac = -h * mu * a * a / (2.0 * Lam);
        for (int c = 0; c < 3; c++) enew[c] = ev[c] + fac * (5.0 * t1[c] - t2[c]);
        const double Lmag = std::sqrt(dot(Lnew, Lnew));
        double n1[3] = {Lnew[0] / Lmag, Lnew[1] / Lmag, Lnew[2] / Lmag};
        double jm = Lmag / Lam;
        if (jm >= 1.0) {   // circular: a takes the angular momentum (rare; its energy change stays visible)
            g_stats.j_clamp++;
            a = Lmag * Lmag / (mu * mu * GM);
            jm = 1.0;
        }
        const double emag = std::sqrt(std::max(0.0, 1.0 - jm * jm));
        const double en = dot(enew, n1);
        for (int c = 0; c < 3; c++) enew[c] -= en * n1[c];
        double Prot[3];
        rotate_min(n0, n1, P0, Prot);
        double P1[3];
        double dvarpi = 0.0;
        const double enorm = std::sqrt(dot(enew, enew));
        if (emag > 1e-12 && enorm > 0.0) {
            for (int c = 0; c < 3; c++) P1[c] = enew[c] / enorm;
            double cx[3];
            cross(Prot, P1, cx);
            dvarpi = std::atan2(dot(cx, n1), dot(Prot, P1));
        } else {
            for (int c = 0; c < 3; c++) P1[c] = Prot[c];
        }
        double Q1[3];
        cross(n1, P1, Q1);
        const double ek = emag > 1e-12 ? emag : 0.0;
        const double M1 = M0 - dvarpi;   // same mean longitude
        const double E1 = ek > 0.0 ? solve_kepler(M1, ek) : M1;
        const double cE = std::cos(E1), sE = std::sin(E1);
        const double rn = a * (1.0 - ek * cE);
        const double vf = std::sqrt(GM * a) / rn;
        double r1[3], v1[3];
        for (int c = 0; c < 3; c++) {
            r1[c] = a * (cE - ek) * P1[c] + a * jm * sE * Q1[c];
            v1[c] = vf * (-sE * P1[c] + jm * cE * Q1[c]);
        }
        // write back as changes of the relative vectors, split by the mass fractions (velocities compensated):
        // the c.o.m. moves only by the rounding of the small changes
        const double fi = mj / M, fj = mi / M;
        double* xs[3] = {x, y, z};
        double* vs[3] = {vx, vy, vz};
        double* cs[3] = {cvx, cvy, cvz};
        for (int c = 0; c < 3; c++) {
            const double dr = r1[c] - r[c], dv = v1[c] - v[c];
            xs[c][i] -= fi * dr;
            xs[c][j] += fj * dr;
            kahan_add(vs[c][i], cs[c][i], -fi * dv);
            kahan_add(vs[c][j], cs[c][j], fj * dv);
        }
        (void)R; (void)V;
    }
    g_bins.clear();
}

}  // namespace cm_tide
