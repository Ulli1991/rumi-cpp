// RUMI++ MSTAR module (GPLv3, see src/mstar/LICENSE)
// Copyright (C) 2026 Ulrich Steinwandel.
// Licensed under the GPLv3 license (or, at your option, any later version).
//
// Design measurement for the production tide hook (NBODY_MSTAR_TIDE, docs/MSTAR.md): an MSTAR group embedded in a
// cluster. A regular 20-body group (central mass 1 with 19 light members on circular orbits of 0.002-0.01, so that
// trajectories can be compared without chaos) on a circular orbit at R = 0.5 in a Plummer cluster (M = 1e4, a = 1; a
// static softened point mass), G = 1, for T = 0.01 (1.6-18 internal orbits, 1/7 of the cluster orbit). Treatments:
//   reference   independent RK4 of the full problem (internal gravity + exact cluster field)
//   impulsive   what the FSI does now: every interval dt the members get the exact cluster acceleration as kicks
//               (kick dt/2, MSTAR internal dynamics in the c.o.m. frame for dt, kick dt/2)
//   hook        c.o.m. kicks (mean acceleration, kick-drift-kick of the c.o.m.) and inside MSTAR the linear tide
//               A(X_mid) about the c.o.m. (ExtraForces, track_work), X_mid the c.o.m. at the middle of the interval
// Reported: internal-configuration error (positions relative to the group c.o.m.) and c.o.m. error against the
// reference, for dt = 1e-3, 5e-4, 2.5e-4, 1e-4 (the innermost orbit 5.6e-4). Measured: the hook is 17x more accurate
// at dt = 1e-3 (1.3e-3 vs 2.2e-2) but has a floor of 1e-3 from the neglected octupole tide (r/R ~ 2 %), which the
// impulsive kicks (second order in dt) undercut at dt = 1e-4 (1.6e-4). Checks: impulsive converges, hook/impulsive
// at 1e-3 and the hook's floor (1.3x measured). ctest mstar_embedded_tide.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <Kokkos_Core.hpp>

#include "mstar_kokkos.h"

namespace {
int g_fail = 0;
void check(const char* what, double v, double bound) {
    const bool ok = v <= bound;
    std::printf("  %s %-52s %.3e (<= %.3e)\n", ok ? "ok  " : "FAIL", what, v, bound);
    if (!ok) g_fail++;
}

constexpr double MCL = 1e4, ACL = 1.0, R0 = 0.5;
void cluster_acc(const double* x, double* a) {
    const double r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + ACL * ACL, f = -MCL / (r2 * std::sqrt(r2));
    for (int k = 0; k < 3; k++) a[k] = f * x[k];
}
// tidal tensor d a_i / d x_j of the Plummer field at X (row-major)
void cluster_tide(const double* X, double* A) {
    const double s2 = X[0] * X[0] + X[1] * X[1] + X[2] * X[2] + ACL * ACL, s = std::sqrt(s2);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) A[3 * i + j] = -MCL / (s2 * s) * ((i == j) - 3.0 * X[i] * X[j] / s2);
}

struct Group { std::vector<double> m, x, v; };
Group make_group() {
    // a regular (non-chaotic) group, so that trajectories can be compared: a central body of mass 1 with 19 light
    // members (1e-6) on circular orbits of radius 0.002-0.01 in random planes (internal periods 5.6e-4 - 6.3e-3)
    const int n = 20;
    Group g;
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<double> U(0, 1);
    g.m.assign(n, 1e-6);
    g.m[0] = 1.0;
    g.x.assign(3 * n, 0.0);
    g.v.assign(3 * n, 0.0);
    for (int i = 1; i < n; i++) {
        const double r = 0.002 + 0.008 * U(rng), vc = std::sqrt(1.0 / r);
        const double ct = 2 * U(rng) - 1, ph = 2 * M_PI * U(rng), st = std::sqrt(1 - ct * ct);
        const double nrm[3] = {st * std::cos(ph), st * std::sin(ph), ct};   // orbit normal
        double e1[3] = {-nrm[1], nrm[0], 0};
        double l = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1]);
        if (l < 1e-8) { e1[0] = 1; e1[1] = 0; l = 1; }
        for (int k = 0; k < 3; k++) e1[k] /= l;
        const double e2[3] = {nrm[1] * e1[2] - nrm[2] * e1[1], nrm[2] * e1[0] - nrm[0] * e1[2], nrm[0] * e1[1] - nrm[1] * e1[0]};
        const double phase = 2 * M_PI * U(rng);
        for (int k = 0; k < 3; k++) {
            g.x[3 * i + k] = r * (std::cos(phase) * e1[k] + std::sin(phase) * e2[k]);
            g.v[3 * i + k] = vc * (-std::sin(phase) * e1[k] + std::cos(phase) * e2[k]);
        }
    }
    double cx[3] = {0, 0, 0}, cv[3] = {0, 0, 0}, M = 0;
    for (int i = 0; i < n; i++) {
        M += g.m[i];
        for (int k = 0; k < 3; k++) { cx[k] += g.m[i] * g.x[3 * i + k]; cv[k] += g.m[i] * g.v[3 * i + k]; }
    }
    double a[3], X[3] = {R0, 0, 0};
    cluster_acc(X, a);
    const double vcl = std::sqrt(-a[0] * R0);
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < 3; k++) { g.x[3 * i + k] -= cx[k] / M; g.v[3 * i + k] -= cv[k] / M; }
        g.x[3 * i] += R0;
        g.v[3 * i + 1] += vcl;
    }
    return g;
}
void com(const Group& g, double* X, double* V) {
    double M = 0;
    for (int k = 0; k < 3; k++) X[k] = V[k] = 0;
    for (size_t i = 0; i < g.m.size(); i++) {
        M += g.m[i];
        for (int k = 0; k < 3; k++) { X[k] += g.m[i] * g.x[3 * i + k]; V[k] += g.m[i] * g.v[3 * i + k]; }
    }
    for (int k = 0; k < 3; k++) { X[k] /= M; V[k] /= M; }
}
void run_mstar(Group& g, double dt, double tol, const rumi_mstar::ExtraForces& ex) {
    rumi_mstar::MstarKokkos M(0, (int)g.m.size());
    M.opt.gbs_relative_tolerance = tol;
    M.opt.output_time_relative_tolerance = 1e-13;
    M.opt.host_engine = true;
    M.opt.extra = ex;
    M.mass = g.m; M.pos = g.x; M.vel = g.v;
    M.run(dt);
    g.x = M.pos; g.v = M.vel;
}
// internal-configuration and c.o.m. differences
void diff(const Group& a, const Group& b, double& d_int, double& d_com) {
    double Xa[3], Va[3], Xb[3], Vb[3];
    com(a, Xa, Va);
    com(b, Xb, Vb);
    d_int = 0;
    double sc = 0;
    for (size_t i = 0; i < a.m.size(); i++)
        for (int k = 0; k < 3; k++) {
            d_int = std::fmax(d_int, std::fabs((a.x[3 * i + k] - Xa[k]) - (b.x[3 * i + k] - Xb[k])));
            sc = std::fmax(sc, std::fabs(b.x[3 * i + k] - Xb[k]));
        }
    d_int /= sc;
    d_com = std::sqrt((Xa[0] - Xb[0]) * (Xa[0] - Xb[0]) + (Xa[1] - Xb[1]) * (Xa[1] - Xb[1]) + (Xa[2] - Xb[2]) * (Xa[2] - Xb[2])) / R0;
}
}  // namespace

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        const double T = 0.01;
        const Group g0 = make_group();
        Group ref = g0;
        {   // independent RK4 of the full problem (internal gravity + exact cluster field), 4e5 steps; MSTAR is not
            // used for the reference: in the cluster frame the group's c.o.m. kinetic energy (~900) swamps its
            // internal U (~4e-3) in T + B, so KETJU's time transformation loses precision there (MSTAR is meant for
            // c.o.m.-frame groups, as the subsystem path uses it)
            const int n = (int)ref.m.size();
            std::vector<double> k1x, k1v, k2x, k2v, k3x, k3v, k4x, k4v, xt(3 * n), vt(3 * n);
            auto acc = [&](const std::vector<double>& X, std::vector<double>& A) {
                A.assign(3 * n, 0.0);
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        if (j == i) continue;
                        double d[3], r2 = 0;
                        for (int k = 0; k < 3; k++) { d[k] = X[3 * j + k] - X[3 * i + k]; r2 += d[k] * d[k]; }
                        const double f = ref.m[j] / (r2 * std::sqrt(r2));
                        for (int k = 0; k < 3; k++) A[3 * i + k] += f * d[k];
                    }
                    double ac[3];
                    cluster_acc(&X[3 * i], ac);
                    for (int k = 0; k < 3; k++) A[3 * i + k] += ac[k];
                }
            };
            const long nst = 400000;
            const double h = T / nst;
            for (long q = 0; q < nst; q++) {
                k1x = ref.v; acc(ref.x, k1v);
                for (int c = 0; c < 3 * n; c++) { xt[c] = ref.x[c] + 0.5 * h * k1x[c]; vt[c] = ref.v[c] + 0.5 * h * k1v[c]; }
                k2x = vt; acc(xt, k2v);
                for (int c = 0; c < 3 * n; c++) { xt[c] = ref.x[c] + 0.5 * h * k2x[c]; vt[c] = ref.v[c] + 0.5 * h * k2v[c]; }
                k3x = vt; acc(xt, k3v);
                for (int c = 0; c < 3 * n; c++) { xt[c] = ref.x[c] + h * k3x[c]; vt[c] = ref.v[c] + h * k3v[c]; }
                k4x = vt; acc(xt, k4v);
                for (int c = 0; c < 3 * n; c++) {
                    ref.x[c] += h / 6 * (k1x[c] + 2 * k2x[c] + 2 * k3x[c] + k4x[c]);
                    ref.v[c] += h / 6 * (k1v[c] + 2 * k2v[c] + 2 * k3v[c] + k4v[c]);
                }
            }
            std::printf("# reference done (RK4)\n");
            std::fflush(stdout);
        }
        std::printf("# dt       impulsive: d_int     d_com      hook: d_int     d_com\n");
        double prev_imp = 1e300;
        const double meas_imp[3] = {0, 0, 0};
        (void)meas_imp;
        for (double dt : {1e-3, 5e-4, 2.5e-4, 1e-4}) {
            const int nint = (int)std::lround(T / dt);
            Group imp = g0, hook = g0;
            for (int s = 0; s < nint; s++) {
                // impulsive: member kicks with the exact cluster acceleration
                auto kick = [&](Group& g, double h) {
                    for (size_t i = 0; i < g.m.size(); i++) {
                        double a[3];
                        cluster_acc(&g.x[3 * i], a);
                        for (int k = 0; k < 3; k++) g.v[3 * i + k] += h * a[k];
                    }
                };
                kick(imp, dt / 2);
                {   // internal dynamics in the c.o.m. frame (as the subsystem path), the c.o.m. drifts
                    double X[3], V[3];
                    com(imp, X, V);
                    Group rel = imp;
                    for (size_t i = 0; i < rel.m.size(); i++)
                        for (int k = 0; k < 3; k++) { rel.x[3 * i + k] -= X[k]; rel.v[3 * i + k] -= V[k]; }
                    run_mstar(rel, dt, 1e-11, rumi_mstar::ExtraForces());
                    for (size_t i = 0; i < rel.m.size(); i++)
                        for (int k = 0; k < 3; k++) {
                            imp.x[3 * i + k] = rel.x[3 * i + k] + X[k] + dt * V[k];
                            imp.v[3 * i + k] = rel.v[3 * i + k] + V[k];
                        }
                }
                kick(imp, dt / 2);
                // hook: c.o.m. kicks, internal dynamics with the linear tide about the c.o.m. at mid-interval
                auto com_kick = [&](Group& g, double h) {
                    double M = 0, ab[3] = {0, 0, 0};
                    for (size_t i = 0; i < g.m.size(); i++) {
                        double a[3];
                        cluster_acc(&g.x[3 * i], a);
                        M += g.m[i];
                        for (int k = 0; k < 3; k++) ab[k] += g.m[i] * a[k];
                    }
                    for (size_t i = 0; i < g.m.size(); i++)
                        for (int k = 0; k < 3; k++) g.v[3 * i + k] += h * ab[k] / M;
                };
                com_kick(hook, dt / 2);
                {
                    double X[3], V[3];
                    com(hook, X, V);
                    Group rel = hook;
                    for (size_t i = 0; i < rel.m.size(); i++)
                        for (int k = 0; k < 3; k++) { rel.x[3 * i + k] -= X[k]; rel.v[3 * i + k] -= V[k]; }
                    rumi_mstar::ExtraForces ex;
                    ex.vindep = true;
                    double Xm[3] = {X[0] + 0.5 * dt * V[0], X[1] + 0.5 * dt * V[1], X[2] + 0.5 * dt * V[2]};
                    cluster_tide(Xm, ex.A);
                    run_mstar(rel, dt, 1e-11, ex);
                    for (size_t i = 0; i < rel.m.size(); i++)
                        for (int k = 0; k < 3; k++) {
                            hook.x[3 * i + k] = rel.x[3 * i + k] + X[k] + dt * V[k];
                            hook.v[3 * i + k] = rel.v[3 * i + k] + V[k];
                        }
                }
                com_kick(hook, dt / 2);
            }
            double di, dc, hi, hc;
            diff(imp, ref, di, dc);
            diff(hook, ref, hi, hc);
            std::printf("  %.0e   %.3e  %.3e   %.3e  %.3e\n", dt, di, dc, hi, hc);
            char lab[96];
            std::snprintf(lab, sizeof lab, "dt %.1e: impulsive error shrinks", dt);
            check(lab, di / prev_imp, 1.0);
            if (dt == 1e-3) check("dt 1e-3: hook / impulsive internal error", hi / di, 0.078);   // measured 0.060
            if (dt == 1e-4) check("dt 1e-4: hook internal error (linear-tide floor)", hi, 1.33e-3);   // measured 1.02e-3
            prev_imp = di;
        }
        std::printf("%s (%d failed checks)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    }
    Kokkos::finalize();
    return g_fail ? 1 : 0;
}
