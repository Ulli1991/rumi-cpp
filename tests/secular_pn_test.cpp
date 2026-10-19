// Orbit-averaged PN two-body evolution (src/collisional/secular.cpp) against the subsystem equations of motion
// (eih_pn_acceleration_pair, checked against Blanchet's review in pn_highorder), on eccentric orbits:
//  (1) mean motion n(eps, j) and periastron advance K(eps, j) (Blanchet LRR eqs. n3PN, K3PN) with eps, j from the
//      state (Kidder 1995 E, J): radial period and pericentre angle of an orbit integrated with PN1 + PN2; residuals
//      O(x^3): ratio 8 between gamma and gamma / 2.
//  (2) radiation reaction: the orbit average of d eps / dt and d j / dt under the PN2.5 + PN3.5 acceleration along a
//      1PN orbit against secular_pn_rates (fluxes of Blanchet eq. Finst and Arun et al. 2009 eq. 4.5, balance eq. 6.4):
//      the 1PN corrections must agree; residual relative to the PN2.5 rate O(x^2): ratio 4.
//  (3) secular-triple closed forms (Arun et al. eqs. 6.6-6.9) against the chain rule through (2), O(x^2) as well;
//      PN2.5 alone equal to Peters (1964).
//  (4) integrate_secular_binary with PN1 + PN2 over 200 orbits against the integrated orbit: orbital phase and
//      pericentre angle.
//  (5) write-back near the pericentre (CASES.md C19): the 1e6-run BH binary (e = 0.99936, peri shift 2.4) whose step
//      ended at the pericentre was written back unbound; now refused (-1, unchanged, LogH). Steps ending within 5
//      pericentre times of the pericentre for a = 0.36-36 AU, 1 - e = 0.1-3e-4: every admitted pair stays bound with
//      eps of the written state = the evolved eps.
// Up to 2026-10 the secular path used Peters with a 3.5PN factor -(1247/336 + 35/12 G m / (a c^2)) (flux coefficient
// for da/dt, G m / (a c^2) in place of nu, no eccentricity terms), test-mass 2PN/3PN precession and the Newtonian mean
// motion of the osculating a (docs/CASES.md C9).
#include "post_newtonian_core.h"
#include "secular.h"
#include "kepler.h"
#include <cmath>
#include <cstdio>
#include <initializer_list>

static int fails = 0;
static void check(bool ok, const char* what) { if (!ok) { std::printf("FAIL %s\n", what); fails++; } }

struct Sys { double m1, m2, M, nu, c; };

// relative acceleration: Newton + PN (bits) with the bodies placed by the 1PN c.o.m. relation (Blanchet eq. CMrel)
static void acc_rel(const Sys& s, const double x[3], const double v[3], int bits, double a[3], bool newton = true) {
    const double r = std::sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
    const double v2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2], rd = (x[0]*v[0] + x[1]*v[1] + x[2]*v[2]) / r;
    const double X1 = s.m1 / s.M, X2 = s.m2 / s.M, D = X1 - X2, c2 = s.c * s.c;
    const double P = (0.5 * v2 - 0.5 * s.M / r) / c2, Pd = -0.5 * s.M * rd / (r * r * c2);
    double x1[3], x2[3], v1[3], vv2[3], a1[3], a2[3];
    for (int k = 0; k < 3; k++) {
        x1[k] = (X2 + s.nu * D * P) * x[k]; x2[k] = (-X1 + s.nu * D * P) * x[k];
        v1[k] = (X2 + s.nu * D * P) * v[k] + s.nu * D * Pd * x[k]; vv2[k] = (-X1 + s.nu * D * P) * v[k] + s.nu * D * Pd * x[k];
    }
    eih_pn_acceleration_pair(x1, v1, s.m1, x2, vv2, s.m2, 1.0, s.c, a1, a2, bits);
    for (int k = 0; k < 3; k++) a[k] = (newton ? -s.M * x[k] / (r * r * r) : 0.0) + a1[k] - a2[k];
}

static void rk4(const Sys& s, double x[3], double v[3], double h, int bits) {
    double k1x[3], k1v[3], k2x[3], k2v[3], k3x[3], k3v[3], k4x[3], k4v[3], xt[3], vt[3];
    for (int k = 0; k < 3; k++) k1x[k] = v[k];
    acc_rel(s, x, v, bits, k1v);
    for (int k = 0; k < 3; k++) { xt[k] = x[k] + 0.5 * h * k1x[k]; vt[k] = v[k] + 0.5 * h * k1v[k]; k2x[k] = vt[k]; }
    acc_rel(s, xt, vt, bits, k2v);
    for (int k = 0; k < 3; k++) { xt[k] = x[k] + 0.5 * h * k2x[k]; vt[k] = v[k] + 0.5 * h * k2v[k]; k3x[k] = vt[k]; }
    acc_rel(s, xt, vt, bits, k3v);
    for (int k = 0; k < 3; k++) { xt[k] = x[k] + h * k3x[k]; vt[k] = v[k] + h * k3v[k]; k4x[k] = vt[k]; }
    acc_rel(s, xt, vt, bits, k4v);
    for (int k = 0; k < 3; k++) {
        x[k] += h / 6.0 * (k1x[k] + 2.0 * k2x[k] + 2.0 * k3x[k] + k4x[k]);
        v[k] += h / 6.0 * (k1v[k] + 2.0 * k2v[k] + 2.0 * k3v[k] + k4v[k]);
    }
}

struct OrbitResult { double P, dphi, deps, dj; };

// One radial period from the pericentre (x along +x, v along +y) with the conservative bits; optionally the average of
// d eps / dt, d j / dt under the dissipative bits rr (directional derivative of eps, j along a_rr, Simpson in time)
static OrbitResult radial_orbit(const Sys& s, double rp, double e, int bits, int rr, int conv) {
    double x[3] = {rp, 0, 0}, v[3] = {0, std::sqrt(s.M * (1.0 + e) / rp), 0};
    const double a = rp / (1.0 - e), P0 = 2.0 * M_PI * std::sqrt(a * a * a / s.M);
    const int nstep = 400000;
    const double h = P0 / nstep;
    auto rates = [&](const double* xx, const double* vv, double& de, double& dj) {
        de = dj = 0.0;
        if (!rr) return;
        double ar[3];
        acc_rel(s, xx, vv, rr, ar, false);
        const double an = std::sqrt(ar[0]*ar[0] + ar[1]*ar[1] + ar[2]*ar[2]);
        const double vn = std::sqrt(vv[0]*vv[0] + vv[1]*vv[1] + vv[2]*vv[2]);
        const double d = 1e-4 * vn / an;
        double ep[5], jp[5];
        for (int q = -2; q <= 2; q++) {
            double vq[3] = {vv[0] + q * d * ar[0], vv[1] + q * d * ar[1], vv[2] + q * d * ar[2]};
            secular_pn_eps_j(xx, vq, s.M, s.nu, s.c, conv, ep[q + 2], jp[q + 2]);
        }
        de = (8.0 * (ep[3] - ep[1]) - (ep[4] - ep[0])) / (12.0 * d);
        dj = (8.0 * (jp[3] - jp[1]) - (jp[4] - jp[0])) / (12.0 * d);
    };
    double t = 0.0, Ie = 0.0, Ij = 0.0, de0, dj0;
    rates(x, v, de0, dj0);
    for (int n = 0;; n++) {
        const double xo[3] = {x[0], x[1], x[2]}, vo[3] = {v[0], v[1], v[2]};
        const double rdo = xo[0]*vo[0] + xo[1]*vo[1];
        rk4(s, x, v, h, bits);
        const double rdn = x[0]*v[0] + x[1]*v[1];
        double de1, dj1;
        rates(x, v, de1, dj1);
        if (n > nstep / 2 && rdo < 0.0 && rdn >= 0.0) {
            // pericentre inside the step: r.v linear in time
            const double f = -rdo / (rdn - rdo);
            Ie += f * h * 0.5 * (de0 + (de0 + f * (de1 - de0)));
            Ij += f * h * 0.5 * (dj0 + (dj0 + f * (dj1 - dj0)));
            const double xp[3] = {xo[0] + f * (x[0] - xo[0]), xo[1] + f * (x[1] - xo[1]), 0.0};
            OrbitResult R;
            R.P = t + f * h;
            R.dphi = std::atan2(xp[1], xp[0]);
            R.deps = Ie / R.P; R.dj = Ij / R.P;
            return R;
        }
        Ie += 0.5 * h * (de0 + de1); Ij += 0.5 * h * (dj0 + dj1);
        de0 = de1; dj0 = dj1;
        t += h;
    }
}

int main() {
    // (1) n and K to 2PN
    for (double q : {1.0, 0.25}) {
        for (double e : {0.45, 0.6}) {
            double resn[2], resk[2];
            for (int t = 0; t < 2; t++) {
                const double gam = t == 0 ? 4e-3 : 2e-3;   // G m / (r_p c^2)
                Sys s{1.0 / (1.0 + q), q / (1.0 + q), 1.0, q / ((1.0 + q) * (1.0 + q)), 0.0};
                const double rp = 1.0;
                s.c = std::sqrt(s.M / (rp * gam));
                const OrbitResult R = radial_orbit(s, rp, e, 1 | 2, 0, 3);
                double eps, j;
                double x0[3] = {rp, 0, 0}, v0[3] = {0, std::sqrt(s.M * (1.0 + e) / rp), 0};
                secular_pn_eps_j(x0, v0, s.M, s.nu, s.c, 3, eps, j);
                const double n = secular_pn_n(eps, j, s.nu, s.M, s.c, 3), k = secular_pn_k(eps, j, s.nu, 3);
                resn[t] = std::fabs(R.P * n / (2.0 * M_PI) - 1.0);
                resk[t] = std::fabs(R.dphi / (2.0 * M_PI * k) - 1.0) * k;   // in units of 2 pi per orbit
            }
            std::printf("q %.2f e %.1f: n residual %.2e ratio %.2f, K residual %.2e ratio %.2f (8)\n", q, e, resn[0],
                        resn[0] / resn[1], resk[0], resk[0] / resk[1]);
            check(std::fabs(resn[0] / resn[1] / 8.0 - 1.0) < 0.15, "radial period differs from n(eps, j) at 2PN");
            check(std::fabs(resk[0] / resk[1] / 8.0 - 1.0) < 0.15, "periastron advance differs from K(eps, j) at 2PN");
        }
    }
    // (2) radiation reaction on eccentric orbits
    for (double q : {1.0, 0.25}) {
        for (double e : {0.3, 0.6}) {
            double rese[2], resj[2];
            for (int t = 0; t < 2; t++) {
                const double gam = t == 0 ? 2e-3 : 1e-3;
                Sys s{1.0 / (1.0 + q), q / (1.0 + q), 1.0, q / ((1.0 + q) * (1.0 + q)), 0.0};
                const double rp = 1.0;
                s.c = std::sqrt(s.M / (rp * gam));
                const OrbitResult R = radial_orbit(s, rp, e, 1, 4 | 16, 1);
                double eps, j, de, dj, de0, dj0;
                double x0[3] = {rp, 0, 0}, v0[3] = {0, std::sqrt(s.M * (1.0 + e) / rp), 0};
                secular_pn_eps_j(x0, v0, s.M, s.nu, s.c, 1, eps, j);
                secular_pn_rates(eps, j, s.nu, s.M, s.c, 1 | 4 | 16, de, dj);
                secular_pn_rates(eps, j, s.nu, s.M, s.c, 4, de0, dj0);
                rese[t] = std::fabs(R.deps - de) / std::fabs(de0);
                resj[t] = std::fabs(R.dj - dj) / std::fabs(dj0);
                if (t == 0)
                    std::printf("q %.2f e %.1f: <deps/dt> EOM / secular %.8f (PN2.5 alone %.8f), <dj/dt> %.8f (%.8f)\n", q, e,
                                R.deps / de, R.deps / de0, R.dj / dj, R.dj / dj0);
            }
            std::printf("    residual relative to PN2.5: eps %.2e ratio %.2f, j %.2e ratio %.2f (4)\n", rese[0],
                        rese[0] / rese[1], resj[0], resj[0] / resj[1]);
            check(std::fabs(rese[0] / rese[1] / 4.0 - 1.0) < 0.25, "averaged energy loss differs from the 1PN flux");
            check(std::fabs(resj[0] / resj[1] / 4.0 - 1.0) < 0.25, "averaged angular-momentum loss differs from the 1PN flux");
        }
    }
    // (3) secular-triple closed forms against the chain rule through secular_pn_rates
    for (double q : {1.0, 0.25}) {
        for (double e : {0.3, 0.6}) {
            double resa[2], rese[2];
            for (int t = 0; t < 2; t++) {
                const double gam = t == 0 ? 2e-3 : 1e-3;   // G m / (a c^2)
                const double nu = q / ((1.0 + q) * (1.0 + q)), Gm = 1.0, a = 1.0, c = std::sqrt(Gm / (a * gam));
                const double n = std::sqrt(Gm / (a * a * a));
                // eps, j with n(eps, j) = n (1PN) and e_t^2(eps, j) = e^2 (1PN): Newton with a numerical Jacobian
                double eps = gam, j = 1.0 - e * e;
                auto fun = [&](double ep, double jj, double& f1, double& f2) {
                    f1 = secular_pn_n(ep, jj, nu, Gm, c, 1) / n - 1.0;
                    f2 = secular_pn_et2(ep, jj, nu, 16) - e * e;
                };
                double J11 = 0, J12 = 0, J21 = 0, J22 = 0;
                auto jac = [&](double ep, double jj) {
                    double f1, f2, g1, g2, h1, h2;
                    const double de = 1e-6 * ep, dj = 1e-6;
                    fun(ep, jj, f1, f2); fun(ep + de, jj, g1, g2); fun(ep, jj + dj, h1, h2);
                    J11 = (g1 - f1) / de; J21 = (g2 - f2) / de; J12 = (h1 - f1) / dj; J22 = (h2 - f2) / dj;
                };
                for (int it = 0; it < 30; it++) {
                    double f1, f2;
                    fun(eps, j, f1, f2);
                    jac(eps, j);
                    const double det = J11 * J22 - J12 * J21;
                    eps -= (J22 * f1 - J12 * f2) / det;
                    j -= (-J21 * f1 + J11 * f2) / det;
                }
                double deps, dj;
                secular_pn_rates(eps, j, nu, Gm, c, 1 | 4 | 16, deps, dj);
                jac(eps, j);   // d(n / n0 - 1, e_t^2) / d(eps, j)
                const double ndot = n * (J11 * deps + J12 * dj), et2dot = J21 * deps + J22 * dj;
                const double da_chain = -2.0 / 3.0 * a * ndot / n, de_chain = et2dot / (2.0 * e);
                double da, de, dw, da0, de0;
                secular_pn_inner_rates(a, e, nu, Gm, c, 1 | 4 | 16, da, de, dw);
                secular_pn_inner_rates(a, e, nu, Gm, c, 4, da0, de0, dw);
                resa[t] = std::fabs(da - da_chain) / std::fabs(da0);
                rese[t] = std::fabs(de - de_chain) / std::fabs(de0);
                if (t == 0) {
                    // PN2.5 alone: Peters (1964)
                    const double e2 = e * e, b = 64.0 / 5.0 * nu * std::pow(Gm, 3) / std::pow(c, 5);
                    const double pa = -b / (a * a * a) * (1.0 + 73.0/24.0 * e2 + 37.0/96.0 * e2 * e2) / std::pow(1.0 - e2, 3.5);
                    const double pe = -19.0/12.0 * b / (a * a * a * a) * e * (1.0 + 121.0/304.0 * e2) / std::pow(1.0 - e2, 2.5);
                    std::printf("q %.2f e %.1f: triple PN2.5 / Peters %.15f %.15f\n", q, e, da0 / pa, de0 / pe);
                    check(std::fabs(da0 / pa - 1.0) < 1e-13 && std::fabs(de0 / pe - 1.0) < 1e-13, "triple PN2.5 differs from Peters");
                }
            }
            std::printf("    triple 1PN rates vs chain rule, relative to PN2.5: a %.2e ratio %.2f, e %.2e ratio %.2f (4)\n",
                        resa[0], resa[0] / resa[1], rese[0], rese[0] / rese[1]);
            check(std::fabs(resa[0] / resa[1] / 4.0 - 1.0) < 0.25 && std::fabs(rese[0] / rese[1] / 4.0 - 1.0) < 0.25,
                  "secular-triple 1PN rates differ from the fluxes");
        }
    }
    // (4) integrate_secular_binary (PN1 + PN2) over 400 orbits against the integrated orbit. Both final states are
    // integrated on to their next pericentre: offsets in time and angle stay bounded at O(x) (the Newtonian orbit shape
    // differs from the PN orbit: e_t, e_r, e_phi differ at 1PN); with a wrong n or K they grow with the orbits
    {
        const double q = 0.5, e = 0.5, gam = 1e-3, a = 1.0;   // G m / (a c^2)
        Sys s{1.0 / (1.0 + q), q / (1.0 + q), 1.0, q / ((1.0 + q) * (1.0 + q)), std::sqrt(1.0 / (a * gam))};
        const double rp = a * (1.0 - e), P = 2.0 * M_PI * std::sqrt(a * a * a / s.M);
        double x[3] = {rp, 0, 0}, v[3] = {0, std::sqrt(s.M * (1.0 + e) / rp), 0};
        const double T = 400.0 * P;
        double p1[3], p2[3], v1[3], v2[3];
        for (int k = 0; k < 3; k++) {
            p1[k] = s.m2 * x[k]; p2[k] = -s.m1 * x[k]; v1[k] = s.m2 * v[k]; v2[k] = -s.m1 * v[k];
        }
        integrate_secular_binary(p1, v1, s.m1, p2, v2, s.m2, 1.0, s.c, 1 | 2, T);
        const int nstep = 800000;
        for (int n = 0; n < nstep; n++) rk4(s, x, v, T / nstep, 1 | 2);
        auto peri = [&](double* xx, double* vv, double& tp, double& ph) {
            const double h = P / 4000.0;
            for (int n = 0; n < 8000; n++) {
                const double xo[3] = {xx[0], xx[1], xx[2]}, vo[3] = {vv[0], vv[1], vv[2]};
                const double rdo = xo[0] * vo[0] + xo[1] * vo[1];
                rk4(s, xx, vv, h, 1 | 2);
                const double rdn = xx[0] * vv[0] + xx[1] * vv[1];
                if (rdo < 0.0 && rdn >= 0.0) {
                    const double f = -rdo / (rdn - rdo);
                    tp = (n + f) * h;
                    ph = std::atan2(xo[1] + f * (xx[1] - xo[1]), xo[0] + f * (xx[0] - xo[0]));
                    return;
                }
            }
        };
        double xs[3] = {p1[0] - p2[0], p1[1] - p2[1], 0.0}, vs[3] = {v1[0] - v2[0], v1[1] - v2[1], 0.0};
        double ts = 0, ps = 0, tr = 0, pr = 0;
        peri(xs, vs, ts, ps);
        peri(x, v, tr, pr);
        const double dt = std::remainder(ts - tr, P) / P, dphi = std::remainder(ps - pr, 2.0 * M_PI);
        std::printf("secular binary after 400 orbits: pericentre time offset %.3e P, angle offset %.3e rad\n", dt, dphi);
        check(std::fabs(dt) < 1.3e-3 && std::fabs(dphi) < 3.4e-3, "secular binary phase drifts from the PN orbit");
    }
    // (5) write-back near the pericentre (C19), cluster units (pc, km/s, Msun)
    {
        const double G = 4.302194e-3, c = 299792.458, TU = 0.977792, AU = 1.0 / 206264.8;
        double m1 = 14.694904891331355, m2 = 28.264946521763406;
        double p1[3] = {-3.9550400154575258, 2.2288385523949472, 0.89460482231650418};
        double v1[3] = {-22.459788311977473, 15.07638740297981, -3.0364789183732142};
        double p2[3] = {-3.9550192408087899, 2.2288195873655932, 0.8946205442104429};
        double v2[3] = {0.88019094481315097, -5.2667128343914067, 12.101863391992737};
        const double q0[12] = {p1[0], p1[1], p1[2], v1[0], v1[1], v1[2], p2[0], p2[1], p2[2], v2[0], v2[1], v2[2]};
        const int n = integrate_secular_binary(p1, v1, m1, p2, v2, m2, G, c, 31, 3.125e-5 / TU);
        const double q1[12] = {p1[0], p1[1], p1[2], v1[0], v1[1], v1[2], p2[0], p2[1], p2[2], v2[0], v2[1], v2[2]};
        bool same = true;
        for (int k = 0; k < 12; k++) same = same && q0[k] == q1[k];
        double r[3], v[3];
        for (int k = 0; k < 3; k++) { r[k] = q0[6 + k] - q0[k]; v[k] = q0[9 + k] - q0[3 + k]; }
        const double M = m1 + m2, nu = m1 * m2 / (M * M);
        const double shift = secular_pn_peri_shift(r, v, G * M, nu, c, 31);
        std::printf("C19 binary: peri shift %.3f, integrate_secular_binary %d, state %s\n", shift, n, same ? "unchanged" : "changed");
        check(n == -1 && same, "C19 binary: secular write-back at the pericentre not refused");

        m1 = 14.69; m2 = 28.26;
        const double Mt = m1 + m2, mu = G * Mt, nut = m1 * m2 / (Mt * Mt);
        unsigned long long rng = 12345;
        auto U = [&]() { rng = rng * 6364136223846793005ULL + 1442695040888963407ULL; return (rng >> 11) * 0x1.0p-53; };
        double worst = 0.0;
        int admitted = 0, refused = 0, unbound = 0;
        for (double a_au : {0.36, 3.6, 36.0})
            for (double om : {1e-1, 1e-2, 3e-3, 1e-3, 6.4e-4, 3e-4})
                for (int t = 0; t < 200; t++) {
                    const double a = a_au * AU, e = 1.0 - om;
                    double x[3], w[3];
                    elements_to_cartesian(a, e, 0.3, 1.0, 2.0, 2.0 * M_PI * U(), mu, x, w);
                    const double P = 2.0 * M_PI * std::sqrt(a * a * a / mu), tp = std::sqrt(std::pow(a * om, 3) / mu);
                    const double dt = kepler_time_to_pericentre(x, w, mu) + P * (3 + (int)(30 * U())) + 10.0 * tp * (U() - 0.5);
                    SecularPnState st;
                    secular_pn_eps_j(x, w, mu, nut, c, 31, st.eps, st.j);
                    secular_evolve_pn(st, dt, nut, mu, c, 31);
                    double a1[3], b1[3], a2[3], b2[3];
                    for (int k = 0; k < 3; k++) {
                        a1[k] = -m2 / Mt * x[k]; a2[k] = m1 / Mt * x[k]; b1[k] = -m2 / Mt * w[k]; b2[k] = m1 / Mt * w[k];
                    }
                    const int ns = integrate_secular_binary(a2, b2, m1, a1, b1, m2, G, c, 31, dt);
                    if (ns < 0) { refused++; continue; }
                    if (st.merged) continue;
                    admitted++;
                    double xr[3], wr[3];
                    for (int k = 0; k < 3; k++) { xr[k] = a2[k] - a1[k]; wr[k] = b2[k] - b1[k]; }
                    const double E = 0.5 * (wr[0]*wr[0] + wr[1]*wr[1] + wr[2]*wr[2]) - mu / std::sqrt(xr[0]*xr[0] + xr[1]*xr[1] + xr[2]*xr[2]);
                    if (!(E < 0.0)) { unbound++; continue; }
                    double eps, j;
                    secular_pn_eps_j(xr, wr, mu, nut, c, 31, eps, j);
                    worst = std::max(worst, std::fabs(eps / st.eps - 1.0));
                }
        std::printf("pericentre write-back: %d admitted, %d refused, %d unbound, max |eps / eps_evolved - 1| %.3e\n",
                    admitted, refused, unbound, worst);
        // 3.71e-10 measured (round-off of eps at the pericentre, ~1e-16 / (1 - e) per term, and the fixed point); before
        // the fix this grid (other phases) wrote 10 of 7200 pairs back unbound, all at peri shift >= 1
        check(unbound == 0 && worst < 4.8e-10, "pericentre write-back: unbound or eps off the evolved one");
    }
    return fails;
}
