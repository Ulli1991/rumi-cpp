// Secular triple (src/collisional/secular.cpp secular_evolve_triple, src/collisional/secular_triple_step.cpp) against
// Hamilton's equations of the double-averaged quadrupole + octupole Hamiltonian it integrates (Naoz et al. 2013 app. A,
// Blaes et al. 2002 eq. 41; tests/secular_triple_ham.h), by long double central differences of the Hamiltonian:
//  (1) rates de1/dt, dg1/dt, dg2/dt, de2/dt, d cos i/dt of one internal (Euler) step against dg/dt = -dH/dG,
//      dG/dt = dH/dg (Naoz sign), e = sqrt(1 - G^2/L^2), cos i = (Gtot^2 - G1^2 - G2^2)/(2 G1 G2): quadrupole alone
//      (m1 = m2, C3 = 0) and with the octupole, 2000 random states each;
//  (2) the averaged energy along the code's rates: dH/dt = sum dH/dq dq/dt = 0;
//  (3) frame independence of secular_triple_step: the same triple in a rotated frame must give the rotated result;
//  (4) the PN terms as stated: GW emission alone (outer body far away) changes |G1| but not the plane, so cos i stays
//      (Blaes et al. 2002 eq. 37); the 1PN precession adds 3 (G m)^(3/2) / (c^2 a^(5/2) (1 - e^2)) to dg1/dt;
//  (5) secular_triple_step against a direct three-body integration (RK4) over two outer periods: the change of the
//      orbit-averaged inner eccentricity, a1/a2 = 0.04, e2 = 0.6, m = (1, 0.3, 1), four (g1, i) with g1 = 0 among them
//      (quadrupole de1/dt = 0 at the start). Residual of a correct secular model (higher multipoles, short-period
//      terms): <= 19 per cent measured; the code before the fix: up to 3.7x the direct change.
// Found 2026-10-17 (with tests/enzyme/secular_triple.cpp): the octupole rates are not Hamiltonian (the code has
// cos phi = cos g1 cos g2 - cos i sin g1 sin g2 and C3 ~ (m2 - m1) with Blaes' signs; Blaes / Naoz: cos phi =
// -cos g1 cos g2 - cos i sin g1 sin g2, C3 ~ (m1 - m2) with Blaes' signs), de2/dt (octupole) is missing, the GW loss of
// G1 tilts the orbits, and g1, g2 are taken from the simulation frame's xy-plane node instead of the mutual node.
#include "secular.h"
#include "secular_triple_step.h"
#include "kepler.h"
#include "secular_triple_ham.h"
#include <cmath>
#include <cstdio>
#include <random>

typedef long double ld;
static int fails = 0;
static void check(bool ok, const char* what, double val, double bound) {
    std::printf("%s %-58s %.3e (bound %.1e)\n", ok ? "ok  " : "FAIL", what, val, bound);
    if (!ok) fails++;
}

// fourth-order central difference with step h (G near L: h ~ L - G, the e^2 = 1 - G^2/L^2 scale)
template <class F> static ld dcen(F f, ld x, ld h) {
    return (8 * (f(x + h) - f(x - h)) - (f(x + 2 * h) - f(x - 2 * h))) / (12 * h);
}

// (1), (2): one Euler step of secular_evolve_triple (pn_order 0) against Hamilton's equations
static void rates_test(bool octupole, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double mx[5] = {0, 0, 0, 0, 0}, mxH = 0.0;
    for (int t = 0; t < 2000; t++) {
        const double m1 = 0.1 + 2.0 * U(rng), m3 = 0.1 + 2.0 * U(rng);
        const double m2 = octupole ? 0.1 + 2.0 * U(rng) : m1;
        const double a2 = 1.0, a1 = 0.01 + 0.09 * U(rng), e1 = 0.05 + 0.9 * U(rng), e2 = 0.05 + 0.85 * U(rng);
        const double ci = -0.99 + 1.98 * U(rng), g1 = 2.0 * M_PI * U(rng), g2 = 2.0 * M_PI * U(rng);
        const double M12 = m1 + m2, Mt = M12 + m3;
        TripleHamParams p{m1, m2, m3, 0.0, 0.0, 0.0, 1, 1};
        const ld L1 = (ld)m1 * m2 / M12 * std::sqrt((ld)M12 * a1), L2 = (ld)M12 * m3 / Mt * std::sqrt((ld)Mt * a2);
        const ld G1 = L1 * std::sqrt(1 - (ld)e1 * e1), G2 = L2 * std::sqrt(1 - (ld)e2 * e2);
        const ld Gt = std::sqrt(G1 * G1 + G2 * G2 + 2 * G1 * G2 * ci);
        p.L1 = (double)L1; p.L2 = (double)L2; p.Gtot = (double)Gt;
        auto H = [&](ld a, ld b, ld c, ld d) { return triple_hamiltonian<ld>(p, a, b, c, d); };
        const ld dHdG1 = dcen([&](ld x) { return H(g1, x, g2, G2); }, G1, 1e-4L * (L1 - G1));
        const ld dHdG2 = dcen([&](ld x) { return H(g1, G1, g2, x); }, G2, 1e-4L * (L2 - G2));
        const ld dHdg1 = dcen([&](ld x) { return H(x, G1, g2, G2); }, (ld)g1, 1e-4L);
        const ld dHdg2 = dcen([&](ld x) { return H(g1, G1, x, G2); }, (ld)g2, 1e-4L);
        const ld rg1 = -dHdG1, rg2 = -dHdG2, rG1 = dHdg1, rG2 = dHdg2;
        const ld re1 = -G1 / (L1 * L1 * e1) * rG1, re2 = -G2 / (L2 * L2 * e2) * rG2;
        const ld rci = (-1 / G2 - ci / G1) * rG1 + (-1 / G1 - ci / G2) * rG2;
        const ld scale = std::fabs(rg1) + std::fabs(rg2) + std::fabs(re1) + std::fabs(re2);

        SecularTriple tri{};
        tri.inner.a = a1; tri.inner.e = e1; tri.inner.omega = g1; tri.inner.m1 = m1; tri.inner.m2 = m2;
        tri.a_out = a2; tri.e_out = e2; tri.omega_out = g2; tri.m3 = m3; tri.cos_i_mut = ci;
        const SecularTriple t0 = tri;
        // one internal Euler step (dt << 0.01 t_KL), changes ~1e-6: e, g linear in dt; cos i (from Gtot) by
        // Richardson extrapolation of dt and 2 dt
        const double dt = 1e-6 / (double)scale;
        SecularTriple t2 = tri;
        secular_evolve_triple(tri, dt, 1.0, 1e30, 0);
        secular_evolve_triple(t2, 2.0 * dt, 1.0, 1e30, 0);
        const ld ce1 = ((ld)tri.inner.e - t0.inner.e) / dt, cg1 = ((ld)tri.inner.omega - t0.inner.omega) / dt;
        const ld cg2 = ((ld)tri.omega_out - t0.omega_out) / dt, ce2 = ((ld)tri.e_out - t0.e_out) / dt;
        const ld cci = 2 * ((ld)tri.cos_i_mut - t0.cos_i_mut) / dt - ((ld)t2.cos_i_mut - t0.cos_i_mut) / (2 * dt);
        const ld d[5] = {ce1 - re1, cg1 - rg1, cg2 - rg2, ce2 - re2, cci - rci};
        for (int k = 0; k < 5; k++) mx[k] = std::max(mx[k], (double)(std::fabs(d[k]) / scale));
        // (2) averaged energy along the code's rates, dH/dt = sum dH/dq dq/dt (q = g1, e1, g2, e2), relative to
        // sum |dH/dq| times the rate scale
        const ld dHde1 = -L1 * L1 * e1 / G1 * dHdG1, dHde2 = -L2 * L2 * e2 / G2 * dHdG2;
        const ld sum = dHdg1 * cg1 + dHde1 * ce1 + dHdg2 * cg2 + dHde2 * ce2;
        const ld den = scale * (std::fabs(dHdg1) + std::fabs(dHde1) + std::fabs(dHdg2) + std::fabs(dHde2));
        mxH = std::max(mxH, (double)(std::fabs(sum) / den));
    }
    // bounds: central differences in long double ~5e-8 (g1 at small e1), one-step rounding of changes ~1e-6: 1e-10
    const char* tag = octupole ? "octupole  " : "quadrupole";
    char s[128];
    const char* name[5] = {"de1/dt", "dg1/dt", "dg2/dt", "de2/dt", "dcos i/dt"};
    for (int k = 0; k < 5; k++) {
        std::snprintf(s, sizeof s, "%s %s vs Hamilton (max |diff| / sum |rates|)", tag, name[k]);
        check(mx[k] < 1e-6, s, mx[k], 1e-6);
    }
    std::snprintf(s, sizeof s, "%s dH_avg/dt along the code's rates (relative)", tag);
    check(mxH < 1e-6, s, mxH, 1e-6);
}

// rotate a 3-vector by the rotation matrix R
static void rot(const double R[3][3], const double x[3], double y[3]) {
    for (int i = 0; i < 3; i++) y[i] = R[i][0] * x[0] + R[i][1] * x[1] + R[i][2] * x[2];
}

// (3) frame independence of secular_triple_step: an octupole-active triple stepped in two frames
static void frame_test(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const double G = 1.0;
    double worst = 0.0;
    for (int t = 0; t < 20; t++) {
        double m[3] = {1.0, 0.4 + 0.5 * U(rng), 0.5 + U(rng)};
        const double a1 = 0.05, e1 = 0.2 + 0.6 * U(rng), a2 = 1.0, e2 = 0.2 + 0.5 * U(rng);
        const double M12 = m[0] + m[1], Mt = M12 + m[2];
        double r1[3], v1[3], r2[3], v2[3];
        elements_to_cartesian(a1, e1, 0.3 + 1.2 * U(rng), 2 * M_PI * U(rng), 2 * M_PI * U(rng), 2 * M_PI * U(rng),
                              G * M12, r1, v1);
        elements_to_cartesian(a2, e2, 0.2 * U(rng), 2 * M_PI * U(rng), 2 * M_PI * U(rng), 2 * M_PI * U(rng),
                              G * Mt, r2, v2);
        double X[3][3], V[3][3];
        for (int k = 0; k < 3; k++) {
            X[0][k] = -m[2] / Mt * r2[k] - m[1] / M12 * r1[k]; V[0][k] = -m[2] / Mt * v2[k] - m[1] / M12 * v1[k];
            X[1][k] = -m[2] / Mt * r2[k] + m[0] / M12 * r1[k]; V[1][k] = -m[2] / Mt * v2[k] + m[0] / M12 * v1[k];
            X[2][k] = M12 / Mt * r2[k]; V[2][k] = M12 / Mt * v2[k];
        }
        // random rotation (axis, angle)
        const double th = 2 * M_PI * U(rng), z = 2 * U(rng) - 1, ph = 2 * M_PI * U(rng);
        const double ax[3] = {std::sqrt(1 - z * z) * std::cos(ph), std::sqrt(1 - z * z) * std::sin(ph), z};
        const double c = std::cos(th), s = std::sin(th);
        double R[3][3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                R[i][j] = (i == j ? c : 0.0) + (1 - c) * ax[i] * ax[j];
        R[0][1] -= s * ax[2]; R[0][2] += s * ax[1]; R[1][0] += s * ax[2];
        R[1][2] -= s * ax[0]; R[2][0] -= s * ax[1]; R[2][1] += s * ax[0];
        double XR[3][3], VR[3][3], M[3] = {m[0], m[1], m[2]}, MR[3] = {m[0], m[1], m[2]};
        for (int b = 0; b < 3; b++) { rot(R, X[b], XR[b]); rot(R, V[b], VR[b]); }
        // a step of a few per cent of the Kozai time t_KL ~ P2^2 / P1 * Mt / m3
        const double P1 = 2 * M_PI * std::sqrt(a1 * a1 * a1 / (G * M12)), P2 = 2 * M_PI * std::sqrt(a2 * a2 * a2 / (G * Mt));
        const double dt = 0.03 * P2 * P2 / P1 * Mt / m[2];
        const int pids[3] = {0, 1, 2};
        secular_triple_step(M, X, V, pids, dt, G, 1e30, 0);
        secular_triple_step(MR, XR, VR, pids, dt, G, 1e30, 0);
        // inner eccentricity vector after the step in both frames (the rotated one rotated back by R^T)
        auto evec = [&](double x[3][3], double v[3][3], double e[3]) {
            double r[3], u[3], h[3];
            for (int k = 0; k < 3; k++) { r[k] = x[1][k] - x[0][k]; u[k] = v[1][k] - v[0][k]; }
            h[0] = r[1] * u[2] - r[2] * u[1]; h[1] = r[2] * u[0] - r[0] * u[2]; h[2] = r[0] * u[1] - r[1] * u[0];
            const double rr = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
            e[0] = (u[1] * h[2] - u[2] * h[1]) / (G * M12) - r[0] / rr;
            e[1] = (u[2] * h[0] - u[0] * h[2]) / (G * M12) - r[1] / rr;
            e[2] = (u[0] * h[1] - u[1] * h[0]) / (G * M12) - r[2] / rr;
        };
        double ea[3], eb[3], ebr[3];
        evec(X, V, ea);
        evec(XR, VR, eb);
        double RT[3][3];
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) RT[i][j] = R[j][i];
        rot(RT, eb, ebr);
        const double de = std::sqrt((ea[0] - ebr[0]) * (ea[0] - ebr[0]) + (ea[1] - ebr[1]) * (ea[1] - ebr[1])
                                    + (ea[2] - ebr[2]) * (ea[2] - ebr[2]));
        worst = std::max(worst, de);
    }
    check(worst < 1e-9, "frame: |e_vec(frame A) - R^T e_vec(frame R A)| after a step", worst, 1e-9);
}

// (4) PN terms: GW alone keeps cos i; the 1PN precession of g1
static void pn_test() {
    const double G = 1.0, c = 30.0;   // G m / (a c^2) ~ 1e-2 at a1 = 0.1, m = 1
    SecularTriple tri{};
    tri.inner.a = 0.1; tri.inner.e = 0.6; tri.inner.omega = 1.0; tri.inner.m1 = 0.6; tri.inner.m2 = 0.4;
    tri.a_out = 1e6; tri.e_out = 0.3; tri.omega_out = 0.5; tri.m3 = 1.0; tri.cos_i_mut = 0.3;   // tidal ratio ~1e-21
    const double M12 = 1.0;
    {   // GW: integrate until G1 dropped by ~1 per cent
        SecularTriple t = tri;
        const double tgw = 0.1 * 0.1 * 0.1 * 0.1 * std::pow(c, 5) * 5.0 / (256.0 * 0.24);   // ~ a^4 / (da^4/dt)
        secular_evolve_triple(t, 3e-3 * tgw, G, c, 4);
        const double j0 = std::sqrt(1 - tri.inner.e * tri.inner.e), j1 = std::sqrt(1 - t.inner.e * t.inner.e);
        const double dG1 = std::sqrt(t.inner.a) * j1 / (std::sqrt(tri.inner.a) * j0) - 1.0;
        std::printf("     GW only: dG1/G1 %.3e, cos i %.6f -> %.6f\n", dG1, tri.cos_i_mut, t.cos_i_mut);
        check(std::fabs(t.cos_i_mut - tri.cos_i_mut) < 1e-6 * std::fabs(dG1),
              "GW alone: |d cos i| / |dG1/G1| (Blaes eq. 37: plane fixed)",
              std::fabs(t.cos_i_mut - tri.cos_i_mut) / std::fabs(dG1), 1e-6);
    }
    {   // 1PN precession: dg1(pn 1) - dg1(pn 0) from one tiny step
        SecularTriple a = tri, b = tri;
        const double dt = 1e-3;
        secular_evolve_triple(a, dt, G, c, 1);
        secular_evolve_triple(b, dt, G, c, 0);
        const double want = 3.0 * std::pow(G * M12, 1.5) / (c * c * std::pow(0.1, 2.5) * (1 - 0.36));
        const double got = (a.inner.omega - b.inner.omega) / dt;
        check(std::fabs(got / want - 1.0) < 1e-9, "1PN: dg1/dt added = 3 (Gm)^1.5 / (c^2 a^2.5 (1-e^2))",
              std::fabs(got / want - 1.0), 1e-9);
    }
}

// (5) secular_triple_step against direct integration
static void acc3(const double m[3], const double x[3][3], double a[3][3]) {
    for (int i = 0; i < 3; i++) for (int k = 0; k < 3; k++) a[i][k] = 0.0;
    for (int i = 0; i < 3; i++)
        for (int j = i + 1; j < 3; j++) {
            double d[3], r2 = 0.0;
            for (int k = 0; k < 3; k++) { d[k] = x[j][k] - x[i][k]; r2 += d[k] * d[k]; }
            const double ir3 = 1.0 / (r2 * std::sqrt(r2));
            for (int k = 0; k < 3; k++) { a[i][k] += m[j] * d[k] * ir3; a[j][k] -= m[i] * d[k] * ir3; }
        }
}
static double ecc_in(const double m[3], const double x[3][3], const double v[3][3]) {
    double r[3], u[3];
    for (int k = 0; k < 3; k++) { r[k] = x[1][k] - x[0][k]; u[k] = v[1][k] - v[0][k]; }
    return cartesian_to_elements(r, u, m[0] + m[1]).e;   // G = 1
}
static void direct_test() {
    double worst = 0.0;
    for (int cas = 0; cas < 4; cas++) {
        const double m[3] = {1.0, 0.3, 1.0}, a1 = 0.04, e1 = 0.4, a2 = 1.0, e2 = 0.6;
        const double inc_mut = (cas % 2) ? 1.2 : 0.6, g1 = (cas < 2) ? 0.0 : 0.5, g2 = 1.0;
        const double M12 = m[0] + m[1], Mt = M12 + m[2];
        // outer plane (inc 0.4, Omega 0.7); mutual node n1 at 0.9 from the outer node in that plane; inner angular
        // momentum Li = Lo cos i + (n1 x Lo) sin i, so that n1 is along Lo x Li; pericentres at g1 from n1, g2 from -n1
        const double io = 0.4, Oo = 0.7;
        const double Lo[3] = {std::sin(io) * std::sin(Oo), -std::sin(io) * std::cos(Oo), std::cos(io)};
        const double No[3] = {std::cos(Oo), std::sin(Oo), 0.0};
        double Qo[3], n1[3], n1xLo[3], Li[3], P1[3], P2[3];
        auto cross = [](const double* a, const double* b, double* c) {
            c[0] = a[1]*b[2] - a[2]*b[1]; c[1] = a[2]*b[0] - a[0]*b[2]; c[2] = a[0]*b[1] - a[1]*b[0];
        };
        cross(Lo, No, Qo);
        for (int k = 0; k < 3; k++) n1[k] = std::cos(0.9) * No[k] + std::sin(0.9) * Qo[k];
        cross(n1, Lo, n1xLo);
        for (int k = 0; k < 3; k++) Li[k] = Lo[k] * std::cos(inc_mut) + n1xLo[k] * std::sin(inc_mut);
        auto peri = [&](const double* L, const double* n, double g, double* e) {
            double q[3];
            cross(L, n, q);
            for (int k = 0; k < 3; k++) e[k] = std::cos(g) * n[k] + std::sin(g) * q[k];
        };
        const double mn1[3] = {-n1[0], -n1[1], -n1[2]};
        peri(Li, n1, g1, P1);
        peri(Lo, mn1, g2, P2);
        auto state = [&](const double* L, const double* P, double a, double e, double M, double mu, double* r, double* v) {
            double Q[3];
            cross(L, P, Q);
            double E = M;
            for (int it = 0; it < 60; it++) E = M + e * std::sin(E);
            const double n = std::sqrt(mu / (a * a * a)), dn = 1.0 - e * std::cos(E), b = std::sqrt(1.0 - e * e);
            for (int k = 0; k < 3; k++) {
                r[k] = a * (std::cos(E) - e) * P[k] + a * b * std::sin(E) * Q[k];
                v[k] = a * n / dn * (-std::sin(E) * P[k] + b * std::cos(E) * Q[k]);
            }
        };
        double r1[3], v1[3], r2[3], v2[3];
        state(Li, P1, a1, e1, 0.3, M12, r1, v1);
        state(Lo, P2, a2, e2, 2.0, Mt, r2, v2);
        double X[3][3], V[3][3];
        for (int k = 0; k < 3; k++) {
            X[0][k] = -m[2] / Mt * r2[k] - m[1] / M12 * r1[k]; V[0][k] = -m[2] / Mt * v2[k] - m[1] / M12 * v1[k];
            X[1][k] = -m[2] / Mt * r2[k] + m[0] / M12 * r1[k]; V[1][k] = -m[2] / Mt * v2[k] + m[0] / M12 * v1[k];
            X[2][k] = M12 / Mt * r2[k]; V[2][k] = M12 / Mt * v2[k];
        }
        const double Pout = 2 * M_PI * std::sqrt(a2 * a2 * a2 / Mt), Pin = 2 * M_PI * std::sqrt(a1 * a1 * a1 / M12);
        const double T = 2.0 * Pout;
        double Xs[3][3], Vs[3][3], Ms[3] = {m[0], m[1], m[2]};
        for (int i = 0; i < 3; i++) for (int k = 0; k < 3; k++) { Xs[i][k] = X[i][k]; Vs[i][k] = V[i][k]; }
        const int pids[3] = {0, 1, 2};
        const double e_start = ecc_in(m, X, V);
        secular_triple_step(Ms, Xs, Vs, pids, T, 1.0, 1e30, 0);
        const double de_sec = ecc_in(m, Xs, Vs) - e_start;
        // RK4, 400 steps per inner orbit; e averaged over the first and the last inner orbit
        const long ns = (long)std::ceil(T / (Pin / 400.0)), navg = 400;
        const double h = T / ns;
        double avg0 = 0.0, avg1 = 0.0;
        for (long st = 0; st < ns; st++) {
            if (st < navg) avg0 += ecc_in(m, X, V) / navg;
            if (st >= ns - navg) avg1 += ecc_in(m, X, V) / navg;
            double kx[4][3][3], kv[4][3][3], Xt[3][3], Vt[3][3];
            for (int s = 0; s < 4; s++) {
                const double f = (s == 0) ? 0.0 : (s == 3 ? 1.0 : 0.5);
                for (int i = 0; i < 3; i++) for (int k = 0; k < 3; k++) {
                    Xt[i][k] = X[i][k] + (s ? f * h * kx[s - 1][i][k] : 0.0);
                    Vt[i][k] = V[i][k] + (s ? f * h * kv[s - 1][i][k] : 0.0);
                    kx[s][i][k] = Vt[i][k];
                }
                acc3(m, Xt, kv[s]);
            }
            for (int i = 0; i < 3; i++) for (int k = 0; k < 3; k++) {
                X[i][k] += h / 6 * (kx[0][i][k] + 2 * kx[1][i][k] + 2 * kx[2][i][k] + kx[3][i][k]);
                V[i][k] += h / 6 * (kv[0][i][k] + 2 * kv[1][i][k] + 2 * kv[2][i][k] + kv[3][i][k]);
            }
        }
        const double de_dir = avg1 - avg0;
        std::printf("     direct: g1 %.1f i %.1f: de direct %+.4e secular %+.4e\n", g1, inc_mut, de_dir, de_sec);
        worst = std::max(worst, std::fabs(de_sec / de_dir - 1.0));
    }
    check(worst < 0.25, "direct 3-body: max |de_sec / de_direct - 1| over 2 P_out", worst, 0.25);
}

int main() {
    std::mt19937_64 rng(20261017);
    rates_test(false, rng);
    rates_test(true, rng);
    frame_test(rng);
    pn_test();
    direct_test();
    if (fails) std::printf("%d check(s) failed\n", fails);
    else std::printf("all secular triple checks passed\n");
    return fails ? 1 : 0;
}
