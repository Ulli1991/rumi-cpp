// Radiation-reaction energy balance with Enzyme gradients (CASES R4): the orbit-averaged power of the dissipative
// (2.5PN + 3.5PN) terms along a conservative 3PN orbit, <dE/dt>_RR = <grad E . flow_RR>, against the gauge-invariant
// orbit-averaged GW flux to relative 1PN (Peters & Mathews 1963 + Blanchet & Schaefer 1989 / Arun et al. 2008, as
// secular_pn_rates). Total time derivatives (Schott terms, gauge terms) average to zero over a radial period, so the
// balance must hold to O(c^-4) relative (2PN flux not included in either) for a consistent 3.5PN term, and fails at
// O(c^-2) if the 3.5PN term is wrong (doubling c divides the residual by 16 vs 4).
//  (a) production harmonic EOM: eih_pn_acceleration_pair (post_newtonian_core.cpp), PN_ORDER 31 minus 11, energy
//      E = harmonic c.o.m. energy to 2PN (Kidder 1995 Eq. 2.7, = secular_pn_eps_j), gradient by Enzyme;
//  (b) ADM: H_ADM to 3PN (tests/proto_admpn/adm_pn.h), RR force F = H_pp^-1 a_RR,KFS, power grad_p H . F with
//      grad_p H by Enzyme, and the Enzyme gradient of H checked against the hand-coded one per PN order.
// Build (repository root):
//   clang++ -std=c++20 -O2 -fplugin=<ClangEnzyme-18.so> -Isrc/collisional tests/enzyme/pn_rr_balance.cpp \
//       src/collisional/secular.cpp src/collisional/kepler.cpp -o pn_rr_balance
#include "../../src/collisional/post_newtonian_core.cpp"
#include "../../src/collisional/secular.h"
#include "../proto_admpn/adm_pn.h"
#include <algorithm>
#include <cstdio>

extern int enzyme_dup, enzyme_const, enzyme_out;
template <typename... T> void __enzyme_autodiff(void*, T...);

static double g_nu = 0.2, g_c = 30.0;

// harmonic c.o.m. energy per reduced mass to 2PN (Kidder 1995 Eq. 2.7), G M = 1
[[gnu::noinline]] static double energy_h(const double* x, const double* v) {
    const double r = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    const double v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    const double rd = (x[0] * v[0] + x[1] * v[1] + x[2] * v[2]) / r, rd2 = rd * rd;
    const double u = 1.0 / r, c2 = g_c * g_c, nu = g_nu, nu2 = nu * nu;
    double E = 0.5 * v2 - u;
    E += (3.0 / 8.0 * (1.0 - 3.0 * nu) * v2 * v2 + 0.5 * (3.0 + nu) * v2 * u + 0.5 * nu * u * rd2 + 0.5 * u * u) / c2;
    E += (5.0 / 16.0 * (1.0 - 7.0 * nu + 13.0 * nu2) * v2 * v2 * v2 - 3.0 / 8.0 * nu * (1.0 - 3.0 * nu) * u * rd2 * rd2
          + 1.0 / 8.0 * (21.0 - 23.0 * nu - 27.0 * nu2) * u * v2 * v2 + 1.0 / 8.0 * (14.0 - 55.0 * nu + 4.0 * nu2) * u * u * v2
          + 0.25 * nu * (1.0 - 15.0 * nu) * u * v2 * rd2 - 0.25 * (2.0 + 15.0 * nu) * u * u * u
          + 1.0 / 8.0 * (4.0 + 69.0 * nu + 12.0 * nu2) * u * u * rd2) / (c2 * c2);
    return E;
}

// production relative acceleration (c.o.m.: body 1 at m2/M x, body 2 at -m1/M x), G = 1, M = 1
static void accel_prod(const double* x, const double* v, int order, double* a) {
    const double m1 = 0.5 * (1 + std::sqrt(1 - 4 * g_nu)), m2 = 1 - m1;
    double x1[3], x2[3], v1[3], v2[3], a1[3], a2[3];
    for (int k = 0; k < 3; k++) { x1[k] = m2 * x[k]; x2[k] = -m1 * x[k]; v1[k] = m2 * v[k]; v2[k] = -m1 * v[k]; }
    eih_pn_acceleration_pair(x1, v1, m1, x2, v2, m2, 1.0, g_c, a1, a2, order);
    const double r = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    for (int k = 0; k < 3; k++) a[k] = -x[k] / (r * r * r) + a1[k] - a2[k];
}

// reduced ADM Hamiltonian (Newton + orders), G M = 1: a plain coefficient table (Enzyme-friendly), filled from
// admpn::PairHam (adm_pn.h)
static admpn::PairHam g_ph;
static double g_mc[32]; static int g_ma[32], g_mb[32], g_mp[32], g_mk[32], g_nm = 0;
static void load_table(double nu, int orders) {
    g_ph.build(nu, orders);
    g_nm = (int)g_ph.t.size();
    for (int i = 0; i < g_nm; i++) { g_mc[i] = g_ph.t[i].c; g_ma[i] = g_ph.t[i].a; g_mb[i] = g_ph.t[i].b; g_mp[i] = g_ph.t[i].pc; g_mk[i] = g_ph.t[i].k; }
}
[[gnu::noinline]] static double ham_adm(const double* x, const double* p) {
    const double r = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    const double u = 1.0 / r, p2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    const double pr = (x[0] * p[0] + x[1] * p[1] + x[2] * p[2]) / r, ic2 = 1.0 / (g_c * g_c);
    double H = 0.5 * p2 - u;
    for (int i = 0; i < g_nm; i++) {
        double t = g_mc[i];
        for (int k = 0; k < g_mk[i]; k++) t *= ic2;
        for (int k = 0; k < g_ma[i]; k++) t *= u;
        for (int k = 0; k < g_mb[i]; k++) t *= p2;
        for (int k = 0; k < g_mp[i]; k++) t *= pr;
        H += t;
    }
    return H;
}
[[gnu::noinline]] static void grad_ham_adm(const double* x, const double* p, double* gx, double* gp) {
    double xc[3] = {x[0], x[1], x[2]}, pc[3] = {p[0], p[1], p[2]};
    for (int k = 0; k < 3; k++) gx[k] = gp[k] = 0;
    __enzyme_autodiff((void*)ham_adm, enzyme_dup, xc, gx, enzyme_dup, pc, gp);
}
[[gnu::noinline]] static void grad_energy_h(const double* x, const double* v, double* gx, double* gv) {
    double xc[3] = {x[0], x[1], x[2]}, vc[3] = {v[0], v[1], v[2]};
    for (int k = 0; k < 3; k++) gx[k] = gv[k] = 0;
    __enzyme_autodiff((void*)energy_h, enzyme_dup, xc, gx, enzyme_dup, vc, gv);
}

// orbit-averaged RR power (harmonic, production) along the conservative 3PN harmonic orbit from apocentre
static void balance_harm(double e, int rr_bits, double& Pavg, double& Fgw) {
    double x[3] = {1 + e, 0, 0}, v[3] = {0, std::sqrt((1 - e) / (1 + e)), 0};
    double eps, jB;
    secular_pn_eps_j(x, v, 1.0, g_nu, g_c, 3, eps, jB);
    double de, dj; secular_pn_rates(eps, jB, g_nu, 1.0, g_c, 31, de, dj);
    Fgw = de * g_c * g_c / 2;   // <F>/mu, > 0
    // RK4 of the conservative EOM (11) until the second apocentre (rd: + -> -), power by Enzyme gradient
    const int cons = 11, full = 11 | rr_bits;
    auto power = [&](const double* xx, const double* vv) {
        double gx[3], gv[3];
        grad_energy_h(xx, vv, gx, gv);
        double af[3], ac[3]; accel_prod(xx, vv, full, af); accel_prod(xx, vv, cons, ac);
        return gv[0] * (af[0] - ac[0]) + gv[1] * (af[1] - ac[1]) + gv[2] * (af[2] - ac[2]);
    };
    const double dt = 2 * M_PI / 200000.0;
    double t = 0, W = 0, rd_prev = 0; int nap = 0;
    auto f = [&](const double* s, double* ds) { accel_prod(s, s + 3, cons, ds + 3); for (int k = 0; k < 3; k++) ds[k] = s[3 + k]; ds[6] = power(s, s + 3); };
    double s[7] = {x[0], x[1], x[2], v[0], v[1], v[2], 0};
    double t_apo1 = 0, W1 = 0;
    for (int it = 0; it < 10000000; it++) {
        double k1[7], k2[7], k3[7], k4[7], tmp[7];
        f(s, k1); for (int i = 0; i < 7; i++) tmp[i] = s[i] + 0.5 * dt * k1[i];
        f(tmp, k2); for (int i = 0; i < 7; i++) tmp[i] = s[i] + 0.5 * dt * k2[i];
        f(tmp, k3); for (int i = 0; i < 7; i++) tmp[i] = s[i] + dt * k3[i];
        f(tmp, k4);
        double sn[7]; for (int i = 0; i < 7; i++) sn[i] = s[i] + dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6;
        const double rd = sn[0] * sn[3] + sn[1] * sn[4] + sn[2] * sn[5];
        if (it > 10 && rd_prev > 0 && rd <= 0) {
            // linear interpolation of the apocentre crossing
            const double w = rd_prev / (rd_prev - rd);
            const double ta = t + w * dt, Wa = s[6] + w * (sn[6] - s[6]);
            nap++;
            if (nap == 1) { t_apo1 = ta; W1 = Wa; }
            else { Pavg = (Wa - W1) / (ta - t_apo1); return; }
        }
        rd_prev = rd;
        for (int i = 0; i < 7; i++) s[i] = sn[i];
        t += dt;
    }
}

// orbit-averaged RR power (ADM) along the conservative 3PN ADM orbit from apocentre (harmonic IC taken as ADM: the
// balance is gauge invariant)
static void balance_adm(double e, int rr_bits, double& Pavg, double& Fgw) {
    load_table(g_nu, 11);
    admpn::PairHam h1; h1.build(g_nu, 11);
    double x[3] = {1 + e, 0, 0}, p[3] = {0, std::sqrt((1 - e) / (1 + e)), 0};
    const double E = ham_adm(x, p);
    const double J = x[0] * p[1] - x[1] * p[0];
    const double eps = -2 * E / (g_c * g_c), jB = -2 * E * J * J;
    double de, dj; secular_pn_rates(eps, jB, g_nu, 1.0, g_c, 31, de, dj);
    Fgw = de * g_c * g_c / 2;
    auto power = [&](const double* xx, const double* pp, const double* gp) {
        double a[3], Hpp[9], dx[3], dp[3], F[3];
        admpn::rr_accel_adm(xx, gp, 1.0, g_nu, g_c, rr_bits, a);
        h1.value_grad(xx, pp, 1.0, 1.0 / (g_c * g_c), dx, dp, Hpp);
        for (int d = 0; d < 3; d++) Hpp[4 * d] += 1.0;
        admpn::solve3(Hpp, a, F);
        return gp[0] * F[0] + gp[1] * F[1] + gp[2] * F[2];
    };
    auto f = [&](const double* s, double* ds) {
        double gx[3], gp[3]; grad_ham_adm(s, s + 3, gx, gp);
        for (int k = 0; k < 3; k++) { ds[k] = gp[k]; ds[3 + k] = -gx[k]; }
        ds[6] = power(s, s + 3, gp);
    };
    const double dt = 2 * M_PI / 200000.0;
    double s[7] = {x[0], x[1], x[2], p[0], p[1], p[2], 0}, t = 0, rd_prev = 0, t_apo1 = 0, W1 = 0; int nap = 0;
    for (int it = 0; it < 10000000; it++) {
        double k1[7], k2[7], k3[7], k4[7], tmp[7];
        f(s, k1); for (int i = 0; i < 7; i++) tmp[i] = s[i] + 0.5 * dt * k1[i];
        f(tmp, k2); for (int i = 0; i < 7; i++) tmp[i] = s[i] + 0.5 * dt * k2[i];
        f(tmp, k3); for (int i = 0; i < 7; i++) tmp[i] = s[i] + dt * k3[i];
        f(tmp, k4);
        double sn[7]; for (int i = 0; i < 7; i++) sn[i] = s[i] + dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6;
        const double rd = sn[0] * sn[3] + sn[1] * sn[4] + sn[2] * sn[5];   // x . p: same sign as x . v (1PN)
        if (it > 10 && rd_prev > 0 && rd <= 0) {
            const double w = rd_prev / (rd_prev - rd);
            const double ta = t + w * dt, Wa = s[6] + w * (sn[6] - s[6]);
            nap++;
            if (nap == 1) { t_apo1 = ta; W1 = Wa; }
            else { Pavg = (Wa - W1) / (ta - t_apo1); return; }
        }
        rd_prev = rd;
        for (int i = 0; i < 7; i++) s[i] = sn[i];
        t += dt;
    }
}

int main() {
    int fails = 0;
    // 1. Enzyme gradient of the scalar H_ADM vs the hand-coded gradient, per order
    {
        double worst = 0;
        for (int o : {1, 2, 8, 11}) {
            load_table(0.21, o); g_c = 3.0;
            double x[3] = {0.9, -0.4, 0.3}, p[3] = {0.35, 0.6, -0.2}, gx[3], gp[3], dx[3], dp[3];
            grad_ham_adm(x, p, gx, gp);
            g_ph.value_grad(x, p, 1.0, 1.0 / 9.0, dx, dp);
            const double r = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
            double err = 0, sc = 0;
            for (int k = 0; k < 3; k++) {
                err = std::max({err, std::fabs(gx[k] - (dx[k] + x[k] / (r * r * r))), std::fabs(gp[k] - (dp[k] + p[k]))});
                sc = std::max({sc, std::fabs(dx[k]), std::fabs(dp[k])});
            }
            std::printf("ADM gradient, order bits %2d: |Enzyme - hand| / |PN part| = %.1e\n", o, err / sc);
            worst = std::max(worst, err / sc);
        }
        if (!(worst < 1e-13)) fails++;
    }
    // 2./3. orbit-averaged RR power vs <F> (Peters + 1PN), e = 0.5 and 0.8, c = 10, 20, 40 (G M = a = 1)
    for (double e : {0.5, 0.8}) {
        for (int bits : {4, 20}) {
            double res_h[3], res_a[3]; int k = 0;
            for (double c : {10.0, 20.0, 40.0}) {
                g_c = c;
                double P, F;
                balance_harm(e, bits, P, F);
                res_h[k] = P / (-F) - 1;
                balance_adm(e, bits, P, F);
                res_a[k] = P / (-F) - 1;
                k++;
            }
            std::printf("e=%.1f RR bits %2d: <P_RR>/(-<F>) - 1 at c = 10/20/40: harmonic(prod) %+.3e %+.3e %+.3e (ratios %.1f %.1f) | "
                        "ADM %+.3e %+.3e %+.3e (ratios %.1f %.1f)\n", e, bits, res_h[0], res_h[1], res_h[2], res_h[0] / res_h[1],
                        res_h[1] / res_h[2], res_a[0], res_a[1], res_a[2], res_a[0] / res_a[1], res_a[1] / res_a[2]);
            if (bits == 20) {   // consistent 3.5PN: residual falls like c^-4 (ratio ~16); a wrong 3.5PN term: ~4
                if (!(res_h[1] / res_h[2] > 10)) { std::printf("  harmonic (production) 3.5PN balance FAILS (c^-2)\n"); fails++; }
                if (!(res_a[1] / res_a[2] > 10)) { std::printf("  ADM 3.5PN balance FAILS (c^-2)\n"); fails++; }
            }
        }
    }
    std::printf(fails ? "FAIL\n" : "PASS\n");
    return fails;
}
