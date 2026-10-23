// Prototype + benchmark: Gauss-Legendre IRK (symplectic, order 2s) on the LogH / Poincare time-transformed
// N-body Hamiltonian vs the production LogH + GBS step (src/collisional/logh_step.cpp, gbs_step, Bulirsch
// sequence, k_max 10, eta_GBS 1e-10, step control as GBSIntegrator::integrate_to_time incl. dyn_cap 3).
// Not part of the build or ctest. Build (worktree root, module gcc/15):
//   g++ -std=c++17 -O3 -march=native -Isrc/collisional tests/proto_symrk/bench.cpp -o <scratch>/symrk_bench
// Usage:
//   symrk_bench verify                                    symplectic defect (exact dual Jacobians), reversibility,
//                                                         P, L, PN1 gradient check
//   symrk_bench gauss <case> <s> <steps_per_orbit> <orbits> [log|poinc]
//   symrk_bench gbs   <case> <orbits> [eta]
//   symrk_bench lf    <case> <steps_per_orbit> <orbits>  plain LogH leapfrog, fixed ds, no GBS
// NBODY_KICK_U=1 switches the production kick to the 1/(-U) of Rantala et al. 2023 (gbs and lf modes; CASES R2 retest).
// Cases: e09, e099 (equal-mass binary a = 1), kozai (tests/data/adv_kozai_triple.txt), quad (2+2), pn09 (1PN BH
// binary e = 0.9, GM/(c^2 a) = 1e-3), harvest (ic.txt of harvest-2026-10-19/s404_s1025_1, path in argv[6] or env
// SYMRK_HARVEST). One orbit = one period of the innermost (closest) pair.
//
// Measured 2026-10-21 (Raven login node, g++ 15 -O3, one core, wall times noisy +-2x). max|dE/E| over the run
// (running max), slope of log(max) vs log(t) (0 bounded, 0.5 random walk, 1 linear drift), force evaluations
// per inner orbit:
//   case (orbits)        LogH+GBS eta 1e-10              Gauss s=8 (order 16), Poincare g = 1/(-U)
//   e=0.9   (1e5)        1.7e-8  slope 1.0   2851        spo 50: 2.1e-11 (end 1e-12) slope 1.1  2876
//   e=0.99  (1e4)        3.3e-9  slope 1.0   4372        spo 50: 1.3e-9 bounded peri swing, secular 6e-14, 3260;
//                                                        spo <= 35: fixed-point iteration diverges at pericentre
//   Kozai triple (2e4)   1.9e-10 slope 0.85  1315        spo 50: 3.2e-11  5514   (spo 35: 1.2e-8, truncation)
//   2+2 quadruple (1e4)  5.0e-11 slope 0.64  3407        spo 25: 1.4e-12  3650
//   1PN binary (1e4)     1.9e-6 (truncated EIH: no       spo 25: 3.3e-13 (ADM H_N + H_1PN, exact invariant) 2030
//                        exact invariant, O(1/c^4))
//   harvest s404 n=5     3.9e-10 slope 0.43  669         spo 25: 8.9e-12  1719
// GBS eta 1e-12: e=0.9 1.3e-11 at 7166, Kozai 2.5e-10 at 2581, quad 5.5e-11 at 8561, harvest 2.2e-9 at 5021.
// Gauss order 8 (s=4), e=0.9 spo 100: 4.1e-13 slope 0.2 at 3047. P exact, L to 1e-14..1e-12 (Gauss).
// verify: symplectic defect 2e-16..9e-16 (log, poinc, poinc+PN1, n=4; control with 3 iterations 3.6e-2),
// reversibility 6e-13 over 2000 steps.
// Kick retest 2026-10-21 (CASES R2), eta 1e-10, max|dE/E| / fevals per orbit, 1/Omega vs 1/(-U):
//   e09 1e5: 1.7e-8/2851 vs 6.3e-9/2457 (end 1.7e-8 vs 3e-10); e099 1e4: 3.3e-9/4372 vs 1.3e-8/3425;
//   kozai 2e4: 1.9e-10/1315 vs 4.3e-9/1090; quad: 5.0e-11/3407 vs 1.1e-10/3115; harvest: 3.9e-10/669 vs 2.0e-9/664;
//   pn09: 1.87e-6 both (EIH truncation swing). Leapfrog fixed ds: both bounded (slope 0), same max swing.
#include "../../src/collisional/logh.cpp"
#include "../../src/collisional/logh_step.cpp"
#include "../../src/collisional/post_newtonian_core.cpp"
#include "nbody_ham.h"
#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>

using namespace symrk;

struct Case {
    NBody sys;
    std::vector<double> q, v;   // positions, velocities
    double P_in = 1.0;          // inner period
    double a_in = 1.0, m1m2 = 1.0;
};

static void kepler_init(double m1, double m2, double a, double e, double G, double inc, double* q, double* v,
                        double phase_apo = 1.0) {
    // two bodies at apocentre (phase_apo = 1) in the x-y plane rotated by inc about x
    const double M = m1 + m2, r = a * (1 + e), vr = std::sqrt(G * M / a * (1 - e) / (1 + e));
    double x[3] = {r, 0, 0}, u[3] = {0, vr * std::cos(inc), vr * std::sin(inc)};
    (void)phase_apo;
    for (int k = 0; k < 3; k++) {
        q[k] = m2 / M * x[k]; q[3 + k] = -m1 / M * x[k];
        v[k] = m2 / M * u[k]; v[3 + k] = -m1 / M * u[k];
    }
}

static void inner_pair(Case& c) {
    const int n = c.sys.n;
    double best = 1e300; int ia = 0, ib = 1;
    for (int a = 0; a < n; a++)
        for (int b = a + 1; b < n; b++) {
            double r2 = 0;
            for (int k = 0; k < 3; k++) r2 += std::pow(c.q[3 * a + k] - c.q[3 * b + k], 2);
            if (r2 < best) { best = r2; ia = a; ib = b; }
        }
    double r = std::sqrt(best), v2 = 0;
    for (int k = 0; k < 3; k++) v2 += std::pow(c.v[3 * ia + k] - c.v[3 * ib + k], 2);
    const double M = c.sys.m[ia] + c.sys.m[ib];
    const double a = 1.0 / (2.0 / r - v2 / (c.sys.G * M));
    c.a_in = a;
    c.P_in = 2 * M_PI * std::sqrt(a * a * a / (c.sys.G * M));
    c.m1m2 = c.sys.m[ia] * c.sys.m[ib];
}

static void load_ic(Case& c, const char* path, double G) {
    FILE* f = std::fopen(path, "r");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    char line[4096]; int n = 0;
    while (std::fgets(line, sizeof line, f)) {
        if (line[0] == '#') continue;
        double m, x, y, z, vx, vy, vz;
        if (std::sscanf(line, "%lf %lf %lf %lf %lf %lf %lf", &m, &x, &y, &z, &vx, &vy, &vz) != 7) continue;
        c.sys.m[n] = m;
        c.q.insert(c.q.end(), {x, y, z}); c.v.insert(c.v.end(), {vx, vy, vz});
        n++;
    }
    std::fclose(f);
    c.sys.n = n; c.sys.G = G;
}

static void com_frame(Case& c) {
    double M = 0, X[3] = {0, 0, 0}, V[3] = {0, 0, 0};
    for (int a = 0; a < c.sys.n; a++) {
        M += c.sys.m[a];
        for (int k = 0; k < 3; k++) { X[k] += c.sys.m[a] * c.q[3 * a + k]; V[k] += c.sys.m[a] * c.v[3 * a + k]; }
    }
    for (int a = 0; a < c.sys.n; a++)
        for (int k = 0; k < 3; k++) { c.q[3 * a + k] -= X[k] / M; c.v[3 * a + k] -= V[k] / M; }
}

static Case make_case(const std::string& name, const char* harvest) {
    Case c;
    c.sys.G = 1.0;
    if (name == "e09" || name == "e099" || name == "pn09" || name == "e05") {
        c.sys.n = 2; c.sys.m[0] = c.sys.m[1] = 0.5;
        const double e = name == "e099" ? 0.99 : (name == "e05" ? 0.5 : 0.9);
        c.q.resize(6); c.v.resize(6);
        kepler_init(0.5, 0.5, 1.0, e, 1.0, 0.3, c.q.data(), c.v.data());
        if (name == "pn09") { c.sys.pn = true; c.sys.c = std::sqrt(std::getenv("SYMRK_C2") ? std::atof(std::getenv("SYMRK_C2")) : 1000.0); }
    } else if (name == "kozai") {
        load_ic(c, "tests/data/adv_kozai_triple.txt", 4.30091e-3);
    } else if (name == "harvest") {
        load_ic(c, harvest, 4.30091e-3);
    } else if (name == "quad") {
        // 2+2: binaries (1,1; a = 1, e = 0.5) and (0.8,0.6; a = 0.7, e = 0.7), outer a = 15, e = 0.3, 50 deg
        c.sys.n = 4; double m[4] = {1.0, 1.0, 0.8, 0.6};
        for (int i = 0; i < 4; i++) c.sys.m[i] = m[i];
        double qa[6], va[6], qb[6], vb[6], qo[6], vo[6];
        kepler_init(1.0, 1.0, 1.0, 0.5, 1.0, 0.0, qa, va);
        kepler_init(0.8, 0.6, 0.7, 0.7, 1.0, 1.1, qb, vb);
        kepler_init(2.0, 1.4, 15.0, 0.3, 1.0, 0.87, qo, vo);
        c.q.resize(12); c.v.resize(12);
        for (int k = 0; k < 3; k++) {
            c.q[k] = qo[k] + qa[k]; c.q[3 + k] = qo[k] + qa[3 + k];
            c.q[6 + k] = qo[3 + k] + qb[k]; c.q[9 + k] = qo[3 + k] + qb[3 + k];
            c.v[k] = vo[k] + va[k]; c.v[3 + k] = vo[k] + va[3 + k];
            c.v[6 + k] = vo[3 + k] + vb[k]; c.v[9 + k] = vo[3 + k] + vb[3 + k];
        }
    } else {
        std::fprintf(stderr, "unknown case %s\n", name.c_str()); std::exit(1);
    }
    com_frame(c);
    inner_pair(c);
    return c;
}

// PN cases: the energy error is sampled only at r > 1.5 a (apocentre side). The production LogH PN integrates the
// truncated harmonic 1PN (EIH) equations, whose 1PN energy is conserved only to O(1/c^4): near pericentre it swings
// by 2e-2 (c^2 = 1e3, e = 0.9; scales as 1/c^4) even with converged GBS. The ADM Hamiltonian H_N + H_1PN of the
// Gauss runs is conserved exactly by its own flow.
static double sep(const double* q, int a, int b) {
    double r2 = 0;
    for (int k = 0; k < 3; k++) r2 += (q[3 * a + k] - q[3 * b + k]) * (q[3 * a + k] - q[3 * b + k]);
    return std::sqrt(r2);
}

static void PL(const NBody& s, const double* q, const double* p, double* P, double* L) {
    for (int k = 0; k < 3; k++) P[k] = L[k] = 0;
    for (int a = 0; a < s.n; a++) {
        const double* x = q + 3 * a; const double* u = p + 3 * a;
        for (int k = 0; k < 3; k++) P[k] += u[k];
        L[0] += x[1] * u[2] - x[2] * u[1]; L[1] += x[2] * u[0] - x[0] * u[2]; L[2] += x[0] * u[1] - x[1] * u[0];
    }
}

// Energy-error history: running max of |dE/E| at log-spaced times; slope of log(max) vs log(t) over the
// last two decades (0: bounded, 0.5: random walk, 1: linear drift)
struct History {
    std::vector<double> t, emax;
    double cur = 0.0, next = 0.0, T = 1.0;
    void init(double Tend) { T = Tend; next = 1e-3 * Tend; }
    void add(double tt, double rel) {
        cur = std::max(cur, std::fabs(rel));
        if (tt >= next) { t.push_back(tt); emax.push_back(cur); next *= 1.1; }
    }
    double slope() const {
        double sx = 0, sy = 0, sxx = 0, sxy = 0; int k = 0;
        for (size_t i = 0; i < t.size(); i++) {
            if (t[i] < 1e-2 * T || emax[i] <= 0) continue;
            double x = std::log(t[i]), y = std::log(emax[i]);
            sx += x; sy += y; sxx += x * x; sxy += x * y; k++;
        }
        if (k < 3) return NAN;
        return (k * sxy - sx * sy) / (k * sxx - sx * sx);
    }
};

static void report(const char* tag, const Case& c, double orbits, double emax, double efin, double slope,
                   double dP, double dL, double evals, double wall, double iters = 0) {
    std::printf("%-34s n=%d orbits=%.0f max|dE/E|=%.2e fin=%+.2e slope=%.2f |dP|=%.1e |dL/L|=%.1e "
                "fevals/orb=%.0f us/orb=%.2f%s",
                tag, c.sys.n, orbits, emax, efin, slope, dP, dL, evals / orbits, 1e6 * wall / orbits,
                iters > 0 ? "" : "\n");
    if (iters > 0) std::printf(" it/step=%.1f\n", iters);
}

static void run_gauss(Case c, int s, double spo, double orbits, Form form) {
    c.sys.form = form;
    if (c.sys.pn) c.sys.form = POINC;
    const int n = c.sys.n, n3 = 3 * n;
    GaussCoeffs g = make_gauss(s);
    CState<double> st;
    st.dim = c.sys.dim();
    for (int i = 0; i < n3; i++) { st.y[i] = c.q[i]; st.y[n3 + i] = c.v[i] * c.sys.m[i / 3]; }
    st.y[2 * n3] = 0.0;
    const double E0 = c.sys.energy(st.y.data());
    st.y[2 * n3 + 1] = -E0;
    double P0[3], L0[3];
    PL(c.sys, st.y.data(), st.y.data() + n3, P0, L0);
    // ds per inner orbit = integral (-U) dt over the orbit ~ G m1 m2 P / a (time average of 1/r = 1/a)
    const double h = c.sys.G * c.m1m2 * c.P_in / c.a_in / spo;
    const double Tend = orbits * c.P_in;
    std::array<std::array<double, DIM_MAX>, S_MAX> Y;
    bool have = false;
    IrkStats stats;
    History hist; hist.init(Tend);
    auto t0 = std::chrono::steady_clock::now();
    double rel = 0;
    while (st.y[2 * n3] < Tend) {
        gauss_step(g, c.sys, st, h, Y, have, stats);
        if (stats.diverged) { std::printf("  fixed-point iteration diverged at t = %.3g orbits\n", st.y[2 * n3] / c.P_in); break; }
        double yy[DIM_MAX];
        for (int i = 0; i < st.dim; i++) yy[i] = st.y[i] + st.lo[i];
        if (c.sys.pn && sep(yy, 0, 1) < 1.5 * c.a_in) continue;
        rel = (c.sys.energy(yy) - E0) / std::fabs(E0);
        hist.add(st.y[2 * n3], rel);
    }
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double P1[3], L1[3];
    PL(c.sys, st.y.data(), st.y.data() + n3, P1, L1);
    double dP = 0, dL = 0, Ln = 0;
    for (int k = 0; k < 3; k++) { dP = std::max(dP, std::fabs(P1[k] - P0[k])); dL += std::pow(L1[k] - L0[k], 2); Ln += L0[k] * L0[k]; }
    char tag[128];
    std::snprintf(tag, sizeof tag, "gauss%d-%s %s spo=%.0f", 2 * s, form == LOG && !c.sys.pn ? "log" : "poinc",
                  "", spo);
    report(tag, c, st.y[2 * n3] / c.P_in, hist.cur, rel, hist.slope(), dP, std::sqrt(dL / Ln),
           (double)stats.rhs_calls, wall, (double)stats.iters / stats.steps);
    if (stats.maxit_hits) std::printf("  (iteration cap hit %lld times)\n", stats.maxit_hits);
}

static const int SEQ[] = {2, 4, 6, 8, 12, 16, 24, 32, 48, 64};

static void run_gbs(Case c, double orbits, double eta) {
    const int n = c.sys.n, n3 = 3 * n;
    LogHState st{};
    st.n_bodies = n;
    for (int i = 0; i < n3; i++) { st.pos[i] = c.q[i]; st.vel[i] = c.v[i]; }
    for (int a = 0; a < n; a++) st.mass[a] = c.sys.m[a];
    const bool pn = c.sys.pn;
    st.c_code = pn ? c.sys.c : 0.0;
    st.use_aux_vel = pn;
    if (pn) for (int i = 0; i < n3; i++) st.aux_vel[i] = st.vel[i];
    const double G = c.sys.G;
    auto energy = [&](const LogHState& s) {
        double E = logh_total_energy(s, G);
        if (pn) E += pn1_energy(s.pos, s.vel, s.mass, n, G, c.sys.c);
        return E;
    };
    const double E0 = energy(st);
    st.B = -logh_total_energy(st, G);
    GbsParams par;
    par.seq = SEQ; par.k_max = 10; par.eta = eta; par.G = G; par.c_code = st.c_code; par.pn_order = pn ? 1 : 0;
    double Mt = 0;
    for (int a = 0; a < n; a++) Mt += c.sys.m[a];
    double P0[3], L0[3], p[24];
    for (int i = 0; i < n3; i++) p[i] = st.vel[i] * st.mass[i / 3];
    PL(c.sys, st.pos, p, P0, L0);
    const double Tend = orbits * c.P_in;
    History hist; hist.init(Tend);
    GbsTable tab;
    double ds = 1e-3 * c.P_in * logh_omega(st);
    long long evals = 0, steps = 0, rej = 0;
    double rel = 0;
    auto t0 = std::chrono::steady_clock::now();
    while (st.t < Tend) {
        double rmin2 = 1e300;
        for (int a = 0; a < n; a++)
            for (int b = a + 1; b < n; b++) {
                double r2 = 0;
                for (int k = 0; k < 3; k++) r2 += std::pow(st.pos[3 * a + k] - st.pos[3 * b + k], 2);
                rmin2 = std::min(rmin2, r2);
            }
        const double rmin = std::sqrt(rmin2);
        const double ds_dyn = std::sqrt(rmin * rmin * rmin / (G * Mt)) * logh_omega(st) / 3.0;
        const double ds_try = std::min(ds, ds_dyn);
        LogHStepStart start;
        logh_step_start(st, G, par.c_code, par.pn_order, start);
        GbsStepOut o;
        gbs_step(par, st, ds_try, tab, -1, &start, o);
        for (int k = 0; k <= o.stage; k++) evals += SEQ[k];
        steps++;
        if (!o.accepted) { rej++; ds = 0.5 * ds_try; continue; }
        ds = o.ds_next;
        if (pn && sep(st.pos, 0, 1) < 1.5 * c.a_in) continue;
        rel = (energy(st) - E0) / std::fabs(E0);
        hist.add(st.t, rel);
        if (std::getenv("SYMRK_TRACE") && steps % 20 == 0) std::printf("trace t=%.4f dE/E=%+.3e\n", st.t / c.P_in, rel);
    }
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double P1[3], L1[3];
    for (int i = 0; i < n3; i++) p[i] = st.vel[i] * st.mass[i / 3];
    PL(c.sys, st.pos, p, P1, L1);
    double dP = 0, dL = 0, Ln = 0;
    for (int k = 0; k < 3; k++) { dP = std::max(dP, std::fabs(P1[k] - P0[k])); dL += std::pow(L1[k] - L0[k], 2); Ln += L0[k] * L0[k]; }
    char tag[128];
    std::snprintf(tag, sizeof tag, "logh-gbs %s eta=%.0e steps/orb=%.1f", std::getenv("NBODY_KICK_U") ? "1/(-U)" : "1/Om", eta, steps / (st.t / c.P_in));
    report(tag, c, st.t / c.P_in, hist.cur, rel, hist.slope(), dP, std::sqrt(dL / Ln), (double)evals, wall);
    if (rej) std::printf("  (rejected %lld of %lld)\n", rej, steps);
}

// Plain LogH leapfrog (production logh_leapfrog_steps, kick per NBODY_KICK_U), fixed ds, no GBS: spo steps per
// inner orbit (ds = G m1 m2 P / a / spo, as the Gauss runs)
static void run_lf(Case c, double spo, double orbits) {
    const int n = c.sys.n, n3 = 3 * n;
    LogHState st{};
    st.n_bodies = n;
    for (int i = 0; i < n3; i++) { st.pos[i] = c.q[i]; st.vel[i] = c.v[i]; }
    for (int a = 0; a < n; a++) st.mass[a] = c.sys.m[a];
    const double G = c.sys.G;
    const double E0 = logh_total_energy(st, G);
    st.B = -E0;
    double P0[3], L0[3], p[24];
    for (int i = 0; i < n3; i++) p[i] = st.vel[i] * st.mass[i / 3];
    PL(c.sys, st.pos, p, P0, L0);
    const double h = G * c.m1m2 * c.P_in / c.a_in / spo, Tend = orbits * c.P_in;
    History hist; hist.init(Tend);
    long long steps = 0; double rel = 0;
    auto t0 = std::chrono::steady_clock::now();
    while (st.t < Tend) {
        logh_leapfrog_steps(st, h, G, 1, nullptr);
        steps++;
        rel = (logh_total_energy(st, G) - E0) / std::fabs(E0);
        hist.add(st.t, rel);
    }
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double P1[3], L1[3];
    for (int i = 0; i < n3; i++) p[i] = st.vel[i] * st.mass[i / 3];
    PL(c.sys, st.pos, p, P1, L1);
    double dP = 0, dL = 0, Ln = 0;
    for (int k = 0; k < 3; k++) { dP = std::max(dP, std::fabs(P1[k] - P0[k])); dL += std::pow(L1[k] - L0[k], 2); Ln += L0[k] * L0[k]; }
    char tag[128];
    std::snprintf(tag, sizeof tag, "leapfrog %s spo=%.0f", std::getenv("NBODY_KICK_U") ? "1/(-U)" : "1/Omega", spo);
    report(tag, c, st.t / c.P_in, hist.cur, rel, hist.slope(), dP, std::sqrt(dL / Ln), (double)steps, wall);
}

// ---------------------------------------------------------------- verification
template <int N>
static double defect_of_step(Case c, int s, double h, Form form, double* det_out = nullptr, int iters = 60) {
    c.sys.form = form;
    if (c.sys.pn) c.sys.form = POINC;
    const int n = c.sys.n, n3 = 3 * n, d = c.sys.dim();
    GaussCoeffs g = make_gauss(s);
    CState<Dual<N>> st;
    st.dim = d;
    double y0[DIM_MAX];
    for (int i = 0; i < n3; i++) { y0[i] = c.q[i]; y0[n3 + i] = c.v[i] * c.sys.m[i / 3]; }
    y0[2 * n3] = 0.0;
    y0[2 * n3 + 1] = -c.sys.energy(y0) * 1.0001;   // slightly off the energy shell: full extended-phase-space map
    for (int i = 0; i < d; i++) { st.y[i] = Dual<N>(y0[i]); st.y[i].d[i] = 1.0; }
    std::array<std::array<Dual<N>, DIM_MAX>, S_MAX> Y;
    bool have = false;
    IrkStats stats;
    gauss_step(g, c.sys, st, h, Y, have, stats, iters);
    // canonical pairs: (q_i, p_i) and (t, p_t); permutation z = (q, t, p, p_t)
    const int D = d, half = D / 2;
    std::vector<int> perm(D);
    for (int i = 0; i < n3; i++) { perm[i] = i; perm[half + i] = n3 + i; }
    perm[n3] = 2 * n3; perm[D - 1] = 2 * n3 + 1;
    std::vector<double> M(D * D);
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++) M[i * D + j] = st.y[perm[i]].d[perm[j]];
    auto J = [&](int i, int j) { return (j == i + half) ? 1.0 : (i == j + half ? -1.0 : 0.0); };
    double def = 0;
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++) {
            double sum = 0;
            for (int k = 0; k < D; k++)
                for (int l = 0; l < D; l++) sum += M[k * D + i] * J(k, l) * M[l * D + j];
            def = std::max(def, std::fabs(sum - J(i, j)));
        }
    (void)det_out;
    return def;
}

static void verify() {
    // 1. PN1 gradient vs central differences
    {
        Case c = make_case("e09", nullptr);
        NBody s = c.sys; s.pn = true; s.c = 3.0; s.n = 3; s.m[2] = 0.3;
        double q[9] = {0.1, 0.2, -0.1, -0.5, 0.3, 0.2, 1.2, -0.7, 0.4}, p[9] = {0.3, -0.1, 0.2, -0.2, 0.25, 0.1, 0.05, -0.1, -0.2};
        double gU[9], Hq[9], Hp[9];
        s.hamiltonian(q, p, s.potential(q, gU), gU, Hq, Hp);
        double err = 0;
        for (int i = 0; i < 18; i++) {
            double* z = i < 9 ? q : p; int k = i % 9; double z0 = z[k], hh = 1e-5, e[2];
            for (int sg = 0; sg < 2; sg++) {
                z[k] = z0 + (sg ? -hh : hh);
                double g2[9], a[9], b[9];
                e[sg] = s.hamiltonian(q, p, s.potential(q, g2), g2, a, b);
            }
            z[k] = z0;
            double fd = (e[0] - e[1]) / (2 * hh), an = i < 9 ? Hq[k] : Hp[k];
            err = std::max(err, std::fabs(fd - an));
        }
        std::printf("verify PN1 ADM gradient vs FD (3 bodies, c = 3): max err %.1e\n", err);
    }
    // 2. symplectic defect of one step (exact dual Jacobian, off the energy shell)
    for (int f = 0; f < 3; f++) {
        Case c = make_case(f == 2 ? "pn09" : "e05", nullptr);
        if (f == 2) { c.sys.c = 3.0; }
        Form form = f == 0 ? LOG : POINC;
        for (double h : {0.05, 0.2}) {
            double d4 = defect_of_step<14>(c, 4, h, form), d8 = defect_of_step<14>(c, 8, h, form);
            std::printf("verify symplectic defect |M^T J M - J| %s%s ds=%.2f: s=4 %.1e  s=8 %.1e\n",
                        form == LOG ? "log" : "poinc", f == 2 ? "+PN1(c=3)" : "", h, d4, d8);
        }
    }
    {
        Case c = make_case("e05", nullptr);
        std::printf("verify control: 3 fixed-point iterations only (not converged), poinc ds=0.2 s=8: defect %.1e\n",
                    defect_of_step<14>(c, 8, 0.2, POINC, nullptr, 3));
    }
    {
        Case c = make_case("quad", nullptr);
        std::printf("verify symplectic defect poinc quad (n=4) ds=0.05: s=8 %.1e\n",
                    defect_of_step<26>(c, 8, 0.05, POINC));
    }
    // 3. reversibility + P, L over 2000 steps forward and back (e = 0.9, s = 8, 50 steps per orbit)
    for (int f = 0; f < 2; f++) {
        Case c = make_case(f ? "pn09" : "e09", nullptr);
        if (f) c.sys.c = 20.0;
        c.sys.form = POINC;
        const int n3 = 3 * c.sys.n;
        GaussCoeffs g = make_gauss(8);
        CState<double> st; st.dim = c.sys.dim();
        for (int i = 0; i < n3; i++) { st.y[i] = c.q[i]; st.y[n3 + i] = c.v[i] * c.sys.m[i / 3]; }
        st.y[2 * n3 + 1] = -c.sys.energy(st.y.data());
        auto y0 = st.y;
        double P0[3], L0[3];
        PL(c.sys, st.y.data(), st.y.data() + n3, P0, L0);
        const double h = c.sys.G * c.m1m2 * c.P_in / c.a_in / 50.0;
        std::array<std::array<double, DIM_MAX>, S_MAX> Y; bool have = false; IrkStats stats;
        for (int k = 0; k < 2000; k++) gauss_step(g, c.sys, st, h, Y, have, stats);
        double P1[3], L1[3];
        PL(c.sys, st.y.data(), st.y.data() + n3, P1, L1);
        have = false;
        for (int k = 0; k < 2000; k++) gauss_step(g, c.sys, st, -h, Y, have, stats);
        double rev = 0;
        for (int i = 0; i < st.dim; i++) rev = std::max(rev, std::fabs(st.y[i] + st.lo[i] - y0[i]));
        double dP = 0, dL = 0;
        for (int k = 0; k < 3; k++) { dP = std::max(dP, std::fabs(P1[k] - P0[k])); dL = std::max(dL, std::fabs(L1[k] - L0[k])); }
        std::printf("verify %s 2000 steps (40 orbits) fwd+back: reversibility %.1e, |dP| %.1e, |dL| %.1e, it/step %.1f\n",
                    f ? "PN1(c=20)" : "Newton e=0.9", rev, dP, dL, (double)stats.iters / stats.steps);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: see header\n"); return 1; }
    std::string mode = argv[1];
    const char* harvest = std::getenv("SYMRK_HARVEST");
    if (!harvest) harvest = "harvest-2026-10-19/s404_s1025_1/ic.txt";
    if (mode == "verify") { verify(); return 0; }
    if (argc < 4) return 1;
    Case c = make_case(argv[2], harvest);
    if (mode == "gauss") {
        int s = std::atoi(argv[3]);
        double spo = std::atof(argv[4]), orbits = std::atof(argv[5]);
        Form form = (argc > 6 && std::string(argv[6]) == "log") ? LOG : POINC;
        run_gauss(c, s, spo, orbits, form);
    } else if (mode == "lf") {
        run_lf(c, std::atof(argv[3]), std::atof(argv[4]));
    } else if (mode == "gbs") {
        run_gbs(c, std::atof(argv[3]), argc > 4 ? std::atof(argv[4]) : 1e-10);
    }
    return 0;
}
