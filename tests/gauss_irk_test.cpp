// Gauss-IRK subsystem integrator (src/collisional/gauss_irk.cpp, docs/CASES.md C22), host only; ctest gauss_irk.
// Normalised units (G = 1), bounds 1.3x the larger measured value of the CMake Release build and -O3 -march=native:
//  1. e = 0.9 binary, 2000 orbits in one call and in 2000 calls of one orbit (production calls restart B): energy
//     error bounded, P exact, L at round-off; the landing hits t_end to 1e-14.
//  2. 2+2 quadruple (tests/proto_symrk quad: (1, 1; a 1, e 0.5) + (0.8, 0.6; a 0.7, e 0.7), outer a 15, e 0.3).
//  3. Simplified Newton (forced on every step) gives the fixed-point result to round-off.
//  4. Hard pericentre: e = 0.99 at 30 steps per orbit without the spo floor, the fixed point and simplified Newton both
//     fail (measured: both fail below 38 steps per orbit at e = 0.99, 12 at e = 0.9; Newton never rescued a step there): the call reports no convergence (fail 2) with the state unchanged, so the caller can run LogH+GBS; with
//     the floor (8 / sqrt(1 - e) = 80 steps per orbit) it converges. Collision watch and step limit fail the same way.
#include "gauss_irk.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace {
// Measured values (bounds are 1.3x)
constexpr double E09_DE_ONE = 4.89e-14, E09_DE_CALLS = 4.02e-14, QUAD_DE = 1.65e-13, NEWTON_DX = 7.69e-14;
constexpr double DL_MAX = 1.09e-13, DP_MAX = 3.11e-15;   // round-off random walk over 1e5 steps

bool check(const char* name, double v, double bound, bool lower = false) {
    const bool ok = lower ? v >= bound : v <= bound;
    std::printf("%s %s = %.3e (%s %.3e)\n", ok ? "ok  " : "FAIL", name, v, lower ? ">=" : "<=", bound);
    return ok;
}

void kepler_pair(double m1, double m2, double a, double e, double inc, double* q, double* v) {
    const double M = m1 + m2, r = a * (1 + e), vr = std::sqrt(M / a * (1 - e) / (1 + e));
    const double x[3] = {r, 0, 0}, u[3] = {0, vr * std::cos(inc), vr * std::sin(inc)};
    for (int k = 0; k < 3; k++) {
        q[k] = m2 / M * x[k]; q[3 + k] = -m1 / M * x[k];
        v[k] = m2 / M * u[k]; v[3 + k] = -m1 / M * u[k];
    }
}

LogHState binary(double e) {
    LogHState s{};
    s.n_bodies = 2;
    s.mass[0] = s.mass[1] = 0.5;
    kepler_pair(0.5, 0.5, 1.0, e, 0.3, s.pos, s.vel);
    return s;
}

LogHState quad() {
    LogHState s{};
    s.n_bodies = 4;
    const double m[4] = {1.0, 1.0, 0.8, 0.6};
    for (int i = 0; i < 4; i++) s.mass[i] = m[i];
    double qa[6], va[6], qb[6], vb[6], qo[6], vo[6];
    kepler_pair(1.0, 1.0, 1.0, 0.5, 0.0, qa, va);
    kepler_pair(0.8, 0.6, 0.7, 0.7, 1.1, qb, vb);
    kepler_pair(2.0, 1.4, 15.0, 0.3, 0.87, qo, vo);
    for (int k = 0; k < 3; k++) {
        s.pos[k] = qo[k] + qa[k]; s.pos[3 + k] = qo[k] + qa[3 + k];
        s.pos[6 + k] = qo[3 + k] + qb[k]; s.pos[9 + k] = qo[3 + k] + qb[3 + k];
        s.vel[k] = vo[k] + va[k]; s.vel[3 + k] = vo[k] + va[3 + k];
        s.vel[6 + k] = vo[3 + k] + vb[k]; s.vel[9 + k] = vo[3 + k] + vb[3 + k];
    }
    return s;
}

void PL(const LogHState& s, double* P, double* L) {
    for (int k = 0; k < 3; k++) P[k] = L[k] = 0.0;
    for (int a = 0; a < s.n_bodies; a++) {
        const double* x = s.pos + 3 * a;
        double u[3];
        for (int k = 0; k < 3; k++) { u[k] = s.mass[a] * s.vel[3 * a + k]; P[k] += u[k]; }
        L[0] += x[1] * u[2] - x[2] * u[1]; L[1] += x[2] * u[0] - x[0] * u[2]; L[2] += x[0] * u[1] - x[1] * u[0];
    }
}

// n_calls calls of T / n_calls each; returns max |dE/E| after a call, |dP|, |dL|/|L|
struct Run { double dE = 0, dP = 0, dL = 0, t_err = 0; long long steps = 0, rhs = 0, newton = 0; bool ok = true; };
Run run(LogHState s, double T, int n_calls, GaussIrkOpts opt = GaussIrkOpts()) {
    Run r;
    const double E0 = logh_total_energy(s, 1.0);
    double P0[3], L0[3];
    PL(s, P0, L0);
    GaussIrkStats st;
    for (int c = 0; c < n_calls; c++) {
        s.t = 0.0;
        s.B = -logh_total_energy(s, 1.0);
        GaussIrkOut o;
        const double dt = T / n_calls;
        gauss_irk_integrate(s, dt, 1.0, opt, 100000000LL, nullptr, o, &st);
        if (!o.ok) { r.ok = false; std::printf("     call %d failed (%d)\n", c, o.fail); return r; }
        r.dE = std::max(r.dE, std::fabs(logh_total_energy(s, 1.0) / E0 - 1.0));
        r.t_err = std::max(r.t_err, std::fabs(s.t - dt) / dt);
    }
    double P1[3], L1[3];
    PL(s, P1, L1);
    double dL2 = 0.0, L2 = 0.0;
    for (int k = 0; k < 3; k++) {
        r.dP = std::max(r.dP, std::fabs(P1[k] - P0[k]));
        dL2 += (L1[k] - L0[k]) * (L1[k] - L0[k]);
        L2 += L0[k] * L0[k];
    }
    r.dL = std::sqrt(dL2 / L2);
    r.steps = st.steps; r.rhs = st.rhs; r.newton = st.newton;
    return r;
}
}  // namespace

int main() {
    bool ok = true;
    const double P = 2.0 * M_PI;   // a = 1, M = 1
    auto t0 = std::chrono::steady_clock::now();

    // 1. e = 0.9 binary
    Run one = run(binary(0.9), 2000 * P, 1);
    std::printf("     e=0.9 one call: %lld steps, %.0f evaluations per orbit\n", one.steps, one.rhs / 2000.0);
    ok &= check("e=0.9, 2000 orbits, one call: max|dE/E|", one.ok ? one.dE : 1.0, 1.3 * E09_DE_ONE);
    ok &= check("e=0.9, one call: |dP|", one.dP, 1.3 * DP_MAX);
    ok &= check("e=0.9, one call: |dL|/|L|", one.dL, 1.3 * DL_MAX);
    Run many = run(binary(0.9), 2000 * P, 2000);
    ok &= check("e=0.9, 2000 calls of one orbit: max|dE/E|", many.ok ? many.dE : 1.0, 1.3 * E09_DE_CALLS);
    ok &= check("e=0.9, 2000 calls: |dL|/|L|", many.dL, 1.3 * DL_MAX);
    ok &= check("e=0.9, 2000 calls: |t - t_end| / dt", many.t_err, 1e-13);

    // 2. 2+2 quadruple, ~1000 orbits of the inner binary (P = 3.1)
    Run q = run(quad(), 3000.0, 30);
    std::printf("     quadruple: %lld steps\n", q.steps);
    ok &= check("2+2 quadruple, 30 calls: max|dE/E|", q.ok ? q.dE : 1.0, 1.3 * QUAD_DE);
    ok &= check("2+2 quadruple: |dP|", q.dP, 1.3 * DP_MAX);
    ok &= check("2+2 quadruple: |dL|/|L|", q.dL, 1.3 * DL_MAX);

    // 3. Simplified Newton (every step) against the fixed point
    {
        LogHState a = binary(0.9), b = binary(0.9);
        GaussIrkOpts on;
        on.force_newton = true;
        GaussIrkOut oa, ob;
        GaussIrkStats sa, sb;
        gauss_irk_integrate(a, 3.3 * P, 1.0, GaussIrkOpts(), 1000000, nullptr, oa, &sa);
        gauss_irk_integrate(b, 3.3 * P, 1.0, on, 1000000, nullptr, ob, &sb);
        double dx = 0.0;
        for (int i = 0; i < 6; i++) dx = std::max(dx, std::fabs(a.pos[i] - b.pos[i]) + std::fabs(a.vel[i] - b.vel[i]));
        ok &= check("simplified Newton: steps solved by Newton", (double)sb.newton, (double)sb.steps, true);
        ok &= check("simplified Newton vs fixed point: max |dx| + |dv|", oa.ok && ob.ok ? dx : 1.0, 1.3 * NEWTON_DX);
    }

    // 4. Hard pericentre, failure reporting
    {
        LogHState s = binary(0.99), s0 = s;
        GaussIrkOpts low;
        low.spo = 30.0;
        low.spo_floor = false;
        GaussIrkOut o;
        gauss_irk_integrate(s, 10 * P, 1.0, low, 1000000, nullptr, o, nullptr);
        ok &= check("e=0.99, 30 steps/orbit, no floor: fail code (2: no convergence)", o.ok ? 0.0 : (double)o.fail, 2.0, true);
        ok &= check("... state unchanged", std::memcmp(&s, &s0, sizeof(s)) == 0 ? 0.0 : 1.0, 0.0);
        low.spo_floor = true;
        Run r = run(binary(0.99), 10 * P, 1, low);
        ok &= check("e=0.99, 30 steps/orbit with the floor: max|dE/E|", r.ok ? r.dE : 1.0, 1e-13);
        double coll[LOGH_MAX_BODIES * LOGH_MAX_BODIES] = {};
        coll[0 * LOGH_MAX_BODIES + 1] = 0.5;   // pericentre 0.01 < 0.5, separation 1.99 > 0.5: stops after the first step
        s = binary(0.99);
        gauss_irk_integrate(s, 10 * P, 1.0, GaussIrkOpts(), 1000000, coll, o, nullptr);
        ok &= check("collision watch: fail code 3", o.ok ? 0.0 : (double)o.fail, 3.0, true);
        ok &= check("... state unchanged", std::memcmp(&s, &s0, sizeof(s)) == 0 ? 0.0 : 1.0, 0.0);
        gauss_irk_integrate(s, 10 * P, 1.0, GaussIrkOpts(), 5, nullptr, o, nullptr);
        ok &= check("step limit: fail code 1", o.ok ? 0.0 : (double)o.fail, 1.0, true);
    }
    std::printf("     wall %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
