// kepler_drift near the parabola with the end point at (or next to) the pericentre (ctest kepler_near_parabolic).
// Bound orbits with 1 - e < 1e-5 and hyperbolae with e - 1 < 1e-4 take the universal-variable route
// (kepler_drift_fg): r = f r0 + g v0 from a far point to one near the pericentre cancels, as the comment in
// kepler_drift says of the old elements route, and the energy error grows as eps / (1 - e)^2. Found with Monte Carlo
// arithmetic (tests/vfc case 5: 1000 drifts at 1 - e = 1e-6 kept 0 digits of dE/E, sigma 3e-11) and natively: one
// drift of 0.3 P at 1 - e = 1e-6 ending near the pericentre changed E by 9e-6 of itself (1.7e-5 for 1.5 P), at
// 1 - e = 1e-8 by 4e-7; one drift from r = a to the pericentre at 1 - e = 1e-6 by 4e-4 of itself (2e-10 of mu / q).
// Requirement: |dE| / (|E| + mu / r) < 1e-13 (the measurement itself cancels at the pericentre) and |dL/L| < 1e-12 for
// drifts from r ~ a to t_peri + {-1, -1e-3, 0, 1e-3, 1} t_q (t_q = sqrt(q^3 / mu), the pericentre passage).
#include "kepler.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

static double nrm(const double* x) { return std::sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]); }
static double energy(const double* r, const double* v) { return 0.5 * (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) - 1.0 / nrm(r); }
static double angmom(const double* r, const double* v) {
    const double h[3] = {r[1]*v[2] - r[2]*v[1], r[2]*v[0] - r[0]*v[2], r[0]*v[1] - r[1]*v[0]};
    return nrm(h);
}

int main() {
    int fails = 0;
    // (e, start radius): bound a = 1, hyperbolic A = 1, mu = 1; incoming (true anomaly < 0), so the drift of t_peri
    // ends at the pericentre
    const double cases[][2] = {{1.0 - 1e-6, 1.0}, {1.0 - 1e-8, 1.5}, {1.0 - 3e-6, 0.3}, {1.0 + 1e-5, 10.0}, {1.0 + 1e-7, 3.0}};
    for (const auto& c : cases) {
        const double e = c[0], rr = c[1];
        const double p = std::fabs(1.0 - e * e);
        const double f = -std::acos(std::min(1.0, std::max(-1.0, (p / rr - 1.0) / e))), vs = std::sqrt(1.0 / p);
        const double ci = std::cos(0.4), si = std::sin(0.4);
        const double r0[3] = {rr * std::cos(f), rr * std::sin(f) * ci, rr * std::sin(f) * si};
        const double v0[3] = {-vs * std::sin(f), vs * (e + std::cos(f)) * ci, vs * (e + std::cos(f)) * si};
        double q = 0.0;
        // pericentre time: kepler_time_to_pericentre, refined by bisection on r.v after the drift (its Barker branch,
        // |1 - e| < 1e-6, is off by O(1) for starts beyond |a|: 0.108 instead of 1.228 in case 2, 9.68 instead of
        // 1.81 in case 5)
        double hi = kepler_time_to_pericentre(r0, v0, 1.0, &q), lo = 0.0;
        auto rv_after = [&](double t) {
            double r[3] = {r0[0], r0[1], r0[2]}, v[3] = {v0[0], v0[1], v0[2]};
            kepler_drift(r, v, 1.0, t);
            return r[0]*v[0] + r[1]*v[1] + r[2]*v[2];
        };
        while (rv_after(hi) < 0.0) { lo = hi; hi *= 2.0; }
        for (int i = 0; i < 200; i++) {
            const double m = 0.5 * (lo + hi);
            if (m == lo || m == hi) break;
            (rv_after(m) < 0.0 ? lo : hi) = m;
        }
        const double tp = 0.5 * (lo + hi);
        const double tq = std::sqrt(q * q * q);
        const double E0 = energy(r0, v0), L0 = angmom(r0, v0);
        double worst_E = 0.0, worst_L = 0.0;
        for (double off : {-1.0, -1e-3, 0.0, 1e-3, 1.0}) {
            double r[3] = {r0[0], r0[1], r0[2]}, v[3] = {v0[0], v0[1], v0[2]};
            kepler_drift(r, v, 1.0, tp + off * tq);
            // relative to |E| + mu / r: at the pericentre the energy itself cancels (T ~ |U| ~ mu / q >> |E|)
            const double scale = std::fabs(E0) + 1.0 / std::min(nrm(r), nrm(r0));
            worst_E = std::max(worst_E, std::fabs(energy(r, v) - E0) / scale);
            worst_L = std::max(worst_L, std::fabs(angmom(r, v) - L0) / L0);
        }
        const bool ok = worst_E < 1e-13 && worst_L < 1e-12;
        std::printf("e - 1 = %+.0e, r0 = %.1f: drift to the pericentre (t = %.4f, q = %.1e): |dE|/(|E|+mu/r) %.2e |dL/L| %.2e  %s\n",
                    e - 1.0, rr, tp, q, worst_E, worst_L, ok ? "ok" : "FAIL");
        if (!ok) fails++;
    }
    std::printf(fails ? "FAIL\n" : "PASS\n");
    return fails;
}
