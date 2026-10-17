// pn1_energy against the PN1 equations of motion (eih_pn_acceleration_pair): the energy T + U + E_pn1 must be
// conserved to O(1/c^4), so dE/dt (4th-order central difference along (v, a)) must drop by 16 when c doubles
// (by 4 if energy and acceleration disagree at 1PN). 2, 3 and 4 bodies (pair formula for two, the genuine N-body
// EIH for more, as LogH): ratio 14-18 for 20 random configurations each. With the pair sums of before, three and
// four bodies gave ~4 (docs/CASES.md C7).
#include "post_newtonian_core.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static const double G = 4.302194e-3;
static double energy(const std::vector<double>& x, const std::vector<double>& v, const std::vector<double>& m, double c) {
    const int n = (int)m.size();
    double E = 0.0;
    for (int A = 0; A < n; A++) {
        E += 0.5 * m[A] * (v[3*A]*v[3*A] + v[3*A+1]*v[3*A+1] + v[3*A+2]*v[3*A+2]);
        for (int B = A + 1; B < n; B++) {
            const double dx = x[3*A]-x[3*B], dy = x[3*A+1]-x[3*B+1], dz = x[3*A+2]-x[3*B+2];
            E -= G * m[A] * m[B] / std::sqrt(dx*dx + dy*dy + dz*dz);
        }
    }
    return E + pn1_energy(x.data(), v.data(), m.data(), n, G, c);
}
static void accel(const std::vector<double>& x, const std::vector<double>& v, const std::vector<double>& m, double c,
                  std::vector<double>& a) {
    const int n = (int)m.size();
    a.assign(3 * n, 0.0);
    for (int A = 0; A < n; A++)
        for (int B = A + 1; B < n; B++) {
            const double d[3] = {x[3*B]-x[3*A], x[3*B+1]-x[3*A+1], x[3*B+2]-x[3*A+2]};
            const double r2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2], r3 = r2 * std::sqrt(r2);
            for (int k = 0; k < 3; k++) { a[3*A+k] += G * m[B] * d[k] / r3; a[3*B+k] -= G * m[A] * d[k] / r3; }
            if (n >= 3) continue;   // as LogH: the pair formula for two bodies, the N-body EIH for more
            double ai[3], aj[3];
            eih_pn_acceleration_pair(&x[3*A], &v[3*A], m[A], &x[3*B], &v[3*B], m[B], G, c, ai, aj, 1);
            for (int k = 0; k < 3; k++) { a[3*A+k] += ai[k]; a[3*B+k] += aj[k]; }
        }
    if (n >= 3) {
        std::vector<double> ap(3 * n);
        eih1_nbody_acceleration(x.data(), v.data(), m.data(), n, G, c, ap.data());
        for (int i = 0; i < 3 * n; i++) a[i] += ap[i];
    }
}
static double dEdt(const std::vector<double>& x, const std::vector<double>& v, const std::vector<double>& m, double c,
                   double h) {
    std::vector<double> a;
    accel(x, v, m, c, a);
    auto at = [&](double s) {
        std::vector<double> xs(x), vs(v);
        for (size_t i = 0; i < x.size(); i++) { xs[i] += s * v[i]; vs[i] += s * a[i]; }
        return energy(xs, vs, m, c);
    };
    return (8.0 * (at(h) - at(-h)) - (at(2 * h) - at(-2 * h))) / (12.0 * h);
}

int main() {
    std::mt19937_64 rng(2026);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    int fails = 0;
    for (int n : {2, 3, 4}) {
        double lo = 1e30, hi = 0.0;
        for (int t = 0; t < 20; t++) {
            std::vector<double> m(n), x(3 * n), v(3 * n);
            double M = 0.0;
            for (int A = 0; A < n; A++) { m[A] = 10.0 + 20.0 * (U(rng) + 1.0); M += m[A]; }
            for (auto& q : x) q = 1e-6 * U(rng);
            const double vs = std::sqrt(G * M / 1e-6);
            for (auto& q : v) q = 0.5 * vs * U(rng);
            const double h = 1e-3 * 1e-6 / vs;
            const double r = std::fabs(dEdt(x, v, m, 10.0 * vs, h) / dEdt(x, v, m, 20.0 * vs, h));
            lo = std::min(lo, r); hi = std::max(hi, r);
        }
        std::printf("n = %d: dE/dt(c) / dE/dt(2c) %.2f - %.2f (16: consistent at 1PN, 4: not)\n", n, lo, hi);
        if (lo < 14.0 || hi > 18.0) { std::printf("FAIL %d-body PN1 energy and EIH acceleration inconsistent\n", n); fails++; }
    }
    return fails;
}
