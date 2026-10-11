// Two-body drift check (collisional/kepler.cpp kepler_advance): bound orbits from e = 0 to 0.9999, the parabola and
// hyperbolae up to e = 31100, intervals from a fraction of an orbit to 1e5 orbits (bound) or 1e6 / n (unbound).
// Per class the worst
//   1v16  |r(one drift of dt) - r(16 drifts of dt / 16)| / |r|   (the exact flow composes: round-off only)
//   rt    |r(dt then -dt) - r0| / |r0|                          (time reversibility)
//   ref   |r - r_ref| / |r_ref| against the element solution of Kepler's equation (elliptic: elements_to_cartesian
//         at M0 + n dt, no conversion of a drifted state; hyperbolic: e sinh F - F = M solved in long double)
//   dE, dL relative change of the two-body energy and angular momentum in one drift
// are printed as "KEPTEST <class> 1v16=... rt=... ref=... dE=... dL=..."; tests/run_case.py (kepler_drift_unit)
// checks them against bounds.
#include "kepler.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace {
double norm(const double* x) { return std::sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]); }
double dist(const double* a, const double* b) {
    const double d[3] = {a[0]-b[0], a[1]-b[1], a[2]-b[2]};
    return norm(d);
}
double energy(const double* r, const double* v, double mu) { return 0.5 * (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) - mu / norm(r); }
double angmom(const double* r, const double* v) {
    const double h[3] = {r[1]*v[2] - r[2]*v[1], r[2]*v[0] - r[0]*v[2], r[0]*v[1] - r[1]*v[0]};
    return norm(h);
}

struct Worst {
    double v1v16 = 0, rt = 0, ref = 0, dE = 0, dL = 0;
    void print(const char* cls) const {
        std::printf("KEPTEST %s 1v16=%.3e rt=%.3e ref=%.3e dE=%.3e dL=%.3e\n", cls, v1v16, rt, ref, dE, dL);
    }
};

// one case: drift (r0, v0) by dt; E_scale: energy scale for the relative energy change (|E|, or mu / q at e = 1)
void run_case(Worst& w, const double* r0, const double* v0, double mu, double dt, const double* r_ref, double E_scale) {
    double r1[3], v1[3], r16[3], v16[3];
    std::memcpy(r1, r0, sizeof r1); std::memcpy(v1, v0, sizeof v1);
    std::memcpy(r16, r0, sizeof r16); std::memcpy(v16, v0, sizeof v16);
    kepler_advance(r1, v1, mu, dt);
    for (int i = 0; i < 16; i++) kepler_advance(r16, v16, mu, dt / 16.0);
    double rb[3], vb[3];
    std::memcpy(rb, r1, sizeof rb); std::memcpy(vb, v1, sizeof vb);
    kepler_advance(rb, vb, mu, -dt);
    const double e1v16 = dist(r1, r16) / norm(r1);
    const double ert = dist(rb, r0) / norm(r0);
    const double eref = r_ref ? dist(r1, r_ref) / norm(r_ref) : 0.0;
    const double edE = std::fabs(energy(r1, v1, mu) - energy(r0, v0, mu)) / E_scale;
    const double edL = std::fabs(angmom(r1, v1) - angmom(r0, v0)) / angmom(r0, v0);
    auto upd = [](double& a, double b) { a = std::isfinite(b) ? std::max(a, b) : INFINITY; };
    upd(w.v1v16, e1v16); upd(w.rt, ert); upd(w.ref, eref); upd(w.dE, edE); upd(w.dL, edL);
}

// rotate a perifocal-plane vector (x, y, 0) by inclination i about x, then by Omega about z
void orient(const double* p, double inc, double Om, double* out) {
    const double y1 = p[1] * std::cos(inc), z1 = p[1] * std::sin(inc);
    out[0] = p[0] * std::cos(Om) - y1 * std::sin(Om);
    out[1] = p[0] * std::sin(Om) + y1 * std::cos(Om);
    out[2] = z1;
}

// hyperbolic state at mean anomaly M (perifocal, then oriented): r = A (e - cosh F, sqrt(e^2 - 1) sinh F),
// v = sqrt(mu / A) (-sinh F, sqrt(e^2 - 1) cosh F) / (e cosh F - 1); A = -a > 0
void hyper_state(double A, double e, double mu, long double M, double inc, double Om, double* r, double* v) {
    long double F = std::asinh(static_cast<double>(M / e));
    for (int k = 0; k < 200; k++) {
        const long double f = e * std::sinh(F) - F - M, fp = e * std::cosh(F) - 1.0L;
        const long double dF = -f / fp;
        F += dF;
        if (std::fabs(static_cast<double>(dF)) <= 1e-19 * std::max(1.0, std::fabs(static_cast<double>(F)))) break;
    }
    const long double sq = std::sqrt(static_cast<long double>(e) * e - 1.0L);
    const double p[3] = {static_cast<double>(A * (e - std::cosh(F))), static_cast<double>(A * sq * std::sinh(F)), 0.0};
    const long double vs = std::sqrt(static_cast<long double>(mu) / A) / (e * std::cosh(F) - 1.0L);
    const double q[3] = {static_cast<double>(-vs * std::sinh(F)), static_cast<double>(vs * sq * std::cosh(F)), 0.0};
    orient(p, inc, Om, r);
    orient(q, inc, Om, v);
}
}  // namespace

int main() {
    const double mu = 1.0, inc = 0.4, Om = 0.3, om = 1.1;
    // Bound orbits, a = 1 (P = 2 pi), starting at mean anomaly 0.7
    const double es[] = {0.0, 1e-12, 1e-8, 1e-6, 1e-4, 1e-2, 0.3, 0.9, 0.99, 0.9999};
    const double norbs[] = {0.37, 2.9, 1000.3};
    Worst low, high, longrun;
    for (double e : es) {
        double r0[3], v0[3];
        elements_to_cartesian(1.0, e, inc, Om, om, 0.7, mu, r0, v0);
        for (double no : norbs) {
            const double dt = no * 2.0 * M_PI;
            double rr[3], vr[3];
            elements_to_cartesian(1.0, e, inc, Om, om, 0.7 + dt, mu, rr, vr);
            run_case(e < 0.5 ? low : high, r0, v0, mu, dt, rr, std::fabs(energy(r0, v0, mu)));
        }
        // 1e5 orbits: the phase itself is known only to eps n dt ~ 1e-10
        const double dt = (1e5 + 0.3) * 2.0 * M_PI;
        double rr[3], vr[3];
        elements_to_cartesian(1.0, e, inc, Om, om, std::fmod(0.7 + dt, 2.0 * M_PI), mu, rr, vr);
        run_case(longrun, r0, v0, mu, dt, rr, std::fabs(energy(r0, v0, mu)));
    }
    low.print("bound_e<0.5");
    high.print("bound_e>=0.5");
    longrun.print("bound_1e5_orbits");

    // Parabola: q = 1, starting inbound at true anomaly -2 (no element reference)
    Worst para;
    {
        const double q = 1.0, p = 2.0 * q, f = -2.0, rm = p / (1.0 + std::cos(f));
        const double pp[3] = {rm * std::cos(f), rm * std::sin(f), 0.0};
        const double vp[3] = {-std::sqrt(mu / p) * std::sin(f), std::sqrt(mu / p) * (1.0 + std::cos(f)), 0.0};
        double r0[3], v0[3];
        orient(pp, inc, Om, r0);
        orient(vp, inc, Om, v0);
        const double tq = std::sqrt(2.0 * q * q * q / mu);   // pericentre crossing time
        for (double x : {0.1, 3.0, 1e3, 1e6}) run_case(para, r0, v0, mu, x * tq, nullptr, mu / q);
    }
    para.print("parabolic");

    // Hyperbolae: A = -a = 1, starting inbound at mean anomaly -5, intervals of 0.1 to 1e3 / n (n = 1), and 1e6 / n
    // (r ~ 1e6 A): there the Cartesian state itself carries |r| |v| eps of L (|r| |v| / L ~ 1e8 at e = 1.0001) and the
    // way back amplifies that, so only 1v16, ref and dE are meaningful
    Worst hyp, far;
    for (double e : {1.0001, 1.5, 5.0, 31100.0}) {
        double r0[3], v0[3];
        hyper_state(1.0, e, mu, -5.0L, inc, Om, r0, v0);
        for (double x : {0.1, 3.0, 1e3, 1e6}) {
            double rr[3], vr[3];
            hyper_state(1.0, e, mu, -5.0L + x, inc, Om, rr, vr);
            run_case(x < 1e4 ? hyp : far, r0, v0, mu, x, rr, std::fabs(energy(r0, v0, mu)));
        }
    }
    hyp.print("hyperbolic");
    far.print("hyperbolic_far");
    return 0;
}
