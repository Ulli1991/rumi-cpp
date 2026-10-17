// Monte Carlo arithmetic (Verificarlo) driver: plain C++ (no Kokkos) runs of the LogH + GBS step, the Kepler drift
// and the secular PN binary, each printing the quantities whose significant digits tests/vfc/vfc_run.py estimates
// over random-rounding samples. Usage: vfc_driver <case 1-6> [b]; one "name value" line per quantity (%.17e).
//   1  e = 0.999 binary, one orbit apocentre -> pericentre (dE/E_mid) -> apocentre (GBS fixed path, 8 stages)
//   2  e = 0.9 binary + perturber (3 bodies), about 8 inner orbits (GBS fixed path, 8 stages)
//   3  kepler_drift, 1e4 calls of one period each at e = 0.99
//   4  integrate_secular_binary, PN 31 at x = G m / (a c^2) = 1e-3, e = 0.7, 1000 orbits
//   5  kepler_drift near the parabola, 1 - e = 1e-6 (universal variables, Stumpff), 1000 calls of one period
//   6  the same at 1 - e = 1e-4 (anomaly route)
// Build: tests/vfc/build_vfc.sh (verificarlo-c++ in the verificarlo/verificarlo:v2.7.0 container) or plain g++.
#include "../../src/collisional/kepler.h"
#include "../../src/collisional/logh_step.h"
#include "../../src/collisional/secular.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

// Deuflhard (harmonic) sequence, GBS_SEQUENCE = deuflhard; argument 2 = b: the default Bulirsch sequence (gbs.h)
static const int SEQ_D[] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
static const int SEQ_B[] = {2, 4, 6, 8, 12, 16, 24, 32, 48, 64};
static const int* SEQ = SEQ_D;

static void pr(const char* name, double v) { std::printf("%s %.17e\n", name, v); }

// Two-body state about the c.o.m., G = 1, a = 1, at true anomaly f (orbit in the x-y plane tilted by 0.3 rad)
static void binary(const double m[2], double e, double f, double* pos, double* vel) {
    const double M = m[0] + m[1], p = 1.0 - e * e, r = p / (1.0 + e * std::cos(f)), vs = std::sqrt(M / p);
    const double rx = r * std::cos(f), ry = r * std::sin(f), vx = -vs * std::sin(f), vy = vs * (e + std::cos(f));
    const double ci = std::cos(0.3), si = std::sin(0.3);
    const double rel_x[3] = {rx, ry * ci, ry * si}, rel_v[3] = {vx, vy * ci, vy * si};
    for (int k = 0; k < 3; k++) {
        pos[k] = -m[1] / M * rel_x[k];  pos[3 + k] = m[0] / M * rel_x[k];
        vel[k] = -m[1] / M * rel_v[k];  vel[3 + k] = m[0] / M * rel_v[k];
    }
}

// T + U measured here (the driver module: not instrumented by the function-level include lists, so their digit
// losses are those of the integration, not of the measurement)
static double meas_energy(const LogHState& s) {
    double T = 0.0, U = 0.0;
    for (int i = 0; i < s.n_bodies; i++) {
        const double* v = s.vel + 3 * i;
        T += 0.5 * s.mass[i] * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        for (int j = i + 1; j < s.n_bodies; j++) {
            const double dx = s.pos[3*j] - s.pos[3*i], dy = s.pos[3*j+1] - s.pos[3*i+1], dz = s.pos[3*j+2] - s.pos[3*i+2];
            U -= s.mass[i] * s.mass[j] / std::sqrt(dx * dx + dy * dy + dz * dz);
        }
    }
    return T + U;
}

// GBS steps of logh_gbs_step (fixed path) with the error estimate of every step kept
// s_total: LogH time of the run (ds = Omega dt, Omega = -U on shell)
static void gbs_run(int n, double* pos, double* vel, const double* mass, double s_total, int n_steps, int k_fixed) {
    GbsParams par;
    par.seq = SEQ;
    par.k_fixed = k_fixed;
    LogHState st{};
    st.n_bodies = n;
    for (int i = 0; i < 3 * n; i++) { st.pos[i] = pos[i]; st.vel[i] = vel[i]; }
    for (int i = 0; i < n; i++) st.mass[i] = mass[i];
    st.B = -logh_total_energy(st, 1.0);
    const double E0 = meas_energy(st);
    const double ds = s_total / n_steps;
    static GbsTable tab;
    double err_max = 0.0, err_last = 0.0, dE_max = 0.0;
    for (int s = 0; s < n_steps; s++) {
        LogHStepStart start;
        logh_step_start(st, 1.0, 0.0, 0, start);
        GbsStepOut o;
        gbs_step(par, st, ds, tab, -1, &start, o);
        err_last = o.err;
        err_max = std::max(err_max, o.err);
        dE_max = std::max(dE_max, std::fabs(meas_energy(st) - E0));
        if (2 * (s + 1) == n_steps) pr("dE/E_mid", (meas_energy(st) - E0) / std::fabs(E0));
    }
    const double E1 = meas_energy(st);
    pr("dE/E", (E1 - E0) / std::fabs(E0));
    pr("max|dE/E|", dE_max / std::fabs(E0));
    pr("t", st.t);
    for (int i = 0; i < 3 * n; i++) { char b[16]; std::snprintf(b, sizeof b, "x%d", i); pr(b, st.pos[i]); }
    for (int i = 0; i < 3 * n; i++) { char b[16]; std::snprintf(b, sizeof b, "v%d", i); pr(b, st.vel[i]); }
    pr("gbs_err_last", err_last);
    pr("gbs_err_max", err_max);
}

static double kepler_energy(const double r[3], const double v[3], double mu) {
    return 0.5 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) - mu / std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
}

int main(int argc, char** argv) {
    const int c = argc > 1 ? std::atoi(argv[1]) : 1;
    if (argc > 2 && argv[2][0] == 'b') SEQ = SEQ_B;
    if (c == 1) {
        const double m[2] = {0.6, 0.4};
        double pos[6], vel[6];
        binary(m, 0.999, M_PI, pos, vel);
        gbs_run(2, pos, vel, m, 2.0 * M_PI * m[0] * m[1], 256, 8);   // one orbit: <-U> = m1 m2 / a
    } else if (c == 2) {
        const double m[3] = {0.5, 0.5, 0.1};
        double pos[9], vel[9], p2[6], v2[6];
        binary(m, 0.9, 2.0, p2, v2);
        for (int i = 0; i < 6; i++) { pos[i] = p2[i]; vel[i] = v2[i]; }
        // perturber at 4 a, moving past at about the local circular speed (mildly bound to the binary)
        pos[6] = 4.0; pos[7] = -1.0; pos[8] = 0.5;
        vel[6] = -0.1; vel[7] = 0.45; vel[8] = 0.0;
        // c.o.m. frame
        double M = 0, cp[3] = {0, 0, 0}, cv[3] = {0, 0, 0};
        for (int b = 0; b < 3; b++) { M += m[b]; for (int k = 0; k < 3; k++) { cp[k] += m[b] * pos[3*b+k]; cv[k] += m[b] * vel[3*b+k]; } }
        for (int b = 0; b < 3; b++) for (int k = 0; k < 3; k++) { pos[3*b+k] -= cp[k] / M; vel[3*b+k] -= cv[k] / M; }
        gbs_run(3, pos, vel, m, 8.0 * 2.0 * M_PI * m[0] * m[1], 800, 8);
    } else if (c == 3) {
        const double m[2] = {0.6, 0.4}, mu = 1.0, e = 0.99;
        double pos[6], vel[6], r[3], v[3];
        binary(m, e, 0.5, pos, vel);
        for (int k = 0; k < 3; k++) { r[k] = pos[3 + k] - pos[k]; v[k] = vel[3 + k] - vel[k]; }
        const double E0 = kepler_energy(r, v, mu), P = 2.0 * M_PI;
        double dE_max = 0.0;
        for (int i = 0; i < 10000; i++) {
            kepler_drift(r, v, mu, P * 1.0001);
            dE_max = std::max(dE_max, std::fabs(kepler_energy(r, v, mu) - E0));
        }
        pr("dE/E", (kepler_energy(r, v, mu) - E0) / std::fabs(E0));
        pr("max|dE/E|", dE_max / std::fabs(E0));
        for (int k = 0; k < 3; k++) { char b[8]; std::snprintf(b, sizeof b, "r%d", k); pr(b, r[k]); }
        for (int k = 0; k < 3; k++) { char b[8]; std::snprintf(b, sizeof b, "v%d", k); pr(b, v[k]); }
    } else if (c == 5 || c == 6) {
        // near-parabolic: 1 - e = 1e-6 (universal-variable f and g, Stumpff functions) and 1e-4 (anomaly route)
        const double m[2] = {0.6, 0.4}, mu = 1.0, e = c == 5 ? 1.0 - 1e-6 : 1.0 - 1e-4;
        double pos[6], vel[6], r[3], v[3];
        binary(m, e, 2.5, pos, vel);
        for (int k = 0; k < 3; k++) { r[k] = pos[3 + k] - pos[k]; v[k] = vel[3 + k] - vel[k]; }
        const double E0 = kepler_energy(r, v, mu), P = 2.0 * M_PI;
        double dE_max = 0.0;
        for (int i = 0; i < 1000; i++) {
            kepler_drift(r, v, mu, P * 1.0001);
            dE_max = std::max(dE_max, std::fabs(kepler_energy(r, v, mu) - E0));
        }
        pr("dE/E", (kepler_energy(r, v, mu) - E0) / std::fabs(E0));
        pr("max|dE/E|", dE_max / std::fabs(E0));
        for (int k = 0; k < 3; k++) { char b[8]; std::snprintf(b, sizeof b, "r%d", k); pr(b, r[k]); }
        for (int k = 0; k < 3; k++) { char b[8]; std::snprintf(b, sizeof b, "v%d", k); pr(b, v[k]); }
    } else if (c == 4) {
        const double m[2] = {0.6, 0.4}, x = 1e-3, cc = std::sqrt(1.0 / x);
        double pos[6], vel[6];
        binary(m, 0.7, 1.0, pos, vel);
        double r[3], v[3];
        for (int k = 0; k < 3; k++) { r[k] = pos[k] - pos[3 + k]; v[k] = vel[k] - vel[3 + k]; }
        const double E0 = kepler_energy(r, v, 1.0);
        const int nsub = integrate_secular_binary(pos, vel, m[0], pos + 3, vel + 3, m[1], 1.0, cc, 31, 1000.0 * 2.0 * M_PI);
        for (int k = 0; k < 3; k++) { r[k] = pos[k] - pos[3 + k]; v[k] = vel[k] - vel[3 + k]; }
        double eps, j;
        secular_pn_eps_j(r, v, 1.0, m[0] * m[1], cc, 31, eps, j);
        pr("dE/E", (kepler_energy(r, v, 1.0) - E0) / std::fabs(E0));
        pr("eps", eps);
        pr("j", j);
        pr("nsub", nsub);
        for (int i = 0; i < 6; i++) { char b[8]; std::snprintf(b, sizeof b, "x%d", i); pr(b, pos[i]); }
        for (int i = 0; i < 6; i++) { char b[8]; std::snprintf(b, sizeof b, "v%d", i); pr(b, vel[i]); }
    }
    return 0;
}
