// Target 1: LogH leapfrog (logh_leapfrog_steps / logh_leapfrog_steps_pn) in extended phase space z = (q, t; p, B)
// with p = m v, p_t = B = -E. Control: textbook LogH leapfrog (Mikkola & Tanikawa 1999; Preto & Tremaine 1999),
// kick with 1/(-U(q)), drift with 1/(T + B): symplectic in z by construction.
// Build (from the repository root; logh_ctrl.cpp: python3.11 tests/enzyme/symplectic/make_ctrl.py <dir>):
//   clang++ -std=c++20 -O2 -fplugin=ClangEnzyme-18.so -Isrc/collisional tests/enzyme/symplectic/symp_logh.cpp -o symp_logh
//   (Enzyme)
#include "../../../src/collisional/logh.cpp"
#include "../../../src/collisional/logh_step.cpp"
#include "../../../src/collisional/post_newtonian_core.cpp"
#include "jac.h"
#include <cstdlib>

static int g_n = 2, g_mode = 0, g_nsteps = 1, g_pn = 0;
static double g_mass[LOGH_MAX_BODIES], g_ds = 1e-2, g_c = 30.0;
// mode 0: code Newtonian; 1: control (1/(-U) kick); 2: code PN (w = v at the start)

static void load(const double* z, LogHState& s) {
    std::memset(&s, 0, sizeof(s));
    const int n = g_n, n3 = 3 * n;
    s.n_bodies = n;
    for (int i = 0; i < n; i++) s.mass[i] = g_mass[i];
    for (int i = 0; i < n3; i++) { s.pos[i] = z[i]; s.vel[i] = z[n3 + 1 + i] / g_mass[i / 3]; }
    s.t = z[n3];
    s.B = z[2 * n3 + 1];
    s.c_code = g_c;
}
static void store(const LogHState& s, double* out) {
    const int n3 = 3 * g_n;
    for (int i = 0; i < n3; i++) { out[i] = s.pos[i]; out[n3 + 1 + i] = s.vel[i] * g_mass[i / 3]; }
    out[n3] = s.t;
    out[2 * n3 + 1] = s.B;
}
static void control_step(LogHState& s, double ds) {
    const int n3 = 3 * s.n_bodies;
    double a[LOGH_MAX_BODIES * 3];
    for (int half = 0; half < 2; half++) {
        logh_internal_accel(s, 1.0, a);
        const double U = logh_potential_energy(s, 1.0);
        for (int i = 0; i < n3; i++) s.vel[i] += 0.5 * ds * a[i] / (-U);
        if (half == 0) {
            const double Om = logh_kinetic_energy(s) + s.B;
            for (int i = 0; i < n3; i++) s.pos[i] += ds * s.vel[i] / Om;
            s.t += ds / Om;
        }
    }
}
static void step_map(const double* z, double* out) {
    LogHState s;
    load(z, s);
    if (g_mode == 0) logh_leapfrog_steps(s, g_ds, 1.0, g_nsteps, nullptr);
    else if (g_mode == 1) for (int k = 0; k < g_nsteps; k++) control_step(s, g_ds);
    else {
        s.use_aux_vel = true;
        for (int i = 0; i < 3 * g_n; i++) s.aux_vel[i] = s.vel[i];
        logh_leapfrog_steps_pn(s, g_ds, 1.0, g_c, g_pn, g_nsteps, nullptr);
    }
    store(s, out);
}

// --- invariance / equivariance of the map's derivative ---
struct Checks { double tr, rot, dP, dL, Pval, Lval; };
static void PL(const double* z, double P[3], double L[3]) {
    const int n3 = 3 * g_n;
    for (int k = 0; k < 3; k++) P[k] = L[k] = 0;
    for (int i = 0; i < g_n; i++) {
        const double* q = z + 3 * i; const double* p = z + n3 + 1 + 3 * i;
        for (int k = 0; k < 3; k++) P[k] += p[k];
        L[0] += q[1] * p[2] - q[2] * p[1]; L[1] += q[2] * p[0] - q[0] * p[2]; L[2] += q[0] * p[1] - q[1] * p[0];
    }
}
// generator of a rotation about axis k (or translation along k) at z
static void gen(const double* z, int k, bool rot, double* xi, int d) {
    const int n3 = 3 * g_n;
    for (int j = 0; j < d; j++) xi[j] = 0;
    const double e[3] = {k == 0 ? 1.0 : 0.0, k == 1 ? 1.0 : 0.0, k == 2 ? 1.0 : 0.0};
    for (int i = 0; i < g_n; i++) {
        if (!rot) { xi[3 * i + k] = 1.0; continue; }
        for (int blk = 0; blk < 2; blk++) {
            const double* x = z + (blk ? n3 + 1 : 0) + 3 * i;
            double* o = xi + (blk ? n3 + 1 : 0) + 3 * i;
            o[0] = e[1] * x[2] - e[2] * x[1]; o[1] = e[2] * x[0] - e[0] * x[2]; o[2] = e[0] * x[1] - e[1] * x[0];
        }
    }
}
static Checks checks(const double* z, const double* zo, const double M[DMAX][DMAX], int d) {
    Checks c{};
    double scale = 0; for (int j = 0; j < d; j++) scale = std::max(scale, std::fabs(zo[j]));
    for (int k = 0; k < 3; k++)
        for (int rot = 0; rot < 2; rot++) {
            double xi[DMAX], xo[DMAX];
            gen(z, k, rot, xi, d); gen(zo, k, rot, xo, d);
            double w = 0;
            for (int i = 0; i < d; i++) { double s = 0; for (int j = 0; j < d; j++) s += M[i][j] * xi[j]; w = std::max(w, std::fabs(s - xo[i])); }
            (rot ? c.rot : c.tr) = std::max(rot ? c.rot : c.tr, w / std::max(1.0, scale));
        }
    // grad P(z') M - grad P(z); grad L likewise (rows)
    const int n3 = 3 * g_n;
    for (int k = 0; k < 3; k++) {
        for (int j = 0; j < d; j++) {
            double s = 0;
            for (int i = 0; i < g_n; i++) s += M[n3 + 1 + 3 * i + k][j];
            double ref = 0; for (int i = 0; i < g_n; i++) if (j == n3 + 1 + 3 * i + k) ref = 1;
            c.dP = std::max(c.dP, std::fabs(s - ref));
        }
        const int k1 = (k + 1) % 3, k2 = (k + 2) % 3;
        for (int j = 0; j < d; j++) {
            double s = 0, ref = 0;
            for (int i = 0; i < g_n; i++) {
                const double* q = zo + 3 * i; const double* p = zo + n3 + 1 + 3 * i;
                const double* q0 = z + 3 * i; const double* p0 = z + n3 + 1 + 3 * i;
                // L_k = q_k1 p_k2 - q_k2 p_k1
                s += p[k2] * M[3 * i + k1][j] + q[k1] * M[n3 + 1 + 3 * i + k2][j] - p[k1] * M[3 * i + k2][j] - q[k2] * M[n3 + 1 + 3 * i + k1][j];
                if (j == 3 * i + k1) ref += p0[k2];
                if (j == n3 + 1 + 3 * i + k2) ref += q0[k1];
                if (j == 3 * i + k2) ref -= p0[k1];
                if (j == n3 + 1 + 3 * i + k1) ref -= q0[k2];
            }
            c.dL = std::max(c.dL, std::fabs(s - ref));
        }
    }
    double P0[3], L0[3], P1[3], L1[3];
    PL(z, P0, L0); PL(zo, P1, L1);
    for (int k = 0; k < 3; k++) { c.Pval = std::max(c.Pval, std::fabs(P1[k] - P0[k])); c.Lval = std::max(c.Lval, std::fabs(L1[k] - L0[k])); }
    return c;
}

// initial conditions: 2-body (a = 1, e, phase f0) or hierarchical 3-body; B = -E
static void ics(int sys, double e, double f0, double* z) {
    const int n3 = 3 * g_n;
    std::memset(z, 0, DMAX * sizeof(double));
    double pos[9] = {}, vel[9] = {};
    auto binary = [&](double m1, double m2, double a, double ee, double f, double* x1, double* x2, double* v1, double* v2) {
        const double mu = m1 + m2, p = a * (1 - ee * ee), r = p / (1 + ee * std::cos(f));
        const double rx[3] = {r * std::cos(f), r * std::sin(f), 0.0};
        const double rv[3] = {-std::sin(f) * std::sqrt(mu / p), (ee + std::cos(f)) * std::sqrt(mu / p), 0.0};
        for (int k = 0; k < 3; k++) { x1[k] = -m2 / mu * rx[k]; x2[k] = m1 / mu * rx[k]; v1[k] = -m2 / mu * rv[k]; v2[k] = m1 / mu * rv[k]; }
    };
    if (sys == 2) {
        g_mass[0] = 0.6; g_mass[1] = 0.4;
        binary(0.6, 0.4, 1.0, e, f0, pos, pos + 3, vel, vel + 3);
        pos[2] += 0.01; vel[5] += 0.02;
    } else {
        g_mass[0] = 0.5; g_mass[1] = 0.3; g_mass[2] = 0.2;
        double xi[3], xo[3], vi[3], vo[3], X1[3], X2[3], V1[3], V2[3];
        binary(0.5, 0.3, 1.0, e, f0, X1, X2, V1, V2);
        binary(0.8, 0.2, 5.0, 0.3, 1.0, xi, xo, vi, vo);
        for (int k = 0; k < 3; k++) {
            pos[k] = X1[k] + xi[k]; pos[3 + k] = X2[k] + xi[k]; pos[6 + k] = xo[k];
            vel[k] = V1[k] + vi[k]; vel[3 + k] = V2[k] + vi[k]; vel[6 + k] = vo[k];
        }
        pos[8] += 0.3; vel[2] += 0.05; vel[5] -= 0.03;   // inclined
    }
    LogHState s{}; s.n_bodies = g_n;
    for (int i = 0; i < g_n; i++) s.mass[i] = g_mass[i];
    for (int i = 0; i < n3; i++) { s.pos[i] = pos[i]; s.vel[i] = vel[i]; z[i] = pos[i]; z[n3 + 1 + i] = vel[i] * g_mass[i / 3]; }
    z[n3] = 0.3;
    z[2 * n3 + 1] = -logh_total_energy(s, 1.0);
}
static void shell_grad(const double* z, double* g) {
    const int n3 = 3 * g_n, d = 2 * (n3 + 1);
    LogHState s; load(z, s);
    double a[LOGH_MAX_BODIES * 3]; logh_internal_accel(s, 1.0, a);
    for (int j = 0; j < d; j++) g[j] = 0;
    for (int i = 0; i < n3; i++) { g[i] = -g_mass[i / 3] * a[i]; g[n3 + 1 + i] = s.vel[i]; }
    g[2 * n3 + 1] = 1.0;
}

static double M[DMAX][DMAX], Mfd[DMAX][DMAX], Mb[DMAX][DMAX];

int main(int argc, char** argv) {
    const char* what = argc > 1 ? argv[1] : "newton";
    std::printf("# sys mode pn c e f0 ds nsteps | symp_defect shell_defect det-1 | rev_value rev_jac | tr rot dP dL Pval Lval | fd_vs_enzyme\n");
    const double es[] = {0.0, 0.5, 0.9, 0.99};
    const double dss[] = {1e-1, 3e-2, 1e-2, 3e-3, 1e-3};
    for (int sys = 2; sys <= 3; sys++) {
        g_n = sys;
        const int d = 2 * (3 * g_n + 1);
        for (int mode : {0, 1, 2}) {
            if (std::strcmp(what, "newton") == 0 && mode == 2) continue;
            if (std::strcmp(what, "pn") == 0 && mode != 2) continue;
            for (double e : es)
                for (double ds : dss) {
                    for (double c : (mode == 2 ? std::vector<double>{30.0, 100.0, 1000.0} : std::vector<double>{0.0})) {
                        g_mode = mode; g_ds = ds; g_c = c; g_pn = mode == 2 ? 7 : 0; g_nsteps = 1;
                        const double f0 = -0.3;   // just before pericentre
                        double z[DMAX], zo[DMAX], zb[DMAX], g[DMAX];
                        ics(sys, e, f0, z);
                        step_map(z, zo);
                        jac_enzyme<step_map>(z, d, M);
                        jac_fd(step_map, z, d, Mfd, 1e-4, 1.0);
                        const double fdd = max_diff(Mfd, M, d);
                        shell_grad(z, g);
                        const double sd = symp_defect(M, d), sh = symp_defect_shell(M, d, g), dt = det(M, d) - 1.0;
                        // reversibility: step(-ds) o step(ds)
                        g_ds = -ds;
                        step_map(zo, zb);
                        jac_enzyme<step_map>(zo, d, Mb);
                        g_ds = ds;
                        double rv = 0; for (int j = 0; j < d; j++) rv = std::max(rv, std::fabs(zb[j] - z[j]) / std::max(1.0, std::fabs(z[j])));
                        const double rj = prod_minus_I(Mb, M, d);
                        Checks ck = checks(z, zo, M, d);
                        std::printf("%d %d %d %6g %4.2f %5.2f %6.0e %d | %9.2e %9.2e %9.2e | %9.2e %9.2e | %8.1e %8.1e %8.1e %8.1e %8.1e %8.1e | %8.1e\n",
                                    sys, mode, g_pn, c, e, f0, ds, g_nsteps, sd, sh, dt, rv, rj, ck.tr, ck.rot, ck.dP, ck.dL, ck.Pval, ck.Lval, fdd);
                    }
                }
        }
    }
}
