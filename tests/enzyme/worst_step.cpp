// Gradient-based worst-case search for the GBS error estimate (adversarial test of the LogH + GBS step; a trial
// tool, not a ctest). For a 3-body subsystem (eccentric binary + perturber) in the normalised units of
// logh_batch.cpp (c.o.m. frame, G = M = 1, V^2 = 2 T / M = 1) one GBS step of ds with K fixed stages (Bulirsch
// sequence, as GBSIntegrator) gives the extrapolation error estimate est (out.err, the number the adaptive path
// compares with eta) and its true error err against the same step with K + 3 stages, both in the norm of
// gbs_extrapolate (max over pos, vel, t, B of |dy| / max(1, |y|)). The search maximises
//   R = ln((err + floor) / (est + floor)) - 10 max(0, ln(est / eta)),   eta = 1e-10
// (floor = 1e-13 by default: ratios at round-off are not hard cases; the penalty keeps the search on steps the
// adaptive path would accept, est < eta: unconverged steps, est ~ 1, have large ratios but are rejected) over 11
// box-bounded parameters (unit cube, see build()): R >> 0 is a step the adaptive path would accept with a real
// error far above its estimate. Gradient: Enzyme reverse mode (checked against central differences), Adam
// with projection onto the box from random starts; baseline: random search with the same wall-clock budget.
// Candidates are re-run on the adaptive path (gbs_step, k_fixed = 0, eta = 1e-10, k_max = 10, halving on
// rejection) against a long-double reference (worst_step_ld.cpp: 64 sub-steps of 8 stages, checked with 32).
//   clang++ -std=c++20 -O2 -fplugin=ClangEnzyme-18.so -Isrc/collisional
//           tests/enzyme/worst_step.cpp tests/enzyme/worst_step_ld.cpp -o worst_step   (one command)
//   ./worst_step [K=4] [starts=20] [steps=300] [seed=1] [floor=1e-13] [particle_file_out]
//   (WORST_LOG_DS_MAX: upper bound of log10 ds / ds_cap, default 0; WORST_PENALTY: weight, default 10)
// Trial 2026-10-17 (K = 3, 4, 5; floor 1e-13, 1e-11; ds up to 1 and 10 caps): no step fools the adaptive path
// (resolved true / estimate <= 2.2). Ratios up to e^2.6 against the double K + 3 reference are its round-off
// (long double: R <= 1.2); with floor 1e-11 the gradient finds no more than random search at equal time.
#include "../../src/collisional/logh.cpp"
#include "../../src/collisional/logh_step.cpp"
#include "../../src/collisional/post_newtonian_core.cpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

extern int enzyme_dup, enzyme_const;
template <typename... T> void __enzyme_autodiff(void*, T...);

static const int SEQ[] = {2, 4, 6, 8, 12, 16, 24, 32, 48, 64};   // GBSIntegrator::STEP_SEQ (bulirsch)
constexpr int NPAR = 11;
static double FLOOR = 1e-13;
static double PENALTY = 10.0;     // WORST_PENALTY
constexpr double ETA = 1e-10;     // the production tolerance (GBSIntegrator::eta_gbs)
static const char* PAR_NAME[NPAR] = {"log10 m2/m1", "log10 m3/(m1+m2)", "e_in", "f_in", "log10 d/a",
                                     "cos th_d", "phi_d", "v3/v_esc", "cos th_v", "phi_v", "log10 ds/ds_cap"};
static const double LO[NPAR] = {-3.0, -3.0, 0.0, -M_PI, std::log10(3.0), -1.0, 0.0, 0.0, -1.0, 0.0, -1.5};
static double HI[NPAR] = {0.0, 0.0, 0.999, M_PI, 2.0, 1.0, 2.0 * M_PI, 1.5, 1.0, 2.0 * M_PI, 0.0};

// Unit cube -> normalised 3-body state and ds. Inner binary (a = 1 before scaling) in the xy plane, pericentre on
// x, true anomaly f_in; perturber at distance d from the binary's c.o.m. (direction th_d, phi_d) with speed
// v3 / v_esc(d) relative to it (direction th_v, phi_v). ds = 10^p ds_cap, ds_cap = t_dyn Omega / 3 the cap of
// GBSIntegrator::integrate_to_time (t_dyn = sqrt(r_min^3 / (G M)), NBODY_GBS_DYNCAP = 3).
static void build(const double* u, LogHState& s, double& ds) {
    double p[NPAR];
    for (int i = 0; i < NPAR; i++) p[i] = LO[i] + u[i] * (HI[i] - LO[i]);
    const double q_in = std::pow(10.0, p[0]), q_out = std::pow(10.0, p[1]);
    double m[3] = {1.0, q_in, q_out * (1.0 + q_in)};
    const double M = m[0] + m[1] + m[2];
    for (int i = 0; i < 3; i++) m[i] /= M;
    const double e = p[2], f = p[3], d = std::pow(10.0, p[4]);
    const double mb = m[0] + m[1], pp = 1.0 - e * e, r = pp / (1.0 + e * std::cos(f)), vk = std::sqrt(mb / pp);
    const double rx[3] = {r * std::cos(f), r * std::sin(f), 0.0};
    const double rv[3] = {-vk * std::sin(f), vk * (e + std::cos(f)), 0.0};
    const double sd = std::sqrt(1.0 - p[5] * p[5]), sv = std::sqrt(1.0 - p[8] * p[8]);
    const double v3 = p[7] * std::sqrt(2.0 * (mb + m[2]) / d);
    const double X[3] = {d * sd * std::cos(p[6]), d * sd * std::sin(p[6]), d * p[5]};
    const double V[3] = {v3 * sv * std::cos(p[9]), v3 * sv * std::sin(p[9]), v3 * p[8]};
    double pos[9], vel[9];
    for (int k = 0; k < 3; k++) {   // binary c.o.m. at -m3 X / M, perturber at mb X / M (total c.o.m. at rest)
        const double cx = -m[2] * X[k], cv = -m[2] * V[k];
        pos[k] = cx - m[1] / mb * rx[k];  pos[3 + k] = cx + m[0] / mb * rx[k];  pos[6 + k] = mb * X[k];
        vel[k] = cv - m[1] / mb * rv[k];  vel[3 + k] = cv + m[0] / mb * rv[k];  vel[6 + k] = mb * V[k];
    }
    double T = 0.0;
    for (int i = 0; i < 3; i++)
        T += 0.5 * m[i] * (vel[3*i] * vel[3*i] + vel[3*i+1] * vel[3*i+1] + vel[3*i+2] * vel[3*i+2]);
    const double vs = std::sqrt(2.0 * T);   // logh_batch.cpp scales: V = sqrt(2 T / M), R = G M / V^2
    s = LogHState{};
    s.n_bodies = 3;
    for (int i = 0; i < 9; i++) { s.pos[i] = pos[i] * vs * vs; s.vel[i] = vel[i] / vs; }
    for (int i = 0; i < 3; i++) s.mass[i] = m[i];
    s.B = -logh_total_energy(s, 1.0);
    double r2min = 1e300;
    for (int i = 0; i < 3; i++)
        for (int j = i + 1; j < 3; j++) {
            double r2 = 0.0;
            for (int k = 0; k < 3; k++) r2 += (s.pos[3*j+k] - s.pos[3*i+k]) * (s.pos[3*j+k] - s.pos[3*i+k]);
            r2min = std::min(r2min, r2);
        }
    const double t_dyn = std::pow(r2min, 0.75);   // sqrt(r^3 / (G M)), G = M = 1
    ds = std::pow(10.0, p[10]) * t_dyn * logh_omega(s) / 3.0;
}

// The norm of gbs_extrapolate between two packed states
static double gbs_norm(const double* a, const double* b, int dim) {
    double e = 0.0;
    for (int d = 0; d < dim; d++) e = std::max(e, std::fabs(a[d] - b[d]) / std::max(1.0, std::fabs(b[d])));
    return e;
}

// One GBS step: fixed (k > 0 stages) or adaptive (k = 0, eta, k_max 10); returns the packed result
static GbsStepOut one_step(const LogHState& s0, double ds, int k, double eta, double* y) {
    GbsParams par;
    par.seq = SEQ;
    par.k_max = 10;
    par.k_fixed = k;
    par.eta = eta;
    LogHState s = s0;
    LogHStepStart start;
    logh_step_start(s, 1.0, 0.0, 0, start);
    GbsTable tab;
    GbsStepOut out;
    gbs_step(par, s, ds, tab, -1, &start, out);
    int dim;
    gbs_pack_state(s, y, dim);
    return out;
}

// Long-double reference (worst_step_ld.cpp): nsub sub-steps of k fixed stages, packed result
void gbs_reference_ld(const double* pos, const double* vel, const double* mass, double B, double ds, int nsub, int k,
                      double* y);
static void reference_ld(const LogHState& s, double ds, int nsub, double* y, int k = 8) {
    gbs_reference_ld(s.pos, s.vel, s.mass, s.B, ds, nsub, k, y);
}

static int g_K = 4;
static double objective_parts(const double* u, double* est_out, double* err_out) {
    LogHState s;
    double ds;
    build(u, s, ds);
    double yk[GBS_DIM_MAX], yr[GBS_DIM_MAX];
    const GbsStepOut o = one_step(s, ds, g_K, 1e-10, yk);
    // (In long double, gbs_reference_ld, the reference would be free of round-off, but Enzyme's gradient through
    // the x87 build is wrong: 1e5 relative against central differences, LTO build with LLDEnzyme-18.so)
    one_step(s, ds, g_K + 3, 1e-10, yr);
    const double err = gbs_norm(yk, yr, 3 * 6 + 2);
    if (est_out) { *est_out = o.err; *err_out = err; }
    // Steps the adaptive path would reject (est > eta) are penalised: their large ratios are harmless
    return std::log((err + FLOOR) / (o.err + FLOOR)) - PENALTY * std::max(0.0, std::log(o.err / ETA));
}
static double objective(const double* u) { return objective_parts(u, nullptr, nullptr); }

static void gradient(const double* u, double* g) {
    for (int i = 0; i < NPAR; i++) g[i] = 0.0;
    __enzyme_autodiff((void*)objective, enzyme_dup, u, g);
}

// Adaptive path on the state of u: GBS steps from ds (halved on rejection as integrate_to_time does) until one
// is accepted; its estimate and true error against the long-double reference (64 sub-steps; ref_err: 32 against
// 64), and against the double reference of 16 sub-steps at eta = 1e-14 (halved up to 4 times; a sub-step whose
// estimate stays above it, at round-off, takes the 10-stage result), whose round-off can exceed the error itself
struct AdaptiveCheck { double ds, est, err, ref_err, err_d, dE; int stage, rejected; };
static AdaptiveCheck adaptive_check(const double* u, double ds_factor) {
    LogHState s;
    double ds;
    build(u, s, ds);
    ds *= ds_factor;
    AdaptiveCheck c{};
    double y[GBS_DIM_MAX];
    GbsStepOut o;
    for (c.rejected = 0; c.rejected < 60; c.rejected++) {
        o = one_step(s, ds, 0, 1e-10, y);
        if (o.accepted) break;
        ds *= 0.5;
    }
    c.ds = ds; c.est = o.err; c.stage = o.stage;
    double y32[GBS_DIM_MAX], y64[GBS_DIM_MAX], yd[GBS_DIM_MAX];
    reference_ld(s, ds, 32, y32);
    reference_ld(s, ds, 64, y64);
    LogHState q = s;
    for (int i = 0; i < 16; i++) {
        const double h = ds / 16;
        double done = 0.0;
        while (done < h * (1.0 - 1e-12)) {
            double hh = h - done;
            GbsStepOut oo;
            for (int t = 0; t < 5; t++, hh *= 0.5) {
                oo = one_step(q, hh, 0, 1e-14, yd);
                if (oo.accepted) break;
            }
            if (!oo.accepted) one_step(q, hh *= 2.0, 10, 0.0, yd);
            gbs_unpack_state(yd, q);
            done += hh;
        }
    }
    int dim;
    gbs_pack_state(q, yd, dim);
    c.err = gbs_norm(y, y64, 20);
    c.ref_err = gbs_norm(y32, y64, 20);
    c.err_d = gbs_norm(y, yd, 20);
    LogHState sa = s;
    gbs_unpack_state(y, sa);
    c.dE = std::fabs(logh_total_energy(sa, 1.0) - logh_total_energy(s, 1.0)) / std::fabs(logh_total_energy(s, 1.0));
    return c;
}

static void print_config(const double* u) {
    for (int i = 0; i < NPAR; i++) std::printf("    %-18s %.6g\n", PAR_NAME[i], LO[i] + u[i] * (HI[i] - LO[i]));
    std::printf("    u =");
    for (int i = 0; i < NPAR; i++) std::printf(" %.17g", u[i]);
    std::printf("\n");
}

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    g_K = argc > 1 ? std::atoi(argv[1]) : 4;
    const int starts = argc > 2 ? std::atoi(argv[2]) : 20, steps = argc > 3 ? std::atoi(argv[3]) : 300;
    const unsigned seed = argc > 4 ? std::atoi(argv[4]) : 1;
    if (argc > 5) FLOOR = std::atof(argv[5]);
    if (std::getenv("WORST_PENALTY")) PENALTY = std::atof(std::getenv("WORST_PENALTY"));
    if (std::getenv("WORST_LOG_DS_MAX")) HI[10] = std::atof(std::getenv("WORST_LOG_DS_MAX"));   // ds above the cap
    const char* out_file = argc > 6 ? argv[6] : nullptr;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::printf("K = %d stages (reference K + 3), %d starts x %d Adam steps, seed %u, floor %.0e, penalty %g, "
                "ds <= 10^%g cap\n", g_K, starts, steps, seed, FLOOR, PENALTY, HI[10]);

    // 1. Gradient check: Enzyme against central differences (h = 1e-5 and 1e-6 in the unit cube) at 3 random
    //    points, with K = 2 and both errors above 1e-7: the true error is a difference of two extrapolated states,
    //    so its round-off (~1e-14) makes central differences of R noise for errors near eta
    double worst_rel = 0.0;
    const int K_run = g_K;
    g_K = 2;
    for (int t = 0; t < 3; t++) {
        double u[NPAR], g[NPAR], est = 0.0, err = 0.0;
        do {
            for (int i = 0; i < NPAR; i++) u[i] = 0.1 + 0.8 * uni(rng);
            objective_parts(u, &est, &err);
        } while (!(est > 1e-7 && err > 1e-7));
        gradient(u, g);
        double dmax = 0.0, fmax = 0.0;
        std::printf("gradient check %d: R = %.4f (estimate %.2e, true error %.2e)\n", t, objective(u), est, err);
        for (int i = 0; i < NPAR; i++) {
            double fd[2];
            for (int k = 0; k < 2; k++) {
                const double h = k == 0 ? 1e-5 : 1e-6;
                double up[NPAR], um[NPAR];
                for (int j = 0; j < NPAR; j++) up[j] = um[j] = u[j];
                up[i] += h; um[i] -= h;
                fd[k] = (objective(up) - objective(um)) / (2.0 * h);
            }
            dmax = std::max(dmax, std::fabs(g[i] - fd[0]));
            fmax = std::max(fmax, std::fabs(fd[0]));
            std::printf("  dR/du[%2d] enzyme % .8e  central (1e-5) % .8e  (1e-6) % .8e\n", i, g[i], fd[0], fd[1]);
        }
        // Relative to the gradient's largest component: components ~1e-5 of it are below the differences' noise
        std::printf("  max |enzyme - central| / max |central| = %.1e\n", dmax / fmax);
        worst_rel = std::max(worst_rel, dmax / fmax);
    }
    std::printf("gradient check: worst relative difference %.1e (h = 1e-5)\n", worst_rel);
    g_K = K_run;

    // 2. Projected Adam from random starts
    double best_g = -1e300, ubest_g[NPAR] = {};
    long n_eval_g = 0;
    const double t0 = now();
    for (int st = 0; st < starts; st++) {
        double u[NPAR], mom[NPAR] = {}, var[NPAR] = {}, best_s = -1e300;
        for (int i = 0; i < NPAR; i++) u[i] = uni(rng);
        const double lr = 0.02, b1 = 0.9, b2 = 0.999;
        for (int it = 1; it <= steps; it++) {
            const double R = objective(u);
            n_eval_g++;
            if (R > best_s) best_s = R;
            if (R > best_g) { best_g = R; for (int i = 0; i < NPAR; i++) ubest_g[i] = u[i]; }
            double g[NPAR];
            gradient(u, g);
            for (int i = 0; i < NPAR; i++) {
                if (!std::isfinite(g[i])) g[i] = 0.0;
                mom[i] = b1 * mom[i] + (1.0 - b1) * g[i];
                var[i] = b2 * var[i] + (1.0 - b2) * g[i] * g[i];
                const double mh = mom[i] / (1.0 - std::pow(b1, it)), vh = var[i] / (1.0 - std::pow(b2, it));
                u[i] = std::min(1.0, std::max(0.0, u[i] + lr * mh / (std::sqrt(vh) + 1e-12)));   // ascent
            }
        }
        std::printf("start %2d: best R = %.3f\n", st, best_s);
    }
    const double t_grad = now() - t0;

    // 3. Random search with the same wall-clock budget (and its best after the same number of evaluations)
    double best_r = -1e300, best_r_n = -1e300, ubest_r[NPAR] = {};
    long n_eval_r = 0;
    const double t1 = now();
    while (now() - t1 < t_grad) {
        for (int b = 0; b < 100; b++) {
            double u[NPAR];
            for (int i = 0; i < NPAR; i++) u[i] = uni(rng);
            const double R = objective(u);
            n_eval_r++;
            if (R > best_r) { best_r = R; for (int i = 0; i < NPAR; i++) ubest_r[i] = u[i]; }
            if (n_eval_r == n_eval_g) best_r_n = best_r;
        }
    }
    if (best_r_n == -1e300) best_r_n = best_r;
    std::printf("gradient: best R = %.3f (%ld objective + gradient evaluations, %.1f s)\n", best_g, n_eval_g, t_grad);
    std::printf("random:   best R = %.3f (%ld evaluations, same time); %.3f after %ld evaluations\n", best_r,
                n_eval_r, best_r_n, n_eval_g);

    // 4. The best of each on the fixed and the adaptive path
    const double* cand[2] = {ubest_g, ubest_r};
    const char* name[2] = {"gradient", "random"};
    double fool_best = 0.0;
    const double* fool_u = nullptr;
    for (int c = 0; c < 2; c++) {
        double est, err;
        const double R = objective_parts(cand[c], &est, &err);
        LogHState s;
        double ds, yk[GBS_DIM_MAX], yl[GBS_DIM_MAX];
        build(cand[c], s, ds);
        one_step(s, ds, g_K, 0.0, yk);
        reference_ld(s, ds, 64, yl);
        const double err_ld = gbs_norm(yk, yl, 20);
        std::printf("%s best: R = %.3f, K = %d estimate %.3e, true error %.3e (K + 3 reference), %.3e (long "
                    "double reference: R = %.3f)\n", name[c], R, g_K, est, err, err_ld,
                    std::log((err_ld + FLOOR) / (est + FLOOR)) - PENALTY * std::max(0.0, std::log(est / ETA)));
        print_config(cand[c]);
        for (double f : {1.0, 1.0 / std::pow(10.0, LO[10] + cand[c][10] * (HI[10] - LO[10]))}) {
            const AdaptiveCheck a = adaptive_check(cand[c], f);
            std::printf("  adaptive (eta 1e-10, ds %s): ds %.4e after %d rejections, stage %d, estimate %.3e, "
                        "true error %.3e (long double reference, its error %.1e; double eta 1e-14 reference: %.3e), "
                        "|dE/E| %.2e, true / estimate %.2g\n", f == 1.0 ? "found" : "= cap", a.ds, a.rejected,
                        a.stage, a.est, a.err, a.ref_err, a.err_d, a.dE, a.err / a.est);
            // Counts only when the reference resolves the error (3x its own) and the error is above round-off
            if (a.err > 3.0 * a.ref_err && a.err > 1e-12 && a.err / a.est > fool_best) {
                fool_best = a.err / a.est;
                fool_u = cand[c];
            }
        }
    }
    std::printf("adaptive path: largest resolved true / estimate %.3g (> 10: fooled)\n", fool_best);

    // 5. Particle file of a fooling configuration (m x y z vx vy vz kw R): 20 Msun, a_in = 1 AU, BHs (R = 0)
    if (out_file && fool_u && fool_best > 10.0) {
        LogHState s;
        double ds;
        build(fool_u, s, ds);
        const double Gp = 4.302194e-3, AU = 4.84813681e-6, Mp = 20.0;
        double r12 = 0.0, v12 = 0.0;   // a_in in normalised units from the inner orbit: 1 / a = 2 / r - v^2 / (G m_b)
        for (int k = 0; k < 3; k++) {
            r12 += (s.pos[3+k] - s.pos[k]) * (s.pos[3+k] - s.pos[k]);
            v12 += (s.vel[3+k] - s.vel[k]) * (s.vel[3+k] - s.vel[k]);
        }
        const double a_in = 1.0 / (2.0 / std::sqrt(r12) - v12 / (s.mass[0] + s.mass[1]));
        const double L = AU / a_in, V = std::sqrt(Gp * Mp / L);   // length, velocity units (G M / (L V^2) = 1)
        FILE* fp = std::fopen(out_file, "w");
        std::fprintf(fp, "# tests/enzyme/worst_step (K = %d, adaptive true / estimate %.3g): binary + perturber, "
                     "20 Msun, a_in = 1 AU, BHs; c.o.m. frame; m x y z vx vy vz kw R\n", g_K, fool_best);
        for (int i = 0; i < 3; i++)
            std::fprintf(fp, "%.17g %.17g %.17g %.17g %.17g %.17g %.17g 14 0\n", s.mass[i] * Mp, s.pos[3*i] * L,
                         s.pos[3*i+1] * L, s.pos[3*i+2] * L, s.vel[3*i] * V, s.vel[3*i+1] * V, s.vel[3*i+2] * V);
        std::fclose(fp);
        std::printf("wrote %s\n", out_file);
    }
    return 0;
}
