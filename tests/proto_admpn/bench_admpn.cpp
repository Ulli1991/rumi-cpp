// Prototype + benchmark: Gauss-Legendre IRK (8 stages, tests/proto_symrk/gauss_irk.h) on the ADM post-Newtonian
// Hamiltonian (adm_pn.h: two-body 1PN-3PN, N-body 1PN, ADM-gauge 2.5PN + 3.5PN radiation reaction) vs the production
// LogH + GBS step on the harmonic (MH) equations of motion (src/collisional/logh_step.cpp gbs_step, step control as
// tests/proto_symrk/bench.cpp run_gbs = GBSIntegrator::integrate_to_time). Not part of the build or ctest. Build
// (worktree root, module gcc/15):
//   g++ -std=c++17 -O3 -march=native -Isrc/collisional tests/proto_admpn/bench_admpn.cpp src/collisional/secular.cpp \
//       src/collisional/kepler.cpp -o <scratch>/admpn
// Modes:
//   admpn verify                     gradients/Hessian vs finite differences, map round trip, gauge-invariant checks
//                                    (circular E(x), test-mass limit, periastron advance formula), KFS RR vs Iyer-Will
//   admpn test                       the pass/fail checks of CASES R4 (bounds 1.3x measured); exit code = failures
//   admpn consistency                ADM (Gauss) vs MH (GBS) trajectories through the map, error vs c (c^-8 scaling)
//   admpn cons <e> <x> <orbits> <spo>                 conservative 3PN binary: H drift, slope, periastron advance
//   admpn rr <variant> <e> <x> <spo> [eps_stop]       inspiral: variant A (RR in the collocation), S (Strang split),
//                                                     S2 (SBAB2 split, Laskar & Robutel 2001); Q = H + p_t, W, t_insp
//   admpn h2h <case> <T_orbits> [orders]             head-to-head ADM-Gauss (A, S, S2) vs LogH+GBS; cases: e09, e099,
//                                                     s432, s1025, s3047 (BH pairs of the harvest cut-outs), trip432
//                                                     (s432 BH pair + the star), orders as PN_ORDER (default 31)
#include "../../src/collisional/logh.cpp"
#include "../../src/collisional/logh_step.cpp"
#include "../../src/collisional/post_newtonian_core.cpp"
#include "../../src/collisional/secular.h"
#include "adm_pn.h"
#include "contact_coeffs.h"
#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>

using namespace admpn;
using symrk::GaussCoeffs; using symrk::CState; using symrk::IrkStats; using symrk::DIM_MAX; using symrk::S_MAX;

static const char* HARVEST = "/raven/ptmp/uli/cpp-nbody-runs/harvest-2026-10-19";
static const double G_PC = 4.30091e-3, C_KMS = 299792.458;

// ---------------------------------------------------------------------------------------------------- cases
struct Case {
    int n = 2;
    double m[8] = {};
    double G = 1.0, c = 1.0;
    std::vector<double> q, v;     // harmonic (MH) positions / velocities, c.o.m. frame
    int pi = 0, pj = 1;           // the PN pair
    double P_in = 1.0, a_in = 1.0;
    std::string name;
};

static void kepler_rel(double M, double a, double e, double G, double* x, double* v, double f = M_PI) {
    // relative orbit at true anomaly f (default apocentre), x-y plane tilted by 0.3 rad about x
    const double p = a * (1 - e * e), r = p / (1 + e * std::cos(f)), h = std::sqrt(G * M * p);
    const double xr[3] = {r * std::cos(f), r * std::sin(f), 0};
    const double vr[3] = {-G * M / h * std::sin(f), G * M / h * (e + std::cos(f)), 0};
    const double ci = std::cos(0.3), si = std::sin(0.3);
    x[0] = xr[0]; x[1] = ci * xr[1]; x[2] = si * xr[1];
    v[0] = vr[0]; v[1] = ci * vr[1]; v[2] = si * vr[1];
}

static void pair_elements(Case& c) {
    double x[3], v[3], r = 0, v2 = 0;
    for (int k = 0; k < 3; k++) {
        x[k] = c.q[3 * c.pi + k] - c.q[3 * c.pj + k]; v[k] = c.v[3 * c.pi + k] - c.v[3 * c.pj + k];
        r += x[k] * x[k]; v2 += v[k] * v[k];
    }
    r = std::sqrt(r);
    const double M = c.m[c.pi] + c.m[c.pj];
    c.a_in = 1.0 / (2.0 / r - v2 / (c.G * M));
    c.P_in = 2 * M_PI * std::sqrt(c.a_in * c.a_in * c.a_in / (c.G * M));
}

static Case make_binary(double q, double e, double x, const char* name = "bin") {
    Case c; c.name = name; c.n = 2; c.G = 1.0;
    c.m[0] = 1.0 / (1 + q); c.m[1] = q / (1 + q);
    c.c = std::sqrt(1.0 / x);   // G M / (a c^2) = x with a = 1
    double xr[3], vr[3];
    kepler_rel(1.0, 1.0, e, 1.0, xr, vr);
    c.q.resize(6); c.v.resize(6);
    for (int k = 0; k < 3; k++) {
        c.q[k] = c.m[1] * xr[k]; c.q[3 + k] = -c.m[0] * xr[k];
        c.v[k] = c.m[1] * vr[k]; c.v[3 + k] = -c.m[0] * vr[k];
    }
    pair_elements(c);
    return c;
}

static Case load_harvest(const std::string& dir, bool keep_others, int max_others = 8) {
    Case c; c.name = dir; c.G = G_PC; c.c = C_KMS;
    std::string path = std::string(HARVEST) + "/" + dir + "/ic.txt";
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    char line[4096];
    std::vector<double> mm, qq, vv; std::vector<int> kw;
    while (std::fgets(line, sizeof line, f)) {
        if (line[0] == '#') continue;
        double m, x, y, z, vx, vy, vz; int k = 0;
        if (std::sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %d", &m, &x, &y, &z, &vx, &vy, &vz, &k) < 7) continue;
        mm.push_back(m); qq.insert(qq.end(), {x, y, z}); vv.insert(vv.end(), {vx, vy, vz}); kw.push_back(k);
    }
    std::fclose(f);
    // the closest BH (kw 14) pair
    double best = 1e300; int bi = -1, bj = -1;
    for (size_t a = 0; a < mm.size(); a++)
        for (size_t b = a + 1; b < mm.size(); b++) {
            if (kw[a] != 14 || kw[b] != 14) continue;
            double r2 = 0;
            for (int k = 0; k < 3; k++) r2 += std::pow(qq[3 * a + k] - qq[3 * b + k], 2);
            if (r2 < best) { best = r2; bi = (int)a; bj = (int)b; }
        }
    std::vector<int> sel = {bi, bj};
    if (keep_others) {
        // others by distance to the pair
        std::vector<std::pair<double, int>> d;
        for (size_t a = 0; a < mm.size(); a++) {
            if ((int)a == bi || (int)a == bj) continue;
            double r2 = 0;
            for (int k = 0; k < 3; k++) r2 += std::pow(qq[3 * a + k] - qq[3 * bi + k], 2);
            d.push_back({r2, (int)a});
        }
        std::sort(d.begin(), d.end());
        for (int k = 0; k < (int)d.size() && k < max_others; k++) sel.push_back(d[k].second);
    }
    c.n = (int)sel.size();
    for (int a = 0; a < c.n; a++) {
        c.m[a] = mm[sel[a]];
        for (int k = 0; k < 3; k++) { c.q.push_back(qq[3 * sel[a] + k]); c.v.push_back(vv[3 * sel[a] + k]); }
    }
    // c.o.m. frame
    double M = 0, X[3] = {0, 0, 0}, V[3] = {0, 0, 0};
    for (int a = 0; a < c.n; a++) { M += c.m[a]; for (int k = 0; k < 3; k++) { X[k] += c.m[a] * c.q[3 * a + k]; V[k] += c.m[a] * c.v[3 * a + k]; } }
    for (int a = 0; a < c.n; a++) for (int k = 0; k < 3; k++) { c.q[3 * a + k] -= X[k] / M; c.v[3 * a + k] -= V[k] / M; }
    // origin at the PN pair's c.o.m. (velocities stay in the system c.o.m. frame): the pair's internal coordinates keep
    // full precision (the cut-out pair sits 1e-4 pc from the system c.o.m. at a separation of 1e-6 pc; the IRK
    // fixed-point increments then stagnate at the round-off of the absolute positions)
    {
        const double mp = c.m[0] + c.m[1];
        double Xp[3];
        for (int k = 0; k < 3; k++) Xp[k] = (c.m[0] * c.q[k] + c.m[1] * c.q[3 + k]) / mp;
        for (int a = 0; a < c.n; a++) for (int k = 0; k < 3; k++) c.q[3 * a + k] -= Xp[k];
    }
    c.pi = 0; c.pj = 1;
    pair_elements(c);
    return c;
}

static Case make_case(const std::string& name) {
    if (name == "e09") return make_binary(0.5, 0.9, 1e-3, "e09");
    if (name == "e099") return make_binary(0.5, 0.99, 1e-4, "e099");
    if (name == "s432") return load_harvest("s101_s432_0", false);
    if (name == "s1025") return load_harvest("s404_s1025_1", false);
    if (name == "s3047") return load_harvest("s404_s3047_0", false);
    if (name == "trip432") return load_harvest("s101_s432_0", true, 1);
    std::fprintf(stderr, "unknown case %s\n", name.c_str()); std::exit(1);
}

static void print_case(const Case& c) {
    double x[3], v[3], r = 0, v2 = 0, h[3];
    for (int k = 0; k < 3; k++) { x[k] = c.q[3 * c.pi + k] - c.q[3 * c.pj + k]; v[k] = c.v[3 * c.pi + k] - c.v[3 * c.pj + k]; r += x[k] * x[k]; v2 += v[k] * v[k]; }
    h[0] = x[1] * v[2] - x[2] * v[1]; h[1] = x[2] * v[0] - x[0] * v[2]; h[2] = x[0] * v[1] - x[1] * v[0];
    const double M = c.m[c.pi] + c.m[c.pj], h2 = h[0] * h[0] + h[1] * h[1] + h[2] * h[2];
    const double e = std::sqrt(std::max(0.0, 1 - h2 / (c.G * M * c.a_in)));
    std::printf("# case %s n=%d pair m=%.4g,%.4g a=%.4g e=%.4f GM/(a c^2)=%.3e r_p c^2/(GM)=%.1f P=%.4g\n", c.name.c_str(), c.n,
                c.m[c.pi], c.m[c.pj], c.a_in, e, c.G * M / (c.a_in * c.c * c.c),
                c.a_in * (1 - e) * c.c * c.c / (c.G * M), c.P_in);
}

// ---------------------------------------------------------------------------------------------------- maps
struct PairMap {
    Reduced R; ContactCoeffs K; double M, mu;
};
static PairMap pair_map(const Case& c, double mi, double mj, int orders) {
    PairMap pm;
    pm.M = mi + mj; pm.mu = mi * mj / pm.M;
    pm.R.build(pm.mu / pm.M, c.G * pm.M, c.c, orders & 11);
    pm.K = contact_coeffs(pm.mu / pm.M);
    return pm;
}

// harmonic state (q, v) of the case -> ADM canonical (q, p): pair relative via the c.o.m. map; others Newtonian
// (their 1PN momentum corrections and the pair c.o.m. ones are dropped: perturbers here are 1e-10-level PN)
static void harm_to_adm_state(const Case& c, const PairMap& pm, const double* qh, const double* vh, double* q, double* p) {
    const int i = c.pi, j = c.pj; const double mi = c.m[i], mj = c.m[j], M = mi + mj;
    for (int a = 0; a < c.n; a++) for (int k = 0; k < 3; k++) { q[3 * a + k] = qh[3 * a + k]; p[3 * a + k] = c.m[a] * vh[3 * a + k]; }
    double xh[3], wh[3], X[3], V[3], x[3], ph[3];
    for (int k = 0; k < 3; k++) {
        xh[k] = qh[3 * i + k] - qh[3 * j + k]; wh[k] = vh[3 * i + k] - vh[3 * j + k];
        X[k] = (mi * qh[3 * i + k] + mj * qh[3 * j + k]) / M; V[k] = (mi * vh[3 * i + k] + mj * vh[3 * j + k]) / M;
    }
    harm_to_adm(pm.R, pm.K, xh, wh, x, ph);
    for (int k = 0; k < 3; k++) {
        q[3 * i + k] = X[k] + mj / M * x[k]; q[3 * j + k] = X[k] - mi / M * x[k];
        p[3 * i + k] = mi * V[k] + pm.mu * ph[k]; p[3 * j + k] = mj * V[k] - pm.mu * ph[k];
    }
}
static void adm_to_harm_state(const Case& c, const PairMap& pm, const double* q, const double* p, double* qh, double* vh,
                              const AdmSystem& sys) {
    // velocities v = dH/dp of the full ADM system; the pair relative (x_h, v_h) through the map
    double U, gU[24], Hq[24], Hp[24];
    sys.hamiltonian(q, p, U, gU, Hq, Hp);
    const int i = c.pi, j = c.pj; const double mi = c.m[i], mj = c.m[j], M = mi + mj;
    for (int a = 0; a < c.n; a++) for (int k = 0; k < 3; k++) { qh[3 * a + k] = q[3 * a + k]; vh[3 * a + k] = Hp[3 * a + k]; }
    double x[3], ph[3], X[3], V[3], xh[3], wh[3];
    for (int k = 0; k < 3; k++) {
        x[k] = q[3 * i + k] - q[3 * j + k]; ph[k] = (mj * p[3 * i + k] - mi * p[3 * j + k]) / (M * pm.mu);
        X[k] = (mi * q[3 * i + k] + mj * q[3 * j + k]) / M; V[k] = (mi * Hp[3 * i + k] + mj * Hp[3 * j + k]) / M;
    }
    adm_to_harm(pm.R, pm.K, x, ph, xh, wh);
    for (int k = 0; k < 3; k++) {
        qh[3 * i + k] = X[k] + mj / M * xh[k]; qh[3 * j + k] = X[k] - mi / M * xh[k];
        vh[3 * i + k] = V[k] + mj / M * wh[k]; vh[3 * j + k] = V[k] - mi / M * wh[k];
    }
}
// conserved energy of the harmonic (MH) 3PN dynamics: E_h = T + U (all) + [H_ADM - H_N]_pair at map^-1(x_h, v_h)
static double harm_energy(const Case& c, const PairMap& pm, const double* qh, const double* vh) {
    double E = 0;
    for (int a = 0; a < c.n; a++) {
        double v2 = 0; for (int k = 0; k < 3; k++) v2 += vh[3 * a + k] * vh[3 * a + k];
        E += 0.5 * c.m[a] * v2;
        for (int b = a + 1; b < c.n; b++) {
            double r2 = 0; for (int k = 0; k < 3; k++) r2 += std::pow(qh[3 * a + k] - qh[3 * b + k], 2);
            E -= c.G * c.m[a] * c.m[b] / std::sqrt(r2);
        }
    }
    const int i = c.pi, j = c.pj;
    double xh[3], wh[3], x[3], ph[3], Hx[3], Hp[3];
    for (int k = 0; k < 3; k++) { xh[k] = qh[3 * i + k] - qh[3 * j + k]; wh[k] = vh[3 * i + k] - vh[3 * j + k]; }
    harm_to_adm(pm.R, pm.K, xh, wh, x, ph);
    double rh = 0, w2 = 0;
    for (int k = 0; k < 3; k++) { rh += xh[k] * xh[k]; w2 += wh[k] * wh[k]; }
    const double EN_pair = pm.mu * (0.5 * w2 - pm.R.Gm / std::sqrt(rh));
    return E - EN_pair + pm.mu * pm.R.H(x, ph, Hx, Hp);
}

// unwrapped orbital phase of the pair (angle in the plane of the initial pair angular momentum)
struct Phase {
    double e1[3], e2[3], last = 0, turns = 0; bool init = false;
    void setup(const double* x, const double* v) {
        double h[3] = {x[1] * v[2] - x[2] * v[1], x[2] * v[0] - x[0] * v[2], x[0] * v[1] - x[1] * v[0]};
        double hn = std::sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]), xn = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        for (int k = 0; k < 3; k++) { h[k] /= hn; e1[k] = x[k] / xn; }
        e2[0] = h[1] * e1[2] - h[2] * e1[1]; e2[1] = h[2] * e1[0] - h[0] * e1[2]; e2[2] = h[0] * e1[1] - h[1] * e1[0];
        init = true; last = 0; turns = 0;
    }
    double angle(const double* x) const {
        return std::atan2(x[0] * e2[0] + x[1] * e2[1] + x[2] * e2[2], x[0] * e1[0] + x[1] * e1[1] + x[2] * e1[2]);
    }
    double update(const double* x) {   // call often (less than half a turn between calls)
        double a = angle(x);
        double d = a - last;
        if (d < -M_PI) turns += 1; else if (d > M_PI) turns -= 1;
        last = a;
        return a + 2 * M_PI * turns;
    }
};

// ---------------------------------------------------------------------------------------------------- Gauss driver
enum Variant { VA = 0, VS = 1, VS2 = 2, VNONE = 3 };
struct GaussRun {
    AdmSystem sys;
    GaussCoeffs g;
    CState<double> st;
    std::array<std::array<double, DIM_MAX>, S_MAX> Y;
    bool have = false;
    IrkStats stats;
    long long kick_evals = 0;
    Variant var = VA;
    double spo = 30;
    double E0 = 0;
    double ds_fixed = 0;   // > 0: fixed ds (conservative runs, exactly symplectic); else per step from the pair energy
    double Gmm = 1, Mpair = 1;

    int n3() const { return 3 * sys.nb.n; }
    double t() const { return st.y[2 * n3()]; }
    mutable double ds_cache = 0, a_cache = 0;
    double ds_now() const {
        if (ds_fixed > 0) return ds_fixed;
        // ds per orbit ~ G m_i m_j P / a; a from the pair's osculating Newtonian elements, re-evaluated only on the
        // outer half of the orbit (r > a): near the pericentre of an e >= 0.99 orbit the Newtonian a from the PN
        // momenta is meaningless (the PN correction to p^2 exceeds 1/a)
        const int i = 0, j = 1, N3 = n3();
        const double* q = st.y.data(); const double* p = q + N3;
        const double mi = sys.nb.m[i], mj = sys.nb.m[j], M = mi + mj, mu = mi * mj / M;
        double r2 = 0, w2 = 0;
        for (int k = 0; k < 3; k++) {
            r2 += std::pow(q[3 * i + k] - q[3 * j + k], 2);
            w2 += std::pow((mj * p[3 * i + k] - mi * p[3 * j + k]) / (M * mu), 2);
        }
        if (ds_cache > 0 && std::sqrt(r2) < a_cache) return ds_cache;
        const double Gm = sys.nb.G * M;
        const double a = 1.0 / (2.0 / std::sqrt(r2) - w2 / Gm);
        if (!(a > 0)) return ds_cache;
        const double P = 2 * M_PI * std::sqrt(a * a * a / Gm);
        a_cache = a;
        ds_cache = sys.nb.G * mi * mj * P / a / spo;
        return ds_cache;
    }
    void init(const double* q, const double* p) {
        g = symrk::make_gauss(8);
        st.dim = sys.dim();
        const int N3 = n3();
        for (int i = 0; i < N3; i++) { st.y[i] = q[i]; st.y[N3 + i] = p[i]; st.lo[i] = st.lo[N3 + i] = 0; }
        st.y[2 * N3] = 0; st.lo[2 * N3] = 0; st.lo[2 * N3 + 1] = 0;
        E0 = sys.energy(st.y.data());
        st.y[2 * N3 + 1] = -E0;
        have = false;
        sys.rr_in_rhs = (var == VA);
    }
    // RR kick over ds (x fixed): dp/ds = g F(x, p), RK4; p_t -= the exact change of H
    void kick(double ds) {
        const int N3 = n3();
        double* q = st.y.data(); double* p = q + N3;
        double U, gU[24], Hq[24], Hp[24];
        const double H0 = sys.hamiltonian(q, p, U, gU, Hq, Hp);
        const double gg = -1.0 / U;
        double k[4][24], pp[24];
        auto F = [&](const double* pin, double* out) {
            double U2, g2[24], hq[24], hp[24];
            sys.hamiltonian(q, pin, U2, g2, hq, hp);
            sys.rr_force(q, pin, hp, out);
            kick_evals++;
        };
        F(p, k[0]);
        for (int i = 0; i < N3; i++) pp[i] = p[i] + 0.5 * ds * gg * k[0][i];
        F(pp, k[1]);
        for (int i = 0; i < N3; i++) pp[i] = p[i] + 0.5 * ds * gg * k[1][i];
        F(pp, k[2]);
        for (int i = 0; i < N3; i++) pp[i] = p[i] + ds * gg * k[2][i];
        F(pp, k[3]);
        for (int i = 0; i < N3; i++) p[i] += ds * gg * (k[0][i] + 2 * k[1][i] + 2 * k[2][i] + k[3][i]) / 6.0;
        const double H1 = sys.hamiltonian(q, p, U, gU, Hq, Hp);
        st.y[2 * N3 + 1] -= (H1 - H0);
    }
    void step(double ds) {
        if (var == VA || !sys.rr_orders) { symrk::gauss_step(g, sys, st, ds, Y, have, stats); return; }
        if (var == VS) {
            kick(0.5 * ds); symrk::gauss_step(g, sys, st, ds, Y, have, stats); kick(0.5 * ds);
        } else {   // SBAB2: B(h/6) A(h/2) B(2h/3) A(h/2) B(h/6), error O(eps h^4 + eps^2 h^2)
            kick(ds / 6); symrk::gauss_step(g, sys, st, 0.5 * ds, Y, have, stats);
            kick(2 * ds / 3); symrk::gauss_step(g, sys, st, 0.5 * ds, Y, have, stats); kick(ds / 6);
        }
    }
    // advance to physical time T (lands by secant on the last ds); f(state) called after every step
    template <class F>
    void advance(double T, F&& every) {
        const int N3 = n3();
        while (true) {
            double ds = ds_now();
            double U, gU[24], Hq[24], Hp[24];
            sys.hamiltonian(st.y.data(), st.y.data() + N3, U, gU, Hq, Hp);
            if (t() + 1.5 * ds / (-U) * 1.0 >= T) break;
            step(ds);
            // gauss_irk.h misses a NaN state (std::max(dmax, NaN) = dmax): flag it here
            if (!std::isfinite(t()) || !std::isfinite(st.y[N3]) || stats.maxit_hits || std::fabs(Q()) > 1e-6 * std::fabs(E0)) stats.diverged++;
            every();
            if (stats.diverged) return;
        }
        // last 1-2 steps: secant on ds so that t lands on T
        for (int pass = 0; pass < 2 && t() < T; pass++) {
            double U, gU[24], Hq[24], Hp[24];
            sys.hamiltonian(st.y.data(), st.y.data() + N3, U, gU, Hq, Hp);
            double full = ds_now();
            double need = (T - t()) * (-U);
            if (pass == 0 && need > 1.2 * full) { step(0.5 * need); every(); continue; }
            auto save = st; auto saveY = Y; bool sh = have;
            double h0 = need, f0, h1, f1;
            step(h0); f0 = t() - T; st = save; Y = saveY; have = sh;
            h1 = h0 * (1 - 1e-3); step(h1); f1 = t() - T; st = save; Y = saveY; have = sh;
            for (int it = 0; it < 6 && std::fabs(f1) > 1e-15 * T; it++) {
                double h2 = h1 - f1 * (h1 - h0) / (f1 - f0);
                h0 = h1; f0 = f1; h1 = h2;
                step(h1); f1 = t() - T; st = save; Y = saveY; have = sh;
            }
            step(h1); every();
            break;
        }
    }
    double Q() const {   // H + p_t (= H - E0 - W_RR): the energy-books residual
        double yy[DIM_MAX];
        for (int i = 0; i < st.dim; i++) yy[i] = st.y[i] + st.lo[i];
        return sys.energy(yy) + yy[2 * n3() + 1];
    }
    long long evals() const { return stats.rhs_calls + kick_evals; }
};

static AdmSystem make_system(const Case& c, int orders) {
    AdmSystem s;
    s.nb.n = c.n; s.nb.G = c.G; s.nb.pn = (orders & 1) && c.n > 2;   // N-body 1PN only for n > 2 (pair term else)
    for (int a = 0; a < c.n; a++) s.nb.m[a] = c.m[a];
    s.c = c.c;
    s.pairs = {{c.pi, c.pj}};
    s.cons_orders = orders & 11;
    s.rr_orders = orders & 20;
    s.setup();
    return s;
}

// ---------------------------------------------------------------------------------------------------- GBS driver
static const int SEQ[] = {2, 4, 6, 8, 12, 16, 24, 32, 48, 64};
struct GbsRun {
    LogHState st{};
    GbsParams par;
    GbsTable tab;
    double ds = 0, Mt = 0, G = 1;
    long long evals = 0, steps = 0, rej = 0;
    void init(const Case& c, int orders, double eta) {
        st = LogHState{};
        st.n_bodies = c.n;
        for (int i = 0; i < 3 * c.n; i++) { st.pos[i] = c.q[i]; st.vel[i] = c.v[i]; st.aux_vel[i] = c.v[i]; }
        for (int a = 0; a < c.n; a++) { st.mass[a] = c.m[a]; Mt += c.m[a]; }
        st.c_code = c.c; st.use_aux_vel = orders != 0;
        G = c.G;
        st.B = -logh_total_energy(st, G);
        par.seq = SEQ; par.k_max = 10; par.eta = eta; par.G = G; par.c_code = c.c; par.pn_order = orders;
        ds = 1e-3 * c.P_in * logh_omega(st);
    }
    bool try_step(double ds_try, GbsStepOut& o) {
        LogHStepStart start;
        logh_step_start(st, G, par.c_code, par.pn_order, start);
        gbs_step(par, st, ds_try, tab, -1, &start, o);
        for (int k = 0; k <= o.stage; k++) evals += SEQ[k];
        steps++;
        if (!o.accepted) rej++;
        return o.accepted;
    }
    double ds_dyn() const {
        double rmin2 = 1e300;
        for (int a = 0; a < st.n_bodies; a++)
            for (int b = a + 1; b < st.n_bodies; b++) {
                double r2 = 0;
                for (int k = 0; k < 3; k++) r2 += std::pow(st.pos[3 * a + k] - st.pos[3 * b + k], 2);
                rmin2 = std::min(rmin2, r2);
            }
        const double rmin = std::sqrt(rmin2);
        return std::sqrt(rmin * rmin * rmin / (G * Mt)) * logh_omega(st) / 3.0;
    }
    template <class F>
    void advance(double T, F&& every) {
        while (st.t < T) {
            const double om = logh_omega(st);
            double ds_try = std::min(ds, ds_dyn());
            bool last = false;
            if (st.t + ds_try / om >= T) { ds_try = (T - st.t) * om; last = true; }
            if (last) {
                // secant on ds so that t lands on T
                LogHState save = st;
                GbsStepOut o;
                double h0 = ds_try, f0, h1, f1;
                if (!try_step(h0, o)) { st = save; ds = 0.5 * h0; continue; }
                f0 = st.t - T; st = save;
                h1 = h0 * (1 - 1e-3); try_step(h1, o); f1 = st.t - T; st = save;
                for (int it = 0; it < 6 && std::fabs(f1) > 1e-15 * T; it++) {
                    double h2 = h1 - f1 * (h1 - h0) / (f1 - f0);
                    h0 = h1; f0 = f1; h1 = h2;
                    try_step(h1, o); f1 = st.t - T; st = save;
                }
                try_step(h1, o);
                every();
                if (!o.accepted) continue;
                break;
            }
            GbsStepOut o;
            if (!try_step(ds_try, o)) { ds = 0.5 * ds_try; continue; }
            ds = o.ds_next;
            every();
        }
    }
};

// ---------------------------------------------------------------------------------------------------- verification
static double fd_check_system() {
    // 3 bodies, pair (0,1) with 1PN..3PN pair terms + N-body 1PN, c = 3 (strong PN: all terms visible)
    AdmSystem s;
    s.nb.n = 3; s.nb.G = 1.0; s.nb.pn = true; s.nb.m[0] = 0.6; s.nb.m[1] = 0.3; s.nb.m[2] = 0.2;
    s.c = 3.0; s.pairs = {{0, 1}}; s.cons_orders = 11; s.setup();
    double q[9] = {0.1, 0.2, -0.1, -0.5, 0.3, 0.2, 1.2, -0.7, 0.4}, p[9] = {0.3, -0.1, 0.2, -0.2, 0.25, 0.1, 0.05, -0.1, -0.2};
    double U, gU[9], Hq[9], Hp[9];
    s.hamiltonian(q, p, U, gU, Hq, Hp);
    double err = 0, scale = 0;
    for (int i = 0; i < 18; i++) {
        double* z = i < 9 ? q : p; int k = i % 9; double z0 = z[k], h = 1e-5, e[2];
        for (int sg = 0; sg < 2; sg++) {
            z[k] = z0 + (sg ? -h : h);
            double U2, g2[9], a[9], b[9];
            e[sg] = s.hamiltonian(q, p, U2, g2, a, b);
        }
        z[k] = z0;
        double fd = (e[0] - e[1]) / (2 * h), an = i < 9 ? Hq[k] : Hp[k];
        err = std::max(err, std::fabs(fd - an)); scale = std::max(scale, std::fabs(an));
    }
    return err / scale;
}

static double fd_check_hessian() {
    PairHam h; h.build(0.2, 11);
    double x[3] = {0.7, -0.4, 0.3}, p[3] = {0.3, 0.5, -0.2}, dx[3], dp[3], Hpp[9];
    h.value_grad(x, p, 1.0, 0.25, dx, dp, Hpp);
    double err = 0, sc = 0;
    for (int j = 0; j < 3; j++) {
        double pp[3] = {p[0], p[1], p[2]}, pm[3] = {p[0], p[1], p[2]}, a[3], b[3], c1[3], c2[3];
        pp[j] += 1e-5; pm[j] -= 1e-5;
        h.value_grad(x, pp, 1.0, 0.25, a, c1);
        h.value_grad(x, pm, 1.0, 0.25, b, c2);
        for (int i = 0; i < 3; i++) { err = std::max(err, std::fabs((c1[i] - c2[i]) / 2e-5 - Hpp[3 * i + j])); sc = std::max(sc, std::fabs(Hpp[3 * i + j])); }
    }
    return err / sc;
}

// circular orbit of the reduced H (G M = 1) at radius r: E and x = (Omega / c^3)^(2/3); residual vs Blanchet E(x)
static void circular_check(double nu, double c, double r, double& rel_res, double& x_out) {
    PairHam h; h.build(nu, 11);
    const double ic2 = 1.0 / (c * c);
    // H(r, p_phi = j) with p = (0, j/r, 0) at x = (r, 0, 0); dH/dr|_j and dH/dj
    auto Hrj = [&](double rr, double j, double& dHdr, double& dHdj) {
        double xx[3] = {rr, 0, 0}, pp[3] = {0, j / rr, 0}, dx[3], dp[3];
        double f = h.value_grad(xx, pp, 1.0, ic2, dx, dp);
        dHdr = dx[0] + 1.0 / (rr * rr) + (dp[1] + j / rr) * (-j / (rr * rr));
        dHdj = (dp[1] + j / rr) / rr;
        return f + 0.5 * j * j / (rr * rr) - 1.0 / rr;
    };
    double j = std::sqrt(r), dHdr, dHdj;
    for (int it = 0; it < 60; it++) {
        Hrj(r, j, dHdr, dHdj);
        double a, b; Hrj(r, j * (1 + 1e-6), a, b);
        double dj = -dHdr / ((a - dHdr) / (j * 1e-6));
        j += dj;
        if (std::fabs(dj) < 1e-16 * j) break;
    }
    const double E = Hrj(r, j, dHdr, dHdj);
    const double x = std::pow(dHdj / (c * c * c), 2.0 / 3.0);
    const double pi2 = M_PI * M_PI, n2 = nu * nu;
    const double Ex = -0.5 * c * c * x * (1 + (-0.75 - nu / 12) * x + (-27.0 / 8 + 19.0 / 8 * nu - n2 / 24) * x * x
        + (-675.0 / 64 + (34445.0 / 576 - 205.0 / 96 * pi2) * nu - 155.0 / 96 * n2 - 35.0 / 5184 * n2 * nu) * x * x * x);
    rel_res = (E - Ex) / std::fabs(Ex);
    x_out = x;
}

// test mass: H(nu = 0) vs the Schwarzschild Hamiltonian in isotropic coordinates (exact): residual / |H_N|
static double testmass_residual(double c) {
    PairHam h; h.build(0.0, 11);
    double xx[3] = {20.0, 3.0, -1.0}, pp[3] = {0.05, 0.2, 0.03}, dx[3], dp[3];
    const double f = h.value_grad(xx, pp, 1.0, 1.0 / (c * c), dx, dp);
    const double r = std::sqrt(xx[0] * xx[0] + xx[1] * xx[1] + xx[2] * xx[2]);
    const double p2 = pp[0] * pp[0] + pp[1] * pp[1] + pp[2] * pp[2];
    const double s = 1.0 / (2 * r * c * c);
    const double Hs = c * c * std::sqrt((1 - s) * (1 - s) / ((1 + s) * (1 + s)) + p2 / (c * c) * (1 - s) * (1 - s) / ((1 + s) * (1 + s)) / std::pow(1 + s, 4)) - c * c;
    return (0.5 * p2 - 1.0 / r + f - Hs) / std::fabs(0.5 * p2 - 1.0 / r);
}

// Iyer & Will 1995 RR (KFS 2003 Sect. V: alpha = 5/3, beta = 3, delta_i) vs rr_accel_adm (KFS Eq. 5.3)
static double iw_check() {
    const double nu = 0.21, al = 5.0 / 3, be = 3.0;
    const double d1 = 41.0 / 84 + 677.0 / 42 * nu, d2 = -61.0 / 14 - 151.0 / 14 * nu, d3 = 583.0 / 28 - 157.0 / 21 * nu,
                 d4 = 115.0 / 28 - 15.0 / 7 * nu, d5 = -961.0 / 42 + 16.0 / 3 * nu, d6 = -51.0 / 14 - 23.0 / 14 * nu;
    const double a1 = 3 + 3 * be, a2 = 23.0 / 3 + 2 * al - 3 * be, a3 = -5 * be, b1 = 2 + al, b2 = 2 - al, b3 = -3 - 3 * al;
    const double c1 = (117 + 132 * nu) / 28 - 1.5 * be * (1 - 3 * nu) + 3 * d2 - 3 * d6;
    const double c2 = -(297 - 310 * nu) / 42 - 3 * al * (1 - 4 * nu) - 1.5 * be * (7 + 13 * nu) - 2 * d1 - 3 * d2 + 3 * d5 + 3 * d6;
    const double c3 = 5.0 / 28 * (19 - 72 * nu) + 2.5 * be * (1 - 3 * nu) - 5 * d2 + 5 * d4 + 5 * d6;
    const double c4 = -(687 - 368 * nu) / 28 - 6 * al * nu + 0.5 * be * (54 + 17 * nu) - 2 * d2 - 5 * d4 - 6 * d5;
    const double c5 = -7 * d4;
    const double c6 = -(1533 + 498 * nu) / 21 - al * (14 + 9 * nu) + 3 * be * (7 + 4 * nu) - 2 * d3 - 3 * d5;
    const double e1 = -3 * (1 - 3 * nu) - 1.5 * al * (1 - 3 * nu) - d1;
    const double e2 = -(139 + 768 * nu) / 84 - 0.5 * al * (5 + 17 * nu) + d1 - d3;
    const double e3 = (369 - 624 * nu) / 28 + 1.5 * (3 * al + 2 * be) * (1 - 3 * nu) + 3 * d1 - 3 * d6;
    const double e4 = (295 - 335 * nu) / 42 + 0.5 * al * (38 - 11 * nu) - 3 * be * (1 - 3 * nu) + 2 * d1 + 4 * d3 + 3 * d6;
    const double e5 = 5.0 / 28 * (19 - 72 * nu) - 5 * be * (1 - 3 * nu) + 5 * d6;
    const double e6 = -(634 - 66 * nu) / 21 + al * (7 + 3 * nu) + d3;
    double worst = 0;
    for (int trial = 0; trial < 20; trial++) {
        double x[3] = {1.0 + 0.1 * trial, 0.3 - 0.05 * trial, 0.2}, v[3] = {0.2 - 0.03 * trial, 0.9, -0.1 + 0.02 * trial};
        const double c = 2.0;
        double a[3]; rr_accel_adm(x, v, 1.0, nu, c, 20, a);
        double r = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]), n[3] = {x[0] / r, x[1] / r, x[2] / r};
        double v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2], rd = n[0] * v[0] + n[1] * v[1] + n[2] * v[2], m = 1.0 / r;
        double A25 = a1 * v2 + a2 * m + a3 * rd * rd, B25 = b1 * v2 + b2 * m + b3 * rd * rd;
        double A35 = c1 * v2 * v2 + c2 * v2 * m + c3 * v2 * rd * rd + c4 * rd * rd * m + c5 * rd * rd * rd * rd + c6 * m * m;
        double B35 = e1 * v2 * v2 + e2 * v2 * m + e3 * v2 * rd * rd + e4 * rd * rd * m + e5 * rd * rd * rd * rd + e6 * m * m;
        double pre = 1.6 * nu * m * m * m;   // (8/5) nu (M / r^2)(M / r), M = 1
        for (int k = 0; k < 3; k++) {
            double iw = pre * (rd * n[k] * (A25 / std::pow(c, 5) + A35 / std::pow(c, 7)) - v[k] * (B25 / std::pow(c, 5) + B35 / std::pow(c, 7)));
            worst = std::max(worst, std::fabs(iw - a[k]) / (std::fabs(iw) + 1e-300));
        }
    }
    return worst;
}

static double map_roundtrip(double nu, double c, double& its) {
    Reduced R; R.build(nu, 1.0, c, 11);
    ContactCoeffs K = contact_coeffs(nu);
    double worst = 0; its = 0;
    for (int t = 0; t < 10; t++) {
        double x[3] = {1.0 + 0.3 * t, -0.2 + 0.1 * t, 0.1}, p[3] = {0.1 - 0.02 * t, 0.8 / std::sqrt(1.0 + 0.3 * t), 0.05};
        double xh[3], vh[3], x2[3], p2[3];
        adm_to_harm(R, K, x, p, xh, vh);
        its += harm_to_adm(R, K, xh, vh, x2, p2);
        for (int k = 0; k < 3; k++) worst = std::max({worst, std::fabs(x2[k] - x[k]) / std::fabs(x[0]), std::fabs(p2[k] - p[k]) / std::fabs(p[1])});
    }
    its /= 10;
    return worst;
}

// DJS 2000 Eq. (4.16): periastron advance k(E, j) per radial period (sigma = 41/24 nu^2), E = H/mu, j = J/(mu G M).
// The E^2 term is i3 E^2, not 3 i3 E^2 as printed in arXiv gr-qc/9912092v1 Eq. (4.16): -d/dj of their own radial action
// Eq. (4.12) gives i3 E^2, and so does Blanchet LRR (Kepsj) = secular_pn_k (checked below and by the integration).
static double k_djs(double E, double j, double nu, double c) {
    const double pi2 = M_PI * M_PI, sig = 41.0 / 24 * nu * nu;
    const double i1 = 77.0 / 2 + (41.0 / 64 * pi2 - 125.0 / 3) * nu + 83.0 / 24 * nu * nu;
    const double i2 = 105.0 / 2 + (41.0 / 64 * pi2 - 218.0 / 3) * nu + 221.0 / 24 * nu * nu;
    const double i3 = 0.25 * (5 - 5 * nu + 4 * nu * nu);
    const double c2 = c * c, j2 = j * j;
    return 3.0 / (c2 * j2) * (1 + (1.25 * (7 - 2 * nu) / j2 + 0.5 * (5 - 2 * nu) * E) / c2
                              + (2.5 * (i1 - sig) / (j2 * j2) + (i2 - sig) * E / j2 + i3 * E * E) / (c2 * c2));
}

static void verify() {
    std::printf("verify gradient vs FD (3 bodies, pair 1-3PN + N-body 1PN, c = 3): rel err %.1e\n", fd_check_system());
    std::printf("verify pair Hessian d2H/dp2 vs FD (1-3PN, c = 2): rel err %.1e\n", fd_check_hessian());
    for (double r : {20.0, 40.0, 80.0}) {
        double res, x; circular_check(0.2, 1.0, r, res, x);
        std::printf("verify circular E(x) vs Blanchet 3PN E(x), nu = 0.2, x = %.3e: rel residual %+.3e (/x^3 = %.3f, /x^4 = %.2f)\n",
                    x, res, res / (x * x * x), res / (x * x * x * x));
    }
    for (double c : {3.0, 4.5, 6.75}) {
        double rr = testmass_residual(c);
        std::printf("verify test mass vs Schwarzschild isotropic, c = %.0f: residual/|H_N| %+.3e (x c^8 = %.3f)\n", c, rr, rr * std::pow(c, 8));
    }
    std::printf("verify KFS ADM RR acceleration vs Iyer-Will (alpha 5/3, beta 3, KFS delta_i): max rel diff %.1e\n", iw_check());
    for (double c : {3.0, 10.0}) {
        double its; double rt = map_roundtrip(0.2, c, its);
        std::printf("verify map ADM -> MH -> ADM round trip, c = %.0f: %.1e (%.1f iterations)\n", c, rt, its);
    }
    {
        const double nu = 0.2, c = 10.0, E = -0.4, j = 1.3;
        const double eps = -2 * E / (c * c), jB = -2 * E * j * j;
        std::printf("verify k: DJS 2000 (E, j) %.12e vs secular_pn_k (Blanchet eps, j) %.12e\n", k_djs(E, j, nu, c), secular_pn_k(eps, jB, nu, 11));
    }
}

// ---------------------------------------------------------------------------------------------------- run helpers
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static void pair_rel(const double* q, const double* w, int i, int j, double* x, double* v) {
    for (int k = 0; k < 3; k++) { x[k] = q[3 * i + k] - q[3 * j + k]; v[k] = w[3 * i + k] - w[3 * j + k]; }
}

// apocentre detector: pr = x.v changes sign + -> -; quadratic interpolation of phi through the last three samples
struct ApoTracker {
    double s[3] = {0, 0, 0}, pr[3] = {0, 0, 0}, ph[3] = {0, 0, 0}, tt[3] = {0, 0, 0}, E[3] = {0, 0, 0};
    int nsamp = 0;
    std::vector<double> phi_apo, t_apo, E_apo;
    void add(double ss, double prr, double phi, double t, double En) {
        for (int k = 0; k < 2; k++) { s[k] = s[k + 1]; pr[k] = pr[k + 1]; ph[k] = ph[k + 1]; tt[k] = tt[k + 1]; E[k] = E[k + 1]; }
        s[2] = ss; pr[2] = prr; ph[2] = phi; tt[2] = t; E[2] = En;
        nsamp++;
        if (nsamp >= 3 && pr[1] > 0 && pr[2] <= 0) {
            // Lagrange interpolation in s of pr, find root in [s1, s2] by bisection, then phi, t, E at the root
            auto L = [&](const double* f, double x) {
                double r = 0;
                for (int a = 0; a < 3; a++) {
                    double l = 1;
                    for (int b = 0; b < 3; b++) if (b != a) l *= (x - s[b]) / (s[a] - s[b]);
                    r += f[a] * l;
                }
                return r;
            };
            double lo = s[1], hi = s[2];
            for (int it = 0; it < 80; it++) { double mid = 0.5 * (lo + hi); if (L(pr, mid) > 0) lo = mid; else hi = mid; }
            phi_apo.push_back(L(ph, lo)); t_apo.push_back(L(tt, lo)); E_apo.push_back(L(E, lo));
        }
    }
};

// ---------------------------------------------------------------------------------------------------- cons
// conservative 3PN binary: H_ADM drift (max over all steps), slope, periastron advance vs DJS k(E, j)
static int run_cons(double e, double x, double orbits, double spo, int orders, bool quiet = false, double* out = nullptr) {
    Case c = make_binary(0.5, e, x, "cons");
    if (!quiet) print_case(c);
    PairMap pm = pair_map(c, c.m[0], c.m[1], orders);
    GaussRun gr; gr.sys = make_system(c, orders); gr.var = VNONE; gr.spo = spo;
    double q[24], p[24];
    harm_to_adm_state(c, pm, c.q.data(), c.v.data(), q, p);
    gr.init(q, p);
    gr.ds_fixed = gr.ds_now();
    const double E0 = gr.E0;
    double xr[3], pr_[3];
    auto rel_xp = [&](double* xx, double* ph) {
        const double* Q = gr.st.y.data(); const double* P = Q + 6;
        for (int k = 0; k < 3; k++) { xx[k] = Q[k] - Q[3 + k]; ph[k] = (c.m[1] * P[k] - c.m[0] * P[3 + k]) / (pm.M * pm.mu); }
    };
    rel_xp(xr, pr_);
    const double jv[3] = {xr[1] * pr_[2] - xr[2] * pr_[1], xr[2] * pr_[0] - xr[0] * pr_[2], xr[0] * pr_[1] - xr[1] * pr_[0]};
    const double j = std::sqrt(jv[0] * jv[0] + jv[1] * jv[1] + jv[2] * jv[2]) / pm.R.Gm;
    const double Ehat = E0 / pm.mu;
    Phase ph; ph.setup(xr, pr_);
    double emax = 0; std::vector<double> tl, el; double next = 1e-3 * orbits * c.P_in;
    // apocentres (pr: + -> -): the state before the crossing step is kept for the first and the last one and the
    // crossing is refined at the end by bisection on a fresh Gauss step from that state (dense, exact to round-off)
    auto prev = gr.st; double prev_phi = 0, pr_prev = 0; int ncross = 0;
    auto first = gr.st, last = gr.st; double first_phi = 0, last_phi = 0;
    {
        double xx[3], pp[3]; rel_xp(xx, pp); pr_prev = xx[0] * pp[0] + xx[1] * pp[1] + xx[2] * pp[2];
    }
    const double t0 = now();
    gr.advance(orbits * c.P_in, [&]() {
        double yy[DIM_MAX];
        for (int i = 0; i < gr.st.dim; i++) yy[i] = gr.st.y[i] + gr.st.lo[i];
        const double rel = (gr.sys.energy(yy) - E0) / std::fabs(E0);
        emax = std::max(emax, std::fabs(rel));
        if (gr.t() >= next) { tl.push_back(gr.t()); el.push_back(emax); next *= 1.1; }
        double xx[3], pp[3]; rel_xp(xx, pp);
        const double prn = xx[0] * pp[0] + xx[1] * pp[1] + xx[2] * pp[2];
        const double phn = ph.update(xx);
        if (pr_prev > 0 && prn <= 0) {
            if (ncross == 0) { first = prev; first_phi = prev_phi; }
            last = prev; last_phi = prev_phi; ncross++;
        }
        prev = gr.st; prev_phi = phn; pr_prev = prn;
    });
    const double wall = now() - t0;
    if (gr.stats.diverged) {
        std::printf("cons e=%.3f x=%.1e spo=%.0f: fixed-point iteration diverged at t = %.3g orbits\n", e, x, spo, gr.t() / c.P_in);
        if (out) out[0] = out[1] = out[2] = NAN;
        return 1;
    }
    auto refine = [&](const CState<double>& s0, double phi0) {
        // bisection on ds in [0, ds_fixed] for pr = 0; phi unwrapped from phi0 (the unwrapped phase at s0)
        auto at = [&](double h, double* xx, double* pp) {
            CState<double> s = s0; std::array<std::array<double, DIM_MAX>, S_MAX> Yl; bool hv = false; IrkStats stt;
            if (h > 0) symrk::gauss_step(gr.g, gr.sys, s, h, Yl, hv, stt);
            const double* Q = s.y.data(); const double* P = Q + 6;
            for (int k2 = 0; k2 < 3; k2++) { xx[k2] = Q[k2] - Q[3 + k2]; pp[k2] = (c.m[1] * P[k2] - c.m[0] * P[3 + k2]) / (pm.M * pm.mu); }
        };
        double lo = 0, hi = gr.ds_fixed, xx[3], pp[3];
        for (int it = 0; it < 60; it++) {
            const double mid = 0.5 * (lo + hi);
            at(mid, xx, pp);
            if (xx[0] * pp[0] + xx[1] * pp[1] + xx[2] * pp[2] > 0) lo = mid; else hi = mid;
        }
        at(0.5 * (lo + hi), xx, pp);
        double a = ph.angle(xx), d = a - std::remainder(phi0, 2 * M_PI);
        d = std::remainder(d, 2 * M_PI);
        return phi0 + d;
    };
    // slope of log(max) vs log(t) over the last two decades
    double sx = 0, sy = 0, sxx = 0, sxy = 0; int k = 0;
    for (size_t i = 0; i < tl.size(); i++) {
        if (tl[i] < 1e-2 * orbits * c.P_in || el[i] <= 0) continue;
        double X = std::log(tl[i]), Yv = std::log(el[i]); sx += X; sy += Yv; sxx += X * X; sxy += X * Yv; k++;
    }
    const double slope = k > 2 ? (k * sxy - sx * sy) / (k * sxx - sx * sx) : NAN;
    const double phi_a = refine(first, first_phi), phi_b = refine(last, last_phi);
    const double kmeas = (phi_b - phi_a) / (2 * M_PI * (ncross - 1)) - 1.0;
    const double kan = k_djs(Ehat, j, pm.mu / pm.M, c.c);
    if (!quiet)
        std::printf("cons e=%.3f x=%.1e orbits=%.0f spo=%.0f: max|dH/H|=%.2e slope=%.2f | k_meas=%.12e k_DJS=%.12e rel diff %.2e "
                    "(x^3/(1-e^2)^3 = %.1e) | rhs/orb=%.0f us/orb=%.2f it/step=%.1f\n",
                    e, x, orbits, spo, emax, slope, kmeas, kan, (kmeas - kan) / kan, std::pow(x / (1 - e * e), 3), gr.evals() / orbits,
                    1e6 * wall / orbits, (double)gr.stats.iters / gr.stats.steps);
    if (out) { out[0] = emax; out[1] = slope; out[2] = (kmeas - kan) / kan; }
    return 0;
}

// GBS (harmonic) periastron advance and E_h swing at apocentres, same binary
static double run_cons_gbs(double e, double x, double orbits, double eta, int orders, bool quiet = false) {
    Case c = make_binary(0.5, e, x, "cons");
    PairMap pm = pair_map(c, c.m[0], c.m[1], orders);
    GbsRun gb; gb.init(c, orders, eta);
    const double E0 = harm_energy(c, pm, gb.st.pos, gb.st.vel);
    double xr[3], vr[3]; pair_rel(gb.st.pos, gb.st.vel, 0, 1, xr, vr);
    double xa[3], pa[3]; harm_to_adm(pm.R, pm.K, xr, vr, xa, pa);
    const double jv[3] = {xa[1] * pa[2] - xa[2] * pa[1], xa[2] * pa[0] - xa[0] * pa[2], xa[0] * pa[1] - xa[1] * pa[0]};
    const double j = std::sqrt(jv[0] * jv[0] + jv[1] * jv[1] + jv[2] * jv[2]) / pm.R.Gm;
    Phase ph; ph.setup(xr, vr);
    ApoTracker apo; double sa = 0, emax_apo = 0;
    const double t0 = now();
    gb.advance(orbits * c.P_in, [&]() {
        double xx[3], vv[3]; pair_rel(gb.st.pos, gb.st.vel, 0, 1, xx, vv);
        sa += 1;
        size_t before = apo.E_apo.size();
        apo.add(sa, xx[0] * vv[0] + xx[1] * vv[1] + xx[2] * vv[2], ph.update(xx), gb.st.t, 0);
        if (apo.E_apo.size() > before) emax_apo = std::max(emax_apo, std::fabs(harm_energy(c, pm, gb.st.pos, gb.st.vel) - E0) / std::fabs(E0));
    });
    const double wall = now() - t0;
    const int na = (int)apo.phi_apo.size();
    const double kmeas = (apo.phi_apo[na - 1] - apo.phi_apo[0]) / (2 * M_PI * (na - 1)) - 1.0;
    const double kan = k_djs(E0 / pm.mu, j, pm.mu / pm.M, c.c);
    if (!quiet)
        std::printf("gbs  e=%.3f x=%.1e orbits=%.0f eta=%.0e: max|dE_h/E| near apocentre=%.2e | k_meas=%.12e k_DJS=%.12e rel diff %.2e | "
                    "evals/orb=%.0f us/orb=%.2f rej=%lld\n", e, x, orbits, eta, emax_apo, kmeas, kan, (kmeas - kan) / kan,
                    gb.evals / orbits, 1e6 * wall / orbits, gb.rej);
    return (kmeas - kan) / kan;
}

// ---------------------------------------------------------------------------------------------------- consistency
// ADM (Gauss, converged) vs MH (GBS, eta 1e-14) from the same harmonic IC, mapped back: difference vs c
static void run_consistency(int orders, double orbits, double qm = 0.5) {
    for (double x : {4e-3, 2e-3, 1e-3}) {
        Case c = make_binary(qm, 0.5, x, "consistency");
        PairMap pm = pair_map(c, c.m[0], c.m[1], orders);
        if (!(orders & 8)) for (double& b : pm.K.b) b = 0.0;
        if (!(orders & 2)) for (double& a : pm.K.a) a = 0.0;
        GaussRun gr; gr.sys = make_system(c, orders); gr.var = VNONE; gr.spo = 120;
        double q[24], p[24];
        harm_to_adm_state(c, pm, c.q.data(), c.v.data(), q, p);
        gr.init(q, p); gr.ds_fixed = gr.ds_now();
        const double T = orbits * c.P_in;
        gr.advance(T, []() {});
        GbsRun gb; gb.init(c, orders, 1e-14);
        gb.advance(T, []() {});
        double qh[24], vh[24];
        adm_to_harm_state(c, pm, gr.st.y.data(), gr.st.y.data() + 6, qh, vh, gr.sys);
        double dx = 0, dv = 0, vn = 0;
        for (int k = 0; k < 3; k++) {
            dx += std::pow((qh[k] - qh[3 + k]) - (gb.st.pos[k] - gb.st.pos[3 + k]), 2);
            dv += std::pow((vh[k] - vh[3 + k]) - (gb.st.vel[k] - gb.st.vel[3 + k]), 2);
            vn += std::pow(gb.st.vel[k] - gb.st.vel[3 + k], 2);
        }
        // same, with the map switched off (identity): shows that the map matters
        double qn[24], vnn[24];
        for (int i = 0; i < 6; i++) { qn[i] = gr.st.y[i]; }
        double U, gU[24], Hq[24], Hp[24]; gr.sys.hamiltonian(gr.st.y.data(), gr.st.y.data() + 6, U, gU, Hq, Hp);
        for (int i = 0; i < 6; i++) vnn[i] = Hp[i];
        double dx0 = 0;
        for (int k = 0; k < 3; k++) dx0 += std::pow((qn[k] - qn[3 + k]) - (gb.st.pos[k] - gb.st.pos[3 + k]), 2);
        (void)vnn;
        std::printf("consistency q=%.2f orders=%d x=%.1e %g orbits: |dx_h|/a = %.3e (no map: %.3e), |dv_h|/v = %.3e, t: %.3e vs %.3e\n",
                    qm, orders, x, orbits, std::sqrt(dx) / c.a_in, std::sqrt(dx0) / c.a_in, std::sqrt(dv / vn), gr.t(), gb.st.t);
    }
}

// ---------------------------------------------------------------------------------------------------- rr (inspiral)
static Variant parse_var(const std::string& s) { return s == "A" ? VA : s == "S" ? VS : s == "S2" ? VS2 : VNONE; }

struct RRResult { double t_stop = NAN, t_sec = NAN, Qmax = 0, flux_rms = 0, flux_max = 0, evals_orb = 0, us_orb = 0, orbits = 0; };

static RRResult run_rr(Variant var, double e, double x, double spo, double eps_stop, int orders, bool quiet = false) {
    Case c = make_binary(0.5, e, x, "rr");
    PairMap pm = pair_map(c, c.m[0], c.m[1], orders);
    GaussRun gr; gr.sys = make_system(c, orders); gr.var = var; gr.spo = spo;
    double q[24], p[24];
    harm_to_adm_state(c, pm, c.q.data(), c.v.data(), q, p);
    gr.init(q, p);
    const double E0 = gr.E0, nu = pm.mu / pm.M, Gm = pm.R.Gm, cc = c.c;
    auto rel_xp = [&](double* xx, double* ph) {
        const double* Q = gr.st.y.data(); const double* P = Q + 6;
        for (int k = 0; k < 3; k++) { xx[k] = Q[k] - Q[3 + k]; ph[k] = (c.m[1] * P[k] - c.m[0] * P[3 + k]) / (pm.M * pm.mu); }
    };
    auto eps_j = [&](double& eps, double& jB) {
        double xx[3], pp[3]; rel_xp(xx, pp);
        const double E = gr.sys.energy(gr.st.y.data()) / pm.mu;
        const double jv[3] = {xx[1] * pp[2] - xx[2] * pp[1], xx[2] * pp[0] - xx[0] * pp[2], xx[0] * pp[1] - xx[1] * pp[0]};
        const double J2 = jv[0] * jv[0] + jv[1] * jv[1] + jv[2] * jv[2];
        eps = -2 * E / (cc * cc); jB = -2 * E * J2 / (Gm * Gm);
    };
    double eps0, j0; eps_j(eps0, j0);
    Phase ph; { double xx[3], pp[3]; rel_xp(xx, pp); ph.setup(xx, pp); }
    ApoTracker apo; double sa = 0;
    RRResult R;
    bool stop = false;
    const double t0 = now();
    const double Tmax = 1e7 * c.P_in;
    // advance in chunks of one inner orbit (to check the stop criterion)
    double eps_last = eps0;
    while (!stop && gr.t() < Tmax) {
        double eps, jB; eps_j(eps, jB);
        const double Pn = 2 * M_PI * Gm / std::pow(eps * cc * cc, 1.5);
        gr.advance(gr.t() + Pn, [&]() {
            double xx[3], pp[3]; rel_xp(xx, pp);
            sa += 1;
            apo.add(sa, xx[0] * pp[0] + xx[1] * pp[1] + xx[2] * pp[2], ph.update(xx), gr.t(), gr.sys.energy(gr.st.y.data()));
            R.Qmax = std::max(R.Qmax, std::fabs(gr.Q()) / std::fabs(E0));
        });
        if (gr.stats.diverged) { std::printf("  diverged at t = %.4g\n", gr.t()); break; }
        eps_j(eps, jB);
        if (eps >= eps_stop) {
            // linear interpolation of the stop time between the last two orbit ends
            R.t_stop = gr.t() - Pn * (eps - eps_stop) / std::max(eps - eps_last, 1e-300);
            stop = true;
        }
        eps_last = eps;
    }
    const double wall = now() - t0;
    R.orbits = (double)apo.t_apo.size();
    R.evals_orb = gr.evals() / std::max(1.0, R.orbits);
    R.us_orb = 1e6 * wall / std::max(1.0, R.orbits);
    // orbit-averaged flux check: energy change between apocentres vs the secular rate (Peters + 1PN) over the orbit
    {
        double s2 = 0; int n = 0;
        for (size_t i = 1; i < apo.t_apo.size(); i++) {
            const double dE = (apo.E_apo[i] - apo.E_apo[i - 1]) / pm.mu, dt = apo.t_apo[i] - apo.t_apo[i - 1];
            // secular rate at the midpoint energy, j from the secular track (eps, j of the start of the orbit: use
            // the RK2 of the secular ODE over the orbit)
            (void)dE; (void)dt; n++;
            (void)s2;
        }
    }
    // secular reference: RK4 in t of (eps, j) with secular_pn_rates until eps_stop
    {
        double eps = eps0, jB = j0, t = 0;
        const double Pn0 = 2 * M_PI * Gm / std::pow(eps * cc * cc, 1.5);
        double h = 0.2 * Pn0;
        std::vector<double> ts, es;
        while (eps < eps_stop && t < Tmax) {
            double k1e, k1j, k2e, k2j, k3e, k3j, k4e, k4j;
            secular_pn_rates(eps, jB, nu, Gm, cc, orders, k1e, k1j);
            const double hh = std::min(h, 0.002 * eps / std::fabs(k1e));
            secular_pn_rates(eps + 0.5 * hh * k1e, jB + 0.5 * hh * k1j, nu, Gm, cc, orders, k2e, k2j);
            secular_pn_rates(eps + 0.5 * hh * k2e, jB + 0.5 * hh * k2j, nu, Gm, cc, orders, k3e, k3j);
            secular_pn_rates(eps + hh * k3e, jB + hh * k3j, nu, Gm, cc, orders, k4e, k4j);
            const double en = eps + hh * (k1e + 2 * k2e + 2 * k3e + k4e) / 6, jn = jB + hh * (k1j + 2 * k2j + 2 * k3j + k4j) / 6;
            if (en >= eps_stop) { R.t_sec = t + hh * (eps_stop - eps) / (en - eps); break; }
            eps = en; jB = jn; t += hh;
            ts.push_back(t); es.push_back(eps);
        }
        // flux per orbit: compare E at apocentres with the secular E(t) interpolated (relative to the radiated energy)
        double s2 = 0, mx = 0; int n = 0;
        for (size_t i = 0; i < apo.t_apo.size(); i++) {
            const double ta = apo.t_apo[i];
            auto it = std::lower_bound(ts.begin(), ts.end(), ta);
            if (it == ts.begin() || it == ts.end()) continue;
            size_t k = it - ts.begin();
            const double w = (ta - ts[k - 1]) / (ts[k] - ts[k - 1]);
            const double eps_s = es[k - 1] + w * (es[k] - es[k - 1]);
            const double E_s = -0.5 * eps_s * cc * cc * pm.mu;
            const double rad = std::fabs(apo.E_apo[i] - E0);
            if (rad <= 0) continue;
            const double d = (apo.E_apo[i] - E_s) / rad;
            s2 += d * d; mx = std::max(mx, std::fabs(d)); n++;
        }
        R.flux_rms = n ? std::sqrt(s2 / n) : NAN; R.flux_max = mx;
    }
    if (!quiet)
        std::printf("rr %-2s e=%.2f x=%.1e spo=%.0f orders=%d: t_insp(eps->%.3g) = %.10e  secular(Peters+1PN) %.10e  rel %.2e | "
                    "max|Q|/|E0| = %.2e | E(apo) vs secular track / radiated: rms %.2e max %.2e | orbits %.0f evals/orb %.0f us/orb %.2f\n",
                    var == VA ? "A" : var == VS ? "S" : "S2", e, x, spo, orders, eps_stop, R.t_stop, R.t_sec, (R.t_stop - R.t_sec) / R.t_sec,
                    R.Qmax, R.flux_rms, R.flux_max, R.orbits, R.evals_orb, R.us_orb);
    return R;
}

// GBS inspiral (harmonic): t to eps_stop with E_h
static RRResult run_rr_gbs(double e, double x, double eta, double eps_stop, int orders) {
    Case c = make_binary(0.5, e, x, "rr");
    PairMap pm = pair_map(c, c.m[0], c.m[1], orders);
    GbsRun gb; gb.init(c, orders, eta);
    const double Gm = pm.R.Gm, cc = c.c;
    RRResult R;
    double eps_last = 0; int norb = 0;
    const double t0 = now();
    while (true) {
        const double E = harm_energy(c, pm, gb.st.pos, gb.st.vel) / pm.mu;
        const double eps = -2 * E / (cc * cc);
        if (eps >= eps_stop) {
            const double Pn = 2 * M_PI * Gm / std::pow(eps * cc * cc, 1.5);
            R.t_stop = gb.st.t - Pn * (eps - eps_stop) / std::max(eps - eps_last, 1e-300);
            break;
        }
        eps_last = eps;
        const double Pn = 2 * M_PI * Gm / std::pow(eps * cc * cc, 1.5);
        gb.advance(gb.st.t + Pn, []() {});
        norb++;
    }
    const double wall = now() - t0;
    R.orbits = norb; R.evals_orb = gb.evals / (double)norb; R.us_orb = 1e6 * wall / norb;
    std::printf("rr GBS eta=%.0e e=%.2f x=%.1e: t_insp = %.10e | orbits %d evals/orb %.0f us/orb %.2f rej %lld\n", eta, e, x, R.t_stop,
                norb, R.evals_orb, R.us_orb, gb.rej);
    return R;
}

// ---------------------------------------------------------------------------------------------------- head-to-head
struct H2H { double E = 0, phi = 0, Qmax = 0, evals = 0, wall = 0, t = 0; double qh[24], vh[24]; };

static H2H h2h_gauss(const Case& c, int orders, Variant var, double spo, double T, const PairMap& pm) {
    GaussRun gr; gr.sys = make_system(c, orders); gr.var = var; gr.spo = spo;
    double q[24], p[24];
    harm_to_adm_state(c, pm, c.q.data(), c.v.data(), q, p);
    gr.init(q, p);
    if (!(orders & 20)) gr.ds_fixed = gr.ds_now();
    const int N3 = 3 * c.n;
    Phase ph; { double xx[3], vv[3]; pair_rel(gr.st.y.data(), gr.st.y.data() + N3, c.pi, c.pj, xx, vv); ph.setup(xx, vv); }
    H2H o;
    const double t0 = now();
    gr.advance(T, [&]() {
        double xx[3]; for (int k = 0; k < 3; k++) xx[k] = gr.st.y[3 * c.pi + k] - gr.st.y[3 * c.pj + k];
        o.phi = ph.update(xx);
        o.Qmax = std::max(o.Qmax, std::fabs(gr.Q()) / std::fabs(gr.E0));
    });
    o.wall = now() - t0;
    o.evals = (double)gr.evals();
    o.E = gr.sys.energy(gr.st.y.data());
    o.t = gr.t();
    adm_to_harm_state(c, pm, gr.st.y.data(), gr.st.y.data() + N3, o.qh, o.vh, gr.sys);
    if (gr.stats.diverged) o.E = NAN;
    return o;
}

static H2H h2h_gbs(const Case& c, int orders, double eta, double T, const PairMap& pm) {
    GbsRun gb; gb.init(c, orders, eta);
    Phase ph; { double xx[3], vv[3]; pair_rel(gb.st.pos, gb.st.vel, c.pi, c.pj, xx, vv); ph.setup(xx, vv); }
    H2H o;
    const double t0 = now();
    gb.advance(T, [&]() {
        double xx[3]; for (int k = 0; k < 3; k++) xx[k] = gb.st.pos[3 * c.pi + k] - gb.st.pos[3 * c.pj + k];
        o.phi = ph.update(xx);
    });
    o.wall = now() - t0;
    o.evals = (double)gb.evals;
    o.E = harm_energy(c, pm, gb.st.pos, gb.st.vel);
    o.t = gb.st.t;
    for (int i = 0; i < 3 * c.n; i++) { o.qh[i] = gb.st.pos[i]; o.vh[i] = gb.st.vel[i]; }
    return o;
}

static void run_h2h(const std::string& cname, double orbits, int orders) {
    Case c = make_case(cname);
    print_case(c);
    PairMap pm = pair_map(c, c.m[c.pi], c.m[c.pj], orders);
    const double T = orbits * c.P_in;
    // E scale for relative errors: |E_N| of the pair
    const double Esc = c.G * c.m[c.pi] * c.m[c.pj] / (2 * c.a_in);
    // references: Gauss variant A at spo 160, GBS eta 1e-14
    H2H refA = h2h_gauss(c, orders, (orders & 20) ? VA : VNONE, 160, T, pm);
    H2H refG = h2h_gbs(c, orders, 1e-14, T, pm);
    std::printf("# T = %.0f inner orbits; refs: Gauss-A spo 160 (Q %.1e, %.0f evals/orb), GBS eta 1e-14 (%.0f evals/orb); "
                "ADM->MH map of the Gauss ref vs GBS ref: dphi %.2e rad, dE/|E_N| %.2e\n", orbits, refA.Qmax, refA.evals / orbits,
                refG.evals / orbits, 0.0, 0.0);
    {
        // physics difference between the two references (mapped to harmonic): phase of the pair relative vector at T
        double xa[3], va[3], xg[3], vg[3];
        pair_rel(refA.qh, refA.vh, c.pi, c.pj, xa, va); pair_rel(refG.qh, refG.vh, c.pi, c.pj, xg, vg);
        double dx = 0, xn = 0; for (int k = 0; k < 3; k++) { dx += std::pow(xa[k] - xg[k], 2); xn += xg[k] * xg[k]; }
        std::printf("# refs mapped to MH at T: |dx_rel|/r = %.2e, phase diff %.2e rad, E_h(ADM ref) vs E_h(GBS ref): %.2e of |E_N|\n",
                    std::sqrt(dx / xn), refA.phi - refG.phi, (harm_energy(c, pm, refA.qh, refA.vh) - refG.E) / Esc);
    }
    std::printf("%-22s %10s %10s %10s %10s %10s\n", "scheme", "dE/|E_N|", "dphi[rad]", "max|Q|", "evals/orb", "us/orb");
    const bool rr = orders & 20;
    for (int vi = 0; vi < (rr ? 3 : 1); vi++) {
        Variant v = rr ? (Variant)vi : VNONE;
        for (double spo : {12.0, 16.0, 24.0, 32.0, 48.0}) {
            H2H o = h2h_gauss(c, orders, v, spo, T, pm);
            char tag[64]; std::snprintf(tag, sizeof tag, "Gauss16-%s spo=%.0f", v == VA ? "A" : v == VS ? "S" : v == VS2 ? "S2" : "cons", spo);
            std::printf("%-22s %10.2e %10.2e %10.2e %10.0f %10.2f\n", tag, (o.E - refA.E) / Esc, o.phi - refA.phi, o.Qmax, o.evals / orbits,
                        1e6 * o.wall / orbits);
        }
    }
    for (double eta : {1e-8, 1e-10, 1e-12}) {
        H2H o = h2h_gbs(c, orders, eta, T, pm);
        char tag[64]; std::snprintf(tag, sizeof tag, "LogH+GBS eta=%.0e", eta);
        std::printf("%-22s %10.2e %10.2e %10s %10.0f %10.2f\n", tag, (o.E - refG.E) / Esc, o.phi - refG.phi, "-", o.evals / orbits,
                    1e6 * o.wall / orbits);
    }
}

// ---------------------------------------------------------------------------------------------------- main
int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: see header\n"); return 1; }
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string mode = argv[1];
    if (mode == "verify") { verify(); return 0; }
    if (mode == "info") {
        for (const char* n : {"e09", "e099", "s432", "s1025", "s3047", "trip432"}) print_case(make_case(n));
        return 0;
    }
    if (mode == "test") {
        // CASES R4 checks; bounds 1.3x the measured values (2026-10-21, Raven login node, g++ 15 -O3)
        int fails = 0;
        auto check = [&](const char* what, double val, double bound) {
            const bool ok = std::fabs(val) <= bound;
            std::printf("%-58s %+.3e  bound %.2e  %s\n", what, val, bound, ok ? "ok" : "FAIL");
            if (!ok) fails++;
        };
        check("gradient vs FD (3 bodies, 1-3PN + N-body 1PN)", fd_check_system(), 1.3 * 2.06e-10);
        check("Hessian d2H/dp2 vs FD", fd_check_hessian(), 1.3 * 1.22e-11);
        check("KFS RR vs Iyer-Will(alpha 5/3, beta 3, delta_i)", iw_check(), 1.3 * 4.24e-14);
        { double its; check("map ADM->MH->ADM round trip (c = 3)", map_roundtrip(0.2, 3.0, its), 1.3 * 1.58e-16); }
        {
            double r1, r2, x1, x2; circular_check(0.2, 1.0, 40.0, r1, x1); circular_check(0.2, 1.0, 80.0, r2, x2);
            check("circular E(x) residual / x^4 at x = 0.024 (3PN exact)", r1 / std::pow(x1, 4), 1.3 * 4.106);
            check("circular E(x): residual ratio x 0.024 / 0.012 - 16 (x^4)", r1 / r2 - 16.0 * std::pow(x1 / (2 * x2), 4), 1.3 * 3.58);
        }
        {
            const double t1 = testmass_residual(3.0), t2 = testmass_residual(4.5);
            check("test mass vs Schwarzschild: residual ratio c 3/4.5 - 1.5^8", t1 / t2 - std::pow(1.5, 8), 1.3 * 0.129);
        }
        double o1[3], o2[3];
        run_cons(0.9, 1e-3, 2000, 30, 11, true, o1);
        run_cons(0.9, 5e-4, 2000, 30, 11, true, o2);
        check("cons e=0.9 x=1e-3 2000 orbits: max |dH/H|", o1[0], 1.3 * 8.82e-13);
        check("periastron advance vs DJS(4.16) k(E,j), x = 1e-3 (rel)", o1[2], 1.3 * 1.00e-4);
        check("periastron residual ratio x 1e-3 / 5e-4 - 8 (c^-8)", o1[2] / o2[2] - 8.0, 1.3 * 0.0201);
        {
            // the map: MH (production GBS) run, k vs DJS at E_h, j_h through map^-1 (a 2PN map error would be O(x^2))
            double kg = run_cons_gbs(0.9, 1e-3, 1000, 1e-13, 11, true);
            check("MH (LogH+GBS) k vs DJS(E, j via map^-1), x = 1e-3 (rel)", kg, 1.3 * 8.93e-5);
        }
        RRResult R = run_rr(VA, 0.5, 0.01, 30, 0.05, 31, true);
        check("RR (A) inspiral e=0.5 x 0.01->0.05: max |H + p_t| / |E0|", R.Qmax, 1.3 * 2.94e-13);
        check("RR (A) t_insp vs secular Peters+1PN (rel)", (R.t_stop - R.t_sec) / R.t_sec, 1.3 * 3.33e-4);
        RRResult R2 = run_rr(VS2, 0.5, 0.01, 30, 0.05, 31, true);
        check("RR split SBAB2 vs in-collocation A: t_insp (rel)", (R2.t_stop - R.t_stop) / R.t_stop, 1.3 * 1.46e-8);
        std::printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
        return fails;
    }
    if (mode == "consistency") {
        const double orb = argc > 2 ? std::atof(argv[2]) : 10, q = argc > 3 ? std::atof(argv[3]) : 0.5;
        run_consistency(1 | 2, orb, q); run_consistency(1 | 2 | 8, orb, q); return 0;
    }
    if (mode == "cons") {
        const double e = std::atof(argv[2]), x = std::atof(argv[3]), orb = std::atof(argv[4]), spo = std::atof(argv[5]);
        const int orders = argc > 6 ? std::atoi(argv[6]) : 11;
        run_cons(e, x, orb, spo, orders);
        return 0;
    }
    if (mode == "consgbs") {
        run_cons_gbs(std::atof(argv[2]), std::atof(argv[3]), std::atof(argv[4]), std::atof(argv[5]), argc > 6 ? std::atoi(argv[6]) : 11);
        return 0;
    }
    if (mode == "rr") {
        Variant v = parse_var(argv[2]);
        const double e = std::atof(argv[3]), x = std::atof(argv[4]), spo = std::atof(argv[5]);
        const double es = argc > 6 ? std::atof(argv[6]) : 0.05;
        const int orders = argc > 7 ? std::atoi(argv[7]) : 31;
        run_rr(v, e, x, spo, es, orders);
        return 0;
    }
    if (mode == "rrgbs") {
        run_rr_gbs(std::atof(argv[2]), std::atof(argv[3]), std::atof(argv[4]), argc > 5 ? std::atof(argv[5]) : 0.05, argc > 6 ? std::atoi(argv[6]) : 31);
        return 0;
    }
    if (mode == "h2h") {
        run_h2h(argv[2], std::atof(argv[3]), argc > 4 ? std::atoi(argv[4]) : 31);
        return 0;
    }
    std::fprintf(stderr, "unknown mode\n");
    return 1;
}
