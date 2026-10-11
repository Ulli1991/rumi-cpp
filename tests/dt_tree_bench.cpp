// Tree vs O(N^2) time-step and neighbour search (gravity/dt_tree.h) on a Plummer cluster with Kroupa masses and
// primordial binaries: checks that the tree results are bitwise identical and times both.
//   dt-tree-bench [N ...]     env: BENCH_BIN (binary fraction of particles, default 0.1), BENCH_CM=1 (pair-step velocity
//   = binary c.o.m. velocity, FLYBY_CM_VELOCITY), BENCH_VC (V_c cap factor, default 0.05), BENCH_REPS, BENCH_RH (pc)
#include "dt_tree.h"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>

using Exec = Kokkos::DefaultExecutionSpace;
using HostExec = Kokkos::DefaultHostExecutionSpace;

static double env_d(const char* k, double d) { return std::getenv(k) ? std::atof(std::getenv(k)) : d; }

struct Cluster {
    std::vector<double> x, y, z, vx, vy, vz, cvx, cvy, cvz, m;
    std::vector<int> sid;
};

static Cluster make_cluster(int N, double fbin, double rh, bool cm, unsigned seed) {
    Cluster c;
    for (auto* v : {&c.x, &c.y, &c.z, &c.vx, &c.vy, &c.vz, &c.cvx, &c.cvy, &c.cvz, &c.m}) v->resize(N);
    c.sid.assign(N, -1);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    auto kroupa = [&]() {   // 0.08-0.5: alpha 1.3, 0.5-150: alpha 2.3 (continuous at 0.5)
        const double a1 = 1.3, a2 = 2.3, m0 = 0.08, m1 = 0.5, m2 = 150.0;
        const double i1 = (std::pow(m1, 1 - a1) - std::pow(m0, 1 - a1)) / (1 - a1);
        const double i2 = std::pow(m1, a2 - a1) * (std::pow(m2, 1 - a2) - std::pow(m1, 1 - a2)) / (1 - a2);
        double u = U(rng) * (i1 + i2);
        if (u < i1) return std::pow(std::pow(m0, 1 - a1) + u * (1 - a1), 1 / (1 - a1));
        u -= i1;
        return std::pow(std::pow(m1, 1 - a2) + u * (1 - a2) / std::pow(m1, a2 - a1), 1 / (1 - a2));
    };
    double M = 0;
    for (int i = 0; i < N; i++) { c.m[i] = kroupa(); M += c.m[i]; }
    const double a = rh / 1.305, G = G_CODE;
    auto iso = [&](double r, double& X, double& Y, double& Z) {
        double ct = 2 * U(rng) - 1, ph = 2 * M_PI * U(rng), st = std::sqrt(1 - ct * ct);
        X = r * st * std::cos(ph); Y = r * st * std::sin(ph); Z = r * ct;
    };
    for (int i = 0; i < N; i++) {
        double r = a / std::sqrt(std::pow(U(rng) * 0.999 + 1e-9, -2.0 / 3.0) - 1.0);
        iso(r, c.x[i], c.y[i], c.z[i]);
        double q, g;
        do { q = U(rng); g = 0.1 * U(rng); } while (g > q * q * std::pow(1 - q * q, 3.5));
        double v = q * std::sqrt(2.0) * std::pow(1 + r * r / (a * a), -0.25) * std::sqrt(G * M / a);
        iso(v, c.vx[i], c.vy[i], c.vz[i]);
    }
    // binaries: consecutive particles (2k, 2k+1) for the first fbin N, a log-uniform in 0.1-100 AU, circular
    const int nb = (int)(fbin * N) / 2;
    for (int k = 0; k < nb; k++) {
        int i = 2 * k, j = 2 * k + 1;
        double ab = std::exp(std::log(0.1) + U(rng) * std::log(1000.0)) * AU_TO_PC;
        double mt = c.m[i] + c.m[j];
        double ex, ey, ez, px, py, pz;
        iso(1.0, ex, ey, ez);
        iso(1.0, px, py, pz);   // perpendicular direction: p - (p.e) e
        double pe = px * ex + py * ey + pz * ez;
        px -= pe * ex; py -= pe * ey; pz -= pe * ez;
        double pn = std::sqrt(px * px + py * py + pz * pz);
        px /= pn; py /= pn; pz /= pn;
        double vo = std::sqrt(G * mt / ab);
        double cx = c.x[i], cy = c.y[i], cz = c.z[i], cvx = c.vx[i], cvy = c.vy[i], cvz = c.vz[i];
        double fi = c.m[j] / mt, fj = c.m[i] / mt;
        c.x[i] = cx + fi * ab * ex; c.y[i] = cy + fi * ab * ey; c.z[i] = cz + fi * ab * ez;
        c.x[j] = cx - fj * ab * ex; c.y[j] = cy - fj * ab * ey; c.z[j] = cz - fj * ab * ez;
        c.vx[i] = cvx + fi * vo * px; c.vy[i] = cvy + fi * vo * py; c.vz[i] = cvz + fi * vo * pz;
        c.vx[j] = cvx - fj * vo * px; c.vy[j] = cvy - fj * vo * py; c.vz[j] = cvz - fj * vo * pz;
        c.sid[i] = c.sid[j] = k;
        if (cm) { c.cvx[i] = c.cvx[j] = cvx; c.cvy[i] = c.cvy[j] = cvy; c.cvz[i] = c.cvz[j] = cvz; }
    }
    for (int i = 0; i < N; i++) {
        if (!(cm && c.sid[i] >= 0)) { c.cvx[i] = c.vx[i]; c.cvy[i] = c.vy[i]; c.cvz[i] = c.vz[i]; }
    }
    // shuffle so binaries are not contiguous in the arrays
    std::vector<int> perm(N);
    for (int i = 0; i < N; i++) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    Cluster s = c;
    for (int i = 0; i < N; i++) {
        int k = perm[i];
        s.x[i] = c.x[k]; s.y[i] = c.y[k]; s.z[i] = c.z[k]; s.vx[i] = c.vx[k]; s.vy[i] = c.vy[k]; s.vz[i] = c.vz[k];
        s.cvx[i] = c.cvx[k]; s.cvy[i] = c.cvy[k]; s.cvz[i] = c.cvz[k]; s.m[i] = c.m[k]; s.sid[i] = c.sid[k];
    }
    return s;
}

template <class Space>
struct DevArrays {
    using V = Kokkos::View<double*, Space>;
    V x, y, z, vx, vy, vz, cvx, cvy, cvz, m, ax, ay, az, dt;
    Kokkos::View<int*, Space> sid;
    explicit DevArrays(const Cluster& c) {
        const int N = (int)c.x.size();
        auto up = [&](const std::vector<double>& h, const char* name) {
            V d(name, N);
            auto hm = Kokkos::create_mirror_view(d);
            for (int i = 0; i < N; i++) hm(i) = h[i];
            Kokkos::deep_copy(d, hm);
            return d;
        };
        x = up(c.x, "x"); y = up(c.y, "y"); z = up(c.z, "z"); vx = up(c.vx, "vx"); vy = up(c.vy, "vy"); vz = up(c.vz, "vz");
        cvx = up(c.cvx, "cvx"); cvy = up(c.cvy, "cvy"); cvz = up(c.cvz, "cvz"); m = up(c.m, "m");
        ax = V("ax", N); ay = V("ay", N); az = V("az", N); dt = V("dt", N);
        sid = Kokkos::View<int*, Space>("sid", N);
        auto hs = Kokkos::create_mirror_view(sid);
        for (int i = 0; i < N; i++) hs(i) = c.sid[i];
        Kokkos::deep_copy(sid, hs);
    }
};

static std::vector<double> to_host(const Kokkos::View<double*>& v) {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    return std::vector<double>(h.data(), h.data() + h.extent(0));
}
static int ndiff(const std::vector<double>& a, const std::vector<double>& b) {
    int n = 0;
    for (size_t k = 0; k < a.size(); k++) n += std::memcmp(&a[k], &b[k], sizeof(double)) != 0;
    return n;
}

template <class F>
static double timeit(int reps, F&& f) {
    f();
    Kokkos::fence();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) f();
    Kokkos::fence();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
}

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        std::vector<int> Ns;
        for (int a = 1; a < argc; a++) Ns.push_back(std::atoi(argv[a]));
        if (Ns.empty()) Ns = {16, 64, 256, 1024, 4096, 16384, 65536, 200000};
        const double fbin = env_d("BENCH_BIN", 0.1), rh_fix = env_d("BENCH_RH", 0.0), vc = env_d("BENCH_VC", 30.0);
        const bool cm = env_d("BENCH_CM", 0) != 0;
        const double G = G_CODE, eta_ff = 0.2, eta_fb = 0.2, eta_jerk = 0.1;
        std::printf("# exec %s, binary fraction %.2f, cm velocity %d, V_c factor %.3f\n", Exec::name(), fbin, (int)cm, vc);
        std::printf("# %8s %9s %9s %9s %9s %9s %9s %9s %9s %9s %7s %7s %9s\n", "N", "n2coll", "n2dt", "build", "query",
                    "sums", "n2ngb", "treengb", "pairs/i", "ngb", "diff", "ndiff", "pairsB/i");
        for (int N : Ns) {
            // density as a 200k cluster with r_h = 3.1 pc scaled to the set size (subsets are denser than the cluster)
            const double rh = rh_fix > 0 ? rh_fix : 3.1 * std::cbrt(N / 200000.0);
            Cluster c = make_cluster(N, fbin, rh, cm, 12345u + N);
            DevArrays<Exec::memory_space> d(c);
            if (std::getenv("BENCH_DIAG")) {   // per-pair bound: how many pairs could not be excluded even pair by pair
                std::mt19937 rs(1);
                double s_bound = 0, s_2 = 0, s_10 = 0;
                const int ns = 200;
                for (int q = 0; q < ns; q++) {
                    int i = rs() % N;
                    std::vector<double> t(N, 1e30), lb(N, 0);
                    double mn = 1e30;
                    for (int j = 0; j < N; j++) {
                        if (j == i || (c.sid[i] >= 0 && c.sid[j] == c.sid[i])) continue;
                        double dx = c.x[j] - c.x[i], dy = c.y[j] - c.y[i], dz = c.z[j] - c.z[i];
                        double r2 = dx * dx + dy * dy + dz * dz, ir = 1 / std::sqrt(r2);
                        double ux = c.cvx[j] - c.cvx[i], uy = c.cvy[j] - c.cvy[i], uz = c.cvz[j] - c.cvz[i];
                        t[j] = pair_dt(r2, ir, dx, dy, dz, ux, uy, uz, G, c.m[i] + c.m[j], eta_ff, eta_fb);
                        double v = std::sqrt(ux * ux + uy * uy + uz * uz);
                        lb[j] = dtt::dt_lower_bound(r2, r2, 0.0, c.m[i] + c.m[j], c.m[i] + c.m[j], v, v, G, eta_ff, eta_fb);
                        mn = std::min(mn, t[j]);
                    }
                    for (int j = 0; j < N; j++) {
                        s_bound += lb[j] < mn; s_2 += t[j] < 2 * mn; s_10 += t[j] < 10 * mn;
                    }
                }
                std::printf("# diag N=%d: per sink, pairs with per-pair bound < min %.1f, step < 2 min %.1f, < 10 min %.1f\n", N,
                            s_bound / ns, s_2 / ns, s_10 / ns);
            }
            const int reps = std::getenv("BENCH_REPS") ? std::atoi(std::getenv("BENCH_REPS"))
                                                       : std::max(1, std::min(200, (int)(3e9 / ((double)N * N + 1e6))));
            RawView X{d.x.data()}, Y{d.y.data()}, Z{d.z.data()}, M{d.m.data()}, VX{d.vx.data()}, VY{d.vy.data()},
                VZ{d.vz.data()}, CX{d.cvx.data()}, CY{d.cvy.data()}, CZ{d.cvz.data()}, AX{d.ax.data()}, AY{d.ay.data()},
                AZ{d.az.data()}, DT{d.dt.data()};
            const int* sid = d.sid.data();
            DtK<true> kc{X, Y, Z, M, VX, VY, VZ, AX, AY, AZ, DT, CX, CY, CZ, RawView{nullptr}, RawView{nullptr}, sid, G, 0.0, eta_ff, eta_fb, eta_jerk};
            const double t_n2c = timeit(reps, [&] { run_n2("dt_n2", 0, N, 0, N, kc); });
            auto dt_ref = to_host(d.dt), ax_ref = to_host(d.ax);
            DtK<false> kn{X, Y, Z, M, CX, CY, CZ, AX, AY, AZ, DT, CX, CY, CZ, RawView{nullptr}, RawView{nullptr}, nullptr, G, 0.0, eta_ff, eta_fb, 0.0};
            const double t_n2d = timeit(reps, [&] { run_n2("dt_n2_nocoll", 0, N, 0, N, kn); });
            auto dtn_ref = to_host(d.dt);

            dtt::PointTree<Exec> T, V;
            const double t_build = timeit(reps, [&] {
                T.build(0, N, X.ptr, Y.ptr, Z.ptr, CX.ptr, CY.ptr, CZ.ptr, M.ptr, false, true);
                V.build(0, N, X.ptr, Y.ptr, Z.ptr, CX.ptr, CY.ptr, CZ.ptr, M.ptr, true);
            });
            dtt::DtParams prm{G, 0.0, eta_ff, eta_fb, vc};
            long long np = 0, npB = 0;
            dtt::min_pair_dt(T, V, 0, N, X.ptr, Y.ptr, Z.ptr, CX.ptr, CY.ptr, CZ.ptr, M.ptr, sid, DT.ptr, prm, &np, &npB);
            if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, Exec::memory_space>::accessible) {
                if (std::getenv("BENCH_DIAG")) {   // leaves of the position tree whose bound fails, by limiting term
                    Kokkos::fence();
                    double cnt_leaf = 0, by_ff = 0, by_lo = 0, by_hi = 0, rlo0 = 0;
                    const int ns = 50;
                    for (int q = 0; q < ns; q++) {
                        const int i = (int)((long long)q * 7919 % N);
                        const double mn = DT.ptr[i];
                        const double ux = c.cvx[i] - T.vmean[0], uy = c.cvy[i] - T.vmean[1], uz = c.cvz[i] - T.vmean[2];
                        const double vtyp = std::sqrt(ux * ux + uy * uy + uz * uz) + 2 * T.vsig;
                        const double rc = mn * 2.2 * vtyp / 0.2;
                        const double Vc = std::min(vc * T.vsig, 2 * G * (c.m[i] + T.node(1, 12)) * mn / (rc * rc));
                        for (int l = 0; l < T.P; l++) {
                            const int nd = T.P + l;
                            if (T.node(nd, 12) < 0) continue;
                            double r2lo, r2hi, v2lo, v2hi;
                            dtt::box_dist2(&T.node(nd, 0), &T.node(nd, 3), c.x[i], c.y[i], c.z[i], r2lo, r2hi);
                            dtt::box_dist2(&T.node(nd, 6), &T.node(nd, 9), c.cvx[i], c.cvy[i], c.cvz[i], v2lo, v2hi);
                            double vlo = std::max(std::sqrt(v2lo), Vc), vhi = std::sqrt(v2hi);
                            if (vlo > vhi) continue;
                            double mh = c.m[i] + T.node(nd, 12), ml = c.m[i] + T.node(nd, 13);
                            double lb = dtt::dt_lower_bound(r2lo, r2hi, 0.0, ml, mh, vlo, vhi, G, eta_ff, eta_fb);
                            if (lb >= mn) continue;
                            cnt_leaf++;
                            if (r2lo == 0) { rlo0++; continue; }
                            double lff = dtt::dt_lower_bound(r2lo, r2hi, 0.0, ml, mh, vhi, vhi, G, eta_ff, 1e10);
                            double llo = dtt::dt_lower_bound(r2lo, r2hi, 0.0, ml, mh, vlo, vlo, G, 1e10, eta_fb);
                            double lhi = dtt::dt_lower_bound(r2lo, r2hi, 0.0, ml, mh, vhi, vhi, G, 1e10, eta_fb);
                            if (lff < mn) by_ff++;
                            if (llo < mn) by_lo++;
                            if (lhi < mn) by_hi++;
                        }
                    }
                    std::printf("# diag leaves with bound < min per sink: %.1f (sink inside box %.1f; ff %.1f, fb at v_lo %.1f, "
                                "fb at v_hi %.1f)\n", cnt_leaf / ns, rlo0 / ns, by_ff / ns, by_lo / ns, by_hi / ns);
                }
            }
            const double t_q = timeit(reps, [&] {
                dtt::min_pair_dt(T, V, 0, N, X.ptr, Y.ptr, Z.ptr, CX.ptr, CY.ptr, CZ.ptr, M.ptr, sid, DT.ptr, prm);
            });
            DtK<true, true> ks{X, Y, Z, M, VX, VY, VZ, AX, AY, AZ, DT, CX, CY, CZ, RawView{nullptr}, RawView{nullptr}, sid, G, 0.0, eta_ff, eta_fb, eta_jerk};
            const double t_s = timeit(reps, [&] {
                dtt::min_pair_dt(T, V, 0, N, X.ptr, Y.ptr, Z.ptr, CX.ptr, CY.ptr, CZ.ptr, M.ptr, sid, DT.ptr, prm);
                run_n2("dt_sums", 0, N, 0, N, ks);
            }) - t_q;
            int diff = ndiff(to_host(d.dt), dt_ref) + ndiff(to_host(d.ax), ax_ref);
            dtt::min_pair_dt(T, V, 0, N, X.ptr, Y.ptr, Z.ptr, CX.ptr, CY.ptr, CZ.ptr, M.ptr, nullptr, DT.ptr, prm);
            diff += ndiff(to_host(d.dt), dtn_ref);

            // neighbours, r_ngb = 1e-3 pc, r_ngb_max = 0.1 pc, pivot 1e-3 TU
            const int maxp = 8 * N + 1000;
            Kokkos::View<int*> pi("pi", maxp), pj("pj", maxp), pb("pb", maxp), cnt("cnt", 1);
            Kokkos::View<double*> pa("pa", maxp), pe("pe", maxp), pE("pE", maxp);
            NgbK kg{X, Y, Z, M, VX, VY, VZ, pi.data(), pj.data(), pb.data(), cnt.data(), pa.data(), pe.data(), pE.data(),
                    maxp, G, 1e-3, 0.01, 1e-3};
            auto sorted_pairs = [&]() {
                int n = 0;
                Kokkos::deep_copy(n, Kokkos::subview(cnt, 0));
                auto hi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pi);
                auto hj = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pj);
                auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pa);
                std::vector<std::pair<long long, double>> v;
                for (int k = 0; k < std::min(n, maxp); k++) v.push_back({(long long)hi(k) * N + hj(k), ha(k)});
                std::sort(v.begin(), v.end());
                return v;
            };
            const double t_n2g = timeit(reps, [&] { Kokkos::deep_copy(cnt, 0); run_n2("ngb_n2", 0, N, 0, N, kg); });
            auto ng_ref = sorted_pairs();
            dtt::NgbOut o{pi.data(), pj.data(), pb.data(), cnt.data(), pa.data(), pe.data(), pE.data(), maxp};
            const double t_tg = timeit(reps, [&] {
                Kokkos::deep_copy(cnt, 0);
                T.build(0, N, X.ptr, Y.ptr, Z.ptr, VX.ptr, VY.ptr, VZ.ptr, M.ptr, false);
                dtt::ngb_pairs(T, 0, N, X.ptr, Y.ptr, Z.ptr, VX.ptr, VY.ptr, VZ.ptr, M.ptr, G, 1e-3, 0.01, 1e-3, o);
            });
            auto ng = sorted_pairs();
            int ngdiff = ng.size() != ng_ref.size();
            for (size_t k = 0; !ngdiff && k < ng.size(); k++)
                ngdiff = ng[k].first != ng_ref[k].first || std::memcmp(&ng[k].second, &ng_ref[k].second, 8) != 0;
            std::printf("  %8d %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e %9.1f %9zu %7d %7d %9.1f\n", N, t_n2c, t_n2d, t_build,
                        t_q, t_s, t_n2g, t_tg, (double)np / N, ng_ref.size(), diff, ngdiff, (double)npB / N);
            if constexpr (!std::is_same_v<Exec, HostExec>) {
                // host-thread tree on host copies (the TREE_DT_CPU_MAX path of an APU), against the device N^2 above
                DevArrays<Kokkos::HostSpace> h(c);
                dtt::PointTree<HostExec> HT, HV;
                const double* hx = h.x.data(); const double* hy = h.y.data(); const double* hz = h.z.data();
                const double* hm = h.m.data();
                const double* hcx = h.cvx.data(); const double* hcy = h.cvy.data(); const double* hcz = h.cvz.data();
                const double* hvx = h.vx.data(); const double* hvy = h.vy.data(); const double* hvz = h.vz.data();
                const double t_hd = timeit(reps, [&] {
                    HT.build(0, N, hx, hy, hz, hcx, hcy, hcz, hm, false, true);
                    HV.build(0, N, hx, hy, hz, hcx, hcy, hcz, hm, true);
                    dtt::min_pair_dt(HT, HV, 0, N, hx, hy, hz, hcx, hcy, hcz, hm, nullptr, h.dt.data(), prm);
                });
                std::vector<double> hdt(h.dt.data(), h.dt.data() + N);
                Kokkos::View<int*, Kokkos::HostSpace> qi("qi", maxp), qj("qj", maxp), qb("qb", maxp), qc("qc", 1);
                Kokkos::View<double*, Kokkos::HostSpace> qa("qa", maxp), qe("qe", maxp), qE("qE", maxp);
                dtt::NgbOut ho{qi.data(), qj.data(), qb.data(), qc.data(), qa.data(), qe.data(), qE.data(), maxp};
                const double t_hg = timeit(reps, [&] {
                    qc(0) = 0;
                    HT.build(0, N, hx, hy, hz, hvx, hvy, hvz, hm, false);
                    dtt::ngb_pairs(HT, 0, N, hx, hy, hz, hvx, hvy, hvz, hm, G, 1e-3, 0.01, 1e-3, ho);
                });
                std::printf("# host tree (%s, %d threads) N=%d: time step %.3e s, neighbours %.3e s (%d pairs); dt differing "
                            "from device N^2 (rsqrt vs 1/sqrt) %d\n", HostExec::name(), HostExec().concurrency(), N, t_hd,
                            t_hg, qc(0), ndiff(hdt, dtn_ref));
            }
            std::fflush(stdout);
        }
    }
    Kokkos::finalize();
    return 0;
}
