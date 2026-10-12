// Kokkos O(N^2) kernel benchmark: the production interactions (n2_kernels.h, run_n2) on a Plummer cluster with Kroupa
// masses and 10 % binary members (or a real particle state from NBODY_DUMP), timed per kernel, with a bitwise checksum
// of the outputs so kernel changes can be checked for bit-identity.
//   n2-bench [ni:nj[:set] ...]      sinks [0, ni) from sources [0, nj); set: sink-set size (SinkSetScope) as when one
//                                   of several ranks computes a slice of a set (default ni)
// env: N2_KERNELS  comma list of force,force0,grad,dt,dt64,accdt (default force,grad,dt,accdt)
//      N2_N        particles of the synthetic cluster (default max nj), N2_DUMP=dump_XXXX.bin: load that state instead
//      N2_REPS     repetitions (default: ~2e11 pairs per kernel and case)
//      N2_OUT      prefix: write each kernel's outputs to <prefix>_<kernel>_<ni>_<nj>.bin (doubles) for error comparisons
//      N2_ACC=1    force kernel: relative error against a long-double reference on 200 sample sinks
#include "n2_kernels.h"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
struct Cluster {
    std::vector<double> x, y, z, vx, vy, vz, m;
    std::vector<int> sid;
};

// Plummer sphere (r_h = 3.1 pc), Kroupa 0.08-150 Msun, the first 10 % of the particles in circular binaries with a
// log-uniform in 0.1-100 AU, shuffled (as tests/dt_tree_bench.cpp)
Cluster make_cluster(int N, unsigned seed) {
    Cluster c;
    for (auto* v : {&c.x, &c.y, &c.z, &c.vx, &c.vy, &c.vz, &c.m}) v->resize(N);
    c.sid.assign(N, -1);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    auto kroupa = [&]() {
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
    const double a = 3.1 / 1.305, G = G_CODE;
    auto iso = [&](double r, double& X, double& Y, double& Z) {
        double ct = 2 * U(rng) - 1, ph = 2 * M_PI * U(rng), st = std::sqrt(1 - ct * ct);
        X = r * st * std::cos(ph); Y = r * st * std::sin(ph); Z = r * ct;
    };
    for (int i = 0; i < N; i++) {
        double r = a / std::sqrt(std::pow(U(rng) * 0.999 + 1e-9, -2.0 / 3.0) - 1.0);
        iso(r, c.x[i], c.y[i], c.z[i]);
        double q, g;
        do { q = U(rng); g = 0.1 * U(rng); } while (g > q * q * std::pow(1 - q * q, 3.5));
        iso(q * std::sqrt(2.0) * std::pow(1 + r * r / (a * a), -0.25) * std::sqrt(G * M / a), c.vx[i], c.vy[i], c.vz[i]);
    }
    const int nb = N / 20;
    for (int k = 0; k < nb; k++) {
        int i = 2 * k, j = 2 * k + 1;
        double ab = std::exp(std::log(0.1) + U(rng) * std::log(1000.0)) / 206264.806;
        double mt = c.m[i] + c.m[j], ex, ey, ez, px, py, pz;
        iso(1.0, ex, ey, ez);
        iso(1.0, px, py, pz);
        double pe = px * ex + py * ey + pz * ez;
        px -= pe * ex; py -= pe * ey; pz -= pe * ez;
        double pn = std::sqrt(px * px + py * py + pz * pz);
        px /= pn; py /= pn; pz /= pn;
        double vo = std::sqrt(G * mt / ab), fi = c.m[j] / mt, fj = c.m[i] / mt;
        double cx = c.x[i], cy = c.y[i], cz = c.z[i], cvx = c.vx[i], cvy = c.vy[i], cvz = c.vz[i];
        c.x[i] = cx + fi * ab * ex; c.y[i] = cy + fi * ab * ey; c.z[i] = cz + fi * ab * ez;
        c.x[j] = cx - fj * ab * ex; c.y[j] = cy - fj * ab * ey; c.z[j] = cz - fj * ab * ez;
        c.vx[i] = cvx + fi * vo * px; c.vy[i] = cvy + fi * vo * py; c.vz[i] = cvz + fi * vo * pz;
        c.vx[j] = cvx - fj * vo * px; c.vy[j] = cvy - fj * vo * py; c.vz[j] = cvz - fj * vo * pz;
        c.sid[i] = c.sid[j] = k;
    }
    std::vector<int> perm(N);
    for (int i = 0; i < N; i++) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    Cluster s = c;
    for (int i = 0; i < N; i++) {
        int k = perm[i];
        s.x[i] = c.x[k]; s.y[i] = c.y[k]; s.z[i] = c.z[k]; s.vx[i] = c.vx[k]; s.vy[i] = c.vy[k]; s.vz[i] = c.vz[k];
        s.m[i] = c.m[k]; s.sid[i] = c.sid[k];
    }
    return s;
}

// NBODY_DUMP file: int N, then rows (pid m x y z vx vy vz subsystem_id) of doubles
bool load_dump(const char* fn, Cluster& c) {
    FILE* f = std::fopen(fn, "rb");
    if (!f) return false;
    int n = 0;
    if (std::fread(&n, sizeof n, 1, f) != 1) { std::fclose(f); return false; }
    for (auto* v : {&c.x, &c.y, &c.z, &c.vx, &c.vy, &c.vz, &c.m}) v->resize(n);
    c.sid.resize(n);
    double r[9];
    for (int i = 0; i < n; i++) {
        if (std::fread(r, sizeof r, 1, f) != 1) { std::fclose(f); return false; }
        c.m[i] = r[1]; c.x[i] = r[2]; c.y[i] = r[3]; c.z[i] = r[4]; c.vx[i] = r[5]; c.vy[i] = r[6]; c.vz[i] = r[7];
        c.sid[i] = (int)r[8];
    }
    std::fclose(f);
    return true;
}

uint64_t fnv(uint64_t h, const std::vector<double>& v) {
    for (double d : v) {
        uint64_t b;
        std::memcpy(&b, &d, 8);
        for (int k = 0; k < 8; k++) { h ^= (b >> (8 * k)) & 0xff; h *= 1099511628211ull; }
    }
    return h;
}

std::vector<double> host(const Kokkos::View<double*>& v, int n) {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    return std::vector<double>(h.data(), h.data() + n);
}
}  // namespace

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        struct Case { int ni, nj, set; };
        std::vector<Case> cases;
        for (int a = 1; a < argc; a++) {
            Case c{0, 0, 0};
            if (std::sscanf(argv[a], "%d:%d:%d", &c.ni, &c.nj, &c.set) >= 2) cases.push_back(c);
        }
        if (cases.empty()) cases = {{200000, 200000, 0}, {50000, 200000, 200000}, {12500, 200000, 0}, {2000, 200000, 0}};
        std::string kl = std::getenv("N2_KERNELS") ? std::getenv("N2_KERNELS") : "force,grad,dt,accdt";
        auto want = [&](const char* k) { return ("," + kl + ",").find("," + std::string(k) + ",") != std::string::npos; };
        int nmax = 0;
        for (auto& c : cases) nmax = std::max({nmax, c.ni, c.nj});
        Cluster cl;
        if (std::getenv("N2_DUMP")) {
            if (!load_dump(std::getenv("N2_DUMP"), cl)) { std::fprintf(stderr, "cannot read %s\n", std::getenv("N2_DUMP")); return 1; }
            std::printf("state: %s (N = %zu)\n", std::getenv("N2_DUMP"), cl.x.size());
        } else {
            int n = std::getenv("N2_N") ? std::atoi(std::getenv("N2_N")) : nmax;
            cl = make_cluster(n, 42);
            std::printf("state: synthetic Plummer/Kroupa, 10 %% binary members (N = %d)\n", n);
        }
        const int N = (int)cl.x.size();
        auto dev = [&](const char* name, const std::vector<double>& h) {
            Kokkos::View<double*> v(name, N);
            auto hv = Kokkos::create_mirror_view(v);
            for (int i = 0; i < N; i++) hv(i) = h[i];
            Kokkos::deep_copy(v, hv);
            return v;
        };
        Kokkos::View<double*> x = dev("x", cl.x), y = dev("y", cl.y), z = dev("z", cl.z), vx = dev("vx", cl.vx),
                              vy = dev("vy", cl.vy), vz = dev("vz", cl.vz), m = dev("m", cl.m);
        Kokkos::View<double*> ax("ax", N), ay("ay", N), az("az", N), dt("dt", N), gm("gm", N), kt("kt", N), kf("kf", N);
        Kokkos::View<double*> sx("sx", N), sy("sy", N), sz("sz", N);
        Kokkos::View<int*> sid("sid", N);
        {
            auto hs = Kokkos::create_mirror_view(sid);
            for (int i = 0; i < N; i++) hs(i) = cl.sid[i];
            Kokkos::deep_copy(sid, hs);
        }
        Kokkos::deep_copy(kt, 1e30);   // TidalSampling::k: non-members
        Kokkos::deep_copy(kf, 0.0);
        const double G = G_CODE, eta = 0.2, dtg = 1e-3 / 0.9778;   // gradient kick step of 1e-3 Myr in code units
        RawView X{x.data()}, Y{y.data()}, Z{z.data()}, VX{vx.data()}, VY{vy.data()}, VZ{vz.data()}, M{m.data()};
        RawView AX{ax.data()}, AY{ay.data()}, AZ{az.data()}, DT{dt.data()}, GM{gm.data()}, KT{kt.data()}, KF{kf.data()};
        RawView SX{sx.data()}, SY{sy.data()}, SZ{sz.data()};
        // snapshot accelerations for the gradient kernel: the forces of all particles
        {
            ForceK<true> k{X, Y, Z, M, SX, SY, SZ, sid.data(), G, 0.0};
            run_n2("bench_snap", 0, N, 0, N, k);
            Kokkos::fence();
        }

        for (const Case& c : cases) {
            if (c.ni > N || c.nj > N) { std::printf("case %d:%d exceeds N = %d\n", c.ni, c.nj, N); continue; }
            n2_detail::SinkSetScope scope(c.set);
            const double pairs = (double)c.ni * c.nj;
            auto bench = [&](const char* name, auto&& run, std::vector<Kokkos::View<double*>> outs) {
                run();
                Kokkos::fence();
                int reps = std::getenv("N2_REPS") ? std::atoi(std::getenv("N2_REPS"))
                                                  : std::min(200, std::max(3, (int)(2e11 / pairs)));
                auto t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < reps; r++) run();
                Kokkos::fence();
                double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
                uint64_t h = 1469598103934665603ull;
                std::vector<double> all;
                for (auto& o : outs) {
                    std::vector<double> v = host(o, c.ni);
                    h = fnv(h, v);
                    all.insert(all.end(), v.begin(), v.end());
                }
                std::printf("%-6s ni=%7d nj=%7d set=%7d %9.3f ms %8.1f Gpairs/s  chk %016llx\n", name, c.ni, c.nj,
                            c.set > 0 ? c.set : c.ni, t * 1e3, pairs / t / 1e9, (unsigned long long)h);
                if (std::getenv("N2_OUT")) {
                    char fn[512];
                    std::snprintf(fn, sizeof fn, "%s_%s_%d_%d.bin", std::getenv("N2_OUT"), name, c.ni, c.nj);
                    if (FILE* f = std::fopen(fn, "wb")) { std::fwrite(all.data(), 8, all.size(), f); std::fclose(f); }
                }
                std::fflush(stdout);
            };
            if (want("force")) {
                ForceK<true> k{X, Y, Z, M, AX, AY, AZ, sid.data(), G, 0.0};
                bench("force", [&] { run_n2("bench_force", 0, c.ni, 0, c.nj, k); }, {ax, ay, az});
                if (std::getenv("N2_ACC")) {
                    std::vector<double> hax = host(ax, c.ni), hay = host(ay, c.ni), haz = host(az, c.ni), err;
                    std::mt19937_64 rng(7);
                    std::uniform_int_distribution<int> pick(0, c.ni - 1);
                    for (int s = 0; s < 200; s++) {
                        int i = pick(rng);
                        long double a[3] = {0, 0, 0};
                        for (int j = 0; j < c.nj; j++) {
                            if (j == i || (cl.sid[i] >= 0 && cl.sid[j] == cl.sid[i])) continue;
                            long double dx = (long double)cl.x[j] - cl.x[i], dy = (long double)cl.y[j] - cl.y[i],
                                        dz = (long double)cl.z[j] - cl.z[i];
                            long double r2 = dx * dx + dy * dy + dz * dz;
                            long double f = (long double)(G * cl.m[j]) / (r2 * sqrtl(r2));
                            a[0] += f * dx; a[1] += f * dy; a[2] += f * dz;
                        }
                        long double d = sqrtl(powl(hax[i] - a[0], 2) + powl(hay[i] - a[1], 2) + powl(haz[i] - a[2], 2));
                        err.push_back((double)(d / sqrtl(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])));
                    }
                    std::sort(err.begin(), err.end());
                    double rms = 0;
                    for (double e : err) rms += e * e;
                    std::printf("       rel force error vs long double: rms %.2e median %.2e max %.2e\n",
                                std::sqrt(rms / err.size()), err[err.size() / 2], err.back());
                }
            }
            if (want("force0")) {   // non-collisional variant (no subsystem exclusion)
                ForceK<false> k{X, Y, Z, M, AX, AY, AZ, nullptr, G, 0.0};
                bench("force0", [&] { run_n2("bench_force0", 0, c.ni, 0, c.nj, k); }, {ax, ay, az});
            }
            if (want("grad")) {
                GradK<true> k{X, Y, Z, M, AX, AY, AZ, SX, SY, SZ, SX, SY, SZ, 0, 0, GM, sid.data(),
                              G * dtg * dtg * GRAD_COEFF, 0.0, 1.0 / (dtg * dtg)};
                bench("grad", [&] { run_n2("bench_grad", 0, c.ni, 0, c.nj, k); }, {ax, ay, az, gm});
            }
            // production time-step kernel: collisional, tidal k present
            if (want("dt")) {
                DtK<true, false, true> k{X, Y, Z, M, VX, VY, VZ, AX, AY, AZ, DT, KT, KF, sid.data(),
                                                G, 0.0, eta, eta, 0.0};
                bench("dt", [&] { run_n2("bench_dt", 0, c.ni, 0, c.nj, k); }, {dt, ax, ay, az});
            }
            if (want("dt64")) {   // without the fp32 pre-reject
                DtK<true, false, false> k{X, Y, Z, M, VX, VY, VZ, AX, AY, AZ, DT, KT, KF, sid.data(),
                                                 G, 0.0, eta, eta, 0.0};
                bench("dt64", [&] { run_n2("bench_dt64", 0, c.ni, 0, c.nj, k); }, {dt, ax, ay, az});
            }
            if (want("accdt")) {
                AccDtK<true> k{X, Y, Z, M, VX, VY, VZ, AX, AY, AZ, DT, KT, KF, sid.data(), G, 0.0, eta, eta,
                               0.0, 0.0};
                bench("accdt", [&] { run_n2("bench_accdt", 0, c.ni, 0, c.nj, k); }, {dt, ax, ay, az});
            }
        }
    }
    Kokkos::finalize();
    return 0;
}
