// Kokkos O(N^2) kernel benchmark: the production interactions (n2_kernels.h, run_n2) on a Plummer cluster with Kroupa
// masses and 10 % binary members (or a real particle state from NBODY_DUMP), timed per kernel, with a bitwise checksum
// of the outputs so kernel changes can be checked for bit-identity.
//   n2-bench [ni:nj[:set] ...]      sinks [0, ni) from sources [0, nj); set: sink-set size (SinkSetScope) as when one
//                                   of several ranks computes a slice of a set (default ni)
// env: N2_KERNELS  comma list of force,force0,grad,dt,dt64,accdt (default force,grad,dt,accdt), and the experimental
//                  force variants that change results (not used by the code; timed and compared with force):
//                  force_nn  ForceK without the RSQRT_NEWTON step (CUDA double rsqrt only, ~1 ulp)
//                  dt_mix    DtMixK<false> (n2_kernels.h; after dt_nj, the production DtK variant; errors against it)
//                  force_mix mixed precision as Sapporo (Gaburov et al. 2009): positions as fp32 hi + lo pairs, the
//                            difference in double-single, everything else fp32 (rsqrt.approx), fp32 sums over 32
//                            sources folded into the fp64 tile sums (needs nj and source offsets multiple of 32)
//      N2_N        particles of the synthetic cluster (default max nj), N2_DUMP=dump_XXXX.bin: load that state instead
//      N2_REPS     repetitions (default: ~2e11 pairs per kernel and case)
//      N2_OUT      prefix: write each kernel's outputs to <prefix>_<kernel>_<ni>_<nj>.bin (doubles) for error comparisons
//      N2_KMEM     PairVelocities::k of binary members (tidal-sampling step; default 1e30 as non-members)
//      dtcheck     (in N2_KERNELS) safety margin of the fp32 time-step pre-rejects
//      dtcm, dtcm64, accdtcm, dtcheckcm  (in N2_KERNELS) dt, dt64, accdt, dtcheck with the pair steps taken with the
//                  subsystem c.o.m. velocities (FLYBY_CM_VELOCITY)
//      N2_ACC=1    force kernels: relative error against a long-double reference on 200 sample sinks
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

// Pair-step velocities of FLYBY_CM_VELOCITY: the mass-weighted c.o.m. velocity of each subsystem for its members,
// the particle velocity otherwise
void cm_velocities(const Cluster& c, std::vector<double>& cx, std::vector<double>& cy, std::vector<double>& cz) {
    const int n = (int)c.x.size();
    cx = c.vx; cy = c.vy; cz = c.vz;
    int smax = -1;
    for (int s : c.sid) smax = std::max(smax, s);
    std::vector<double> M(smax + 1, 0.0), px(smax + 1, 0.0), py(smax + 1, 0.0), pz(smax + 1, 0.0);
    for (int i = 0; i < n; i++)
        if (c.sid[i] >= 0) {
            M[c.sid[i]] += c.m[i]; px[c.sid[i]] += c.m[i] * c.vx[i]; py[c.sid[i]] += c.m[i] * c.vy[i];
            pz[c.sid[i]] += c.m[i] * c.vz[i];
        }
    for (int i = 0; i < n; i++)
        if (c.sid[i] >= 0) { cx[i] = px[c.sid[i]] / M[c.sid[i]]; cy[i] = py[c.sid[i]] / M[c.sid[i]]; cz[i] = pz[c.sid[i]] / M[c.sid[i]]; }
}

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

// ---- experimental force variants (results differ from ForceK; for the speed / accuracy trade-off only)
KOKKOS_INLINE_FUNCTION double rsqrt_plain(double x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return rsqrt(x);
#else
    return 1.0 / std::sqrt(x);
#endif
}
KOKKOS_INLINE_FUNCTION float rsqrt_ftz(float x) {
#if defined(__CUDA_ARCH__)
    float y;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
#else
    return 1.0f / std::sqrt(x);
#endif
}

// ForceK (collisional, soft2 = 0) with the plain device rsqrt: 4 FP64 operations less per pair
struct ForceNNK : ForceK<true, false> {
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, double* sum, double*) const {
        const bool w = pair_weight(i, j, a.sid, n2_lo(s[4]));
        double dx = s[0] - a.x, dy = s[1] - a.y, dz = s[2] - a.z;
        double r2 = w ? dx * dx + dy * dy + dz * dz : 1.0;
        double ir = rsqrt_plain(r2);
        double f = (w ? s[3] : 0.0) * ir * ir * ir;
        sum[0] += f * dx; sum[1] += f * dy; sum[2] += f * dz;
    }
};

// Mixed precision: x = hi + lo (fp32 each), dx = (xj_hi - xi_hi) + (xj_lo - xi_lo) in fp32 (the hi difference is exact
// for nearby pairs), r^-3 and G m r^-3 dx in fp32, fp32 partial sums over 32 sources added to the fp64 sum
struct ForceMixK {
    static constexpr int NSRC = 4, NS = 3, NM = 0;
    RawView x, y, z, m, ax, ay, az;
    const int* sid;
    double G;
    struct Sink { float xh, xl, yh, yl, zh, zl; int sid; mutable float b0, b1, b2; };
    static KOKKOS_INLINE_FUNCTION float lo(double v, float h) { return (float)(v - (double)h); }
    KOKKOS_INLINE_FUNCTION Sink sink(int i) const {
        const float xh = (float)x(i), yh = (float)y(i), zh = (float)z(i);
        return {xh, lo(x(i), xh), yh, lo(y(i), yh), zh, lo(z(i), zh), sid[i], 0.0f, 0.0f, 0.0f};
    }
    KOKKOS_INLINE_FUNCTION void source(int j, double* s) const {
        const float xh = (float)x(j), yh = (float)y(j), zh = (float)z(j);
        s[0] = n2_pack(xh, lo(x(j), xh)); s[1] = n2_pack(yh, lo(y(j), yh)); s[2] = n2_pack(zh, lo(z(j), zh));
        s[3] = n2_pack(sid[j], (float)(G * m(j)));
    }
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, double* sum, double*) const {
        const bool w = pair_weight(i, j, a.sid, n2_lo(s[3]));
        const float dx = (n2_lo_f(s[0]) - a.xh) + (n2_hi_f(s[0]) - a.xl);
        const float dy = (n2_lo_f(s[1]) - a.yh) + (n2_hi_f(s[1]) - a.yl);
        const float dz = (n2_lo_f(s[2]) - a.zh) + (n2_hi_f(s[2]) - a.zl);
        const float r2 = w ? dx * dx + dy * dy + dz * dz : 1.0f;
        const float ir = rsqrt_ftz(r2);
        const float f = (w ? n2_hi_f(s[3]) : 0.0f) * ir * ir * ir;
        a.b0 += f * dx; a.b1 += f * dy; a.b2 += f * dz;
        if ((j & 31) == 31) {
            sum[0] += (double)a.b0; sum[1] += (double)a.b1; sum[2] += (double)a.b2;
            a.b0 = a.b1 = a.b2 = 0.0f;
        }
    }
    KOKKOS_INLINE_FUNCTION void finish(int i, const Sink&, const double* sum, const double*) const {
        ax(i) = sum[0]; ay(i) = sum[1]; az(i) = sum[2];
    }
};

// Safety check of the fp32 pre-rejects (dtcheck): every pair evaluated in fp64; for the pairs the pre-rejects would skip,
// the smallest ratio of the exact step to the running minimum they were tested against (must stay >= 1; the design
// margin is 1.49 for the pair step, (1 + 4e-5) / (1 - 4e-5) for the tidal-sampling step), per sink in dt and ax.
// CV: pair-step velocities = the subsystem c.o.m. velocities (FLYBY_CM_VELOCITY), as DtK<..., CV = true>
template <bool CV>
struct DtCheckK : DtK<true, false, CV, true, false, false> {
    using B = DtK<true, false, CV, true, false, false>;
    using typename B::Sink;
    using B::KI; using B::G; using B::eta_ff; using B::eta_fb; using B::dt; using B::ax;
    static constexpr int NS = 0, NM = 3;
    KOKKOS_INLINE_FUNCTION void pair(const Sink& a, int i, const double* s, int j, double*, double* mn) const {
        const bool w = pair_weight(i, j, a.sid, n2_lo(s[7]));
        double dx = s[0] - a.x, dy = s[1] - a.y, dz = s[2] - a.z;
        double dvx = s[4] - a.vx, dvy = s[5] - a.vy, dvz = s[6] - a.vz;
        if constexpr (CV) { dvx = s[8] - a.cvx; dvy = s[9] - a.cvy; dvz = s[10] - a.cvz; }
        double r2s = w ? dx * dx + dy * dy + dz * dz : 1.0;
        double ir = fast_rsqrt(r2s);
        bool rp = false, rt = false;
#ifdef N2_DT_F32
        const float r2f = (float)r2s, irf = rsqrt_f32_normal(r2f);
        rp = pair_dt_reject_f32(a.cur3, r2f, irf, dvx * dvx + dvy * dvy + dvz * dvz, dx * dvx + dy * dvy + dz * dvz,
                                (float)G * (a.m32 + n2_hi_f(s[7])), (float)eta_ff, (float)eta_fb);
        rt = tidal_dt_reject_f32(a.curt, r2f, r2f * irf, a.k32, n2_hi_f(s[KI + 2]), n2_lo_f(s[KI + 2]), a.rm32);
#endif
        const double tp = pair_dt(r2s, ir, dx, dy, dz, dvx, dvy, dvz, G, a.m + s[3], eta_ff, eta_fb);
        const double tt = tidal_sampling_dt(a.k, s[KI], a.m, s[3], r2s, a.kf, s[KI + 1]);
        if (w && rp) mn[1] = Kokkos::fmin(mn[1], tp / mn[0]);
        if (w && rt) mn[2] = Kokkos::fmin(mn[2], tt / mn[0]);
        dt_min(a, mn, w ? Kokkos::fmin(tp, tt) : 1e30);
    }
    KOKKOS_INLINE_FUNCTION void finish(int i, const Sink&, const double*, const double* mn) const {
        dt(i) = mn[1]; ax(i) = mn[2];
    }
};

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
        Kokkos::deep_copy(kt, 1e30);   // PairVelocities::k: non-members; N2_KMEM: k of subsystem members (default 1e30)
        if (std::getenv("N2_KMEM")) {
            auto hk = Kokkos::create_mirror_view(kt);
            for (int i = 0; i < N; i++) hk(i) = cl.sid[i] >= 0 ? std::atof(std::getenv("N2_KMEM")) : 1e30;
            Kokkos::deep_copy(kt, hk);
        }
        Kokkos::deep_copy(kf, 0.0);
        const double G = G_CODE, eta = 0.2, dtg = 1e-3 / 0.9778;   // gradient kick step of 1e-3 Myr in code units
        RawView X{x.data()}, Y{y.data()}, Z{z.data()}, VX{vx.data()}, VY{vy.data()}, VZ{vz.data()}, M{m.data()};
        RawView AX{ax.data()}, AY{ay.data()}, AZ{az.data()}, DT{dt.data()}, GM{gm.data()}, KT{kt.data()}, KF{kf.data()};
        RawView SX{sx.data()}, SY{sy.data()}, SZ{sz.data()};
        std::vector<double> hcx, hcy, hcz;
        cm_velocities(cl, hcx, hcy, hcz);
        Kokkos::View<double*> cvx = dev("cvx", hcx), cvy = dev("cvy", hcy), cvz = dev("cvz", hcz);
        RawView CX{cvx.data()}, CY{cvy.data()}, CZ{cvz.data()};
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
            // relative force error per sink, |a - a_ref| / |a_ref|: against the production kernel on all sinks (fref)
            // and, with N2_ACC, against a long-double sum on 200 sample sinks
            std::vector<double> fref[3];
            auto force_errors = [&](const char* name) {
                std::vector<double> hx = host(ax, c.ni), hy = host(ay, c.ni), hz = host(az, c.ni);
                auto stats = [&](std::vector<double>& e, const char* what) {
                    std::sort(e.begin(), e.end());
                    double rms = 0;
                    for (double v : e) rms += v * v;
                    std::printf("       %-9s rel. force error vs %s: rms %.2e median %.2e max %.2e\n", name, what,
                                std::sqrt(rms / e.size()), e[e.size() / 2], e.back());
                };
                if (!fref[0].empty() && std::string(name) != "force") {
                    std::vector<double> e(c.ni);
                    for (int i = 0; i < c.ni; i++) {
                        const double d = std::sqrt(std::pow(hx[i] - fref[0][i], 2) + std::pow(hy[i] - fref[1][i], 2) +
                                                   std::pow(hz[i] - fref[2][i], 2));
                        e[i] = d / std::sqrt(fref[0][i] * fref[0][i] + fref[1][i] * fref[1][i] + fref[2][i] * fref[2][i]);
                    }
                    stats(e, "force");
                }
                if (std::getenv("N2_ACC")) {
                    std::vector<double> e;
                    std::mt19937_64 rng(7);
                    std::uniform_int_distribution<int> pick(0, c.ni - 1);
                    for (int smp = 0; smp < 200; smp++) {
                        int i = pick(rng);
                        long double acc[3] = {0, 0, 0};
                        for (int j = 0; j < c.nj; j++) {
                            if (j == i || (cl.sid[i] >= 0 && cl.sid[j] == cl.sid[i])) continue;
                            long double dx = (long double)cl.x[j] - cl.x[i], dy = (long double)cl.y[j] - cl.y[i],
                                        dz = (long double)cl.z[j] - cl.z[i];
                            long double r2 = dx * dx + dy * dy + dz * dz;
                            long double f = (long double)(G * cl.m[j]) / (r2 * sqrtl(r2));
                            acc[0] += f * dx; acc[1] += f * dy; acc[2] += f * dz;
                        }
                        long double d = sqrtl(powl(hx[i] - acc[0], 2) + powl(hy[i] - acc[1], 2) + powl(hz[i] - acc[2], 2));
                        e.push_back((double)(d / sqrtl(acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2])));
                    }
                    stats(e, "long dbl");
                }
            };
            const bool variants = want("force_nn") || want("force_mix");
            if (want("force") || variants) {
                ForceK<true> k{X, Y, Z, M, AX, AY, AZ, sid.data(), G, 0.0};
                bench("force", [&] { run_n2("bench_force", 0, c.ni, 0, c.nj, k); }, {ax, ay, az});
                fref[0] = host(ax, c.ni); fref[1] = host(ay, c.ni); fref[2] = host(az, c.ni);
                force_errors("force");
            }
            if (want("force_nn")) {
                ForceNNK k{ForceK<true, false>{X, Y, Z, M, AX, AY, AZ, sid.data(), G, 0.0}};
                bench("force_nn", [&] { run_n2("bench_force_nn", 0, c.ni, 0, c.nj, k); }, {ax, ay, az});
                force_errors("force_nn");
            }
            if (want("force_mix")) {
                ForceMixK k{X, Y, Z, M, AX, AY, AZ, sid.data(), G};
                bench("force_mix", [&] { run_n2("bench_force_mix", 0, c.ni, 0, c.nj, k); }, {ax, ay, az});
                force_errors("force_mix");
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
                DtK<true, false, false, true> k{X, Y, Z, M, VX, VY, VZ, AX, AY, AZ, DT, VX, VY, VZ, KT, KF, sid.data(),
                                                G, 0.0, eta, eta, 0.0};
                bench("dt", [&] { run_n2("bench_dt", 0, c.ni, 0, c.nj, k); }, {dt, ax, ay, az});
            }
            if (want("dt_mix")) {   // reference: the production DtK variant (no jerk, unsoftened), then the mixed one
                DtK<true, false, false, true> k0{X, Y, Z, M, VX, VY, VZ, AX, AY, AZ, DT, VX, VY, VZ, KT, KF, sid.data(),
                                                 G, 0.0, eta, eta, 0.0};
                auto k = k0.template with<false, false>();
                bench("dt_nj", [&] { run_n2("bench_dt_nj", 0, c.ni, 0, c.nj, k); }, {dt, ax, ay, az});
                std::vector<double> rd = host(dt, c.ni), rx = host(ax, c.ni), ry = host(ay, c.ni), rz = host(az, c.ni);
                DtMixK<false> km{X, Y, Z, M, VX, VY, VZ, AX, AY, AZ, DT, KT, KF, sid.data(), (float)G, (float)eta, (float)eta};
                bench("dt_mix", [&] { run_n2("bench_dt_mix", 0, c.ni, 0, c.nj, km); }, {dt, ax, ay, az});
                std::vector<double> hd = host(dt, c.ni), hx = host(ax, c.ni), hy = host(ay, c.ni), hz = host(az, c.ni);
                std::vector<double> ed(c.ni), ea(c.ni);
                int lev = 0, bad = 0;
                for (int i = 0; i < c.ni; i++) {
                    if (!(hd[i] > 0) || !std::isfinite(hd[i])) bad++;
                    ed[i] = std::fabs(hd[i] - rd[i]) / rd[i];
                    lev += (int)std::floor(std::log2(1e-3 / 0.9778 / rd[i])) != (int)std::floor(std::log2(1e-3 / 0.9778 / hd[i]));
                    ea[i] = std::sqrt(std::pow(hx[i] - rx[i], 2) + std::pow(hy[i] - ry[i], 2) + std::pow(hz[i] - rz[i], 2)) /
                            std::sqrt(rx[i] * rx[i] + ry[i] * ry[i] + rz[i] * rz[i]);
                }
                for (auto* e : {&ed, &ea}) std::sort(e->begin(), e->end());
                std::printf("       dt_mix    rel. dt error median %.2e max %.2e; level changed %d of %d; bad %d\n",
                            ed[c.ni / 2], ed.back(), lev, c.ni, bad);
                std::printf("       dt_mix    rel. acc error median %.2e max %.2e\n", ea[c.ni / 2], ea.back());
            }
            if (want("dtcm")) {   // FLYBY_CM_VELOCITY: pair steps with the subsystem c.o.m. velocities
                DtK<true, false, true, true> k{X, Y, Z, M, VX, VY, VZ, AX, AY, AZ, DT, CX, CY, CZ, KT, KF, sid.data(),
                                               G, 0.0, eta, eta, 0.0};
                bench("dtcm", [&] { run_n2("bench_dtcm", 0, c.ni, 0, c.nj, k); }, {dt, ax, ay, az});
            }
            if (want("dtcm64")) {
                DtK<true, false, true, false> k{X, Y, Z, M, VX, VY, VZ, AX, AY, AZ, DT, CX, CY, CZ, KT, KF, sid.data(),
                                                G, 0.0, eta, eta, 0.0};
                bench("dtcm64", [&] { run_n2("bench_dtcm64", 0, c.ni, 0, c.nj, k); }, {dt, ax, ay, az});
            }
            if (want("dt64")) {   // without the fp32 pre-reject
                DtK<true, false, false, false> k{X, Y, Z, M, VX, VY, VZ, AX, AY, AZ, DT, VX, VY, VZ, KT, KF, sid.data(),
                                                 G, 0.0, eta, eta, 0.0};
                bench("dt64", [&] { run_n2("bench_dt64", 0, c.ni, 0, c.nj, k); }, {dt, ax, ay, az});
            }
            if (want("accdt")) {
                AccDtK<true> k{X, Y, Z, M, VX, VY, VZ, AX, AY, AZ, DT, VX, VY, VZ, KT, KF, sid.data(), G, 0.0, eta, eta,
                               0.0, 0.0};
                bench("accdt", [&] { run_n2("bench_accdt", 0, c.ni, 0, c.nj, k); }, {dt, ax, ay, az});
            }
            if (want("accdtcm")) {
                AccDtK<true, true, true> k{X, Y, Z, M, VX, VY, VZ, AX, AY, AZ, DT, CX, CY, CZ, KT, KF, sid.data(), G, 0.0,
                                           eta, eta, 0.0, 0.0};
                bench("accdtcm", [&] { run_n2("bench_accdtcm", 0, c.ni, 0, c.nj, k); }, {dt, ax, ay, az});
            }
            auto dtcheck = [&](const char* name, auto k) {
                run_n2(name, 0, c.ni, 0, c.nj, k);
                std::vector<double> r = host(dt, c.ni), rt = host(ax, c.ni);
                std::printf("%s ni=%7d nj=%7d  smallest exact step / running minimum over pre-rejected pairs: "
                            "pair step %.4f, tidal-sampling step %.6f\n", name + 6, c.ni, c.nj,
                            *std::min_element(r.begin(), r.end()), *std::min_element(rt.begin(), rt.end()));
            };
            if (want("dtcheck"))
                dtcheck("bench_dtcheck", DtCheckK<false>{DtK<true, false, false, true, false, false>{X, Y, Z, M, VX, VY, VZ, AX,
                                         AY, AZ, DT, VX, VY, VZ, KT, KF, sid.data(), G, 0.0, eta, eta, 0.0}});
            if (want("dtcheckcm"))   // the pre-rejects with the c.o.m. pair-step velocities
                dtcheck("bench_dtcheckcm", DtCheckK<true>{DtK<true, false, true, true, false, false>{X, Y, Z, M, VX, VY, VZ, AX,
                                           AY, AZ, DT, CX, CY, CZ, KT, KF, sid.data(), G, 0.0, eta, eta, 0.0}});
        }
    }
    Kokkos::finalize();
    return 0;
}
