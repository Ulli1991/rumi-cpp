// Standalone HIP benchmark of the direct-summation force kernel (forces only, Kahan summation, the same
// pair interaction and subsystem-exclusion test as gravity_interaction_kahan in src/core/constants.h).
//
//   hipcc -O3 --offload-arch=gfx942 -o nbody_bench nbody_bench.hip
//   ./nbody_bench [n_sink n_source]...
//
// Kernels:
//   base    one sink per thread, 128-thread blocks, 128-source LDS tiles (layout of the Kokkos kernel)
//   b256    one sink per thread, 256-thread blocks
//   r2, r4  2 or 4 sinks per thread (register blocking), 128-thread blocks
//   split   base layout, the source range split into chunks over extra blocks; per-chunk partial sums
//           and compensations are reduced in a second kernel in chunk order (deterministic)
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); std::exit(1); } } while (0)

__device__ __forceinline__ double rsq(double x) {
    double y = rsqrt(x);
    return y * (1.5 - 0.5 * x * y * y);   // RSQRT_NEWTON
}
__device__ __forceinline__ void kadd(double& s, double& c, double v) {
    double y = v - c, t = s + y;
    c = (t - s) - y;
    s = t;
}
__device__ __forceinline__ void pair(double xi, double yi, double zi, int sidi, const double4& pj, int sidj,
                                     double soft2, double& ax, double& ay, double& az,
                                     double& cx, double& cy, double& cz) {
    if (sidi >= 0 && sidj == sidi) return;
    double dx = pj.x - xi, dy = pj.y - yi, dz = pj.z - zi;
    double r2 = dx * dx + dy * dy + dz * dz + soft2;
    double ir = rsq(r2);
    double f = pj.w * ir * ir * ir;
    kadd(ax, cx, f * dx);
    kadd(ay, cy, f * dy);
    kadd(az, cz, f * dz);
}

// Sinks [0, ni) of pos, sources [j0, j1) of pos (G folded into the mass)
template <int BS, int R>
__global__ void __launch_bounds__(BS) k_tiled(const double4* __restrict__ pos, const int* __restrict__ sid,
                                               int ni, int j0, int j1, double soft2,
                                               double* __restrict__ out, double* __restrict__ comp) {
    __shared__ double4 tp[BS];
    __shared__ int ts[BS];
    int base = (blockIdx.x * BS) * R + threadIdx.x;
    double xi[R], yi[R], zi[R], ax[R], ay[R], az[R], cx[R], cy[R], cz[R];
    int si[R];
    for (int r = 0; r < R; r++) {
        int i = base + r * BS;
        double4 p = i < ni ? pos[i] : make_double4(0, 0, 0, 0);
        xi[r] = p.x; yi[r] = p.y; zi[r] = p.z;
        si[r] = i < ni ? sid[i] : -1;
        ax[r] = ay[r] = az[r] = cx[r] = cy[r] = cz[r] = 0;
    }
    for (int t0 = j0; t0 < j1; t0 += BS) {
        int n = min(BS, j1 - t0);
        if (threadIdx.x < n) { tp[threadIdx.x] = pos[t0 + threadIdx.x]; ts[threadIdx.x] = sid[t0 + threadIdx.x]; }
        __syncthreads();
        for (int jt = 0; jt < n; jt++) {
            double4 pj = tp[jt];
            int sj = ts[jt];
#pragma unroll
            for (int r = 0; r < R; r++) pair(xi[r], yi[r], zi[r], si[r], pj, sj, soft2, ax[r], ay[r], az[r], cx[r], cy[r], cz[r]);
        }
        __syncthreads();
    }
    for (int r = 0; r < R; r++) {
        int i = base + r * BS;
        if (i < ni) {
            out[3 * i] = ax[r]; out[3 * i + 1] = ay[r]; out[3 * i + 2] = az[r];
            comp[3 * i] = cx[r]; comp[3 * i + 1] = cy[r]; comp[3 * i + 2] = cz[r];
        }
    }
}

template <int MODE, bool BF>
__device__ __forceinline__ void pair2(double xi, double yi, double zi, int sidi, const double4& pj, int sidj,
                                      double soft2, double& ax, double& ay, double& az,
                                      double& cx, double& cy, double& cz) {
    double mj = pj.w;
    if (BF) mj = (sidi >= 0 && sidj == sidi) ? 0.0 : mj;
    else if (sidi >= 0 && sidj == sidi) return;
    double dx = pj.x - xi, dy = pj.y - yi, dz = pj.z - zi;
    double r2 = dx * dx + dy * dy + dz * dz + soft2;
    double ir = rsq(r2);
    double f = mj * ir * ir * ir;
    if (MODE == 0) { kadd(ax, cx, f * dx); kadd(ay, cy, f * dy); kadd(az, cz, f * dz); }
    else { ax += f * dx; ay += f * dy; az += f * dz; }
}
// MODE 2: plain sums over one tile, compensated sum of the tile sums
template <int BS, int R, int MODE, bool BF>
__global__ void __launch_bounds__(BS) k_var(const double4* __restrict__ pos, const int* __restrict__ sid,
                                             int ni, int j0, int j1, double soft2,
                                             double* __restrict__ out, double* __restrict__ comp) {
    __shared__ double4 tp[BS];
    __shared__ int ts[BS];
    int base = (blockIdx.x * BS) * R + threadIdx.x;
    double xi[R], yi[R], zi[R], ax[R], ay[R], az[R], cx[R], cy[R], cz[R];
    int si[R];
    for (int r = 0; r < R; r++) {
        int i = base + r * BS;
        double4 p = i < ni ? pos[i] : make_double4(0, 0, 0, 0);
        xi[r] = p.x; yi[r] = p.y; zi[r] = p.z;
        si[r] = i < ni ? sid[i] : -1;
        ax[r] = ay[r] = az[r] = cx[r] = cy[r] = cz[r] = 0;
    }
    for (int t0 = j0; t0 < j1; t0 += BS) {
        int n = min(BS, j1 - t0);
        if (threadIdx.x < n) { tp[threadIdx.x] = pos[t0 + threadIdx.x]; ts[threadIdx.x] = sid[t0 + threadIdx.x]; }
        __syncthreads();
        if (MODE == 2) {
            double tx[R], ty[R], tz[R], dum = 0;
            for (int r = 0; r < R; r++) tx[r] = ty[r] = tz[r] = 0;
            for (int jt = 0; jt < n; jt++) {
                double4 pj = tp[jt]; int sj = ts[jt];
#pragma unroll
                for (int r = 0; r < R; r++) pair2<1, BF>(xi[r], yi[r], zi[r], si[r], pj, sj, soft2, tx[r], ty[r], tz[r], dum, dum, dum);
            }
            for (int r = 0; r < R; r++) { kadd(ax[r], cx[r], tx[r]); kadd(ay[r], cy[r], ty[r]); kadd(az[r], cz[r], tz[r]); }
        } else {
            for (int jt = 0; jt < n; jt++) {
                double4 pj = tp[jt]; int sj = ts[jt];
#pragma unroll
                for (int r = 0; r < R; r++) pair2<MODE, BF>(xi[r], yi[r], zi[r], si[r], pj, sj, soft2, ax[r], ay[r], az[r], cx[r], cy[r], cz[r]);
            }
        }
        __syncthreads();
    }
    for (int r = 0; r < R; r++) {
        int i = base + r * BS;
        if (i < ni) {
            out[3 * i] = ax[r]; out[3 * i + 1] = ay[r]; out[3 * i + 2] = az[r];
            comp[3 * i] = cx[r]; comp[3 * i + 1] = cy[r]; comp[3 * i + 2] = cz[r];
        }
    }
}

// Source-split: blockIdx.y selects the chunk; partial results go to out/comp + chunk * 3 * ni
template <int BS>
__global__ void __launch_bounds__(BS) k_split(const double4* __restrict__ pos, const int* __restrict__ sid,
                                               int ni, int j0, int j1, int chunk, double soft2,
                                               double* __restrict__ out, double* __restrict__ comp) {
    int c0 = j0 + blockIdx.y * chunk, c1 = min(j1, c0 + chunk);
    size_t off = (size_t)blockIdx.y * 3 * ni;
    __shared__ double4 tp[BS];
    __shared__ int ts[BS];
    int i = blockIdx.x * BS + threadIdx.x;
    double4 p = i < ni ? pos[i] : make_double4(0, 0, 0, 0);
    int si = i < ni ? sid[i] : -1;
    double ax = 0, ay = 0, az = 0, cx = 0, cy = 0, cz = 0;
    for (int t0 = c0; t0 < c1; t0 += BS) {
        int n = min(BS, c1 - t0);
        if (threadIdx.x < n) { tp[threadIdx.x] = pos[t0 + threadIdx.x]; ts[threadIdx.x] = sid[t0 + threadIdx.x]; }
        __syncthreads();
        for (int jt = 0; jt < n; jt++) pair(p.x, p.y, p.z, si, tp[jt], ts[jt], soft2, ax, ay, az, cx, cy, cz);
        __syncthreads();
    }
    if (i < ni) {
        out[off + 3 * i] = ax; out[off + 3 * i + 1] = ay; out[off + 3 * i + 2] = az;
        comp[off + 3 * i] = cx; comp[off + 3 * i + 1] = cy; comp[off + 3 * i + 2] = cz;
    }
}
__global__ void k_reduce(int ni, int nchunk, const double* __restrict__ part, const double* __restrict__ pcomp,
                         double* __restrict__ out, double* __restrict__ comp) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;   // component 3*i + d
    if (k >= 3 * ni) return;
    double s = 0, c = 0;
    for (int q = 0; q < nchunk; q++) {
        kadd(s, c, part[(size_t)q * 3 * ni + k]);
        kadd(s, c, -pcomp[(size_t)q * 3 * ni + k]);   // a chunk's compensation is the negated lost low part
    }
    out[k] = s; comp[k] = c;
}

int main(int argc, char** argv) {
    std::vector<std::pair<int, int>> cases;
    for (int a = 1; a + 1 < argc; a += 2) cases.push_back({std::atoi(argv[a]), std::atoi(argv[a + 1])});
    if (cases.empty()) cases = {{65536, 65536}, {262144, 262144}, {524288, 524288},
                                {50000, 200000}, {12500, 200000}, {2000, 200000}};
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device %s, %d CUs\n", prop.name, prop.multiProcessorCount);

    for (auto [ni, nj] : cases) {
        int n = std::max(ni, nj);
        std::vector<double4> h(n);
        std::vector<int> hs(n, -1);
        std::mt19937_64 rng(42);
        std::normal_distribution<double> g(0.0, 1.0);
        for (int i = 0; i < n; i++) h[i] = make_double4(g(rng), g(rng), g(rng), 1.0 / n);
        for (int i = 0; i + 1 < n / 10; i += 2) hs[i] = hs[i + 1] = i;   // 10 % of particles in pairs
        double4* d; int* ds; double *o, *c, *po, *pc;
        int maxchunk = 64;
        CHECK(hipMalloc(&d, n * sizeof(double4)));
        CHECK(hipMalloc(&ds, n * sizeof(int)));
        CHECK(hipMalloc(&o, 3 * (size_t)ni * sizeof(double)));
        CHECK(hipMalloc(&c, 3 * (size_t)ni * sizeof(double)));
        CHECK(hipMalloc(&po, 3 * (size_t)ni * maxchunk * sizeof(double)));
        CHECK(hipMalloc(&pc, 3 * (size_t)ni * maxchunk * sizeof(double)));
        CHECK(hipMemcpy(d, h.data(), n * sizeof(double4), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(ds, hs.data(), n * sizeof(int), hipMemcpyHostToDevice));
        double soft2 = 1e-6;
        std::vector<double> ref(3 * ni), res(3 * ni);
        hipEvent_t e0, e1;
        CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
        double pairs = (double)ni * nj;

        auto timeit = [&](const char* name, auto launch, bool is_ref) {
            launch();   // warm-up
            CHECK(hipDeviceSynchronize());
            int reps = std::max(3, (int)(2e11 / pairs));
            reps = std::min(reps, 200);
            CHECK(hipEventRecord(e0));
            for (int r = 0; r < reps; r++) launch();
            CHECK(hipEventRecord(e1));
            CHECK(hipEventSynchronize(e1));
            float ms; CHECK(hipEventElapsedTime(&ms, e0, e1));
            double t = ms / 1e3 / reps;
            CHECK(hipMemcpy(res.data(), o, 3 * (size_t)ni * sizeof(double), hipMemcpyDeviceToHost));
            if (is_ref) ref = res;
            double maxrel = 0;
            for (int k = 0; k < 3 * ni; k += 3) {
                double a = std::sqrt(ref[k] * ref[k] + ref[k + 1] * ref[k + 1] + ref[k + 2] * ref[k + 2]);
                double dd = std::sqrt(std::pow(res[k] - ref[k], 2) + std::pow(res[k + 1] - ref[k + 1], 2) + std::pow(res[k + 2] - ref[k + 2], 2));
                if (a > 0) maxrel = std::max(maxrel, dd / a);
            }
            std::printf("  %-12s %9.3f ms %8.1f Gpairs/s   max rel diff vs base %.1e\n", name, t * 1e3, pairs / t / 1e9, maxrel);
        };
        std::printf("n_sink=%d n_source=%d\n", ni, nj);
        timeit("base", [&] { k_tiled<128, 1><<<(ni + 127) / 128, 128>>>(d, ds, ni, 0, nj, soft2, o, c); }, true);
        timeit("b256", [&] { k_tiled<256, 1><<<(ni + 255) / 256, 256>>>(d, ds, ni, 0, nj, soft2, o, c); }, false);
        timeit("b64", [&] { k_tiled<64, 1><<<(ni + 63) / 64, 64>>>(d, ds, ni, 0, nj, soft2, o, c); }, false);
        timeit("r2", [&] { k_tiled<128, 2><<<(ni + 255) / 256, 128>>>(d, ds, ni, 0, nj, soft2, o, c); }, false);
        timeit("r4", [&] { k_tiled<128, 4><<<(ni + 511) / 512, 128>>>(d, ds, ni, 0, nj, soft2, o, c); }, false);
        timeit("bf", [&] { k_var<128, 1, 0, true><<<(ni + 127) / 128, 128>>>(d, ds, ni, 0, nj, soft2, o, c); }, false);
        timeit("plain", [&] { k_var<128, 1, 1, true><<<(ni + 127) / 128, 128>>>(d, ds, ni, 0, nj, soft2, o, c); }, false);
        timeit("tileK", [&] { k_var<128, 1, 2, true><<<(ni + 127) / 128, 128>>>(d, ds, ni, 0, nj, soft2, o, c); }, false);
        timeit("tileK r2", [&] { k_var<128, 2, 2, true><<<(ni + 255) / 256, 128>>>(d, ds, ni, 0, nj, soft2, o, c); }, false);
        timeit("tileK r4", [&] { k_var<128, 4, 2, true><<<(ni + 511) / 512, 128>>>(d, ds, ni, 0, nj, soft2, o, c); }, false);
        timeit("tileK b256", [&] { k_var<256, 1, 2, true><<<(ni + 255) / 256, 256>>>(d, ds, ni, 0, nj, soft2, o, c); }, false);
        // split: enough blocks for ~8 per CU
        int iblocks = (ni + 127) / 128;
        int nchunk = std::min(maxchunk, std::max(1, (8 * prop.multiProcessorCount + iblocks - 1) / iblocks));
        int chunk = ((nj + nchunk - 1) / nchunk + 127) / 128 * 128;
        nchunk = (nj + chunk - 1) / chunk;
        char nm[32]; std::snprintf(nm, sizeof nm, "split x%d", nchunk);
        timeit(nm, [&] {
            k_split<128><<<dim3(iblocks, nchunk), 128>>>(d, ds, ni, 0, nj, chunk, soft2, po, pc);
            k_reduce<<<(3 * ni + 255) / 256, 256>>>(ni, nchunk, po, pc, o, c);
        }, false);
        CHECK(hipFree(d)); CHECK(hipFree(ds)); CHECK(hipFree(o)); CHECK(hipFree(c)); CHECK(hipFree(po)); CHECK(hipFree(pc));
    }
    return 0;
}
