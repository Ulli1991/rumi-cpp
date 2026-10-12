#pragma once
#include "n2_sinkset.h"

// Generic tiled direct-summation driver for all pairwise kernels (forces, gradient correction, time steps).
//
// A kernel is an "interaction" struct K that supplies:
//   static constexpr int NSRC;   doubles per source particle kept in team scratch
//   static constexpr int NS;     summed outputs (forces, jerks, ...), accumulated with tile-blocked Kahan
//   static constexpr int NM;     minimised outputs (time steps)
//   struct Sink { ... };         per-sink data held in registers
//   Sink sink(int i) const;
//   void source(int j, double* s) const;                                   fill NSRC doubles
//   void pair(const Sink&, int i, const double* s, int j, double* sum, double* mn) const;
//   void finish(int i, const Sink&, const double* sum, const double* mn) const;
// for sinks [i0, i0 + ni) against sources [j0, j0 + nj). pair() must leave sum and mn unchanged for an
// excluded pair (self pair, partner in the same subsystem), preferably without branching (see pair_weight).
//
// Accuracy: per sink, plain sums over blocks of N2_KAHAN_BLOCK sources, Kahan summation of the block sums
// (error grows over one block, ~1e-15 relative, instead of per pair). finish() receives sum - compensation.
//
// Occupancy: with few sinks (multi-GPU slices, mid HHS levels) one team per TILE_SIZE sinks leaves most of
// the GPU idle. The source range is then split into chunks over extra teams; the partial sums are combined
// in chunk order in a second kernel, so results do not depend on scheduling. Host backends never split. The chunk
// count follows the whole sink set (n2_sinkset.h), so one rank's slice of a large set can still leave SMs idle; such
// launches put P = 2 or 4 threads on each sink (run_tiled), which leaves every sum bit-identical.

#include "constants.h"
#include "small_for.h"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <type_traits>

constexpr int N2_KAHAN_BLOCK = 128;

// 0 for an excluded pair: the same particle, or both members of one subsystem
KOKKOS_INLINE_FUNCTION bool pair_weight(int i, int j, int sid_i, int sid_j) {
    return (i != j) & !((sid_i >= 0) & (sid_j == sid_i));
}

// Integer (and single-precision) source data in a double slot of the source tile, as raw bits: an int read back from
// a double-valued slot costs a 64-bit conversion per pair (F2I.F64 at half the FP64 rate on A100), a bit move is free
KOKKOS_INLINE_FUNCTION double n2_pack(int lo, int hi = 0) {
    return Kokkos::bit_cast<double>((unsigned long long)(unsigned)lo | ((unsigned long long)(unsigned)hi << 32));
}
KOKKOS_INLINE_FUNCTION double n2_pack(int lo, float hi) { return n2_pack(lo, Kokkos::bit_cast<int>(hi)); }
KOKKOS_INLINE_FUNCTION double n2_pack(float lo, float hi) { return n2_pack(Kokkos::bit_cast<int>(lo), Kokkos::bit_cast<int>(hi)); }
KOKKOS_INLINE_FUNCTION int n2_lo(double d) { return (int)(unsigned)Kokkos::bit_cast<unsigned long long>(d); }
KOKKOS_INLINE_FUNCTION int n2_hi(double d) { return (int)(unsigned)(Kokkos::bit_cast<unsigned long long>(d) >> 32); }
KOKKOS_INLINE_FUNCTION float n2_lo_f(double d) { return Kokkos::bit_cast<float>(n2_lo(d)); }
KOKKOS_INLINE_FUNCTION float n2_hi_f(double d) { return Kokkos::bit_cast<float>(n2_hi(d)); }

namespace n2_detail {
inline Kokkos::View<double*>& partial_buffer() {
    static Kokkos::View<double*> buf("n2_partials", 0);
    static bool hook = (Kokkos::push_finalize_hook([] { buf = Kokkos::View<double*>(); }), true);
    (void)hook;
    return buf;
}

// Number of source chunks: about 8 teams per compute unit, chunks of at least 4 tiles (ni: the whole set's sinks)
inline int chunk_count(int ni, int nj, int TS) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
    const int target_teams = std::max(1, Kokkos::DefaultExecutionSpace().concurrency() / (2 * TS));
    int iblocks = (ni + TS - 1) / TS;
    int n = (target_teams + iblocks - 1) / iblocks;
    n = std::min(n, std::max(1, nj / (4 * TS)));
    return std::max(1, std::min(n, 64));
#else
    (void)ni; (void)nj; (void)TS;
    return 1;
#endif
}
}  // namespace n2_detail

// Pair counts up to this run on the host where it can access device memory (NBODY_SMALL_N2, default 4096)
inline long long n2_host_max() {
    static const long long v = std::getenv("NBODY_SMALL_N2") ? std::atoll(std::getenv("NBODY_SMALL_N2")) : 4096;
    return v;
}

namespace n2_detail {
// One team: TS threads, TS / P sinks, P threads per sink. Each sink's sources are taken in tiles of TS (aligned to the
// chunk start; a tile is the Kahan block, as TS >= N2_KAHAN_BLOCK on the devices), and the P threads of a sink take
// P consecutive tiles at a time: thread p computes the plain sum of tile p, the first thread then Kahan-adds the P tile
// sums in tile order. That is exactly the P = 1 accumulation (plain sum per tile, Kahan over tiles in order), so the
// sums are bit-identical for every P; minima are order-independent (the pre-rejects only skip pairs that cannot lower
// the minimum). P > 1 multiplies the teams by P (finer load balance; threads_per_sink).
template <int TS, int P, class K>
void run_tiled(const char* name, int i0, int ni, int j0, int nj, const K& k) {
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    constexpr int NS = K::NS, NM = K::NM, NSRC = K::NSRC;
    constexpr int NSa = NS > 0 ? NS : 1, NMa = NM > 0 ? NM : 1;
    constexpr int S = TS / P;                 // sinks per team
    constexpr int GROUP = P * TS;             // sources staged per step
    static_assert(P == 1 || (TS % P == 0 && TS >= N2_KAHAN_BLOCK), "P > 1 needs tiles that are whole Kahan blocks");
    constexpr int XCH = P > 1 ? (P - 1) * S * (NS + NM) : 0;   // tile sums / minima handed to the first thread
    const size_t scratch = (size_t)std::max(GROUP * NSRC, XCH) * sizeof(double);
    const int iblocks = (ni + S - 1) / S;
    const int nchunk = nj > 0 ? n2_detail::chunk_count(n2_detail::set_size(ni), nj, TS) : 1;
    const int chunk = nchunk > 1 ? ((nj + nchunk - 1) / nchunk + TS - 1) / TS * TS : nj;
    const int nc = nchunk > 1 ? (nj + chunk - 1) / chunk : 1;
    constexpr int STRIDE = 2 * NS + NM;   // per sink and chunk: sums, compensations, minima

    double* part = nullptr;
    if (nc > 1) {
        auto& buf = n2_detail::partial_buffer();
        size_t need = (size_t)nc * ni * STRIDE;
        if (buf.extent(0) < need) Kokkos::realloc(buf, need);
        part = buf.data();
    }

    Kokkos::parallel_for(name,
        team_policy(iblocks * nc, TS).set_scratch_size(0, Kokkos::PerTeam(scratch)),
        KOKKOS_LAMBDA(const member_type& team) {
            const int q = team.league_rank() / iblocks;          // source chunk
            const int ib = team.league_rank() - q * iblocks;
            const int t = team.team_rank();
            const int p = P > 1 ? t / S : 0;                     // tile of the group taken by this thread
            const int ii = ib * S + (P > 1 ? t - p * S : t);
            const int i = i0 + ii;
            const bool active = ii < ni;
            const int c0 = j0 + q * chunk, c1 = Kokkos::min(j0 + nj, c0 + chunk);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged> tile(team.team_scratch(0),
                                                                               scratch / sizeof(double));

            typename K::Sink s{};
            if (active) s = k.sink(i);
            double sum[NSa], comp[NSa], blk[NSa], mn[NMa];
            for (int a = 0; a < NSa; a++) sum[a] = comp[a] = blk[a] = 0.0;
            for (int a = 0; a < NMa; a++) mn[a] = 1e30;
            int in_block = 0;
            (void)in_block;

            for (int g0 = c0; g0 < c1; g0 += GROUP) {
                const int ng = Kokkos::min(GROUP, c1 - g0);
                for (int u = t; u < ng; u += TS) k.source(g0 + u, &tile(u * NSRC));
                team.team_barrier();
                const int b0 = p * TS, n = Kokkos::min(TS, ng - b0);   // this thread's tile of the group
                if (active && n > 0) {
                    for (int jt = 0; jt < n; jt++) k.pair(s, i, &tile((b0 + jt) * NSRC), g0 + b0 + jt, blk, mn);
                    if (P == 1) {   // not constexpr-if: nvcc extended lambdas cannot first-capture there
                        in_block += n;
                        if (in_block >= N2_KAHAN_BLOCK || g0 + n >= c1) {
                            for (int a = 0; a < NS; a++) { kahan_add(sum[a], comp[a], blk[a]); blk[a] = 0.0; }
                            in_block = 0;
                        }
                    }
                }
                team.team_barrier();
                if (P > 1) {
                    const int ntile = (ng + TS - 1) / TS;        // tiles in this group (the last may be partial)
                    double* x = &tile(0);
                    if (p > 0 && active && p < ntile) {
                        double* o = x + ((size_t)(p - 1) * S + (t - p * S)) * (NS + NM);
                        for (int a = 0; a < NS; a++) o[a] = blk[a];
                        for (int a = 0; a < NM; a++) o[NS + a] = mn[a];
                    }
                    team.team_barrier();
                    if (p == 0 && active) {
                        for (int a = 0; a < NS; a++) { kahan_add(sum[a], comp[a], blk[a]); blk[a] = 0.0; }
                        for (int r = 1; r < ntile; r++) {
                            const double* o = x + ((size_t)(r - 1) * S + t) * (NS + NM);
                            for (int a = 0; a < NS; a++) kahan_add(sum[a], comp[a], o[a]);
                            for (int a = 0; a < NM; a++) mn[a] = Kokkos::fmin(mn[a], o[NS + a]);
                        }
                    }
                    if (p > 0) for (int a = 0; a < NS; a++) blk[a] = 0.0;
                    team.team_barrier();
                }
            }
            if (!active || p > 0) return;
            if (nc == 1) {
                double out[NSa];
                for (int a = 0; a < NS; a++) out[a] = sum[a] - comp[a];
                k.finish(i, s, out, mn);
            } else {
                double* pq = part + ((size_t)q * ni + ii) * STRIDE;
                for (int a = 0; a < NS; a++) { pq[a] = sum[a]; pq[NS + a] = comp[a]; }
                for (int a = 0; a < NM; a++) pq[2 * NS + a] = mn[a];
            }
        });

    if (nc > 1) {
        Kokkos::parallel_for("n2_reduce_chunks", ni, KOKKOS_LAMBDA(int ii) {
            const int i = i0 + ii;
            double sum[NSa], comp[NSa], mn[NMa];
            for (int a = 0; a < NSa; a++) sum[a] = comp[a] = 0.0;
            for (int a = 0; a < NMa; a++) mn[a] = 1e30;
            for (int q = 0; q < nc; q++) {
                const double* pq = part + ((size_t)q * ni + ii) * STRIDE;
                for (int a = 0; a < NS; a++) {
                    kahan_add(sum[a], comp[a], pq[a]);
                    kahan_add(sum[a], comp[a], -pq[NS + a]);   // a chunk's value is its sum minus its compensation
                }
                for (int a = 0; a < NM; a++) mn[a] = Kokkos::fmin(mn[a], pq[2 * NS + a]);
            }
            double out[NSa];
            for (int a = 0; a < NS; a++) out[a] = sum[a] - comp[a];
            k.finish(i, k.sink(i), out, mn);
        });
    }
}

// Threads per sink (P of run_tiled). Results do not depend on P, so the choice may follow the local slice (unlike the
// chunk count). Measured on one A100 (n2-bench, 200k Plummer/Kroupa, 10 % binaries; sinks:sources:set):
//   force 200k:200k 161 -> 176 Gpairs/s with P = 4, 35k:140k:140k 107 -> 146, 12.5k:50k:50k 140 -> 164, never slower;
//   grad (P = 4 exceeds 48 KB at TS = 256) 103 -> 109 and 70 -> 92 with P = 2;
//   time steps (DtK, AccDtK: FILL_ONLY) gain only where a launch has fewer than ~1.5 teams per SM (35k:140k:140k,
//   137 teams of 256: 43 -> 49) and lose elsewhere (fewer pre-rejected pairs per thread, three barriers per group).
// So: the largest P <= 4 whose staged sources fit in 48 KB, for FILL_ONLY kernels P = 2 below 1.5 teams per SM.
// NBODY_N2_P = 1, 2 or 4 forces P (where it fits).
template <class K, class = void> struct fill_only : std::false_type {};
template <class K> struct fill_only<K, std::void_t<decltype(K::FILL_ONLY)>> : std::bool_constant<K::FILL_ONLY> {};

inline int threads_per_sink(long long teams, int TS, int nsrc, int ns_nm, bool fill_only_kernel) {
    static const int forced = std::getenv("NBODY_N2_P") ? std::atoi(std::getenv("NBODY_N2_P")) : 0;
    if (ns_nm == 0) return 1;   // append-only kernels (NgbK, EdgeK) keep one thread per sink
    auto fits = [&](int P) { return (size_t)P * TS * nsrc * sizeof(double) <= 48 * 1024; };
    if (forced == 1 || forced == 2 || forced == 4) return fits(forced) ? forced : 1;
    if (fill_only_kernel) {
        const double sms = Kokkos::DefaultExecutionSpace().concurrency() / 2048.0;
        return (teams < 1.5 * sms && fits(2)) ? 2 : 1;
    }
    return fits(4) ? 4 : fits(2) ? 2 : 1;
}

template <int TS, class K>
void run_tiled_p(const char* name, int i0, int ni, int j0, int nj, const K& k) {
    const int nchunk = nj > 0 ? chunk_count(set_size(ni), nj, TS) : 1;
    const int chunk = nchunk > 1 ? ((nj + nchunk - 1) / nchunk + TS - 1) / TS * TS : nj;
    const int nc = nchunk > 1 ? (nj + chunk - 1) / chunk : 1;
    const long long teams = (long long)((ni + TS - 1) / TS) * nc;
    const int P = threads_per_sink(teams, TS, K::NSRC, K::NS + K::NM, fill_only<K>::value);
    if (P == 4) run_tiled<TS, 4>(name, i0, ni, j0, nj, k);
    else if (P == 2) run_tiled<TS, 2>(name, i0, ni, j0, nj, k);
    else run_tiled<TS, 1>(name, i0, ni, j0, nj, k);
}
}  // namespace n2_detail

template <class K>
void run_n2(const char* name, int i0, int ni, int j0, int nj, const K& k) {
    if (ni <= 0) return;
    if constexpr (kHostAccessibleDevice) {
        if ((long long)n2_detail::set_size(ni) * nj <= n2_host_max()) {   // tiny set: host loops, same accumulation as the kernel
            constexpr int NSa = K::NS > 0 ? K::NS : 1, NMa = K::NM > 0 ? K::NM : 1;
            Kokkos::fence("n2_driver.h:160");
            double src[K::NSRC > 0 ? K::NSRC : 1];
            for (int ii = 0; ii < ni; ii++) {
                const int i = i0 + ii;
                typename K::Sink s = k.sink(i);
                double sum[NSa], comp[NSa], blk[NSa], mn[NMa];
                for (int a = 0; a < NSa; a++) sum[a] = comp[a] = blk[a] = 0.0;
                for (int a = 0; a < NMa; a++) mn[a] = 1e30;
                int in_block = 0;
                for (int jj = 0; jj < nj; jj++) {
                    k.source(j0 + jj, src);
                    k.pair(s, i, src, j0 + jj, blk, mn);
                    if (++in_block >= N2_KAHAN_BLOCK || jj == nj - 1) {
                        for (int a = 0; a < K::NS; a++) { kahan_add(sum[a], comp[a], blk[a]); blk[a] = 0.0; }
                        in_block = 0;
                    }
                }
                double out[NSa];
                for (int a = 0; a < K::NS; a++) out[a] = sum[a] - comp[a];
                k.finish(i, s, out, mn);
            }
            return;
        }
    }
    // 256-thread tiles once the sinks alone fill the GPU (+5-10 % at 0.25-0.5M in tools/hip_bench), 128 otherwise,
    // where the source split matters more
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
    static const int big_tile_min = std::getenv("NBODY_TILE256_MIN") ? std::atoi(std::getenv("NBODY_TILE256_MIN")) : 131072;
    if (n2_detail::set_size(ni) >= big_tile_min) { n2_detail::run_tiled_p<256>(name, i0, ni, j0, nj, k); return; }
    n2_detail::run_tiled_p<TILE_SIZE>(name, i0, ni, j0, nj, k);
#else
    n2_detail::run_tiled<TILE_SIZE, 1>(name, i0, ni, j0, nj, k);
#endif
}
