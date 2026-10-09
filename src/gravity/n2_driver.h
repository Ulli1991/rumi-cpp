#pragma once

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
// in chunk order in a second kernel, so results do not depend on scheduling. Host backends never split.

#include "constants.h"
#include "small_for.h"
#include <Kokkos_Core.hpp>
#include <algorithm>

constexpr int N2_KAHAN_BLOCK = 128;

// 0 for an excluded pair: the same particle, or both members of one subsystem
KOKKOS_INLINE_FUNCTION bool pair_weight(int i, int j, int sid_i, int sid_j) {
    return (i != j) & !((sid_i >= 0) & (sid_j == sid_i));
}

namespace n2_detail {
inline Kokkos::View<double*>& partial_buffer() {
    static Kokkos::View<double*> buf("n2_partials", 0);
    static bool hook = (Kokkos::push_finalize_hook([] { buf = Kokkos::View<double*>(); }), true);
    (void)hook;
    return buf;
}
// Number of source chunks: about 8 teams per compute unit, chunks of at least 4 tiles
inline int chunk_count(int ni, int nj) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
    static const int target_teams = std::max(1, Kokkos::DefaultExecutionSpace().concurrency() / (2 * TILE_SIZE));
    int iblocks = (ni + TILE_SIZE - 1) / TILE_SIZE;
    int n = (target_teams + iblocks - 1) / iblocks;
    n = std::min(n, std::max(1, nj / (4 * TILE_SIZE)));
    return std::max(1, std::min(n, 64));
#else
    (void)ni; (void)nj;
    return 1;
#endif
}
}  // namespace n2_detail

// Pair counts up to this run on the host where it can access device memory (NBODY_SMALL_N2, default 4096)
inline long long n2_host_max() {
    static const long long v = std::getenv("NBODY_SMALL_N2") ? std::atoll(std::getenv("NBODY_SMALL_N2")) : 4096;
    return v;
}

template <class K>
void run_n2(const char* name, int i0, int ni, int j0, int nj, const K& k) {
    if (ni <= 0) return;
    if constexpr (kHostAccessibleDevice) {
        if ((long long)ni * nj <= n2_host_max()) {   // tiny set: host loops, same accumulation as the kernel
            constexpr int NSa = K::NS > 0 ? K::NS : 1, NMa = K::NM > 0 ? K::NM : 1;
            Kokkos::fence();
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
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    constexpr int NS = K::NS, NM = K::NM, NSRC = K::NSRC;
    constexpr int NSa = NS > 0 ? NS : 1, NMa = NM > 0 ? NM : 1;
    const size_t scratch = TILE_SIZE * NSRC * sizeof(double);
    const int iblocks = (ni + TILE_SIZE - 1) / TILE_SIZE;
    const int nchunk = nj > 0 ? n2_detail::chunk_count(ni, nj) : 1;
    const int chunk = nchunk > 1 ? ((nj + nchunk - 1) / nchunk + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE : nj;
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
        team_policy(iblocks * nc, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch)),
        KOKKOS_LAMBDA(const member_type& team) {
            const int q = team.league_rank() / iblocks;          // source chunk
            const int ib = team.league_rank() - q * iblocks;
            const int ii = ib * TILE_SIZE + team.team_rank();
            const int i = i0 + ii;
            const bool active = ii < ni;
            const int c0 = j0 + q * chunk, c1 = Kokkos::min(j0 + nj, c0 + chunk);
            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged> tile(team.team_scratch(0), TILE_SIZE * NSRC);

            typename K::Sink s{};
            if (active) s = k.sink(i);
            double sum[NSa], comp[NSa], blk[NSa], mn[NMa];
            for (int a = 0; a < NSa; a++) sum[a] = comp[a] = blk[a] = 0.0;
            for (int a = 0; a < NMa; a++) mn[a] = 1e30;
            int in_block = 0;

            for (int t0 = c0; t0 < c1; t0 += TILE_SIZE) {
                const int n = Kokkos::min(TILE_SIZE, c1 - t0);
                const int t = team.team_rank();
                if (t < n) k.source(t0 + t, &tile(t * NSRC));
                team.team_barrier();
                if (active) {
                    for (int jt = 0; jt < n; jt++) k.pair(s, i, &tile(jt * NSRC), t0 + jt, blk, mn);
                    in_block += n;
                    if (in_block >= N2_KAHAN_BLOCK || t0 + n >= c1) {
                        for (int a = 0; a < NS; a++) { kahan_add(sum[a], comp[a], blk[a]); blk[a] = 0.0; }
                        in_block = 0;
                    }
                }
                team.team_barrier();
            }
            if (!active) return;
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
