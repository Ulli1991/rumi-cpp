// Kokkos counterpart of tools/hip_bench/nbody_bench.hip: the production force interaction (ForceK<true>, run_n2:
// tile-blocked Kahan, branch-free subsystem exclusion, source split) on the same particle distribution and cases.
//   n2-bench [n_sink n_source]...
#include "n2_kernels.h"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        std::vector<std::pair<int, int>> cases;
        for (int a = 1; a + 1 < argc; a += 2) cases.push_back({std::atoi(argv[a]), std::atoi(argv[a + 1])});
        if (cases.empty()) cases = {{65536, 65536}, {262144, 262144}, {524288, 524288},
                                    {50000, 200000}, {12500, 200000}, {2000, 200000}};
        for (auto [ni, nj] : cases) {
            const int n = std::max(ni, nj);
            Kokkos::View<double*> x("x", n), y("y", n), z("z", n), m("m", n), ax("ax", n), ay("ay", n), az("az", n);
            Kokkos::View<int*> sid("sid", n);
            auto hx = Kokkos::create_mirror_view(x); auto hy = Kokkos::create_mirror_view(y);
            auto hz = Kokkos::create_mirror_view(z); auto hm = Kokkos::create_mirror_view(m);
            auto hs = Kokkos::create_mirror_view(sid);
            std::mt19937_64 rng(42);
            std::normal_distribution<double> g(0.0, 1.0);
            for (int i = 0; i < n; i++) { hx(i) = g(rng); hy(i) = g(rng); hz(i) = g(rng); hm(i) = 1.0 / n; hs(i) = -1; }
            for (int i = 0; i + 1 < n / 10; i += 2) hs(i) = hs(i + 1) = i;   // 10 % of particles in pairs
            Kokkos::deep_copy(x, hx); Kokkos::deep_copy(y, hy); Kokkos::deep_copy(z, hz); Kokkos::deep_copy(m, hm);
            Kokkos::deep_copy(sid, hs);
            // sinks [0, ni) from sources [0, nj), as the HIP benchmark (G = 1, soft2 = 1e-6)
            ForceK<true> k{{x.data()}, {y.data()}, {z.data()}, {m.data()}, {ax.data()}, {ay.data()}, {az.data()},
                           sid.data(), 1.0, 1e-6};
            run_n2("bench_force", 0, ni, 0, nj, k);
            Kokkos::fence();
            const double pairs = (double)ni * nj;
            int reps = std::min(200, std::max(3, (int)(2e11 / pairs)));
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; r++) run_n2("bench_force", 0, ni, 0, nj, k);
            Kokkos::fence();
            double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
            std::printf("kokkos n_sink=%d n_source=%d  %9.3f ms %8.1f Gpairs/s\n", ni, nj, t * 1e3, pairs / t / 1e9);
        }
    }
    Kokkos::finalize();
    return 0;
}
