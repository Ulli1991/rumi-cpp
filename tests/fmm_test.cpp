// FMM check: Plummer sphere, FMM accelerations against direct summation (same kernel), momentum, timings
// against the direct GPU kernel (run_n2).   fmm-test N p theta [n_leaf] [kernel 0|2] [eps] [n_sample]
#include "fmm.h"
#include "n2_kernels.h"
#include <Kokkos_Core.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        if (argc < 4) { std::printf("usage: %s N p theta [n_leaf] [kernel] [eps] [n_sample]\n", argv[0]); Kokkos::finalize(); return 1; }
        const int N = std::atoi(argv[1]);
        FmmParams prm;
        prm.p = std::atoi(argv[2]); prm.theta = std::atof(argv[3]);
        prm.n_leaf = argc > 4 ? std::atoi(argv[4]) : 64;
        prm.kernel = argc > 5 ? std::atoi(argv[5]) : 0;
        prm.eps = argc > 6 ? std::atof(argv[6]) : 0.0;
        const int n_sample = argc > 7 ? std::atoi(argv[7]) : 1000;
        std::mt19937_64 rng(1);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        Kokkos::View<double*> x("x", N), y("y", N), z("z", N), m("m", N), ax("ax", N), ay("ay", N), az("az", N);
        auto hx = Kokkos::create_mirror_view(x); auto hy = Kokkos::create_mirror_view(y);
        auto hz = Kokkos::create_mirror_view(z); auto hm = Kokkos::create_mirror_view(m);
        for (int i = 0; i < N; i++) {
            double r;
            do { r = 1.0 / std::sqrt(std::pow(U(rng), -2.0 / 3.0) - 1.0); } while (r > 50.0);
            double ct = 2 * U(rng) - 1, ph = 2 * M_PI * U(rng), st = std::sqrt(1 - ct * ct);
            hx(i) = r * st * std::cos(ph); hy(i) = r * st * std::sin(ph); hz(i) = r * ct; hm(i) = 1.0 / N;
        }
        Kokkos::deep_copy(x, hx); Kokkos::deep_copy(y, hy); Kokkos::deep_copy(z, hz); Kokkos::deep_copy(m, hm);
        Kokkos::View<int*> nosid;
        FmmStats st;
        fmm_accelerations(x, y, z, m, nosid, 0, N, prm, ax, ay, az, &st);   // warm-up (setup of rotations)
        auto t0 = std::chrono::steady_clock::now();
        fmm_accelerations(x, y, z, m, nosid, 0, N, prm, ax, ay, az, &st);
        double t_fmm = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        // direct reference (and timing) with the driver; kernel 0 only (Newtonian, no softening)
        Kokkos::View<double*> dx("dx", N), dy("dy", N), dz("dz", N);
        ForceK<false> k{{x.data()}, {y.data()}, {z.data()}, {m.data()}, {dx.data()}, {dy.data()}, {dz.data()}, nullptr, 1.0, 0.0};
        run_n2("direct_ref", 0, N, 0, N, k);
        Kokkos::fence();
        auto t1 = std::chrono::steady_clock::now();
        run_n2("direct_ref", 0, N, 0, N, k);
        Kokkos::fence();
        double t_dir = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        auto hax = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ax), hay = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ay),
             haz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), az);
        auto hdx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dx), hdy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dy),
             hdz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dz);
        std::vector<double> err;
        long double P[3] = {0, 0, 0}, Pa = 0, Tq[3] = {0, 0, 0}, Ta = 0, Td[3] = {0, 0, 0};
        for (int i = 0; i < N; i++) {
            P[0] += hm(i) * hax(i); P[1] += hm(i) * hay(i); P[2] += hm(i) * haz(i);
            double amag = std::sqrt(hax(i) * hax(i) + hay(i) * hay(i) + haz(i) * haz(i));
            Pa += hm(i) * amag;
            // net torque sum m x cross a (zero for exact pair forces), relative to sum m |x| |a|
            Tq[0] += hm(i) * (hy(i) * haz(i) - hz(i) * hay(i));
            Tq[1] += hm(i) * (hz(i) * hax(i) - hx(i) * haz(i));
            Tq[2] += hm(i) * (hx(i) * hay(i) - hy(i) * hax(i));
            Td[0] += hm(i) * (hy(i) * hdz(i) - hz(i) * hdy(i));
            Td[1] += hm(i) * (hz(i) * hdx(i) - hx(i) * hdz(i));
            Td[2] += hm(i) * (hx(i) * hdy(i) - hy(i) * hdx(i));
            Ta += hm(i) * std::sqrt(hx(i) * hx(i) + hy(i) * hy(i) + hz(i) * hz(i)) * amag;
        }
        double dL = (double)(std::sqrt(Tq[0] * Tq[0] + Tq[1] * Tq[1] + Tq[2] * Tq[2]) / Ta);
        double dLd = (double)(std::sqrt(Td[0] * Td[0] + Td[1] * Td[1] + Td[2] * Td[2]) / Ta);
        std::uniform_int_distribution<int> pick(0, N - 1);
        for (int s = 0; s < n_sample; s++) {
            int i = pick(rng);
            double e = std::sqrt(std::pow(hax(i) - hdx(i), 2) + std::pow(hay(i) - hdy(i), 2) + std::pow(haz(i) - hdz(i), 2)) /
                       std::sqrt(hdx(i) * hdx(i) + hdy(i) * hdy(i) + hdz(i) * hdz(i));
            err.push_back(e);
        }
        std::sort(err.begin(), err.end());
        double rms = 0;
        for (double e : err) rms += e * e;
        rms = std::sqrt(rms / err.size());
        std::printf("FMM N=%d p=%d theta=%.2f leaf=%d | err rms=%.2e median=%.2e max=%.2e | dP=%.1e torque=%.1e (direct %.1e) | fmm %.3f s "
                    "(tree %.3f up %.3f lists %.3f m2l %.3f p2p %.3f down %.3f; cells %d leaves %d m2l %lld) | direct %.3f s\n",
                    N, prm.p, prm.theta, prm.n_leaf, rms, err[err.size() / 2], err.back(),
                    (double)(std::sqrt(P[0] * P[0] + P[1] * P[1] + P[2] * P[2]) / Pa), dL, dLd, t_fmm, st.t_tree, st.t_up, st.t_lists,
                    st.t_m2l, st.t_p2p, st.t_down, st.n_cells, st.n_leaves, st.n_m2l, t_dir);
        fmm_release();
    }
    Kokkos::finalize();
    return 0;
}
