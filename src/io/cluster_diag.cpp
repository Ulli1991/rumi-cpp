#include "cluster_diag.h"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <unordered_map>
#include <vector>

constexpr double ClusterDiag::LAG_FRAC[];

namespace {
struct Body { double x[3], v[3], m; };

double dist2(const double* a, const double* b) {
    const double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}
}  // namespace

ClusterDiag compute_cluster_diag(const Particles& p, double G, double t_myr, double r_pair_max) {
    ClusterDiag d;
    d.t_myr = t_myr;
    const int N0 = p.N;
    if (N0 <= 0) return d;
    auto r = Kokkos::make_pair(0, N0);
    auto hx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.x, r));
    auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.y, r));
    auto hz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.z, r));
    auto hvx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.vx, r));
    auto hvy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.vy, r));
    auto hvz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.vz, r));
    auto hm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.mass, r));
    std::vector<Body> b;
    b.reserve(N0);
    for (int i = 0; i < N0; i++)
        if (hm(i) > 0.0) b.push_back({{hx(i), hy(i), hz(i)}, {hvx(i), hvy(i), hvz(i)}, hm(i)});
    const int N = (int)b.size();
    if (N == 0) return d;
    for (const auto& q : b) d.M_total += q.m;

    // 1. Shrinking spheres (Power et al. 2003): mass centre of the particles inside R, R reduced by 2.5 % per pass
    double c[3] = {0, 0, 0};
    for (const auto& q : b) for (int k = 0; k < 3; k++) c[k] += q.m * q.x[k] / d.M_total;
    double R2 = 0.0;
    for (const auto& q : b) R2 = std::max(R2, dist2(q.x, c));
    const int n_stop = std::max(std::min(N, 1000), N / 100);
    for (int it = 0; it < 2000; it++) {
        R2 *= 0.975 * 0.975;
        double cn[3] = {0, 0, 0}, mm = 0.0;
        int n_in = 0;
        for (const auto& q : b)
            if (dist2(q.x, c) < R2) { for (int k = 0; k < 3; k++) cn[k] += q.m * q.x[k]; mm += q.m; n_in++; }
        if (n_in < n_stop || mm <= 0.0) break;
        for (int k = 0; k < 3; k++) c[k] = cn[k] / mm;
    }

    // 2. Casertano & Hut density centre and core from the n_s particles nearest that centre (brute-force 6-NN among
    // them: the densest particles, which carry the weight, have all their neighbours inside the sample)
    constexpr int K = 6;
    const int n_s = std::min(N, 20000);
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::nth_element(idx.begin(), idx.begin() + (n_s - 1), idx.end(),
                     [&](int a, int bb) { return dist2(b[a].x, c) < dist2(b[bb].x, c); });
    idx.resize(n_s);
    std::vector<double> rho(n_s, 0.0);
    if (n_s > K) {
        Kokkos::parallel_for("cluster_diag_knn", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n_s),
                             [&](int a) {
            double nd[K]; int ni[K];
            for (int k = 0; k < K; k++) { nd[k] = 1e300; ni[k] = -1; }
            const Body& qa = b[idx[a]];
            for (int s = 0; s < n_s; s++) {
                if (s == a) continue;
                const double r2 = dist2(qa.x, b[idx[s]].x);
                if (r2 >= nd[K - 1]) continue;
                int k = K - 1;
                while (k > 0 && nd[k - 1] > r2) { nd[k] = nd[k - 1]; ni[k] = ni[k - 1]; k--; }
                nd[k] = r2; ni[k] = s;
            }
            double m_in = 0.0;   // M_(k-1): the k - 1 nearest, without the k-th (Casertano & Hut)
            for (int k = 0; k < K - 1; k++) m_in += b[idx[ni[k]]].m;
            const double rk = std::sqrt(nd[K - 1]);
            rho[a] = rk > 0.0 ? 3.0 * m_in / (4.0 * M_PI * rk * rk * rk) : 0.0;
        });
        double s1 = 0.0, s2 = 0.0, cc[3] = {0, 0, 0};
        for (int a = 0; a < n_s; a++) {
            s1 += rho[a]; s2 += rho[a] * rho[a];
            for (int k = 0; k < 3; k++) cc[k] += rho[a] * b[idx[a]].x[k];
        }
        if (s1 > 0.0) {
            for (int k = 0; k < 3; k++) c[k] = cc[k] / s1;
            double rc2 = 0.0;
            for (int a = 0; a < n_s; a++) rc2 += rho[a] * rho[a] * dist2(b[idx[a]].x, c);
            d.r_core = std::sqrt(rc2 / s2);
            d.rho_core = s2 / s1;
        }
        d.n_core_sample = n_s;
    }
    for (int k = 0; k < 3; k++) d.center[k] = c[k];

    // 3. Lagrangian radii about the density centre
    std::vector<std::pair<double, double>> rm(N);
    for (int i = 0; i < N; i++) rm[i] = {dist2(b[i].x, c), b[i].m};
    std::sort(rm.begin(), rm.end());
    double cum = 0.0;
    int l = 0;
    for (int i = 0; i < N && l < ClusterDiag::NLAG; i++) {
        cum += rm[i].second;
        while (l < ClusterDiag::NLAG && cum >= ClusterDiag::LAG_FRAC[l] * d.M_total) d.r_lag[l++] = std::sqrt(rm[i].first);
    }

    // 4. Velocity centre and 1D dispersion inside the 10 % Lagrangian radius
    {
        const double r10 = d.r_lag[2] * d.r_lag[2];
        double mv[3] = {0, 0, 0}, mm = 0.0;
        for (const auto& q : b) if (dist2(q.x, c) <= r10) { for (int k = 0; k < 3; k++) mv[k] += q.m * q.v[k]; mm += q.m; }
        if (mm > 0.0) {
            for (int k = 0; k < 3; k++) d.vcenter[k] = mv[k] / mm;
            double s = 0.0;
            for (const auto& q : b)
                if (dist2(q.x, c) <= r10) s += q.m * dist2(q.v, d.vcenter);
            d.sigma_c = std::sqrt(s / (3.0 * mm));
        }
    }

    // 5. Bound pairs: every particle's most bound partner within r_pair_max (spatial hash, cell = r_pair_max); a
    // binary is a mutual most-bound pair with E < 0. Hard: |E_b| > <m> sigma_c^2
    {
        const double h = r_pair_max;
        auto key = [&](long long ix, long long iy, long long iz) { return (ix * 73856093LL) ^ (iy * 19349663LL) ^ (iz * 83492791LL); };
        auto cell = [&](double x) { return (long long)std::floor(x / h); };
        std::unordered_map<long long, std::vector<int>> grid;
        grid.reserve(N);
        for (int i = 0; i < N; i++) grid[key(cell(b[i].x[0]), cell(b[i].x[1]), cell(b[i].x[2]))].push_back(i);
        std::vector<int> best(N, -1);
        std::vector<double> bestE(N, 0.0);
        Kokkos::parallel_for("cluster_diag_pairs", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, N),
                             [&](int i) {
            const long long cx = cell(b[i].x[0]), cy = cell(b[i].x[1]), cz = cell(b[i].x[2]);
            for (int dx = -1; dx <= 1; dx++) for (int dy = -1; dy <= 1; dy++) for (int dz = -1; dz <= 1; dz++) {
                auto it = grid.find(key(cx + dx, cy + dy, cz + dz));
                if (it == grid.end()) continue;
                for (int j : it->second) {
                    if (j == i) continue;
                    const double r2 = dist2(b[i].x, b[j].x);
                    if (r2 >= h * h || r2 <= 0.0) continue;
                    const double mu = b[i].m * b[j].m / (b[i].m + b[j].m);
                    const double E = 0.5 * mu * dist2(b[i].v, b[j].v) - G * b[i].m * b[j].m / std::sqrt(r2);
                    if (E < bestE[i]) { bestE[i] = E; best[i] = j; }
                }
            }
        });
        const double m_mean = d.M_total / N, e_hard = m_mean * d.sigma_c * d.sigma_c;
        for (int i = 0; i < N; i++) {
            const int j = best[i];
            if (j > i && best[j] == i) {
                d.n_bin++;
                d.E_bin += bestE[i];
                if (-bestE[i] > e_hard) d.n_bin_hard++;
            }
        }
    }
    return d;
}

void append_cluster_diag(const std::string& path, int step, const ClusterDiag& d) {
    std::FILE* f = std::fopen(path.c_str(), "a");
    if (!f) return;
    if (std::ftell(f) == 0) {
        std::fprintf(f, "# step t_Myr M_total cx cy cz r_core rho_core sigma_c");
        for (double fr : ClusterDiag::LAG_FRAC) std::fprintf(f, " r_lag_%g", fr);
        std::fprintf(f, " n_bin n_bin_hard E_bin\n");
    }
    std::fprintf(f, "%d %.9e %.9e %.9e %.9e %.9e %.9e %.9e %.9e", step, d.t_myr, d.M_total, d.center[0], d.center[1],
                 d.center[2], d.r_core, d.rho_core, d.sigma_c);
    for (double rl : d.r_lag) std::fprintf(f, " %.9e", rl);
    std::fprintf(f, " %d %d %.9e\n", d.n_bin, d.n_bin_hard, d.E_bin);
    std::fclose(f);
}
