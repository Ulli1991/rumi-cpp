#include "timestep.h"
#include <Kokkos_Core.hpp>

std::vector<double> build_levels(double dt_max, double dt_min) {
    std::vector<double> levels;
    double dt = dt_max;
    while (dt >= dt_min) {
        levels.push_back(dt);
        dt *= 0.5;
    }
    if (levels.empty()) levels.push_back(dt_max);
    return levels;
}

void assign_levels(Particles& p, const std::vector<double>& levels) {
    int nl = (int)levels.size();
    p.n_levels = nl;

    // Copy levels to device
    Kokkos::View<double*> d_levels("levels", nl);
    auto h_levels = Kokkos::create_mirror_view(d_levels);
    for (int k = 0; k < nl; k++) h_levels(k) = levels[k];
    Kokkos::deep_copy(d_levels, h_levels);

    auto p_level = p.level;
    auto p_dt_frost = p.dt_frost;
    int N = p.N;

    Kokkos::parallel_for("assign_levels", N, KOKKOS_LAMBDA(int i) {
        int l = nl - 1;
        for (int k = 0; k < nl; k++) {
            if (d_levels(k) <= p_dt_frost(i)) {
                l = k;
                break;
            }
        }
        p_level(i) = l;
    });
    Kokkos::fence();
}

int find_coarsest_active(const Particles& p) {
    int coarsest = p.n_levels - 1;
    auto p_level = p.level;
    int N = p.N;

    Kokkos::parallel_reduce("find_coarsest", N,
        KOKKOS_LAMBDA(int i, int& lmin) {
            if (p_level(i) < lmin) lmin = p_level(i);
        },
        Kokkos::Min<int>(coarsest));
    Kokkos::fence();
    return coarsest;
}
