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

    // Only allocate and copy levels if size changed or first call
    if (p.d_level_dts.extent(0) != (size_t)nl) {
        p.d_level_dts = Kokkos::View<double*>("d_level_dts", nl);
        auto h_levels = Kokkos::create_mirror_view(p.d_level_dts);
        for (int k = 0; k < nl; k++) h_levels(k) = levels[k];
        Kokkos::deep_copy(p.d_level_dts, h_levels);
    }

    auto d_levels = p.d_level_dts;
    auto p_level = p.level;
    auto p_dt_adapt = p.dt_adapt;
    int N = p.N;

    Kokkos::parallel_for("assign_levels", N, KOKKOS_LAMBDA(int i) {
        int l = nl - 1;
        for (int k = 0; k < nl; k++) {
            if (d_levels(k) <= p_dt_adapt(i)) {
                l = k;
                break;
            }
        }
        p_level(i) = l;
    });
}

