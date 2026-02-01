#include "timestep.h"

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
    for (int i = 0; i < p.N; i++) {
        // Find coarsest level where dt_level <= dt_frost
        int l = nl - 1; // default to finest
        for (int k = 0; k < nl; k++) {
            if (levels[k] <= p.dt_frost[i]) {
                l = k;
                break;
            }
        }
        p.level[i] = l;
    }
}

int find_coarsest_active(const Particles& p) {
    int coarsest = p.n_levels - 1;
    for (int i = 0; i < p.N; i++) {
        if (p.level[i] < coarsest) coarsest = p.level[i];
    }
    return coarsest;
}
