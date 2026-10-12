#pragma once

#include "particles.h"
#include <string>

// Cluster structure diagnostics for long runs (HARD_PROBLEMS.md G2): density centre and core (Casertano & Hut 1985,
// ApJ 298, 80: density of each central particle from its k = 6 nearest neighbours, rho_j = 3 M_(k-1) / (4 pi r_k^3);
// centre = sum rho_j x_j / sum rho_j, core radius r_c = sqrt(sum rho_j^2 |x_j - c|^2 / sum rho_j^2), central density
// sum rho_j^2 / sum rho_j), Lagrangian radii about the density centre, and a census of bound pairs (binaries found
// directly from the particles, independent of the integrator's subsystem lists). Computed on the host from a full
// particle copy; replicated MPI runs call it on rank 0.
struct ClusterDiag {
    static constexpr int NLAG = 7;
    static constexpr double LAG_FRAC[NLAG] = {0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9};
    double t_myr = 0.0;
    double M_total = 0.0;
    double center[3] = {0, 0, 0}, vcenter[3] = {0, 0, 0};   // density centre (pc) and its velocity (km/s)
    double r_lag[NLAG] = {};                               // pc
    double r_core = 0.0, rho_core = 0.0;                   // pc, Msun / pc^3
    int n_core_sample = 0;                                 // particles used for the density estimate
    int n_bin = 0, n_bin_hard = 0;                         // bound pairs (mutual most-bound partners), hard: |E_b| > <m> sigma^2
    double E_bin = 0.0;                                    // total binding energy of those pairs (code units, < 0)
    double sigma_c = 0.0;                                  // 1D velocity dispersion inside r_lag(10 %) (km/s)
};

// G: gravitational constant in code units; r_pair_max: largest pair separation searched for binaries (pc)
ClusterDiag compute_cluster_diag(const Particles& p, double G, double t_myr, double r_pair_max = 0.02);

// One line per call (header written when the file is new): t, centre, r_c, rho_c, Lagrangian radii, binaries
void append_cluster_diag(const std::string& path, int step, const ClusterDiag& d);
