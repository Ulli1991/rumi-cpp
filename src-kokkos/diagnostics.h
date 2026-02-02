#pragma once

#include "particles.h"
#include <vector>

struct DiagnosticState {
    double KE, PE, E_total;
    double Px, Py, Pz;
    double Lx, Ly, Lz;
};

DiagnosticState compute_diagnostics(const Particles& p, double G, double soft2);

void compute_errors(const DiagnosticState& cur, const DiagnosticState& ini,
                    const Particles& p, bool is_cluster,
                    double& dE, double& dP, double& dL);

void print_timebin_table(const Particles& p, const std::vector<double>& levels,
                         double dE, double dP, double dL,
                         double t_yr, int step, bool is_cluster);

void print_final(const DiagnosticState& cur, const DiagnosticState& ini,
                 const Particles& p, const std::vector<double>& levels,
                 bool is_cluster, double wall_sec, int steps,
                 const char* const* names);
