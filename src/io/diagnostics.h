#pragma once

#include "particles.h"
#include <vector>

struct DiagnosticState {
    double KE, PE, E_total;
    double Px, Py, Pz;
    double Lx, Ly, Lz;
};

// Energy bookkeeping for collisional mode (tracks energy sinks)
// Conservation check: |E_total + Σ(sinks) - E_initial| / |E_initial|
struct EnergyBookkeeping {
    double E_escape = 0.0;    // energy carried away by escapers
    double E_gw = 0.0;        // gravitational wave losses (PN2.5)
    double E_merge = 0.0;     // merger dissipation
    double E_sse = 0.0;       // stellar evolution mass loss

    double total_sink() const { return E_escape + E_gw + E_merge + E_sse; }
    void reset() { E_escape = E_gw = E_merge = E_sse = 0.0; }
};

// Global energy bookkeeping instance
extern EnergyBookkeeping g_energy_books;

DiagnosticState compute_diagnostics(const Particles& p, double G, double soft2);

void compute_errors(const DiagnosticState& cur, const DiagnosticState& ini,
                    const Particles& p, bool is_cluster,
                    double& dE, double& dP, double& dL);

void print_timebin_table(const Particles& p, const std::vector<double>& levels,
                         double dE, double dP, double dL,
                         double t_myr, int step, bool is_cluster);

void print_final(const DiagnosticState& cur, const DiagnosticState& ini,
                 const Particles& p, const std::vector<double>& levels,
                 bool is_cluster, double wall_sec, int steps,
                 const char* const* names);
