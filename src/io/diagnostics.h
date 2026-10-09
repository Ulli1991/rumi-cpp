#pragma once

#include "particles.h"
#include <vector>

struct DiagnosticState {
    double KE, PE, E_total;
    double Px, Py, Pz;
    double Lx, Ly, Lz;
    double E_pn1 = 0.0;   // PN1 energy correction
    double E_pn2 = 0.0;   // PN2 energy correction
};

// Energy bookkeeping for collisional mode (tracks energy sinks)
// Conservation check: |E_total + Σ(sinks) - E_initial| / |E_initial|
struct EnergyBookkeeping {
    double E_escape = 0.0;    // energy carried away by escapers
    double E_gw = 0.0;        // gravitational wave losses (PN2.5)
    double E_merge = 0.0;     // merger dissipation
    double E_sse = 0.0;       // stellar evolution mass loss
    double E_pn = 0.0;        // Newtonian energy changed by PN forces in subsystems (diagnostics stay Newtonian)

    // Momentum and angular momentum (about the origin) carried off by the same events: escapers, merger
    // mass loss and kicks, orbital L of merged pairs (remnant spin is not tracked), SSE winds and kicks
    double P_sink[3] = {0.0, 0.0, 0.0};
    double L_sink[3] = {0.0, 0.0, 0.0};

    double total_sink() const { return E_escape + E_gw + E_merge + E_sse + E_pn; }
    void reset() { *this = EnergyBookkeeping(); }

    // Book what a body of mass m at r with velocity v takes along (sign -1: what it brings)
    void sink_PL(double m, const double r[3], const double v[3], double sign = 1.0) {
        for (int k = 0; k < 3; k++) P_sink[k] += sign * m * v[k];
        L_sink[0] += sign * m * (r[1] * v[2] - r[2] * v[1]);
        L_sink[1] += sign * m * (r[2] * v[0] - r[0] * v[2]);
        L_sink[2] += sign * m * (r[0] * v[1] - r[1] * v[0]);
    }
};

// Global energy bookkeeping instance
extern EnergyBookkeeping g_energy_books;

DiagnosticState compute_diagnostics(const Particles& p, double G, double soft2);

// Tiled GPU potential energy computation (shared-memory tiled, O(N²))
// Computes PE for i in [i_start, i_start+i_count) against all N particles (j > i only)
double compute_potential_energy_tiled(const Particles& p, double G, double soft2,
                                      int i_start, int i_count);

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

#ifdef ENABLE_COLLISIONAL
void print_subsystem_diagnostics(const Particles& p, double G);
#endif
