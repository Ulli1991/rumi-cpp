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
    double E_sd = 0.0;        // SDAR: physical energy changed by the slow-down (kappa jumps, Wang+2020 Eq. 24)
    double E_tidal = 0.0;     // external tidal field: minus the kinetic energy its kicks put in (gravity/tidal_field.h)

    // Momentum and angular momentum (about the origin) carried off by the same events: escapers, merger
    // mass loss and kicks, orbital L of merged pairs (remnant spin is not tracked), SSE winds and kicks
    double P_sink[3] = {0.0, 0.0, 0.0};
    double L_sink[3] = {0.0, 0.0, 0.0};

    double total_sink() const { return E_escape + E_gw + E_merge + E_sse + E_pn + E_sd + E_tidal; }
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

// NBODY_EATTR diagnostics: per print interval, the Newtonian energy change of LogH subsystems that is not
// booked (GBS error of non-PN subsystems), with the worst single subsystem advance
struct EnergyAttribution {
    bool on = false;
    double logh = 0.0;          // sum of E_post - E_pre over unbooked LogH advances
    double logh_abs = 0.0;      // sum of |E_post - E_pre|
    double worst = 0.0;         // largest |E_post - E_pre| of one advance (signed value)
    int worst_n = 0, worst_pid0 = -1, worst_pid1 = -1, worst_type = -1;
    double worst_a = 0.0, worst_e = 0.0, worst_dt = 0.0;
    double worst_m[4] = {0.0, 0.0, 0.0, 0.0};   // masses of its first 4 members (Msun)
    int n_logh = 0;
    // tidal kick estimate mu |f| |v| dt of 2-body bound subsystems (subsystem_timesteps), max over the interval
    double dt_cap = 1e30;
    double tid_max = 0.0, tid_a = 0.0, tid_e = 0.0, tid_gamma = 0.0, tid_dt = 0.0, tid_P = 0.0, tid_m1 = 0.0, tid_m2 = 0.0;
    int tid_pid0 = -1, tid_pid1 = -1;
    void reset() { bool o = on; double c = dt_cap; *this = EnergyAttribution(); on = o; dt_cap = c; }
};
extern EnergyAttribution g_eattr;
// Number of the top-level step in progress (main loop; SUBSYS_ERR lines)
extern int g_step_in_progress;

DiagnosticState compute_diagnostics(const Particles& p, double G, double soft2);

// Tiled GPU potential energy computation (shared-memory tiled, O(N²))
// Computes PE for i in [i_start, i_start+i_count) against all N particles (j > i only)
// PE of all pairs split over the MPI ranks (area-balanced rows, rank-order sum: the same bits on every rank)
double compute_potential_energy_split(const Particles& p, double G, double soft2);
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
