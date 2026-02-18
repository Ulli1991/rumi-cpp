#pragma once

#include "particles.h"
#include "subsystem.h"
#include "diagnostics.h"

// Merger detection and execution for collisional dynamics

enum class MergerType {
    GW_COALESCENCE,      // Peters timescale < dt
    ISCO,                // r < 6 G M / c^2
    TIDAL_DISRUPTION,    // r_peri < R_td
    STELLAR_COLLISION,   // r < R1 + R2
    NONE
};

struct MergerEvent {
    MergerType type;
    int pid1, pid2;          // particle IDs
    double m_remnant;        // remnant mass
    double v_kick[3];        // GW kick velocity
    double spin_remnant[3];  // remnant spin vector
    double E_radiated;       // energy radiated (GW, dissipation)
};

// Check for merger conditions in all active subsystems
// Returns list of merger events to process
std::vector<MergerEvent> detect_mergers(const SubsystemManager& mgr,
                                         const Particles& p,
                                         double G, double c_code, double dt);

// Execute a merger: replace two particles with one remnant
// Updates Particles, SubsystemManager, and EnergyBookkeeping
void execute_merger(const MergerEvent& event, Particles& p,
                    SubsystemManager& mgr, EnergyBookkeeping& books, double G);

// BH-BH merger remnant properties (Zlochower+ 2015 fitting formulas)
void bh_bh_remnant(double m1, double m2, const double chi1[3], const double chi2[3],
                   const double L_orb[3], double& m_rem, double chi_rem[3],
                   double v_kick[3], double G, double c_code);

// Peters (1964) GW coalescence timescale
double gw_coalescence_time(double a, double e, double m1, double m2,
                           double G, double c_code);

// Tidal disruption radius
double tidal_disruption_radius(double m_star, double R_star, double M_bh);

// Escaper removal: remove particles beyond escape radius
// Returns number of particles removed
int remove_escapers(Particles& p, double G, double R_escape,
                    SubsystemManager& sm, EnergyBookkeeping& books);
