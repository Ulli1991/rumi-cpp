#pragma once

#include <cmath>

// Maximum bodies in a LogH subsystem (MSTAR handles larger)
constexpr int LOGH_MAX_BODIES = 8;

// LogH regularized state: time-transformed leapfrog (TTL)
// Follows BIFROST (Rantala+ 2023) Section 3.3 / FROST Section 2.3.
// Omega = T + B  where T=kinetic, B=-E₀ (positive for bound systems).
// BIFROST convention: ds/dt = Omega, all EOMs have 1/Omega factor.
// B is constant for Newtonian dynamics; evolves when PN forces are present
// (BIFROST Eq. 4): dB/ds = -Σ m_i v_i · g_i / Omega.
struct LogHState {
    double t;                              // physical time
    double B;                              // binding energy
    int n_bodies;
    double pos[LOGH_MAX_BODIES * 3];
    double vel[LOGH_MAX_BODIES * 3];
    double mass[LOGH_MAX_BODIES];
    double ext_accel[LOGH_MAX_BODIES * 3]; // external perturbation

    // Auxiliary velocities for velocity-dependent forces (PN)
    double aux_vel[LOGH_MAX_BODIES * 3];
    bool use_aux_vel;                       // whether PN is active

    // Spin vectors for BH binaries (dimensionless chi vectors; S = chi G m^2 / c) and their auxiliary
    // copies Z (Rantala+ 2017): S precesses with Omega(w, Z), Z with Omega(v, S)
    double spin1[3], spin2[3];
    double aux_spin1[3], aux_spin2[3];
    bool has_spins;                          // true for 2-body BH binaries with spins
    double c_code;                           // speed of light (normalized units)

};

// Compute internal accelerations for LogH system
void logh_internal_accel(const LogHState& state, double G,
                         double accel[LOGH_MAX_BODIES * 3]);

// Compute kinetic energy T of the system
double logh_kinetic_energy(const LogHState& state);

// Compute potential energy W of the system
double logh_potential_energy(const LogHState& state, double G);

// Single leapfrog step in regularized time s
void logh_leapfrog_step(LogHState& state, double ds, double G);

// Single leapfrog step with PN corrections using auxiliary velocity technique
// (Hellstrom & Mikkola 2010): v gets g(w), w gets g(v)
void logh_leapfrog_step_pn(LogHState& state, double ds, double G,
                            double c_code, int pn_order);

// Compute total energy (kinetic + potential) for error checking
double logh_total_energy(const LogHState& state, double G);
