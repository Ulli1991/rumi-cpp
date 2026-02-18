#pragma once

#include <cmath>

// Maximum bodies in a LogH subsystem (MSTAR handles larger)
constexpr int LOGH_MAX_BODIES = 8;

// LogH regularized state: time-transformed leapfrog (TTL)
// Omega = T + B  where T=kinetic, B=-E₀ (positive for bound systems)
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

// Compute total energy (kinetic + potential) for error checking
double logh_total_energy(const LogHState& state, double G);
