#pragma once

#include "particles.h"

struct EnergyBookkeeping;

// External tidal field of a host galaxy (HARD_PROBLEMS.md G3), as in PeTar's Galpy interface (Wang et al. 2020,
// src/petar.hpp: externalForce, correctPtclVelCM): particle positions and velocities stay relative to a reference
// point (the cluster's guiding centre) that moves through the galactic potential; each particle gets the
// differential acceleration a_G(X + x) - a_G(X), the reference V += a_G(X). Applied as a half kick before and after
// every top-level step (kick-drift-kick of the external Hamiltonian around the internal integration; the tidal
// timescale is ~100 Myr against 1e-3 Myr steps). Relative coordinates keep the precision of close binaries, which
// galactocentric positions (|X| ~ kpc) would destroy. The energy, momentum and angular momentum each kick puts into
// the cluster are booked (EnergyBookkeeping::E_tidal, P_sink, L_sink), so the energy check still measures the
// integration error alone.
struct TidalField {
    enum Type { NONE = 0, POINT_MASS = 1, LOG_HALO = 2 };
    Type type = NONE;
    double M_gal = 0.0;   // POINT_MASS: galaxy mass (Msun)
    double V_c = 0.0;     // LOG_HALO: circular velocity (km/s), Phi = V_c^2 / 2 ln(R_c^2 + r^2)
    double R_c = 0.0;     // LOG_HALO core radius (pc)
    double X[3] = {0, 0, 0}, V[3] = {0, 0, 0};   // reference point (pc from the galaxy centre, km/s)
    bool on() const { return type != NONE; }
    void acc(const double x[3], double G, double a[3]) const;
    double v_circ(double R, double G) const;
    // Circular orbit at galactocentric radius R in the x-y plane, starting on the +x axis
    void init_circular(double R, double G);
};

extern TidalField g_tidal;

// Half kick of length hdt to every particle of p (relative velocities) and to the reference point; books the
// energy, momentum and angular momentum put into the particles. All ranks of a replicated run do the same.
void tidal_kick(Particles& p, double hdt, double G, EnergyBookkeeping& books);
// The reference point drifts with its velocity over dt (the particles' frame moves with it: nothing else changes)
void tidal_drift_reference(double dt);
