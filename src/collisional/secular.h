#pragma once

#include "subsystem.h"

// Secular (orbit-averaged) evolution for tight binaries and stable triples
// Peters (1964) GW dissipation + Kozai-Lidov oscillations

struct SecularBinary {
    double a, e;              // semi-major axis, eccentricity
    double omega;             // argument of periapsis
    double inc, Omega;        // inclination, longitude of ascending node
    double m1, m2;            // component masses
    int pid1, pid2;           // particle IDs

    // Peters (1964) rates
    double da_dt;             // da/dt from GW emission
    double de_dt;             // de/dt from GW emission
    double domega_dt;         // periapsis advance (PN1.0 + PN2.0)
};

struct SecularTriple {
    SecularBinary inner;
    double a_out, e_out;      // outer orbital elements
    double omega_out, inc_out, Omega_out;
    double m3;                // outer companion mass
    int pid3;                 // outer companion particle ID
    double cos_i_mut;         // mutual inclination cosine
};

// Evolve a secular binary by dt; returns the number of internal steps
int secular_evolve_binary(SecularBinary& bin, double dt, double G, double c_code, int pn_order);

// Evolve a secular triple by dt (Kozai-Lidov + GW dissipation)
void secular_evolve_triple(SecularTriple& tri, double dt, double G, double c_code, int pn_order);

// Check if a subsystem should transition to secular mode
bool should_be_secular(const Subsystem& sub, double dt_step, int n_orb_threshold);

// Reconstruct (r, v) from secular elements (relative coordinates)
void secular_to_cartesian(const SecularBinary& bin, double G,
                          double mean_anomaly, double r[3], double v[3]);

// Full secular integration of a binary: Cartesian -> elements -> secular evolve -> advance M -> Cartesian
// Works in absolute coordinates (handles CM frame internally); returns the number of internal steps
int integrate_secular_binary(
    double pos1[3], double vel1[3], double m1,
    double pos2[3], double vel2[3], double m2,
    double G, double c_code, int pn_order, double dt);
