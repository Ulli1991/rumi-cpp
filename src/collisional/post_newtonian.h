#pragma once

#include "logh.h"

// Post-Newtonian accelerations up to PN3.5 for pairs inside regularized subsystems
// and global PN1.0 for compact objects outside subsystems

// Compute PN acceleration for a pair (in code units: pc, M_sun, km/s)
// r[3]: relative position (j - i)
// v[3]: relative velocity (j - i)
// a_pn[3]: output PN acceleration on particle i due to j
// pn_order: bitmask (1=PN1, 2=PN2, 4=PN2.5, 8=PN3, 16=PN3.5)
void pn_acceleration(const double r[3], const double v[3],
                     double m1, double m2, double G, double c_code,
                     double a_pn[3], int pn_order);

// Auxiliary velocity kick for explicit velocity-dependent PN in leapfrog
// (BIFROST Sec. 3.1): extends phase space with auxiliary velocity w
void pn_auxiliary_kick(LogHState& state, double ds, double G, double c_code, int pn_order);

// Spin precession rate for a spinning body in a binary
// S_i: spin vector of body i, S_j: spin vector of body j
// Omega_prec: output precession angular velocity
void spin_precession(const double r[3], const double v[3],
                     double m1, double m2, const double S1[3], const double S2[3],
                     double G, double c_code,
                     double Omega_prec_1[3], double Omega_prec_2[3]);

// Evolve spin vectors by dt using precession equations
void evolve_spins(double S1[3], double S2[3],
                  const double Omega_prec_1[3], const double Omega_prec_2[3],
                  double dt);
