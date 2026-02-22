#pragma once

#include "particles.h"

// Post-Newtonian accelerations up to PN3.5 for pairs inside regularized subsystems
// and global PN1.0 for compact objects outside subsystems

// ============================================================================
// EIH (Einstein-Infeld-Hoffmann) N-body PN acceleration
// ============================================================================
// Computes the PN correction to the acceleration of body A due to body B.
// Uses individual particle positions/velocities/masses directly — no
// relative-coordinate decomposition needed. (Poisson & Will 2014 / BIFROST)
//
// a_pn_A[3]: output PN acceleration on body A (added to Newtonian)
// pn_order: bitmask (1=PN1, 2=PN2, 4=PN2.5)
void eih_pn_acceleration(const double x_A[3], const double v_A[3], double m_A,
                         const double x_B[3], const double v_B[3], double m_B,
                         double G, double c_code, double a_pn_A[3], int pn_order);

// ============================================================================
// PN conserved energy corrections (Poisson & Will 2014, Section 9.3)
// ============================================================================
// E_conserved = T + U + E_pn1 + E_pn2 (+ cumulative GW losses for PN2.5)

// PN1 energy correction for an N-body system
// pos[i*3+k], vel[i*3+k], mass[i] for i=0..n-1
double pn1_energy(const double* pos, const double* vel, const double* mass,
                  int n, double G, double c_code);

// PN2.5 instantaneous GW power (dissipative — track cumulative loss)
double pn25_power(const double* pos, const double* vel, const double* mass,
                  int n, double G, double c_code);

// ============================================================================
// Legacy relative-coordinate PN acceleration (kept for regression testing)
// ============================================================================
// r[3]: relative position (j - i)
// v[3]: relative velocity (j - i)
// a_pn[3]: output relative PN acceleration (two-body EOM for r = r_j - r_i)
//          Individual: a_i = -(m_j/M)*a_pn, a_j = +(m_i/M)*a_pn
// pn_order: bitmask (1=PN1, 2=PN2, 4=PN2.5)
void pn_acceleration_relative(const double r[3], const double v[3],
                              double m1, double m2, double G, double c_code,
                              double a_pn[3], int pn_order);

// Spin precession rate for a spinning body in a binary
void spin_precession(const double r[3], const double v[3],
                     double m1, double m2, const double S1[3], const double S2[3],
                     double G, double c_code,
                     double Omega_prec_1[3], double Omega_prec_2[3]);

// Evolve spin vectors by dt using precession equations
void evolve_spins(double S1[3], double S2[3],
                  const double Omega_prec_1[3], const double Omega_prec_2[3],
                  double dt);

// Global PN1.0 velocity kick for compact objects (mass >= m_pn) outside subsystems
void global_pn1_kick(Particles& p, int start, int count,
                     double G, double c_code, double m_pn, double dt);
