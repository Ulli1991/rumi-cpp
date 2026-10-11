#pragma once

#include "particles.h"

// PN_ORDER bitmask: 1 = PN1.0, 2 = PN2.0, 4 = PN2.5, 8 = PN3.0, 16 = PN3.5, 32 = spin couplings in 2-body LogH
// subsystems (spin-orbit, spin-spin, spin-quadrupole accelerations and spin precession; BIFROST sec. 3.7)
constexpr int PN_SPIN = 32;

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

// Both bodies at once: a_i = eih_pn_acceleration(i from j), a_j = eih_pn_acceleration(j from i), bit-identical
// to the two calls, with the geometry and the pair-symmetric PN2.5-3.5 coefficients computed once
void eih_pn_acceleration_pair(const double x_i[3], const double v_i[3], double m_i,
                              const double x_j[3], const double v_j[3], double m_j,
                              double G, double c_code, double a_i[3], double a_j[3], int pn_order);

// ============================================================================
// PN conserved energy corrections (Poisson & Will 2014, Section 9.3)
// ============================================================================
// E_conserved = T + U + E_pn1 + E_pn2 (+ cumulative GW losses for PN2.5)

// PN1 energy correction for an N-body system
// pos[i*3+k], vel[i*3+k], mass[i] for i=0..n-1
double pn1_energy(const double* pos, const double* vel, const double* mass,
                  int n, double G, double c_code);

// Spin-dependent PN binary dynamics (BIFROST sec. 3.7; Kidder 1995 eqs. 2.2c-d, 2.4, 2.9; Kerr quadrupole of
// Racine 2008). Relative coordinates x = x1 - x2, v = v1 - v2; S1, S2 physical spins (chi G m^2 / c).
// Precession vectors: dS_i/dt = Omega_i × S_i (spin-orbit + spin-spin + quadrupole)
void spin_precession(const double x[3], const double v[3],
                     double m1, double m2, const double S1[3], const double S2[3],
                     double G, double c_code,
                     double Omega1[3], double Omega2[3]);

// Relative acceleration a1 - a2 from spin-orbit (1.5PN), spin-spin and spin-quadrupole (2PN) couplings
void spin_pn_acceleration(const double x[3], const double v[3],
                          double m1, double m2, const double S1[3], const double S2[3],
                          double G, double c_code, double a_rel[3]);

// Conserved total angular momentum J = L_N (+ L_PN1 if pn1) + L_SO + S1 + S2 of a binary (Kidder eq. 2.9)
void binary_total_angular_momentum(const double x[3], const double v[3],
                                   double m1, double m2, const double S1[3], const double S2[3],
                                   double G, double c_code, bool pn1, double J[3]);

// Rotate S about Omega by the angle |Omega| dt (exact solution of dS/dt = Omega × S for constant Omega)
void rotate_about(double S[3], const double Omega[3], double dt);

// Global PN1.0 velocity kick for compact objects (mass >= m_pn) outside subsystems
void global_pn1_kick(Particles& p, int start, int count,
                     double G, double c_code, double m_pn, double dt);
