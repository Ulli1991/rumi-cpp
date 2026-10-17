#pragma once

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

// Genuine N-body EIH 1PN acceleration (Einstein, Infeld & Hoffmann 1938; Poisson & Will 2014 eq. 9.127), the PN1
// part only, all bodies at once (a[3n] is overwritten): the two-body terms of every pair plus the three-body terms
// (a third body's potential in the A-B interaction). From the EIH Lagrangian (Landau & Lifshitz sec. 106); for n = 2
// the same as the pair formula (eih_pn_acceleration with pn_order 1) up to rounding.
void eih1_nbody_acceleration(const double* pos, const double* vel, const double* mass, int n, double G,
                             double c_code, double* a);

// ============================================================================
// The PN accelerations of a LogH subsystem (eih1_nbody_acceleration and eih_pn_acceleration_pair over all pairs) at
// several velocities and the same positions: a leapfrog step evaluates them 5 times at two position sets (K_aux and
// K_phys change only v, w). PnGeom holds what depends only on the masses (pn_geom_set_masses: total masses, nu, the
// nu polynomials of PN3.0 / 3.5, 1/c^n) and only on the positions (pn_geom_set_positions: separations, 1/r, unit
// vectors, G m / r, the prefactors, the EIH potentials and fields), computed once; eih1_nbody_acceleration_geom and
// eih_pn_pairs_geom then do only the velocity-dependent part.
// Bit-identical to eih1_nbody_acceleration and eih_pn_acceleration_pair: each cached value is computed with the
// operations of those functions and is one that they round on its own (sqrt, quotient, a product that only feeds
// products, a complete sum), never a product that they fuse into an addition (FMA contraction), and the
// velocity-dependent code is theirs statement for statement, with loads of the cached values in place of their
// computation: the compiler's FMA choices (GCC: in statement order; clang -ffp-contract=on: within an expression)
// come out the same. Not so under clang -ffp-contract=fast (the HIP default), where already the pair and the
// single-body functions differ in the last bit. Checked bitwise on random configurations of 2-8 bodies over all
// orders (tests/pn_geom_test.cpp, ctest pn_geom_bitwise; g++ 13 and 15, nvcc_wrapper, clang 18 -ffp-contract=on)
// and on every evaluation of the run_case tests (NBODY_CHECK_EIH=1). A structure-of-arrays variant vectorised over
// the pairs was faster but not bit-identical under g++ 15 (the vectoriser reorders the statements, and with them
// which product of a sum GCC fuses).
// ============================================================================
constexpr int PN_GEOM_MAX_BODIES = 8;
constexpr int PN_GEOM_MAX_PAIRS = PN_GEOM_MAX_BODIES * (PN_GEOM_MAX_BODIES - 1) / 2;
constexpr int PN_GEOM_MAX_DIR = 2 * PN_GEOM_MAX_PAIRS;   // bodies of the pairs (ordered pairs)

struct PnBodyGeom {          // body A of a pair from body B (eih_pn_acceleration with A, B)
    double n[3];             // (x_A - x_B) * (1/r)
    double Gm_A, Gm_B;       // G m_A, G m_B
    double Gm_A_r, Gm_B_r;   // G m_A / r, G m_B / r (as G m * (1/r))
    double pf1, pf2;         // PN1, PN2 prefactors G m_B (1/r)^2 / c^2, / c^4
    double fA;               // m_B / M: body A's share of the PN2.5-3.5 relative acceleration
};

struct PnPairGeom {          // pair i < j
    int i, j;
    // masses
    double M, Gm, nu;        // m_i + m_j, G M, m_i m_j / M^2
    double f25;              // 8/5 nu
    double P3A[10], P3B[6];  // nu polynomials of the PN3.0 coefficients A, B
    double P35A[5], P35B[5]; // nu polynomials of the PN3.5 coefficients A, B
    // positions
    double rij[3];           // x_i - x_j
    double inv_r;
    double u, uu, uuu;       // G M / r and its square, cube
    double c25, c35;         // (1/c^5) (8/5) nu u, (1/c^7) nu u
    double g;                // -G M / r^2
    PnBodyGeom body[2];      // [0]: i from j, [1]: j from i
};

struct PnGeom {
    int n = 0;
    bool eih_nbody = false;  // eih1_nbody_acceleration part set up (positions: phi, g, nAB, ...)
    int pair_order = 0;      // orders of the pair formula (0: no pair part)
    double G = 0.0;
    double inv_c2 = 0.0, inv_c4 = 0.0, inv_c5 = 0.0, inv_c6 = 0.0, inv_c7 = 0.0;
    double Gm[PN_GEOM_MAX_BODIES];
    // eih1_nbody_acceleration: potentials phi_A, fields g_A and per ordered pair (A, B) n_AB, G m_B / r^2,
    // 3.5 G m_B / r, (x_B - x_A).g_B
    double phi[PN_GEOM_MAX_BODIES], g[PN_GEOM_MAX_BODIES][3];
    double nAB[PN_GEOM_MAX_BODIES][PN_GEOM_MAX_BODIES][3];
    double Gm_r2[PN_GEOM_MAX_BODIES][PN_GEOM_MAX_BODIES];
    double c35_r[PN_GEOM_MAX_BODIES][PN_GEOM_MAX_BODIES];
    double xg[PN_GEOM_MAX_BODIES][PN_GEOM_MAX_BODIES];
    int n_pairs = 0;
    PnPairGeom pair[PN_GEOM_MAX_PAIRS];   // (0,1), (0,2), ..., (1,2), ...: the order of the loop i < j
};

// Masses and constants (n <= PN_GEOM_MAX_BODIES; eih_nbody: the PN1 part by eih1_nbody_acceleration_geom; pair_order:
// the orders of eih_pn_pairs_geom, 0 for none). Then pn_geom_set_positions at every new position set.
void pn_geom_set_masses(const double* mass, int n, double G, double c_code, bool eih_nbody, int pair_order,
                        PnGeom& geom);
void pn_geom_set_positions(const double* pos, PnGeom& geom);
// = eih1_nbody_acceleration(pos, vel, mass, n, G, c_code, a) at the positions and masses of geom (a[3n] overwritten)
void eih1_nbody_acceleration_geom(const PnGeom& geom, const double* vel, double* a);
// = eih_pn_acceleration_pair(x_i, v_i, m_i, x_j, v_j, m_j, G, c_code, a_i, a_j, pn_order) of every pair q of geom
// (bodies geom.pair[q].i, .j): a[6 q + k] = a_i[k], a[6 q + 3 + k] = a_j[k]
void eih_pn_pairs_geom(const PnGeom& geom, const double* vel, int pn_order, double* a);

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
