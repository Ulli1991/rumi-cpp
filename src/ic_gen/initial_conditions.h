#pragma once

#include "particles.h"
#include "sse.h"
#include <string>

void init_outer_planets(Particles& p);

// r_half_pc <= 0: r_h from scic_mass_size_r_half(M) of the generated masses. r_break_au > 0: SCIC artificial-binary
// avoidance (BIFROST App. A) on singles and binary centres of mass.
void init_plummer(Particles& p, int N, double M_total_msun, double r_half_pc, int seed,
                  const std::string& mass_distribution = "equal",
                  double age_gyr = 1.0, double metallicity = 0.02,
                  bool remove_bh = false, double binary_frac = 0.0,
                  double bin_a_min_au = 0.01, double bin_a_max_au = 100.0,
                  bool add_imbh = false, double imbh_m1 = 500.0, double imbh_m2 = 500.0,
                  double imbh_a_pc = 1e-4, double imbh_e = 0.5, double imbh_spin = 0.0,
                  double r_break_au = 0.0,
                  const SSEConfig* sse_cfg = nullptr,
                  // Primordial binary population: "bifrost" (BIFROST App. A: flat q above 5 M_sun, random
                  // pairing below) or "mass_rank" (heaviest with next heaviest); pericentre condition
                  // r_p = a(1-e) > contact distance (FROST-CLUSTERS II Sec. 2.3)
                  // Binary centre of mass: the primary's own Plummer f(E) draw (SCIC/mcluster: a binary is one
                  // point particle until its members are placed)
                  const std::string& binary_pop = "bifrost", bool binary_peri_check = true);

// Half-mass radius [pc] of a star cluster of mass M [M_sun] from the SCIC mass-size relations
// (BIFROST, Rantala et al. 2023, App. A, eqs. A2-A3)
double scic_mass_size_r_half(double M_msun);

// Isolated binary test case (2 particles): configurable orbital elements
// a_pc: semi-major axis in parsecs, e: eccentricity, m1/m2: masses in M_sun
void init_binary_test(Particles& p, double a_pc, double e,
                      double m1_msun, double m2_msun, double rot_deg = 0.0);
void init_binary_test_peri(Particles& p, double a_pc, double e,
                           double m1_msun, double m2_msun);

// Hierarchical triple test case (3 particles):
// Inner binary (m1, m2) with semi-major axis a_in, eccentricity e_in in x-y plane.
// Outer body m3 orbits inner COM with semi-major axis a_out, eccentricity e_out,
// tilted by inclination i_deg (degrees) for Kozai-Lidov oscillations.
void init_particle_file(Particles& p, const std::string& path);
void init_quad22_test(Particles& p, double m1, double m2, double a1, double e1, double m3, double m4, double a2,
                      double e2, double a_out, double e_out, double incl_deg);
void init_triple_test(Particles& p,
                      double a_in_pc, double e_in, double m1_msun, double m2_msun,
                      double a_out_pc, double e_out, double m3_msun,
                      double i_deg, bool outer_first = false, double in_rot_deg = 0.0, double in_start_q = 5.0,
                      double out_start_q = 5.0);
