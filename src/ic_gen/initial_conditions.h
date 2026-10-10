#pragma once

#include "particles.h"
#include "sse.h"
#include <string>

void init_outer_planets(Particles& p);

void init_plummer(Particles& p, int N, double M_total_msun, double r_half_pc, int seed,
                  const std::string& mass_distribution = "equal",
                  double age_gyr = 1.0, double metallicity = 0.02,
                  bool remove_bh = false, double binary_frac = 0.0,
                  double bin_a_min_au = 0.01, double bin_a_max_au = 100.0,
                  bool add_imbh = false, double imbh_m1 = 500.0, double imbh_m2 = 500.0,
                  double imbh_a_pc = 1e-4, double imbh_e = 0.5, double imbh_spin = 0.0,
                  double r_break_au = 0.0,
                  const SSEConfig* sse_cfg = nullptr);

// Isolated binary test case (2 particles): configurable orbital elements
// a_pc: semi-major axis in parsecs, e: eccentricity, m1/m2: masses in M_sun
void init_binary_test(Particles& p, double a_pc, double e,
                      double m1_msun, double m2_msun);
void init_binary_test_peri(Particles& p, double a_pc, double e,
                           double m1_msun, double m2_msun);

// Hierarchical triple test case (3 particles):
// Inner binary (m1, m2) with semi-major axis a_in, eccentricity e_in in x-y plane.
// Outer body m3 orbits inner COM with semi-major axis a_out, eccentricity e_out,
// tilted by inclination i_deg (degrees) for Kozai-Lidov oscillations.
void init_triple_test(Particles& p,
                      double a_in_pc, double e_in, double m1_msun, double m2_msun,
                      double a_out_pc, double e_out, double m3_msun,
                      double i_deg, bool outer_first = false);
