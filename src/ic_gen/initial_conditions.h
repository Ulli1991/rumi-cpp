#pragma once

#include "particles.h"
#include <string>

void init_outer_planets(Particles& p);

void init_plummer(Particles& p, int N, double M_total_msun, double r_half_pc, int seed,
                  const std::string& mass_distribution = "equal",
                  double age_gyr = 1.0, double metallicity = 0.02);

// Isolated binary test case (2 particles): configurable orbital elements
// a_pc: semi-major axis in parsecs, e: eccentricity, m1/m2: masses in M_sun
void init_binary_test(Particles& p, double a_pc, double e,
                      double m1_msun, double m2_msun);
