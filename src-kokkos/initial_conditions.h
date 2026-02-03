#pragma once

#include "particles.h"
#include <string>

void init_outer_planets(Particles& p);

void init_plummer(Particles& p, int N, double M_total_msun, double r_half_pc, int seed,
                  const std::string& mass_distribution = "equal",
                  double age_gyr = 1.0, double metallicity = 0.02);
