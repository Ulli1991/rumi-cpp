#pragma once

#include "particles.h"
#include <string>

// 5-body outer solar system: Sun + Jupiter, Saturn, Uranus, Neptune
void init_outer_planets(Particles& p);

// Plummer sphere
// mass_distribution: "equal", "kroupa", "kroupa_evolved"
void init_plummer(Particles& p, int N, double M_total_msun, double r_half_pc, int seed,
                  const std::string& mass_distribution = "equal",
                  double age_gyr = 1.0, double metallicity = 0.02);
