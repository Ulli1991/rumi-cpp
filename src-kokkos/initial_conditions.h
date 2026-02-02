#pragma once

#include "particles.h"

void init_outer_planets(Particles& p);

void init_plummer(Particles& p, int N, double M_total_msun, double r_half_pc, int seed);
