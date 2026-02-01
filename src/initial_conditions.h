#pragma once

#include "particles.h"

// 5-body outer solar system: Sun + Jupiter, Saturn, Uranus, Neptune
void init_outer_planets(Particles& p);

// Plummer sphere with equal-mass particles
void init_plummer(Particles& p, int N, double M_total_msun, double r_half_pc, int seed);
