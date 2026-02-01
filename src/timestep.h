#pragma once

#include "particles.h"
#include <vector>

// Build power-of-2 timestep hierarchy: [dt_max, dt_max/2, ..., >= dt_min]
std::vector<double> build_levels(double dt_max, double dt_min);

// Assign each particle to coarsest level where dt_level <= dt_frost
void assign_levels(Particles& p, const std::vector<double>& levels);

// Find coarsest (smallest index) active level
int find_coarsest_active(const Particles& p);
