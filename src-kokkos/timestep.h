#pragma once

#include "particles.h"
#include <vector>

std::vector<double> build_levels(double dt_max, double dt_min);

void assign_levels(Particles& p, const std::vector<double>& levels);

// Assign levels for a subset of particles [start, start+count)
void assign_levels_subset(Particles& p, int start, int count, const std::vector<double>& levels);

int find_coarsest_active(const Particles& p);
