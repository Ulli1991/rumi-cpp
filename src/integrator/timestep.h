#pragma once

#include "particles.h"
#include <vector>

std::vector<double> build_levels(double dt_max, double dt_min);

void assign_levels(Particles& p, const std::vector<double>& levels);
