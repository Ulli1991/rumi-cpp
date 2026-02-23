#pragma once

#include "particles.h"
#include "diagnostics.h"
#include "sse.h"

// Runtime stellar evolution: advance stellar properties by dt
// Updates mass, stellar_type, stellar_radius, luminosity, core_mass
// Accounts for mass loss in EnergyBookkeeping
void advance_stellar_evolution(Particles& p, SSEContext& sse, double dt_myr,
                                EnergyBookkeeping& books, double G);
