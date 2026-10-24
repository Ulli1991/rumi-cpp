#pragma once

#include "particles.h"
#include "diagnostics.h"
#include "sse.h"

// Runtime stellar evolution: advance stellar properties by dt
// Updates mass, stellar_type, stellar_radius, luminosity, core_mass
// Accounts for mass loss in EnergyBookkeeping
void advance_stellar_evolution(Particles& p, SSEContext& sse, double dt_myr,
                                EnergyBookkeeping& books, double G);

// Potential-energy change of the mass update of the stars touched (indices), d = m' - m, mbar_t = (m + m') / 2, with
// p.mass already holding m' (exact, unsoftened; collective under MPI). The E_sse books use it; so does the library.
double potential_change(const Particles& p, const std::vector<int>& touched, const std::vector<double>& d,
                        const std::vector<double>& mbar_t, double G);
