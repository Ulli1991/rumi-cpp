// Ketju-integrator (MSTAR)
// Copyright (C) 2020-2022 Matias Mannerkoski and contributors.
// Licensed under the GPLv3 license.
// See LICENSE.md or <https://www.gnu.org/licenses/> for details.
// Modified by Ulrich Steinwandel, 2026: rewritten in C++ with Kokkos support (GPU).
// Copyright (C) 2026 Ulrich Steinwandel. Licensed under the GPLv3 license.
//
// BH mergers of the MSTAR port (from reference/ketju/src/bh_mergers.c and the PN centre-of-mass velocity of pn.c):
// remnant mass, spin and recoil from the Zlochower & Lousto (2015) fits, the PN-corrected c.o.m. velocity of the
// pair (de Andrade et al. 2001 to 3PN with the 2.5PN momentum term, the leading spin term of Keppel et al. 2009).
// Separations are Cartesian (KETJU uses its tree coordinates where the pair is close in the tree).

#pragma once

#include <array>
#include <vector>

namespace rumi_mstar {

using Vec3 = std::array<double, 3>;

struct MergerInput {
    double G, c;
    double m1, m2;
    Vec3 x1, x2, v1, v2, S1, S2;
};
struct RemnantProperties {
    double m;
    Vec3 vkick, S;
};

// ketju_calculate_bh_merger_remnant_properties
RemnantProperties bh_merger_remnant(const MergerInput& in);

// ketju_binary_PN_CoM_velocity for PN particles i, j of the state (PN flags as KETJU's bits)
Vec3 binary_pn_com_velocity(const std::vector<double>& mass, const std::vector<double>& pos,
                            const std::vector<double>& vel, const std::vector<double>& spin, int i, int j, double G,
                            double c, unsigned flags);

class MstarKokkos;
// ketju_do_bh_mergers on the host state of M (merger condition: separation < bh_merger_distance_in_Rs 2 G (m1 + m2)
// / c^2); returns the number of mergers; arrays shift as in KETJU, extra_index moves merged-away entries to the end
int do_bh_mergers(MstarKokkos& M);
bool bh_merger_due(const MstarKokkos& M);

}  // namespace rumi_mstar
