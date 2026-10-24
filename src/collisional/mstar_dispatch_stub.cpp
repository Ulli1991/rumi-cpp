// SPDX-License-Identifier: BSD-3-Clause
// RUMI++ core: MSTAR dispatch without the module (ENABLE_MSTAR=OFF). See mstar_dispatch.h.
#include "mstar_dispatch.h"

bool mstar_available() { return false; }
int mstar_crossover_n() { return 1 << 30; }
bool mstar_integrate(MstarGroup& g) {
    g.ok = false;
    return false;
}
