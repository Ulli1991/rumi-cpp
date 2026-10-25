// SPDX-License-Identifier: BSD-3-Clause
// RUMI++ core: Euclidean minimum spanning tree of a point set (dense Prim, O(N^2)), used by the KETJU reference
// cases (tests/mstar_reference). (The MSTAR integrator itself, with its own tree code, is the optional GPLv3
// module src/mstar.)
#pragma once

// Tree from vertex 0: order[k] the k-th vertex added, pairs[2e], pairs[2e+1] the (parent, child) of edge e,
// n_edges = n - 1. mass is unused (kept for the interface of the former MSTAR scaffolding).
void build_mst(const double pos[], const double mass[], int n, int order[], int pairs[], int& n_edges);
