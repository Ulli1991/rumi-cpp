// SPDX-License-Identifier: BSD-3-Clause
#include "mst.h"

#include <cmath>
#include <limits>
#include <vector>

void build_mst(const double pos[], const double mass[], int n, int order[], int pairs[], int& n_edges) {
    (void)mass;
    n_edges = n > 0 ? n - 1 : 0;
    if (n <= 0) return;
    std::vector<char> in(n, 0);
    std::vector<double> best(n, std::numeric_limits<double>::infinity());
    std::vector<int> from(n, -1);
    best[0] = 0.0;
    for (int k = 0; k < n; k++) {
        int u = -1;
        for (int i = 0; i < n; i++)
            if (!in[i] && (u < 0 || best[i] < best[u])) u = i;
        in[u] = 1;
        order[k] = u;
        if (k > 0) { pairs[2 * (k - 1)] = from[u]; pairs[2 * (k - 1) + 1] = u; }
        for (int i = 0; i < n; i++) {
            if (in[i]) continue;
            double r2 = 0;
            for (int c = 0; c < 3; c++) { const double d = pos[3 * i + c] - pos[3 * u + c]; r2 += d * d; }
            const double r = std::sqrt(r2);
            if (r < best[i]) { best[i] = r; from[i] = u; }
        }
    }
}
