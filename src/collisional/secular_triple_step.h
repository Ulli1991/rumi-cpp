#pragma once
// One secular step of a stable hierarchical triple (Kozai-Lidov + GW, secular.h), energy- and angular-momentum-exact
// rebuild (docs/CASES.md C4, C6). Plain C++ (tests/enzyme differentiates it).
struct SecularTripleStep {
    bool hierarchical = true;    // false: no longer hierarchical (a_in or a_out <= 0), integrate it directly (LogH)
    double booked_gw = 0.0;      // energy radiated by the inner binary (E_gw)
    int unfixed = 0;             // corrections no orbit could take (angular momentum, energy)
    double a_in = 0.0, e_in = 0.0, a_out = 0.0, cos_i = 0.0;   // elements before the step (log)
};
// bodies 0, 1: inner binary (0 = inner_binary_id), 2: outer companion; m, x, v (Msun, pc, km/s) in and out
SecularTripleStep secular_triple_step(double m[3], double x[3][3], double v[3][3], const int pids[3], double dt,
                                      double G, double c_code, int pn_order);
