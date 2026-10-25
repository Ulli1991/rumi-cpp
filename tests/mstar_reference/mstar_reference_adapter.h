// Interface between the KETJU reference harness (mstar_reference_test.cpp) and RUMI++ (mstar_reference_adapter.cpp).
// NotApplicable (with skip_reason) is reported as skipped (exit code 77).
#pragma once

#include <string>
#include <utility>
#include <vector>

struct MstarRefBody {
    int id = 0;         // 1..N, stable across the run; a merger remnant keeps one of its components' ids
    bool pn = false;    // PN (BH) body; spins and mergers only for these
    double m = 0;
    double x[3] = {0, 0, 0}, v[3] = {0, 0, 0};
    double s[3] = {0, 0, 0}; // spin angular momentum S (chi = c S / (G m^2))
};

struct MstarRefMerger {
    int id1 = 0, id2 = 0;  // the two merging bodies (either may label the remnant)
    double m1 = 0, m2 = 0, m_remnant = 0;
};

struct MstarRefCase {
    std::string name, kind;   // kind: integration | merger | mst
    double G = 1, c = 1;
    std::string pn;           // none | 1pn | 1pn+threebody | dynamic_all (PN 1-3.5) | all (dynamic_all + spin terms)
    unsigned ketju_pn_flags = 0;
    double gbs_tol = 1e-12, t_end = 0;
    double merger_distance_Rs = 12; // PN bodies merge within merger_distance_Rs * 2 G (m1 + m2) / c^2
    std::string mst_algorithm;      // mst cases: prim | dnc
    std::vector<MstarRefBody> bodies;
};

struct MstarRefResult {
    double t = 0;                               // physical time reached
    std::vector<MstarRefBody> bodies;           // final state, any order
    std::vector<MstarRefMerger> mergers;        // in the order they happened
    std::vector<std::pair<int, int>> mst_edges; // mst cases: N - 1 edges as body-id pairs
    double relative_energy_error = 0;           // (E_end - E_0) / |E_0|, Newtonian energy (checked for pn = none)
    bool has_spins = true;                      // false: final spins not reported, not compared
    std::string skip_reason;                    // NotApplicable: why
};

enum class MstarRefStatus { Ok, NotApplicable };

// integration/merger: integrate c.bodies for c.t_end; mst: build the spanning tree of c.bodies only
MstarRefStatus mstar_reference_run(const MstarRefCase& c, MstarRefResult& r);
