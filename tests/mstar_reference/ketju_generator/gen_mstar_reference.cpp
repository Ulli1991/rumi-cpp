// MSTAR reference data from the KETJU integrator, for RUMI++'s tests/mstar_reference.
// The cases are those of ketju-integrator/unit-tests (integration.cpp, mergers.cpp, mst_construction.cpp).
// Copyright (C) 2020-2022 Matias Mannerkoski and contributors (the test cases);
// Copyright (C) 2026 Ulrich Steinwandel (this generator).
// Licensed under the GPLv3 license, see LICENSE in this directory or <https://www.gnu.org/licenses/>.
//
// Usage: gen-mstar-reference <output directory>   (run on one MPI task)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <mpi.h>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ketju_integrator/internal.h"
#include "ketju_integrator/ketju_integrator.h"

namespace {

struct Body {
    double m, x[3], v[3], s[3];
};

struct Spec {
    std::string name, origin, pn_name, note;
    unsigned pn_flags = KETJU_PN_NONE;
    double G = 1, c = 1, t_end = 0, merger_Rs = 12;
    int n_pn = 0;
    std::vector<Body> bodies;
    // KETJU's tolerances: per-component relative (rel_k * gbs_tol * steps) or absolute (abs_k * steps * scale)
    double rel_k = 0, abs_k = 0, abs_pos_scale = 0, abs_vel_scale = 0, fixed_abs = 0;
    double mass_lo = -1, mass_hi = -1; // remnant mass range checks
    std::vector<Body> expected;        // the original test's hard-coded final state (old AR-CHAIN integrator), if any
};

std::string g_outdir;

void header(FILE *f, const std::string &name, const std::string &kind, const std::string &origin) {
    std::fprintf(f, "# MSTAR reference case, produced by running the KETJU integrator (ketju_generator/, GPLv3).\n");
    std::fprintf(f, "# Origin: ketju-integrator unit-tests/%s\n", origin.c_str());
    std::fprintf(f, "case %s\nkind %s\n", name.c_str(), kind.c_str());
}

void run_integration(const Spec &sp) {
    const int n = (int)sp.bodies.size();
    struct ketju_system R;
    ketju_create_system(&R, sp.n_pn, n - sp.n_pn, MPI_COMM_WORLD);
    std::vector<int> ids(n);
    for (int i = 0; i < n; ++i) ids[i] = i + 1;
    R.particle_extra_data = ids.data();
    R.particle_extra_data_elem_size = sizeof(int);
    R.constants->G = sp.G;
    R.constants->c = sp.c;
    R.options->PN_flags = sp.pn_flags;
    R.options->gbs_relative_tolerance = 1e-12;
    R.options->output_time_relative_tolerance = 1e-12;
    R.options->bh_merger_distance_in_Rs = sp.merger_Rs;
    for (int i = 0; i < n; ++i) {
        R.physical_state->mass[i] = sp.bodies[i].m;
        for (int k = 0; k < 3; ++k) {
            R.physical_state->pos[i][k] = sp.bodies[i].x[k];
            R.physical_state->vel[i][k] = sp.bodies[i].v[k];
            if (i < sp.n_pn) R.physical_state->spin[i][k] = sp.bodies[i].s[k];
        }
    }
    ketju_run_integrator(&R, sp.t_end);
    const int steps = R.perf->successful_steps;

    FILE *f = std::fopen((g_outdir + "/" + sp.name + ".txt").c_str(), "w");
    header(f, sp.name, R.num_mergers || sp.mass_lo > 0 ? "merger" : "integration", sp.origin);
    if (!sp.note.empty()) std::fprintf(f, "# %s\n", sp.note.c_str());
    std::fprintf(f, "# Units: G and c as given; spin = spin angular momentum S (chi = c S / (G m^2)).\n");
    std::fprintf(f, "# BH mergers between PN bodies at merger_distance_Rs * 2 G (m1 + m2) / c^2.\n");
    std::fprintf(f, "G %.17g\nc %.17g\npn %s\nketju_pn_flags %u\ngbs_tol 1e-12\nt_end %.17g\nmerger_distance_Rs %.17g\n",
                 sp.G, sp.c, sp.pn_name.c_str(), sp.pn_flags, sp.t_end, sp.merger_Rs);
    std::fprintf(f, "# body id pn mass x y z vx vy vz sx sy sz\n");
    for (int i = 0; i < n; ++i) {
        const Body &b = sp.bodies[i];
        std::fprintf(f, "body %d %d %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n", i + 1, i < sp.n_pn ? 1 : 0, b.m,
                     b.x[0], b.x[1], b.x[2], b.v[0], b.v[1], b.v[2], b.s[0], b.s[1], b.s[2]);
    }
    std::fprintf(f, "# KETJU result\nketju_steps %d\nketju_relative_energy_error %.6e\n", steps, R.perf->relative_energy_error);
    std::fprintf(f, "final_time %.17g\nfinal_n %d\nfinal_n_pn %d\n", R.physical_state->time, R.num_particles, R.num_pn_particles);
    if (sp.rel_k > 0) std::fprintf(f, "tol_rel %.6e\n", sp.rel_k * 1e-12 * steps);
    if (sp.abs_k > 0)
        std::fprintf(f, "tol_abs_pos %.6e\ntol_abs_vel %.6e\n", sp.abs_k * steps * sp.abs_pos_scale, sp.abs_k * steps * sp.abs_vel_scale);
    if (sp.fixed_abs > 0) std::fprintf(f, "tol_abs %.6e\n", sp.fixed_abs);
    if (sp.mass_lo > 0) std::fprintf(f, "remnant_mass_range %.17g %.17g\n", sp.mass_lo, sp.mass_hi);
    for (int i = 0; i < R.num_particles; ++i) {
        std::fprintf(f, "final_body %d %d %.17g", ids[i], i < R.num_pn_particles ? 1 : 0, R.physical_state->mass[i]);
        for (int k = 0; k < 3; ++k) std::fprintf(f, " %.17g", R.physical_state->pos[i][k]);
        for (int k = 0; k < 3; ++k) std::fprintf(f, " %.17g", R.physical_state->vel[i][k]);
        for (int k = 0; k < 3; ++k) std::fprintf(f, " %.17g", i < R.num_pn_particles ? R.physical_state->spin[i][k] : 0.);
        std::fprintf(f, "\n");
    }
    if (!sp.expected.empty()) {
        std::fprintf(f, "# the original test's expected final state (old AR-CHAIN integrator, gbs tol 1e-14): the reference "
                        "for other codes\n");
        for (int i = 0; i < (int)sp.expected.size(); ++i) {
            const Body &b = sp.expected[i];
            std::fprintf(f, "expected_body %d 1 %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n", i + 1, b.m, b.x[0],
                         b.x[1], b.x[2], b.v[0], b.v[1], b.v[2], b.s[0], b.s[1], b.s[2]);
        }
    }
    std::fprintf(f, "# merger seq id_survivor id_absorbed m1 m2 m_remnant chi_remnant v_kick t_merger\n");
    for (int k = 0; k < R.num_mergers; ++k) {
        const auto &mi = R.merger_infos[k];
        std::fprintf(f, "merger %d %d %d %.17g %.17g %.17g %.17g %.17g %.17g\n", k, ids[mi.extra_index1], ids[mi.extra_index2], mi.m1,
                     mi.m2, mi.m_remnant, mi.chi_remnant, mi.v_kick, mi.t_merger);
    }
    std::fclose(f);
    std::printf("%-40s steps=%d n=%d mergers=%d dE/E=%.2e\n", sp.name.c_str(), steps, R.num_particles, R.num_mergers,
                R.perf->relative_energy_error);
    ketju_free_system(&R);
}

// exact Euclidean MST length (O(N^2) Prim in double), independent of KETJU's float edge weights
double exact_mst_length(const std::vector<std::array<double, 3>> &x) {
    const int n = (int)x.size();
    std::vector<double> d(n, std::numeric_limits<double>::max());
    std::vector<bool> in(n, false);
    d[0] = 0;
    double L = 0;
    for (int s = 0; s < n; ++s) {
        int u = -1;
        for (int i = 0; i < n; ++i)
            if (!in[i] && (u < 0 || d[i] < d[u])) u = i;
        in[u] = true;
        L += d[u];
        for (int v = 0; v < n; ++v) {
            if (in[v]) continue;
            double r = std::sqrt((x[u][0] - x[v][0]) * (x[u][0] - x[v][0]) + (x[u][1] - x[v][1]) * (x[u][1] - x[v][1]) +
                                 (x[u][2] - x[v][2]) * (x[u][2] - x[v][2]));
            d[v] = std::min(d[v], r);
        }
    }
    return L;
}

void run_mst(enum ketju_mst_algorithm algo, int N, std::mt19937 &gen) {
    const std::string alg = algo == KETJU_MST_PRIM ? "prim" : "dnc";
    const std::string name = "mst_" + alg + "_" + std::to_string(N);
    struct ketju_system R;
    ketju_create_system(&R, 0, N, MPI_COMM_WORLD);
    std::uniform_real_distribution<> dist(-1, 1);
    std::vector<std::array<double, 3>> x(N);
    for (int i = 0; i < N; ++i) {
        R.physical_state->mass[i] = 1;
        for (int k = 0; k < 3; ++k) x[i][k] = R.physical_state->pos[i][k] = dist(gen);
    }
    R.options->mst_algorithm = algo; // the original test always set KETJU_MST_PRIM here
    ketju_construct_MST(&R);
    auto *mst = R.internal_state->mst;
    std::vector<std::pair<int, int>> edges;
    double L = 0;
    for (int e = 0; e < N - 1; ++e) {
        int a = mst->edge_in_mst[e].vertex1, b = mst->edge_in_mst[e].vertex2;
        edges.emplace_back(std::min(a, b) + 1, std::max(a, b) + 1);
        L += std::sqrt((x[a][0] - x[b][0]) * (x[a][0] - x[b][0]) + (x[a][1] - x[b][1]) * (x[a][1] - x[b][1]) +
                       (x[a][2] - x[b][2]) * (x[a][2] - x[b][2]));
    }
    std::sort(edges.begin(), edges.end());
    const double Lx = exact_mst_length(x);

    FILE *f = std::fopen((g_outdir + "/" + name + ".txt").c_str(), "w");
    header(f, name, "mst", "mst_construction.cpp");
    std::fprintf(f, "# Positions: std::mt19937(0x5eed) + uniform(-1, 1), drawn in gtest order (Prim 2..1000, then DnC 2..1000).\n");
    std::fprintf(f, "# Edge weight = Euclidean distance (all masses 1). Edges as sorted id pairs (ids 1..N).\n");
    std::fprintf(f, "mst_algorithm %s\n# body id pn mass x y z vx vy vz sx sy sz\n", alg.c_str());
    for (int i = 0; i < N; ++i)
        std::fprintf(f, "body %d 0 1 %.17g %.17g %.17g 0 0 0 0 0 0\n", i + 1, x[i][0], x[i][1], x[i][2]);
    std::fprintf(f, "# KETJU result\nmst_length_exact %.17g\nmst_length_ketju %.17g\n", Lx, L);
    for (auto &e : edges) std::fprintf(f, "mst_edge %d %d\n", e.first, e.second);
    std::fclose(f);
    std::printf("%-40s L_ketju/L_exact-1=%.2e\n", name.c_str(), L / Lx - 1);
    ketju_free_system(&R);
}

Body B(double m, double x, double y, double vx, double vy, double sx = 0, double sy = 0) {
    return Body{m, {x, y, 0}, {vx, vy, 0}, {sx, sy, 0}};
}

} // namespace

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <output directory>\n", argv[0]);
        return 1;
    }
    g_outdir = argv[1];
    std::vector<Spec> specs;
    {
        Spec s; // Integration_versus_old_integrator.binary_system_full_pn_no_spin
        s.name = "binary_full_pn_no_spin"; s.origin = "integration.cpp (binary_system_full_pn_no_spin)";
        s.pn_name = "dynamic_all"; s.pn_flags = KETJU_PN_DYNAMIC_ALL; s.t_end = 1e6; s.n_pn = 2; s.rel_k = 10;
        s.bodies = {B(1, 40, 0, 0, .085, 1, 0), B(1, -40, 0, 0, -.085, 0, 1)};
        s.note = "PN 1-3.5 incl. three-body terms, no spin terms (spins are carried unchanged).";
        s.expected = {Body{1, {-15.675129114282553, -24.12600028245775, 0}, {0.07932781836871444, -0.051044405469834665, 0}, {1, 0, 0}},
                      Body{1, {15.675129114282553, 24.12600028245775, 0}, {-0.07932781836871444, 0.051044405469834665, 0}, {0, 1, 0}}};
        specs.push_back(s);
    }
    {
        Spec s; // Integration_versus_old_integrator.binary_system_no_pn
        s.name = "binary_no_pn"; s.origin = "integration.cpp (binary_system_no_pn)";
        s.pn_name = "none"; s.t_end = 1e6; s.n_pn = 2; s.rel_k = 20;
        s.bodies = {B(1, 40, 0, 0, .085, 1, 0), B(1, -40, 0, 0, -.085, 0, 1)};
        s.note = "Newtonian; ketju_relative_energy_error is the energy error of the run.";
        s.expected = {Body{1, {28.093181660139404, -31.029347533391817, 0}, {0.05450807268322124, 0.060820810188436096, 0}, {1, 0, 0}},
                      Body{1, {-28.093181660139404, 31.029347533391817, 0}, {-0.05450807268322124, -0.060820810188436096, 0}, {0, 1, 0}}};
        specs.push_back(s);
    }
    for (int tb = 1; tb >= 0; --tb) { // Integration_threebody_terms: the two runs must agree (three-body terms vanish for N = 2)
        Spec s;
        s.name = tb ? "binary_1pn_threebody" : "binary_1pn"; s.origin = "integration.cpp (binary_system_with_and_without_threebody_terms)";
        s.pn_name = tb ? "1pn+threebody" : "1pn"; s.pn_flags = KETJU_PN_1_0_ACC | (tb ? KETJU_PN_THREEBODY : 0);
        s.t_end = 1e5; s.n_pn = 2; s.abs_k = 2.5e-12; s.abs_pos_scale = 40; s.abs_vel_scale = .085;
        s.bodies = {B(.5, 40, 0, 0, .085), B(1.5, -40, 0, 0, -.085)};
        s.note = "binary_1pn and binary_1pn_threebody must agree within the same tolerance.";
        specs.push_back(s);
    }
    {
        Spec s; // Integration_units.binary_system_full_pn_spin_generic_units, natural units
        s.name = "binary_full_pn_spin"; s.origin = "integration.cpp (binary_system_full_pn_spin_generic_units)";
        s.pn_name = "all"; s.pn_flags = KETJU_PN_ALL; s.t_end = 1e6; s.n_pn = 2; s.rel_k = 200;
        s.bodies = {B(.5, 40, 0, 0, .085, 1, 0), B(1.5, -40, 0, 0, -.085, 0, 1)};
        s.note = "PN 1-3.5 + spin-orbit/spin-spin + spin evolution; same physics as binary_full_pn_spin_generic_units.";
        specs.push_back(s);
        const double c = 12354, G = 0.0634, ul = G / (c * c), ut = G / (c * c * c), uv = c, us = G / c;
        s.name = "binary_full_pn_spin_generic_units"; s.G = G; s.c = c; s.t_end = 1e6 * ut;
        s.bodies = {B(.5, 40 * ul, 0, 0, .085 * uv, 1 * us, 0), B(1.5, -40 * ul, 0, 0, -.085 * uv, 0, 1 * us)};
        s.note = "binary_full_pn_spin in G = 0.0634, c = 12354: lengths / (G/c^2), times / (G/c^3), velocities / c, spins / (G/c) "
                 "must reproduce it.";
        specs.push_back(s);
    }
    {
        Spec s; // Integration_with_mergers.head_on_merger
        s.name = "head_on_merger"; s.origin = "mergers.cpp (head_on_merger)";
        s.pn_name = "all"; s.pn_flags = KETJU_PN_ALL; s.t_end = 1e6; s.n_pn = 2; s.fixed_abs = 1e-8;
        s.mass_lo = 1.99; s.mass_hi = 2.0;
        s.bodies = {B(1, 40, 0, 0, 0), B(1, -40, 0, 0, 0)};
        s.note = "One remnant at rest at the origin with zero spin (tol_abs); remnant mass in remnant_mass_range.";
        specs.push_back(s);
    }
    {
        Spec s; // Integration_with_mergers.merger_from_orbit
        s.name = "merger_from_orbit"; s.origin = "mergers.cpp (merger_from_orbit)";
        s.pn_name = "all"; s.pn_flags = KETJU_PN_ALL; s.t_end = 1e6; s.n_pn = 2; s.mass_lo = 1.8; s.mass_hi = 2.01;
        s.bodies = {B(1, 40, 0, 0, .05), B(1, -40, 0, 0, 0)};
        s.note = "Only the merger and remnant_mass_range are checked (the remnant state depends on the GR fits).";
        specs.push_back(s);
    }
    {
        Spec s; // Instant_mergers: mergers detected at the first checks, tend = 1
        s.pn_name = "all"; s.pn_flags = KETJU_PN_ALL; s.t_end = 1; s.merger_Rs = 6;
        s.note = "Merger order and ids are checked (KETJU's extra_data index bookkeeping in the original test).";
        s.name = "instant_binary_merger"; s.origin = "mergers.cpp (binary_merger_with_extra_data)"; s.n_pn = 2;
        s.bodies = {B(1, 4, 0, 0, .05), B(1, -4, 0, 0, 0), B(1e-3, 1e4, 0, 0, 0)};
        specs.push_back(s);
        s.name = "instant_multiple_merger1"; s.origin = "mergers.cpp (multiple_merger_with_extra_data1)"; s.n_pn = 3;
        s.bodies = {B(1, 4, 0, 0, 0), B(1, 0, -24, 0, 0), B(1, -4, 0, 0, 0), B(1e-3, 1e4, 0, 0, 0)};
        specs.push_back(s);
        s.name = "instant_multiple_merger2"; s.origin = "mergers.cpp (multiple_merger_with_extra_data2)"; s.n_pn = 3;
        s.bodies = {B(1, 4, 0, 0, 0), B(.9, -22, 0, 0, 0), B(1, -23, 0, 0, 0), B(1e-3, 1e4, 0, 0, 0)};
        specs.push_back(s);
        s.name = "instant_multiple_merger3"; s.origin = "mergers.cpp (multiple_merger_with_extra_data3)"; s.n_pn = 4;
        s.bodies = {B(1, 10, 0, 0, 0), B(.9, 10 + (12 + 1e-12) * 1.9, 0, 0, 0), B(1, -100, 0, 0, 0), B(2, -105, 0, 0, 0),
                    B(1e-3, 1e4, 0, 0, 0)};
        specs.push_back(s);
    }
    for (auto &s : specs) run_integration(s);
    std::mt19937 gen(0x5eed);
    for (auto algo : {KETJU_MST_PRIM, KETJU_MST_DIVIDE_AND_CONQUER_PRIM})
        for (int N : {2, 4, 10, 20, 50, 100, 500, 1000}) run_mst(algo, N, gen);
    MPI_Finalize();
    return 0;
}
