// KETJU reference cases (tests/mstar_reference/data, produced by running the KETJU integrator) run through RUMI++
// via mstar_reference_run(). Usage: mstar-reference-test <case file>. Exit 0 pass, 1 fail, 77 skipped (not applicable).
// Checks per kind (README.md):
//   integration  body count, time, per-id mass/pos/vel/spin within the case tolerance; |dE/E| < 1e-10 for pn = none
//   merger       merger count and id groups (any order), component masses, remnant mass (range or (0.8, 1] (m1 + m2),
//                = final mass), final ids; final state only if tol_abs is given
//   mst          spanning tree on the ids; prim: length = exact MST and KETJU's edge set; dnc: length <= KETJU's
#include "mstar_reference_adapter.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>

namespace {

struct Reference {
    MstarRefCase c;
    double final_time = 0;
    int final_n = -1, final_n_pn = -1;
    double tol_rel = 0, tol_abs_pos = 0, tol_abs_vel = 0, tol_abs = 0;
    double mass_lo = -1, mass_hi = -1;
    double mst_length_exact = 0, mst_length_ketju = 0;
    std::vector<MstarRefBody> final_bodies;
    std::vector<MstarRefBody> expected_bodies; // the original test's reference (old integrator): used if present
    std::vector<MstarRefMerger> mergers;
    std::vector<std::pair<int, int>> mst_edges;
};

int g_fail = 0;

void fail(const char* fmt, double a = 0, double b = 0, double c = 0) {
    std::printf("FAIL ");
    std::printf(fmt, a, b, c);
    std::printf("\n");
    ++g_fail;
}

MstarRefBody read_body(std::istringstream& in) {
    MstarRefBody b;
    int pn = 0;
    in >> b.id >> pn >> b.m >> b.x[0] >> b.x[1] >> b.x[2] >> b.v[0] >> b.v[1] >> b.v[2] >> b.s[0] >> b.s[1] >> b.s[2];
    b.pn = pn != 0;
    return b;
}

bool read_reference(const char* path, Reference& R) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream in(line);
        std::string key;
        in >> key;
        if (key == "case") in >> R.c.name;
        else if (key == "kind") in >> R.c.kind;
        else if (key == "G") in >> R.c.G;
        else if (key == "c") in >> R.c.c;
        else if (key == "pn") in >> R.c.pn;
        else if (key == "ketju_pn_flags") in >> R.c.ketju_pn_flags;
        else if (key == "gbs_tol") in >> R.c.gbs_tol;
        else if (key == "t_end") in >> R.c.t_end;
        else if (key == "merger_distance_Rs") in >> R.c.merger_distance_Rs;
        else if (key == "mst_algorithm") in >> R.c.mst_algorithm;
        else if (key == "body") R.c.bodies.push_back(read_body(in));
        else if (key == "final_body") R.final_bodies.push_back(read_body(in));
        else if (key == "expected_body") R.expected_bodies.push_back(read_body(in));
        else if (key == "final_time") in >> R.final_time;
        else if (key == "final_n") in >> R.final_n;
        else if (key == "final_n_pn") in >> R.final_n_pn;
        else if (key == "tol_rel") in >> R.tol_rel;
        else if (key == "tol_abs_pos") in >> R.tol_abs_pos;
        else if (key == "tol_abs_vel") in >> R.tol_abs_vel;
        else if (key == "tol_abs") in >> R.tol_abs;
        else if (key == "remnant_mass_range") in >> R.mass_lo >> R.mass_hi;
        else if (key == "mst_length_exact") in >> R.mst_length_exact;
        else if (key == "mst_length_ketju") in >> R.mst_length_ketju;
        else if (key == "mst_edge") {
            std::pair<int, int> e;
            in >> e.first >> e.second;
            R.mst_edges.push_back(e);
        } else if (key == "merger") {
            MstarRefMerger m;
            int seq;
            double chi, vk, t;
            in >> seq >> m.id1 >> m.id2 >> m.m1 >> m.m2 >> m.m_remnant >> chi >> vk >> t;
            R.mergers.push_back(m);
        }
        // ketju_steps, ketju_relative_energy_error: provenance only
    }
    return !R.c.name.empty();
}

bool close_rel(double a, double b, double rel) { return std::abs(a - b) <= rel * std::max(std::abs(a), std::abs(b)); }

// component tolerance: relative to the component, floored at 1e-3 of the vector's largest component
void compare_vec(const char* what, int id, const double* got, const double* ref, double rel, double abs_tol) {
    const double scale = std::max({std::abs(ref[0]), std::abs(ref[1]), std::abs(ref[2])});
    for (int k = 0; k < 3; ++k) {
        const double allowed = abs_tol > 0 ? abs_tol : rel * std::max(std::abs(ref[k]), 1e-3 * scale);
        if (!(std::abs(got[k] - ref[k]) <= allowed)) {
            std::printf("FAIL %s id %d [%d]: got %.17g ref %.17g |d| %.3e > %.3e\n", what, id, k, got[k], ref[k],
                        std::abs(got[k] - ref[k]), allowed);
            ++g_fail;
        }
    }
}

int find_group(std::vector<int>& p, int i) { return p[i] == i ? i : p[i] = find_group(p, p[i]); }

void check_dynamics(const Reference& R, const MstarRefResult& r) {
    const bool is_merger = R.c.kind == "merger";
    if (!(std::abs(r.t - R.final_time) <= 1e-12 * std::abs(R.c.t_end))) fail("time %.17g, reference %.17g", r.t, R.final_time);
    if ((int)r.bodies.size() != R.final_n) fail("final body count %g, reference %g", r.bodies.size(), R.final_n);
    const int n_pn = (int)std::count_if(r.bodies.begin(), r.bodies.end(), [](const MstarRefBody& b) { return b.pn; });
    if (n_pn != R.final_n_pn) fail("final PN body count %g, reference %g", n_pn, R.final_n_pn);

    // mergers: the pairs as id groups, in any order (KETJU's order follows its merger radius); masses unordered
    if (r.mergers.size() != R.mergers.size()) fail("merger count %g, reference %g", r.mergers.size(), R.mergers.size());
    // groups of ids that merged in the reference; either member may label the remnant
    const int nid = (int)R.c.bodies.size() + 1;
    std::vector<int> grp(nid);
    std::iota(grp.begin(), grp.end(), 0);
    for (const auto& a : R.mergers) grp[find_group(grp, a.id1)] = find_group(grp, a.id2);
    std::vector<bool> used(R.mergers.size(), false);
    std::map<int, double> initial_mass, current_mass;
    for (const auto& bb : R.c.bodies) initial_mass[bb.id] = bb.m;
    for (size_t k = 0; k < r.mergers.size(); ++k) {
        const auto& b = r.mergers[k];
        if (b.id1 <= 0 || b.id1 >= nid || b.id2 <= 0 || b.id2 >= nid || find_group(grp, b.id1) != find_group(grp, b.id2)) {
            fail("merger %g: ids not merged in the reference", k);
            continue;
        }
        bool found = false;
        for (size_t j = 0; j < R.mergers.size() && !found; ++j)
            if (!used[j] && find_group(grp, R.mergers[j].id1) == find_group(grp, b.id1)) used[j] = found = true;
        if (!found) fail("merger %g: more mergers in this group than in the reference", k);
        // component masses: an initial body's own mass, or the product mass of its previous merger
        for (int side = 0; side < 2; ++side) {
            const int id = side ? b.id2 : b.id1;
            const double mc = side ? b.m2 : b.m1;
            const double expect = current_mass.count(id) ? current_mass[id] : initial_mass[id];
            if (!close_rel(mc, expect, 1e-12)) fail("merger %g: component mass %.17g, expected %.17g", k, mc, expect);
        }
        current_mass[b.id1] = current_mass[b.id2] = b.m_remnant;
        if (R.mass_lo > 0) {
            if (!(b.m_remnant >= R.mass_lo && b.m_remnant <= R.mass_hi))
                fail("merger remnant mass %.17g outside [%g, %g]", b.m_remnant, R.mass_lo, R.mass_hi);
        } else if (!(b.m_remnant <= (b.m1 + b.m2) * (1 + 1e-12) && b.m_remnant > 0.8 * (b.m1 + b.m2))) {
            fail("merger %g: remnant mass %.17g outside (0.8, 1] (m1 + m2)", k, b.m_remnant);
        }
    }
    // remnant mass of each merger group as booked by RUMI++'s last merger in it
    std::map<int, double> remnant_mass;
    for (const auto& b : r.mergers)
        if (b.id1 > 0 && b.id1 < nid) remnant_mass[find_group(grp, b.id1)] = b.m_remnant;

    // final bodies, matched through the merger groups
    std::map<int, const MstarRefBody*> ref_by_group;
    for (const auto& b : R.expected_bodies.empty() ? R.final_bodies : R.expected_bodies) ref_by_group[find_group(grp, b.id)] = &b;
    std::set<int> seen;
    for (const auto& b : r.bodies) {
        if (b.id <= 0 || b.id >= nid) {
            fail("final body id %g out of range", b.id);
            continue;
        }
        const int g = find_group(grp, b.id);
        auto it = ref_by_group.find(g);
        if (it == ref_by_group.end() || !seen.insert(g).second) {
            fail("final body id %g has no (unique) reference counterpart", b.id);
            continue;
        }
        const MstarRefBody& ref = *it->second;
        if (remnant_mass.count(g)) {
            if (!close_rel(b.m, remnant_mass[g], 1e-12)) fail("id %g: mass %.17g, its merger record says %.17g", b.id, b.m, remnant_mass[g]);
        } else if (!close_rel(b.m, ref.m, 1e-12)) {
            fail("id %g: mass %.17g, reference %.17g", b.id, b.m, ref.m);
        }
        if (is_merger && R.tol_abs <= 0) continue; // only merger bookkeeping is checked
        const double tp = R.tol_abs > 0 ? R.tol_abs : R.tol_abs_pos, tv = R.tol_abs > 0 ? R.tol_abs : R.tol_abs_vel;
        compare_vec("pos", b.id, b.x, ref.x, R.tol_rel, tp);
        compare_vec("vel", b.id, b.v, ref.v, R.tol_rel, tv);
        if (b.pn && r.has_spins) compare_vec("spin", b.id, b.s, ref.s, R.tol_rel, R.tol_abs > 0 ? R.tol_abs : (R.tol_rel > 0 ? 0 : 1e-12));
    }
    if (R.c.pn == "none" && !(std::abs(r.relative_energy_error) < 1e-10))
        fail("relative energy error %.3e >= 1e-10", r.relative_energy_error);
}

void check_mst(const Reference& R, const MstarRefResult& r) {
    const int n = (int)R.c.bodies.size();
    if ((int)r.mst_edges.size() != n - 1) fail("MST has %g edges, expected %g", r.mst_edges.size(), n - 1);
    std::vector<int> grp(n + 1);
    std::iota(grp.begin(), grp.end(), 0);
    std::map<int, const MstarRefBody*> byid;
    for (const auto& b : R.c.bodies) byid[b.id] = &b;
    double L = 0;
    for (auto e : r.mst_edges) {
        if (!byid.count(e.first) || !byid.count(e.second) || e.first == e.second) {
            fail("invalid MST edge %g-%g", e.first, e.second);
            continue;
        }
        const int a = find_group(grp, e.first), b = find_group(grp, e.second);
        if (a == b) fail("MST edge %g-%g closes a cycle", e.first, e.second);
        grp[a] = b;
        const double* p = byid[e.first]->x;
        const double* q = byid[e.second]->x;
        L += std::sqrt((p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) + (p[2] - q[2]) * (p[2] - q[2]));
    }
    if (g_fail) return;
    if (L < R.mst_length_exact * (1 - 1e-12)) fail("MST length %.17g below the exact minimum %.17g", L, R.mst_length_exact);
    if (R.c.mst_algorithm == "prim") {
        if (L > R.mst_length_exact * (1 + 1e-12)) fail("MST length %.17g, exact %.17g", L, R.mst_length_exact);
        std::vector<std::pair<int, int>> got;
        for (auto e : r.mst_edges) got.emplace_back(std::min(e.first, e.second), std::max(e.first, e.second));
        std::sort(got.begin(), got.end());
        if (got != R.mst_edges) fail("MST edge set differs from KETJU's");
    } else if (L > R.mst_length_ketju * (1 + 1e-12)) {
        fail("MST length %.17g longer than KETJU's %.17g", L, R.mst_length_ketju);
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <case file>\n", argv[0]);
        return 1;
    }
    Reference R;
    if (!read_reference(argv[1], R)) {
        std::fprintf(stderr, "cannot read reference case %s\n", argv[1]);
        return 1;
    }
    MstarRefResult r;
    if (mstar_reference_run(R.c, r) == MstarRefStatus::NotApplicable) {
        std::printf("SKIP %s: %s\n", R.c.name.c_str(), r.skip_reason.c_str());
        return 77;
    }
    if (!r.has_spins && R.c.kind != "mst") std::printf("NOTE %s: spins not reported, not compared\n", R.c.name.c_str());
    for (size_t k = 0; k < r.mergers.size(); ++k)
        std::printf("INFO merger %zu: ids %d + %d, remnant %.6f (KETJU %.6f)\n", k, r.mergers[k].id1, r.mergers[k].id2,
                    r.mergers[k].m_remnant, k < R.mergers.size() ? R.mergers[k].m_remnant : 0.0);
    if (R.c.kind == "mst") check_mst(R, r);
    else check_dynamics(R, r);
    std::printf("%s %s (%d failed checks)\n", g_fail ? "FAIL" : "PASS", R.c.name.c_str(), g_fail);
    return g_fail ? 1 : 0;
}
