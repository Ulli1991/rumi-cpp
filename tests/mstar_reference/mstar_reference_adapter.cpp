// RUMI++ side of the KETJU reference harness.
// integration/merger: the library path (src/lib/rumi_lib.h: HHS with LogH+GBS subsystems), the case scaled from
// KETJU's units to RUMI++'s (mass unit M_UNIT Msun, length G M_UNIT / c^2, velocity c); mst: build_mst (mstar.h).
#include "mstar_reference_adapter.h"

#include "constants.h"
#include "mstar.h"
#include "rumi_lib.h"

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr double M_UNIT = 1e6; // Msun per KETJU mass unit: BHs far above M_PN, lengths ~1e-6 pc

int pn_order(const std::string& pn) {
    if (pn == "none") return 0;
    if (pn == "1pn" || pn == "1pn+threebody") return 1; // three-body 1PN terms vanish for the binaries that use them
    if (pn == "dynamic_all") return 31;
    if (pn == "all") return 63;
    return -1;
}

MstarRefStatus run_mst(const MstarRefCase& c, MstarRefResult& r) {
    const int n = (int)c.bodies.size();
    std::vector<double> pos(3 * n), mass(n);
    for (int i = 0; i < n; ++i) {
        mass[i] = c.bodies[i].m;
        for (int k = 0; k < 3; ++k) pos[3 * i + k] = c.bodies[i].x[k];
    }
    std::vector<int> order(n), pairs(2 * n);
    int n_edges = 0;
    build_mst(pos.data(), mass.data(), n, order.data(), pairs.data(), n_edges);
    for (int e = 0; e < n_edges; ++e) r.mst_edges.emplace_back(c.bodies[pairs[2 * e]].id, c.bodies[pairs[2 * e + 1]].id);
    return MstarRefStatus::Ok;
}

} // namespace

MstarRefStatus mstar_reference_run(const MstarRefCase& c, MstarRefResult& r) {
    if (c.kind == "mst") return run_mst(c, r);
    const int pn = pn_order(c.pn);
    if (pn < 0) {
        r.skip_reason = "unknown PN set " + c.pn;
        return MstarRefStatus::NotApplicable;
    }
    // KETJU units -> natural (G = c = 1) -> RUMI++
    const double G = G_CODE, cl = C_CODE;
    const double Lk = c.G / (c.c * c.c), Vk = c.c, Tk = c.G / (c.c * c.c * c.c);
    const double Lr = G * M_UNIT / (cl * cl), Vr = cl, Tr = G * M_UNIT / (cl * cl * cl);
    const double fx = Lr / Lk, fv = Vr / Vk, ft = Tr / Tk;

    // LogH end-time tolerance = KETJU's output_time_relative_tolerance of the cases (RUMI++ default 1e-4)
    setenv("NBODY_GBS_TTOL", "1e-12", 0);
    FILE* f = std::fopen("ketju_reference.cfg", "w");
    if (!f) return MstarRefStatus::NotApplicable;
    std::fprintf(f, "COLLISIONAL = true\nPRESET = cluster\nSSE_ENABLED = false\nTIDAL_FIELD = none\nSOFTENING_PC = 0\n"
                    "PN_ORDER = %d\nPN_FRAME = lab\nETA_GBS = %.3g\nN_ORB_SECULAR = 1000000000\nSECULAR_TRIPLES = false\nSDAR = false\n"
                    "PRINT_INTERVAL = 0\nPROFILING = off\nGW_CENSUS_TGW_MYR = 0\n",
                 pn, c.gbs_tol);
    std::fclose(f);
    if (rumi_lib::kokkos_initialize(1, -1) != 0 || rumi_lib::configure("ketju_reference.cfg", 0) != 0) {
        r.skip_reason = "rumi_lib configure failed";
        return MstarRefStatus::NotApplicable;
    }
    const int n = (int)c.bodies.size();
    std::vector<long long> id(n);
    std::vector<double> m(n), x(3 * n), v(3 * n), rad(n, 0.0);
    std::vector<int> kw(n);
    for (int i = 0; i < n; ++i) {
        const auto& b = c.bodies[i];
        id[i] = b.id;
        m[i] = b.m * M_UNIT;
        kw[i] = b.pn ? 14 : 1; // PN bodies are BHs; the others main-sequence stars of zero radius
        for (int k = 0; k < 3; ++k) {
            x[3 * i + k] = b.x[k] * fx;
            v[3 * i + k] = b.v[k] * fv;
        }
    }
    const double t_end = c.t_end * ft;
    // host steps: KETJU_REF_HOST_STEPS (default 1000), as a host code would call RUMI++
    const char* hs = std::getenv("KETJU_REF_HOST_STEPS");
    const int n_host = hs ? std::max(1, std::atoi(hs)) : 1000;
    if (rumi_lib::set_particles(n, id.data(), m.data(), x.data(), v.data(), kw.data(), rad.data(), G, t_end / n_host) != 0) {
        r.skip_reason = "rumi_lib set_particles failed";
        return MstarRefStatus::NotApplicable;
    }
    // spins: KETJU's S -> chi = c S / (G m^2) (unclamped: KETJU's body 1 of the spin cases has chi = 4)
    std::vector<double> chi(3 * n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k) chi[3 * i + k] = c.bodies[i].s[k] * c.c / (c.G * c.bodies[i].m * c.bodies[i].m);
    rumi_lib::set_spins(chi.data());
    double E0, B0, E1, B1;
    rumi_lib::energy(G, &E0, &B0);
    for (int s = 0; s < n_host; ++s)
        if (rumi_lib::advance(t_end / n_host, G) != 0) {
            r.skip_reason = "rumi_lib advance failed";
            return MstarRefStatus::NotApplicable;
        }
    rumi_lib::energy(G, &E1, &B1);
    r.relative_energy_error = ((E1 + B1) - (E0 + B0)) / std::abs(E0);
    r.t = c.t_end;

    std::vector<int> kept, gone;
    std::vector<double> mprod;
    rumi_lib::merger_events(kept, gone, mprod);
    std::vector<double> mslot(m);
    for (size_t k = 0; k < kept.size(); ++k) {
        MstarRefMerger e;
        e.id1 = (int)id[kept[k]];
        e.id2 = (int)id[gone[k]];
        e.m1 = mslot[kept[k]] / M_UNIT;
        e.m2 = mslot[gone[k]] / M_UNIT;
        e.m_remnant = mprod[k] / M_UNIT;
        mslot[kept[k]] = mprod[k];
        mslot[gone[k]] = 0;
        r.mergers.push_back(e);
    }
    rumi_lib::get_state(m.data(), x.data(), v.data());
    rumi_lib::get_spins(chi.data());
    for (int i = 0; i < n; ++i) {
        if (m[i] <= 0) continue;
        MstarRefBody b;
        b.id = (int)id[i];
        b.pn = c.bodies[i].pn;
        b.m = m[i] / M_UNIT;
        for (int k = 0; k < 3; ++k) {
            b.x[k] = x[3 * i + k] / fx;
            b.v[k] = v[3 * i + k] / fv;
            b.s[k] = chi[3 * i + k] * c.G * b.m * b.m / c.c;
        }
        r.bodies.push_back(b);
    }
    rumi_lib::kokkos_finalize();
    return MstarRefStatus::Ok;
}
