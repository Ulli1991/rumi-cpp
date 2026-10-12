#include "runtime_sse.h"
#include "stellar_cache.h"
#include "constants.h"
#include "diagnostics.h"
#include "collisional_config.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

// ============================================================================
// Natal kicks of new neutron stars and black holes: the "standard, fallback-controlled" kick of Banerjee et al. 2020
// (A&A 639, A41, arXiv 1902.07718, sec. 3.1, eq. 1; the updated SSE/BSE that BIFROST sec. 3.13 cites):
//   v_kick = v_NS (1 - f_fb),  v_NS from a Maxwellian of 1D dispersion sigma_NS = 265 km/s (Hobbs et al. 2005),
// isotropic. f_fb is the fallback fraction of the remnant prescription (SSEResult::fbfac): Fryer et al. 2012
// rapid/delayed, Belczynski et al. 2002/2008 (f_fb = (M_CO - 5)/2.6 between M_CO = 5 and 7.6 Msun, 1 above; the
// StarTrack kick v = v_NS (1 - f_fb) of Belczynski et al. 2008 is the same rule); direct collapse (f_fb = 1) gives
// no kick. HURLEY and ELDRIDGE_TOUT have no fallback (f_fb = 0): every NS and BH gets the full NS kick, as BSE
// (Hurley et al. 2002) does for every NS and BH. Electron-capture SN NSs: Maxwellian sigma = 3 km/s, no fallback
// scaling (Banerjee et al. 2020 sec. 3.1 and 3.2, the ECS engine is a full explosion; Gessner & Janka 2018).
// FROST-CLUSTERS I-III use SEVN's "unified" kick model (Giacobbo & Mapelli 2020; FC-I tab. B1) instead, which
// needs SEVN's ejecta masses.
// The draws are a deterministic hash of (particle id, PLUMMER_SEED, new stellar type): identical on every MPI rank
// and across restarts (a particle forms its remnant once, since remnants are not evolved further).
// ============================================================================
namespace {
constexpr double SIGMA_NS_KMS   = 265.0;   // Hobbs et al. 2005, MNRAS 360, 974
constexpr double SIGMA_ECSN_KMS = 3.0;     // Banerjee et al. 2020 sec. 3.2 (Gessner & Janka 2018, ApJ 865, 61)

std::uint64_t splitmix64(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Kick velocity (km/s = code units) of a remnant of type kw_new formed by particle pid
void natal_kick(int pid, int seed, int kw_new, double fbfac, bool ecsn, double dv[3]) {
    dv[0] = dv[1] = dv[2] = 0.0;
    const double scale = ecsn ? SIGMA_ECSN_KMS : SIGMA_NS_KMS * (1.0 - std::min(1.0, std::max(0.0, fbfac)));
    if (scale <= 0.0) return;   // direct collapse
    // FNV-1a of the inputs (as bh_bh_remnant in mergers.cpp), then splitmix64 for the uniforms
    std::uint64_t h = 1469598103934665603ULL;
    for (std::int64_t v : {(std::int64_t)pid, (std::int64_t)seed, (std::int64_t)kw_new, (std::int64_t)0x4b49434bLL})
        h = (h ^ (std::uint64_t)v) * 1099511628211ULL;
    double u[4];
    for (double& x : u) x = ((double)(splitmix64(h) >> 11) + 0.5) * 0x1.0p-53;   // in (0, 1)
    // Box-Muller: three independent N(0, 1) components; the vector is Maxwellian in magnitude and isotropic
    const double two_pi = 6.283185307179586;
    const double r1 = std::sqrt(-2.0 * std::log(u[0])), r2 = std::sqrt(-2.0 * std::log(u[2]));
    dv[0] = scale * r1 * std::cos(two_pi * u[1]);
    dv[1] = scale * r1 * std::sin(two_pi * u[1]);
    dv[2] = scale * r2 * std::cos(two_pi * u[3]);
}
}  // namespace

void advance_stellar_evolution(Particles& p, SSEContext& sse, double dt_myr,
                                EnergyBookkeeping& books, double G) {
    if (dt_myr <= 0.0) return;

    int N = p.N;
    auto h_x     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
    auto h_y     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
    auto h_z     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);

    // Copy stellar fields to host
    auto h_m     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
    auto h_vx    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
    auto h_vy    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
    auto h_vz    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
    auto h_stype = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_type);
    auto h_mzams = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.m_zams);
    auto h_sage  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_age);
    auto h_srad  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_radius);
    auto h_lum   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.luminosity);
    auto h_cmass = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.core_mass);
    auto h_pid   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);

    bool changed = false;
    double total_dm = 0;
    int n_type_change = 0;
    // State before the update, for the exact energy, momentum and angular momentum sinks
    std::vector<int> touched;
    std::vector<double> m_before, v_before;
    double PE_before = 0.0;

    for (int i = 0; i < N; i++) {
        int kw = h_stype(i);

        // Skip remnants (WD=10-12, NS=13, BH=14, massless=15)
        if (kw >= 10) continue;

        double m_zams = h_mzams(i);
        double age_old = h_sage(i);
        double m_old = h_m(i);
        double age_new = age_old + dt_myr;

        // Evolve using SSE fitting formulae
        SSEResult res = sse.evolve(m_zams, age_new);

        // Mass loss
        double dm = m_old - res.mass;
        if (dm > 0) total_dm += dm;
        if (dm != 0.0 || res.kw != kw) {
            // PE before the first change: on a host build the mirrors ARE the particle views, so the masses of p
            // change with h_m below. Before 2026-10-12 PE_before was taken after the loop: on host builds it already
            // saw the new masses and E_sse missed the potential-energy change (|dE/E| 1.4e-3 per 1 Myr of winds
            // in a 500-star cluster of 12 Msun stars, 2 after their supernovae); GPU builds were right
            if (touched.empty()) PE_before = compute_potential_energy_tiled(p, G, 0.0, 0, N);
            touched.push_back(i);
            m_before.push_back(m_old);
            v_before.insert(v_before.end(), {h_vx(i), h_vy(i), h_vz(i)});
        }

        // Update particle properties
        h_m(i) = res.mass;
        h_sage(i) = age_new;
        h_cmass(i) = res.mass_core;
        h_srad(i) = res.radius;       // collision radii follow the evolution
        h_lum(i) = res.luminosity;

        // Type change detection
        if (res.kw != kw) {
            n_type_change++;
            // Natal kick of a new NS or BH (see natal_kick above); the remnant's velocity only, the ejecta carry the
            // recoil (booked below in sink_PL and E_sse). Before 2026-10-12: a fixed 265 (NS) or 50 (BH) km/s along
            // the current velocity, scaled by 1 - m_rem / m_old
            if ((res.kw == 13 || res.kw == 14) && kw < 13 && g_collisional.sse_natal_kicks == "hobbs") {
                double dv[3];
                natal_kick(h_pid(i), g_collisional.kick_seed, res.kw, res.fbfac, res.ecsn, dv);
                h_vx(i) += dv[0];
                h_vy(i) += dv[1];
                h_vz(i) += dv[2];
                std::printf("SSE natal kick: pid %d kw %d -> %d m %.6g Msun f_fb %.6g ecsn %d dv %.9e %.9e %.9e km/s\n",
                            h_pid(i), kw, res.kw, res.mass, res.fbfac, (int)res.ecsn, dv[0], dv[1], dv[2]);
            }
        }

        h_stype(i) = res.kw;
        changed = true;
    }

    if (!changed) return;

    // Exact sinks: E_sse = E(before) - E(after) with the potential of every pair, so E_total + sinks is
    // conserved through winds and natal kicks; P and L carried off by the lost mass (and the kick recoil)
    double KE_change = 0.0;
    for (size_t k = 0; k < touched.size(); k++) {
        int i = touched[k];
        const double* vb = &v_before[3 * k];
        double r[3] = {h_x(i), h_y(i), h_z(i)}, va[3] = {h_vx(i), h_vy(i), h_vz(i)};
        KE_change += 0.5 * h_m(i) * (va[0]*va[0] + va[1]*va[1] + va[2]*va[2])
                   - 0.5 * m_before[k] * (vb[0]*vb[0] + vb[1]*vb[1] + vb[2]*vb[2]);
        books.sink_PL(m_before[k], r, vb);
        books.sink_PL(h_m(i), r, va, -1.0);
    }

    // Copy back to device
    Kokkos::deep_copy(p.mass, h_m);
    Kokkos::deep_copy(p.vx, h_vx);
    Kokkos::deep_copy(p.vy, h_vy);
    Kokkos::deep_copy(p.vz, h_vz);
    p.zero_kick_compensation();   // kicked velocities are new
    if (!touched.empty())
        books.E_sse += -KE_change + PE_before - compute_potential_energy_tiled(p, G, 0.0, 0, N);
    Kokkos::deep_copy(p.stellar_type, h_stype);
    Kokkos::deep_copy(p.stellar_age, h_sage);
    Kokkos::deep_copy(p.core_mass, h_cmass);
    Kokkos::deep_copy(p.stellar_radius, h_srad);
    Kokkos::deep_copy(p.luminosity, h_lum);
    StellarPropsCache::get().invalidate();   // types and radii changed

    if (total_dm > 0 || n_type_change > 0) {
        std::printf("SSE: dm_total=%.4e M_sun, %d type changes, E_sse=%.4e\n",
                    total_dm, n_type_change, books.E_sse);
    }
}
