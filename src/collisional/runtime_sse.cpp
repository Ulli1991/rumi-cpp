#include "runtime_sse.h"
#include "constants.h"
#include "diagnostics.h"
#include <cmath>
#include <cstdio>
#include <vector>

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

    bool changed = false;
    double total_dm = 0;
    int n_type_change = 0;
    // State before the update, for the exact energy, momentum and angular momentum sinks
    std::vector<int> touched;
    std::vector<double> m_before, v_before;

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
            touched.push_back(i);
            m_before.push_back(m_old);
            v_before.insert(v_before.end(), {h_vx(i), h_vy(i), h_vz(i)});
        }

        // Update particle properties
        h_m(i) = res.mass;
        h_sage(i) = age_new;
        h_cmass(i) = res.mass_core;

        // Type change detection
        if (res.kw != kw) {
            n_type_change++;
            // Natal kick for newly-formed NS/BH
            if ((res.kw == 13 || res.kw == 14) && kw < 13) {
                // Simplified natal kick: Hobbs et al. (2005) Maxwellian with sigma=265 km/s for NS
                // BH kicks reduced by fallback fraction
                double sigma = (res.kw == 13) ? 265.0 : 50.0;  // km/s
                // Deterministic kick magnitude from mass ratio (instead of random)
                double fb = (res.kw == 14) ? std::max(0.0, 1.0 - res.mass / m_old) : 0.0;
                double v_kick = sigma * (1.0 - fb);
                // Apply kick in current velocity direction (simplified)
                double v_mag = std::sqrt(h_vx(i)*h_vx(i) + h_vy(i)*h_vy(i) + h_vz(i)*h_vz(i));
                if (v_mag > 0) {
                    h_vx(i) += v_kick * h_vx(i) / v_mag;
                    h_vy(i) += v_kick * h_vy(i) / v_mag;
                    h_vz(i) += v_kick * h_vz(i) / v_mag;
                } else {
                    h_vz(i) += v_kick;
                }
            }
        }

        h_stype(i) = res.kw;
        changed = true;
    }

    if (!changed) return;

    // Exact sinks: E_sse = E(before) - E(after) with the potential of every pair, so E_total + sinks is
    // conserved through winds and natal kicks; P and L carried off by the lost mass (and the kick recoil)
    double PE_before = 0.0, KE_change = 0.0;
    if (!touched.empty()) PE_before = compute_potential_energy_tiled(p, G, 0.0, 0, N);
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

    if (total_dm > 0 || n_type_change > 0) {
        std::printf("SSE: dm_total=%.4e M_sun, %d type changes, E_sse=%.4e\n",
                    total_dm, n_type_change, books.E_sse);
    }
}
