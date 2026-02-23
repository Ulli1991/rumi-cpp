#include "runtime_sse.h"
#include "constants.h"
#include <cmath>
#include <cstdio>

void advance_stellar_evolution(Particles& p, SSEContext& sse, double dt_myr,
                                EnergyBookkeeping& books, double G) {
    if (dt_myr <= 0.0) return;

    int N = p.N;

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
        if (dm > 0) {
            // Energy correction: kinetic energy of lost mass
            double v2 = h_vx(i)*h_vx(i) + h_vy(i)*h_vy(i) + h_vz(i)*h_vz(i);
            books.E_sse += 0.5 * dm * v2;
            total_dm += dm;
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

    // Copy back to device
    Kokkos::deep_copy(p.mass, h_m);
    Kokkos::deep_copy(p.vx, h_vx);
    Kokkos::deep_copy(p.vy, h_vy);
    Kokkos::deep_copy(p.vz, h_vz);
    Kokkos::deep_copy(p.stellar_type, h_stype);
    Kokkos::deep_copy(p.stellar_age, h_sage);
    Kokkos::deep_copy(p.core_mass, h_cmass);

    if (total_dm > 0 || n_type_change > 0) {
        std::printf("SSE: dm_total=%.4e M_sun, %d type changes, E_sse=%.4e\n",
                    total_dm, n_type_change, books.E_sse);
    }
}
