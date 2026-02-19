#include "initial_conditions.h"
#include "constants.h"
#ifdef ENABLE_COLLISIONAL
#include "collisional_config.h"
#include "kepler.h"
#define COLL_ENABLED g_collisional.enabled
#else
#define COLL_ENABLED false
#endif
#include "sse.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <random>
#include <algorithm>
#include <vector>
#include <string>

void init_outer_planets(Particles& p) {
    p.allocate(5, COLL_ENABLED);

    double pos_au[][3] = {
        {0.0, 0.0, 0.0},
        {-3.5023653, -3.8169847, -1.5507963},
        {9.0755314, -3.0458353, -1.6483708},
        {8.3101420, -16.2901086, -7.2521278},
        {11.4707666, -25.7294829, -10.8169456}
    };
    double vel_aud[][3] = {
        {0.0, 0.0, 0.0},
        {0.00565429, -0.00412490, -0.00190589},
        {0.00168318, 0.00483525, 0.00192462},
        {0.00354178, 0.00137102, 0.00055029},
        {0.00288930, 0.00114527, 0.00039677}
    };
    double mass_msun[] = {
        1.0, 9.54786104043e-4, 2.85583733151e-4,
        4.37273164546e-5, 5.17759138449e-5
    };

    auto h_x = Kokkos::create_mirror_view(p.x);
    auto h_y = Kokkos::create_mirror_view(p.y);
    auto h_z = Kokkos::create_mirror_view(p.z);
    auto h_vx = Kokkos::create_mirror_view(p.vx);
    auto h_vy = Kokkos::create_mirror_view(p.vy);
    auto h_vz = Kokkos::create_mirror_view(p.vz);
    auto h_mass = Kokkos::create_mirror_view(p.mass);

    // Convert: positions AU→pc, velocities AU/day→km/s, masses already in M_sun
    constexpr double AU_PER_DAY_TO_KMS = 149597870.7 / 86400.0;  // 1 AU/day ≈ 1731.457 km/s
    for (int i = 0; i < 5; i++) {
        h_x(i)  = pos_au[i][0] * AU_TO_PC;
        h_y(i)  = pos_au[i][1] * AU_TO_PC;
        h_z(i)  = pos_au[i][2] * AU_TO_PC;
        h_vx(i) = vel_aud[i][0] * AU_PER_DAY_TO_KMS;
        h_vy(i) = vel_aud[i][1] * AU_PER_DAY_TO_KMS;
        h_vz(i) = vel_aud[i][2] * AU_PER_DAY_TO_KMS;
        h_mass(i) = mass_msun[i];
    }

    Kokkos::deep_copy(p.x, h_x);
    Kokkos::deep_copy(p.y, h_y);
    Kokkos::deep_copy(p.z, h_z);
    Kokkos::deep_copy(p.vx, h_vx);
    Kokkos::deep_copy(p.vy, h_vy);
    Kokkos::deep_copy(p.vz, h_vz);
    Kokkos::deep_copy(p.mass, h_mass);
}

// =========================================================================
// Stellar evolution via SSE (Hurley et al. 2000)
// =========================================================================

static double turnoff_mass(double age_gyr, double Z) {
    // Binary search: find the mass where SSE transitions from MS to post-MS
    SSEContext sse(Z);
    double age_myr = age_gyr * 1000.0;
    double m_low = 0.1, m_high = 100.0;
    while (m_high - m_low > 0.001) {
        double m_mid = (m_low + m_high) / 2.0;
        SSEResult res = sse.evolve(m_mid, age_myr);
        if (res.kw <= 1) // still on MS
            m_low = m_mid;
        else
            m_high = m_mid;
    }
    return (m_low + m_high) / 2.0;
}

static std::vector<double> sample_kroupa_imf(int N, std::mt19937_64& rng,
                                              double m_min = 0.08, double m_max = 100.0) {
    const double m_break = 0.5, alpha1 = 1.3, alpha2 = 2.3;
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    double I1 = (m_min < m_break)
        ? (std::pow(m_break, 1-alpha1) - std::pow(m_min, 1-alpha1)) / (1-alpha1) : 0;
    double C = std::pow(m_break, alpha2 - alpha1);
    double I2 = (m_max > m_break)
        ? C * (std::pow(std::min(m_max, 150.0), 1-alpha2) - std::pow(std::max(m_break, m_min), 1-alpha2)) / (1-alpha2) : 0;
    double p1 = I1 / (I1 + I2);

    std::vector<double> masses(N);
    for (int i = 0; i < N; i++) {
        if (uni(rng) < p1) {
            double u = uni(rng);
            masses[i] = std::pow(std::pow(m_min, 1-alpha1) + u * (std::pow(m_break, 1-alpha1) - std::pow(m_min, 1-alpha1)), 1.0/(1-alpha1));
        } else {
            double u = uni(rng);
            masses[i] = std::pow(std::pow(m_break, 1-alpha2) + u * (std::pow(m_max, 1-alpha2) - std::pow(m_break, 1-alpha2)), 1.0/(1-alpha2));
        }
    }
    return masses;
}

// =========================================================================
// Plummer sphere generator
// =========================================================================

void init_plummer(Particles& p, int N, double M_total_msun, double r_half_pc, int seed,
                  const std::string& mass_distribution, double age_gyr, double metallicity) {
    p.allocate(N, COLL_ENABLED);

    double M_total = M_total_msun;            // M_sun = code mass unit
    double a_pc = r_half_pc / PLUMMER_SCALE_FACTOR;
    double a = a_pc;                          // pc = code length unit

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    auto h_x = Kokkos::create_mirror_view(p.x);
    auto h_y = Kokkos::create_mirror_view(p.y);
    auto h_z = Kokkos::create_mirror_view(p.z);
    auto h_vx = Kokkos::create_mirror_view(p.vx);
    auto h_vy = Kokkos::create_mirror_view(p.vy);
    auto h_vz = Kokkos::create_mirror_view(p.vz);
    auto h_mass = Kokkos::create_mirror_view(p.mass);

    // Generate masses
    if (mass_distribution == "equal") {
        double m_per = M_total / N;
        for (int i = 0; i < N; i++) h_mass(i) = m_per;
        std::printf("  Equal mass: %.4f M_sun per particle\n", M_total_msun / N);
    } else if (mass_distribution == "kroupa") {
        auto masses = sample_kroupa_imf(N, rng);
        double sum = 0;
        for (int i = 0; i < N; i++) sum += masses[i];
        for (int i = 0; i < N; i++) h_mass(i) = masses[i] * (M_total / sum);
        double mn = *std::min_element(masses.begin(), masses.end()) * (M_total_msun / sum);
        double mx = *std::max_element(masses.begin(), masses.end()) * (M_total_msun / sum);
        std::printf("  Raw Kroupa IMF (not evolved)\n");
        std::printf("    Mass range: %.2f - %.2f M_sun\n", mn, mx);
    } else if (mass_distribution == "kroupa_evolved") {
        SSEContext sse(metallicity);
        double age_myr = age_gyr * 1000.0;
        auto m_initial = sample_kroupa_imf(N, rng);
        int n_ms = 0, n_evolved = 0, n_wd = 0, n_ns = 0, n_bh = 0;
        std::vector<double> masses(N);
        std::vector<int> kw_types(N);
        for (int i = 0; i < N; i++) {
            SSEResult res = sse.evolve(m_initial[i], age_myr);
            masses[i] = res.mass;
            kw_types[i] = res.kw;
            if (res.kw <= 1) n_ms++;
            else if (res.kw >= 2 && res.kw <= 9) n_evolved++;
            else if (res.kw >= 10 && res.kw <= 12) n_wd++;
            else if (res.kw == 13) n_ns++;
            else if (res.kw == 14) n_bh++;
        }
        // Use physical SSE masses directly (no rescaling)
        // M_total is determined by the IMF sampling + SSE, not prescribed
        M_total_msun = 0;
        for (int i = 0; i < N; i++) M_total_msun += masses[i];
        M_total = M_total_msun;
        for (int i = 0; i < N; i++) h_mass(i) = masses[i];

        double m_to = turnoff_mass(age_gyr, metallicity);
        std::printf("  Stellar population at %.1f Gyr (Z=%.4f, full SSE):\n", age_gyr, metallicity);
        std::printf("    Turn-off mass: %.2f M_sun\n", m_to);
        std::printf("    Total mass:    %.0f M_sun\n", M_total_msun);
        std::printf("    Main sequence: %d (%.1f%%)\n", n_ms, 100.0 * n_ms / N);
        std::printf("    Evolved:       %d (%.1f%%)\n", n_evolved, 100.0 * n_evolved / N);
        std::printf("    White dwarfs:  %d (%.1f%%)\n", n_wd, 100.0 * n_wd / N);
        std::printf("    Neutron stars: %d (%.1f%%)\n", n_ns, 100.0 * n_ns / N);
        std::printf("    Black holes:   %d (%.1f%%)\n", n_bh, 100.0 * n_bh / N);
        double mn = *std::min_element(masses.begin(), masses.end());
        double mx = *std::max_element(masses.begin(), masses.end());
        std::printf("    Mass range: %.4f - %.4f M_sun\n", mn, mx);
        // Per-type mass ranges
        double wd_mn = 1e30, wd_mx = 0, ns_mn = 1e30, ns_mx = 0, bh_mn = 1e30, bh_mx = 0;
        for (int i = 0; i < N; i++) {
            if (kw_types[i] >= 10 && kw_types[i] <= 12) { wd_mn = std::min(wd_mn, masses[i]); wd_mx = std::max(wd_mx, masses[i]); }
            else if (kw_types[i] == 13) { ns_mn = std::min(ns_mn, masses[i]); ns_mx = std::max(ns_mx, masses[i]); }
            else if (kw_types[i] == 14) { bh_mn = std::min(bh_mn, masses[i]); bh_mx = std::max(bh_mx, masses[i]); }
        }
        if (n_wd > 0) std::printf("    WD masses:  %.4f - %.4f M_sun\n", wd_mn, wd_mx);
        if (n_ns > 0) std::printf("    NS masses:  %.4f - %.4f M_sun\n", ns_mn, ns_mx);
        if (n_bh > 0) std::printf("    BH masses:  %.4f - %.4f M_sun\n", bh_mn, bh_mx);
    }

    // Crossing time: t_cr = sqrt(R_h^3 / (G * M)) in code TU, then convert to Myr
    double t_cr_code = std::sqrt(r_half_pc * r_half_pc * r_half_pc / (G_CODE * M_total_msun));
    double t_cr_myr = t_cr_code * TIME_UNIT_MYR;
    std::printf("  Crossing time: %.3f Myr\n", t_cr_myr);

    // Sample positions and velocities
    for (int i = 0; i < N; i++) {
        double U = std::clamp(uni(rng), 1e-10, 1.0 - 1e-10);
        double r = a / std::sqrt(std::pow(U, -2.0/3.0) - 1.0);

        double cos_theta = 2.0 * uni(rng) - 1.0;
        double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
        double phi = 2.0 * M_PI * uni(rng);

        h_x(i) = r * sin_theta * std::cos(phi);
        h_y(i) = r * sin_theta * std::sin(phi);
        h_z(i) = r * cos_theta;

        double v_escape = std::sqrt(2.0 * G_CODE * M_total / std::sqrt(r*r + a*a));
        while (true) {
            double q = uni(rng);
            double g_q = q * q * std::pow(1.0 - q * q, 3.5);
            if (uni(rng) * 0.1 < g_q) {
                double v_mag = q * v_escape;
                double vcos = 2.0 * uni(rng) - 1.0;
                double vsin = std::sqrt(1.0 - vcos * vcos);
                double vphi = 2.0 * M_PI * uni(rng);
                h_vx(i) = v_mag * vsin * std::cos(vphi);
                h_vy(i) = v_mag * vsin * std::sin(vphi);
                h_vz(i) = v_mag * vcos;
                break;
            }
        }
    }

    // Center of mass correction
    double com_x = 0, com_y = 0, com_z = 0;
    double com_vx = 0, com_vy = 0, com_vz = 0;
    for (int i = 0; i < N; i++) {
        com_x += h_mass(i) * h_x(i);   com_y += h_mass(i) * h_y(i);   com_z += h_mass(i) * h_z(i);
        com_vx += h_mass(i) * h_vx(i); com_vy += h_mass(i) * h_vy(i); com_vz += h_mass(i) * h_vz(i);
    }
    com_x /= M_total; com_y /= M_total; com_z /= M_total;
    com_vx /= M_total; com_vy /= M_total; com_vz /= M_total;
    for (int i = 0; i < N; i++) {
        h_x(i) -= com_x; h_y(i) -= com_y; h_z(i) -= com_z;
        h_vx(i) -= com_vx; h_vy(i) -= com_vy; h_vz(i) -= com_vz;
    }

    Kokkos::deep_copy(p.x, h_x);   Kokkos::deep_copy(p.y, h_y);   Kokkos::deep_copy(p.z, h_z);
    Kokkos::deep_copy(p.vx, h_vx); Kokkos::deep_copy(p.vy, h_vy); Kokkos::deep_copy(p.vz, h_vz);
    Kokkos::deep_copy(p.mass, h_mass);
}

// =========================================================================
// Isolated binary test case
// =========================================================================

void init_binary_test(Particles& p, double a_pc, double e,
                      double m1_msun, double m2_msun) {
    p.allocate(2, COLL_ENABLED);

    double mu = G_CODE * (m1_msun + m2_msun);
    double M_total = m1_msun + m2_msun;

    // At mean anomaly = 0 (periapsis), inc=Omega=omega=0:
    //   r_rel = (a*(1-e), 0, 0),  v_rel = (0, sqrt(mu/a * (1+e)/(1-e)), 0)
    double r_peri = a_pc * (1.0 - e);
    double v_peri = std::sqrt(mu / a_pc * (1.0 + e) / (1.0 - e + 1e-30));

    // Split into two particles around COM at origin
    double f1 = m2_msun / M_total;  // particle 0 gets -f1 * r_rel
    double f2 = m1_msun / M_total;  // particle 1 gets +f2 * r_rel

    auto h_x  = Kokkos::create_mirror_view(p.x);
    auto h_y  = Kokkos::create_mirror_view(p.y);
    auto h_z  = Kokkos::create_mirror_view(p.z);
    auto h_vx = Kokkos::create_mirror_view(p.vx);
    auto h_vy = Kokkos::create_mirror_view(p.vy);
    auto h_vz = Kokkos::create_mirror_view(p.vz);
    auto h_mass = Kokkos::create_mirror_view(p.mass);

    h_x(0) = -f1 * r_peri;  h_y(0) = 0.0;  h_z(0) = 0.0;
    h_x(1) =  f2 * r_peri;  h_y(1) = 0.0;  h_z(1) = 0.0;
    h_vx(0) = 0.0; h_vy(0) = -f1 * v_peri; h_vz(0) = 0.0;
    h_vx(1) = 0.0; h_vy(1) =  f2 * v_peri; h_vz(1) = 0.0;
    h_mass(0) = m1_msun;
    h_mass(1) = m2_msun;

    Kokkos::deep_copy(p.x, h_x);   Kokkos::deep_copy(p.y, h_y);   Kokkos::deep_copy(p.z, h_z);
    Kokkos::deep_copy(p.vx, h_vx); Kokkos::deep_copy(p.vy, h_vy); Kokkos::deep_copy(p.vz, h_vz);
    Kokkos::deep_copy(p.mass, h_mass);

    // Informational output
    double period_code = 2.0 * M_PI * std::sqrt(a_pc * a_pc * a_pc / mu);
    double period_yr = period_code * TIME_UNIT_YR;
    double r_apo  = a_pc * (1.0 + e);

    std::printf("Binary test: m1=%.1f m2=%.1f M_sun, a=%.4e pc (%.2f AU)\n",
                m1_msun, m2_msun, a_pc, a_pc / AU_TO_PC);
    std::printf("  e=%.4f, r_peri=%.4e pc (%.2f AU), r_apo=%.4e pc (%.2f AU)\n",
                e, r_peri, r_peri / AU_TO_PC, r_apo, r_apo / AU_TO_PC);
    std::printf("  Period = %.4f yr (%.4e Myr)\n", period_yr, period_yr / 1e6);
    std::printf("  Separation at periapsis: %.4e pc\n", r_peri);
}
