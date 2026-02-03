#include "initial_conditions.h"
#include "constants.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <algorithm>
#include <vector>
#include <string>

void init_outer_planets(Particles& p) {
    p.allocate(5);

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

    for (int i = 0; i < 5; i++) {
        p.x[i]  = pos_au[i][0] * LENGTH_UNIT;
        p.y[i]  = pos_au[i][1] * LENGTH_UNIT;
        p.z[i]  = pos_au[i][2] * LENGTH_UNIT;
        p.vx[i] = vel_aud[i][0] * VELOCITY_UNIT;
        p.vy[i] = vel_aud[i][1] * VELOCITY_UNIT;
        p.vz[i] = vel_aud[i][2] * VELOCITY_UNIT;
        p.mass[i] = mass_msun[i] * MASS_UNIT;
    }
}

// =========================================================================
// Stellar evolution helpers (SSE-based, matching JAX version)
// =========================================================================

// Main sequence lifetime using SSE formulae (Hurley et al. 2000 / Tout et al. 1996)
static double main_sequence_lifetime(double m_init, double Z) {
    double lzs = std::log10(Z / 0.02);

    // SSE coefficients from zdata.h
    double a17 = 1593.890 + lzs * (2053.038 + lzs * (1231.226 + lzs * 232.7785));
    double a18 = 2706.708 + lzs * (1483.131 + lzs * (577.2723 + lzs * 74.11230));
    double a19 = 146.6143 + lzs * (-104.8442 + lzs * (-67.95374 + lzs * (-13.91127)));
    double a20 = 0.0414196 + lzs * (0.0456488 + lzs * (0.0295854 + lzs * 0.00557148));
    double a21 = 0.3426349;

    double m = m_init;
    double num = a17 + a18 * std::pow(m, 4) + a19 * std::pow(m, 5.5) + std::pow(m, 7);
    double den = a20 * m * m + a21 * std::pow(m, 7);
    double t_ms_myr = num / den;
    return t_ms_myr / 1000.0; // Gyr
}

// Turn-off mass at given age (binary search)
static double turnoff_mass(double age_gyr, double Z) {
    double m_low = 0.1, m_high = 100.0;
    while (m_high - m_low > 0.001) {
        double m_mid = (m_low + m_high) / 2.0;
        if (main_sequence_lifetime(m_mid, Z) > age_gyr)
            m_low = m_mid;
        else
            m_high = m_mid;
    }
    return (m_low + m_high) / 2.0;
}

// Remnant type and mass (SSE prescriptions)
struct Remnant {
    char type; // 'W' = WD, 'N' = NS, 'B' = BH
    double mass;
};

static Remnant remnant_type_and_mass(double m_init, double Z) {
    const double mch = 1.44;
    const double M_WD_max = 8.0;
    const double M_NS_max = 25.0;

    if (m_init < M_WD_max) {
        double mc;
        if (Z >= 1.0e-08) {
            mc = std::min(0.36 + 0.104 * m_init, 0.58 + 0.061 * m_init);
            mc = std::max(0.54 + 0.042 * m_init, mc);
            if (m_init < 1.0) mc = 0.46;
        } else {
            mc = std::min(0.29 + 0.178 * m_init, 0.65 + 0.062 * m_init);
            mc = std::max(0.54 + 0.073 * m_init, mc);
        }
        return {'W', std::min(mc, mch)};
    } else if (m_init < M_NS_max) {
        double mc_core = 0.15 * m_init + 0.5;
        double m_ns;
        if (mc_core < 2.5)
            m_ns = 0.161767 * mc_core + 1.067055;
        else
            m_ns = 0.314154 * mc_core + 0.686088;
        m_ns = std::clamp(m_ns, 1.1, 2.5);
        return {'N', m_ns};
    } else {
        double m_bh;
        if (Z >= 0.01)
            m_bh = 0.2 * m_init + 2.0;
        else
            m_bh = 0.4 * m_init;
        m_bh = std::clamp(m_bh, 3.0, 50.0);
        return {'B', m_bh};
    }
}

// Evolve star from initial mass to mass at given age
static Remnant evolve_stellar_mass(double m_init, double age_gyr, double Z) {
    double t_ms = main_sequence_lifetime(m_init, Z);
    if (age_gyr < t_ms)
        return {'M', m_init}; // still on MS
    return remnant_type_and_mass(m_init, Z);
}

// Sample from Kroupa (2001) IMF
static std::vector<double> sample_kroupa_imf(int N, std::mt19937_64& rng,
                                              double m_min = 0.08, double m_max = 100.0) {
    const double m_break = 0.5;
    const double alpha1 = 1.3, alpha2 = 2.3;
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    double I1 = 0;
    if (m_min < m_break)
        I1 = (std::pow(m_break, 1 - alpha1) - std::pow(m_min, 1 - alpha1)) / (1 - alpha1);

    double C = std::pow(m_break, alpha2 - alpha1);
    double I2 = 0;
    if (m_max > m_break) {
        double m_upper = std::min(m_max, 150.0);
        double m_lower = std::max(m_break, m_min);
        I2 = C * (std::pow(m_upper, 1 - alpha2) - std::pow(m_lower, 1 - alpha2)) / (1 - alpha2);
    }

    double I_total = I1 + I2;
    double p1 = I1 / I_total;

    std::vector<double> masses(N);
    for (int i = 0; i < N; i++) {
        if (uni(rng) < p1) {
            double u = uni(rng);
            double m1_term = std::pow(m_min, 1 - alpha1);
            double m2_term = std::pow(m_break, 1 - alpha1);
            masses[i] = std::pow(m1_term + u * (m2_term - m1_term), 1.0 / (1 - alpha1));
        } else {
            double u = uni(rng);
            double m1_term = std::pow(m_break, 1 - alpha2);
            double m2_term = std::pow(m_max, 1 - alpha2);
            masses[i] = std::pow(m1_term + u * (m2_term - m1_term), 1.0 / (1 - alpha2));
        }
    }
    return masses;
}

// =========================================================================
// Plummer sphere generator
// =========================================================================

void init_plummer(Particles& p, int N, double M_total_msun, double r_half_pc, int seed,
                  const std::string& mass_distribution, double age_gyr, double metallicity) {
    p.allocate(N);

    double M_total = M_total_msun * MASS_UNIT;
    double a_pc = r_half_pc / PLUMMER_SCALE_FACTOR;
    double a = a_pc * PC_TO_CM;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    // Generate masses
    if (mass_distribution == "equal") {
        double m_per = M_total / N;
        for (int i = 0; i < N; i++) p.mass[i] = m_per;
        std::printf("  Equal mass: %.4f M_sun per particle\n", M_total_msun / N);
    } else if (mass_distribution == "kroupa") {
        auto masses = sample_kroupa_imf(N, rng);
        double sum = 0;
        for (int i = 0; i < N; i++) sum += masses[i];
        for (int i = 0; i < N; i++) p.mass[i] = masses[i] * (M_total / sum);
        std::printf("  Raw Kroupa IMF (not evolved)\n");
        double mn = *std::min_element(masses.begin(), masses.end()) * (M_total_msun / sum);
        double mx = *std::max_element(masses.begin(), masses.end()) * (M_total_msun / sum);
        std::printf("    Mass range: %.2f - %.2f M_sun\n", mn, mx);
    } else if (mass_distribution == "kroupa_evolved") {
        auto m_initial = sample_kroupa_imf(N, rng);
        int n_ms = 0, n_wd = 0, n_ns = 0, n_bh = 0;
        std::vector<double> masses(N);
        for (int i = 0; i < N; i++) {
            auto rem = evolve_stellar_mass(m_initial[i], age_gyr, metallicity);
            masses[i] = rem.mass;
            switch (rem.type) {
                case 'M': n_ms++; break;
                case 'W': n_wd++; break;
                case 'N': n_ns++; break;
                case 'B': n_bh++; break;
            }
        }
        double sum = 0;
        for (int i = 0; i < N; i++) sum += masses[i];
        for (int i = 0; i < N; i++) p.mass[i] = masses[i] * (M_total / sum);

        double m_to = turnoff_mass(age_gyr, metallicity);
        std::printf("  Stellar population at %.1f Gyr (Z=%.3f):\n", age_gyr, metallicity);
        std::printf("    Turn-off mass: %.2f M_sun\n", m_to);
        std::printf("    Main sequence: %d (%.1f%%)\n", n_ms, 100.0 * n_ms / N);
        std::printf("    White dwarfs:  %d (%.1f%%)\n", n_wd, 100.0 * n_wd / N);
        std::printf("    Neutron stars: %d (%.1f%%)\n", n_ns, 100.0 * n_ns / N);
        std::printf("    Black holes:   %d (%.1f%%)\n", n_bh, 100.0 * n_bh / N);
        double mn = *std::min_element(masses.begin(), masses.end()) * (M_total_msun / sum);
        double mx = *std::max_element(masses.begin(), masses.end()) * (M_total_msun / sum);
        std::printf("    Mass range: %.2f - %.2f M_sun\n", mn, mx);
    }

    // Sample positions from Plummer distribution
    for (int i = 0; i < N; i++) {
        double U = std::clamp(uni(rng), 1e-10, 1.0 - 1e-10);
        double r = a / std::sqrt(std::pow(U, -2.0/3.0) - 1.0);

        double cos_theta = 2.0 * uni(rng) - 1.0;
        double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
        double phi = 2.0 * M_PI * uni(rng);

        p.x[i] = r * sin_theta * std::cos(phi);
        p.y[i] = r * sin_theta * std::sin(phi);
        p.z[i] = r * cos_theta;

        // Velocity: rejection sampling
        double v_escape = std::sqrt(2.0 * G_CGS * M_total / std::sqrt(r*r + a*a));
        while (true) {
            double q = uni(rng);
            double g_q = q * q * std::pow(1.0 - q * q, 3.5);
            if (uni(rng) * 0.1 < g_q) {
                double v_mag = q * v_escape;
                double vcos = 2.0 * uni(rng) - 1.0;
                double vsin = std::sqrt(1.0 - vcos * vcos);
                double vphi = 2.0 * M_PI * uni(rng);
                p.vx[i] = v_mag * vsin * std::cos(vphi);
                p.vy[i] = v_mag * vsin * std::sin(vphi);
                p.vz[i] = v_mag * vcos;
                break;
            }
        }
    }

    // Center of mass correction
    double com_x = 0, com_y = 0, com_z = 0;
    double com_vx = 0, com_vy = 0, com_vz = 0;
    for (int i = 0; i < N; i++) {
        com_x += p.mass[i] * p.x[i];
        com_y += p.mass[i] * p.y[i];
        com_z += p.mass[i] * p.z[i];
        com_vx += p.mass[i] * p.vx[i];
        com_vy += p.mass[i] * p.vy[i];
        com_vz += p.mass[i] * p.vz[i];
    }
    com_x /= M_total; com_y /= M_total; com_z /= M_total;
    com_vx /= M_total; com_vy /= M_total; com_vz /= M_total;

    for (int i = 0; i < N; i++) {
        p.x[i] -= com_x; p.y[i] -= com_y; p.z[i] -= com_z;
        p.vx[i] -= com_vx; p.vy[i] -= com_vy; p.vz[i] -= com_vz;
    }
}
