#include "initial_conditions.h"
#include "constants.h"
#include <cmath>
#include <random>

void init_outer_planets(Particles& p) {
    p.allocate(5);

    // Positions in AU -> cm
    double pos_au[][3] = {
        {0.0, 0.0, 0.0},
        {-3.5023653, -3.8169847, -1.5507963},
        {9.0755314, -3.0458353, -1.6483708},
        {8.3101420, -16.2901086, -7.2521278},
        {11.4707666, -25.7294829, -10.8169456}
    };
    // Velocities in AU/day -> cm/s
    double vel_aud[][3] = {
        {0.0, 0.0, 0.0},
        {0.00565429, -0.00412490, -0.00190589},
        {0.00168318, 0.00483525, 0.00192462},
        {0.00354178, 0.00137102, 0.00055029},
        {0.00288930, 0.00114527, 0.00039677}
    };
    // Masses in M_sun -> g
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

void init_plummer(Particles& p, int N, double M_total_msun, double r_half_pc, int seed) {
    p.allocate(N);

    double M_total = M_total_msun * MASS_UNIT;
    double a_pc = r_half_pc / PLUMMER_SCALE_FACTOR;
    double a = a_pc * PC_TO_CM;
    double m_per = M_total / N;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    for (int i = 0; i < N; i++) p.mass[i] = m_per;

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
