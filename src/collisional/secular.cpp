#include "secular.h"
#include "kepler.h"
#include "constants.h"
#include <cmath>
#include <algorithm>

// ============================================================================
// Peters (1964) orbit-averaged GW dissipation
// ============================================================================

static double peters_F(double e) {
    double e2 = e * e;
    return (1.0 + 73.0/24.0 * e2 + 37.0/96.0 * e2*e2) / std::pow(1.0 - e2, 3.5);
}

static double peters_G(double e) {
    double e2 = e * e;
    return (1.0 + 7.0/8.0 * e2) / std::pow(1.0 - e2, 2.5);
}

static double peters_beta(double m1, double m2, double G, double c) {
    // beta = 64/5 * G^3 * m1 * m2 * (m1+m2) / c^5
    double c5 = c * c * c * c * c;
    return 64.0/5.0 * G*G*G * m1 * m2 * (m1 + m2) / c5;
}

// ============================================================================
// Secular binary evolution
// ============================================================================

void secular_evolve_binary(SecularBinary& bin, double dt, double G, double c_code, int pn_order) {
    double beta = peters_beta(bin.m1, bin.m2, G, c_code);
    double c2 = c_code * c_code;
    double mu = G * (bin.m1 + bin.m2);

    // Time integration with leapfrog (auxiliary variable method)
    int n_steps = std::max(1, (int)(dt / (0.01 * std::fabs(bin.a / (bin.da_dt + 1e-30)))));
    n_steps = std::min(n_steps, 10000);
    double h = dt / n_steps;

    for (int step = 0; step < n_steps; step++) {
        double a = bin.a;
        double e = bin.e;
        double e2 = e * e;
        double one_me2 = 1.0 - e2;

        // GW dissipation rates (Peters 1964)
        if (pn_order & 4) {  // PN2.5
            bin.da_dt = -beta / (a * a * a) * peters_F(e);
            bin.de_dt = -304.0/15.0 * beta / (a * a * a * a * std::pow(one_me2, 2.5)) * e * peters_G(e);
        } else {
            bin.da_dt = 0.0;
            bin.de_dt = 0.0;
        }

        // PN1.0 periapsis advance
        if (pn_order & 1) {
            double P = 2.0 * M_PI * std::sqrt(a * a * a / mu);
            bin.domega_dt = 6.0 * M_PI * mu / (c2 * P * a * one_me2);
        } else {
            bin.domega_dt = 0.0;
        }

        // Leapfrog step
        bin.a += h * bin.da_dt;
        bin.e += h * bin.de_dt;
        bin.omega += h * bin.domega_dt;

        // Clamp eccentricity
        bin.e = std::max(0.0, std::min(bin.e, 0.9999));

        // Check for coalescence
        if (bin.a < 0.0 || bin.a * (1.0 - bin.e) < 6.0 * mu / c2) {
            bin.a = 0.0;  // signal merger
            break;
        }
    }
}

// ============================================================================
// Kozai-Lidov secular triple evolution
// ============================================================================

void secular_evolve_triple(SecularTriple& tri, double dt, double G, double c_code, int pn_order) {
    double mu_in = G * (tri.inner.m1 + tri.inner.m2);
    double mu_out = G * (tri.inner.m1 + tri.inner.m2 + tri.m3);

    int n_steps = 100;
    double h = dt / n_steps;

    for (int step = 0; step < n_steps; step++) {
        double a_in = tri.inner.a;
        double e_in = tri.inner.e;
        double a_out = tri.a_out;
        double e_out = tri.e_out;

        if (a_in <= 0 || a_out <= 0) break;

        double e_in2 = e_in * e_in;
        double e_out2 = e_out * e_out;
        double cos_i = tri.cos_i_mut;
        double cos_i2 = cos_i * cos_i;

        // Quadrupole Kozai-Lidov timescale
        double P_in = 2.0 * M_PI * std::sqrt(a_in * a_in * a_in / mu_in);
        double P_out = 2.0 * M_PI * std::sqrt(a_out * a_out * a_out / mu_out);
        double t_KL = P_out * P_out / P_in * (tri.inner.m1 + tri.inner.m2 + tri.m3) / tri.m3 *
                      std::pow(1.0 - e_out2, 1.5);

        // Quadrupole de_in/dt
        double de_in_dt = 15.0 / (4.0 * t_KL) * e_in * std::sqrt(1.0 - e_in2) *
                          sin(2.0 * tri.inner.omega) * (1.0 - cos_i2);

        // Quadrupole d(cos_i)/dt
        double dcos_i_dt = -15.0 / (4.0 * t_KL) * e_in2 / std::sqrt(1.0 - e_in2) *
                           sin(2.0 * tri.inner.omega) * cos_i * (1.0 - cos_i2);

        // Quadrupole domega_in/dt
        double domega_dt = 3.0 / t_KL * (2.0 * (1.0 - e_in2) + 5.0 * e_in2 * sin(tri.inner.omega) * sin(tri.inner.omega)) *
                           (cos_i2 - 1.0/3.0) / (2.0 * std::sqrt(1.0 - e_in2));

        // Add PN1.0 precession to suppress high-e oscillations
        if (pn_order & 1) {
            double c2 = c_code * c_code;
            domega_dt += 6.0 * M_PI * mu_in / (c2 * P_in * a_in * (1.0 - e_in2));
        }

        // GW dissipation on inner binary
        double da_in_dt = 0.0, de_in_gw_dt = 0.0;
        if (pn_order & 4) {
            double beta = peters_beta(tri.inner.m1, tri.inner.m2, G, c_code);
            da_in_dt = -beta / (a_in * a_in * a_in) * peters_F(e_in);
            de_in_gw_dt = -304.0/15.0 * beta / (a_in * a_in * a_in * a_in *
                          std::pow(1.0 - e_in2, 2.5)) * e_in * peters_G(e_in);
        }

        // Integrate
        tri.inner.a += h * da_in_dt;
        tri.inner.e += h * (de_in_dt + de_in_gw_dt);
        tri.inner.omega += h * domega_dt;
        tri.cos_i_mut += h * dcos_i_dt;

        // Clamp
        tri.inner.e = std::max(0.0, std::min(tri.inner.e, 0.9999));
        tri.cos_i_mut = std::max(-1.0, std::min(tri.cos_i_mut, 1.0));

        if (tri.inner.a <= 0.0) break;
    }
}

// ============================================================================
// Transition checks
// ============================================================================

bool should_be_secular(const Subsystem& sub, double dt_step, int n_orb_threshold) {
    if (sub.type != SubsystemType::BINARY && sub.type != SubsystemType::TRIPLE_STABLE)
        return false;
    return sub.n_orb_per_dt >= n_orb_threshold;
}

void secular_to_cartesian(const SecularBinary& bin, double G,
                          double r[3], double v[3]) {
    double mu = G * (bin.m1 + bin.m2);
    // Use a random mean anomaly (uniformly distributed phase)
    // In practice this is called with the current phase
    double M = 0.0;  // placeholder — caller should provide
    elements_to_cartesian(bin.a, bin.e, bin.inc, bin.Omega, bin.omega, M, mu, r, v);
}
