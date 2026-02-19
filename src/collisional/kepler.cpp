#include "kepler.h"
#include <cmath>
#include <algorithm>

// ============================================================================
// Kepler equation solver: Mikkola (1987) cubic start with BIFROST empirical
// correction (a_K = 6 - 0.768688675*M) + Newton-Raphson + bisection fallback
// ============================================================================

double solve_kepler(double M, double e, double tol) {
    // Reduce M to [0, 2*pi)
    M = std::fmod(M, 2.0 * M_PI);
    if (M < 0.0) M += 2.0 * M_PI;

    // Map to [0, pi] with sign tracking (Mikkola cubic is symmetric about pi)
    double sign = 1.0;
    double M_reduced = M;
    if (M_reduced > M_PI) {
        M_reduced = 2.0 * M_PI - M_reduced;
        sign = -1.0;
    }

    // Mikkola (1987) cubic starting value with BIFROST empirical correction
    // (Section 3.10): a_K = 6 - 0.768688675 * M makes the cubic exact at M=pi
    double a_K = 6.0 - 0.768688675 * M_reduced;
    // Guard against a_K becoming too small or negative for very large M_reduced
    if (a_K < 0.5) a_K = 0.5;
    double K = std::sqrt((1.0 - e) / (1.0 / a_K + 4.0 / 3.0 * e));
    double s = K * std::sinh(1.0 / 3.0 * std::asinh(M_reduced / ((1.0 - e) * K)));
    double E = M_reduced + e * (3.0 * s - 4.0 * s * s * s);

    // Restore full [0, 2*pi) range
    if (sign < 0.0) {
        E = 2.0 * M_PI - E;
    }

    // Newton-Raphson refinement (typically converges in 2-3 iterations)
    bool converged = false;
    for (int iter = 0; iter < 15; iter++) {
        double f = E - e * std::sin(E) - M;
        double fp = 1.0 - e * std::cos(E);
        double dE = -f / fp;
        E += dE;
        if (std::fabs(dE) < tol) {
            converged = true;
            break;
        }
    }

    // Bisection fallback if Newton-Raphson did not converge
    // (guaranteed convergence for any e in [0,1))
    if (!converged) {
        // Bracket: f(E) = E - e*sin(E) - M is monotonically increasing,
        // so f(0) = -M <= 0 and f(2*pi) = 2*pi - M >= 0
        double E_lo = 0.0;
        double E_hi = 2.0 * M_PI;

        // Tighten bracket using current best estimate
        double f_mid = E - e * std::sin(E) - M;
        if (f_mid > 0.0) {
            E_hi = E;
        } else {
            E_lo = E;
        }

        for (int iter = 0; iter < 64; iter++) {
            E = 0.5 * (E_lo + E_hi);
            double f = E - e * std::sin(E) - M;
            if (f > 0.0) {
                E_hi = E;
            } else {
                E_lo = E;
            }
            if (E_hi - E_lo < tol) break;
        }
        E = 0.5 * (E_lo + E_hi);
    }

    return E;
}

// ============================================================================
// Cartesian to orbital elements
// ============================================================================

OrbitalElements cartesian_to_elements(const double r[3], const double v[3], double mu) {
    OrbitalElements oe{};

    double r_mag = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    double v_mag = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);

    // Specific orbital energy
    oe.E_orb = 0.5 * v_mag * v_mag - mu / r_mag;

    // Angular momentum vector L = r x v
    double Lx = r[1]*v[2] - r[2]*v[1];
    double Ly = r[2]*v[0] - r[0]*v[2];
    double Lz = r[0]*v[1] - r[1]*v[0];
    oe.L_mag = std::sqrt(Lx*Lx + Ly*Ly + Lz*Lz);

    // Semi-major axis
    if (std::fabs(oe.E_orb) > 1e-30) {
        oe.a = -mu / (2.0 * oe.E_orb);
    } else {
        oe.a = 1e30; // parabolic
    }

    // Eccentricity from vis-viva + angular momentum
    double p = oe.L_mag * oe.L_mag / mu;  // semi-latus rectum
    if (oe.a > 0 && std::fabs(oe.a) > 1e-30) {
        oe.e = std::sqrt(std::max(0.0, 1.0 - p / oe.a));
    } else {
        // Hyperbolic
        oe.e = std::sqrt(1.0 + 2.0 * oe.E_orb * oe.L_mag * oe.L_mag / (mu * mu));
    }

    // Inclination
    oe.inc = std::acos(std::min(1.0, std::max(-1.0, Lz / (oe.L_mag + 1e-30))));

    // Node line N = z_hat x L
    double Nx = -Ly;
    double Ny = Lx;
    double N_mag = std::sqrt(Nx*Nx + Ny*Ny);

    // Longitude of ascending node
    if (N_mag > 1e-30) {
        oe.Omega = std::acos(std::min(1.0, std::max(-1.0, Nx / N_mag)));
        if (Ny < 0.0) oe.Omega = 2.0 * M_PI - oe.Omega;
    } else {
        oe.Omega = 0.0;
    }

    // Eccentricity vector e_vec = (v x L) / mu - r_hat
    double r_dot_v = r[0]*v[0] + r[1]*v[1] + r[2]*v[2];
    double ex = (v[1]*Lz - v[2]*Ly) / mu - r[0] / r_mag;
    double ey = (v[2]*Lx - v[0]*Lz) / mu - r[1] / r_mag;
    double ez = (v[0]*Ly - v[1]*Lx) / mu - r[2] / r_mag;

    // Argument of periapsis
    if (N_mag > 1e-30 && oe.e > 1e-10) {
        double cos_omega = (Nx*ex + Ny*ey) / (N_mag * oe.e);
        cos_omega = std::min(1.0, std::max(-1.0, cos_omega));
        oe.omega = std::acos(cos_omega);
        if (ez < 0.0) oe.omega = 2.0 * M_PI - oe.omega;
    } else {
        oe.omega = 0.0;
    }

    // True anomaly
    if (oe.e > 1e-10) {
        double cos_f = (ex * r[0] + ey * r[1] + ez * r[2]) / (oe.e * r_mag);
        cos_f = std::min(1.0, std::max(-1.0, cos_f));
        oe.true_anom = std::acos(cos_f);
        if (r_dot_v < 0.0) oe.true_anom = 2.0 * M_PI - oe.true_anom;
    } else {
        // Circular orbit: use angle from node line
        if (N_mag > 1e-30) {
            double cos_u = (Nx * r[0] + Ny * r[1]) / (N_mag * r_mag);
            cos_u = std::min(1.0, std::max(-1.0, cos_u));
            oe.true_anom = std::acos(cos_u);
            if (r[2] < 0.0) oe.true_anom = 2.0 * M_PI - oe.true_anom;
        } else {
            oe.true_anom = std::atan2(r[1], r[0]);
            if (oe.true_anom < 0.0) oe.true_anom += 2.0 * M_PI;
        }
    }

    // Mean anomaly (from eccentric anomaly)
    if (oe.e < 1.0 && oe.a > 0) {
        double cos_E = (oe.e + std::cos(oe.true_anom)) / (1.0 + oe.e * std::cos(oe.true_anom));
        double E = std::acos(std::min(1.0, std::max(-1.0, cos_E)));
        if (oe.true_anom > M_PI) E = 2.0 * M_PI - E;
        oe.mean_anom = E - oe.e * std::sin(E);
    } else {
        oe.mean_anom = 0.0;
    }

    // Orbital period
    if (oe.a > 0 && oe.E_orb < 0) {
        oe.period = 2.0 * M_PI * std::sqrt(oe.a * oe.a * oe.a / mu);
    } else {
        oe.period = -1.0; // unbound
    }

    return oe;
}

// ============================================================================
// Orbital elements to Cartesian state
// ============================================================================

void elements_to_cartesian(double a, double e, double inc, double Omega,
                           double omega, double mean_anomaly, double mu,
                           double r[3], double v[3]) {
    // Solve Kepler's equation
    double E = solve_kepler(mean_anomaly, e);

    // Position and velocity in orbital plane
    double cos_E = std::cos(E);
    double sin_E = std::sin(E);
    double sqrt_1_e2 = std::sqrt(std::fabs(1.0 - e * e));

    double x_orb = a * (cos_E - e);
    double y_orb = a * sqrt_1_e2 * sin_E;

    double denom = 1.0 - e * cos_E;
    double n = std::sqrt(mu / (a * a * a));  // mean motion
    double vx_orb = -a * n * sin_E / denom;
    double vy_orb = a * n * sqrt_1_e2 * cos_E / denom;

    // Rotation to inertial frame
    double cos_o = std::cos(omega);
    double sin_o = std::sin(omega);
    double cos_O = std::cos(Omega);
    double sin_O = std::sin(Omega);
    double cos_i = std::cos(inc);
    double sin_i = std::sin(inc);

    // Rotation matrix elements
    double P1 = cos_o * cos_O - sin_o * sin_O * cos_i;
    double P2 = -sin_o * cos_O - cos_o * sin_O * cos_i;
    double Q1 = cos_o * sin_O + sin_o * cos_O * cos_i;
    double Q2 = -sin_o * sin_O + cos_o * cos_O * cos_i;
    double R1 = sin_o * sin_i;
    double R2 = cos_o * sin_i;

    r[0] = P1 * x_orb + P2 * y_orb;
    r[1] = Q1 * x_orb + Q2 * y_orb;
    r[2] = R1 * x_orb + R2 * y_orb;

    v[0] = P1 * vx_orb + P2 * vy_orb;
    v[1] = Q1 * vx_orb + Q2 * vy_orb;
    v[2] = R1 * vx_orb + R2 * vy_orb;
}

// ============================================================================
// Kepler advance: advance 2-body orbit analytically by dt
// ============================================================================

void kepler_advance(double r[3], double v[3], double mu, double dt) {
    // Convert to orbital elements
    OrbitalElements oe = cartesian_to_elements(r, v, mu);

    if (oe.E_orb >= 0.0 || oe.a <= 0.0) {
        // Unbound orbit — use simple leapfrog as fallback
        double r_mag = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
        double inv_r3 = 1.0 / (r_mag * r_mag * r_mag);
        // Half kick
        v[0] -= 0.5 * dt * mu * r[0] * inv_r3;
        v[1] -= 0.5 * dt * mu * r[1] * inv_r3;
        v[2] -= 0.5 * dt * mu * r[2] * inv_r3;
        // Full drift
        r[0] += dt * v[0];
        r[1] += dt * v[1];
        r[2] += dt * v[2];
        // Half kick
        r_mag = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
        inv_r3 = 1.0 / (r_mag * r_mag * r_mag);
        v[0] -= 0.5 * dt * mu * r[0] * inv_r3;
        v[1] -= 0.5 * dt * mu * r[1] * inv_r3;
        v[2] -= 0.5 * dt * mu * r[2] * inv_r3;
        return;
    }

    // Advance mean anomaly
    double n = std::sqrt(mu / (oe.a * oe.a * oe.a));
    double M_new = oe.mean_anom + n * dt;

    // Reconstruct Cartesian state
    elements_to_cartesian(oe.a, oe.e, oe.inc, oe.Omega, oe.omega, M_new, mu, r, v);
}
