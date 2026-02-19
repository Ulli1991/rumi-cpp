#pragma once

// Kepler solver and orbital element utilities for subsystem detection

// (r, v) -> orbital elements
struct OrbitalElements {
    double a;          // semi-major axis (negative if unbound)
    double e;          // eccentricity
    double inc;        // inclination
    double Omega;      // longitude of ascending node
    double omega;      // argument of periapsis
    double true_anom;  // true anomaly
    double mean_anom;  // mean anomaly
    double E_orb;      // specific orbital energy
    double L_mag;      // specific angular momentum magnitude
    double period;     // orbital period (negative if unbound)
};

// Convert Cartesian state to orbital elements
// r[3], v[3]: relative position and velocity
// mu: G*(m1+m2)
OrbitalElements cartesian_to_elements(const double r[3], const double v[3], double mu);

// Convert orbital elements back to Cartesian state
// Outputs: r[3], v[3]
void elements_to_cartesian(double a, double e, double inc, double Omega,
                           double omega, double mean_anomaly, double mu,
                           double r[3], double v[3]);

// Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E
// Uses Mikkola (1987) cubic start with BIFROST empirical correction
// + Newton-Raphson refinement + bisection fallback for guaranteed convergence
double solve_kepler(double M, double e, double tol = 1e-14);

// Advance a 2-body orbit by dt using the Kepler drift (analytical)
void kepler_advance(double r[3], double v[3], double mu, double dt);
