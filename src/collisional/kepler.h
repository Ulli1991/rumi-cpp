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

// Exact two-body (Kepler) drift of the relative orbit r, v over dt (any sign) under mu = G (m1 + m2): universal
// variables with Stumpff c-functions and Gauss f and g functions, no orbital elements or angles, so it holds for
// every e (circular, radial, parabolic, hyperbolic) and any interval (bound orbits drop whole periods first).
// Stumpff (1962); Danby (1992, Fundamentals of Celestial Mechanics, sec. 6.9); Mikkola & Innanen (1999); c-functions
// with the quarter-angle formulae and f, g in the incremental form of WHFast (Rein & Tamayo 2015, MNRAS 452, 376,
// sec. 2.2-2.5); Laguerre-Conway iteration safeguarded by bisection (Conway 1986; Wisdom & Hernandez 2015, MNRAS
// 453, 3015). Returns the number of iterations of the Kepler equation.
int kepler_drift(double r[3], double v[3], double mu, double dt);

// Advance a 2-body orbit by dt (analytical): kepler_drift. The elements route it replaced (r, v -> a, e, angles,
// M + n dt -> r, v) lost the phase at small e (1 drift vs 16: 1.6e-4 at e = 1e-6, order 1 at e <= 1e-8) and
// leapfrogged unbound orbits
void kepler_advance(double r[3], double v[3], double mu, double dt);

// Time to the next pericentre passage of a two-body orbit from its osculating elements (FROST-CLUSTERS I,
// Rantala et al. 2024, App. B3; the same use as TPERI in NBODY6's chain, Aarseth 2003). Bound orbits: the next
// pericentre, 0 <= tau < P; unbound orbits: the only pericentre, +inf once it has passed. r_peri (optional):
// osculating pericentre distance h^2 / (mu (1 + e)), well conditioned for e -> 1.
double kepler_time_to_pericentre(const double r[3], const double v[3], double mu, double* r_peri = nullptr);
