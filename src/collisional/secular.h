#pragma once


// Secular (orbit-averaged) evolution for tight binaries and stable triples
// Orbit-averaged PN evolution (gauge-invariant E, J; Blanchet 2014, Arun et al. 2009) + Kozai-Lidov oscillations

struct SecularBinary {
    double a, e;              // semi-major axis, eccentricity
    double omega;             // argument of periapsis
    double inc, Omega;        // inclination, longitude of ascending node
    double m1, m2;            // component masses
    int pid1, pid2;           // particle IDs

    // Peters (1964) rates
    double da_dt;             // da/dt from GW emission
    double de_dt;             // de/dt from GW emission
    double domega_dt;         // periapsis advance (PN1.0 + PN2.0)
};

struct SecularTriple {
    SecularBinary inner;
    double a_out, e_out;      // outer orbital elements
    double omega_out, inc_out, Omega_out;
    double m3;                // outer companion mass
    int pid3;                 // outer companion particle ID
    double cos_i_mut;         // mutual inclination cosine
};

// Orbit-averaged PN two-body evolution in eps = -2E/c^2, j = -2EJ^2/(Gm)^2 (E, J per reduced mass, c.o.m., harmonic);
// dw, dM: pericentre angle and mean anomaly advanced over dt. Returns the number of internal steps
struct SecularPnState {
    double eps, j;
    double dw = 0.0, dM = 0.0;
    bool merged = false;
};
int secular_evolve_pn(SecularPnState& s, double dt, double nu, double Gm, double c, int pn_order);
void secular_pn_eps_j(const double r[3], const double v[3], double Gm, double nu, double c, int pn_order,
                      double& eps, double& j);
double secular_pn_n(double eps, double j, double nu, double Gm, double c, int pn_order);
double secular_pn_k(double eps, double j, double nu, int pn_order);
double secular_pn_et2(double eps, double j, double nu, int pn_order);
void secular_pn_fluxes(double x, double et2, double nu, double Gm, double c, int pn_order, double& F, double& Gf);
void secular_pn_rates(double eps, double j, double nu, double Gm, double c, int pn_order, double& deps, double& dj);
// Secular-triple inner orbit: (a, e) as mean-motion semi-major axis and e_t -> da/dt, de/dt, d omega/dt
void secular_pn_inner_rates(double a, double e, double nu, double Gm, double c, int pn_order,
                            double& da, double& de, double& dw);

// Evolve a secular triple by dt (Kozai-Lidov + GW dissipation)
void secular_evolve_triple(SecularTriple& tri, double dt, double G, double c_code, int pn_order);


// Reconstruct (r, v) from secular elements (relative coordinates)
void secular_to_cartesian(const SecularBinary& bin, double G,
                          double mean_anomaly, double r[3], double v[3]);

// Full secular integration of a binary: Cartesian -> elements -> secular evolve -> advance M -> Cartesian
// Works in absolute coordinates (handles CM frame internally); returns the number of internal steps
int integrate_secular_binary(
    double pos1[3], double vel1[3], double m1,
    double pos2[3], double vel2[3], double m2,
    double G, double c_code, int pn_order, double dt);
