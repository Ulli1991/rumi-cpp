#include "secular.h"
#include "kepler.h"
#include "constants.h"
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <initializer_list>

// ============================================================================
// Peters (1964) orbit-averaged GW dissipation
// ============================================================================

static double peters_F(double e) {
    double e2 = e * e;
    return (1.0 + 73.0/24.0 * e2 + 37.0/96.0 * e2*e2) / std::pow(1.0 - e2, 3.5);
}

static double peters_G(double e) {
    double e2 = e * e;
    return (1.0 + 121.0/304.0 * e2) / std::pow(1.0 - e2, 2.5);
}

static double peters_beta(double m1, double m2, double G, double c) {
    // beta = 64/5 * G^3 * m1 * m2 * (m1+m2) / c^5
    double c5 = c * c * c * c * c;
    return 64.0/5.0 * G*G*G * m1 * m2 * (m1 + m2) / c5;
}

// ============================================================================
// Rate computation helpers for the auxiliary variable leapfrog
// ============================================================================

// GW dissipation rate da/dt (Peters 1964, Eq. 16)
// Note: peters_beta includes the 64/5 prefactor, so da/dt = -beta/a^3 * F(e)
static double compute_da_dt(double a, double e, double beta, int pn_order) {
    if (!(pn_order & 4)) return 0.0;  // PN2.5 required
    return -beta / (a * a * a) * peters_F(e);
}

// GW dissipation rate de/dt (Peters 1964, Eq. 16)
// Paper: de/dt = -304/15 * beta_raw/a^4 * e * G(e)  where beta_raw = G^3 m1 m2 M / c^5
// Code:  peters_beta = 64/5 * beta_raw, so beta_raw = 5/64 * peters_beta
// Therefore: de/dt = -304/15 * (5/64) * beta/a^4 * e * G(e) = -19/12 * beta/a^4 * e * G(e)
static double compute_de_dt(double a, double e, double beta, int pn_order) {
    if (!(pn_order & 4)) return 0.0;  // PN2.5 required
    return -19.0/12.0 * beta / (a * a * a * a) * e * peters_G(e);
}

// PN1.0 + PN2.0 + PN3.0 periapsis advance rate
static double compute_domega_dt(double a, double e, double mu, double c2, int pn_order) {
    if (!(pn_order & 1)) return 0.0;
    double e2 = e * e;
    double one_me2 = 1.0 - e2;
    double P = 2.0 * M_PI * std::sqrt(a * a * a / mu);
    double Gm_ap = mu / (a * one_me2);
    double rate = 6.0 * M_PI * Gm_ap / (c2 * P);
    // PN2.0 term
    if (pn_order & 2) {
        double c4 = c2 * c2;
        rate += 3.0 * (18.0 + e2) * M_PI * Gm_ap * Gm_ap / (2.0 * c4 * P);
    }
    // PN3.0 term (Blanchet 2014, periapsis advance at 3PN)
    if (pn_order & 8) {
        double c6 = c2 * c2 * c2;
        double Gm_ap3 = Gm_ap * Gm_ap * Gm_ap;
        rate += M_PI * Gm_ap3 / (c6 * P) *
                (1.0/4.0 * (3024.0 + 1152.0*e2 + 41.0*M_PI*M_PI));
    }
    return rate;
}

// PN3.5 corrections to da/dt and de/dt (higher-order radiation reaction)
// Corrections to Peters 1964 formulas at order c^-7
static double compute_da_dt_pn35(double a, double e, double beta, double mu,
                                   double c2, int pn_order) {
    if (!(pn_order & 16)) return 0.0;
    double e2 = e * e;
    double one_me2 = 1.0 - e2;
    double c7_corr = mu / (a * one_me2 * c2);  // GM/(a*(1-e^2)*c^2)

    // Blanchet & Iyer (2003): relative correction to PN2.5 da/dt at PN3.5
    // da/dt_{3.5} = da/dt_{2.5} * correction * c7_corr
    double correction = -(1247.0/336.0 + 35.0/12.0 * (mu / (a * c2)));
    return compute_da_dt(a, e, beta, pn_order) * correction * c7_corr;
}

static double compute_de_dt_pn35(double a, double e, double beta, double mu,
                                   double c2, int pn_order) {
    if (!(pn_order & 16)) return 0.0;
    double e2 = e * e;
    double one_me2 = 1.0 - e2;
    double c7_corr = mu / (a * one_me2 * c2);

    double correction = -(1247.0/336.0 + 35.0/12.0 * (mu / (a * c2)));
    return compute_de_dt(a, e, beta, pn_order) * correction * c7_corr;
}

// ============================================================================
// Secular binary evolution — auxiliary variable leapfrog
// (Hellstrom & Mikkola 2010)
// ============================================================================
//
// Since da/dt = f(a,e) and de/dt = g(a,e) — the rates depend on the state —
// the ODE is not separable into independent position/velocity parts for a
// standard leapfrog.  The auxiliary variable trick introduces copies (a_aux,
// e_aux) and cross-evaluates:
//
//   da/dt      = f(a_aux, e_aux)     da_aux/dt = f(a, e)
//   de/dt      = g(a_aux, e_aux)     de_aux/dt = g(a, e)
//
// This makes each pair explicit and amenable to KDK leapfrog:
//   1) half-kick:  update (a, e) by dt/2 from rates at (a_aux, e_aux)
//                  update (a_aux, e_aux) by dt/2 from rates at (a, e)
//   2) drift:      advance omega by dt using rates at the half-stepped (a, e)
//   3) half-kick:  same as step 1

int secular_evolve_binary(SecularBinary& bin, double dt, double G, double c_code, int pn_order) {
    double beta = peters_beta(bin.m1, bin.m2, G, c_code);
    double c2 = c_code * c_code;
    double mu = G * (bin.m1 + bin.m2);

    // Secular timestep criterion: eta_sec * min(|a/da_dt|, |e/de_dt|) from the current state, re-evaluated every
    // sub-step (t_GW ~ a^4 shrinks within one interval of an inspiral; a count fixed from the initial rates
    // overshot coalescence). BIFROST Eq. 30 also has omega/omega-dot, but omega is an angle (near omega = 0
    // every call ran the 100000-step cap), and the drift of omega is the midpoint rule of omega-dot(a, e):
    // exact while a and e are constant; with a, e changing on t_GW its error over dt is ~ eta_sec^2 of the
    // precession angle. A precession term (1/omega-dot) made a 10+10 Msun binary at 0.01 AU take the
    // 100000-step cap in every evolution (48 ms each). NBODY_SECULAR_PREC_STEP=1 restores 1/|omega-dot|.
    static const bool prec_step = std::getenv("NBODY_SECULAR_PREC_STEP") && std::atoi(std::getenv("NBODY_SECULAR_PREC_STEP"));
    const double eta_sec = 0.01;
    constexpr int max_steps = 100000;
    auto step_size = [&](double a, double e) {
        const double da = compute_da_dt(a, e, beta, pn_order) + compute_da_dt_pn35(a, e, beta, mu, c2, pn_order);
        const double de = compute_de_dt(a, e, beta, pn_order) + compute_de_dt_pn35(a, e, beta, mu, c2, pn_order);
        double t_sec = 1e30;
        if (std::fabs(da) > 0.0) t_sec = std::min(t_sec, std::fabs(a / da));
        if (std::fabs(de) > 0.0 && e > 0.0) t_sec = std::min(t_sec, std::fabs(e / de));
        if (prec_step) {
            const double dw = compute_domega_dt(a, e, mu, c2, pn_order);
            if (std::fabs(dw) > 0.0) t_sec = std::min(t_sec, 1.0 / std::fabs(dw));
        }
        return std::max(eta_sec * t_sec, dt / max_steps);
    };

    // Initialize auxiliary variables equal to the physical state
    double a_aux = bin.a;
    double e_aux = bin.e;

    int n_steps = 0;
    for (double t_done = 0.0; t_done < dt && n_steps < max_steps; n_steps++) {
        double h = step_size(bin.a, bin.e);
        if (h >= (dt - t_done) * (1.0 - 1e-12)) h = dt - t_done;   // last sub-step lands on dt
        t_done = (h == dt - t_done) ? dt : t_done + h;
        double a = bin.a;
        double e = bin.e;

        // --- Half-kick (step 1) ---
        // Rates from auxiliary state for physical variables
        double da_from_aux = compute_da_dt(a_aux, e_aux, beta, pn_order)
                           + compute_da_dt_pn35(a_aux, e_aux, beta, mu, c2, pn_order);
        double de_from_aux = compute_de_dt(a_aux, e_aux, beta, pn_order)
                           + compute_de_dt_pn35(a_aux, e_aux, beta, mu, c2, pn_order);
        // Rates from physical state for auxiliary variables
        double da_from_phys = compute_da_dt(a, e, beta, pn_order)
                            + compute_da_dt_pn35(a, e, beta, mu, c2, pn_order);
        double de_from_phys = compute_de_dt(a, e, beta, pn_order)
                            + compute_de_dt_pn35(a, e, beta, mu, c2, pn_order);

        // Half-kick physical
        bin.a += 0.5 * h * da_from_aux;
        bin.e += 0.5 * h * de_from_aux;
        // Half-kick auxiliary
        a_aux += 0.5 * h * da_from_phys;
        e_aux += 0.5 * h * de_from_phys;

        // Clamp eccentricity after half-kick
        bin.e = std::max(0.0, std::min(bin.e, 0.9999));
        e_aux = std::max(0.0, std::min(e_aux, 0.9999));

        // --- Drift (step 2): advance omega using half-stepped (a, e) ---
        double dw = compute_domega_dt(bin.a, bin.e, mu, c2, pn_order);
        bin.omega += h * dw;

        // --- Second half-kick (step 3) ---
        // Recompute rates from the updated states
        da_from_aux = compute_da_dt(a_aux, e_aux, beta, pn_order)
                    + compute_da_dt_pn35(a_aux, e_aux, beta, mu, c2, pn_order);
        de_from_aux = compute_de_dt(a_aux, e_aux, beta, pn_order)
                    + compute_de_dt_pn35(a_aux, e_aux, beta, mu, c2, pn_order);
        da_from_phys = compute_da_dt(bin.a, bin.e, beta, pn_order)
                     + compute_da_dt_pn35(bin.a, bin.e, beta, mu, c2, pn_order);
        de_from_phys = compute_de_dt(bin.a, bin.e, beta, pn_order)
                     + compute_de_dt_pn35(bin.a, bin.e, beta, mu, c2, pn_order);

        // Half-kick physical
        bin.a += 0.5 * h * da_from_aux;
        bin.e += 0.5 * h * de_from_aux;
        // Half-kick auxiliary
        a_aux += 0.5 * h * da_from_phys;
        e_aux += 0.5 * h * de_from_phys;

        // Clamp eccentricity
        bin.e = std::max(0.0, std::min(bin.e, 0.9999));
        e_aux = std::max(0.0, std::min(e_aux, 0.9999));

        // Check for coalescence: pericenter < 6 * G*M / c^2 (innermost stable circular orbit)
        if (bin.a < 0.0 || bin.a * (1.0 - bin.e) < 6.0 * mu / c2) {
            bin.a = 0.0;  // signal merger
            break;
        }
    }

    // Store final rates for external use (e.g., secular timestep criterion)
    bin.da_dt = compute_da_dt(bin.a, bin.e, beta, pn_order)
              + compute_da_dt_pn35(bin.a, bin.e, beta, mu, c2, pn_order);
    bin.de_dt = compute_de_dt(bin.a, bin.e, beta, pn_order)
              + compute_de_dt_pn35(bin.a, bin.e, beta, mu, c2, pn_order);
    bin.domega_dt = compute_domega_dt(bin.a, bin.e, mu, c2, pn_order);
    return n_steps;
}

// ============================================================================
// Kozai-Lidov secular triple evolution (quadrupole + octupole)
// Following Blaes, Lee & Socrates (2002) / Naoz et al. (2013) conventions
// with C2, C3, G1, G2 coefficients from the Antognini (2015) kozai package.
// Octupole terms enable the "eccentric Kozai-Lidov" (EKL) mechanism.
// ============================================================================

void secular_evolve_triple(SecularTriple& tri, double dt, double G, double c_code, int pn_order) {
    double m1 = tri.inner.m1;
    double m2 = tri.inner.m2;
    double m3 = tri.m3;
    double M12 = m1 + m2;
    double M_tot = M12 + m3;
    double mu_in = G * M12;

    double a_out = tri.a_out;
    double e_out = tri.e_out;
    if (a_out <= 0.0) return;

    double e_out2 = e_out * e_out;
    double j2 = std::sqrt(1.0 - e_out2);  // outer angular momentum parameter

    // Blaes et al. (2002) coefficients (Eq. 19-20)
    // C2: quadrupole coefficient
    double C2 = G * m1 * m2 * m3 / (16.0 * M12 * a_out * std::pow(j2, 3)) *
                (tri.inner.a / a_out) * (tri.inner.a / a_out);
    // C3: octupole coefficient (note: m2-m1 sign convention from Blaes et al.)
    double C3 = 15.0 * G * m1 * m2 * m3 * (m2 - m1) /
                (64.0 * M12 * M12 * a_out * std::pow(j2, 5)) *
                (tri.inner.a / a_out) * (tri.inner.a / a_out) * (tri.inner.a / a_out);

    // Delaunay angular momenta
    // G1 = mu_red1 * sqrt(G * M12 * a1) * j1
    // G2 = mu_red2 * sqrt(G * M_tot * a2) * j2
    // Initial values (for H conservation)
    double j1_init = std::sqrt(1.0 - tri.inner.e * tri.inner.e);
    double G1_init = m1 * m2 / M12 * std::sqrt(G * M12 * tri.inner.a) * j1_init;
    double G2 = (M12 * m3 / M_tot) * std::sqrt(G * M_tot * a_out) * j2;

    // Conserved total angular momentum component
    // H^2 = G1^2 + G2^2 + 2*G1*G2*cos(i_mut)
    double H2 = G1_init * G1_init + G2 * G2 + 2.0 * G1_init * G2 * tri.cos_i_mut;

    // Adaptive internal stepping
    // Estimate KL timescale from C2 for step size control
    double G1_est = G1_init;
    double rate_est = std::fabs(30.0 * C2 * tri.inner.e * (1.0 - tri.inner.e * tri.inner.e) /
                     std::max(G1_est, 1e-30) * (1.0 - tri.cos_i_mut * tri.cos_i_mut));
    double t_sec = (rate_est > 0.0) ? tri.inner.e / rate_est : 1e30;
    const double eta_sec = 0.01;
    int n_steps = std::max(1, (int)std::ceil(dt / (eta_sec * t_sec)));
    n_steps = std::min(n_steps, 100000);
    double h = dt / n_steps;

    for (int step = 0; step < n_steps; step++) {
        double a_in = tri.inner.a;
        double e1 = tri.inner.e;
        double g1 = tri.inner.omega;
        double g2 = tri.omega_out;
        double th = tri.cos_i_mut;

        if (a_in <= 0.0) break;

        double e1_2 = e1 * e1;
        double j1 = std::sqrt(1.0 - e1_2);
        double th2 = th * th;
        double sin2i = 1.0 - th2;  // sin^2(i_mut)

        // Delaunay angular momentum for inner orbit
        double G1 = m1 * m2 / M12 * std::sqrt(G * M12 * a_in) * j1;
        if (G1 < 1e-30 || G2 < 1e-30) break;

        // Recompute C2, C3 with current a_in (a_in changes due to GW)
        double alpha = a_in / a_out;
        double C2_cur = G * m1 * m2 * m3 / (16.0 * M12 * a_out * std::pow(j2, 3)) * alpha * alpha;
        double C3_cur = 15.0 * G * m1 * m2 * m3 * (m2 - m1) /
                        (64.0 * M12 * M12 * a_out * std::pow(j2, 5)) * alpha * alpha * alpha;

        // Trigonometric quantities
        double sing1 = std::sin(g1), cosg1 = std::cos(g1);
        double sing2 = std::sin(g2), cosg2 = std::cos(g2);

        // Octupole auxiliary quantities (Blaes et al. 2002)
        double B = 2.0 + 5.0 * e1_2 - 7.0 * e1_2 * std::cos(2.0 * g1);
        double A = 4.0 + 3.0 * e1_2 - 2.5 * sin2i * B;
        double cosphi = cosg1 * cosg2 - th * sing1 * sing2;

        // ---- de1/dt ----
        // Quadrupole (Blaes et al. 2002, Eq. 13)
        double de1_dt = 30.0 * C2_cur * e1 * (1.0 - e1_2) / G1 *
                        sin2i * std::sin(2.0 * g1);
        // Octupole
        if (std::fabs(C3_cur) > 0.0) {
            de1_dt += -C3_cur * e_out * (1.0 - e1_2) / G1 * (
                35.0 * cosphi * sin2i * e1_2 * std::sin(2.0 * g1)
                - 10.0 * th * (1.0 - e1_2) * sin2i * cosg1 * sing2
                - A * (sing1 * cosg2 - th * cosg1 * sing2)
            );
        }

        // ---- dg1/dt ----
        // Quadrupole
        double dg1_dt = 6.0 * C2_cur * (
            (1.0 / G1) * (4.0 * th2 + (5.0 * std::cos(2.0 * g1) - 1.0) * (1.0 - e1_2 - th2))
            + th / G2 * (2.0 + e1_2 * (3.0 - 5.0 * std::cos(2.0 * g1)))
        );
        // Octupole
        if (std::fabs(C3_cur) > 0.0 && e1 > 1e-10) {
            dg1_dt += C3_cur * e_out * e1 * (1.0 / G2 + th / G1) * (
                sing1 * sing2 * (A + 10.0 * (3.0 * th2 - 1.0) * (1.0 - e1_2))
                - 5.0 * th * B * cosphi
            );
            dg1_dt += -C3_cur * e_out * (1.0 - e1_2) / (e1 * G1) * (
                10.0 * th * sin2i * (1.0 - 3.0 * e1_2) * sing1 * sing2
                + cosphi * (3.0 * A - 10.0 * th2 + 2.0)
            );
        }

        // ---- dg2/dt ----
        // Quadrupole
        double dg2_dt = 3.0 * C2_cur * (
            2.0 * th / G1 * (2.0 + e1_2 * (3.0 - 5.0 * std::cos(2.0 * g1)))
            + (1.0 / G2) * (4.0 + 6.0 * e1_2 + (5.0 * th2 - 3.0) *
                            (2.0 + 3.0 * e1_2 - 5.0 * e1_2 * std::cos(2.0 * g1)))
        );
        // Octupole
        if (std::fabs(C3_cur) > 0.0) {
            dg2_dt += -C3_cur * e1 * sing1 * sing2 * (
                (4.0 * e_out2 + 1.0) / (e_out * G2) * 10.0 * th * sin2i * (1.0 - e1_2)
                - e_out * (1.0 / G1 + th / G2) * (A + 10.0 * (3.0 * th2 - 1.0) * (1.0 - e1_2))
            );
            dg2_dt += -C3_cur * e1 * cosphi * (
                5.0 * B * th * e_out * (1.0 / G1 + th / G2)
                + (4.0 * e_out2 + 1.0) / (e_out * G2) * A
            );
        }

        // PN precession on inner orbit (suppresses high-e Kozai)
        // Uses compute_domega_dt which includes PN1.0, PN2.0, and PN3.0
        {
            double c2_loc = c_code * c_code;
            dg1_dt += compute_domega_dt(a_in, e1, mu_in, c2_loc, pn_order);
        }

        // GW dissipation on inner binary — Peters (1964) Eq. 16 + PN3.5 correction
        double da_in_dt = 0.0, de1_gw_dt = 0.0;
        if (pn_order & 4) {
            double beta = peters_beta(m1, m2, G, c_code);
            da_in_dt = -beta / (a_in * a_in * a_in) * peters_F(e1);
            de1_gw_dt = -19.0/12.0 * beta / (a_in * a_in * a_in * a_in) * e1 * peters_G(e1);
            // PN3.5 higher-order radiation reaction correction
            if (pn_order & 16) {
                double c2_loc = c_code * c_code;
                da_in_dt += compute_da_dt_pn35(a_in, e1, beta, mu_in, c2_loc, pn_order);
                de1_gw_dt += compute_de_dt_pn35(a_in, e1, beta, mu_in, c2_loc, pn_order);
            }
        }

        // Integrate
        tri.inner.a += h * da_in_dt;
        tri.inner.e += h * (de1_dt + de1_gw_dt);
        tri.inner.omega += h * dg1_dt;
        tri.omega_out += h * dg2_dt;

        // Clamp eccentricity
        tri.inner.e = std::max(1e-6, std::min(tri.inner.e, 0.9999));

        // Update mutual inclination from H conservation
        // H^2 = G1^2 + G2^2 + 2*G1*G2*cos(i_mut) => cos(i_mut) = (H^2-G1^2-G2^2)/(2*G1*G2)
        double j1_new = std::sqrt(1.0 - tri.inner.e * tri.inner.e);
        double G1_new = m1 * m2 / M12 * std::sqrt(G * M12 * std::max(tri.inner.a, 1e-30)) * j1_new;
        if (G1_new > 1e-30) {
            double cos_i_new = (H2 - G1_new * G1_new - G2 * G2) / (2.0 * G1_new * G2);
            tri.cos_i_mut = std::max(-1.0, std::min(cos_i_new, 1.0));
        }

        if (tri.inner.a <= 0.0) break;
    }
}

// ============================================================================
// Transition checks
// ============================================================================

bool should_be_secular(const Subsystem& sub, double dt_step, int n_orb_threshold) {
    (void)dt_step;
    if (sub.type != SubsystemType::BINARY && sub.type != SubsystemType::TRIPLE_STABLE)
        return false;
    return sub.n_orb_per_dt >= n_orb_threshold;
}

void secular_to_cartesian(const SecularBinary& bin, double G,
                          double mean_anomaly, double r[3], double v[3]) {
    double mu = G * (bin.m1 + bin.m2);
    elements_to_cartesian(bin.a, bin.e, bin.inc, bin.Omega, bin.omega,
                          mean_anomaly, mu, r, v);
}

// ============================================================================
// Secular binary integration in absolute coordinates (handles the c.o.m. internally): the exact Kepler drift of the
// relative orbit (kepler_drift, universal variables), then the orbit-averaged PN changes of (a, e, omega) applied to
// the drifted state as a rotation about L and a change of shape at fixed orientation and mean anomaly. No orbital
// elements or angles are formed, so it holds at every e. The elements route it replaced (r, v -> a, e, omega, M ->
// r, v) lost the phase at small e: one drift vs sixteen disagreed by 4e-8 at e = 1e-4, 1.6e-4 at e = 1e-6 and by
// order 1 at e <= 1e-8 (the pericentre direction and the true anomaly are separately undefined there), and it froze
// unbound pairs. Without PN this is the analytic Kepler solution, to which BIFROST's secular integrator reduces
// (Rantala et al. 2023, sec. 3.10).
// ============================================================================

namespace {
double dot3(const double* a, const double* b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
void cross3(const double* a, const double* b, double* c) {
    c[0] = a[1]*b[2] - a[2]*b[1]; c[1] = a[2]*b[0] - a[0]*b[2]; c[2] = a[0]*b[1] - a[1]*b[0];
}

// Rotate r and v by the angle dw about the unit vector n (Rodrigues): the pericentre advance of the orbit
void rotate_about(double r[3], double v[3], const double n[3], double dw) {
    const double c = std::cos(dw), s = std::sin(dw);
    for (double* x : {r, v}) {
        double nx[3];
        cross3(n, x, nx);
        const double nd = dot3(n, x) * (1.0 - c);
        for (int k = 0; k < 3; k++) x[k] = x[k] * c + nx[k] * s + n[k] * nd;
    }
}

// Bound relative state r, v (mu) -> the orbit with semi-major axis a1, eccentricity vector k e_vec (same pericentre
// direction and plane) and mean anomaly M + dM. In the orbit plane r = a (cos E - e) P + a sqrt(1 - e^2) sin E Q and
// v = sqrt(mu / a) (-sin E P + sqrt(1 - e^2) cos E Q) / (1 - e cos E); with u = cos E P + sin E Q, w = -sin E P +
// cos E Q, the eccentricity vector e_vec = e P, q = L_hat x e_vec = e Q, C = e cos E = 1 - r / a and S = e sin E =
// r.v / sqrt(mu a) this is r = a (u - e_vec - S q / (1 + sqrt(1 - e^2))), v = sqrt(mu / a) (w - C q / (1 +
// sqrt(1 - e^2))) / (1 - C): every term stays defined as e -> 0, where P and E separately do not.
void reshape_orbit(double r[3], double v[3], double mu, double a1, double k, double dM) {
    const double rr = std::sqrt(dot3(r, r));
    const double beta = 2.0 * mu / rr - dot3(v, v);
    if (!(beta > 0.0) || !(a1 > 0.0)) return;
    const double a = mu / beta;
    double h[3], vh[3], ev[3], q[3], hh[3];
    cross3(r, v, h);
    const double hn = std::sqrt(dot3(h, h));
    if (!(hn > 0.0)) return;   // radial orbit: no plane
    cross3(v, h, vh);
    for (int c = 0; c < 3; c++) { ev[c] = vh[c] / mu - r[c] / rr; hh[c] = h[c] / hn; }
    cross3(hh, ev, q);
    const double e2 = std::min(dot3(ev, ev), 1.0 - 1e-15);
    const double be = std::sqrt(1.0 - e2);
    const double omc = rr / a;                       // 1 - C, exact (no cancellation near the pericentre of e -> 1)
    const double C = 1.0 - omc, S = dot3(r, v) / std::sqrt(mu * a);
    const double sv = std::sqrt(mu / a);
    double u[3], w[3];
    for (int c = 0; c < 3; c++) {
        u[c] = r[c] / a + ev[c] + S * q[c] / (1.0 + be);
        w[c] = v[c] * omc / sv + C * q[c] / (1.0 + be);
    }
    // Eccentric-anomaly change d at the new mean anomaly: (E + d) - k e sin(E + d) = E - e sin E + dM, i.e.
    // f(d) = d - k (S cos d + C sin d) + S - dM = 0; f' = 1 - k e cos(E + d) > 0 and |k (S cos d + C sin d)| <= k e
    // bracket the root in dM - S -+ k e. Newton, bisection where it leaves the bracket (high e near the pericentre)
    const double ke = k * std::sqrt(e2);
    double lo = dM - S - ke, hi = dM - S + ke, d = dM;
    if (!(d > lo && d < hi)) d = 0.5 * (lo + hi);
    for (int it = 0; it < 100; it++) {
        const double cd = std::cos(d), sd = std::sin(d);
        const double f = d - k * (S * cd + C * sd) + S - dM;
        if (f == 0.0) break;
        if (f < 0.0) lo = std::max(lo, d); else hi = std::min(hi, d);
        double dn = d - f / (1.0 - k * (C * cd - S * sd));
        if (!(dn > lo && dn < hi)) dn = 0.5 * (lo + hi);
        if (dn == d || !(hi > lo)) break;
        const double step = std::fabs(dn - d);
        d = dn;
        if (step <= 1e-16 * std::max(1.0, std::fabs(d))) break;
    }
    const double cd = std::cos(d), sd = std::sin(d);
    const double C1 = k * (C * cd - S * sd), S1 = k * (S * cd + C * sd);
    // 1 - C1 = (1 - k) + k (1 - C + 2 C sin^2(d / 2) + S sin d), without cancellation
    const double sh = std::sin(0.5 * d);
    const double omc1 = (1.0 - k) + k * (omc + 2.0 * C * sh * sh + S * sd);
    const double be1 = std::sqrt(std::max(0.0, 1.0 - k * k * e2));
    const double sv1 = std::sqrt(mu / a1);
    for (int c = 0; c < 3; c++) {
        const double u1 = cd * u[c] + sd * w[c], w1 = -sd * u[c] + cd * w[c];
        r[c] = a1 * (u1 - k * ev[c] - S1 * k * q[c] / (1.0 + be1));
        v[c] = sv1 * (w1 - C1 * k * q[c] / (1.0 + be1)) / omc1;
    }
}
}  // namespace

int integrate_secular_binary(
    double pos1[3], double vel1[3], double m1,
    double pos2[3], double vel2[3], double m2,
    double G, double c_code, int pn_order, double dt) {

    const double M_tot = m1 + m2;
    const double mu = G * M_tot;

    double cm_pos[3], cm_vel[3], r[3], v[3];
    for (int k = 0; k < 3; k++) {
        cm_pos[k] = (m1 * pos1[k] + m2 * pos2[k]) / M_tot;
        cm_vel[k] = (m1 * vel1[k] + m2 * vel2[k]) / M_tot;
        r[k] = pos1[k] - pos2[k];
        v[k] = vel1[k] - vel2[k];
    }
    const double r_in[3] = {r[0], r[1], r[2]}, v_in[3] = {v[0], v[1], v[2]};

    int n_sub = 1;
    double a1 = 0.0, k_e = 1.0, dw = 0.0, dM = 0.0;
    double hhat[3] = {0.0, 0.0, 1.0};
    bool reshape = false;
    if (pn_order != 0) {
        // Orbit-averaged PN evolution of (a, e, omega) from the state's a (energy) and e (|e_vec|, accurate to
        // round-off at small e, unlike sqrt(1 - p / a)); only the increments are used
        const double rr = std::sqrt(dot3(r, r));
        const double beta = 2.0 * mu / rr - dot3(v, v);
        double h[3], vh[3];
        cross3(r, v, h);
        cross3(v, h, vh);
        double e2 = 0.0;
        for (int c = 0; c < 3; c++) { const double ec = vh[c] / mu - r[c] / rr; e2 += ec * ec; }
        const double e0 = std::sqrt(e2);
        const double hn = std::sqrt(dot3(h, h));
        if (!(beta > 0.0) || e0 >= 1.0 || !(hn > 0.0)) return 0;   // unbound or radial: not a secular binary
        for (int c = 0; c < 3; c++) hhat[c] = h[c] / hn;
        SecularBinary bin;
        bin.a = mu / beta;
        bin.e = e0;
        bin.omega = 0.0;
        bin.inc = 0.0;
        bin.Omega = 0.0;
        bin.m1 = m1;
        bin.m2 = m2;
        const double a0 = bin.a;
        const double e_ref = std::min(e0, 0.9999);   // secular_evolve_binary clamps e to 0.9999: a change, not the clamp
        n_sub = secular_evolve_binary(bin, dt, G, c_code, pn_order);

        // Merger (secular_evolve_binary sets a = 0 at coalescence) or unbound after the evolution: leave the
        // particles as they are (as before)
        if (bin.a <= 0.0) {
            printf("  [SEC] WARNING: binary merged during secular evolution "
                   "(pericenter < ISCO)\n");
            return n_sub;
        }
        if (bin.e >= 1.0) {
            printf("  [SEC] WARNING: binary became unbound during secular evolution "
                   "(e=%.6f)\n", bin.e);
            return n_sub;
        }
        a1 = bin.a;
        k_e = e_ref > 0.0 ? bin.e / e_ref : 1.0;
        dw = bin.omega;
        // Mean motion: the drift advances M by n(a0) dt; the trapezoid of n(a) over the interval adds the rest. (The
        // elements route used n(a1) dt, first order: for the 10 + 10 Msun BH binary at 0.01 AU, 4.5e4 orbits and
        // da/a = 2.5e-6 per 1e-5 Myr, that is 0.5 rad of phase per step)
        if (a1 != a0) dM = 0.5 * (std::sqrt(mu / (a1 * a1 * a1)) - std::sqrt(mu / (a0 * a0 * a0))) * dt;
        reshape = (a1 != a0 || k_e != 1.0 || dM != 0.0);
    }

    kepler_drift(r, v, mu, dt);
    if (dw != 0.0) rotate_about(r, v, hhat, dw);
    if (reshape) reshape_orbit(r, v, mu, a1, k_e, dM);

    // c.o.m. moves on a straight line (internal forces only); pos1 = cm + (m2 / M) r, pos2 = cm - (m1 / M) r
    for (int k = 0; k < 3; k++) {
        const double cm_new = cm_pos[k] + cm_vel[k] * dt;
        pos1[k] = cm_new + (m2 / M_tot) * r[k];
        vel1[k] = cm_vel[k] + (m2 / M_tot) * v[k];
        pos2[k] = cm_new - (m1 / M_tot) * r[k];
        vel2[k] = cm_vel[k] - (m1 / M_tot) * v[k];
    }
    // The two-body energy is set exactly (without PN: conserved), but the members are stored in absolute coordinates: at
    // |x| / |r| ~ 1e7 (a 0.02 AU binary 1 pc from the centre) rounding the new positions changes E_orb by ~1e-9 of
    // itself, which the energy check sees (043190d booked that change as E_gw, which hid it). Scaling the relative
    // velocity, stored to ~eps of itself, restores the orbit's energy (manifold correction for Kepler energy
    // consistency, Fukushima 2003, AJ 126, 1097); two passes for the rounding of the velocities themselves
    // With PN the target is the evolved orbit's energy -mu / (2 a1): without the correction the rounding error of a
    // (~1e-9 of itself far from the origin) became a phase error ~ (3/2) da/a 2 pi N_orb over the 1e4 - 1e5 orbits of a
    // secular step (1e-3 of the orbit for a 10 + 10 Msun binary at 0.01 AU boosted by 300 km/s, test pn_boost_secular)
    {
        const double E_target = reshape ? -mu / (2.0 * a1) : 0.5 * dot3(v_in, v_in) - mu / std::sqrt(dot3(r_in, r_in));
        for (int pass = 0; pass < 2; pass++) {
            double ro[3], vo[3];
            for (int k = 0; k < 3; k++) { ro[k] = pos1[k] - pos2[k]; vo[k] = vel1[k] - vel2[k]; }
            const double vo2 = dot3(vo, vo);
            const double arg = 2.0 * (E_target + mu / std::sqrt(dot3(ro, ro)));
            if (!(arg > 0.0) || !(vo2 > 0.0)) break;
            const double sm1 = std::sqrt(arg / vo2) - 1.0;
            if (sm1 == 0.0) break;
            for (int k = 0; k < 3; k++) {
                const double dv = sm1 * vo[k];
                vel1[k] += (m2 / M_tot) * dv;
                vel2[k] -= (m1 / M_tot) * dv;
            }
        }
    }
    return n_sub;
}
