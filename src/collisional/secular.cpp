#include "secular.h"
#include "kepler.h"
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <initializer_list>

// ============================================================================
// Orbit-averaged PN evolution of a two-body orbit
// ============================================================================
//
// Variables: E, J per reduced mass (c.o.m. frame, harmonic coordinates), eps = -2E/c^2, j = -2 E J^2 / (Gm)^2: gauge
// invariant and constant along the conservative PN orbit, unlike osculating Newtonian a, e, which oscillate at O(x)
// around a PN orbit (the 1PN radiation-reaction correction, PN3.5, is a relative O(x) effect too, so it is only
// defined in such variables). Mean motion n and periastron advance k = K - 1 in eps, j (Blanchet, Living Rev. Rel. 17, 2
// (2014, v3 2024), eqs. n3PN, K3PN); fluxes in x and e_t (eq. Finst, AvEMh0-1; Arun et al. 2009, PRD 80, 124018,
// eqs. 4.5-4.6), with x, e_t from eps, j (eqs. xepsj, etMH) to 1PN; the balance mu dE/dt = -<F>, mu dJ/dt = -<G>
// (Arun et al. eq. 6.4). ADM and MH e_t differ from 2PN on: not needed here.

static double sec_one_m(double et2) { return std::max(1.0 - et2, 1e-12); }

// x and e_t^2 from eps, j; 1PN terms with PN3.5 (they enter only the 1PN-corrected fluxes)
static void sec_x_et2(double eps, double j, double nu, int pn_order, double& x, double& et2) {
    x = eps;
    et2 = 1.0 - j;
    if (pn_order & 16) {
        x = eps * (1.0 + eps * (-1.25 + nu / 12.0 + 2.0 / j));
        et2 += 0.25 * eps * (-8.0 + 8.0 * nu + j * (17.0 - 7.0 * nu));
    }
}

void secular_pn_eps_j(const double r[3], const double v[3], double Gm, double nu, double c, int pn_order,
                      double& eps, double& j) {
    // Kidder 1995, PRD 52, 821, eqs. 2.7-2.8 (c.o.m., harmonic), to the conservative orders of pn_order
    const double rr = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    const double v2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    const double rd = (r[0]*v[0] + r[1]*v[1] + r[2]*v[2]) / rr, rd2 = rd * rd;
    const double u = Gm / rr, c2 = c * c, nu2 = nu * nu;
    double E = 0.5 * v2 - u, fJ = 1.0;
    if (pn_order & 1) {
        E += (3.0/8.0 * (1.0 - 3.0 * nu) * v2 * v2 + 0.5 * (3.0 + nu) * v2 * u + 0.5 * nu * u * rd2 + 0.5 * u * u) / c2;
        fJ += (0.5 * v2 * (1.0 - 3.0 * nu) + (3.0 + nu) * u) / c2;
    }
    if (pn_order & 2) {
        const double c4 = c2 * c2;
        E += (5.0/16.0 * (1.0 - 7.0 * nu + 13.0 * nu2) * v2 * v2 * v2 - 3.0/8.0 * nu * (1.0 - 3.0 * nu) * u * rd2 * rd2
              + 1.0/8.0 * (21.0 - 23.0 * nu - 27.0 * nu2) * u * v2 * v2 + 1.0/8.0 * (14.0 - 55.0 * nu + 4.0 * nu2) * u * u * v2
              + 0.25 * nu * (1.0 - 15.0 * nu) * u * v2 * rd2 - 0.25 * (2.0 + 15.0 * nu) * u * u * u
              + 1.0/8.0 * (4.0 + 69.0 * nu + 12.0 * nu2) * u * u * rd2) / c4;
        fJ += (3.0/8.0 * (1.0 - 7.0 * nu + 13.0 * nu2) * v2 * v2 - 0.5 * nu * (2.0 + 5.0 * nu) * u * rd2
               + 0.5 * (7.0 - 10.0 * nu - 9.0 * nu2) * u * v2 + 0.25 * (14.0 - 41.0 * nu + 4.0 * nu2) * u * u) / c4;
    }
    double h[3] = {r[1]*v[2] - r[2]*v[1], r[2]*v[0] - r[0]*v[2], r[0]*v[1] - r[1]*v[0]};
    const double J2 = (h[0]*h[0] + h[1]*h[1] + h[2]*h[2]) * fJ * fJ;
    eps = -2.0 * E / c2;
    j = -2.0 * E * J2 / (Gm * Gm);
}

double secular_pn_peri_shift(const double r[3], const double v[3], double Gm, double nu, double c, int pn_order) {
    const double rr = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    const double v2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    const double beta = 2.0 * Gm / rr - v2;   // Gm / a
    const double h[3] = {r[1]*v[2] - r[2]*v[1], r[2]*v[0] - r[0]*v[2], r[0]*v[1] - r[1]*v[0]};
    const double h2 = h[0]*h[0] + h[1]*h[1] + h[2]*h[2];
    if (!(beta > 0.0) || !(h2 > 0.0)) return INFINITY;
    // e from 1 - e^2 = h^2 beta / (Gm)^2, r_p = h^2 / (Gm (1 + e)), v_p = h / r_p (no cancellation as e -> 1)
    const double e = std::sqrt(std::max(0.0, 1.0 - h2 * beta / (Gm * Gm)));
    const double rp = h2 / (Gm * (1.0 + e)), vp = std::sqrt(h2) / rp;
    const double xp[3] = {rp, 0.0, 0.0}, vpv[3] = {0.0, vp, 0.0};
    double eps, j;
    secular_pn_eps_j(xp, vpv, Gm, nu, c, pn_order, eps, j);
    return std::fabs(1.0 - eps * c * c / beta);
}

double secular_pn_n(double eps, double j, double nu, double Gm, double c, int pn_order) {
    double s = 1.0;
    if (pn_order & 1) s += eps / 8.0 * (-15.0 + nu);
    if (pn_order & 2) s += eps * eps / 128.0 * (555.0 + 30.0 * nu + 11.0 * nu * nu + 192.0 / std::sqrt(j) * (-5.0 + 2.0 * nu));
    if (pn_order & 8)
        s += eps * eps * eps / 3072.0 * (-29385.0 - 4995.0 * nu - 315.0 * nu * nu + 135.0 * nu * nu * nu
             + 5760.0 / std::sqrt(j) * (17.0 - 9.0 * nu + 2.0 * nu * nu)
             - 16.0 / (j * std::sqrt(j)) * (10080.0 + (-13952.0 + 123.0 * M_PI * M_PI) * nu + 1440.0 * nu * nu));
    return c * c * c / Gm * eps * std::sqrt(eps) * s;
}

double secular_pn_k(double eps, double j, double nu, int pn_order) {
    double k = 0.0;
    if (pn_order & 1) k += 3.0 * eps / j;
    if (pn_order & 2) k += eps * eps / 4.0 * (3.0 / j * (-5.0 + 2.0 * nu) + 15.0 / (j * j) * (7.0 - 2.0 * nu));
    if (pn_order & 8)
        k += eps * eps * eps / 128.0 * (24.0 / j * (5.0 - 5.0 * nu + 4.0 * nu * nu)
             - 1.0 / (j * j) * (10080.0 + (-13952.0 + 123.0 * M_PI * M_PI) * nu + 1440.0 * nu * nu)
             + 5.0 / (j * j * j) * (7392.0 + (-8000.0 + 123.0 * M_PI * M_PI) * nu + 336.0 * nu * nu));
    return k;
}

// Orbit-averaged energy and angular-momentum fluxes per reduced mass, <F>/mu and <G>/mu (PN2.5, + PN3.5 1PN terms)
void secular_pn_fluxes(double x, double et2, double nu, double Gm, double c, int pn_order, double& F, double& Gf) {
    F = 0.0; Gf = 0.0;
    if (!(pn_order & 4)) return;
    const double e2 = std::max(et2, 0.0), e4 = e2 * e2, om = sec_one_m(e2);
    double I = (1.0 + 73.0/24.0 * e2 + 37.0/96.0 * e4) / std::pow(om, 3.5);
    double H = (8.0 + 7.0 * e2) / (om * om);
    if (pn_order & 16) {
        I += x * (-1247.0/336.0 - 35.0/12.0 * nu + e2 * (10475.0/672.0 - 1081.0/36.0 * nu)
                  + e4 * (10043.0/384.0 - 311.0/12.0 * nu) + e4 * e2 * (2179.0/1792.0 - 851.0/576.0 * nu)) / std::pow(om, 4.5);
        H += x * (-1247.0/42.0 - 70.0/3.0 * nu + e2 * (3019.0/42.0 - 335.0/3.0 * nu)
                  + e4 * (8399.0/336.0 - 275.0/12.0 * nu)) / (om * om * om);
    }
    const double x2 = x * x;
    F = 32.0/5.0 * std::pow(c, 5) * nu * x2 * x2 * x / Gm * I;
    Gf = 0.8 * c * c * nu * x2 * x * std::sqrt(x) * H;
}

void secular_pn_rates(double eps, double j, double nu, double Gm, double c, int pn_order, double& deps, double& dj) {
    double x, et2, F, Gf;
    sec_x_et2(eps, j, nu, pn_order, x, et2);
    secular_pn_fluxes(x, et2, nu, Gm, c, pn_order, F, Gf);
    deps = 2.0 * F / (c * c);
    dj = 2.0 * F * j / (eps * c * c) - 2.0 * c * std::sqrt(eps * j) * Gf / Gm;
}

double secular_pn_et2(double eps, double j, double nu, int pn_order) {
    double x, et2;
    sec_x_et2(eps, j, nu, pn_order, x, et2);
    return et2;
}

// ============================================================================
// Secular evolution of (eps, j) — auxiliary variable leapfrog (Hellstrom & Mikkola 2010)
// ============================================================================
//
// The rates depend on the state, so the ODE is not separable; copies (eps_aux, j_aux) are kicked with the rates of
// the physical state and vice versa (KDK), and the drift advances the pericentre angle (k n) and the mean anomaly (n)
// at the half-stepped state.
int secular_evolve_pn(SecularPnState& s, double dt, double nu, double Gm, double c, int pn_order) {
    // Secular timestep criterion: eta_sec * min(eps / |eps'|, e_t^2 / |(e_t^2)'|) from the current state, re-evaluated
    // every sub-step (t_GW ~ a^4 shrinks within one interval of an inspiral). BIFROST Eq. 30 also has omega/omega-dot,
    // but omega is an angle (near omega = 0 every call ran the step cap), and the drift of omega is the midpoint rule
    // of omega-dot(eps, j): exact while eps, j are constant (the A/B of 4ebc740).
    const double eta_sec = 0.01;
    constexpr int max_steps = 100000;
    auto step_size = [&](double eps, double j) {
        double de, dj;
        secular_pn_rates(eps, j, nu, Gm, c, pn_order, de, dj);
        double t_sec = 1e30;
        if (std::fabs(de) > 0.0) t_sec = std::min(t_sec, eps / std::fabs(de));
        const double et2 = secular_pn_et2(eps, j, nu, pn_order);
        if (et2 > 1e-8) {
            const double h = 1e-7 * dt;
            const double det2 = (secular_pn_et2(eps + h * de, j + h * dj, nu, pn_order) - et2) / h;
            if (std::fabs(det2) > 0.0) t_sec = std::min(t_sec, 2.0 * et2 / std::fabs(det2));
        }
        return std::max(eta_sec * t_sec, dt / max_steps);
    };
    auto drift_rate = [&](double eps, double j, double& dw, double& dM) {
        dM = secular_pn_n(eps, j, nu, Gm, c, pn_order);
        dw = secular_pn_k(eps, j, nu, pn_order) * dM;
    };

    double eps_aux = s.eps, j_aux = s.j;
    s.dw = 0.0; s.dM = 0.0; s.merged = false;
    int n_steps = 0;
    for (double t_done = 0.0; t_done < dt && n_steps < max_steps; n_steps++) {
        double h = step_size(s.eps, s.j);
        if (h >= (dt - t_done) * (1.0 - 1e-12)) h = dt - t_done;   // last sub-step lands on dt
        t_done = (h == dt - t_done) ? dt : t_done + h;
        for (int half = 0; half < 2; half++) {
            double de_a, dj_a, de_p, dj_p;
            secular_pn_rates(eps_aux, j_aux, nu, Gm, c, pn_order, de_a, dj_a);
            secular_pn_rates(s.eps, s.j, nu, Gm, c, pn_order, de_p, dj_p);
            s.eps += 0.5 * h * de_a; s.j += 0.5 * h * dj_a;
            eps_aux += 0.5 * h * de_p; j_aux += 0.5 * h * dj_p;
            if (half == 0) {
                double dw, dM;
                drift_rate(s.eps, s.j, dw, dM);
                s.dw += h * dw; s.dM += h * dM;
            }
        }
        // Coalescence: pericentre a (1 - e) < 20 G m / c^2 (10 Schwarzschild radii, collision_distance), a = G m / (eps c^2)
        const double e = std::sqrt(std::max(secular_pn_et2(s.eps, s.j, nu, pn_order), 0.0));
        if (!(s.eps > 0.0) || !(s.j > 0.0) || (1.0 - e) < 20.0 * s.eps) { s.merged = true; break; }
    }
    return n_steps;
}

// Secular triple inner orbit: (a, e) read as the mean-motion semi-major axis a = (G m / n^2)^(1/3) and e_t. Rates of
// n and e_t from Arun et al. 2009 eqs. 6.6-6.9 (1PN terms with PN3.5) in x = zeta^(2/3) (1 + 2 zeta^(2/3) / (1 - e_t^2))
// (eq. 4.13, zeta = G m n / c^3); periastron advance k n with eps, j from x, e_t to 1PN (eq. 6.11, etMH) when a 1PN
// accurate eps, j matters (PN2 or PN3.5).
void secular_pn_inner_rates(double a, double e, double nu, double Gm, double c, int pn_order,
                            double& da, double& de, double& dw) {
    da = 0.0; de = 0.0; dw = 0.0;
    const double n = std::sqrt(Gm / (a * a * a)), z23 = Gm / (a * c * c);
    const double e2 = e * e, e4 = e2 * e2, om = sec_one_m(e2);
    const bool pn1_elements = (pn_order & 18) != 0;
    const double x = pn1_elements ? z23 * (1.0 + 2.0 * z23 / om) : z23;
    if (pn_order & 4) {
        double N = (96.0/5.0 + 292.0/5.0 * e2 + 37.0/5.0 * e4) / std::pow(om, 3.5);
        double Ee = (304.0/15.0 + 121.0/15.0 * e2) / std::pow(om, 2.5);
        if (pn_order & 16) {
            N += x * (-4846.0/35.0 - 264.0/5.0 * nu + e2 * (5001.0/35.0 - 570.0 * nu) + e4 * (2489.0/4.0 - 5061.0/10.0 * nu)
                      + e4 * e2 * (11717.0/280.0 - 148.0/5.0 * nu)) / std::pow(om, 4.5);
            Ee += x * (-939.0/35.0 - 4084.0/45.0 * nu + e2 * (29917.0/105.0 - 7753.0/30.0 * nu)
                       + e4 * (13929.0/280.0 - 1664.0/45.0 * nu)) / std::pow(om, 3.5);
        }
        const double c3 = c * c * c, x2 = x * x;
        const double ndot = c3 * c3 * nu / (Gm * Gm) * x2 * x2 * x * std::sqrt(x) * N;
        da = -2.0/3.0 * a * ndot / n;
        de = -c3 / Gm * e * nu * x2 * x2 * Ee;
    }
    double eps = x, j = om;
    if (pn1_elements) {
        eps = x * (1.0 + x / om * (-0.75 - nu / 12.0 + e2 * (-1.25 + nu / 12.0)));
        j = (1.0 - e2 + 0.25 * eps * (-8.0 + 8.0 * nu)) / (1.0 - 0.25 * eps * (17.0 - 7.0 * nu));
    }
    dw = secular_pn_k(eps, j, nu, pn_order) * n;
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
    // C3 (octupole, (m1 - m2) with Blaes et al. 2002's equation signs; Antognini's kozai code has m2 - m1 with its own
    // signs) is recomputed per substep below

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
    // GW shrinkage of the inner orbit (C24): the step also resolves a / |da/dt| and e / |de/dt| of the PN rates. With
    // the Kozai time alone a BH inner binary with t_GW << dt took one explicit step past a = 0 (NaN members).
    // t_GW ~ a^4 only shortens within the call; a coalescing inner binary is refused by secular_triple_step
    {
        double da0 = 0.0, de0 = 0.0, dw0;
        secular_pn_inner_rates(tri.inner.a, tri.inner.e, m1 * m2 / (M12 * M12), mu_in, c_code, pn_order, da0, de0, dw0);
        double t_gw = 1e30;
        if (std::fabs(da0) > 0.0) t_gw = std::min(t_gw, tri.inner.a / std::fabs(da0));
        if (std::fabs(de0) > 0.0 && tri.inner.e > 1e-6) t_gw = std::min(t_gw, tri.inner.e / std::fabs(de0));
        if (t_gw < 1e30) n_steps = std::max(n_steps, (int)std::min(std::ceil(dt / (eta_sec * t_gw)), 1e5));
    }
    n_steps = std::min(n_steps, 100000);
    double h = dt / n_steps;

    for (int step = 0; step < n_steps; step++) {
        double a_in = tri.inner.a;
        double e1 = tri.inner.e;
        double g1 = tri.inner.omega;
        double g2 = tri.omega_out;
        double th = tri.cos_i_mut;

        if (a_in <= 0.0) break;
        e_out = tri.e_out; e_out2 = e_out * e_out; j2 = std::sqrt(1.0 - e_out2);
        G2 = (M12 * m3 / M_tot) * std::sqrt(G * M_tot * a_out) * j2;

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
        double C3_cur = 15.0 * G * m1 * m2 * m3 * (m1 - m2) /
                        (64.0 * M12 * M12 * a_out * std::pow(j2, 5)) * alpha * alpha * alpha;

        // Trigonometric quantities
        double sing1 = std::sin(g1), cosg1 = std::cos(g1);
        double sing2 = std::sin(g2), cosg2 = std::cos(g2);

        // Octupole auxiliary quantities (Blaes et al. 2002)
        double B = 2.0 + 5.0 * e1_2 - 7.0 * e1_2 * std::cos(2.0 * g1);
        double A = 4.0 + 3.0 * e1_2 - 2.5 * sin2i * B;
        double cosphi = -cosg1 * cosg2 - th * sing1 * sing2;

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

        // ---- de2/dt ---- (octupole; Blaes et al. 2002 eq. de2dt)
        double de2_dt = 0.0;
        if (std::fabs(C3_cur) > 0.0)
            de2_dt = C3_cur * e1 * (1.0 - e_out2) / G2 * (10.0 * th * sin2i * (1.0 - e1_2) * sing1 * cosg2
                                                          + A * (cosg1 * sing2 - th * sing1 * cosg2));

        // PN on the inner orbit: periastron advance (suppresses high-e Kozai) and GW dissipation (PN2.5, PN3.5)
        double da_in_dt = 0.0, de1_gw_dt = 0.0;
        {
            double dw_pn;
            secular_pn_inner_rates(a_in, e1, m1 * m2 / (M12 * M12), mu_in, c_code, pn_order, da_in_dt, de1_gw_dt, dw_pn);
            dg1_dt += dw_pn;
        }

        // Integrate
        // interaction (H conserved) first: cos i from H, G1 and G2 after the Kozai-Lidov change
        const double e_kl = std::max(1e-6, std::min(tri.inner.e + h * de1_dt, 0.9999));
        tri.e_out = std::max(1e-6, std::min(tri.e_out + h * de2_dt, 0.9999));
        const double G2_new = (M12 * m3 / M_tot) * std::sqrt(G * M_tot * a_out) * std::sqrt(1.0 - tri.e_out * tri.e_out);
        const double G1_kl = m1 * m2 / M12 * std::sqrt(G * M12 * a_in) * std::sqrt(1.0 - e_kl * e_kl);
        tri.cos_i_mut = std::max(-1.0, std::min((H2 - G1_kl * G1_kl - G2_new * G2_new) / (2.0 * G1_kl * G2_new), 1.0));
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
        // GW changes |G1| but not the planes (Blaes et al. 2002 eq. 37): H follows at fixed cos i; the radiated part
        tri.dG1_gw += G1_new - G1_kl;
        H2 = G1_new * G1_new + G2_new * G2_new + 2.0 * G1_new * G2_new * tri.cos_i_mut;

        if (tri.inner.a <= 0.0) break;
    }
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
    double a1 = 0.0;
    bool reshape = false;
    if (pn_order != 0) {
        // Orbit-averaged PN evolution of the gauge-invariant eps, j of the state (secular_evolve_pn), the pericentre
        // angle (k n) and the mean anomaly (n): the Kepler drift advances M by the Newtonian n(a0), the PN mean motion
        // differs by O(x) (for a 10 + 10 Msun binary at 0.01 AU, x = 2e-5 over 4.5e4 orbits: 6 rad per step)
        const double rr = std::sqrt(dot3(r, r));
        const double beta = 2.0 * mu / rr - dot3(v, v);
        double h[3], vh[3], hhat[3];
        cross3(r, v, h);
        cross3(v, h, vh);
        double e2 = 0.0;
        for (int c = 0; c < 3; c++) { const double ec = vh[c] / mu - r[c] / rr; e2 += ec * ec; }
        const double e0 = std::sqrt(e2);
        const double hn = std::sqrt(dot3(h, h));
        if (!(beta > 0.0) || e0 >= 1.0 || !(hn > 0.0)) return 0;   // unbound or radial: not a secular binary
        for (int c = 0; c < 3; c++) hhat[c] = h[c] / hn;
        const double nu = m1 * m2 / (M_tot * M_tot);
        if (secular_pn_peri_shift(r, v, mu, nu, c_code, pn_order) > SECULAR_PN_PERI_SHIFT_MAX) return -1;
        SecularPnState s;
        secular_pn_eps_j(r, v, mu, nu, c_code, pn_order, s.eps, s.j);
        if (!(s.eps > 0.0) || !(s.j > 0.0)) return 0;
        n_sub = secular_evolve_pn(s, dt, nu, mu, c_code, pn_order);
        // Merger (pericentre inside 6 G m / c^2): leave the particles as they are (as before)
        if (s.merged) {
            printf("  [SEC] WARNING: binary merged during secular evolution "
                   "(pericentre < 10 Schwarzschild radii)\n");
            return n_sub;
        }
        const double a0 = mu / beta;
        const double dM = s.dM - std::sqrt(mu / (a0 * a0 * a0)) * dt;
        kepler_drift(r, v, mu, dt);
        if (s.dw != 0.0) rotate_about(r, v, hhat, s.dw);
        // Newtonian orbit (a1, k e0) at mean anomaly M + dM whose eps, j are the evolved ones; the drifted state is not
        // on them (a Newtonian orbit is not a PN orbit: eps, j of its states vary by O(x) along it). Fixed point: eps
        // ~ 1 / a, e_t^2 ~ e^2 to leading order, converging by O(x) per iteration
        const double et2_1 = secular_pn_et2(s.eps, s.j, nu, pn_order);
        const double r_d[3] = {r[0], r[1], r[2]}, v_d[3] = {v[0], v[1], v[2]};
        double eN2 = e2;
        a1 = a0;
        // Fixed-point residual of the state written back; the iteration contracts by ~ the peri shift (<= 0.1 here),
        // and the eps of a state near the pericentre carries round-off ~ 1e-16 v^2 / |E| ~ 1e-16 / (1 - e)
        double fp_res = INFINITY;
        for (int it = 0; it < 60; it++) {
            for (int c = 0; c < 3; c++) { r[c] = r_d[c]; v[c] = v_d[c]; }
            reshape_orbit(r, v, mu, a1, e0 > 0.0 ? std::sqrt(eN2) / e0 : 1.0, dM);
            double eps, j;
            secular_pn_eps_j(r, v, mu, nu, c_code, pn_order, eps, j);
            const double d_et2 = et2_1 - secular_pn_et2(eps, j, nu, pn_order);
            const double f_eps = eps / s.eps;
            fp_res = (a1 > 0.0 && eps > 0.0) ? std::fabs(f_eps - 1.0) + std::fabs(d_et2) : INFINITY;
            if (std::fabs(f_eps - 1.0) < 1e-15 && (std::fabs(d_et2) < 1e-15 || !(e0 > 0.0) || (eN2 == 0.0 && d_et2 < 0.0))) break;
            a1 *= f_eps;
            eN2 = std::min(std::max(eN2 + d_et2, 0.0), 1.0 - 1e-12);
        }
        // No Newtonian orbit at this phase carries the evolved eps, j (the write-back diverged, C19): leave the pair
        // unchanged for LogH instead of writing an orbit of the wrong (possibly positive) energy
        if (!(fp_res < 1e-6)) {
            for (int c = 0; c < 3; c++) { r[c] = r_in[c]; v[c] = v_in[c]; }
            return -1;
        }
        reshape = true;
    } else {
        kepler_drift(r, v, mu, dt);
    }

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
