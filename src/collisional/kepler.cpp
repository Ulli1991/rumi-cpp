#include "kepler.h"
#include <cmath>
#include <algorithm>
#include <limits>

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
        // 3D case: omega relative to ascending node
        double cos_omega = (Nx*ex + Ny*ey) / (N_mag * oe.e);
        cos_omega = std::min(1.0, std::max(-1.0, cos_omega));
        oe.omega = std::acos(cos_omega);
        if (ez < 0.0) oe.omega = 2.0 * M_PI - oe.omega;
    } else if (oe.e > 1e-10) {
        // Planar orbit (inc ≈ 0): use longitude of periapsis
        // = angle of eccentricity vector from x-axis
        oe.omega = std::atan2(ey, ex);
        if (oe.omega < 0.0) oe.omega += 2.0 * M_PI;
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
// Universal-variable Kepler drift (kepler.h)
// ============================================================================

namespace {

// Stumpff c-functions c_n(z) = sum_j (-z)^j / (n + 2 j)!, n = 0..3. z is divided by 4 until |z| <= 0.1, c4 and c5 are
// summed there to convergence, and the quarter-angle formulae c4(z) = c3(z/4) (1 + c1(z/4)) / 8,
// c5(z) = (c5 + c4 + c3 c2)(z/4) / 16 (Mikkola 1997) with c_n = 1/n! - z c_{n+2} give the rest (WHFast, Rein & Tamayo
// 2015, sec. 2.4; their pseudo code has a sign typo in the c4 formula, their Eq. 9 is right). For z < 0 (hyperbolic)
// every term is positive, so the recursion has no cancellation at any |z|.
inline void stumpff_c(double z, double& c0, double& c1, double& c2, double& c3) {
    int n = 0;
    while (std::fabs(z) > 0.1) { z *= 0.25; n++; }
    // c4 = 1/4! - z/6! + z^2/8! ..., c5 = 1/5! - z/7! + ...
    double c4 = 1.0 / 24.0, c5 = 1.0 / 120.0;
    double term4 = c4, term5 = c5;
    for (int k = 0; k < 12; k++) {
        const double d4 = (6.0 + 2.0 * k) * (5.0 + 2.0 * k), d5 = (7.0 + 2.0 * k) * (6.0 + 2.0 * k);
        term4 *= -z / d4;
        term5 *= -z / d5;
        const double c4n = c4 + term4;
        c5 += term5;
        if (c4n == c4) break;
        c4 = c4n;
    }
    c3 = 1.0 / 6.0 - z * c5;
    c2 = 0.5 - z * c4;
    c1 = 1.0 - z * c3;
    for (; n > 0; n--) {
        z *= 4.0;
        c5 = (c5 + c4 + c3 * c2) * 0.0625;
        c4 = c3 * (1.0 + c1) * 0.125;
        c3 = 1.0 / 6.0 - z * c5;
        c2 = 0.5 - z * c4;
        c1 = 1.0 - z * c3;
    }
    c0 = 1.0 - z * c2;
}

}  // namespace

int kepler_drift(double r[3], double v[3], double mu, double dt) {
    const double r0 = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    if (dt == 0.0 || !(r0 > 0.0) || !(mu > 0.0)) return 0;
    const double v2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    const double eta0 = r[0]*v[0] + r[1]*v[1] + r[2]*v[2];
    const double beta = 2.0 * mu / r0 - v2;          // mu / a (> 0 bound, < 0 hyperbolic)
    const double zeta0 = mu - beta * r0;             // mu e cos E0 for bound orbits
    constexpr double inf = std::numeric_limits<double>::infinity();

    // Kepler's equation in the universal variable X (ds = dt / r): F(X) = r0 X + eta0 G2 + zeta0 G3 - h = 0 with
    // G_n = X^n c_n(beta X^2) (WHFast Eq. 13); F is monotonic (dF/dX = r > 0), so [lo, hi] brackets the root
    double h = dt;
    double lo = -inf, hi = inf, X;
    const double X_lin = h / r0 * (1.0 - 0.5 * eta0 * h / (r0 * r0));   // second order in h (WHFast Eq. 18)
    if (beta > 0.0) {
        // Bound: whole periods are the identity, so only h = dt mod P (in [-P/2, P/2]) is integrated; the phase
        // error of that reduction, eps n dt, is the conditioning of the problem itself
        const double sb = std::sqrt(beta);
        const double P = 2.0 * M_PI * mu / (beta * sb);
        if (std::fabs(h) > 0.5 * P) h = std::remainder(h, P);
        if (h == 0.0) return 0;
        // X = dE / sqrt(beta) with dE - dM = C sin dE + S (cos dE - 1), C = e cos E0, S = e sin E0, |dE - dM| <= 2 e
        const double dM = h * beta * sb / mu;
        const double C = zeta0 / mu, S = eta0 * sb / mu;
        lo = (dM - 2.0) / sb;
        hi = (dM + 2.0) / sb;
        X = (std::fabs(X_lin * sb) < 0.3) ? X_lin : (dM + C * std::sin(dM) + S * (std::cos(dM) - 1.0)) / sb;
    } else {
        X = X_lin;
        if (beta < 0.0) {
            // Hyperbolic, |h| beyond the crossing time: F ~ e^{k|X|} (r0 k^2 +- eta0 k + mu) / (2 k^3), k = sqrt(-beta)
            // (the bracket is > 0 for either sign), so |X| ~ ln(2 |h| k^3 / (r0 k^2 +- eta0 k + mu)) / k; the linear
            // guess overshoots exponentially there
            const double k = std::sqrt(-beta);
            if (std::fabs(X_lin) * k > 1.0) {
                const double sg = h > 0.0 ? 1.0 : -1.0;
                const double den = r0 * k * k + sg * eta0 * k + mu;
                const double arg = 2.0 * std::fabs(h) * k * k * k / den;
                if (den > 0.0 && arg > 1.0) X = sg * std::log(arg) / k;
            }
        }
    }
    if (h > 0.0) lo = std::max(lo, 0.0); else hi = std::min(hi, 0.0);
    if (!(X > lo && X < hi)) X = (std::isfinite(lo) && std::isfinite(hi)) ? 0.5 * (lo + hi) : (h > 0.0 ? 1.0 : -1.0) * std::fabs(X_lin);

    // Laguerre-Conway iteration (n = 5; Conway 1986, as WHFast and Wisdom & Hernandez 2015 for long steps and high e),
    // kept inside the bracket by bisection (or doubling while one side is open). Stops when X repeats (WHFast: the
    // iteration may cycle between neighbouring floating-point values once converged)
    double c0, c1, c2, c3, G1 = 0.0, G2 = 0.0, G3 = 0.0;
    double hist[8];
    int n_hist = 0, it = 0;
    double X_eval = std::numeric_limits<double>::quiet_NaN();
    for (; it < 100; it++) {
        stumpff_c(beta * X * X, c0, c1, c2, c3);
        X_eval = X;
        G1 = X * c1; G2 = X * X * c2; G3 = X * X * X * c3;
        const double F = r0 * X + eta0 * G2 + zeta0 * G3 - h;
        if (F == 0.0) break;
        if (F < 0.0) lo = std::max(lo, X); else hi = std::min(hi, X);
        const double Fp = r0 + eta0 * G1 + zeta0 * G2;     // r(X)
        const double Fpp = eta0 * c0 + zeta0 * G1;         // dr/dX
        const double disc = 16.0 * Fp * Fp - 20.0 * F * Fpp;
        double Xn = X - 5.0 * F / (Fp + std::copysign(std::sqrt(std::fabs(disc)), Fp));
        if (!(Xn > lo && Xn < hi)) {
            if (std::isfinite(lo) && std::isfinite(hi)) Xn = 0.5 * (lo + hi);
            else Xn = 2.0 * (std::isfinite(lo) ? std::max(std::fabs(lo), std::fabs(X)) : -std::max(std::fabs(hi), std::fabs(X)));
            if (!(Xn > lo && Xn < hi)) break;   // bracket exhausted at round-off
        }
        bool seen = (Xn == X);
        for (int q = 0; q < n_hist && !seen; q++) seen = (hist[q] == Xn);
        hist[n_hist < 8 ? n_hist++ : (it % 8)] = X;
        X = Xn;
        if (seen) break;
    }
    if (X != X_eval) {
        stumpff_c(beta * X * X, c0, c1, c2, c3);
        G1 = X * c1; G2 = X * X * c2; G3 = X * X * X * c3;
    }

    // Gauss f and g functions in incremental form, r = r0 + (f - 1) r0 + g v0 (WHFast Eqs. 15-16, 20)
    const double rr = r0 + eta0 * G1 + zeta0 * G2;
    const double fm1 = -mu * G2 / r0;
    const double g = h - mu * G3;
    const double fd = -mu * G1 / (r0 * rr);
    const double gdm1 = -mu * G2 / rr;
    for (int k = 0; k < 3; k++) {
        const double x = r[k], u = v[k];
        r[k] = x + (fm1 * x + g * u);
        v[k] = u + (fd * x + gdm1 * u);
    }
    return it + 1;
}

void kepler_advance(double r[3], double v[3], double mu, double dt) {
    kepler_drift(r, v, mu, dt);
}


// x - sin x and sinh x - x without cancellation for small |x|
static double x_minus_sin(double x) {
    if (std::fabs(x) > 0.1) return x - std::sin(x);
    const double x2 = x * x;
    return x * x2 / 6.0 * (1.0 - x2 / 20.0 * (1.0 - x2 / 42.0 * (1.0 - x2 / 72.0)));
}
static double sinh_minus_x(double x) {
    if (std::fabs(x) > 0.1) return std::sinh(x) - x;
    const double x2 = x * x;
    return x * x2 / 6.0 * (1.0 + x2 / 20.0 * (1.0 + x2 / 42.0 * (1.0 + x2 / 72.0)));
}

double kepler_time_to_pericentre(const double r[3], const double v[3], double mu, double* r_peri) {
    const double rr = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    const double v2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    const double rv = r[0]*v[0] + r[1]*v[1] + r[2]*v[2];
    const double hx = r[1]*v[2] - r[2]*v[1], hy = r[2]*v[0] - r[0]*v[2], hz = r[0]*v[1] - r[1]*v[0];
    const double p = (hx*hx + hy*hy + hz*hz) / mu;              // semi-latus rectum
    const double eps = 0.5 * v2 - mu / rr;                      // specific energy
    const double e = std::sqrt(std::max(0.0, 1.0 + 2.0 * eps * p / mu));
    if (r_peri) *r_peri = p / (1.0 + e);
    constexpr double inf = std::numeric_limits<double>::infinity();
    if (e < 1e-12) return 0.0;                                  // circular: every point is the pericentre
    if (std::fabs(1.0 - e) < 1e-6) {
        // Near-parabolic: Barker's equation, t - t_p = sqrt(p^3 / mu) (D + D^3 / 3) / 2 with D = tan(nu / 2) =
        // r.v / sqrt(mu p) (relative error O(|1 - e|); the caller iterates on the pericentre anyway)
        const double D = rv / std::sqrt(mu * p);
        const double t_since = 0.5 * std::sqrt(p * p * p / mu) * (D + D * D * D / 3.0);
        if (t_since <= 0.0) return -t_since;
        if (e >= 1.0) return inf;
        // bound by a hair: the next pericentre one (very long) period later
        const double a = -mu / (2.0 * eps);
        return 2.0 * M_PI * std::sqrt(a * a * a / mu) - t_since;
    }
    if (eps < 0.0) {
        // Ellipse: r.v = e sqrt(mu a) sin E, r = a (1 - e cos E); M = E - e sin E with 1 - e = (p / a) / (1 + e)
        const double a = -mu / (2.0 * eps);
        const double n = std::sqrt(mu / (a * a * a));
        const double E = std::atan2(rv / (e * std::sqrt(mu * a)), (1.0 - rr / a) / e);
        const double M = x_minus_sin(E) + (p / a) / (1.0 + e) * std::sin(E);
        return M < 0.0 ? -M / n : (2.0 * M_PI - M) / n;
    }
    // Hyperbola: r.v = e sqrt(mu A) sinh F (A = -a); M_h = e sinh F - F with e - 1 = (p / A) / (e + 1)
    if (rv >= 0.0) return inf;
    const double A = mu / (2.0 * eps);
    const double n = std::sqrt(mu / (A * A * A));
    const double F = std::asinh(rv / (e * std::sqrt(mu * A)));
    const double Mh = sinh_minus_x(F) + (p / A) / (e + 1.0) * std::sinh(F);
    return -Mh / n;
}
