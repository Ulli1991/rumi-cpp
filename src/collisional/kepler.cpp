#include "kepler.h"
#include <cmath>
#include <algorithm>
#include <limits>

// ============================================================================
// Kepler equation solver: Mikkola (1987) cubic start with the empirical
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

    // Mikkola (1987) cubic starting value with the empirical correction of Rantala et al. (2023)
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

// Universal-variable drift: Kepler's equation in X from the state itself; the new state from f and g (peri = false)
// or from the pericentre (peri = true; kepler_drift below chooses)
static int kepler_drift_fg(double r[3], double v[3], double mu, double dt, bool peri) {
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
    if (beta <= 0.0) {
        // Parabolic regime (|h| beyond the pericentre time, small |beta|): F ~ mu X^3 / 6; of the guesses, the
        // smallest |X| (each overshoots outside its own regime)
        const double Xc = std::cbrt(6.0 * h / mu);
        if (std::fabs(Xc) < std::fabs(X)) X = Xc;
    }
    // dF/dX = r >= q (pericentre distance): the root lies within h / q of 0, so the bracket is always finite
    {
        double hv[3];
        hv[0] = r[1]*v[2] - r[2]*v[1]; hv[1] = r[2]*v[0] - r[0]*v[2]; hv[2] = r[0]*v[1] - r[1]*v[0];
        const double p = (hv[0]*hv[0] + hv[1]*hv[1] + hv[2]*hv[2]) / mu;
        const double q = p / (1.0 + std::sqrt(std::max(0.0, 1.0 - p * beta / mu)));
        if (q > 0.0) {
            const double Xmax = std::fabs(h) / q * 1.001;   // margin: e (so q) from sqrt(1 - p beta / mu) is only good to ~1e-8
            lo = std::max(lo, -Xmax);
            hi = std::min(hi, Xmax);
        }
    }
    if (h > 0.0) lo = std::max(lo, 0.0); else hi = std::min(hi, 0.0);
    if (!(X > lo && X < hi)) X = (std::isfinite(lo) && std::isfinite(hi)) ? 0.5 * (lo + hi) : (h > 0.0 ? 1.0 : -1.0) * std::fabs(X_lin);

    // Laguerre-Conway iteration (n = 5; Conway 1986, as WHFast and Wisdom & Hernandez 2015 for long steps and high e),
    // kept inside the bracket by bisection (doubling while one side is open: radial orbits only). Stops when X repeats (WHFast: the
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

    const double rr = r0 + eta0 * G1 + zeta0 * G2;     // r at the end
    // r = f r0 + g v0 cancels when the end is much closer to the pericentre than the start (|f r0|, |g v0| ~ r0 >> r:
    // per drift from r0 ~ a to the pericentre at 1 - e = 1e-6 an energy error of 2e-10 mu / q, |dL/L| 1.5e-10,
    // ctest kepler_near_parabolic). There (peri: near the parabola, rr < r0 / 2) the end state is built from the
    // pericentre instead. Not for every drift: from the pericentre a short drift far out sums s1 = s0 + X with
    // |s0| >> |X|, which is accurate in value but its derivative cancels (Enzyme symplectic defect 3e-5 at the
    // apocentre, 1 - e = 1e-8, against 8e-16 for f and g)
    if (peri && rr < 0.5 * r0) {
        // New state from the pericentre. Universal anomaly s from the pericentre
        // (ds = dt / r): r = q + mu e G2(s), r.v = dr/ds = mu e G1(s), and in the perifocal frame (P along e_vec,
        // Q = L_hat x P) the f and g functions from the pericentre (r_q = q P, v_q = (h / q) Q) give
        //   x = q - mu G2, y = h G1, vx = -mu G1 / r, vy = h c0 / r   (q - mu (1 - e) G2 = q (1 - beta G2) = q c0)
        // with no cancellation at the pericentre: an error in s (the X of the start plus the start's anomaly s0)
        // only moves the point along the orbit, E and L are the round-off of the construction (universal variables:
        // Danby 1992, ch. 6; the anomaly route of kepler_drift does the same with E or F)
        const double hv[3] = {r[1]*v[2] - r[2]*v[1], r[2]*v[0] - r[0]*v[2], r[0]*v[1] - r[1]*v[0]};
        const double hn = std::sqrt(hv[0]*hv[0] + hv[1]*hv[1] + hv[2]*hv[2]);
        const double Lh[3] = {hv[0] / hn, hv[1] / hn, hv[2] / hn};
        double Ph[3] = {(v[1]*hv[2] - v[2]*hv[1]) / mu - r[0] / r0, (v[2]*hv[0] - v[0]*hv[2]) / mu - r[1] / r0,
                        (v[0]*hv[1] - v[1]*hv[0]) / mu - r[2] / r0};
        const double e = std::sqrt(Ph[0]*Ph[0] + Ph[1]*Ph[1] + Ph[2]*Ph[2]);
        const double pl = Ph[0]*Lh[0] + Ph[1]*Lh[1] + Ph[2]*Lh[2];   // e_vec . L_hat: 0 up to round-off
        for (int k = 0; k < 3; k++) Ph[k] -= pl * Lh[k];
        const double pn = std::sqrt(Ph[0]*Ph[0] + Ph[1]*Ph[1] + Ph[2]*Ph[2]);
        for (int k = 0; k < 3; k++) Ph[k] /= pn;
        const double Qh[3] = {Lh[1]*Ph[2] - Lh[2]*Ph[1], Lh[2]*Ph[0] - Lh[0]*Ph[2], Lh[0]*Ph[1] - Lh[1]*Ph[0]};
        const double q = hn * hn / mu / (1.0 + e), mue = mu * e;
        // Anomaly of the start: mu e G1(s0) = r0.v0, mu e c0(beta s0^2) = zeta0 (= mu - beta r0), so with
        // t = eta0 / zeta0: s0 = atan(sqrt(beta) t) / sqrt(beta) (bound), atanh(sqrt(-beta) t) / sqrt(-beta)
        // (hyperbolic), = t Phi(beta t^2) with Phi(z) = sum_k (-z)^k / (2k + 1), analytic through beta = 0 (the
        // parabola; separate formulas per sign of beta would give a wrong derivative there). The series for
        // |beta t^2| < 1e-2, else atan2 (any E0) and asinh (no cancellation far out on the hyperbola)
        double s1;
        const double t0 = zeta0 > 0.0 ? eta0 / zeta0 : 0.0, z0 = beta * t0 * t0;
        if (zeta0 > 0.0 && std::fabs(z0) < 1e-2) {
            double phi = 1.0, term = 1.0;
            for (int k = 1; k < 30; k++) {
                term *= -z0;
                const double phin = phi + term / (2 * k + 1);
                if (phin == phi) break;
                phi = phin;
            }
            s1 = t0 * phi + X;
        } else if (beta > 0.0) {
            const double sb = std::sqrt(beta);
            s1 = std::atan2(eta0 * sb, zeta0) / sb + X;
        } else {
            const double k = std::sqrt(-beta);
            s1 = std::asinh(eta0 * k / mue) / k + X;
        }
        if (beta > 0.0) {
            // G_n periodic in s (whole orbits): reduce to |beta s1^2| <= pi^2 (n fixed, so the derivative keeps -n dP)
            const double Ps = 2.0 * M_PI / std::sqrt(beta);
            if (std::fabs(s1) > 0.5 * Ps) s1 -= Ps * std::nearbyint(s1 / Ps);
        }
        stumpff_c(beta * s1 * s1, c0, c1, c2, c3);
        const double S1 = s1 * c1, S2 = s1 * s1 * c2;
        const double rq = q + mue * S2;
        const double x = q - mu * S2, y = hn * S1, vx = -mu * S1 / rq, vy = hn * c0 / rq;
        for (int k = 0; k < 3; k++) {
            r[k] = x * Ph[k] + y * Qh[k];
            v[k] = vx * Ph[k] + vy * Qh[k];
        }
        return it + 1;
    }

    // Gauss f and g functions in incremental form, r = r0 + (f - 1) r0 + g v0 (WHFast Eqs. 15-16, 20)
    const double fm1 = -mu * G2 / r0;
    const double g = r0 * G1 + eta0 * G2;
    const double fd = -mu * G1 / (r0 * rr);
    const double gdm1 = -mu * G2 / rr;
    for (int k = 0; k < 3; k++) {
        const double x = r[k], u = v[k];
        r[k] = x + (fm1 * x + g * u);
        v[k] = u + (fd * x + gdm1 * u);
    }
    return it + 1;
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

int kepler_drift(double r[3], double v[3], double mu, double dt) {
    const double r0 = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    if (dt == 0.0 || !(r0 > 0.0) || !(mu > 0.0)) return 0;
    // Invariants: L = r x v, p = L^2 / mu, e_vec = v x L / mu - r / r, beta = mu / a
    const double h[3] = {r[1]*v[2] - r[2]*v[1], r[2]*v[0] - r[0]*v[2], r[0]*v[1] - r[1]*v[0]};
    const double hn = std::sqrt(h[0]*h[0] + h[1]*h[1] + h[2]*h[2]);
    const double ev[3] = {(v[1]*h[2] - v[2]*h[1]) / mu - r[0] / r0, (v[2]*h[0] - v[0]*h[2]) / mu - r[1] / r0,
                          (v[0]*h[1] - v[1]*h[0]) / mu - r[2] / r0};
    const double e = std::sqrt(ev[0]*ev[0] + ev[1]*ev[1] + ev[2]*ev[2]);
    const double v2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    const double eta0 = r[0]*v[0] + r[1]*v[1] + r[2]*v[2];
    const double beta = 2.0 * mu / r0 - v2;
    // Near-circular orbits (and radial ones, no plane, and the exact parabola): universal f and g from the state, where
    // elements are undefined. Eccentric orbits: f and g from a far point to one near the pericentre cancel (r = f r0
    // + g v0 with |f r0|, |g v0| >> r; the energy error grew as eps / (1 - e)^2, 5e-10 at e = 0.9998 from the
    // apocentre side), so the new state is built in the perifocal frame from the anomaly, every component without
    // cancellation (1 - e = (p / a) / (1 + e), cos E - e = (1 - e) - 2 sin^2(E / 2); Danby 1992, sec. 6.3-6.4): its
    // error is the round-off of the new point itself, as in the elements route, while the frame (P along e_vec,
    // Q = L_hat x P) and the anomaly (atan2 / asinh of r.v and e cos E) come from the state without acos
    // Near the parabola the anomaly form loses the phase (bound, 1 - e < 1e-5: 1 vs 16 drifts 3e-11 ... 4e-4 at
    // 1 - e = 1e-12) or fails (hyperbolic, e - 1 < 1e-4), where the universal variables stay at round-off; there the
    // drift that ends near the pericentre builds the end state from it (kepler_drift_fg, peri)
    if (e < 0.1 || !(hn > 0.0) || beta == 0.0 || (e < 1.0 && 1.0 - e < 1e-5) || (e >= 1.0 && e - 1.0 < 1e-4))
        return kepler_drift_fg(r, v, mu, dt, hn > 0.0 && e >= 0.1);
    const double zeta0 = mu - beta * r0;                 // mu e cos E (bound), mu e cosh F (hyperbolic)
    const double p = hn * hn / mu;
    double Ph[3], Qh[3];
    for (int c = 0; c < 3; c++) Ph[c] = ev[c] / e;
    const double Lh[3] = {h[0] / hn, h[1] / hn, h[2] / hn};
    Qh[0] = Lh[1]*Ph[2] - Lh[2]*Ph[1]; Qh[1] = Lh[2]*Ph[0] - Lh[0]*Ph[2]; Qh[2] = Lh[0]*Ph[1] - Lh[1]*Ph[0];
    double x, y, vx, vy;
    int it = 0;
    if (beta > 0.0) {
        const double a = mu / beta, sb = std::sqrt(beta), n = beta * sb / mu;
        const double ome = (p / a) / (1.0 + e);            // 1 - e
        const double E0 = std::atan2(eta0 * sb, zeta0);
        // Mean anomaly advanced over dt mod P; Kepler's equation for the change dE (Newton in a bracket):
        // dE - e cos E0 sin dE + e sin E0 (1 - cos dE) = dM, |dE - dM| <= 2 e
        double h_red = dt;
        const double P = 2.0 * M_PI / n;
        if (std::fabs(h_red) > 0.5 * P) h_red = std::remainder(h_red, P);
        const double dM = n * h_red;
        const double C = zeta0 / mu, S = eta0 * sb / mu;   // e cos E0, e sin E0
        double lo = dM - 2.0 * e, hi = dM + 2.0 * e;
        double d = dM + C * std::sin(dM) - S * (1.0 - std::cos(dM));
        if (!(d > lo && d < hi)) d = dM;
        for (it = 0; it < 100; it++) {
            const double sd = std::sin(d), sh = std::sin(0.5 * d);
            const double f = d - C * sd + 2.0 * S * sh * sh - dM;
            if (f == 0.0) break;
            if (f < 0.0) lo = std::max(lo, d); else hi = std::min(hi, d);
            double dn = d - f / (1.0 - C * std::cos(d) + S * sd);
            if (!(dn > lo && dn < hi)) dn = 0.5 * (lo + hi);
            if (dn == d) break;
            const double step = std::fabs(dn - d);
            d = dn;
            if (step <= 4e-16 * std::max(1.0, std::fabs(d)) || !(hi > lo)) break;
        }
        const double E1 = E0 + d;
        const double s1 = std::sin(E1), c1 = std::cos(E1), sh1 = std::sin(0.5 * E1);
        const double b_a = std::sqrt(p / a);                // sqrt(1 - e^2)
        const double den = ome + 2.0 * e * sh1 * sh1;       // 1 - e cos E1
        x = a * (ome - 2.0 * sh1 * sh1);                    // a (cos E1 - e)
        y = a * b_a * s1;
        const double vs = std::sqrt(mu / a) / den;
        vx = -vs * s1;
        vy = vs * b_a * c1;
    } else {
        const double A = -mu / beta, k = std::sqrt(-beta), n = k * k * k / mu;
        const double em1 = (p / A) / (e + 1.0);             // e - 1
        const double F0 = std::asinh(eta0 * k / (mu * e));
        // e sinh F - F = M0 + n dt, with e sinh F - F = (e - 1) sinh F + (sinh F - F); monotonic in F
        const double M1 = em1 * std::sinh(F0) + sinh_minus_x(F0) + n * dt;
        auto kep = [&](double F) { return em1 * std::sinh(F) + sinh_minus_x(F) - M1; };
        double F = std::fabs(M1) / e > 1.0 ? std::copysign(std::log(2.0 * std::fabs(M1) / e), M1) : std::asinh(M1 / e);
        double lo = -std::numeric_limits<double>::infinity(), hi = std::numeric_limits<double>::infinity();
        for (it = 0; it < 200; it++) {
            const double f = kep(F);
            if (f == 0.0) break;
            if (f < 0.0) lo = std::max(lo, F); else hi = std::min(hi, F);
            double Fn = F - f / (e * std::cosh(F) - 1.0);
            if (!(Fn > lo && Fn < hi))
                Fn = (std::isfinite(lo) && std::isfinite(hi)) ? 0.5 * (lo + hi) : (f < 0.0 ? F + 1.0 : F - 1.0);
            if (Fn == F) break;
            const double step = std::fabs(Fn - F);
            F = Fn;
            if (step <= 4e-16 * std::max(1.0, std::fabs(F))) break;
        }
        const double sF = std::sinh(F), cF = std::cosh(F), shF = std::sinh(0.5 * F);
        const double b_A = std::sqrt(p / A);                // sqrt(e^2 - 1)
        const double den = em1 + 2.0 * e * shF * shF;       // e cosh F - 1
        x = A * (em1 - 2.0 * shF * shF);                    // A (e - cosh F)
        y = A * b_A * sF;
        const double vs = std::sqrt(mu / A) / den;
        vx = -vs * sF;
        vy = vs * b_A * cF;
    }
    for (int c = 0; c < 3; c++) {
        r[c] = x * Ph[c] + y * Qh[c];
        v[c] = vx * Ph[c] + vy * Qh[c];
    }
    return it + 1;
}

void kepler_advance(double r[3], double v[3], double mu, double dt) {
    kepler_drift(r, v, mu, dt);
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
    if (std::fabs(1.0 - e) < 1e-6 && std::fabs(eps) * rr < 1e-6 * mu) {   // parabolic in energy too: a bound radial orbit (e = 1, p = 0) takes the ellipse branch
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
