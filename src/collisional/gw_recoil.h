#pragma once

// Gravitational-wave recoil of a BH-BH merger with spins (Campanelli et al. 2007, ApJ 659, L5; Gonzalez et al. 2007;
// Lousto et al. 2012, PRD 85, 084015, Eqs. 1-4; as used by CMC, Rodriguez et al. 2018, and by Gerosa & Kesden 2016's
// "precession"): v = v_m e1 + v_perp (cos xi e1 + sin xi e2) + v_par L_hat with
//   v_m    = A eta^2 (1 - q) / (1 + q) (1 + B eta)                               (mass asymmetry)
//   v_perp = H eta^2 / (1 + q) (chi1_par - q chi2_par)                          (aligned spins)
//   v_par  = 16 eta^2 / (1 + q) [V11 + VA S + VB S^2 + VC S^3] |chi1_perp - q chi2_perp| cos(phi)   ("superkick")
// with q = m2 / m1 <= 1 (body 1 the heavier), eta = q / (1 + q)^2, S = 2 (chi1_par + q^2 chi2_par) / (1 + q)^2, the
// spin components along / perpendicular to the orbital angular momentum, A = 1.2e4 km/s, B = -0.93, H = 6.9e3 km/s,
// xi = 145 deg, V11 = 3677.76, VA = 2481.21, VB = 1792.45, VC = 1506.52 km/s. phi (the angle between the in-plane
// spin difference and the infall direction at merger) and the orientation of e1 in the orbital plane are not known
// from the orbit-averaged state; they are drawn from `u1`, `u2` in [0, 1) (deterministic hashes in the caller, so all
// MPI ranks and restarts agree).

#include <cmath>

inline void gw_recoil_kick(double m1, double m2, const double chi1[3], const double chi2[3], const double L_orb[3],
                           double u1, double u2, double v_kick[3]) {
    // body 1 = the heavier
    const double* c1 = chi1;
    const double* c2 = chi2;
    if (m2 > m1) { const double t = m1; m1 = m2; m2 = t; c1 = chi2; c2 = chi1; }
    const double q = m2 / m1, eta = q / ((1.0 + q) * (1.0 + q));
    double Lh[3] = {0.0, 0.0, 1.0};
    const double Ln = std::sqrt(L_orb[0] * L_orb[0] + L_orb[1] * L_orb[1] + L_orb[2] * L_orb[2]);
    if (Ln > 0.0) for (int k = 0; k < 3; k++) Lh[k] = L_orb[k] / Ln;
    // In-plane basis: a = any unit vector perpendicular to L_hat, b = L_hat x a; e1 rotated by 2 pi u2 in the plane
    double a[3];
    if (std::fabs(Lh[2]) < 0.9) { a[0] = Lh[1]; a[1] = -Lh[0]; a[2] = 0.0; }
    else { a[0] = 0.0; a[1] = Lh[2]; a[2] = -Lh[1]; }
    const double an = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    for (double& x : a) x /= an;
    const double b[3] = {Lh[1] * a[2] - Lh[2] * a[1], Lh[2] * a[0] - Lh[0] * a[2], Lh[0] * a[1] - Lh[1] * a[0]};
    const double th = 2.0 * M_PI * u2;
    double e1[3], e2[3];
    for (int k = 0; k < 3; k++) {
        e1[k] = std::cos(th) * a[k] + std::sin(th) * b[k];
        e2[k] = -std::sin(th) * a[k] + std::cos(th) * b[k];
    }
    const double c1p = c1[0] * Lh[0] + c1[1] * Lh[1] + c1[2] * Lh[2];
    const double c2p = c2[0] * Lh[0] + c2[1] * Lh[1] + c2[2] * Lh[2];
    double dperp[3];   // chi1_perp - q chi2_perp
    for (int k = 0; k < 3; k++) dperp[k] = (c1[k] - c1p * Lh[k]) - q * (c2[k] - c2p * Lh[k]);
    const double dperp_n = std::sqrt(dperp[0] * dperp[0] + dperp[1] * dperp[1] + dperp[2] * dperp[2]);

    constexpr double A = 1.2e4, B = -0.93, H = 6.9e3, XI = 145.0 * M_PI / 180.0;
    constexpr double V11 = 3677.76, VA = 2481.21, VB = 1792.45, VC = 1506.52;
    const double v_m = A * eta * eta * (1.0 - q) / (1.0 + q) * (1.0 + B * eta);
    const double v_perp = H * eta * eta / (1.0 + q) * (c1p - q * c2p);
    const double S = 2.0 * (c1p + q * q * c2p) / ((1.0 + q) * (1.0 + q));
    const double v_par = 16.0 * eta * eta / (1.0 + q) * (V11 + VA * S + VB * S * S + VC * S * S * S) * dperp_n *
                         std::cos(2.0 * M_PI * u1);
    for (int k = 0; k < 3; k++)
        v_kick[k] = v_m * e1[k] + v_perp * (std::cos(XI) * e1[k] + std::sin(XI) * e2[k]) + v_par * Lh[k];
}
