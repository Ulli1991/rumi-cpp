#include "logh.h"
#include "post_newtonian_core.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================================
// LogH: Logarithmic Hamiltonian regularization (Mikkola & Tanikawa 1999)
// Implementation follows Rantala et al. 2023, Section 3.3 and
// FROST (Rantala+ 2021) Section 2.3.
//
// Time-Transformed Leapfrog (TTL) with Omega = T + B
// where T = kinetic energy, B = -E₀ (positive for bound systems).
//
// Convention of Rantala et al. (2023), Eqs. 2-3: ds/dt = Omega, so dt/ds = 1/Omega.
// ALL equations of motion have 1/Omega:
//   dv/ds = a / Omega
//   dx/ds = v / Omega
//   dt/ds = 1 / Omega
//
// This IS a proper time reparameterization: the continuous-time limit
// preserves the Laplace-Runge-Lenz vector for Kepler orbits, so GBS
// extrapolation converges to precession-free dynamics.
//
// Leapfrog in fictitious time s (KDK):
//   Each kick solves the implicit midpoint EXACTLY via cubic equation:
//     Ω³ - Ω₀·Ω² - C·Ω - D = 0   (Newton's method, machine precision)
//   This guarantees exact time-reversibility → even-power GBS error expansion.
//
//   1. Half-kick:  v += (ds/2) * a / Ω_cubic       (exact implicit midpoint)
//   2. Full drift: x += ds * v / Omega(v_half)
//   3. Time:       t += ds / Omega(v_half)
//   4. Forces:     a = F(x)
//   5. Half-kick:  v += (ds/2) * a / Ω_cubic       (exact implicit midpoint)
//
// B is constant for purely Newtonian dynamics (external perturbation handled
// by FSI kicks, not inside LogH). When velocity-dependent forces (PN) are
// present, B evolves per Rantala et al. 2023, Eq. (4):
//   dB/ds = -Σ m_i * v_i · g_i / Omega
// discretized with SYMMETRIC (trapezoidal) rule: half before kick, half after.
//
// PN corrections use the Hellstrom & Mikkola (2010) auxiliary velocity
// technique with proper ABA operator splitting for time-reversibility.
// Wrapped in GBS extrapolation for high-accuracy convergence.
// ============================================================================

void logh_internal_accel(const LogHState& state, double G,
                         double accel[LOGH_MAX_BODIES * 3]) {
    int n = state.n_bodies;
    std::memset(accel, 0, n * 3 * sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double dx = state.pos[j*3+0] - state.pos[i*3+0];
            double dy = state.pos[j*3+1] - state.pos[i*3+1];
            double dz = state.pos[j*3+2] - state.pos[i*3+2];
            double r2 = dx*dx + dy*dy + dz*dz;
            double r = std::sqrt(r2);
            double inv_r3 = 1.0 / (r2 * r);

            double fx = G * inv_r3 * dx;
            double fy = G * inv_r3 * dy;
            double fz = G * inv_r3 * dz;

            accel[i*3+0] += state.mass[j] * fx;
            accel[i*3+1] += state.mass[j] * fy;
            accel[i*3+2] += state.mass[j] * fz;
            accel[j*3+0] -= state.mass[i] * fx;
            accel[j*3+1] -= state.mass[i] * fy;
            accel[j*3+2] -= state.mass[i] * fz;
        }
    }
}

double logh_kinetic_energy(const LogHState& state) {
    double T = 0.0;
    for (int i = 0; i < state.n_bodies; i++) {
        double v2 = state.vel[i*3+0]*state.vel[i*3+0]
                   + state.vel[i*3+1]*state.vel[i*3+1]
                   + state.vel[i*3+2]*state.vel[i*3+2];
        T += 0.5 * state.mass[i] * v2;
    }
    return T;
}

double logh_potential_energy(const LogHState& state, double G) {
    double W = 0.0;
    for (int i = 0; i < state.n_bodies; i++) {
        for (int j = i + 1; j < state.n_bodies; j++) {
            double dx = state.pos[j*3+0] - state.pos[i*3+0];
            double dy = state.pos[j*3+1] - state.pos[i*3+1];
            double dz = state.pos[j*3+2] - state.pos[i*3+2];
            double r = std::sqrt(dx*dx + dy*dy + dz*dz);
            W -= G * state.mass[i] * state.mass[j] / r;
        }
    }
    return W;
}

double logh_total_energy(const LogHState& state, double G) {
    return logh_kinetic_energy(state) + logh_potential_energy(state, G);
}

// ============================================================================
// Single TTL leapfrog step in regularized time s
// Rantala et al. (2023) convention: all operations scaled by 1/Omega for proper
// time reparameterization (preserves LRL vector in continuous limit).
//
// Implicit midpoint kick solved EXACTLY via cubic equation:
//   v_new = v_old + Δ/Ω  where Δ = (ds/2)*a
//   v_mid = (v_old + v_new)/2 = v_old + Δ/(2Ω)
//   Ω = T(v_mid) + B = Ω₀ + C/Ω + D/Ω²
//   → Ω³ - Ω₀·Ω² - C·Ω - D = 0  (cubic, solved by Newton's method)
//   C = ½ Σ mᵢ vᵢ·Δᵢ,  D = ⅛ Σ mᵢ|Δᵢ|²
//
// Properties:
//   - Exact implicit midpoint → exact time-reversibility
//   - Even-power error expansion (h², h⁴,...) for GBS extrapolation
//   - No spurious precession (proper 1/Ω reparameterization)
// ============================================================================

// Solve the implicit midpoint kick: v_new = v_old + delta/Omega
// where Omega satisfies Omega^3 - Omega0*Omega^2 - C*Omega - D = 0
// Omega0 = T(v_old) + B,  C = 0.5*sum(m*v*delta),  D = 0.125*sum(m*delta^2)
static double solve_kick_omega(double Omega0, double C, double D) {
    double Omega = Omega0;
    for (int iter = 0; iter < 5; iter++) {
        double O2 = Omega * Omega;
        double f = O2 * Omega - Omega0 * O2 - C * Omega - D;
        double fp = 3.0 * O2 - 2.0 * Omega0 * Omega - C;
        if (std::fabs(fp) < 1e-30) break;
        // Stop at a fixed point of the iteration: the following iterations would repeat the same operations on
        // the same Omega, so the result is the one of the 5 iterations (period-2 round-off cycles run on)
        const double Onew = Omega - f / fp;
        if (Onew == Omega) break;
        Omega = Onew;
    }
    return Omega;
}

// ============================================================================
// SDAR: slow-down algorithmic regularization (Mikkola & Aarseth 1996, CeMDA 64, 197; Wang, Nitadori & Makino
// 2020, MNRAS 493, 3398 = W20; FROST-CLUSTERS I, Rantala et al. 2024, App. A2 = FC-I).
//
// Slow-down Hamiltonian (W20 Eq. 19, FC-I Eq. A3) with n_sd disjoint inner binaries k = (a, b):
//   H_sd = sum_k H_b,k / kappa_k + (H - sum_k H_b,k),   H_b,k = mu_k v_ab^2 / 2 - G m_a m_b / r_ab
// split as T_sd + U_sd (FC-I Eq. A4, W20 Eq. 50):
//   T_sd = sum_k T_b,k / kappa_k + sum_k T_cm,k + sum_singles T_i,   U_sd = U + sum_k (1/kappa_k - 1) U_b,k
// With the binary internal kinetic energy T_b = mu v_ab^2 / 2, T_sd is the quadratic form
//   T_sd = <v, v>_sd / 2,  <u, w>_sd = sum_i m_i u_i.w_i + sum_k (1/kappa_k - 1) mu_k u_ab.w_ab.
// Equations of motion (W20 Eq. 20-21, FC-I Eq. A6-A7), here in the LogH time ds = Omega dt with
// Omega = T_sd + B, B = -H_sd on shell (FC-I Eq. A5):
//   binary member i of k:  dx_i/dt = v_cm + (v_i - v_cm) / kappa_k,  dv_i/dt = a_i + (1/kappa_k - 1) a_bin,i
//   single:                dx_i/dt = v_i,                           dv_i/dt = a_i
// The kick and drift of logh_leapfrog_step carry over with the inner product <.,.>_sd in the cubic for Omega.
// Momentum and angular momentum are conserved by H_sd (W20 sec. 2.5.2), so the subsystem c.o.m. still drifts
// uniformly. n_sd = 0 takes the unchanged plain-LogH code paths.
// ============================================================================

static double sd_dot(const LogHState& s, const double* u, const double* w) {
    double d = 0.0;
    for (int b = 0; b < s.n_bodies; b++)
        d += s.mass[b] * (u[b*3+0]*w[b*3+0] + u[b*3+1]*w[b*3+1] + u[b*3+2]*w[b*3+2]);
    for (int k = 0; k < s.n_sd; k++) {
        const int a = s.sd_a[k], b = s.sd_b[k];
        const double mu = s.mass[a] * s.mass[b] / (s.mass[a] + s.mass[b]);
        double p = 0.0;
        for (int c = 0; c < 3; c++) p += (u[a*3+c] - u[b*3+c]) * (w[a*3+c] - w[b*3+c]);
        d += (1.0 / s.sd_kappa[k] - 1.0) * mu * p;
    }
    return d;
}

double logh_sd_kinetic_energy(const LogHState& state) {
    if (state.n_sd == 0) return logh_kinetic_energy(state);
    return 0.5 * sd_dot(state, state.vel, state.vel);
}

double logh_sd_potential_energy(const LogHState& state, double G) {
    double U = logh_potential_energy(state, G);
    for (int k = 0; k < state.n_sd; k++) {
        const int a = state.sd_a[k], b = state.sd_b[k];
        double r2 = 0.0;
        for (int c = 0; c < 3; c++) {
            const double d = state.pos[b*3+c] - state.pos[a*3+c];
            r2 += d * d;
        }
        U += (1.0 / state.sd_kappa[k] - 1.0) * (-G * state.mass[a] * state.mass[b] / std::sqrt(r2));
    }
    return U;
}

double logh_omega(const LogHState& state) {
    return logh_sd_kinetic_energy(state) + state.B;
}

// a_i -> a_i + (1/kappa - 1) a_bin,i for the members of each slowed binary (W20 Eq. 20)
static void sd_scale_binary_accel(const LogHState& s, double G, double* accel) {
    for (int k = 0; k < s.n_sd; k++) {
        const int a = s.sd_a[k], b = s.sd_b[k];
        double d[3], r2 = 0.0;
        for (int c = 0; c < 3; c++) {
            d[c] = s.pos[b*3+c] - s.pos[a*3+c];
            r2 += d[c] * d[c];
        }
        const double f = (1.0 / s.sd_kappa[k] - 1.0) * G / (r2 * std::sqrt(r2));
        for (int c = 0; c < 3; c++) {
            accel[a*3+c] += f * s.mass[b] * d[c];
            accel[b*3+c] -= f * s.mass[a] * d[c];
        }
    }
}

// dx/dt of every body: v, with the internal velocity of slowed binary members divided by kappa (W20 Eq. 20)
static void sd_drift_velocity(const LogHState& s, double* xdot) {
    std::memcpy(xdot, s.vel, s.n_bodies * 3 * sizeof(double));
    for (int k = 0; k < s.n_sd; k++) {
        const int a = s.sd_a[k], b = s.sd_b[k];
        const double ma = s.mass[a], mb = s.mass[b], inv_k = 1.0 / s.sd_kappa[k];
        for (int c = 0; c < 3; c++) {
            const double vcm = (ma * s.vel[a*3+c] + mb * s.vel[b*3+c]) / (ma + mb);
            xdot[a*3+c] = vcm + (s.vel[a*3+c] - vcm) * inv_k;
            xdot[b*3+c] = vcm + (s.vel[b*3+c] - vcm) * inv_k;
        }
    }
}

// Implicit-midpoint kick of logh_leapfrog_step with the slowed metric: Omega = T_sd(v_mid) + B gives the same
// cubic Omega^3 - Omega0 Omega^2 - C Omega - D = 0 with C = <v, Delta>_sd / 2, D = <Delta, Delta>_sd / 8
static void sd_kick(LogHState& s, double h, const double* accel) {
    const int n3 = s.n_bodies * 3;
    double delta[LOGH_MAX_BODIES * 3];
    for (int i = 0; i < n3; i++) delta[i] = h * accel[i];
    const double Omega0 = logh_sd_kinetic_energy(s) + s.B;
    const double C = 0.5 * sd_dot(s, s.vel, delta), D = 0.125 * sd_dot(s, delta, delta);
    const double inv_Omega = 1.0 / solve_kick_omega(Omega0, C, D);
    for (int i = 0; i < n3; i++) s.vel[i] += delta[i] * inv_Omega;
}

static void logh_leapfrog_step_sd(LogHState& state, double ds, double G) {
    const int n3 = state.n_bodies * 3;
    double accel[LOGH_MAX_BODIES * 3], xdot[LOGH_MAX_BODIES * 3];
    logh_internal_accel(state, G, accel);
    sd_scale_binary_accel(state, G, accel);
    sd_kick(state, 0.5 * ds, accel);

    const double inv_Omega_half = 1.0 / (logh_sd_kinetic_energy(state) + state.B);
    sd_drift_velocity(state, xdot);
    for (int i = 0; i < n3; i++) state.pos[i] += ds * xdot[i] * inv_Omega_half;
    state.t += ds * inv_Omega_half;

    logh_internal_accel(state, G, accel);
    sd_scale_binary_accel(state, G, accel);
    sd_kick(state, 0.5 * ds, accel);
}

double logh_sd_dyn_time(const LogHState& state, double G, double M_total) {
    double t_min = 1e300;
    for (int i = 0; i < state.n_bodies; i++)
        for (int j = i + 1; j < state.n_bodies; j++) {
            double r2 = 0.0;
            for (int c = 0; c < 3; c++) {
                const double d = state.pos[j*3+c] - state.pos[i*3+c];
                r2 += d * d;
            }
            double t = std::sqrt(r2 * std::sqrt(r2) / (G * M_total));
            for (int k = 0; k < state.n_sd; k++)
                if (state.sd_a[k] == i && state.sd_b[k] == j) t *= state.sd_kappa[k];
            t_min = std::min(t_min, t);
        }
    return t_min;
}

// Jumping kappa (W20 sec. 2.4, steps i-iii). Inner binaries: bound pairs taken greedily from the smallest
// semi-major axis, each body in at most one pair (SDAR builds a Kepler binary tree for the same purpose, W20
// sec. 4); every other subsystem member is a perturber. Per pair:
//   kappa_org = k_ref (m_a m_b / r_apo^3) / sum_j (m_ab m_j / |r_j - r_cm|^3)        (W20 Eq. 22, FC-I Eq. A8)
// with the perturbations summed as in the SDAR library (pert_out += m_cm m_j / r^3; for one perturber this is
// exactly Eq. 22, whose printed form sums |r_j - r_cm|^3 / m_j and would let the weakest perturber set kappa),
//   kappa_max = c |<r_j - r_cm>_m| / (P |<v_j - v_cm>_m|)                             (W20 Eq. 28, FC-I Eq. A9)
// with mass-weighted means over the perturbers, kappa = min(kappa_org, kappa_max); pairs with kappa <= 1 are
// not slowed, nor collision candidates, pericentre < coll_factor (R_a + R_b) (FC-I App. B3).
// The sum over perturbers includes those outside the subsystem, m_ab par.ext_tidal (C21; PeTar sums all perturbers
// of a group, members and outside neighbours). Their tide acts through the FSI kicks on the slowed members, as the
// external potential U of W20 Eq. H2: H_sd is conserved, but the physical energy drifts by (1 - 1/kappa) of the
// binary's energy change by that tide, which no book holds (s3_s324_0: 8e-8 E_REF per interval). A pair whose
// outside tidal energy swing per orbit exceeds par.ext_share (FLYBY_CM_EPS E_REF) therefore keeps kappa = 1, as
// 2-body subsystems do (see logh_batch.cpp).
// The energy jump delta H_sd = H_sd(kappa') - H_sd(kappa) at fixed (x, v) (W20 Eq. 24) is booked into B, so
// that T_sd + U_sd + B stays the accumulated integration error (W20 Eq. 25; B is the LogH momentum of time).
double logh_sd_update(LogHState& state, double G, const SdarParams& par, bool clear) {
    const double H_old = logh_sd_kinetic_energy(state) + logh_sd_potential_energy(state, G);
    const int n = state.n_bodies;
    state.n_sd = 0;
    double P_eff_min = 1e300;
    if (!clear && n >= 3) {
        struct Cand { double a, e; int i, j; };
        Cand cand[LOGH_MAX_BODIES * (LOGH_MAX_BODIES - 1) / 2];
        int nc = 0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) {
                double r[3], v[3], r2 = 0.0, v2 = 0.0;
                for (int c = 0; c < 3; c++) {
                    r[c] = state.pos[j*3+c] - state.pos[i*3+c];
                    v[c] = state.vel[j*3+c] - state.vel[i*3+c];
                    r2 += r[c] * r[c];
                    v2 += v[c] * v[c];
                }
                const double mu = G * (state.mass[i] + state.mass[j]);
                const double eps = 0.5 * v2 - mu / std::sqrt(r2);
                if (eps >= 0.0) continue;
                const double hx = r[1]*v[2] - r[2]*v[1], hy = r[2]*v[0] - r[0]*v[2], hz = r[0]*v[1] - r[1]*v[0];
                const double e2 = 1.0 + 2.0 * eps * (hx*hx + hy*hy + hz*hz) / (mu * mu);
                cand[nc++] = {-mu / (2.0 * eps), std::sqrt(std::max(0.0, e2)), i, j};
            }
        std::sort(cand, cand + nc, [](const Cand& x, const Cand& y) { return x.a < y.a; });
        bool used[LOGH_MAX_BODIES] = {};
        for (int q = 0; q < nc; q++) {
            const int i = cand[q].i, j = cand[q].j;
            const double a = cand[q].a, e = cand[q].e;
            const double mi = state.mass[i], mj = state.mass[j], mb = mi + mj;
            const double P = 2.0 * M_PI * std::sqrt(a * a * a / (G * mb));
            if (used[i] || used[j]) continue;
            used[i] = used[j] = true;   // the tightest pair of a body is its inner orbit, slowed or not
            if (par.pn[i] || par.pn[j]) continue;   // PN pair: kappa = 1 (FC-I A2)
            const double r_apo = a * (1.0 + e);
            const double r_coll = par.coll_pair ? par.coll_pair[i * LOGH_MAX_BODIES + j]
                                                : par.coll_radius[i] + par.coll_radius[j];
            if (r_coll > 0.0 && a * (1.0 - e) < par.coll_factor * r_coll) continue;
            double xcm[3], vcm[3];
            for (int c = 0; c < 3; c++) {
                xcm[c] = (mi * state.pos[i*3+c] + mj * state.pos[j*3+c]) / mb;
                vcm[c] = (mi * state.vel[i*3+c] + mj * state.vel[j*3+c]) / mb;
            }
            double pert_out = 0.0, m_p = 0.0, R[3] = {0, 0, 0}, V[3] = {0, 0, 0};
            for (int k = 0; k < n; k++) {
                if (k == i || k == j) continue;
                double d[3], d2 = 0.0;
                for (int c = 0; c < 3; c++) {
                    d[c] = state.pos[k*3+c] - xcm[c];
                    d2 += d[c] * d[c];
                    R[c] += state.mass[k] * d[c];
                    V[c] += state.mass[k] * (state.vel[k*3+c] - vcm[c]);
                }
                m_p += state.mass[k];
                pert_out += mb * state.mass[k] / (d2 * std::sqrt(d2));
            }
            pert_out += mb * par.ext_tidal;
            if (pert_out <= 0.0) continue;
            const double kappa_org = par.k_ref * (mi * mj / (r_apo * r_apo * r_apo)) / pert_out;
            const double Rn = std::sqrt(R[0]*R[0] + R[1]*R[1] + R[2]*R[2]) / m_p;
            const double Vn = std::sqrt(V[0]*V[0] + V[1]*V[1] + V[2]*V[2]) / m_p;
            const double kappa_max = Vn > 0.0 ? par.c * Rn / (P * Vn) : kappa_org;
            double kappa = std::min(kappa_org, kappa_max);
            // Outside tide budget (C21): the binary's tidal energy swing over an orbit, est = mu |T| v P =
            // 2 pi mu G ext_tidal a^2 (T = G ext_tidal a, v P = 2 pi a), against the share FLYBY_CM_EPS |E0| of the
            // tidal-sampling steps; above it kappa = 1
            if (par.ext_share > 0.0 && par.ext_tidal > 0.0 && kappa > 1.0) {
                const double est = 2.0 * M_PI * mi * mj / mb * G * par.ext_tidal * a * a;
                if (est > par.ext_share) kappa = 1.0;
            }
            if (!(kappa > 1.0)) continue;
            state.sd_a[state.n_sd] = i;
            state.sd_b[state.n_sd] = j;
            state.sd_kappa[state.n_sd] = kappa;
            state.n_sd++;
            state.sd_kappa_peak = std::max(state.sd_kappa_peak, kappa);
        }
        // Shortest orbit over the bound pairs for the GBS step limit: a slowed pair's orbit takes kappa P, x
        // sqrt(1 - e) since eccentric orbits need more GBS steps per orbit (e = 0.9: ~140 measured, vs ~17 at low
        // e); an unslowed pair counts P as in plain LogH (same budget before a call gives up)
        for (int q = 0; q < nc; q++) {
            double P = 2.0 * M_PI * std::sqrt(cand[q].a * cand[q].a * cand[q].a /
                                              (G * (state.mass[cand[q].i] + state.mass[cand[q].j])));
            for (int k = 0; k < state.n_sd; k++)
                if (state.sd_a[k] == cand[q].i && state.sd_b[k] == cand[q].j)
                    P *= state.sd_kappa[k] * std::sqrt(std::max(1e-4, 1.0 - cand[q].e));
            P_eff_min = std::min(P_eff_min, P);
        }
    }
    const double dH = logh_sd_kinetic_energy(state) + logh_sd_potential_energy(state, G) - H_old;
    state.B -= dH;
    state.sd_jump += dH;
    return P_eff_min;
}

// Kick v += (ds/2) a / Omega with the exact implicit midpoint: Omega solves the cubic with
// Omega0 = T(v_old) + B (the caller's), C = ½ Σ mᵢ vᵢ·Δᵢ, D = ⅛ Σ mᵢ|Δᵢ|², Δ = (ds/2) a
static inline void kick_cubic(LogHState& state, const double* accel, double ds_half, double Omega0) {
    const int n = state.n_bodies, n3 = n * 3;
    double C = 0.0, D = 0.0;
    for (int b = 0; b < n; b++) {
        double vdot = 0.0, dsq = 0.0;
        for (int d = 0; d < 3; d++) {
            int idx = b * 3 + d;
            double delta = ds_half * accel[idx];
            vdot += state.vel[idx] * delta;
            dsq += delta * delta;
        }
        C += state.mass[b] * vdot;
        D += state.mass[b] * dsq;
    }
    C *= 0.5;
    D *= 0.125;

    double Omega = solve_kick_omega(Omega0, C, D);
    double inv_Omega = 1.0 / Omega;
    for (int i = 0; i < n3; i++)
        state.vel[i] += ds_half * accel[i] * inv_Omega;
}

// nsteps TTL leapfrog steps (kick, drift, kick). The acceleration at the start of a step is the one of the
// previous step's second kick (same positions), and that kick's Omega0 = T + B is the Omega of the drift (the drift
// changes neither v nor B): 1 acceleration and 2 kinetic energies per step instead of 2 and 3, with the same
// operations on the same values as single steps (bit-identical). start: acceleration and T + B at the first step
// (the same for all GBS stages of a step; nullptr: computed here). Slowed-down (SDAR) states take single steps.
void logh_leapfrog_steps(LogHState& state, double ds, double G, int nsteps, const LogHStepStart* start) {
    if (state.n_sd > 0) {
        for (int s = 0; s < nsteps; s++) logh_leapfrog_step_sd(state, ds, G);
        return;
    }
    const int n3 = state.n_bodies * 3;
    const double ds_half = 0.5 * ds;

    double accel[LOGH_MAX_BODIES * 3];   // internal Newtonian acceleration at the current positions
    double Omega0;                       // T + B at the current velocities
    if (start) {
        std::memcpy(accel, start->accel, n3 * sizeof(double));
        Omega0 = start->omega;
    } else {
        logh_internal_accel(state, G, accel);
        Omega0 = logh_kinetic_energy(state) + state.B;
    }

    for (int s = 0; s < nsteps; s++) {
        if (s > 0) Omega0 = logh_kinetic_energy(state) + state.B;   // velocities changed by the last kick

        // First half-kick: exact implicit midpoint via cubic solve
        kick_cubic(state, accel, ds_half, Omega0);

        // Drift
        double Omega_half = logh_kinetic_energy(state) + state.B;
        double inv_Omega_half = 1.0 / Omega_half;
        for (int i = 0; i < n3; i++)
            state.pos[i] += ds * state.vel[i] * inv_Omega_half;
        state.t += ds * inv_Omega_half;

        // Accelerations at the new positions; second half-kick (Omega0 = T + B unchanged by the drift)
        logh_internal_accel(state, G, accel);
        kick_cubic(state, accel, ds_half, Omega_half);
    }
}

void logh_leapfrog_step(LogHState& state, double ds, double G) {
    logh_leapfrog_steps(state, ds, G, 1);
}

// ============================================================================
// Helper: compute PN accelerations for all pairs using a given velocity array
// for the PN evaluation. Positions come from state.pos.
// pn_out[i*3+k] = individual PN acceleration on body i
// ============================================================================

// Physical spin S = chi G m^2 / c of body b
static void spin_from_chi(const LogHState& state, const double chi[3], int b, double G, double S[3]) {
    double f = G * state.mass[b] * state.mass[b] / state.c_code;
    for (int k = 0; k < 3; k++) S[k] = chi[k] * f;
}

// Spin-dependent PN accelerations (spin-orbit, spin-spin, Kerr quadrupole; Rantala et al. 2023, sec. 3.7) of a 2-body
// subsystem, evaluated at velocities eval_vel and dimensionless spins chi1, chi2; added to pn_out
static void add_spin_accel(const LogHState& state, const double* eval_vel, const double chi1[3],
                           const double chi2[3], double G, double* pn_out) {
    double x[3], v[3], S1[3], S2[3], a[3];
    for (int k = 0; k < 3; k++) {
        x[k] = state.pos[k] - state.pos[3 + k];
        v[k] = eval_vel[k] - eval_vel[3 + k];
    }
    spin_from_chi(state, chi1, 0, G, S1);
    spin_from_chi(state, chi2, 1, G, S2);
    spin_pn_acceleration(x, v, state.mass[0], state.mass[1], S1, S2, G, state.c_code, a);
    const double M = state.mass[0] + state.mass[1];
    for (int k = 0; k < 3; k++) {
        pn_out[k] += state.mass[1] / M * a[k];
        pn_out[3 + k] -= state.mass[0] / M * a[k];
    }
}

// Rotate the spins rot1, rot2 by the precession vectors evaluated at (x, eval_vel, chi1, chi2) over the
// physical time dt: exact (norm-preserving, time-symmetric) solution of dS/dt = Omega × S with Omega frozen
static void rotate_spins(const LogHState& state, const double* eval_vel, const double chi1[3],
                         const double chi2[3], double G, double dt, double rot1[3], double rot2[3]) {
    double x[3], v[3], S1[3], S2[3], O1[3], O2[3];
    for (int k = 0; k < 3; k++) {
        x[k] = state.pos[k] - state.pos[3 + k];
        v[k] = eval_vel[k] - eval_vel[3 + k];
    }
    spin_from_chi(state, chi1, 0, G, S1);
    spin_from_chi(state, chi2, 1, G, S2);
    spin_precession(x, v, state.mass[0], state.mass[1], S1, S2, G, state.c_code, O1, O2);
    rotate_about(rot1, O1, dt);
    rotate_about(rot2, O2, dt);
}

// The PN geometry of the state (masses, the EIH choice and the pair orders of compute_pn_all_pairs; positions)
static void pn_geom_init(const LogHState& state, double G, double c_code, int pn_order, PnGeom& geom) {
    // Three or more bodies: the PN1 part from the genuine N-body EIH equations (three-body terms included; pair sums
    // of the two-body formula have no conserved 1PN energy, docs/CASES.md C7); the higher orders stay pairwise
    // (binary formulas). Two bodies, and slowed (SDAR) pairs present: the pair formula for every order, as before
    const int n = state.n_bodies;
    const bool eih_nbody = n >= 3 && (pn_order & 1) && state.n_sd == 0;
    pn_geom_set_masses(state.mass, n, G, c_code, eih_nbody, eih_nbody ? (pn_order & ~1) : pn_order, geom);
    pn_geom_set_positions(state.pos, geom);
}

// PN accelerations at the positions of geom (pn_geom_init / pn_geom_set_positions at state.pos) and the velocities
// eval_vel: the position- and mass-only factors come from geom, computed once per position set (5 evaluations per
// leapfrog step at 2 position sets), bit-identical to eih1_nbody_acceleration and eih_pn_acceleration_pair
static void compute_pn_all_pairs(const LogHState& state, const PnGeom& geom, const double* eval_vel,
                                  double G, double c_code,
                                  double* pn_out, const double* chi1 = nullptr, const double* chi2 = nullptr) {
    int n = state.n_bodies;
    std::memset(pn_out, 0, n * 3 * sizeof(double));
    // NBODY_CHECK_EIH=1 (tests): every evaluation must equal the direct functions (eih1_nbody_acceleration, and
    // eih_pn_acceleration of each body of a pair) bit for bit
    static const bool check_eih = std::getenv("NBODY_CHECK_EIH") != nullptr;
    if (geom.eih_nbody) {
        eih1_nbody_acceleration_geom(geom, eval_vel, pn_out);
        if (check_eih) {
            double ref[LOGH_MAX_BODIES * 3];
            eih1_nbody_acceleration(state.pos, eval_vel, state.mass, n, G, c_code, ref);
            if (std::memcmp(ref, pn_out, n * 3 * sizeof(double))) {
                std::fprintf(stderr, "EIH CHECK FAILED: N-body PN1, n = %d: %a %a %a vs %a %a %a\n", n, pn_out[0],
                             pn_out[1], pn_out[2], ref[0], ref[1], ref[2]);
                std::abort();
            }
        }
    }
    const int pair_order = geom.pair_order;
    double a_pairs[PN_GEOM_MAX_DIR * 3];   // both bodies of every pair
    if (pair_order != 0) eih_pn_pairs_geom(geom, eval_vel, pair_order, a_pairs);
    int q = -1;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            q++;
            // Slowed (SDAR) pairs have no member above M_PN: their own PN term is left out. Applied at full strength
            // to the slowed orbit it acts kappa times longer per orbit (PN1 precession x kappa, ~0.3 rad per slowed
            // orbit for kappa 9e4 at 0.019 AU), which breaks the slow-down approximation
            bool slowed = false;
            for (int k = 0; k < state.n_sd; k++)
                if (state.sd_a[k] == i && state.sd_b[k] == j) slowed = true;
            if (slowed) continue;
            if (pair_order == 0) continue;
            const double* a_pn_i = &a_pairs[6*q];
            const double* a_pn_j = &a_pairs[6*q + 3];
            if (check_eih) {
                double ref_i[3], ref_j[3];
                eih_pn_acceleration(&state.pos[i*3], &eval_vel[i*3], state.mass[i],
                                    &state.pos[j*3], &eval_vel[j*3], state.mass[j], G, c_code, ref_i, pair_order);
                eih_pn_acceleration(&state.pos[j*3], &eval_vel[j*3], state.mass[j],
                                    &state.pos[i*3], &eval_vel[i*3], state.mass[i], G, c_code, ref_j, pair_order);
                if (std::memcmp(ref_i, a_pn_i, sizeof ref_i) || std::memcmp(ref_j, a_pn_j, sizeof ref_j)) {
                    std::fprintf(stderr, "EIH CHECK FAILED: pair %d-%d order %d: %a %a %a vs %a %a %a\n", i, j,
                                 pair_order, a_pn_i[0], a_pn_i[1], a_pn_i[2], ref_i[0], ref_i[1], ref_i[2]);
                    std::abort();
                }
            }
            for (int k = 0; k < 3; k++) {
                pn_out[i*3+k] += a_pn_i[k];
                pn_out[j*3+k] += a_pn_j[k];
            }
        }
    }
    if (state.has_spins && n == 2 && chi1) add_spin_accel(state, eval_vel, chi1, chi2, G, pn_out);
}

// ============================================================================
// Single TTL leapfrog step with PN corrections via auxiliary velocity technique
// (Hellstrom & Mikkola 2010, Rantala et al. 2023, Section 3.4)
//
// Rantala et al. (2023) convention: ALL operations scaled by 1/Omega for proper time
// reparameterization (no spurious precession in continuous limit).
//
// Each half-kick K(ε) splits as ABA:
//   K_aux(ε/2): w += (ε/2) * [a + g(v)] / Ω   [EXPLICIT: v,B unchanged → Ω constant]
//   K_phys(ε):  v += ε * [a + g(w)] / Ω_cubic  [IMPLICIT: cubic solve for Ω]
//               B -= ε * W_mid / Ω_cubic        [midpoint B evolution folded into cubic]
//   K_aux(ε/2): w += (ε/2) * [a + g(v')] / Ω  [EXPLICIT: v',B' known → Ω constant]
//
// K_phys cubic: the implicit midpoint for simultaneous (v,B) update gives
//   Ω³ - Ω₀·Ω² - C'·Ω - D' = 0
// where C' = C - (ε/2)*W_old, D' = D - (ε/4)*P incorporate B evolution.
//   C = ½ Σ mᵢ vᵢ·Δᵢ, D = ⅛ Σ mᵢ|Δᵢ|², Δ = ε*(a+g(w))
//   W_old = Σ mᵢ vᵢ·gᵢ(w), P = Σ mᵢ Δᵢ·gᵢ(w)
//
// The full step K(ds/2)·D(ds)·K(ds/2) has 6 PN evaluations (3 per half-kick); in a chain of steps the first
// one equals the last one of the previous step (logh_leapfrog_steps_pn), 5 per step.
// ============================================================================

// Helper: perform one K_phys step with cubic-solved Omega.
// Advances both v and B via exact implicit midpoint.
//   accel[] = Newtonian acceleration (constant during K_phys)
//   pn[]    = PN acceleration g(w) from auxiliary velocity (constant during K_phys)
//   eps     = K_phys step size in fictitious time s
//
// Uses Ω = T(v_mid) + B_mid (implicit midpoint in BOTH v and B).
// This ensures exact time-reversibility, which is required for GBS's
// even-power error expansion. The T(v_mid) terms (C, D) are essential:
// without them, the kick loses time-reversibility and GBS convergence degrades.
//
// Cubic: Ω³ - Ω₀·Ω² - C'·Ω - D' = 0
//   Ω₀ = T(v_old) + B_old
//   C' = C - (eps/2)*W_old,  D' = D - (eps/4)*P
//   C = ½ Σ mᵢ vᵢ·Δᵢ, D = ⅛ Σ mᵢ|Δᵢ|², Δ = eps*(a+g(w))
//   W_old = Σ mᵢ vᵢ·gᵢ(w), P = Σ mᵢ Δᵢ·gᵢ(w)
static void k_phys_cubic(LogHState& state, const double* accel, const double* pn,
                          double eps, double G, double Omega0) {
    int n = state.n_bodies;
    int n3 = n * 3;

    // Total impulse: Δ = eps * [a + g(w)]
    double delta[LOGH_MAX_BODIES * 3];
    for (int i = 0; i < n3; i++)
        delta[i] = eps * (accel[i] + pn[i]);

    // Cubic coefficients from T(v_mid); Omega0 = T(v_old) + B_old (the caller's)
    double C = 0.0, D = 0.0;
    for (int b = 0; b < n; b++) {
        double vdot = 0.0, dsq = 0.0;
        for (int d = 0; d < 3; d++) {
            int idx = b * 3 + d;
            vdot += state.vel[idx] * delta[idx];
            dsq += delta[idx] * delta[idx];
        }
        C += state.mass[b] * vdot;
        D += state.mass[b] * dsq;
    }
    C *= 0.5;
    D *= 0.125;

    // B evolution correction terms
    // W_old = Σ m_i v_i · g_i(w)  (PN work rate at current v)
    // P     = Σ m_i Δ_i · g_i(w)  (impulse-PN coupling)
    double W_old = 0.0, P = 0.0;
    for (int b = 0; b < n; b++) {
        for (int d = 0; d < 3; d++) {
            int idx = b * 3 + d;
            double mg = state.mass[b] * pn[idx];
            W_old += mg * state.vel[idx];
            P     += mg * delta[idx];
        }
    }

    // Modified coefficients: B_mid = B_old - (eps/2)*W_old/Ω - (eps/4)*P/Ω²
    double C_prime = C - 0.5 * eps * W_old;
    double D_prime = D - 0.25 * eps * P;

    // Solve cubic: Ω³ - Ω₀·Ω² - C'·Ω - D' = 0
    double Omega = solve_kick_omega(Omega0, C_prime, D_prime);
    double inv_Omega = 1.0 / Omega;

    // Apply velocity kick: v += Δ / Ω
    for (int i = 0; i < n3; i++)
        state.vel[i] += delta[i] * inv_Omega;

    // Apply B evolution: B -= eps * W_mid / Ω
    // W_mid = W_old + P/(2Ω) = Σ m_i v_mid · g(w)
    double W_mid = W_old + P / (2.0 * Omega);
    state.B -= eps * W_mid * inv_Omega;

    // Spins: S rotated by Omega(x, w, Z) over the physical time eps / Ω of this kick (auxiliary spins Z,
    // Rantala+ 2017; w and Z are frozen during K_phys)
    if (state.has_spins && n == 2)
        rotate_spins(state, state.aux_vel, state.aux_spin1, state.aux_spin2, G, eps * inv_Omega,
                     state.spin1, state.spin2);
}

// K_aux(ds/4): w += (ds/4) [a + g(v)] / Omega, Z rotated by Omega(x, v, S)   [EXPLICIT: v, B unchanged, so
// Omega = T(v) + B is the caller's]
static inline void k_aux(LogHState& state, const double* accel, const double* pn_v, double ds_quarter,
                         double Omega, double G) {
    const int n = state.n_bodies, n3 = n * 3;
    double inv_Omega = 1.0 / Omega;
    for (int i = 0; i < n3; i++)
        state.aux_vel[i] += ds_quarter * (accel[i] + pn_v[i]) * inv_Omega;
    if (state.has_spins && n == 2)
        rotate_spins(state, state.vel, state.spin1, state.spin2, G, ds_quarter * inv_Omega,
                     state.aux_spin1, state.aux_spin2);
}

void logh_step_start(const LogHState& state, double G, double c_code, int pn_order, LogHStepStart& start) {
    logh_internal_accel(state, G, start.accel);
    start.omega = logh_kinetic_energy(state) + state.B;
    if (pn_order > 0 && state.use_aux_vel) {
        PnGeom geom;
        pn_geom_init(state, G, c_code, pn_order, geom);
        compute_pn_all_pairs(state, geom, state.vel, G, c_code, start.pn_v, state.spin1, state.spin2);
    }
}

// nsteps PN leapfrog steps. A step needs the Newtonian acceleration a(x), g(v) = PN(x, v, S) and Omega = T(v) + B
// at its start: those at the end of the previous step (its last K_aux leaves x, v, S and B unchanged and rotates
// only the auxiliary spins Z). They are computed once per chain (or taken from start: the same for all GBS
// stages of a step) and then carried. The drift and the K_aux / K_phys after it use the Omega of the K_aux
// before the drift (K_aux and drift change neither v nor B). 5 PN evaluations and 2 kinetic energies per step
// instead of 6 and 7, with the same operations on the same values as single steps: bit-identical.
// SDAR in a PN subsystem (PLAN item 4c): slowed inner binaries without a PN member (logh_sd_update) in the auxiliary-
// velocity step. With the slowed metric <.,.>_sd of T_sd, dH_sd/dt = sum_i m_i xdot_i . g_i = <v, g>_sd, so B evolves
// as dB/ds = -<v, g>_sd / Omega, and K_phys is k_phys_cubic with every inner product taken in <.,.>_sd; the Newtonian
// acceleration carries the (1/kappa - 1) a_bin terms and the drift the slowed internal velocities, as in
// logh_leapfrog_step_sd. g(v), g(w) act on the slowed binaries as perturbations (W20 sec. 2.3).
static void k_phys_cubic_sd(LogHState& state, const double* accel, const double* pn, double eps, double Omega0) {
    const int n3 = state.n_bodies * 3;
    double delta[LOGH_MAX_BODIES * 3];
    for (int i = 0; i < n3; i++) delta[i] = eps * (accel[i] + pn[i]);
    const double C = 0.5 * sd_dot(state, state.vel, delta), D = 0.125 * sd_dot(state, delta, delta);
    const double W_old = sd_dot(state, state.vel, pn), P = sd_dot(state, delta, pn);
    const double Omega = solve_kick_omega(Omega0, C - 0.5 * eps * W_old, D - 0.25 * eps * P);
    const double inv_Omega = 1.0 / Omega;
    for (int i = 0; i < n3; i++) state.vel[i] += delta[i] * inv_Omega;
    state.B -= eps * (W_old + P / (2.0 * Omega)) * inv_Omega;
}

static void logh_leapfrog_step_pn_sd(LogHState& state, double ds, double G, double c_code, int pn_order) {
    const int n3 = state.n_bodies * 3;
    const double ds_half = 0.5 * ds, ds_quarter = 0.25 * ds;
    double accel[LOGH_MAX_BODIES * 3], pn_v[LOGH_MAX_BODIES * 3], pn[LOGH_MAX_BODIES * 3], xdot[LOGH_MAX_BODIES * 3];
    logh_internal_accel(state, G, accel);
    sd_scale_binary_accel(state, G, accel);
    PnGeom geom;
    pn_geom_init(state, G, c_code, pn_order, geom);
    for (int half = 0; half < 2; half++) {
        if (half == 1) {   // drift with the Omega of the K_aux before it, then the forces at the new positions
            const double inv_Omega = 1.0 / logh_omega(state);
            sd_drift_velocity(state, xdot);
            for (int i = 0; i < n3; i++) state.pos[i] += ds * xdot[i] * inv_Omega;
            state.t += ds * inv_Omega;
            logh_internal_accel(state, G, accel);
            sd_scale_binary_accel(state, G, accel);
            pn_geom_set_positions(state.pos, geom);
        }
        double Omega = logh_omega(state);
        compute_pn_all_pairs(state, geom, state.vel, G, c_code, pn_v);
        k_aux(state, accel, pn_v, ds_quarter, Omega, G);
        compute_pn_all_pairs(state, geom, state.aux_vel, G, c_code, pn);
        k_phys_cubic_sd(state, accel, pn, ds_half, Omega);
        Omega = logh_omega(state);
        compute_pn_all_pairs(state, geom, state.vel, G, c_code, pn_v);
        k_aux(state, accel, pn_v, ds_quarter, Omega, G);
    }
}

void logh_leapfrog_steps_pn(LogHState& state, double ds, double G, double c_code, int pn_order, int nsteps,
                            const LogHStepStart* start) {
    if (state.n_sd > 0) {
        for (int s = 0; s < nsteps; s++) logh_leapfrog_step_pn_sd(state, ds, G, c_code, pn_order);
        return;
    }
    const int n3 = state.n_bodies * 3;
    const double ds_half = 0.5 * ds;
    const double ds_quarter = 0.25 * ds;

    double accel[LOGH_MAX_BODIES * 3];   // Newtonian acceleration at x
    double pn_v[LOGH_MAX_BODIES * 3];    // g(v) = PN acceleration at (x, v, S)
    double pn[LOGH_MAX_BODIES * 3];      // g(w) = PN acceleration at (x, w, Z)
    double Omega;                        // T(v) + B
    PnGeom geom;                         // position- and mass-only factors of g at x (set again after each drift)
    pn_geom_init(state, G, c_code, pn_order, geom);
    if (start) {
        std::memcpy(accel, start->accel, n3 * sizeof(double));
        std::memcpy(pn_v, start->pn_v, n3 * sizeof(double));
        Omega = start->omega;
    } else {
        logh_internal_accel(state, G, accel);
        Omega = logh_kinetic_energy(state) + state.B;
        compute_pn_all_pairs(state, geom, state.vel, G, c_code, pn_v, state.spin1, state.spin2);
    }

    for (int s = 0; s < nsteps; s++) {
        // ============ FIRST HALF-KICK: K(ds/2) = K_aux(ds/4) · K_phys(ds/2) · K_aux(ds/4) ============
        // Rantala et al. 2023, Eq. (5): dw/ds = (1/Ω)[a + g(v)],  dv/ds = (1/Ω)[a + g(w)]
        // K_aux kicks w by [a + g(v)],  K_phys kicks v by [a + g(w)]
        // External perturbations handled via FSI kicks, not inside LogH.
        k_aux(state, accel, pn_v, ds_quarter, Omega, G);

        // K_phys(ds/2): cubic solve for implicit midpoint (v and B)
        compute_pn_all_pairs(state, geom, state.aux_vel, G, c_code, pn, state.aux_spin1, state.aux_spin2);
        k_phys_cubic(state, accel, pn, ds_half, G, Omega);

        // K_aux(ds/4) with g(v') at the updated v', B'
        Omega = logh_kinetic_energy(state) + state.B;
        compute_pn_all_pairs(state, geom, state.vel, G, c_code, pn_v, state.spin1, state.spin2);
        k_aux(state, accel, pn_v, ds_quarter, Omega, G);

        // ============ DRIFT: x += ds * v / Omega,  t += ds / Omega  [EXPLICIT] ============
        {
            double inv_Omega = 1.0 / Omega;
            for (int i = 0; i < n3; i++)
                state.pos[i] += ds * state.vel[i] * inv_Omega;
            state.t += ds * inv_Omega;
        }

        // ============ Newtonian acceleration and g(v) at the new positions ============
        logh_internal_accel(state, G, accel);
        pn_geom_set_positions(state.pos, geom);
        compute_pn_all_pairs(state, geom, state.vel, G, c_code, pn_v, state.spin1, state.spin2);

        // ============ SECOND HALF-KICK: K(ds/2) = K_aux(ds/4) · K_phys(ds/2) · K_aux(ds/4) ============
        k_aux(state, accel, pn_v, ds_quarter, Omega, G);

        compute_pn_all_pairs(state, geom, state.aux_vel, G, c_code, pn, state.aux_spin1, state.aux_spin2);
        k_phys_cubic(state, accel, pn, ds_half, G, Omega);

        Omega = logh_kinetic_energy(state) + state.B;
        compute_pn_all_pairs(state, geom, state.vel, G, c_code, pn_v, state.spin1, state.spin2);
        k_aux(state, accel, pn_v, ds_quarter, Omega, G);
    }
}

void logh_leapfrog_step_pn(LogHState& state, double ds, double G,
                            double c_code, int pn_order) {
    logh_leapfrog_steps_pn(state, ds, G, c_code, pn_order, 1);
}
