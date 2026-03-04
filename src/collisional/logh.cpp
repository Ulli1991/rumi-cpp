#include "logh.h"
#include "post_newtonian.h"
#include <cstring>

// ============================================================================
// LogH: Logarithmic Hamiltonian regularization (Mikkola & Tanikawa 1999)
// Implementation follows BIFROST (Rantala+ 2023) Section 3.3 and
// FROST (Rantala+ 2021) Section 2.3.
//
// Time-Transformed Leapfrog (TTL) with Omega = T + B
// where T = kinetic energy, B = -E₀ (positive for bound systems).
//
// BIFROST convention (Eq. 2-3): ds/dt = Omega, so dt/ds = 1/Omega.
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
// present, B evolves per BIFROST Eq. (4):
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
// BIFROST convention: all operations scaled by 1/Omega for proper
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
        Omega -= f / fp;
    }
    return Omega;
}

void logh_leapfrog_step(LogHState& state, double ds, double G) {
    int n = state.n_bodies;
    int n3 = n * 3;

    // Compute internal Newtonian accelerations at current positions
    double accel[LOGH_MAX_BODIES * 3];
    logh_internal_accel(state, G, accel);

    double ds_half = 0.5 * ds;

    // First half-kick: exact implicit midpoint via cubic solve
    {
        double Omega0 = logh_kinetic_energy(state) + state.B;
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

    // Drift
    double Omega_half = logh_kinetic_energy(state) + state.B;
    double inv_Omega_half = 1.0 / Omega_half;
    for (int i = 0; i < n3; i++)
        state.pos[i] += ds * state.vel[i] * inv_Omega_half;
    state.t += ds * inv_Omega_half;

    // Recompute accelerations
    logh_internal_accel(state, G, accel);

    // Second half-kick
    {
        double Omega0 = logh_kinetic_energy(state) + state.B;
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
}

// ============================================================================
// Helper: compute PN accelerations for all pairs using a given velocity array
// for the PN evaluation. Positions come from state.pos.
// pn_out[i*3+k] = individual PN acceleration on body i
// ============================================================================

static void compute_pn_all_pairs(const LogHState& state, const double* eval_vel,
                                  double G, double c_code, int pn_order,
                                  double* pn_out) {
    int n = state.n_bodies;
    std::memset(pn_out, 0, n * 3 * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            // EIH: compute individual PN accelerations directly
            double a_pn_i[3], a_pn_j[3];
            eih_pn_acceleration(&state.pos[i*3], &eval_vel[i*3], state.mass[i],
                                &state.pos[j*3], &eval_vel[j*3], state.mass[j],
                                G, c_code, a_pn_i, pn_order);
            eih_pn_acceleration(&state.pos[j*3], &eval_vel[j*3], state.mass[j],
                                &state.pos[i*3], &eval_vel[i*3], state.mass[i],
                                G, c_code, a_pn_j, pn_order);
            for (int k = 0; k < 3; k++) {
                pn_out[i*3+k] += a_pn_i[k];
                pn_out[j*3+k] += a_pn_j[k];
            }
        }
    }
}

// ============================================================================
// Single TTL leapfrog step with PN corrections via auxiliary velocity technique
// (Hellstrom & Mikkola 2010, BIFROST Section 3.4)
//
// BIFROST convention: ALL operations scaled by 1/Omega for proper time
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
// The full step K(ds/2)·D(ds)·K(ds/2) has 6 PN evaluations (3 per half-kick).
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
// Helper: evolve spins for a 2-body BH binary during a K_phys step
static void evolve_spins_in_kphys(LogHState& state, double dt_phys, double G) {
    if (!state.has_spins || state.n_bodies != 2 || state.c_code <= 0.0) return;

    // Relative position and velocity
    double r[3] = {state.pos[1*3+0]-state.pos[0*3+0],
                   state.pos[1*3+1]-state.pos[0*3+1],
                   state.pos[1*3+2]-state.pos[0*3+2]};
    double v[3] = {state.vel[1*3+0]-state.vel[0*3+0],
                   state.vel[1*3+1]-state.vel[0*3+1],
                   state.vel[1*3+2]-state.vel[0*3+2]};

    double m1 = state.mass[0], m2 = state.mass[1];

    // Compute precession rates
    double Omega1[3], Omega2[3];
    spin_precession(r, v, m1, m2, state.spin1, state.spin2,
                    G, state.c_code, Omega1, Omega2);

    // Evolve spins by dt_phys
    evolve_spins(state.spin1, state.spin2, Omega1, Omega2, dt_phys);
}

static void k_phys_cubic(LogHState& state, const double* accel, const double* pn,
                          double eps, double G) {
    int n = state.n_bodies;
    int n3 = n * 3;

    // Total impulse: Δ = eps * [a + g(w)]
    double delta[LOGH_MAX_BODIES * 3];
    for (int i = 0; i < n3; i++)
        delta[i] = eps * (accel[i] + pn[i]);

    // Cubic coefficients from T(v_mid)
    double Omega0 = logh_kinetic_energy(state) + state.B;
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

    // Evolve spins: physical time elapsed in this K_phys step = eps / Ω
    if (state.has_spins) {
        double dt_phys = eps * inv_Omega;
        evolve_spins_in_kphys(state, dt_phys, G);
    }
}

void logh_leapfrog_step_pn(LogHState& state, double ds, double G,
                            double c_code, int pn_order) {
    int n = state.n_bodies;
    int n3 = n * 3;
    double ds_half = 0.5 * ds;
    double ds_quarter = 0.25 * ds;

    // Compute Newtonian accelerations at current positions
    double accel[LOGH_MAX_BODIES * 3];
    logh_internal_accel(state, G, accel);

    double pn[LOGH_MAX_BODIES * 3];

    // ============ FIRST HALF-KICK: K(ds/2) = K_aux(ds/4) · K_phys(ds/2) · K_aux(ds/4) ============
    // BIFROST Eq. (5): dw/ds = (1/Ω)[a + g(v)],  dv/ds = (1/Ω)[a + g(w)]
    // K_aux kicks w by [a + g(v)],  K_phys kicks v by [a + g(w)]
    // External perturbations handled via FSI kicks, not inside LogH.

    // K_aux(ds/4): w += (ds/4) * [a + g(v)] / Omega   [EXPLICIT: v,B unchanged → Ω constant]
    {
        double Omega = logh_kinetic_energy(state) + state.B;
        double inv_Omega = 1.0 / Omega;
        compute_pn_all_pairs(state, state.vel, G, c_code, pn_order, pn);
        for (int i = 0; i < n3; i++)
            state.aux_vel[i] += ds_quarter * (accel[i] + pn[i]) * inv_Omega;
    }

    // K_phys(ds/2): cubic solve for implicit midpoint (v and B)
    compute_pn_all_pairs(state, state.aux_vel, G, c_code, pn_order, pn);
    k_phys_cubic(state, accel, pn, ds_half, G);

    // K_aux(ds/4): w += (ds/4) * [a + g(v')] / Omega   [EXPLICIT, v' and B' updated]
    {
        double Omega = logh_kinetic_energy(state) + state.B;
        double inv_Omega = 1.0 / Omega;
        compute_pn_all_pairs(state, state.vel, G, c_code, pn_order, pn);
        for (int i = 0; i < n3; i++)
            state.aux_vel[i] += ds_quarter * (accel[i] + pn[i]) * inv_Omega;
    }

    // ============ DRIFT: x += ds * v / Omega,  t += ds / Omega  [EXPLICIT] ============
    {
        double Omega = logh_kinetic_energy(state) + state.B;
        double inv_Omega = 1.0 / Omega;
        for (int i = 0; i < n3; i++)
            state.pos[i] += ds * state.vel[i] * inv_Omega;
        state.t += ds * inv_Omega;
    }

    // ============ Recompute Newtonian accelerations at new positions ============
    logh_internal_accel(state, G, accel);

    // ============ SECOND HALF-KICK: K(ds/2) = K_aux(ds/4) · K_phys(ds/2) · K_aux(ds/4) ============

    // K_aux(ds/4)   [EXPLICIT]
    {
        double Omega = logh_kinetic_energy(state) + state.B;
        double inv_Omega = 1.0 / Omega;
        compute_pn_all_pairs(state, state.vel, G, c_code, pn_order, pn);
        for (int i = 0; i < n3; i++)
            state.aux_vel[i] += ds_quarter * (accel[i] + pn[i]) * inv_Omega;
    }

    // K_phys(ds/2): cubic solve with PN
    compute_pn_all_pairs(state, state.aux_vel, G, c_code, pn_order, pn);
    k_phys_cubic(state, accel, pn, ds_half, G);

    // K_aux(ds/4)   [EXPLICIT]
    {
        double Omega = logh_kinetic_energy(state) + state.B;
        double inv_Omega = 1.0 / Omega;
        compute_pn_all_pairs(state, state.vel, G, c_code, pn_order, pn);
        for (int i = 0; i < n3; i++)
            state.aux_vel[i] += ds_quarter * (accel[i] + pn[i]) * inv_Omega;
    }
}
