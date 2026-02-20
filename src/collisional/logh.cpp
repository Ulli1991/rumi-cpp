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
//   0. Estimate:   Omega = T(v + ds/2 * a) + B    (plain half-kick for Omega)
//   1. Half-kick:  v += (ds/2) * a / Omega
//   2. Full drift: x += ds * v / Omega
//   3. Time:       t += ds / Omega
//   4. Forces:     a = F(x)
//   5. Half-kick:  v += (ds/2) * a / Omega         (same Omega as step 0)
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
// time reparameterization (preserves LRL vector in continuous limit)
// ============================================================================

void logh_leapfrog_step(LogHState& state, double ds, double G) {
    int n = state.n_bodies;

    // Compute accelerations at current positions
    double accel[LOGH_MAX_BODIES * 3];
    logh_internal_accel(state, G, accel);

    double ds_half = 0.5 * ds;

    // Estimate half-step velocity with plain half-kick for Omega evaluation
    double v_save[LOGH_MAX_BODIES * 3];
    std::memcpy(v_save, state.vel, n * 3 * sizeof(double));
    for (int i = 0; i < n * 3; i++) {
        state.vel[i] += ds_half * accel[i];
    }
    double T = logh_kinetic_energy(state);
    double Omega = T + state.B;
    double inv_Omega = 1.0 / Omega;

    // Restore velocity and do actual half-kick: v += (ds/2) * a / Omega
    std::memcpy(state.vel, v_save, n * 3 * sizeof(double));
    for (int i = 0; i < n * 3; i++) {
        state.vel[i] += ds_half * accel[i] * inv_Omega;
    }

    // Full drift: x += ds * v / Omega,  t += ds / Omega
    for (int i = 0; i < n * 3; i++) {
        state.pos[i] += ds * state.vel[i] * inv_Omega;
    }
    state.t += ds * inv_Omega;

    // Recompute accelerations at new positions
    logh_internal_accel(state, G, accel);

    // Second half-kick with same Omega: v += (ds/2) * a / Omega
    for (int i = 0; i < n * 3; i++) {
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
            double r[3] = { state.pos[j*3] - state.pos[i*3],
                           state.pos[j*3+1] - state.pos[i*3+1],
                           state.pos[j*3+2] - state.pos[i*3+2] };
            double M_ij = state.mass[i] + state.mass[j];
            double fi = state.mass[j] / M_ij;
            double fj = state.mass[i] / M_ij;
            double vrel[3] = { eval_vel[j*3] - eval_vel[i*3],
                              eval_vel[j*3+1] - eval_vel[i*3+1],
                              eval_vel[j*3+2] - eval_vel[i*3+2] };
            double a_rel[3];
            pn_acceleration(r, vrel, state.mass[i], state.mass[j], G, c_code, a_rel, pn_order);
            for (int k = 0; k < 3; k++) {
                pn_out[i*3+k] -= fi * a_rel[k];
                pn_out[j*3+k] += fj * a_rel[k];
            }
        }
    }
}

// ============================================================================
// Single TTL leapfrog step with PN corrections via auxiliary velocity technique
// (Hellstrom & Mikkola 2010, BIFROST Section 3.4)
//
// BIFROST convention: ALL kick and B evolution operators scaled by 1/Omega
// for proper time reparameterization. Omega estimated from plain Newtonian
// half-kick at the start of each step and held constant throughout.
//
// Each half-kick K(ε) splits as ABA:
//   K_aux(ε/2): w += (ε/2) * [a + g(v)] / Omega
//   K_phys(ε):  B -= (ε/2) * Σ m_i v · g(w) / Omega   (pre-kick B)
//               v += ε * [a + g(w)] / Omega             (velocity kick)
//               B -= (ε/2) * Σ m_i v'· g(w) / Omega    (post-kick B)
//   K_aux(ε/2): w += (ε/2) * [a + g(v')] / Omega
//
// B evolution (BIFROST Eq. 4): dB/ds = -Σ m_i v_i · g_i / Omega
// Discretized with SYMMETRIC (trapezoidal) rule for time-reversibility.
//
// The full step K(ds/2)·D(ds)·K(ds/2) has 6 PN evaluations (3 per half-kick).
// ============================================================================

void logh_leapfrog_step_pn(LogHState& state, double ds, double G,
                            double c_code, int pn_order) {
    int n = state.n_bodies;
    double ds_half = 0.5 * ds;
    double ds_quarter = 0.25 * ds;

    // Compute Newtonian accelerations at current positions
    double accel[LOGH_MAX_BODIES * 3];
    logh_internal_accel(state, G, accel);

    // Estimate Omega from plain Newtonian half-kick
    double v_save[LOGH_MAX_BODIES * 3];
    std::memcpy(v_save, state.vel, n * 3 * sizeof(double));
    for (int i = 0; i < n * 3; i++) {
        state.vel[i] += ds_half * accel[i];
    }
    double T_est = logh_kinetic_energy(state);
    double Omega = T_est + state.B;
    double inv_Omega = 1.0 / Omega;
    std::memcpy(state.vel, v_save, n * 3 * sizeof(double));

    double pn[LOGH_MAX_BODIES * 3];

    // ============ FIRST HALF-KICK: K(ds/2) = K_aux(ds/4) · K_phys(ds/2) · K_aux(ds/4) ============

    // K_aux(ds/4): w += (ds/4) * [a + g(v)] / Omega
    compute_pn_all_pairs(state, state.vel, G, c_code, pn_order, pn);
    for (int i = 0; i < n * 3; i++) {
        state.aux_vel[i] += ds_quarter * (accel[i] + pn[i]) * inv_Omega;
    }

    // K_phys(ds/2): symmetric B update + v kick using PN from UPDATED w
    compute_pn_all_pairs(state, state.aux_vel, G, c_code, pn_order, pn);
    // Pre-kick B: ΔB₁ = -(ds/4) * Σ m_i v₀ · g(w) / Omega
    double work_pn = 0.0;
    for (int i = 0; i < n; i++) {
        work_pn += state.mass[i] * (state.vel[i*3+0] * pn[i*3+0]
                                   + state.vel[i*3+1] * pn[i*3+1]
                                   + state.vel[i*3+2] * pn[i*3+2]);
    }
    state.B -= ds_quarter * work_pn * inv_Omega;
    // Kick: v += (ds/2) * (a + g(w)) / Omega
    for (int i = 0; i < n * 3; i++) {
        state.vel[i] += ds_half * (accel[i] + pn[i]) * inv_Omega;
    }
    // Post-kick B: ΔB₂ = -(ds/4) * Σ m_i v₁ · g(w) / Omega  [using updated v]
    work_pn = 0.0;
    for (int i = 0; i < n; i++) {
        work_pn += state.mass[i] * (state.vel[i*3+0] * pn[i*3+0]
                                   + state.vel[i*3+1] * pn[i*3+1]
                                   + state.vel[i*3+2] * pn[i*3+2]);
    }
    state.B -= ds_quarter * work_pn * inv_Omega;

    // K_aux(ds/4): w += (ds/4) * [a + g(v')] / Omega  (v' = updated physical velocity)
    compute_pn_all_pairs(state, state.vel, G, c_code, pn_order, pn);
    for (int i = 0; i < n * 3; i++) {
        state.aux_vel[i] += ds_quarter * (accel[i] + pn[i]) * inv_Omega;
    }

    // ============ DRIFT: x += ds * v / Omega,  t += ds / Omega ============
    for (int i = 0; i < n * 3; i++) {
        state.pos[i] += ds * state.vel[i] * inv_Omega;
    }
    state.t += ds * inv_Omega;

    // ============ Recompute Newtonian accelerations at new positions ============
    logh_internal_accel(state, G, accel);

    // ============ SECOND HALF-KICK: K(ds/2) = K_aux(ds/4) · K_phys(ds/2) · K_aux(ds/4) ============

    // K_aux(ds/4): w += (ds/4) * [a' + g'(v)] / Omega
    compute_pn_all_pairs(state, state.vel, G, c_code, pn_order, pn);
    for (int i = 0; i < n * 3; i++) {
        state.aux_vel[i] += ds_quarter * (accel[i] + pn[i]) * inv_Omega;
    }

    // K_phys(ds/2): symmetric B update + v kick using PN from UPDATED w
    compute_pn_all_pairs(state, state.aux_vel, G, c_code, pn_order, pn);
    // Pre-kick B: ΔB₁ = -(ds/4) * Σ m_i v₀ · g(w) / Omega
    work_pn = 0.0;
    for (int i = 0; i < n; i++) {
        work_pn += state.mass[i] * (state.vel[i*3+0] * pn[i*3+0]
                                   + state.vel[i*3+1] * pn[i*3+1]
                                   + state.vel[i*3+2] * pn[i*3+2]);
    }
    state.B -= ds_quarter * work_pn * inv_Omega;
    // Kick: v += (ds/2) * (a + g(w)) / Omega
    for (int i = 0; i < n * 3; i++) {
        state.vel[i] += ds_half * (accel[i] + pn[i]) * inv_Omega;
    }
    // Post-kick B: ΔB₂ = -(ds/4) * Σ m_i v₁ · g(w) / Omega  [using updated v]
    work_pn = 0.0;
    for (int i = 0; i < n; i++) {
        work_pn += state.mass[i] * (state.vel[i*3+0] * pn[i*3+0]
                                   + state.vel[i*3+1] * pn[i*3+1]
                                   + state.vel[i*3+2] * pn[i*3+2]);
    }
    state.B -= ds_quarter * work_pn * inv_Omega;

    // K_aux(ds/4): w += (ds/4) * [a' + g'(v')] / Omega
    compute_pn_all_pairs(state, state.vel, G, c_code, pn_order, pn);
    for (int i = 0; i < n * 3; i++) {
        state.aux_vel[i] += ds_quarter * (accel[i] + pn[i]) * inv_Omega;
    }
}
