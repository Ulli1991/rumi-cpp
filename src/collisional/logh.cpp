#include "logh.h"
#include "post_newtonian.h"
#include <cstring>

// ============================================================================
// LogH: Logarithmic Hamiltonian regularization (Mikkola & Tanikawa 1999)
//
// Time-Transformed Leapfrog (TTL) with Omega = T + B
// where T = kinetic energy, B = -E₀ (positive for bound systems)
//
// Leapfrog in fictitious time s (KDK):
//   1. Half-kick:  v += (ds/2) * a          (plain, no Omega)
//   2. Evaluate:   Omega = T(v) + B
//   3. Full drift: x += ds * Omega * v
//   4. Time:       t += ds * Omega
//   5. Forces:     a = F(x)
//   6. Half-kick:  v += (ds/2) * a          (plain, no Omega)
//
// B is constant (external perturbation handled by FSI kicks, not inside LogH).
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
// Kick-Drift-Kick with Omega = T + B evaluated at half-step velocity
// ============================================================================

void logh_leapfrog_step(LogHState& state, double ds, double G) {
    int n = state.n_bodies;

    // Compute accelerations at current positions
    double accel[LOGH_MAX_BODIES * 3];
    logh_internal_accel(state, G, accel);

    // Half-kick: v += (ds/2) * a  (plain, no Omega scaling)
    double ds_half = 0.5 * ds;
    for (int i = 0; i < n * 3; i++) {
        state.vel[i] += ds_half * accel[i];
    }

    // Compute Omega = T + B at half-stepped velocities
    double T = logh_kinetic_energy(state);
    double Omega = T + state.B;

    // Full drift: x += ds * Omega * v,  t += ds * Omega
    for (int i = 0; i < n * 3; i++) {
        state.pos[i] += ds * Omega * state.vel[i];
    }
    state.t += ds * Omega;

    // Recompute accelerations at new positions
    logh_internal_accel(state, G, accel);

    // Second half-kick: v += (ds/2) * a  (plain)
    for (int i = 0; i < n * 3; i++) {
        state.vel[i] += ds_half * accel[i];
    }
}

// ============================================================================
// Single TTL leapfrog step with PN corrections via auxiliary velocity technique
// (Hellstrom & Mikkola 2010)
//
// dv/ds = a_newton + pn(r, w)   -- physical vel gets PN from auxiliary vel
// dw/ds = a_newton + pn(r, v)   -- auxiliary vel gets PN from physical vel
// ============================================================================

void logh_leapfrog_step_pn(LogHState& state, double ds, double G,
                            double c_code, int pn_order) {
    int n = state.n_bodies;

    // Compute Newtonian accelerations at current positions
    double accel[LOGH_MAX_BODIES * 3];
    logh_internal_accel(state, G, accel);

    // Compute PN accelerations for each pair
    // pn_from_w[i] = PN contribution to vel[i] (evaluated at aux_vel)
    // pn_from_v[i] = PN contribution to aux_vel[i] (evaluated at vel)
    double pn_from_w[LOGH_MAX_BODIES * 3] = {};
    double pn_from_v[LOGH_MAX_BODIES * 3] = {};

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double r[3] = { state.pos[j*3] - state.pos[i*3],
                           state.pos[j*3+1] - state.pos[i*3+1],
                           state.pos[j*3+2] - state.pos[i*3+2] };
            double M_ij = state.mass[i] + state.mass[j];
            double fi = state.mass[j] / M_ij;  // fraction for body i
            double fj = state.mass[i] / M_ij;  // fraction for body j

            // PN from auxiliary velocity w -> kicks physical velocity v
            double vw[3] = { state.aux_vel[j*3] - state.aux_vel[i*3],
                            state.aux_vel[j*3+1] - state.aux_vel[i*3+1],
                            state.aux_vel[j*3+2] - state.aux_vel[i*3+2] };
            double a_rel_w[3];
            pn_acceleration(r, vw, state.mass[i], state.mass[j], G, c_code, a_rel_w, pn_order);
            // Individual: a_i = -(m_j/M)*a_rel, a_j = +(m_i/M)*a_rel
            for (int k = 0; k < 3; k++) {
                pn_from_w[i*3+k] -= fi * a_rel_w[k];
                pn_from_w[j*3+k] += fj * a_rel_w[k];
            }

            // PN from physical velocity v -> kicks auxiliary velocity w
            double vv[3] = { state.vel[j*3] - state.vel[i*3],
                            state.vel[j*3+1] - state.vel[i*3+1],
                            state.vel[j*3+2] - state.vel[i*3+2] };
            double a_rel_v[3];
            pn_acceleration(r, vv, state.mass[i], state.mass[j], G, c_code, a_rel_v, pn_order);
            for (int k = 0; k < 3; k++) {
                pn_from_v[i*3+k] -= fi * a_rel_v[k];
                pn_from_v[j*3+k] += fj * a_rel_v[k];
            }
        }
    }

    // Half-kick: v += (ds/2) * (a + pn_from_w), w += (ds/2) * (a + pn_from_v)
    double ds_half = 0.5 * ds;
    for (int i = 0; i < n * 3; i++) {
        state.vel[i]     += ds_half * (accel[i] + pn_from_w[i]);
        state.aux_vel[i] += ds_half * (accel[i] + pn_from_v[i]);
    }

    // Compute Omega = T(vel) + B at half-stepped velocities
    double T = logh_kinetic_energy(state);
    double Omega = T + state.B;

    // Full drift: x += ds * Omega * v,  t += ds * Omega
    for (int i = 0; i < n * 3; i++) {
        state.pos[i] += ds * Omega * state.vel[i];
    }
    state.t += ds * Omega;

    // Recompute Newtonian accelerations at new positions
    logh_internal_accel(state, G, accel);

    // Recompute PN at new positions
    std::memset(pn_from_w, 0, n * 3 * sizeof(double));
    std::memset(pn_from_v, 0, n * 3 * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double r[3] = { state.pos[j*3] - state.pos[i*3],
                           state.pos[j*3+1] - state.pos[i*3+1],
                           state.pos[j*3+2] - state.pos[i*3+2] };
            double M_ij = state.mass[i] + state.mass[j];
            double fi = state.mass[j] / M_ij;
            double fj = state.mass[i] / M_ij;

            double vw[3] = { state.aux_vel[j*3] - state.aux_vel[i*3],
                            state.aux_vel[j*3+1] - state.aux_vel[i*3+1],
                            state.aux_vel[j*3+2] - state.aux_vel[i*3+2] };
            double a_rel_w[3];
            pn_acceleration(r, vw, state.mass[i], state.mass[j], G, c_code, a_rel_w, pn_order);
            for (int k = 0; k < 3; k++) {
                pn_from_w[i*3+k] -= fi * a_rel_w[k];
                pn_from_w[j*3+k] += fj * a_rel_w[k];
            }

            double vv[3] = { state.vel[j*3] - state.vel[i*3],
                            state.vel[j*3+1] - state.vel[i*3+1],
                            state.vel[j*3+2] - state.vel[i*3+2] };
            double a_rel_v[3];
            pn_acceleration(r, vv, state.mass[i], state.mass[j], G, c_code, a_rel_v, pn_order);
            for (int k = 0; k < 3; k++) {
                pn_from_v[i*3+k] -= fi * a_rel_v[k];
                pn_from_v[j*3+k] += fj * a_rel_v[k];
            }
        }
    }

    // Second half-kick
    for (int i = 0; i < n * 3; i++) {
        state.vel[i]     += ds_half * (accel[i] + pn_from_w[i]);
        state.aux_vel[i] += ds_half * (accel[i] + pn_from_v[i]);
    }
}
