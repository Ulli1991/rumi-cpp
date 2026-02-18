#include "mstar.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <limits>

// ============================================================================
// Minimum Spanning Tree via Prim's algorithm
// ============================================================================

void build_mst(const double pos[], const double mass[], int n,
               int chain_order[], int chain_pairs[], int& n_edges) {
    n_edges = n - 1;

    // Weighted distance: d_ij / sqrt(m_i * m_j) for gravitational focus
    std::vector<bool> in_tree(n, false);
    std::vector<double> min_dist(n, std::numeric_limits<double>::max());
    std::vector<int> parent(n, -1);

    // Start from particle 0
    min_dist[0] = 0.0;

    for (int step = 0; step < n; step++) {
        // Find minimum distance vertex not yet in tree
        int u = -1;
        double best = std::numeric_limits<double>::max();
        for (int i = 0; i < n; i++) {
            if (!in_tree[i] && min_dist[i] < best) {
                best = min_dist[i];
                u = i;
            }
        }
        if (u < 0) break;

        in_tree[u] = true;
        chain_order[step] = u;

        if (parent[u] >= 0) {
            int edge_idx = step - 1;
            chain_pairs[edge_idx * 2 + 0] = parent[u];
            chain_pairs[edge_idx * 2 + 1] = u;
        }

        // Update distances
        for (int v = 0; v < n; v++) {
            if (in_tree[v]) continue;
            double dx = pos[v*3+0] - pos[u*3+0];
            double dy = pos[v*3+1] - pos[u*3+1];
            double dz = pos[v*3+2] - pos[u*3+2];
            double r = std::sqrt(dx*dx + dy*dy + dz*dz);
            // Weight by inverse sqrt of mass product (closer + heavier = preferred)
            double w = r / std::sqrt(mass[u] * mass[v] + 1e-30);
            if (w < min_dist[v]) {
                min_dist[v] = w;
                parent[v] = u;
            }
        }
    }
}

// ============================================================================
// MSTAR leapfrog step (placeholder — uses absolute coordinates for now)
// Full chain coordinate implementation requires coordinate transforms
// ============================================================================

void mstar_leapfrog_step(MSTARState& state, double ds, double G) {
    int n = state.n_bodies;

    // Compute accelerations (direct N^2)
    double accel[MSTAR_MAX_BODIES * 3];
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

    // Compute T and Omega
    double T = 0.0;
    for (int i = 0; i < n; i++) {
        T += 0.5 * state.mass[i] * (state.vel[i*3+0]*state.vel[i*3+0] +
                                      state.vel[i*3+1]*state.vel[i*3+1] +
                                      state.vel[i*3+2]*state.vel[i*3+2]);
    }
    double Omega = T + state.B;

    // Half kick
    double ds_half = 0.5 * ds;
    for (int i = 0; i < n; i++) {
        state.vel[i*3+0] += ds_half * (Omega * accel[i*3+0] + state.ext_accel[i*3+0]);
        state.vel[i*3+1] += ds_half * (Omega * accel[i*3+1] + state.ext_accel[i*3+1]);
        state.vel[i*3+2] += ds_half * (Omega * accel[i*3+2] + state.ext_accel[i*3+2]);
    }

    // Update Omega
    T = 0.0;
    for (int i = 0; i < n; i++) {
        T += 0.5 * state.mass[i] * (state.vel[i*3+0]*state.vel[i*3+0] +
                                      state.vel[i*3+1]*state.vel[i*3+1] +
                                      state.vel[i*3+2]*state.vel[i*3+2]);
    }
    Omega = T + state.B;

    // Full drift
    for (int i = 0; i < n; i++) {
        state.pos[i*3+0] += ds * Omega * state.vel[i*3+0];
        state.pos[i*3+1] += ds * Omega * state.vel[i*3+1];
        state.pos[i*3+2] += ds * Omega * state.vel[i*3+2];
    }
    state.t += ds * Omega;

    // Recompute accelerations
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

    // Second half kick
    Omega = T + state.B;
    for (int i = 0; i < n; i++) {
        state.vel[i*3+0] += ds_half * (Omega * accel[i*3+0] + state.ext_accel[i*3+0]);
        state.vel[i*3+1] += ds_half * (Omega * accel[i*3+1] + state.ext_accel[i*3+1]);
        state.vel[i*3+2] += ds_half * (Omega * accel[i*3+2] + state.ext_accel[i*3+2]);
    }
}

// ============================================================================
// Dispatch: LogH for small subsystems, MSTAR for large
// ============================================================================

void regularized_integrate(Subsystem& sub, Particles& p,
                           double G, double dt, double eta_gbs,
                           int mstar_threshold) {
    if (sub.n_bodies <= mstar_threshold) {
        // Use LogH (handled by logh_gpu batch integration)
        return;
    }

    // MSTAR path for large subsystems
    // TODO: Full implementation with chain coordinates
    // For now, build MST and use direct N^2 in absolute coordinates
    MSTARState state{};
    state.n_bodies = sub.n_bodies;
    state.t = 0.0;

    // Pack particle data
    auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
    auto h_y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
    auto h_z = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
    auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
    auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
    auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
    auto h_m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
    auto h_pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);

    int max_pid = 0;
    for (int i = 0; i < p.N; i++) max_pid = std::max(max_pid, h_pid(i));
    std::vector<int> pid_to_idx(max_pid + 1, -1);
    for (int i = 0; i < p.N; i++) pid_to_idx[h_pid(i)] = i;

    for (int b = 0; b < sub.n_bodies && b < MSTAR_MAX_BODIES; b++) {
        int idx = pid_to_idx[sub.particle_ids[b]];
        state.pos[b*3+0] = h_x(idx);
        state.pos[b*3+1] = h_y(idx);
        state.pos[b*3+2] = h_z(idx);
        state.vel[b*3+0] = h_vx(idx);
        state.vel[b*3+1] = h_vy(idx);
        state.vel[b*3+2] = h_vz(idx);
        state.mass[b] = h_m(idx);
        state.ext_accel[b*3+0] = 0.0;
        state.ext_accel[b*3+1] = 0.0;
        state.ext_accel[b*3+2] = 0.0;
    }

    // Build MST
    build_mst(state.pos, state.mass, state.n_bodies,
              state.chain_order, state.chain_pairs, state.n_edges);

    // Compute binding energy
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
    state.B = -W;

    // Integrate with GBS
    double T = 0.0;
    for (int i = 0; i < state.n_bodies; i++) {
        T += 0.5 * state.mass[i] * (state.vel[i*3+0]*state.vel[i*3+0] +
                                      state.vel[i*3+1]*state.vel[i*3+1] +
                                      state.vel[i*3+2]*state.vel[i*3+2]);
    }
    double Omega = T + state.B;
    double ds = (Omega > 1e-30) ? dt / Omega : dt;

    // Simple integration with fixed step (GBS for MSTAR needs separate impl)
    int n_steps = std::max(1, (int)(dt / ds));
    ds = dt / n_steps;  // TODO: use GBS extrapolation
    for (int s = 0; s < n_steps; s++) {
        mstar_leapfrog_step(state, ds, G);
    }

    // Unpack results
    for (int b = 0; b < sub.n_bodies && b < MSTAR_MAX_BODIES; b++) {
        int idx = pid_to_idx[sub.particle_ids[b]];
        h_x(idx) = state.pos[b*3+0];
        h_y(idx) = state.pos[b*3+1];
        h_z(idx) = state.pos[b*3+2];
        h_vx(idx) = state.vel[b*3+0];
        h_vy(idx) = state.vel[b*3+1];
        h_vz(idx) = state.vel[b*3+2];
    }

    Kokkos::deep_copy(p.x, h_x);
    Kokkos::deep_copy(p.y, h_y);
    Kokkos::deep_copy(p.z, h_z);
    Kokkos::deep_copy(p.vx, h_vx);
    Kokkos::deep_copy(p.vy, h_vy);
    Kokkos::deep_copy(p.vz, h_vz);
}
