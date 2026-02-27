#include "subsystem.h"
#include "collisional_config.h"
#include "constants.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>

// Global subsystem manager instance
SubsystemManager g_subsystem_mgr;

// ============================================================================
// GPU buffer allocation
// ============================================================================

void SubsystemManager::allocate_ngb_buffers(int max_pairs) {
    if (max_pairs <= max_ngb_pairs) return;
    max_ngb_pairs = max_pairs;
    ngb_pair_i     = Kokkos::View<int*>("ngb_i", max_pairs);
    ngb_pair_j     = Kokkos::View<int*>("ngb_j", max_pairs);
    ngb_pair_a     = Kokkos::View<double*>("ngb_a", max_pairs);
    ngb_pair_e     = Kokkos::View<double*>("ngb_e", max_pairs);
    ngb_pair_E     = Kokkos::View<double*>("ngb_E", max_pairs);
    ngb_pair_bound = Kokkos::View<int*>("ngb_bound", max_pairs);
    ngb_count      = Kokkos::View<int*>("ngb_count", 1);
}

void SubsystemManager::reset_ngb_count() {
    if (ngb_count.extent(0) > 0) {
        Kokkos::deep_copy(ngb_count, 0);
    }
}

// ============================================================================
// Union-Find for connected components
// ============================================================================

struct UnionFind {
    std::vector<int> parent, rank_uf;

    UnionFind(int n) : parent(n), rank_uf(n, 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    void unite(int x, int y) {
        x = find(x); y = find(y);
        if (x == y) return;
        if (rank_uf[x] < rank_uf[y]) std::swap(x, y);
        parent[y] = x;
        if (rank_uf[x] == rank_uf[y]) rank_uf[x]++;
    }
};

// ============================================================================
// Mardling (2001) stability criterion
// ============================================================================

bool SubsystemManager::mardling_stability(double a_in, double e_in,
                                           double a_out, double e_out,
                                           double m1, double m2, double m3) {
    // Mardling & Aarseth (2001) stability criterion:
    // a_out/a_in > 2.8 * ((1+q_out)*(1+e_out)/sqrt(1-e_out))^{2/5} * (1 - 0.3*inc/pi)
    // Using the simplified version without inclination:
    double q_out = m3 / (m1 + m2);
    double ratio_crit = 2.8 * std::pow((1.0 + q_out) * (1.0 + e_out) / std::sqrt(1.0 - e_out), 0.4);
    double ratio_actual = a_out * (1.0 - e_out) / (a_in * (1.0 + e_in));

    return ratio_actual > ratio_crit;
}

// ============================================================================
// Detect subsystems from neighbor pairs
// ============================================================================

void SubsystemManager::detect_from_pairs(Particles& p, double G, double dt_step,
                                          double r_ngb, double r_ngb_max,
                                          int n_orb_secular) {
    // BIFROST Section 3.5: fresh detection at each HHS level.
    // Save existing subsystems for persistent state transfer, then start fresh.
    std::vector<Subsystem> old_active = std::move(active);
    active.clear();

    // Copy neighbor count from device
    auto h_count = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ngb_count);
    int n_pairs = h_count(0);
    if (n_pairs == 0) {
        // No neighbors at this level — no subsystems
        last_level_pairs = 0;
        last_level_bound = 0;
        return;
    }
    n_pairs = std::min(n_pairs, max_ngb_pairs);
    last_level_pairs = n_pairs;

    // Copy neighbor pairs to host
    auto h_pi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                    Kokkos::subview(ngb_pair_i, std::make_pair(0, n_pairs)));
    auto h_pj = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                    Kokkos::subview(ngb_pair_j, std::make_pair(0, n_pairs)));
    auto h_pa = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                    Kokkos::subview(ngb_pair_a, std::make_pair(0, n_pairs)));
    auto h_pe = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                    Kokkos::subview(ngb_pair_e, std::make_pair(0, n_pairs)));
    auto h_pE = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                    Kokkos::subview(ngb_pair_E, std::make_pair(0, n_pairs)));
    auto h_pb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                    Kokkos::subview(ngb_pair_bound, std::make_pair(0, n_pairs)));

    // Count bound pairs for diagnostics
    {
        int nb = 0;
        for (int k = 0; k < n_pairs; k++) { if (h_pb(k)) nb++; }
        last_level_bound = nb;
    }

    // Get particle IDs and adaptive timesteps on host
    auto h_pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);
    auto h_dt  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.dt_adapt);

    // Build mapping from particle_id to array index
    int max_pid = 0;
    for (int i = 0; i < p.N; i++) max_pid = std::max(max_pid, h_pid(i));
    std::vector<int> pid_to_idx(max_pid + 1, -1);
    for (int i = 0; i < p.N; i++) pid_to_idx[h_pid(i)] = i;

    // Sort pairs by (min_pid, max_pid) for determinism across MPI ranks
    struct PairInfo {
        int pid_lo, pid_hi;
        double a, e, E_orb;
        bool bound;
    };
    std::vector<PairInfo> pairs(n_pairs);
    for (int k = 0; k < n_pairs; k++) {
        int pi = h_pi(k), pj = h_pj(k);
        // Convert array indices to particle IDs
        int pid_i = h_pid(pi);
        int pid_j = h_pid(pj);
        pairs[k].pid_lo = std::min(pid_i, pid_j);
        pairs[k].pid_hi = std::max(pid_i, pid_j);
        pairs[k].a = h_pa(k);
        pairs[k].e = h_pe(k);
        pairs[k].E_orb = h_pE(k);
        pairs[k].bound = (h_pb(k) != 0);
    }
    std::sort(pairs.begin(), pairs.end(), [](const PairInfo& a, const PairInfo& b) {
        if (a.pid_lo != b.pid_lo) return a.pid_lo < b.pid_lo;
        return a.pid_hi < b.pid_hi;
    });

    // Remove duplicates (same pair may be found by different threads)
    pairs.erase(std::unique(pairs.begin(), pairs.end(),
        [](const PairInfo& a, const PairInfo& b) {
            return a.pid_lo == b.pid_lo && a.pid_hi == b.pid_hi;
        }), pairs.end());

    // Union-find to identify connected components
    // Merge ALL pairs (bound and unbound fly-bys both form subsystems)
    UnionFind uf(max_pid + 1);
    for (auto& pr : pairs) {
        uf.unite(pr.pid_lo, pr.pid_hi);
    }

    // Build components from ALL pairs (bound and unbound)
    std::map<int, std::vector<int>> components;
    for (auto& pr : pairs) {
        int root = uf.find(pr.pid_lo);
        if (components.find(root) == components.end()) {
            // Collect all members
            std::set<int> members;
            for (auto& pr2 : pairs) {
                if (uf.find(pr2.pid_lo) == root) {
                    members.insert(pr2.pid_lo);
                    members.insert(pr2.pid_hi);
                }
            }
            components[root] = std::vector<int>(members.begin(), members.end());
        }
    }

    // Build lookup from sorted pid set → old subsystem for persistent state transfer
    // (B_phys, aux_vel, normalization scales carried over for identical member sets)
    auto make_key = [](const std::vector<int>& pids) -> std::vector<int> {
        std::vector<int> k = pids;
        std::sort(k.begin(), k.end());
        return k;
    };
    std::map<std::vector<int>, Subsystem*> old_by_pids;
    for (auto& old_sub : old_active) {
        old_by_pids[make_key(old_sub.particle_ids)] = &old_sub;
    }

    auto h_mass = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
    // Position/velocity mirrors created lazily (only for triples, avoids N-sized D2H for binaries)
    bool have_pos = false;
    auto h_x  = Kokkos::create_mirror_view(Kokkos::HostSpace(), p.x);
    auto h_y  = Kokkos::create_mirror_view(Kokkos::HostSpace(), p.y);
    auto h_z  = Kokkos::create_mirror_view(Kokkos::HostSpace(), p.z);
    auto h_vx = Kokkos::create_mirror_view(Kokkos::HostSpace(), p.vx);
    auto h_vy = Kokkos::create_mirror_view(Kokkos::HostSpace(), p.vy);
    auto h_vz = Kokkos::create_mirror_view(Kokkos::HostSpace(), p.vz);

    for (auto& [root, members] : components) {
        // Check if this component exactly matches an existing subsystem
        auto key = make_key(members);
        auto it = old_by_pids.find(key);

        Subsystem sub{};
        if (it != old_by_pids.end()) {
            // Same members — carry over persistent LogH state
            sub.id = it->second->id;
            sub.B_phys = it->second->B_phys;
            sub.has_aux_vel = it->second->has_aux_vel;
            std::memcpy(sub.aux_vel_phys, it->second->aux_vel_phys, sizeof(sub.aux_vel_phys));
            // Carry over fixed normalization scales
            sub.V_scale = it->second->V_scale;
            sub.R_scale = it->second->R_scale;
            sub.T_scale = it->second->T_scale;
            sub.E_scale = it->second->E_scale;
            sub.scales_set = it->second->scales_set;
        } else {
            sub.id = next_id++;
        }
        sub.particle_ids = members;
        sub.n_bodies = (int)members.size();
        sub.is_secular = false;
        sub.inner_binary_id = -1;
        sub.outer_companion_pid = -1;

        if (sub.n_bodies == 2) {
            // Binary or fly-by
            int idx_i = pid_to_idx[members[0]];
            int idx_j = pid_to_idx[members[1]];

            // Find the pair info
            bool pair_bound = false;
            for (auto& pr : pairs) {
                if ((pr.pid_lo == members[0] && pr.pid_hi == members[1]) ||
                    (pr.pid_lo == members[1] && pr.pid_hi == members[0])) {
                    sub.orb.a = pr.a;
                    sub.orb.e = pr.e;
                    sub.orb.E_orb = pr.E_orb;
                    pair_bound = pr.bound;
                    break;
                }
            }

            // Compute period (only meaningful for bound orbits with a > 0)
            if (sub.orb.a > 0 && idx_i >= 0 && idx_j >= 0) {
                double mu = G * (h_mass(idx_i) + h_mass(idx_j));
                sub.orb.period = 2.0 * M_PI * std::sqrt(sub.orb.a * sub.orb.a * sub.orb.a / mu);
                sub.n_orb_per_dt = (int)(dt_step / sub.orb.period);
                sub.is_secular = (sub.n_orb_per_dt >= n_orb_secular);
            }

            sub.type = pair_bound ? SubsystemType::BINARY : SubsystemType::FLYBY;
            // Record min(dt_adapt) at detection — before exclusion strips internal pairs
            {
                double dt_detect = 1e30;
                for (int pid : sub.particle_ids) {
                    int idx = pid_to_idx[pid];
                    if (idx >= 0) dt_detect = std::min(dt_detect, h_dt(idx));
                }
                sub.dt_sub = dt_detect;
            }
        } else if (sub.n_bodies == 3) {
            // Triple — check stability
            // Find the tightest pair (inner binary)
            double a_min = 1e30;
            int inner_i = -1, inner_j = -1;
            for (auto& pr : pairs) {
                if (pr.bound && pr.a > 0 && pr.a < a_min) {
                    int root_lo = uf.find(pr.pid_lo);
                    if (root_lo == root) {
                        a_min = pr.a;
                        inner_i = pr.pid_lo;
                        inner_j = pr.pid_hi;
                    }
                }
            }

            // Outer companion is the remaining member
            int outer_pid = -1;
            for (int pid : members) {
                if (pid != inner_i && pid != inner_j) {
                    outer_pid = pid;
                    break;
                }
            }

            if (inner_i >= 0 && inner_j >= 0 && outer_pid >= 0) {
                int idx_i = pid_to_idx[inner_i];
                int idx_j = pid_to_idx[inner_j];
                int idx_o = pid_to_idx[outer_pid];

                if (idx_i >= 0 && idx_j >= 0 && idx_o >= 0) {
                    double m1 = h_mass(idx_i), m2 = h_mass(idx_j), m3 = h_mass(idx_o);

                    // Find inner and outer orbital elements
                    double a_in = a_min, e_in = 0.0;
                    for (auto& pr : pairs) {
                        if ((pr.pid_lo == inner_i && pr.pid_hi == inner_j) ||
                            (pr.pid_lo == inner_j && pr.pid_hi == inner_i)) {
                            e_in = pr.e;
                            break;
                        }
                    }

                    // Compute outer orbit relative to inner binary COM
                    // (pairwise elements use wrong mu and wrong relative vector)
                    if (!have_pos) {
                        Kokkos::deep_copy(h_x, p.x);   Kokkos::deep_copy(h_y, p.y);
                        Kokkos::deep_copy(h_z, p.z);   Kokkos::deep_copy(h_vx, p.vx);
                        Kokkos::deep_copy(h_vy, p.vy);  Kokkos::deep_copy(h_vz, p.vz);
                        have_pos = true;
                    }
                    double M_in = m1 + m2;
                    double mu_out = G * (M_in + m3);
                    double cx  = (m1*h_x(idx_i)  + m2*h_x(idx_j))  / M_in;
                    double cy  = (m1*h_y(idx_i)  + m2*h_y(idx_j))  / M_in;
                    double cz  = (m1*h_z(idx_i)  + m2*h_z(idx_j))  / M_in;
                    double cvx = (m1*h_vx(idx_i) + m2*h_vx(idx_j)) / M_in;
                    double cvy = (m1*h_vy(idx_i) + m2*h_vy(idx_j)) / M_in;
                    double cvz = (m1*h_vz(idx_i) + m2*h_vz(idx_j)) / M_in;
                    double r_out[3] = { h_x(idx_o)-cx,  h_y(idx_o)-cy,  h_z(idx_o)-cz };
                    double v_out[3] = { h_vx(idx_o)-cvx, h_vy(idx_o)-cvy, h_vz(idx_o)-cvz };
                    auto oe_out = cartesian_to_elements(r_out, v_out, mu_out);
                    double a_out = oe_out.a;
                    double e_out = oe_out.e;

                    // Always store inner binary elements for triples
                    // (needed for timestep clamping in enforce_shared_timestep)
                    if (a_in > 0) {
                        sub.orb.a = a_in;
                        sub.orb.e = e_in;
                    }

                    if (a_out > 0) {
                        bool stable = mardling_stability(a_in, e_in, a_out, e_out, m1, m2, m3);
                        sub.type = stable ? SubsystemType::TRIPLE_STABLE : SubsystemType::TRIPLE_UNSTABLE;
                        sub.inner_binary_id = inner_i;  // store inner binary info
                        sub.outer_companion_pid = outer_pid;

                        // Secular flag for stable triples: based on inner binary period
                        if (stable && a_in > 0) {
                            double mu_in = G * (m1 + m2);
                            double P_in = 2.0 * M_PI * std::sqrt(a_in * a_in * a_in / mu_in);
                            sub.n_orb_per_dt = (int)(dt_step / P_in);
                            sub.is_secular = (sub.n_orb_per_dt >= n_orb_secular);
                        }
                    } else {
                        sub.type = SubsystemType::TRIPLE_UNSTABLE;
                    }
                }
            } else {
                sub.type = SubsystemType::TRIPLE_UNSTABLE;
            }
            // Record min(dt_adapt) at detection — before exclusion strips internal pairs
            {
                double dt_detect = 1e30;
                for (int pid : sub.particle_ids) {
                    int idx = pid_to_idx[pid];
                    if (idx >= 0) dt_detect = std::min(dt_detect, h_dt(idx));
                }
                sub.dt_sub = dt_detect;
            }
        } else {
            sub.type = SubsystemType::MULTIPLET;
            // Record min(dt_adapt) at detection — before exclusion strips internal pairs
            {
                double dt_detect = 1e30;
                for (int pid : sub.particle_ids) {
                    int idx = pid_to_idx[pid];
                    if (idx >= 0) dt_detect = std::min(dt_detect, h_dt(idx));
                }
                sub.dt_sub = dt_detect;
            }
        }

        active.push_back(sub);
    }
}

// ============================================================================
// Update subsystem_id in Particles
// ============================================================================

void SubsystemManager::update_particle_flags(Particles& p) const {
    if (active.empty()) return;

    // Reset all to -1
    auto sid = p.subsystem_id;
    int N = p.N;
    Kokkos::parallel_for("reset_sid", N, KOKKOS_LAMBDA(int i) { sid(i) = -1; });

    // Build mapping on host: particle_id -> subsystem_id
    auto h_pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);
    auto h_sid = Kokkos::create_mirror_view(p.subsystem_id);
    for (int i = 0; i < N; i++) h_sid(i) = -1;

    for (const auto& sub : active) {
        for (int pid : sub.particle_ids) {
            for (int i = 0; i < N; i++) {
                if (h_pid(i) == pid) {
                    h_sid(i) = sub.id;
                    break;
                }
            }
        }
    }
    Kokkos::deep_copy(p.subsystem_id, h_sid);
}

// ============================================================================
// Remap indices after partition
// ============================================================================

void SubsystemManager::remap_indices(const Particles& p) {
    // Subsystem tracks permanent particle_ids, not array indices.
    // No remapping needed — we always look up by particle_id.
}

// ============================================================================
// Enforce shared timestep for subsystem members (BIFROST Section 3.4.2)
// ============================================================================

void SubsystemManager::enforce_shared_timestep(Particles& p) const {
    if (active.empty()) return;

    auto h_dt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.dt_adapt);
    auto h_pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);

    // Build pid->idx mapping
    int max_pid = 0;
    for (int i = 0; i < p.N; i++) max_pid = std::max(max_pid, h_pid(i));
    std::vector<int> pid_to_idx(max_pid + 1, -1);
    for (int i = 0; i < p.N; i++) pid_to_idx[h_pid(i)] = i;

    // For each subsystem, find min dt and set all members to it
    for (const auto& sub : active) {
        double dt_min = 1e30;
        for (int pid : sub.particle_ids) {
            int idx = pid_to_idx[pid];
            if (idx >= 0) dt_min = std::min(dt_min, h_dt(idx));
        }

        // dt_sub clamp DISABLED for testing — see if natural timestep criteria
        // handle this case without needing the detection-time floor.
        // if (sub.dt_sub > 0.0 && sub.dt_sub < 1e30) {
        //     dt_min = std::min(dt_min, sub.dt_sub);
        // }

        for (int pid : sub.particle_ids) {
            int idx = pid_to_idx[pid];
            if (idx >= 0) h_dt(idx) = dt_min;
        }
    }

    Kokkos::deep_copy(p.dt_adapt, h_dt);
}

// ============================================================================
// Perturbation timestep for subsystems (BIFROST Section 3.4.3, Eq. 30)
//
// 2-body: Gauss planetary equations for da/dt, de/dt from tidal perturbation
//   ε_pert = η_pert * min{ |a/ȧ|, |1/ė| }
// N>2 body: total energy and angular momentum rates
//   ε_pert = η_pert * min{ |E/Ė|, |L/L̇| }
// ============================================================================

void SubsystemManager::apply_perturbation_timestep(Particles& p, double G, double eta_pert) const {
    if (active.empty() || eta_pert <= 0.0) return;

    auto h_x  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
    auto h_y  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
    auto h_z  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
    auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
    auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
    auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
    auto h_ax = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.ax);
    auto h_ay = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.ay);
    auto h_az = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.az);
    auto h_m  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
    auto h_dt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.dt_adapt);
    auto h_pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);

    // Build pid->idx mapping
    int max_pid = 0;
    for (int i = 0; i < p.N; i++) max_pid = std::max(max_pid, h_pid(i));
    std::vector<int> pid_to_idx(max_pid + 1, -1);
    for (int i = 0; i < p.N; i++) pid_to_idx[h_pid(i)] = i;

    bool modified = false;

    for (const auto& sub : active) {
        double dt_pert = 1e30;

        if (sub.n_bodies == 2) {
            // --- 2-body: Gauss equations for ȧ, ė from tidal perturbation ---
            int idx_i = pid_to_idx[sub.particle_ids[0]];
            int idx_j = pid_to_idx[sub.particle_ids[1]];
            if (idx_i < 0 || idx_j < 0) continue;

            // Relative position
            double rx = h_x(idx_j) - h_x(idx_i);
            double ry = h_y(idx_j) - h_y(idx_i);
            double rz = h_z(idx_j) - h_z(idx_i);
            double r2 = rx*rx + ry*ry + rz*rz;
            double r_mag = std::sqrt(r2);
            if (r_mag < 1e-30) continue;
            double r_inv = 1.0 / r_mag;

            // Relative velocity
            double dvx = h_vx(idx_j) - h_vx(idx_i);
            double dvy = h_vy(idx_j) - h_vy(idx_i);
            double dvz = h_vz(idx_j) - h_vz(idx_i);
            double v2 = dvx*dvx + dvy*dvy + dvz*dvz;

            double mu = G * (h_m(idx_i) + h_m(idx_j));

            // Tidal perturbation = differential external acceleration
            // (force kernel excludes internal subsystem pairs, so ax/ay/az are external only)
            double fx = h_ax(idx_j) - h_ax(idx_i);
            double fy = h_ay(idx_j) - h_ay(idx_i);
            double fz = h_az(idx_j) - h_az(idx_i);
            double f_mag = std::sqrt(fx*fx + fy*fy + fz*fz);
            if (f_mag < 1e-30) continue;

            // Orbital elements from current state
            double E_orb = 0.5 * v2 - mu * r_inv;
            double a = -mu / (2.0 * E_orb);  // negative for unbound (hyperbolic)

            // Angular momentum vector: L = r × v
            double Lx = ry*dvz - rz*dvy;
            double Ly = rz*dvx - rx*dvz;
            double Lz = rx*dvy - ry*dvx;
            double L2 = Lx*Lx + Ly*Ly + Lz*Lz;
            double L_mag = std::sqrt(L2);

            // Eccentricity
            double p_param = L2 / mu;  // semi-latus rectum
            double e2 = 1.0 - p_param / a;
            if (e2 < 0.0) e2 = 0.0;
            double e = std::sqrt(e2);

            // Decompose perturbation into radial (R) and tangential (T) components
            // r_hat = r/|r|, t_hat = (L × r) / |L × r| (in orbital plane, perpendicular to r)
            double rhat_x = rx * r_inv, rhat_y = ry * r_inv, rhat_z = rz * r_inv;
            double f_R = fx*rhat_x + fy*rhat_y + fz*rhat_z;  // radial component

            // Tangential: project out radial, then the in-plane component
            // t_hat = (v - (v·r_hat)*r_hat) / |...| but simpler via cross product:
            // t_hat = (L_hat × r_hat)
            double L_inv = (L_mag > 1e-30) ? 1.0 / L_mag : 0.0;
            double Lhat_x = Lx * L_inv, Lhat_y = Ly * L_inv, Lhat_z = Lz * L_inv;
            double that_x = Lhat_y*rhat_z - Lhat_z*rhat_y;
            double that_y = Lhat_z*rhat_x - Lhat_x*rhat_z;
            double that_z = Lhat_x*rhat_y - Lhat_y*rhat_x;
            double f_T = fx*that_x + fy*that_y + fz*that_z;  // tangential component

            // Gauss planetary equations (e.g. Murray & Dermott Ch. 2.9):
            //   ȧ = 2/(n√(1-e²)) * [e·sin(ν)·f_R + (1+e·cos(ν))·f_T]
            //   ė = √(1-e²)/(na) * [sin(ν)·f_R + (cos(ν)+cos(E))·f_T]
            //
            // We compute sin(ν), cos(ν) from state vectors:
            //   cos(ν) = (p/r - 1) / e   where p = L²/μ
            //   sin(ν) = (r·v) * √p / (e·μ·r)  [via h = r² dν/dt]
            double abs_a = std::fabs(a);
            double n = std::sqrt(mu / (abs_a * abs_a * abs_a));  // mean motion
            double sqrt_1me2 = std::sqrt(std::fabs(1.0 - e2));

            double cos_nu, sin_nu;
            if (e > 1e-12) {
                cos_nu = (p_param * r_inv - 1.0) / e;
                double rdot = (rx*dvx + ry*dvy + rz*dvz) * r_inv;  // r·v / |r|
                sin_nu = rdot * std::sqrt(p_param) / (e * std::sqrt(mu));
            } else {
                // Nearly circular: ν from position angle
                cos_nu = 1.0;
                sin_nu = 0.0;
            }

            // cos(E) from cos(ν): cos(E) = (e + cos(ν)) / (1 + e·cos(ν))
            double cos_E = (e + cos_nu) / (1.0 + e * cos_nu);

            double adot = 0.0, edot = 0.0;
            if (n > 1e-30 && sqrt_1me2 > 1e-30) {
                adot = 2.0 / (n * sqrt_1me2) *
                       (e * sin_nu * f_R + (1.0 + e * cos_nu) * f_T);

                edot = sqrt_1me2 / (n * abs_a) *
                       (sin_nu * f_R + (cos_nu + cos_E) * f_T);
            }

            // dt_pert = η * min{ |a/ȧ|, |1/ė| }
            if (std::fabs(adot) > 1e-30)
                dt_pert = std::min(dt_pert, std::fabs(a / adot));
            if (std::fabs(edot) > 1e-30)
                dt_pert = std::min(dt_pert, 1.0 / std::fabs(edot));

            dt_pert *= eta_pert;

        } else {
            // --- N>2 body: |E/Ė| and |L/L̇| from tidal perturbation ---
            // Compute subsystem COM position, velocity, acceleration
            double cx = 0, cy = 0, cz = 0;
            double cvx = 0, cvy = 0, cvz = 0;
            double cax = 0, cay = 0, caz = 0;
            double M_tot = 0;
            std::vector<int> idxs;
            bool skip = false;
            for (int pid : sub.particle_ids) {
                int idx = (pid <= max_pid) ? pid_to_idx[pid] : -1;
                if (idx < 0) { skip = true; break; }
                idxs.push_back(idx);
                double m = h_m(idx);
                M_tot += m;
                cx += m * h_x(idx);  cy += m * h_y(idx);  cz += m * h_z(idx);
                cvx += m * h_vx(idx); cvy += m * h_vy(idx); cvz += m * h_vz(idx);
                cax += m * h_ax(idx); cay += m * h_ay(idx); caz += m * h_az(idx);
            }
            if (skip || M_tot < 1e-30) continue;
            double M_inv = 1.0 / M_tot;
            cx *= M_inv; cy *= M_inv; cz *= M_inv;
            cvx *= M_inv; cvy *= M_inv; cvz *= M_inv;
            cax *= M_inv; cay *= M_inv; caz *= M_inv;

            // Compute internal KE, PE, L and their time derivatives from external perturbation
            double KE = 0, PE = 0;
            double Lx = 0, Ly = 0, Lz = 0;
            double dE_dt = 0;
            double dLx_dt = 0, dLy_dt = 0, dLz_dt = 0;

            for (int k = 0; k < sub.n_bodies; k++) {
                int idx = idxs[k];
                double m = h_m(idx);
                // Relative to COM
                double rx = h_x(idx) - cx, ry = h_y(idx) - cy, rz = h_z(idx) - cz;
                double vx = h_vx(idx) - cvx, vy = h_vy(idx) - cvy, vz = h_vz(idx) - cvz;
                // External perturbation on this particle (= a_external - a_COM_external)
                double fx = h_ax(idx) - cax, fy = h_ay(idx) - cay, fz = h_az(idx) - caz;

                KE += 0.5 * m * (vx*vx + vy*vy + vz*vz);
                Lx += m * (ry*vz - rz*vy);
                Ly += m * (rz*vx - rx*vz);
                Lz += m * (rx*vy - ry*vx);

                // dE/dt from external perturbation: Σ m_k v_k · f_k
                dE_dt += m * (vx*fx + vy*fy + vz*fz);
                // dL/dt from external perturbation: Σ m_k (r_k × f_k)
                dLx_dt += m * (ry*fz - rz*fy);
                dLy_dt += m * (rz*fx - rx*fz);
                dLz_dt += m * (rx*fy - ry*fx);

                // PE from internal pairwise interactions
                for (int l = k + 1; l < sub.n_bodies; l++) {
                    int idx2 = idxs[l];
                    double dx = h_x(idx2) - h_x(idx);
                    double dy = h_y(idx2) - h_y(idx);
                    double dz = h_z(idx2) - h_z(idx);
                    double r = std::sqrt(dx*dx + dy*dy + dz*dz);
                    if (r > 1e-30)
                        PE -= G * m * h_m(idx2) / r;
                }
            }

            double E_int = KE + PE;
            double L_mag = std::sqrt(Lx*Lx + Ly*Ly + Lz*Lz);
            double dL_dt = (L_mag > 1e-30) ?
                (Lx*dLx_dt + Ly*dLy_dt + Lz*dLz_dt) / L_mag : 0.0;

            // dt_pert = η * min{ |E/Ė|, |L/L̇| }
            if (std::fabs(dE_dt) > 1e-30 && std::fabs(E_int) > 1e-30)
                dt_pert = std::min(dt_pert, std::fabs(E_int / dE_dt));
            if (std::fabs(dL_dt) > 1e-30 && L_mag > 1e-30)
                dt_pert = std::min(dt_pert, L_mag / std::fabs(dL_dt));

            dt_pert *= eta_pert;
        }

        if (dt_pert >= 1e30) continue;

        // Clamp dt_adapt of all members
        for (int pid : sub.particle_ids) {
            int idx = (pid <= max_pid) ? pid_to_idx[pid] : -1;
            if (idx >= 0 && dt_pert < h_dt(idx)) {
                h_dt(idx) = dt_pert;
                modified = true;
            }
        }
    }

    if (modified) {
        Kokkos::deep_copy(p.dt_adapt, h_dt);
    }
}

void SubsystemManager::reset_step_diagnostics() {
    step_total_pairs = 0;
    step_total_bound = 0;
    step_total_subsystems = 0;
    step_n_bin = 0; step_n_fly = 0; step_n_trip_s = 0; step_n_trip_u = 0; step_n_multi = 0;
    step_n_secular = 0;
    step_total_members = 0;
    last_level_pairs = 0;
    last_level_bound = 0;
    step_logh_calls = 0;
    step_logh_gbs_total = 0;
    step_logh_worst_dE = 0;
    step_logh_best_dE = 1e30;
    step_logh_cap_hit = 0;
    step_logh_restarts = 0;
    step_logh_restart_ok = 0;
    step_secular_evolved = 0;
    step_secular_triples = 0;
    step_mergers = 0;
    step_escapers = 0;
}

void SubsystemManager::accumulate_diagnostics() {
    // Use pair counts already computed by detect_from_pairs
    step_total_pairs += last_level_pairs;
    step_total_bound += last_level_bound;
    step_total_subsystems += (int)active.size();
    for (const auto& sub : active) {
        step_total_members += sub.n_bodies;
        switch (sub.type) {
            case SubsystemType::BINARY:         step_n_bin++; break;
            case SubsystemType::FLYBY:          step_n_fly++; break;
            case SubsystemType::TRIPLE_STABLE:  step_n_trip_s++; break;
            case SubsystemType::TRIPLE_UNSTABLE:step_n_trip_u++; break;
            case SubsystemType::MULTIPLET:      step_n_multi++; break;
        }
        if (sub.is_secular) step_n_secular++;
    }
}

void SubsystemManager::print_step_summary() const {
    // --- Collisional ---
    std::printf("\n  --- Collisional ");
    for (int i = 0; i < 62; i++) std::printf("-");
    std::printf("\n");

    std::printf("    Subsystems: %d pairs (%d bound), %d active (%d particles)\n",
                step_total_pairs, step_total_bound, step_total_subsystems, step_total_members);
    std::printf("      Types:  bin=%d  fly=%d  tri_s=%d  tri_u=%d  multi=%d\n",
                step_n_bin, step_n_fly, step_n_trip_s, step_n_trip_u, step_n_multi);

    // Integration summary
    std::printf("    Integration: LogH=%d", step_logh_calls);
    if (step_logh_calls > 0) {
        std::printf(" (gbs=%d, dE=[%.1e,%.1e]",
                    step_logh_gbs_total,
                    step_logh_best_dE, step_logh_worst_dE);
        if (step_logh_cap_hit > 0)
            std::printf(", cap=%d", step_logh_cap_hit);
        if (step_logh_restarts > 0)
            std::printf(", restart=%d/%d", step_logh_restart_ok, step_logh_restarts);
        std::printf(")");
    }
    std::printf("  Secular=%d  SecTriple=%d\n", step_secular_evolved, step_secular_triples);

    // Events
    std::printf("    Events: mergers=%d  escapers=%d\n", step_mergers, step_escapers);
}

void SubsystemManager::clear() {
    active.clear();
}
