#include "subsystem.h"
#include "collisional_config.h"
#include "constants.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdio>

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
    // Copy neighbor count from device
    auto h_count = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ngb_count);
    int n_pairs = h_count(0);
    if (n_pairs == 0) {
        // No new pairs detected. Keep existing subsystems — they persist
        // until prune() dissolves them (unbound) or particles escape.
        // Same-subsystem pairs are excluded by the force/timestep kernel,
        // so they won't appear in the neighbor buffer. Clearing here would
        // destroy active subsystems every time re-detection runs.
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

    // Create subsystems
    active.clear();
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
        Subsystem sub{};
        sub.id = next_id++;
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
// Prune unbound subsystems
// ============================================================================

void SubsystemManager::prune(Particles& p, double G) {
    if (active.empty()) return;

    auto h_x  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
    auto h_y  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
    auto h_z  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
    auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
    auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
    auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
    auto h_m  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
    auto h_pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);

    // Build pid->idx
    int max_pid = 0;
    for (int i = 0; i < p.N; i++) max_pid = std::max(max_pid, h_pid(i));
    std::vector<int> pid_to_idx(max_pid + 1, -1);
    for (int i = 0; i < p.N; i++) pid_to_idx[h_pid(i)] = i;

    active.erase(std::remove_if(active.begin(), active.end(),
        [&](Subsystem& sub) {
            if (sub.n_bodies != 2) return false;  // only prune binaries for now

            int idx_i = pid_to_idx[sub.particle_ids[0]];
            int idx_j = pid_to_idx[sub.particle_ids[1]];
            if (idx_i < 0 || idx_j < 0) return true;

            double dx = h_x(idx_j) - h_x(idx_i);
            double dy = h_y(idx_j) - h_y(idx_i);
            double dz = h_z(idx_j) - h_z(idx_i);
            double dvx = h_vx(idx_j) - h_vx(idx_i);
            double dvy = h_vy(idx_j) - h_vy(idx_i);
            double dvz = h_vz(idx_j) - h_vz(idx_i);

            double r = std::sqrt(dx*dx + dy*dy + dz*dz);
            double v2 = dvx*dvx + dvy*dvy + dvz*dvz;
            double mu = G * (h_m(idx_i) + h_m(idx_j));
            double E = 0.5 * v2 - mu / r;

            return (E >= 0.0);  // unbound → remove
        }), active.end());
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
// Perturbation timestep for subsystems (BIFROST Section 3.4.3)
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
        if (sub.n_bodies != 2) continue;  // only binaries for now

        int idx_i = pid_to_idx[sub.particle_ids[0]];
        int idx_j = pid_to_idx[sub.particle_ids[1]];
        if (idx_i < 0 || idx_j < 0) continue;

        // Relative position and velocity
        double rx = h_x(idx_j) - h_x(idx_i);
        double ry = h_y(idx_j) - h_y(idx_i);
        double rz = h_z(idx_j) - h_z(idx_i);
        double r2 = rx*rx + ry*ry + rz*rz;
        double r_mag = std::sqrt(r2);
        if (r_mag < 1e-30) continue;

        double vx = h_vx(idx_j) - h_vx(idx_i);
        double vy = h_vy(idx_j) - h_vy(idx_i);
        double vz = h_vz(idx_j) - h_vz(idx_i);
        double v_mag = std::sqrt(vx*vx + vy*vy + vz*vz);

        // Tidal perturbation = difference in EXTERNAL (field) accelerations.
        // Since the force kernel already excludes internal subsystem pairs (COLL),
        // h_ax/h_ay/h_az contain only external forces. The perturbation is simply
        // the differential external acceleration across the binary.
        // (No need to subtract internal a_int — it's already excluded.)
        double mu = G * (h_m(idx_i) + h_m(idx_j));
        double da_tid_x = h_ax(idx_j) - h_ax(idx_i);
        double da_tid_y = h_ay(idx_j) - h_ay(idx_i);
        double da_tid_z = h_az(idx_j) - h_az(idx_i);
        double da_tid = std::sqrt(da_tid_x*da_tid_x + da_tid_y*da_tid_y + da_tid_z*da_tid_z);

        if (da_tid < 1e-30) continue;

        // Perturbation timestep: dt_pert = eta_pert * |v_rel| / |delta_a_tidal|
        // This is the timescale over which the tidal perturbation significantly
        // changes the relative velocity (and hence orbital elements).
        double dt_pert = eta_pert * std::max(v_mag, std::sqrt(mu / std::max(sub.orb.a, r_mag))) / da_tid;

        // Clamp dt_adapt of both members
        for (int pid : sub.particle_ids) {
            int idx = pid_to_idx[pid];
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
    std::printf("  Subsystems: %d pairs found (%d bound), %d subsystems (%d particles)",
                step_total_pairs, step_total_bound, step_total_subsystems, step_total_members);
    if (step_total_subsystems > 0) {
        std::printf(" [");
        bool first = true;
        if (step_n_bin > 0)    { std::printf("%s%d bin", first ? "" : ", ", step_n_bin); first = false; }
        if (step_n_fly > 0)    { std::printf("%s%d fly", first ? "" : ", ", step_n_fly); first = false; }
        if (step_n_trip_s > 0) { std::printf("%s%d triple(s)", first ? "" : ", ", step_n_trip_s); first = false; }
        if (step_n_trip_u > 0) { std::printf("%s%d triple(u)", first ? "" : ", ", step_n_trip_u); first = false; }
        if (step_n_multi > 0)  { std::printf("%s%d multi", first ? "" : ", ", step_n_multi); first = false; }
        if (step_n_secular > 0){ std::printf("%s%d secular", first ? "" : ", ", step_n_secular); first = false; }
        std::printf("]");
    }
    std::printf("\n");
}

void SubsystemManager::clear() {
    active.clear();
}
