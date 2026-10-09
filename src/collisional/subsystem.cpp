#include "small_for.h"
#include "subsystem.h"
#include "collisional_config.h"
#include "constants.h"
#include <algorithm>
#include <unordered_map>
#include <numeric>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>

// Global subsystem manager instance
SubsystemManager g_subsystem_mgr;

// ============================================================================
// Cached pid → array index map
// ============================================================================

void SubsystemManager::rebuild_pid_to_idx(const Particles& p) {
    // Local managers (LogH batches) borrow the global map
    if (this != &g_subsystem_mgr) {
        g_subsystem_mgr.rebuild_pid_to_idx(p);
        shared_map_ = &g_subsystem_mgr;   // no copy of the O(max_pid) vector
        return;
    }
    // Incremental: only the slots touched since the last build (O(count) instead of O(N) per call)
    const int lo = p.layout_dirty_lo, hi = std::min(p.layout_dirty_hi, p.N);
    if (map_source_ == &p && !pid_to_idx_.empty() && p.layout_dirty_hi != INT_MAX) {
        for (int pid : p.removed_pid_log)
            if (pid >= 0 && pid < (int)pid_to_idx_.size()) pid_to_idx_[pid] = -1;
        bool ok = true;
        if (lo < hi) {
            auto h_rng = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                             Kokkos::subview(p.particle_id, Kokkos::pair<int,int>(lo, hi)));
            for (int i = lo; i < hi && ok; i++) {
                int pid = h_rng(i - lo);
                if (pid < 0 || pid >= (int)pid_to_idx_.size()) ok = false;
                else pid_to_idx_[pid] = i;
            }
        }
        if (ok) {
            p.layout_dirty_lo = INT_MAX; p.layout_dirty_hi = 0; p.removed_pid_log.clear();
            if (std::getenv("NBODY_CHECK_PIDMAP")) check_pid_to_idx(p);
            return;
        }
    }
    auto h_pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);
    int max_pid = 0;
    for (int i = 0; i < p.N; i++) max_pid = std::max(max_pid, h_pid(i));
    if ((int)pid_to_idx_.size() < max_pid + 1) {
        pid_to_idx_.resize(max_pid + 1, -1);
    } else {
        std::fill(pid_to_idx_.begin(), pid_to_idx_.end(), -1);
    }
    for (int i = 0; i < p.N; i++) pid_to_idx_[h_pid(i)] = i;
    pid_to_idx_max_ = max_pid;
    map_source_ = &p;
    p.layout_dirty_lo = INT_MAX; p.layout_dirty_hi = 0; p.removed_pid_log.clear();
}

void SubsystemManager::check_pid_to_idx(const Particles& p) const {
    auto h_pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);
    std::vector<int> full(pid_to_idx_.size(), -1);
    for (int i = 0; i < p.N; i++) if (h_pid(i) >= 0 && h_pid(i) < (int)full.size()) full[h_pid(i)] = i;
    for (size_t k = 0; k < full.size(); k++)
        if (full[k] != pid_to_idx_[k]) {
            std::fprintf(stderr, "PIDMAP MISMATCH pid=%zu map=%d actual=%d\n", k, pid_to_idx_[k], full[k]);
            std::abort();
        }
}

namespace {
// Host copy of the subsystem members' state only (instead of full O(N) arrays per HHS level). The
// columns keep the h_x(idx) call syntax of the code that used full-array mirrors.
struct MemberData {
    enum { X, Y, Z, VX, VY, VZ, AX, AY, AZ, M, DT, PID, NF };
    std::unordered_map<int, int> loc;   // particle slot -> member row
    std::vector<int> idx;
    std::vector<double> f[NF];

    struct Col {
        std::vector<double>* v; const std::unordered_map<int, int>* loc;
        double& operator()(int i) const {
            auto it = loc->find(i);
            if (it == loc->end()) {   // a slot outside the gathered members: map and gather disagree
                std::fprintf(stderr, "MemberData: slot %d was not gathered\n", i);
                std::abort();
            }
            return (*v)[it->second];
        }
    };
    Col col(int k) { return Col{&f[k], &loc}; }

    MemberData(const std::vector<Subsystem>& subs, const std::vector<int>& p2i, const Particles& p) {
        for (const auto& sub : subs)
            for (int pid : sub.particle_ids) {
                int i = (pid >= 0 && pid < (int)p2i.size()) ? p2i[pid] : -1;
                if (i >= 0 && i < p.N && loc.emplace(i, (int)idx.size()).second) idx.push_back(i);
            }
        gather(p);
    }
    MemberData(const std::vector<int>& slots, const Particles& p) {
        for (int i : slots)
            if (i >= 0 && i < p.N && loc.emplace(i, (int)idx.size()).second) idx.push_back(i);
        gather(p);
    }

    void gather(const Particles& p) {
        const int n = (int)idx.size();
        for (auto& v : f) v.resize(n);
        if (n == 0) return;
        Kokkos::View<int*> d_idx("member_idx", n);
        Kokkos::View<double*> d_buf("member_gather", (size_t)n * NF);
        auto h_idx = Kokkos::create_mirror_view(d_idx);
        std::copy(idx.begin(), idx.end(), h_idx.data());
        Kokkos::deep_copy(d_idx, h_idx);
        auto px = p.x, py = p.y, pz = p.z, pvx = p.vx, pvy = p.vy, pvz = p.vz;
        auto pax = p.ax, pay = p.ay, paz = p.az, pm = p.mass, pdt = p.dt_adapt;
        auto ppid = p.particle_id;
        small_for("member_gather", n, KOKKOS_LAMBDA(int k) {
            int i = d_idx(k);
            double* b = &d_buf(k * NF);
            b[X] = px(i); b[Y] = py(i); b[Z] = pz(i); b[VX] = pvx(i); b[VY] = pvy(i); b[VZ] = pvz(i);
            b[AX] = pax(i); b[AY] = pay(i); b[AZ] = paz(i); b[M] = pm(i); b[DT] = pdt(i);
            b[PID] = (double)ppid(i);   // ids are exactly representable
        });
        auto h_buf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_buf);
        for (int k = 0; k < n; k++)
            for (int c = 0; c < NF; c++) f[c][k] = h_buf(k * NF + c);
    }

    // Write the members' dt back to the device
    void scatter_dt(Particles& p) const {
        const int n = (int)idx.size();
        if (n == 0) return;
        Kokkos::View<int*> d_idx("member_idx", n);
        Kokkos::View<double*> d_dt("member_dt", n);
        auto h_idx = Kokkos::create_mirror_view(d_idx);
        auto h_dt = Kokkos::create_mirror_view(d_dt);
        for (int k = 0; k < n; k++) { h_idx(k) = idx[k]; h_dt(k) = f[DT][k]; }
        Kokkos::deep_copy(d_idx, h_idx);
        Kokkos::deep_copy(d_dt, h_dt);
        auto pdt = p.dt_adapt;
        small_for("member_scatter_dt", n, KOKKOS_LAMBDA(int k) { pdt(d_idx(k)) = d_dt(k); });
    }
};
}  // namespace

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
        if (x < 0 || x >= (int)parent.size()) {
            std::fprintf(stderr, "UnionFind: index %d outside [0, %zu)\n", x, parent.size());
            std::abort();
        }
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
    (void)r_ngb;
    (void)r_ngb_max;
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

    // Ids, time-steps, masses (and positions for triples) of the pair members only, not all N
    std::vector<int> pair_slots(2 * (size_t)n_pairs);
    for (int k = 0; k < n_pairs; k++) { pair_slots[2 * k] = h_pi(k); pair_slots[2 * k + 1] = h_pj(k); }
    MemberData md(pair_slots, p);
    auto h_pid_d = md.col(MemberData::PID);
    auto h_pid = [&](int i) { return (int)h_pid_d(i); };
    auto h_dt  = md.col(MemberData::DT);

    // Rebuild cached pid → index mapping
    rebuild_pid_to_idx(p);
    const auto& pid_to_idx = pid_to_idx_;
    int max_pid = pid_to_idx_max_;

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
    // Union-find on compact ids (O(pairs), not sized by the largest particle id)
    (void)max_pid;
    std::unordered_map<int, int> compact;
    for (auto& pr : pairs) { compact.emplace(pr.pid_lo, (int)compact.size()); compact.emplace(pr.pid_hi, (int)compact.size()); }
    UnionFind uf((int)compact.size());
    for (auto& pr : pairs) {
        uf.unite(compact[pr.pid_lo], compact[pr.pid_hi]);
    }

    // Components: (root, pid) sorted and unique (members ascending, as before); each component's pairs in their
    // sorted order, for the triple search below
    std::vector<std::pair<int, int>> root_pid;
    root_pid.reserve(2 * pairs.size());
    std::unordered_map<int, std::vector<int>> pairs_of_root;
    for (int k = 0; k < (int)pairs.size(); k++) {
        const auto& pr = pairs[k];
        int root = uf.find(compact[pr.pid_lo]);
        root_pid.push_back({root, pr.pid_lo});
        root_pid.push_back({root, pr.pid_hi});
        pairs_of_root[root].push_back(k);
    }
    std::sort(root_pid.begin(), root_pid.end());
    root_pid.erase(std::unique(root_pid.begin(), root_pid.end()), root_pid.end());
    std::map<int, std::vector<int>> components;
    for (size_t k = 0; k < root_pid.size();) {
        size_t e = k;
        std::vector<int> members;
        while (e < root_pid.size() && root_pid[e].first == root_pid[k].first) members.push_back(root_pid[e++].second);
        components.emplace_hint(components.end(), root_pid[k].first, std::move(members));
        k = e;
    }
    // Pair of two particle ids in the sorted, unique pair list (binary search), or null
    auto find_pair = [&](int a, int b) -> const PairInfo* {
        int lo = std::min(a, b), hi = std::max(a, b);
        auto it = std::lower_bound(pairs.begin(), pairs.end(), std::make_pair(lo, hi),
                                   [](const PairInfo& p, const std::pair<int, int>& k) {
                                       return p.pid_lo != k.first ? p.pid_lo < k.first : p.pid_hi < k.second;
                                   });
        return (it != pairs.end() && it->pid_lo == lo && it->pid_hi == hi) ? &*it : nullptr;
    };

    // A group larger than LogH's storage is usually a chain linked partly by unbound fly-bys (dense
    // cores). Split it along its bound pairs only: its binaries and triples still become subsystems
    // instead of the whole group being integrated orbit by orbit by the FSI.
    {
        std::vector<int> oversize;
        for (auto& [root, members] : components)
            if ((int)members.size() > LOGH_MAX_BODIES) oversize.push_back(root);
        int next_key = -1;
        for (int root : oversize) {
            std::unordered_map<int, int> in_group;
            for (int pid : components[root]) in_group.emplace(pid, (int)in_group.size());
            components.erase(root);
            UnionFind bound_uf((int)in_group.size());
            for (const auto& pr : pairs) {
                if (!pr.bound) continue;
                auto a = in_group.find(pr.pid_lo), b = in_group.find(pr.pid_hi);
                if (a != in_group.end() && b != in_group.end()) bound_uf.unite(a->second, b->second);
            }
            std::map<int, std::vector<int>> parts;
            for (const auto& [pid, k] : in_group) parts[bound_uf.find(k)].push_back(pid);
            for (auto& [r, part] : parts) {
                if (part.size() < 2) continue;   // a single star: field particle
                std::sort(part.begin(), part.end());
                components[next_key--] = part;   // still > LOGH_MAX_BODIES: left to the FSI below
            }
        }
    }

    // Build lookup from sorted pid set → old subsystem for persistent state transfer
    // (aux_vel, normalization scales carried over for identical member sets)
    auto make_key = [](const std::vector<int>& pids) -> std::vector<int> {
        std::vector<int> k = pids;
        std::sort(k.begin(), k.end());
        return k;
    };
    std::map<std::vector<int>, Subsystem*> old_by_pids;
    for (auto& old_sub : old_active) {
        old_by_pids[make_key(old_sub.particle_ids)] = &old_sub;
    }

    auto h_mass = md.col(MemberData::M);
    auto h_x  = md.col(MemberData::X);
    auto h_y  = md.col(MemberData::Y);
    auto h_z  = md.col(MemberData::Z);
    auto h_vx = md.col(MemberData::VX);
    auto h_vy = md.col(MemberData::VY);
    auto h_vz = md.col(MemberData::VZ);

    // Helper: min dt_adapt over subsystem members (used for binary, triple, multiplet)
    auto min_dt_members = [&](const Subsystem& s) {
        double dt_min = 1e30;
        for (int pid : s.particle_ids) {
            int idx = pid_to_idx[pid];
            if (idx >= 0) dt_min = std::min(dt_min, h_dt(idx));
        }
        return dt_min;
    };

    for (auto& [root, members] : components) {
        // LogH has fixed storage for LOGH_MAX_BODIES; MSTAR, which BIFROST uses for larger groups, is
        // not wired in. Larger groups stay field particles, integrated by the FSI directly.
        if ((int)members.size() > LOGH_MAX_BODIES) {
            static bool warned = false;
            if (!warned) {
                std::printf("WARNING: %d-body group exceeds LOGH_MAX_BODIES=%d, left to the FSI "
                            "(reported once)\n", (int)members.size(), LOGH_MAX_BODIES);
                warned = true;
            }
            continue;
        }

        // Check if this component exactly matches an existing subsystem
        auto key = make_key(members);
        auto it = old_by_pids.find(key);

        Subsystem sub{};
        if (it != old_by_pids.end()) {
            // Same members — carry over persistent LogH state
            sub.id = it->second->id;
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
            if (const PairInfo* pr = find_pair(members[0], members[1])) {
                sub.orb.a = pr->a;
                sub.orb.e = pr->e;
                sub.orb.E_orb = pr->E_orb;
                pair_bound = pr->bound;
            }

            // Compute period (only meaningful for bound orbits with a > 0)
            if (sub.orb.a > 0 && idx_i >= 0 && idx_j >= 0) {
                double mu = G * (h_mass(idx_i) + h_mass(idx_j));
                sub.orb.period = 2.0 * M_PI * std::sqrt(sub.orb.a * sub.orb.a * sub.orb.a / mu);
                sub.n_orb_per_dt = (int)(dt_step / sub.orb.period);
                sub.is_secular = (sub.n_orb_per_dt >= n_orb_secular);
            }

            sub.type = pair_bound ? SubsystemType::BINARY : SubsystemType::FLYBY;
            sub.dt_sub = min_dt_members(sub);
        } else if (sub.n_bodies == 3) {
            // Triple — check stability
            // Find the tightest pair (inner binary)
            double a_min = 1e30;
            int inner_i = -1, inner_j = -1;
            for (int k : pairs_of_root[root]) {   // this component's pairs, in the order of the full list
                const auto& pr = pairs[k];
                if (pr.bound && pr.a > 0 && pr.a < a_min) {
                    a_min = pr.a;
                    inner_i = pr.pid_lo;
                    inner_j = pr.pid_hi;
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
                    if (const PairInfo* pr = find_pair(inner_i, inner_j)) e_in = pr->e;

                    // Compute outer orbit relative to inner binary COM
                    // (pairwise elements use wrong mu and wrong relative vector)
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
            sub.dt_sub = min_dt_members(sub);
        } else {
            sub.type = SubsystemType::MULTIPLET;
            sub.dt_sub = min_dt_members(sub);
        }

        active.push_back(sub);
    }
}

// ============================================================================
// Update subsystem_id in Particles
// ============================================================================

void SubsystemManager::restore_active(const std::vector<Subsystem>& saved) {
    active = saved;
    if (removed_pids.empty()) return;
    active.erase(std::remove_if(active.begin(), active.end(), [&](const Subsystem& sub) {
        for (int pid : sub.particle_ids)
            if (removed_pids.count(pid)) return true;
        return false;
    }), active.end());
}

void SubsystemManager::update_particle_flags(Particles& p) {
    // subsystem_id exists only in collisional mode; the HHS recursion calls this unconditionally
    if (!p.collisional) return;
    // Reset all to -1 on the device, then scatter only the members (no O(N) host loop or upload)
    auto sid = p.subsystem_id;
    int N = p.N;
    Kokkos::parallel_for("reset_sid", N, KOKKOS_LAMBDA(int i) { sid(i) = -1; });

    if (active.empty()) return;
    rebuild_pid_to_idx(p);   // incremental: partitions may have moved particles since the last build

    const auto& p2i = pid_to_idx_;
    std::vector<int> idx_sid;
    idx_sid.reserve(4 * active.size());
    for (const auto& sub : active) {
        for (int pid : sub.particle_ids) {
            int idx = (pid >= 0 && pid < (int)p2i.size()) ? p2i[pid] : -1;
            if (idx >= 0 && idx < N) { idx_sid.push_back(idx); idx_sid.push_back(sub.id); }
        }
    }
    const int n = (int)idx_sid.size() / 2;
    if (n == 0) return;
    if ((int)flag_buf_.extent(0) < 2 * n) flag_buf_ = Kokkos::View<int*>("sid_scatter", 4 * n);
    auto h_buf = Kokkos::create_mirror_view(flag_buf_);
    std::copy(idx_sid.begin(), idx_sid.end(), h_buf.data());
    Kokkos::deep_copy(Kokkos::subview(flag_buf_, Kokkos::pair<int,int>(0, 2 * n)),
                      Kokkos::subview(h_buf, Kokkos::pair<int,int>(0, 2 * n)));
    auto buf = flag_buf_;
    small_for("scatter_sid", n, KOKKOS_LAMBDA(int k) { sid(buf(2 * k)) = buf(2 * k + 1); });
}

// ============================================================================
// Remap indices after partition
// ============================================================================

void SubsystemManager::remap_indices(const Particles& p) {
    (void)p;
    // Subsystem tracks permanent particle_ids, not array indices.
    // No remapping needed — we always look up by particle_id.
}

// ============================================================================
// Enforce shared timestep for subsystem members (BIFROST Section 3.4.2)
// ============================================================================


void SubsystemManager::fill_cm_velocity(const Particles& p, Kokkos::View<double*> cvx, Kokkos::View<double*> cvy,
                                         Kokkos::View<double*> cvz) const {
    const int N = p.N;
    auto vx = p.vx, vy = p.vy, vz = p.vz;
    Kokkos::parallel_for("cm_vel_copy", N, KOKKOS_LAMBDA(int i) { cvx(i) = vx(i); cvy(i) = vy(i); cvz(i) = vz(i); });
    if (active.empty()) return;
    const auto& p2i = pid_to_idx();
    MemberData md(active, p2i, p);
    auto mvx = md.col(MemberData::VX), mvy = md.col(MemberData::VY), mvz = md.col(MemberData::VZ);
    auto mm = md.col(MemberData::M);
    std::vector<int> slot;
    std::vector<double> val;   // cm velocity per listed slot
    for (const auto& sub : active) {
        double M = 0, px = 0, py = 0, pz = 0;
        std::vector<int> members;
        for (int pid : sub.particle_ids) {
            int i = (pid >= 0 && pid < (int)p2i.size()) ? p2i[pid] : -1;
            if (i < 0 || i >= N) continue;
            members.push_back(i);
            M += mm(i); px += mm(i) * mvx(i); py += mm(i) * mvy(i); pz += mm(i) * mvz(i);
        }
        if (M <= 0) continue;
        for (int i : members) { slot.push_back(i); val.push_back(px / M); val.push_back(py / M); val.push_back(pz / M); }
    }
    const int n = (int)slot.size();
    if (n == 0) return;
    Kokkos::View<int*> d_slot("cm_slot", n);
    Kokkos::View<double*> d_val("cm_val", 3 * (size_t)n);
    auto h_slot = Kokkos::create_mirror_view(d_slot);
    auto h_val = Kokkos::create_mirror_view(d_val);
    std::copy(slot.begin(), slot.end(), h_slot.data());
    std::copy(val.begin(), val.end(), h_val.data());
    Kokkos::deep_copy(d_slot, h_slot);
    Kokkos::deep_copy(d_val, h_val);
    small_for("cm_vel_scatter", n, KOKKOS_LAMBDA(int k) {
        int i = d_slot(k);
        cvx(i) = d_val(3 * k); cvy(i) = d_val(3 * k + 1); cvz(i) = d_val(3 * k + 2);
    });
}

// Shared time step of each subsystem on gathered member data (min over its members)
static void shared_timestep_md(const std::vector<Subsystem>& active, const std::vector<int>& pid_to_idx, MemberData& md) {
    auto h_dt = md.col(MemberData::DT);
    for (const auto& sub : active) {
        double dt_min = 1e30;
        for (int pid : sub.particle_ids) {
            int idx = pid_to_idx[pid];
            if (idx >= 0) dt_min = std::min(dt_min, h_dt(idx));
        }
        for (int pid : sub.particle_ids) {
            int idx = pid_to_idx[pid];
            if (idx >= 0) h_dt(idx) = dt_min;
        }
    }
}

void SubsystemManager::enforce_shared_timestep(Particles& p) const {
    if (active.empty()) return;
    MemberData md(active, pid_to_idx_, p);
    shared_timestep_md(active, pid_to_idx_, md);
    md.scatter_dt(p);
}

// ============================================================================
// Perturbation timestep for subsystems (BIFROST Section 3.4.3, Eq. 30)
//
// 2-body: Gauss planetary equations for da/dt, de/dt from tidal perturbation
//   ε_pert = η_pert * min{ |a/ȧ|, |1/ė| }
// N>2 body: total energy and angular momentum rates
//   ε_pert = η_pert * min{ |E/Ė|, |L/L̇| }
// ============================================================================

// Perturbation time step (BIFROST Eq. 30) on gathered member data; returns whether any member was clamped
static bool perturbation_timestep_md(const std::vector<Subsystem>& active, const std::vector<int>& pid_to_idx,
                                     MemberData& md, double G, double eta_pert) {
    auto h_x  = md.col(MemberData::X);
    auto h_y  = md.col(MemberData::Y);
    auto h_z  = md.col(MemberData::Z);
    auto h_vx = md.col(MemberData::VX);
    auto h_vy = md.col(MemberData::VY);
    auto h_vz = md.col(MemberData::VZ);
    auto h_ax = md.col(MemberData::AX);
    auto h_ay = md.col(MemberData::AY);
    auto h_az = md.col(MemberData::AZ);
    auto h_m  = md.col(MemberData::M);
    auto h_dt = md.col(MemberData::DT);

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
            double dt_a = 1e30, dt_e = 1e30;
            if (std::fabs(adot) > 1e-30)
                dt_a = std::fabs(a / adot);
            if (std::fabs(edot) > 1e-30)
                dt_e = 1.0 / std::fabs(edot);
            dt_pert = std::min(dt_a, dt_e) * eta_pert;


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
                int idx = (pid < (int)pid_to_idx.size()) ? pid_to_idx[pid] : -1;
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
            int idx = (pid < (int)pid_to_idx.size()) ? pid_to_idx[pid] : -1;
            if (idx >= 0 && dt_pert < h_dt(idx)) {
                h_dt(idx) = dt_pert;
                modified = true;
            }
        }
    }

    return modified;
}

void SubsystemManager::apply_perturbation_timestep(Particles& p, double G, double eta_pert) const {
    if (active.empty() || eta_pert <= 0.0) return;
    MemberData md(active, pid_to_idx_, p);
    if (perturbation_timestep_md(active, pid_to_idx_, md, G, eta_pert)) md.scatter_dt(p);
}

void SubsystemManager::subsystem_timesteps(Particles& p, double G, double eta_pert) const {
    if (active.empty()) return;
    MemberData md(active, pid_to_idx_, p);   // one gather and one scatter for both passes
    shared_timestep_md(active, pid_to_idx_, md);
    if (eta_pert > 0.0) perturbation_timestep_md(active, pid_to_idx_, md, G, eta_pert);
    md.scatter_dt(p);
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
