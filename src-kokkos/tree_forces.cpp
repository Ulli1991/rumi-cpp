#include "tree.h"
#include "integrator.h"
#include "constants.h"
#include <Kokkos_Core.hpp>
#include <cfloat>

// ==========================================================================
// Barnes-Hut tree traversal kernel (GPU)
//
// Group-based traversal following Bonsai2:
// - Each Kokkos team handles one particle group (TREE_GROUP_SIZE particles)
// - MAC evaluated against GROUP bounding box (all threads take same branch)
// - Shared-memory stack for depth-first traversal
// - Direct P2P at leaf cells, monopole approximation for distant cells
// ==========================================================================

void tree_self_acceleration(Particles& p, const Octree& tree,
                            int start, int count, double G, double soft2) {
    if (count <= 0 || tree.n_cells == 0) return;

    const int n_groups = tree.n_groups;

    // Device Views from tree
    auto cell_com = tree.cell_com;
    auto cell_size_sq_theta = tree.cell_size_sq_theta;
    auto cell_children = tree.cell_children;
    auto cell_pbeg = tree.cell_pbeg;
    auto cell_pend = tree.cell_pend;
    auto cell_is_leaf = tree.cell_is_leaf;
    auto sorted_pos = tree.sorted_pos;
    auto sorted_indices = tree.sorted_indices;
    auto grp_start = tree.group_start;
    auto grp_count = tree.group_count;
    auto grp_center = tree.group_center;
    auto grp_size = tree.group_size;

    // Output acceleration arrays
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    // Scratch: stack (ints) + particle positions (doubles)
    // Stack: TREE_STACK_SIZE ints
    // Particle cache: TREE_GROUP_SIZE * 4 doubles (for P2P leaf interactions)
    size_t stack_bytes = TREE_STACK_SIZE * sizeof(int);
    size_t particle_cache_bytes = TREE_GROUP_SIZE * 4 * sizeof(double);
    size_t total_scratch = stack_bytes + particle_cache_bytes;

    Kokkos::parallel_for("tree_traversal",
        team_policy(n_groups, TREE_GROUP_SIZE)
            .set_scratch_size(0, Kokkos::PerTeam(total_scratch)),
        KOKKOS_LAMBDA(const member_type& team) {
            int gid = team.league_rank();
            int tid = team.team_rank();
            int team_size = team.team_size();

            // Shared memory regions
            // Stack occupies first TREE_STACK_SIZE ints
            Kokkos::View<int*, scratch_space, Kokkos::MemoryUnmanaged>
                stack(team.team_scratch(0), TREE_STACK_SIZE);

            // Get group info
            int g_start = grp_start(gid);
            int g_count_val = grp_count(gid);
            double gcx = grp_center(gid, 0);
            double gcy = grp_center(gid, 1);
            double gcz = grp_center(gid, 2);
            double gsx = grp_size(gid, 0);
            double gsy = grp_size(gid, 1);
            double gsz = grp_size(gid, 2);

            // Each thread loads its particle position
            int my_sorted_idx = g_start + tid;
            double xi = 0, yi = 0, zi = 0;
            int my_orig_idx = -1;
            if (tid < g_count_val) {
                double mi_unused;
                load4(&sorted_pos(my_sorted_idx, 0), xi, yi, zi, mi_unused);
                my_orig_idx = sorted_indices(my_sorted_idx);
            }

            double axi = 0, ayi = 0, azi = 0;

            // Initialize stack with root node
            if (tid == 0) {
                stack(0) = 0;  // root cell index
            }
            int stack_top = 1;  // number of items on stack
            team.team_barrier();

            // Tree walk
            while (stack_top > 0) {
                // Pop node from stack (thread 0 manages stack)
                int node;
                if (tid == 0) {
                    node = stack(stack_top - 1);
                    stack_top--;
                }
                // Broadcast node to all threads
                // Use team-level broadcast via shared memory
                if (tid == 0) {
                    stack(TREE_STACK_SIZE - 1) = node;  // temp broadcast slot
                }
                team.team_barrier();
                node = stack(TREE_STACK_SIZE - 1);
                int current_stack_top = stack_top;  // capture for all threads

                // Read cell data
                double ccx, ccy, ccz, cmass;
                load4(&cell_com(node, 0), ccx, ccy, ccz, cmass);
                double size_sq_th = cell_size_sq_theta(node);
                int is_leaf = cell_is_leaf(node);

                // MAC test: improved Barnes-Hut using group bounding box
                // Distance from group center to cell center, minus group extent
                double drx = Kokkos::fabs(gcx - ccx) - gsx;
                double dry = Kokkos::fabs(gcy - ccy) - gsy;
                double drz = Kokkos::fabs(gcz - ccz) - gsz;
                drx = Kokkos::fmax(drx, 0.0);
                dry = Kokkos::fmax(dry, 0.0);
                drz = Kokkos::fmax(drz, 0.0);
                double dist_sq = drx * drx + dry * dry + drz * drz;

                // Accept (use monopole) if cell is small enough relative to distance
                bool accept = (size_sq_th < dist_sq) && (dist_sq > 0.0);

                if (accept) {
                    // Monopole approximation: all threads compute force from cell CoM
                    if (tid < g_count_val) {
                        double dx = ccx - xi;
                        double dy = ccy - yi;
                        double dz = ccz - zi;
                        double r2 = dx * dx + dy * dy + dz * dz + soft2;
                        double inv_r = fast_rsqrt(r2);
                        double inv_r3 = inv_r * inv_r * inv_r;
                        double f = G * cmass * inv_r3;
                        axi += f * dx;
                        ayi += f * dy;
                        azi += f * dz;
                    }
                } else if (is_leaf) {
                    // Direct P2P: loop over all particles in this leaf cell
                    int lpbeg = cell_pbeg(node);
                    int lpend = cell_pend(node);

                    if (tid < g_count_val) {
                        for (int j = lpbeg; j < lpend; j++) {
                            // Skip self-interaction
                            if (sorted_indices(j) == my_orig_idx) continue;

                            double pjx, pjy, pjz, mj;
                            load4(&sorted_pos(j, 0), pjx, pjy, pjz, mj);
                            double dx = pjx - xi;
                            double dy = pjy - yi;
                            double dz = pjz - zi;
                            double r2 = dx * dx + dy * dy + dz * dz + soft2;
                            double inv_r = fast_rsqrt(r2);
                            double inv_r3 = inv_r * inv_r * inv_r;
                            double f = G * mj * inv_r3;
                            axi += f * dx;
                            ayi += f * dy;
                            azi += f * dz;
                        }
                    }
                } else {
                    // Open node: push non-empty children onto stack
                    // Thread 0 handles stack management
                    if (tid == 0) {
                        stack_top = current_stack_top;
                        for (int c = 0; c < 8; c++) {
                            int child = cell_children(node, c);
                            if (child >= 0 && stack_top < TREE_STACK_SIZE - 1) {
                                stack(stack_top++) = child;
                            }
                        }
                    }
                }

                team.team_barrier();
                // Update stack_top for all threads
                if (tid == 0) {
                    stack(TREE_STACK_SIZE - 2) = stack_top;
                }
                team.team_barrier();
                stack_top = stack(TREE_STACK_SIZE - 2);
            }

            // Write results back via sorted_indices mapping
            if (tid < g_count_val) {
                pax(my_orig_idx) = axi;
                pay(my_orig_idx) = ayi;
                paz(my_orig_idx) = azi;
            }
        });
    Kokkos::fence();
}

// ==========================================================================
// Fused tree traversal: forces + adaptive timestep
// For monopole nodes: timestep from cell mass + distance (free-fall only)
// For P2P leaves: exact pairwise timestep (free-fall + fly-by)
// ==========================================================================
void tree_self_acceleration_and_timestep(Particles& p, const Octree& tree,
                                          int start, int count,
                                          double G, double soft2, double eta) {
    if (count <= 0 || tree.n_cells == 0) return;

    const int n_groups = tree.n_groups;

    auto cell_com = tree.cell_com;
    auto cell_size_sq_theta = tree.cell_size_sq_theta;
    auto cell_children = tree.cell_children;
    auto cell_pbeg = tree.cell_pbeg;
    auto cell_pend = tree.cell_pend;
    auto cell_is_leaf = tree.cell_is_leaf;
    auto sorted_pos = tree.sorted_pos;
    auto sorted_indices = tree.sorted_indices;
    auto grp_start = tree.group_start;
    auto grp_count = tree.group_count;
    auto grp_center = tree.group_center;
    auto grp_size = tree.group_size;

    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView dt_f{p.dt_adapt.data()};
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    RawView pm{p.mass.data()};

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t stack_bytes = TREE_STACK_SIZE * sizeof(int);
    size_t total_scratch = stack_bytes;

    Kokkos::parallel_for("tree_traversal_dt",
        team_policy(n_groups, TREE_GROUP_SIZE)
            .set_scratch_size(0, Kokkos::PerTeam(total_scratch)),
        KOKKOS_LAMBDA(const member_type& team) {
            int gid = team.league_rank();
            int tid = team.team_rank();

            Kokkos::View<int*, scratch_space, Kokkos::MemoryUnmanaged>
                stack(team.team_scratch(0), TREE_STACK_SIZE);

            int g_start = grp_start(gid);
            int g_count_val = grp_count(gid);
            double gcx = grp_center(gid, 0);
            double gcy = grp_center(gid, 1);
            double gcz = grp_center(gid, 2);
            double gsx = grp_size(gid, 0);
            double gsy = grp_size(gid, 1);
            double gsz = grp_size(gid, 2);

            int my_sorted_idx = g_start + tid;
            double xi = 0, yi = 0, zi = 0, mi = 0;
            double vxi = 0, vyi = 0, vzi = 0;
            int my_orig_idx = -1;
            if (tid < g_count_val) {
                load4(&sorted_pos(my_sorted_idx, 0), xi, yi, zi, mi);
                my_orig_idx = sorted_indices(my_sorted_idx);
                vxi = pvx(my_orig_idx);
                vyi = pvy(my_orig_idx);
                vzi = pvz(my_orig_idx);
            }

            double axi = 0, ayi = 0, azi = 0;
            double t_min_sq = 1e60;

            if (tid == 0) stack(0) = 0;
            int stack_top = 1;
            team.team_barrier();

            while (stack_top > 0) {
                int node;
                if (tid == 0) {
                    node = stack(stack_top - 1);
                    stack_top--;
                    stack(TREE_STACK_SIZE - 1) = node;
                }
                team.team_barrier();
                node = stack(TREE_STACK_SIZE - 1);
                int current_stack_top = stack_top;

                double ccx, ccy, ccz, cmass;
                load4(&cell_com(node, 0), ccx, ccy, ccz, cmass);
                double size_sq_th = cell_size_sq_theta(node);
                int is_leaf = cell_is_leaf(node);

                double drx = Kokkos::fabs(gcx - ccx) - gsx;
                double dry = Kokkos::fabs(gcy - ccy) - gsy;
                double drz = Kokkos::fabs(gcz - ccz) - gsz;
                drx = Kokkos::fmax(drx, 0.0);
                dry = Kokkos::fmax(dry, 0.0);
                drz = Kokkos::fmax(drz, 0.0);
                double dist_sq = drx * drx + dry * dry + drz * drz;
                bool accept = (size_sq_th < dist_sq) && (dist_sq > 0.0);

                if (accept) {
                    if (tid < g_count_val) {
                        double dx = ccx - xi;
                        double dy = ccy - yi;
                        double dz = ccz - zi;
                        double r2 = dx * dx + dy * dy + dz * dz + soft2;
                        double inv_r = fast_rsqrt(r2);
                        double inv_r3 = inv_r * inv_r * inv_r;
                        double f = G * cmass * inv_r3;
                        axi += f * dx;
                        ayi += f * dy;
                        azi += f * dz;

                        // Monopole timestep: free-fall only (conservative)
                        double mass_sum = mi + cmass;
                        double r3 = r2 * r2 * inv_r;
                        double t_ff_sq = r3 / (G * mass_sum + EPS);
                        t_min_sq = Kokkos::fmin(t_min_sq, t_ff_sq);
                    }
                } else if (is_leaf) {
                    int lpbeg = cell_pbeg(node);
                    int lpend = cell_pend(node);
                    if (tid < g_count_val) {
                        for (int j = lpbeg; j < lpend; j++) {
                            if (sorted_indices(j) == my_orig_idx) continue;
                            double pjx, pjy, pjz, mj;
                            load4(&sorted_pos(j, 0), pjx, pjy, pjz, mj);
                            double dx = pjx - xi;
                            double dy = pjy - yi;
                            double dz = pjz - zi;
                            double r2 = dx * dx + dy * dy + dz * dz + soft2;
                            double inv_r = fast_rsqrt(r2);
                            double inv_r3 = inv_r * inv_r * inv_r;
                            double f = G * mj * inv_r3;
                            axi += f * dx;
                            ayi += f * dy;
                            azi += f * dz;

                            // Exact pairwise timestep: free-fall + fly-by
                            int j_orig = sorted_indices(j);
                            double dvx = pvx(j_orig) - vxi;
                            double dvy = pvy(j_orig) - vyi;
                            double dvz = pvz(j_orig) - vzi;
                            double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
                            double mass_sum = mi + mj;
                            double r3 = r2 * r2 * inv_r;
                            double t_ff_sq = r3 / (G * mass_sum + EPS);
                            double t_fb_sq = r2 / (v_sq + EPS);
                            t_min_sq = Kokkos::fmin(t_min_sq,
                                       Kokkos::fmin(t_ff_sq, t_fb_sq));
                        }
                    }
                } else {
                    if (tid == 0) {
                        stack_top = current_stack_top;
                        for (int c = 0; c < 8; c++) {
                            int child = cell_children(node, c);
                            if (child >= 0 && stack_top < TREE_STACK_SIZE - 1) {
                                stack(stack_top++) = child;
                            }
                        }
                    }
                }

                team.team_barrier();
                if (tid == 0) stack(TREE_STACK_SIZE - 2) = stack_top;
                team.team_barrier();
                stack_top = stack(TREE_STACK_SIZE - 2);
            }

            if (tid < g_count_val) {
                pax(my_orig_idx) = axi;
                pay(my_orig_idx) = ayi;
                paz(my_orig_idx) = azi;
                dt_f(my_orig_idx) = Kokkos::sqrt(t_min_sq) * eta;
            }
        });
    Kokkos::fence();
}

// ==========================================================================
// Post-hoc momentum correction for tree forces
//
// Tree forces violate Newton's 3rd law because different groups may use
// different approximations (monopole vs P2P) for the same interaction pair.
// This O(N) correction subtracts the net acceleration (weighted by mass)
// to enforce exact linear momentum conservation: sum(m_i * a_i) = 0.
// ==========================================================================
void tree_momentum_correction(Particles& p, int start, int count) {
    if (count <= 1) return;

    auto ax_v = p.ax;
    auto ay_v = p.ay;
    auto az_v = p.az;
    auto mass_v = p.mass;

    double sum_mx = 0, sum_my = 0, sum_mz = 0, sum_m = 0;

    Kokkos::parallel_reduce("mom_mx", count, KOKKOS_LAMBDA(int ii, double& v) {
        int i = start + ii;
        v += mass_v(i) * ax_v(i);
    }, sum_mx);
    Kokkos::parallel_reduce("mom_my", count, KOKKOS_LAMBDA(int ii, double& v) {
        int i = start + ii;
        v += mass_v(i) * ay_v(i);
    }, sum_my);
    Kokkos::parallel_reduce("mom_mz", count, KOKKOS_LAMBDA(int ii, double& v) {
        int i = start + ii;
        v += mass_v(i) * az_v(i);
    }, sum_mz);
    Kokkos::parallel_reduce("mom_m", count, KOKKOS_LAMBDA(int ii, double& v) {
        int i = start + ii;
        v += mass_v(i);
    }, sum_m);

    double inv_m = 1.0 / sum_m;
    double cx = sum_mx * inv_m;
    double cy = sum_my * inv_m;
    double cz = sum_mz * inv_m;

    Kokkos::parallel_for("mom_correct", count, KOKKOS_LAMBDA(int ii) {
        int i = start + ii;
        ax_v(i) -= cx;
        ay_v(i) -= cy;
        az_v(i) -= cz;
    });
}
