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
                            int start, int count, double G, double soft2,
                            int group_offset, int num_groups_param) {
    if (count <= 0 || tree.n_cells == 0) return;

    const int n_groups = (num_groups_param >= 0) ? num_groups_param : tree.n_groups;

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

    const int g_offset = group_offset;
    Kokkos::parallel_for("tree_traversal",
        team_policy(n_groups, TREE_GROUP_SIZE)
            .set_scratch_size(0, Kokkos::PerTeam(total_scratch)),
        KOKKOS_LAMBDA(const member_type& team) {
            int gid = g_offset + team.league_rank();
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

}

// ==========================================================================
// Fused tree traversal: forces + adaptive timestep
// For monopole nodes: timestep from cell mass + distance (free-fall only)
// For P2P leaves: exact pairwise timestep (free-fall + fly-by)
// ==========================================================================
void tree_self_acceleration_and_timestep(Particles& p, const Octree& tree,
                                          int start, int count,
                                          double G, double soft2, double eta,
                                          int group_offset, int num_groups_param) {
    if (count <= 0 || tree.n_cells == 0) return;

    const int n_groups = (num_groups_param >= 0) ? num_groups_param : tree.n_groups;

    auto cell_com = tree.cell_com;
    auto cell_vel_v = tree.cell_vel;
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

    const int g_offset = group_offset;
    Kokkos::parallel_for("tree_traversal_dt",
        team_policy(n_groups, TREE_GROUP_SIZE)
            .set_scratch_size(0, Kokkos::PerTeam(total_scratch)),
        KOKKOS_LAMBDA(const member_type& team) {
            int gid = g_offset + team.league_rank();
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

                        // Symmetrized monopole timestep (FROST Eq. 23)
                        double dvx = cell_vel_v(node, 0) - vxi;
                        double dvy = cell_vel_v(node, 1) - vyi;
                        double dvz = cell_vel_v(node, 2) - vzi;
                        double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
                        double mass_sum = mi + cmass;
                        double inv_Gm = 1.0 / (G * mass_sum + EPS);
                        double inv_vsq = 1.0 / (v_sq + EPS);

                        double r3 = r2 * r2 * inv_r;
                        double t_ff_sq = r3 * inv_Gm;
                        double t_fb_sq = r2 * inv_vsq;
                        double t_ff = t_ff_sq * fast_rsqrt(t_ff_sq + EPS);
                        double t_fb = t_fb_sq * fast_rsqrt(t_fb_sq + EPS);

                        double inv_r2 = inv_r * inv_r;
                        double r_dot_v = dx*dvx + dy*dvy + dz*dvz;

                        double c_ff = 1.5 * r_dot_v * inv_r2;
                        double c_fb = r_dot_v * inv_r2 * (1.0 + G * mass_sum * inv_r * inv_vsq);

                        double ct_ff = c_ff * t_ff_sq;
                        double s_ff = 1.0 + c_ff * ct_ff;
                        double u_ff = s_ff * fast_rsqrt(s_ff);
                        double t_sym_ff = t_ff + ct_ff / (u_ff + 1.0);

                        double ct_fb = c_fb * t_fb_sq;
                        double s_fb = 1.0 + c_fb * ct_fb;
                        double u_fb = s_fb * fast_rsqrt(s_fb);
                        double t_sym_fb = t_fb + ct_fb / (u_fb + 1.0);

                        t_min_sq = Kokkos::fmin(t_min_sq,
                                   Kokkos::fmin(t_sym_ff * t_sym_ff, t_sym_fb * t_sym_fb));
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

                            // Symmetrized pairwise timestep (FROST Eq. 23)
                            int j_orig = sorted_indices(j);
                            double dvx = pvx(j_orig) - vxi;
                            double dvy = pvy(j_orig) - vyi;
                            double dvz = pvz(j_orig) - vzi;
                            double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
                            double mass_sum = mi + mj;
                            double inv_Gm = 1.0 / (G * mass_sum + EPS);
                            double inv_vsq = 1.0 / (v_sq + EPS);

                            double r3 = r2 * r2 * inv_r;
                            double t_ff_sq = r3 * inv_Gm;
                            double t_fb_sq = r2 * inv_vsq;
                            double t_ff = t_ff_sq * fast_rsqrt(t_ff_sq + EPS);
                            double t_fb = t_fb_sq * fast_rsqrt(t_fb_sq + EPS);

                            double inv_r2 = inv_r * inv_r;
                            double r_dot_v = dx*dvx + dy*dvy + dz*dvz;

                            double c_ff = 1.5 * r_dot_v * inv_r2;
                            double c_fb = r_dot_v * inv_r2 * (1.0 + G * mass_sum * inv_r * inv_vsq);

                            // Rationalized form: t_sym = t + c*t²/(sqrt(1 + c²*t²) + 1)
                            double ct_ff = c_ff * t_ff_sq;
                            double s_ff = 1.0 + c_ff * ct_ff;
                            double u_ff = s_ff * fast_rsqrt(s_ff);
                            double t_sym_ff = t_ff + ct_ff / (u_ff + 1.0);

                            double ct_fb = c_fb * t_fb_sq;
                            double s_fb = 1.0 + c_fb * ct_fb;
                            double u_fb = s_fb * fast_rsqrt(s_fb);
                            double t_sym_fb = t_fb + ct_fb / (u_fb + 1.0);

                            t_min_sq = Kokkos::fmin(t_min_sq,
                                       Kokkos::fmin(t_sym_ff * t_sym_ff, t_sym_fb * t_sym_fb));
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

}

// ==========================================================================
// Cross-force via tree: force on i-particles from j-tree
// i_tree provides spatial groups for i-particles
// j_tree provides tree structure to walk (cells, sorted_pos for P2P)
// No self-interaction check (i and j are disjoint subsets)
// ==========================================================================
void tree_cross_acceleration(Particles& p,
                             const Octree& i_tree,
                             const Octree& j_tree,
                             double G, double soft2) {
    if (i_tree.n_particles <= 0 || j_tree.n_cells == 0) return;

    const int n_groups = i_tree.n_groups;

    // i-particle data from i_tree
    auto i_sorted_pos = i_tree.sorted_pos;
    auto i_sorted_indices = i_tree.sorted_indices;
    auto i_grp_start = i_tree.group_start;
    auto i_grp_count = i_tree.group_count;
    auto i_grp_center = i_tree.group_center;
    auto i_grp_size = i_tree.group_size;

    // j-tree structure
    auto j_cell_com = j_tree.cell_com;
    auto j_cell_size_sq_theta = j_tree.cell_size_sq_theta;
    auto j_cell_children = j_tree.cell_children;
    auto j_cell_pbeg = j_tree.cell_pbeg;
    auto j_cell_pend = j_tree.cell_pend;
    auto j_cell_is_leaf = j_tree.cell_is_leaf;
    auto j_sorted_pos = j_tree.sorted_pos;

    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t stack_bytes = TREE_STACK_SIZE * sizeof(int);

    Kokkos::parallel_for("tree_cross_force",
        team_policy(n_groups, TREE_GROUP_SIZE)
            .set_scratch_size(0, Kokkos::PerTeam(stack_bytes)),
        KOKKOS_LAMBDA(const member_type& team) {
            int gid = team.league_rank();
            int tid = team.team_rank();

            Kokkos::View<int*, scratch_space, Kokkos::MemoryUnmanaged>
                stack(team.team_scratch(0), TREE_STACK_SIZE);

            int g_start = i_grp_start(gid);
            int g_count_val = i_grp_count(gid);
            double gcx = i_grp_center(gid, 0);
            double gcy = i_grp_center(gid, 1);
            double gcz = i_grp_center(gid, 2);
            double gsx = i_grp_size(gid, 0);
            double gsy = i_grp_size(gid, 1);
            double gsz = i_grp_size(gid, 2);

            int my_sorted_idx = g_start + tid;
            double xi = 0, yi = 0, zi = 0;
            int my_orig_idx = -1;
            if (tid < g_count_val) {
                double mi_unused;
                load4(&i_sorted_pos(my_sorted_idx, 0), xi, yi, zi, mi_unused);
                my_orig_idx = i_sorted_indices(my_sorted_idx);
            }

            double axi = 0, ayi = 0, azi = 0;

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
                load4(&j_cell_com(node, 0), ccx, ccy, ccz, cmass);
                double size_sq_th = j_cell_size_sq_theta(node);
                int is_leaf = j_cell_is_leaf(node);

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
                    }
                } else if (is_leaf) {
                    int lpbeg = j_cell_pbeg(node);
                    int lpend = j_cell_pend(node);
                    if (tid < g_count_val) {
                        for (int j = lpbeg; j < lpend; j++) {
                            double pjx, pjy, pjz, mj;
                            load4(&j_sorted_pos(j, 0), pjx, pjy, pjz, mj);
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
                    if (tid == 0) {
                        stack_top = current_stack_top;
                        for (int c = 0; c < 8; c++) {
                            int child = j_cell_children(node, c);
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

            // OVERWRITE accelerations for i-particles
            if (tid < g_count_val) {
                pax(my_orig_idx) = axi;
                pay(my_orig_idx) = ayi;
                paz(my_orig_idx) = azi;
            }
        });

}

// ==========================================================================
// Self-gradient correction via tree traversal
// Dehnen jerk formula: correction_i += G*m_j/r^5 * [r²*(a_j-a_i) - 3*(r·(a_j-a_i))*r] * coeff
// Monopole: a_j = cell CoM acceleration, P2P: a_j from per-particle data
// ==========================================================================
void tree_self_gradient_correction(Particles& p, const Octree& tree,
                                    double G, double soft2, double dt,
                                    int group_offset, int num_groups_param) {
    if (tree.n_particles <= 0 || tree.n_cells == 0) return;

    const int n_groups = (num_groups_param >= 0) ? num_groups_param : tree.n_groups;
    const double coeff = G * dt * dt * GRAD_COEFF;

    auto cell_com = tree.cell_com;
    auto cell_accel_v = tree.cell_accel;
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

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t stack_bytes = TREE_STACK_SIZE * sizeof(int);

    const int g_offset = group_offset;
    Kokkos::parallel_for("tree_self_grad",
        team_policy(n_groups, TREE_GROUP_SIZE)
            .set_scratch_size(0, Kokkos::PerTeam(stack_bytes)),
        KOKKOS_LAMBDA(const member_type& team) {
            int gid = g_offset + team.league_rank();
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
            double xi = 0, yi = 0, zi = 0;
            double axi = 0, ayi = 0, azi = 0;
            int my_orig_idx = -1;
            if (tid < g_count_val) {
                double mi_unused;
                load4(&sorted_pos(my_sorted_idx, 0), xi, yi, zi, mi_unused);
                my_orig_idx = sorted_indices(my_sorted_idx);
                axi = pax(my_orig_idx);
                ayi = pay(my_orig_idx);
                azi = paz(my_orig_idx);
            }

            double cx = 0, cy = 0, cz = 0;

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
                    // Monopole jerk: use cell CoM acceleration
                    if (tid < g_count_val) {
                        double dx = ccx - xi;
                        double dy = ccy - yi;
                        double dz = ccz - zi;
                        double dax = cell_accel_v(node, 0) - axi;
                        double day = cell_accel_v(node, 1) - ayi;
                        double daz = cell_accel_v(node, 2) - azi;

                        double r2 = dx * dx + dy * dy + dz * dz + soft2;
                        double inv_r = fast_rsqrt(r2);
                        double inv_r2 = inv_r * inv_r;
                        double inv_r5 = inv_r2 * inv_r2 * inv_r;

                        double r_dot_a = dx * dax + dy * day + dz * daz;
                        double f = coeff * cmass * inv_r5;

                        cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                        cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                        cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
                    }
                } else if (is_leaf) {
                    // P2P jerk: per-particle accelerations
                    int lpbeg = cell_pbeg(node);
                    int lpend = cell_pend(node);
                    if (tid < g_count_val) {
                        for (int j = lpbeg; j < lpend; j++) {
                            if (sorted_indices(j) == my_orig_idx) continue;
                            double pjx, pjy, pjz, mj;
                            load4(&sorted_pos(j, 0), pjx, pjy, pjz, mj);
                            int j_orig = sorted_indices(j);

                            double dx = pjx - xi;
                            double dy = pjy - yi;
                            double dz = pjz - zi;
                            double dax = pax(j_orig) - axi;
                            double day = pay(j_orig) - ayi;
                            double daz = paz(j_orig) - azi;

                            double r2 = dx * dx + dy * dy + dz * dz + soft2;
                            double inv_r = fast_rsqrt(r2);
                            double inv_r2 = inv_r * inv_r;
                            double inv_r5 = inv_r2 * inv_r2 * inv_r;

                            double r_dot_a = dx * dax + dy * day + dz * daz;
                            double f = coeff * mj * inv_r5;

                            cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                            cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                            cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
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

            // ADD gradient correction to existing accelerations
            if (tid < g_count_val) {
                pax(my_orig_idx) = axi + cx;
                pay(my_orig_idx) = ayi + cy;
                paz(my_orig_idx) = azi + cz;
            }
        });

}

// ==========================================================================
// Cross-gradient correction via tree: jerk on i-particles from j-tree
// i_tree provides groups, j_tree provides tree structure + cell_accel
// ==========================================================================
void tree_cross_gradient_correction(Particles& p,
                                     const Octree& i_tree,
                                     const Octree& j_tree,
                                     double G, double soft2, double dt) {
    if (i_tree.n_particles <= 0 || j_tree.n_cells == 0) return;

    const int n_groups = i_tree.n_groups;
    const double coeff = G * dt * dt * GRAD_COEFF;

    // i-particle data from i_tree
    auto i_sorted_pos = i_tree.sorted_pos;
    auto i_sorted_indices = i_tree.sorted_indices;
    auto i_grp_start = i_tree.group_start;
    auto i_grp_count = i_tree.group_count;
    auto i_grp_center = i_tree.group_center;
    auto i_grp_size = i_tree.group_size;

    // j-tree structure + cell accelerations
    auto j_cell_com = j_tree.cell_com;
    auto j_cell_accel = j_tree.cell_accel;
    auto j_cell_size_sq_theta = j_tree.cell_size_sq_theta;
    auto j_cell_children = j_tree.cell_children;
    auto j_cell_pbeg = j_tree.cell_pbeg;
    auto j_cell_pend = j_tree.cell_pend;
    auto j_cell_is_leaf = j_tree.cell_is_leaf;
    auto j_sorted_pos = j_tree.sorted_pos;
    auto j_sorted_indices = j_tree.sorted_indices;

    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t stack_bytes = TREE_STACK_SIZE * sizeof(int);

    Kokkos::parallel_for("tree_cross_grad",
        team_policy(n_groups, TREE_GROUP_SIZE)
            .set_scratch_size(0, Kokkos::PerTeam(stack_bytes)),
        KOKKOS_LAMBDA(const member_type& team) {
            int gid = team.league_rank();
            int tid = team.team_rank();

            Kokkos::View<int*, scratch_space, Kokkos::MemoryUnmanaged>
                stack(team.team_scratch(0), TREE_STACK_SIZE);

            int g_start = i_grp_start(gid);
            int g_count_val = i_grp_count(gid);
            double gcx = i_grp_center(gid, 0);
            double gcy = i_grp_center(gid, 1);
            double gcz = i_grp_center(gid, 2);
            double gsx = i_grp_size(gid, 0);
            double gsy = i_grp_size(gid, 1);
            double gsz = i_grp_size(gid, 2);

            int my_sorted_idx = g_start + tid;
            double xi = 0, yi = 0, zi = 0;
            double axi = 0, ayi = 0, azi = 0;
            int my_orig_idx = -1;
            if (tid < g_count_val) {
                double mi_unused;
                load4(&i_sorted_pos(my_sorted_idx, 0), xi, yi, zi, mi_unused);
                my_orig_idx = i_sorted_indices(my_sorted_idx);
                axi = pax(my_orig_idx);
                ayi = pay(my_orig_idx);
                azi = paz(my_orig_idx);
            }

            double cx = 0, cy = 0, cz = 0;

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
                load4(&j_cell_com(node, 0), ccx, ccy, ccz, cmass);
                double size_sq_th = j_cell_size_sq_theta(node);
                int is_leaf = j_cell_is_leaf(node);

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
                        double dax = j_cell_accel(node, 0) - axi;
                        double day = j_cell_accel(node, 1) - ayi;
                        double daz = j_cell_accel(node, 2) - azi;

                        double r2 = dx * dx + dy * dy + dz * dz + soft2;
                        double inv_r = fast_rsqrt(r2);
                        double inv_r2 = inv_r * inv_r;
                        double inv_r5 = inv_r2 * inv_r2 * inv_r;

                        double r_dot_a = dx * dax + dy * day + dz * daz;
                        double f = coeff * cmass * inv_r5;

                        cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                        cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                        cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
                    }
                } else if (is_leaf) {
                    int lpbeg = j_cell_pbeg(node);
                    int lpend = j_cell_pend(node);
                    if (tid < g_count_val) {
                        for (int j = lpbeg; j < lpend; j++) {
                            double pjx, pjy, pjz, mj;
                            load4(&j_sorted_pos(j, 0), pjx, pjy, pjz, mj);
                            int j_orig = j_sorted_indices(j);

                            double dx = pjx - xi;
                            double dy = pjy - yi;
                            double dz = pjz - zi;
                            double dax = pax(j_orig) - axi;
                            double day = pay(j_orig) - ayi;
                            double daz = paz(j_orig) - azi;

                            double r2 = dx * dx + dy * dy + dz * dz + soft2;
                            double inv_r = fast_rsqrt(r2);
                            double inv_r2 = inv_r * inv_r;
                            double inv_r5 = inv_r2 * inv_r2 * inv_r;

                            double r_dot_a = dx * dax + dy * day + dz * daz;
                            double f = coeff * mj * inv_r5;

                            cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                            cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                            cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
                        }
                    }
                } else {
                    if (tid == 0) {
                        stack_top = current_stack_top;
                        for (int c = 0; c < 8; c++) {
                            int child = j_cell_children(node, c);
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

            // ADD gradient correction to existing accelerations
            if (tid < g_count_val) {
                pax(my_orig_idx) += cx;
                pay(my_orig_idx) += cy;
                paz(my_orig_idx) += cz;
            }
        });

}

// ==========================================================================
// Global-tree cross-force: walks a SINGLE global tree with dual monopoles.
// source_flag: 0 = slow sources (target=fast), 1 = fast sources (target=slow)
// Target particles = those with flag != source_flag.
// Uses cell_com_slow/cell_com_fast for monopole approximation.
// P2P filters by sorted_subset_flag to only interact with source particles.
// Results OVERWRITE p.ax, p.ay, p.az for TARGET particles only.
// ==========================================================================
void tree_global_cross_acceleration(Particles& p, const Octree& tree,
                                     int source_flag, double G, double soft2) {
    if (tree.n_particles <= 0 || tree.n_cells == 0) return;

    const int n_groups = tree.n_groups;
    int target_flag = 1 - source_flag;

    auto cell_com_src = (source_flag == 0) ? tree.cell_com_slow : tree.cell_com_fast;
    auto cell_size_sq_theta = tree.cell_size_sq_theta;
    auto cell_children = tree.cell_children;
    auto cell_pbeg = tree.cell_pbeg;
    auto cell_pend = tree.cell_pend;
    auto cell_is_leaf = tree.cell_is_leaf;
    auto sorted_pos = tree.sorted_pos;
    auto sorted_indices = tree.sorted_indices;
    auto subset_flag = tree.sorted_subset_flag;
    auto grp_start = tree.group_start;
    auto grp_count = tree.group_count;
    auto grp_center = tree.group_center;
    auto grp_size = tree.group_size;

    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t stack_bytes = TREE_STACK_SIZE * sizeof(int);
    int tgt_flag = target_flag;
    int src_flag = source_flag;

    Kokkos::parallel_for("tree_global_cross",
        team_policy(n_groups, TREE_GROUP_SIZE)
            .set_scratch_size(0, Kokkos::PerTeam(stack_bytes)),
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
            double xi = 0, yi = 0, zi = 0;
            int my_orig_idx = -1;
            bool is_target = false;
            if (tid < g_count_val) {
                double mi_unused;
                load4(&sorted_pos(my_sorted_idx, 0), xi, yi, zi, mi_unused);
                my_orig_idx = sorted_indices(my_sorted_idx);
                is_target = (subset_flag(my_sorted_idx) == tgt_flag);
            }

            double axi = 0, ayi = 0, azi = 0;

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

                // Read SOURCE subset monopole
                double ccx, ccy, ccz, cmass;
                load4(&cell_com_src(node, 0), ccx, ccy, ccz, cmass);

                // Skip cells with zero source mass
                if (cmass <= 0.0) {
                    team.team_barrier();
                    if (tid == 0) stack(TREE_STACK_SIZE - 2) = current_stack_top;
                    team.team_barrier();
                    stack_top = stack(TREE_STACK_SIZE - 2);
                    continue;
                }

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
                    // Monopole approximation using source subset CoM
                    if (is_target) {
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
                    // P2P: only interact with SOURCE particles in leaf
                    if (is_target) {
                        int lpbeg = cell_pbeg(node);
                        int lpend = cell_pend(node);
                        for (int j = lpbeg; j < lpend; j++) {
                            if (subset_flag(j) != src_flag) continue;  // skip non-source
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

            // Write results for TARGET particles only
            if (is_target) {
                pax(my_orig_idx) = axi;
                pay(my_orig_idx) = ayi;
                paz(my_orig_idx) = azi;
            }
        });

}

// ==========================================================================
// Global-tree cross-gradient correction using dual monopoles.
// source_flag: 0 = slow sources, 1 = fast sources
// Uses cell_accel_slow/fast for monopole jerk, per-particle accel for P2P.
// Results ADDED to p.ax, p.ay, p.az for TARGET particles.
// ==========================================================================
void tree_global_cross_gradient_correction(Particles& p, const Octree& tree,
                                            int source_flag,
                                            double G, double soft2, double dt) {
    if (tree.n_particles <= 0 || tree.n_cells == 0) return;

    const int n_groups = tree.n_groups;
    const double coeff = G * dt * dt * GRAD_COEFF;
    int target_flag = 1 - source_flag;

    auto cell_com_src = (source_flag == 0) ? tree.cell_com_slow : tree.cell_com_fast;
    auto cell_accel_src = (source_flag == 0) ? tree.cell_accel_slow : tree.cell_accel_fast;
    auto cell_size_sq_theta = tree.cell_size_sq_theta;
    auto cell_children = tree.cell_children;
    auto cell_pbeg = tree.cell_pbeg;
    auto cell_pend = tree.cell_pend;
    auto cell_is_leaf = tree.cell_is_leaf;
    auto sorted_pos = tree.sorted_pos;
    auto sorted_indices = tree.sorted_indices;
    auto subset_flag = tree.sorted_subset_flag;
    auto grp_start = tree.group_start;
    auto grp_count = tree.group_count;
    auto grp_center = tree.group_center;
    auto grp_size = tree.group_size;

    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t stack_bytes = TREE_STACK_SIZE * sizeof(int);
    int tgt_flag = target_flag;
    int src_flag = source_flag;

    Kokkos::parallel_for("tree_global_cross_grad",
        team_policy(n_groups, TREE_GROUP_SIZE)
            .set_scratch_size(0, Kokkos::PerTeam(stack_bytes)),
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
            double xi = 0, yi = 0, zi = 0;
            double axi = 0, ayi = 0, azi = 0;
            int my_orig_idx = -1;
            bool is_target = false;
            if (tid < g_count_val) {
                double mi_unused;
                load4(&sorted_pos(my_sorted_idx, 0), xi, yi, zi, mi_unused);
                my_orig_idx = sorted_indices(my_sorted_idx);
                is_target = (subset_flag(my_sorted_idx) == tgt_flag);
                if (is_target) {
                    axi = pax(my_orig_idx);
                    ayi = pay(my_orig_idx);
                    azi = paz(my_orig_idx);
                }
            }

            double cx = 0, cy = 0, cz = 0;

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
                load4(&cell_com_src(node, 0), ccx, ccy, ccz, cmass);

                if (cmass <= 0.0) {
                    team.team_barrier();
                    if (tid == 0) stack(TREE_STACK_SIZE - 2) = current_stack_top;
                    team.team_barrier();
                    stack_top = stack(TREE_STACK_SIZE - 2);
                    continue;
                }

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
                    if (is_target) {
                        double dx = ccx - xi;
                        double dy = ccy - yi;
                        double dz = ccz - zi;
                        double dax = cell_accel_src(node, 0) - axi;
                        double day = cell_accel_src(node, 1) - ayi;
                        double daz = cell_accel_src(node, 2) - azi;

                        double r2 = dx * dx + dy * dy + dz * dz + soft2;
                        double inv_r = fast_rsqrt(r2);
                        double inv_r2 = inv_r * inv_r;
                        double inv_r5 = inv_r2 * inv_r2 * inv_r;

                        double r_dot_a = dx * dax + dy * day + dz * daz;
                        double f = coeff * cmass * inv_r5;

                        cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                        cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                        cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
                    }
                } else if (is_leaf) {
                    if (is_target) {
                        int lpbeg = cell_pbeg(node);
                        int lpend = cell_pend(node);
                        for (int j = lpbeg; j < lpend; j++) {
                            if (subset_flag(j) != src_flag) continue;
                            double pjx, pjy, pjz, mj;
                            load4(&sorted_pos(j, 0), pjx, pjy, pjz, mj);
                            int j_orig = sorted_indices(j);

                            double dx = pjx - xi;
                            double dy = pjy - yi;
                            double dz = pjz - zi;
                            double dax = pax(j_orig) - axi;
                            double day = pay(j_orig) - ayi;
                            double daz = paz(j_orig) - azi;

                            double r2 = dx * dx + dy * dy + dz * dz + soft2;
                            double inv_r = fast_rsqrt(r2);
                            double inv_r2 = inv_r * inv_r;
                            double inv_r5 = inv_r2 * inv_r2 * inv_r;

                            double r_dot_a = dx * dax + dy * day + dz * daz;
                            double f = coeff * mj * inv_r5;

                            cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                            cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                            cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
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

            // ADD gradient correction to existing accelerations for targets
            if (is_target) {
                pax(my_orig_idx) += cx;
                pay(my_orig_idx) += cy;
                paz(my_orig_idx) += cz;
            }
        });

}

// ==========================================================================
// Global-tree self-force: walks a SINGLE global tree with dual monopoles.
// subset_flag: 0 = slow subset, 1 = fast subset
// Both targets AND sources are the SAME subset (self-interaction).
// Uses cell_com_slow/cell_com_fast for monopole approximation.
// P2P filters by sorted_subset_flag + self-skip.
// Results OVERWRITE p.ax, p.ay, p.az for subset particles only.
// ==========================================================================
void tree_global_self_acceleration(Particles& p, const Octree& tree,
                                    int subset_flag, double G, double soft2) {
    if (tree.n_particles <= 0 || tree.n_cells == 0) return;

    const int n_groups = tree.n_groups;

    auto cell_com_src = (subset_flag == 0) ? tree.cell_com_slow : tree.cell_com_fast;
    auto cell_size_sq_theta = tree.cell_size_sq_theta;
    auto cell_children = tree.cell_children;
    auto cell_pbeg = tree.cell_pbeg;
    auto cell_pend = tree.cell_pend;
    auto cell_is_leaf = tree.cell_is_leaf;
    auto sorted_pos = tree.sorted_pos;
    auto sorted_indices = tree.sorted_indices;
    auto sflag = tree.sorted_subset_flag;
    auto grp_start = tree.group_start;
    auto grp_count = tree.group_count;
    auto grp_center = tree.group_center;
    auto grp_size = tree.group_size;

    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t stack_bytes = TREE_STACK_SIZE * sizeof(int);
    int sub_flag = subset_flag;

    Kokkos::parallel_for("tree_global_self",
        team_policy(n_groups, TREE_GROUP_SIZE)
            .set_scratch_size(0, Kokkos::PerTeam(stack_bytes)),
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
            double xi = 0, yi = 0, zi = 0;
            int my_orig_idx = -1;
            bool is_target = false;
            if (tid < g_count_val) {
                double mi_unused;
                load4(&sorted_pos(my_sorted_idx, 0), xi, yi, zi, mi_unused);
                my_orig_idx = sorted_indices(my_sorted_idx);
                is_target = (sflag(my_sorted_idx) == sub_flag);
            }

            double axi = 0, ayi = 0, azi = 0;

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
                load4(&cell_com_src(node, 0), ccx, ccy, ccz, cmass);

                if (cmass <= 0.0) {
                    team.team_barrier();
                    if (tid == 0) stack(TREE_STACK_SIZE - 2) = current_stack_top;
                    team.team_barrier();
                    stack_top = stack(TREE_STACK_SIZE - 2);
                    continue;
                }

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
                    if (is_target) {
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
                    if (is_target) {
                        int lpbeg = cell_pbeg(node);
                        int lpend = cell_pend(node);
                        for (int j = lpbeg; j < lpend; j++) {
                            if (sflag(j) != sub_flag) continue;  // skip non-subset
                            if (j == my_sorted_idx) continue;     // self-skip
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

            if (is_target) {
                pax(my_orig_idx) = axi;
                pay(my_orig_idx) = ayi;
                paz(my_orig_idx) = azi;
            }
        });

}

// ==========================================================================
// Global-tree self-gradient correction using dual monopoles.
// subset_flag: 0 = slow subset, 1 = fast subset
// Uses cell_accel_slow/fast for monopole jerk, per-particle accel for P2P.
// Results ADDED to p.ax, p.ay, p.az for subset particles.
// ==========================================================================
void tree_global_self_gradient_correction(Particles& p, const Octree& tree,
                                           int subset_flag,
                                           double G, double soft2, double dt) {
    if (tree.n_particles <= 0 || tree.n_cells == 0) return;

    const int n_groups = tree.n_groups;
    const double coeff = G * dt * dt * GRAD_COEFF;

    auto cell_com_src = (subset_flag == 0) ? tree.cell_com_slow : tree.cell_com_fast;
    auto cell_accel_src = (subset_flag == 0) ? tree.cell_accel_slow : tree.cell_accel_fast;
    auto cell_size_sq_theta = tree.cell_size_sq_theta;
    auto cell_children = tree.cell_children;
    auto cell_pbeg = tree.cell_pbeg;
    auto cell_pend = tree.cell_pend;
    auto cell_is_leaf = tree.cell_is_leaf;
    auto sorted_pos = tree.sorted_pos;
    auto sorted_indices = tree.sorted_indices;
    auto sflag = tree.sorted_subset_flag;
    auto grp_start = tree.group_start;
    auto grp_count = tree.group_count;
    auto grp_center = tree.group_center;
    auto grp_size = tree.group_size;

    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t stack_bytes = TREE_STACK_SIZE * sizeof(int);
    int sub_flag = subset_flag;

    Kokkos::parallel_for("tree_global_self_grad",
        team_policy(n_groups, TREE_GROUP_SIZE)
            .set_scratch_size(0, Kokkos::PerTeam(stack_bytes)),
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
            double xi = 0, yi = 0, zi = 0;
            double axi = 0, ayi = 0, azi = 0;
            int my_orig_idx = -1;
            bool is_target = false;
            if (tid < g_count_val) {
                double mi_unused;
                load4(&sorted_pos(my_sorted_idx, 0), xi, yi, zi, mi_unused);
                my_orig_idx = sorted_indices(my_sorted_idx);
                is_target = (sflag(my_sorted_idx) == sub_flag);
                if (is_target) {
                    axi = pax(my_orig_idx);
                    ayi = pay(my_orig_idx);
                    azi = paz(my_orig_idx);
                }
            }

            double cx = 0, cy = 0, cz = 0;

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
                load4(&cell_com_src(node, 0), ccx, ccy, ccz, cmass);

                if (cmass <= 0.0) {
                    team.team_barrier();
                    if (tid == 0) stack(TREE_STACK_SIZE - 2) = current_stack_top;
                    team.team_barrier();
                    stack_top = stack(TREE_STACK_SIZE - 2);
                    continue;
                }

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
                    if (is_target) {
                        double dx = ccx - xi;
                        double dy = ccy - yi;
                        double dz = ccz - zi;
                        double dax = cell_accel_src(node, 0) - axi;
                        double day = cell_accel_src(node, 1) - ayi;
                        double daz = cell_accel_src(node, 2) - azi;

                        double r2 = dx * dx + dy * dy + dz * dz + soft2;
                        double inv_r = fast_rsqrt(r2);
                        double inv_r2 = inv_r * inv_r;
                        double inv_r5 = inv_r2 * inv_r2 * inv_r;

                        double r_dot_a = dx * dax + dy * day + dz * daz;
                        double f = coeff * cmass * inv_r5;

                        cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                        cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                        cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
                    }
                } else if (is_leaf) {
                    if (is_target) {
                        int lpbeg = cell_pbeg(node);
                        int lpend = cell_pend(node);
                        for (int j = lpbeg; j < lpend; j++) {
                            if (sflag(j) != sub_flag) continue;
                            if (j == my_sorted_idx) continue;  // self-skip
                            double pjx, pjy, pjz, mj;
                            load4(&sorted_pos(j, 0), pjx, pjy, pjz, mj);
                            int j_orig = sorted_indices(j);

                            double dx = pjx - xi;
                            double dy = pjy - yi;
                            double dz = pjz - zi;
                            double dax = pax(j_orig) - axi;
                            double day = pay(j_orig) - ayi;
                            double daz = paz(j_orig) - azi;

                            double r2 = dx * dx + dy * dy + dz * dz + soft2;
                            double inv_r = fast_rsqrt(r2);
                            double inv_r2 = inv_r * inv_r;
                            double inv_r5 = inv_r2 * inv_r2 * inv_r;

                            double r_dot_a = dx * dax + dy * day + dz * daz;
                            double f = coeff * mj * inv_r5;

                            cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                            cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                            cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
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

            if (is_target) {
                pax(my_orig_idx) += cx;
                pay(my_orig_idx) += cy;
                paz(my_orig_idx) += cz;
            }
        });

}

// ==========================================================================
// Global-tree fused self-force + timestep for a subset.
// Uses dual monopoles (cell_com_slow/fast) and dual velocities (cell_vel_slow/fast).
// P2P filters by sorted_subset_flag + self-skip.
// Results OVERWRITE p.ax, p.ay, p.az and p.dt_adapt for subset particles.
// ==========================================================================
void tree_global_self_acceleration_and_timestep(Particles& p, const Octree& tree,
                                                 int subset_flag,
                                                 double G, double soft2, double eta) {
    if (tree.n_particles <= 0 || tree.n_cells == 0) return;

    const int n_groups = tree.n_groups;

    auto cell_com_src = (subset_flag == 0) ? tree.cell_com_slow : tree.cell_com_fast;
    auto cell_vel_src = (subset_flag == 0) ? tree.cell_vel_slow : tree.cell_vel_fast;
    auto cell_size_sq_theta = tree.cell_size_sq_theta;
    auto cell_children = tree.cell_children;
    auto cell_pbeg = tree.cell_pbeg;
    auto cell_pend = tree.cell_pend;
    auto cell_is_leaf = tree.cell_is_leaf;
    auto sorted_pos = tree.sorted_pos;
    auto sorted_indices = tree.sorted_indices;
    auto sflag = tree.sorted_subset_flag;
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
    int sub_flag = subset_flag;

    Kokkos::parallel_for("tree_global_self_dt",
        team_policy(n_groups, TREE_GROUP_SIZE)
            .set_scratch_size(0, Kokkos::PerTeam(stack_bytes)),
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
            bool is_target = false;
            if (tid < g_count_val) {
                load4(&sorted_pos(my_sorted_idx, 0), xi, yi, zi, mi);
                my_orig_idx = sorted_indices(my_sorted_idx);
                is_target = (sflag(my_sorted_idx) == sub_flag);
                if (is_target) {
                    vxi = pvx(my_orig_idx);
                    vyi = pvy(my_orig_idx);
                    vzi = pvz(my_orig_idx);
                }
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
                load4(&cell_com_src(node, 0), ccx, ccy, ccz, cmass);

                if (cmass <= 0.0) {
                    team.team_barrier();
                    if (tid == 0) stack(TREE_STACK_SIZE - 2) = current_stack_top;
                    team.team_barrier();
                    stack_top = stack(TREE_STACK_SIZE - 2);
                    continue;
                }

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
                    if (is_target) {
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

                        // Symmetrized monopole timestep
                        double dvx = cell_vel_src(node, 0) - vxi;
                        double dvy = cell_vel_src(node, 1) - vyi;
                        double dvz = cell_vel_src(node, 2) - vzi;
                        double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
                        double mass_sum = mi + cmass;
                        double inv_Gm = 1.0 / (G * mass_sum + EPS);
                        double inv_vsq = 1.0 / (v_sq + EPS);

                        double r3 = r2 * r2 * inv_r;
                        double t_ff_sq = r3 * inv_Gm;
                        double t_fb_sq = r2 * inv_vsq;
                        double t_ff = t_ff_sq * fast_rsqrt(t_ff_sq + EPS);
                        double t_fb = t_fb_sq * fast_rsqrt(t_fb_sq + EPS);

                        double inv_r2 = inv_r * inv_r;
                        double r_dot_v = dx*dvx + dy*dvy + dz*dvz;

                        double c_ff = 1.5 * r_dot_v * inv_r2;
                        double c_fb = r_dot_v * inv_r2 * (1.0 + G * mass_sum * inv_r * inv_vsq);

                        double ct_ff = c_ff * t_ff_sq;
                        double s_ff = 1.0 + c_ff * ct_ff;
                        double u_ff = s_ff * fast_rsqrt(s_ff);
                        double t_sym_ff = t_ff + ct_ff / (u_ff + 1.0);

                        double ct_fb = c_fb * t_fb_sq;
                        double s_fb = 1.0 + c_fb * ct_fb;
                        double u_fb = s_fb * fast_rsqrt(s_fb);
                        double t_sym_fb = t_fb + ct_fb / (u_fb + 1.0);

                        t_min_sq = Kokkos::fmin(t_min_sq,
                                   Kokkos::fmin(t_sym_ff * t_sym_ff, t_sym_fb * t_sym_fb));
                    }
                } else if (is_leaf) {
                    if (is_target) {
                        int lpbeg = cell_pbeg(node);
                        int lpend = cell_pend(node);
                        for (int j = lpbeg; j < lpend; j++) {
                            if (sflag(j) != sub_flag) continue;
                            if (j == my_sorted_idx) continue;
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

                            // Symmetrized pairwise timestep
                            int j_orig = sorted_indices(j);
                            double dvx = pvx(j_orig) - vxi;
                            double dvy = pvy(j_orig) - vyi;
                            double dvz = pvz(j_orig) - vzi;
                            double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
                            double mass_sum = mi + mj;
                            double inv_Gm = 1.0 / (G * mass_sum + EPS);
                            double inv_vsq = 1.0 / (v_sq + EPS);

                            double r3 = r2 * r2 * inv_r;
                            double t_ff_sq = r3 * inv_Gm;
                            double t_fb_sq = r2 * inv_vsq;
                            double t_ff = t_ff_sq * fast_rsqrt(t_ff_sq + EPS);
                            double t_fb = t_fb_sq * fast_rsqrt(t_fb_sq + EPS);

                            double inv_r2 = inv_r * inv_r;
                            double r_dot_v = dx*dvx + dy*dvy + dz*dvz;

                            double c_ff = 1.5 * r_dot_v * inv_r2;
                            double c_fb = r_dot_v * inv_r2 * (1.0 + G * mass_sum * inv_r * inv_vsq);

                            double ct_ff = c_ff * t_ff_sq;
                            double s_ff = 1.0 + c_ff * ct_ff;
                            double u_ff = s_ff * fast_rsqrt(s_ff);
                            double t_sym_ff = t_ff + ct_ff / (u_ff + 1.0);

                            double ct_fb = c_fb * t_fb_sq;
                            double s_fb = 1.0 + c_fb * ct_fb;
                            double u_fb = s_fb * fast_rsqrt(s_fb);
                            double t_sym_fb = t_fb + ct_fb / (u_fb + 1.0);

                            t_min_sq = Kokkos::fmin(t_min_sq,
                                       Kokkos::fmin(t_sym_ff * t_sym_ff, t_sym_fb * t_sym_fb));
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

            if (is_target) {
                pax(my_orig_idx) = axi;
                pay(my_orig_idx) = ayi;
                paz(my_orig_idx) = azi;
                dt_f(my_orig_idx) = Kokkos::sqrt(t_min_sq) * eta;
            }
        });

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

// ==========================================================================
// GADGET-4 cross-force decomposition: F_all traversal functions.
// Compute forces from ALL particles using total cell_com monopoles.
// Targets: particles with sflag != 2 (both slow and fast at current level).
// P2P: all sorted particles (no flag filter), only self-skip.
// ==========================================================================

// Force from ALL particles on active subset (sflag != 2).
void tree_all_forces_on_subset(Particles& p, const Octree& tree,
                                double G, double soft2) {
    if (tree.n_particles <= 0 || tree.n_cells == 0) return;

    const int n_groups = tree.n_groups;

    auto cell_com_src = tree.cell_com;  // TOTAL monopoles (not slow/fast)
    auto cell_size_sq_theta = tree.cell_size_sq_theta;
    auto cell_children = tree.cell_children;
    auto cell_pbeg = tree.cell_pbeg;
    auto cell_pend = tree.cell_pend;
    auto cell_is_leaf = tree.cell_is_leaf;
    auto sorted_pos = tree.sorted_pos;
    auto sorted_indices = tree.sorted_indices;
    auto sflag = tree.sorted_subset_flag;
    auto grp_start = tree.group_start;
    auto grp_count = tree.group_count;
    auto grp_center = tree.group_center;
    auto grp_size = tree.group_size;

    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t stack_bytes = TREE_STACK_SIZE * sizeof(int);

    Kokkos::parallel_for("tree_all_forces",
        team_policy(n_groups, TREE_GROUP_SIZE)
            .set_scratch_size(0, Kokkos::PerTeam(stack_bytes)),
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
            double xi = 0, yi = 0, zi = 0;
            int my_orig_idx = -1;
            bool is_target = false;
            if (tid < g_count_val) {
                double mi_unused;
                load4(&sorted_pos(my_sorted_idx, 0), xi, yi, zi, mi_unused);
                my_orig_idx = sorted_indices(my_sorted_idx);
                is_target = (sflag(my_sorted_idx) != 2);  // active = slow or fast
            }

            double axi = 0, ayi = 0, azi = 0;

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
                load4(&cell_com_src(node, 0), ccx, ccy, ccz, cmass);

                if (cmass <= 0.0) {
                    team.team_barrier();
                    if (tid == 0) stack(TREE_STACK_SIZE - 2) = current_stack_top;
                    team.team_barrier();
                    stack_top = stack(TREE_STACK_SIZE - 2);
                    continue;
                }

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
                    if (is_target) {
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
                    if (is_target) {
                        int lpbeg = cell_pbeg(node);
                        int lpend = cell_pend(node);
                        for (int j = lpbeg; j < lpend; j++) {
                            // No flag filter — ALL particles are sources
                            if (j == my_sorted_idx) continue;  // self-skip only
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

            if (is_target) {
                pax(my_orig_idx) = axi;
                pay(my_orig_idx) = ayi;
                paz(my_orig_idx) = azi;
            }
        });
}

// Gradient correction from ALL particles on active subset (sflag != 2).
// Uses cell_com + cell_accel (total). Results ADDED to p.ax, p.ay, p.az.
void tree_all_gradient_on_subset(Particles& p, const Octree& tree,
                                  double G, double soft2, double dt) {
    if (tree.n_particles <= 0 || tree.n_cells == 0) return;

    const int n_groups = tree.n_groups;
    const double coeff = G * dt * dt * GRAD_COEFF;

    auto cell_com_src = tree.cell_com;          // TOTAL monopoles
    auto cell_accel_src = tree.cell_accel;      // TOTAL cell accelerations
    auto cell_size_sq_theta = tree.cell_size_sq_theta;
    auto cell_children = tree.cell_children;
    auto cell_pbeg = tree.cell_pbeg;
    auto cell_pend = tree.cell_pend;
    auto cell_is_leaf = tree.cell_is_leaf;
    auto sorted_pos = tree.sorted_pos;
    auto sorted_indices = tree.sorted_indices;
    auto sflag = tree.sorted_subset_flag;
    auto grp_start = tree.group_start;
    auto grp_count = tree.group_count;
    auto grp_center = tree.group_center;
    auto grp_size = tree.group_size;

    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t stack_bytes = TREE_STACK_SIZE * sizeof(int);

    Kokkos::parallel_for("tree_all_grad",
        team_policy(n_groups, TREE_GROUP_SIZE)
            .set_scratch_size(0, Kokkos::PerTeam(stack_bytes)),
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
            double xi = 0, yi = 0, zi = 0;
            double axi = 0, ayi = 0, azi = 0;
            int my_orig_idx = -1;
            bool is_target = false;
            if (tid < g_count_val) {
                double mi_unused;
                load4(&sorted_pos(my_sorted_idx, 0), xi, yi, zi, mi_unused);
                my_orig_idx = sorted_indices(my_sorted_idx);
                is_target = (sflag(my_sorted_idx) != 2);
                if (is_target) {
                    axi = pax(my_orig_idx);
                    ayi = pay(my_orig_idx);
                    azi = paz(my_orig_idx);
                }
            }

            double cx = 0, cy = 0, cz = 0;

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
                load4(&cell_com_src(node, 0), ccx, ccy, ccz, cmass);

                if (cmass <= 0.0) {
                    team.team_barrier();
                    if (tid == 0) stack(TREE_STACK_SIZE - 2) = current_stack_top;
                    team.team_barrier();
                    stack_top = stack(TREE_STACK_SIZE - 2);
                    continue;
                }

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
                    if (is_target) {
                        double dx = ccx - xi;
                        double dy = ccy - yi;
                        double dz = ccz - zi;
                        double dax = cell_accel_src(node, 0) - axi;
                        double day = cell_accel_src(node, 1) - ayi;
                        double daz = cell_accel_src(node, 2) - azi;

                        double r2 = dx * dx + dy * dy + dz * dz + soft2;
                        double inv_r = fast_rsqrt(r2);
                        double inv_r2 = inv_r * inv_r;
                        double inv_r5 = inv_r2 * inv_r2 * inv_r;

                        double r_dot_a = dx * dax + dy * day + dz * daz;
                        double f = coeff * cmass * inv_r5;

                        cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                        cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                        cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
                    }
                } else if (is_leaf) {
                    if (is_target) {
                        int lpbeg = cell_pbeg(node);
                        int lpend = cell_pend(node);
                        for (int j = lpbeg; j < lpend; j++) {
                            // No flag filter — ALL particles are sources
                            if (j == my_sorted_idx) continue;  // self-skip only
                            double pjx, pjy, pjz, mj;
                            load4(&sorted_pos(j, 0), pjx, pjy, pjz, mj);
                            int j_orig = sorted_indices(j);

                            double dx = pjx - xi;
                            double dy = pjy - yi;
                            double dz = pjz - zi;
                            double dax = pax(j_orig) - axi;
                            double day = pay(j_orig) - ayi;
                            double daz = paz(j_orig) - azi;

                            double r2 = dx * dx + dy * dy + dz * dz + soft2;
                            double inv_r = fast_rsqrt(r2);
                            double inv_r2 = inv_r * inv_r;
                            double inv_r5 = inv_r2 * inv_r2 * inv_r;

                            double r_dot_a = dx * dax + dy * day + dz * daz;
                            double f = coeff * mj * inv_r5;

                            cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                            cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                            cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
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

            if (is_target) {
                pax(my_orig_idx) += cx;
                pay(my_orig_idx) += cy;
                paz(my_orig_idx) += cz;
            }
        });
}

// Fused force + timestep from ALL particles on active subset (sflag != 2).
// Uses cell_com + cell_vel (total). Results OVERWRITE p.ax, p.ay, p.az, p.dt_adapt.
void tree_all_forces_and_timestep_on_subset(Particles& p, const Octree& tree,
                                             double G, double soft2, double eta) {
    if (tree.n_particles <= 0 || tree.n_cells == 0) return;

    const int n_groups = tree.n_groups;

    auto cell_com_src = tree.cell_com;      // TOTAL monopoles
    auto cell_vel_src = tree.cell_vel;      // TOTAL cell velocities
    auto cell_size_sq_theta = tree.cell_size_sq_theta;
    auto cell_children = tree.cell_children;
    auto cell_pbeg = tree.cell_pbeg;
    auto cell_pend = tree.cell_pend;
    auto cell_is_leaf = tree.cell_is_leaf;
    auto sorted_pos = tree.sorted_pos;
    auto sorted_indices = tree.sorted_indices;
    auto sflag = tree.sorted_subset_flag;
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

    Kokkos::parallel_for("tree_all_forces_dt",
        team_policy(n_groups, TREE_GROUP_SIZE)
            .set_scratch_size(0, Kokkos::PerTeam(stack_bytes)),
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
            bool is_target = false;
            if (tid < g_count_val) {
                load4(&sorted_pos(my_sorted_idx, 0), xi, yi, zi, mi);
                my_orig_idx = sorted_indices(my_sorted_idx);
                is_target = (sflag(my_sorted_idx) != 2);
                if (is_target) {
                    vxi = pvx(my_orig_idx);
                    vyi = pvy(my_orig_idx);
                    vzi = pvz(my_orig_idx);
                }
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
                load4(&cell_com_src(node, 0), ccx, ccy, ccz, cmass);

                if (cmass <= 0.0) {
                    team.team_barrier();
                    if (tid == 0) stack(TREE_STACK_SIZE - 2) = current_stack_top;
                    team.team_barrier();
                    stack_top = stack(TREE_STACK_SIZE - 2);
                    continue;
                }

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
                    if (is_target) {
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

                        // Symmetrized monopole timestep
                        double dvx = cell_vel_src(node, 0) - vxi;
                        double dvy = cell_vel_src(node, 1) - vyi;
                        double dvz = cell_vel_src(node, 2) - vzi;
                        double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
                        double mass_sum = mi + cmass;
                        double inv_Gm = 1.0 / (G * mass_sum + EPS);
                        double inv_vsq = 1.0 / (v_sq + EPS);

                        double r3 = r2 * r2 * inv_r;
                        double t_ff_sq = r3 * inv_Gm;
                        double t_fb_sq = r2 * inv_vsq;
                        double t_ff = t_ff_sq * fast_rsqrt(t_ff_sq + EPS);
                        double t_fb = t_fb_sq * fast_rsqrt(t_fb_sq + EPS);

                        double inv_r2 = inv_r * inv_r;
                        double r_dot_v = dx*dvx + dy*dvy + dz*dvz;

                        double c_ff = 1.5 * r_dot_v * inv_r2;
                        double c_fb = r_dot_v * inv_r2 * (1.0 + G * mass_sum * inv_r * inv_vsq);

                        double ct_ff = c_ff * t_ff_sq;
                        double s_ff = 1.0 + c_ff * ct_ff;
                        double u_ff = s_ff * fast_rsqrt(s_ff);
                        double t_sym_ff = t_ff + ct_ff / (u_ff + 1.0);

                        double ct_fb = c_fb * t_fb_sq;
                        double s_fb = 1.0 + c_fb * ct_fb;
                        double u_fb = s_fb * fast_rsqrt(s_fb);
                        double t_sym_fb = t_fb + ct_fb / (u_fb + 1.0);

                        t_min_sq = Kokkos::fmin(t_min_sq,
                                   Kokkos::fmin(t_sym_ff * t_sym_ff, t_sym_fb * t_sym_fb));
                    }
                } else if (is_leaf) {
                    if (is_target) {
                        int lpbeg = cell_pbeg(node);
                        int lpend = cell_pend(node);
                        for (int j = lpbeg; j < lpend; j++) {
                            // No flag filter — ALL particles are sources
                            if (j == my_sorted_idx) continue;
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

                            // Symmetrized pairwise timestep
                            int j_orig = sorted_indices(j);
                            double dvx = pvx(j_orig) - vxi;
                            double dvy = pvy(j_orig) - vyi;
                            double dvz = pvz(j_orig) - vzi;
                            double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
                            double mass_sum = mi + mj;
                            double inv_Gm = 1.0 / (G * mass_sum + EPS);
                            double inv_vsq = 1.0 / (v_sq + EPS);

                            double r3 = r2 * r2 * inv_r;
                            double t_ff_sq = r3 * inv_Gm;
                            double t_fb_sq = r2 * inv_vsq;
                            double t_ff = t_ff_sq * fast_rsqrt(t_ff_sq + EPS);
                            double t_fb = t_fb_sq * fast_rsqrt(t_fb_sq + EPS);

                            double inv_r2 = inv_r * inv_r;
                            double r_dot_v = dx*dvx + dy*dvy + dz*dvz;

                            double c_ff = 1.5 * r_dot_v * inv_r2;
                            double c_fb = r_dot_v * inv_r2 * (1.0 + G * mass_sum * inv_r * inv_vsq);

                            double ct_ff = c_ff * t_ff_sq;
                            double s_ff = 1.0 + c_ff * ct_ff;
                            double u_ff = s_ff * fast_rsqrt(s_ff);
                            double t_sym_ff = t_ff + ct_ff / (u_ff + 1.0);

                            double ct_fb = c_fb * t_fb_sq;
                            double s_fb = 1.0 + c_fb * ct_fb;
                            double u_fb = s_fb * fast_rsqrt(s_fb);
                            double t_sym_fb = t_fb + ct_fb / (u_fb + 1.0);

                            t_min_sq = Kokkos::fmin(t_min_sq,
                                       Kokkos::fmin(t_sym_ff * t_sym_ff, t_sym_fb * t_sym_fb));
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

            if (is_target) {
                pax(my_orig_idx) = axi;
                pay(my_orig_idx) = ayi;
                paz(my_orig_idx) = azi;
                dt_f(my_orig_idx) = Kokkos::sqrt(t_min_sq) * eta;
            }
        });
}
