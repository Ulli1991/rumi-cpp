#include "tree.h"
#include "integrator.h"
#include "constants.h"
#ifdef ENABLE_COLLISIONAL
#include "collisional_config.h"
#endif
#include <Kokkos_Core.hpp>
#include <cfloat>

// ==========================================================================
// Barnes-Hut tree traversal kernels (GPU)
//
// Group-based traversal following Bonsai2:
// - Each Kokkos team handles one particle group (TREE_GROUP_SIZE particles)
// - MAC evaluated against GROUP bounding box (all threads take same branch)
// - Shared-memory stack for depth-first traversal
// - Direct P2P at leaf cells, monopole approximation for distant cells
//
// All 13 traversal functions are generated from one template via Op/Filt
// compile-time parameters. The compiler eliminates dead branches.
// ==========================================================================

// Operation type (what to compute at each interaction)
enum { OP_FORCE = 0, OP_GRADIENT = 1, OP_FORCE_DT = 2 };

// Particle filter mode (which particles are active, P2P skip rules)
//   FILT_ALL:      all threads active, self-skip by original index
//   FILT_DISJOINT: all threads active, no P2P skip (cross-tree)
//   FILT_SELF:     subset flag match, P2P flag+self-skip (global self)
//   FILT_CROSS:    subset flag match, P2P source flag filter (global cross)
//   FILT_ACTIVE:   sflag != 2, P2P self-skip only (GADGET-4 all-particle)
enum { FILT_ALL = 0, FILT_DISJOINT = 1, FILT_SELF = 2, FILT_CROSS = 3, FILT_ACTIVE = 4 };

// All Views and scalars needed by the tree walk template.
// Unused fields for a given Op/Filt combination are default-constructed
// (null Views) and optimized away by the compiler.
struct TreeWalkData {
    // Cell monopole (caller selects: total, slow, or fast)
    Kokkos::View<double*[4], Kokkos::LayoutRight> cell_com;
    // Tree structure
    Kokkos::View<double*> cell_size_sq_theta;
    Kokkos::View<int*[8], Kokkos::LayoutRight> cell_children;
    Kokkos::View<int*> cell_pbeg, cell_pend, cell_is_leaf;
    // I-particle sorted data (for loading target positions + original indices)
    Kokkos::View<double*[4], Kokkos::LayoutRight> sorted_pos;
    Kokkos::View<int*> sorted_indices;
    // J-particle sorted data (for P2P sources; same as above unless cross-tree)
    Kokkos::View<double*[4], Kokkos::LayoutRight> j_sorted_pos;
    Kokkos::View<int*> j_sorted_indices;
    // Groups
    Kokkos::View<int*> grp_start, grp_count;
    Kokkos::View<double*[3]> grp_center, grp_size;
    // Particle acceleration arrays (read for gradient, write for all)
    RawView pax, pay, paz;
    // Gradient: cell acceleration monopole
    Kokkos::View<double*[3], Kokkos::LayoutRight> cell_accel;
    // Timestep: particle velocity/mass/dt + cell velocity monopole
    RawView pvx, pvy, pvz, pm, dt_f;
    Kokkos::View<double*[3], Kokkos::LayoutRight> cell_vel;
    // Subset flags (sorted order)
    Kokkos::View<int*> subset_flag;
    // Scalars
    double G, soft2, coeff;
    double eta_ff_sq, eta_fb_sq;
    int match_flag, src_flag;
    int g_offset, n_groups;
};

// Fill common fields shared by all single-tree functions.
static TreeWalkData make_base(Particles& p, const Octree& tree,
                               double G, double soft2, int g_off, int ng) {
    TreeWalkData d{};
    d.cell_size_sq_theta = tree.cell_size_sq_theta;
    d.cell_children = tree.cell_children;
    d.cell_pbeg = tree.cell_pbeg;
    d.cell_pend = tree.cell_pend;
    d.cell_is_leaf = tree.cell_is_leaf;
    d.sorted_pos = tree.sorted_pos;
    d.sorted_indices = tree.sorted_indices;
    d.j_sorted_pos = tree.sorted_pos;      // same tree for P2P
    d.j_sorted_indices = tree.sorted_indices;
    d.grp_start = tree.group_start;
    d.grp_count = tree.group_count;
    d.grp_center = tree.group_center;
    d.grp_size = tree.group_size;
    d.pax = {p.ax.data()};
    d.pay = {p.ay.data()};
    d.paz = {p.az.data()};
    d.G = G;
    d.soft2 = soft2;
    d.g_offset = g_off;
    d.n_groups = ng;
    return d;
}

// ============================================================================
// Universal tree walk template — handles all 13 traversal variants.
// Op selects the interaction type, Filt selects the particle filter.
// Dead branches are eliminated at compile time.
// ============================================================================

template <int Op, int Filt>
static void tree_walk(const TreeWalkData& d, const char* name) {
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    size_t stack_bytes = TREE_STACK_SIZE * sizeof(int);

    // Capture Views (Kokkos requires value captures for GPU lambdas)
    auto cell_com  = d.cell_com;
    auto cell_sst  = d.cell_size_sq_theta;
    auto cell_ch   = d.cell_children;
    auto cell_pb   = d.cell_pbeg;
    auto cell_pe   = d.cell_pend;
    auto cell_lf   = d.cell_is_leaf;
    auto spos      = d.sorted_pos;
    auto sidx      = d.sorted_indices;
    auto j_spos    = d.j_sorted_pos;
    auto j_sidx    = d.j_sorted_indices;
    auto gs        = d.grp_start;
    auto gc        = d.grp_count;
    auto gcen      = d.grp_center;
    auto grp_sz    = d.grp_size;
    auto pax = d.pax, pay = d.pay, paz = d.paz;
    auto caccel    = d.cell_accel;
    auto pvx = d.pvx, pvy = d.pvy, pvz = d.pvz;
    auto dtf = d.dt_f;
    auto cvel      = d.cell_vel;
    auto sflag     = d.subset_flag;
    double G = d.G, soft2 = d.soft2, coeff = d.coeff;
    double eta_ff_sq = d.eta_ff_sq, eta_fb_sq = d.eta_fb_sq;
    int match_flag = d.match_flag, src_flag_val = d.src_flag;
    int g_off = d.g_offset;

    Kokkos::parallel_for(name,
        team_policy(d.n_groups, TREE_GROUP_SIZE)
            .set_scratch_size(0, Kokkos::PerTeam(stack_bytes)),
        KOKKOS_LAMBDA(const member_type& team) {
            int gid = g_off + team.league_rank();
            int tid = team.team_rank();

            Kokkos::View<int*, scratch_space, Kokkos::MemoryUnmanaged>
                stack(team.team_scratch(0), TREE_STACK_SIZE);

            // Load group bounding box
            int g_start_val = gs(gid);
            int g_count_val = gc(gid);
            double gcx = gcen(gid, 0), gcy = gcen(gid, 1), gcz = gcen(gid, 2);
            double gsx = grp_sz(gid, 0), gsy = grp_sz(gid, 1), gsz = grp_sz(gid, 2);

            // Load this thread's particle
            int my_sorted_idx = g_start_val + tid;
            double xi = 0, yi = 0, zi = 0, mi = 0;
            double vxi = 0, vyi = 0, vzi = 0;
            int my_orig_idx = -1;
            bool active = false;

            if (tid < g_count_val) {
                load4(&spos(my_sorted_idx, 0), xi, yi, zi, mi);
                my_orig_idx = sidx(my_sorted_idx);

                // Active condition
                if (Filt == FILT_ALL || Filt == FILT_DISJOINT)
                    active = true;
                else if (Filt == FILT_SELF || Filt == FILT_CROSS)
                    active = (sflag(my_sorted_idx) == match_flag);
                else // FILT_ACTIVE
                    active = (sflag(my_sorted_idx) != 2);

                // Load velocity for timestep
                if (Op == OP_FORCE_DT && active) {
                    vxi = pvx(my_orig_idx);
                    vyi = pvy(my_orig_idx);
                    vzi = pvz(my_orig_idx);
                }
            }

            // Accumulation variables
            double axi = 0, ayi = 0, azi = 0;
            double cx = 0, cy = 0, cz = 0;
            double t_min_sq = 1e60;

            // Gradient: load current acceleration for jerk computation
            if (Op == OP_GRADIENT && active) {
                axi = pax(my_orig_idx);
                ayi = pay(my_orig_idx);
                azi = paz(my_orig_idx);
            }

            // Initialize stack with root
            if (tid == 0) stack(0) = 0;
            int stack_top = 1;
            team.team_barrier();

            // ---- Tree walk ----
            while (stack_top > 0) {
                // Pop node (thread 0 manages stack)
                int node;
                if (tid == 0) {
                    node = stack(stack_top - 1);
                    stack_top--;
                    stack(TREE_STACK_SIZE - 1) = node;
                }
                team.team_barrier();
                node = stack(TREE_STACK_SIZE - 1);
                int saved_top = stack_top;

                // Read monopole
                double ccx, ccy, ccz, cmass;
                load4(&cell_com(node, 0), ccx, ccy, ccz, cmass);

                // Skip zero-mass cells (subset modes only)
                if ((Filt == FILT_SELF || Filt == FILT_CROSS || Filt == FILT_ACTIVE)
                    && cmass <= 0.0) {
                    team.team_barrier();
                    if (tid == 0) stack(TREE_STACK_SIZE - 2) = saved_top;
                    team.team_barrier();
                    stack_top = stack(TREE_STACK_SIZE - 2);
                    continue;
                }

                double size_sq_th = cell_sst(node);
                int is_leaf = cell_lf(node);

                bool accept = mac_accept(gcx, gcy, gcz, gsx, gsy, gsz,
                                          ccx, ccy, ccz, size_sq_th);

                if (accept) {
                    // ---- Monopole approximation ----
                    if (active) {
                        double dx = ccx - xi, dy = ccy - yi, dz = ccz - zi;
                        double r2 = dx*dx + dy*dy + dz*dz + soft2;

                        if (Op == OP_FORCE || Op == OP_FORCE_DT) {
                            double inv_r = gravity_interaction(dx, dy, dz, r2, G * cmass, axi, ayi, azi);
                            if (Op == OP_FORCE_DT) {
                                double dvx = cvel(node, 0) - vxi;
                                double dvy = cvel(node, 1) - vyi;
                                double dvz = cvel(node, 2) - vzi;
                                double t_ff, t_fb;
                                symmetrized_timestep(r2, inv_r, dx, dy, dz, dvx, dvy, dvz, G, mi + cmass, t_ff, t_fb);
                                t_min_sq = Kokkos::fmin(t_min_sq,
                                    Kokkos::fmin(eta_ff_sq * t_ff * t_ff, eta_fb_sq * t_fb * t_fb));
                            }
                        }
                        if (Op == OP_GRADIENT) {
                            double dax = caccel(node, 0) - axi;
                            double day = caccel(node, 1) - ayi;
                            double daz = caccel(node, 2) - azi;
                            gradient_correction_interaction(dx, dy, dz, dax, day, daz, r2, coeff * cmass, cx, cy, cz);
                        }
                    }
                } else if (is_leaf) {
                    // ---- Direct P2P ----
                    if (active) {
                        int lpbeg = cell_pb(node);
                        int lpend = cell_pe(node);
                        for (int j = lpbeg; j < lpend; j++) {
                            // P2P skip conditions
                            if (Filt == FILT_ALL) {
                                if (j_sidx(j) == my_orig_idx) continue;
                            }
                            // FILT_DISJOINT: no skip (trees are disjoint)
                            if (Filt == FILT_SELF) {
                                if (sflag(j) != match_flag) continue;
                                if (j == my_sorted_idx) continue;
                            }
                            if (Filt == FILT_CROSS) {
                                if (sflag(j) != src_flag_val) continue;
                            }
                            if (Filt == FILT_ACTIVE) {
                                if (j == my_sorted_idx) continue;
                            }

                            double pjx, pjy, pjz, mj;
                            load4(&j_spos(j, 0), pjx, pjy, pjz, mj);
                            double dx = pjx - xi, dy = pjy - yi, dz = pjz - zi;
                            double r2 = dx*dx + dy*dy + dz*dz + soft2;

                            if (Op == OP_FORCE || Op == OP_FORCE_DT) {
                                double inv_r = gravity_interaction(dx, dy, dz, r2, G * mj, axi, ayi, azi);
                                if (Op == OP_FORCE_DT) {
                                    int j_orig = j_sidx(j);
                                    double dvx = pvx(j_orig) - vxi;
                                    double dvy = pvy(j_orig) - vyi;
                                    double dvz = pvz(j_orig) - vzi;
                                    double t_ff, t_fb;
                                    symmetrized_timestep(r2, inv_r, dx, dy, dz, dvx, dvy, dvz, G, mi + mj, t_ff, t_fb);
                                    t_min_sq = Kokkos::fmin(t_min_sq,
                                        Kokkos::fmin(eta_ff_sq * t_ff * t_ff, eta_fb_sq * t_fb * t_fb));
                                }
                            }
                            if (Op == OP_GRADIENT) {
                                int j_orig = j_sidx(j);
                                double dax = pax(j_orig) - axi;
                                double day = pay(j_orig) - ayi;
                                double daz = paz(j_orig) - azi;
                                gradient_correction_interaction(dx, dy, dz, dax, day, daz, r2, coeff * mj, cx, cy, cz);
                            }
                        }
                    }
                } else {
                    // ---- Open cell: push children ----
                    if (tid == 0) {
                        stack_top = saved_top;
                        for (int c = 0; c < 8; c++) {
                            int child = cell_ch(node, c);
                            if (child >= 0 && stack_top < TREE_STACK_SIZE - 1)
                                stack(stack_top++) = child;
                        }
                    }
                }

                team.team_barrier();
                if (tid == 0) stack(TREE_STACK_SIZE - 2) = stack_top;
                team.team_barrier();
                stack_top = stack(TREE_STACK_SIZE - 2);
            }

            // ---- Write results ----
            if (active) {
                if (Op == OP_FORCE || Op == OP_FORCE_DT) {
                    pax(my_orig_idx) = axi;
                    pay(my_orig_idx) = ayi;
                    paz(my_orig_idx) = azi;
                }
                if (Op == OP_FORCE_DT) {
                    dtf(my_orig_idx) = Kokkos::sqrt(t_min_sq);
                }
                if (Op == OP_GRADIENT) {
                    pax(my_orig_idx) = axi + cx;
                    pay(my_orig_idx) = ayi + cy;
                    paz(my_orig_idx) = azi + cz;
                }
            }
        });
}

// Helper: compute eta_ff² and eta_fb² for timestep kernels
static void get_eta_sq(double eta, double& eta_ff_sq, double& eta_fb_sq) {
#ifdef ENABLE_COLLISIONAL
    (void)eta;
    eta_ff_sq = g_collisional.eta_ff * g_collisional.eta_ff;
    eta_fb_sq = g_collisional.eta_fb * g_collisional.eta_fb;
#else
    eta_ff_sq = eta * eta;
    eta_fb_sq = eta * eta;
#endif
}

// ==========================================================================
// Public API: thin wrappers that fill TreeWalkData and call tree_walk<>
// ==========================================================================

// ---- Basic self-interaction (no subset flags) ----

void tree_self_acceleration(Particles& p, const Octree& tree,
                            int start, int count, double G, double soft2,
                            int group_offset, int num_groups_param) {
    (void)start;
    if (count <= 0 || tree.n_cells == 0) return;
    int ng = (num_groups_param >= 0) ? num_groups_param : tree.n_groups;
    auto d = make_base(p, tree, G, soft2, group_offset, ng);
    d.cell_com = tree.cell_com;
    tree_walk<OP_FORCE, FILT_ALL>(d, "tree_traversal");
}

void tree_self_acceleration_and_timestep(Particles& p, const Octree& tree,
                                          int start, int count,
                                          double G, double soft2, double eta,
                                          int group_offset, int num_groups_param) {
    (void)start;
    if (count <= 0 || tree.n_cells == 0) return;
    int ng = (num_groups_param >= 0) ? num_groups_param : tree.n_groups;
    auto d = make_base(p, tree, G, soft2, group_offset, ng);
    d.cell_com = tree.cell_com;
    d.cell_vel = tree.cell_vel;
    d.pvx = {p.vx.data()}; d.pvy = {p.vy.data()}; d.pvz = {p.vz.data()};
    d.pm = {p.mass.data()}; d.dt_f = {p.dt_adapt.data()};
    get_eta_sq(eta, d.eta_ff_sq, d.eta_fb_sq);
    tree_walk<OP_FORCE_DT, FILT_ALL>(d, "tree_traversal_dt");
}

void tree_self_gradient_correction(Particles& p, const Octree& tree,
                                    double G, double soft2, double dt,
                                    int group_offset, int num_groups_param) {
    if (tree.n_particles <= 0 || tree.n_cells == 0) return;
    int ng = (num_groups_param >= 0) ? num_groups_param : tree.n_groups;
    auto d = make_base(p, tree, G, soft2, group_offset, ng);
    d.cell_com = tree.cell_com;
    d.cell_accel = tree.cell_accel;
    d.coeff = G * dt * dt * GRAD_COEFF;
    tree_walk<OP_GRADIENT, FILT_ALL>(d, "tree_self_grad");
}

// ---- Cross-tree (i_tree groups, j_tree cells — disjoint particle sets) ----

void tree_cross_acceleration(Particles& p,
                             const Octree& i_tree,
                             const Octree& j_tree,
                             double G, double soft2) {
    if (i_tree.n_particles <= 0 || j_tree.n_cells == 0) return;
    TreeWalkData d{};
    // Groups + i-particles from i_tree
    d.sorted_pos = i_tree.sorted_pos;
    d.sorted_indices = i_tree.sorted_indices;
    d.grp_start = i_tree.group_start;
    d.grp_count = i_tree.group_count;
    d.grp_center = i_tree.group_center;
    d.grp_size = i_tree.group_size;
    // Cell structure + j-particles from j_tree
    d.cell_com = j_tree.cell_com;
    d.cell_size_sq_theta = j_tree.cell_size_sq_theta;
    d.cell_children = j_tree.cell_children;
    d.cell_pbeg = j_tree.cell_pbeg;
    d.cell_pend = j_tree.cell_pend;
    d.cell_is_leaf = j_tree.cell_is_leaf;
    d.j_sorted_pos = j_tree.sorted_pos;
    d.j_sorted_indices = j_tree.sorted_indices;
    // Output
    d.pax = {p.ax.data()}; d.pay = {p.ay.data()}; d.paz = {p.az.data()};
    d.G = G; d.soft2 = soft2;
    d.g_offset = 0; d.n_groups = i_tree.n_groups;
    tree_walk<OP_FORCE, FILT_DISJOINT>(d, "tree_cross_force");
}

void tree_cross_gradient_correction(Particles& p,
                                     const Octree& i_tree,
                                     const Octree& j_tree,
                                     double G, double soft2, double dt) {
    if (i_tree.n_particles <= 0 || j_tree.n_cells == 0) return;
    TreeWalkData d{};
    // Groups + i-particles from i_tree
    d.sorted_pos = i_tree.sorted_pos;
    d.sorted_indices = i_tree.sorted_indices;
    d.grp_start = i_tree.group_start;
    d.grp_count = i_tree.group_count;
    d.grp_center = i_tree.group_center;
    d.grp_size = i_tree.group_size;
    // Cell structure + j-particles from j_tree
    d.cell_com = j_tree.cell_com;
    d.cell_accel = j_tree.cell_accel;
    d.cell_size_sq_theta = j_tree.cell_size_sq_theta;
    d.cell_children = j_tree.cell_children;
    d.cell_pbeg = j_tree.cell_pbeg;
    d.cell_pend = j_tree.cell_pend;
    d.cell_is_leaf = j_tree.cell_is_leaf;
    d.j_sorted_pos = j_tree.sorted_pos;
    d.j_sorted_indices = j_tree.sorted_indices;
    // Output
    d.pax = {p.ax.data()}; d.pay = {p.ay.data()}; d.paz = {p.az.data()};
    d.G = G; d.soft2 = soft2;
    d.coeff = G * dt * dt * GRAD_COEFF;
    d.g_offset = 0; d.n_groups = i_tree.n_groups;
    tree_walk<OP_GRADIENT, FILT_DISJOINT>(d, "tree_cross_grad");
}

// ---- Global-tree cross-force (single tree, dual monopoles) ----

void tree_global_cross_acceleration(Particles& p, const Octree& tree,
                                     int source_flag, double G, double soft2) {
    if (tree.n_particles <= 0 || tree.n_cells == 0) return;
    auto d = make_base(p, tree, G, soft2, 0, tree.n_groups);
    d.cell_com = (source_flag == 0) ? tree.cell_com_slow : tree.cell_com_fast;
    d.subset_flag = tree.sorted_subset_flag;
    d.match_flag = 1 - source_flag;  // target flag
    d.src_flag = source_flag;
    tree_walk<OP_FORCE, FILT_CROSS>(d, "tree_global_cross");
}

void tree_global_cross_gradient_correction(Particles& p, const Octree& tree,
                                            int source_flag,
                                            double G, double soft2, double dt) {
    if (tree.n_particles <= 0 || tree.n_cells == 0) return;
    auto d = make_base(p, tree, G, soft2, 0, tree.n_groups);
    d.cell_com = (source_flag == 0) ? tree.cell_com_slow : tree.cell_com_fast;
    d.cell_accel = (source_flag == 0) ? tree.cell_accel_slow : tree.cell_accel_fast;
    d.subset_flag = tree.sorted_subset_flag;
    d.coeff = G * dt * dt * GRAD_COEFF;
    d.match_flag = 1 - source_flag;
    d.src_flag = source_flag;
    tree_walk<OP_GRADIENT, FILT_CROSS>(d, "tree_global_cross_grad");
}

// ---- Global-tree self-force (single tree, subset = targets AND sources) ----

void tree_global_self_acceleration(Particles& p, const Octree& tree,
                                    int subset_flag, double G, double soft2) {
    if (tree.n_particles <= 0 || tree.n_cells == 0) return;
    auto d = make_base(p, tree, G, soft2, 0, tree.n_groups);
    d.cell_com = (subset_flag == 0) ? tree.cell_com_slow : tree.cell_com_fast;
    d.subset_flag = tree.sorted_subset_flag;
    d.match_flag = subset_flag;
    tree_walk<OP_FORCE, FILT_SELF>(d, "tree_global_self");
}

void tree_global_self_gradient_correction(Particles& p, const Octree& tree,
                                           int subset_flag,
                                           double G, double soft2, double dt) {
    if (tree.n_particles <= 0 || tree.n_cells == 0) return;
    auto d = make_base(p, tree, G, soft2, 0, tree.n_groups);
    d.cell_com = (subset_flag == 0) ? tree.cell_com_slow : tree.cell_com_fast;
    d.cell_accel = (subset_flag == 0) ? tree.cell_accel_slow : tree.cell_accel_fast;
    d.subset_flag = tree.sorted_subset_flag;
    d.coeff = G * dt * dt * GRAD_COEFF;
    d.match_flag = subset_flag;
    tree_walk<OP_GRADIENT, FILT_SELF>(d, "tree_global_self_grad");
}

void tree_global_self_acceleration_and_timestep(Particles& p, const Octree& tree,
                                                 int subset_flag,
                                                 double G, double soft2, double eta) {
    if (tree.n_particles <= 0 || tree.n_cells == 0) return;
    auto d = make_base(p, tree, G, soft2, 0, tree.n_groups);
    d.cell_com = (subset_flag == 0) ? tree.cell_com_slow : tree.cell_com_fast;
    d.cell_vel = (subset_flag == 0) ? tree.cell_vel_slow : tree.cell_vel_fast;
    d.subset_flag = tree.sorted_subset_flag;
    d.pvx = {p.vx.data()}; d.pvy = {p.vy.data()}; d.pvz = {p.vz.data()};
    d.pm = {p.mass.data()}; d.dt_f = {p.dt_adapt.data()};
    d.match_flag = subset_flag;
    get_eta_sq(eta, d.eta_ff_sq, d.eta_fb_sq);
    tree_walk<OP_FORCE_DT, FILT_SELF>(d, "tree_global_self_dt");
}

// ---- GADGET-4 all-particle forces on active subset (sflag != 2) ----

void tree_all_forces_on_subset(Particles& p, const Octree& tree,
                                double G, double soft2) {
    if (tree.n_particles <= 0 || tree.n_cells == 0) return;
    auto d = make_base(p, tree, G, soft2, 0, tree.n_groups);
    d.cell_com = tree.cell_com;
    d.subset_flag = tree.sorted_subset_flag;
    tree_walk<OP_FORCE, FILT_ACTIVE>(d, "tree_all_forces");
}

void tree_all_gradient_on_subset(Particles& p, const Octree& tree,
                                  double G, double soft2, double dt) {
    if (tree.n_particles <= 0 || tree.n_cells == 0) return;
    auto d = make_base(p, tree, G, soft2, 0, tree.n_groups);
    d.cell_com = tree.cell_com;
    d.cell_accel = tree.cell_accel;
    d.subset_flag = tree.sorted_subset_flag;
    d.coeff = G * dt * dt * GRAD_COEFF;
    tree_walk<OP_GRADIENT, FILT_ACTIVE>(d, "tree_all_grad");
}

void tree_all_forces_and_timestep_on_subset(Particles& p, const Octree& tree,
                                             double G, double soft2, double eta) {
    if (tree.n_particles <= 0 || tree.n_cells == 0) return;
    auto d = make_base(p, tree, G, soft2, 0, tree.n_groups);
    d.cell_com = tree.cell_com;
    d.cell_vel = tree.cell_vel;
    d.subset_flag = tree.sorted_subset_flag;
    d.pvx = {p.vx.data()}; d.pvy = {p.vy.data()}; d.pvz = {p.vz.data()};
    d.pm = {p.mass.data()}; d.dt_f = {p.dt_adapt.data()};
    get_eta_sq(eta, d.eta_ff_sq, d.eta_fb_sq);
    tree_walk<OP_FORCE_DT, FILT_ACTIVE>(d, "tree_all_forces_dt");
}

// ==========================================================================
// Post-hoc momentum correction for tree forces
// ==========================================================================

void tree_momentum_correction(Particles& p, int start, int count) {
    if (count <= 1) return;

    auto ax_v = p.ax, ay_v = p.ay, az_v = p.az, mass_v = p.mass;

    double sum_mx = 0, sum_my = 0, sum_mz = 0, sum_m = 0;

    Kokkos::parallel_reduce("mom_mx", count, KOKKOS_LAMBDA(int ii, double& v) {
        v += mass_v(start + ii) * ax_v(start + ii);
    }, sum_mx);
    Kokkos::parallel_reduce("mom_my", count, KOKKOS_LAMBDA(int ii, double& v) {
        v += mass_v(start + ii) * ay_v(start + ii);
    }, sum_my);
    Kokkos::parallel_reduce("mom_mz", count, KOKKOS_LAMBDA(int ii, double& v) {
        v += mass_v(start + ii) * az_v(start + ii);
    }, sum_mz);
    Kokkos::parallel_reduce("mom_m", count, KOKKOS_LAMBDA(int ii, double& v) {
        v += mass_v(start + ii);
    }, sum_m);

    double inv_m = 1.0 / sum_m;
    double cx = sum_mx * inv_m, cy = sum_my * inv_m, cz = sum_mz * inv_m;

    Kokkos::parallel_for("mom_correct", count, KOKKOS_LAMBDA(int ii) {
        int i = start + ii;
        ax_v(i) -= cx; ay_v(i) -= cy; az_v(i) -= cz;
    });
}
