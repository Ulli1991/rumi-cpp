#pragma once

#include "particles.h"
#include <Kokkos_Core.hpp>

// Barnes-Hut octree for O(N log N) force computation
// Follows Bonsai2 algorithmic approach, adapted to Kokkos (no raw CUDA)

// Tree traversal group size (threads per team in traversal kernel)
constexpr int TREE_GROUP_SIZE = 64;

// Team size of the per-cell reduction kernels; host backends cap teams at the thread count
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
constexpr int TREE_CELL_TEAM = 32;
#else
constexpr int TREE_CELL_TEAM = 1;
#endif

// Max tree depth (log8(10^6) ~ 7, with margin)
constexpr int TREE_MAX_DEPTH = 25;

// Max stack size per team during traversal (8 children * max_depth)
constexpr int TREE_STACK_SIZE = 256;

struct Octree {
    // Cell data (flat arrays, indexed by cell ID)
    // cell_com(i, 0..3) = {cx, cy, cz, mass} — center of mass + total mass
    Kokkos::View<double*[4], Kokkos::LayoutRight> cell_com;

    // cell_vel(i, 0..2) = {vx, vy, vz} — center-of-mass velocity
    Kokkos::View<double*[3], Kokkos::LayoutRight> cell_vel;

    // cell_accel(i, 0..2) = {ax, ay, az} — center-of-mass acceleration
    // Computed bottom-up after forces are evaluated (not during build)
    Kokkos::View<double*[3], Kokkos::LayoutRight> cell_accel;

    // cell_dip(i, 0..8) = D_lq = sum m delta_l (a_q - a_bar_q), delta = x - cell_com; (i, 9..11) = sum m delta
    Kokkos::View<double*[12], Kokkos::LayoutRight> cell_dip;

    // cell_size_sq_theta(i) = cell_half_width² / theta² (pre-computed for MAC)
    Kokkos::View<double*> cell_size_sq_theta;

    // cell_children(i, 0..7) = child cell indices
    //   >= 0: child cell index
    //   -1: empty octant
    Kokkos::View<int*[8], Kokkos::LayoutRight> cell_children;

    // Particle range in sorted order per cell
    Kokkos::View<int*> cell_pbeg;  // first particle (inclusive)
    Kokkos::View<int*> cell_pend;  // last particle (exclusive)

    // Whether cell is a leaf (no children, contains particles directly)
    Kokkos::View<int*> cell_is_leaf;

    // Sorted particle data (spatially ordered by Morton key)
    Kokkos::View<int*>  sorted_indices;  // sorted position -> original particle index
    Kokkos::View<double*[4], Kokkos::LayoutRight> sorted_pos;  // {x, y, z, mass}

    // Permanent particle IDs in sorted order (survives particle reordering by partition_by_pivot)
    Kokkos::View<int*>  sorted_orig_ids;  // sorted position -> permanent particle ID (orig_index)

    // Per-sorted-particle subset flag: 0=slow, 1=fast (set by update_subsets)
    Kokkos::View<int*>  sorted_subset_flag;

    // Dual monopoles for slow/fast subsets (recomputed at each HHS level)
    Kokkos::View<double*[4], Kokkos::LayoutRight> cell_com_slow;  // {cx, cy, cz, mass} slow only
    Kokkos::View<double*[4], Kokkos::LayoutRight> cell_com_fast;  // {cx, cy, cz, mass} fast only

    // Dual cell velocities for slow/fast subsets (for fused timestep)
    Kokkos::View<double*[3], Kokkos::LayoutRight> cell_vel_slow;
    Kokkos::View<double*[3], Kokkos::LayoutRight> cell_vel_fast;

    // Dual cell accelerations for gradient correction
    Kokkos::View<double*[3], Kokkos::LayoutRight> cell_accel_slow;
    Kokkos::View<double*[3], Kokkos::LayoutRight> cell_accel_fast;
    Kokkos::View<double*[12], Kokkos::LayoutRight> cell_dip_slow;
    Kokkos::View<double*[12], Kokkos::LayoutRight> cell_dip_fast;

    // Group data for tree traversal (groups of spatially nearby particles)
    Kokkos::View<int*> group_start;     // start index in sorted array
    Kokkos::View<int*> group_count;     // particle count per group
    Kokkos::View<double*[3]> group_center;  // geometric center of group
    Kokkos::View<double*[3]> group_size;    // half-extent of group bounding box

    // Tree parameters
    int n_cells = 0;
    int n_groups = 0;
    int n_particles = 0;
    double theta = 0.5;
    int n_leaf = 32;      // max particles per leaf cell
    int n_group = 64;     // particles per traversal group
    bool grad_dipole = true;  // FSI gradient: cell acceleration dipole term (cell_dip) on accepted cells

    // Bounding box of the tree
    double box_cx, box_cy, box_cz;  // center
    double box_half;                 // half-width

    // Build octree from particles [start, start+count) in p
    // Computes Morton keys, sorts, builds tree on host, uploads to device
    void build(Particles& p, int start, int count, double theta_param);

    // O(N) position refresh: updates sorted_pos from live particle positions
    // using sorted_indices mapping. Tree structure (cells, MAC) stays unchanged.
    // Use after drift operations to avoid full tree rebuild.
    void refresh_positions(Particles& p);

    // Compute cell center-of-mass accelerations bottom-up from particle data.
    // Call AFTER force computation (p.ax/ay/az must be valid).
    // Leaf cells: mass-weighted average of particle accelerations.
    // Internal cells: mass-weighted average of child cell accelerations.
    void compute_cell_accelerations(Particles& p);

    // Create traversal groups from sorted particles
    void make_groups();

    // Rebuild sorted_indices after particle reordering (e.g. partition_by_pivot).
    // Uses sorted_orig_ids + reverse map to find current particle index. O(N) GPU.
    void rebuild_sorted_indices(Particles& p);

    // Full update after partition: rebuild indices, refresh positions, mark subsets,
    // recompute dual monopoles. Call once per HHS level after partition_by_pivot.
    // Two-flag version: particles in [slow_start, slow_start+slow_count) = 0 (slow), rest = 1 (fast)
    void update_subsets(Particles& p, int slow_start, int slow_count);

    // Three-flag version for GADGET-4 cross-force decomposition:
    // [start, start+slow_count) = 0 (slow), [start+slow_count, start+count) = 1 (fast), rest = 2 (other)
    void update_subsets_3flag(Particles& p, int start, int count, int slow_count);

    // Recompute total cell monopoles (cell_com) from sorted_pos.
    // Call after refresh_positions() when total monopoles need to be current.
    void recompute_total_monopoles();

    // Recompute dual monopoles (cell_com_slow/fast, cell_vel_slow/fast) from sorted_pos + flags.
    // Bottom-up on host. Call after refresh_positions + mark_subsets.
    // Takes Particles& for velocity access via sorted_indices.
    void recompute_dual_monopoles(Particles& p);

    // Compute dual cell accelerations (cell_accel_slow, cell_accel_fast) bottom-up.
    // Call after cross-force computation (p.ax/ay/az contain cross-forces).
    void compute_dual_cell_accelerations(Particles& p);

    // Compute cell_accel from active particles only (sflag != 2).
    // For GADGET-4: after computing F_all on active subset, get cell monopole accels.
    void compute_active_cell_accelerations(Particles& p);

    // Deallocate all Views
    void deallocate();
};

// Compute tree self-acceleration for particles [start, start+count)
// Results written to p.ax, p.ay, p.az
// group_offset/num_groups: process a subset of groups (for MPI distribution)
//   -1 for num_groups means process all groups
void tree_self_acceleration(Particles& p, const Octree& tree,
                            int start, int count, double G, double soft2,
                            int group_offset = 0, int num_groups = -1);

// Fused tree self-acceleration + adaptive timestep
// Computes forces AND pairwise timesteps during tree traversal
// Results written to p.ax, p.ay, p.az and p.dt_adapt
void tree_self_acceleration_and_timestep(Particles& p, const Octree& tree,
                                          int start, int count,
                                          double G, double soft2, double eta,
                                          int group_offset = 0, int num_groups = -1);

// Cross-force via tree: force on i-particles (grouped by i_tree) from j_tree
// i_tree provides spatial groups for i-particles; j_tree is walked for forces
// Results OVERWRITE p.ax, p.ay, p.az for i-particles
void tree_cross_acceleration(Particles& p,
                             const Octree& i_tree,
                             const Octree& j_tree,
                             double G, double soft2);

// Self-gradient correction via tree traversal (Dehnen jerk formula)
// Uses tree.cell_accel for monopole, per-particle accel for P2P
// Results ADDED to p.ax, p.ay, p.az
void tree_self_gradient_correction(Particles& p, const Octree& tree,
                                    double G, double soft2, double dt,
                                    int group_offset = 0, int num_groups = -1);

// Cross-gradient correction: jerk on i-particles from j_tree
// i_tree provides groups, j_tree provides tree structure + cell_accel
// Results ADDED to p.ax, p.ay, p.az for i-particles
void tree_cross_gradient_correction(Particles& p,
                                     const Octree& i_tree,
                                     const Octree& j_tree,
                                     double G, double soft2, double dt);

// Global-tree cross-force: force on target_subset particles from source_subset
// Uses dual monopoles (cell_com_slow/fast) and sorted_subset_flag for P2P filtering.
// source_flag: 0=slow sources, 1=fast sources
// Results OVERWRITE p.ax, p.ay, p.az for target particles only.
void tree_global_cross_acceleration(Particles& p, const Octree& tree,
                                     int source_flag, double G, double soft2);

// Global-tree cross-gradient correction using dual monopoles + cell accelerations.
// source_flag: 0=slow sources (uses cell_com_slow, cell_accel_slow), 1=fast sources
// Results ADDED to p.ax, p.ay, p.az for target particles.
void tree_global_cross_gradient_correction(Particles& p, const Octree& tree,
                                            int source_flag,
                                            double G, double soft2, double dt);

// Global-tree self-force: force on subset particles from same subset.
// Uses dual monopoles (cell_com_slow/fast) and sorted_subset_flag for P2P filtering.
// subset_flag: 0=slow subset, 1=fast subset (targets AND sources are same subset)
// Results OVERWRITE p.ax, p.ay, p.az for subset particles only.
void tree_global_self_acceleration(Particles& p, const Octree& tree,
                                    int subset_flag, double G, double soft2);

// Global-tree self-gradient correction using dual monopoles + cell accelerations.
// subset_flag: 0=slow subset, 1=fast subset
// Results ADDED to p.ax, p.ay, p.az for subset particles.
void tree_global_self_gradient_correction(Particles& p, const Octree& tree,
                                           int subset_flag,
                                           double G, double soft2, double dt);

// Global-tree fused self-force + timestep for a subset.
// Uses dual monopoles and subset velocities for timestep criterion.
// subset_flag: 0=slow, 1=fast
// Results OVERWRITE p.ax, p.ay, p.az and p.dt_adapt for subset particles.
void tree_global_self_acceleration_and_timestep(Particles& p, const Octree& tree,
                                                 int subset_flag,
                                                 double G, double soft2, double eta);

// Post-hoc momentum correction: subtracts net force / total mass
// to enforce exact linear momentum conservation (tree forces are not
// antisymmetric due to asymmetric MAC decisions between groups)
void tree_momentum_correction(Particles& p, int start, int count);

// ==========================================================================
// GADGET-4 cross-force decomposition: F_cross = F_all - F_self
// These compute forces from ALL particles (using total cell_com monopoles)
// on particles at the current HHS level (sflag != 2).
// ==========================================================================

// Force from ALL particles on active subset (sflag != 2).
// Uses cell_com (total monopoles). P2P: all sorted particles, self-skip only.
// Results OVERWRITE p.ax, p.ay, p.az for active particles (sflag != 2).
void tree_all_forces_on_subset(Particles& p, const Octree& tree,
                                double G, double soft2);

// Gradient correction from ALL particles on active subset (sflag != 2).
// Uses cell_com + cell_accel (total). Results ADDED to p.ax, p.ay, p.az.
void tree_all_gradient_on_subset(Particles& p, const Octree& tree,
                                  double G, double soft2, double dt);

// Fused force + timestep from ALL particles on active subset (sflag != 2).
// Uses cell_com + cell_vel (total). Results OVERWRITE p.ax, p.ay, p.az, p.dt_adapt.
void tree_all_forces_and_timestep_on_subset(Particles& p, const Octree& tree,
                                             double G, double soft2, double eta);
