#pragma once

#include "particles.h"
#include <Kokkos_Core.hpp>

// Barnes-Hut octree for O(N log N) force computation
// Follows Bonsai2 algorithmic approach, adapted to Kokkos (no raw CUDA)

// Tree traversal group size (threads per team in traversal kernel)
constexpr int TREE_GROUP_SIZE = 64;

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

// Post-hoc momentum correction: subtracts net force / total mass
// to enforce exact linear momentum conservation (tree forces are not
// antisymmetric due to asymmetric MAC decisions between groups)
void tree_momentum_correction(Particles& p, int start, int count);
