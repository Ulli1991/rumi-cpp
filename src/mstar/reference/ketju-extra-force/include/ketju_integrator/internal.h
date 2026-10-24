// Header for internal implementation details

#ifndef KETJU_INTEGRATOR_INTERNAL_H
#define KETJU_INTEGRATOR_INTERNAL_H

#include <stddef.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#include <mpi.h>

#include "ketju_integrator.h"

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

// For compatibility with C++ programs
#ifdef __cplusplus
extern "C" {
#endif

struct octree_node {
    struct octree_node *children[8];
    struct octree_node *parent;
    double center[3];
    double half_size;
    int *object_list;
    int Npart;
    int level;
    bool is_leaf;
    int ID;
};

struct octree_data {
    struct octree_node *root_node, **particle_in_node;
    int **indexlist;
};

struct weight_index {
    double weight;
    int index;
};
struct weight_index_2 {
    double weight;
    int index1;
    int index2;
    int id1;
    int id2;
};

struct graph_edge {
    int id;
    int vertex1;
    int vertex2;
    float weight;
};

struct graph_vertex {
    int id;
    int degree;
    int parent;
    int edgetoparent;
    int level;
    bool inMST;
    int cell[3];
};

struct spanning_tree_coordinate_system {
    int num_vertices;
    int num_edges;    

    struct graph_vertex *vertex;

    // TODO check if some of these can be removed
    struct graph_edge *local_edge, *local_edge_subset, *edge_in_mst;

    // Absolute coordinates of the tree root.
    // These are integrated to allow for forces that do not conserve the CoM of
    // the system.
    double root_pos[3]; 
    double root_vel[3];
    double root_aux_vel[3];
    // Relative coordinates of the edges
    double (*edge_rel_pos)[3];
    double (*edge_rel_vel)[3];
    double (*edge_rel_aux_vel)[3];
    double (*edge_rel_acc)[3];
    double (*edge_rel_acc_vd)[3];
};

struct group_info {
    int num_gbs_groups;
    int *gbs_group_sizes;
};

// info on MPI groups this task is in
struct task_info {
    // main task group = all tasks participating in the integration
    int main_group_size;
    int main_group_rank;
    MPI_Comm main_comm;
    
    // gbs task group = tasks participating in a given set of substep levels
    // i.e. force calculations done in this group
    // TODO maybe could be named as force_group, since mostly appears in force
    // calculations
    int gbs_group_size;
    int gbs_group_rank;
    int gbs_group_index; //< which of the gbs groups this task belongs in
    MPI_Comm gbs_comm;
    MPI_Comm *gbs_data_comms;

    bool gbs_comms_initialized;
    
    int global_rank; // <rank in MPI_COMM_WORLD
};

// A wrapper struct for all the internal state needed by the integrator
struct ketju_system_internal_state {

    struct task_info *task_info;
    struct group_info *group_info;

    struct spanning_tree_coordinate_system *mst;
    struct octree_data *octree_data;

    double U; // potential energy
    double T;
    double B;

    int gbs_k;

    double (*aux_vel)[3]; 
    double (*acc)[3]; // velocity independent acceleration
    double (*acc_vd)[3]; // velocity dependent acceleration (for all particles at least for now)
    double (*aux_spin)[3]; 
    double (*spin_derivative)[3]; 
};


// Small helper functions.
// Inline definitions allow using them in different
// compilation units without introducing extra public symbols.

// timer functions for basic profiling, used as
//   tick(timer)
//   ... code ...
//   tock(timer)
// the value of *timer is not meaningful between the calls.
//
static inline void tick(double *timer) {
    *timer -= MPI_Wtime();   
}
static inline void tock(double *timer) {
    *timer += MPI_Wtime();   
}

// for exiting on catastrophic errors
// TODO most uses now could be replaced with error codes allowing potentially
// recovering from errors during integration / saving the state so far.
static inline void terminate(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

static inline void get_v1_v2_fast(const struct ketju_system *R, int i, int *v1,
                                  int *v2) {
    *v1 = R->internal_state->mst->edge_in_mst[i].vertex1;
    *v2 = R->internal_state->mst->edge_in_mst[i].vertex2;
}

static inline void vector_cross(const double a[3], const double b[3],
                                double res[3]) {
    res[0] = a[1] * b[2] - b[1] * a[2];
    res[1] = a[2] * b[0] - b[2] * a[0];
    res[2] = a[0] * b[1] - b[0] * a[1];
}

static inline double vector_norm(const double a[3]) {
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

static inline double vector_dot(const double a[3], const double b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static inline void unit_vector(const double a[3], double res[3]) {
    double norm = vector_norm(a);
    res[0] = a[0] / norm;
    res[1] = a[1] / norm;
    res[2] = a[2] / norm;
}

static inline double norm_and_unit_vector(const double a[3], double res[3]) {
    double norm = vector_norm(a);
    res[0] = a[0] / norm;
    res[1] = a[1] / norm;
    res[2] = a[2] / norm;
    return norm;
}


// Divide work for a i = (0..N-1), j = (i+1..N-1) loop between Nt tasks.
// For the given task, computes the starting value for i (istart), the starting
// value for j (jstart) for the loop with i==istart, the number of elements
// handled by the given task (ThisBlock) and the number of elements before the
// block handled by this task (CumNum).
// The loops for i,j go up to N (exclusive) until ThisBlock elements have been
// processed.
static inline void loop_scheduling_N2(int N, int Nt, int task, int *istart,
                                      int *jstart, int *ThisBlock, int *CumNum
                                      ) {
    int Np = (N * (N - 1)) / 2;
    int block = (int)floor(Np / Nt);
    int limit = (block + 1) * Nt - Np;
    int ThisCumNum;
    if (task < limit) {
        ThisCumNum = task * block;
        *ThisBlock = block;
    } else {
        ThisCumNum = limit * block + (task - limit) * (block + 1);
        *ThisBlock = block + 1;
    }

    double drow =
        1.0 * N - 0.5 -
        0.5 * sqrt((1.0 - 2.0 * N) * (1.0 - 2.0 * N) - 8.0 * ThisCumNum);
    int row = (int)floor(drow);
    int rowstart =
        (int)round(-0.5 * (1.0 * row + 1.0) * (row - 2.0 * N) - 1.0 * N);
    *istart = row;
    *jstart = ThisCumNum - rowstart + row + 1;
    *CumNum = ThisCumNum;
}

// declarations of internal functions used between compilation units

void ketju_binary_PN_CoM_velocity(struct ketju_system *R, int i, int j,
                                  double CoM_vel[3]);
int ketju_do_bh_mergers(struct ketju_system *R);

void ketju_init_gbs_comms(struct ketju_system *R);
void ketju_free_gbs_comms(struct ketju_system *R);

void ketju_build_mst(struct ketju_system *R);

void ketju_compute_PN_acc(struct ketju_system *, double (*vel_to_use)[3],
                          double (*spin_to_use)[3]);

// These functions are called through the helper below
bool ketju_check_relative_proximity_general(int v1, int v2, const int Nd,
                                    const struct ketju_system *R, int *d,
                                    int path[/*Nd*/], int sign[/*Nd*/]);
bool ketju_check_relative_proximity_ND_2(int v1, int v2,
                                    const struct ketju_system *R, int *d,
                                    int path[2], int sign[2]);

static inline bool ketju_check_relative_proximity(int v1, int v2, const int Nd,
                                    const struct ketju_system *R, int *d,
                                    int path[/*Nd*/], int sign[/*Nd*/]) {

#ifndef NDEBUG
    const struct graph_vertex *vertex = R->internal_state->mst->vertex;
    // Debugging stuff for some tree construction bug that may still be present
    if (vertex[v1].level == 0 && vertex[v2].level == 0) {
        char buf[100];
        sprintf(buf,
                "ketju_check_relative_proximity: This should not happen! "
                "thistask: %d --- v1 v2 %d %d --- levels: %d %d\n",
                R->internal_state->task_info->global_rank, vertex[v1].id,
                vertex[v2].id, vertex[v1].level, vertex[v2].level);
        terminate(buf);
    }
#endif

    if (Nd == 2) {
        // Use the specialized version for this common case
        // TODO: measure if this actually helps.
        return ketju_check_relative_proximity_ND_2(v1, v2, R, d, path, sign);
    }

    return ketju_check_relative_proximity_general(v1, v2, Nd, R, d, path, sign);
}

#ifdef __cplusplus
}
#endif

#endif // KETJU_INTEGRATOR_INTERNAL_H
