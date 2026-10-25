// Functions for constructing and working with the MST coordinate system

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>


#include "ketju_integrator/ketju_integrator.h"
#include "ketju_integrator/internal.h"

bool ketju_check_relative_proximity_ND_2(int v1, int v2,
                                          const struct ketju_system *R, int *d,
                                          int path[2], int sign[2]) {

    const struct graph_vertex *vertex = R->internal_state->mst->vertex;

    int diff = abs(vertex[v1].level - vertex[v2].level);

    if (diff == 1) {
        if (vertex[v2].parent == v1) {
            *d = 1;
            sign[0] = -1;
            path[0] = vertex[v2].edgetoparent;
            return true;
        } else if (vertex[v1].parent == v2) {
            *d = 1;
            sign[0] = +1;
            path[0] = vertex[v1].edgetoparent;
            return true;
        }
        return false;
    } else if (diff == 2) {
        // Need to check that there actually is a valid parent before checking
        // the parents parent.
        if (vertex[v2].parent >= 0 &&
            vertex[vertex[v2].parent].parent == v1) {
            *d = 2;
            sign[0] = -1;
            sign[1] = -1;
            path[0] = vertex[v2].edgetoparent;
            path[1] = vertex[vertex[v2].parent].edgetoparent;
            return true;
        } else if (vertex[v1].parent >= 0 &&
                   vertex[vertex[v1].parent].parent == v2) {
            *d = 2;
            sign[0] = +1;
            sign[1] = +1;
            path[0] = vertex[v1].edgetoparent;
            path[1] = vertex[vertex[v1].parent].edgetoparent;
            return 1;
        }
        return false;
    } else { // diff == 0
        if (vertex[v1].parent == vertex[v2].parent) {
            *d = 2;
            sign[0] = -1;
            sign[1] = +1;
            path[0] = vertex[v2].edgetoparent;
            path[1] = vertex[v1].edgetoparent;
            return true;
        }
        return false;
    }
    return false;
}

bool ketju_check_relative_proximity_general(int v1, int v2, const int Nd,
                                    const struct ketju_system *R, int *d,
                                    int path[Nd], int sign[Nd]) {

    const struct graph_vertex *vertex = R->internal_state->mst->vertex;

    for (int i = 0; i < Nd; i++) {
        path[i] = -1000;
        sign[i] = 0;
    }

    if (vertex[v1].level == vertex[v2].level) {

        int halfNd = (int)floor(0.5 * Nd);
        int pathend[Nd];
        memset(pathend, 0, sizeof pathend);

        for (int i = 0; i < halfNd; i++) {
            sign[i] = +1;
            path[i] = vertex[v1].edgetoparent;
            pathend[i] = vertex[v2].edgetoparent;
            v1 = vertex[v1].parent;
            v2 = vertex[v2].parent;
            if (v1 == v2) {
                *d = 2 * (i + 1);
                int c = 0;
                for (int m = i + 1; m < (*d); m++) {
                    sign[m] = -1;
                    path[m] = pathend[i - c];
                    c++;
                }
                return true;
            }
        }

    } else {
        *d = abs(vertex[v1].level - vertex[v2].level);
        if (*d <= Nd) {
            int lo, hi, thissign;
            if (vertex[v1].level > vertex[v2].level) {
                lo = v2;
                hi = v1;
                thissign = +1;
            } else {
                lo = v1;
                hi = v2;
                thissign = -1;
            }
            for (int i = 0; i < *d; i++) {
                path[i] = vertex[hi].edgetoparent;
                sign[i] = thissign;
                hi = vertex[hi].parent;
            }
            if (lo == hi) { return true; }
        }
    }

    return false;
}

static void get_v1_v2(int N, int target, int *v1, int *v2) {

    int final_element_on_row = 0;
    int elements_on_this_row;

    for (int row = 0; row < N - 1; row++) {
        if (row == 0) {
            elements_on_this_row = N - 2 - row;
        } else {
            elements_on_this_row = N - 1 - row;
        }
        final_element_on_row += elements_on_this_row;
        if (target <= final_element_on_row) {
            *v1 = row;
            int diff = final_element_on_row - target;
            *v2 = N - 1 - diff;
            break;
        }
    }
}

static int cmp_weight_index2(const void *a, const void *b) {
    struct weight_index_2 *a1 = (struct weight_index_2 *)a;
    struct weight_index_2 *a2 = (struct weight_index_2 *)b;
    if ((*a1).weight > (*a2).weight)
        return 1;
    else if ((*a1).weight < (*a2).weight)
        return -1;
    else
        return 0;
}

static int cmp(const void *a, const void *b) {
    struct graph_edge *a1 = (struct graph_edge *)a;
    struct graph_edge *a2 = (struct graph_edge *)b;
    if ((*a1).weight > (*a2).weight)
        return 1;
    else if ((*a1).weight < (*a2).weight)
        return -1;
    else
        return 0;
}

static int edge_search_for_prim(struct ketju_system *R, int num_local_edges) {

    tick(&R->perf->tprim3);
    const struct spanning_tree_coordinate_system *mst = R->internal_state->mst;
    struct weight_index wi;

    wi.weight = DBL_MAX;
    wi.index = -1;
    for (int i = 0; i < num_local_edges; i++) {
        if (mst->vertex[mst->local_edge[i].vertex1].inMST !=
            mst->vertex[mst->local_edge[i].vertex2].inMST) {
            wi.weight = mst->local_edge[i].weight;
            wi.index = mst->local_edge[i].id;
            break;
        }
    }

    tock(&R->perf->tprim3);
    tick(&R->perf->tprim4);

    MPI_Allreduce(MPI_IN_PLACE, &wi, 1, MPI_DOUBLE_INT, MPI_MINLOC,
                  R->internal_state->task_info->main_comm);

    tock(&R->perf->tprim4);

    return wi.index;
}

static void PrimMST(struct ketju_system *R) {

    const struct ketju_system_physical_state *ps = R->physical_state;
    const struct task_info *ti = R->internal_state->task_info;
    struct spanning_tree_coordinate_system *mst = R->internal_state->mst;

    tick(&R->perf->time_mst);

    // Construct the edges handled by this task
    int block_size, cum_num_edge, istart, jstart;
    loop_scheduling_N2(mst->num_vertices, ti->main_group_size,
                       ti->main_group_rank, &istart, &jstart, &block_size,
                       &cum_num_edge);

    int num_local_edges = 0;
    for (int i = istart; i < mst->num_vertices; i++) {

        if (i > istart) { jstart = i + 1; }
        for (int j = jstart; j < mst->num_vertices; j++) {
            if (num_local_edges >= block_size) {
                // If there are more tasks than edges, ThisBlock may be 0
                goto after_edge_construction; // exit both loops
            }
            double w = 0;
            for (int k = 0; k < 3; k++) {
                double ds = ps->pos[i][k] - ps->pos[j][k];
                w += ds * ds;
            }
            w = sqrt(w);

            mst->local_edge[num_local_edges].weight = w;
            mst->local_edge[num_local_edges].vertex1 = i;
            mst->local_edge[num_local_edges].vertex2 = j;
            mst->local_edge[num_local_edges].id =
                (int)(cum_num_edge + num_local_edges);

            num_local_edges++;

        }
    }
    after_edge_construction:

    qsort(mst->local_edge, num_local_edges, sizeof(struct graph_edge), cmp);

    // Setup vertices
    for (int i = 0; i < mst->num_vertices; i++) {
        mst->vertex[i].id = i;
        mst->vertex[i].degree = 0;
        mst->vertex[i].edgetoparent = -1;
        mst->vertex[i].parent = -1;
        mst->vertex[i].level = 0;
        mst->vertex[i].inMST = false;
    }
    
    // Find the root of the tree = the particle closest to the CoM.
    double rmin = DBL_MAX;
    int root = -1;
    double CoM_pos[3], CoM_vel[3];
    ketju_compute_CoM(R, CoM_pos, CoM_vel);

    for (int i = 0; i < R->num_particles; i++) {
        double r2 = 0;
        for (int k = 0; k < 3; k++) {
            double ds = ps->pos[i][k] - CoM_pos[k];
            r2 += ds * ds;
        }
        if (rmin > sqrt(r2)) {
            rmin = sqrt(r2);
            root = i;
        }
    }
    mst->vertex[root].inMST = true;
    for (int k = 0; k < 3; k++) {
        mst->root_pos[k] = ps->pos[root][k];
        mst->root_vel[k] = ps->vel[root][k];
    }

    // Construct the MST
    for (int c = 0; c < mst->num_vertices - 1; ++c) {
        int next_edge = edge_search_for_prim(R, num_local_edges);

        int v1 = -1, v2 = -1;
        get_v1_v2(mst->num_vertices, next_edge, &v1, &v2);

        mst->edge_in_mst[c].vertex1 = v1;
        mst->edge_in_mst[c].vertex2 = v2;

        mst->vertex[v1].degree++;
        mst->vertex[v2].degree++;

        if (!mst->vertex[v1].inMST) {

            mst->vertex[v1].parent = v2;
            mst->vertex[v1].level = mst->vertex[v2].level + 1;
            mst->vertex[v1].inMST = true;
            mst->vertex[v1].edgetoparent = c;
            for (int k = 0; k < 3; k++) {
                mst->edge_rel_pos[c][k] = ps->pos[v1][k] - ps->pos[v2][k];
                mst->edge_rel_vel[c][k] = ps->vel[v1][k] - ps->vel[v2][k];
            }
        } else {
            mst->vertex[v2].parent = v1;
            mst->vertex[v2].level = mst->vertex[v1].level + 1;
            mst->vertex[v2].inMST = true;
            mst->vertex[v2].edgetoparent = c;
            for (int k = 0; k < 3; k++) {
                mst->edge_rel_pos[c][k] = ps->pos[v2][k] - ps->pos[v1][k];
                mst->edge_rel_vel[c][k] = ps->vel[v2][k] - ps->vel[v1][k];
            }
        }
    }
    // now all the mst edges are constructed
    mst->num_edges = mst->num_vertices -1; 
    tock(&R->perf->time_mst);
}

static double get_rmax(struct ketju_system *R) {
    double rmax2 = 0;
    double CoM_pos[3], CoM_vel[3];
    ketju_compute_CoM(R, CoM_pos, CoM_vel);
    for (int i = 0; i < R->num_particles; i++) {
        double ds;
        double r2 = 0;
        for (int k = 0; k < 3; k++) {
            ds = R->physical_state->pos[i][k] - CoM_pos[k];
            r2 += ds * ds;
        }
        if (r2 > rmax2) { rmax2 = r2; }
    }
    return sqrt(rmax2);
}

static int get_octant(int triplet[3]) {
    if (triplet[0] == 1) {
        if (triplet[1] == 1) {
            if (triplet[2] == -1) {
                return 6;
            } else {
                return 7;
            }
        } else {
            if (triplet[2] == -1) {
                return 4;
            } else {
                return 5;
            }
        }
    } else {
        if (triplet[1] == 1) {
            if (triplet[2] == -1) {
                return 2;
            } else {
                return 3;
            }
        } else {
            if (triplet[2] == -1) {
                return 0;
            } else {
                return 1;
            }
        }
    }
    return -1;
}

static void get_triplet_from_octant(int octant, int *triplet) {

    if (octant == 0) {
        triplet[0] = -1;
        triplet[1] = -1;
        triplet[2] = -1;
        return;
    }
    if (octant == 1) {
        triplet[0] = -1;
        triplet[1] = -1;
        triplet[2] = +1;
        return;
    }
    if (octant == 2) {
        triplet[0] = -1;
        triplet[1] = +1;
        triplet[2] = -1;
        return;
    }
    if (octant == 3) {
        triplet[0] = -1;
        triplet[1] = +1;
        triplet[2] = +1;
        return;
    }
    if (octant == 4) {
        triplet[0] = +1;
        triplet[1] = -1;
        triplet[2] = -1;
        return;
    }
    if (octant == 5) {
        triplet[0] = +1;
        triplet[1] = -1;
        triplet[2] = +1;
        return;
    }
    if (octant == 6) {
        triplet[0] = +1;
        triplet[1] = +1;
        triplet[2] = -1;
        return;
    }
    if (octant == 7) {
        triplet[0] = +1;
        triplet[1] = +1;
        triplet[2] = +1;
        return;
    }
    return;
}

static void list_octree_level(struct octree_node *Node, int *LeafSizes,
                       int **LeafParticles, int *Nleafs) {

    if (Node->is_leaf) {
        LeafSizes[*Nleafs] = Node->Npart;
        LeafParticles[*Nleafs] = calloc(Node->Npart, sizeof(int));
        for (int i = 0; i < Node->Npart; i++) {
            LeafParticles[*Nleafs][i] = Node->object_list[i];
        }
        (*Nleafs)++;
    } else {
        for (int oct = 0; oct < 8; oct++) {
            if (Node->children[oct]) {
                list_octree_level(Node->children[oct], LeafSizes, LeafParticles,
                                  Nleafs);
            }
        }
    }
}

static void GetNumberOfLeafs(struct octree_node *Node, int *Nleaf) {
    if (Node->is_leaf) {
        (*Nleaf)++;
    } else {
        for (int oct = 0; oct < 8; oct++) {
            if (Node->children[oct]) {
                GetNumberOfLeafs(Node->children[oct], Nleaf);
            }
        }
    }
}

static void free_octree(struct octree_node *Node) {
    for (int oct = 0; oct < 8; oct++) {
        if (Node->children[oct]) {
            free_octree(Node->children[oct]);
        }
    }
    free(Node->object_list);
    free(Node);
}

static void allocate_new_octree_node(struct octree_node *Node, int oct,
                                     int *label) {

    Node->children[oct] = calloc(1, sizeof(struct octree_node));
    Node->children[oct]->parent = Node;
    Node->children[oct]->half_size = Node->half_size / 2;
    Node->children[oct]->Npart = 0;
    Node->children[oct]->level = Node->level + 1;
    Node->children[oct]->is_leaf = 0;
    Node->children[oct]->ID = *label;

    (*label)++;
}

static void octree(struct ketju_system *R, struct octree_node *Node,
                   int MaxLeafSize, int *label) {

    struct octree_data *octd = R->internal_state->octree_data;

    int NpartOctant[8] = {0};
    int triplet[3];
    for (int i = 0; i < Node->Npart; i++) {
        int j = Node->object_list[i];
        for (int k = 0; k < 3; k++) {
            triplet[k] =
                R->physical_state->pos[j][k] - Node->center[k] >= 0 ? 1 : -1;
        }
        int oct = get_octant(triplet);

        if (NpartOctant[oct] == 0) {
            allocate_new_octree_node(Node, oct, label);
        }

        octd->indexlist[oct][NpartOctant[oct]] = j;

        NpartOctant[oct]++;
        Node->children[oct]->Npart++;
    }

    for (int oct = 0; oct < 8; oct++) {

        if (NpartOctant[oct] == 0) { continue; }

        Node->children[oct]->object_list =
            calloc(Node->children[oct]->Npart, sizeof(int));

        for (int i = 0; i < Node->children[oct]->Npart; i++) {
            Node->children[oct]->object_list[i] = octd->indexlist[oct][i];
        }

        get_triplet_from_octant(oct, triplet);

        for (int k = 0; k < 3; k++) {
            Node->children[oct]->center[k] =
                Node->center[k] + triplet[k] * Node->children[oct]->half_size;
        }

        for (int i = 0; i < Node->children[oct]->Npart; i++) {
            octd->particle_in_node[Node->children[oct]->object_list[i]] =
                Node->children[oct];
        }

        if (Node->children[oct]->Npart == 1) {
            Node->children[oct]->is_leaf = 1;
        }
    }

    for (int oct = 0; oct < 8; oct++) {
        if (Node->children[oct]) {
            if (Node->children[oct]->Npart > MaxLeafSize) {
                octree(R, Node->children[oct], MaxLeafSize, label);
            } else {
                Node->children[oct]->is_leaf = 1;
            }
        }
    }
}

static void spatialpartition(struct ketju_system *R, int MaxLeafSize) {

    struct octree_data *octd = R->internal_state->octree_data;

    double rmax = get_rmax(R);

    octd->root_node = calloc(1, sizeof(struct octree_node));

    octd->root_node->ID = 0;
    octd->root_node->parent = NULL;
    octd->root_node->half_size = 1.001 * rmax;
    octd->root_node->Npart = R->num_particles;
    octd->root_node->object_list = calloc(R->num_particles, sizeof(int));
    octd->root_node->level = 0;
    for (int i = 0; i < 3; i++) {
        octd->root_node->center[i] = 0.0;
    }

    for (int i = 0; i < R->num_particles; i++) {
        octd->root_node->object_list[i] = i;
    }

    int label = 1;
    octree(R, octd->root_node, MaxLeafSize, &label);
}

static int get_longest(double w[3], int id[3]) {

    double L = -1;
    int ind = -1;
    for (int m = 0; m < 3; m++) {
        if (w[m] > L) {
            L = w[m];
            ind = m;
        }
    }
    return id[ind];
}

static int get_edge(int Nleafs, int va, int vb) {

    int N = Nleafs - 2;
    int lo, hi;
    lo = va, hi = vb;
    if (va > vb) {
        lo = vb;
        hi = va;
    }
    int index = hi - 1;
    for (int p = 0; p < lo; p++) {
        index += N;
        N--;
    }
    return index;
}

static void SolveMetaMST(struct ketju_system *R, int Nleafs, int *Npart,
                         int **Particles, int *NumEdge) {

    const struct task_info *ti = R->internal_state->task_info;
    const struct ketju_system_physical_state *ps = R->physical_state;
    struct spanning_tree_coordinate_system *mst = R->internal_state->mst;
    struct octree_data *octd = R->internal_state->octree_data;

    int Nedge = Nleafs * (Nleafs - 1) / 2;
    struct weight_index_2 *W = calloc(Nedge, sizeof(struct weight_index_2));
    int *InMst = calloc(Nleafs, sizeof(int));

    int CumNumEdge = 0, istart = 0;
    int ThisBlock, jstart;
    loop_scheduling_N2(Nleafs, ti->main_group_size, ti->main_group_rank,
                       &istart, &jstart, &ThisBlock, &CumNumEdge);

    struct octree_node *Node_i, *Node_j;
    int counter = 0, ind1 = -1, ind2 = -1;

    int *ind1s = calloc(Nedge, sizeof(int));
    int *ind2s = calloc(Nedge, sizeof(int));
    double *weights = calloc(Nedge, sizeof(double));

    for (int i = istart; i < Nleafs; i++) {

        Node_i = octd->particle_in_node[Particles[i][0]];

        if (i > istart) { jstart = i + 1; }
        for (int j = jstart; j < Nleafs; j++) {

            if (counter >= ThisBlock) {
                goto after_loops;
            }

            Node_j = octd->particle_in_node[Particles[j][0]];

            double r2 = 0;
            for (int k = 0; k < 3; k++) {
                double ds = Node_i->center[k] - Node_j->center[k];
                r2 = ds * ds;
            }
            double NodeSeparation = sqrt(r2);
            double NodeSize =
                1.01 * sqrt(2) *
                (Node_i->half_size + Node_j->half_size); // with safety buffer

            if (NodeSeparation <= NodeSize) {

                double MinWeight = DBL_MAX;
                for (int q = 0; q < Npart[i]; q++) {
                    for (int p = 0; p < Npart[j]; p++) {
                        double r2 = 0;
                        int i1 = Particles[i][q];
                        int i2 = Particles[j][p];
                        for (int k = 0; k < 3; k++) {
                            double ds = ps->pos[i1][k] - ps->pos[i2][k];
                            r2 += ds * ds;
                        }
                        double r = sqrt(r2);
                        if (r < MinWeight) {
                            MinWeight = r;
                            ind1 = i1;
                            ind2 = i2;
                        }
                    }
                }

            } else {

                double MinWeight = DBL_MAX;
                for (int q = 0; q < Npart[i]; q++) {
                    int i1 = Particles[i][q];
                    double r2 = 0;
                    for (int k = 0; k < 3; k++) {
                        double ds = ps->pos[i1][k] - Node_j->center[k];
                        r2 += ds * ds;
                    }
                    double r = sqrt(r2);
                    if (r < MinWeight) {
                        MinWeight = r;
                        ind1 = i1;
                    }
                }
                MinWeight = DBL_MAX;
                for (int p = 0; p < Npart[j]; p++) {
                    int i2 = Particles[j][p];
                    double r2 = 0;
                    for (int k = 0; k < 3; k++) {
                        double ds = ps->pos[i2][k] - Node_i->center[k];
                        r2 += ds * ds;
                    }
                    double r = sqrt(r2);
                    if (r < MinWeight) {
                        MinWeight = r;
                        ind2 = i2;
                    }
                }
            }

            r2 = 0;
            for (int k = 0; k < 3; k++) {
                double ds = ps->pos[ind1][k] - ps->pos[ind2][k];
                r2 += ds * ds;
            }
            double r = sqrt(r2);

            weights[CumNumEdge + counter] = r;
            ind1s[CumNumEdge + counter] = ind1;
            ind2s[CumNumEdge + counter] = ind2;
            counter++;
        }
    }
    after_loops:

    // TODO: this is probably just gathering non-zero data, can it be replaced
    // with a gather call?
    MPI_Allreduce(MPI_IN_PLACE, ind1s, Nedge, MPI_INT, MPI_SUM, ti->main_comm);
    MPI_Allreduce(MPI_IN_PLACE, ind2s, Nedge, MPI_INT, MPI_SUM, ti->main_comm);
    MPI_Allreduce(MPI_IN_PLACE, weights, Nedge, MPI_DOUBLE, MPI_SUM,
                  ti->main_comm);

    counter = 0;
    for (int i = 0; i < Nleafs; i++) {
        for (int j = i + 1; j < Nleafs; j++) {
            W[counter].weight = weights[counter];
            W[counter].id1 = ind1s[counter];
            W[counter].id2 = ind2s[counter];
            W[counter].index1 = i;
            W[counter].index2 = j;
            counter++;
        }
    }

    free(weights);
    free(ind1s);
    free(ind2s);

    // Filter out edges that clearly aren't part of the final tree, these get 
    // the corresponding entry in the flag table set
    istart = 0;
    int istop = Nedge;

    int block = (int)floor(Nedge / ti->main_group_size);
    istart = ti->main_group_rank * block;
    istop = (ti->main_group_rank + 1) * block;
    if (ti->main_group_rank == ti->main_group_size - 1) istop = Nedge;

    bool *flag = calloc(Nedge, sizeof(bool));

    int Noff = 0;
    double w[3];
    int ind[3];
    for (int e1 = istart; e1 < istop; e1++) {

        int v1 = W[e1].index1;
        int v2 = W[e1].index2;
        ind[0] = e1;
        w[0] = W[e1].weight;

        if (w[0] < 0) continue;

        for (int v3 = 0; v3 < Nleafs; v3++) {

            if (w[0] < 0) continue;

            if (v3 == v1 || v3 == v2) continue;

            ind[1] = get_edge(Nleafs, v1, v3);
            ind[2] = get_edge(Nleafs, v2, v3);

            w[1] = W[ind[1]].weight;
            w[2] = W[ind[2]].weight;
            if (w[1] < 0 || w[2] < 0) continue;
            int elong = get_longest(w, ind);

            flag[elong] = true;
            W[elong].weight *= -1;
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, flag, Nedge, MPI_C_BOOL, MPI_LOR, ti->main_comm);

    // How many are filtered
    for (int e1 = 0; e1 < Nedge; e1++) {
        if (flag[e1]) Noff++;
    }

    // Collect the remaining edges
    struct weight_index_2 *Wfilter =
        calloc(Nedge - Noff, sizeof(struct weight_index_2));
    
    counter = 0;
    for (int e1 = 0; e1 < Nedge; e1++) {
        if (flag[e1] == 0) {
            Wfilter[counter].weight = W[e1].weight;
            Wfilter[counter].id1 = W[e1].id1;
            Wfilter[counter].id2 = W[e1].id2;
            Wfilter[counter].index1 = W[e1].index1;
            Wfilter[counter].index2 = W[e1].index2;
            counter++;
        }
    }

    Nedge -= Noff;
    free(W);
    free(flag);

    qsort(Wfilter, Nedge, sizeof(struct weight_index_2), cmp_weight_index2);

    InMst[0] = 1;

    int remaining = Nleafs - 1;
    istart = 0;
    istop = Nedge;

    block = (int)floor(Nedge / ti->main_group_size);
    istart = ti->main_group_rank * block;
    istop = (ti->main_group_rank + 1) * block;
    if (ti->main_group_rank == ti->main_group_size - 1) istop = Nedge;

    do {
        struct weight_index wi;
        wi.weight = DBL_MAX;
        wi.index = -1;

        for (int i = istart; i < istop; i++) {
            int i1 = Wfilter[i].index1;
            int i2 = Wfilter[i].index2;
            if (InMst[i1] != InMst[i2]) {
                wi.index = i;
                wi.weight = Wfilter[i].weight;
                break;
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, &wi, 1, MPI_DOUBLE_INT, MPI_MINLOC,
                      ti->main_comm);

        InMst[Wfilter[wi.index].index1] = 1;
        InMst[Wfilter[wi.index].index2] = 1;

        mst->edge_in_mst[*NumEdge].vertex1 = Wfilter[wi.index].id1;
        mst->edge_in_mst[*NumEdge].vertex2 = Wfilter[wi.index].id2;
        mst->edge_in_mst[*NumEdge].weight = Wfilter[wi.index].weight;
        (*NumEdge)++;

        remaining--;
    } while (remaining > 0);

    free(Wfilter);
    free(InMst);
}

static void SolveSubMST(struct ketju_system *R, int Npart, int *Particles,
                        int *NumEdge) {
    if (Npart == 1) return;

    const struct ketju_system_physical_state *ps = R->physical_state;
    struct spanning_tree_coordinate_system *mst = R->internal_state->mst;

    int Nedge = Npart * (Npart - 1) / 2;
    struct weight_index_2 *W = calloc(Nedge, sizeof(struct weight_index_2));
    int *InMst = calloc(Npart, sizeof(int));

    int counter = 0;
    for (int i = 0; i < Npart; i++) {
        int ind1 = Particles[i];
        for (int j = i + 1; j < Npart; j++) {
            int ind2 = Particles[j];
            double r2 = 0;
            for (int k = 0; k < 3; k++) {
                double ds = ps->pos[ind1][k] - ps->pos[ind2][k];
                r2 += ds * ds;
            }
            double r = sqrt(r2);
            W[counter].weight = r;
            W[counter].index1 = i;
            W[counter].index2 = j;
            W[counter].id1 = ind1;
            W[counter].id2 = ind2;
            counter++;
        }
    }

    qsort(W, Nedge, sizeof(struct weight_index_2), cmp_weight_index2);

    InMst[0] = 1;

    int remaining = Npart - 1;

    do {

        for (int i = 0; i < Nedge; i++) {
            int i1 = W[i].index1;
            int i2 = W[i].index2;
            if (InMst[i1] != InMst[i2]) {
                if (InMst[i1] == 1) {
                    InMst[i2] = 1;
                } else {
                    InMst[i1] = 1;
                }

                mst->edge_in_mst[*NumEdge].vertex1 = W[i].id1;
                mst->edge_in_mst[*NumEdge].vertex2 = W[i].id2;
                mst->edge_in_mst[*NumEdge].weight = W[i].weight;
                (*NumEdge)++;
                break;
            }
        }

        remaining--;
    } while (remaining > 0);

    free(W);
    free(InMst);
}

static void SwapEdge(struct ketju_system *R, int i, int j) {
    struct spanning_tree_coordinate_system *mst = R->internal_state->mst;

    int v1 = mst->edge_in_mst[i].vertex1;
    int v2 = mst->edge_in_mst[i].vertex2;
    double w = mst->edge_in_mst[i].weight;

    mst->edge_in_mst[i].vertex1 = mst->edge_in_mst[j].vertex1;
    mst->edge_in_mst[i].vertex2 = mst->edge_in_mst[j].vertex2;
    mst->edge_in_mst[i].weight = mst->edge_in_mst[j].weight;

    mst->edge_in_mst[j].vertex1 = v1;
    mst->edge_in_mst[j].vertex2 = v2;
    mst->edge_in_mst[j].weight = w;
}

static void CombineAllMST(struct ketju_system *R, int NumEdge) {

    const struct ketju_system_physical_state *ps = R->physical_state;
    struct spanning_tree_coordinate_system *mst = R->internal_state->mst;

    int root = -1;
    double rmin = DBL_MAX;
    double CoM_pos[3], CoM_vel[3];
    ketju_compute_CoM(R, CoM_pos, CoM_vel);

    for (int i = 0; i < mst->num_vertices; i++) {
        double r2 = 0;
        for (int k = 0; k < 3; k++) {
            double ds = ps->pos[i][k] - CoM_pos[k];
            r2 += ds * ds;
        }
        double r = sqrt(r2);
        if (rmin > r) {
            rmin = r;
            root = i;
        }
    }

    int counter = 0;
    mst->vertex[root].inMST = true;
    for (int k = 0; k < 3; k++) {
        mst->root_pos[k] = ps->pos[root][k];
        mst->root_vel[k] = ps->vel[root][k];
    }

    int istop = NumEdge;

    do {

        struct weight_index wi;
        wi.weight = DBL_MAX;
        wi.index = -1;

        for (int i = counter; i < istop; i++) {
            int v1 = mst->edge_in_mst[i].vertex1;
            int v2 = mst->edge_in_mst[i].vertex2;
            if (mst->vertex[v1].inMST != mst->vertex[v2].inMST) {
                if (mst->edge_in_mst[i].weight < wi.weight) {
                    wi.weight = mst->edge_in_mst[i].weight;
                    wi.index = i;
                }
            }
        }

        int new = mst->edge_in_mst[wi.index].vertex1;
        int old = mst->edge_in_mst[wi.index].vertex2;
        if (mst->vertex[new].inMST == 1) {
            old = mst->edge_in_mst[wi.index].vertex1;
            new = mst->edge_in_mst[wi.index].vertex2;
        }

        mst->vertex[new].inMST = 1;
        mst->vertex[new].parent = old;
        mst->vertex[new].level = mst->vertex[old].level + 1;
        mst->vertex[new].edgetoparent = counter;

        for (int k = 0; k < 3; k++) {
            mst->edge_rel_pos[counter][k] = ps->pos[new][k] - ps->pos[old][k];
            mst->edge_rel_vel[counter][k] = ps->vel[new][k] - ps->vel[old][k];
        }

        SwapEdge(R, counter, wi.index);
        counter++;

    } while (counter < NumEdge);
}

static void DivideAndConquerPrim(struct ketju_system *R) {

    tick(&R->perf->time_mst);
    struct spanning_tree_coordinate_system *mst = R->internal_state->mst;
    struct octree_data *octd = R->internal_state->octree_data;

    for (int i = 0; i < mst->num_vertices; i++) {
        mst->vertex[i].id = i;
        mst->vertex[i].degree = 0;
        mst->vertex[i].edgetoparent = -1;
        mst->vertex[i].parent = -1;
        mst->vertex[i].level = 0;
        mst->vertex[i].inMST = 0;
    }

    int Nleafs = 0;
    int MaxLeafSize = (int)floor(3 * sqrt(mst->num_vertices));

    tick(&R->perf->tprim1);
    spatialpartition(&R[0], MaxLeafSize);

    GetNumberOfLeafs(octd->root_node, &Nleafs);
    int *LeafSizes = calloc(Nleafs, sizeof(int));
    int **LeafParticles = calloc(Nleafs, sizeof(int *));

    tock(&R->perf->tprim1);

    tick(&R->perf->tprim2);
    Nleafs = 0;
    list_octree_level(octd->root_node, LeafSizes, LeafParticles, &Nleafs);

    tock(&R->perf->tprim2);

    tick(&R->perf->tprim3);
    int NumEdge = 0;
    for (int i = 0; i < Nleafs; i++) {
        SolveSubMST(R, LeafSizes[i], LeafParticles[i], &NumEdge);
    }
    tock(&R->perf->tprim3);

    tick(&R->perf->tprim4);
    SolveMetaMST(R, Nleafs, LeafSizes, LeafParticles, &NumEdge);
    tock(&R->perf->tprim4);

    tick(&R->perf->tprim5);
    CombineAllMST(R, NumEdge);
    tock(&R->perf->tprim5);

    free_octree(octd->root_node);
    for (int i = 0; i < Nleafs; i++) {
        free(LeafParticles[i]);
    }
    free(LeafParticles);
    free(LeafSizes);

    // now all the mst edges are constructed
    mst->num_edges = mst->num_vertices -1; 

    tock(&R->perf->time_mst);
}

void ketju_build_mst(struct ketju_system *R) {
    // Use the basic prim method for small systems always, since divide and
    // conquer is broken for small enough systems.
    if (R->options->mst_algorithm == KETJU_MST_PRIM || R->num_particles < 10) {
        PrimMST(R);
    } else if (R->options->mst_algorithm == KETJU_MST_DIVIDE_AND_CONQUER_PRIM) {
        DivideAndConquerPrim(R);
    } else {
        char msg[100];
        sprintf(msg, "Invalid mst_algorithm value %d!",
                R->options->mst_algorithm);
        terminate(msg);
    }
}
