#pragma once

#include "particles.h"

// FLYBY_CM_VELOCITY: pair time steps use the subsystem centre-of-mass velocity of members
// (FROST-CLUSTERS II, Rantala et al. 2025, arXiv:2506.04330, sec. 2.1)
extern bool flyby_cm_velocity;
// Velocities that enter the pair time steps (fly-by criterion and the time symmetrisation) of slots [lo, hi):
// vx, vy, vz, or in collisional mode with FLYBY_CM_VELOCITY the subsystem c.o.m. velocity of the members of
// g_subsystem_mgr.active. Device pointers, valid until the next call. Used by every path that computes pair
// time steps (DtK, AccDtK, the HHS_CC split, the CPU small-set path).
// k: per slot, tidal-sampling constant of FLYBY_CM_EPS (null: none): a pair also takes dt <= min(k_i / m_j,
// k_j / m_i) r^3, so that a slow particle's kicks, which sample a subsystem's internal motion at the pair's step,
// change its internal energy by at most FLYBY_CM_EPS x |E0| per step
// kf: per slot, the floor of that step (TIDAL_STEP_RES x P (1 - e)^(3/2) of a member's binary, 0 otherwise)
struct PairVelocities { double* x; double* y; double* z; double* k; double* kf; };
PairVelocities pair_velocities(Particles& p, int lo, int hi);
// Exact tree time-step / neighbour searches (dt_tree.h): device tree for source sets >= tree_dt_min_n, host tree for
// sets <= tree_dt_cpu_max (host-accessible memory only); 0 = off. tree_dt_vc_fac: low-speed split (multiplier of V_c, default 30)
extern int tree_dt_min_n;
extern int tree_ngb_min_n;   // the same for the neighbour search
extern int tree_dt_cpu_max;
extern double tree_dt_vc_fac;

// Full N-body
void compute_acceleration_and_timestep(Particles& p, double G, double soft2, double eta);

// Self-interaction within [start, start+count)
void self_acceleration(Particles& p, int start, int count,
                       double G, double soft2);

void self_gradient_correction(Particles& p, int start, int count,
                              double G, double soft2, double dt);

// Cross-interaction between two disjoint groups
void cross_acceleration(Particles& p,
                        int a_start, int a_count,
                        int b_start, int b_count,
                        double G, double soft2);

void cross_gradient_correction(Particles& p,
                               int a_start, int a_count,
                               int b_start, int b_count,
                               double G, double soft2, double dt);

// Compute adaptive timesteps for a subset of particles [start, start+count)
// Only considers mutual interactions within the subset
void compute_timestep_subset(Particles& p, int start, int count,
                          double G, double soft2, double eta);

// Range versions: sinks [i0, i0+ni) from sources [j0, j0+nj). The MPI paths pass their slice of the sinks.
void acceleration_and_timestep_range(Particles& p, int i0, int ni, int j0, int nj, double G, double soft2,
                                     double eta);
void acceleration_range(Particles& p, int i0, int ni, int j0, int nj, double G, double soft2);
void timestep_range(Particles& p, int i0, int ni, int j0, int nj, double G, double soft2, double eta);

// Snapshot of uncorrected accelerations for the gradient correction: element i - offset belongs to particle i
struct GradSnapshot { double* x; double* y; double* z; int offset; };
void gradient_correction_range(Particles& p, int i0, int ni, int j0, int nj, const GradSnapshot& si,
                               const GradSnapshot& sj, double G, double soft2, double dt, bool write_gmag);
#ifdef ENABLE_COLLISIONAL
// GRAD_FRESH: grad_mag(i) = |g'_i| of sinks [i0, i0+ni) from sources [j0, j0+nj) and the accelerations in ax, ay, az
// (left unchanged), subsystem partners excluded
void gradient_magnitude_range(Particles& p, int i0, int ni, int j0, int nj, double G, double soft2);
#endif

// Standalone neighbor search (BIFROST Section 3.5)
// Lightweight N² tiled kernel: detects neighbor pairs without computing forces
// or timesteps. No subsystem exclusion — all pairs are checked.
// Writes detected pairs to g_subsystem_mgr neighbor buffers as array indices.
void neighbor_search_subset(Particles& p, int start, int count,
                            double G, double r_ngb, double r_ngb_max,
                            double pivot_dt);


// HHS with connected components (HHS_CC): split [start, start+count) into components (pair time step below
// pivot, or subsystem partners, connect), reorder it so components are contiguous, return their sizes in order
// (one entry if connected or too dense)
#include <vector>
std::vector<int> cc_components(Particles& p, int start, int count, double pivot, double G, double soft2, double eta);
// Accelerations and gradient corrections from other components (cid per particle), added to ax, ay, az
void cc_inter_acceleration(Particles& p, int start, int count, const int* cid, double G, double soft2);
void cc_inter_gradient(Particles& p, int start, int count, const int* cid, const GradSnapshot& s, double G,
                       double soft2, double dt);
