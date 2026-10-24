// The public interface header with all the user-facing types and function
// prototypes.

#ifndef KETJU_INTEGRATOR_H
#define KETJU_INTEGRATOR_H

#include <stdbool.h>
#include <mpi.h>

// For compatibility with C++ programs
#ifdef __cplusplus
extern "C" {
#endif

// Types and enums
// Using bitflags for the PN terms allows more fine-grained control of the used
// terms (e.g. only PN2.5 radiation reaction or only conservative terms).
// XXX: this could end up being too complicated, but should be fairly easy to
// use especially if all commonly used combinations are pre-defined.
enum ketju_PN_flags {
    KETJU_PN_NONE = 0,
    KETJU_PN_1_0_ACC = 1 << 0,
    KETJU_PN_THREEBODY = 1 << 1,
    KETJU_PN_2_0_ACC = 1 << 2,
    KETJU_PN_2_5_ACC = 1 << 3,
    KETJU_PN_3_0_ACC = 1 << 4,
    KETJU_PN_3_5_ACC = 1 << 5,
    KETJU_PN_CONSERVATIVE = KETJU_PN_1_0_ACC | KETJU_PN_2_0_ACC |
                            KETJU_PN_3_0_ACC | KETJU_PN_THREEBODY,
    KETJU_PN_DYNAMIC_ALL =
        0xFF, // all the spin independent terms, is this a good name?

    KETJU_PN_1_5_SPIN_ACC = 1 << 8,
    KETJU_PN_2_0_SPIN_ACC = 1 << 9,
    KETJU_PN_SPIN_EVOL = 1 << 10,
    KETJU_PN_SPIN_ALL = 0x700,

    KETJU_PN_ALL = KETJU_PN_DYNAMIC_ALL | KETJU_PN_SPIN_ALL,
};

enum ketju_mst_algorithm {
    KETJU_MST_PRIM,
    KETJU_MST_DIVIDE_AND_CONQUER_PRIM,
};

struct ketju_integration_options {
    int gbs_kmax; //< Maximum k (number of rows) the integrator can use

    double gbs_relative_tolerance;

    double output_time_relative_tolerance;
    //< Maximum size of relative error in time iteration
    double output_time_absolute_tolerance;
    //< Minimum absolute size of error in time iteration

    unsigned int PN_flags; //< Bitflags for enabling different PN terms

    int steps_between_mst_reconstruction;
    int max_tree_distance;
    //< Maximum separation in the tree structure before using CoM coordinate
    //distances
    int max_step_count;
    //< Maximum number of attempted (successful or not) steps before abandoning
    //  the integration and dumping state.
    //  Set to 0 or a negative value to disable.

    enum ketju_mst_algorithm mst_algorithm;

    double star_star_softening;
    //< force softening length between non-pn particles, disabled if <= 0
    bool no_star_star_gravity;
    //< Disable gravity between stars (non-pn particles), only interactions
    // including pn-particles are computed.
    bool enable_bh_mergers;
    //< Check for mergers between BHs (pn particles).
    double bh_merger_distance_in_Rs;
    //< If BH mergers are enabled, the particles are merged when they come within
    // this factor * 2G/c^2 * (m1 + m2).
    // The PN forces break down below about 5, and for suppressing PN CoM
    // oscillations values of ~>10 are recommended.
    bool enable_bh_merger_kicks;
    //< Include the merger kick in BH mergers.
    // If disabled the merger remnant moves at the CoM velocity of the merged
    // BHs.
};

// Default values applied when creating the system
static const struct ketju_integration_options
    ketju_default_integration_options = {
        .gbs_kmax = 16,
        .gbs_relative_tolerance = 1e-9,
        .output_time_relative_tolerance = 1e-6,
        .output_time_absolute_tolerance = 1e-300,
        .PN_flags = KETJU_PN_ALL,
        .steps_between_mst_reconstruction = 10,
        .max_tree_distance = 2,
        .max_step_count = -1,
        .mst_algorithm = KETJU_MST_PRIM,
        .star_star_softening = 0.,
        .no_star_star_gravity = false,
        .enable_bh_mergers = true,
        .bh_merger_distance_in_Rs = 12,
        .enable_bh_merger_kicks = true,
};

struct ketju_natural_constants {
    double G;
    double c;
};
// Gadget default values for now
static const struct ketju_natural_constants ketju_default_natural_constants = {
    .G = 43007.1, .c = 299792.0};


struct ketju_system_physical_state {
    double time;
    double *mass;
    double (*pos)[3];
    double (*vel)[3];
    double (*spin)[3]; //spin only for pn particles
};

// The physical state of the system.
// The coordinate system is arbitrary, but it is numerically favorable to have
// the CoM of the system stay near the origin.
// Note that the PN corrections depend slightly on the coordinate velocities
// so that different choices of the reference frame lead to slightly different
// results. The PN corrected equations of motion are approximately
// Lorentz-invariant, while the standard Newtonian equations are
// Galilei-invariant, so for general systems mixing these particle types there
// is no real 'correct' way to choose the coordinates. The differences caused by
// these effects are very small for reasonable choices of rest frame.
struct ketju_performance_counters {
    double time_force;
    double time_force_comm;
    double time_coord;
    double time_mst;
    double time_gbscomm;
    double time_gbsextr;
    double time_total;

    // TODO: check what these actually mean
    double tprim1, tprim2, tprim3, tprim4, tprim5;

    int successful_steps;
    int failed_steps;
    double total_work; //< ~ Number of force evaluations done
    double relative_energy_error;
    //< Relative change in the energy of the system.
    //  Gives information on accuracy only when PN terms are disabled
};

struct ketju_bh_merger_info {
    double t_merger; // time of the merger
    double m1, m2, m_remnant; // masses
    double chi1, chi2, chi_remnant; // dimensionless spin magnitudes
    double v_kick; // magnitude of kick velocity
    int extra_index1,
        extra_index2; // Indices of the components in particle_extra_data
};

struct ketju_system; // forward declaration for the following function types

// Function pointer types for user specified extra forces.
// The vindep_force may only depend on the positions and constant properties
// of the particles, while vdep_force may depend on the velocities and spins
// as well.
// For the auxiliary velocity trick used by the integrator to work correctly,
// the velocity and spin fields must be accessed through the pointer arguments
// to the function.
// For both functions, the additional accelerations must be added to the acc_out
// argument values, which must not be modified otherwise.
typedef void (*ketju_extra_vindep_force)(const struct ketju_system*,
                                         double (*acc_out)[3]);
typedef void (*ketju_extra_vdep_force)(const struct ketju_system*,
                                       double (*vel)[3],
                                       double (*spin)[3],
                                       double (*acc_out)[3]);

struct ketju_system {
    int num_particles; //< Total number of particles
    int num_pn_particles; //< Number of PN enabled particles

    struct ketju_system_physical_state *physical_state;
    //< Physical state of the system (as a pointer for making temporary copies)
    // TODO looking like we're not going to use the pointer nature for making
    // copies, maybe make into a direct member?

    struct ketju_system_internal_state *internal_state; 
    //< Internal state (implementation detail)

    // TODO is it better to have this stuff (which doesn't need to be copied
    // into temp variables) as direct members to avoid extra pointer derefs, or
    // as pointers for consistency with the above and perhaps some cache
    // behaviour benefits?
    struct ketju_performance_counters *perf; 
    struct ketju_integration_options *options;

    struct ketju_natural_constants *constants;
    //< Values of natural constants which define the unit system
    
    void *particle_extra_data;
    //< Optional extra data (e.g. IDs) associated with the particles, which is
    // reordered in case of mergers so that the removed particle data are found
    // at the end of the array.
    // The caller is responsible for managing this memory.

    size_t particle_extra_data_elem_size;
    //< Size of the extra data for a single particle.

    ketju_extra_vindep_force extra_vindep_force;
    //< Optional extra velocity independent force component
    ketju_extra_vdep_force extra_vdep_force;
    //< Optional extra velocity dependent force component

    int num_mergers; //< How many mergers there have been
    struct ketju_bh_merger_info *merger_infos; //< contains num_mergers elements
};

// Function prototypes
void ketju_create_system(struct ketju_system *R, int num_pn_particles,
                         int num_other_particles, MPI_Comm comm);
void ketju_free_system(struct ketju_system *R);

// Compute the centre of mass of the system
void ketju_compute_CoM(const struct ketju_system *R, double CoM_pos[3], double CoM_vel[3]);
// Shift the coordinate origin of the system to the CoM, outputs the original values
void ketju_into_CoM_frame(struct ketju_system *R, double CoM_pos[3], double CoM_vel[3]);
// Runs the integrator for a given time interval.
// The time interval can be negative for backwards-in-time integration.
void ketju_run_integrator(struct ketju_system *R, double time_interval);

// debug helpers
void ketju_debug_dump_integrator_state(const struct ketju_system *R,
                                       const char *fname);
bool ketju_debug_read_integrator_state(struct ketju_system *R,
                                       const char *fname, MPI_Comm comm);

// Compute the separation vector rv between particles i and j
void ketju_particle_separation(const struct ketju_system *R, int i,
                                       int j, double rv[3]);

// Compute the separation vector rv and relative velocity vv between particles
// i and j, using the velocity data in vel
void ketju_particle_separation_and_relative_velocity(
    const struct ketju_system *R, int i, int j, double rv[3], double vv[3],
    double (*vel)[3]);

#ifdef __cplusplus
}
#endif

#endif // KETJU_INTEGRATOR_H
