#include "params.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>

// ---------------------------------------------------------------------------
// The parameter table. One row per key: {key, type, default, section, level, allowed values, doc}.
// Defaults are the production values (config.txt of 2026-10-12) for every key that is not level user; a level-user
// key's default applies only when config.txt leaves it out. scripts/gen_config_md.py parses this table (keep one
// row per entry, string literals only) into docs/CONFIG.md and checks config.txt and the cfg.get_* calls against it.
// ---------------------------------------------------------------------------
namespace {
constexpr ParamType B = ParamType::Bool, I = ParamType::Int, D = ParamType::Double, S = ParamType::String;
constexpr ParamLevel USER = ParamLevel::User, EXPERT = ParamLevel::Expert, TEST = ParamLevel::Test,
                     INTERNAL = ParamLevel::Internal, EXPERIMENT = ParamLevel::Experiment;

const std::vector<ParamSpec> kParams = {
    // ---- Run ----
    {"TEST_CASE", S, "outer_planets", "Run", USER, "plummer|outer_planets|binary_test|binary_test_peri|triple_test|quad22_test|particle_file",
     "Initial conditions: a star cluster (plummer) or a few-body test problem (--test overrides)"},
    {"TOTAL_TIME_MYR", D, "0.23", "Run", USER, "", "End time (Myr)"},
    {"DT_MAX_MYR", D, "1.6e-3", "Run", USER, "",
     "Integration interval (Myr): the level-0 step; diagnostics, restarts and snapshots count these steps"},
    {"DT_MIN_MYR", D, "1.0e-14", "Run", EXPERT, "", "Smallest time step (Myr): the deepest HHS level"},
    {"INTEGRATOR", S, "hhs", "Run", INTERNAL, "hhs|fsi",
     "hhs: adaptive HHS-FSI (FROST Algorithm 4); fsi: every particle on the fastest occupied level (tests)"},

    // ---- Star cluster ----
    {"PLUMMER_N", I, "1000", "Star cluster", USER, "", "Number of stars"},
    {"PLUMMER_M_TOTAL", D, "3.79e2", "Star cluster", USER, "", "Total mass (Msun); not used by kroupa_evolved"},
    {"PLUMMER_R_HALF", S, "mass_size", "Star cluster", USER, "mass_size|<number>",
     "Half-mass radius (pc), or mass_size: the SCIC mass-size relation (BIFROST App. A eq. A2-A3)"},
    {"PLUMMER_MASS_DIST", S, "equal", "Star cluster", USER, "equal|kroupa|kroupa_evolved",
     "Masses: equal, Kroupa IMF (ZAMS), or Kroupa IMF evolved with SSE to PLUMMER_AGE_GYR"},
    {"PLUMMER_AGE_GYR", D, "1.0", "Star cluster", USER, "", "Age of the population for kroupa_evolved (Gyr)"},
    {"PLUMMER_METALLICITY", D, "0.02", "Star cluster", USER, "", "Metallicity Z of the population for kroupa_evolved"},
    {"PLUMMER_SEED", I, "42", "Star cluster", USER, "", "Random seed of the initial conditions"},
    {"REMOVE_BH", B, "false", "Star cluster", USER, "", "Strip the black holes from the initial conditions"},

    // ---- Primordial binaries ----
    {"BINARY_FRAC", D, "0.0", "Primordial binaries", USER, "", "Primordial binary fraction (0-1)"},
    {"BINARY_POP", S, "bifrost", "Primordial binaries", USER, "bifrost|mass_rank",
     "Pairing: bifrost (BIFROST App. A: flat q in [0.1, 1) above 5 Msun, random pairs below); mass_rank: "
     "heaviest with next heaviest"},
    {"BINARY_A_MIN_AU", D, "0.01", "Primordial binaries", USER, "", "Smallest semi-major axis (AU), log-uniform in a"},
    {"BINARY_A_MAX_AU", D, "100.0", "Primordial binaries", USER, "", "Largest semi-major axis (AU)"},
    {"R_BREAK_AU", D, "0.0", "Primordial binaries", USER, "",
     "Remove artificial binaries with a < r_break (AU) from the ICs (BIFROST App. A); 0: IC_R_BREAK"},
    {"IC_R_BREAK", D, "0.0", "Primordial binaries", EXPERT, "",
     "r_break in units of BINARY_A_MAX_AU (BIFROST Table A1: a few); used only when R_BREAK_AU = 0"},
    {"BINARY_PERI_CHECK", B, "true", "Primordial binaries", EXPERT, "",
     "Redraw e (or a) until r_p = a (1 - e) > R1 + R2 (TDE radius / ISCO with BHs), FROST-CLUSTERS II sec. 2.3"},

    // ---- IMBH binary ----
    {"ADD_IMBH", B, "false", "IMBH binary", USER, "", "Put an IMBH binary at the cluster centre"},
    {"IMBH_M1", D, "500.0", "IMBH binary", USER, "", "Primary mass (Msun)"},
    {"IMBH_M2", D, "500.0", "IMBH binary", USER, "", "Secondary mass (Msun)"},
    {"IMBH_A_PC", D, "1e-4", "IMBH binary", USER, "", "Semi-major axis (pc)"},
    {"IMBH_E", D, "0.5", "IMBH binary", USER, "", "Eccentricity"},
    {"IMBH_SPIN", D, "0.0", "IMBH binary", USER, "", "Dimensionless spin of both BHs (along z)"},

    // ---- Physics ----
    {"COLLISIONAL", B, "false", "Physics", USER, "",
     "Unsoftened direct N-body with subsystems, PN and mergers (BIFROST); false: softened N-body"},
    {"PN_ORDER", I, "7", "Physics", USER, "",
     "PN bitmask: 1 PN1, 2 PN2, 4 PN2.5, 8 PN3, 16 PN3.5, 32 BH spin couplings (63 all; 0 off)"},
    {"M_PN", D, "5.0", "Physics", EXPERT, "", "PN terms only for pairs with a member of at least this mass (Msun)"},
    {"PN_BH_ONLY", B, "true", "Physics", EXPERT, "",
     "PN terms only for black holes (stellar type 14) of at least M_PN (false: every body of at least M_PN, 3-5x slower young clusters)"},
    {"SOFTENING_PC", D, "1e-3", "Physics", EXPERT, "", "Plummer softening (pc); 0 with COLLISIONAL"},

    // ---- Stellar evolution ----
    {"SSE_ENABLED", B, "true", "Stellar evolution", USER, "", "Single stellar evolution during the run (COLLISIONAL)"},
    {"SSE_METALLICITY", D, "0.02", "Stellar evolution", USER, "", "Metallicity Z of the runtime SSE"},
    {"SSE_WIND_MODEL", S, "HURLEY", "Stellar evolution", USER, "HURLEY|HURLEY_LBV|VINK|VINK_NO_BISTAB",
     "Stellar winds (also for kroupa_evolved ICs)"},
    {"SSE_REMNANT_MODEL", S, "BELCZYNSKI02", "Stellar evolution", USER,
     "HURLEY|BELCZYNSKI02|BELCZYNSKI08|FRYER_RAPID|FRYER_DELAYED|ELDRIDGE_TOUT", "Remnant masses"},
    {"SSE_PPISN_MODEL", S, "NONE", "Stellar evolution", USER, "NONE|STRONG|MODERATE|WEAK",
     "Pulsational pair-instability mass loss"},
    {"SSE_ECSN_MODEL", S, "OFF", "Stellar evolution", USER, "OFF|ON", "Electron-capture supernovae"},
    {"SSE_MXNS", D, "-1.0", "Stellar evolution", USER, "",
     "Largest neutron-star mass (Msun); -1: 1.8 for HURLEY, 2.5 otherwise"},
    {"SSE_FLBV", D, "1.5", "Stellar evolution", USER, "", "LBV wind scaling factor"},
    {"SSE_NATAL_KICKS", S, "hobbs", "Stellar evolution", USER, "hobbs|none",
     "Natal kicks of new NSs and BHs: Maxwellian sigma = 265 km/s (Hobbs 2005) times (1 - f_fb), ECSN 3 km/s "
     "(Banerjee 2020); none: no kick"},

    // ---- Collisions and mergers ----
    {"COLL_MASS_LOSS_C", D, "0.0", "Collisions and mergers", USER, "",
     "Stellar-collision mass loss: m_rem = (m1 + m2)(1 - C q / (1 + q)^2) (Glebbeek 2013; FROST-CLUSTERS III: 0.3)"},
    {"TDE_ACCRETION_FRACTION", D, "0.5", "Collisions and mergers", USER, "",
     "Fraction of a tidally disrupted star the BH accretes (FROST-CLUSTERS I, DRAGON-II)"},
    {"BH_MERGE_RADIUS", S, "total", "Collisions and mergers", USER, "total|max_single",
     "BH-BH merger at r < 10 R_S = 20 G (m1 + m2) / c^2 (total) or max(6 G m_i / c^2) (max_single, FROST-CLUSTERS I)"},

    // ---- Escapers and tidal field ----
    {"R_ESCAPE_RH", D, "100.0", "Escapers and tidal field", EXPERT, "", "Escape radius in half-mass radii"},
    {"R_ESCAPE_RJ", D, "2.0", "Escapers and tidal field", EXPERT, "",
     "With TIDAL_FIELD: escape radius in Jacobi radii (NBODY6); 0: R_ESCAPE_RH"},
    {"TIDAL_FIELD", S, "none", "Escapers and tidal field", USER, "none|pointmass|loghalo",
     "Host galaxy tide on a cluster on a circular orbit (PeTar-style reference point)"},
    {"GALAXY_R_PC", D, "8000.0", "Escapers and tidal field", USER, "", "Galactocentric orbit radius (pc)"},
    {"GALAXY_VC_KMS", D, "220.0", "Escapers and tidal field", USER, "", "loghalo circular velocity (km/s)"},
    {"GALAXY_RC_PC", D, "0.0", "Escapers and tidal field", USER, "", "loghalo core radius (pc)"},
    {"GALAXY_MASS_MSUN", D, "0.0", "Escapers and tidal field", USER, "", "pointmass galaxy mass (Msun)"},

    // ---- Time steps ----
    {"ETA_FF", D, "0.2", "Time steps", EXPERT, "", "Free-fall time-step eta (BIFROST sec. 3.4)"},
    {"ETA_FB", D, "0.2", "Time steps", EXPERT, "", "Fly-by time-step eta"},
    {"ETA_GRAD", D, "0.2", "Time steps", EXPERT, "", "Gradient time-step eta (FROST-CLUSTERS; BIFROST 0.1)"},
    {"ETA_JERK", D, "0", "Time steps", EXPERT, "", "Jerk time-step eta (0: off, FROST-CLUSTERS; BIFROST 0.1)"},
    {"ETA", D, "0", "Time steps", EXPERT, "",
     "Velocity / acceleration time-step eta (0: off, FROST-CLUSTERS; BIFROST 0.2)"},
    {"ETA_PERT", D, "0.02", "Time steps", EXPERT, "", "Perturbation time-step eta of subsystems (BIFROST Eq. 30)"},
    {"ETA_GBS", D, "1e-10", "Time steps", EXPERT, "", "GBS extrapolation tolerance of the LogH integrator"},
    {"ETA_RESTART", D, "1e-8", "Time steps", EXPERT, "",
     "Energy-error threshold of the per-subsystem LogH restart and the step-level restart (0: off)"},
    {"N_ORB_SECULAR", I, "3", "Time steps", EXPERT, "",
     "Binaries with more orbits than this per step take the secular (orbit-averaged) path"},
    {"GRAD_FRESH", B, "false", "Time steps", EXPERIMENT, "", "Gradient time step from a fresh |g'| at level 0"},

    // ---- Subsystems ----
    {"R_NGB", D, "1e-3", "Subsystems", EXPERT, "", "Neighbour radius (pc) of the subsystem search"},
    {"R_NGB_MAX", D, "0.1", "Subsystems", EXPERT, "", "Largest separation (pc) of the orbital-element check"},
    {"SECULAR_TRIPLES", B, "true", "Subsystems", EXPERIMENT, "",
     "Stable triples with >= N_ORB_SECULAR inner orbits per step take the double-averaged secular path; false: LogH"},
    {"SDAR", B, "true", "Subsystems", INTERNAL, "",
     "Slow-down of weakly perturbed inner binaries in LogH (Wang+2020; FROST-CLUSTERS I App. A2); env NBODY_SDAR "
     "overrides"},
    {"SDAR_KREF", D, "1e-6", "Subsystems", EXPERIMENT, "", "SDAR k_ref of the slow-down factor (W20 Eq. 22)"},
    {"SDAR_C", D, "0.1", "Subsystems", EXPERIMENT, "", "SDAR kappa_max coefficient c (W20 Eq. 28)"},
    {"GBS_SEQUENCE", S, "bulirsch", "Subsystems", EXPERIMENT, "bulirsch|deuflhard",
     "GBS substep sequence: bulirsch (2, 4, 6, 8, 12, 16, ...) or deuflhard (2, 4, 6, 8, 10, ...; MSTAR)"},
    {"GBS_KMAX", I, "10", "Subsystems", EXPERIMENT, "", "GBS extrapolation stages (FROST-CLUSTERS I: 8)"},
    {"FLYBY_CM_VELOCITY", B, "false", "Subsystems", EXPERIMENT, "",
     "Fly-by step of subsystem members with the c.o.m. velocity (FROST-CLUSTERS II sec. 2.1)"},
    {"FLYBY_CM_EPS", D, "1e-11", "Subsystems", INTERNAL, "",
     "Tidal-sampling pair step: energy a slow particle's kicks put into a subsystem per step <= this x |E0| (0: off)"},
    {"TIDAL_STEP_RES", D, "1.0", "Subsystems", INTERNAL, "",
     "Floor of the FLYBY_CM_EPS step: this x P (1 - e)^(3/2) of a bound binary, this x the shortest pair pericentre passage 2 pi r_p / v_p of an unbound pair or larger subsystem (0: none)"},
    {"FLYBY_NGB_CM", B, "true", "Subsystems", INTERNAL, "",
     "Distant fly-by neighbour test of a binary member against the binary's c.o.m."},
    {"NGB_MASS_SCALING", B, "true", "Subsystems", INTERNAL, "",
     "Pair neighbour radius r_ngb max(1, m / <m>)^(1/3) (PeTar Eq. 12); the extra pairs join only as strong "
     "perturbers of a binary"},
    {"WEAK_SPLIT_GAMMA", D, "0", "Subsystems", EXPERT, "",
     "Fly-by or unstable triple whose third body perturbs the inner binary below this tidal ratio: binary subsystem + field star (0: off; 1e-6: 35x faster stalled steps, |dE/E| per step up to 30x worse)"},
    {"NGB_MASS_GAMMA", D, "1e-3", "Subsystems", INTERNAL, "",
     "Tidal ratio above which a mass-scaled neighbour joins a binary"},
    {"NGB_MREF", D, "0.0", "Subsystems", TEST, "", "<m> of NGB_MASS_SCALING (Msun) instead of the cluster's (0)"},
    {"E_REF", D, "0.0", "Subsystems", TEST, "",
     "|E0| of FLYBY_CM_EPS (Msun (km/s)^2) instead of the run's (0): a cut-out of a cluster as in the cluster"},
    {"NGB_WIDE_FAC", D, "0", "Subsystems", EXPERIMENT, "",
     "Bound pairs with a < NGB_WIDE_FAC x R_NGB are neighbours (wide binaries; 0: off, BIFROST)"},
    {"MSTAR_THRESHOLD", I, "8", "Subsystems", EXPERIMENT, "",
     "Subsystems of at least this many bodies for MSTAR (not wired: printed only, every subsystem uses LogH)"},

    // ---- Output ----
    {"PRINT_INTERVAL", I, "1000", "Output", USER, "", "Diagnostics every this many steps"},
    {"PROFILING", B, "true", "Output", USER, "", "Timers and the performance breakdown at each print"},
    {"PROFILING_FENCES", B, "true", "Output", INTERNAL, "",
     "Fence the device at each timer stop (exact per-component times, ~4 % of a 200k step)"},
    {"RESTART_INTERVAL", I, "0", "Output", USER, "",
     "Restart file every this many steps and at the end (0: off; needs USE_HDF5)"},
    {"WALLTIME_LIMIT", D, "0.0", "Output", USER, "", "Hours: restart file and clean stop before this wall time (0: off)"},
    {"RESTART_DIR", S, ".", "Output", USER, "", "Directory of restart_<step>.h5"},
    {"RESTART_FILE", S, "", "Output", USER, "",
     "Continue from this file; latest: newest in RESTART_DIR; auto: that or a fresh start (--restart overrides)"},
    {"RESTART_KEEP", I, "2", "Output", EXPERT, "", "Restart files kept (the newest)"},
    {"STALL_FACTOR", D, "5", "Output", EXPERT, "",
     "A step slower than this x the median of the previous steps prints STALL lines: the costliest LogH integrations and the deepest set (0: off)"},
    {"STALL_DE_FACTOR", D, "10", "Output", EXPERT, "",
     "A step whose energy change (ETA_RESTART check) exceeds this x the median of the previous steps prints STALL lines (0: off)"},
    {"SUBSYS_ERR_TOL", D, "1e-11", "Output", EXPERT, "",
     "A subsystem advance (LogH, secular binary or triple) with unbooked |dE| above this x |E0| prints a SUBSYS_ERR line with its members' state (scripts/subsys_cases.py; 0: off)"},
    {"STALL_RESTART", B, "false", "Output", EXPERT, "",
     "Restart file before every step (RESTART_DIR/stall_roll), kept as RESTART_DIR/stall_<step>.h5 for a stalled step (scripts/cutout.py)"},
    {"SNAPSHOT_INTERVAL", I, "0", "Output", USER, "",
     "LUNA-layout snapshot every this many steps, at step 0 and at the end (0: off)"},
    {"SNAPSHOT_DIR", S, ".", "Output", USER, "", "Directory of snapdir_NNN/snapshot_NNN.0.hdf5"},
    {"SNAPSHOT_BASE", S, "snapshot", "Output", EXPERT, "", "Snapshot file base name"},
    {"SNAPSHOT_POTENTIAL", B, "true", "Output", EXPERT, "", "Write the potential of each particle into snapshots"},
    {"SNAPSHOT_CLUSTER_DIAG", B, "true", "Output", EXPERT, "",
     "Density centre, core, Lagrangian radii, binaries (and r_J) in each snapshot Header"},
    {"CLUSTER_DIAG_INTERVAL", I, "0", "Output", USER, "",
     "Cluster diagnostics (density centre, core, Lagrangian radii, binaries) every this many steps (0: off)"},
    {"CLUSTER_DIAG_FILE", S, "cluster_diag.dat", "Output", EXPERT, "", "File of the cluster diagnostics (rank 0)"},
    {"CLUSTER_DIAG_RPAIR_PC", D, "0.02", "Output", EXPERT, "", "Largest pair separation in the binary census (pc)"},

    // ---- Performance and parallelism ----
    {"SERIAL_N2_GPU", I, "16384", "Performance and parallelism", INTERNAL, "",
     "Below this count: full N^2 on every rank, no MPI distribution (0: always distributed)"},
    {"HOST_SUBTREE_MAX", I, "64", "Performance and parallelism", INTERNAL, "",
     "Sets of at most this many particles run their whole HHS sub-recursion on the host (one copy in and out; 0: off; "
     "run B 630 -> 240 s/step, 2026-10-16)"},
    {"REPLICA_CHECK_INTERVAL", I, "10", "Performance and parallelism", INTERNAL, "",
     "Distributed runs: check every this many steps that the replicas agree (0: off)"},
    {"MPI_NON_BLOCKING", B, "false", "Performance and parallelism", INTERNAL, "",
     "MPI_Iallgatherv for overlapped transfers"},
    {"MPI_TRANSPORT", S, "auto", "Performance and parallelism", INTERNAL, "auto|nccl|host",
     "Allgathers: nccl (USE_NCCL builds; auto there), host (staged through pinned host memory; auto otherwise)"},
    {"USE_TREE", B, "false", "Performance and parallelism", INTERNAL, "",
     "Barnes-Hut tree forces above TREE_THRESHOLD (not with COLLISIONAL)"},
    {"TREE_THRESHOLD", I, "100000", "Performance and parallelism", INTERNAL, "", "Tree forces above this count"},
    {"TREE_THETA", D, "0.7", "Performance and parallelism", INTERNAL, "", "Tree opening angle"},
    {"TREE_N_LEAF", I, "32", "Performance and parallelism", INTERNAL, "", "Largest leaf cell"},
    {"TREE_N_GROUP", I, "64", "Performance and parallelism", INTERNAL, "", "Particles per tree traversal group"},
    {"TREE_DT_MIN_N", I, "-1", "Performance and parallelism", INTERNAL, "",
     "Exact device-tree time-step search for sets of at least this many (0: off; -1: host builds 2048, GPU 65536, "
     "300000 with FLYBY_CM_VELOCITY)"},
    {"TREE_NGB_MIN_N", I, "-1", "Performance and parallelism", INTERNAL, "",
     "The same for the neighbour search (-1: host builds 2048, GPU 32768)"},
    {"TREE_DT_CPU_MAX", I, "0", "Performance and parallelism", INTERNAL, "",
     "Host-thread tree search for sets up to this size (host builds / APUs; 0: off)"},
    {"TREE_DT_VC_FAC", D, "30.0", "Performance and parallelism", INTERNAL, "",
     "Low-speed split of the tree time-step search (multiple of V_c)"},
    {"HHS_CC", B, "false", "Performance and parallelism", EXPERIMENT, "",
     "Fast HHS sets split into connected components (Janes et al. 2014)"},
    {"HHS_CC_MAX", I, "-1", "Performance and parallelism", EXPERIMENT, "",
     "Largest set split, run locally on every rank (-1: max(SERIAL_N2_GPU, 16384))"},
    {"HHS_CC_MIN", I, "0", "Performance and parallelism", EXPERIMENT, "",
     "Components smaller than this recurse together as one group (0: off)"},
    {"INTEGER_POSITIONS", B, "false", "Performance and parallelism", EXPERIMENT, "",
     "int64 positions (exact drift, MPI-safe)"},
    {"DOMAIN_HALFSIZE_PC", D, "200", "Performance and parallelism", EXPERIMENT, "",
     "Half-size of the INTEGER_POSITIONS domain (pc)"},

    // ---- Test problems ----
    {"BINARY_A_PC", D, "1e-8", "Test problems", TEST, "", "binary_test: semi-major axis (pc)"},
    {"BINARY_E", D, "0.9", "Test problems", TEST, "", "binary_test: eccentricity"},
    {"BINARY_M1", D, "10.0", "Test problems", TEST, "", "binary_test: primary mass (Msun)"},
    {"BINARY_M2", D, "10.0", "Test problems", TEST, "", "binary_test: secondary mass (Msun)"},
    {"BINARY_ROT_DEG", D, "0.0", "Test problems", TEST, "", "binary_test: rotation of the orbit in its plane (deg)"},
    {"BINARY_CHI1", S, "", "Test problems", TEST, "", "binary_test: dimensionless spin vector \"cx,cy,cz\" of body 1"},
    {"BINARY_CHI2", S, "", "Test problems", TEST, "", "binary_test: dimensionless spin vector of body 2"},
    {"TRIPLE_A_IN_PC", D, "1e-8", "Test problems", TEST, "", "triple_test: inner semi-major axis (pc)"},
    {"TRIPLE_E_IN", D, "0.1", "Test problems", TEST, "", "triple_test: inner eccentricity"},
    {"TRIPLE_M1", D, "10.0", "Test problems", TEST, "", "triple_test: inner primary mass (Msun)"},
    {"TRIPLE_M2", D, "10.0", "Test problems", TEST, "", "triple_test: inner secondary mass (Msun)"},
    {"TRIPLE_A_OUT_PC", D, "5e-8", "Test problems", TEST, "", "triple_test: outer semi-major axis (pc)"},
    {"TRIPLE_E_OUT", D, "0.1", "Test problems", TEST, "", "triple_test: outer eccentricity"},
    {"TRIPLE_M3", D, "10.0", "Test problems", TEST, "", "triple_test: outer body mass (Msun)"},
    {"QUAD_M1", D, "20.7", "Test problems", TEST, "", "quad22_test: binary A primary (Msun)"},
    {"QUAD_M2", D, "14.1", "Test problems", TEST, "", "quad22_test: binary A secondary (Msun)"},
    {"QUAD_A1_PC", D, "2.82e-4", "Test problems", TEST, "", "quad22_test: binary A semi-major axis (pc)"},
    {"QUAD_E1", D, "0.853", "Test problems", TEST, "", "quad22_test: binary A eccentricity"},
    {"QUAD_M3", D, "0.791", "Test problems", TEST, "", "quad22_test: binary B primary (Msun)"},
    {"QUAD_M4", D, "0.105", "Test problems", TEST, "", "quad22_test: binary B secondary (Msun)"},
    {"QUAD_A2_PC", D, "9.0e-8", "Test problems", TEST, "", "quad22_test: binary B semi-major axis (pc)"},
    {"QUAD_E2", D, "0.62", "Test problems", TEST, "", "quad22_test: binary B eccentricity"},
    {"QUAD_A_OUT_PC", D, "2.7e-3", "Test problems", TEST, "", "quad22_test: outer orbit semi-major axis (pc)"},
    {"QUAD_E_OUT", D, "0.48", "Test problems", TEST, "", "quad22_test: outer orbit eccentricity"},
    {"IC_FILE", S, "", "Test problems", TEST, "", "particle_file: \"m x y z vx vy vz [kw [R]]\" per line (Msun, pc, km/s, R_sun)"},
    {"QUAD_INCL_DEG", D, "30.0", "Test problems", TEST, "", "quad22_test: inclination of binary B's plane (deg)"},
    {"TRIPLE_INCL_DEG", D, "45.0", "Test problems", TEST, "", "triple_test: mutual inclination (deg)"},
    {"TRIPLE_OUTER_FIRST", B, "false", "Test problems", TEST, "", "triple_test: outer body at the lowest particle id"},
    {"TRIPLE_IN_ROT_DEG", D, "0.0", "Test problems", TEST, "", "triple_test: rotation of the inner orbit (deg)"},
    {"TRIPLE_IN_START_Q", D, "5.0", "Test problems", TEST, "",
     "triple_test, unbound inner orbit: start at this many pericentre distances"},
    {"TRIPLE_OUT_START_Q", D, "5.0", "Test problems", TEST, "",
     "triple_test, unbound outer orbit: start at this many pericentre distances"},
    {"TEST_STELLAR_RADIUS_RSUN", D, "0.0", "Test problems", TEST, "",
     "Few-body tests: stellar radius of every non-BH body (Rsun; 0: none)"},
    {"TEST_BOOST_KMS", S, "", "Test problems", TEST, "",
     "Few-body tests: uniform velocity \"vx,vy,vz\" (km/s) added to every body"},
};

// NBODY_* environment switches (debugging and A/B checks; not config keys)
const std::vector<DebugEnvSpec> kDebugEnv = {
    {"NBODY_CHECK_CC", "Check that the connected-component split sees the pair steps of the time-step kernel"},
    {"NBODY_CHECK_DT_FP32", "=1: check the fp32 time-step kernels against fp64"},
    {"NBODY_CHECK_HOSTSUB", "=1: abort on a device pointer met inside a host subtree (HOST_SUBTREE_MAX)"},
    {"NBODY_CHECK_EIH", "Check that every LogH PN evaluation (cached geometry) equals the direct EIH functions bit for bit"},
    {"NBODY_CHECK_PIDMAP", "Check every incremental pid-map update against a full rebuild"},
    {"NBODY_CHECK_STELLAR_CACHE", "Check every cached stellar type and radius against the device values"},
    {"NBODY_CHECK_SUBSYS", "Check that every subsystem is advanced exactly once per subsystem call"},
    {"NBODY_CHECK_TREE_DT", "=1: check every tree time-step / neighbour search against O(N^2)"},
    {"NBODY_KICK_U", "A/B (C12): LogH kick with 1/(-U) (BIFROST, symplectic) instead of 1/(T+B)"},
    {"NBODY_LOGH_KAHAN", "A/B: compensated (Kahan) x, v, t sums in the Newtonian LogH leapfrog chain"},
    {"NBODY_DEEP_LOG", "=d: log the particles and pairs that keep HHS sets at depth >= d"},
    {"NBODY_FENCE_STATS", "=1/2: count device fences and copies per call site (fence_stats.h; NVTX ranges for nsys)"},
    {"NBODY_FORCE_ERR", "=eps: synthetic relative force error before every field kick of HHS sets >= NBODY_FORCE_ERR_MIN (FMM accuracy study; not bit-identical)"},
    {"NBODY_FORCE_ERR_MIN", "=n: smallest HHS set that gets NBODY_FORCE_ERR (default 16384)"},
    {"NBODY_FORCE_ERR_MODEL", "=random (default; eps |a_i| times an isotropic unit vector per particle and call) / frozen (the same vector per particle at every call) / smooth (eps a_rms times a random linear field in position over the set, per call)"},
    {"NBODY_MERGE_CHECK", "=1: total energy and books before and after each per-level merger (merger energy booking check)"},
    {"NBODY_LEAN_SYNC", "=0: keep every device fence of the original code (default on: lean copies, bit-identical)"},
    {"NBODY_DUMP", "Particle state at each print to dump_<step>.bin"},
    {"NBODY_DUMP_DT", "=n: pid, dt_adapt, grad_mag, ax of the set after the time-step criteria, first n calls"},
    {"NBODY_EATTR", "Per-print energy attribution"},
    {"NBODY_ELEDGER", "=1: energy ledger of the HHS recursion (error per call and level; one rank, replays only)"},
    {"NBODY_ECHECK_CMP", "Energy check: the split sum against the replicated whole-range sum"},
    {"NBODY_PE_STREAMS", "=0: the potential-energy blocks one after the other on the default stream (GPU default: concurrent streams, same bits)"},
    {"NBODY_PE_REUSE", "=0: the print diagnostics recompute the energy check's potential energy (same bits); =check: recompute and compare"},
    {"NBODY_ESPLIT", "Total energy and the internal energy of a fixed list of binaries at each print"},
    {"NBODY_GBS_DYNCAP", "Initial LogH step: t_dyn / this (default 3)"},
    {"NBODY_GBS_STAGE_PAR", "Batches of up to this many subsystems run their GBS stages on host threads (default 2)"},
    {"NBODY_GBS_STATS", "GBS step and stage statistics"},
    {"NBODY_GBS_STEPS_PER_ORBIT", "LogH GBS step limit per orbit (default 64)"},
    {"NBODY_GBS_TTOL", "Relative tolerance of the LogH end time (default 1e-4)"},
    {"NBODY_LOGH_SLOW", "=k: name every LogH integration that took at least k GBS steps"},
    {"NBODY_SDAR", "=0/1: overrides SDAR"},
    {"NBODY_SDAR_PN", "=1: SDAR slow-down of inner binaries without a PN member inside PN subsystems (PLAN item 4c; A/B)"},
    {"NBODY_SLOW0_STATS", "Statistics of small all-fast HHS sets, printed at the end"},
    {"NBODY_SMALL_FOR", "Largest loop run on the host where it can access device memory (default 256; 0: off)"},
    {"NBODY_SMALL_N2", "Largest pair count run on the host where it can access device memory (default 4096)"},
    {"NBODY_TILE256_MIN", "N^2 kernel: 256-thread tiles from this set size (GPUs; default 131072)"},
    {"NBODY_N2_P", "N^2 kernels: force P = 1, 2, 4 threads per sink (default per kernel, results identical)"},
    {"NBODY_N2_R", "N^2 kernels: force teams of TS / R threads, R = 1, 2 (default per kernel, results identical)"},
};

const char* type_name(ParamType t) {
    switch (t) {
        case ParamType::Bool: return "bool";
        case ParamType::Int: return "int";
        case ParamType::Double: return "double";
        default: return "string";
    }
}

[[noreturn]] void config_error(const std::string& msg) {
    std::fprintf(stderr, "CONFIG ERROR: %s\n", msg.c_str());
    std::exit(1);
}

size_t edit_distance(const std::string& a, const std::string& b) {
    std::vector<size_t> d(b.size() + 1);
    for (size_t j = 0; j <= b.size(); j++) d[j] = j;
    for (size_t i = 1; i <= a.size(); i++) {
        size_t prev = d[0];
        d[0] = i;
        for (size_t j = 1; j <= b.size(); j++) {
            const size_t cur = d[j];
            d[j] = std::min({d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] == b[j - 1] ? 0 : 1)});
            prev = cur;
        }
    }
    return d[b.size()];
}

std::string nearest_key(const std::string& key) {
    const char* best = "";
    size_t best_d = SIZE_MAX;
    for (const auto& p : kParams) {
        const size_t d = edit_distance(key, p.key);
        if (d < best_d) { best_d = d; best = p.key; }
    }
    return best;
}

bool parse_bool(const std::string& v, bool& out) {
    if (v == "true" || v == "TRUE" || v == "True" || v == "1" || v == "yes") { out = true; return true; }
    if (v == "false" || v == "FALSE" || v == "False" || v == "0" || v == "no") { out = false; return true; }
    return false;
}

bool parse_int(const std::string& v, int& out) {
    try {
        size_t pos = 0;
        out = std::stoi(v, &pos);
        return pos == v.size();
    } catch (const std::exception&) {
        return false;
    }
}

bool parse_double(const std::string& v, double& out) {
    try {
        size_t pos = 0;
        out = std::stod(v, &pos);
        return pos == v.size();
    } catch (const std::exception&) {
        return false;
    }
}

// Empty when `value` is a valid value of `p`, else the reason
std::string check_value(const ParamSpec& p, const std::string& value) {
    bool b; int i; double d;
    switch (p.type) {
        case ParamType::Bool: return parse_bool(value, b) ? "" : "expected true or false";
        case ParamType::Int: return parse_int(value, i) ? "" : "expected an integer";
        case ParamType::Double: return parse_double(value, d) ? "" : "expected a number";
        default: break;
    }
    if (!*p.values) return "";
    const std::string vals = p.values;
    size_t s = 0;
    while (s <= vals.size()) {
        size_t e = vals.find('|', s);
        if (e == std::string::npos) e = vals.size();
        const std::string opt = vals.substr(s, e - s);
        if (opt == value || (opt == "<number>" && parse_double(value, d))) return "";
        s = e + 1;
    }
    return std::string("expected one of ") + p.values;
}
}  // namespace

const std::vector<ParamSpec>& param_table() { return kParams; }
const std::vector<DebugEnvSpec>& debug_env_table() { return kDebugEnv; }

const ParamSpec* find_param(const std::string& key) {
    static const std::map<std::string, const ParamSpec*> index = [] {
        std::map<std::string, const ParamSpec*> m;
        for (const auto& p : kParams) {
            if (!m.emplace(p.key, &p).second) config_error(std::string("duplicate table entry ") + p.key);
            // The table's own defaults must be valid values
            const std::string why = check_value(p, p.def);
            if (!why.empty()) config_error(std::string("table default of ") + p.key + ": " + why);
        }
        return m;
    }();
    auto it = index.find(key);
    return it == index.end() ? nullptr : it->second;
}

void Config::load(const std::string& file, bool verbose) {
    filename = file;
    std::ifstream f(file);
    if (!f.is_open()) {
        std::fprintf(stderr, "Cannot open config file: %s\n", file.c_str());
        std::exit(1);
    }
    auto trim = [](std::string& s) {
        s.erase(0, s.find_first_not_of(" \t\r\n"));
        s.erase(s.find_last_not_of(" \t\r\n") + 1);
    };
    std::vector<std::string> errors;
    std::string line;
    int lineno = 0;
    while (std::getline(f, line)) {
        lineno++;
        text += line + "\n";
        auto pos = line.find('#');
        if (pos != std::string::npos) line.erase(pos);
        pos = line.find('=');
        if (pos == std::string::npos) {
            trim(line);
            if (!line.empty())
                errors.push_back(file + ":" + std::to_string(lineno) + ": not KEY = value: '" + line + "'");
            continue;
        }
        std::string key = line.substr(0, pos);
        std::string val = line.substr(pos + 1);
        trim(key); trim(val);
        if (key.empty()) continue;
        const std::string where = file + ":" + std::to_string(lineno) + ": ";
        const ParamSpec* p = find_param(key);
        if (!p) {
            errors.push_back(where + "unknown key " + key + " (did you mean " + nearest_key(key) + "?)");
            continue;
        }
        if (params.count(key)) errors.push_back(where + key + " set twice");
        const std::string why = check_value(*p, val);
        if (!why.empty()) errors.push_back(where + key + " = '" + val + "': " + why);
        params[key] = val;
    }
    if (!errors.empty()) {
        if (verbose) {
            for (const auto& e : errors) std::fprintf(stderr, "CONFIG ERROR: %s\n", e.c_str());
            std::fprintf(stderr, "Valid keys, types and defaults: docs/CONFIG.md\n");
        }
        std::exit(1);
    }
}

void Config::set_override(const std::string& key, const std::string& value) {
    const ParamSpec* p = find_param(key);
    if (!p) config_error("override of unknown key " + key);
    const std::string why = check_value(*p, value);
    if (!why.empty()) config_error(key + " = '" + value + "': " + why);
    overrides[key] = value;
}

const std::string& Config::raw(const std::string& key, ParamType type) const {
    const ParamSpec* p = find_param(key);
    if (!p) {
        std::fprintf(stderr, "INTERNAL ERROR: config key %s is not in the parameter table (src/core/params.cpp)\n",
                     key.c_str());
        std::abort();
    }
    if (p->type != type) {
        std::fprintf(stderr, "INTERNAL ERROR: config key %s is a %s, read as a %s\n", key.c_str(), type_name(p->type),
                     type_name(type));
        std::abort();
    }
    auto o = overrides.find(key);
    if (o != overrides.end()) return o->second;
    auto it = params.find(key);
    if (it != params.end()) return it->second;
    static thread_local std::string def;
    def = p->def;
    return def;
}

std::string Config::get_str(const std::string& key) const { return raw(key, ParamType::String); }

double Config::get_double(const std::string& key) const { return to_double(key, raw(key, ParamType::Double)); }

int Config::get_int(const std::string& key) const {
    int v = 0;
    if (!parse_int(raw(key, ParamType::Int), v)) config_error(key + ": expected an integer");
    return v;
}

bool Config::get_bool(const std::string& key) const {
    bool v = false;
    if (!parse_bool(raw(key, ParamType::Bool), v)) config_error(key + ": expected true or false");
    return v;
}

double Config::to_double(const std::string& key, const std::string& s) {
    double v = 0.0;
    if (!parse_double(s, v)) config_error(key + " = '" + s + "': expected a number");
    return v;
}

void Config::write_effective(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "WARNING: cannot write %s\n", path.c_str());
        return;
    }
    std::fprintf(f, "# Effective parameters of this run (config file: %s); docs/CONFIG.md describes them\n",
                 filename.c_str());
    const char* section = "";
    for (const auto& p : kParams) {
        if (std::string(section) != p.section) {
            section = p.section;
            std::fprintf(f, "\n# ---- %s ----\n", section);
        }
        const char* from = "default";
        std::string v = p.def;
        if (overrides.count(p.key)) { from = "command line"; v = overrides.at(p.key); }
        else if (params.count(p.key)) { from = "file"; v = params.at(p.key); }
        std::fprintf(f, "%-26s = %-20s # %s\n", p.key, v.c_str(), from);
    }
    std::fclose(f);
}
