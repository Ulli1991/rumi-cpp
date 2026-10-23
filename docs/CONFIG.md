# Config reference

All 153 keys of the config file (`config.txt`, or `--config <file>`), generated from the parameter table in `src/core/params.cpp` by `scripts/gen_config_md.py`. Every key has one default, in that table; an unknown key in a config file is an error. Each run writes the effective value of every key, and whether it came from the file or the default, to `run_params.txt`.

Levels:

- **user**: the choices of a run; these 51 keys are in `config.txt`.
- **expert**: accuracy and output knobs; the default is the production value.
- **test**: parameters of the few-body test problems.
- **internal**: performance tuning thresholds; results do not depend on them (or only through the reduction order).
- **experiment**: open or decided A/B switches; temporary, to be removed once decided.

## Run

| key | type | default | level | values | description |
|---|---|---|---|---|---|
| `TEST_CASE` | string | `outer_planets` | user | plummer\|outer_planets\|binary_test\|binary_test_peri\|triple_test\|quad22_test\|particle_file | Initial conditions: a star cluster (plummer) or a few-body test problem (--test overrides) |
| `TOTAL_TIME_MYR` | double | `0.23` | user |  | End time (Myr) |
| `DT_MAX_MYR` | double | `1.6e-3` | user |  | Integration interval (Myr): the level-0 step; diagnostics, restarts and snapshots count these steps |
| `DT_MIN_MYR` | double | `1.0e-14` | expert |  | Smallest time step (Myr): the deepest HHS level |
| `INTEGRATOR` | string | `hhs` | internal | hhs\|fsi | hhs: adaptive HHS-FSI (FROST Algorithm 4); fsi: every particle on the fastest occupied level (tests) |

## Star cluster

| key | type | default | level | values | description |
|---|---|---|---|---|---|
| `PLUMMER_N` | int | `1000` | user |  | Number of stars |
| `PLUMMER_M_TOTAL` | double | `3.79e2` | user |  | Total mass (Msun); not used by kroupa_evolved |
| `PLUMMER_R_HALF` | string | `mass_size` | user | mass_size\|a number | Half-mass radius (pc), or mass_size: the SCIC mass-size relation (Rantala et al. 2023, App. A eq. A2-A3) |
| `PLUMMER_MASS_DIST` | string | `equal` | user | equal\|kroupa\|kroupa_evolved | Masses: equal, Kroupa IMF (ZAMS), or Kroupa IMF evolved with SSE to PLUMMER_AGE_GYR |
| `PLUMMER_AGE_GYR` | double | `1.0` | user |  | Age of the population for kroupa_evolved (Gyr) |
| `PLUMMER_METALLICITY` | double | `0.02` | user |  | Metallicity Z of the population for kroupa_evolved |
| `PLUMMER_SEED` | int | `42` | user |  | Random seed of the initial conditions |
| `REMOVE_BH` | bool | `false` | user |  | Strip the black holes from the initial conditions |

## Primordial binaries

| key | type | default | level | values | description |
|---|---|---|---|---|---|
| `BINARY_FRAC` | double | `0.0` | user |  | Primordial binary fraction (0-1) |
| `BINARY_POP` | string | `flatq_massive` | user | flatq_massive\|mass_rank\|bifrost | Pairing: flatq_massive (Rantala et al. 2023, App. A: flat q in [0.1, 1) above 5 Msun, random pairs below); mass_rank: heaviest with next heaviest; bifrost: deprecated alias of flatq_massive |
| `BINARY_A_MIN_AU` | double | `0.01` | user |  | Smallest semi-major axis (AU), log-uniform in a |
| `BINARY_A_MAX_AU` | double | `100.0` | user |  | Largest semi-major axis (AU) |
| `R_BREAK_AU` | double | `0.0` | user |  | Remove artificial binaries with a < r_break (AU) from the ICs (Rantala et al. 2023, App. A); 0: IC_R_BREAK |
| `IC_R_BREAK` | double | `0.0` | expert |  | r_break in units of BINARY_A_MAX_AU (Rantala et al. 2023, Table A1: a few); used only when R_BREAK_AU = 0 |
| `BINARY_PERI_CHECK` | bool | `true` | expert |  | Redraw e (or a) until r_p = a (1 - e) > R1 + R2 (TDE radius / ISCO with BHs), FROST-CLUSTERS II sec. 2.3 |

## IMBH binary

| key | type | default | level | values | description |
|---|---|---|---|---|---|
| `ADD_IMBH` | bool | `false` | user |  | Put an IMBH binary at the cluster centre |
| `IMBH_M1` | double | `500.0` | user |  | Primary mass (Msun) |
| `IMBH_M2` | double | `500.0` | user |  | Secondary mass (Msun) |
| `IMBH_A_PC` | double | `1e-4` | user |  | Semi-major axis (pc) |
| `IMBH_E` | double | `0.5` | user |  | Eccentricity |
| `IMBH_SPIN` | double | `0.0` | user |  | Dimensionless spin of both BHs (along z) |

## Physics

| key | type | default | level | values | description |
|---|---|---|---|---|---|
| `COLLISIONAL` | bool | `false` | user |  | Unsoftened direct N-body with subsystems, PN and mergers; false: softened N-body |
| `PN_ORDER` | int | `7` | user |  | PN bitmask: 1 PN1, 2 PN2, 4 PN2.5, 8 PN3, 16 PN3.5, 32 BH spin couplings (63 all; 0 off) |
| `M_PN` | double | `5.0` | expert |  | PN terms only for pairs with a member of at least this mass (Msun) |
| `PN_BH_ONLY` | bool | `true` | expert |  | PN terms only for black holes (stellar type 14) of at least M_PN (false: every body of at least M_PN, 3-5x slower young clusters) |
| `SOFTENING_PC` | double | `1e-3` | expert |  | Plummer softening (pc); 0 with COLLISIONAL |

## Stellar evolution

| key | type | default | level | values | description |
|---|---|---|---|---|---|
| `SSE_ENABLED` | bool | `true` | user |  | Single stellar evolution during the run (COLLISIONAL) |
| `SSE_METALLICITY` | double | `0.02` | user |  | Metallicity Z of the runtime SSE |
| `SSE_WIND_MODEL` | string | `HURLEY` | user | HURLEY\|HURLEY_LBV\|VINK\|VINK_NO_BISTAB | Stellar winds (also for kroupa_evolved ICs) |
| `SSE_REMNANT_MODEL` | string | `BELCZYNSKI02` | user | HURLEY\|BELCZYNSKI02\|BELCZYNSKI08\|FRYER_RAPID\|FRYER_DELAYED\|ELDRIDGE_TOUT | Remnant masses |
| `SSE_PPISN_MODEL` | string | `NONE` | user | NONE\|STRONG\|MODERATE\|WEAK | Pulsational pair-instability mass loss |
| `SSE_ECSN_MODEL` | string | `OFF` | user | OFF\|ON | Electron-capture supernovae |
| `SSE_MXNS` | double | `-1.0` | user |  | Largest neutron-star mass (Msun); -1: 1.8 for HURLEY, 2.5 otherwise |
| `SSE_FLBV` | double | `1.5` | user |  | LBV wind scaling factor |
| `SSE_NATAL_KICKS` | string | `hobbs` | user | hobbs\|none | Natal kicks of new NSs and BHs: Maxwellian sigma = 265 km/s (Hobbs 2005) times (1 - f_fb), ECSN 3 km/s (Banerjee 2020); none: no kick |

## Collisions and mergers

| key | type | default | level | values | description |
|---|---|---|---|---|---|
| `COLL_MASS_LOSS_C` | double | `0.0` | user |  | Stellar-collision mass loss: m_rem = (m1 + m2)(1 - C q / (1 + q)^2) (Glebbeek 2013; FROST-CLUSTERS III: 0.3) |
| `TDE_ACCRETION_FRACTION` | double | `0.5` | user |  | Fraction of a tidally disrupted star the BH accretes (FROST-CLUSTERS I, DRAGON-II) |
| `BH_MERGE_RADIUS` | string | `total` | user | total\|max_single | BH-BH merger at r < 10 R_S = 20 G (m1 + m2) / c^2 (total) or max(6 G m_i / c^2) (max_single, FROST-CLUSTERS I) |
| `GW_CENSUS_TGW_MYR` | double | `13800` | user |  | gw_events.dat: at every snapshot also the bound NS/BH pairs of the subsystems with Peters merger time below this (Myr; LISA band, ejected binaries); 0: merger rows only |

## Escapers and tidal field

| key | type | default | level | values | description |
|---|---|---|---|---|---|
| `R_ESCAPE_RH` | double | `100.0` | expert |  | Escape radius in half-mass radii |
| `R_ESCAPE_RJ` | double | `2.0` | expert |  | With TIDAL_FIELD: escape radius in Jacobi radii (NBODY6); 0: R_ESCAPE_RH |
| `TIDAL_FIELD` | string | `none` | user | none\|pointmass\|loghalo | Host galaxy tide on a cluster on a circular orbit (PeTar-style reference point) |
| `GALAXY_R_PC` | double | `8000.0` | user |  | Galactocentric orbit radius (pc) |
| `GALAXY_VC_KMS` | double | `220.0` | user |  | loghalo circular velocity (km/s) |
| `GALAXY_RC_PC` | double | `0.0` | user |  | loghalo core radius (pc) |
| `GALAXY_MASS_MSUN` | double | `0.0` | user |  | pointmass galaxy mass (Msun) |

## Time steps

| key | type | default | level | values | description |
|---|---|---|---|---|---|
| `ETA_FF` | double | `0.2` | expert |  | Free-fall time-step eta (Rantala et al. 2023, sec. 3.4) |
| `ETA_FB` | double | `0.2` | expert |  | Fly-by time-step eta |
| `ETA_GRAD` | double | `0.2` | expert |  | Gradient time-step eta (FROST-CLUSTERS; Rantala et al. 2023: 0.1) |
| `ETA_JERK` | double | `0` | expert |  | Jerk time-step eta (0: off, FROST-CLUSTERS; Rantala et al. 2023: 0.1) |
| `ETA` | double | `0` | expert |  | Velocity / acceleration time-step eta (0: off, FROST-CLUSTERS; Rantala et al. 2023: 0.2) |
| `ETA_PERT` | double | `0.02` | expert |  | Perturbation time-step eta of subsystems (Rantala et al. 2023, Eq. 30) |
| `ETA_GBS` | double | `1e-10` | expert |  | GBS extrapolation tolerance of the LogH integrator |
| `ETA_RESTART` | double | `1e-8` | expert |  | Energy-error threshold of the per-subsystem LogH restart and the step-level restart (0: off) |
| `N_ORB_SECULAR` | int | `3` | expert |  | Binaries with more orbits than this per step take the secular (orbit-averaged) path |
| `GRAD_FRESH` | bool | `false` | **experiment** |  | Gradient time step from a fresh \|g'\| at level 0 |

## Subsystems

| key | type | default | level | values | description |
|---|---|---|---|---|---|
| `R_NGB` | double | `1e-3` | expert |  | Neighbour radius (pc) of the subsystem search |
| `R_NGB_MAX` | double | `0.1` | expert |  | Largest separation (pc) of the orbital-element check |
| `SECULAR_TRIPLES` | bool | `true` | **experiment** |  | Stable triples with >= N_ORB_SECULAR inner orbits per step take the double-averaged secular path; false: LogH |
| `SDAR` | bool | `true` | internal |  | Slow-down of weakly perturbed inner binaries in LogH (Wang+2020; FROST-CLUSTERS I App. A2); env NBODY_SDAR overrides |
| `SDAR_KREF` | double | `1e-6` | **experiment** |  | SDAR k_ref of the slow-down factor (W20 Eq. 22) |
| `SDAR_C` | double | `0.1` | **experiment** |  | SDAR kappa_max coefficient c (W20 Eq. 28) |
| `GBS_SEQUENCE` | string | `bulirsch` | **experiment** | bulirsch\|deuflhard | GBS substep sequence: bulirsch (2, 4, 6, 8, 12, 16, ...) or deuflhard (2, 4, 6, 8, 10, ...; MSTAR) |
| `GBS_KMAX` | int | `10` | **experiment** |  | GBS extrapolation stages (FROST-CLUSTERS I: 8) |
| `FLYBY_CM_VELOCITY` | bool | `false` | **experiment** |  | Fly-by step of subsystem members with the c.o.m. velocity (FROST-CLUSTERS II sec. 2.1) |
| `FLYBY_CM_EPS` | double | `1e-11` | internal |  | Tidal-sampling pair step: energy a slow particle's kicks put into a subsystem per step <= this x \|E0\| (0: off) |
| `TIDAL_STEP_RES` | double | `1.0` | internal |  | Floor of the FLYBY_CM_EPS step: this x P (1 - e)^(3/2) of a bound binary, this x the shortest pair pericentre passage 2 pi r_p / v_p of an unbound pair or larger subsystem (0: none) |
| `TIDAL_FLOOR_ADAPT` | bool | `false` | internal |  | Bound binaries whose local tidal estimate at the TIDAL_STEP_RES floor, mu \|T\| v_p floor (T: tidal acceleration of all perturbers), exceeds their share FLYBY_CM_EPS x \|E0\| take the floor x sqrt(share / estimate) (resolved kicks: error ~ dt^2; C5); others unchanged |
| `FLYBY_NGB_CM` | bool | `true` | internal |  | Distant fly-by neighbour test of a binary member against the binary's c.o.m. |
| `NGB_MASS_SCALING` | bool | `true` | internal |  | Pair neighbour radius r_ngb max(1, m / <m>)^(1/3) (PeTar Eq. 12); the extra pairs join only as strong perturbers of a binary |
| `WEAK_SPLIT_GAMMA` | double | `0` | expert |  | Fly-by or unstable triple whose third body perturbs the inner binary below this tidal ratio: binary subsystem + field star (0: off; 1e-6: 35x faster stalled steps, \|dE/E\| per step up to 30x worse) |
| `NGB_MASS_GAMMA` | double | `1e-3` | internal |  | Tidal ratio above which a mass-scaled neighbour joins a binary |
| `NGB_MREF` | double | `0.0` | test |  | <m> of NGB_MASS_SCALING (Msun) instead of the cluster's (0) |
| `E_REF` | double | `0.0` | test |  | \|E0\| of FLYBY_CM_EPS (Msun (km/s)^2) instead of the run's (0): a cut-out of a cluster as in the cluster |
| `NGB_WIDE_FAC` | double | `0` | **experiment** |  | Bound pairs with a < NGB_WIDE_FAC x R_NGB are neighbours (wide binaries; 0: off) |
| `MSTAR_THRESHOLD` | int | `8` | **experiment** |  | Subsystems of at least this many bodies for MSTAR (not wired: printed only, every subsystem uses LogH) |

## Output

| key | type | default | level | values | description |
|---|---|---|---|---|---|
| `PRINT_INTERVAL` | int | `1000` | user |  | Diagnostics every this many steps |
| `PROFILING` | string | `on` | user | off\|on\|detailed\|true\|false | off (false): no timers, counters or profiling fences (zero overhead); on (true): bucket tree of each print interval (every second of wall time named, core/profiling.h) and the component table; detailed: also per-rank min/max, calls, force pairs, work counters (copies, MPI bytes, subsystems, GBS steps, SSE stars) and per-depth sets. on and detailed append every report to profile.jsonl in the run directory |
| `PROFILING_FENCES` | bool | `true` | internal |  | Fence the device at each timer stop (exact per-component times, ~4 % of a 200k step) |
| `RESTART_INTERVAL` | int | `0` | user |  | Restart file every this many steps and at the end (0: off; needs USE_HDF5) |
| `WALLTIME_LIMIT` | double | `0.0` | user |  | Hours: restart file and clean stop before this wall time (0: off) |
| `RESTART_DIR` | string | `.` | user |  | Directory of restart_<step>.h5 |
| `RESTART_FILE` | string | (empty) | user |  | Continue from this file; latest: newest in RESTART_DIR; auto: that or a fresh start (--restart overrides) |
| `RESTART_KEEP` | int | `2` | expert |  | Restart files kept (the newest) |
| `STALL_FACTOR` | double | `5` | expert |  | A step slower than this x the median of the previous steps prints STALL lines: the costliest LogH integrations and the deepest set (0: off) |
| `STALL_DE_FACTOR` | double | `10` | expert |  | A step whose energy change (ETA_RESTART check) exceeds this x the median of the previous steps prints STALL lines (0: off) |
| `SUBSYS_ERR_TOL` | double | `1e-11` | expert |  | A subsystem advance (LogH, secular binary or triple) with unbooked \|dE\| above this x \|E0\| prints a SUBSYS_ERR line with its members' state (scripts/subsys_cases.py; 0: off) |
| `STALL_RESTART` | bool | `false` | expert |  | Restart file before every step (RESTART_DIR/stall_roll), kept as RESTART_DIR/stall_<step>.h5 for a stalled step (scripts/cutout.py) |
| `SNAPSHOT_INTERVAL` | int | `0` | user |  | LUNA-layout snapshot every this many steps, at step 0 and at the end (0: off) |
| `SNAPSHOT_DIR` | string | `.` | user |  | Directory of snapdir_NNN/snapshot_NNN.0.hdf5 |
| `SNAPSHOT_BASE` | string | `snapshot` | expert |  | Snapshot file base name |
| `SNAPSHOT_POTENTIAL` | bool | `true` | expert |  | Write the potential of each particle into snapshots |
| `SNAPSHOT_CLUSTER_DIAG` | bool | `true` | expert |  | Density centre, core, Lagrangian radii, binaries (and r_J) in each snapshot Header |
| `CLUSTER_DIAG_INTERVAL` | int | `0` | user |  | Cluster diagnostics (density centre, core, Lagrangian radii, binaries) every this many steps (0: off) |
| `CLUSTER_DIAG_FILE` | string | `cluster_diag.dat` | expert |  | File of the cluster diagnostics (rank 0) |
| `CLUSTER_DIAG_RPAIR_PC` | double | `0.02` | expert |  | Largest pair separation in the binary census (pc) |

## Performance and parallelism

| key | type | default | level | values | description |
|---|---|---|---|---|---|
| `SERIAL_N2_GPU` | int | `16384` | internal |  | Below this count: full N^2 on every rank, no MPI distribution (0: always distributed) |
| `HOST_SUBTREE_MAX` | int | `64` | internal |  | Sets of at most this many particles run their whole HHS sub-recursion on the host (one copy in and out; 0: off; run B 630 -> 240 s/step, 2026-10-16) |
| `REPLICA_CHECK_INTERVAL` | int | `10` | internal |  | Distributed runs: check every this many steps that the replicas agree (0: off) |
| `MPI_NON_BLOCKING` | bool | `false` | internal |  | MPI_Iallgatherv for overlapped transfers |
| `MPI_TRANSPORT` | string | `auto` | internal | auto\|nccl\|host | Allgathers: nccl (USE_NCCL builds; auto there), host (staged through pinned host memory; auto otherwise) |
| `USE_TREE` | bool | `false` | internal |  | Barnes-Hut tree forces above TREE_THRESHOLD (not with COLLISIONAL) |
| `TREE_THRESHOLD` | int | `100000` | internal |  | Tree forces above this count |
| `TREE_THETA` | double | `0.7` | internal |  | Tree opening angle |
| `TREE_N_LEAF` | int | `32` | internal |  | Largest leaf cell |
| `TREE_N_GROUP` | int | `64` | internal |  | Particles per tree traversal group |
| `TREE_DT_MIN_N` | int | `-1` | internal |  | Exact device-tree time-step search for sets of at least this many (0: off; -1: host builds 2048, GPU 65536, 300000 with FLYBY_CM_VELOCITY) |
| `TREE_NGB_MIN_N` | int | `-1` | internal |  | The same for the neighbour search (-1: host builds 2048, GPU 32768) |
| `TREE_DT_CPU_MAX` | int | `0` | internal |  | Host-thread tree search for sets up to this size (host builds / APUs; 0: off) |
| `TREE_DT_VC_FAC` | double | `30.0` | internal |  | Low-speed split of the tree time-step search (multiple of V_c) |
| `HHS_CC` | bool | `false` | **experiment** |  | Fast HHS sets split into connected components (Janes et al. 2014) |
| `HHS_CC_MAX` | int | `-1` | **experiment** |  | Largest set split, run locally on every rank (-1: max(SERIAL_N2_GPU, 16384)) |
| `HHS_CC_MIN` | int | `0` | **experiment** |  | Components smaller than this recurse together as one group (0: off) |
| `INTEGER_POSITIONS` | bool | `false` | **experiment** |  | int64 positions (exact drift, MPI-safe) |
| `DOMAIN_HALFSIZE_PC` | double | `200` | **experiment** |  | Half-size of the INTEGER_POSITIONS domain (pc) |

## Test problems

| key | type | default | level | values | description |
|---|---|---|---|---|---|
| `BINARY_A_PC` | double | `1e-8` | test |  | binary_test: semi-major axis (pc) |
| `BINARY_E` | double | `0.9` | test |  | binary_test: eccentricity |
| `BINARY_M1` | double | `10.0` | test |  | binary_test: primary mass (Msun) |
| `BINARY_M2` | double | `10.0` | test |  | binary_test: secondary mass (Msun) |
| `BINARY_ROT_DEG` | double | `0.0` | test |  | binary_test: rotation of the orbit in its plane (deg) |
| `BINARY_CHI1` | string | (empty) | test |  | binary_test: dimensionless spin vector "cx,cy,cz" of body 1 |
| `BINARY_CHI2` | string | (empty) | test |  | binary_test: dimensionless spin vector of body 2 |
| `TRIPLE_A_IN_PC` | double | `1e-8` | test |  | triple_test: inner semi-major axis (pc) |
| `TRIPLE_E_IN` | double | `0.1` | test |  | triple_test: inner eccentricity |
| `TRIPLE_M1` | double | `10.0` | test |  | triple_test: inner primary mass (Msun) |
| `TRIPLE_M2` | double | `10.0` | test |  | triple_test: inner secondary mass (Msun) |
| `TRIPLE_A_OUT_PC` | double | `5e-8` | test |  | triple_test: outer semi-major axis (pc) |
| `TRIPLE_E_OUT` | double | `0.1` | test |  | triple_test: outer eccentricity |
| `TRIPLE_M3` | double | `10.0` | test |  | triple_test: outer body mass (Msun) |
| `QUAD_M1` | double | `20.7` | test |  | quad22_test: binary A primary (Msun) |
| `QUAD_M2` | double | `14.1` | test |  | quad22_test: binary A secondary (Msun) |
| `QUAD_A1_PC` | double | `2.82e-4` | test |  | quad22_test: binary A semi-major axis (pc) |
| `QUAD_E1` | double | `0.853` | test |  | quad22_test: binary A eccentricity |
| `QUAD_M3` | double | `0.791` | test |  | quad22_test: binary B primary (Msun) |
| `QUAD_M4` | double | `0.105` | test |  | quad22_test: binary B secondary (Msun) |
| `QUAD_A2_PC` | double | `9.0e-8` | test |  | quad22_test: binary B semi-major axis (pc) |
| `QUAD_E2` | double | `0.62` | test |  | quad22_test: binary B eccentricity |
| `QUAD_A_OUT_PC` | double | `2.7e-3` | test |  | quad22_test: outer orbit semi-major axis (pc) |
| `QUAD_E_OUT` | double | `0.48` | test |  | quad22_test: outer orbit eccentricity |
| `IC_FILE` | string | (empty) | test |  | particle_file: "m x y z vx vy vz [kw [R]]" per line (Msun, pc, km/s, R_sun) |
| `QUAD_INCL_DEG` | double | `30.0` | test |  | quad22_test: inclination of binary B's plane (deg) |
| `TRIPLE_INCL_DEG` | double | `45.0` | test |  | triple_test: mutual inclination (deg) |
| `TRIPLE_OUTER_FIRST` | bool | `false` | test |  | triple_test: outer body at the lowest particle id |
| `TRIPLE_IN_ROT_DEG` | double | `0.0` | test |  | triple_test: rotation of the inner orbit (deg) |
| `TRIPLE_IN_START_Q` | double | `5.0` | test |  | triple_test, unbound inner orbit: start at this many pericentre distances |
| `TRIPLE_OUT_START_Q` | double | `5.0` | test |  | triple_test, unbound outer orbit: start at this many pericentre distances |
| `TEST_STELLAR_RADIUS_RSUN` | double | `0.0` | test |  | Few-body tests: stellar radius of every non-BH body (Rsun; 0: none) |
| `TEST_BOOST_KMS` | string | (empty) | test |  | Few-body tests: uniform velocity "vx,vy,vz" (km/s) added to every body |

## Debug environment variables

`NBODY_*` switches read with getenv (diagnostics and A/B checks; not config keys). Unset: off / the stated default.

| variable | effect |
|---|---|
| `NBODY_CHECK_CC` | Check that the connected-component split sees the pair steps of the time-step kernel |
| `NBODY_CHECK_DT_FP32` | =1: check the fp32 time-step kernels against fp64 |
| `NBODY_CHECK_HOSTSUB` | =1: abort on a device pointer met inside a host subtree (HOST_SUBTREE_MAX) |
| `NBODY_CHECK_EIH` | Check that every LogH PN evaluation (cached geometry) equals the direct EIH functions bit for bit |
| `NBODY_CHECK_PIDMAP` | Check every incremental pid-map update against a full rebuild |
| `NBODY_CHECK_STELLAR_CACHE` | Check every cached stellar type and radius against the device values |
| `NBODY_CHECK_SUBSYS` | Check that every subsystem is advanced exactly once per subsystem call |
| `NBODY_CHECK_TREE_DT` | =1: check every tree time-step / neighbour search against O(N^2) |
| `NBODY_DEEP_LOG` | =d: log the particles and pairs that keep HHS sets at depth >= d |
| `NBODY_FENCE_STATS` | =1/2: count device fences and copies per call site (fence_stats.h; NVTX ranges for nsys) |
| `NBODY_FORCE_ERR` | =eps: synthetic relative force error before every field kick of HHS sets >= NBODY_FORCE_ERR_MIN (FMM accuracy study; not bit-identical) |
| `NBODY_FORCE_ERR_MIN` | =n: smallest HHS set that gets NBODY_FORCE_ERR (default 16384) |
| `NBODY_FORCE_ERR_MODEL` | =random (default; eps \|a_i\| times an isotropic unit vector per particle and call) / frozen (the same vector per particle at every call) / smooth (eps a_rms times a random linear field in position over the set, per call) / cell (eps \|a_i\| times hashed unit vectors of NBODY_FORCE_ERR_LEVELS nested grid cells, frozen in space: FMM-like) |
| `NBODY_FORCE_ERR_LEVELS` | =L: nested grids of the cell model (default 6; edges r_rms 2^-l) |
| `NBODY_FORCE_ERR_LFIX` | =1: also remove the net torque of the error over each kicked range (angular momentum exact; implies PFIX) |
| `NBODY_FORCE_ERR_PFIX` | =1: remove the mass-weighted mean error over each kicked range (momentum exact; default 1 for cell, 0 otherwise) |
| `NBODY_GW_OUTPUT` | =0: no gravitational-wave table gw_events.dat (A/B: the dynamics must be bit-identical) |
| `NBODY_MERGE_CHECK` | =1: total energy and books before and after each per-level merger (merger energy booking check) |
| `NBODY_SSE_CHECK` | =1: runtime SSE potential change booked against two full triangle sums (SSE energy booking check) |
| `NBODY_LEAN_SYNC` | =0: keep every device fence of the original code (default on: lean copies, bit-identical) |
| `NBODY_DUMP` | Particle state at each print to dump_<step>.bin |
| `NBODY_DUMP_DT` | =n: pid, dt_adapt, grad_mag, ax of the set after the time-step criteria, first n calls |
| `NBODY_EATTR` | Per-print energy attribution |
| `NBODY_ELEDGER` | =1: energy ledger of the HHS recursion (error per call and level; one rank, replays only) |
| `NBODY_ECHECK_CMP` | Energy check: the split sum against the replicated whole-range sum |
| `NBODY_PE_STREAMS` | =0: the potential-energy blocks one after the other on the default stream (GPU default: concurrent streams, same bits) |
| `NBODY_PE_REUSE` | =0: the print diagnostics recompute the energy check's potential energy (same bits); =check: recompute and compare |
| `NBODY_ESPLIT` | Total energy and the internal energy of a fixed list of binaries at each print |
| `NBODY_GBS_DYNCAP` | Initial LogH step: t_dyn / this (default 3) |
| `NBODY_GBS_STAGE_PAR` | Batches of up to this many subsystems run their GBS stages on host threads (default 2) |
| `NBODY_GBS_STATS` | GBS step and stage statistics |
| `NBODY_GBS_STEPS_PER_ORBIT` | LogH GBS step limit per orbit (default 64) |
| `NBODY_GBS_TTOL` | Relative tolerance of the LogH end time (default 1e-4) |
| `NBODY_LOGH_SLOW` | =k: name every LogH integration that took at least k GBS steps |
| `NBODY_SDAR` | =0/1: overrides SDAR |
| `NBODY_SEC_INNER` | =0: no secular inner binaries in PN subsystems (CASES.md C20; A/B) |
| `NBODY_SDAR_PN` | =0: no SDAR slow-down of inner binaries without a PN member inside PN subsystems (default on since C20; PLAN item 4c) |
| `NBODY_SDAR_EXT` | =0: SDAR slow-down factors ignore the perturbers outside the LogH subsystem (no outside tide in kappa, no tidal energy cap; before C21) |
| `NBODY_DT_ATTR` | =1 (one rank): print which time-step criterion set each particle's step, per level and class (diagnostic, CASES.md R6) |
| `NBODY_DT_SCALE` | <class>:<f>: scale the steps of bh \| sub \| single \| all \| deepN particles by f (diagnostic scan, CASES.md R6) |
| `NBODY_CM_TIDE` | =1: field kicks of a bound 2-body subsystem whose kick spacing exceeds P/NBODY_CM_TIDE_NRES (gamma < NBODY_CM_TIDE_GAMMA, slowly moving sources; sets <= 64 on the host) go to its c.o.m., the tide acts on the internal orbit as the orbit-averaged quadrupole (Milankovitch), with E, P, L exact by construction (CASES.md C25; default off, A/B). At exit: `CM_TIDE:` line |
| `NBODY_CM_TIDE_CHECK` | =1: report every NBODY_CM_TIDE kick that changes the internal semi-major axis (must never print) |
| `NBODY_CM_TIDE_GAMMA` | =g: tidal strength 2 (a(1+e))^3 sum m_k/d_k^3 / M below which NBODY_CM_TIDE takes a binary (default 1e-3; leaves it above 4 g) |
| `NBODY_CM_TIDE_NRES` | =n: kicks per orbit that resolve a binary for NBODY_CM_TIDE (default 16: c.o.m. mode above spacing P/16, stays down to P/32) |
| `NBODY_GAUSS_IRK` | =1: compact Newtonian subsystems (2-4 bodies, no PN member, no collision candidate, no e >= 0.99 pair or hard encounter, no stable MA hierarchy, no SDAR slow-down) by the symplectic 8-stage Gauss-Legendre IRK instead of LogH+GBS, LogH+GBS if it does not converge (CASES.md C22; default off, A/B). At exit: `Gauss-IRK stats` line; step summary `irk=` / `irk_fb=`; SUBSYS_ERR kind `unbooked-irk` |
| `NBODY_SLOW0_STATS` | Statistics of small all-fast HHS sets, printed at the end |
| `NBODY_SMALL_FOR` | Largest loop run on the host where it can access device memory (default 256; 0: off) |
| `NBODY_SMALL_N2` | Largest pair count run on the host where it can access device memory (default 4096) |
| `NBODY_TILE256_MIN` | N^2 kernel: 256-thread tiles from this set size (GPUs; default 131072) |
| `NBODY_N2_P` | N^2 kernels: force P = 1, 2, 4 threads per sink (default per kernel, results identical) |
| `NBODY_N2_R` | N^2 kernels: force teams of TS / R threads, R = 1, 2 (default per kernel, results identical) |

<!-- notes: written by hand below this line; gen_config_md.py keeps them -->

## Notes

- A user key's table default applies only when a config file leaves the key out; for several of them (TEST_CASE,
  PLUMMER_*, PN_ORDER, COLLISIONAL, SSE_*) it is not the production value, which `config.txt` sets.
- Every other key's default is the production value (config.txt before 2026-10-12, which listed 108 keys). The ctest
  cases (`tests/run_case.py`) predate that and pin the old code defaults of 15 keys in `OLD_DEFAULTS`.
- `R_BREAK_AU` and `IC_R_BREAK` set the same radius (absolute, or in units of `BINARY_A_MAX_AU`); `IC_R_BREAK` applies
  only when `R_BREAK_AU` = 0.
- `MSTAR_THRESHOLD` is printed but not used: every subsystem is integrated with LogH.
- Adding a key: a row in `kParams` (src/core/params.cpp), `cfg.get_*("KEY")` at the use site (no default there), a line
  in `config.txt` if it is level user, then `python3 scripts/gen_config_md.py`. The ctest case `config_table` checks
  all of this.
- Output files of collisional runs: `merger_tree.dat`, `escaper_log.dat` (rewritten at every print) and
  `gw_events.dat` (appended; docs/CASES.md E2): one row per NS/BH merger (masses, spins, a, e, Wen peak frequency, e
  at 10 Hz and 10 mHz from Peters evolution, chi_eff, chi_p, remnant, path, environment, channel) and, at every
  snapshot, one per bound NS/BH pair in a subsystem with Peters time below `GW_CENSUS_TGW_MYR`. The column header is
  in the file; a resumed run truncates it to the rows of its restart file. `NBODY_GW_OUTPUT=0` turns it off.
