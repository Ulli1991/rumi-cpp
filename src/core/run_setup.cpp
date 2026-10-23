#include "run_setup.h"
#include <cstdlib>
#include <type_traits>
#include "particles.h"
#include "forces.h"
#include "integrator.h"
#include "timestep.h"
#include "diagnostics.h"
#include "constants.h"
#include "mpi_comm.h"
#include "profiling.h"
#ifdef ENABLE_COLLISIONAL
#include "collisional_config.h"
#include "subsystem.h"
#include "gw_output.h"
#endif

#ifdef ENABLE_COLLISIONAL
CollisionalConfig g_collisional;
#endif

int apply_run_settings(const Config& cfg, RunSettings& rs) {
    serial_n2_gpu = cfg.get_int("SERIAL_N2_GPU");
    host_subtree_max = cfg.get_int("HOST_SUBTREE_MAX");
    hhs_cc = cfg.get_bool("HHS_CC");
    hhs_cc_max = cfg.get_int("HHS_CC_MAX");
    hhs_cc_min = cfg.get_int("HHS_CC_MIN");
    rs.ngb_mass_scaling = false;
    rs.ngb_mref = 0.0;
    rs.e_ref_set = 0.0;
    flyby_cm_velocity = cfg.get_bool("FLYBY_CM_VELOCITY");   // FROST-CLUSTERS II sec. 2.1
    g_eattr.on = std::getenv("NBODY_EATTR") != nullptr;   // per-print energy attribution (diagnostics.h)
    use_integer_positions = cfg.get_bool("INTEGER_POSITIONS");
    use_nonblocking_mpi = cfg.get_bool("MPI_NON_BLOCKING");
    mpi_transport = parse_mpi_transport(cfg.get_str("MPI_TRANSPORT"));   // AUDIT G row 4

    // Barnes-Hut tree configuration
    use_tree = cfg.get_bool("USE_TREE");
    tree_threshold = cfg.get_int("TREE_THRESHOLD");
    tree_theta = cfg.get_double("TREE_THETA");
    tree_n_leaf = cfg.get_int("TREE_N_LEAF");
    tree_n_group = cfg.get_int("TREE_N_GROUP");
    // Exact tree searches for the HHS time steps and neighbours (gravity/dt_tree.h); forces stay direct
    // Exact tree searches on by default above the measured break-even set sizes (AUDIT F7; results are
    // bit-identical to O(N^2), so only speed changes). Host builds: ~1000-1500 for both searches (32 threads).
    // GPUs (A100): neighbours ~25k, time steps ~30k with component velocities, ~150k with FLYBY_CM_VELOCITY
    // (fewer prunable pairs). GPU values also serve APUs until measured there. With FLYBY_CM_VELOCITY and the
    // tidal-sampling step the 200k cluster ran 4.12 s/step with the tree against 3.96 s without (verify-gpuopt,
    // 2026-10-11), so the c.o.m. threshold sits above 200k.
    constexpr bool host_build = std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>;
    const int dt_default = host_build ? 2048 : (flyby_cm_velocity ? 300000 : 65536);
    tree_dt_min_n = cfg.get_int("TREE_DT_MIN_N");        // -1: the build's default
    if (tree_dt_min_n < 0) tree_dt_min_n = dt_default;
    tree_ngb_min_n = cfg.get_int("TREE_NGB_MIN_N");
    if (tree_ngb_min_n < 0) tree_ngb_min_n = host_build ? 2048 : 32768;
    tree_dt_cpu_max = cfg.get_int("TREE_DT_CPU_MAX");
    tree_dt_vc_fac = cfg.get_double("TREE_DT_VC_FAC");

#ifdef ENABLE_COLLISIONAL
    // ETA_FF/ETA_FB always parsed (used by force kernels even without subsystems)
    g_collisional.eta_ff          = cfg.get_double("ETA_FF");
    g_collisional.eta_fb          = cfg.get_double("ETA_FB");

    // Collisional (unsoftened direct N-body) configuration
    g_collisional.enabled = cfg.get_bool("COLLISIONAL");
    if (g_collisional.enabled && use_tree) {
        // Merger compaction re-indexes particles under a built tree, and tree forces lose 4th order and
        // L conservation (force-error floor ~1e-5 at theta 0.5): not supported for collisional runs
        if (MPIComm::rank() == 0)
            std::fprintf(stderr, "ERROR: USE_TREE is not supported with COLLISIONAL = true\n");
        return 1;
    }
    if (g_collisional.enabled) {
        g_collisional.r_ngb           = cfg.get_double("R_NGB");
        g_collisional.r_ngb_max       = cfg.get_double("R_NGB_MAX");
        g_collisional.eta_grad        = cfg.get_double("ETA_GRAD");
        g_collisional.eta_jerk        = cfg.get_double("ETA_JERK");
        g_collisional.eta_va          = cfg.get_double("ETA");
        g_collisional.eta_gbs         = cfg.get_double("ETA_GBS");
        g_collisional.eta_pert        = cfg.get_double("ETA_PERT");
        g_collisional.flyby_ngb_cm    = cfg.get_bool("FLYBY_NGB_CM");
        rs.ngb_mass_scaling           = cfg.get_bool("NGB_MASS_SCALING");
        rs.ngb_mref                   = cfg.get_double("NGB_MREF");   // <m> override [Msun] (tests)
        rs.e_ref_set                  = cfg.get_double("E_REF");      // |E0| override (tests)
        g_collisional.ngb_mass_gamma  = cfg.get_double("NGB_MASS_GAMMA");
        g_collisional.weak_split_gamma = cfg.get_double("WEAK_SPLIT_GAMMA");
        g_collisional.subsys_err_tol = cfg.get_double("SUBSYS_ERR_TOL");
        g_collisional.tidal_step_res  = cfg.get_double("TIDAL_STEP_RES");
        g_collisional.tidal_floor_adapt = cfg.get_bool("TIDAL_FLOOR_ADAPT");
        g_collisional.secular_triples = cfg.get_bool("SECULAR_TRIPLES");
        g_collisional.sdar            = cfg.get_bool("SDAR");
        if (const char* e = std::getenv("NBODY_SDAR")) g_collisional.sdar = std::atoi(e) != 0;
        g_collisional.sdar_kref       = cfg.get_double("SDAR_KREF");
        g_collisional.sdar_c          = cfg.get_double("SDAR_C");
        g_collisional.tde_accretion = cfg.get_double("TDE_ACCRETION_FRACTION");
        g_collisional.coll_mass_loss_c = cfg.get_double("COLL_MASS_LOSS_C");
        g_collisional.bh_merge_max_single = cfg.get_str("BH_MERGE_RADIUS") == "max_single";
        // Gravitational-wave table gw_events.dat (gw_output.h); NBODY_GW_OUTPUT=0: off (tests: identical dynamics)
        g_gw_log.census_tgw = cfg.get_double("GW_CENSUS_TGW_MYR") / TIME_UNIT_MYR;
        if (const char* e = std::getenv("NBODY_GW_OUTPUT")) g_gw_log.on = std::atoi(e) != 0;
        g_collisional.grad_fresh = cfg.get_bool("GRAD_FRESH");
        g_collisional.gbs_deuflhard = cfg.get_str("GBS_SEQUENCE") == "deuflhard";
        g_collisional.gbs_kmax = cfg.get_int("GBS_KMAX");
        g_collisional.r_ngb_wide = cfg.get_double("NGB_WIDE_FAC") * g_collisional.r_ngb;
        g_collisional.flyby_cm_eps    = cfg.get_double("FLYBY_CM_EPS");
        g_collisional.eta_restart     = cfg.get_double("ETA_RESTART");
        g_collisional.n_orb_secular   = cfg.get_int("N_ORB_SECULAR");
        g_collisional.m_pn            = cfg.get_double("M_PN");
        g_collisional.pn_bh_only      = cfg.get_bool("PN_BH_ONLY");
        g_collisional.pn_order        = cfg.get_int("PN_ORDER");
        g_collisional.r_escape_rh     = cfg.get_double("R_ESCAPE_RH");
        g_collisional.r_escape_rj     = cfg.get_double("R_ESCAPE_RJ");
        g_collisional.mstar_threshold = cfg.get_int("MSTAR_THRESHOLD");
        g_collisional.sse_enabled     = cfg.get_bool("SSE_ENABLED");
        g_collisional.sse_metallicity = cfg.get_double("SSE_METALLICITY");
        g_collisional.sse_wind        = cfg.get_str("SSE_WIND_MODEL");
        g_collisional.sse_remnant     = cfg.get_str("SSE_REMNANT_MODEL");
        g_collisional.sse_ppisn       = cfg.get_str("SSE_PPISN_MODEL");
        g_collisional.sse_ecsn        = cfg.get_str("SSE_ECSN_MODEL");
        g_collisional.sse_mxns        = cfg.get_double("SSE_MXNS");
        g_collisional.sse_flbv        = cfg.get_double("SSE_FLBV");
        g_collisional.sse_natal_kicks = cfg.get_str("SSE_NATAL_KICKS");
        g_collisional.kick_seed       = cfg.get_int("PLUMMER_SEED");
        g_collisional.compute_derived();

        // Collisional mode requires zero softening
        if (rs.soft_pc > 0.0 && MPIComm::rank() == 0) {
            std::printf("WARNING: COLLISIONAL=true but SOFTENING_PC=%.2e > 0. "
                        "Setting softening to 0.\n", rs.soft_pc);
        }
        rs.soft_pc = 0.0;
    }
#endif
    return 0;
}
