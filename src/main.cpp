#include "particles.h"
#include "forces.h"
#include "integrator.h"
#include "timestep.h"
#include "initial_conditions.h"
#include "diagnostics.h"
#include "constants.h"
#include "mpi_comm.h"
#include "profiling.h"
#ifdef ENABLE_COLLISIONAL
#include "collisional_config.h"
#include "subsystem.h"
#include "kepler.h"
CollisionalConfig g_collisional;

// Binary orbit tracker for PN precession validation
struct BinaryTracker {
    double mu;               // G*(m1+m2)
    double omega_prev;       // previous omega for unwrapping
    double omega_cumul;      // cumulative unwrapped omega
    double orbits_cumul;     // cumulative orbit count (from elapsed time)
    double omega_gr_per_orbit; // analytic GR precession rate
    double a0, e0;           // initial orbital elements
    double period0;          // initial orbital period
    bool initialized;

    // Apoapsis-based measurement (avoids intra-orbit PN oscillation aliasing)
    double rdot_prev;        // previous r_dot for sign-change detection
    int n_apo;               // number of apoapsis passages
    double omega_at_apo_prev;// omega at previous apoapsis
    double omega_cumul_apo;  // cumulative precession measured at apoapsis
    double e_at_apo;         // eccentricity at last apoapsis

    BinaryTracker() : initialized(false), rdot_prev(0), n_apo(0),
                       omega_at_apo_prev(0), omega_cumul_apo(0), e_at_apo(0) {}

    void init(Particles& p, double G) {
        if (p.N != 2) return;
        auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
        auto h_y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
        auto h_z = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
        auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
        auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
        auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
        auto h_m  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);

        mu = G * (h_m(0) + h_m(1));
        double r[3] = { h_x(1)-h_x(0), h_y(1)-h_y(0), h_z(1)-h_z(0) };
        double v[3] = { h_vx(1)-h_vx(0), h_vy(1)-h_vy(0), h_vz(1)-h_vz(0) };
        auto oe = cartesian_to_elements(r, v, mu);

        a0 = oe.a;
        e0 = oe.e;
        period0 = oe.period;
        omega_prev = oe.omega;
        omega_cumul = 0.0;
        orbits_cumul = 0.0;

        // Analytic PN1 precession: dw/orbit = 6*pi*G*M / (a*c^2*(1-e^2))
        // Sign: prograde (same direction as orbital motion).
        // Lz > 0 (CCW) → dw > 0;  Lz < 0 (CW) → dw < 0.
        double c2 = C_CODE * C_CODE;
        double Lz = r[0]*v[1] - r[1]*v[0];
        double sign = (Lz >= 0) ? 1.0 : -1.0;
        omega_gr_per_orbit = sign * 6.0 * M_PI * mu / (oe.a * c2 * (1.0 - oe.e * oe.e));

        // Initial r_dot for apoapsis detection
        double rmag = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
        rdot_prev = (r[0]*v[0] + r[1]*v[1] + r[2]*v[2]) / rmag;
        omega_at_apo_prev = oe.omega;
        omega_cumul_apo = 0.0;
        n_apo = 0;
        e_at_apo = oe.e;

        std::printf("BinaryTracker: a=%.4e pc (%.4f AU), e=%.4f, P=%.4e yr\n",
                    oe.a, oe.a / AU_TO_PC, oe.e, oe.period * TIME_UNIT_YR);
        std::printf("  PN1 precession: %.4e rad/orbit (%.4f deg/orbit)\n",
                    omega_gr_per_orbit, omega_gr_per_orbit * 180.0 / M_PI);
        std::fflush(stdout);
        initialized = true;
    }

    void update(Particles& p, double G, double t_code) {
        if (!initialized || p.N != 2) return;
        auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
        auto h_y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
        auto h_z = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
        auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
        auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
        auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);

        double r[3] = { h_x(1)-h_x(0), h_y(1)-h_y(0), h_z(1)-h_z(0) };
        double v[3] = { h_vx(1)-h_vx(0), h_vy(1)-h_vy(0), h_vz(1)-h_vz(0) };
        auto oe = cartesian_to_elements(r, v, mu);

        // Count orbits from elapsed time / initial period
        orbits_cumul = t_code / period0;

        // Unwrap omega to track cumulative precession (raw, at arbitrary phase)
        double dw = oe.omega - omega_prev;
        if (dw < -M_PI) dw += 2.0 * M_PI;
        if (dw > M_PI) dw -= 2.0 * M_PI;
        omega_cumul += dw;
        omega_prev = oe.omega;

        // Apoapsis detection: r_dot changes from positive to negative
        double rmag = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
        double rdot = (r[0]*v[0] + r[1]*v[1] + r[2]*v[2]) / rmag;

        if (rdot_prev > 0.0 && rdot <= 0.0) {
            // Passed through apoapsis — measure omega here
            double dw_apo = oe.omega - omega_at_apo_prev;
            if (dw_apo < -M_PI) dw_apo += 2.0 * M_PI;
            if (dw_apo > M_PI) dw_apo -= 2.0 * M_PI;
            omega_cumul_apo += dw_apo;
            omega_at_apo_prev = oe.omega;
            e_at_apo = oe.e;
            n_apo++;
        }
        rdot_prev = rdot;

        // Report: both raw (arbitrary phase) and apoapsis-based measurements
        double omega_gr_expected = orbits_cumul * omega_gr_per_orbit;
        double ratio_raw = (std::fabs(omega_gr_expected) > 1e-30)
                           ? omega_cumul / omega_gr_expected : 0.0;

        // ratio_apo uses omega measured only at apoapsis phases (less aliasing)
        // but normalized by total orbits from elapsed time (n_apo undercounts
        // because we only sample once per outer step, missing most apoapsides)
        double ratio_apo = 0.0;
        if (n_apo > 0 && std::fabs(omega_gr_expected) > 1e-30) {
            ratio_apo = omega_cumul_apo / omega_gr_expected;
        }

        std::printf("  Binary: N_orb=%.2f  a=%.4e  e=%.6f"
                    "  ratio_raw=%.4f  ratio_apo=%.4f (n_apo=%d, e_apo=%.6f)\n",
                    orbits_cumul, oe.a, oe.e,
                    ratio_raw, ratio_apo, n_apo, e_at_apo);
        std::fflush(stdout);
    }
};
#endif

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <chrono>
#include <iostream>
#include <cmath>

// ---------------------------------------------------------------------------
// Config file parser
// ---------------------------------------------------------------------------
struct Config {
    std::map<std::string, std::string> params;

    void load(const std::string& filename) {
        std::ifstream f(filename);
        if (!f.is_open()) {
            std::fprintf(stderr, "Cannot open config file: %s\n", filename.c_str());
            std::exit(1);
        }
        std::string line;
        while (std::getline(f, line)) {
            auto pos = line.find('#');
            if (pos != std::string::npos) line.erase(pos);
            pos = line.find('=');
            if (pos == std::string::npos) continue;
            std::string key = line.substr(0, pos);
            std::string val = line.substr(pos + 1);
            auto trim = [](std::string& s) {
                s.erase(0, s.find_first_not_of(" \t\r\n"));
                s.erase(s.find_last_not_of(" \t\r\n") + 1);
            };
            trim(key); trim(val);
            if (!key.empty()) params[key] = val;
        }
    }

    std::string get_str(const std::string& key, const std::string& def) const {
        auto it = params.find(key);
        return (it != params.end()) ? it->second : def;
    }
    double get_double(const std::string& key, double def) const {
        auto it = params.find(key);
        return (it != params.end()) ? std::atof(it->second.c_str()) : def;
    }
    int get_int(const std::string& key, int def) const {
        auto it = params.find(key);
        return (it != params.end()) ? std::atoi(it->second.c_str()) : def;
    }
};

// ---------------------------------------------------------------------------
static void print_usage(const char* prog) {
    std::printf("Usage: %s [--config <file>] [--test <case>]\n", prog);
    std::printf("  --config <file>   Config file (default: config.txt)\n");
    std::printf("  --test <case>     Override TEST_CASE from config\n");
    std::printf("  -h, --help        Show this help\n");
}

static const char* outer_planet_names[] = {"Sun", "Jupiter", "Saturn", "Uranus", "Neptune"};

int main(int argc, char** argv) {
    // Initialize MPI before Kokkos (MPI may set device affinity)
    MPIComm::init(&argc, &argv);

    Kokkos::initialize(argc, argv);
    {
        int my_rank = MPIComm::rank();
        int n_ranks = MPIComm::size();

        if (my_rank == 0) {
            std::printf("Kokkos configuration:\n"); std::fflush(stdout);
            Kokkos::print_configuration(std::cout);
            std::cout << std::flush;
            if (n_ranks > 1) {
                std::printf("\nMPI: %d ranks\n", n_ranks);
#if EFFECTIVE_GPU_AWARE_MPI
                std::printf("GPU-aware MPI: ENABLED\n");
#else
                std::printf("GPU-aware MPI: disabled (host-staged transfers)\n");
#endif
            }
        }
        MPIComm::barrier();

        std::string config_file = "config.txt";
        std::string test_override;

        for (int i = 1; i < argc; i++) {
            if ((!std::strcmp(argv[i], "--config") || !std::strcmp(argv[i], "-c")) && i+1 < argc)
                config_file = argv[++i];
            else if (!std::strcmp(argv[i], "--test") && i+1 < argc)
                test_override = argv[++i];
            else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
                print_usage(argv[0]); Kokkos::finalize(); MPIComm::finalize(); return 0;
            }
            // Silently skip Kokkos args (--kokkos-*)
            else if (std::strncmp(argv[i], "--kokkos", 8) == 0) {
                // Skip value arg if present
                if (i+1 < argc && argv[i+1][0] != '-') i++;
            }
            else {
                std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
                print_usage(argv[0]); Kokkos::finalize(); MPIComm::finalize(); return 1;
            }
        }

        Config cfg;
        cfg.load(config_file);

        std::string test = test_override.empty() ? cfg.get_str("TEST_CASE", "outer_planets") : test_override;
        std::string integrator = cfg.get_str("INTEGRATOR", "hhs");
        double t_final_myr  = cfg.get_double("TOTAL_TIME_MYR", 0.23);
        double dt_max_myr   = cfg.get_double("DT_MAX_MYR", 1.6e-3);
        double dt_min_myr   = cfg.get_double("DT_MIN_MYR", 1.0e-9);
        double eta          = cfg.get_double("ETA", 0.02);
        double soft_pc      = cfg.get_double("SOFTENING_PC", 1e-3);
        int print_interval  = cfg.get_int("PRINT_INTERVAL", 1000);
        int N_plummer       = cfg.get_int("PLUMMER_N", 1000);
        double M_total_msun = cfg.get_double("PLUMMER_M_TOTAL", 3.79e2);
        double r_half_pc    = cfg.get_double("PLUMMER_R_HALF", 3.5);
        int seed            = cfg.get_int("PLUMMER_SEED", 42);
        std::string mass_dist = cfg.get_str("PLUMMER_MASS_DIST", "equal");
        double age_gyr      = cfg.get_double("PLUMMER_AGE_GYR", 1.0);
        double metallicity  = cfg.get_double("PLUMMER_METALLICITY", 0.02);

        bool profiling_enabled = (cfg.get_str("PROFILING", "true") == "true");
        serial_n2_gpu = std::stoi(cfg.get_str("SERIAL_N2_GPU", "0"));
        serial_n2_cpu = std::stoi(cfg.get_str("SERIAL_N2_CPU", "0"));
        use_integer_positions = (cfg.get_str("INTEGER_POSITIONS", "false") == "true");
        double domain_halfsize = cfg.get_double("DOMAIN_HALFSIZE_PC", 200.0);
        use_nonblocking_mpi = (cfg.get_str("MPI_NON_BLOCKING", "false") == "true");
        symmetric_timesteps = (cfg.get_str("SYMMETRIC_TIMESTEPS", "false") == "true");

        // Barnes-Hut tree configuration
        use_tree = (cfg.get_str("USE_TREE", "false") == "true");
        tree_threshold = cfg.get_int("TREE_THRESHOLD", 10000);
        tree_theta = cfg.get_double("TREE_THETA", 0.5);
        tree_n_leaf = cfg.get_int("TREE_N_LEAF", 32);
        tree_n_group = cfg.get_int("TREE_N_GROUP", 64);
        gadget4_cross = (cfg.get_str("GADGET4_CROSS", "false") == "true");

#ifdef ENABLE_COLLISIONAL
        // Collisional (unsoftened direct N-body) configuration
        g_collisional.enabled = (cfg.get_str("COLLISIONAL", "false") == "true");
        if (g_collisional.enabled) {
            g_collisional.r_ngb           = cfg.get_double("R_NGB", 1e-3);
            g_collisional.r_ngb_max       = cfg.get_double("R_NGB_MAX", 0.1);
            g_collisional.eta_ff          = cfg.get_double("ETA_FF", 0.2);
            g_collisional.eta_fb          = cfg.get_double("ETA_FB", 0.2);
            g_collisional.eta_grad        = cfg.get_double("ETA_GRAD", 0.1);
            g_collisional.eta_jerk        = cfg.get_double("ETA_JERK", 0.1);
            g_collisional.eta_gbs         = cfg.get_double("ETA_GBS", 1e-10);
            g_collisional.eta_pert        = cfg.get_double("ETA_PERT", 0.02);
            g_collisional.eta_restart     = cfg.get_double("ETA_RESTART", 1e-6);
            g_collisional.n_orb_secular   = cfg.get_int("N_ORB_SECULAR", 3);
            g_collisional.m_pn            = cfg.get_double("M_PN", 5.0);
            g_collisional.pn_order        = cfg.get_int("PN_ORDER", 7);
            g_collisional.r_escape_rh     = cfg.get_double("R_ESCAPE_RH", 100.0);
            g_collisional.mstar_threshold = cfg.get_int("MSTAR_THRESHOLD", 8);
            g_collisional.sse_enabled     = (cfg.get_str("SSE_ENABLED", "true") == "true");
            g_collisional.sse_metallicity = cfg.get_double("SSE_METALLICITY", 0.02);
            g_collisional.compute_derived();

            // Collisional mode requires zero softening
            if (soft_pc > 0.0 && my_rank == 0) {
                std::printf("WARNING: COLLISIONAL=true but SOFTENING_PC=%.2e > 0. "
                            "Setting softening to 0.\n", soft_pc);
            }
            soft_pc = 0.0;
        }
#endif

        double G = G_CODE;
        double soft2 = soft_pc * soft_pc;
        double dt_max = dt_max_myr / TIME_UNIT_MYR;   // Myr → code TU
        double dt_min = dt_min_myr / TIME_UNIT_MYR;   // Myr → code TU
        double t_final = t_final_myr / TIME_UNIT_MYR;  // Myr → code TU

        if (use_integer_positions) {
            int_pos_scale = domain_halfsize / (double)(1LL << 62);
            int_pos_inv_scale = 1.0 / int_pos_scale;
        }

        Particles p;
        DistributedParticles dp;  // For multi-GPU mode
        const char* const* names = nullptr;
        bool use_distributed = MPIComm::is_distributed();

        if (my_rank == 0) {
            std::printf("Initializing particles...\n"); std::fflush(stdout);
        }
        auto t_init0 = std::chrono::high_resolution_clock::now();

        // Initialize particles on rank 0 (or all ranks for single-GPU)
        // Binary test parameters
        double binary_a_pc  = cfg.get_double("BINARY_A_PC", 4.848e-5);
        double binary_e     = cfg.get_double("BINARY_E", 0.0);
        double binary_m1    = cfg.get_double("BINARY_M1", 10.0);
        double binary_m2    = cfg.get_double("BINARY_M2", 10.0);

        if (my_rank == 0 || !use_distributed) {
            if (test == "outer_planets") {
                init_outer_planets(p);
                names = outer_planet_names;
            } else if (test == "plummer") {
                init_plummer(p, N_plummer, M_total_msun, r_half_pc, seed, mass_dist, age_gyr, metallicity);
            } else if (test == "binary_test") {
                init_binary_test(p, binary_a_pc, binary_e, binary_m1, binary_m2);
            } else {
                std::fprintf(stderr, "Unknown test case: %s\n", test.c_str());
                MPIComm::finalize();
                Kokkos::finalize();
                return 1;
            }
        }

        // Convert initial positions to integer representation (before MPI distribution)
        if (use_integer_positions && (my_rank == 0 || !use_distributed)) {
            auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
            auto h_y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
            auto h_z = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
            auto h_ix = Kokkos::create_mirror_view(p.ix);
            auto h_iy = Kokkos::create_mirror_view(p.iy);
            auto h_iz = Kokkos::create_mirror_view(p.iz);
            for (int i = 0; i < p.N; i++) {
                h_ix(i) = llrint(h_x(i) * int_pos_inv_scale);
                h_iy(i) = llrint(h_y(i) * int_pos_inv_scale);
                h_iz(i) = llrint(h_z(i) * int_pos_inv_scale);
                h_x(i) = (double)h_ix(i) * int_pos_scale;  // snap to integer grid
                h_y(i) = (double)h_iy(i) * int_pos_scale;
                h_z(i) = (double)h_iz(i) * int_pos_scale;
            }
            Kokkos::deep_copy(p.ix, h_ix);
            Kokkos::deep_copy(p.iy, h_iy);
            Kokkos::deep_copy(p.iz, h_iz);
            Kokkos::deep_copy(p.x, h_x);
            Kokkos::deep_copy(p.y, h_y);
            Kokkos::deep_copy(p.z, h_z);
        }

        // Distribute particles across ranks if using MPI
        if (use_distributed) {
            dp.distribute_initial(p);
            // After distribution, each rank works with dp.local
            // Rank 0's 'p' is no longer valid - use dp for all operations
        }

        auto t_init1 = std::chrono::high_resolution_clock::now();
        if (my_rank == 0) {
            std::printf("Particle init: %.3f s\n", std::chrono::duration<double>(t_init1-t_init0).count());
            std::fflush(stdout);
        }

        bool is_cluster = (test == "plummer");

        // Get particle count (local or total depending on mode)
        int N_display = use_distributed ? dp.N_total : p.N;

        const char* mode_str = (integrator == "hhs") ? "hhs" : "fsi";
        if (my_rank == 0) {
            std::printf("=== Kokkos N-Body: %s, %s test, N=%d ===\n",
                        mode_str, test.c_str(), N_display);
            if (use_distributed) {
                std::printf("    Distributed mode: %d ranks, %d local particles\n",
                            n_ranks, dp.N_local);
            }
            std::fflush(stdout);
            std::printf("G=%.4e  soft=%.4e pc  dt_max=%.4e Myr  dt_min=%.4e Myr  eta=%.4f\n",
                        G, soft_pc, dt_max_myr, dt_min_myr, eta);
            std::printf("t_final=%.4e Myr  print_interval=%d  profiling=%s\n",
                        t_final_myr, print_interval,
                        profiling_enabled ? "true" : "false");
            if (use_integer_positions) {
                std::printf("Integer positions: domain=%.1f pc, resolution=%.2e pc\n",
                            domain_halfsize, int_pos_scale);
            }
            if (use_tree) {
                std::printf("Barnes-Hut tree: theta=%.2f, n_leaf=%d, n_group=%d, threshold=%d%s\n",
                            tree_theta, tree_n_leaf, tree_n_group, tree_threshold,
                            gadget4_cross ? ", GADGET4_CROSS" : "");
            }
#ifdef ENABLE_COLLISIONAL
            if (g_collisional.enabled) {
                std::printf("Collisional mode: r_ngb=%.2e, r_ngb_max=%.2e, eta_gbs=%.2e\n",
                            g_collisional.r_ngb, g_collisional.r_ngb_max, g_collisional.eta_gbs);
                std::printf("  PN order=%d, M_PN=%.1f M_sun, MSTAR threshold=%d\n",
                            g_collisional.pn_order, g_collisional.m_pn, g_collisional.mstar_threshold);
                if (g_collisional.sse_enabled) {
                    std::printf("  SSE: Z=%.4f\n", g_collisional.sse_metallicity);
                }
            }
#endif
            std::printf("\n");
            std::fflush(stdout);
            if (profiling_enabled) {
                g_profiler.print_memory_report(N_display, n_ranks);
            }
        }

        auto levels = build_levels(dt_max, dt_min);

        // Get reference to working particles (local for distributed, p for single)
        Particles& work_p = use_distributed ? dp.local : p;

#ifdef ENABLE_COLLISIONAL
        // Allocate neighbor pair buffers before initial force computation
        // so the force kernels can detect close pairs
        if (g_collisional.enabled) {
            int max_pairs = std::max(1000, work_p.N / 10);
            g_subsystem_mgr.allocate_ngb_buffers(max_pairs);
            g_subsystem_mgr.reset_ngb_count();
            if (my_rank == 0) {
                std::printf("Collisional: neighbor buffer allocated for %d pairs\n", max_pairs);
                std::fflush(stdout);
            }
        }
#endif

        // Initial force computation + adaptive timestep
        if (my_rank == 0) {
            std::printf("Computing initial forces...\n"); std::fflush(stdout);
        }
        auto t0 = std::chrono::high_resolution_clock::now();

        if (use_distributed) {
            compute_acceleration_and_timestep_distributed(dp, G, soft2, eta);
        } else {
            compute_acceleration_and_timestep(work_p, G, soft2, eta);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        if (my_rank == 0) {
            std::printf("Initial forces: %.3f s\n", std::chrono::duration<double>(t1-t0).count());
            std::fflush(stdout);
        }

#ifdef ENABLE_COLLISIONAL
        // Run initial subsystem detection from pairs found during force computation.
        // For N=2 binary test with FSI, this is the only detection needed —
        // the binary is detected once and LogH handles it from there.
        if (g_collisional.enabled) {
            Kokkos::fence();
            g_subsystem_mgr.detect_from_pairs(work_p, G, dt_max,
                                               g_collisional.r_ngb, g_collisional.r_ngb_max,
                                               g_collisional.n_orb_secular);
            g_subsystem_mgr.update_particle_flags(work_p);
            g_subsystem_mgr.prune(work_p, G);
            g_subsystem_mgr.update_particle_flags(work_p);
            g_subsystem_mgr.reset_step_diagnostics();
            g_subsystem_mgr.accumulate_diagnostics();
            if (my_rank == 0) g_subsystem_mgr.print_step_summary();
        }
#endif

        assign_levels(work_p, levels);
        work_p.partition_by_level();

        if (my_rank == 0) {
            print_timebin_table(work_p, levels, -1.0, 0.0, 0.0, 0.0, 0, is_cluster);
        }

        // Print dt range
        if (my_rank == 0) {
            auto h_dt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), work_p.dt_adapt);
            double mn = 1e30, mx = 0;
            for (int i = 0; i < work_p.N; i++) {
                if (h_dt(i) < mn) mn = h_dt(i);
                if (h_dt(i) > mx) mx = h_dt(i);
            }
            std::printf("Particle dt range: [%.4e, %.4e] Myr\n\n", mn*TIME_UNIT_MYR, mx*TIME_UNIT_MYR);
        }

#ifdef ENABLE_COLLISIONAL
        BinaryTracker bin_tracker;
        if (test == "binary_test" && g_collisional.enabled && my_rank == 0) {
            bin_tracker.init(work_p, G);
        }
#endif

        auto diag0 = compute_diagnostics(work_p, G, soft2);

        if (my_rank == 0) {
            std::printf("Running...\n"); std::fflush(stdout);
        }

        auto wall_start = std::chrono::high_resolution_clock::now();

        double t = 0;
        int step = 0;

        auto print_progress = [&](double t_now, int step_now) {
            auto diag = compute_diagnostics(work_p, G, soft2);
            double dE, dP, dL;
            compute_errors(diag, diag0, work_p, is_cluster, dE, dP, dL);
            if (my_rank == 0) {
                print_timebin_table(work_p, levels, dE, dP, dL, t_now*TIME_UNIT_MYR, step_now, is_cluster);
#ifdef ENABLE_COLLISIONAL
                if (bin_tracker.initialized) {
                    bin_tracker.update(work_p, G, t_now);
                }
#endif
            }
        };

        if (integrator == "fsi") {
            double dt_global = levels.back();
            while (t < t_final) {
                auto step_t0 = std::chrono::high_resolution_clock::now();
                // In replicated mode, work_p contains all particles on all ranks
                fsi_step(work_p, 0, work_p.N, G, soft2, dt_global);
                t += dt_global;
                step++;
                auto step_t1 = std::chrono::high_resolution_clock::now();
                double step_wall = std::chrono::duration<double>(step_t1 - step_t0).count();
                g_profiler.step_wall_total += step_wall;
                g_profiler.step_count++;
                if (step % print_interval == 0) {
                    compute_acceleration_and_timestep(work_p, G, soft2, eta);
                    assign_levels(work_p, levels);
                    work_p.partition_by_level();
                    print_progress(t, step);
                    if (my_rank == 0 && profiling_enabled) {
                        g_profiler.print_report(N_display, n_ranks);
                        g_subset_hist.print_report();
                    }
                    g_profiler.reset();
                    g_subset_hist.reset();
                }
            }
        } else {
            // HHS-FSI (per paper's Algorithm 4)
            // Note: the post-step force+timestep recomputation is only needed for
            // diagnostics. HHS recomputes adaptive timesteps at the start of each step from
            // the same state, so skipping it for non-diagnostic steps saves a
            // full O(N^2) pass per step.
            while (t < t_final) {
                auto step_t0 = std::chrono::high_resolution_clock::now();

                if (use_distributed) {
                    hhs_fsi_step_distributed(dp, dt_max, G, soft2, dt_min, eta, 2);
                } else {
                    hhs_fsi_step_dynamic(p, 0, p.N, dt_max, G, soft2, dt_min, eta);
                }

                t += dt_max;
                step++;

                auto step_t1 = std::chrono::high_resolution_clock::now();
                double step_wall = std::chrono::duration<double>(step_t1 - step_t0).count();
                g_profiler.step_wall_total += step_wall;
                g_profiler.step_count++;

                if (my_rank == 0 && step <= 5) {
                    std::printf("Step %d: %.3fs, dt=%.2e Myr\n",
                        step, step_wall, dt_max*TIME_UNIT_MYR);
                    std::fflush(stdout);
                }

                if (step % print_interval == 0) {
                    // Recompute forces + timestep only for diagnostics
                    if (use_distributed) {
                        compute_acceleration_and_timestep_distributed(dp, G, soft2, eta);
                    } else {
                        compute_acceleration_and_timestep(work_p, G, soft2, eta);
                    }
                    assign_levels(work_p, levels);
                    work_p.partition_by_level();
                    print_progress(t, step);
                    if (my_rank == 0 && profiling_enabled) {
                        g_profiler.print_report(N_display, n_ranks);
                        g_subset_hist.print_report();
                    }
                    g_profiler.reset();
                    g_subset_hist.reset();
                }
            }
        }

        auto wall_end = std::chrono::high_resolution_clock::now();
        double wall_sec = std::chrono::duration<double>(wall_end - wall_start).count();

        // Final diagnostics: recompute forces + timestep if not already done
        if (step % print_interval != 0) {
            if (use_distributed) {
                compute_acceleration_and_timestep_distributed(dp, G, soft2, eta);
            } else {
                compute_acceleration_and_timestep(work_p, G, soft2, eta);
            }
            assign_levels(work_p, levels);
            work_p.partition_by_level();
        }
        print_progress(t, step);
        if (my_rank == 0 && profiling_enabled && g_profiler.step_count > 0) {
            g_profiler.print_report(N_display, n_ranks);
            g_subset_hist.print_report();
        }
        g_profiler.reset();
        g_subset_hist.reset();

        auto diag_final = compute_diagnostics(work_p, G, soft2);
        if (my_rank == 0) {
            print_final(diag_final, diag0, work_p, levels, is_cluster, wall_sec, step, names);
        }

        if (use_distributed) {
            dp.local.deallocate();
        } else {
            p.deallocate();
        }
    }
    Kokkos::finalize();
    MPIComm::finalize();
    return 0;
}
