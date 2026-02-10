#include "particles.h"
#include "forces.h"
#include "integrator.h"
#include "timestep.h"
#include "initial_conditions.h"
#include "diagnostics.h"
#include "constants.h"
#include "mpi_comm.h"
#include "profiling.h"

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
        direct_n2_threshold = std::stoi(cfg.get_str("DIRECT_N2_ON_DEEP_LEVELS", "0"));
        cpu_serial_threshold = std::stoi(cfg.get_str("DIRECT_CPU_N2_ON_VERY_DEEP_LEVELS", "0"));
        fuse_gradient = (cfg.get_str("FUSE_GRADIENT", "false") == "true");
        use_integer_positions = (cfg.get_str("INTEGER_POSITIONS", "false") == "true");
        double domain_halfsize = cfg.get_double("DOMAIN_HALFSIZE_PC", 200.0);
        use_nonblocking_mpi = (cfg.get_str("MPI_NON_BLOCKING", "false") == "true");
        symmetric_timesteps = (cfg.get_str("SYMMETRIC_TIMESTEPS", "false") == "true");

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
        if (my_rank == 0 || !use_distributed) {
            if (test == "outer_planets") {
                init_outer_planets(p);
                names = outer_planet_names;
            } else if (test == "plummer") {
                init_plummer(p, N_plummer, M_total_msun, r_half_pc, seed, mass_dist, age_gyr, metallicity);
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
            std::printf("\n");
            std::fflush(stdout);
            if (profiling_enabled) {
                g_profiler.print_memory_report(N_display, n_ranks);
            }
        }

        auto levels = build_levels(dt_max, dt_min);

        // Get reference to working particles (local for distributed, p for single)
        Particles& work_p = use_distributed ? dp.local : p;

        // Initial force computation + adaptive timestep
        if (my_rank == 0) {
            std::printf("Computing initial forces...\n"); std::fflush(stdout);
        }
        auto t0 = std::chrono::high_resolution_clock::now();

        if (use_distributed) {
            // Scatter-compute-gather: each rank computes its slice, then allgather
            compute_acceleration_and_timestep_distributed(dp, G, soft2, eta);
        } else {
            compute_acceleration_and_timestep(work_p, G, soft2, eta);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        if (my_rank == 0) {
            std::printf("Initial forces: %.3f s\n", std::chrono::duration<double>(t1-t0).count());
            std::fflush(stdout);
        }

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
                    }
                    g_profiler.reset();
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
                    }
                    g_profiler.reset();
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
        }
        g_profiler.reset();

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
