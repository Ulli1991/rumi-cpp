#include "particles.h"
#include "forces.h"
#include "integrator.h"
#include "timestep.h"
#include "initial_conditions.h"
#include "diagnostics.h"
#include "constants.h"

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
    Kokkos::initialize(argc, argv);
    {
        std::printf("Kokkos configuration:\n"); std::fflush(stdout);
        Kokkos::print_configuration(std::cout);
        std::cout << std::flush;

        std::string config_file = "config.txt";
        std::string test_override;

        for (int i = 1; i < argc; i++) {
            if ((!std::strcmp(argv[i], "--config") || !std::strcmp(argv[i], "-c")) && i+1 < argc)
                config_file = argv[++i];
            else if (!std::strcmp(argv[i], "--test") && i+1 < argc)
                test_override = argv[++i];
            else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
                print_usage(argv[0]); Kokkos::finalize(); return 0;
            }
            // Silently skip Kokkos args (--kokkos-*)
            else if (std::strncmp(argv[i], "--kokkos", 8) == 0) {
                // Skip value arg if present
                if (i+1 < argc && argv[i+1][0] != '-') i++;
            }
            else {
                std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
                print_usage(argv[0]); Kokkos::finalize(); return 1;
            }
        }

        Config cfg;
        cfg.load(config_file);

        std::string test = test_override.empty() ? cfg.get_str("TEST_CASE", "outer_planets") : test_override;
        std::string integrator = cfg.get_str("INTEGRATOR", "hhs");
        double t_final_yr   = cfg.get_double("TOTAL_TIME_YEARS", 2.5e8);
        double dt_max_yr    = cfg.get_double("DT_MAX_YEARS", 1e4);
        double dt_min_yr    = cfg.get_double("DT_MIN_YEARS", 100.0);
        double eta          = cfg.get_double("ETA", 0.02);
        double soft_au      = cfg.get_double("SOFTENING_AU", 206.265);
        int print_interval  = cfg.get_int("PRINT_INTERVAL", 1000);
        int N_plummer       = cfg.get_int("PLUMMER_N", 1000);
        double M_total_msun = cfg.get_double("PLUMMER_M_TOTAL", 3.79e2);
        double r_half_pc    = cfg.get_double("PLUMMER_R_HALF", 3.5);
        int seed            = cfg.get_int("PLUMMER_SEED", 42);
        std::string mass_dist = cfg.get_str("PLUMMER_MASS_DIST", "equal");
        double age_gyr      = cfg.get_double("PLUMMER_AGE_GYR", 1.0);
        double metallicity  = cfg.get_double("PLUMMER_METALLICITY", 0.02);

        grad_n3l = (cfg.get_str("GRAD_N3L", "false") == "true");
        use_kahan = (cfg.get_str("USE_KAHAN", "false") == "true");
        use_memory_tiling = (cfg.get_str("MEMORY_TILING", "true") == "true");

        // USE_N3L: default true on CPU-like backends, false on GPU
        bool use_n3l_default = true;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
        use_n3l_default = false;
#endif
        bool use_n3l = (cfg.get_str("USE_N3L", use_n3l_default ? "true" : "false") == "true");

        // Set force dispatch
        set_force_dispatch(use_n3l);

        double G = G_CGS;
        double soft_cm = soft_au * LENGTH_UNIT;
        double soft2 = soft_cm * soft_cm;
        double dt_max = dt_max_yr * TIME_UNIT;
        double dt_min = dt_min_yr * TIME_UNIT;
        double t_final = t_final_yr * TIME_UNIT;

        Particles p;
        const char* const* names = nullptr;
        std::printf("Initializing particles...\n"); std::fflush(stdout);
        auto t_init0 = std::chrono::high_resolution_clock::now();
        if (test == "outer_planets") {
            init_outer_planets(p);
            names = outer_planet_names;
        } else if (test == "plummer") {
            init_plummer(p, N_plummer, M_total_msun, r_half_pc, seed, mass_dist, age_gyr, metallicity);
        } else {
            std::fprintf(stderr, "Unknown test case: %s\n", test.c_str());
            Kokkos::finalize(); return 1;
        }
        auto t_init1 = std::chrono::high_resolution_clock::now();
        std::printf("Particle init: %.3f s\n", std::chrono::duration<double>(t_init1-t_init0).count());
        std::fflush(stdout);

        bool is_cluster = (test == "plummer");

        const char* mode_str = (integrator == "hhs") ? "hhs" : "fsi";
        std::printf("=== Kokkos N-Body: %s, %s test, N=%d, N3L=%s ===\n",
                    mode_str, test.c_str(), p.N, use_n3l ? "true" : "false");
        std::fflush(stdout);
        std::printf("G=%.4e  soft=%.4e cm  dt_max=%.4e yr  dt_min=%.4e yr  eta=%.4f\n",
                    G, soft_cm, dt_max_yr, dt_min_yr, eta);
        std::printf("t_final=%.4e yr  print_interval=%d  memory_tiling=%s\n\n",
                    t_final_yr, print_interval, use_memory_tiling ? "true" : "false");
        std::fflush(stdout);

        auto levels = build_levels(dt_max, dt_min);

        // Initial force computation + FROST (always use N3L for FROST - single O(N²) computation)
        std::printf("Computing initial forces...\n"); std::fflush(stdout);
        auto t0 = std::chrono::high_resolution_clock::now();
        compute_acceleration_and_frost<true>(p, G, soft2, eta);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::printf("Initial forces: %.3f s\n", std::chrono::duration<double>(t1-t0).count());
        std::fflush(stdout);

        assign_levels(p, levels);
        p.partition_by_level();

        print_timebin_table(p, levels, -1.0, 0.0, 0.0, 0.0, 0, is_cluster);

        // Print dt range
        {
            auto h_dt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.dt_frost);
            double mn = 1e30, mx = 0;
            for (int i = 0; i < p.N; i++) {
                if (h_dt(i) < mn) mn = h_dt(i);
                if (h_dt(i) > mx) mx = h_dt(i);
            }
            std::printf("Particle dt range: [%.4e, %.4e] yr\n\n", mn/TIME_UNIT, mx/TIME_UNIT);
        }

        auto diag0 = compute_diagnostics(p, G, soft2);

        std::printf("Running...\n"); std::fflush(stdout);

        auto wall_start = std::chrono::high_resolution_clock::now();

        double t = 0;
        int step = 0;

        auto print_progress = [&](double t_now, int step_now) {
            auto diag = compute_diagnostics(p, G, soft2);
            double dE, dP, dL;
            compute_errors(diag, diag0, p, is_cluster, dE, dP, dL);
            print_timebin_table(p, levels, dE, dP, dL, t_now/TIME_UNIT, step_now, is_cluster);
        };

        if (integrator == "fsi") {
            double dt_global = levels.back();
            while (t < t_final) {
                fsi_step(p, 0, p.N, G, soft2, dt_global);
                t += dt_global;
                step++;
                if (step % print_interval == 0) {
                    compute_acceleration_and_frost<true>(p, G, soft2, eta);
                    assign_levels(p, levels);
                    p.partition_by_level();
                    print_progress(t, step);
                }
            }
        } else {
            // HHS-FSI (per paper's Algorithm 4)
            while (t < t_final) {
                auto step_t0 = std::chrono::high_resolution_clock::now();

                auto hhs_t0 = std::chrono::high_resolution_clock::now();
                hhs_fsi_step_dynamic(p, 0, p.N, dt_max, G, soft2, dt_min, eta, use_n3l);
                auto hhs_t1 = std::chrono::high_resolution_clock::now();

                // Recompute forces and FROST for diagnostics and next step (always N3L)
                auto frost_t0 = std::chrono::high_resolution_clock::now();
                compute_acceleration_and_frost<true>(p, G, soft2, eta);
                auto frost_t1 = std::chrono::high_resolution_clock::now();

                // Restore level structure for diagnostics
                assign_levels(p, levels);
                p.partition_by_level();

                t += dt_max;
                step++;

                auto step_t1 = std::chrono::high_resolution_clock::now();

                // Print timing for first few steps
                if (step <= 5) {
                    std::printf("Step %d: HHS=%.3fs, FROST=%.3fs, total=%.3fs, dt=%.2e yr\n",
                        step,
                        std::chrono::duration<double>(hhs_t1-hhs_t0).count(),
                        std::chrono::duration<double>(frost_t1-frost_t0).count(),
                        std::chrono::duration<double>(step_t1-step_t0).count(),
                        dt_max/TIME_UNIT);
                    std::fflush(stdout);
                }

                if (step % print_interval == 0) {
                    print_progress(t, step);
                }
            }
        }

        auto wall_end = std::chrono::high_resolution_clock::now();
        double wall_sec = std::chrono::duration<double>(wall_end - wall_start).count();

        print_progress(t, step);

        auto diag_final = compute_diagnostics(p, G, soft2);
        print_final(diag_final, diag0, p, levels, is_cluster, wall_sec, step, names);

        p.deallocate();
    }
    Kokkos::finalize();
    return 0;
}
