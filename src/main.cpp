#include "particles.h"
#include "forces.h"
#include "integrator.h"
#include "timestep.h"
#include "initial_conditions.h"
#include "diagnostics.h"
#include "constants.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <chrono>

// ---------------------------------------------------------------------------
// Config file parser: reads KEY = VALUE lines, ignores # comments
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

// Particle names for few-body output
static const char* outer_planet_names[] = {"Sun", "Jupiter", "Saturn", "Uranus", "Neptune"};

int main(int argc, char** argv) {
    std::string config_file = "config.txt";
    std::string test_override;

    for (int i = 1; i < argc; i++) {
        if ((!std::strcmp(argv[i], "--config") || !std::strcmp(argv[i], "-c")) && i+1 < argc)
            config_file = argv[++i];
        else if (!std::strcmp(argv[i], "--test") && i+1 < argc)
            test_override = argv[++i];
        else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
            print_usage(argv[0]); return 0;
        }
        else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]); return 1;
        }
    }

    Config cfg;
    cfg.load(config_file);

    std::string test = test_override.empty() ? cfg.get_str("TEST_CASE", "outer_planets") : test_override;
    std::string integrator = cfg.get_str("INTEGRATOR", "hhs");
    HHSMode hhs_mode = parse_hhs_mode(cfg.get_str("HHS_MODE", "recursive"));
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

    double G = G_CGS;
    double soft_cm = soft_au * LENGTH_UNIT;
    double soft2 = soft_cm * soft_cm;
    double dt_max = dt_max_yr * TIME_UNIT;
    double dt_min = dt_min_yr * TIME_UNIT;
    double t_final = t_final_yr * TIME_UNIT;

    // Initialize particles
    Particles p;
    const char* const* names = nullptr;
    if (test == "outer_planets") {
        init_outer_planets(p);
        names = outer_planet_names;
    } else if (test == "plummer") {
        init_plummer(p, N_plummer, M_total_msun, r_half_pc, seed, mass_dist, age_gyr, metallicity);
    } else {
        std::fprintf(stderr, "Unknown test case: %s\n", test.c_str());
        return 1;
    }

    bool is_cluster = (test == "plummer");

    const char* mode_str = (integrator == "hhs")
        ? (hhs_mode == HHSMode::Stack ? "hhs-stack" : "hhs-recursive") : "fsi";
    std::printf("=== C++ N-Body: %s, %s test, N=%d ===\n",
                mode_str, test.c_str(), p.N);
    std::printf("G=%.4e  soft=%.4e cm  dt_max=%.4e yr  dt_min=%.4e yr  eta=%.4f\n",
                G, soft_cm, dt_max_yr, dt_min_yr, eta);
    std::printf("t_final=%.4e yr  print_interval=%d\n\n", t_final_yr, print_interval);

    // Build timestep levels
    auto levels = build_levels(dt_max, dt_min);

    // Initial force computation + FROST
    compute_acceleration_and_frost(p, G, soft2, eta);
    assign_levels(p, levels);
    p.partition_by_level();

    // Print initial timebin table (no diagnostics yet)
    print_timebin_table(p, levels, -1.0, 0.0, 0.0, 0.0, 0, is_cluster);

    std::printf("Particle dt range: [%.4e, %.4e] yr\n\n",
                [&]{ double mn=1e30; for(int i=0;i<p.N;i++) mn=std::min(mn,p.dt_frost[i]); return mn/TIME_UNIT; }(),
                [&]{ double mx=0;    for(int i=0;i<p.N;i++) mx=std::max(mx,p.dt_frost[i]); return mx/TIME_UNIT; }());

    auto diag0 = compute_diagnostics(p, G, soft2);

    std::printf("Running...\n");

    // Pre-allocate HHS stack
    std::vector<StackEntry> hhs_stack;
    hhs_stack.reserve(7 * p.n_levels + 1);

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
                compute_acceleration_and_frost(p, G, soft2, eta);
                assign_levels(p, levels);
                p.partition_by_level();
                print_progress(t, step);
            }
        }
    } else {
        while (t < t_final) {
            int coarsest = find_coarsest_active(p);
            double dt_outer = levels[coarsest];

            if (hhs_mode == HHSMode::Stack)
                hhs_fsi_step_stack(p, coarsest, dt_outer, G, soft2, hhs_stack);
            else
                hhs_fsi_step(p, coarsest, dt_outer, G, soft2, levels);

            compute_acceleration_and_frost(p, G, soft2, eta);
            assign_levels(p, levels);
            p.partition_by_level();

            t += dt_outer;
            step++;

            if (step % print_interval == 0) {
                print_progress(t, step);
            }
        }
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_sec = std::chrono::duration<double>(wall_end - wall_start).count();

    // Final timebin table + results
    print_progress(t, step);

    auto diag_final = compute_diagnostics(p, G, soft2);
    print_final(diag_final, diag0, p, levels, is_cluster, wall_sec, step, names);

    p.deallocate();
    return 0;
}
