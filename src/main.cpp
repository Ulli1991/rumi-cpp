#include "particles.h"
#include <numeric>
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
#include "logh_batch.h"
#include "kepler.h"
#include "checkpoint.h"
#include "mergers.h"
#include "merger_tree.h"
#include "runtime_sse.h"
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
        (void)G;
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

// Triple orbit tracker for Kozai-Lidov validation
// Tracks inner binary (a, e, omega) and mutual inclination at each print step.
// Conserved Kozai integral: C_K = (1 - e_in^2) * cos^2(i_mut) = const.
struct TripleTracker {
    double mu_in;            // G*(m1+m2)
    double mu_out;           // G*(m1+m2+m3)
    double m1, m2, m3;
    double C_K0;             // initial Kozai integral
    double a_in0, e_in0;     // initial inner orbital elements
    double i_mut0;           // initial mutual inclination (rad)
    bool initialized;

    TripleTracker() : initialized(false) {}

    // Find array index of particle with given permanent ID
    static int find_by_pid(Particles& p, int target_pid) {
        auto h_pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);
        for (int i = 0; i < p.N; i++) {
            if (h_pid(i) == target_pid) return i;
        }
        return -1;
    }

    // Compute inner binary relative state and outer relative state
    // (outer body relative to inner COM) using particle_id lookup.
    void compute_elements(Particles& p, double G,
                          OrbitalElements& oe_in, OrbitalElements& oe_out,
                          double& i_mut, double& C_K) {
        (void)G;
        // Look up particles by permanent ID (survives partition_by_level reordering)
        int i0 = find_by_pid(p, 0);  // inner body 1
        int i1 = find_by_pid(p, 1);  // inner body 2
        int i2 = find_by_pid(p, 2);  // outer body
        if (i0 < 0 || i1 < 0 || i2 < 0) return;

        auto h_x  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
        auto h_y  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
        auto h_z  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
        auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
        auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
        auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
        auto h_m  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);

        // Inner binary: relative vector
        double r_in[3] = { h_x(i1)-h_x(i0),  h_y(i1)-h_y(i0),  h_z(i1)-h_z(i0)  };
        double v_in[3] = { h_vx(i1)-h_vx(i0), h_vy(i1)-h_vy(i0), h_vz(i1)-h_vz(i0) };
        oe_in = cartesian_to_elements(r_in, v_in, mu_in);

        // Inner COM
        double M_in = m1 + m2;
        double cx  = (m1*h_x(i0)  + m2*h_x(i1))  / M_in;
        double cy  = (m1*h_y(i0)  + m2*h_y(i1))  / M_in;
        double cz  = (m1*h_z(i0)  + m2*h_z(i1))  / M_in;
        double cvx = (m1*h_vx(i0) + m2*h_vx(i1)) / M_in;
        double cvy = (m1*h_vy(i0) + m2*h_vy(i1)) / M_in;
        double cvz = (m1*h_vz(i0) + m2*h_vz(i1)) / M_in;

        // Outer orbit: outer body relative to inner COM
        double r_out[3] = { h_x(i2)-cx,  h_y(i2)-cy,  h_z(i2)-cz  };
        double v_out[3] = { h_vx(i2)-cvx, h_vy(i2)-cvy, h_vz(i2)-cvz };
        oe_out = cartesian_to_elements(r_out, v_out, mu_out);

        // Angular momentum vectors
        double L_in[3]  = { r_in[1]*v_in[2]  - r_in[2]*v_in[1],
                            r_in[2]*v_in[0]  - r_in[0]*v_in[2],
                            r_in[0]*v_in[1]  - r_in[1]*v_in[0] };
        double L_out[3] = { r_out[1]*v_out[2] - r_out[2]*v_out[1],
                            r_out[2]*v_out[0] - r_out[0]*v_out[2],
                            r_out[0]*v_out[1] - r_out[1]*v_out[0] };

        double L_in_mag  = std::sqrt(L_in[0]*L_in[0]  + L_in[1]*L_in[1]  + L_in[2]*L_in[2]);
        double L_out_mag = std::sqrt(L_out[0]*L_out[0] + L_out[1]*L_out[1] + L_out[2]*L_out[2]);

        double cos_i = (L_in[0]*L_out[0] + L_in[1]*L_out[1] + L_in[2]*L_out[2])
                       / (L_in_mag * L_out_mag + 1e-30);
        cos_i = std::max(-1.0, std::min(1.0, cos_i));
        i_mut = std::acos(cos_i);
        C_K = (1.0 - oe_in.e * oe_in.e) * cos_i * cos_i;
    }

    void init(Particles& p, double G) {
        if (p.N != 3) return;
        int i0 = find_by_pid(p, 0);
        int i1 = find_by_pid(p, 1);
        int i2 = find_by_pid(p, 2);
        if (i0 < 0 || i1 < 0 || i2 < 0) return;
        auto h_m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
        m1 = h_m(i0); m2 = h_m(i1); m3 = h_m(i2);
        mu_in  = G * (m1 + m2);
        mu_out = G * (m1 + m2 + m3);

        OrbitalElements oe_in, oe_out;
        double i_mut;
        compute_elements(p, G, oe_in, oe_out, i_mut, C_K0);
        a_in0  = oe_in.a;
        e_in0  = oe_in.e;
        i_mut0 = i_mut;

        std::printf("TripleTracker: inner a=%.4e pc (%.4f AU), e=%.6f, P=%.4e yr\n",
                    oe_in.a, oe_in.a / AU_TO_PC, oe_in.e, oe_in.period * TIME_UNIT_YR);
        std::printf("  outer a=%.4e pc (%.4f AU), e=%.6f, P=%.4e yr\n",
                    oe_out.a, oe_out.a / AU_TO_PC, oe_out.e, oe_out.period * TIME_UNIT_YR);
        std::printf("  i_mut=%.2f deg, Kozai C_K=%.8f\n",
                    i_mut * 180.0 / M_PI, C_K0);
        std::fflush(stdout);
        initialized = true;
    }

    void update(Particles& p, double G, double t_code) {
        if (!initialized || p.N != 3) return;

        OrbitalElements oe_in, oe_out;
        double i_mut, C_K;
        compute_elements(p, G, oe_in, oe_out, i_mut, C_K);

        double dC = (std::fabs(C_K0) > 1e-30) ? (C_K - C_K0) / C_K0 : C_K - C_K0;
        double t_yr = t_code * TIME_UNIT_YR;

        std::printf("  Triple: t=%.4f yr  a_in=%.4e  e_in=%.6f  i_mut=%.2f deg"
                    "  a_out=%.4e  e_out=%.6f  C_K=%.8f  dC/C0=%.2e\n",
                    t_yr, oe_in.a, oe_in.e, i_mut * 180.0 / M_PI,
                    oe_out.a, oe_out.e, C_K, dC);
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
#include <set>
#include <fstream>
#include <chrono>
#include <iostream>
#include <cmath>

// ---------------------------------------------------------------------------
// Config file parser
// ---------------------------------------------------------------------------
struct Config {
    std::map<std::string, std::string> params;
    mutable std::set<std::string> consumed_keys;

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
        consumed_keys.insert(key);
        auto it = params.find(key);
        return (it != params.end()) ? it->second : def;
    }
    double get_double(const std::string& key, double def) const {
        consumed_keys.insert(key);
        auto it = params.find(key);
        if (it == params.end()) return def;
        try {
            return std::stod(it->second);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "WARNING: Cannot parse '%s' as double for key '%s': %s\n",
                         it->second.c_str(), key.c_str(), e.what());
            return def;
        }
    }
    int get_int(const std::string& key, int def) const {
        consumed_keys.insert(key);
        auto it = params.find(key);
        if (it == params.end()) return def;
        try {
            return std::stoi(it->second);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "WARNING: Cannot parse '%s' as int for key '%s': %s\n",
                         it->second.c_str(), key.c_str(), e.what());
            return def;
        }
    }
    bool get_bool(const std::string& key, bool def) const {
        consumed_keys.insert(key);
        auto it = params.find(key);
        if (it == params.end()) return def;
        const std::string& v = it->second;
        if (v == "true" || v == "TRUE" || v == "True" || v == "1" || v == "yes") return true;
        if (v == "false" || v == "FALSE" || v == "False" || v == "0" || v == "no") return false;
        std::fprintf(stderr, "WARNING: Cannot parse '%s' as bool for key '%s', using default\n",
                     v.c_str(), key.c_str());
        return def;
    }
    void warn_unknown_keys() const {
        for (const auto& kv : params) {
            if (consumed_keys.find(kv.first) == consumed_keys.end()) {
                std::fprintf(stderr, "WARNING: Unknown config key '%s' (value: '%s')\n",
                             kv.first.c_str(), kv.second.c_str());
            }
        }
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
    int exit_code = 0;
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
        double eta          = cfg.get_double("ETA", 0.2);
        double soft_pc      = cfg.get_double("SOFTENING_PC", 1e-3);
        int print_interval  = cfg.get_int("PRINT_INTERVAL", 1000);
        int replica_check_interval = cfg.get_int("REPLICA_CHECK_INTERVAL", 10);  // steps; 0 = off
        int N_plummer       = cfg.get_int("PLUMMER_N", 1000);
        double M_total_msun = cfg.get_double("PLUMMER_M_TOTAL", 3.79e2);
        double r_half_pc    = cfg.get_double("PLUMMER_R_HALF", 3.5);
        int seed            = cfg.get_int("PLUMMER_SEED", 42);
        std::string mass_dist = cfg.get_str("PLUMMER_MASS_DIST", "equal");
        double age_gyr      = cfg.get_double("PLUMMER_AGE_GYR", 1.0);
        double metallicity  = cfg.get_double("PLUMMER_METALLICITY", 0.02);
        bool remove_bh      = cfg.get_bool("REMOVE_BH", false);
        double binary_frac  = cfg.get_double("BINARY_FRAC", 0.0);
        double bin_a_min_au = cfg.get_double("BINARY_A_MIN_AU", 0.01);
        double bin_a_max_au = cfg.get_double("BINARY_A_MAX_AU", 100.0);
        bool add_imbh       = cfg.get_bool("ADD_IMBH", false);
        double imbh_m1      = cfg.get_double("IMBH_M1", 500.0);
        double imbh_m2      = cfg.get_double("IMBH_M2", 500.0);
        double imbh_a_pc    = cfg.get_double("IMBH_A_PC", 1e-4);
        double imbh_e       = cfg.get_double("IMBH_E", 0.5);
        double imbh_spin    = cfg.get_double("IMBH_SPIN", 0.0);
        double r_break_au   = cfg.get_double("R_BREAK_AU", 0.0);

        // SSE model config (used by IC generator and runtime SSE)
        std::string sse_wind_str    = cfg.get_str("SSE_WIND_MODEL", "HURLEY");
        std::string sse_remnant_str = cfg.get_str("SSE_REMNANT_MODEL", "BELCZYNSKI02");
        std::string sse_ppisn_str   = cfg.get_str("SSE_PPISN_MODEL", "NONE");
        std::string sse_ecsn_str    = cfg.get_str("SSE_ECSN_MODEL", "OFF");
        double sse_mxns_val         = cfg.get_double("SSE_MXNS", -1.0);
        double sse_flbv_val         = cfg.get_double("SSE_FLBV", 1.5);
        SSEConfig ic_sse_cfg = make_sse_config(sse_wind_str, sse_remnant_str,
                                                sse_ppisn_str, sse_ecsn_str,
                                                sse_mxns_val, sse_flbv_val);

        bool profiling_enabled = cfg.get_bool("PROFILING", true);
        serial_n2_gpu = cfg.get_int("SERIAL_N2_GPU", 0);
        serial_n2_cpu = cfg.get_int("SERIAL_N2_CPU", 0);
        all_slow_split = cfg.get_bool("ALL_SLOW_SPLIT", false);
        use_integer_positions = cfg.get_bool("INTEGER_POSITIONS", false);
        double domain_halfsize = cfg.get_double("DOMAIN_HALFSIZE_PC", 200.0);
        use_nonblocking_mpi = cfg.get_bool("MPI_NON_BLOCKING", false);

        // Barnes-Hut tree configuration
        use_tree = cfg.get_bool("USE_TREE", false);
        tree_threshold = cfg.get_int("TREE_THRESHOLD", 10000);
        tree_theta = cfg.get_double("TREE_THETA", 0.5);
        tree_n_leaf = cfg.get_int("TREE_N_LEAF", 32);
        tree_n_group = cfg.get_int("TREE_N_GROUP", 64);

#ifdef ENABLE_COLLISIONAL
        // ETA_FF/ETA_FB always parsed (used by force kernels even without subsystems)
        g_collisional.eta_ff          = cfg.get_double("ETA_FF", 0.2);
        g_collisional.eta_fb          = cfg.get_double("ETA_FB", 0.2);

        // Collisional (unsoftened direct N-body) configuration
        g_collisional.enabled = cfg.get_bool("COLLISIONAL", false);
        if (g_collisional.enabled) {
            g_collisional.r_ngb           = cfg.get_double("R_NGB", 1e-3);
            g_collisional.r_ngb_max       = cfg.get_double("R_NGB_MAX", 0.1);
            g_collisional.eta_grad        = cfg.get_double("ETA_GRAD", 0.1);
            g_collisional.eta_jerk        = cfg.get_double("ETA_JERK", 0.1);
            g_collisional.eta_gbs         = cfg.get_double("ETA_GBS", 1e-10);
            g_collisional.eta_pert        = cfg.get_double("ETA_PERT", 0.02);
            g_collisional.eta_restart     = cfg.get_double("ETA_RESTART", 0.0);
            g_collisional.n_orb_secular   = cfg.get_int("N_ORB_SECULAR", 3);
            g_collisional.m_pn            = cfg.get_double("M_PN", 5.0);
            g_collisional.pn_order        = cfg.get_int("PN_ORDER", 7);
            g_collisional.r_escape_rh     = cfg.get_double("R_ESCAPE_RH", 100.0);
            g_collisional.mstar_threshold = cfg.get_int("MSTAR_THRESHOLD", 8);
            g_collisional.sse_enabled     = cfg.get_bool("SSE_ENABLED", true);
            g_collisional.sse_metallicity = cfg.get_double("SSE_METALLICITY", 0.02);
            g_collisional.sse_wind        = sse_wind_str;
            g_collisional.sse_remnant     = sse_remnant_str;
            g_collisional.sse_ppisn       = sse_ppisn_str;
            g_collisional.sse_ecsn        = sse_ecsn_str;
            g_collisional.sse_mxns        = sse_mxns_val;
            g_collisional.sse_flbv        = sse_flbv_val;
            g_collisional.compute_derived();

            // Collisional mode requires zero softening
            if (soft_pc > 0.0 && my_rank == 0) {
                std::printf("WARNING: COLLISIONAL=true but SOFTENING_PC=%.2e > 0. "
                            "Setting softening to 0.\n", soft_pc);
            }
            soft_pc = 0.0;
        }
#endif

        // Warn about unknown config keys (typos, deprecated options)
        if (my_rank == 0) cfg.warn_unknown_keys();

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

        // Triple test parameters
        double triple_a_in  = cfg.get_double("TRIPLE_A_IN_PC", 1e-8);
        double triple_e_in  = cfg.get_double("TRIPLE_E_IN", 0.1);
        double triple_m1    = cfg.get_double("TRIPLE_M1", 10.0);
        double triple_m2    = cfg.get_double("TRIPLE_M2", 10.0);
        double triple_a_out = cfg.get_double("TRIPLE_A_OUT_PC", 2e-7);
        double triple_e_out = cfg.get_double("TRIPLE_E_OUT", 0.3);
        double triple_m3    = cfg.get_double("TRIPLE_M3", 10.0);
        double triple_incl  = cfg.get_double("TRIPLE_INCL_DEG", 85.0);

        if (my_rank == 0 || !use_distributed) {
            if (test == "outer_planets") {
                init_outer_planets(p);
                names = outer_planet_names;
            } else if (test == "plummer") {
                init_plummer(p, N_plummer, M_total_msun, r_half_pc, seed, mass_dist, age_gyr, metallicity,
                             remove_bh, binary_frac, bin_a_min_au, bin_a_max_au,
                             add_imbh, imbh_m1, imbh_m2, imbh_a_pc, imbh_e, imbh_spin,
                             r_break_au, &ic_sse_cfg);
            } else if (test == "binary_test") {
                init_binary_test(p, binary_a_pc, binary_e, binary_m1, binary_m2);
            } else if (test == "binary_test_peri") {
                init_binary_test_peri(p, binary_a_pc, binary_e, binary_m1, binary_m2);
            } else if (test == "triple_test") {
                init_triple_test(p, triple_a_in, triple_e_in, triple_m1, triple_m2,
                                 triple_a_out, triple_e_out, triple_m3, triple_incl);
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

        // Check for duplicate positions in initial conditions (sort-based O(N log N))
        if (my_rank == 0) {
            Particles& check_p = use_distributed ? dp.local : p;
            auto hx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), check_p.x);
            auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), check_p.y);
            auto hz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), check_p.z);
            std::vector<int> idx(check_p.N);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](int a, int b) {
                if (hx(a) != hx(b)) return hx(a) < hx(b);
                if (hy(a) != hy(b)) return hy(a) < hy(b);
                return hz(a) < hz(b);
            });
            int n_dup = 0;
            for (int k = 1; k < check_p.N; k++) {
                int i = idx[k-1], j = idx[k];
                if (hx(i) == hx(j) && hy(i) == hy(j) && hz(i) == hz(j)) {
                    std::printf("  IC DUPLICATE: idx %d and %d at (%.15e, %.15e, %.15e)\n",
                                i, j, hx(i), hy(i), hz(i));
                    if (++n_dup >= 20) break;
                }
            }
            if (n_dup == 0)
                std::printf("  IC check: no duplicate positions found\n");
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
            std::printf("G=%.4e  soft=%.4e pc  dt_max=%.4e Myr  dt_min=%.4e Myr",
                        G, soft_pc, dt_max_myr, dt_min_myr);
#ifdef ENABLE_COLLISIONAL
            std::printf("  eta_ff=%.4f  eta_fb=%.4f\n",
                        g_collisional.eta_ff, g_collisional.eta_fb);
#else
            std::printf("  eta=%.4f\n", eta);
#endif
            std::printf("t_final=%.4e Myr  print_interval=%d  profiling=%s\n",
                        t_final_myr, print_interval,
                        profiling_enabled ? "true" : "false");
            if (use_integer_positions) {
                std::printf("Integer positions: domain=%.1f pc, resolution=%.2e pc\n",
                            domain_halfsize, int_pos_scale);
            }
            if (use_tree) {
                std::printf("Barnes-Hut tree: theta=%.2f, n_leaf=%d, n_group=%d, threshold=%d\n",
                            tree_theta, tree_n_leaf, tree_n_group, tree_threshold);
            }
#ifdef ENABLE_COLLISIONAL
            if (g_collisional.enabled) {
                std::printf("Collisional mode: r_ngb=%.2e, r_ngb_max=%.2e, eta_gbs=%.2e\n",
                            g_collisional.r_ngb, g_collisional.r_ngb_max, g_collisional.eta_gbs);
                std::printf("  PN order=%d, M_PN=%.1f M_sun, MSTAR threshold=%d\n",
                            g_collisional.pn_order, g_collisional.m_pn, g_collisional.mstar_threshold);
                if (g_collisional.sse_enabled) {
                    std::printf("  SSE: Z=%.4f, wind=%s, remnant=%s, ppisn=%s, ecsn=%s\n",
                                g_collisional.sse_metallicity,
                                g_collisional.sse_wind.c_str(),
                                g_collisional.sse_remnant.c_str(),
                                g_collisional.sse_ppisn.c_str(),
                                g_collisional.sse_ecsn.c_str());
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

        // Initialize SSE context for runtime stellar evolution
        SSEContext* g_sse_ctx = nullptr;
        if (g_collisional.enabled && g_collisional.sse_enabled) {
            auto sse_cfg = make_sse_config(g_collisional.sse_wind,
                                            g_collisional.sse_remnant,
                                            g_collisional.sse_ppisn,
                                            g_collisional.sse_ecsn,
                                            g_collisional.sse_mxns,
                                            g_collisional.sse_flbv);
            g_sse_ctx = new SSEContext(g_collisional.sse_metallicity, sse_cfg);
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
        // Initial subsystem detection using standalone neighbor search
        if (g_collisional.enabled) {
            g_subsystem_mgr.reset_ngb_count();
            neighbor_search_subset(work_p, 0, work_p.N, G,
                                   g_collisional.r_ngb, g_collisional.r_ngb_max, dt_max);
            g_subsystem_mgr.detect_from_pairs(work_p, G, dt_max,
                                               g_collisional.r_ngb, g_collisional.r_ngb_max,
                                               g_collisional.n_orb_secular);
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
        if ((test == "binary_test" || test == "binary_test_peri") && g_collisional.enabled && my_rank == 0) {
            bin_tracker.init(work_p, G);
        }
        TripleTracker triple_tracker;
        if (test == "triple_test" && g_collisional.enabled && my_rank == 0) {
            triple_tracker.init(work_p, G);
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
                if (triple_tracker.initialized) {
                    triple_tracker.update(work_p, G, t_now);
                }
                if (g_collisional.enabled && !g_subsystem_mgr.active.empty()) {
                    print_subsystem_diagnostics(work_p, G);
                }
#endif
            }
        };

        if (integrator == "fsi") {
            // All particles at the fastest occupied level (FROST sec. 6.2), not the floor of the
            // level table (levels.back() is DT_MIN: ~1e11 steps for a 1000 yr run)
            double dt_min_particle = 1e300;
            {
                auto h_dt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), work_p.dt_adapt);
                for (int i = 0; i < work_p.N; i++) dt_min_particle = std::min(dt_min_particle, h_dt(i));
            }
            double dt_global = levels.back();
            for (double lv : levels) {
                if (lv <= dt_min_particle) { dt_global = lv; break; }
            }
            if (my_rank == 0) std::printf("FSI global step: %.4e Myr\n", dt_global * TIME_UNIT_MYR);
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
                    g_subset_hist.reset();
                }
            }
        } else {
            // HHS-FSI (per paper's Algorithm 4)
            // Note: the post-step force+timestep recomputation is only needed for
            // diagnostics. HHS recomputes adaptive timesteps at the start of each step from
            // the same state, so skipping it for non-diagnostic steps saves a
            // full O(N^2) pass per step.
#ifdef ENABLE_COLLISIONAL
            Checkpoint checkpoint;
#endif
            while (t < t_final) {
                auto step_t0 = std::chrono::high_resolution_clock::now();
#ifdef ENABLE_COLLISIONAL
                g_sim_time = t;
#endif

#ifdef ENABLE_COLLISIONAL
                // Runtime stellar evolution before each step
                if (g_collisional.enabled && g_collisional.sse_enabled && g_sse_ctx) {
                    double dt_sse_myr = dt_max * TIME_UNIT_MYR;
                    advance_stellar_evolution(work_p, *g_sse_ctx, dt_sse_myr,
                                              g_energy_books, G);
                }

                // ============================================================
                // Adaptive restart loop (BIFROST §3.19)
                // If energy error is too large, restore checkpoint, reduce all
                // eta parameters by 50%, increase r_ngb by 15%, and retry.
                // Escape valves: max retries exhausted, or error not decreasing.
                // ============================================================
                if (g_collisional.enabled && g_collisional.eta_restart > 0.0) {
                    // Per-step reference energy (BIFROST §3.19: "beginning of the interval")
                    // BIFROST Eq.: |ΔE_s/E| = |E(t) - E(t-τ)| / |E(t₀)|
                    //   numerator: per-step change (E_after - E_step_start)
                    //   denominator: initial energy E(t₀) = diag0.E_total
                    g_profiler.start();
                    double E_step_start = compute_total_energy(work_p, g_energy_books, G, soft2);
                    g_profiler.stop(StepProfiler::ENERGY_CHECK);

                    // Save originals ONCE before the loop
                    double eta_ff_orig   = g_collisional.eta_ff;
                    double eta_fb_orig   = g_collisional.eta_fb;
                    double eta_gbs_orig  = g_collisional.eta_gbs;
                    double eta_grad_orig = g_collisional.eta_grad;
                    double eta_jerk_orig = g_collisional.eta_jerk;
                    double eta_pert_orig = g_collisional.eta_pert;
                    double r_ngb_orig    = g_collisional.r_ngb;

                    constexpr int max_restarts = 10;
                    double prev_dE = 1e30;
                    Checkpoint best;  // post-step state of the attempt with the smallest error
                    double best_dE = 1e30;

                    for (int restart = 0; restart <= max_restarts; restart++) {
                        // Save checkpoint
                        checkpoint.save(work_p, g_subsystem_mgr, g_energy_books);

                        // Run HHS step
                        if (use_distributed) {
                            hhs_fsi_step_distributed(dp, dt_max, G, soft2, dt_min, eta, 2);
                        } else {
                            hhs_fsi_step_dynamic(p, 0, p.N, dt_max, G, soft2, dt_min, eta);
                        }

                        // Per-step error: |E(t) - E(t-τ)| / |E(t₀)|  (BIFROST §3.19)
                        g_profiler.start();
                        double E_after = compute_total_energy(work_p, g_energy_books, G, soft2);
                        g_profiler.stop(StepProfiler::ENERGY_CHECK);
                        double dE = std::fabs((E_after - E_step_start) / (std::fabs(diag0.E_total) + 1e-30));

                        if (dE < g_collisional.eta_restart) {
                            // Accepted
                            break;
                        }

                        bool give_up = restart == max_restarts || (restart > 0 && dE >= prev_dE);
                        if (give_up) {
                            // Max retries exhausted, or the error grew again (BIFROST: accept the
                            // attempt with the smallest error rather than the last one)
                            if (dE > best_dE) {
                                best.restore(work_p, g_subsystem_mgr, g_energy_books);
                                if (use_distributed) dp.sync_count();
                            }
                            std::printf("RESTART: %s, accepting |dE/E|=%.2e\n",
                                        restart == max_restarts ? "max retries reached" : "dE not decreasing",
                                        std::min(dE, best_dE));
                            break;
                        }

                        prev_dE = dE;
                        if (dE < best_dE) {
                            best_dE = dE;
                            best.save(work_p, g_subsystem_mgr, g_energy_books);
                        }

                        // Restore checkpoint
                        checkpoint.restore(work_p, g_subsystem_mgr, g_energy_books);
                        if (use_distributed) dp.sync_count();

                        // Reduce ALL eta parameters by 50% (BIFROST §3.19)
                        g_collisional.eta_ff   *= 0.5;
                        g_collisional.eta_fb   *= 0.5;
                        g_collisional.eta_gbs  *= 0.5;
                        g_collisional.eta_grad *= 0.5;
                        g_collisional.eta_jerk *= 0.5;
                        g_collisional.eta_pert *= 0.5;

                        // Increase r_ngb by 15%
                        g_collisional.r_ngb *= 1.15;
                        g_collisional.r_ngb_sq = g_collisional.r_ngb * g_collisional.r_ngb;

                        std::printf("RESTART[%d]: |dE/E|=%.2e, reducing etas by 50%%, r_ngb=%.4e\n",
                                    restart + 1, dE, g_collisional.r_ngb);
                    }

                    // Restore original parameters after step (BIFROST: "original parameters are restored")
                    g_collisional.eta_ff   = eta_ff_orig;
                    g_collisional.eta_fb   = eta_fb_orig;
                    g_collisional.eta_gbs  = eta_gbs_orig;
                    g_collisional.eta_grad = eta_grad_orig;
                    g_collisional.eta_jerk = eta_jerk_orig;
                    g_collisional.eta_pert = eta_pert_orig;
                    g_collisional.r_ngb    = r_ngb_orig;
                    g_collisional.r_ngb_sq = r_ngb_orig * r_ngb_orig;
                } else {
#endif
                    // Non-collisional or eta_restart disabled: single step, no restart
                    if (use_distributed) {
                        hhs_fsi_step_distributed(dp, dt_max, G, soft2, dt_min, eta, 2);
                    } else {
                        hhs_fsi_step_dynamic(p, 0, p.N, dt_max, G, soft2, dt_min, eta);
                    }
#ifdef ENABLE_COLLISIONAL
                }
#endif

                t += dt_max;
                step++;

                // Replicated MPI state must stay bitwise identical; stop instead of diverging silently
                if (use_distributed && replica_check_interval > 0 && step % replica_check_interval == 0 &&
                    !dp.replicas_agree(step)) {
                    exit_code = 3;
                    break;
                }

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
#ifdef ENABLE_COLLISIONAL
                    // Periodic escaper removal
                    if (g_collisional.enabled && is_cluster && g_collisional.r_escape_rh > 0) {
                        int n_esc = remove_escapers(work_p, G, g_collisional.r_escape_rh,
                                                     g_subsystem_mgr, g_energy_books);
                        if (n_esc > 0) {
                            g_subsystem_mgr.step_escapers += n_esc;
                            work_p.compact_remove_zero_mass();
                            if (use_distributed) dp.sync_count();
                        }
                    }
                    // Write event logs at snapshot intervals
                    if (my_rank == 0) {
                        g_merger_tree.write_ascii("merger_tree.dat");
                        g_escaper_log.write_ascii("escaper_log.dat");
                    }
#endif
                    if (my_rank == 0 && profiling_enabled) {
                        g_profiler.print_report(N_display, n_ranks);
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
        }
        g_profiler.reset();
        g_subset_hist.reset();

        auto diag_final = compute_diagnostics(work_p, G, soft2);
        if (my_rank == 0) {
            print_final(diag_final, diag0, work_p, levels, is_cluster, wall_sec, step, names);
#ifdef ENABLE_COLLISIONAL
            g_merger_tree.write_ascii("merger_tree.dat");
            g_merger_tree.print_summary();
            g_escaper_log.write_ascii("escaper_log.dat");
#endif
        }

        if (use_distributed) {
            dp.local.deallocate();
        } else {
            p.deallocate();
        }
#ifdef ENABLE_COLLISIONAL
        delete g_sse_ctx;
        // Global Views must be released before Kokkos::finalize
        g_logh_batch = SubsystemBatch();
        g_subsystem_mgr = SubsystemManager();
#endif
    }
    release_integrator_globals();
    Kokkos::finalize();
    MPIComm::finalize();
    return exit_code;
}
