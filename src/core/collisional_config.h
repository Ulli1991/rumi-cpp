#pragma once

#include <string>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <initializer_list>

// Collisional N-body configuration (BIFROST-style)
// All parameters under "Unsoftened direct N-body options" in config file

struct CollisionalConfig {
    bool enabled = false;               // master toggle (COLLISIONAL)

    // Neighbor search
    double r_ngb = 1e-3;               // neighbor search radius (pc)
    double r_ngb_max = 0.1;            // max distance for orbital element check (pc)

    // Timestep parameters
    double eta_ff = 0.2;               // free-fall timestep parameter
    double eta_fb = 0.2;               // fly-by timestep parameter
    double eta_grad = 0.1;             // gradient timestep parameter
    double eta_jerk = 0.1;             // jerk timestep parameter
    double eta_va = 0.2;               // velocity-per-acceleration timestep parameter (ETA; BIFROST Eq. 24)
    double eta_gbs = 1e-10;            // GBS accuracy for LogH
    double eta_pert = 0.02;            // perturbation timestep for subsystems
    double tidal_gamma = 0.0;          // TIDAL_GAMMA: binaries with |tidal accel| / |internal accel| above this resolve
    double tidal_res = 0.05;           //   their orbit, dt <= TIDAL_RES x period (0 = off; dense cores)
    double eta_restart = 1e-6;         // energy error restart threshold

    // Secular evolution
    int n_orb_secular = 3;             // orbits/timestep threshold for secular

    // Post-Newtonian
    double m_pn = 5.0;                 // minimum mass (M_sun) for global PN1.0
    int pn_order = 7;                  // bitmask: 1=PN1, 2=PN2, 4=PN2.5, 8=PN3, 16=PN3.5

    // Escaper removal
    double r_escape_rh = 100.0;        // escape radius in half-mass radii

    // MSTAR
    int mstar_threshold = 8;           // bodies threshold for MSTAR vs LogH

    // Stellar evolution
    bool sse_enabled = true;           // runtime stellar evolution
    double sse_metallicity = 0.02;     // metallicity for SSE
    std::string sse_wind = "HURLEY";            // Wind model
    std::string sse_remnant = "BELCZYNSKI02";   // Remnant model
    std::string sse_ppisn = "NONE";             // PPISN model
    std::string sse_ecsn = "OFF";               // ECSN model
    double sse_mxns = -1.0;                     // Max NS mass (-1 = auto)
    double sse_flbv = 1.5;                      // LBV scaling factor

    // Derived quantities (computed at init)
    double r_ngb_sq = 0.0;            // r_ngb²
    double r_ngb_max_sq = 0.0;        // r_ngb_max²
    double c_code = 0.0;              // speed of light in code units

    void compute_derived() {
        // Validate parameters
        auto check = [](bool cond, const char* msg) {
            if (!cond) { std::fprintf(stderr, "CollisionalConfig error: %s\n", msg); std::exit(1); }
        };
        check(r_ngb > 0, "r_ngb must be > 0");
        check(r_ngb_max >= r_ngb, "r_ngb_max must be >= r_ngb");
        check(eta_gbs > 0, "eta_gbs must be > 0");
        check(eta_ff > 0, "eta_ff must be > 0");
        check(eta_fb > 0, "eta_fb must be > 0");
        check(eta_pert > 0, "eta_pert must be > 0");

        auto one_of = [](const std::string& val, std::initializer_list<const char*> opts, const char* name) {
            for (auto o : opts) if (val == o) return;
            std::fprintf(stderr, "CollisionalConfig error: unknown %s '%s'\n", name, val.c_str());
            std::exit(1);
        };
        one_of(sse_wind, {"HURLEY","HURLEY_LBV","VINK","VINK_NO_BISTAB"}, "SSE_WIND_MODEL");
        one_of(sse_remnant, {"HURLEY","BELCZYNSKI02","BELCZYNSKI08","FRYER_RAPID","FRYER_DELAYED","ELDRIDGE_TOUT"}, "SSE_REMNANT_MODEL");
        one_of(sse_ppisn, {"NONE","STRONG","MODERATE","WEAK"}, "SSE_PPISN_MODEL");
        one_of(sse_ecsn, {"OFF","ON"}, "SSE_ECSN_MODEL");

        r_ngb_sq = r_ngb * r_ngb;
        r_ngb_max_sq = r_ngb_max * r_ngb_max;
        // Speed of light: 299792.458 km/s in code velocity units (1 km/s)
        c_code = 299792.458;
    }
};

// Global collisional config (accessed from integrator, forces, etc.)
extern CollisionalConfig g_collisional;
