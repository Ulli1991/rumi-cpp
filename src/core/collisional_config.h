#pragma once

#include <string>

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
    double eta_gbs = 1e-10;            // GBS accuracy for LogH
    double eta_pert = 0.02;            // perturbation timestep for subsystems
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
        r_ngb_sq = r_ngb * r_ngb;
        r_ngb_max_sq = r_ngb_max * r_ngb_max;
        // Speed of light: 299792.458 km/s in code velocity units (1 km/s)
        c_code = 299792.458;
    }
};

// Global collisional config (accessed from integrator, forces, etc.)
extern CollisionalConfig g_collisional;
