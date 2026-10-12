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
    bool secular_triples = true;        // SECULAR_TRIPLES: stable triples with >= N_ORB_SECULAR inner orbits per step use the
                                        // double-averaged secular path; false sends them to LogH (FROST-CLUSTERS I dropped it for SDAR)
    double weak_perturber_gamma = 0.0;  // WEAK_PERTURBER_GAMMA (experimental, off): a non-secular triple whose third body perturbs the inner
                                        // binary below this tidal ratio is split (binary subsystem + field star); 0: off
    double tidal_gamma = 0.0;          // TIDAL_GAMMA: binaries with |tidal accel| / |internal accel| above this resolve
    double flyby_cm_eps = 0.0;         // FLYBY_CM_EPS (either fly-by velocity; 0: off): tidal-sampling pair step that bounds the
                                       //   energy a slow particle's kicks put into a subsystem per step by this x |E0|
    double e_ref = 0.0;                //   |E0| of the run
    double tidal_step_res = 1.0;       // TIDAL_STEP_RES: floor of the FLYBY_CM_EPS step, this x P (1 - e)^(3/2) of the binary (0: none)
    bool flyby_ngb_cm = true;
    double ngb_mass_gamma = 1e-3;      // NGB_MASS_GAMMA: tidal ratio above which a mass-scaled neighbour joins a binary
    double ngb_inv_mref = 0.0;         // NGB_MASS_SCALING: 1 / mean particle mass (pair neighbour radius r_ngb max(1, m/<m>)^(1/3)), 0: off          // FLYBY_NGB_CM: distant fly-by neighbour test of a binary member against the binary's c.o.m.
    double tidal_res = 0.05;           //   their orbit, dt <= TIDAL_RES x period (0 = off; dense cores)
    double eta_restart = 1e-6;         // energy error restart threshold

    // SDAR slow-down of weakly perturbed inner binaries in LogH subsystems (Wang, Nitadori & Makino 2020;
    // FROST-CLUSTERS I App. A2, which replaced plain LogH by it). Off: plain LogH (BIFROST 2023)
    bool sdar = false;                 // SDAR (env NBODY_SDAR=0/1 overrides)
    double sdar_kref = 1e-6;           // SDAR_KREF: k_ref of the slow-down factor (W20 Eq. 22, FC-I Eq. A8)
    double sdar_c = 0.1;               // SDAR_C: kappa_max coefficient c (W20 Eq. 28, FC-I Eq. A9)

    // SUBSYS_COLLISIONS: collisions decided inside the regularised subsystem integration (FROST-CLUSTERS I App. B3):
    // pairs closer than their collision distance merge there; candidates (osculating pericentre inside it) get a
    // step forced onto their pericentre and are decided there. Off: only the per-level check of 2-body subsystems
    bool subsys_collisions = true;
    double r_ngb_wide = 0.0;           // NGB_WIDE_FAC x r_ngb: bound pairs with a below it are neighbours (wide binaries; 0: off)
    double tde_accretion = 0.5;        // TDE_ACCRETION_FRACTION: fraction of a disrupted star the BH accretes (FROST-CLUSTERS I, DRAGON-II: 0.5)

    // Secular evolution
    int n_orb_secular = 3;             // orbits/timestep threshold for secular

    // KEPLER_2BODY: an isolated 2-body subsystem without PN takes the analytic Kepler drift (kepler_drift) at any
    // orbit count, bound or unbound, instead of LogH + GBS below N_ORB_SECULAR orbits per step. Exact: with BIFROST's
    // splitting of the Hamiltonian (Rantala et al. 2023, sec. 3.1) all external forces enter through the FSI kicks,
    // so the internal motion of a 2-body subsystem between kicks is a Kepler orbit; BIFROST sec. 3.10: without PN
    // the secular integrator reduces to a Kepler solver; NBODY6 advances unperturbed binaries analytically (Aarseth
    // 2003). Stays on LogH: PN pairs (a member >= M_PN with PN on), collision candidates whose pericentre falls in
    // the interval (FROST-CLUSTERS I App. B3, decided there), pairs with a merged-away (massless) member, and
    // every subsystem of 3 or more bodies
    bool kepler_2body = true;          // default since 2026-10-12 (12-27 % faster; old E_gw booking hid the 200k difference)
    bool kepler_2body_pair(double m1, double m2) const {
        return kepler_2body && !(pn_order > 0 && c_code > 0.0 && std::max(m1, m2) >= m_pn);
    }

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
