#pragma once

#include <cmath>

// Maximum bodies in a LogH subsystem (MSTAR handles larger)
constexpr int LOGH_MAX_BODIES = 8;

// LogH regularized state: time-transformed leapfrog (TTL)
// Follows BIFROST (Rantala+ 2023) Section 3.3 / FROST Section 2.3.
// Omega = T + B  where T=kinetic, B=-E₀ (positive for bound systems).
// BIFROST convention: ds/dt = Omega, all EOMs have 1/Omega factor.
// B is constant for Newtonian dynamics; evolves when PN forces are present
// (BIFROST Eq. 4): dB/ds = -Σ m_i v_i · g_i / Omega.
struct LogHState {
    double t;                              // physical time
    double B;                              // binding energy
    int n_bodies;
    double pos[LOGH_MAX_BODIES * 3];
    double vel[LOGH_MAX_BODIES * 3];
    double mass[LOGH_MAX_BODIES];
    double ext_accel[LOGH_MAX_BODIES * 3]; // external perturbation

    // Auxiliary velocities for velocity-dependent forces (PN)
    double aux_vel[LOGH_MAX_BODIES * 3];
    bool use_aux_vel;                       // whether PN is active

    // Spin vectors for BH binaries (dimensionless chi vectors; S = chi G m^2 / c) and their auxiliary
    // copies Z (Rantala+ 2017): S precesses with Omega(w, Z), Z with Omega(v, S)
    double spin1[3], spin2[3];
    double aux_spin1[3], aux_spin2[3];
    bool has_spins;                          // true for 2-body BH binaries with spins
    double c_code;                           // speed of light (normalized units)

    // SDAR slow-down (Mikkola & Aarseth 1996; Wang, Nitadori & Makino 2020, MNRAS 493, 3398, hereafter W20;
    // FROST-CLUSTERS I, Rantala et al. 2024, App. A2, hereafter FC-I): n_sd disjoint inner binaries
    // (sd_a[k], sd_b[k]) whose internal motion runs 1/kappa_k times slower, W20 Eq. 19 / FC-I Eq. A3:
    //   H_sd = sum_k H_b,k / kappa_k + (H - sum_k H_b,k).
    // kappa is constant within a GBS step and jumps only between steps (W20 sec. 2.4); n_sd = 0 is plain LogH.
    int n_sd;
    int sd_a[LOGH_MAX_BODIES / 2], sd_b[LOGH_MAX_BODIES / 2];
    double sd_kappa[LOGH_MAX_BODIES / 2];
    double sd_jump;   // sum of the jumps delta H_sd (W20 Eq. 24) booked into B over this integration
    double sd_kappa_peak;   // largest kappa set during this integration (statistics)
};

// SDAR parameters of one subsystem integration (not part of the integrated state)
struct SdarParams {
    double k_ref = 1e-6;                    // W20 Eq. 22, FC-I Eq. A8
    double c = 0.1;                         // kappa_max coefficient, W20 Eq. 28, FC-I Eq. A9
    double coll_factor = 1.0;               // collision candidate: osculating pericentre < coll_factor x collision
                                            // distance -> kappa = 1 (FC-I App. B3: r_p < r_coll)
    double coll_radius[LOGH_MAX_BODIES] = {};   // collision radius per body (normalized units, 0: none)
    const double* coll_pair = nullptr;          // pair collision distances [i * LOGH_MAX_BODIES + j] (normalised;
                                                // replaces the per-body sum when set: TDE radii are not additive)
    double m_pn = 0.0;                      // PN subsystem: pairs with a member of at least this mass keep kappa = 1
                                            // (normalised; 0: no PN, every pair may be slowed)
    double steps_per_orbit = 0.0;           // GBS step limit per slowed orbit (0: only the caller's max_iter)
};

// SDAR: slowed kinetic energy T_sd (FC-I Eq. A4; W20 Eq. 50) and potential U_sd; plain T, U when n_sd = 0
double logh_sd_kinetic_energy(const LogHState& state);
double logh_sd_potential_energy(const LogHState& state, double G);
// Omega = T_sd + B (the time transformation dt/ds = 1/Omega, FC-I Eq. A5)
double logh_omega(const LogHState& state);
// Jumping kappa (W20 sec. 2.4): re-identify the inner binaries, set kappa from W20 Eqs. 22 and 28, and apply
// the instantaneous correction delta H_sd = H_sd(new) - H_sd(old) at the current phase-space point (W20 Eq. 24)
// to B (B = -H_sd on shell) and to state.sd_jump. clear = true removes every slow-down (end of an integration).
// Returns the shortest orbital period over bound pairs, kappa P sqrt(1 - e) for slowed ones (GBS step limit;
// 1e300: no bound pair).
double logh_sd_update(LogHState& state, double G, const SdarParams& par, bool clear = false);
// Smallest dynamical time sqrt(r^3 / (G M)) over pairs, x kappa for a slowed pair (its orbit takes kappa P)
double logh_sd_dyn_time(const LogHState& state, double G, double M_total);

// Compute internal accelerations for LogH system
void logh_internal_accel(const LogHState& state, double G,
                         double accel[LOGH_MAX_BODIES * 3]);

// Compute kinetic energy T of the system
double logh_kinetic_energy(const LogHState& state);

// Compute potential energy W of the system
double logh_potential_energy(const LogHState& state, double G);

// Single leapfrog step in regularized time s
void logh_leapfrog_step(LogHState& state, double ds, double G);

// Single leapfrog step with PN corrections using auxiliary velocity technique
// (Hellstrom & Mikkola 2010): v gets g(w), w gets g(v)
void logh_leapfrog_step_pn(LogHState& state, double ds, double G,
                            double c_code, int pn_order);

// Values a chain of leapfrog steps needs at its start, which every GBS stage of a step shares (all stages start
// from the same state): the Newtonian acceleration, Omega = T + B, and for PN steps g(v) = PN(x, v, S)
struct LogHStepStart {
    double accel[LOGH_MAX_BODIES * 3];
    double pn_v[LOGH_MAX_BODIES * 3];
    double omega;
};
void logh_step_start(const LogHState& state, double G, double c_code, int pn_order, LogHStepStart& start);

// nsteps logh_leapfrog_step steps with the acceleration and T + B carried between steps (bit-identical to single
// steps); start: the values at the first step (nullptr: computed here; unused for SDAR states, n_sd > 0)
void logh_leapfrog_steps(LogHState& state, double ds, double G, int nsteps, const LogHStepStart* start = nullptr);

// nsteps logh_leapfrog_step_pn steps with the values at the start of each step carried from the end of the
// previous one (bit-identical to single steps); start: the values at the first step (nullptr: computed here)
void logh_leapfrog_steps_pn(LogHState& state, double ds, double G, double c_code, int pn_order, int nsteps,
                            const LogHStepStart* start = nullptr);

// Compute total energy (kinetic + potential) for error checking
double logh_total_energy(const LogHState& state, double G);
