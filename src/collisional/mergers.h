#pragma once

#include "particles.h"
#include "subsystem.h"
#include "diagnostics.h"

// Merger detection and execution for collisional dynamics

enum class MergerType {
    GW_COALESCENCE,      // Peters timescale < dt
    ISCO,                // BH pair r < 20 G M / c^2 (10 R_S); BH swallowing a star whole r < 6 G M / c^2
    TIDAL_DISRUPTION,    // r_peri < R_td
    STELLAR_COLLISION,   // r < R1 + R2
    NONE
};

struct MergerEvent {
    MergerType type;
    int pid1, pid2;          // particle IDs
    double m1, m2;           // progenitor masses
    double m_remnant;        // remnant mass
    double v_kick[3];        // GW kick velocity
    double spin_remnant[3];  // remnant spin vector
    double E_radiated;       // energy radiated (GW, dissipation)
};

// Check for merger conditions in all active subsystems
// Returns list of merger events to process
std::vector<MergerEvent> detect_mergers(const SubsystemManager& mgr,
                                         const Particles& p,
                                         double G, double c_code, double dt);

// Execute a merger: replace two particles with one remnant
// Updates Particles, SubsystemManager, and EnergyBookkeeping
void execute_merger(const MergerEvent& event, Particles& p,
                    SubsystemManager& mgr, EnergyBookkeeping& books,
                    double G, double c_code = 0.0);

// BH-BH merger remnant properties: quasi-circular fits scaled with the orbital angular momentum (head-on limit);
// L_orb: orbital angular momentum (direction), dx, dv: relative separation and velocity at the decision
void bh_bh_remnant(double m1, double m2, const double chi1[3], const double chi2[3],
                   const double L_orb[3], const double dx[3], const double dv[3], double& m_rem, double chi_rem[3],
                   double v_kick[3], double G, double c_code);

// Peters (1964) GW coalescence timescale
double gw_coalescence_time(double a, double e, double m1, double m2,
                           double G, double c_code);

// Tidal disruption radius 1.3 (M_bh / m_star)^{1/3} R_star (FROST-CLUSTERS I App. B3.3, after Kochanek 1992)
double tidal_disruption_radius(double m_star, double R_star, double M_bh);

// Collision distance r_coll of a pair (FROST-CLUSTERS I App. B3; lengths in pc, kw = stellar type) and the event
// it gives when the pair comes closer: two BHs (kw 14) type ISCO at 10 Schwarzschild radii, 20 G (m1 + m2) / c^2
// (the conservative option of Rantala et al. 2023, sec. 3.14.1; FC-I's larger single-BH ISCO 6 G max(m) / c^2 with BH_MERGE_RADIUS); a BH and any other body the tidal
// radius above, at least R_star (a BH lighter than ~0.46 m_star is inside the star before r_t), or the ISCO when
// that is larger (swallowed whole, type ISCO); two non-BHs R1 + R2. c <= 0: no ISCO.
double collision_distance(int kw1, double m1, double R1, int kw2, double m2, double R2, double G, double c,
                          MergerType& type);

// Remnant mass of a collision of the given type (m1, m2 with kinds kw1, kw2) before GW losses: a TDE leaves the BH
// plus TDE_ACCRETION_FRACTION of the star (default 0.5), everything else keeps m1 + m2
double collision_remnant_mass(MergerType type, int kw1, double m1, int kw2, double m2);

// A collision decided inside a regularised subsystem integration (logh_batch.cpp, FROST-CLUSTERS I App. B3). The
// integrator merges the pair at the collision point into one body of mass m1 + m2 at the pair's centre of mass
// with its velocity (momentum exact), books nothing itself and continues with the other members. dE_int is the
// change of the subsystem's internal energy T + U at that moment (exact; E_after - E_before), d / u the members'
// offsets and velocities relative to the pair's centre of mass there. pid_keep keeps the remnant, pid_gone stays
// as a massless placeholder ("ghost", co-moving member of the subsystem) until the next compaction of its range.
struct SubsystemCollision {
    MergerType type = MergerType::NONE;
    int pid_keep = -1, pid_gone = -1;
    int kw_keep = 0, kw_gone = 0;
    double m_keep = 0, m_gone = 0;
    double dE_int = 0;                       // internal energy change at the collision (physical)
    double d_keep[3] = {}, d_gone[3] = {};   // offsets from the pair c.o.m. at the collision (pc)
    double u_keep[3] = {}, u_gone[3] = {};   // velocities relative to the pair c.o.m. (km/s)
    double r_sep = 0, r_coll = 0;            // separation and collision distance at the decision (pc)
    double t_rel = 0;                        // time of the collision since the start of the call (code units)
    int n_members = 0;
    std::vector<int> member_pids;            // the subsystem's members at the call (n_members; internal to the books above)
    bool pl_booked = false;                  // P and L of the merge already booked by the caller (PN subsystems)
};

// Book and complete a subsystem collision after the integration (end of the subsystem advance): E_merge gets
// -dE_int plus the change of the potential with the bodies outside the subsystem (pair at its frozen offsets
// around the remnant's final position vs. the remnant), P / L sinks of the pair's orbital motion; then the remnant
// physics at the remnant's final state (GW mass loss, kick and spin of a BH merger, TDE mass loss), booked from
// the full particle set as execute_merger does; types, spins, m_zams; the ghost is registered for compaction and
// the merger tree records the event.
void finalize_subsystem_collision(const SubsystemCollision& c, Particles& p, SubsystemManager& mgr,
                                  EnergyBookkeeping& books, double G, double c_code);

// Escaper removal: remove particles beyond escape radius
// Returns number of particles removed
int remove_escapers(Particles& p, double G, double R_escape,
                    SubsystemManager& sm, EnergyBookkeeping& books);
