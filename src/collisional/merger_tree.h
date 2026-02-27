#pragma once

#include "mergers.h"
#include "subsystem.h"
#include <vector>
#include <unordered_map>
#include <cstdio>

// Global simulation time (set by main loop, read by merger/escaper recording)
extern double g_sim_time;

// ============================================================================
// Merger tree: tracks assembly history of all mergers
// ============================================================================

struct MergerTreeNode {
    int id;                    // sequential merger ID
    int pid1, pid2;            // progenitor particle IDs (pid1 = less massive)
    int pid_remnant;           // remnant particle ID (= more massive progenitor)
    double time;               // simulation time of merger [code units]
    MergerType type;
    double m1, m2, m_remnant;  // masses [code units = M_sun]
    double a, e;               // orbital elements at merger
    double v_kick;             // kick speed [km/s]
    double E_radiated;         // energy dissipated [code units]
    double spin_remnant[3];    // remnant spin vector
};

struct MergerTree {
    std::vector<MergerTreeNode> events;
    int next_id = 0;

    // pid -> list of event indices involving that particle
    std::unordered_map<int, std::vector<int>> history;

    // Record a merger event. Called from integrator after execute_merger().
    // sub: the subsystem that triggered the merger (for orbital elements)
    // ev: the merger event (for masses, type, kick, spin)
    void record(const MergerEvent& ev, const Subsystem& sub, double t_sim);

    // Write all events to a tab-separated file (overwrite mode)
    void write_ascii(const char* filename) const;

    // One-line summary for step log (e.g. "MergerTree: 3 events")
    void print_summary() const;

    void clear() { events.clear(); history.clear(); next_id = 0; }
};

extern MergerTree g_merger_tree;

// ============================================================================
// Escaper log: records every particle removed as an escaper
// ============================================================================

struct EscaperEvent {
    int pid;                   // permanent particle ID
    double time;               // simulation time [code units]
    double mass;               // mass at removal [M_sun]
    double pos[3], vel[3];     // position [pc] and velocity [km/s] at removal
    double r;                  // distance from COM [pc]
    double v;                  // speed relative to COM [km/s]
    double E_specific;         // specific energy at escape [code units]
};

struct EscaperLog {
    std::vector<EscaperEvent> events;

    void record(int pid, double t, double mass,
                double x, double y, double z,
                double vx, double vy, double vz,
                double r, double v, double E_spec);

    void write_ascii(const char* filename) const;

    void clear() { events.clear(); }
};

extern EscaperLog g_escaper_log;

// Helper: find subsystem matching a merger event (by pid pair).
// Returns pointer to subsystem or nullptr if not found.
// Must be called BEFORE execute_merger() which removes the subsystem.
inline const Subsystem* find_subsystem_for_merger(const SubsystemManager& mgr,
                                                   const MergerEvent& ev) {
    for (const auto& sub : mgr.active) {
        if (sub.n_bodies != 2) continue;
        bool match = (sub.particle_ids[0] == ev.pid1 && sub.particle_ids[1] == ev.pid2) ||
                     (sub.particle_ids[0] == ev.pid2 && sub.particle_ids[1] == ev.pid1);
        if (match) return &sub;
    }
    return nullptr;
}
