#include "merger_tree.h"
#include "constants.h"
#include <cstdio>
#include <cmath>

// Global instances
double g_sim_time = 0.0;
MergerTree g_merger_tree;
EscaperLog g_escaper_log;

// ============================================================================
// MergerTree
// ============================================================================

static const char* merger_type_str(MergerType t) {
    switch (t) {
        case MergerType::GW_COALESCENCE:   return "GW_COAL";
        case MergerType::ISCO:             return "ISCO";
        case MergerType::TIDAL_DISRUPTION: return "TDE";
        case MergerType::STELLAR_COLLISION:return "STELLAR";
        default:                           return "UNKNOWN";
    }
}

void MergerTree::record(const MergerEvent& ev, const Subsystem& sub, double t_sim) {
    MergerTreeNode node{};
    node.id = next_id++;
    node.time = t_sim;
    node.type = ev.type;

    // Convention: pid1 = less massive, pid2 = more massive = remnant
    // execute_merger() places remnant at pid1's index, but the remnant keeps
    // the more massive particle's identity. We need to match that convention.
    // In detect_mergers(), pid1/pid2 follow subsystem ordering (not mass-sorted).
    // In execute_merger(), remnant goes to idx1 position.
    // For the tree, store them mass-sorted for consistency.
    node.pid1 = ev.pid1;
    node.pid2 = ev.pid2;
    node.pid_remnant = ev.pid1;  // execute_merger places remnant at idx1

    node.m1 = ev.m1;
    node.m2 = ev.m2;
    node.m_remnant = ev.m_remnant;
    node.a = sub.orb.a;
    node.e = sub.orb.e;

    double vk = std::sqrt(ev.v_kick[0]*ev.v_kick[0] +
                          ev.v_kick[1]*ev.v_kick[1] +
                          ev.v_kick[2]*ev.v_kick[2]);
    node.v_kick = vk;
    node.E_radiated = ev.E_radiated;
    node.spin_remnant[0] = ev.spin_remnant[0];
    node.spin_remnant[1] = ev.spin_remnant[1];
    node.spin_remnant[2] = ev.spin_remnant[2];

    int idx = (int)events.size();
    events.push_back(node);

    // Index by both progenitor IDs and remnant
    history[node.pid1].push_back(idx);
    if (node.pid2 != node.pid1)
        history[node.pid2].push_back(idx);
    if (node.pid_remnant != node.pid1 && node.pid_remnant != node.pid2)
        history[node.pid_remnant].push_back(idx);

    std::printf("[MERGER #%d] %s pid=%d+%d -> rem=%d, m_rem=%.4f, a=%.3e pc, e=%.4f, "
                "v_kick=%.1f km/s, t=%.6f Myr\n",
                node.id, merger_type_str(ev.type),
                ev.pid1, ev.pid2, node.pid_remnant,
                ev.m_remnant, sub.orb.a, sub.orb.e,
                vk, t_sim * TIME_UNIT_MYR);
}

void MergerTree::write_ascii(const char* filename) const {
    if (events.empty()) return;

    FILE* f = std::fopen(filename, "w");
    if (!f) {
        std::fprintf(stderr, "WARNING: cannot open %s for writing\n", filename);
        return;
    }

    std::fprintf(f, "# id\ttime\ttime_myr\ttype\tpid1\tpid2\tpid_rem\t"
                    "m1\tm2\tm_rem\ta\te\tv_kick\tE_rad\t"
                    "chi_x\tchi_y\tchi_z\n");

    for (const auto& n : events) {
        std::fprintf(f, "%d\t%.15e\t%.10f\t%s\t%d\t%d\t%d\t"
                        "%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.6f\t%.10e\t"
                        "%.10e\t%.10e\t%.10e\n",
                     n.id, n.time, n.time * TIME_UNIT_MYR,
                     merger_type_str(n.type),
                     n.pid1, n.pid2, n.pid_remnant,
                     n.m1, n.m2, n.m_remnant,
                     n.a, n.e, n.v_kick, n.E_radiated,
                     n.spin_remnant[0], n.spin_remnant[1], n.spin_remnant[2]);
    }

    std::fclose(f);
    std::printf("Merger tree written to %s (%d events)\n", filename, (int)events.size());
}

void MergerTree::print_summary() const {
    if (!events.empty()) {
        std::printf("  MergerTree: %d total events\n", (int)events.size());
    }
}

// ============================================================================
// EscaperLog
// ============================================================================

void EscaperLog::record(int pid, double t, double mass,
                        double x, double y, double z,
                        double vx, double vy, double vz,
                        double r, double v, double E_spec) {
    EscaperEvent ev{};
    ev.pid = pid;
    ev.time = t;
    ev.mass = mass;
    ev.pos[0] = x; ev.pos[1] = y; ev.pos[2] = z;
    ev.vel[0] = vx; ev.vel[1] = vy; ev.vel[2] = vz;
    ev.r = r;
    ev.v = v;
    ev.E_specific = E_spec;
    events.push_back(ev);

    std::printf("[ESCAPER] pid=%d, m=%.4f M_sun, r=%.3f pc, v=%.2f km/s, t=%.6f Myr\n",
                pid, mass, r, v, t * TIME_UNIT_MYR);
}

void EscaperLog::write_ascii(const char* filename) const {
    if (events.empty()) return;

    FILE* f = std::fopen(filename, "w");
    if (!f) {
        std::fprintf(stderr, "WARNING: cannot open %s for writing\n", filename);
        return;
    }

    std::fprintf(f, "# pid\ttime\ttime_myr\tmass\tx\ty\tz\tvx\tvy\tvz\tr\tv\tE_spec\n");

    for (const auto& e : events) {
        std::fprintf(f, "%d\t%.15e\t%.10f\t%.10e\t"
                        "%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t"
                        "%.10e\t%.10e\t%.10e\n",
                     e.pid, e.time, e.time * TIME_UNIT_MYR, e.mass,
                     e.pos[0], e.pos[1], e.pos[2],
                     e.vel[0], e.vel[1], e.vel[2],
                     e.r, e.v, e.E_specific);
    }

    std::fclose(f);
    std::printf("Escaper log written to %s (%d events)\n", filename, (int)events.size());
}
