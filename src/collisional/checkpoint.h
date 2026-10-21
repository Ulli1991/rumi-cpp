#pragma once

#include "particles.h"
#include "subsystem.h"
#include "diagnostics.h"
#include "merger_tree.h"
#include "gw_output.h"
#include <unordered_set>
#include <vector>

// Adaptive error restarts: save state and rollback on energy error

struct Checkpoint {
    // Particle state snapshot
    int N;
    std::vector<double> x, y, z;
    std::vector<double> vx, vy, vz;
    std::vector<double> ax, ay, az;
    std::vector<double> mass, dt_adapt;
    std::vector<int> level, orig_index;
    std::vector<int64_t> ix, iy, iz;

    // Collisional fields
    std::vector<int> particle_id, subsystem_id, stellar_type;
    std::vector<double> m_zams, stellar_age, stellar_radius, luminosity, core_mass;
    std::vector<double> spin_vec_flat;  // 3*N
    std::vector<double> spin_mag, dt_level0, grad_mag;

    // Subsystem manager state
    std::vector<Subsystem> subsystems;
    int next_subsystem_id;
    int step_mergers;
    std::unordered_set<int> removed_pids;

    // Merger history (a rolled-back step must not leave its mergers recorded)
    MergerTree merger_tree;
    std::vector<GwRow> gw_pending;   // rows of the gravitational-wave table not yet in the file

    // Energy bookkeeping
    EnergyBookkeeping energy_books;

    bool valid = false;

    // Save current state
    void save(const Particles& p, const SubsystemManager& mgr,
              const EnergyBookkeeping& books);

    // Restore saved state
    void restore(Particles& p, SubsystemManager& mgr,
                 EnergyBookkeeping& books) const;
};

// Compute total energy E = KE + PE + sinks (BIFROST §3.18 bookkeeping).
double compute_particle_energy(const Particles& p, double G, double soft2, double* pe_out = nullptr);   // KE + PE (PE also in *pe_out)
double compute_total_energy(const Particles& p, const EnergyBookkeeping& books,
                            double G, double soft2);

// Compute relative energy error |dE/E| for adaptive restart check (BIFROST §3.19).
// Pure function — no side effects. Caller handles restart logic.
double compute_energy_error(const Particles& p, const EnergyBookkeeping& books,
                            double E_initial, double G, double soft2);
