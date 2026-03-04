#pragma once

#include "particles.h"
#include "subsystem.h"
#include "diagnostics.h"
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
    std::vector<double> spin_mag, dt_level0;

    // Subsystem manager state
    std::vector<Subsystem> subsystems;
    int next_subsystem_id;

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
double compute_total_energy(const Particles& p, const EnergyBookkeeping& books,
                            double G, double soft2);

// Compute relative energy error |dE/E| for adaptive restart check (BIFROST §3.19).
// Pure function — no side effects. Caller handles restart logic.
double compute_energy_error(const Particles& p, const EnergyBookkeeping& books,
                            double E_initial, double G, double soft2);
