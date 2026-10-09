#pragma once

#include "particles.h"
#include <Kokkos_Core.hpp>
#include <vector>
#include <cstdint>

// =============================================================================
// CPU Serial Path for Small Subsets
// Host-side N² force computation used when particle count falls below
// serial_n2_cpu threshold in the HHS recursion.
// =============================================================================

// Host-side data for CPU serial computation
struct CpuSubset {
    std::vector<double> x, y, z, vx, vy, vz, ax, ay, az, mass, dt_adapt;
    std::vector<double> cvx, cvy, cvz;  // Kahan compensation of the kicks, as on the device
    std::vector<int64_t> ix, iy, iz;
    std::vector<int> sid;  // subsystem_id (collisional mode)
#ifdef ENABLE_COLLISIONAL
    bool collisional = false;
    std::vector<double> grad_mag;     // gradient magnitude (stale from previous step)
    std::vector<double> dt_level0;    // level-0 timestep for clamping
    std::vector<int> particle_id;     // unique particle IDs for neighbor pair output
#else
    static constexpr bool collisional = false;
#endif
    int count;
    int device_start = 0;  // offset into device Particles arrays
    std::vector<int> src;  // src[i]: device slot (relative to device_start) that host slot i came from
    bool order_dirty = false;

    void resize(int n);

    // Copy subset [start, start+count) from device to host
    void copy_from_device(Particles& p, int start);

    // Copy results back to device (positions, velocities, mass, dt_adapt, ix/iy/iz)
    void copy_to_device(Particles& p, int start);

    // Reorder the device slots to match the host order after partitions, so fields that only
    // live on the device (particle_id, stellar data, spin, kick compensation) follow their particle
    void sync_order_to_device(Particles& p);

    // Write host slots [host_start, host_start+n) of the kick compensation to device [dev_start, ...)
    void copy_compensation_to_device(Particles& p, int host_start, int dev_start, int n) const;
};

// Entry point: copy subset to host, run CPU HHS, copy back
int hhs_fsi_recursive_cpu(Particles& p, int start, int count,
                           double pivot_dt, double G, double soft2,
                           double dt_min, double eta, bool dt_valid,
                           int depth);
