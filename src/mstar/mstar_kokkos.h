// Ketju-integrator (MSTAR)
// Copyright (C) 2020-2022 Antti Rantala, Matias Mannerkoski, and contributors.
// Licensed under the GPLv3 license.
// See LICENSE.md or <https://www.gnu.org/licenses/> for details.
// Modified by Ulrich Steinwandel, 2026: rewritten in C++ with Kokkos support (GPU).
// Copyright (C) 2026 Ulrich Steinwandel. Licensed under the GPLv3 license.
//
// MSTAR on Kokkos. The algorithm is KETJU's (Rantala et al. 2020, MNRAS 492, 4131; Mannerkoski et al. 2023,
// MNRAS 524, 4062): logarithmic-Hamiltonian leapfrog in minimum-spanning-tree relative coordinates, GBS
// extrapolation with adaptive order (Hairer, Norsett & Wanner II.9), PN 1-3.5 + spin terms (pn.c), BH mergers
// (bh_mergers.c, C++ port in mstar_mergers.cpp, on the host). Differences from the C library:
//  - all GBS rows (substep divisions) of a step run concurrently on the device, in lockstep: one launch for the
//    forces and one for kick+drift per substep stage, 2 k_max stages per step; small groups run every row inside
//    one team (one launch per step);
//  - vertex positions/velocities are rebuilt from the tree by per-vertex root-to-vertex path sums (the same
//    floating-point order as KETJU's edge loop, no level barriers);
//  - reductions (T, U, forces) are deterministic team reductions, so results agree with the C library to round-off
//    (not bitwise) and are bitwise reproducible run to run;
//  - one device->host copy per GBS step (errors, energies, time, BH positions);
//  - the MST is built on the host by dense Prim, O(N^2), giving KETJU's tree (Prim, float edge weights);
//  - not ported: star-star softening, no_star_star_gravity, the divide-and-conquer MST, MPI.

#pragma once

#include <Kokkos_Core.hpp>
#include <vector>

namespace rumi_mstar {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
using MemSpace = ExecSpace::memory_space;
template <class Exec> struct Engine;   // the integrator on one execution space (mstar_kokkos.cpp)
template <class T> using DView = Kokkos::View<T, Kokkos::LayoutRight, MemSpace>;

// PN flags: KETJU's enum ketju_PN_flags values
enum : unsigned {
    PN_1_0 = 1u << 0, PN_THREEBODY = 1u << 1, PN_2_0 = 1u << 2, PN_2_5 = 1u << 3, PN_3_0 = 1u << 4,
    PN_3_5 = 1u << 5, PN_1_5_SPIN = 1u << 8, PN_2_0_SPIN = 1u << 9, PN_SPIN_EVOL = 1u << 10,
    PN_SPIN_ALL = PN_1_5_SPIN | PN_2_0_SPIN | PN_SPIN_EVOL
};

// Extra forces (C++ port of KETJU's experimental extra-force branch, Mannerkoski 2022, b60d16e): forces besides
// the Newtonian and PN ones, evaluated on the device. Velocity-independent part (positions only; added to the Newtonian
// accelerations): a linear field a = A (x - x0) (e.g. a tidal tensor) and static point masses (Plummer-softened).
// Velocity-dependent part (through the auxiliary-velocity scheme exactly as PN; its work enters B like the PN work):
// a = D v. All constant during one run() call; off by default.
struct ExtraForces {
    static constexpr int MAX_PM = 4;
    bool vindep = false;
    double A[9] = {};            // row-major 3x3
    double x0[3] = {};
    int n_pm = 0;
    double pm_m[MAX_PM] = {};
    double pm_x[MAX_PM][3] = {};
    double pm_eps2 = 0;
    bool track_work = true;      // B follows their work (RUMI++ fix; false: KETJU's branch, B unchanged)
    bool vdep = false;
    double D[9] = {};            // row-major 3x3
};

// KETJU's ketju_integration_options (same defaults) + the natural constants; softening options not ported
struct Options {
    int gbs_kmax = 10;
    double gbs_relative_tolerance = 1e-9;
    double gbs_absolute_tolerance = 1e-100;
    double output_time_relative_tolerance = 1e-6;
    double output_time_absolute_tolerance = 1e-300;
    unsigned PN_flags = 0;
    int steps_between_mst_reconstruction = 10;
    int max_tree_distance = 2;
    long max_step_count = -1;          // > 0: run() stops (returns false) after this many attempted steps
    bool enable_bh_mergers = true;
    double bh_merger_distance_in_Rs = 12;
    bool enable_bh_merger_kicks = true;
    double G = 1.0, c = 1.0;
    int team_rows_max_n = 96;          // N <= this: every GBS row inside one team (one launch per step)
    bool host_engine = false;          // run on the host execution space (small groups in GPU builds)
    ExtraForces extra;                 // extra forces (off)
};

struct Perf {
    long successful_steps = 0, failed_steps = 0;
    double total_work = 0;              // KETJU's step_total_work sum
    double relative_energy_error = 0;   // (B_end - B_start) / B_start over the last run()
    double t_mst = 0, t_total = 0, t_sync = 0;   // wall seconds
    long launches = 0;                  // kernel launches issued
};

struct MergerInfo {
    double t_merger, m1, m2, m_remnant, chi1, chi2, chi_remnant, v_kick;
    int extra_index1, extra_index2;     // caller indices (extra_index) of the components
};

class MstarKokkos {
  public:
    MstarKokkos(int num_pn, int num_other);
    ~MstarKokkos();
    // Reuse for another group (device buffers are kept when the sizes match): resizes the host arrays, clears
    // perf and mergers, keeps opt
    void reset(int num_pn, int num_other);
    MstarKokkos(const MstarKokkos&) = delete;
    MstarKokkos& operator=(const MstarKokkos&) = delete;

    // Host state, KETJU's ketju_system_physical_state layout (PN particles first); the caller fills it before
    // run() and reads it after. Mergers shrink n / n_pn and shift the arrays like KETJU (extra_index follows).
    int n = 0, n_pn = 0;
    double time = 0;
    std::vector<double> mass, pos, vel, spin;   // [n], [3n], [3n], [3 n_pn]
    std::vector<int> extra_index;               // caller indices; merged-away particles move to the end
    Options opt;
    Perf perf;
    // Collision check after every accepted step (RUMI++ extension; rule: mstar_collision_distance in
    // src/collisional/mstar_dispatch.h): coll_R[i] radius, coll_bh[i] black hole; empty coll_R: off. A hit stops
    // run() at that step (coll_i < coll_j set, time = the step's end). Not maintained across KETJU BH mergers (the
    // check is switched off after one).
    std::vector<double> coll_R;
    std::vector<char> coll_bh;
    double coll_c = 0;
    bool coll_bh_max_single = false;
    int coll_i = -1, coll_j = -1, coll_type = 0;
    double coll_rsep = 0, coll_rcoll = 0;
    // Work ledger of the non-Newtonian forces (RUMI++ extension): sum over accepted steps of the extrapolated
    // change of B = U - T by the PN and extra-force kicks (B is carried as an extra GBS variable outside the error
    // control); the Newtonian T + U changed by -pn_dB through these forces, the rest of its change is integration
    // error. Zero without them.
    double pn_dB = 0;
    std::vector<MergerInfo> mergers;

    // ketju_run_integrator: advance by time_interval. Returns false if max_step_count was exceeded (state left at
    // the last accepted step).
    bool run(double time_interval);

  private:
    template <class Exec> bool run_on(Engine<Exec>& im, double time_interval);
    Engine<ExecSpace>* dev_ = nullptr;       // device engine (the default execution space)
    Engine<HostExecSpace>* host_ = nullptr;  // host engine (small groups; the same object as dev_ in host builds)
};

// Launch + fence cost of an empty kernel and of a minimal reduction read back to the host (microseconds),
// measured over reps calls: the per-substep latency floor of a device-resident GBS step
void measure_launch_latency(int reps, double& us_launch_fence, double& us_reduce_readback, double& us_async_launch);

}  // namespace rumi_mstar
