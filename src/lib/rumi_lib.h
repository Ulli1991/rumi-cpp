#pragma once
// RUMI++ as a library: a host code hands RUMI++ a set of bodies and advances them over host steps.
//
// - The header is plain C++ (no Kokkos, no MPI): every array is host memory, the library stages it to the Kokkos
//   default memory space (OpenMP: the same memory; CUDA/HIP: device) with explicit deep copies.
// - Units are RUMI++'s: pc, Msun, km/s, time pc/(km/s). G is the host's, passed at every call in pc^3 Msun^-1 (km/s)^2.
// - The library owns the bodies' state between calls (no round trip through host units). The host changes it only
//   through set_particles (a new set), set_mass, kick and shift.
// - Slots: bodies are numbered 0..n-1 in the order of the last set_particles call; ids are the host's.
// - Ranks: one process, or the ranks of a communicator given to attach_comm (RUMI++'s distributed path: replicated
//   bodies, force slices per rank). Then every call below is collective over that communicator; host arrays are read
//   on its rank 0 only (the library broadcasts them) and results are valid on every rank.
// - Lifecycle: kokkos_initialize after MPI_Init, [attach_comm], configure, ..., kokkos_finalize before MPI_Finalize.

#include <cstddef>
#include <string>
#include <vector>

// Interface version: bumped on every change of this header's calls or their meaning (docs/LIBRARY.md). The library
// embeds "RUMI_LIB_ID api=<version> commit=<git describe>" (build_id); hosts check both at build and run time.
#define RUMI_LIB_API_VERSION 2

namespace rumi_lib {

struct Timers {                // seconds, cumulative since configure
    double advance = 0;        // hhs steps (kernels)
    double rebuild = 0;        // set_particles: staging, initial forces, subsystem detection
    double h2d = 0, d2h = 0;   // host <-> device staging of kicks, shifts, masses and results
    long long n_advance = 0, n_rebuild = 0, n_set_mass = 0;
};

// Energy ledger, RUMI++ units, cumulative since configure: E_bodies + E_books - W_kick - E_set is constant up to the
// integration error (E_books holds RUMI++'s own sinks: mergers, and E_sse = the energy mass changes removed)
struct Ledger {
    double W_kick = 0;         // work of the host's kicks: sum 1/2 m (|v + dv|^2 - |v|^2)
    double W_kick_abs = 0;     // the same with |.| per kick call
    double E_set = 0;          // energy jumps of set_particles (bodies added or removed)
};

// Kokkos lifecycle. num_threads <= 0 and device_id < 0: Kokkos defaults (OMP_NUM_THREADS, KOKKOS_DEVICE_ID).
// Leaves an already initialised Kokkos alone (and then does not finalise it). Returns 0 on success.
int kokkos_initialize(int num_threads, int device_id);
void kokkos_finalize();
const char* backend_name();

// Run on the ranks of an MPI communicator (Fortran handle, MPI_Comm_c2f) instead of one process: needs a build
// with USE_MPI. Before configure; the host keeps the communicator alive. Returns 0 on success.
int attach_comm(int fortran_comm);
int comm_rank();               // rank in that communicator (0 without one)

// RUMI++ config file (the standalone keys; TEST_CASE, IC and output keys are ignored). Returns 0 on success.
int configure(const char* config_file, int verbose);

// Replace the body set (n bodies; x, v as [3n]; kw = stellar type or nullptr for main sequence; radius_rsun or
// nullptr for 0). dt: the step the next advance takes (subsystem detection). Returns 0 on success.
int set_particles(int n, const long long* id, const double* m, const double* x, const double* v, const int* kw,
                  const double* radius_rsun, double G, double dt);

int n_slots();                 // bodies of the last set_particles call (merged ones included)
// The state by slot. Merged-away bodies: m = 0 and their last state. Returns the number still present.
int get_state(double* m, double* x, double* v);
// Velocity and position changes by slot (dv, dx as [3 n_slots]; nullptr: none); kicks book their work (Ledger)
void kick(const double* dv);
void shift(const double* dx);
// New masses (and radii, nullptr: unchanged) by slot, in place: no subsystem rebuild; the energy change is booked
// exactly in RUMI++'s E_sse (as its runtime stellar evolution does). Slots with m <= 0 or merged are skipped.
void set_mass(const double* m, const double* radius_rsun, double G);

// Advance all bodies by dt with gravitational constant G. Returns 0 on success.
int advance(double dt, double G);

// Kinetic + potential energy of the bodies and the energy booked as lost (mergers, mass changes); G as in advance
void energy(double G, double* E_bodies, double* E_books);
// The energy books one by one: escape, gw, merge, sse, pn, sd (SDAR), tidal (as E_books, cumulative)
void books(double out[7]);
const Ledger& ledger();
// |E| of FLYBY_CM_EPS (the tidal-sampling budget eps |E_ref|): E_REF if the config sets it, else the internal energy
// (centre-of-mass frame) of the set at the last set_particles
double e_ref();
// Mergers and collisions since the last call: the slot that keeps the product, the slot merged away, the
// product's mass. Returns the number of events.
int merger_events(std::vector<int>& kept_slot, std::vector<int>& gone_slot, std::vector<double>& m_product);
// Write the current state as a RUMI++ IC_FILE (m x y z vx vy vz kw R per line, %.17g): bit-exact replays
int dump_ic(const char* path);

// Restart: the whole library state (bodies, subsystems, books, ledger, slots) plus an opaque host blob, in RUMI++'s
// HDF5 restart format in directory dir (one file, the previous one replaced). A run continued from it is bit-identical
// to the uninterrupted one. Needs a build with HDF5. Return 0 on success; load fills host_blob.
int save_state(const char* dir, const std::vector<char>& host_blob);
int load_state(const char* dir, std::vector<char>& host_blob);

// "RUMI_LIB_ID api=<RUMI_LIB_API_VERSION> commit=<git describe of the RUMI++ sources>" of the linked library
const char* build_id();

const Timers& timers();
int n_mergers();               // bodies merged away since the last set_particles

}  // namespace rumi_lib
