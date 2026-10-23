#pragma once
// RUMI++ as a library: a host code hands RUMI++ a set of bodies and advances them over host steps.
//
// - The header is plain C++ (no Kokkos, no MPI): every array is host memory, the library stages it to the Kokkos
//   default memory space (OpenMP: the same memory; CUDA/HIP: device) with explicit deep copies.
// - Units are RUMI++'s: pc, Msun, km/s, time pc/(km/s). G is the host's, passed at every call in pc^3 Msun^-1 (km/s)^2.
// - The library owns the bodies' state between calls (no round trip through host units). The host changes it only
//   through set_particles (a new set), kick (velocity changes, e.g. external-field kicks) and shift.
// - Slots: bodies are numbered 0..n-1 in the order of the last set_particles call; ids are the host's.
// - One process runs the library; it calls no MPI (build with USE_MPI=OFF).
// - Lifecycle: kokkos_initialize after MPI_Init, configure, ..., kokkos_finalize before MPI_Finalize.

namespace rumi_lib {

struct Timers {                // seconds, cumulative since configure
    double advance = 0;        // hhs steps (kernels)
    double rebuild = 0;        // set_particles: staging, initial forces, subsystem detection
    double h2d = 0, d2h = 0;   // host <-> device staging of kicks, shifts and results
    long long n_advance = 0, n_rebuild = 0;
};

// Kokkos lifecycle. num_threads <= 0 and device_id < 0: Kokkos defaults (OMP_NUM_THREADS, KOKKOS_DEVICE_ID).
// Leaves an already initialised Kokkos alone (and then does not finalise it). Returns 0 on success.
int kokkos_initialize(int num_threads, int device_id);
void kokkos_finalize();
const char* backend_name();

// RUMI++ config file (the standalone keys; TEST_CASE, IC and output keys are ignored). Returns 0 on success.
int configure(const char* config_file, int verbose);

// Replace the body set (n bodies; x, v as [3n]; kw = stellar type or nullptr for main sequence; radius_rsun or
// nullptr for 0). dt: the step the next advance takes (subsystem detection). Returns 0 on success.
int set_particles(int n, const long long* id, const double* m, const double* x, const double* v, const int* kw,
                  const double* radius_rsun, double G, double dt);

int n_slots();                 // bodies of the last set_particles call (merged ones included)
// The state by slot. Merged-away bodies: m = 0 and the merger product's x, v. Returns the number still present.
int get_state(double* m, double* x, double* v);
// Velocity and position changes by slot (dv, dx as [3 n_slots]; nullptr: none)
void kick(const double* dv);
void shift(const double* dx);

// Advance all bodies by dt with gravitational constant G. Returns 0 on success.
int advance(double dt, double G);

// Kinetic + potential energy of the bodies and the energy booked as lost (mergers, GW); G as in advance
void energy(double G, double* E_bodies, double* E_books);
// The energy books one by one: escape, gw, merge, sse, pn, sd (SDAR), tidal (as E_books, cumulative)
void books(double out[7]);
// Write the current state as a RUMI++ IC_FILE (m x y z vx vy vz kw R per line, %.17g): bit-exact replays
int dump_ic(const char* path);

const Timers& timers();
int n_mergers();               // bodies merged away since the last set_particles

}  // namespace rumi_lib
