#pragma once

#include "particles.h"
#include <Kokkos_Core.hpp>
#include <vector>
#include <cstdio>

#ifdef USE_MPI
#include <mpi.h>

// MPI error checking macro — wraps any MPI call, aborts on failure
#define MPI_CHECK(call) do { \
    int _mpi_err = (call); \
    if (_mpi_err != MPI_SUCCESS) { \
        char _mpi_msg[MPI_MAX_ERROR_STRING]; int _mpi_len; \
        MPI_Error_string(_mpi_err, _mpi_msg, &_mpi_len); \
        std::fprintf(stderr, "MPI error at %s:%d: %s\n", __FILE__, __LINE__, _mpi_msg); \
        MPI_Abort(MPI_COMM_WORLD, _mpi_err); \
    } \
} while(0)
#endif

// =============================================================================
// GPU-aware MPI detection
// =============================================================================
// Set USE_GPU_AWARE_MPI=1 at compile time if your MPI supports direct GPU buffers
// - NVIDIA: CUDA-aware OpenMPI (--mca mpi_cuda_support 1)
// - AMD: ROCm-aware OpenMPI
//
// When enabled, MPI calls use device pointers directly (no host staging)
// When disabled, data is copied to host before MPI transfer (portable fallback)
// =============================================================================

#ifndef USE_GPU_AWARE_MPI
#define USE_GPU_AWARE_MPI 0
#endif

// Check if we're on a GPU backend
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
#define HAVE_GPU_BACKEND 1
#else
#define HAVE_GPU_BACKEND 0
#endif

// Effective GPU-aware MPI: only use if both compiled with support AND on GPU backend
#define EFFECTIVE_GPU_AWARE_MPI (USE_GPU_AWARE_MPI && HAVE_GPU_BACKEND)

// =============================================================================
// MPI Communication Module for Multi-GPU N-Body
//
// Architecture: Scatter-compute-gather paradigm (matching FROST paper)
//
// All ranks maintain a REPLICATED copy of ALL particles. The HHS recursion
// (partitioning, drifts, kicks) runs identically on all ranks, keeping them
// synchronized WITHOUT explicit communication.
//
// Only force computation is distributed: each rank computes forces for
// particles [my_start, my_start + my_count), then Allgather accelerations.
//
// This gives O(N) communication per force computation instead of O(N * 2^L)
// for ghost-exchange approaches.
// =============================================================================

// MPI wrapper that compiles with or without MPI
namespace MPIComm {

// Initialize MPI (no-op if MPI not enabled)
void init(int* argc, char*** argv);

// Finalize MPI
void finalize();

// Get rank and size
int rank();
int size();

// Barrier
void barrier();

// Check if MPI is active (compiled with MPI and size > 1)
bool is_distributed();

} // namespace MPIComm

// =============================================================================
// Distributed Particles Manager
// =============================================================================
struct DistributedParticles {
    // ALL particles, replicated on every rank
    Particles local;

    // This rank's responsibility for force computation
    // Rank r computes forces for particles [my_start, my_start + my_count)
    int my_start = 0;
    int my_count = 0;

    // Number of particles (same on all ranks in replicated mode)
    int N_local = 0;

    // Total particles (same as N_local in replicated mode)
    int N_total = 0;

    // Pre-allocated MPI communication buffers (packed [ax,ay,az] interleaved)
    // Avoids cudaMalloc/cudaFree per allgather call
    Kokkos::View<double*> mpi_acc_send_buf;  // size: 3 * max_per_rank (slot 0)
    Kokkos::View<double*> mpi_acc_recv_buf;  // size: 3 * N_total     (slot 0)
    Kokkos::View<double*> mpi_acc_send_buf2; // size: 3 * max_per_rank (slot 1, for concurrent iallgather)
    Kokkos::View<double*> mpi_acc_recv_buf2; // size: 3 * N_total     (slot 1, for concurrent iallgather)

#ifdef USE_MPI
    MPI_Request mpi_reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
#endif

    // Pre-allocated per-rank count/displacement arrays for MPI_Allgatherv
    // Avoids heap allocation on every allgather call
    std::vector<int> mpi_counts;        // size: n_ranks
    std::vector<int> mpi_displs;        // size: n_ranks
    std::vector<int> mpi_packed_counts; // size: n_ranks (3x for packed ax/ay/az)
    std::vector<int> mpi_packed_displs; // size: n_ranks

    // Per-slot copies for non-blocking iallgather (MPI reads these asynchronously,
    // so concurrent slots must not share the same arrays)
    std::vector<int> mpi_packed_counts_slot[2];
    std::vector<int> mpi_packed_displs_slot[2];

    // Persistent host-staged buffers (reused across calls, avoids per-call allocation)
    std::vector<double> host_send_buf;   // packed [ax,ay,az] for send
    std::vector<double> host_recv_buf;   // packed [ax,ay,az] for recv
    std::vector<double> host_recv_ax, host_recv_ay, host_recv_az;  // unpacked
    std::vector<double> host_send_dt, host_recv_dt;  // timestep staging

    // Initialize distributed particles from a global particle set
    // Only rank 0 should have the full particle set initially
    // After this, ALL ranks have ALL particles (replicated)
    void distribute_initial(Particles& global_particles);

    // Recompute N_total and this rank's force slice after local.N changed (merger or escaper removal;
    // every rank removes the same particles from its replica)
    void sync_count();

    // Hash the replicated state (N, positions, velocities, masses; ids and spins in collisional
    // mode) and compare across ranks. Returns false, after printing the step, if any rank differs.
    bool replicas_agree(int step) const;

    // Allgather accelerations after distributed force computation
    // Each rank has computed accelerations for [my_start, my_start + my_count)
    // After this call, all ranks have full acceleration array
    void allgather_accelerations();

    // Allgather accelerations for a SUBSET of particles [start, start+count)
    // Packs ax/ay/az into a single MPI message (3x fewer MPI calls)
    void allgather_accelerations_subset(int start, int count);

    // Non-blocking allgather: pack + MPI_Iallgatherv, returns immediately
    // slot=0 uses primary buffers, slot=1 uses secondary buffers
    void iallgather_accelerations_subset(int start, int count, int slot);

    // Wait for iallgather to complete + unpack recv buffer into ax/ay/az
    void wait_and_unpack_accelerations(int start, int count, int slot);

    // Allgather adaptive timesteps after distributed computation
    void allgather_timestep();

    // Allgather adaptive timesteps for a SUBSET of particles [start, start+count)
    void allgather_timestep_subset(int start, int count);

    // Gather particle data from all ranks to rank 0
    // In replicated mode, just returns local (all ranks have all data)
    void gather_to_root(Particles& global_particles);

    // Compute the slice of particles [start, start+count) that this rank
    // is responsible for in distributed force computation
    void get_force_slice(int start, int count, int& slice_start, int& slice_count) const;

    // =========================================================================
    // Allreduce for ring topology and tree-distributed force computation
    // Ring: partial forces from j-chunks → allreduce(SUM)
    // Tree: each rank processes a subset of tree groups → allreduce(SUM)
    // =========================================================================

    // Allreduce accelerations in-place for particles [start, start+count)
    // Packs ax/ay/az, calls MPI_Allreduce(MPI_IN_PLACE, MPI_SUM), unpacks
    void allreduce_accelerations_subset(int start, int count);

    // Allreduce timesteps in-place for particles [start, start+count)
    // Uses MPI_Allreduce(MPI_IN_PLACE, MPI_MIN) since timestep = min over j-interactions
    void allreduce_timestep_subset(int start, int count);
};

// =============================================================================
// MPI data types and tags
// =============================================================================
#ifdef USE_MPI

// Message tags
enum MPITags {
    TAG_PARTICLE_COUNT = 100,
    TAG_PARTICLE_DATA = 101,
};

#endif // USE_MPI
