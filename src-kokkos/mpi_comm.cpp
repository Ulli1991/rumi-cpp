#include "mpi_comm.h"
#include <cstdio>
#include <algorithm>
#include <numeric>

// =============================================================================
// MPIComm namespace implementation
// =============================================================================

namespace MPIComm {

static int s_rank = 0;
static int s_size = 1;

void init(int* argc, char*** argv) {
#ifdef USE_MPI
    int provided;
    MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &s_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &s_size);

    if (s_rank == 0) {
        std::printf("MPI initialized: %d ranks\n", s_size);
    }
#else
    (void)argc; (void)argv;
    s_rank = 0;
    s_size = 1;
#endif
}

void finalize() {
#ifdef USE_MPI
    MPI_Finalize();
#endif
}

int rank() { return s_rank; }
int size() { return s_size; }

void barrier() {
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}

bool is_distributed() {
#ifdef USE_MPI
    return s_size > 1;
#else
    return false;
#endif
}

} // namespace MPIComm

// =============================================================================
// DistributedParticles implementation
// =============================================================================

void DistributedParticles::distribute_initial(Particles& global_particles) {
#ifdef USE_MPI
    int my_rank = MPIComm::rank();
    int n_ranks = MPIComm::size();

    if (n_ranks == 1) {
        // Single rank: just copy
        local = global_particles;
        N_local = local.N;
        N_total = N_local;
        my_start = 0;
        my_count = N_local;
        return;
    }

    // =========================================================================
    // Replicated particles approach (scatter-compute-gather)
    // All ranks get ALL particles. Each rank computes forces for a slice.
    // =========================================================================

    // Step 1: Rank 0 broadcasts total count
    if (my_rank == 0) {
        N_total = global_particles.N;
    }
    MPI_Bcast(&N_total, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Step 2: Compute force computation responsibility for each rank
    int base_count = N_total / n_ranks;
    int remainder = N_total % n_ranks;

    my_start = 0;
    for (int r = 0; r < my_rank; r++) {
        my_start += base_count + (r < remainder ? 1 : 0);
    }
    my_count = base_count + (my_rank < remainder ? 1 : 0);

    // N_local = N_total because all ranks have all particles
    N_local = N_total;

    // Step 3: Allocate local particles (full size on all ranks)
    local.allocate(N_total);

    // Step 4: Broadcast ALL particles from rank 0 to all ranks
    if (my_rank == 0) {
        // Copy to host for broadcast
        auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), global_particles.x);
        auto h_y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), global_particles.y);
        auto h_z = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), global_particles.z);
        auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), global_particles.vx);
        auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), global_particles.vy);
        auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), global_particles.vz);
        auto h_ax = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), global_particles.ax);
        auto h_ay = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), global_particles.ay);
        auto h_az = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), global_particles.az);
        auto h_mass = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), global_particles.mass);
        auto h_dt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), global_particles.dt_adapt);
        auto h_level = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), global_particles.level);
        auto h_orig = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), global_particles.orig_index);

        // Broadcast each array
        MPI_Bcast(h_x.data(), N_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_y.data(), N_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_z.data(), N_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_vx.data(), N_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_vy.data(), N_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_vz.data(), N_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_ax.data(), N_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_ay.data(), N_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_az.data(), N_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_mass.data(), N_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_dt.data(), N_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_level.data(), N_total, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_orig.data(), N_total, MPI_INT, 0, MPI_COMM_WORLD);

        // Copy to device
        Kokkos::deep_copy(local.x, h_x);
        Kokkos::deep_copy(local.y, h_y);
        Kokkos::deep_copy(local.z, h_z);
        Kokkos::deep_copy(local.vx, h_vx);
        Kokkos::deep_copy(local.vy, h_vy);
        Kokkos::deep_copy(local.vz, h_vz);
        Kokkos::deep_copy(local.ax, h_ax);
        Kokkos::deep_copy(local.ay, h_ay);
        Kokkos::deep_copy(local.az, h_az);
        Kokkos::deep_copy(local.mass, h_mass);
        Kokkos::deep_copy(local.dt_adapt, h_dt);
        Kokkos::deep_copy(local.level, h_level);
        Kokkos::deep_copy(local.orig_index, h_orig);
    } else {
        // Other ranks: allocate host buffers and receive broadcast
        auto h_x = Kokkos::create_mirror_view(local.x);
        auto h_y = Kokkos::create_mirror_view(local.y);
        auto h_z = Kokkos::create_mirror_view(local.z);
        auto h_vx = Kokkos::create_mirror_view(local.vx);
        auto h_vy = Kokkos::create_mirror_view(local.vy);
        auto h_vz = Kokkos::create_mirror_view(local.vz);
        auto h_ax = Kokkos::create_mirror_view(local.ax);
        auto h_ay = Kokkos::create_mirror_view(local.ay);
        auto h_az = Kokkos::create_mirror_view(local.az);
        auto h_mass = Kokkos::create_mirror_view(local.mass);
        auto h_dt = Kokkos::create_mirror_view(local.dt_adapt);
        auto h_level = Kokkos::create_mirror_view(local.level);
        auto h_orig = Kokkos::create_mirror_view(local.orig_index);

        // Receive broadcast
        MPI_Bcast(h_x.data(), N_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_y.data(), N_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_z.data(), N_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_vx.data(), N_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_vy.data(), N_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_vz.data(), N_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_ax.data(), N_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_ay.data(), N_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_az.data(), N_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_mass.data(), N_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_dt.data(), N_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_level.data(), N_total, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_orig.data(), N_total, MPI_INT, 0, MPI_COMM_WORLD);

        // Copy to device
        Kokkos::deep_copy(local.x, h_x);
        Kokkos::deep_copy(local.y, h_y);
        Kokkos::deep_copy(local.z, h_z);
        Kokkos::deep_copy(local.vx, h_vx);
        Kokkos::deep_copy(local.vy, h_vy);
        Kokkos::deep_copy(local.vz, h_vz);
        Kokkos::deep_copy(local.ax, h_ax);
        Kokkos::deep_copy(local.ay, h_ay);
        Kokkos::deep_copy(local.az, h_az);
        Kokkos::deep_copy(local.mass, h_mass);
        Kokkos::deep_copy(local.dt_adapt, h_dt);
        Kokkos::deep_copy(local.level, h_level);
        Kokkos::deep_copy(local.orig_index, h_orig);
    }

    // Pre-allocate MPI communication buffers to avoid cudaMalloc per allgather
    int max_per_rank = (N_total + n_ranks - 1) / n_ranks;
    mpi_acc_send_buf = Kokkos::View<double*>("mpi_acc_send", 3 * max_per_rank);
    mpi_acc_recv_buf = Kokkos::View<double*>("mpi_acc_recv", 3 * N_total);

    // Pre-allocate count/displacement vectors to avoid heap allocs per allgather
    mpi_counts.resize(n_ranks);
    mpi_displs.resize(n_ranks);
    mpi_packed_counts.resize(n_ranks);
    mpi_packed_displs.resize(n_ranks);

    if (my_rank == 0) {
        std::printf("Replicated particles: %d total, rank 0 computes forces for [%d, %d)\n",
                    N_total, my_start, my_start + my_count);
#if EFFECTIVE_GPU_AWARE_MPI
        std::printf("  GPU-aware MPI: enabled (packed allgathers)\n");
#else
        std::printf("  GPU-aware MPI: disabled (host-staged)\n");
#endif
    }

    MPI_Barrier(MPI_COMM_WORLD);
#else
    // Non-MPI: just use global particles directly
    local = global_particles;
    N_local = local.N;
    N_total = N_local;
    my_start = 0;
    my_count = N_local;
#endif
}

void DistributedParticles::get_force_slice(int start, int count, int& slice_start, int& slice_count) const {
#ifdef USE_MPI
    int n_ranks = MPIComm::size();
    int my_rank = MPIComm::rank();

    if (n_ranks == 1) {
        slice_start = start;
        slice_count = count;
        return;
    }

    // Distribute [start, start+count) evenly among ranks
    int base = count / n_ranks;
    int remainder = count % n_ranks;

    slice_start = start;
    for (int r = 0; r < my_rank; r++) {
        slice_start += base + (r < remainder ? 1 : 0);
    }
    slice_count = base + (my_rank < remainder ? 1 : 0);

    // Clamp to valid range
    if (slice_start >= start + count) {
        slice_start = start + count;
        slice_count = 0;
    } else if (slice_start + slice_count > start + count) {
        slice_count = start + count - slice_start;
    }
#else
    slice_start = start;
    slice_count = count;
#endif
}

void DistributedParticles::allgather_accelerations() {
    allgather_accelerations_subset(0, N_total);
}

void DistributedParticles::allgather_accelerations_subset(int start, int count) {
#ifdef USE_MPI
    if (MPIComm::size() == 1 || count == 0) return;

    int n_ranks = MPIComm::size();
    int my_slice_start, my_slice_count;
    get_force_slice(start, count, my_slice_start, my_slice_count);

    // Compute per-rank counts and displacements using pre-allocated vectors
    int base = count / n_ranks;
    int remainder = count % n_ranks;
    mpi_displs[0] = 0;
    for (int r = 0; r < n_ranks; r++) {
        mpi_counts[r] = base + (r < remainder ? 1 : 0);
        if (r > 0) mpi_displs[r] = mpi_displs[r-1] + mpi_counts[r-1];
    }

    // Packed: 3 doubles (ax,ay,az) per particle in single MPI message
    for (int r = 0; r < n_ranks; r++) {
        mpi_packed_counts[r] = 3 * mpi_counts[r];
        mpi_packed_displs[r] = 3 * mpi_displs[r];
    }

    Kokkos::fence();

#if EFFECTIVE_GPU_AWARE_MPI
    // GPU-aware: pack on device, single MPI call, unpack on device
    // Uses pre-allocated buffers (no cudaMalloc per call)
    auto send = mpi_acc_send_buf;
    auto recv = mpi_acc_recv_buf;
    auto lax = local.ax; auto lay = local.ay; auto laz = local.az;
    int ss = my_slice_start;

    Kokkos::parallel_for("pack_acc", my_slice_count, KOKKOS_LAMBDA(int ii) {
        int i = ss + ii;
        send(3*ii)   = lax(i);
        send(3*ii+1) = lay(i);
        send(3*ii+2) = laz(i);
    });
    Kokkos::fence();

    MPI_Allgatherv(send.data(), 3 * my_slice_count, MPI_DOUBLE,
                   recv.data(), mpi_packed_counts.data(), mpi_packed_displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    int st = start;
    Kokkos::parallel_for("unpack_acc", count, KOKKOS_LAMBDA(int ii) {
        int i = st + ii;
        lax(i) = recv(3*ii);
        lay(i) = recv(3*ii+1);
        laz(i) = recv(3*ii+2);
    });
    Kokkos::fence();

#else
    // Host-staged: copy only the slice (not full array), single packed MPI call
    auto h_ax = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(local.ax, Kokkos::make_pair(my_slice_start, my_slice_start + my_slice_count)));
    auto h_ay = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(local.ay, Kokkos::make_pair(my_slice_start, my_slice_start + my_slice_count)));
    auto h_az = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(local.az, Kokkos::make_pair(my_slice_start, my_slice_start + my_slice_count)));

    // Pack interleaved
    std::vector<double> send_buf(3 * my_slice_count);
    for (int i = 0; i < my_slice_count; i++) {
        send_buf[3*i]   = h_ax(i);
        send_buf[3*i+1] = h_ay(i);
        send_buf[3*i+2] = h_az(i);
    }

    std::vector<double> recv_buf(3 * count);
    MPI_Allgatherv(send_buf.data(), 3 * my_slice_count, MPI_DOUBLE,
                   recv_buf.data(), mpi_packed_counts.data(), mpi_packed_displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    // Unpack and copy only the subset back to device
    std::vector<double> recv_ax(count), recv_ay(count), recv_az(count);
    for (int i = 0; i < count; i++) {
        recv_ax[i] = recv_buf[3*i];
        recv_ay[i] = recv_buf[3*i+1];
        recv_az[i] = recv_buf[3*i+2];
    }

    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h_recv_ax(recv_ax.data(), count);
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h_recv_ay(recv_ay.data(), count);
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h_recv_az(recv_az.data(), count);

    Kokkos::deep_copy(Kokkos::subview(local.ax, Kokkos::make_pair(start, start + count)), h_recv_ax);
    Kokkos::deep_copy(Kokkos::subview(local.ay, Kokkos::make_pair(start, start + count)), h_recv_ay);
    Kokkos::deep_copy(Kokkos::subview(local.az, Kokkos::make_pair(start, start + count)), h_recv_az);
#endif

#endif // USE_MPI
}

void DistributedParticles::allgather_timestep() {
    allgather_timestep_subset(0, N_total);
}

void DistributedParticles::allgather_timestep_subset(int start, int count) {
#ifdef USE_MPI
    if (MPIComm::size() == 1 || count == 0) return;

    int n_ranks = MPIComm::size();
    int slice_start, slice_count;
    get_force_slice(start, count, slice_start, slice_count);

    // Compute per-rank counts and displacements using pre-allocated vectors
    int base = count / n_ranks;
    int remainder = count % n_ranks;
    mpi_displs[0] = 0;
    for (int r = 0; r < n_ranks; r++) {
        mpi_counts[r] = base + (r < remainder ? 1 : 0);
        if (r > 0) mpi_displs[r] = mpi_displs[r-1] + mpi_counts[r-1];
    }

    Kokkos::fence();

#if EFFECTIVE_GPU_AWARE_MPI
    // GPU-aware: use device pointers directly, reuse recv buffer
    auto recv = Kokkos::subview(mpi_acc_recv_buf, Kokkos::make_pair(0, count));

    MPI_Allgatherv(local.dt_adapt.data() + slice_start, slice_count, MPI_DOUBLE,
                   recv.data(), mpi_counts.data(), mpi_displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    auto dt_f = local.dt_adapt;
    int st = start;
    Kokkos::parallel_for("unpack_dt", count, KOKKOS_LAMBDA(int i) {
        dt_f(st + i) = recv(i);
    });
    Kokkos::fence();

#else
    // Host-staged: copy only the slice, not the full array
    auto h_dt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(local.dt_adapt, Kokkos::make_pair(slice_start, slice_start + slice_count)));

    std::vector<double> send_dt(slice_count);
    for (int i = 0; i < slice_count; i++) send_dt[i] = h_dt(i);

    std::vector<double> recv_dt(count);
    MPI_Allgatherv(send_dt.data(), slice_count, MPI_DOUBLE,
                   recv_dt.data(), mpi_counts.data(), mpi_displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h_recv_dt(recv_dt.data(), count);
    Kokkos::deep_copy(Kokkos::subview(local.dt_adapt, Kokkos::make_pair(start, start + count)), h_recv_dt);
#endif

#endif // USE_MPI
}

void DistributedParticles::gather_to_root(Particles& global_particles) {
    // In replicated mode, all ranks have all particles
    global_particles = local;
}
