#include "mpi_comm.h"
#include "profiling.h"
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
    MPI_CHECK(MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &s_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &s_size));

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
    MPI_CHECK(MPI_Finalize());
#endif
}

int rank() { return s_rank; }
int size() { return s_size; }

void barrier() {
#ifdef USE_MPI
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
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
    MPI_CHECK(MPI_Bcast(&N_total, 1, MPI_INT, 0, MPI_COMM_WORLD));

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

    // Step 3: Broadcast collisional flag before allocating (rank 0 set it in init)
    int coll_flag = (my_rank == 0) ? (int)global_particles.collisional : 0;
    MPI_CHECK(MPI_Bcast(&coll_flag, 1, MPI_INT, 0, MPI_COMM_WORLD));

    // Step 4: Allocate local particles (full size on all ranks)
    local.allocate(N_total, (bool)coll_flag);

    // Step 4: Broadcast ALL particles from rank 0 to all ranks
    // Pack 11 double fields into 1 broadcast, 2 int fields into 1, 3 int64 fields into 1
    {
        constexpr int N_DBL_FIELDS = 11;  // x,y,z,vx,vy,vz,ax,ay,az,mass,dt
        constexpr int N_INT_FIELDS = 2;   // level, orig_index
        constexpr int N_I64_FIELDS = 3;   // ix, iy, iz

        std::vector<double> dbl_buf(N_DBL_FIELDS * (size_t)N_total);
        std::vector<int> int_buf(N_INT_FIELDS * (size_t)N_total);
        std::vector<int64_t> i64_buf(N_I64_FIELDS * (size_t)N_total);

        Particles& src = (my_rank == 0) ? global_particles : local;

        // Rank 0: copy from device to packed host buffers
        if (my_rank == 0) {
            auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.x);
            auto h_y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.y);
            auto h_z = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.z);
            auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.vx);
            auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.vy);
            auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.vz);
            auto h_ax = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.ax);
            auto h_ay = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.ay);
            auto h_az = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.az);
            auto h_mass = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.mass);
            auto h_dt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.dt_adapt);
            auto h_level = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.level);
            auto h_orig = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.orig_index);
            auto h_ix = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.ix);
            auto h_iy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.iy);
            auto h_iz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.iz);
            for (int i = 0; i < N_total; i++) {
                double* dp = &dbl_buf[i * N_DBL_FIELDS];
                dp[0]=h_x(i); dp[1]=h_y(i); dp[2]=h_z(i);
                dp[3]=h_vx(i); dp[4]=h_vy(i); dp[5]=h_vz(i);
                dp[6]=h_ax(i); dp[7]=h_ay(i); dp[8]=h_az(i);
                dp[9]=h_mass(i); dp[10]=h_dt(i);
                int_buf[i*N_INT_FIELDS+0] = h_level(i);
                int_buf[i*N_INT_FIELDS+1] = h_orig(i);
                i64_buf[i*N_I64_FIELDS+0] = h_ix(i);
                i64_buf[i*N_I64_FIELDS+1] = h_iy(i);
                i64_buf[i*N_I64_FIELDS+2] = h_iz(i);
            }
        }

        // 3 broadcasts instead of 16
        MPI_CHECK(MPI_Bcast(dbl_buf.data(), N_DBL_FIELDS * N_total, MPI_DOUBLE, 0, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Bcast(int_buf.data(), N_INT_FIELDS * N_total, MPI_INT, 0, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Bcast(i64_buf.data(), N_I64_FIELDS * N_total, MPI_INT64_T, 0, MPI_COMM_WORLD));

        // Unpack into per-field host mirrors and copy to device
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
        auto h_ix = Kokkos::create_mirror_view(local.ix);
        auto h_iy = Kokkos::create_mirror_view(local.iy);
        auto h_iz = Kokkos::create_mirror_view(local.iz);
        for (int i = 0; i < N_total; i++) {
            const double* dp = &dbl_buf[i * N_DBL_FIELDS];
            h_x(i)=dp[0]; h_y(i)=dp[1]; h_z(i)=dp[2];
            h_vx(i)=dp[3]; h_vy(i)=dp[4]; h_vz(i)=dp[5];
            h_ax(i)=dp[6]; h_ay(i)=dp[7]; h_az(i)=dp[8];
            h_mass(i)=dp[9]; h_dt(i)=dp[10];
            h_level(i) = int_buf[i*N_INT_FIELDS+0];
            h_orig(i) = int_buf[i*N_INT_FIELDS+1];
            h_ix(i) = i64_buf[i*N_I64_FIELDS+0];
            h_iy(i) = i64_buf[i*N_I64_FIELDS+1];
            h_iz(i) = i64_buf[i*N_I64_FIELDS+2];
        }
        Kokkos::deep_copy(local.x, h_x);
        Kokkos::deep_copy(local.y, h_y);
        Kokkos::deep_copy(local.z, h_z);
        Kokkos::deep_copy(local.vx, h_vx);
        Kokkos::deep_copy(local.vy, h_vy);
        Kokkos::deep_copy(local.vz, h_vz);
        local.zero_kick_compensation();   // received velocities carry no local compensation
        Kokkos::deep_copy(local.ax, h_ax);
        Kokkos::deep_copy(local.ay, h_ay);
        Kokkos::deep_copy(local.az, h_az);
        Kokkos::deep_copy(local.mass, h_mass);
        Kokkos::deep_copy(local.dt_adapt, h_dt);
        Kokkos::deep_copy(local.level, h_level);
        Kokkos::deep_copy(local.orig_index, h_orig);
        Kokkos::deep_copy(local.ix, h_ix);
        Kokkos::deep_copy(local.iy, h_iy);
        Kokkos::deep_copy(local.iz, h_iz);
    }

    // Broadcast collisional fields if enabled (packed: 6 doubles + 3 ints → 2 broadcasts)
    if (local.collisional) {
        constexpr int N_COLL_DBL = 6;  // mzams, sage, srad, lum, cmass, smag
        constexpr int N_COLL_INT = 3;  // pid, sid, stype

        std::vector<double> coll_dbl(N_COLL_DBL * (size_t)N_total);
        std::vector<int> coll_int(N_COLL_INT * (size_t)N_total);

        Particles& src = (my_rank == 0) ? global_particles : local;
        if (my_rank == 0) {
            auto h_pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.particle_id);
            auto h_sid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.subsystem_id);
            auto h_stype = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.stellar_type);
            auto h_mzams = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.m_zams);
            auto h_sage = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.stellar_age);
            auto h_srad = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.stellar_radius);
            auto h_lum = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.luminosity);
            auto h_cmass = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.core_mass);
            auto h_smag = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.spin_mag);
            for (int i = 0; i < N_total; i++) {
                double* dp = &coll_dbl[i * N_COLL_DBL];
                dp[0]=h_mzams(i); dp[1]=h_sage(i); dp[2]=h_srad(i);
                dp[3]=h_lum(i); dp[4]=h_cmass(i); dp[5]=h_smag(i);
                int* ip = &coll_int[i * N_COLL_INT];
                ip[0]=h_pid(i); ip[1]=h_sid(i); ip[2]=h_stype(i);
            }
        }

        MPI_CHECK(MPI_Bcast(coll_dbl.data(), N_COLL_DBL * N_total, MPI_DOUBLE, 0, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Bcast(coll_int.data(), N_COLL_INT * N_total, MPI_INT, 0, MPI_COMM_WORLD));

        auto h_pid = Kokkos::create_mirror_view(local.particle_id);
        auto h_sid = Kokkos::create_mirror_view(local.subsystem_id);
        auto h_stype = Kokkos::create_mirror_view(local.stellar_type);
        auto h_mzams = Kokkos::create_mirror_view(local.m_zams);
        auto h_sage = Kokkos::create_mirror_view(local.stellar_age);
        auto h_srad = Kokkos::create_mirror_view(local.stellar_radius);
        auto h_lum = Kokkos::create_mirror_view(local.luminosity);
        auto h_cmass = Kokkos::create_mirror_view(local.core_mass);
        auto h_smag = Kokkos::create_mirror_view(local.spin_mag);
        for (int i = 0; i < N_total; i++) {
            const double* dp = &coll_dbl[i * N_COLL_DBL];
            h_mzams(i)=dp[0]; h_sage(i)=dp[1]; h_srad(i)=dp[2];
            h_lum(i)=dp[3]; h_cmass(i)=dp[4]; h_smag(i)=dp[5];
            const int* ip = &coll_int[i * N_COLL_INT];
            h_pid(i)=ip[0]; h_sid(i)=ip[1]; h_stype(i)=ip[2];
        }
        Kokkos::deep_copy(local.particle_id, h_pid);
        Kokkos::deep_copy(local.subsystem_id, h_sid);
        Kokkos::deep_copy(local.stellar_type, h_stype);
        Kokkos::deep_copy(local.m_zams, h_mzams);
        Kokkos::deep_copy(local.stellar_age, h_sage);
        Kokkos::deep_copy(local.stellar_radius, h_srad);
        Kokkos::deep_copy(local.luminosity, h_lum);
        Kokkos::deep_copy(local.core_mass, h_cmass);
        Kokkos::deep_copy(local.spin_mag, h_smag);
    }

    // Pre-allocate MPI communication buffers to avoid cudaMalloc per allgather
    int max_per_rank = (N_total + n_ranks - 1) / n_ranks;
    mpi_acc_send_buf = Kokkos::View<double*>("mpi_acc_send", 3 * max_per_rank);
    mpi_acc_recv_buf = Kokkos::View<double*>("mpi_acc_recv", 3 * N_total);
    mpi_acc_send_buf2 = Kokkos::View<double*>("mpi_acc_send2", 3 * max_per_rank);
    mpi_acc_recv_buf2 = Kokkos::View<double*>("mpi_acc_recv2", 3 * N_total);

    // Pre-allocate count/displacement vectors to avoid heap allocs per allgather
    mpi_counts.resize(n_ranks);
    mpi_displs.resize(n_ranks);
    mpi_packed_counts.resize(n_ranks);
    mpi_packed_displs.resize(n_ranks);
    // Per-slot copies for concurrent non-blocking iallgathers
    for (int s = 0; s < 2; s++) {
        mpi_packed_counts_slot[s].resize(n_ranks);
        mpi_packed_displs_slot[s].resize(n_ranks);
    }

    if (my_rank == 0) {
        std::printf("Replicated particles: %d total, rank 0 computes forces for [%d, %d)\n",
                    N_total, my_start, my_start + my_count);
        std::printf("  MPI topology: i-partitioned (allgatherv)\n");
#if EFFECTIVE_GPU_AWARE_MPI
        std::printf("  GPU-aware MPI: enabled (packed allgathers)\n");
#else
        std::printf("  GPU-aware MPI: disabled (host-staged)\n");
#endif
    }

    // Pre-allocate persistent host-staged buffers
    host_send_buf.resize(3 * (max_per_rank + 1));
    host_recv_buf.resize(3 * N_total);
    host_recv_ax.resize(N_total);
    host_recv_ay.resize(N_total);
    host_recv_az.resize(N_total);
    host_send_dt.resize(max_per_rank + 1);
    host_recv_dt.resize(N_total);

    // MPI_Barrier removed: MPI_Bcast is already synchronizing
#else
    // Non-MPI: just use global particles directly
    local = global_particles;
    N_local = local.N;
    N_total = N_local;
    my_start = 0;
    my_count = N_local;
#endif
}

void DistributedParticles::sync_count() {
    N_total = local.N;
    N_local = N_total;
#ifdef USE_MPI
    int n_ranks = MPIComm::size();
    int my_rank = MPIComm::rank();
    int base_count = N_total / n_ranks;
    int remainder = N_total % n_ranks;
    my_start = 0;
    for (int r = 0; r < my_rank; r++) my_start += base_count + (r < remainder ? 1 : 0);
    my_count = base_count + (my_rank < remainder ? 1 : 0);
#else
    my_start = 0;
    my_count = N_total;
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

    g_profiler.mpi_start();
    MPI_CHECK(MPI_Allgatherv(send.data(), 3 * my_slice_count, MPI_DOUBLE,
                   recv.data(), mpi_packed_counts.data(), mpi_packed_displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD));
    g_profiler.mpi_stop();

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

    // Pack interleaved (reuse persistent buffers)
    for (int i = 0; i < my_slice_count; i++) {
        host_send_buf[3*i]   = h_ax(i);
        host_send_buf[3*i+1] = h_ay(i);
        host_send_buf[3*i+2] = h_az(i);
    }

    g_profiler.mpi_start();
    MPI_CHECK(MPI_Allgatherv(host_send_buf.data(), 3 * my_slice_count, MPI_DOUBLE,
                   host_recv_buf.data(), mpi_packed_counts.data(), mpi_packed_displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD));
    g_profiler.mpi_stop();

    // Unpack and copy only the subset back to device
    for (int i = 0; i < count; i++) {
        host_recv_ax[i] = host_recv_buf[3*i];
        host_recv_ay[i] = host_recv_buf[3*i+1];
        host_recv_az[i] = host_recv_buf[3*i+2];
    }

    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h_recv_ax(host_recv_ax.data(), count);
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h_recv_ay(host_recv_ay.data(), count);
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h_recv_az(host_recv_az.data(), count);

    Kokkos::deep_copy(Kokkos::subview(local.ax, Kokkos::make_pair(start, start + count)), h_recv_ax);
    Kokkos::deep_copy(Kokkos::subview(local.ay, Kokkos::make_pair(start, start + count)), h_recv_ay);
    Kokkos::deep_copy(Kokkos::subview(local.az, Kokkos::make_pair(start, start + count)), h_recv_az);
#endif

#endif // USE_MPI
}

void DistributedParticles::iallgather_accelerations_subset(int start, int count, int slot) {
#ifdef USE_MPI
    if (MPIComm::size() == 1 || count == 0) return;

    int n_ranks = MPIComm::size();
    int my_slice_start, my_slice_count;
    get_force_slice(start, count, my_slice_start, my_slice_count);

    // Compute per-rank packed counts and displacements (cheap, avoids storing per-slot)
    int base = count / n_ranks;
    int remainder = count % n_ranks;
    mpi_displs[0] = 0;
    for (int r = 0; r < n_ranks; r++) {
        mpi_counts[r] = base + (r < remainder ? 1 : 0);
        if (r > 0) mpi_displs[r] = mpi_displs[r-1] + mpi_counts[r-1];
    }
    for (int r = 0; r < n_ranks; r++) {
        mpi_packed_counts[r] = 3 * mpi_counts[r];
        mpi_packed_displs[r] = 3 * mpi_displs[r];
    }

    Kokkos::fence();

#if EFFECTIVE_GPU_AWARE_MPI
    // GPU-aware: pack on device, issue non-blocking MPI_Iallgatherv
    auto send = (slot == 0) ? mpi_acc_send_buf : mpi_acc_send_buf2;
    auto recv = (slot == 0) ? mpi_acc_recv_buf : mpi_acc_recv_buf2;
    auto lax = local.ax; auto lay = local.ay; auto laz = local.az;
    int ss = my_slice_start;

    Kokkos::parallel_for("ipack_acc", my_slice_count, KOKKOS_LAMBDA(int ii) {
        int i = ss + ii;
        send(3*ii)   = lax(i);
        send(3*ii+1) = lay(i);
        send(3*ii+2) = laz(i);
    });
    Kokkos::fence();

    // Copy counts/displs to per-slot arrays — MPI reads these asynchronously,
    // so concurrent slots must not share the same underlying memory
    for (int r = 0; r < n_ranks; r++) {
        mpi_packed_counts_slot[slot][r] = mpi_packed_counts[r];
        mpi_packed_displs_slot[slot][r] = mpi_packed_displs[r];
    }

    MPI_CHECK(MPI_Iallgatherv(send.data(), 3 * my_slice_count, MPI_DOUBLE,
                     recv.data(), mpi_packed_counts_slot[slot].data(),
                     mpi_packed_displs_slot[slot].data(), MPI_DOUBLE,
                     MPI_COMM_WORLD, &mpi_reqs[slot]));
    // Returns immediately — no wait, no unpack
#else
    // Host-staged: fall back to full blocking allgather (same as allgather_accelerations_subset)
    auto h_ax = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(local.ax, Kokkos::make_pair(my_slice_start, my_slice_start + my_slice_count)));
    auto h_ay = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(local.ay, Kokkos::make_pair(my_slice_start, my_slice_start + my_slice_count)));
    auto h_az = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(local.az, Kokkos::make_pair(my_slice_start, my_slice_start + my_slice_count)));

    // Reuse persistent buffers
    for (int i = 0; i < my_slice_count; i++) {
        host_send_buf[3*i]   = h_ax(i);
        host_send_buf[3*i+1] = h_ay(i);
        host_send_buf[3*i+2] = h_az(i);
    }

    MPI_CHECK(MPI_Allgatherv(host_send_buf.data(), 3 * my_slice_count, MPI_DOUBLE,
                   host_recv_buf.data(), mpi_packed_counts.data(), mpi_packed_displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD));

    for (int i = 0; i < count; i++) {
        host_recv_ax[i] = host_recv_buf[3*i];
        host_recv_ay[i] = host_recv_buf[3*i+1];
        host_recv_az[i] = host_recv_buf[3*i+2];
    }

    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h_recv_ax(host_recv_ax.data(), count);
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h_recv_ay(host_recv_ay.data(), count);
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h_recv_az(host_recv_az.data(), count);

    Kokkos::deep_copy(Kokkos::subview(local.ax, Kokkos::make_pair(start, start + count)), h_recv_ax);
    Kokkos::deep_copy(Kokkos::subview(local.ay, Kokkos::make_pair(start, start + count)), h_recv_ay);
    Kokkos::deep_copy(Kokkos::subview(local.az, Kokkos::make_pair(start, start + count)), h_recv_az);

    mpi_reqs[slot] = MPI_REQUEST_NULL;  // signal no wait needed
#endif

#endif // USE_MPI
}

void DistributedParticles::wait_and_unpack_accelerations(int start, int count, int slot) {
#ifdef USE_MPI
    if (MPIComm::size() == 1 || count == 0) return;

#if EFFECTIVE_GPU_AWARE_MPI
    MPI_CHECK(MPI_Wait(&mpi_reqs[slot], MPI_STATUS_IGNORE));

    auto recv = (slot == 0) ? mpi_acc_recv_buf : mpi_acc_recv_buf2;
    auto lax = local.ax; auto lay = local.ay; auto laz = local.az;
    int st = start;

    Kokkos::parallel_for("iunpack_acc", count, KOKKOS_LAMBDA(int ii) {
        int i = st + ii;
        lax(i) = recv(3*ii);
        lay(i) = recv(3*ii+1);
        laz(i) = recv(3*ii+2);
    });
    Kokkos::fence();
#else
    // Host-staged: blocking allgather already completed in initiate phase, no-op
    (void)start; (void)count; (void)slot;
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

    g_profiler.mpi_start();
    MPI_CHECK(MPI_Allgatherv(local.dt_adapt.data() + slice_start, slice_count, MPI_DOUBLE,
                   recv.data(), mpi_counts.data(), mpi_displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD));
    g_profiler.mpi_stop();

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

    // Reuse persistent buffers
    for (int i = 0; i < slice_count; i++) host_send_dt[i] = h_dt(i);

    g_profiler.mpi_start();
    MPI_CHECK(MPI_Allgatherv(host_send_dt.data(), slice_count, MPI_DOUBLE,
                   host_recv_dt.data(), mpi_counts.data(), mpi_displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD));
    g_profiler.mpi_stop();

    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h_recv_dt(host_recv_dt.data(), count);
    Kokkos::deep_copy(Kokkos::subview(local.dt_adapt, Kokkos::make_pair(start, start + count)), h_recv_dt);
#endif

#endif // USE_MPI
}

void DistributedParticles::gather_to_root(Particles& global_particles) {
    // In replicated mode, all ranks have all particles
    global_particles = local;
}

// Allreduce used by ring topology and tree-distributed force computation
void DistributedParticles::allreduce_accelerations_subset(int start, int count) {
#ifdef USE_MPI
    if (MPIComm::size() == 1 || count == 0) return;

    Kokkos::fence();

#if EFFECTIVE_GPU_AWARE_MPI
    // GPU-aware: pack ax/ay/az into buffer, allreduce in-place, unpack
    auto buf = mpi_acc_recv_buf;  // reuse recv buffer (size: 3 * N_total)
    auto lax = local.ax; auto lay = local.ay; auto laz = local.az;
    int st = start;

    Kokkos::parallel_for("pack_allreduce", count, KOKKOS_LAMBDA(int ii) {
        int i = st + ii;
        buf(3*ii)   = lax(i);
        buf(3*ii+1) = lay(i);
        buf(3*ii+2) = laz(i);
    });
    Kokkos::fence();

    g_profiler.mpi_start();
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, buf.data(), 3 * count, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD));
    g_profiler.mpi_stop();

    Kokkos::parallel_for("unpack_allreduce", count, KOKKOS_LAMBDA(int ii) {
        int i = st + ii;
        lax(i) = buf(3*ii);
        lay(i) = buf(3*ii+1);
        laz(i) = buf(3*ii+2);
    });
    Kokkos::fence();

#else
    // Host-staged: copy to host, allreduce, copy back
    std::vector<double> buf(3 * count);
    auto h_ax = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(local.ax, Kokkos::make_pair(start, start + count)));
    auto h_ay = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(local.ay, Kokkos::make_pair(start, start + count)));
    auto h_az = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(local.az, Kokkos::make_pair(start, start + count)));

    for (int i = 0; i < count; i++) {
        buf[3*i]   = h_ax(i);
        buf[3*i+1] = h_ay(i);
        buf[3*i+2] = h_az(i);
    }

    g_profiler.mpi_start();
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, buf.data(), 3 * count, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD));
    g_profiler.mpi_stop();

    for (int i = 0; i < count; i++) {
        h_ax(i) = buf[3*i];
        h_ay(i) = buf[3*i+1];
        h_az(i) = buf[3*i+2];
    }

    Kokkos::deep_copy(Kokkos::subview(local.ax, Kokkos::make_pair(start, start + count)), h_ax);
    Kokkos::deep_copy(Kokkos::subview(local.ay, Kokkos::make_pair(start, start + count)), h_ay);
    Kokkos::deep_copy(Kokkos::subview(local.az, Kokkos::make_pair(start, start + count)), h_az);
#endif

#endif // USE_MPI
}

void DistributedParticles::allreduce_timestep_subset(int start, int count) {
#ifdef USE_MPI
    if (MPIComm::size() == 1 || count == 0) return;

    Kokkos::fence();

#if EFFECTIVE_GPU_AWARE_MPI
    // GPU-aware: allreduce dt_adapt in-place via a contiguous buffer
    auto buf = Kokkos::subview(mpi_acc_recv_buf, Kokkos::make_pair(0, count));
    auto dt_f = local.dt_adapt;
    int st = start;

    // Copy to contiguous buffer
    Kokkos::parallel_for("pack_dt_allreduce", count, KOKKOS_LAMBDA(int ii) {
        buf(ii) = dt_f(st + ii);
    });
    Kokkos::fence();

    g_profiler.mpi_start();
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, buf.data(), count, MPI_DOUBLE,
                  MPI_MIN, MPI_COMM_WORLD));
    g_profiler.mpi_stop();

    Kokkos::parallel_for("unpack_dt_allreduce", count, KOKKOS_LAMBDA(int ii) {
        dt_f(st + ii) = buf(ii);
    });
    Kokkos::fence();

#else
    // Host-staged
    auto h_dt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(local.dt_adapt, Kokkos::make_pair(start, start + count)));

    std::vector<double> buf(count);
    for (int i = 0; i < count; i++) buf[i] = h_dt(i);

    g_profiler.mpi_start();
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, buf.data(), count, MPI_DOUBLE,
                  MPI_MIN, MPI_COMM_WORLD));
    g_profiler.mpi_stop();

    for (int i = 0; i < count; i++) h_dt(i) = buf[i];
    Kokkos::deep_copy(Kokkos::subview(local.dt_adapt, Kokkos::make_pair(start, start + count)), h_dt);
#endif

#endif // USE_MPI
}

// end allreduce functions
