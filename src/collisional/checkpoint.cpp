#include "checkpoint.h"
#include "stellar_cache.h"
#include "collisional_config.h"
#include "diagnostics.h"
#include "merger_tree.h"
#include "mpi_comm.h"
#include <cstdlib>
#include <cstdio>
#include <cmath>

// ============================================================================
// Save full state to checkpoint
// ============================================================================

void Checkpoint::save(const Particles& p, const SubsystemManager& mgr,
                      const EnergyBookkeeping& books) {
    N = p.N;

    // Resize host vectors
    x.resize(N); y.resize(N); z.resize(N);
    vx.resize(N); vy.resize(N); vz.resize(N);
    ax.resize(N); ay.resize(N); az.resize(N);
    mass.resize(N); dt_adapt.resize(N);
    level.resize(N); orig_index.resize(N);
    ix.resize(N); iy.resize(N); iz.resize(N);

    // Copy from device
    auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
    auto h_y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
    auto h_z = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
    auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
    auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
    auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
    auto h_ax = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.ax);
    auto h_ay = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.ay);
    auto h_az = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.az);
    auto h_m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
    auto h_dt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.dt_adapt);
    auto h_lev = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.level);
    auto h_orig = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.orig_index);
    auto h_ix = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.ix);
    auto h_iy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.iy);
    auto h_iz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.iz);

    for (int i = 0; i < N; i++) {
        x[i] = h_x(i); y[i] = h_y(i); z[i] = h_z(i);
        vx[i] = h_vx(i); vy[i] = h_vy(i); vz[i] = h_vz(i);
        ax[i] = h_ax(i); ay[i] = h_ay(i); az[i] = h_az(i);
        mass[i] = h_m(i); dt_adapt[i] = h_dt(i);
        level[i] = h_lev(i); orig_index[i] = h_orig(i);
        ix[i] = h_ix(i); iy[i] = h_iy(i); iz[i] = h_iz(i);
    }

    // Collisional fields
    if (p.collisional) {
        particle_id.resize(N); subsystem_id.resize(N); stellar_type.resize(N);
        m_zams.resize(N); stellar_age.resize(N); stellar_radius.resize(N);
        luminosity.resize(N); core_mass.resize(N);
        spin_vec_flat.resize(N * 3);
        spin_mag.resize(N); dt_level0.resize(N); grad_mag.resize(N);

        auto h_pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);
        auto h_sid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.subsystem_id);
        auto h_stype = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_type);
        auto h_mzams = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.m_zams);
        auto h_sage = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_age);
        auto h_srad = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_radius);
        auto h_lum = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.luminosity);
        auto h_cmass = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.core_mass);
        auto h_spin = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.spin_vec);
        auto h_smag = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.spin_mag);
        auto h_dtl0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.dt_level0);
        auto h_gmag = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.grad_mag);

        for (int i = 0; i < N; i++) {
            particle_id[i] = h_pid(i);
            subsystem_id[i] = h_sid(i);
            stellar_type[i] = h_stype(i);
            m_zams[i] = h_mzams(i);
            stellar_age[i] = h_sage(i);
            stellar_radius[i] = h_srad(i);
            luminosity[i] = h_lum(i);
            core_mass[i] = h_cmass(i);
            spin_vec_flat[i*3+0] = h_spin(i, 0);
            spin_vec_flat[i*3+1] = h_spin(i, 1);
            spin_vec_flat[i*3+2] = h_spin(i, 2);
            spin_mag[i] = h_smag(i);
            dt_level0[i] = h_dtl0(i);
            grad_mag[i] = h_gmag(i);
        }
    }

    // Subsystem state
    subsystems = mgr.active;
    next_subsystem_id = mgr.next_id;
    step_mergers = mgr.step_mergers;
    removed_pids = mgr.removed_pids;
    energy_books = books;
    merger_tree = g_merger_tree;

    valid = true;
}

// ============================================================================
// Restore state from checkpoint
// ============================================================================

void Checkpoint::restore(Particles& p, SubsystemManager& mgr,
                         EnergyBookkeeping& books) const {
    if (!valid) return;
    p.N = N;  // mergers during the rolled-back step shortened the arrays
    p.mark_layout_dirty(0, INT_MAX);
    p.removed_pid_log.clear();

    // Copy back to device
    auto h_x = Kokkos::create_mirror_view(p.x);
    auto h_y = Kokkos::create_mirror_view(p.y);
    auto h_z = Kokkos::create_mirror_view(p.z);
    auto h_vx = Kokkos::create_mirror_view(p.vx);
    auto h_vy = Kokkos::create_mirror_view(p.vy);
    auto h_vz = Kokkos::create_mirror_view(p.vz);
    auto h_ax = Kokkos::create_mirror_view(p.ax);
    auto h_ay = Kokkos::create_mirror_view(p.ay);
    auto h_az = Kokkos::create_mirror_view(p.az);
    auto h_m = Kokkos::create_mirror_view(p.mass);
    auto h_dt = Kokkos::create_mirror_view(p.dt_adapt);
    auto h_lev = Kokkos::create_mirror_view(p.level);
    auto h_orig = Kokkos::create_mirror_view(p.orig_index);
    auto h_ix = Kokkos::create_mirror_view(p.ix);
    auto h_iy = Kokkos::create_mirror_view(p.iy);
    auto h_iz = Kokkos::create_mirror_view(p.iz);

    for (int i = 0; i < N; i++) {
        h_x(i) = x[i]; h_y(i) = y[i]; h_z(i) = z[i];
        h_vx(i) = vx[i]; h_vy(i) = vy[i]; h_vz(i) = vz[i];
        h_ax(i) = ax[i]; h_ay(i) = ay[i]; h_az(i) = az[i];
        h_m(i) = mass[i]; h_dt(i) = dt_adapt[i];
        h_lev(i) = level[i]; h_orig(i) = orig_index[i];
        h_ix(i) = ix[i]; h_iy(i) = iy[i]; h_iz(i) = iz[i];
    }

    Kokkos::deep_copy(p.x, h_x);
    Kokkos::deep_copy(p.y, h_y);
    Kokkos::deep_copy(p.z, h_z);
    Kokkos::deep_copy(p.vx, h_vx);
    Kokkos::deep_copy(p.vy, h_vy);
    Kokkos::deep_copy(p.vz, h_vz);
    p.zero_kick_compensation();   // restored velocities; compensation is not checkpointed
    Kokkos::deep_copy(p.ax, h_ax);
    Kokkos::deep_copy(p.ay, h_ay);
    Kokkos::deep_copy(p.az, h_az);
    Kokkos::deep_copy(p.mass, h_m);
    Kokkos::deep_copy(p.dt_adapt, h_dt);
    Kokkos::deep_copy(p.level, h_lev);
    Kokkos::deep_copy(p.orig_index, h_orig);
    Kokkos::deep_copy(p.ix, h_ix);
    Kokkos::deep_copy(p.iy, h_iy);
    Kokkos::deep_copy(p.iz, h_iz);

    // Collisional fields
    if (p.collisional && !particle_id.empty()) {
        auto h_pid = Kokkos::create_mirror_view(p.particle_id);
        auto h_sid = Kokkos::create_mirror_view(p.subsystem_id);
        auto h_stype = Kokkos::create_mirror_view(p.stellar_type);
        auto h_mzams = Kokkos::create_mirror_view(p.m_zams);
        auto h_sage = Kokkos::create_mirror_view(p.stellar_age);
        auto h_srad = Kokkos::create_mirror_view(p.stellar_radius);
        auto h_lum = Kokkos::create_mirror_view(p.luminosity);
        auto h_cmass = Kokkos::create_mirror_view(p.core_mass);
        auto h_spin = Kokkos::create_mirror_view(p.spin_vec);
        auto h_smag = Kokkos::create_mirror_view(p.spin_mag);
        auto h_dtl0 = Kokkos::create_mirror_view(p.dt_level0);
        auto h_gmag = Kokkos::create_mirror_view(p.grad_mag);

        for (int i = 0; i < N; i++) {
            h_pid(i) = particle_id[i];
            h_sid(i) = subsystem_id[i];
            h_stype(i) = stellar_type[i];
            h_mzams(i) = m_zams[i];
            h_sage(i) = stellar_age[i];
            h_srad(i) = stellar_radius[i];
            h_lum(i) = luminosity[i];
            h_cmass(i) = core_mass[i];
            h_spin(i, 0) = spin_vec_flat[i*3+0];
            h_spin(i, 1) = spin_vec_flat[i*3+1];
            h_spin(i, 2) = spin_vec_flat[i*3+2];
            h_smag(i) = spin_mag[i];
            h_dtl0(i) = dt_level0[i];
            h_gmag(i) = grad_mag[i];
        }

        Kokkos::deep_copy(p.particle_id, h_pid);
        Kokkos::deep_copy(p.subsystem_id, h_sid);
        Kokkos::deep_copy(p.stellar_type, h_stype);
        Kokkos::deep_copy(p.m_zams, h_mzams);
        Kokkos::deep_copy(p.stellar_age, h_sage);
        Kokkos::deep_copy(p.stellar_radius, h_srad);
        Kokkos::deep_copy(p.luminosity, h_lum);
        Kokkos::deep_copy(p.core_mass, h_cmass);
        Kokkos::deep_copy(p.spin_vec, h_spin);
        StellarPropsCache::get().invalidate();   // types and radii of the saved state
        Kokkos::deep_copy(p.spin_mag, h_smag);
        Kokkos::deep_copy(p.dt_level0, h_dtl0);
        Kokkos::deep_copy(p.grad_mag, h_gmag);
    }

    mgr.active = subsystems;
    mgr.next_id = next_subsystem_id;
    mgr.step_mergers = step_mergers;
    mgr.removed_pids = removed_pids;
    mgr.ghost_pids.clear();   // checkpoints are taken between steps, where no placeholder is left
    if (p.collisional) mgr.rebuild_pid_to_idx(p);
    books = energy_books;
    g_merger_tree = merger_tree;
}

// ============================================================================
// Compute total energy E = KE + PE + sinks (BIFROST §3.18 bookkeeping)
// ============================================================================

double compute_total_energy(const Particles& p, const EnergyBookkeeping& books,
                            double G, double soft2) {
    return compute_particle_energy(p, G, soft2) + books.total_sink();
}

// KE + PE (PE split over the MPI ranks, compute_potential_energy_split)
double compute_particle_energy(const Particles& p, double G, double soft2, double* pe_out) {
    double KE = 0;
    auto pvx = p.vx; auto pvy = p.vy; auto pvz = p.vz;
    auto pm = p.mass;
    int N = p.N;

    Kokkos::parallel_reduce("chk_ke", N,
        KOKKOS_LAMBDA(int i, double& lKE) {
            double v2 = pvx(i)*pvx(i) + pvy(i)*pvy(i) + pvz(i)*pvz(i);
            lKE += 0.5 * pm(i) * v2;
        }, KE);

    double PE = compute_potential_energy_split(p, G, soft2);
    // NBODY_ECHECK_CMP (diagnostics): the split sum against the replicated whole-range sum of the previous code
    static const bool cmp = std::getenv("NBODY_ECHECK_CMP") != nullptr;
    if (cmp) {
        const double PE_full = compute_potential_energy_tiled(p, G, soft2, 0, N);
        if (MPIComm::rank() == 0)
            std::printf("ECHECK_CMP PE_split %.17g PE_full %.17g rel %.3e (KE+PE) rel %.3e\n", PE, PE_full,
                        (PE - PE_full) / std::fabs(PE_full), (PE - PE_full) / std::fabs(KE + PE_full));
    }
    if (pe_out) *pe_out = PE;
    return KE + PE;
}

// ============================================================================
// Compute energy error for adaptive restart check (BIFROST §3.19)
// Returns |E_now - E_ref| / |E_norm|
// ============================================================================

double compute_energy_error(const Particles& p, const EnergyBookkeeping& books,
                            double E_initial, double G, double soft2) {
    double E_now = compute_total_energy(p, books, G, soft2);
    return std::fabs((E_now - E_initial) / (std::fabs(E_initial) + 1e-30));
}
