#include "runtime_sse.h"
#include "stellar_cache.h"
#include "constants.h"
#include "diagnostics.h"
#include "collisional_config.h"
#include "mpi_comm.h"
#include "profiling.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>
#ifdef USE_MPI
#include <mpi.h>
#endif

// ============================================================================
// Natal kicks of new neutron stars and black holes: the "standard, fallback-controlled" kick of Banerjee et al. 2020
// (A&A 639, A41, arXiv 1902.07718, sec. 3.1, eq. 1; the updated SSE/BSE that Rantala et al. (2023, sec. 3.13) cite):
//   v_kick = v_NS (1 - f_fb),  v_NS from a Maxwellian of 1D dispersion sigma_NS = 265 km/s (Hobbs et al. 2005),
// isotropic. f_fb is the fallback fraction of the remnant prescription (SSEResult::fbfac): Fryer et al. 2012
// rapid/delayed, Belczynski et al. 2002/2008 (f_fb = (M_CO - 5)/2.6 between M_CO = 5 and 7.6 Msun, 1 above; the
// StarTrack kick v = v_NS (1 - f_fb) of Belczynski et al. 2008 is the same rule); direct collapse (f_fb = 1) gives
// no kick. HURLEY and ELDRIDGE_TOUT have no fallback (f_fb = 0): every NS and BH gets the full NS kick, as BSE
// (Hurley et al. 2002) does for every NS and BH. Electron-capture SN NSs: Maxwellian sigma = 3 km/s, no fallback
// scaling (Banerjee et al. 2020 sec. 3.1 and 3.2, the ECS engine is a full explosion; Gessner & Janka 2018).
// FROST-CLUSTERS I-III use SEVN's "unified" kick model (Giacobbo & Mapelli 2020; FC-I tab. B1) instead, which
// needs SEVN's ejecta masses.
// The draws are a deterministic hash of (particle id, PLUMMER_SEED, new stellar type): identical on every MPI rank
// and across restarts (a particle forms its remnant once, since remnants are not evolved further).
// ============================================================================
namespace {
constexpr double SIGMA_NS_KMS   = 265.0;   // Hobbs et al. 2005, MNRAS 360, 974
constexpr double SIGMA_ECSN_KMS = 3.0;     // Banerjee et al. 2020 sec. 3.2 (Gessner & Janka 2018, ApJ 865, 61)

std::uint64_t splitmix64(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Kick velocity (km/s = code units) of a remnant of type kw_new formed by particle pid
void natal_kick(int pid, int seed, int kw_new, double fbfac, bool ecsn, double dv[3]) {
    dv[0] = dv[1] = dv[2] = 0.0;
    const double scale = ecsn ? SIGMA_ECSN_KMS : SIGMA_NS_KMS * (1.0 - std::min(1.0, std::max(0.0, fbfac)));
    if (scale <= 0.0) return;   // direct collapse
    // FNV-1a of the inputs (as bh_bh_remnant in mergers.cpp), then splitmix64 for the uniforms
    std::uint64_t h = 1469598103934665603ULL;
    for (std::int64_t v : {(std::int64_t)pid, (std::int64_t)seed, (std::int64_t)kw_new, (std::int64_t)0x4b49434bLL})
        h = (h ^ (std::uint64_t)v) * 1099511628211ULL;
    double u[4];
    for (double& x : u) x = ((double)(splitmix64(h) >> 11) + 0.5) * 0x1.0p-53;   // in (0, 1)
    // Box-Muller: three independent N(0, 1) components; the vector is Maxwellian in magnitude and isotropic
    const double two_pi = 6.283185307179586;
    const double r1 = std::sqrt(-2.0 * std::log(u[0])), r2 = std::sqrt(-2.0 * std::log(u[2]));
    dv[0] = scale * r1 * std::cos(two_pi * u[1]);
    dv[1] = scale * r1 * std::sin(two_pi * u[1]);
    dv[2] = scale * r2 * std::cos(two_pi * u[3]);
}

// Potential-energy change of a mass update m -> m' that touches the stars T (d_i = m'_i - m_i, zero elsewhere):
//   U' - U = -G sum_{i<j} (m'_i m'_j - m_i m_j) / r_ij,   m'_i m'_j - m_i m_j = d_i mbar_j + mbar_i d_j  (exact),
// mbar = (m + m') / 2, hence
//   dU = -G sum_{i in T} d_i sum_{j != i} mbar_j / r_ij,
// one pass over T x N instead of two full triangles (N^2), and no cancellation of two O(|U|) sums. Each rank sums a
// contiguous slice of T on its device; the per-star terms are allreduced (one nonzero contribution per entry, so the
// reduction is exact) and added on the host in T order: every rank count gives the same bits. Unsoftened, as the
// energy books (E_sse) always were.
double potential_change(const Particles& p, const std::vector<int>& touched, const std::vector<double>& d,
                        const std::vector<double>& mbar_t, double G) {
    const int nT = (int)touched.size(), N = p.N;
    if (nT == 0) return 0.0;
    int k0 = 0, k1 = nT;
#ifdef USE_MPI
    const int n_ranks = MPIComm::size(), r = MPIComm::rank();
    if (n_ranks > 1) { k0 = (int)((long long)r * nT / n_ranks); k1 = (int)((long long)(r + 1) * nT / n_ranks); }
#endif
    std::vector<double> term(nT, 0.0);
    if (k1 > k0) {
        Kokkos::View<int*> idx("sse_du_idx", nT);
        Kokkos::View<double*> mbt("sse_du_mbt", nT), out("sse_du_out", k1 - k0);
        Kokkos::deep_copy(idx, Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(touched.data(), nT));
        Kokkos::deep_copy(mbt, Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(mbar_t.data(), nT));
        Kokkos::View<double*> mbar("sse_du_mbar", N);
        Kokkos::deep_copy(mbar, Kokkos::subview(p.mass, std::make_pair(0, N)));   // the new masses (views hold capacity >= N)
        Kokkos::parallel_for("sse_du_mbar_set", nT, KOKKOS_LAMBDA(int k) { mbar(idx(k)) = mbt(k); });
        auto px = p.x; auto py = p.y; auto pz = p.z;
        using team_policy = Kokkos::TeamPolicy<>;
        using member_type = team_policy::member_type;
        Kokkos::parallel_for("sse_du", team_policy(k1 - k0, Kokkos::AUTO),
            KOKKOS_LAMBDA(const member_type& team) {
                const int i = idx(k0 + team.league_rank());
                const double xi = px(i), yi = py(i), zi = pz(i);
                double phi = 0.0;
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, N), [&](int j, double& s) {
                    if (j == i) return;
                    const double dx = px(j) - xi, dy = py(j) - yi, dz = pz(j) - zi;
                    s += mbar(j) * fast_rsqrt(dx*dx + dy*dy + dz*dz);
                }, phi);
                Kokkos::single(Kokkos::PerTeam(team), [&]() { out(team.league_rank()) = phi; });
            });
        auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
        for (int k = k0; k < k1; k++) term[k] = -G * d[k] * h_out(k - k0);
    }
#ifdef USE_MPI
    if (n_ranks > 1) MPI_Allreduce(MPI_IN_PLACE, term.data(), nT, MPI_DOUBLE, MPI_SUM, MPIComm::comm());
#endif
    double dU = 0.0;
    for (double t : term) dU += t;
    return dU;
}
}  // namespace

void advance_stellar_evolution(Particles& p, SSEContext& sse, double dt_myr,
                                EnergyBookkeeping& books, double G) {
    if (dt_myr <= 0.0) return;

    int N = p.N;
    // NBODY_SSE_CHECK (debug): the booked potential change against the difference of two full triangle sums
    static const bool sse_check = std::getenv("NBODY_SSE_CHECK") != nullptr;
    const double PE_ref0 = sse_check ? compute_potential_energy_tiled(p, G, 0.0, 0, N) : 0.0;
    auto h_x     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
    auto h_y     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
    auto h_z     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);

    // Copy stellar fields to host
    auto h_m     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
    auto h_vx    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
    auto h_vy    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
    auto h_vz    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
    auto h_stype = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_type);
    auto h_mzams = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.m_zams);
    auto h_sage  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_age);
    auto h_srad  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_radius);
    auto h_lum   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.luminosity);
    auto h_cmass = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.core_mass);
    auto h_pid   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);

    // The SSE evaluations, in parallel on the host threads: SSEContext::evolve is a const, pure function of
    // (m_zams, age) (it re-integrates the track from age 0), so each star's result does not depend on the thread
    // count. Before 2026-10-18 this loop was serial on every rank (9.1 s per call at N = 1e6, 9 us per star) and, with
    // the two replicated full potential-energy sums below (2 x 2.7 s on an A100), made the unprofiled "Other" 32 % of
    // an interval at N = 1e6 (fig7-2026-10-17 rerun; docs/CASES.md C13)
    std::vector<int> evo;
    evo.reserve(N);
    for (int i = 0; i < N; i++)
        if (h_stype(i) < 10) evo.push_back(i);   // remnants (WD=10-12, NS=13, BH=14, massless=15) are not evolved
    // With MPI (particles replicated) each rank evaluates a contiguous slice and the results are allgathered, so every
    // rank applies the same results (SSEResult is trivially copyable; byte counts fit an int up to ~3.8e7 stars)
    const int n_evo = (int)evo.size();
    int e0 = 0, e1 = n_evo;
#ifdef USE_MPI
    const int n_ranks = MPIComm::size(), my_rank = MPIComm::rank();
    auto slice = [&](int r) { return (int)((long long)r * n_evo / n_ranks); };
    if (n_ranks > 1) { e0 = slice(my_rank); e1 = slice(my_rank + 1); }
#endif
    std::vector<SSEResult> evo_res(n_evo);
    Kokkos::parallel_for("sse_evolve", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(e0, e1),
                         [&](int k) { const int i = evo[k]; evo_res[k] = sse.evolve(h_mzams(i), h_sage(i) + dt_myr); });
    Kokkos::DefaultHostExecutionSpace().fence();
#ifdef USE_MPI
    if (n_ranks > 1) {
        static_assert(std::is_trivially_copyable<SSEResult>::value, "SSEResult is sent as bytes");
        std::vector<int> cnt(n_ranks), dsp(n_ranks);
        for (int r = 0; r < n_ranks; r++) {
            dsp[r] = slice(r) * (int)sizeof(SSEResult);
            cnt[r] = (slice(r + 1) - slice(r)) * (int)sizeof(SSEResult);
        }
        g_profiler.mpi_start();
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, evo_res.data(), cnt.data(), dsp.data(), MPI_BYTE,
                       MPIComm::comm());
        g_profiler.mpi_stop();
    }
#endif

    bool changed = false;
    double total_dm = 0;
    int n_type_change = 0;
    // State before the update, for the exact energy, momentum and angular momentum sinks
    std::vector<int> touched;
    std::vector<double> m_before, v_before;

    for (size_t k = 0; k < evo.size(); k++) {
        const int i = evo[k];
        int kw = h_stype(i);
        double age_old = h_sage(i);
        double m_old = h_m(i);
        double age_new = age_old + dt_myr;
        const SSEResult& res = evo_res[k];

        // Mass loss
        double dm = m_old - res.mass;
        if (dm > 0) total_dm += dm;
        if (dm != 0.0 || res.kw != kw) {
            touched.push_back(i);
            m_before.push_back(m_old);
            v_before.insert(v_before.end(), {h_vx(i), h_vy(i), h_vz(i)});
        }

        // Update particle properties
        h_m(i) = res.mass;
        h_sage(i) = age_new;
        h_cmass(i) = res.mass_core;
        h_srad(i) = res.radius;       // collision radii follow the evolution
        h_lum(i) = res.luminosity;

        // Type change detection
        if (res.kw != kw) {
            n_type_change++;
            // Natal kick of a new NS or BH (see natal_kick above); the remnant's velocity only, the ejecta carry the
            // recoil (booked below in sink_PL and E_sse). Before 2026-10-12: a fixed 265 (NS) or 50 (BH) km/s along
            // the current velocity, scaled by 1 - m_rem / m_old
            if ((res.kw == 13 || res.kw == 14) && kw < 13 && g_collisional.sse_natal_kicks == "hobbs") {
                double dv[3];
                natal_kick(h_pid(i), g_collisional.kick_seed, res.kw, res.fbfac, res.ecsn, dv);
                h_vx(i) += dv[0];
                h_vy(i) += dv[1];
                h_vz(i) += dv[2];
                std::printf("SSE natal kick: pid %d kw %d -> %d m %.6g Msun f_fb %.6g ecsn %d dv %.9e %.9e %.9e km/s\n",
                            h_pid(i), kw, res.kw, res.mass, res.fbfac, (int)res.ecsn, dv[0], dv[1], dv[2]);
            }
        }

        h_stype(i) = res.kw;
        changed = true;
    }

    if (!changed) return;

    // Exact sinks: E_sse = E(before) - E(after) with the potential of every pair, so E_total + sinks is
    // conserved through winds and natal kicks; P and L carried off by the lost mass (and the kick recoil)
    double KE_change = 0.0;
    std::vector<double> dmass(touched.size()), mbar_t(touched.size());
    for (size_t k = 0; k < touched.size(); k++) {
        int i = touched[k];
        const double* vb = &v_before[3 * k];
        double r[3] = {h_x(i), h_y(i), h_z(i)}, va[3] = {h_vx(i), h_vy(i), h_vz(i)};
        KE_change += 0.5 * h_m(i) * (va[0]*va[0] + va[1]*va[1] + va[2]*va[2])
                   - 0.5 * m_before[k] * (vb[0]*vb[0] + vb[1]*vb[1] + vb[2]*vb[2]);
        books.sink_PL(m_before[k], r, vb);
        books.sink_PL(h_m(i), r, va, -1.0);
        dmass[k] = h_m(i) - m_before[k];
        mbar_t[k] = 0.5 * (h_m(i) + m_before[k]);
    }

    // Copy back to device
    Kokkos::deep_copy(p.mass, h_m);
    Kokkos::deep_copy(p.vx, h_vx);
    Kokkos::deep_copy(p.vy, h_vy);
    Kokkos::deep_copy(p.vz, h_vz);
    p.zero_kick_compensation();   // kicked velocities are new
    // Potential change from the touched stars only (potential_change above), on the devices of all ranks. Before
    // 2026-10-18: PE(before) - PE(after) from two full O(N^2) triangle sums, replicated on every rank and carrying
    // the round-off of two O(|U|) sums
    if (!touched.empty()) {
        const double dU = potential_change(p, touched, dmass, mbar_t, G);
        books.E_sse += -KE_change - dU;
        if (sse_check && MPIComm::rank() == 0) {
            const double full = compute_potential_energy_tiled(p, G, 0.0, 0, N) - PE_ref0;
            std::printf("SSE_CHECK touched %zu dU %.17g full %.17g rel %.3e\n", touched.size(), dU, full,
                        (dU - full) / std::fabs(PE_ref0));
        }
    }
    Kokkos::deep_copy(p.stellar_type, h_stype);
    Kokkos::deep_copy(p.stellar_age, h_sage);
    Kokkos::deep_copy(p.core_mass, h_cmass);
    Kokkos::deep_copy(p.stellar_radius, h_srad);
    Kokkos::deep_copy(p.luminosity, h_lum);
    StellarPropsCache::get().invalidate();   // types and radii changed

    if (total_dm > 0 || n_type_change > 0) {
        std::printf("SSE: dm_total=%.4e M_sun, %d type changes, %zu stars changed mass, E_sse=%.4e\n",
                    total_dm, n_type_change, touched.size(), books.E_sse);
    }
}
