#include "n2_sinkset.h"
#include "small_for.h"
#include "lean_sync.h"
#include "integrator.h"
#include "hhs_counters.h"
#include "forces.h"
#include "timestep.h"
#include "constants.h"
#include "mpi_comm.h"
#include "profiling.h"
#include "tree.h"
#ifdef ENABLE_COLLISIONAL
#include "collisional_config.h"
#include "subsystem.h"
#include "logh_batch.h"
#include "post_newtonian.h"
#include "secular.h"
#include "secular_triple_step.h"
#include "mergers.h"
#include "merger_tree.h"
#include "checkpoint.h"
#include "diagnostics.h"
#include "kepler.h"
#include "stellar_cache.h"
#include "device_staging.h"
#include "stall.h"
#include "cm_tide.h"
#endif
#include <Kokkos_Core.hpp>
#include <vector>
#include <set>
#include <map>
#include <tuple>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <chrono>

// Global profiler instance
StepProfiler g_profiler;
SubsetHistogram g_subset_hist;

// HHS recursion debug counters
HHSCounters g_hhs_counters;

// Serial N² GPU threshold: skip MPI, compute full N² locally (no tree)
int serial_n2_gpu = 0;

// Serial N² CPU threshold: use CPU serial loops for full N²
int host_subtree_max = 64;

// Fused force+gradient: use stale accelerations to compute gradient in same N² pass as force

bool use_integer_positions = false;
double int_pos_scale = 0.0;
double int_pos_inv_scale = 0.0;

bool use_nonblocking_mpi = false;

// symmetric_timesteps removed — always enabled (hardcoded true)

// Barnes-Hut tree configuration
bool use_tree = false;
bool hhs_cc = false;
int hhs_cc_max = -1;   // -1: SERIAL_N2_GPU
int hhs_cc_min = 0;
int tree_threshold = 10000;
double tree_theta = 0.5;
bool tree_grad_dipole = true;
int tree_n_leaf = 32;
int tree_n_group = 64;
// Global tree instance (rebuilt each top-level step)
static Octree g_octree;

// Energy booked as E_gw by the secular path for binaries without PN terms (round-off of the Kepler drift; diagnostic)
double g_sec_nonpn_gw = 0.0;

void release_integrator_globals() { g_octree = Octree(); }  // before Kokkos::finalize

// =============================================================================
// Synthetic force errors (debug, plan I-1): what force accuracy would an FMM for the large HHS sets need?
// NBODY_FORCE_ERR=eps: before every field kick of an HHS set with at least NBODY_FORCE_ERR_MIN (default 16384)
// particles (the self kicks of fsi_step / fsi_step_distributed, both cross kicks and the gradient-corrected middle
// kicks), the complete accelerations of the kicked range (after the allgathers / allreduces, identically on every
// rank) are perturbed in place; the next force evaluation overwrites them. Subsystem-internal forces are not touched.
// NBODY_FORCE_ERR_MODEL:
//   random (default): delta_a_i = eps |a_i| u_i, u_i an isotropic unit vector from a hash of (particle id, force-call
//     counter): white noise in space and time, the error model of Hernquist, Hut & Makino (1993); breaks momentum
//     conservation at ~eps |a| m sqrt(N) per call, as a tree code without momentum correction does.
//   frozen: as random, but u_i from the particle id alone (the same direction at every call): an error that is
//     coherent in time, the worst case of an FMM/tree error that changes only as the configuration does (the
//     random-walk energy error of the white models becomes a drift; Dehnen 2014 App. A).
//   smooth (white in time, like random): delta_a_i = eps a_rms M (x_i - x_c) / r_rms, x_c the mass centre and r_rms the mass-weighted rms radius
//     of the range, a_rms the rms |a| over the range, M a 3x3 matrix of N(0, 1/3) entries drawn per force call (so the
//     rms |delta_a| is ~eps a_rms): a spatially smooth error field like the truncation error of a low-order multipole
//     expansion (Dehnen 2014 sec. 2-3: correlated, not white). sum_i m_i delta_a_i = 0 (momentum conserved, as
//     Dehnen's FMM); the antisymmetric part of M exerts a torque.
//   cell: an FMM-like error that is correlated in space and coherent in time: delta_a_i = eps |a_i| sum_l u_{c_l(i)} /
//     sqrt(L), over L = NBODY_FORCE_ERR_LEVELS (default 6) nested cubic grids of edge r0 2^-l (l = 0..L-1; r0 = the
//     mass-weighted rms radius of the first perturbed set, then fixed; grids anchored at the origin), u_c an isotropic
//     unit vector from a hash of (level, cell): all sinks in a cell share the error of that level (the error of one
//     M2L/L2P chain is the same smooth field for the sinks of a cell, Dehnen 2014 sec. 3), the field is frozen while
//     the configuration is (time-coherent), and it jumps when a particle crosses a cell face (Dehnen 2014 App. B3).
// NBODY_FORCE_ERR_PFIX = 1 (default for cell, 0 for random/frozen): subtract the mass-weighted mean of delta_a over
//   the kicked range (rank 0's mean, broadcast), so sum_i m_i delta_a_i = 0 and the momentum stays exact, as a
//   mutual (momentum-conserving) FMM keeps it (Dehnen 2000/2002 falcON; Zhu 2021 Taichi).
// NBODY_FORCE_ERR_LFIX = 1 (implies PFIX; random/frozen/cell): also remove the rigid-rotation part alpha x (x_i - x_c),
//   I alpha = sum_i m_i (x_i - x_c) x delta_a_i (I the inertia tensor of the range about its mass centre), so the
//   error exerts no net torque either: an FMM with a global torque correction (Marcello 2017 corrects pairwise).
// Off (unset or 0): no code runs, bit-identical.
// =============================================================================
namespace {
struct ForceErr {
    double eps = 0.0;
    int nmin = 16384;
    bool smooth = false;
    bool frozen = false;   // random directions fixed per particle for the whole run (time-correlated)
    bool cell = false;     // nested-grid cell errors, frozen in space (FMM-like)
    int levels = 6;        // cell model: number of nested grids
    int pfix = -1;         // subtract the mass-weighted mean (momentum exact); -1: model default
    int lfix = 0;          // also remove the net torque of the error (angular momentum exact)
};
const ForceErr& force_err_cfg() {
    static const ForceErr f = [] {
        ForceErr c;
        if (const char* v = std::getenv("NBODY_FORCE_ERR")) c.eps = std::atof(v);
        if (const char* v = std::getenv("NBODY_FORCE_ERR_MIN")) c.nmin = std::atoi(v);
        if (const char* v = std::getenv("NBODY_FORCE_ERR_MODEL")) {
            const std::string m(v);
            if (m == "smooth") c.smooth = true;
            else if (m == "frozen") c.frozen = true;
            else if (m == "cell") c.cell = true;
            else if (m != "random" && !m.empty()) {
                std::fprintf(stderr, "NBODY_FORCE_ERR_MODEL = '%s': random, frozen, smooth or cell\n", v);
                std::exit(1);
            }
        }
        if (const char* v = std::getenv("NBODY_FORCE_ERR_LEVELS")) c.levels = std::max(1, std::atoi(v));
        if (const char* v = std::getenv("NBODY_FORCE_ERR_PFIX")) c.pfix = std::atoi(v);
        if (const char* v = std::getenv("NBODY_FORCE_ERR_LFIX")) c.lfix = std::atoi(v);
        if (c.pfix < 0) c.pfix = c.cell ? 1 : 0;
        if (c.lfix) c.pfix = 1;
        if (c.eps > 0.0 && MPIComm::rank() == 0)
            std::printf("NBODY_FORCE_ERR: eps %.3e model %s for HHS sets >= %d (momentum fix %d, torque fix %d%s)\n", c.eps,
                        c.smooth ? "smooth" : c.frozen ? "frozen" : c.cell ? "cell" : "random", c.nmin, c.pfix, c.lfix,
                        c.cell ? (", levels " + std::to_string(c.levels)).c_str() : "");
        return c;
    }();
    return f;
}
std::uint64_t g_force_err_calls = 0;   // per-run force-call counter: identical on every rank (replicated control flow)

KOKKOS_INLINE_FUNCTION std::uint64_t ferr_mix(std::uint64_t z) {   // splitmix64 finaliser
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
KOKKOS_INLINE_FUNCTION double ferr_u01(std::uint64_t h) { return ((double)(h >> 11) + 0.5) * 0x1.0p-53; }

KOKKOS_INLINE_FUNCTION void ferr_unit(std::uint64_t h1, double& ux, double& uy, double& uz) {
    const std::uint64_t h2 = ferr_mix(h1);
    const double cz = 2.0 * ferr_u01(h1) - 1.0, phi = 6.283185307179586 * ferr_u01(h2);
    const double sz = Kokkos::sqrt(Kokkos::fmax(0.0, 1.0 - cz * cz));
    ux = sz * Kokkos::cos(phi); uy = sz * Kokkos::sin(phi); uz = cz;
}
double g_force_err_r0 = 0.0;   // cell model: coarsest grid edge (rms radius of the first perturbed set, then fixed)

// The perturbation of particle i for the random / frozen / cell models (a pure function of the state: evaluated
// twice when the mass-weighted mean is removed).
struct FerrField {
    Kokkos::View<double*> ax, ay, az, x, y, z;
    Kokkos::View<int*> pid;
    std::uint64_t key;
    double eps, inv_h0, norm;
    int levels;
    bool cell;
    KOKKOS_INLINE_FUNCTION void operator()(int i, double& dx, double& dy, double& dz) const {
        const double a = eps * Kokkos::sqrt(ax(i) * ax(i) + ay(i) * ay(i) + az(i) * az(i));
        double ux, uy, uz;
        if (!cell) {
            ferr_unit(ferr_mix(key ^ (std::uint64_t)(unsigned)pid(i)), ux, uy, uz);
            dx = a * ux; dy = a * uy; dz = a * uz;
            return;
        }
        double sx = 0.0, sy = 0.0, sz = 0.0, s = inv_h0;
        for (int l = 0; l < levels; ++l, s *= 2.0) {
            const std::uint64_t cx = (std::uint64_t)(long long)Kokkos::floor(x(i) * s);
            const std::uint64_t cy = (std::uint64_t)(long long)Kokkos::floor(y(i) * s);
            const std::uint64_t cz = (std::uint64_t)(long long)Kokkos::floor(z(i) * s);
            const std::uint64_t h = ferr_mix(key ^ ferr_mix((std::uint64_t)l ^ ferr_mix(cx ^ ferr_mix(cy ^ ferr_mix(cz)))));
            ferr_unit(h, ux, uy, uz);
            sx += ux; sy += uy; sz += uz;
        }
        dx = a * norm * sx; dy = a * norm * sy; dz = a * norm * sz;
    }
};

void force_err_inject(Particles& p, int start, int count, int set_count) {
    const ForceErr& c = force_err_cfg();
    if (!(c.eps > 0.0) || set_count < c.nmin || count <= 0) return;
    const std::uint64_t call = ++g_force_err_calls;
    const double eps = c.eps;
    auto ax = p.ax, ay = p.ay, az = p.az;
    if (!c.smooth) {
        auto x = p.x, y = p.y, z = p.z, m = p.mass;
        if (c.cell && !(g_force_err_r0 > 0.0)) {   // coarsest grid from the first perturbed set (rank 0's value)
            double s[4] = {0, 0, 0, 0}, r2 = 0.0;
            Kokkos::parallel_reduce("force_err_cm", count, KOKKOS_LAMBDA(int ii, double& sm, double& sx, double& sy,
                                                                          double& sz) {
                const int i = start + ii;
                sm += m(i); sx += m(i) * x(i); sy += m(i) * y(i); sz += m(i) * z(i);
            }, s[0], s[1], s[2], s[3]);
            const double xc = s[1] / s[0], yc = s[2] / s[0], zc = s[3] / s[0];
            Kokkos::parallel_reduce("force_err_r0", count, KOKKOS_LAMBDA(int ii, double& acc) {
                const int i = start + ii;
                const double dx = x(i) - xc, dy = y(i) - yc, dz = z(i) - zc;
                acc += m(i) * (dx * dx + dy * dy + dz * dz);
            }, r2);
            double r0 = std::sqrt(r2 / s[0]);
#ifdef USE_MPI
            MPI_CHECK(MPI_Bcast(&r0, 1, MPI_DOUBLE, 0, MPIComm::comm()));
#endif
            g_force_err_r0 = r0 > 0.0 ? r0 : 1.0;
            if (MPIComm::rank() == 0) std::printf("NBODY_FORCE_ERR: cell grids of edge %.4e / 2^l\n", g_force_err_r0);
        }
        FerrField F{ax, ay, az, x, y, z, p.particle_id,
                    ferr_mix(((c.frozen || c.cell) ? 0 : call) ^ 0x46455252ULL), eps,
                    c.cell ? 1.0 / g_force_err_r0 : 0.0, 1.0 / std::sqrt((double)c.levels), c.levels, c.cell};
        double mean[3] = {0.0, 0.0, 0.0};
        if (c.pfix) {
            double s[4] = {0, 0, 0, 0};
            Kokkos::parallel_reduce("force_err_mean", count, KOKKOS_LAMBDA(int ii, double& sm, double& sx, double& sy,
                                                                            double& sz) {
                const int i = start + ii;
                double dx, dy, dz;
                F(i, dx, dy, dz);
                sm += m(i); sx += m(i) * dx; sy += m(i) * dy; sz += m(i) * dz;
            }, s[0], s[1], s[2], s[3]);
            if (s[0] > 0.0) { mean[0] = s[1] / s[0]; mean[1] = s[2] / s[0]; mean[2] = s[3] / s[0]; }
#ifdef USE_MPI
            MPI_CHECK(MPI_Bcast(mean, 3, MPI_DOUBLE, 0, MPIComm::comm()));
#endif
        }
        const double m0 = mean[0], m1 = mean[1], m2 = mean[2];
        double al[6] = {0, 0, 0, 0, 0, 0};   // alpha, mass centre
        if (c.lfix) {
            double cm[4] = {0, 0, 0, 0};
            Kokkos::parallel_reduce("force_err_lcm", count, KOKKOS_LAMBDA(int ii, double& sm, double& sx, double& sy,
                                                                           double& sz) {
                const int i = start + ii;
                sm += m(i); sx += m(i) * x(i); sy += m(i) * y(i); sz += m(i) * z(i);
            }, cm[0], cm[1], cm[2], cm[3]);
            const double xc = cm[1] / cm[0], yc = cm[2] / cm[0], zc = cm[3] / cm[0];
            double t[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};   // torque (3), Ixx Iyy Izz Ixy Ixz Iyz
            Kokkos::parallel_reduce("force_err_torque", count, KOKKOS_LAMBDA(int ii, double& tx, double& ty, double& tz,
                    double& ixx, double& iyy, double& izz, double& ixy, double& ixz, double& iyz) {
                const int i = start + ii;
                double dx, dy, dz;
                F(i, dx, dy, dz);
                dx -= m0; dy -= m1; dz -= m2;
                const double rx = x(i) - xc, ry = y(i) - yc, rz = z(i) - zc, mi = m(i);
                tx += mi * (ry * dz - rz * dy); ty += mi * (rz * dx - rx * dz); tz += mi * (rx * dy - ry * dx);
                ixx += mi * (ry * ry + rz * rz); iyy += mi * (rx * rx + rz * rz); izz += mi * (rx * rx + ry * ry);
                ixy -= mi * rx * ry; ixz -= mi * rx * rz; iyz -= mi * ry * rz;
            }, t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8]);
            // alpha = I^-1 T (Cramer)
            const double a = t[3], b = t[6], cc = t[7], d = t[4], e = t[8], g = t[5];   // [[a b c][b d e][c e g]]
            const double c00 = d * g - e * e, c01 = cc * e - b * g, c02 = b * e - cc * d;
            const double c11 = a * g - cc * cc, c12 = b * cc - a * e, c22 = a * d - b * b;
            const double det = a * c00 + b * c01 + cc * c02;
            if (det > 0.0) {
                al[0] = (c00 * t[0] + c01 * t[1] + c02 * t[2]) / det;
                al[1] = (c01 * t[0] + c11 * t[1] + c12 * t[2]) / det;
                al[2] = (c02 * t[0] + c12 * t[1] + c22 * t[2]) / det;
            }
            al[3] = xc; al[4] = yc; al[5] = zc;
#ifdef USE_MPI
            MPI_CHECK(MPI_Bcast(al, 6, MPI_DOUBLE, 0, MPIComm::comm()));
#endif
        }
        const double wx = al[0], wy = al[1], wz = al[2], xc = al[3], yc = al[4], zc = al[5];
        Kokkos::parallel_for("force_err_random", count, KOKKOS_LAMBDA(int ii) {
            const int i = start + ii;
            double dx, dy, dz;
            F(i, dx, dy, dz);
            const double rx = x(i) - xc, ry = y(i) - yc, rz = z(i) - zc;
            ax(i) += dx - m0 - (wy * rz - wz * ry);
            ay(i) += dy - m1 - (wz * rx - wx * rz);
            az(i) += dz - m2 - (wx * ry - wy * rx);
        });
        return;
    }
    // smooth: mass centre, rms radius, rms |a| of the range (rank 0's values broadcast: every rank adds the same field)
    auto x = p.x, y = p.y, z = p.z, m = p.mass;
    double s[5] = {0, 0, 0, 0, 0};   // sum m, sum m x, sum m y, sum m z, sum |a|^2
    Kokkos::parallel_reduce("force_err_moments", count, KOKKOS_LAMBDA(int ii, double& sm, double& sx, double& sy,
                                                                       double& sz, double& sa) {
        const int i = start + ii;
        sm += m(i); sx += m(i) * x(i); sy += m(i) * y(i); sz += m(i) * z(i);
        sa += ax(i) * ax(i) + ay(i) * ay(i) + az(i) * az(i);
    }, s[0], s[1], s[2], s[3], s[4]);
    if (!(s[0] > 0.0)) return;
    const double xc = s[1] / s[0], yc = s[2] / s[0], zc = s[3] / s[0];
    double r2 = 0.0;
    Kokkos::parallel_reduce("force_err_rms", count, KOKKOS_LAMBDA(int ii, double& acc) {
        const int i = start + ii;
        const double dx = x(i) - xc, dy = y(i) - yc, dz = z(i) - zc;
        acc += m(i) * (dx * dx + dy * dy + dz * dz);
    }, r2);
    double g[6] = {xc, yc, zc, std::sqrt(r2 / s[0]), std::sqrt(s[4] / count), 0.0};
#ifdef USE_MPI
    MPI_CHECK(MPI_Bcast(g, 6, MPI_DOUBLE, 0, MPIComm::comm()));
#endif
    if (!(g[3] > 0.0)) return;
    // M: Box-Muller N(0, 1/3) entries from the call counter
    double M[9];
    std::uint64_t h = ferr_mix(call ^ 0x534d4f4fULL);
    for (int k = 0; k < 9; k += 2) {
        const double u1 = ferr_u01(h = ferr_mix(h)), u2 = ferr_u01(h = ferr_mix(h));
        const double r = std::sqrt(-2.0 * std::log(u1) / 3.0);
        M[k] = r * std::cos(6.283185307179586 * u2);
        if (k + 1 < 9) M[k + 1] = r * std::sin(6.283185307179586 * u2);
    }
    const double f = eps * g[4] / g[3];
    const double m00 = f * M[0], m01 = f * M[1], m02 = f * M[2], m10 = f * M[3], m11 = f * M[4], m12 = f * M[5],
                 m20 = f * M[6], m21 = f * M[7], m22 = f * M[8];
    const double cx = g[0], cy = g[1], cz = g[2];
    Kokkos::parallel_for("force_err_smooth", count, KOKKOS_LAMBDA(int ii) {
        const int i = start + ii;
        const double dx = x(i) - cx, dy = y(i) - cy, dz = z(i) - cz;
        ax(i) += m00 * dx + m01 * dy + m02 * dz;
        ay(i) += m10 * dx + m11 * dy + m12 * dz;
        az(i) += m20 * dx + m21 * dy + m22 * dz;
    });
}
}  // namespace

// TILE_SIZE defined in constants.h

// =============================================================================
// Distributed force + timestep computation for between-step updates
// Each rank computes for its slice [my_start, my_count) from ALL N particles
// Then allgather both accelerations and timestep values
// Uses tiled GPU kernel with shared memory for efficiency
// =============================================================================
void compute_acceleration_and_timestep_distributed(DistributedParticles& dp,
                                                 double G, double soft2, double eta) {
#ifdef USE_MPI
    if (MPIComm::size() == 1) {
        // Single rank: use optimized single-GPU version
        compute_acceleration_and_timestep(dp.local, G, soft2, eta);
        return;
    }

    const int N = dp.N_total;
    const int my_start = dp.my_start;
    const int my_count = dp.my_count;

    if (my_count == 0) {
        dp.allgather_dt_acc_subset(0, N);
        return;
    }



    (void)N;
    acceleration_and_timestep_range(dp.local, my_start, my_count, 0, N, G, soft2, eta);
    Kokkos::fence("integrator.cpp:104");

    // Accelerations and time steps in one collective
    dp.allgather_dt_acc_subset(0, N);
#else
    compute_acceleration_and_timestep(dp.local, G, soft2, eta);
#endif
}


// Debug: check for duplicate positions in the global tail of the particle array
// Uses FULL array copy (not subview) to match diagnostic's copy method
// Helper: drift positions for [start, start+count)
// Not static - also used by distributed FSI
void drift(Particles& p, int start, int count, double dt) {
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    if (use_integer_positions) {
        auto pix = raw(p.ix); auto piy = raw(p.iy); auto piz = raw(p.iz);
        double scale = int_pos_scale;
        double inv_scale = int_pos_inv_scale;
        small_for("drift_int", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            pix(i) += llrint(pvx(i) * dt * inv_scale);
            piy(i) += llrint(pvy(i) * dt * inv_scale);
            piz(i) += llrint(pvz(i) * dt * inv_scale);
            px(i) = (double)pix(i) * scale;
            py(i) = (double)piy(i) * scale;
            pz(i) = (double)piz(i) * scale;
        });
    } else {
        small_for("drift", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            px(i) += pvx(i) * dt;
            py(i) += pvy(i) * dt;
            pz(i) += pvz(i) * dt;
        });
    }
    // No fence - next operation will handle dependency
}

#ifdef ENABLE_COLLISIONAL
void drift_field_only(Particles& p, int start, int count, double dt) {
    if (!p.collisional) {
        drift(p, start, count, dt);
        return;
    }
    RawView px{p.x.data()}, py{p.y.data()}, pz{p.z.data()};
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    auto p_sid = raw(p.subsystem_id);
    if (use_integer_positions) {
        auto pix = raw(p.ix); auto piy = raw(p.iy); auto piz = raw(p.iz);
        double scale = int_pos_scale;
        double inv_scale = int_pos_inv_scale;
        small_for("drift_field_int", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            if (p_sid(i) >= 0) return;  // skip subsystem members
            pix(i) += llrint(pvx(i) * dt * inv_scale);
            piy(i) += llrint(pvy(i) * dt * inv_scale);
            piz(i) += llrint(pvz(i) * dt * inv_scale);
            px(i) = (double)pix(i) * scale;
            py(i) = (double)piy(i) * scale;
            pz(i) = (double)piz(i) * scale;
        });
    } else {
        small_for("drift_field", count, KOKKOS_LAMBDA(int ii) {
            int i = start + ii;
            if (p_sid(i) >= 0) return;  // skip subsystem members
            px(i) += pvx(i) * dt;
            py(i) += pvy(i) * dt;
            pz(i) += pvz(i) * dt;
        });
    }
}

// =============================================================================

// Collisions decided inside the LogH integrations of the current advance_subsystems_in_range call
static std::vector<SubsystemCollision> g_pending_collisions;

// Wall time per phase of advance_subsystems_in_range, summed over a step (printed with the step summary)
// Phases: mirror = host copy of the range; map+select = pid map, eligibility, classification; logh_prep = sort and
// save; logh_pack / logh_run / logh_unpack = the batch calls (with their host<->device part *_xfer, the integration
// loop's wall time integ, its longest subsystem crit, and the summed subsystem time work); logh_post = restarts,
// PN bookkeeping and state propagation. Counts: calls with eligible subsystems, eligible, secular, LogH.
// det_* split "Subsystem detect" (neighbour search, pair classification, flags + diagnostics); dt_wait is the
// asynchronous time-step kernels finishing, dt_subsys the shared/perturbation time steps, dt_merge the merger
// detection (the three make up "Subsystem timestep").
struct SubsystemPhaseTimes {
    enum { MIRROR, MAP, SECULAR, SEC_TRIPLE, LOGH_PREP, LOGH_PACK, LOGH_RUN, LOGH_UNPACK, LOGH_RETRY,
           PACK_XFER, RUN_XFER, INTEG, CRIT, WORK, UNPACK_XFER,
           DET_NGB, DET_PAIRS, DET_FLAGS, DT_WAIT, DT_SUBSYS, DT_MERGE, N };
    enum { C_CALLS, C_WORK, C_ELIG, C_SEC, C_LOGH, NC };
    static constexpr int MAXD = 40;
    double t[N] = {};
    double td[MAXD][N] = {};   // the same per HHS depth
    long long calls[MAXD] = {};
    long long cnt[MAXD][NC] = {};
    static int depth() { return std::min(g_profiler.cur_depth, MAXD - 1); }
    void add(int phase, double dt) {
        t[phase] += dt;
        td[depth()][phase] += dt;
        // the wall-time phases (not the batch split *_xfer, integ, crit, work) are buckets of the step profiler
        static constexpr int bucket[N] = {
            StepProfiler::SUBSYS_MIRROR, StepProfiler::SUBSYS_MAP, StepProfiler::SUBSYS_SECULAR,
            StepProfiler::SUBSYS_SEC_TRIPLE, StepProfiler::SUBSYS_LOGH_PREP, StepProfiler::SUBSYS_LOGH_PACK,
            StepProfiler::SUBSYS_LOGH, StepProfiler::SUBSYS_LOGH_UNPACK, StepProfiler::SUBSYS_LOGH_POST,
            -1, -1, -1, -1, -1, -1,
            StepProfiler::DET_NGB, StepProfiler::DET_PAIRS, StepProfiler::DET_FLAGS,
            StepProfiler::DT_WAIT, StepProfiler::DT_SUBSYS, StepProfiler::DT_MERGE};
        if (bucket[phase] >= 0) g_profiler.add_child_time((StepProfiler::Timer)bucket[phase], dt);
    }
    void count(int c, long long n) { cnt[depth()][c] += n; }
    void take_batch_prof() {   // the batch calls' split since the last take
        add(PACK_XFER, g_logh_prof.pack_xfer); add(RUN_XFER, g_logh_prof.run_xfer);
        add(INTEG, g_logh_prof.integ); add(CRIT, g_logh_prof.crit); add(WORK, g_logh_prof.work);
        add(UNPACK_XFER, g_logh_prof.unpack_xfer);
        g_logh_prof = LoghBatchProf();
    }
    bool any() const { for (double v : t) if (v > 0.0) return true; return false; }
    void report_and_reset() {
        static const char* name[N] = {"mirror", "map+select", "secular", "sec_triple", "logh_prep", "logh_pack",
                                      "logh_run", "logh_unpack", "logh_post", "pack_xfer", "run_xfer", "integ",
                                      "crit", "work", "unpack_xfer", "det_ngb", "det_pairs", "det_flags",
                                      "dt_wait", "dt_subsys", "dt_merge"};
        static const char* cname[NC] = {"calls", "w/subs", "elig", "secular", "logh"};
        std::printf("  Subsystem phases [s]:");
        for (int i = 0; i < N; i++) std::printf(" %s=%.3f", name[i], t[i]);
        std::printf("\n  %-6s", "Depth");
        for (int c = 0; c < NC; c++) std::printf(" %8s", cname[c]);
        for (int i = 0; i < N; i++) std::printf(" %11s", name[i]);
        std::printf("\n");
        for (int d = 0; d < MAXD; d++) {
            if (calls[d] == 0) continue;
            std::printf("  d%-5d %8lld", d, calls[d]);
            for (int c = 1; c < NC; c++) std::printf(" %8lld", cnt[d][c]);
            for (int i = 0; i < N; i++) std::printf(" %11.4f", td[d][i]);
            std::printf("\n");
        }
        *this = SubsystemPhaseTimes();
    }
};
static SubsystemPhaseTimes g_subsys_phase;
namespace {
struct PhaseClock {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double lap() {
        auto t1 = std::chrono::steady_clock::now();
        double d = std::chrono::duration<double>(t1 - t0).count();
        t0 = t1;
        return d;
    }
};
}  // namespace

// Host copy of x, y, z, vx, vy, vz, mass of the member particles of the subsystems one subsystem advance owns,
// all inside [start, start + count). The host columns span the range (access as col[f](idx - start)), but only the
// member slots are transferred: one gather into a packed device buffer and one transfer per direction (before
// 2026-10-17 the whole range went both ways). Nothing reads or writes another slot of the columns. Buffers persist.
// The "mirror" phase time is mostly the wait for the FSI kernels still queued when the first copy fences (50k
// cluster, 10 % binaries: 0.45 s per step with the whole range and with the members only)
namespace {
struct RangeMirror {
    enum { X, Y, Z, VX, VY, VZ, M, NF };
    using HostCol = Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
    int start = 0, count = 0;
    HostCol col[NF];
    std::vector<int> members;   // global indices of the member slots, ascending
    static Kokkos::View<double*, Kokkos::HostSpace>& hbuf() {
        static Kokkos::View<double*, Kokkos::HostSpace> b("range_mirror_host", 0);
        static bool hook = (Kokkos::push_finalize_hook([] { b = Kokkos::View<double*, Kokkos::HostSpace>(); }), true);
        (void)hook;
        return b;
    }
    static DeviceStaging& staging() { return DeviceStaging::get<7>(); }   // a host twin in a host subtree
    // nf = 6: positions and velocities, 7: also masses
    void pull(Particles& p, int nf) {
        DeviceStaging& st = staging();
        const int n = (int)members.size();
        auto d_idx = raw(st.d_idx);
        auto o = raw(st.d_buf);
        RawView x{p.x.data()}, y{p.y.data()}, z{p.z.data()}, vx{p.vx.data()}, vy{p.vy.data()}, vz{p.vz.data()},
            m{p.mass.data()};
        small_for("range_mirror_pack", n, KOKKOS_LAMBDA(int k) {
            const int i = d_idx(k);
            o(k) = x(i); o(n + k) = y(i); o(2 * n + k) = z(i);
            o(3 * n + k) = vx(i); o(4 * n + k) = vy(i); o(5 * n + k) = vz(i);
            if (nf > 6) o(6 * n + k) = m(i);
        });
        st.download_buf((size_t)nf * n);
        for (int f = 0; f < nf; f++) {
            const double* src = st.h_buf.data() + (size_t)f * n;
            for (int k = 0; k < n; k++) col[f](members[k] - start) = src[k];
        }
    }
    void push(Particles& p) {   // positions and velocities
        DeviceStaging& st = staging();
        const int n = (int)members.size();
        for (int f = 0; f < 6; f++) {
            double* dst = st.h_buf.data() + (size_t)f * n;
            for (int k = 0; k < n; k++) dst[k] = col[f](members[k] - start);
        }
        st.upload_buf((size_t)6 * n);
        auto d_idx = raw(st.d_idx);
        auto o = raw(st.d_buf);
        RawView x{p.x.data()}, y{p.y.data()}, z{p.z.data()}, vx{p.vx.data()}, vy{p.vy.data()}, vz{p.vz.data()};
        small_for("range_mirror_unpack", n, KOKKOS_LAMBDA(int k) {
            const int i = d_idx(k);
            x(i) = o(k); y(i) = o(n + k); z(i) = o(2 * n + k);
            vx(i) = o(3 * n + k); vy(i) = o(4 * n + k); vz(i) = o(5 * n + k);
        });
    }
    // member_slots: global indices in [s, s + n) (any order, repeats allowed)
    RangeMirror(Particles& p, int s, int n, std::vector<int> member_slots)
        : start(s), count(n), members(std::move(member_slots)) {
        std::sort(members.begin(), members.end());
        members.erase(std::unique(members.begin(), members.end()), members.end());
        if ((int)hbuf().extent(0) < NF * n) Kokkos::realloc(Kokkos::WithoutInitializing, hbuf(), NF * n);
        for (int f = 0; f < NF; f++) col[f] = HostCol(hbuf().data() + (size_t)f * n, n);
        const int nm = (int)members.size();
        DeviceStaging& st = staging();
        st.reserve(nm, NF);
        std::copy(members.begin(), members.end(), st.h_idx.data());
        st.upload_idx(nm);
        pull(p, NF);
    }
};

// NBODY_CHECK_SUBSYS=1: every subsystem this call owns must be advanced exactly once, over the full dt, by
// one of the paths (secular binary, secular triple, LogH). The FSI drifts no subsystem members, so a path that
// skips a subsystem (a `continue`, an early return) or stops short freezes its members with no energy change,
// invisible to the energy check and the restarts (AUDIT_2026-10.md F5). Aborts with the offending subsystem.
struct SubsysAdvanceCheck {
    bool on = false;
    double dt = 0.0;
    std::vector<int> owned;           // indices into g_subsystem_mgr.active this call must advance
    std::vector<int> n_adv;           // per active index: number of advances
    std::vector<double> t_adv;        // per active index: time advanced
    std::vector<const char*> path;    // per active index: last path that advanced it

    static bool enabled() {
        static const bool e = [] { const char* v = std::getenv("NBODY_CHECK_SUBSYS"); return v && v[0] && v[0] != '0'; }();
        return e;
    }
    void begin(int n_active, double dt_) {
        on = enabled();
        if (!on) return;
        dt = dt_;
        owned.clear();
        n_adv.assign(n_active, 0);
        t_adv.assign(n_active, 0.0);
        path.assign(n_active, "none");
    }
    void own(int s) { if (on) owned.push_back(s); }
    void advanced(int s, double t, const char* via) {
        if (!on) return;
        n_adv[s]++;
        t_adv[s] += t;
        path[s] = via;
    }
    void set_time(int s, double t) { if (on) t_adv[s] = t; }   // LogH retry replaces the first attempt
    void verify(int start, int count) const {
        if (!on) return;
        // GBS lands on the end time to a relative 1e-4 (NBODY_GBS_TTOL); anything further off is a partial advance
        const double tol = 1e-3 * std::fabs(dt);
        for (int s : owned) {
            if (n_adv[s] == 1 && std::fabs(t_adv[s] - dt) <= tol) continue;
            const auto& sub = g_subsystem_mgr.active[s];
            std::fprintf(stderr,
                "NBODY_CHECK_SUBSYS: subsystem %d (n_bodies=%d, type=%d, is_secular=%d, pids", s, sub.n_bodies,
                (int)sub.type, (int)sub.is_secular);
            for (int pid : sub.particle_ids) std::fprintf(stderr, " %d", pid);
            std::fprintf(stderr, ") advanced %d times over %.6e of dt=%.6e (last path: %s) in range [%d,%d)\n",
                         n_adv[s], t_adv[s], dt, path[s], start, start + count);
            std::fflush(stderr);
            std::abort();
        }
    }
};
}  // namespace

static void advance_subsystems_in_range_impl(Particles& p, int start, int count, double G, double dt);
// NBODY_ELEDGER: per-path booking residues (members' T + U change from the states + the path's own booking),
// summed over a step, code units; printed and reset with the ELEDGER lines
static double g_led_sec2 = 0.0, g_led_sec2_pn = 0.0, g_led_sec3 = 0.0;
static int g_led_n_sec2 = 0, g_led_n_sec3 = 0;
static inline double pair_energy(double m1, const double* r1, const double* v1, double m2, const double* r2,
                                 const double* v2, double G) {
    const double dx = r2[0] - r1[0], dy = r2[1] - r1[1], dz = r2[2] - r1[2];
    return 0.5 * m1 * (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]) +
           0.5 * m2 * (v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]) - G * m1 * m2 / std::sqrt(dx * dx + dy * dy + dz * dz);
}
void advance_subsystems_in_range_impl(Particles& p, int start, int count,
                                       double G, double dt) {
    PhaseClock phase_clock;
    if (!p.collisional || g_subsystem_mgr.active.empty()) return;
    NvtxScope nvtx_subsys("SUBSYS");
    g_subsys_phase.calls[std::min(g_profiler.cur_depth, SubsystemPhaseTimes::MAXD - 1)]++;

    // Rebuild pid→idx mapping: partition_by_pivot() reorders particles,
    // making the mapping from detect_from_pairs() stale.
    g_subsystem_mgr.rebuild_pid_to_idx(p);

    // Find which subsystems have ALL members in [start, start+count)
    const auto& pid_to_idx = g_subsystem_mgr.pid_to_idx();

    // Filter: only subsystems fully contained in [start, start+count)
    std::vector<int> eligible;
    int end = start + count;
    auto rng = Kokkos::pair<int,int>(start, end);
    for (int s = 0; s < (int)g_subsystem_mgr.active.size(); s++) {
        const auto& sub = g_subsystem_mgr.active[s];
        bool all_in = true;
        for (int pid : sub.particle_ids) {
            int idx = (pid < (int)pid_to_idx.size()) ? pid_to_idx[pid] : -1;
            if (idx < start || idx >= end) { all_in = false; break; }
        }
        if (all_in) eligible.push_back(s);
    }

    if (eligible.empty()) {
        g_subsys_phase.add(SubsystemPhaseTimes::MAP, phase_clock.lap());
        return;
    }
    g_subsys_phase.count(SubsystemPhaseTimes::C_WORK, 1);
    g_subsys_phase.count(SubsystemPhaseTimes::C_ELIG, (long long)eligible.size());
    g_subsys_phase.add(SubsystemPhaseTimes::MAP, phase_clock.lap());

    // Host copy of the members' particle data; all accessed indices are members in [start, start+count),
    // access as h_x(idx - start). Changes go back with mirror.push(p).
    std::vector<int> member_slots;
    for (int s : eligible)
        for (int pid : g_subsystem_mgr.active[s].particle_ids) member_slots.push_back(pid_to_idx[pid]);
    RangeMirror mirror(p, start, count, std::move(member_slots));
    g_subsys_phase.add(SubsystemPhaseTimes::MIRROR, phase_clock.lap());
    auto h_x = mirror.col[RangeMirror::X], h_y = mirror.col[RangeMirror::Y], h_z = mirror.col[RangeMirror::Z];
    auto h_vx = mirror.col[RangeMirror::VX], h_vy = mirror.col[RangeMirror::VY], h_vz = mirror.col[RangeMirror::VZ];
    auto h_m = mirror.col[RangeMirror::M];

    // Re-evaluate secular/LogH classification from current orbital state.
    // Subsystems persist across steps, so the detection-time is_secular may be stale.
    {

        int n_orb_secular = g_collisional.n_orb_secular;

        for (int s : eligible) {
            auto& sub = g_subsystem_mgr.active[s];

            if (sub.n_bodies == 2) {
                int idx_i = g_subsystem_mgr.safe_pid_to_idx(sub.particle_ids[0]);
                int idx_j = g_subsystem_mgr.safe_pid_to_idx(sub.particle_ids[1]);
                if (idx_i < 0 || idx_j < 0) continue;

                double rx = h_x(idx_j - start) - h_x(idx_i - start);
                double ry = h_y(idx_j - start) - h_y(idx_i - start);
                double rz = h_z(idx_j - start) - h_z(idx_i - start);
                double vx = h_vx(idx_j - start) - h_vx(idx_i - start);
                double vy = h_vy(idx_j - start) - h_vy(idx_i - start);
                double vz = h_vz(idx_j - start) - h_vz(idx_i - start);
                double r2 = rx*rx + ry*ry + rz*rz;
                double v2 = vx*vx + vy*vy + vz*vz;
                double r_mag = std::sqrt(r2);
                double mu = G * (h_m(idx_i - start) + h_m(idx_j - start));
                double E_orb = 0.5 * v2 - mu / r_mag;
                // Without PN the pair is a Kepler problem at any orbit count (collisional_config.h kepler_2body_pair)
                StellarPropsCache& kwc = StellarPropsCache::get();
                const bool kep2 = g_collisional.kepler_2body_pair(h_m(idx_i - start), kwc.type_of(p, sub.particle_ids[0]),
                                                                  h_m(idx_j - start), kwc.type_of(p, sub.particle_ids[1]));

                // Update orbital elements
                sub.orb.E_orb = E_orb;
                if (E_orb < 0.0) {
                    double a = -mu / (2.0 * E_orb);
                    double L2 = (ry*vz-rz*vy)*(ry*vz-rz*vy) +
                                (rz*vx-rx*vz)*(rz*vx-rx*vz) +
                                (rx*vy-ry*vx)*(rx*vy-ry*vx);
                    double e2 = 1.0 - L2 / (mu * a);
                    sub.orb.a = a;
                    sub.orb.e = (e2 > 0.0) ? std::sqrt(e2) : 0.0;
                    sub.orb.period = 2.0 * M_PI * std::sqrt(a * a * a / mu);
                    sub.n_orb_per_dt = (int)(dt / sub.orb.period);
                    sub.is_secular = (sub.n_orb_per_dt >= n_orb_secular) || kep2;

                    // Transition: was FLYBY but now bound → BINARY
                    if (sub.type == SubsystemType::FLYBY) sub.type = SubsystemType::BINARY;
                } else {
                    sub.orb.a = -mu / (2.0 * E_orb);  // hyperbolic
                    sub.is_secular = kep2;            // the universal-variable drift covers unbound orbits

                    // Transition: was BINARY but now unbound → FLYBY
                    if (sub.type == SubsystemType::BINARY) sub.type = SubsystemType::FLYBY;
                }

            } else if (sub.n_bodies == 3 && sub.inner_binary_id >= 0 && sub.outer_companion_pid >= 0) {
                // Recheck stability for triples with known hierarchy
                int inner_i = sub.inner_binary_id;
                int outer_pid = sub.outer_companion_pid;
                int inner_j = -1;
                for (int pid : sub.particle_ids) {
                    if (pid != inner_i && pid != outer_pid) { inner_j = pid; break; }
                }
                if (inner_j < 0) continue;

                int idx_i = (inner_i < (int)pid_to_idx.size()) ? pid_to_idx[inner_i] : -1;
                int idx_j = (inner_j < (int)pid_to_idx.size()) ? pid_to_idx[inner_j] : -1;
                int idx_o = (outer_pid < (int)pid_to_idx.size()) ? pid_to_idx[outer_pid] : -1;
                if (idx_i < 0 || idx_j < 0 || idx_o < 0) continue;

                double m1 = h_m(idx_i - start), m2 = h_m(idx_j - start), m3 = h_m(idx_o - start);
                double mu_in = G * (m1 + m2);

                // Inner binary elements
                double r_in[3] = {h_x(idx_j - start)-h_x(idx_i - start), h_y(idx_j - start)-h_y(idx_i - start), h_z(idx_j - start)-h_z(idx_i - start)};
                double v_in[3] = {h_vx(idx_j - start)-h_vx(idx_i - start), h_vy(idx_j - start)-h_vy(idx_i - start), h_vz(idx_j - start)-h_vz(idx_i - start)};
                auto oe_in = cartesian_to_elements(r_in, v_in, mu_in);

                if (oe_in.a > 0) {
                    sub.orb.a = oe_in.a;
                    sub.orb.e = oe_in.e;

                    // Outer orbit relative to inner COM
                    double M_in = m1 + m2;
                    double cx  = (m1*h_x(idx_i - start)  + m2*h_x(idx_j - start))  / M_in;
                    double cy  = (m1*h_y(idx_i - start)  + m2*h_y(idx_j - start))  / M_in;
                    double cz  = (m1*h_z(idx_i - start)  + m2*h_z(idx_j - start))  / M_in;
                    double cvx = (m1*h_vx(idx_i - start) + m2*h_vx(idx_j - start)) / M_in;
                    double cvy = (m1*h_vy(idx_i - start) + m2*h_vy(idx_j - start)) / M_in;
                    double cvz = (m1*h_vz(idx_i - start) + m2*h_vz(idx_j - start)) / M_in;
                    double r_out[3] = {h_x(idx_o - start)-cx, h_y(idx_o - start)-cy, h_z(idx_o - start)-cz};
                    double v_out[3] = {h_vx(idx_o - start)-cvx, h_vy(idx_o - start)-cvy, h_vz(idx_o - start)-cvz};
                    auto oe_out = cartesian_to_elements(r_out, v_out, G * (M_in + m3));

                    if (oe_out.a > 0) {
                        bool stable = SubsystemManager::mardling_stability(
                            oe_in.a, oe_out.a, oe_out.e, m1, m2, m3,
                            SubsystemManager::mutual_inclination(r_in, v_in, r_out, v_out));
                        sub.type = stable ? SubsystemType::TRIPLE_STABLE : SubsystemType::TRIPLE_UNSTABLE;

                        if (stable) {
                            double P_in = 2.0 * M_PI * std::sqrt(oe_in.a*oe_in.a*oe_in.a / mu_in);
                            sub.n_orb_per_dt = (int)(dt / P_in);
                            sub.is_secular = g_collisional.secular_triples && sub.n_orb_per_dt >= n_orb_secular;
                        } else {
                            sub.is_secular = false;
                        }
                    } else {
                        sub.type = SubsystemType::TRIPLE_UNSTABLE;
                        sub.is_secular = false;
                    }
                } else {
                    sub.type = SubsystemType::TRIPLE_UNSTABLE;
                    sub.is_secular = false;
                }
            }
        }
    }

    // Partition eligible subsystems into secular binaries, secular triples, and LogH (every rank advances all of
    // them: the replicas stay identical without an exchange)
    std::vector<int> secular_idx, secular_triple_idx, logh_idx;
    SubsysAdvanceCheck adv_check;
    adv_check.begin((int)g_subsystem_mgr.active.size(), dt);
    // Orbit-averaged paths cannot resolve a collision: a secular binary (or a secular triple's inner binary) whose
    // osculating pericentre lies inside its collision distance (FROST-CLUSTERS I App. B3 candidate), or a subsystem
    // with a massless placeholder member, goes to the regularised integration, which decides it at the pericentre
    std::vector<char> to_logh(g_subsystem_mgr.active.size(), 0);
    if (g_collisional.c_code > 0.0) {
        std::vector<int> sec_rows;   // member indices of secular candidates, two per subsystem
        std::vector<int> sec_subs;
        std::vector<int> sec_pids;   // their particle ids
        for (int s : eligible) {
            const auto& sub = g_subsystem_mgr.active[s];
            for (int pid : sub.particle_ids) {
                const int i = g_subsystem_mgr.safe_pid_to_idx(pid);
                if (i >= 0 && h_m(i - start) <= 0.0) to_logh[s] = 1;
            }
            if (to_logh[s] || !sub.is_secular) continue;
            int pa = -1, pb = -1;
            if (sub.n_bodies == 2) { pa = sub.particle_ids[0]; pb = sub.particle_ids[1]; }
            else if (sub.n_bodies == 3 && sub.inner_binary_id >= 0 && sub.outer_companion_pid >= 0) {
                pa = sub.inner_binary_id;
                for (int pid : sub.particle_ids)
                    if (pid != sub.inner_binary_id && pid != sub.outer_companion_pid) pb = pid;
            }
            const int ia = pa >= 0 ? g_subsystem_mgr.safe_pid_to_idx(pa) : -1;
            const int ib = pb >= 0 ? g_subsystem_mgr.safe_pid_to_idx(pb) : -1;
            if (ia < 0 || ib < 0) continue;
            sec_subs.push_back(s);
            sec_rows.push_back(ia);
            sec_rows.push_back(ib);
            sec_pids.push_back(pa);
            sec_pids.push_back(pb);
        }
        if (!sec_subs.empty()) {
            // Stellar types and radii of those members from the host cache by particle id (stellar_cache.h): the
            // device gather it replaced cost ~40 us + 0.25 us per binary per call (0.4-0.6 s per step at 200k)
            StellarPropsCache& props = StellarPropsCache::get();
            props.ensure(p);
            bool all = true;
            for (int pid : sec_pids) all = all && props.has(pid);
            if (!all) { props.invalidate(); props.ensure(p); }
            if (StellarPropsCache::check_enabled()) props.verify(p, sec_rows, sec_pids, "secular candidates");
            auto kw_of = [&](int row) { return props.type_of(sec_pids[row]); };
            auto rad_of = [&](int row) { return props.radius_of(sec_pids[row]); };
            constexpr double R_sun_to_pc = 2.254e-8;   // as mergers.cpp
            for (size_t q = 0; q < sec_subs.size(); q++) {
                const int ia = sec_rows[2 * q] - start, ib = sec_rows[2 * q + 1] - start;
                double r[3] = {h_x(ib) - h_x(ia), h_y(ib) - h_y(ia), h_z(ib) - h_z(ia)};
                double v[3] = {h_vx(ib) - h_vx(ia), h_vy(ib) - h_vy(ia), h_vz(ib) - h_vz(ia)};
                double rp = 0.0;
                const double tau = kepler_time_to_pericentre(r, v, G * (h_m(ia) + h_m(ib)), &rp);
                const double r_now = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
                MergerType ct;
                const int qa = 2 * (int)q, qb = qa + 1;
                const double rc = collision_distance(kw_of(qa), h_m(ia), rad_of(qa) * R_sun_to_pc, kw_of(qb), h_m(ib),
                                                     rad_of(qb) * R_sun_to_pc, G, g_collisional.c_code, ct);
                // A candidate goes to LogH when its pericentre falls in this interval (or it is inside r_coll now);
                // otherwise the exact drift cannot reach the collision distance before the interval ends (a pair
                // with >= 1 orbit per step always has its pericentre inside: unchanged for the secular binaries).
                // The margin covers the O(|1 - e|) of the near-parabolic pericentre time
                if (rp < rc && (tau <= dt * (1.0 + 1e-5) || r_now < rc)) to_logh[sec_subs[q]] = 1;
                // A PN binary whose pericentre is too relativistic for the Newtonian write-back of the secular step
                // (secular_pn_peri_shift, CASES.md C19) goes to LogH as well
                const double mA = h_m(ia), mB = h_m(ib);
                if (g_subsystem_mgr.active[sec_subs[q]].n_bodies == 2 &&
                    (g_collisional.pn_body(mA, kw_of(qa)) || g_collisional.pn_body(mB, kw_of(qb))) &&
                    secular_pn_peri_shift(r, v, G * (mA + mB), mA * mB / ((mA + mB) * (mA + mB)), g_collisional.c_code,
                                          g_collisional.pn_order) > SECULAR_PN_PERI_SHIFT_MAX) {
                    if (!to_logh[sec_subs[q]]) g_subsystem_mgr.step_secular_peri_logh++;
                    to_logh[sec_subs[q]] = 1;
                }
            }
        }
    }
    for (int k = 0; k < (int)eligible.size(); k++) {
        const int s = eligible[k];
        adv_check.own(s);
        const auto& sub = g_subsystem_mgr.active[s];
        if (subsys_advance_count(sub.particle_ids, dt, sub.orb.a, sub.orb.e, (int)sub.type)) {
            constexpr double R_sun_to_pc = 2.254e-8;   // as mergers.cpp
            auto& sc = StellarPropsCache::get();   // the members as RUMI++ has them, and its contact distance
            std::vector<int> kw; std::vector<double> m, R;
            for (int pid : sub.particle_ids) {
                const int ii = g_subsystem_mgr.safe_pid_to_idx(pid);
                kw.push_back(sc.type_of(p, pid)); R.push_back(sc.radius_of(pid)); m.push_back(ii >= 0 ? h_m(ii - start) : -1.0);
                std::printf("SUBSYS_ADVANCE_LIMIT member pid %d m %.6g Msun kw %d R %.6g Rsun\n", pid, m.back(), kw.back(), R.back());
            }
            if (sub.n_bodies == 2) {
                MergerType mt;
                const double rc = collision_distance(kw[0], m[0], R[0] * R_sun_to_pc, kw[1], m[1], R[1] * R_sun_to_pc, G,
                                                     g_collisional.c_code, mt);
                std::printf("SUBSYS_ADVANCE_LIMIT pericentre %.6e pc = %.4g Rsun, contact distance %.6e pc = %.4g Rsun\n",
                            sub.orb.a * (1 - sub.orb.e), sub.orb.a * (1 - sub.orb.e) / R_sun_to_pc, rc, rc / R_sun_to_pc);
            }
            std::fflush(stdout);
            std::abort();
        }
        if (to_logh[s]) {
            logh_idx.push_back(s);
        } else if (sub.is_secular && sub.n_bodies == 2) {
            secular_idx.push_back(s);
        } else if (sub.is_secular && sub.type == SubsystemType::TRIPLE_STABLE) {
            secular_triple_idx.push_back(s);
        } else {
            logh_idx.push_back(s);
        }
    }

    g_subsys_phase.add(SubsystemPhaseTimes::MAP, phase_clock.lap());
    g_subsys_phase.count(SubsystemPhaseTimes::C_SEC, (long long)secular_idx.size());


    // --- Secular binaries: evolve on host ---
    if (!secular_idx.empty()) {
        // Binaries are independent (disjoint members): evolve them on all host threads. The GW energy is
        // summed afterwards in a fixed order, so every MPI rank books the same value.
        const int n_sec = (int)secular_idx.size();
        std::vector<double> dE_gw(n_sec, 0.0);
        std::vector<double> sec_res(n_sec, 0.0);   // NBODY_ELEDGER, SUBSYS_ERR_TOL: booking residue per binary
        std::vector<double> sec_pre(14 * (size_t)n_sec, 0.0);   // SUBSYS_ERR_TOL: m x y z vx vy vz of both members before
        std::vector<char> sec_pn(n_sec, 0);
        std::vector<char> evolved(n_sec, 0);
        std::vector<int> substeps(n_sec, 0);
        std::vector<EnergyBookkeeping> sec_books(n_sec);  // P/L the secular evolution moves (PN, GW)
        std::vector<int> sec_kw(2 * (size_t)n_sec);         // members' stellar types (looked up before the threads)
        for (int k = 0; k < n_sec; k++) {
            const auto& sub = g_subsystem_mgr.active[secular_idx[k]];
            sec_kw[2 * k] = StellarPropsCache::get().type_of(p, sub.particle_ids[0]);
            sec_kw[2 * k + 1] = StellarPropsCache::get().type_of(p, sub.particle_ids[1]);
        }
        auto evolve = [&](int k) {
            const auto& sub = g_subsystem_mgr.active[secular_idx[k]];
            int idx0 = g_subsystem_mgr.safe_pid_to_idx(sub.particle_ids[0]);
            int idx1 = g_subsystem_mgr.safe_pid_to_idx(sub.particle_ids[1]);
            if (idx0 < 0 || idx1 < 0) return;

            double p1[3] = {h_x(idx0 - start), h_y(idx0 - start), h_z(idx0 - start)};
            double v1[3] = {h_vx(idx0 - start), h_vy(idx0 - start), h_vz(idx0 - start)};
            double p2[3] = {h_x(idx1 - start), h_y(idx1 - start), h_z(idx1 - start)};
            double v2[3] = {h_vx(idx1 - start), h_vy(idx1 - start), h_vz(idx1 - start)};

            evolved[k] = 1;

            // PN terms only for BH binaries (CollisionalConfig::pn_body), as in LogH and Rantala et al. (2023): stellar binaries
            // are a Kepler drift. With PN on, a tight stellar binary took thousands of 1PN-precession sub-steps
            // per evolution (5-7 ms secular calls at depths 3-6 of the 200k cluster)
            double m1_sec = h_m(idx0 - start), m2_sec = h_m(idx1 - start);
            double mu_sec = G * (m1_sec + m2_sec);
            const bool pn_binary = g_collisional.pn_body(m1_sec, sec_kw[2 * k]) ||
                                   g_collisional.pn_body(m2_sec, sec_kw[2 * k + 1]);
            sec_pn[k] = pn_binary;
            const int pn_b = pn_binary ? g_collisional.pn_order : 0;

            // GW energy bookkeeping (PN evolution only): energy radiated = E_orb_before - E_orb_after, from the
            // states (any sign of the orbital energy: 043190d-C19 booked it from a > 0 only, so an orbit written back
            // unbound left its whole change unbooked). Without PN the drift conserves E_orb exactly; booking its
            // round-off as "GW" energy hid the drift's error from the energy check
            auto e_orb = [&](const double* q1, const double* w1, const double* q2, const double* w2) {
                const double dr[3] = {q2[0]-q1[0], q2[1]-q1[1], q2[2]-q1[2]}, dv[3] = {w2[0]-w1[0], w2[1]-w1[1], w2[2]-w1[2]};
                const double mred = m1_sec * m2_sec / (m1_sec + m2_sec);
                return mred * (0.5 * (dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2])
                               - mu_sec / std::sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]));
            };
            const double eorb_pre = pn_b ? e_orb(p1, v1, p2, v2) : 0.0;

            static const bool led_sec = std::getenv("NBODY_ELEDGER") != nullptr;
            const bool err_log = g_collisional.subsys_err_tol > 0.0 && g_collisional.e_ref > 0.0;
            const double e_sec0 = (led_sec || err_log) ? pair_energy(m1_sec, p1, v1, m2_sec, p2, v2, G) : 0.0;
            if (err_log) {
                double* q = &sec_pre[14 * k];
                q[0] = m1_sec; q[7] = m2_sec;
                for (int c = 0; c < 3; c++) { q[1 + c] = p1[c]; q[4 + c] = v1[c]; q[8 + c] = p2[c]; q[11 + c] = v2[c]; }
            }
            substeps[k] = integrate_secular_binary(p1, v1, m1_sec, p2, v2, m2_sec, G, g_collisional.c_code, pn_b, dt);
            if (substeps[k] < 0) {   // PN write-back not possible (C19): unchanged, integrated by LogH below
                substeps[k] = 0;
                evolved[k] = 0;
                return;
            }

            if (pn_b) dE_gw[k] = eorb_pre - e_orb(p1, v1, p2, v2);
            if (led_sec || err_log) sec_res[k] = pair_energy(m1_sec, p1, v1, m2_sec, p2, v2, G) - e_sec0 + dE_gw[k];

            // P and L moved by the evolution (Kepler motion and the CoM drift conserve both)
            {
                double r1o[3] = {h_x(idx0 - start), h_y(idx0 - start), h_z(idx0 - start)};
                double v1o[3] = {h_vx(idx0 - start), h_vy(idx0 - start), h_vz(idx0 - start)};
                double r2o[3] = {h_x(idx1 - start), h_y(idx1 - start), h_z(idx1 - start)};
                double v2o[3] = {h_vx(idx1 - start), h_vy(idx1 - start), h_vz(idx1 - start)};
                sec_books[k].sink_PL(m1_sec, r1o, v1o);
                sec_books[k].sink_PL(m2_sec, r2o, v2o);
                sec_books[k].sink_PL(m1_sec, p1, v1, -1.0);
                sec_books[k].sink_PL(m2_sec, p2, v2, -1.0);
            }

            // Write back to host mirrors
            h_x(idx0 - start) = p1[0]; h_y(idx0 - start) = p1[1]; h_z(idx0 - start) = p1[2];
            h_vx(idx0 - start) = v1[0]; h_vy(idx0 - start) = v1[1]; h_vz(idx0 - start) = v1[2];
            h_x(idx1 - start) = p2[0]; h_y(idx1 - start) = p2[1]; h_z(idx1 - start) = p2[2];
            h_vx(idx1 - start) = v2[0]; h_vy(idx1 - start) = v2[1]; h_vz(idx1 - start) = v2[2];
        };
        // A few binaries (deep levels, thousands of calls per step) cost less than starting the host threads
        if (n_sec < 32) {
            for (int k = 0; k < n_sec; k++) evolve(k);
        } else {
            Kokkos::parallel_for("secular_binaries",
                Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>(0, n_sec),
                evolve);
        }
        if (g_collisional.subsys_err_tol > 0.0 && g_collisional.e_ref > 0.0 && MPIComm::rank() == 0)
            for (int k = 0; k < n_sec; k++)
                if (evolved[k] && std::fabs(sec_res[k]) > g_collisional.subsys_err_tol * g_collisional.e_ref &&
                    subsys_err_first(g_subsystem_mgr.active[secular_idx[k]].particle_ids)) {
                    const auto& sub = g_subsystem_mgr.active[secular_idx[k]];
                    const double* q = &sec_pre[14 * k];
                    auto& sc = StellarPropsCache::get();
                    const int k0 = sc.type_of(p, sub.particle_ids[0]), k1 = sc.type_of(p, sub.particle_ids[1]);
                    std::printf("SUBSYS_ERR step %d secular%s dE/E0 %.3e type %d n 2 dt %.3e a %.4e e %.5f m %.4g %.4g pids %d %d "
                                "state [%.17g %.17g %.17g %.17g %.17g %.17g %.17g %d %.17g] [%.17g %.17g %.17g %.17g %.17g %.17g %.17g %d %.17g]\n",
                                g_step_in_progress, sec_pn[k] ? "_pn" : "", sec_res[k] / g_collisional.e_ref, (int)sub.type,
                                dt * TIME_UNIT_MYR, sub.orb.a, sub.orb.e, q[0], q[7], sub.particle_ids[0], sub.particle_ids[1],
                                q[0], q[1], q[2], q[3], q[4], q[5], q[6], k0, k0 >= 0 ? sc.radius_of(sub.particle_ids[0]) : 0.0,
                                q[7], q[8], q[9], q[10], q[11], q[12], q[13], k1, k1 >= 0 ? sc.radius_of(sub.particle_ids[1]) : 0.0);
                }
        for (int k = 0; k < n_sec; k++) {
            if (evolved[k]) { g_led_sec2 += sec_res[k]; if (sec_pn[k]) g_led_sec2_pn += sec_res[k]; g_led_n_sec2++; }
            g_energy_books.E_gw += dE_gw[k];
            if (!sec_pn[k]) g_sec_nonpn_gw += dE_gw[k];   // NBODY_EATTR: the part booked for non-PN binaries
            for (int c = 0; c < 3; c++) {
                g_energy_books.P_sink[c] += sec_books[k].P_sink[c];
                g_energy_books.L_sink[c] += sec_books[k].L_sink[c];
            }
            g_subsystem_mgr.step_secular_evolved += evolved[k];
            if (evolved[k]) adv_check.advanced(secular_idx[k], dt, "secular binary");
            g_subsystem_mgr.step_secular_substeps += substeps[k];
            const auto& sk = g_subsystem_mgr.active[secular_idx[k]];
            if (evolved[k] &&
                (sk.orb.E_orb >= 0.0 || sk.n_orb_per_dt < g_collisional.n_orb_secular))
                g_subsystem_mgr.step_kepler2++;
        }

        // Pairs the secular write-back refused (integrate_secular_binary -1, members unchanged) go to LogH
        for (int k = 0; k < n_sec; k++)
            if (!evolved[k] && g_subsystem_mgr.safe_pid_to_idx(g_subsystem_mgr.active[secular_idx[k]].particle_ids[0]) >= 0 &&
                g_subsystem_mgr.safe_pid_to_idx(g_subsystem_mgr.active[secular_idx[k]].particle_ids[1]) >= 0) {
                logh_idx.push_back(secular_idx[k]);
                g_subsystem_mgr.step_secular_peri_logh++;
            }

        // Copy modified host subviews back to device
        mirror.push(p);

        // Clear aux_vel for secular subsystems (not used by secular path;
        // if subsystem transitions back to LogH, false triggers fresh init)
        for (int s : secular_idx) {
            auto& dst = g_subsystem_mgr.active[s];
            dst.has_aux_vel = false;
        }
    }

    g_subsys_phase.add(SubsystemPhaseTimes::SECULAR, phase_clock.lap());

    // --- Secular triples: Kozai-Lidov + GW dissipation ---
    struct LedTri { int i0, i1, i2; double e0, booked; };
    std::vector<LedTri> led_tri_pending;
    if (!secular_triple_idx.empty()) {
        for (int s : secular_triple_idx) {
            g_subsystem_mgr.step_secular_triples++;
            const auto& sub = g_subsystem_mgr.active[s];
            // A triple this path cannot evolve goes to LogH: skipping it would leave its members frozen for the
            // whole step (the FSI drifts no subsystem members)
            if (sub.n_bodies != 3 || sub.inner_binary_id < 0 || sub.outer_companion_pid < 0) {
                logh_idx.push_back(s);
                continue;
            }

            // Inner binary and outer companion from the classification (particle_ids are sorted by id, so
            // their order says nothing about the hierarchy)
            int inner_j = -1;
            for (int pid : sub.particle_ids)
                if (pid != sub.inner_binary_id && pid != sub.outer_companion_pid) inner_j = pid;
            int idx0 = g_subsystem_mgr.safe_pid_to_idx(sub.inner_binary_id);
            int idx1 = inner_j >= 0 ? g_subsystem_mgr.safe_pid_to_idx(inner_j) : -1;
            int idx2 = g_subsystem_mgr.safe_pid_to_idx(sub.outer_companion_pid);
            if (idx0 < 0 || idx1 < 0 || idx2 < 0) {
                logh_idx.push_back(s);
                continue;
            }

            double TM[3], TX[3][3], TV[3][3];
            const int tidx[3] = {idx0 - start, idx1 - start, idx2 - start};
            for (int b = 0; b < 3; b++) {
                const int i = tidx[b];
                TM[b] = h_m(i);
                TX[b][0] = h_x(i); TX[b][1] = h_y(i); TX[b][2] = h_z(i);
                TV[b][0] = h_vx(i); TV[b][1] = h_vy(i); TV[b][2] = h_vz(i);
            }
            static const bool led_tri = std::getenv("NBODY_ELEDGER") != nullptr;
            auto tri_energy = [&]() {   // T + U of the three members in the particle arrays
                double E = 0.0;
                for (int b = 0; b < 3; b++) {
                    const int i = tidx[b];
                    E += 0.5 * h_m(i) * (h_vx(i) * h_vx(i) + h_vy(i) * h_vy(i) + h_vz(i) * h_vz(i));
                    for (int c = b + 1; c < 3; c++) {
                        const int j = tidx[c];
                        const double dx = h_x(j) - h_x(i), dy = h_y(j) - h_y(i), dz = h_z(j) - h_z(i);
                        E -= G * h_m(i) * h_m(j) / std::sqrt(dx * dx + dy * dy + dz * dz);
                    }
                }
                return E;
            };
            const double E_tri0 = tri_energy();
            double tri_pre[21];   // SUBSYS_ERR_TOL: m x y z vx vy vz of the three members before the step
            for (int b = 0; b < 3; b++) {
                double* q = tri_pre + 7 * b;
                q[0] = TM[b]; q[1] = TX[b][0]; q[2] = TX[b][1]; q[3] = TX[b][2]; q[4] = TV[b][0]; q[5] = TV[b][1]; q[6] = TV[b][2];
            }
            const double m1 = TM[0], m2 = TM[1], m3 = TM[2];
            const int pn_tri = g_collisional.pn_body(m1, StellarPropsCache::get().type_of(p, sub.inner_binary_id)) ||
                               g_collisional.pn_body(m2, StellarPropsCache::get().type_of(p, inner_j))
                                   ? g_collisional.pn_order : 0;
            const int tpids[3] = {sub.inner_binary_id, inner_j, sub.outer_companion_pid};
            const SecularTripleStep st = secular_triple_step(TM, TX, TV, tpids, dt, G, g_collisional.c_code, pn_tri);
            if (!st.hierarchical) {   // no longer hierarchical: integrate it directly
                logh_idx.push_back(s);
                continue;
            }
            printf("  [SEC_TRI] pid=%d,%d,%d  a_in=%.4e  e_in=%.6f  a_out=%.4e  cos_i=%.4f  dt=%.4e\n",
                   sub.inner_binary_id, inner_j, sub.outer_companion_pid, st.a_in, st.e_in, st.a_out, st.cos_i, dt);
            adv_check.advanced(s, dt, "secular triple");
            if (st.booked_gw != 0.0) g_energy_books.E_gw += st.booked_gw;   // GW energy of the inner binary's inspiral
            for (int c = 0; c < 3; c++) g_energy_books.L_sink[c] -= st.dL_gw[c];   // and its angular momentum
            if (led_tri) led_tri_pending.push_back({idx0, idx1, idx2, E_tri0, st.booked_gw});
            g_subsystem_mgr.step_sec_triple_unfixed += st.unfixed;
            for (int b = 0; b < 3; b++) {
                const int i = tidx[b];
                h_x(i) = TX[b][0]; h_y(i) = TX[b][1]; h_z(i) = TX[b][2];
                h_vx(i) = TV[b][0]; h_vy(i) = TV[b][1]; h_vz(i) = TV[b][2];
            }
            {
                const double res = tri_energy() - (E_tri0 - st.booked_gw);   // what is left after the fix (round-off)
                if (g_collisional.subsys_err_tol > 0.0 && g_collisional.e_ref > 0.0 && MPIComm::rank() == 0 &&
                    std::fabs(res) > g_collisional.subsys_err_tol * g_collisional.e_ref && subsys_err_first(sub.particle_ids)) {
                    std::printf("SUBSYS_ERR step %d secular_triple dE/E0 %.3e type %d n 3 dt %.3e a %.4e e %.5f m %.4g %.4g %.4g "
                                "pids %d %d %d state", g_step_in_progress, res / g_collisional.e_ref, (int)sub.type,
                                dt * TIME_UNIT_MYR, st.a_in, st.e_in, m1, m2, m3, sub.inner_binary_id, inner_j,
                                sub.outer_companion_pid);
                    auto& sc = StellarPropsCache::get();
                    for (int b = 0; b < 3; b++) {
                        const int kw = sc.type_of(p, tpids[b]);
                        std::printf(" [%.17g %.17g %.17g %.17g %.17g %.17g %.17g %d %.17g]", tri_pre[7 * b], tri_pre[7 * b + 1],
                                    tri_pre[7 * b + 2], tri_pre[7 * b + 3], tri_pre[7 * b + 4], tri_pre[7 * b + 5],
                                    tri_pre[7 * b + 6], kw, kw >= 0 ? sc.radius_of(tpids[b]) : 0.0);
                    }
                    std::printf("\n");
                }
            }
        }

        mirror.push(p);

        for (int s : secular_triple_idx) {
            auto& dst = g_subsystem_mgr.active[s];
            dst.has_aux_vel = false;
        }
    }

    g_subsys_phase.add(SubsystemPhaseTimes::SEC_TRIPLE, phase_clock.lap());

    for (const auto& t : led_tri_pending) {   // NBODY_ELEDGER: secular triples' residues
        const int ii[3] = {t.i0 - start, t.i1 - start, t.i2 - start};
        double E = 0.0;
        for (int b = 0; b < 3; b++) {
            const int i = ii[b];
            E += 0.5 * h_m(i) * (h_vx(i) * h_vx(i) + h_vy(i) * h_vy(i) + h_vz(i) * h_vz(i));
            for (int c = b + 1; c < 3; c++) {
                const int j = ii[c];
                const double dx = h_x(j) - h_x(i), dy = h_y(j) - h_y(i), dz = h_z(j) - h_z(i);
                E -= G * h_m(i) * h_m(j) / std::sqrt(dx * dx + dy * dy + dz * dz);
            }
        }
        g_led_sec3 += E - t.e0 + t.booked;
        g_led_n_sec3++;
    }
    // --- LogH subsystems: batch integrate with per-subsystem restart (Rantala et al. 2023, §3.19) ---
    if (!logh_idx.empty()) {
        // Most expensive first (longest-processing-time order), so a long integration does not start last
        // while the other host threads idle: cost ~ N^2 x orbits per step
        auto cost = [&](int s) {
            const auto& sub = g_subsystem_mgr.active[s];
            double n_orb = sub.orb.period > 0.0 ? dt / sub.orb.period : 1.0;
            return (double)sub.n_bodies * sub.n_bodies * (1.0 + n_orb);
        };
        std::stable_sort(logh_idx.begin(), logh_idx.end(), [&](int a, int b) { return cost(a) > cost(b); });
        SubsystemManager logh_mgr;
        for (int s : logh_idx) {
            logh_mgr.active.push_back(g_subsystem_mgr.active[s]);
        }
        logh_mgr.rebuild_pid_to_idx(p);

        const int n_sub = (int)logh_mgr.active.size();
        const double eta_restart = g_collisional.eta_restart;
        constexpr int MAX_SUBDIV = 8;  // max 2^8 = 256 sub-steps

        // Save pre-integration state for subsystem members (needed for restart)
        // Reuse host mirrors from top of function (already up-to-date after secular evolution)
        struct SavedBody { double x, y, z, vx, vy, vz, m; };
        std::vector<std::vector<SavedBody>> saved_bodies(n_sub);
        std::vector<Subsystem> saved_subs(logh_mgr.active.begin(), logh_mgr.active.end());
        for (int si = 0; si < n_sub; si++) {
            const auto& sub = logh_mgr.active[si];
            saved_bodies[si].resize(sub.n_bodies);
            for (int b = 0; b < sub.n_bodies; b++) {
                int idx = g_subsystem_mgr.safe_pid_to_idx(sub.particle_ids[b]);
                if (idx < 0) continue;
                saved_bodies[si][b] = {h_x(idx - start), h_y(idx - start), h_z(idx - start),
                                       h_vx(idx - start), h_vy(idx - start), h_vz(idx - start), h_m(idx - start)};
            }
        }

        // Initial batch integration
        g_subsys_phase.count(SubsystemPhaseTimes::C_LOGH, n_sub);
        g_subsys_phase.take_batch_prof();
        g_subsys_phase.add(SubsystemPhaseTimes::LOGH_PREP, phase_clock.lap());
        // The members are in the host mirror (up to date after the secular paths): pack from it, and write the
        // results into it as well as to the device, so no further copy of the range is needed
        HostSource mirror_src;
        mirror_src.start = start;
        mirror_src.count = count;
        mirror_src.x = h_x.data(); mirror_src.y = h_y.data(); mirror_src.z = h_z.data();
        mirror_src.vx = h_vx.data(); mirror_src.vy = h_vy.data(); mirror_src.vz = h_vz.data();
        mirror_src.m = h_m.data();
        g_logh_batch.pack(logh_mgr, p, G, dt, &mirror_src);
        g_subsys_phase.add(SubsystemPhaseTimes::LOGH_PACK, phase_clock.lap());
        logh_integrate_batch_gpu(g_logh_batch, G, g_collisional.eta_gbs);
        g_subsys_phase.add(SubsystemPhaseTimes::LOGH_RUN, phase_clock.lap());

        // Check per-subsystem energy errors
        std::vector<int> failed;
        for (int si = 0; si < n_sub; si++) {
            double dE = g_logh_batch.energy_error[si];
            if (eta_restart > 0 && (dE > eta_restart || std::isnan(dE))) {
                failed.push_back(si);
            }
        }
        const bool any_retry = !failed.empty();

        // SDAR: book the physical energy change of the slow-down (sum of kappa jumps, Wang+2020 Eq. 24) of the
        // accepted integrations; failed ones are booked from the accepted retry below. sd_J[si]: booked jump
        std::vector<double> sd_J(n_sub, 0.0);
        {   // also the jumps of secular inner binaries (CASES.md C20), which do not need SDAR
            std::vector<char> is_failed(n_sub, 0);
            for (int si : failed) is_failed[si] = 1;
            for (int si = 0; si < n_sub; si++)
                if (!is_failed[si]) sd_J[si] = g_logh_batch.sd_jump[si];
            g_subsystem_mgr.step_sdar_calls += g_logh_batch.stat_sd_calls;
            g_subsystem_mgr.step_sdar_kappa_max = std::max(g_subsystem_mgr.step_sdar_kappa_max,
                                                           g_logh_batch.stat_sd_kappa_max);
            g_subsystem_mgr.step_sec_inner += g_logh_batch.stat_sec_inner;
            g_subsystem_mgr.step_irk += g_logh_batch.stat_irk;
            g_subsystem_mgr.step_irk_fb += g_logh_batch.stat_irk_fb;
        }

        // Collisions decided inside the accepted integrations (FROST-CLUSTERS I App. B3); failed ones are taken from
        // the accepted retry below. coll_J[si]: their internal energy change (sum of dE_int)
        std::vector<std::vector<SubsystemCollision>> sub_colls(n_sub);
        std::vector<double> coll_J(n_sub, 0.0);
        {
            std::vector<char> is_failed(n_sub, 0);
            for (int si : failed) is_failed[si] = 1;
            for (int si = 0; si < n_sub; si++)
                if (!is_failed[si]) { sub_colls[si] = g_logh_batch.collisions[si]; coll_J[si] = g_logh_batch.coll_dE[si]; }
            g_subsystem_mgr.step_peri_landings += g_logh_batch.stat_peri_landings;
            g_subsystem_mgr.step_peri_misses += g_logh_batch.stat_peri_misses;
        }

        // Unpack all results (passed get final state; failed get best-effort temporarily)
        g_logh_batch.unpack(logh_mgr, p, &mirror_src);
        g_subsys_phase.add(SubsystemPhaseTimes::LOGH_UNPACK, phase_clock.lap());
        g_subsys_phase.take_batch_prof();

        // Accumulate LogH stats into step-level counters
        g_subsystem_mgr.step_logh_calls += n_sub;
        for (int si = 0; si < n_sub; si++) adv_check.advanced(logh_idx[si], g_logh_batch.t_reached[si], "LogH");
        g_subsystem_mgr.step_logh_gbs_total += g_logh_batch.stat_total_gbs;
        if (g_logh_batch.stat_worst_dE > g_subsystem_mgr.step_logh_worst_dE)
            g_subsystem_mgr.step_logh_worst_dE = g_logh_batch.stat_worst_dE;
        if (g_logh_batch.stat_best_dE < g_subsystem_mgr.step_logh_best_dE)
            g_subsystem_mgr.step_logh_best_dE = g_logh_batch.stat_best_dE;
        g_subsystem_mgr.step_logh_cap_hit += g_logh_batch.stat_cap_hit;

        // Retry failed subsystems with subdivided timestep (Rantala et al. 2023, §3.19)
        if (!failed.empty()) {
            g_subsystem_mgr.step_logh_restarts += (int)failed.size();
            std::printf("  LogH restart: %d/%d subs exceeded eta_restart=%.1e\n",
                        (int)failed.size(), n_sub, eta_restart);

            // A failed attempt with a collision changed its members' masses: those go back to the device too
            // (mirror.push moves x and v only). Without a collision the masses stay as the attempt left them, as
            // before (bitwise the same runs)
            std::vector<char> mass_dirty(n_sub, 0);
            for (int si : failed) mass_dirty[si] = !g_logh_batch.collisions[si].empty();
            auto push_masses = [&]() {
                std::vector<int> idx;
                std::vector<double> m;
                for (int si : failed)
                    for (int b = 0; b < saved_subs[si].n_bodies && mass_dirty[si]; b++) {
                        const int i = g_subsystem_mgr.safe_pid_to_idx(saved_subs[si].particle_ids[b]);
                        if (i >= 0) { idx.push_back(i); m.push_back(saved_bodies[si][b].m); }
                    }
                if (idx.empty()) return;
                DeviceStaging& st = DeviceStaging::get<5>();
                st.reserve((int)idx.size(), 1);
                std::copy(idx.begin(), idx.end(), st.h_idx.data());
                std::copy(m.begin(), m.end(), st.h_buf.data());
                st.upload_idx((int)idx.size());
                st.upload_buf(idx.size());
                auto d_idx = raw(st.d_idx);
                auto d_buf = raw(st.d_buf);
                auto pm = raw(p.mass);
                small_for("restore_masses", (int)idx.size(), KOKKOS_LAMBDA(int k) { pm(d_idx(k)) = d_buf(k); });
            };
            // Helper: restore failed subsystems' particle state from saved copy
            // Refresh host mirrors from device (LogH modified positions), then overwrite failed members
            auto restore_failed = [&]() {
                mirror.pull(p, 6);
                for (int si : failed) {
                    const auto& sub = saved_subs[si];
                    for (int b = 0; b < sub.n_bodies; b++) {
                        int idx = g_subsystem_mgr.safe_pid_to_idx(sub.particle_ids[b]);
                        if (idx < 0) continue;
                        h_x(idx - start)  = saved_bodies[si][b].x;
                        h_y(idx - start)  = saved_bodies[si][b].y;
                        h_z(idx - start)  = saved_bodies[si][b].z;
                        h_vx(idx - start) = saved_bodies[si][b].vx;
                        h_vy(idx - start) = saved_bodies[si][b].vy;
                        h_vz(idx - start) = saved_bodies[si][b].vz;
                        if (mass_dirty[si]) h_m(idx - start) = saved_bodies[si][b].m;
                    }
                    logh_mgr.active[si] = saved_subs[si];
                }
                mirror.push(p);
                push_masses();
                for (int si : failed) mass_dirty[si] = 0;
            };

            restore_failed();

            // Subdivision retry loop: 2, 4, 8, ... sub-steps
            for (int subdiv = 1; subdiv <= MAX_SUBDIV; subdiv++) {
                int n_steps = 1 << subdiv;
                double dt_sub = dt / n_steps;

                SubsystemManager retry_mgr;
                for (int si : failed) retry_mgr.active.push_back(logh_mgr.active[si]);
                retry_mgr.rebuild_pid_to_idx(p);

                bool all_passed = true;
                double worst_dE = 0;
                const bool last_try = (subdiv == MAX_SUBDIV);
                std::vector<double> t_sum(failed.size(), 0.0), j_sum(failed.size(), 0.0), c_sum(failed.size(), 0.0);
                std::vector<std::vector<SubsystemCollision>> c_list(failed.size());

                for (int step = 0; step < n_steps; step++) {
                    SubsystemBatch retry_batch;
                    retry_batch.pack(retry_mgr, p, G, dt_sub);
                    logh_integrate_batch_gpu(retry_batch, G, g_collisional.eta_gbs);

                    for (int r = 0; r < (int)retry_mgr.active.size(); r++) {
                        double dE = retry_batch.energy_error[r];
                        if (dE > eta_restart || std::isnan(dE)) all_passed = false;
                        worst_dE = std::max(worst_dE, std::isnan(dE) ? 1e30 : dE);
                    }

                    retry_batch.unpack(retry_mgr, p);
                    for (size_t r = 0; r < failed.size(); r++) {
                        t_sum[r] += retry_batch.t_reached[r];
                        j_sum[r] += retry_batch.sd_jump[r];
                        c_sum[r] += retry_batch.coll_dE[r];
                        for (const auto& c : retry_batch.collisions[r]) c_list[r].push_back(c);
                        if (!retry_batch.collisions[r].empty()) mass_dirty[failed[r]] = 1;
                    }
                    g_subsystem_mgr.step_peri_landings += retry_batch.stat_peri_landings;
                    g_subsystem_mgr.step_irk += retry_batch.stat_irk;
                    g_subsystem_mgr.step_irk_fb += retry_batch.stat_irk_fb;
                    g_subsystem_mgr.step_peri_misses += retry_batch.stat_peri_misses;

                    // No point continuing if a sub-step failed, except on the last try: there the result stands,
                    // and stopping early would leave the members short of dt (frozen for the rest of the step)
                    if (!all_passed && !last_try) break;
                }
                if (all_passed || last_try)
                    for (size_t r = 0; r < failed.size(); r++) {
                        adv_check.set_time(logh_idx[failed[r]], t_sum[r]);
                        sd_J[failed[r]] = j_sum[r];
                        coll_J[failed[r]] = c_sum[r];
                        sub_colls[failed[r]] = c_list[r];
                    }

                if (all_passed) {
                    std::printf("  LogH restart: OK with %d sub-steps (worst dE=%.2e)\n",
                                n_steps, worst_dE);
                    g_subsystem_mgr.step_logh_restart_ok += (int)failed.size();
                    // Copy final persistent state back to logh_mgr
                    for (size_t r = 0; r < failed.size(); r++)
                        logh_mgr.active[failed[r]] = retry_mgr.active[r];
                    failed.clear();
                    break;
                }

                std::printf("  LogH restart: %d sub-steps failed (worst dE=%.2e)\n",
                            n_steps, worst_dE);

                if (!last_try) {
                    restore_failed();
                } else {   // keep the persistent state (aux velocities, scales) consistent with the members
                    for (size_t r = 0; r < failed.size(); r++) logh_mgr.active[failed[r]] = retry_mgr.active[r];
                }
            }

            if (!failed.empty())
                std::printf("  LogH restart: GAVE UP after %d subdivisions\n", MAX_SUBDIV);
        }

        // Physical energy is the conserved quantity only up to the slow-down: E_now + E_sd = E_0 + integration error
        for (int si = 0; si < n_sub; si++) g_energy_books.E_sd -= sd_J[si];

        // PN-active subsystems: the global diagnostics are Newtonian, so book the change of the members'
        // Newtonian energy, momentum and angular momentum over this advance (Newtonian LogH conserves
        // them; what changed is the PN dynamics, plus GBS error at its tolerance)
        const bool pn_on = g_collisional.pn_order > 0 && g_collisional.c_code > 0.0;
        // SUBSYS_ERR_TOL = x: one SUBSYS_ERR line per subsystem advance whose unbooked energy change (integration
        // error) exceeds x |E0|, with the members' state before the advance (Msun, pc, km/s; scripts/subsys_cases.py
        // reruns it in isolation). PN subsystems book their whole change (E_pn), so they cannot show up in dE/E;
        // the secular paths log the same way (secular binaries incl. Kepler drifts, secular triples)
        const double subsys_log = g_collisional.subsys_err_tol;
        auto log_subsys = [&](const Subsystem& sub, int nb, const char* kind, double d, int si) {
            if (subsys_log <= 0.0 || g_collisional.e_ref <= 0.0 || MPIComm::rank() != 0 ||
                std::fabs(d) <= subsys_log * g_collisional.e_ref) return;
            if (std::strcmp(kind, "unbooked-irk") != 0 && !subsys_err_first(sub.particle_ids)) return;   // Gauss-IRK A/B: every call logged
            std::string msg;
            char buf[96];
            std::snprintf(buf, sizeof(buf), "SUBSYS_ERR step %d %s dE/E0 %.3e type %d n %d dt %.3e a %.4e e %.5f m",
                          g_step_in_progress, kind, d / g_collisional.e_ref, (int)sub.type, nb, dt * TIME_UNIT_MYR,
                          sub.orb.a, sub.orb.e);
            msg = buf;
            for (int b = 0; b < nb; b++) {
                std::snprintf(buf, sizeof(buf), " %.4g", saved_bodies[si][b].m);
                msg += buf;
            }
            msg += " pids";
            for (int b = 0; b < nb; b++) {
                std::snprintf(buf, sizeof(buf), " %d", sub.particle_ids[b]);
                msg += buf;
            }
            msg += " state";
            for (int b = 0; b < nb; b++) {
                const auto& q = saved_bodies[si][b];
                char sb[200];
                auto& sc = StellarPropsCache::get();
                const int kw = sc.type_of(p, sub.particle_ids[b]);
                std::snprintf(sb, sizeof(sb), " [%.17g %.17g %.17g %.17g %.17g %.17g %.17g %d %.17g]", q.m, q.x, q.y, q.z,
                              q.vx, q.vy, q.vz, kw, kw >= 0 ? sc.radius_of(sub.particle_ids[b]) : 0.0);
                msg += sb;
            }
            std::printf("%s\n", msg.c_str());
        };
        if (pn_on || g_eattr.on || subsys_log > 0.0) {
            if (any_retry) mirror.pull(p, 7);   // otherwise the unpack wrote the results (and masses) into the mirror
            for (int si = 0; si < n_sub; si++) {
                const auto& sub = saved_subs[si];
                int nb = sub.n_bodies;
                int idx[LOGH_MAX_BODIES];
                bool pn_active = false, ok = true;
                for (int b = 0; b < nb; b++) {
                    idx[b] = g_subsystem_mgr.safe_pid_to_idx(sub.particle_ids[b]);
                    if (idx[b] < 0) { ok = false; break; }
                    if (g_collisional.pn_body(saved_bodies[si][b].m, StellarPropsCache::get().type_of(p, sub.particle_ids[b])))
                        pn_active = true;
                }
                pn_active = pn_active && pn_on;
                if (!ok || (!pn_active && !g_eattr.on && subsys_log <= 0.0)) continue;
                double E_pre = 0.0, E_post = 0.0;
                for (int b = 0; b < nb; b++) {
                    const auto& s0 = saved_bodies[si][b];
                    int i = idx[b] - start;
                    const double m0 = s0.m, m = h_m(i);   // before / after (a collision in the call changes them)
                    double r0[3] = {s0.x, s0.y, s0.z}, v0[3] = {s0.vx, s0.vy, s0.vz};
                    double r1[3] = {h_x(i), h_y(i), h_z(i)}, v1[3] = {h_vx(i), h_vy(i), h_vz(i)};
                    E_pre  += 0.5 * m0 * (v0[0]*v0[0] + v0[1]*v0[1] + v0[2]*v0[2]);
                    E_post += 0.5 * m * (v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]);
                    if (pn_active) {
                        g_energy_books.sink_PL(m0, r0, v0);
                        g_energy_books.sink_PL(m, r1, v1, -1.0);
                    }
                    for (int c = b + 1; c < nb; c++) {
                        const auto& s1 = saved_bodies[si][c];
                        int j = idx[c] - start;
                        E_pre  -= G * m0 * s1.m / std::sqrt((s0.x-s1.x)*(s0.x-s1.x) + (s0.y-s1.y)*(s0.y-s1.y) + (s0.z-s1.z)*(s0.z-s1.z));
                        if (m > 0.0 && h_m(j) > 0.0)
                            E_post -= G * m * h_m(j) / std::sqrt((h_x(i)-h_x(j))*(h_x(i)-h_x(j)) + (h_y(i)-h_y(j))*(h_y(i)-h_y(j)) +
                                                                 (h_z(i)-h_z(j))*(h_z(i)-h_z(j)));
                    }
                }
                // Collisions in the call: their energy change coll_J goes to E_merge (finalize_subsystem_collision),
                // their P and L to the P / L books just made here (PN), so the collision books leave them out
                if (pn_active) {
                    g_energy_books.E_pn += E_pre - E_post + coll_J[si] + sd_J[si];   // slow-down jumps: E_sd
                    for (auto& c : sub_colls[si]) c.pl_booked = true;
                } else {
                    const double d = E_post - E_pre - sd_J[si] - coll_J[si];   // unbooked part
                    // Gauss-IRK integrations (CASES.md C21) logged as unbooked-irk (a retried one: as its retry left it)
                    const bool irk = !any_retry && si < (int)g_logh_batch.irk_used.size() && g_logh_batch.irk_used[si];
                    log_subsys(sub, nb, irk ? "unbooked-irk" : "unbooked", d, si);
                    g_eattr.logh += d; g_eattr.logh_abs += std::fabs(d); g_eattr.n_logh++;
                    if (std::fabs(d) > std::fabs(g_eattr.worst)) {
                        g_eattr.worst = d; g_eattr.worst_n = nb; g_eattr.worst_type = (int)sub.type;
                        g_eattr.worst_pid0 = sub.particle_ids[0]; g_eattr.worst_pid1 = sub.particle_ids[1];
                        g_eattr.worst_a = sub.orb.a; g_eattr.worst_e = sub.orb.e; g_eattr.worst_dt = dt;
                        for (int b = 0; b < 4; b++) g_eattr.worst_m[b] = b < nb ? saved_bodies[si][b].m : 0.0;
                    }
                }
            }
        }

        for (int si = 0; si < n_sub; si++)
            for (const auto& c : sub_colls[si]) g_pending_collisions.push_back(c);

        // Propagate persistent state back from logh_mgr to g_subsystem_mgr
        // (B is always recomputed fresh — only aux_vel and scales need persistence)
        for (int i = 0; i < n_sub; i++) {
            auto& dst = g_subsystem_mgr.active[logh_idx[i]];
            const auto& src = logh_mgr.active[i];
            dst.has_aux_vel = src.has_aux_vel;
            if (src.has_aux_vel) {
                std::memcpy(dst.aux_vel_phys, src.aux_vel_phys,
                            LOGH_MAX_BODIES * 3 * sizeof(double));
            }
            dst.V_scale = src.V_scale;
            dst.R_scale = src.R_scale;
            dst.T_scale = src.T_scale;
            dst.E_scale = src.E_scale;
            dst.scales_set = src.scales_set;
        }
        g_subsys_phase.add(SubsystemPhaseTimes::LOGH_RETRY, phase_clock.lap());
        g_logh_prof = LoghBatchProf();   // retry batches are inside logh_post
    }
    adv_check.verify(start, count);

    // Collisions decided inside the subsystem integrations: books, remnant physics, placeholders
    std::vector<SubsystemCollision> colls = std::move(g_pending_collisions);
    g_pending_collisions.clear();
    if (!colls.empty()) {
        g_subsystem_mgr.rebuild_pid_to_idx(p);
        for (const auto& c : colls)
            finalize_subsystem_collision(c, p, g_subsystem_mgr, g_energy_books, G, g_collisional.c_code);
        g_subsystem_mgr.step_subsys_collisions += (int)colls.size();
        g_subsystem_mgr.update_particle_flags(p);
    }

    // INTEGER_POSITIONS: the drift takes positions from the integer coordinates, but the subsystem integrators
    // (LogH / SDAR, Kepler, secular) wrote the double positions of the members they advanced. Their integer
    // coordinates follow those positions here; otherwise a member's next field drift restarted from the position it
    // had before the advance (200k, 10 % binaries: |dE/E| 2e-7 against 1.4e-11 with double positions)
    if (use_integer_positions) {
        g_subsystem_mgr.rebuild_pid_to_idx(p);
        std::vector<int> idx;
        for (int s : eligible) {
            if (s >= (int)g_subsystem_mgr.active.size()) continue;
            for (int pid : g_subsystem_mgr.active[s].particle_ids) {
                const int i = g_subsystem_mgr.safe_pid_to_idx(pid);
                if (i >= 0) idx.push_back(i);
            }
        }
        if (!idx.empty()) {
            DeviceStaging& st = DeviceStaging::get<6>();
            st.reserve((int)idx.size(), 1);
            std::copy(idx.begin(), idx.end(), st.h_idx.data());
            st.upload_idx((int)idx.size());
            auto d_idx = raw(st.d_idx);
            auto px = raw(p.x), py = raw(p.y), pz = raw(p.z);
            auto pix = raw(p.ix), piy = raw(p.iy), piz = raw(p.iz);
            const double inv = int_pos_inv_scale;
            xfor("int_pos_sync", (int)idx.size(), KOKKOS_LAMBDA(int k) {
                const int i = d_idx(k);
                pix(i) = llrint(px(i) * inv);
                piy(i) = llrint(py(i) * inv);
                piz(i) = llrint(pz(i) * inv);
            });
        }
    }
}
#endif

void fsi_step(Particles& p, int start, int count,
              double G, double soft2, double dt, bool forces_valid,
              bool force_direct_n2) {
    g_subset_hist.record(SubsetHistogram::FSI_STEP, count);
    bool tree_mode = !force_direct_n2 && (use_tree && count >= tree_threshold);
    bool tree_built = false;
#ifdef ENABLE_COLLISIONAL
    cm_tide::KickSrcScope cmt_src(start, count, dt, G);   // self kicks: the set is its own source
#endif

    // U kick (dt/6)
    if (forces_valid) {
        g_profiler.start();
        force_err_inject(p, start, count, count);
        kick_velocities(p, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else if (tree_mode) {
        g_profiler.start();
        g_octree.n_leaf = tree_n_leaf;
        g_octree.grad_dipole = tree_grad_dipole;
        g_octree.n_group = tree_n_group;
        g_octree.build(p, start, count, tree_theta);
        tree_self_acceleration(p, g_octree, start, count, G, soft2);
        tree_momentum_correction(p, start, count);
        tree_built = true;
        g_profiler.stop(StepProfiler::SELF_FORCE);
        g_profiler.start();
        force_err_inject(p, start, count, count);
        kick_velocities(p, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else {
        g_profiler.start();
        self_acceleration(p, start, count, G, soft2);
        g_profiler.stop(StepProfiler::SELF_FORCE, (long long)count * count);
        g_profiler.start();
        force_err_inject(p, start, count, count);
        kick_velocities(p, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // H_S (dt/2) then T' drift (dt/2) — Rantala et al. 2023, Algorithm 1 ordering
    g_profiler.start();
#ifdef ENABLE_COLLISIONAL
    if (p.collisional && !g_subsystem_mgr.active.empty()) {
        advance_subsystems_in_range(p, start, count, G, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::SUBSYSTEM_INTEG);
        g_profiler.start();
        drift_field_only(p, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else
#endif
    {
        drift(p, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // U-tilde kick (2dt/3)
    if (tree_mode) {
        g_profiler.start();
        if (!tree_built) {
            g_octree.n_leaf = tree_n_leaf;
            g_octree.grad_dipole = tree_grad_dipole;
            g_octree.n_group = tree_n_group;
            g_octree.build(p, start, count, tree_theta);
            tree_built = true;
        } else {
            g_octree.refresh_positions(p);
        }
        tree_self_acceleration(p, g_octree, start, count, G, soft2);
        tree_momentum_correction(p, start, count);
        g_profiler.stop(StepProfiler::SELF_FORCE);
        g_profiler.start();
        g_octree.compute_cell_accelerations(p);   // FSI gradient term, as in fsi_step_distributed
        tree_self_gradient_correction(p, g_octree, G, soft2, dt);
        tree_momentum_correction(p, start, count);
        g_profiler.stop(StepProfiler::SELF_GRAD);
        g_profiler.start();
        force_err_inject(p, start, count, count);
        kick_velocities(p, start, count, dt * FSI_MOD_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else {
        g_profiler.start();
        self_acceleration(p, start, count, G, soft2);
        g_profiler.stop(StepProfiler::SELF_FORCE, (long long)count * count);
        g_profiler.start();
        self_gradient_correction(p, start, count, G, soft2, dt);
        g_profiler.stop(StepProfiler::SELF_GRAD, (long long)count * count);
        g_profiler.start();
        force_err_inject(p, start, count, count);
        kick_velocities(p, start, count, dt * FSI_MOD_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // H_S (dt/2) then T' drift (dt/2) — Rantala et al. 2023, Algorithm 1 ordering
    g_profiler.start();
#ifdef ENABLE_COLLISIONAL
    if (p.collisional && !g_subsystem_mgr.active.empty()) {
        advance_subsystems_in_range(p, start, count, G, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::SUBSYSTEM_INTEG);
        g_profiler.start();
        drift_field_only(p, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else
#endif
    {
        drift(p, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // U kick (dt/6)
    if (tree_mode) {
        g_profiler.start();
        g_octree.refresh_positions(p);
        tree_self_acceleration(p, g_octree, start, count, G, soft2);
        tree_momentum_correction(p, start, count);
        g_profiler.stop(StepProfiler::SELF_FORCE);
        g_profiler.start();
        force_err_inject(p, start, count, count);
        kick_velocities(p, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else {
        g_profiler.start();
        self_acceleration(p, start, count, G, soft2);
        g_profiler.stop(StepProfiler::SELF_FORCE, (long long)count * count);
        g_profiler.start();
        force_err_inject(p, start, count, count);
        kick_velocities(p, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }
}

// ==========================================================================
// Dynamic pivot-based partitioning for HHS (FROST Algorithm 3)
// Partitions particles [start, start+count) into:
//   - slow: dt_adapt >= pivot (at beginning)
//   - fast: dt_adapt < pivot (at end)
// Returns the count of slow particles
//
// IMPORTANT: This must be DETERMINISTIC across MPI ranks! We use prefix sums
// instead of atomics to ensure all ranks produce identical permutations.
// ==========================================================================
#ifdef ENABLE_COLLISIONAL
// NBODY_SLOW0_STATS (diagnostics, AUDIT G row 1a): sets of <= 4 particles in which every particle is fast, so the
// recursion halves the pivot without a slow set. Per (size, distinct subsystems, singles): how often, and in how many
// of them every particle (pert_all) or at least one (pert_any) had a step >= the pivot before subsystem_timesteps
// (shared + perturbation step), i.e. the subsystem step drove the set down. Printed by rank 0 at the end of the run.
namespace {
struct Slow0Stats {
    bool on = std::getenv("NBODY_SLOW0_STATS") != nullptr;
    Kokkos::View<double*> dt_pre{"slow0_dt_pre", 0};   // by particle id: dt_adapt before subsystem_timesteps
    struct Row { long long calls = 0, pert_all = 0, pert_any = 0; };
    std::map<std::tuple<int, int, int>, Row> rows;
    // Pairs of two singles: bound or not, semi-major axis bins (AU), distinct pairs (by pid)
    long long pair_unbound = 0;
    std::map<int, long long> pair_abin;   // upper edge in AU -> calls
    std::set<std::pair<int, int>> pair_ids;
    std::map<int, long long> pair_rbin;   // separation / r_ngb bin -> calls
    void print() const {
        if (!on || MPIComm::rank() != 0) return;
        std::printf("SLOW0_STATS: size subsystems singles | calls pert_all pert_any%s\n", rows.empty() ? " (none)" : "");
        for (const auto& kv : rows)
            std::printf("SLOW0_STATS: %d %d %d | %lld %lld %lld\n", std::get<0>(kv.first), std::get<1>(kv.first),
                        std::get<2>(kv.first), kv.second.calls, kv.second.pert_all, kv.second.pert_any);
        std::printf("SLOW0_PAIRS: two-single sets: distinct pairs %zu, unbound calls %lld\n", pair_ids.size(), pair_unbound);
        for (const auto& kv : pair_abin) std::printf("SLOW0_PAIRS: bound a < %d AU: %lld calls\n", kv.first, kv.second);
        for (const auto& kv : pair_rbin) std::printf("SLOW0_PAIRS: separation < %d r_ngb: %lld calls\n", kv.first, kv.second);
        std::fflush(stdout);
    }
    void record_pre(const Particles& p, int start, int count) {
        if (!on || count <= 0) return;
        auto pid = p.particle_id;
        int maxpid = 0;
        Kokkos::parallel_reduce("slow0_maxpid", count, KOKKOS_LAMBDA(int ii, int& m) {
            if (pid(start + ii) > m) m = pid(start + ii);
        }, Kokkos::Max<int>(maxpid));
        if ((int)dt_pre.extent(0) <= maxpid) Kokkos::resize(dt_pre, 2 * (maxpid + 1));
        auto pre = dt_pre;
        auto dt = p.dt_adapt;
        Kokkos::parallel_for("slow0_pre", count, KOKKOS_LAMBDA(int ii) { pre(pid(start + ii)) = dt(start + ii); });
    }
    void record_set(const Particles& p, int start, int count, double pivot, double G) {
        if (!on || count > 4) return;
        auto r = Kokkos::make_pair(start, start + count);
        auto hp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.particle_id, r));
        auto hs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.subsystem_id, r));
        std::set<int> subs;
        int singles = 0, slow_pre = 0;
        for (int k = 0; k < count; k++) {
            if (hs(k) >= 0) subs.insert(hs(k)); else singles++;
            if (hp(k) < (int)dt_pre.extent(0)) {
                double v = 0;
                Kokkos::deep_copy(v, Kokkos::subview(dt_pre, hp(k)));
                if (v >= pivot) slow_pre++;
            }
        }
        if (count == 2 && singles == 2) {
            auto hx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.x, r));
            auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.y, r));
            auto hz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.z, r));
            auto hvx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.vx, r));
            auto hvy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.vy, r));
            auto hvz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.vz, r));
            auto hm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.mass, r));
            const double dx = hx(1) - hx(0), dy = hy(1) - hy(0), dz = hz(1) - hz(0);
            const double dvx = hvx(1) - hvx(0), dvy = hvy(1) - hvy(0), dvz = hvz(1) - hvz(0);
            const double rr = std::sqrt(dx * dx + dy * dy + dz * dz), v2 = dvx * dvx + dvy * dvy + dvz * dvz;
            const double mu = G * (hm(0) + hm(1));
            const double beta = 2.0 * mu / rr - v2;   // 1 / a * mu
            pair_ids.insert({std::min(hp(0), hp(1)), std::max(hp(0), hp(1))});
            const int rb[] = {1, 2, 5, 10, 100, 1000000};
            for (int e : rb) if (rr < e * g_collisional.r_ngb) { pair_rbin[e]++; break; }
            if (!(beta > 0.0)) pair_unbound++;
            else {
                const double a_au = mu / beta / AU_TO_PC;
                const int ab[] = {206, 500, 1000, 2000, 5000, 1000000000};
                for (int e : ab) if (a_au < e) { pair_abin[e]++; break; }
            }
        }
        Row& row = rows[std::make_tuple(count, (int)subs.size(), singles)];
        row.calls++;
        if (slow_pre == count) row.pert_all++;
        if (slow_pre > 0) row.pert_any++;
    }
};
Slow0Stats& slow0() {
    static Slow0Stats* s = nullptr;
    if (!s) {
        s = new Slow0Stats();
        Kokkos::push_finalize_hook([] { s->print(); delete s; s = nullptr; });
    }
    return *s;
}
}  // namespace
#endif

static int partition_by_pivot(Particles& p, int start, int count, double pivot) {
    if (count <= 1) return count;

    RawView dt_f{p.dt_adapt.data()};
    auto perm = raw(p.scratch_i);
    auto is_slow = raw(p.scratch_i2);  // temporary: 1 if slow, 0 if fast

    // Fused mark + exclusive prefix sum in single kernel
    // is_slow(ii) = 1 if slow, 0 if fast
    // slow_prefix(ii) = number of slow particles with index < ii
    auto slow_prefix = raw(p.scratch_i3);
    // The scan's total is the slow count: returned by the scan itself (one copy and stream sync instead of two
    // single-element copies, each with its own sync; integer sums, identical result)
    int slow_count = 0;
    xscan("mark_prefix_slow", count,
        KOKKOS_LAMBDA(int ii, int& partial_sum, bool is_final) {
            int val = (dt_f(start + ii) >= pivot) ? 1 : 0;
            if (is_final) {
                is_slow(ii) = val;
                slow_prefix(ii) = partial_sum;
            }
            partial_sum += val;
        }, slow_count);
    lean_fence("integrator.cpp:1514");

    if (slow_count == 0 || slow_count == count) {
        return slow_count;
    }

    // Step 3: Build permutation deterministically
    // - Slow particle at position ii goes to slow_prefix(ii)
    // - Fast particle at position ii goes to slow_count + (ii - slow_prefix(ii) - (1-is_slow(ii)))
    //   = slow_count + (ii - slow_prefix(ii)) for fast particles
    int sc = slow_count;
    xfor("build_perm_deterministic", count, KOKKOS_LAMBDA(int ii) {
        int dest;
        if (is_slow(ii)) {
            dest = slow_prefix(ii);
        } else {
            // Fast particles: position among fast = ii - slow_prefix(ii)
            // (since slow_prefix(ii) counts slow particles before ii)
            dest = sc + (ii - slow_prefix(ii));
        }
        perm(dest) = start + ii;
    });

    p.apply_permutation(start, count);

    return slow_count;
}

// Forward declaration: unified HHS-FSI recursion (defined after all helper functions).
// Returns the number of particles that mergers removed from [start, start+count).
static int hhs_fsi_recursive(Particles& p, DistributedParticles* dp,
    int start, int count, double pivot_dt,
    double G, double soft2, double dt_min, double eta,
    bool dt_valid, bool global_tree_available = false, int depth = 0);

// After compact_remove_zero_mass: the cached pid map points at old slots, and in MPI mode the
// per-rank slices still cover the old particle count.
static void sync_after_removal(Particles& p, DistributedParticles* dp) {
#ifdef ENABLE_COLLISIONAL
    if (p.collisional) g_subsystem_mgr.rebuild_pid_to_idx(p);
#endif
    if (dp) dp->sync_count();
}

#ifdef ENABLE_COLLISIONAL
int compact_zero_mass_range(Particles& p, int start, int count) {
    const size_t log_from = p.removed_pid_log.size();
    const int k = p.compact_remove_zero_mass(start, start + count);
    if (k == 0) return 0;
    // Placeholders of bodies merged inside subsystem integrations become removed ids; their subsystems keep the
    // remaining members (strip_removed)
    for (size_t q = log_from; q < p.removed_pid_log.size(); q++) {
        const int pid = p.removed_pid_log[q];
        g_subsystem_mgr.ghost_pids.erase(pid);
        g_subsystem_mgr.removed_pids.insert(pid);
    }
    g_subsystem_mgr.strip_removed();
    return k;
}

// Massless particles in [start, start + count)? (one reduction; only while placeholders exist)
static bool range_has_ghost(const Particles& p, int start, int count) {
    if (g_subsystem_mgr.ghost_pids.empty() || count <= 0) return false;
    auto m = raw(p.mass);
    int n = 0;
    xreduce("ghost_count", start, start + count,
                            KOKKOS_LAMBDA(int i, int& c) { if (m(i) <= 0.0) c++; }, n);
    return n > 0;
}
#endif

#ifdef ENABLE_COLLISIONAL
// Work counters of the step for the profiler (the subsystem manager's per-step statistics)
static void profile_subsystem_counters() {
    const auto& m = g_subsystem_mgr;
    g_profiler.count(StepProfiler::C_SUBSYS_ACTIVE, (double)m.active.size());
    g_profiler.count(StepProfiler::C_LOGH_SUBSYS, m.step_logh_calls);
    g_profiler.count(StepProfiler::C_GBS_STEPS, m.step_logh_gbs_total);
    g_profiler.count(StepProfiler::C_SDAR_CALLS, m.step_sdar_calls);
    g_profiler.count(StepProfiler::C_SECULAR, m.step_secular_evolved);
    g_profiler.count(StepProfiler::C_SEC_SUBSTEPS, (double)m.step_secular_substeps);
    g_profiler.count(StepProfiler::C_KEPLER2, m.step_kepler2);
    g_profiler.count(StepProfiler::C_SEC_TRIPLE, m.step_secular_triples);
    g_profiler.count(StepProfiler::C_MERGERS, m.step_mergers);
}
#endif

void hhs_fsi_step_dynamic(Particles& p, int start, int count, double pivot_dt,
                          double G, double soft2, double dt_min,
                          double eta) {
#ifdef ENABLE_COLLISIONAL
    if (p.collisional) g_subsystem_mgr.reset_step_diagnostics();
    // Global PN1.0 half-kicks around the FSI step (Rantala et al. 2023, Section 3.7), as in hhs_fsi_step_distributed
    bool do_global_pn = p.collisional && (g_collisional.pn_order & 1) && g_collisional.c_code > 0.0;
    if (do_global_pn) {
        StepProfiler::Region r_(g_profiler, StepProfiler::GLOBAL_PN);
        global_pn1_kick(p, start, count, G, g_collisional.c_code, g_collisional.m_pn, pivot_dt * 0.5);
    }
#endif
    if (hhs_cc) g_hhs_counters.reset();
    int n_removed = hhs_fsi_recursive(p, nullptr, start, count, pivot_dt, G, soft2, dt_min, eta, false);
#ifdef ENABLE_COLLISIONAL
    // Placeholders still left (merged in a range not entered again this step) go before the diagnostics
    if (p.collisional && range_has_ghost(p, start, count - n_removed)) {
        StepProfiler::Region r_(g_profiler, StepProfiler::GHOST_COMPACT);
        const int k = compact_zero_mass_range(p, start, count - n_removed);
        if (k > 0) { sync_after_removal(p, nullptr); g_subsystem_mgr.update_particle_flags(p); n_removed += k; }
    }
#endif
#ifdef ENABLE_COLLISIONAL
    if (p.collisional) profile_subsystem_counters();
#endif
    StepProfiler::Region r_print_(g_profiler, StepProfiler::STEP_PRINT);   // to the end: output, and global PN
    if (hhs_cc) {   // component statistics per depth
        std::printf("  HHS_CC components:");
        for (int d = 0; d < HHSCounters::MAX_DEPTH; d++)
            if (g_hhs_counters.cc_calls[d] > 0)
                std::printf(" d%d %lld x %.1f (largest %.0f)", d, g_hhs_counters.cc_calls[d],
                            (double)g_hhs_counters.cc_comps[d] / g_hhs_counters.cc_calls[d],
                            (double)g_hhs_counters.cc_largest[d] / g_hhs_counters.cc_calls[d]);
        std::printf("\n");
    }
#ifdef ENABLE_COLLISIONAL
    if (do_global_pn) {
        StepProfiler::Region r_(g_profiler, StepProfiler::GLOBAL_PN);
        global_pn1_kick(p, start, count - n_removed, G, g_collisional.c_code, g_collisional.m_pn, pivot_dt * 0.5);
    }
    // Per-step subsystem summary, as hhs_fsi_step_distributed prints it (tests read its counters)
    if (p.collisional) g_subsystem_mgr.print_step_summary();
    if (p.collisional && g_subsys_phase.any()) g_subsys_phase.report_and_reset();
#else
    (void)n_removed;
#endif
}

// =============================================================================
// Distributed (Multi-GPU) HHS Integration
// =============================================================================
// NEW DESIGN: Scatter-compute-gather paradigm (matching FROST paper)
//
// All ranks maintain replicated particle data. The HHS recursion runs
// IDENTICALLY on all ranks (partitioning, drifts, kicks). Only force
// computation is distributed:
//   1. Each rank computes forces for its slice of particles
//   2. Allgather accelerations after each force computation
//
// This achieves O(N) communication per step regardless of HHS depth.
// =============================================================================

// Kick velocities using pre-computed accelerations
void kick_velocities(Particles& p, int start, int count, double dt) {
    RawView pvx{p.vx.data()}, pvy{p.vy.data()}, pvz{p.vz.data()};
    RawView pax{p.ax.data()}, pay{p.ay.data()}, paz{p.az.data()};
    RawView cvx{p.cvx.data()}, cvy{p.cvy.data()}, cvz{p.cvz.data()};
#ifdef ENABLE_COLLISIONAL
    // NBODY_CM_TIDE (cm_tide.h, CASES.md C24): unresolved binaries get c.o.m. kicks and the orbit-averaged tide
    const bool cmt = !use_integer_positions && cm_tide::enabled() && cm_tide::pre(p, start, count, dt);
#endif

    // Compensated: a plain v += a dt rounds at ~eps|v| per kick and sets the energy floor
    small_for("kick", count, KOKKOS_LAMBDA(int ii) {
        int i = start + ii;
        kahan_add(pvx(i), cvx(i), pax(i) * dt);
        kahan_add(pvy(i), cvy(i), pay(i) * dt);
        kahan_add(pvz(i), cvz(i), paz(i) * dt);
    });
#ifdef ENABLE_COLLISIONAL
    if (cmt) cm_tide::post(p, dt);
#endif
}

// =============================================================================
// Distributed force computation: scatter-compute-gather with tiled GPU kernels
// Each rank computes forces for particles in its slice FROM ALL particles
// in the working set [start, start+count). Then allgather accelerations.
// Uses shared memory tiling per FROST paper.
// =============================================================================
static void self_acceleration_distributed(DistributedParticles& dp, int start, int count,
                                          double G, double soft2) {
    if (count <= 0) return;


    int slice_start, slice_count;
    dp.get_force_slice(start, count, slice_start, slice_count);
    n2_detail::SinkSetScope n2_set_1(count);   // chunking of the whole set (rank-count independent)
    if (slice_count == 0) {
        dp.allgather_accelerations_subset(start, count);
        return;
    }

    acceleration_range(dp.local, slice_start, slice_count, start, count, G, soft2);
    Kokkos::fence("integrator.cpp:1678");

    if (use_nonblocking_mpi) {
        dp.iallgather_accelerations_subset(start, count, 0);
        dp.wait_and_unpack_accelerations(start, count, 0);
    } else {
        dp.allgather_accelerations_subset(start, count);
    }
}

// =============================================================================
// Distributed gradient correction: computes gradient correction for my slice,
// applies it to accelerations, then allgather corrected accelerations.
// =============================================================================
static void self_gradient_correction_distributed(DistributedParticles& dp, int start, int count,
                                                  double G, double soft2, double grad_dt) {
    if (count <= 0) return;

    // Snapshot pre-correction accelerations for the full range.
    // The gradient kernel reads j-particle accelerations while writing corrected
    // i-particle accelerations to the same array — without a snapshot, j-particles
    // in the local MPI slice can read partially-corrected values (data race),
    // breaking the antisymmetry corr_ij = -corr_ji and violating momentum conservation.
    static Kokkos::View<double*> ax_save("ax_save_grad", 0);
    static Kokkos::View<double*> ay_save("ay_save_grad", 0);
    static Kokkos::View<double*> az_save("az_save_grad", 0);
    static bool released_at_finalize = (Kokkos::push_finalize_hook([] {
        ax_save = {}; ay_save = {}; az_save = {};
    }), true);
    (void)released_at_finalize;
    if ((int)ax_save.extent(0) < count) {
        Kokkos::realloc(ax_save, count);
        Kokkos::realloc(ay_save, count);
        Kokkos::realloc(az_save, count);
    }
    dcopy(Kokkos::subview(ax_save, Kokkos::make_pair(0, count)),
                      Kokkos::subview(dp.local.ax, Kokkos::make_pair(start, start + count)));
    dcopy(Kokkos::subview(ay_save, Kokkos::make_pair(0, count)),
                      Kokkos::subview(dp.local.ay, Kokkos::make_pair(start, start + count)));
    dcopy(Kokkos::subview(az_save, Kokkos::make_pair(0, count)),
                      Kokkos::subview(dp.local.az, Kokkos::make_pair(start, start + count)));

    int slice_start, slice_count;
    dp.get_force_slice(start, count, slice_start, slice_count);
    n2_detail::SinkSetScope n2_set_2(count);   // chunking of the whole set (rank-count independent)

    if (slice_count > 0) {
        GradSnapshot snap{ax_save.data(), ay_save.data(), az_save.data(), start};
        // grad_mag in the kernel for the slice, as on one rank (it was |a_after - a_before| / dt^2 from the gathered
        // accelerations: other rounding, so the gradient time step depended on the rank count)
        gradient_correction_range(dp.local, slice_start, slice_count, start, count, snap, snap, G, soft2, grad_dt,
                                  dp.local.collisional);
        Kokkos::fence();
    }

    // Allgather corrected accelerations
    if (use_nonblocking_mpi) {
        dp.iallgather_accelerations_subset(start, count, 0);
        dp.wait_and_unpack_accelerations(start, count, 0);
    } else {
        dp.allgather_accelerations_subset(start, count);
    }

#ifdef ENABLE_COLLISIONAL
    // grad_mag of the other ranks' slices
    if (dp.local.collisional) {
        const int f[1] = {4};
        dp.allgather_fields(&start, &count, 1, f, 1);
    }
#endif
}


// =============================================================================
// Tree MPI distribution helpers: divide tree groups across MPI ranks
// Each rank computes forces for its subset of groups, then allreduce(SUM)
// =============================================================================
static void get_tree_group_range(const Octree& tree, int& g_offset, int& g_count) {
    int rank = MPIComm::rank();
    int size = MPIComm::size();
    int total_groups = tree.n_groups;
    g_offset = rank * total_groups / size;
    int g_end = (rank + 1) * total_groups / size;
    g_count = g_end - g_offset;
}

static void zero_accelerations(Particles& p, int start, int count) {
    auto ax = raw(p.ax); auto ay = raw(p.ay); auto az = raw(p.az);
    small_for("zero_acc", count, KOKKOS_LAMBDA(int ii) {
        int i = start + ii;
        ax(i) = 0.0; ay(i) = 0.0; az(i) = 0.0;
    });
}

static void init_timesteps_max(Particles& p, int start, int count) {
    auto dt_f = raw(p.dt_adapt);
    small_for("init_dt_max", count, KOKKOS_LAMBDA(int ii) {
        dt_f(start + ii) = 1e30;
    });
}

// =============================================================================
// Distributed FSI step using scatter-compute-gather
// All ranks execute identical code; only force computation is distributed
// Uses fused force+kick kernels to reduce kernel launches and memory traffic
//
// tree_prebuilt: if true, g_octree is already built for [start, count) —
//   skip first tree build (reuse from timestep computation or previous call)
// global_tree_subset: -1 = build own tree (default), 0 or 1 = use global tree
//   with subset monopoles (0=slow, 1=fast). No MPI allreduce needed.
// =============================================================================
static void fsi_step_distributed(DistributedParticles& dp, int start, int count,
                                 double G, double soft2, double dt,
                                 bool forces_valid = false,
                                 bool tree_prebuilt = false,
                                 int global_tree_subset = -1) {
    if (count <= 0) return;

    // Global tree mode: use subset monopoles from the once-built global tree
    if (global_tree_subset >= 0) {
        long long self_pairs = (long long)count * count;

        // Step 1: force + kick (dt/6)
        if (forces_valid) {
            g_profiler.start();
            force_err_inject(dp.local, start, count, count);
            kick_velocities(dp.local, start, count, dt * FSI_KICK);
            g_profiler.stop(StepProfiler::DRIFT_KICK);
        } else {
            g_profiler.start();
            tree_global_self_acceleration(dp.local, g_octree, global_tree_subset, G, soft2);
            tree_momentum_correction(dp.local, start, count);
            g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
            g_profiler.start();
            force_err_inject(dp.local, start, count, count);
            kick_velocities(dp.local, start, count, dt * FSI_KICK);
            g_profiler.stop(StepProfiler::DRIFT_KICK);
        }

        // H_S (dt/2) then T' drift (dt/2) — Rantala et al. 2023, Algorithm 1 ordering
        g_profiler.start();
#ifdef ENABLE_COLLISIONAL
        if (dp.local.collisional && !g_subsystem_mgr.active.empty()) {
            advance_subsystems_in_range(dp.local, start, count, G, dt * FSI_DRIFT);
            g_profiler.stop(StepProfiler::SUBSYSTEM_INTEG);
            g_profiler.start();
            drift_field_only(dp.local, start, count, dt * FSI_DRIFT);
            g_profiler.stop(StepProfiler::DRIFT_KICK);
        } else
#endif
        {
            drift(dp.local, start, count, dt * FSI_DRIFT);
            g_profiler.stop(StepProfiler::DRIFT_KICK);
        }

        // Refresh positions in global tree after drift
        g_profiler.start();
        g_octree.refresh_positions(dp.local);
        g_profiler.stop(StepProfiler::TREE_MAINT);

        // Step 3: force + gradient + kick (2dt/3)
        g_profiler.start();
        tree_global_self_acceleration(dp.local, g_octree, global_tree_subset, G, soft2);
        tree_momentum_correction(dp.local, start, count);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        g_profiler.start();
        g_octree.compute_dual_cell_accelerations(dp.local);
        tree_global_self_gradient_correction(dp.local, g_octree, global_tree_subset, G, soft2, dt);
        tree_momentum_correction(dp.local, start, count);
        g_profiler.stop(StepProfiler::SELF_GRAD, self_pairs);
        g_profiler.start();
        force_err_inject(dp.local, start, count, count);
        kick_velocities(dp.local, start, count, dt * FSI_MOD_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);

        // H_S (dt/2) then T' drift (dt/2) — Rantala et al. 2023, Algorithm 1 ordering
        g_profiler.start();
#ifdef ENABLE_COLLISIONAL
        if (dp.local.collisional && !g_subsystem_mgr.active.empty()) {
            advance_subsystems_in_range(dp.local, start, count, G, dt * FSI_DRIFT);
            g_profiler.stop(StepProfiler::SUBSYSTEM_INTEG);
            g_profiler.start();
            drift_field_only(dp.local, start, count, dt * FSI_DRIFT);
            g_profiler.stop(StepProfiler::DRIFT_KICK);
        } else
#endif
        {
            drift(dp.local, start, count, dt * FSI_DRIFT);
            g_profiler.stop(StepProfiler::DRIFT_KICK);
        }

        // Refresh positions in global tree after drift
        g_profiler.start();
        g_octree.refresh_positions(dp.local);
        g_profiler.stop(StepProfiler::TREE_MAINT);

        // Step 5: force + kick (dt/6)
        g_profiler.start();
        tree_global_self_acceleration(dp.local, g_octree, global_tree_subset, G, soft2);
        tree_momentum_correction(dp.local, start, count);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        g_profiler.start();
        force_err_inject(dp.local, start, count, count);
        kick_velocities(dp.local, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);

        return;
    }

    bool tree_mode = use_tree && count >= tree_threshold;
    long long self_pairs = (long long)count * count;

    // Sync packed positions for vectorized kernel loads
    g_profiler.start();
    dp.local.pack_pos_range(start, count);
    g_profiler.stop(StepProfiler::PACK_DATA);

    // U kick (dt/6) - force then kick (separate, matching FROST)
    if (forces_valid) {
        g_profiler.start();
        force_err_inject(dp.local, start, count, count);
        kick_velocities(dp.local, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else if (tree_mode) {
        g_profiler.start();
        if (!tree_prebuilt) {
            g_octree.n_leaf = tree_n_leaf;
            g_octree.grad_dipole = tree_grad_dipole;
            g_octree.n_group = tree_n_group;
            g_octree.build(dp.local, start, count, tree_theta);
            tree_prebuilt = true;
        }
        int my_g_off, my_g_cnt;
        get_tree_group_range(g_octree, my_g_off, my_g_cnt);
        zero_accelerations(dp.local, start, count);
        tree_self_acceleration(dp.local, g_octree, start, count, G, soft2, my_g_off, my_g_cnt);
        dp.allreduce_accelerations_subset(start, count);
        tree_momentum_correction(dp.local, start, count);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        g_profiler.start();
        force_err_inject(dp.local, start, count, count);
        kick_velocities(dp.local, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else {
        g_profiler.start();
        self_acceleration_distributed(dp, start, count, G, soft2);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        g_profiler.start();
        force_err_inject(dp.local, start, count, count);
        kick_velocities(dp.local, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // H_S (dt/2) then T' drift (dt/2) — Rantala et al. 2023, Algorithm 1 ordering
    g_profiler.start();
#ifdef ENABLE_COLLISIONAL
    if (dp.local.collisional && !g_subsystem_mgr.active.empty()) {
        advance_subsystems_in_range(dp.local, start, count, G, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::SUBSYSTEM_INTEG);
        g_profiler.start();
        drift_field_only(dp.local, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else
#endif
    {
        drift(dp.local, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // Re-sync packed positions after drift
    g_profiler.start();
    dp.local.pack_pos_range(start, count);
    g_profiler.stop(StepProfiler::PACK_DATA);

    // U-tilde kick (2dt/3) with gradient correction
    if (tree_mode) {
        // Tree mode: force + gradient correction + kick
        // Refresh positions (O(N)) instead of rebuilding tree (O(N log N))
        g_profiler.start();
        if (!tree_prebuilt) {
            g_octree.n_leaf = tree_n_leaf;
            g_octree.grad_dipole = tree_grad_dipole;
            g_octree.n_group = tree_n_group;
            g_octree.build(dp.local, start, count, tree_theta);
            tree_prebuilt = true;
        } else {
            g_octree.refresh_positions(dp.local);
        }
        int my_g_off, my_g_cnt;
        get_tree_group_range(g_octree, my_g_off, my_g_cnt);
        zero_accelerations(dp.local, start, count);
        tree_self_acceleration(dp.local, g_octree, start, count, G, soft2, my_g_off, my_g_cnt);
        dp.allreduce_accelerations_subset(start, count);
        tree_momentum_correction(dp.local, start, count);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        // Gradient correction via tree: all ranks compute (not distributed)
        // Forces must be correct (allreduced) before gradient computation
        g_profiler.start();
        g_octree.compute_cell_accelerations(dp.local);
        tree_self_gradient_correction(dp.local, g_octree, G, soft2, dt);
        tree_momentum_correction(dp.local, start, count);
        g_profiler.stop(StepProfiler::SELF_GRAD, self_pairs);
        g_profiler.start();
        force_err_inject(dp.local, start, count, count);
        kick_velocities(dp.local, start, count, dt * FSI_MOD_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else {
        // Separate: force → allgather → gradient correction → allgather → kick
        g_profiler.start();
        self_acceleration_distributed(dp, start, count, G, soft2);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        g_profiler.start();
        self_gradient_correction_distributed(dp, start, count, G, soft2, dt);
        g_profiler.stop(StepProfiler::SELF_GRAD, self_pairs);
        g_profiler.start();
        force_err_inject(dp.local, start, count, count);
        kick_velocities(dp.local, start, count, dt * FSI_MOD_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // H_S (dt/2) then T' drift (dt/2) — Rantala et al. 2023, Algorithm 1 ordering
    g_profiler.start();
#ifdef ENABLE_COLLISIONAL
    if (dp.local.collisional && !g_subsystem_mgr.active.empty()) {
        advance_subsystems_in_range(dp.local, start, count, G, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::SUBSYSTEM_INTEG);
        g_profiler.start();
        drift_field_only(dp.local, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else
#endif
    {
        drift(dp.local, start, count, dt * FSI_DRIFT);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }

    // Re-sync packed positions after drift
    g_profiler.start();
    dp.local.pack_pos_range(start, count);
    g_profiler.stop(StepProfiler::PACK_DATA);

    // U kick (dt/6) - force then kick
    if (tree_mode) {
        g_profiler.start();
        g_octree.refresh_positions(dp.local);
        int my_g_off, my_g_cnt;
        get_tree_group_range(g_octree, my_g_off, my_g_cnt);
        zero_accelerations(dp.local, start, count);
        tree_self_acceleration(dp.local, g_octree, start, count, G, soft2, my_g_off, my_g_cnt);
        dp.allreduce_accelerations_subset(start, count);
        tree_momentum_correction(dp.local, start, count);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        g_profiler.start();
        force_err_inject(dp.local, start, count, count);
        kick_velocities(dp.local, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    } else {
        g_profiler.start();
        self_acceleration_distributed(dp, start, count, G, soft2);
        g_profiler.stop(StepProfiler::SELF_FORCE, self_pairs);
        g_profiler.start();
        force_err_inject(dp.local, start, count, count);
        kick_velocities(dp.local, start, count, dt * FSI_KICK);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    }
}

// =============================================================================
// Distributed cross-acceleration between slow and fast particles
// Tiled GPU kernel per FROST paper
// =============================================================================
static void cross_acceleration_distributed(DistributedParticles& dp,
                                           int slow_start, int slow_count,
                                           int fast_start, int fast_count,
                                           double G, double soft2) {
    if (slow_count <= 0 || fast_count <= 0) return;


    // Slow particles from fast, fast from slow: each rank its slice of the sinks
    int slice_start, slice_count;
    dp.get_force_slice(slow_start, slow_count, slice_start, slice_count);
    n2_detail::SinkSetScope n2_set_3(slow_count);   // chunking of the whole set (rank-count independent)
    acceleration_range(dp.local, slice_start, slice_count, fast_start, fast_count, G, soft2);
    int slice_start_f, slice_count_f;
    dp.get_force_slice(fast_start, fast_count, slice_start_f, slice_count_f);
    n2_detail::SinkSetScope n2_set_4(fast_count);   // chunking of the whole set (rank-count independent)
    acceleration_range(dp.local, slice_start_f, slice_count_f, slow_start, slow_count, G, soft2);
    Kokkos::fence("integrator.cpp:2061");

    // Each rank computed its slice of slow and its slice of fast: one gather of both slices (AUDIT G row 5)
    if (use_nonblocking_mpi) {
        dp.iallgather_accelerations_subset(slow_start, slow_count, 0);
        dp.iallgather_accelerations_subset(fast_start, fast_count, 1);
        dp.wait_and_unpack_accelerations(slow_start, slow_count, 0);
        dp.wait_and_unpack_accelerations(fast_start, fast_count, 1);
    } else {
        dp.allgather_accelerations_two(slow_start, slow_count, fast_start, fast_count);   // one collective
    }
}

// Cross-acceleration with gradient correction - tiled GPU kernel
static void cross_acceleration_and_gradient_distributed(DistributedParticles& dp,
                                                        int slow_start, int slow_count,
                                                        int fast_start, int fast_count,
                                                        double G, double soft2, double dt) {
    // First compute cross-accelerations (already tiled)
    cross_acceleration_distributed(dp, slow_start, slow_count,
                                   fast_start, fast_count, G, soft2);


    RawView pax{dp.local.ax.data()}, pay{dp.local.ay.data()}, paz{dp.local.az.data()};

    // Snapshot accelerations for both slow and fast ranges BEFORE gradient kernels.
    // The slow-from-fast kernel modifies slow accelerations; without a snapshot the
    // fast-from-slow kernel would read corrected (not original) slow values, breaking
    // the antisymmetry corr_ij = -corr_ji and violating momentum conservation.
    static Kokkos::View<double*> ax_snap_s("ax_snap_cross_s", 0);
    static Kokkos::View<double*> ay_snap_s("ay_snap_cross_s", 0);
    static Kokkos::View<double*> az_snap_s("az_snap_cross_s", 0);
    static Kokkos::View<double*> ax_snap_f("ax_snap_cross_f", 0);
    static Kokkos::View<double*> ay_snap_f("ay_snap_cross_f", 0);
    static Kokkos::View<double*> az_snap_f("az_snap_cross_f", 0);
    static bool released_at_finalize = (Kokkos::push_finalize_hook([] {
        ax_snap_s = {}; ay_snap_s = {}; az_snap_s = {}; ax_snap_f = {}; ay_snap_f = {}; az_snap_f = {};
    }), true);
    (void)released_at_finalize;
    if ((int)ax_snap_s.extent(0) < slow_count) {
        Kokkos::realloc(ax_snap_s, slow_count);
        Kokkos::realloc(ay_snap_s, slow_count);
        Kokkos::realloc(az_snap_s, slow_count);
    }
    if ((int)ax_snap_f.extent(0) < fast_count) {
        Kokkos::realloc(ax_snap_f, fast_count);
        Kokkos::realloc(ay_snap_f, fast_count);
        Kokkos::realloc(az_snap_f, fast_count);
    }
    RawView sax_s{ax_snap_s.data()}, say_s{ay_snap_s.data()}, saz_s{az_snap_s.data()};
    RawView sax_f{ax_snap_f.data()}, say_f{ay_snap_f.data()}, saz_f{az_snap_f.data()};
    small_for("snap_cross_slow", slow_count, KOKKOS_LAMBDA(int ii) {
        sax_s(ii) = pax(slow_start + ii);
        say_s(ii) = pay(slow_start + ii);
        saz_s(ii) = paz(slow_start + ii);
    });
    small_for("snap_cross_fast", fast_count, KOKKOS_LAMBDA(int ii) {
        sax_f(ii) = pax(fast_start + ii);
        say_f(ii) = pay(fast_start + ii);
        saz_f(ii) = paz(fast_start + ii);
    });

    GradSnapshot snap_s{ax_snap_s.data(), ay_snap_s.data(), az_snap_s.data(), slow_start};
    GradSnapshot snap_f{ax_snap_f.data(), ay_snap_f.data(), az_snap_f.data(), fast_start};
    int slice_start, slice_count;
    dp.get_force_slice(slow_start, slow_count, slice_start, slice_count);
    n2_detail::SinkSetScope n2_set_5(slow_count);   // chunking of the whole set (rank-count independent)
    // grad_mag written as by cross_gradient_correction on one rank (rank-count independence), gathered below
    const bool gm_on = dp.local.collisional;
    gradient_correction_range(dp.local, slice_start, slice_count, fast_start, fast_count, snap_s, snap_f,
                              G, soft2, dt, gm_on);
    int slice_start_f, slice_count_f;
    dp.get_force_slice(fast_start, fast_count, slice_start_f, slice_count_f);
    n2_detail::SinkSetScope n2_set_6(fast_count);   // chunking of the whole set (rank-count independent)
    gradient_correction_range(dp.local, slice_start_f, slice_count_f, slow_start, slow_count, snap_f, snap_s,
                              G, soft2, dt, gm_on);
    Kokkos::fence();

    // Slices of slow and fast computed independently: one gather of both (AUDIT G row 5)
    if (use_nonblocking_mpi) {
        dp.iallgather_accelerations_subset(slow_start, slow_count, 0);
        dp.iallgather_accelerations_subset(fast_start, fast_count, 1);
        dp.wait_and_unpack_accelerations(slow_start, slow_count, 0);
        dp.wait_and_unpack_accelerations(fast_start, fast_count, 1);
    } else {
        dp.allgather_accelerations_two(slow_start, slow_count, fast_start, fast_count);   // one collective
    }
    if (gm_on) {
        const int st[2] = {slow_start, fast_start}, ct[2] = {slow_count, fast_count}, f[1] = {4};
        dp.allgather_fields(st, ct, 2, f, 1);
    }
}

// Distributed adaptive timestep computation - tiled GPU kernel
static void compute_timestep_distributed(DistributedParticles& dp, int start, int count,
                                      double G, double soft2, double eta) {
    (void)eta;
    if (count <= 1) return;


    int slice_start, slice_count;
    dp.get_force_slice(start, count, slice_start, slice_count);
    n2_detail::SinkSetScope n2_set_7(count);   // chunking of the whole set (rank-count independent)

    if (slice_count > 0) {
        timestep_range(dp.local, slice_start, slice_count, start, count, G, soft2, eta);
        Kokkos::fence();
    }

    // dt_adapt, and in collisional mode the partner-free accelerations, in one collective (AUDIT G row 5)
#ifdef ENABLE_COLLISIONAL
    if (dp.local.collisional) {
        dp.allgather_dt_acc_subset(start, count);
        return;
    }
#endif
    dp.allgather_timestep_subset(start, count);
}

// hhs_fsi_recursive_local removed — merged into hhs_fsi_recursive

// =============================================================================
// Tree-based cross-force using global tree with dual monopoles
// One global tree (g_octree) built at top of step; dual monopoles recomputed
// at each HHS level after partition. No separate tree builds needed.
// =============================================================================

// Cross-force via global tree: both directions (slow←fast, fast←slow)
static void cross_acceleration_global_tree(Particles& p,
                                            int slow_start, int slow_count,
                                            int fast_start, int fast_count,
                                            double G, double soft2) {
    (void)fast_start;
    if (slow_count <= 0 || fast_count <= 0) return;

    // slow ← fast (source_flag=1 means fast sources, target=slow)
    tree_global_cross_acceleration(p, g_octree, 1, G, soft2);
    // fast ← slow (source_flag=0 means slow sources, target=fast)
    tree_global_cross_acceleration(p, g_octree, 0, G, soft2);

    // Momentum correction on combined range
    tree_momentum_correction(p, slow_start, slow_count + fast_count);
}

// Cross-force + gradient via global tree
static void cross_acceleration_and_gradient_global_tree(Particles& p,
                                                         int slow_start, int slow_count,
                                                         int fast_start, int fast_count,
                                                         double G, double soft2, double dt) {
    (void)fast_start;
    if (slow_count <= 0 || fast_count <= 0) return;

    // Cross-forces (overwrites ax/ay/az for both subsets)
    tree_global_cross_acceleration(p, g_octree, 1, G, soft2);
    tree_global_cross_acceleration(p, g_octree, 0, G, soft2);
    tree_momentum_correction(p, slow_start, slow_count + fast_count);

    // Compute dual cell accelerations for gradient correction
    g_octree.compute_dual_cell_accelerations(p);

    // Gradient correction: slow from fast, fast from slow
    tree_global_cross_gradient_correction(p, g_octree, 1, G, soft2, dt);
    tree_global_cross_gradient_correction(p, g_octree, 0, G, soft2, dt);
    tree_momentum_correction(p, slow_start, slow_count + fast_count);
}

// =============================================================================
// HHS with connected components (HHS_CC, Janes et al. 2014): the fast set F splits into components (pair time
// step below the pivot, or subsystem partners, connect). Interactions S<->F and between components are kicked at
// this level (with the gradient term in the middle kick); each component recurses alone, so the deep levels become
// independent pieces of work. Same kick-drift structure as FROST Algorithm 3.
// =============================================================================
static int hhs_cc_step(Particles& p, DistributedParticles* dp, int slow_start, int slow_count, int fast_start,
                       std::vector<int> comps, double pivot_dt, double G, double soft2, double dt_min, double eta,
                       int depth) {
    static Kokkos::View<int*> cid("cc_cid", 0);
    static Kokkos::View<double*> snx("cc_snx", 0), sny("cc_sny", 0), snz("cc_snz", 0);
    static bool hook = (Kokkos::push_finalize_hook([] { cid = Kokkos::View<int*>(); snx = sny = snz = Kokkos::View<double*>(); }), true);
    (void)hook;
    int n_removed = 0;
    auto fast_count = [&]() { int n = 0; for (int c : comps) n += c; return n; };
    auto set_cid = [&]() {   // component index per particle of F (nested levels reuse the array)
        if ((int)cid.extent(0) < p.N) Kokkos::realloc(cid, p.N);
        std::vector<int> h;
        for (size_t c = 0; c < comps.size(); c++) h.insert(h.end(), comps[c], (int)c);
        if (h.empty()) return;
        Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hv(h.data(), h.size());
        dcopy(Kokkos::subview(cid, Kokkos::make_pair(fast_start, fast_start + (int)h.size())), hv);
    };
    auto cross = [&](bool with_grad) {
        const int nf = fast_count();
        g_profiler.start();
        cross_acceleration(p, slow_start, slow_count, fast_start, nf, G, soft2);   // nf = 0 or slow = 0: zero
        set_cid();
        cc_inter_acceleration(p, fast_start, nf, cid.data(), G, soft2);
        if (with_grad) {
            if ((int)snx.extent(0) < nf) { Kokkos::realloc(snx, nf); Kokkos::realloc(sny, nf); Kokkos::realloc(snz, nf); }
            RawView ax{p.ax.data()}, ay{p.ay.data()}, az{p.az.data()}, ox{snx.data()}, oy{sny.data()}, oz{snz.data()};
            const int f0 = fast_start;
            small_for("cc_snap", nf, KOKKOS_LAMBDA(int ii) {
                ox(ii) = ax(f0 + ii); oy(ii) = ay(f0 + ii); oz(ii) = az(f0 + ii);
            });
            if (slow_count > 0)
                cross_gradient_correction(p, slow_start, slow_count, fast_start, nf, G, soft2, pivot_dt);
            cc_inter_gradient(p, fast_start, nf, cid.data(), GradSnapshot{snx.data(), sny.data(), snz.data(), fast_start},
                              G, soft2, pivot_dt);
        }
        g_profiler.stop(with_grad ? StepProfiler::CROSS_GRAD : StepProfiler::CROSS_FORCE);
    };
    auto kick = [&](double f) {
        g_profiler.start();
        force_err_inject(p, slow_start, slow_count + fast_count(), slow_count + fast_count());
        kick_velocities(p, slow_start, slow_count, pivot_dt * f);
        kick_velocities(p, fast_start, fast_count(), pivot_dt * f);
        g_profiler.stop(StepProfiler::DRIFT_KICK);
    };
    auto recurse_all = [&]() {
        int s0 = fast_start;
        for (int& c : comps) {
            if (c == 1) {   // a lone particle has no internal interactions: its own Hamiltonian is a free drift
                g_profiler.start();
                drift(p, s0, 1, pivot_dt * 0.5);
                g_profiler.stop(StepProfiler::DRIFT_KICK);
                s0 += 1;
                continue;
            }
            int k = hhs_fsi_recursive(p, dp, s0, c, pivot_dt * 0.5, G, soft2, dt_min, eta, false, false, depth + 1);
            c -= k;
            n_removed += k;
            s0 += c;
        }
    };
#ifdef ENABLE_COLLISIONAL
    auto saved_active = g_subsystem_mgr.active;
    auto restore = [&]() {
        g_profiler.start();
        g_subsystem_mgr.restore_active(saved_active);
        if (!g_subsystem_mgr.active.empty()) g_subsystem_mgr.rebuild_pid_to_idx(p);
        g_subsystem_mgr.update_particle_flags(p);
        g_profiler.stop(StepProfiler::SUBSYSTEM_BOOKKEEP);
    };
#else
    auto restore = []() {};
#endif
    cross(false); kick(FSI_KICK);
    if (slow_count > 0) fsi_step(p, slow_start, slow_count, G, soft2, pivot_dt * 0.5, false, true);
    recurse_all();
    restore();
    cross(true); kick(FSI_MOD_KICK);
    recurse_all();
    restore();
    if (slow_count > 0) fsi_step(p, slow_start, slow_count, G, soft2, pivot_dt * 0.5, false, true);
    cross(false); kick(FSI_KICK);
    return n_removed;
}

// =============================================================================
// Host-resident subtrees (HOST_SUBTREE_MAX, host_exec.h). A set of at most host_subtree_max particles runs its whole
// sub-recursion on the host: one packed copy of its particles in, the same recursion code on a host copy (time steps,
// detection, subsystems, mergers, kicks, drifts), one packed copy out. On a discrete GPU each deep call is a chain of
// tiny launches and synchronisations (0.2-0.3 ms per call for 4-13 particles in run B, 2026-10-15), a few us on the
// host. The device code runs unchanged: it reads the particle arrays through raw pointers, which here point to host
// memory. Host and device arithmetic differ in the last bits (rsqrt, FMA contraction), so the trajectories change
// within round-off; the placement depends only on the set size (deterministic, rank-count independent).
// =============================================================================
namespace {
struct HostSubtree {
    // Particle fields moved in and out: doubles, ints and int64s, field-major ([field][slot]) in one buffer each.
    // The host buffers are the host copy's storage (its views point into them)
    enum { X, Y, Z, VX, VY, VZ, CVX, CVY, CVZ, AX, AY, AZ, M, DT, ND_BASE,
           MZAMS = ND_BASE, AGE, RAD, LUM, CMASS, SMAG, DTL0, GMAG, SPX, SPY, SPZ, ND_COLL };
    enum { LEVEL, ORIG, NI_BASE, PID = NI_BASE, SID, STYPE, NI_COLL };
    enum { IX, IY, IZ, NL };
    using PinD = Kokkos::View<double*, Kokkos::SharedHostPinnedSpace>;
    using PinI = Kokkos::View<int*, Kokkos::SharedHostPinnedSpace>;
    using PinL = Kokkos::View<int64_t*, Kokkos::SharedHostPinnedSpace>;
    Kokkos::View<double*> d_dbl;
    Kokkos::View<int*> d_int;
    Kokkos::View<int64_t*> d_l;
    PinD h_dbl;
    PinI h_int;
    PinL h_l;
    std::vector<double> scr_d;     // scratch views of the host copy
    std::vector<int> scr_i;
    std::vector<int64_t> scr_l;
    std::vector<double> packed;    // pos_packed, vel_packed (unused on the local path, sized for safety)
    std::vector<double> spin;      // spin_vec (LayoutRight [n][3])
    Particles q;
    int cap = 0;

    // the device arrays of the moved fields (spin_vec apart: interleaved)
    struct PtrsD { double* v[ND_COLL - 3]; };
    struct PtrsI { int* v[NI_COLL]; };
    static PtrsD ptrs_d(const Particles& p) {
        const bool c = p.collisional;
        return {{p.x.data(), p.y.data(), p.z.data(), p.vx.data(), p.vy.data(), p.vz.data(), p.cvx.data(), p.cvy.data(),
                 p.cvz.data(), p.ax.data(), p.ay.data(), p.az.data(), p.mass.data(), p.dt_adapt.data(),
                 c ? p.m_zams.data() : nullptr, c ? p.stellar_age.data() : nullptr, c ? p.stellar_radius.data() : nullptr,
                 c ? p.luminosity.data() : nullptr, c ? p.core_mass.data() : nullptr, c ? p.spin_mag.data() : nullptr,
                 c ? p.dt_level0.data() : nullptr, c ? p.grad_mag.data() : nullptr}};
    }
    static PtrsI ptrs_i(const Particles& p) {
        const bool c = p.collisional;
        return {{p.level.data(), p.orig_index.data(), c ? p.particle_id.data() : nullptr,
                 c ? p.subsystem_id.data() : nullptr, c ? p.stellar_type.data() : nullptr}};
    }
    static int nd(bool coll) { return coll ? ND_COLL : ND_BASE; }
    static int ni(bool coll) { return coll ? NI_COLL : NI_BASE; }

    void reserve(int n) {
        if (n <= cap) return;
        cap = n;
        d_dbl = Kokkos::View<double*>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "hs_d"), (size_t)ND_COLL * n);
        d_int = Kokkos::View<int*>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "hs_i"), (size_t)NI_COLL * n);
        d_l = Kokkos::View<int64_t*>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "hs_l"), (size_t)NL * n);
        h_dbl = PinD(Kokkos::view_alloc(Kokkos::WithoutInitializing, "hs_hd"), (size_t)ND_COLL * n);
        h_int = PinI(Kokkos::view_alloc(Kokkos::WithoutInitializing, "hs_hi"), (size_t)NI_COLL * n);
        h_l = PinL(Kokkos::view_alloc(Kokkos::WithoutInitializing, "hs_hl"), (size_t)NL * n);
        scr_d.assign((size_t)11 * n, 0.0);
        scr_i.assign((size_t)3 * n, 0);
        scr_l.assign((size_t)3 * n, 0);
        packed.assign((size_t)8 * n, 0.0);
        spin.assign((size_t)3 * n, 0.0);
    }

    // The host copy's views over the buffers, for a set of n particles
    void wrap(const Particles& p, int n) {
        using VD = Kokkos::View<double*>;
        using VI = Kokkos::View<int*>;
        using VL = Kokkos::View<int64_t*>;
        auto d = [&](int f) { return VD(h_dbl.data() + (size_t)f * n, n); };
        auto i = [&](int f) { return VI(h_int.data() + (size_t)f * n, n); };
        auto l = [&](int f) { return VL(h_l.data() + (size_t)f * n, n); };
        q.N = n;
        q.collisional = p.collisional;
        q.x = d(X); q.y = d(Y); q.z = d(Z); q.vx = d(VX); q.vy = d(VY); q.vz = d(VZ);
        q.cvx = d(CVX); q.cvy = d(CVY); q.cvz = d(CVZ); q.ax = d(AX); q.ay = d(AY); q.az = d(AZ);
        q.mass = d(M); q.dt_adapt = d(DT);
        q.level = i(LEVEL); q.orig_index = i(ORIG);
        q.ix = l(IX); q.iy = l(IY); q.iz = l(IZ);
        VD* sd[11] = {&q.scratch1, &q.scratch2, &q.scratch3, &q.scratch4, &q.scratch5, &q.scratch6, &q.scratch7,
                      &q.scratch8, &q.scratch9, &q.scratch10, &q.scratch11};
        for (int k = 0; k < 11; k++) *sd[k] = VD(scr_d.data() + (size_t)k * n, n);
        q.scratch_i = VI(scr_i.data(), n); q.scratch_i2 = VI(scr_i.data() + n, n); q.scratch_i3 = VI(scr_i.data() + 2 * n, n);
        q.scratch_l1 = VL(scr_l.data(), n); q.scratch_l2 = VL(scr_l.data() + n, n); q.scratch_l3 = VL(scr_l.data() + 2 * n, n);
        q.pos_packed = decltype(q.pos_packed)(packed.data(), n);
        q.vel_packed = decltype(q.vel_packed)(packed.data() + 4 * (size_t)n, n);
        q.layout_dirty_lo = INT_MAX; q.layout_dirty_hi = 0; q.removed_pid_log.clear();
        if (!p.collisional) return;
        q.m_zams = d(MZAMS); q.stellar_age = d(AGE); q.stellar_radius = d(RAD); q.luminosity = d(LUM);
        q.core_mass = d(CMASS); q.spin_mag = d(SMAG); q.dt_level0 = d(DTL0); q.grad_mag = d(GMAG);
        q.spin_vec = decltype(q.spin_vec)(spin.data(), n);
        q.particle_id = i(PID); q.subsystem_id = i(SID); q.stellar_type = i(STYPE);
    }

    // [start, start + n) of p into the host copy: one pack kernel and one transfer per buffer
    void copy_in(Particles& p, int start, int n) {
        reserve(n);
        const bool coll = p.collisional;
        RawView o{d_dbl.data()};
        Raw1<int> oi{d_int.data()};
        Raw1<int64_t> ol{d_l.data()};
        const PtrsD pd = ptrs_d(p);
        const PtrsI pi = ptrs_i(p);
        const int64_t* lx = p.ix.data(); const int64_t* ly = p.iy.data(); const int64_t* lz = p.iz.data();
        const double* sp = coll ? p.spin_vec.data() : nullptr;
        const int ndd = nd(coll), nii = ni(coll);
        Kokkos::parallel_for("hs_pack", n, KOKKOS_LAMBDA(int k) {
            const int s = start + k;
            for (int f = 0; f < ND_BASE; f++) o((size_t)f * n + k) = pd.v[f][s];
            if (coll) {
                for (int f = ND_BASE; f < ND_COLL - 3; f++) o((size_t)f * n + k) = pd.v[f][s];
                for (int c = 0; c < 3; c++) o((size_t)(SPX + c) * n + k) = sp[3 * (size_t)s + c];
            }
            for (int f = 0; f < nii; f++) oi((size_t)f * n + k) = pi.v[f][s];
            ol(k) = lx[s]; ol((size_t)n + k) = ly[s]; ol(2 * (size_t)n + k) = lz[s];
        });
        auto cp = [&](auto& h, auto& dv, size_t len) {
            dcopy(Kokkos::subview(h, Kokkos::make_pair((size_t)0, len)), Kokkos::subview(dv, Kokkos::make_pair((size_t)0, len)));
        };
        cp(h_dbl, d_dbl, (size_t)ndd * n);
        cp(h_int, d_int, (size_t)nii * n);
        cp(h_l, d_l, (size_t)NL * n);
        wrap(p, n);
        if (coll)
            for (int k = 0; k < n; k++)
                for (int c = 0; c < 3; c++) spin[3 * (size_t)k + c] = h_dbl((size_t)(SPX + c) * n + k);
    }

    // The host copy back to [start, start+n): its first n_out slots (a compaction on the host keeps the field stride
    // n), then n - n_out removed slots (mass 0, ids `gone`)
    void copy_out(Particles& p, int start, int n, int n_out, const std::vector<int>& gone) {
        const bool coll = p.collisional;
        const int ndd = nd(coll), nii = ni(coll);
        if (coll)
            for (int k = 0; k < n; k++)
                for (int c = 0; c < 3; c++) h_dbl((size_t)(SPX + c) * n + k) = spin[3 * (size_t)k + c];
        for (int k = n_out; k < n; k++) {
            h_dbl((size_t)M * n + k) = 0.0;
            if (coll) { h_int((size_t)PID * n + k) = gone[k - n_out]; h_int((size_t)SID * n + k) = -1; }
        }
        auto cp = [&](auto& dv, auto& h, size_t len) {
            dcopy(Kokkos::subview(dv, Kokkos::make_pair((size_t)0, len)), Kokkos::subview(h, Kokkos::make_pair((size_t)0, len)));
        };
        cp(d_dbl, h_dbl, (size_t)ndd * n);
        cp(d_int, h_int, (size_t)nii * n);
        cp(d_l, h_l, (size_t)NL * n);
        RawView o{d_dbl.data()};
        Raw1<int> oi{d_int.data()};
        Raw1<int64_t> ol{d_l.data()};
        const PtrsD pd = ptrs_d(p);
        const PtrsI pi = ptrs_i(p);
        int64_t* lx = p.ix.data(); int64_t* ly = p.iy.data(); int64_t* lz = p.iz.data();
        double* sp = coll ? p.spin_vec.data() : nullptr;
        Kokkos::parallel_for("hs_unpack", n, KOKKOS_LAMBDA(int k) {
            const int s = start + k;
            for (int f = 0; f < ND_BASE; f++) pd.v[f][s] = o((size_t)f * n + k);
            if (coll) {
                for (int f = ND_BASE; f < ND_COLL - 3; f++) pd.v[f][s] = o((size_t)f * n + k);
                for (int c = 0; c < 3; c++) sp[3 * (size_t)s + c] = o((size_t)(SPX + c) * n + k);
            }
            for (int f = 0; f < nii; f++) pi.v[f][s] = oi((size_t)f * n + k);
            lx[s] = ol(k); ly[s] = ol((size_t)n + k); lz[s] = ol(2 * (size_t)n + k);
        });
    }
};
struct HostSubtreeStats {   // printed at exit (tests read it)
    long long calls = 0, particles = 0;
    ~HostSubtreeStats() {
        if (calls) std::printf("HOST_SUBTREE: %lld calls, %lld particles\n", calls, particles);
        std::fflush(stdout);
    }
} g_host_subtree_stats;
HostSubtree& host_subtree() {
    static HostSubtree* h = nullptr;
    if (!h) {
        h = new HostSubtree();
        Kokkos::push_finalize_hook([] { delete h; h = nullptr; });
    }
    return *h;
}
}  // namespace

// Whether a set of `count` particles runs as a host subtree here: local path (no MPI distribution, no tree), and
// none of the experiments the host copy does not carry (integer positions, HHS_CC buffers)
static bool host_subtree_eligible(const Particles& p, int count, bool use_mpi, bool tree_eligible) {
    if (host_subtree_max <= 0 || count > host_subtree_max || host_exec() || use_mpi || tree_eligible) return false;
    if (use_integer_positions || hhs_cc) return false;
#ifdef ENABLE_COLLISIONAL
#endif
    (void)p;
    return true;
}

static int hhs_fsi_host_subtree(Particles& p, DistributedParticles* dp, int start, int count, double pivot_dt,
                                double G, double soft2, double dt_min, double eta, bool dt_valid, int depth) {
    HostSubtree& hs = host_subtree();
    NvtxScope nvtx_hostsub("HOSTSUB");
    StepProfiler::Region prof_hostsub(g_profiler, StepProfiler::HOST_SUBTREE);
    g_host_subtree_stats.calls++;
    g_host_subtree_stats.particles += count;
    g_profiler.start();
    hs.copy_in(p, start, count);
    g_profiler.stop(StepProfiler::CPU_PATH);
    Particles& q = hs.q;
#ifdef ENABLE_COLLISIONAL
    // The pid map points the set's ids at their host slots for the duration (the map entries of all other ids stay
    // untouched: no subsystem of the host copy refers to them); subsystems with a member outside the set are not the
    // set's (parent levels); the stellar cache must hold every particle before (a refresh inside sees only the set)
    SubsystemManager& mgr = g_subsystem_mgr;
    const long long cache_refreshes = StellarPropsCache::get().refreshes;
    if (p.collisional) {
        if (!StellarPropsCache::get().valid) StellarPropsCache::get().ensure(p);
        mgr.rebuild_pid_to_idx(p);
        const int* pid = q.particle_id.data();
        for (int k = 0; k < count; k++) mgr.pid_to_idx_[pid[k]] = k;
        mgr.map_source_ = &q;
        std::vector<Subsystem> mine;
        for (const auto& sub : mgr.active) {
            bool in = true;
            for (int m : sub.particle_ids) {
                const int i = mgr.safe_pid_to_idx(m);
                in = in && i >= 0 && i < count && pid[i] == m;
            }
            if (in) mine.push_back(sub);
        }
        mgr.active.swap(mine);
        mgr.use_host_buffers(true);
    }
#endif
    std::vector<int> pids_in;
    if (p.collisional) pids_in.assign(q.particle_id.data(), q.particle_id.data() + count);
    int n_removed = 0;
    {
        HostExecScope on;
        n_removed = hhs_fsi_recursive(q, nullptr, 0, count, pivot_dt, G, soft2, dt_min, eta, dt_valid, false, depth);
    }
    const int n_out = count - n_removed;
    std::vector<int> gone;   // ids merged away on the host
    if (n_removed > 0) {
        std::vector<int> kept(q.particle_id.data(), q.particle_id.data() + n_out);
        std::sort(kept.begin(), kept.end());
        for (int pid : pids_in)
            if (!std::binary_search(kept.begin(), kept.end(), pid)) gone.push_back(pid);
        if ((int)gone.size() != n_removed) {
            std::fprintf(stderr, "host subtree: %d particles removed, %zu ids gone\n", n_removed, gone.size());
            std::abort();
        }
    }
#ifdef ENABLE_COLLISIONAL
    if (p.collisional) {
        mgr.use_host_buffers(false);
        const int* pid = q.particle_id.data();
        for (int k = 0; k < n_out; k++) mgr.pid_to_idx_[pid[k]] = start + k;
        for (int g : gone) mgr.pid_to_idx_[g] = -1;
        mgr.map_source_ = &p;
        if (StellarPropsCache::get().refreshes != cache_refreshes) StellarPropsCache::get().invalidate();
    }
#endif
    g_profiler.start();
    hs.copy_out(p, start, count, n_out, gone);
    g_profiler.stop(StepProfiler::CPU_PATH);
#ifdef ENABLE_COLLISIONAL
    if (n_removed > 0) {
        // the removed particles' slots (mass 0) leave the device arrays: the tail shifts down as after a merger here
        const int k = p.compact_remove_zero_mass(start + n_out, start + count);
        if (k != n_removed) {
            std::fprintf(stderr, "host subtree: %d particles removed on the host, %d on the device\n", n_removed, k);
            std::abort();
        }
        sync_after_removal(p, dp);
        g_subsystem_mgr.update_particle_flags(p);
    }
#endif
    if (dp) p.pack_pos_range(start, n_out);
    return n_removed;
}

// =============================================================================
// Unified HHS-FSI recursion (FROST Algorithm 3)
//
// Single function handles all GPU paths: MPI-distributed, local GPU, and tree.
// dp == nullptr → single-rank mode (no MPI, no tree).
// dp != nullptr → dispatch MPI vs local GPU based on count vs serial_n2_gpu.
// Tree operations require dp != nullptr (tree_threshold >> serial_n2_gpu).
// Sets of at most host_subtree_max particles run as a host subtree (hhs_fsi_host_subtree).
// =============================================================================
static int hhs_fsi_recursive_impl(Particles& p, DistributedParticles* dp, int start, int count, double pivot_dt,
                                  double G, double soft2, double dt_min, double eta, bool dt_valid,
                                  bool global_tree_available, int depth);

// NBODY_ELEDGER=1 (debug, one rank, no MPI distribution): an energy ledger of the HHS recursion. A call integrates
// only the Hamiltonian of its own set [start, start + count) (everything outside acts through the parent's kicks), so
// E_set(after) + books(after) - E_set(before) - books(before) is the integration error made inside the call; minus
// the errors of its child calls, the error made at its own level (its slow/fast split, kicks, subsystem coupling).
// After the root call the step's error and the calls with the largest own errors (depth, set size, pivot, ids of
// sets of up to 32) are printed as ELEDGER lines, in units of |E0| (E_REF). Costs O(count^2) per call: replays only.
namespace {
struct EnergyLedger {
    // type: the branch the call took (NBODY_ELEDGER call types; set by hhs_fsi_recursive_impl through ledger_type)
    struct Frame { double child = 0.0; bool valid = true; int type = 0; };
    struct Rec { int depth = 0, n = 0; double pivot = 0.0, own = 0.0, total = 0.0; std::vector<int> pids; };
    std::vector<Frame> stack;
    std::vector<Rec> top;   // largest |own| first
    double root = 0.0, sum_own = 0.0, eref = 0.0;
    int calls = 0;
    // budget diagnostics: own errors and subsystem-advance residues per depth (frame depth), per-class internal
    // energy changes of the advances, book deltas made inside the advances
    static constexpr int MD = 40;
    double depth_own[MD] = {}, depth_adv[MD] = {};
    int depth_calls[MD] = {};
    double adv_unattr = 0.0;   // advances whose subsystem list changed (not attributed per class)
    // own errors per call type (0 other/early return, 1 min-dt FSI, 2 all slow, 3 all fast, 4 slow + fast
    // (FROST Algorithm 3), 5 one subsystem, 6 HHS_CC, 7 host subtree) and |own| per depth (signed sums cancel)
    static constexpr int NT = 8;
    double type_own[NT] = {}, type_abs[NT] = {}, depth_abs[MD] = {};
    int type_calls[NT] = {};
    // classes: 0 sec2 nonPN, 1 sec2 PN, 2 sec3, 3 LogH n2 nonPN, 4 LogH n3 nonPN, 5 LogH n4+ nonPN, 6 LogH PN (any n)
    static constexpr int NC = 7;
    double cls_dE[NC] = {}, cls_abs[NC] = {};
    long long cls_n[NC] = {};
    double bk_gw = 0.0, bk_pn = 0.0, bk_sd = 0.0, bk_merge = 0.0, bk_other = 0.0;
    // subsystem advances (advance_subsystems_in_range): change of the members' internal energy (T + U among the
    // members of each subsystem) minus the books' change, summed; the subsystems with the largest internal change
    double adv_unbooked = 0.0;
    struct SubRec { double dE = 0.0; int type = -1, n = 0; bool secular = false; std::vector<int> pids; };
    std::vector<SubRec> sub_top;
    void note_sub(SubRec r) {
        if (sub_top.size() >= 8 && std::fabs(r.dE) <= std::fabs(sub_top.back().dE)) return;
        sub_top.insert(std::upper_bound(sub_top.begin(), sub_top.end(), r, [](const SubRec& a, const SubRec& b) {
                           return std::fabs(a.dE) > std::fabs(b.dE); }), std::move(r));
        if (sub_top.size() > 8) sub_top.pop_back();
    }
    void note(Rec r) {
        sum_own += r.own;
        if (top.size() >= 12 && std::fabs(r.own) <= std::fabs(top.back().own)) return;
        top.insert(std::upper_bound(top.begin(), top.end(), r, [](const Rec& a, const Rec& b) {
                       return std::fabs(a.own) > std::fabs(b.own); }), std::move(r));
        if (top.size() > 12) top.pop_back();
    }
};
EnergyLedger g_ledger;
inline void ledger_type(int t) { if (!g_ledger.stack.empty()) g_ledger.stack.back().type = t; }

// ---- FSI time-step experiments (debug; both unset: no-op, bit-identical) ------------------------------------------
// NBODY_DT_SCALE=<class>:<f>[,<class>:<f>...]: the final time step of the particles of a class (after every criterion
// and the subsystem adjustments, at every depth) times f, i.e. all criteria of that class with eta x f: attribution of
// the FSI error by intervention. Classes: all, bh (stellar type 14), sub (subsystem members), single (neither),
// deep<L> / shallow<L> (step below / at or above dt_max / 2^L before the scaling).
// NBODY_DT_ATTR=1 (one rank): at depth 0, which criterion set each particle's step (pair free-fall, pair fly-by,
// tidal sampling, gradient / v-a, subsystem adjustment, other) per level and class; DTATTR lines. The pair and
// tidal-sampling steps are recomputed in fp64 over all pairs (O(N^2) once per step).
struct DtScaleRule { int cls = 0; int lvl = 0; double f = 1.0; };   // cls: 0 all 1 bh 2 sub 3 single 4 deep 5 shallow
const std::vector<DtScaleRule>& dt_scale_rules() {
    static const std::vector<DtScaleRule> rules = [] {
        std::vector<DtScaleRule> r;
        const char* e = std::getenv("NBODY_DT_SCALE");
        if (!e || !*e) return r;
        std::string s(e);
        size_t pos = 0;
        while (pos < s.size()) {
            size_t c = s.find(',', pos);
            std::string tok = s.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
            pos = c == std::string::npos ? s.size() : c + 1;
            const size_t k = tok.find(':');
            if (k == std::string::npos) continue;
            const std::string name = tok.substr(0, k);
            DtScaleRule q;
            q.f = std::atof(tok.c_str() + k + 1);
            if (name == "all") q.cls = 0;
            else if (name == "bh") q.cls = 1;
            else if (name == "sub") q.cls = 2;
            else if (name == "single") q.cls = 3;
            else if (name.rfind("deep", 0) == 0) { q.cls = 4; q.lvl = std::atoi(name.c_str() + 4); }
            else if (name.rfind("shallow", 0) == 0) { q.cls = 5; q.lvl = std::atoi(name.c_str() + 7); }
            else { std::fprintf(stderr, "NBODY_DT_SCALE: unknown class %s\n", name.c_str()); std::abort(); }
            if (!(q.f > 0.0)) { std::fprintf(stderr, "NBODY_DT_SCALE: factor of %s\n", name.c_str()); std::abort(); }
            r.push_back(q);
            if (MPIComm::rank() == 0) std::printf("NBODY_DT_SCALE rule %s x %g\n", name.c_str(), q.f);
        }
        return r;
    }();
    return rules;
}
bool dt_attr_on() { static const bool on = std::getenv("NBODY_DT_ATTR") != nullptr; return on; }
double g_dt_root = 0.0;   // pivot of the depth-0 call (dt_max) of the current step

void dt_scale_apply(Particles& p, int start, int count) {
    const auto& rules = dt_scale_rules();
    if (rules.empty() || count <= 0) return;
    auto dt_f = raw(p.dt_adapt);
    auto sid = raw(p.subsystem_id);
    auto st = raw(p.stellar_type);
    const bool coll = p.collisional;
    const double root = g_dt_root;
    for (const auto& q : rules) {
        const int cls = q.cls;
        const double f = q.f, thr = root > 0.0 ? std::ldexp(root, -q.lvl) : 0.0;
        small_for("dt_scale", count, KOKKOS_LAMBDA(int ii) {
            const int i = start + ii;
            const bool bh = coll && st(i) == 14, sub = coll && sid(i) >= 0;
            bool in = false;
            switch (cls) {
                case 0: in = true; break;
                case 1: in = bh; break;
                case 2: in = sub; break;
                case 3: in = !bh && !sub; break;
                case 4: in = dt_f(i) < thr; break;
                default: in = dt_f(i) >= thr; break;
            }
            if (in) dt_f(i) *= f;
        });
    }
}

// NBODY_DT_ATTR snapshots of dt_adapt at depth 0: after the pair kernels (kern) and before the subsystem adjustments
Kokkos::View<double*> g_dta_kern, g_dta_pre;
void dt_attr_snap(Particles& p, int start, int count, Kokkos::View<double*>& dst) {
    if (!dt_attr_on() || MPIComm::size() > 1 || host_exec()) return;
    static bool hook = (Kokkos::push_finalize_hook([] { g_dta_kern = {}; g_dta_pre = {}; }), true);
    (void)hook;
    if ((int)dst.extent(0) < start + count) dst = Kokkos::View<double*>("dt_attr_snap", p.N);
    Kokkos::deep_copy(Kokkos::subview(dst, Kokkos::make_pair(start, start + count)),
                      Kokkos::subview(p.dt_adapt, Kokkos::make_pair(start, start + count)));
}
void dt_attr_report(Particles& p, int start, int count, double G, double soft2) {
    if (!dt_attr_on() || MPIComm::size() > 1 || host_exec() || count <= 1) return;
    const int e = start + count;
    PairVelocities pv = pair_velocities(p, start, e);
    const double eff = g_collisional.eta_ff, efb = g_collisional.eta_fb;
    Kokkos::View<double*> mff("dta_ff", count), mfb("dta_fb", count), mtd("dta_td", count);
    auto x = p.x, y = p.y, z = p.z, m = p.mass;
    auto sid = p.subsystem_id;
    RawView vx{pv.x}, vy{pv.y}, vz{pv.z};
    const bool has_k = pv.k != nullptr;
    RawView kk{pv.k}, kf{pv.kf};
    Kokkos::parallel_for("dt_attr_pairs", count, KOKKOS_LAMBDA(int ii) {
        const int i = start + ii;
        double bff = 1e300, bfb = 1e300, btd = 1e300;
        const int si = sid(i);
        for (int j = start; j < e; j++) {
            if (j == i || (si >= 0 && sid(j) == si)) continue;
            const double dx = x(j) - x(i), dy = y(j) - y(i), dz = z(j) - z(i);
            const double dvx = vx(j) - vx(i), dvy = vy(j) - vy(i), dvz = vz(j) - vz(i);
            const double r2 = dx * dx + dy * dy + dz * dz + soft2;
            const double ir = 1.0 / Kokkos::sqrt(r2);
            double tff, tfb;
            symmetrized_timestep(r2, ir, dx, dy, dz, dvx, dvy, dvz, G, m(i) + m(j), eff, efb, tff, tfb);
            bff = Kokkos::fmin(bff, eff * tff);
            bfb = Kokkos::fmin(bfb, efb * tfb);
            if (has_k)
                btd = Kokkos::fmin(btd, Kokkos::fmax(Kokkos::fmin(kk(i) / m(j), kk(j) / m(i)) * r2 * Kokkos::sqrt(r2),
                                                     Kokkos::fmax(kf(i), kf(j))));
        }
        mff(ii) = bff; mfb(ii) = bfb; mtd(ii) = btd;
    });
    auto cp = [&](auto v) { return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v); };
    auto r = Kokkos::make_pair(start, e);
    auto hff = cp(mff), hfb = cp(mfb), htd = cp(mtd);
    auto hk = cp(Kokkos::subview(g_dta_kern, r)), hp = cp(Kokkos::subview(g_dta_pre, r)), hf = cp(Kokkos::subview(p.dt_adapt, r));
    auto hs = cp(Kokkos::subview(p.subsystem_id, r)), ht = cp(Kokkos::subview(p.stellar_type, r));
    // criteria: 0 ff, 1 fb, 2 tidal sampling, 3 gradient / v-a, 4 subsystem adjustment, 5 other (kernel step not
    // matched, e.g. tree path or jerk); classes 0 bh, 1 subsystem member, 2 single
    constexpr int NL = 48, NCR = 6;
    static long long cnt[NL][NCR][3];
    std::memset(cnt, 0, sizeof cnt);
    const double root = g_dt_root > 0.0 ? g_dt_root : 1.0;
    for (int k = 0; k < count; k++) {
        const double df = hf(k);
        int lvl = df >= root ? 0 : (int)std::ceil(std::log2(root / df) - 1e-12);
        lvl = std::min(std::max(lvl, 0), NL - 1);
        const int cl = ht(k) == 14 ? 0 : hs(k) >= 0 ? 1 : 2;
        int cr;
        if (df < hp(k) * (1.0 - 1e-12)) cr = 4;
        else if (hp(k) < hk(k) * (1.0 - 1e-12)) cr = 3;
        else {
            const double d = hk(k), c[3] = {hff(k), hfb(k), htd(k)};
            int b = 0;
            for (int q = 1; q < 3; q++) if (c[q] < c[b]) b = q;
            cr = std::fabs(c[b] - d) <= 1e-4 * d ? b : 5;
        }
        cnt[lvl][cr][cl]++;
    }
    static const char* cn[NCR] = {"ff", "fb", "tid", "grad", "subadj", "other"};
    for (int l = 0; l < NL; l++) {
        long long n = 0;
        for (int c = 0; c < NCR; c++) n += cnt[l][c][0] + cnt[l][c][1] + cnt[l][c][2];
        if (n == 0) continue;
        std::printf("DTATTR step %d L%d n %lld (bh/sub/single):", g_step_in_progress, l, n);
        for (int c = 0; c < NCR; c++) std::printf(" %s %lld/%lld/%lld", cn[c], cnt[l][c][0], cnt[l][c][1], cnt[l][c][2]);
        std::printf("\n");
    }
    std::fflush(stdout);
}

// KE + PE of the particles [s, s + n) among themselves
double set_energy(const Particles& p, int s, int n, double G) {
    if (n <= 0) return 0.0;
    auto x = p.x, y = p.y, z = p.z, vx = p.vx, vy = p.vy, vz = p.vz, m = p.mass;
    const int e = s + n;
    if (host_exec()) {   // host subtree: device-typed views over host memory (raw pointers, as host_exec.h)
        const double *hx = x.data(), *hy = y.data(), *hz = z.data(), *hvx = vx.data(), *hvy = vy.data(),
                     *hvz = vz.data(), *hm = m.data();
        double tot = 0.0;
        for (int i = s; i < e; i++) {
            double t = 0.5 * hm[i] * (hvx[i] * hvx[i] + hvy[i] * hvy[i] + hvz[i] * hvz[i]);
            for (int j = i + 1; j < e; j++) {
                const double dx = hx[j] - hx[i], dy = hy[j] - hy[i], dz = hz[j] - hz[i];
                t -= G * hm[i] * hm[j] / std::sqrt(dx * dx + dy * dy + dz * dz);
            }
            tot += t;
        }
        return tot;
    }
    double tot = 0.0;
    Kokkos::parallel_reduce("ledger_energy", Kokkos::RangePolicy<>(s, e), KOKKOS_LAMBDA(int i, double& acc) {
        double t = 0.5 * m(i) * (vx(i) * vx(i) + vy(i) * vy(i) + vz(i) * vz(i));
        for (int j = i + 1; j < e; j++) {
            const double dx = x(j) - x(i), dy = y(j) - y(i), dz = z(j) - z(i);
            t -= G * m(i) * m(j) / Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
        }
        acc += t;
    }, tot);
    return tot;
}

#ifdef ENABLE_COLLISIONAL
// T + U among the members of each subsystem fully inside [start, start + count), by subsystem position in active
std::vector<double> member_energies(const Particles& p, int start, int count, double G, std::vector<char>* pn_flag = nullptr) {
    const auto& subs = g_subsystem_mgr.active;
    std::vector<double> out(subs.size(), std::nan(""));
    g_subsystem_mgr.rebuild_pid_to_idx(p);
    const auto& p2i = g_subsystem_mgr.pid_to_idx();
    const int e = start + count;
    std::vector<double> hx, hy, hz, hvx, hvy, hvz, hm;
    auto get = [&](const auto& v, std::vector<double>& dst) {
        dst.resize(count);
        if (host_exec()) { std::copy(v.data() + start, v.data() + e, dst.begin()); return; }
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(v, Kokkos::make_pair(start, e)));
        std::copy(h.data(), h.data() + count, dst.begin());
    };
    get(p.x, hx); get(p.y, hy); get(p.z, hz); get(p.vx, hvx); get(p.vy, hvy); get(p.vz, hvz); get(p.mass, hm);
    for (size_t s = 0; s < subs.size(); s++) {
        int idx[LOGH_MAX_BODIES];
        const int nb = std::min((int)subs[s].particle_ids.size(), LOGH_MAX_BODIES);
        bool in = nb >= 2;
        for (int b = 0; b < nb && in; b++) {
            const int pid = subs[s].particle_ids[b];
            const int i = (pid >= 0 && pid < (int)p2i.size()) ? p2i[pid] : -1;
            in = i >= start && i < e;
            idx[b] = i - start;
        }
        if (!in) continue;
        if (pn_flag) {
            if (pn_flag->size() != subs.size()) pn_flag->assign(subs.size(), 0);
            const bool pn_on = g_collisional.pn_order > 0 && g_collisional.c_code > 0.0;
            for (int b = 0; b < nb && pn_on; b++)
                if (g_collisional.pn_body(hm[idx[b]], StellarPropsCache::get().type_of(p, subs[s].particle_ids[b])))
                    (*pn_flag)[s] = 1;
        }
        double E = 0.0;
        for (int b = 0; b < nb; b++) {
            const int i = idx[b];
            E += 0.5 * hm[i] * (hvx[i] * hvx[i] + hvy[i] * hvy[i] + hvz[i] * hvz[i]);
            for (int c = b + 1; c < nb; c++) {
                const int j = idx[c];
                const double dx = hx[j] - hx[i], dy = hy[j] - hy[i], dz = hz[j] - hz[i];
                E -= G * hm[i] * hm[j] / std::sqrt(dx * dx + dy * dy + dz * dz);
            }
        }
        out[s] = E;
    }
    return out;
}
#endif
}  // namespace

// NBODY_ELEDGER: the subsystem advances with their members' energy change against the books (see EnergyLedger)
void advance_subsystems_in_range(Particles& p, int start, int count, double G, double dt) {
    static const bool ledger_on = std::getenv("NBODY_ELEDGER") != nullptr;
#ifdef ENABLE_COLLISIONAL
    if (!ledger_on || !p.collisional || g_subsystem_mgr.active.empty() || MPIComm::size() > 1 ||
        g_ledger.stack.empty()) {
        advance_subsystems_in_range_impl(p, start, count, G, dt);
        return;
    }
    const std::vector<Subsystem> before_subs = g_subsystem_mgr.active;
    std::vector<char> pnf(before_subs.size(), 0);
    const std::vector<double> e0 = member_energies(p, start, count, G, &pnf);
    const double b0 = g_energy_books.total_sink();
    const EnergyBookkeeping bk0 = g_energy_books;
    advance_subsystems_in_range_impl(p, start, count, G, dt);
    const double db = g_energy_books.total_sink() - b0;
    {
        const double er0 = g_ledger.eref > 0.0 ? g_ledger.eref : 1.0;
        const auto& b1 = g_energy_books;
        g_ledger.bk_gw += (b1.E_gw - bk0.E_gw) / er0; g_ledger.bk_pn += (b1.E_pn - bk0.E_pn) / er0;
        g_ledger.bk_sd += (b1.E_sd - bk0.E_sd) / er0; g_ledger.bk_merge += (b1.E_merge - bk0.E_merge) / er0;
        g_ledger.bk_other += ((b1.E_escape - bk0.E_escape) + (b1.E_sse - bk0.E_sse) + (b1.E_tidal - bk0.E_tidal)) / er0;
    }
    const double eref = g_ledger.eref > 0.0 ? g_ledger.eref : 1.0;
    // the same subsystems after the advance (by member list; the advance may reclassify, not regroup)
    const std::vector<double> e1 = member_energies(p, start, count, G);
    double dsum = 0.0;
    bool all = g_subsystem_mgr.active.size() == before_subs.size();
    for (size_t s = 0; s < before_subs.size() && all; s++) {
        if (std::isnan(e0[s])) continue;
        if (g_subsystem_mgr.active[s].particle_ids != before_subs[s].particle_ids || std::isnan(e1[s])) { all = false; break; }
        const double d = (e1[s] - e0[s]) / eref;
        dsum += d;
        EnergyLedger::SubRec r;
        r.dE = d; r.type = (int)before_subs[s].type; r.n = before_subs[s].n_bodies; r.secular = before_subs[s].is_secular;
        r.pids = before_subs[s].particle_ids;
        {
            const auto& sa = g_subsystem_mgr.active[s];   // classification after the advance (re-evaluated in it)
            const int nb = sa.n_bodies;
            int c;
            if (sa.is_secular && nb == 2) c = pnf[s] ? 1 : 0;
            else if (sa.is_secular && sa.type == SubsystemType::TRIPLE_STABLE) c = 2;
            else if (pnf[s]) c = 6;
            else c = nb == 2 ? 3 : nb == 3 ? 4 : 5;
            g_ledger.cls_dE[c] += d; g_ledger.cls_abs[c] += std::fabs(d); g_ledger.cls_n[c]++;
        }
        g_ledger.note_sub(std::move(r));
    }
    {
        const int dd = std::min((int)g_ledger.stack.size() - 1, EnergyLedger::MD - 1);
        if (all) g_ledger.depth_adv[std::max(dd, 0)] += dsum + db / eref;
        else g_ledger.adv_unattr += db / eref;
    }
    if (all) {
        g_ledger.adv_unbooked += dsum + db / eref;
    }
#else
    (void)ledger_on;
    advance_subsystems_in_range_impl(p, start, count, G, dt);
#endif
}

static int hhs_fsi_recursive(
    Particles& p,
    DistributedParticles* dp,
    int start, int count,
    double pivot_dt,
    double G, double soft2,
    double dt_min, double eta,
    bool dt_valid,
    bool global_tree_available,
    int depth) {
    static const bool ledger_on = std::getenv("NBODY_ELEDGER") != nullptr;
    if (!ledger_on || count == 0 || dp != nullptr || MPIComm::size() > 1)
        return hhs_fsi_recursive_impl(p, dp, start, count, pivot_dt, G, soft2, dt_min, eta, dt_valid,
                                      global_tree_available, depth);
#ifdef ENABLE_COLLISIONAL
    auto books = [] { return g_energy_books.total_sink(); };
    const double eref_run = g_collisional.e_ref;
#else
    auto books = [] { return 0.0; };
    const double eref_run = 0.0;
#endif
    const double e0 = set_energy(p, start, count, G) + books();
    if (g_ledger.stack.empty()) g_ledger.eref = eref_run > 0.0 ? eref_run : std::fabs(e0);   // |E0|, else |E| of the root set
    const double eref = g_ledger.eref > 0.0 ? g_ledger.eref : 1.0;
    g_ledger.stack.push_back({});
    const int k = hhs_fsi_recursive_impl(p, dp, start, count, pivot_dt, G, soft2, dt_min, eta, dt_valid,
                                         global_tree_available, depth);
    EnergyLedger::Frame fr = g_ledger.stack.back();
    g_ledger.stack.pop_back();
    g_ledger.calls++;
    const double dE = (set_energy(p, start, count - k, G) + books() - e0) / eref;
    if (k == 0 && fr.valid) {
        EnergyLedger::Rec r;
        r.depth = depth; r.n = count; r.pivot = pivot_dt * TIME_UNIT_MYR; r.total = dE; r.own = dE - fr.child;
        {
            const int dd = std::min((int)g_ledger.stack.size(), EnergyLedger::MD - 1);   // frame depth (root 0)
            g_ledger.depth_own[dd] += r.own; g_ledger.depth_calls[dd]++;
            g_ledger.depth_abs[dd] += std::fabs(r.own);
            const int t = std::min(std::max(fr.type, 0), EnergyLedger::NT - 1);
            g_ledger.type_own[t] += r.own; g_ledger.type_abs[t] += std::fabs(r.own); g_ledger.type_calls[t]++;
        }
        if (count <= 32) {
            if (host_exec()) {
                r.pids.assign(p.particle_id.data() + start, p.particle_id.data() + start + count);
            } else {
                auto hp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                              Kokkos::subview(p.particle_id, Kokkos::make_pair(start, start + count)));
                r.pids.assign(hp.data(), hp.data() + count);
            }
        }
        g_ledger.note(std::move(r));
    }
    if (!g_ledger.stack.empty()) {
        g_ledger.stack.back().child += dE;
        if (k != 0 || !fr.valid) g_ledger.stack.back().valid = false;
    } else {
        std::printf("ELEDGER root dE/E0 %.6e (sum of own errors %.6e, %d calls%s)\n", dE, g_ledger.sum_own,
                    g_ledger.calls, fr.valid && k == 0 ? "" : ", merger inside: not attributed");
        for (const auto& r : g_ledger.top) {
            std::printf("ELEDGER own %.6e total %.6e depth %d n %d pivot %.3e Myr pids", r.own, r.total, r.depth, r.n,
                        r.pivot);
            for (int pid : r.pids) std::printf(" %d", pid);
            std::printf("\n");
        }
        std::printf("ELEDGER depth (own error incl. subsystem residue made there | subsystem residue | calls):");
        for (int d = 0; d < EnergyLedger::MD; d++)
            if (g_ledger.depth_calls[d] > 0 || g_ledger.depth_adv[d] != 0.0)
                std::printf(" d%d %.4e %.4e %d", d, g_ledger.depth_own[d], g_ledger.depth_adv[d], g_ledger.depth_calls[d]);
        std::printf(" unattr_adv_books %.4e\n", g_ledger.adv_unattr);
        std::printf("ELEDGER depth_abs (sum |own| per depth):");
        for (int d = 0; d < EnergyLedger::MD; d++)
            if (g_ledger.depth_calls[d] > 0) std::printf(" d%d %.4e", d, g_ledger.depth_abs[d]);
        std::printf("\n");
        {
            static const char* tn[EnergyLedger::NT] = {"other", "mindt", "allslow", "allfast", "mixed", "onesub", "cc", "hostsub"};
            std::printf("ELEDGER types (sum own | sum |own| | calls):");
            for (int t = 0; t < EnergyLedger::NT; t++)
                std::printf(" %s %.4e %.4e %d", tn[t], g_ledger.type_own[t], g_ledger.type_abs[t], g_ledger.type_calls[t]);
            std::printf("\n");
        }
        std::printf("ELEDGER classes (sum dE_int | sum |dE_int| | n): sec2 %.4e %.4e %lld  sec2pn %.4e %.4e %lld  sec3 %.4e %.4e %lld"
                    "  logh2 %.4e %.4e %lld  logh3 %.4e %.4e %lld  logh4p %.4e %.4e %lld  loghpn %.4e %.4e %lld\n",
                    g_ledger.cls_dE[0], g_ledger.cls_abs[0], g_ledger.cls_n[0], g_ledger.cls_dE[1], g_ledger.cls_abs[1],
                    g_ledger.cls_n[1], g_ledger.cls_dE[2], g_ledger.cls_abs[2], g_ledger.cls_n[2], g_ledger.cls_dE[3],
                    g_ledger.cls_abs[3], g_ledger.cls_n[3], g_ledger.cls_dE[4], g_ledger.cls_abs[4], g_ledger.cls_n[4],
                    g_ledger.cls_dE[5], g_ledger.cls_abs[5], g_ledger.cls_n[5], g_ledger.cls_dE[6], g_ledger.cls_abs[6],
                    g_ledger.cls_n[6]);
        std::printf("ELEDGER books in advances: gw %.4e pn %.4e sd %.4e merge %.4e other %.4e\n", g_ledger.bk_gw,
                    g_ledger.bk_pn, g_ledger.bk_sd, g_ledger.bk_merge, g_ledger.bk_other);
        std::printf("ELEDGER subsystem advances: members' energy change + books change %.6e (types: 0 binary, 1 fly-by, "
                    "2 stable triple, 3 unstable triple, 4 multiple)\n", g_ledger.adv_unbooked);
        {
            const double er = g_ledger.eref > 0.0 ? g_ledger.eref : 1.0;
            std::printf("ELEDGER path residues: secular binaries %.6e (%d calls; PN ones %.6e)  secular triples %.6e (%d calls)\n",
                        g_led_sec2 / er, g_led_n_sec2, g_led_sec2_pn / er, g_led_sec3 / er, g_led_n_sec3);
            g_led_sec2 = g_led_sec2_pn = g_led_sec3 = 0.0;
            g_led_n_sec2 = g_led_n_sec3 = 0;
        }
        for (const auto& r : g_ledger.sub_top) {
            std::printf("ELEDGER sub dE_int %.6e type %d n %d secular %d pids", r.dE, r.type, r.n, (int)r.secular);
            for (int pid : r.pids) std::printf(" %d", pid);
            std::printf("\n");
        }
        std::fflush(stdout);
        g_ledger = EnergyLedger{};
    }
    return k;
}

static int hhs_fsi_recursive_impl(
    Particles& p,
    DistributedParticles* dp,
    int start, int count,
    double pivot_dt,
    double G, double soft2,
    double dt_min, double eta,
    bool dt_valid,
    bool global_tree_available,
    int depth) {

    if (count == 0) return 0;
    int n_removed = 0;  // merged-away particles; [start, start+count) is always a suffix of [0, N)
#ifdef ENABLE_COLLISIONAL
    // Placeholders of bodies merged inside subsystem integrations (finalize_subsystem_collision) leave before this
    // range computes time steps or detects subsystems. Only [start, start + count) is compacted, the tail shifts
    // down: what the callers expect of a merger in this range (HHS_CC components included)
    if (p.collisional && range_has_ghost(p, start, count)) {
        const int k = compact_zero_mass_range(p, start, count);
        if (k > 0) {
            sync_after_removal(p, dp);
            g_subsystem_mgr.update_particle_flags(p);
            n_removed += k;
            count -= k;
            if (count <= 0) return n_removed;
        }
    }
#endif
    struct DepthScope {   // profiler charges work to this depth until the call returns
        int saved;
        explicit DepthScope(int d) : saved(g_profiler.cur_depth) {
            g_profiler.cur_depth = d;
#if defined(NBODY_NVTX) && defined(KOKKOS_ENABLE_CUDA)
            char nv[16]; std::snprintf(nv, sizeof nv, "D%02d", d); NBODY_NVTX_PUSH(nv);   // per-depth NVTX range
#endif
        }
        ~DepthScope() { NBODY_NVTX_POP(); g_profiler.cur_depth = saved; }
    } depth_scope(depth);

    // HHS_CC splits sets of at most cc_max particles; they run locally on every rank (replicated), so that the
    // integration does not depend on the number of ranks
    const int cc_max = hhs_cc_max >= 0 ? hhs_cc_max : std::max(serial_n2_gpu, 16384);
    const bool cc_set = hhs_cc && count <= cc_max;
    // Dispatch flags: MPI for large subsets when distributed
    bool use_mpi = dp && (serial_n2_gpu == 0 || count > serial_n2_gpu) && !cc_set;
    bool tree_eligible = dp && use_tree && count >= tree_threshold;

    // Host-resident subtree (HOST_SUBTREE_MAX): the whole sub-recursion of this set on the host
    if (host_subtree_eligible(p, count, use_mpi, tree_eligible)) {
        g_hhs_counters.cpu++;
        ledger_type(7);
        return n_removed + hhs_fsi_host_subtree(p, dp, start, count, pivot_dt, G, soft2, dt_min, eta, dt_valid, depth);
    }

    // Counters
    g_profiler.record_set(depth, count);
    if (use_mpi) g_hhs_counters.distributed++;
    else g_hhs_counters.local++;
    if (depth > g_hhs_counters.max_depth) g_hhs_counters.max_depth = depth;
    if (depth < HHSCounters::MAX_DEPTH) {
        if (use_mpi) {
            g_hhs_counters.dist_depth_calls[depth]++;
            g_hhs_counters.dist_depth_count[depth] += count;
        } else {
            g_hhs_counters.local_depth_calls[depth]++;
            g_hhs_counters.local_depth_count[depth] += count;
        }
    }
    g_subset_hist.record(use_mpi ? SubsetHistogram::DISTRIBUTED : SubsetHistogram::LOCAL_GPU, count);


    // Base case: at minimum timestep, use FSI on the subset
    if (pivot_dt * 0.5 < dt_min) {
        ledger_type(1);
        if (dp && global_tree_available && tree_eligible) {
            g_profiler.start();
            g_octree.update_subsets(p, start, count);
            g_profiler.stop(StepProfiler::TREE_MAINT);
            fsi_step_distributed(*dp, start, count, G, soft2, pivot_dt, false, false, 0);
        } else if (use_mpi) {
            fsi_step_distributed(*dp, start, count, G, soft2, pivot_dt);
        } else {
            fsi_step(p, start, count, G, soft2, pivot_dt, false, true);
        }
        return 0;
    }

    // Rantala et al. 2023, Section 3.5: fresh neighbor search + detection, then timestep
    bool forces_valid = false;
    bool tree_built_for_subset = false;
    // A merger detected below (per-level, Rantala et al. 2023, sec. 3.14) leaves the remnant with the time step of its progenitors
    // although its mass and velocity changed (GW recoil kick, up to ~1e3 km/s): the set's neighbours, subsystems and
    // time steps are computed again with the remnant. With FLYBY_CM_VELOCITY the progenitors' step is the c.o.m.
    // fly-by step (slow set), and a kicked remnant crossed 0.1 pc in one step of the slow set (200k seed 21 step 38:
    // dE/E -1.55e-9, docs/CASES.md C13); with component velocities it was the fast set's, recomputed one depth down.
    if (depth == 0) g_dt_root = pivot_dt;   // NBODY_DT_SCALE deep<L> / shallow<L>, NBODY_DT_ATTR levels
    for (int dt_pass = 0; !dt_valid && count > 1; dt_pass++) {
        bool merged_here = false;
#ifdef ENABLE_COLLISIONAL
        // Step 1: Standalone neighbor search + detection (before timestep)
        if (p.collisional) {
            g_profiler.start();
            PhaseClock det_clock;
            g_subsystem_mgr.reset_ngb_count();
            neighbor_search_subset(p, start, count, G,
                                   g_collisional.r_ngb, g_collisional.r_ngb_max, pivot_dt);
            lean_fence("integrator.cpp:2421");
            g_subsys_phase.add(SubsystemPhaseTimes::DET_NGB, det_clock.lap());
            g_subsystem_mgr.detect_from_pairs(p, G, pivot_dt,
                                               g_collisional.r_ngb, g_collisional.r_ngb_max,
                                               g_collisional.n_orb_secular);
            g_subsys_phase.add(SubsystemPhaseTimes::DET_PAIRS, det_clock.lap());
            g_subsystem_mgr.update_particle_flags(p);
            g_subsystem_mgr.accumulate_diagnostics();
            lean_fence("integrator.cpp:2429");
            g_subsys_phase.add(SubsystemPhaseTimes::DET_FLAGS, det_clock.lap());
            if (depth < HHSCounters::MAX_DEPTH)
                g_hhs_counters.dist_depth_ngb_pairs[depth] += g_subsystem_mgr.last_level_pairs;
            g_profiler.stop(StepProfiler::SUBSYSTEM_DETECT);
        }
#endif
        // Step 2: Timestep computation (with subsystem exclusion active)
        g_profiler.start();
        if (global_tree_available && tree_eligible) {
            g_octree.update_subsets(p, start, count);
            tree_global_self_acceleration_and_timestep(p, g_octree, 0, G, soft2, eta);
            tree_momentum_correction(p, start, count);
            forces_valid = true;
        } else if (!global_tree_available && tree_eligible) {
            g_octree.n_leaf = tree_n_leaf;
            g_octree.grad_dipole = tree_grad_dipole;
            g_octree.n_group = tree_n_group;
            g_octree.build(p, start, count, tree_theta);
            int my_g_off, my_g_cnt;
            get_tree_group_range(g_octree, my_g_off, my_g_cnt);
            zero_accelerations(p, start, count);
            init_timesteps_max(p, start, count);
            tree_self_acceleration_and_timestep(p, g_octree, start, count, G, soft2, eta, my_g_off, my_g_cnt);
            dp->allreduce_accelerations_subset(start, count);
            dp->allreduce_timestep_subset(start, count);
            tree_momentum_correction(p, start, count);
            forces_valid = true;
            tree_built_for_subset = true;
        } else if (use_mpi) {
            p.pack_pos_range(start, count);
            p.pack_vel_range(start, count);
            compute_timestep_distributed(*dp, start, count, G, soft2, eta);
        } else {
            compute_timestep_subset(p, start, count, G, soft2, eta);
        }
        if (depth == 0) dt_attr_snap(p, start, count, g_dta_kern);   // NBODY_DT_ATTR (off: no-op)
#ifdef ENABLE_COLLISIONAL
        // Gradient + v/a timestep criteria (Rantala et al. 2023, Section 3.4.4, Eq. 24/26/27), each off at eta = 0
        if (p.collisional && (g_collisional.eta_grad > 0.0 || g_collisional.eta_va > 0.0)) {
            // GRAD_FRESH (fix 3, b7dfabe): |g'| of every particle from the current accelerations at level 0 (the set is
            // every particle, the sum over all j != i of Rantala et al. 2023); deeper levels inherit the step through the level-0
            // clamp (sec. 3.4.1). Off: grad_mag as the last gradient kick left it (partial: self or slow-fast)
            const bool fresh = g_collisional.grad_fresh && g_collisional.eta_grad > 0.0;
            if (fresh && depth == 0) {
                if (use_mpi) {
                    int s0 = start, sc = count;
                    dp->get_force_slice(start, count, s0, sc);
                    n2_detail::SinkSetScope n2_set_gm(count);
                    gradient_magnitude_range(p, s0, sc, start, count, G, soft2);
                    Kokkos::fence();
                    const int f[1] = {4};
                    dp->allgather_fields(&start, &count, 1, f, 1);
                } else {
                    gradient_magnitude_range(p, start, count, start, count, G, soft2);
                }
            }
            auto dt_f = raw(p.dt_adapt);
            auto gm = raw(p.grad_mag);
            RawView agx{p.ax.data()}, agy{p.ay.data()}, agz{p.az.data()};
            RawView vvx{p.vx.data()}, vvy{p.vy.data()}, vvz{p.vz.data()};
            double eta_g = (fresh && depth > 0) ? 0.0 : g_collisional.eta_grad;
            double eta_va = g_collisional.eta_va;   // halved by the restart loop like the other etas
            small_for("dt_grad", count, KOKKOS_LAMBDA(int ii) {
                int i = start + ii;
                double a_mag = Kokkos::sqrt(agx(i)*agx(i) + agy(i)*agy(i) + agz(i)*agz(i));
                double v_mag = Kokkos::sqrt(vvx(i)*vvx(i) + vvy(i)*vvy(i) + vvz(i)*vvz(i));
                double g = gm(i);
                if (eta_g > 0.0 && g > EPS) {
                    if (a_mag > EPS) {
                        double dt_g2 = eta_g * Kokkos::sqrt(a_mag / g);
                        if (dt_g2 < dt_f(i)) dt_f(i) = dt_g2;
                    }
                    if (v_mag > EPS) {
                        double dt_g1 = eta_g * Kokkos::cbrt(v_mag / g);
                        if (dt_g1 < dt_f(i)) dt_f(i) = dt_g1;
                    }
                }
                if (eta_va > 0.0 && a_mag > EPS && v_mag > EPS) {
                    double dt_va = eta_va * v_mag / a_mag;
                    if (dt_va < dt_f(i)) dt_f(i) = dt_va;
                }
            });
        }
        // NBODY_DUMP_DT=n (debug): pid, dt_adapt, grad_mag, ax of the set after the time-step criteria, first n calls
        {
            static int dump_calls = 0;
            static const int dump_max = std::getenv("NBODY_DUMP_DT") ? std::atoi(std::getenv("NBODY_DUMP_DT")) : 0;
            if (dump_calls < dump_max && MPIComm::rank() == 0) {
                auto r = Kokkos::make_pair(start, start + count);
                auto hp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.particle_id, r));
                auto hd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.dt_adapt, r));
                auto hg = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.grad_mag, r));
                auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.ax, r));
                char fn[64]; std::snprintf(fn, sizeof fn, "dtdump_%05d.txt", dump_calls);
                if (FILE* f = std::fopen(fn, "w")) {
                    for (int k = 0; k < count; k++) std::fprintf(f, "%d %a %a %a\n", hp(k), hd(k), hg(k), ha(k));
                    std::fclose(f);
                }
            }
            dump_calls++;
        }
        // STALL_FACTOR: the first set at the deepest depth of the step (named if the step stalls)
        if (g_stall.deeper(depth) && count <= StallRecorder::MAX_DEEP && MPIComm::rank() == 0 &&
            (int)p.particle_id.extent(0) >= start + count) {
            auto hp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                          Kokkos::subview(p.particle_id, Kokkos::make_pair(start, start + count)));
            g_stall.note_deep(depth, pivot_dt * TIME_UNIT_MYR, std::vector<int>(hp.data(), hp.data() + count));
        }
        // NBODY_DEEP_LOG=d (debug), at depth 0: every particle with dt below the depth-d pivot and its 4 nearest
        // neighbours (pid, mass, subsystem id, separation, relative speed): the pairs that set the level-0 steps
        {
            static const int deep_log0 = std::getenv("NBODY_DEEP_LOG") ? std::atoi(std::getenv("NBODY_DEEP_LOG")) : 0;
            if (deep_log0 > 0 && depth == 0 && MPIComm::rank() == 0) {
                const double thr = pivot_dt / std::ldexp(1.0, deep_log0);
                auto r = Kokkos::make_pair(start, start + count);
                auto cp = [&](auto v) { return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(v, r)); };
                auto hd = cp(p.dt_adapt);
                std::vector<int> low;
                for (int k = 0; k < count && low.size() < 10; k++) if (hd(k) < thr) low.push_back(k);
                if (!low.empty()) {
                    auto hp = cp(p.particle_id); auto hm = cp(p.mass);
                    auto hx = cp(p.x); auto hy = cp(p.y); auto hz = cp(p.z);
                    auto hvx = cp(p.vx); auto hvy = cp(p.vy); auto hvz = cp(p.vz);
#ifdef ENABLE_COLLISIONAL
                    auto hs = cp(p.subsystem_id);
#endif
                    for (int k : low) {
                        std::vector<std::pair<double, int>> nb;
                        for (int j = 0; j < count; j++) {
                            if (j == k) continue;
                            const double dx = hx(j) - hx(k), dy = hy(j) - hy(k), dz = hz(j) - hz(k);
                            nb.push_back({dx*dx + dy*dy + dz*dz, j});
                        }
                        std::partial_sort(nb.begin(), nb.begin() + std::min<size_t>(4, nb.size()), nb.end());
                        int sk = -1;
#ifdef ENABLE_COLLISIONAL
                        sk = hs(k);
#endif
                        std::printf("DEEP0 pid %d m %.4g dt %.3e Myr sid %d:", hp(k), hm(k), hd(k) * TIME_UNIT_MYR, sk);
                        for (size_t q = 0; q < std::min<size_t>(4, nb.size()); q++) {
                            const int j = nb[q].second;
                            const double ux = hvx(j) - hvx(k), uy = hvy(j) - hvy(k), uz = hvz(j) - hvz(k);
                            int sj = -1;
#ifdef ENABLE_COLLISIONAL
                            sj = hs(j);
#endif
                            std::printf(" | pid %d m %.4g sid %d dt %.3e r %.3e v %.3e", hp(j), hm(j), sj,
                                        hd(j) * TIME_UNIT_MYR, std::sqrt(nb[q].first), std::sqrt(ux*ux + uy*uy + uz*uz));
                        }
                        std::printf("\n");
                    }
                    std::fflush(stdout);
                }
            }
        }
        // NBODY_DEEP_LOG=d (debug): the first 4 sets at each depth >= d (count <= 32): members with mass, dt, subsystem
        // id, and every pair's separation, relative speed and two-body energy (what keeps the set this deep)
        {
            static const int deep_log = std::getenv("NBODY_DEEP_LOG") ? std::atoi(std::getenv("NBODY_DEEP_LOG")) : 0;
            static int deep_seen[64] = {};
            if (deep_log > 0 && depth >= deep_log && depth < 64 && count <= 32 && deep_seen[depth] < 4 &&
                MPIComm::rank() == 0) {
                deep_seen[depth]++;
                auto r = Kokkos::make_pair(start, start + count);
                auto cp = [&](auto v) { return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(v, r)); };
                auto hp = cp(p.particle_id); auto hd = cp(p.dt_adapt); auto hm = cp(p.mass); auto hl = cp(p.dt_level0);
                auto hx = cp(p.x); auto hy = cp(p.y); auto hz = cp(p.z);
                auto hvx = cp(p.vx); auto hvy = cp(p.vy); auto hvz = cp(p.vz);
#ifdef ENABLE_COLLISIONAL
                auto hs = cp(p.subsystem_id);
#endif
                std::printf("DEEP depth %d pivot %.3e Myr count %d\n", depth, pivot_dt * TIME_UNIT_MYR, count);
                for (int k = 0; k < count; k++) {
                    int sid = -1;
#ifdef ENABLE_COLLISIONAL
                    sid = hs(k);
#endif
                    std::printf("DEEP   pid %d m %.4g dt %.3e dt_level0 %.3e Myr sid %d x %.9e %.9e %.9e\n", hp(k), hm(k),
                                hd(k) * TIME_UNIT_MYR, hl(k) * TIME_UNIT_MYR, sid, hx(k), hy(k), hz(k));
                }
                for (int a = 0; a < count; a++)
                    for (int c = a + 1; c < count; c++) {
                        const double dx = hx(c) - hx(a), dy = hy(c) - hy(a), dz = hz(c) - hz(a);
                        const double ux = hvx(c) - hvx(a), uy = hvy(c) - hvy(a), uz = hvz(c) - hvz(a);
                        const double rr = std::sqrt(dx*dx + dy*dy + dz*dz), vv = std::sqrt(ux*ux + uy*uy + uz*uz);
                        const double eps = 0.5 * vv * vv - G * (hm(a) + hm(c)) / rr;
                        std::printf("DEEP   pair %d-%d r %.3e pc v %.3e km/s eps %.3e a %.3e pc\n", hp(a), hp(c), rr, vv,
                                    eps, eps < 0.0 ? -G * (hm(a) + hm(c)) / (2.0 * eps) : 0.0);
                    }
                std::fflush(stdout);
            }
        }
        // Level-0 clamping (Rantala et al. 2023, Section 3.4.1)
        if (p.collisional && depth > 0) {
            auto dt_f = raw(p.dt_adapt);
            auto dt_l0 = raw(p.dt_level0);
            int n_clamped = 0;
            xreduce("clamp_dt_level0", count,
                KOKKOS_LAMBDA(int ii, int& cnt) {
                    int i = start + ii;
                    if (dt_f(i) > dt_l0(i)) {
                        dt_f(i) = dt_l0(i);
                        cnt++;
                    }
                }, n_clamped);
            if (depth < HHSCounters::MAX_DEPTH) g_hhs_counters.dist_depth_l0_clamped[depth] += n_clamped;
        }
#endif
        g_profiler.stop(StepProfiler::ADAPTIVE_DT, (long long)count * count);
#ifdef ENABLE_COLLISIONAL
        // Step 3: Subsystem timestep adjustments
        if (p.collisional) {
            g_profiler.start();
            PhaseClock dt_clock;
            if (g_profiler.fence) Kokkos::fence();   // profile: the time-step kernels above are asynchronous
            g_subsys_phase.add(SubsystemPhaseTimes::DT_WAIT, dt_clock.lap());
            slow0().record_pre(p, start, count);
            if (depth == 0) dt_attr_snap(p, start, count, g_dta_pre);   // NBODY_DT_ATTR (off: no-op)
            g_subsystem_mgr.subsystem_timesteps(p, G, g_collisional.eta_pert, depth == 0);
            if (depth == 0) dt_attr_report(p, start, count, G, soft2);   // NBODY_DT_ATTR (off: no-op)
            dt_scale_apply(p, start, count);                            // NBODY_DT_SCALE (unset: no-op)
            g_subsys_phase.add(SubsystemPhaseTimes::DT_SUBSYS, dt_clock.lap());

            // Level-0 store: capture dt AFTER subsystem adjustments
            if (depth == 0) {
                auto dt_f = raw(p.dt_adapt);
                auto dt_l0 = raw(p.dt_level0);
                small_for("copy_dt_level0", count, KOKKOS_LAMBDA(int ii) {
                    int i = start + ii;
                    dt_l0(i) = dt_f(i);
                });
            }

            // Merger detection at every HHS level (Rantala et al. 2023, §3.14)
            if (g_collisional.c_code > 0.0) {
                auto mergers = detect_mergers(g_subsystem_mgr, p, G,
                                               g_collisional.c_code, pivot_dt);
                for (const auto& ev : mergers) {   // execute_merger records the event in the merger tree
                    static const bool merge_check = std::getenv("NBODY_MERGE_CHECK") != nullptr;   // debug
                    double e_chk0 = 0.0;
                    if (merge_check)
                        e_chk0 = compute_diagnostics(p, G, soft2).E_total + g_energy_books.total_sink();
                    execute_merger(ev, p, g_subsystem_mgr, g_energy_books,
                                   G, g_collisional.c_code);
                    if (merge_check && MPIComm::rank() == 0) {
                        const double e_chk1 = compute_diagnostics(p, G, soft2).E_total + g_energy_books.total_sink();
                        std::printf("MERGE_CHECK depth %d pids %d %d dE(total+books) %.6e E_merge %.9e E_pn %.9e\n",
                                    depth, ev.pid1, ev.pid2, e_chk1 - e_chk0, g_energy_books.E_merge,
                                    g_energy_books.E_pn);
                    } else if (merge_check) {
                        compute_diagnostics(p, G, soft2);   // collective under MPI
                    }
                }
                g_subsystem_mgr.step_mergers += (int)mergers.size();
                if (!mergers.empty()) {
                    merged_here = true;
                    // Compaction of this range only; the tail shifts down, which callers see as a shorter range
                    int k = compact_zero_mass_range(p, start, count);
                    sync_after_removal(p, dp);
                    n_removed += k;
                    count -= k;
                    if (count <= 0) {
                        g_subsys_phase.add(SubsystemPhaseTimes::DT_MERGE, dt_clock.lap());
                        g_profiler.stop(StepProfiler::SUBSYSTEM_DT);
                        return n_removed;
                    }
                    g_subsystem_mgr.update_particle_flags(p);
                }
            }
            g_subsys_phase.add(SubsystemPhaseTimes::DT_MERGE, dt_clock.lap());
            g_profiler.stop(StepProfiler::SUBSYSTEM_DT);
        }
#endif
        if (!merged_here || dt_pass >= 3) break;
    }

#ifdef ENABLE_COLLISIONAL
    // A set that is one subsystem (or one particle) has no interaction left for the recursion to resolve: the
    // members' mutual force is the subsystem's (LogH), and external perturbations reach it only from the parent
    // levels. Recursing would only cut its LogH integration into 2^d pieces; one FSI step over the pivot (drift and
    // one subsystem advance) is the same Hamiltonian, exact to the GBS tolerance.
    if (p.collisional && count <= LOGH_MAX_BODIES && pivot_dt * 0.5 >= dt_min) {
        bool one_subsystem = (count == 1);
        if (!one_subsystem) {
            int s0 = -2, same = 1;
            auto h_sid = host_copy(Kokkos::subview(p.subsystem_id, Kokkos::make_pair(start, start + count)));
            for (int k = 0; k < count; k++) {
                int sk = h_sid(k);
                if (sk < 0 || (s0 != -2 && sk != s0)) { same = 0; break; }
                s0 = sk;
            }
            // Not for a triple: whether a (stable, re-evaluated in the advance) triple goes secular depends on the
            // step it is advanced with (inner orbits per dt), so the longer pivot would hand it to the orbit-averaged
            // evolution where the recursion uses LogH (200k, 10 % binaries: |dE/E| 5e-10 -> 3e-9 within three steps
            // of the first secular triple)
            one_subsystem = same && count != 3;
        }
        if (one_subsystem) {
            ledger_type(5);
            if (depth < HHSCounters::MAX_DEPTH) g_hhs_counters.isolated_sets[depth]++;
            fsi_step(p, start, count, G, soft2, pivot_dt, false, true);
            return n_removed;
        }
    }
#endif

    // Partition into slow (dt >= pivot) and fast (dt < pivot)
    g_profiler.start();
    int slow_count = partition_by_pivot(p, start, count, pivot_dt);
    g_profiler.stop(StepProfiler::PARTITION);
    int slow_start = start;
    int fast_start = start + slow_count;
    int fast_count = count - slow_count;

    // Tree availability for sub-calls (either built here or inherited)
    bool tree_for_subs = global_tree_available || tree_built_for_subset;

    // Base case: all particles are slow — reuse fused forces for first FSI kick
    if (fast_count == 0) {
        ledger_type(2);
        // FROST Algorithm 3 with an empty fast set: fsi(S, tau/2) twice, on every path (one step of tau: 16x the
        // energy error at the same time, 6655494)
        const double h = pivot_dt * 0.5;
        if (dp && global_tree_available && tree_eligible) {
            for (int k = 0; k < 2; k++)
                fsi_step_distributed(*dp, start, count, G, soft2, h, k == 0 && forces_valid, false, 0);
        } else if (use_mpi) {
            for (int k = 0; k < 2; k++)
                fsi_step_distributed(*dp, start, count, G, soft2, h, k == 0 && forces_valid,
                                     k == 0 && tree_built_for_subset);
        } else {
            fsi_step(p, start, count, G, soft2, pivot_dt * 0.5, forces_valid, true);
            fsi_step(p, start, count, G, soft2, pivot_dt * 0.5, false, true);
        }
        return n_removed;
    }

    // HHS_CC: split the fast set into components when it falls apart (local path only)
    if (cc_set && !tree_eligible && fast_count >= 2) {
        g_profiler.start();
        std::vector<int> comps = cc_components(p, fast_start, fast_count, pivot_dt, G, soft2, eta, hhs_cc_min);
        g_profiler.stop(StepProfiler::PARTITION);
        if (depth < HHSCounters::MAX_DEPTH) {
            g_hhs_counters.cc_calls[depth]++;
            g_hhs_counters.cc_comps[depth] += (long long)comps.size();
            g_hhs_counters.cc_largest[depth] += *std::max_element(comps.begin(), comps.end());
        }
        if (comps.size() > 1) ledger_type(6);
        if (comps.size() > 1)
            return n_removed + hhs_cc_step(p, dp, slow_start, slow_count, fast_start, std::move(comps), pivot_dt,
                                           G, soft2, dt_min, eta, depth);
    }

    // Base case: all particles are fast
    if (slow_count == 0) {
        ledger_type(3);
        if (use_mpi) g_hhs_counters.slow0_dist++;
        else g_hhs_counters.slow0_local++;
#ifdef ENABLE_COLLISIONAL
        if (p.collisional) slow0().record_set(p, start, count, pivot_dt, G);
#endif
        double sub_pivot = pivot_dt * 0.5;
        int k = hhs_fsi_recursive(p, dp, start, count, sub_pivot,
                                  G, soft2, dt_min, eta, true, tree_for_subs, depth + 1);
        n_removed += k;
        count -= k;
        n_removed += hhs_fsi_recursive(p, dp, start, count, sub_pivot,
                                       G, soft2, dt_min, eta, false, tree_for_subs, depth + 1);
        return n_removed;
    }

    ledger_type(4);
    long long cross_pairs = 2LL * slow_count * fast_count;

    // Re-sync packed positions after partition reorder (only needed for MPI)
    if (use_mpi) {
        g_profiler.start();
        p.pack_pos_range(start, count);
        g_profiler.stop(StepProfiler::PACK_DATA);
    }

    // ==========================================================================
    // ALGORITHM 4: HHS-FSI with slow (S) and fast (F) particles
    // ==========================================================================

    // Tree flags for cross-force and self-force
    bool use_tree_cross = tree_for_subs && use_tree && count >= tree_threshold;
    bool use_tree_self = tree_for_subs && use_tree && slow_count >= tree_threshold;

    // Mark slow/fast/other in global tree
    if (use_tree_cross || use_tree_self) {
        g_profiler.start();
        if (use_tree_cross)
            g_octree.update_subsets_3flag(p, start, count, slow_count);
        else
            g_octree.update_subsets(p, slow_start, slow_count);
        g_profiler.stop(StepProfiler::TREE_MAINT);
    }

#ifdef ENABLE_COLLISIONAL
    // Save subsystem state before recursive calls can overwrite it
    auto saved_active = g_subsystem_mgr.active;
#endif

    // Step 1: Cross-force S<->F then kick (dt/6)
    g_profiler.start();
    if (use_tree_cross) {
        cross_acceleration_global_tree(p, slow_start, slow_count,
                                        fast_start, fast_count, G, soft2);
    } else if (use_mpi) {
        p.pack_pos_range(slow_start, slow_count + fast_count);
        cross_acceleration_distributed(*dp, slow_start, slow_count, fast_start, fast_count, G, soft2);
    } else {
        cross_acceleration(p, slow_start, slow_count, fast_start, fast_count, G, soft2);
    }
    g_profiler.stop(StepProfiler::CROSS_FORCE, cross_pairs);
    g_profiler.start();
    force_err_inject(p, slow_start, slow_count + fast_count, slow_count + fast_count);
    {
#ifdef ENABLE_COLLISIONAL
        cm_tide::KickSrcScope cmt_src(fast_start, fast_count, pivot_dt, G);   // cross kicks: sources = the other set
#endif
        kick_velocities(p, slow_start, slow_count, pivot_dt * FSI_KICK);
    }
    {
#ifdef ENABLE_COLLISIONAL
        cm_tide::KickSrcScope cmt_src(slow_start, slow_count, pivot_dt, G);
#endif
        kick_velocities(p, fast_start, fast_count, pivot_dt * FSI_KICK);
    }
    g_profiler.stop(StepProfiler::DRIFT_KICK);

    // Step 2: FSI half-step on slow
    if (use_tree_self || use_tree_cross) {
        fsi_step_distributed(*dp, slow_start, slow_count, G, soft2, pivot_dt * 0.5,
                             false, false, 0);
    } else if (use_mpi) {
        fsi_step_distributed(*dp, slow_start, slow_count, G, soft2, pivot_dt * 0.5);
    } else {
        fsi_step(p, slow_start, slow_count, G, soft2, pivot_dt * 0.5, false, true);
    }

    // Step 3: Recurse on fast (first half) — always compute fresh timesteps
    int k = hhs_fsi_recursive(p, dp, fast_start, fast_count, pivot_dt * 0.5,
                              G, soft2, dt_min, eta, false, tree_for_subs, depth + 1);
    n_removed += k;
    fast_count -= k;
    count -= k;

    // Restore subsystem state after recursion overwrote it
#ifdef ENABLE_COLLISIONAL
    g_profiler.start();
    g_subsystem_mgr.restore_active(saved_active);
    // The flags go by particle id; partitions below moved particles since the map was last built
    if (!g_subsystem_mgr.active.empty()) g_subsystem_mgr.rebuild_pid_to_idx(p);
    g_subsystem_mgr.update_particle_flags(p);
    g_profiler.stop(StepProfiler::SUBSYSTEM_BOOKKEEP);
#endif

    // Restore tree state after steps 2-3
    if (use_tree_cross || use_tree_self) {
        g_profiler.start();
        if (use_tree_cross)
            g_octree.update_subsets_3flag(p, start, count, slow_count);
        else
            g_octree.update_subsets(p, slow_start, slow_count);
        g_profiler.stop(StepProfiler::TREE_MAINT);
    }

    // Step 4: Cross-kick S<->F with gradient (2dt/3)
    // Re-sync packed positions: step 3's recursion may have run locally (use_mpi=false)
    // modifying SoA positions via drifts without updating pos_packed.
    if (use_mpi) {
        g_profiler.start();
        p.pack_pos_range(slow_start, slow_count + fast_count);
        g_profiler.stop(StepProfiler::PACK_DATA);
    }
    g_profiler.start();
    if (use_tree_cross) {
        cross_acceleration_and_gradient_global_tree(p, slow_start, slow_count,
                                                     fast_start, fast_count, G, soft2, pivot_dt);
    } else if (use_mpi) {
        cross_acceleration_and_gradient_distributed(*dp, slow_start, slow_count,
                                                    fast_start, fast_count, G, soft2, pivot_dt);
    } else {
        cross_acceleration(p, slow_start, slow_count, fast_start, fast_count, G, soft2);
        cross_gradient_correction(p, slow_start, slow_count, fast_start, fast_count, G, soft2, pivot_dt);
    }
    g_profiler.stop(StepProfiler::CROSS_GRAD, cross_pairs * 2);
    g_profiler.start();
    force_err_inject(p, slow_start, slow_count + fast_count, slow_count + fast_count);
    {
#ifdef ENABLE_COLLISIONAL
        cm_tide::KickSrcScope cmt_src(fast_start, fast_count, pivot_dt, G);   // cross kicks: sources = the other set
#endif
        kick_velocities(p, slow_start, slow_count, pivot_dt * FSI_MOD_KICK);
    }
    {
#ifdef ENABLE_COLLISIONAL
        cm_tide::KickSrcScope cmt_src(slow_start, slow_count, pivot_dt, G);
#endif
        kick_velocities(p, fast_start, fast_count, pivot_dt * FSI_MOD_KICK);
    }
    g_profiler.stop(StepProfiler::DRIFT_KICK);

    // Step 5: Recurse on fast (second half)
    k = hhs_fsi_recursive(p, dp, fast_start, fast_count, pivot_dt * 0.5,
                          G, soft2, dt_min, eta, false, tree_for_subs, depth + 1);
    n_removed += k;
    fast_count -= k;
    count -= k;

    // Restore again for the second half-step on slow
#ifdef ENABLE_COLLISIONAL
    g_profiler.start();
    g_subsystem_mgr.restore_active(saved_active);
    // The flags go by particle id; partitions below moved particles since the map was last built
    if (!g_subsystem_mgr.active.empty()) g_subsystem_mgr.rebuild_pid_to_idx(p);
    g_subsystem_mgr.update_particle_flags(p);
    g_profiler.stop(StepProfiler::SUBSYSTEM_BOOKKEEP);
#endif

    // Restore tree state for step 6 and 7
    if (use_tree_cross || use_tree_self) {
        g_profiler.start();
        if (use_tree_cross)
            g_octree.update_subsets_3flag(p, start, count, slow_count);
        else
            g_octree.update_subsets(p, slow_start, slow_count);
        g_profiler.stop(StepProfiler::TREE_MAINT);
    }

    // Step 6: FSI half-step on slow
    if (use_tree_self || use_tree_cross) {
        fsi_step_distributed(*dp, slow_start, slow_count, G, soft2, pivot_dt * 0.5,
                             false, false, 0);
    } else if (use_mpi) {
        fsi_step_distributed(*dp, slow_start, slow_count, G, soft2, pivot_dt * 0.5);
    } else {
        fsi_step(p, slow_start, slow_count, G, soft2, pivot_dt * 0.5, false, true);
    }

    // Refresh tree for step 7 cross-force (step 6 FSI drifted slow particles)
    if (use_tree_cross) {
        g_profiler.start();
        g_octree.update_subsets_3flag(p, start, count, slow_count);
        g_profiler.stop(StepProfiler::TREE_MAINT);
    }

    // Step 7: Cross-force S<->F then kick (dt/6)
    g_profiler.start();
    if (use_tree_cross) {
        cross_acceleration_global_tree(p, slow_start, slow_count,
                                        fast_start, fast_count, G, soft2);
    } else if (use_mpi) {
        p.pack_pos_range(slow_start, slow_count + fast_count);
        cross_acceleration_distributed(*dp, slow_start, slow_count, fast_start, fast_count, G, soft2);
    } else {
        cross_acceleration(p, slow_start, slow_count, fast_start, fast_count, G, soft2);
    }
    g_profiler.stop(StepProfiler::CROSS_FORCE, cross_pairs);
    g_profiler.start();
    force_err_inject(p, slow_start, slow_count + fast_count, slow_count + fast_count);
    {
#ifdef ENABLE_COLLISIONAL
        cm_tide::KickSrcScope cmt_src(fast_start, fast_count, pivot_dt, G);   // cross kicks: sources = the other set
#endif
        kick_velocities(p, slow_start, slow_count, pivot_dt * FSI_KICK);
    }
    {
#ifdef ENABLE_COLLISIONAL
        cm_tide::KickSrcScope cmt_src(slow_start, slow_count, pivot_dt, G);
#endif
        kick_velocities(p, fast_start, fast_count, pivot_dt * FSI_KICK);
    }
    g_profiler.stop(StepProfiler::DRIFT_KICK);
    return n_removed;
}

// Distributed HHS-FSI - main entry point
void hhs_fsi_step_distributed(DistributedParticles& dp, double pivot_dt,
                              double G, double soft2, double dt_min,
                              double eta, int global_levels) {
    if (dp.N_total == 0) return;

    (void)global_levels;  // Not used in new implementation

    // Reset debug counters
    g_hhs_counters.reset();

#ifdef ENABLE_COLLISIONAL
    if (dp.local.collisional) g_subsystem_mgr.reset_step_diagnostics();
#endif

#ifdef USE_MPI
    bool is_distributed = MPIComm::size() > 1;
#else
    bool is_distributed = false;
#endif

    // Global PN1.0 half-kick before FSI (Rantala et al. 2023, Section 3.7)
#ifdef ENABLE_COLLISIONAL
    bool do_global_pn = dp.local.collisional && (g_collisional.pn_order & 1) &&
                        g_collisional.c_code > 0.0;
    if (do_global_pn) {
        StepProfiler::Region r_(g_profiler, StepProfiler::GLOBAL_PN);
        global_pn1_kick(dp.local, 0, dp.N_total, G, g_collisional.c_code,
                        g_collisional.m_pn, pivot_dt * 0.5);
    }
#endif

    // Unified recursion: dp != nullptr enables MPI dispatch for large subsets
    // Single-rank mode: MPI allgather is a no-op, all forces use local GPU
    hhs_fsi_recursive(dp.local, is_distributed ? &dp : nullptr,
                      0, dp.N_total, pivot_dt, G, soft2, dt_min, eta, false);
#ifdef ENABLE_COLLISIONAL
    // Placeholders still left (merged in a range not entered again this step) go before the diagnostics
    if (dp.local.collisional && range_has_ghost(dp.local, 0, dp.local.N)) {
        StepProfiler::Region r_(g_profiler, StepProfiler::GHOST_COMPACT);
        if (compact_zero_mass_range(dp.local, 0, dp.local.N) > 0) {
            sync_after_removal(dp.local, &dp);
            g_subsystem_mgr.update_particle_flags(dp.local);
        }
    }
#endif

    // Global PN1.0 half-kick after FSI (Rantala et al. 2023, Section 3.7)
#ifdef ENABLE_COLLISIONAL
    if (do_global_pn) {
        StepProfiler::Region r_(g_profiler, StepProfiler::GLOBAL_PN);
        global_pn1_kick(dp.local, 0, dp.N_total, G, g_collisional.c_code,
                        g_collisional.m_pn, pivot_dt * 0.5);
    }
    // print_step_summary moved into unified block below
#endif

#ifdef ENABLE_COLLISIONAL
    if (dp.local.collisional) profile_subsystem_counters();
#endif
    // Unified post-step summary: HHS hierarchy + subsystem stats
    StepProfiler::Region r_print_(g_profiler, StepProfiler::STEP_PRINT);
#ifdef USE_MPI
    if (MPIComm::rank() == 0)
#endif
    {
        // --- HHS Hierarchy ---
        std::printf("\n  --- HHS Hierarchy (mpi=%d gpu=%d cpu=%d) ",
                    g_hhs_counters.distributed, g_hhs_counters.local, g_hhs_counters.cpu);
        for (int i = 0; i < 40; i++) std::printf("-");
        std::printf("\n");

        // Find last populated depth (across all paths: distributed, local GPU, CPU serial)
        int last_depth = -1;
        for (int d = HHSCounters::MAX_DEPTH - 1; d >= 0; d--) {
            if (g_hhs_counters.dist_depth_calls[d] > 0 || g_hhs_counters.local_depth_calls[d] > 0 || g_hhs_counters.cpu_depth_calls[d] > 0) {
                last_depth = d;
                break;
            }
        }
        for (int d = 0; d <= last_depth; d++) {
            int dc = g_hhs_counters.dist_depth_calls[d];
            int lc = g_hhs_counters.local_depth_calls[d];
            int cc = g_hhs_counters.cpu_depth_calls[d];
            int total_calls = dc + lc + cc;
            long long total_count = g_hhs_counters.dist_depth_count[d] + g_hhs_counters.local_depth_count[d] + g_hhs_counters.cpu_depth_count[d];
            long long avg_n = total_calls > 0 ? total_count / total_calls : 0;
            std::printf("    L%d: %3d calls", d, total_calls);
            if (dc > 0 || lc > 0 || cc > 0) {
                std::printf(" [");
                bool first = true;
                if (dc > 0) { std::printf("mpi=%d", dc); first = false; }
                if (lc > 0) { std::printf("%sgpu=%d", first ? "" : " ", lc); first = false; }
                if (cc > 0) { std::printf("%scpu=%d", first ? "" : " ", cc); first = false; }
                std::printf("]");
            }
            std::printf("  avg_n=%5lld", avg_n);
            if (g_hhs_counters.dist_depth_ngb_pairs[d] > 0)
                std::printf("  ngb=%d", g_hhs_counters.dist_depth_ngb_pairs[d]);
            if (g_hhs_counters.dist_depth_l0_clamped[d] > 0)
                std::printf("  l0_clamp=%d", g_hhs_counters.dist_depth_l0_clamped[d]);
            if (g_hhs_counters.isolated_sets[d] > 0) std::printf("  isolated=%lld", g_hhs_counters.isolated_sets[d]);
            if (g_hhs_counters.cc_calls[d] > 0)
                std::printf("  cc: %lld calls, %.1f comps, largest %.0f", g_hhs_counters.cc_calls[d],
                            (double)g_hhs_counters.cc_comps[d] / g_hhs_counters.cc_calls[d],
                            (double)g_hhs_counters.cc_largest[d] / g_hhs_counters.cc_calls[d]);
            if (g_hhs_counters.local_depth_time[d] > 0.001)
                std::printf("  time=%.3fs", g_hhs_counters.local_depth_time[d]);
            std::printf("\n");
        }

#ifdef ENABLE_COLLISIONAL
        if (dp.local.collisional) {
            g_subsystem_mgr.print_step_summary();
            g_subsys_phase.report_and_reset();
        }
#endif
    }
}

