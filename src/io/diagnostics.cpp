#include "diagnostics.h"
#include "constants.h"
#include "mpi_comm.h"
#ifdef ENABLE_COLLISIONAL
#include "collisional_config.h"
#include "post_newtonian.h"
#include "subsystem.h"
#include "kepler.h"
#include <unordered_map>
#endif
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <vector>
#ifdef USE_MPI
#include <mpi.h>
#endif

// Global energy bookkeeping instance
EnergyBookkeeping g_energy_books;

// ==========================================================================
// Tiled GPU potential energy computation (shared-memory, TILE_SIZE tiles)
// Computes PE for i in [i_start, i_start+i_count) against all j > i
// ==========================================================================
double compute_potential_energy_tiled(const Particles& p, double G, double soft2,
                                      int i_start, int i_count) {
    const int N = p.N;
    auto px = p.x; auto py = p.y; auto pz = p.z;
    auto pm = p.mass;

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;

    // Tile: x,y,z,m per j-particle
    size_t scratch_size = TILE_SIZE * 4 * sizeof(double);
    int num_teams = (i_count + TILE_SIZE - 1) / TILE_SIZE;
    if (num_teams == 0) return 0.0;

    double PE = 0;
    Kokkos::parallel_reduce("pe_tiled",
        team_policy(num_teams, TILE_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type& team, double& lPE) {
            int i = i_start + team.league_rank() * TILE_SIZE + team.team_rank();

            Kokkos::View<double*, scratch_space, Kokkos::MemoryUnmanaged>
                tile(team.team_scratch(0), TILE_SIZE * 4);

            double xi = 0, yi = 0, zi = 0, mi = 0;
            if (i < i_start + i_count) {
                xi = px(i); yi = py(i); zi = pz(i); mi = pm(i);
            }

            double pe_i = 0;
            for (int tile_start = 0; tile_start < N; tile_start += TILE_SIZE) {
                int tile_end = (tile_start + TILE_SIZE < N) ? tile_start + TILE_SIZE : N;
                int tile_count = tile_end - tile_start;

                // Collaborative tile load
                int t = team.team_rank();
                if (t < tile_count) {
                    int idx = tile_start + t;
                    int base = t * 4;
                    tile(base)     = px(idx);
                    tile(base + 1) = py(idx);
                    tile(base + 2) = pz(idx);
                    tile(base + 3) = pm(idx);
                }
                team.team_barrier();

                if (i < i_start + i_count) {
                    for (int jt = 0; jt < tile_count; jt++) {
                        int j = tile_start + jt;
                        if (j <= i) continue;  // unique pairs only (j > i)
                        int jb = jt * 4;
                        double dx = tile(jb)     - xi;
                        double dy = tile(jb + 1) - yi;
                        double dz = tile(jb + 2) - zi;
                        double mj = tile(jb + 3);
                        double inv_r = fast_rsqrt(dx*dx + dy*dy + dz*dz + soft2);
                        pe_i -= G * mi * mj * inv_r;
                    }
                }
                team.team_barrier();
            }
            lPE += pe_i;
        }, PE);

    return PE;
}

// ==========================================================================
// Compute PE distributed across MPI ranks (tiled kernel + allreduce)
// ==========================================================================
static double compute_potential_energy_distributed(const Particles& p, double G, double soft2) {
    const int N = p.N;
#ifdef USE_MPI
    if (MPIComm::size() > 1) {
        int n_ranks = MPIComm::size();
        int my_rank = MPIComm::rank();
        int base = N / n_ranks;
        int rem  = N % n_ranks;
        int i_start = my_rank * base + (my_rank < rem ? my_rank : rem);
        int i_count = base + (my_rank < rem ? 1 : 0);

        double local_PE = compute_potential_energy_tiled(p, G, soft2, i_start, i_count);

        double total_PE = 0;
        MPI_Allreduce(&local_PE, &total_PE, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return total_PE;
    }
#endif
    return compute_potential_energy_tiled(p, G, soft2, 0, N);
}

DiagnosticState compute_diagnostics(const Particles& p, double G, double soft2) {
    const int N = p.N;
    auto px = p.x; auto py = p.y; auto pz = p.z;
    auto pvx = p.vx; auto pvy = p.vy; auto pvz = p.vz;
    auto pm = p.mass;

    // Kinetic energy + momentum + angular momentum (O(N), fast)
    double KE = 0, Px = 0, Py = 0, Pz = 0, Lx = 0, Ly = 0, Lz = 0;
    Kokkos::parallel_reduce("diag_ke", N,
        KOKKOS_LAMBDA(int i, double& lKE) {
            double v2 = pvx(i)*pvx(i) + pvy(i)*pvy(i) + pvz(i)*pvz(i);
            lKE += 0.5 * pm(i) * v2;
        }, KE);

    Kokkos::parallel_reduce("diag_px", N,
        KOKKOS_LAMBDA(int i, double& lPx) { lPx += pm(i) * pvx(i); }, Px);
    Kokkos::parallel_reduce("diag_py", N,
        KOKKOS_LAMBDA(int i, double& lPy) { lPy += pm(i) * pvy(i); }, Py);
    Kokkos::parallel_reduce("diag_pz", N,
        KOKKOS_LAMBDA(int i, double& lPz) { lPz += pm(i) * pvz(i); }, Pz);

    Kokkos::parallel_reduce("diag_lx", N,
        KOKKOS_LAMBDA(int i, double& lLx) {
            lLx += pm(i) * (py(i)*pvz(i) - pz(i)*pvy(i));
        }, Lx);
    Kokkos::parallel_reduce("diag_ly", N,
        KOKKOS_LAMBDA(int i, double& lLy) {
            lLy += pm(i) * (pz(i)*pvx(i) - px(i)*pvz(i));
        }, Ly);
    Kokkos::parallel_reduce("diag_lz", N,
        KOKKOS_LAMBDA(int i, double& lLz) {
            lLz += pm(i) * (px(i)*pvy(i) - py(i)*pvx(i));
        }, Lz);

    // Potential energy: tiled GPU kernel + MPI-distributed
    double PE = compute_potential_energy_distributed(p, G, soft2);

    Kokkos::fence();

    DiagnosticState d{};
    d.KE = KE; d.PE = PE; d.E_total = KE + PE;
    d.Px = Px; d.Py = Py; d.Pz = Pz;
    d.Lx = Lx; d.Ly = Ly; d.Lz = Lz;

    // Note: PN1 energy diagnostic (pn1_energy) removed — host-side O(N²) is
    // prohibitively slow for large N. PN forces in the integrator are unaffected.

    return d;
}

static double vec_mag(double x, double y, double z) {
    return std::sqrt(x*x + y*y + z*z);
}

void compute_errors(const DiagnosticState& cur, const DiagnosticState& ini,
                    const Particles& p, bool is_cluster,
                    double& dE, double& dP, double& dL) {
    // Conserved quantity: E_total + all energy sinks (GW, mergers, escapers, SSE mass loss)
    double E_cur = cur.E_total;
    double E_ini = ini.E_total;
#ifdef ENABLE_COLLISIONAL
    if (g_collisional.enabled) {
        E_cur += g_energy_books.total_sink();
    }
#endif
    dE = (E_ini != 0.0) ? std::fabs((E_cur - E_ini) / E_ini) : 0.0;

    // For systems initialized in the COM frame (binary/triple tests), P0 ≈ 0
    // and |dP/P0| is meaningless. Use |dP|/M (absolute drift per unit mass) instead.
    {
        auto h_mass = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
        double M_total = 0;
        for (int i = 0; i < p.N; i++) M_total += h_mass(i);
        double dpx = cur.Px - ini.Px, dpy = cur.Py - ini.Py, dpz = cur.Pz - ini.Pz;
        dP = vec_mag(dpx, dpy, dpz) / M_total;
    }

    double L0_mag = vec_mag(ini.Lx, ini.Ly, ini.Lz);
    double L_mag  = vec_mag(cur.Lx, cur.Ly, cur.Lz);
    dL = (L0_mag > 0.0) ? std::fabs((L_mag - L0_mag) / L0_mag) : 0.0;
}

void print_timebin_table(const Particles& p, const std::vector<double>& levels,
                         double dE, double dP, double dL,
                         double t_myr, int step, bool is_cluster) {
    int nl = (int)levels.size();

    // Mirror level to host for counting
    auto h_level = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.level);

    std::vector<int> counts(nl, 0);
    for (int i = 0; i < p.N; i++) {
        if (h_level(i) >= 0 && h_level(i) < nl)
            counts[h_level(i)]++;
    }

    const char* p_label = "|dP|/M [km/s]";

    const char* e_label = "|dE/E|";

    std::printf("  %-6s %14s %6s   %12s %14s %12s %14s %8s\n",
                "Level", "dt [Myr]", "count", e_label, p_label, "|dL/L|", "Time [Myr]", "Step");
    std::printf("----------------------------------------------------------------------------------------------------\n");

    // Find last populated level
    int last_pop = 0;
    for (int l = nl - 1; l >= 0; l--) {
        if (counts[l] > 0) { last_pop = l; break; }
    }

    for (int l = 0; l <= last_pop; l++) {
        double dt_myr = levels[l] * TIME_UNIT_MYR;
        std::printf("  L%-5d %14.6e %6d", l, dt_myr, counts[l]);
        if (l == 0 && dE >= 0.0) {
            std::printf("   %12.6e %14.6e %12.6e %14.6f %8d", dE, dP, dL, t_myr, step);
        }
        std::printf("\n");
    }
    std::printf("\n");
}

#ifdef ENABLE_COLLISIONAL
void print_subsystem_diagnostics(const Particles& p, double G) {
    if (g_subsystem_mgr.active.empty()) return;

    // Mirror particle data to host (once)
    auto h_x  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
    auto h_y  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
    auto h_z  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
    auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
    auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
    auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
    auto h_m  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
    auto h_pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);
    auto h_spin = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.spin_vec);
    auto h_stype = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_type);

    // Build pid -> array index map
    std::unordered_map<int, int> pid_to_idx;
    for (int i = 0; i < p.N; i++) pid_to_idx[h_pid(i)] = i;

    std::printf("  Subsystem orbital elements:\n");
    std::printf("  %-4s %-6s %3s %12s %12s %12s %12s %12s %10s %10s %8s\n",
                "ID", "Type", "N", "a [pc]", "e", "omega [rad]",
                "P [yr]", "r_peri [pc]", "|chi_1|", "|chi_2|", "S.L/|S||L|");
    std::printf("  ");
    for (int i = 0; i < 117; i++) std::printf("-");
    std::printf("\n");

    for (const auto& sub : g_subsystem_mgr.active) {
        // Type label
        const char* type_str = "???";
        switch (sub.type) {
            case SubsystemType::BINARY:          type_str = "BIN"; break;
            case SubsystemType::FLYBY:           type_str = "FLY"; break;
            case SubsystemType::TRIPLE_STABLE:   type_str = "TRI_S"; break;
            case SubsystemType::TRIPLE_UNSTABLE: type_str = "TRI_U"; break;
            case SubsystemType::MULTIPLET:       type_str = "MULTI"; break;
        }

        if (sub.n_bodies < 2 || sub.particle_ids.size() < 2) continue;

        // Identify the inner pair
        int pid0, pid1;
        if (sub.type == SubsystemType::TRIPLE_STABLE && sub.inner_binary_id >= 0 &&
            sub.particle_ids.size() >= 3) {
            // For stable triples, inner binary is particle_ids[0], [1] by convention
            // (set during detection with inner_binary_id)
            pid0 = sub.particle_ids[0];
            pid1 = sub.particle_ids[1];
        } else {
            pid0 = sub.particle_ids[0];
            pid1 = sub.particle_ids[1];
        }

        auto it0 = pid_to_idx.find(pid0);
        auto it1 = pid_to_idx.find(pid1);
        if (it0 == pid_to_idx.end() || it1 == pid_to_idx.end()) continue;
        int i0 = it0->second, i1 = it1->second;

        double m1 = h_m(i0), m2 = h_m(i1);
        double mu = G * (m1 + m2);

        // Relative position and velocity
        double r[3] = { h_x(i1) - h_x(i0), h_y(i1) - h_y(i0), h_z(i1) - h_z(i0) };
        double v[3] = { h_vx(i1) - h_vx(i0), h_vy(i1) - h_vy(i0), h_vz(i1) - h_vz(i0) };

        auto oe = cartesian_to_elements(r, v, mu);

        double r_peri = oe.a * (1.0 - oe.e);
        double P_yr = oe.period * TIME_UNIT_YR;

        // Spin magnitudes for BH-BH pairs (stellar_type == 14)
        double chi1 = -1.0, chi2 = -1.0;
        double sdotl = -99.0;  // sentinel for "not applicable"
        bool is_bh_pair = (h_stype(i0) == 14 && h_stype(i1) == 14);
        if (is_bh_pair) {
            double s1[3] = { h_spin(i0, 0), h_spin(i0, 1), h_spin(i0, 2) };
            double s2[3] = { h_spin(i1, 0), h_spin(i1, 1), h_spin(i1, 2) };
            chi1 = std::sqrt(s1[0]*s1[0] + s1[1]*s1[1] + s1[2]*s1[2]);
            chi2 = std::sqrt(s2[0]*s2[0] + s2[1]*s2[1] + s2[2]*s2[2]);

            // Orbital angular momentum direction L_hat
            double Lx = r[1]*v[2] - r[2]*v[1];
            double Ly = r[2]*v[0] - r[0]*v[2];
            double Lz = r[0]*v[1] - r[1]*v[0];
            double Lmag = std::sqrt(Lx*Lx + Ly*Ly + Lz*Lz);

            // Total spin S = m1^2 * chi1_vec + m2^2 * chi2_vec (dimensionless convention)
            double Sx = m1*m1*s1[0] + m2*m2*s2[0];
            double Sy = m1*m1*s1[1] + m2*m2*s2[1];
            double Sz = m1*m1*s1[2] + m2*m2*s2[2];
            double Smag = std::sqrt(Sx*Sx + Sy*Sy + Sz*Sz);

            if (Lmag > 0 && Smag > 0) {
                sdotl = (Sx*Lx + Sy*Ly + Sz*Lz) / (Smag * Lmag);
            }
        }

        // Print main line
        std::printf("  %-4d %-6s %3d %12.4e %12.4e %12.4e %12.4e %12.4e",
                    sub.id, type_str, sub.n_bodies,
                    oe.a, oe.e, oe.omega, P_yr, r_peri);
        if (is_bh_pair) {
            std::printf(" %10.4f %10.4f %8.4f", chi1, chi2, sdotl);
        } else {
            std::printf(" %10s %10s %8s", "--", "--", "--");
        }
        std::printf("\n");

        // For triples: print outer orbit on a second line
        if ((sub.type == SubsystemType::TRIPLE_STABLE ||
             sub.type == SubsystemType::TRIPLE_UNSTABLE) &&
            sub.particle_ids.size() >= 3 && sub.outer_companion_pid >= 0) {
            int pid_out = sub.outer_companion_pid;
            auto it_out = pid_to_idx.find(pid_out);
            if (it_out != pid_to_idx.end()) {
                int i_out = it_out->second;
                // CM of inner binary
                double mtot_in = m1 + m2;
                double cx = (m1*h_x(i0) + m2*h_x(i1)) / mtot_in;
                double cy = (m1*h_y(i0) + m2*h_y(i1)) / mtot_in;
                double cz = (m1*h_z(i0) + m2*h_z(i1)) / mtot_in;
                double cvx = (m1*h_vx(i0) + m2*h_vx(i1)) / mtot_in;
                double cvy = (m1*h_vy(i0) + m2*h_vy(i1)) / mtot_in;
                double cvz = (m1*h_vz(i0) + m2*h_vz(i1)) / mtot_in;

                double r_out[3] = { h_x(i_out) - cx, h_y(i_out) - cy, h_z(i_out) - cz };
                double v_out[3] = { h_vx(i_out) - cvx, h_vy(i_out) - cvy, h_vz(i_out) - cvz };
                double mu_out = G * (mtot_in + h_m(i_out));

                auto oe_out = cartesian_to_elements(r_out, v_out, mu_out);
                std::printf("  %4s outer: %3s %12.4e %12.4e\n",
                            "", "", oe_out.a, oe_out.e);
            }
        }
    }
    std::printf("\n");
    std::fflush(stdout);
}
#endif

void print_final(const DiagnosticState& cur, const DiagnosticState& ini,
                 const Particles& p, const std::vector<double>& /*levels*/,
                 bool is_cluster, double wall_sec, int steps,
                 const char* const* names) {
    double dE, dP, dL;
    compute_errors(cur, ini, p, is_cluster, dE, dP, dL);

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("RESULTS\n");
    std::printf("============================================================\n");

    std::printf("|dE/E0| = %.15e\n", dE);
    std::printf("|dP|/M  = %.15e km/s\n", dP);
    std::printf("|dL/L0| = %.15e\n", dL);
#ifdef ENABLE_COLLISIONAL
    if (g_collisional.enabled) {
        // E_pn1 computed but not printed (Newtonian E_total only)
        if (g_energy_books.E_gw != 0.0)
            std::printf("E_gw    = %.15e (cumulative GW losses)\n", g_energy_books.E_gw);
        if (g_energy_books.E_merge != 0.0)
            std::printf("E_merge = %.15e (merger dissipation)\n", g_energy_books.E_merge);
        if (g_energy_books.E_escape != 0.0)
            std::printf("E_esc   = %.15e (escaper removal)\n", g_energy_books.E_escape);
        if (g_energy_books.E_sse != 0.0)
            std::printf("E_sse   = %.15e (stellar mass loss)\n", g_energy_books.E_sse);
    }
#endif

    std::printf("\nWall: %.4fs, Steps: %d", wall_sec, steps);
    if (steps > 0) std::printf(", %.4fms/step", wall_sec / steps * 1000.0);
    std::printf("\n");

    // Mirror data to host for printing
    auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
    auto h_y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
    auto h_z = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
    auto h_orig = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.orig_index);

    if (p.N <= 10 && names != nullptr) {
        std::printf("\nPositions (pc):\n");
        std::vector<int> cur_pos(p.N);
        for (int i = 0; i < p.N; i++) cur_pos[h_orig(i)] = i;
        for (int orig = 0; orig < p.N; orig++) {
            int i = cur_pos[orig];
            double ppx = h_x(i);
            double ppy = h_y(i);
            double ppz = h_z(i);
            double r = std::sqrt(ppx*ppx + ppy*ppy + ppz*ppz);
            std::printf("  %-10s: (%12.6e,%12.6e,%12.6e) r=%.6e\n",
                        names[orig], ppx, ppy, ppz, r);
        }
    } else if (p.N > 10) {
        std::printf("\nPosition statistics (pc):\n");
        double r_min = 1e30, r_max = 0, r_sum = 0;
        for (int i = 0; i < p.N; i++) {
            double ppx = h_x(i);
            double ppy = h_y(i);
            double ppz = h_z(i);
            double r = std::sqrt(ppx*ppx + ppy*ppy + ppz*ppz);
            if (r < r_min) r_min = r;
            if (r > r_max) r_max = r;
            r_sum += r;
        }
        std::printf("  r_min=%.4f, r_max=%.4f, r_mean=%.4f\n",
                    r_min, r_max, r_sum / p.N);
    }

#ifdef ENABLE_COLLISIONAL
    if (g_collisional.enabled && !g_subsystem_mgr.active.empty()) {
        std::printf("\n");
        print_subsystem_diagnostics(p, G_CODE);
    }
#endif

    std::printf("\nDone!\n");
}
