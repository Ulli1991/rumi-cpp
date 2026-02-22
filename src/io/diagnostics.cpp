#include "diagnostics.h"
#include "constants.h"
#ifdef ENABLE_COLLISIONAL
#include "collisional_config.h"
#include "post_newtonian.h"
#endif
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <vector>

// Global energy bookkeeping instance
EnergyBookkeeping g_energy_books;

DiagnosticState compute_diagnostics(const Particles& p, double G, double soft2) {
    const int N = p.N;
    auto px = p.x; auto py = p.y; auto pz = p.z;
    auto pvx = p.vx; auto pvy = p.vy; auto pvz = p.vz;
    auto pm = p.mass;

    // Kinetic energy + momentum + angular momentum
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

    // Potential energy: O(N^2)
    double PE = 0;
    Kokkos::parallel_reduce("diag_pe", N,
        KOKKOS_LAMBDA(int i, double& lPE) {
            double pe_i = 0;
            for (int j = i + 1; j < N; j++) {
                double dx = px(j) - px(i);
                double dy = py(j) - py(i);
                double dz = pz(j) - pz(i);
                double inv_r = fast_rsqrt(dx*dx + dy*dy + dz*dz + soft2);
                pe_i -= G * pm(i) * pm(j) * inv_r;
            }
            lPE += pe_i;
        }, PE);

    Kokkos::fence();

    DiagnosticState d{};
    d.KE = KE; d.PE = PE; d.E_total = KE + PE;
    d.Px = Px; d.Py = Py; d.Pz = Pz;
    d.Lx = Lx; d.Ly = Ly; d.Lz = Lz;

#ifdef ENABLE_COLLISIONAL
    // Add PN energy corrections when post-Newtonian corrections are active
    if (g_collisional.enabled && g_collisional.pn_order > 0 && g_collisional.c_code > 0.0) {
        // Copy positions/velocities/masses to host flat arrays for pn1_energy()
        auto h_x  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
        auto h_y  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
        auto h_z  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
        auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
        auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
        auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
        auto h_m  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);

        std::vector<double> flat_pos(N * 3), flat_vel(N * 3), flat_mass(N);
        for (int i = 0; i < N; i++) {
            flat_pos[i*3+0] = h_x(i);
            flat_pos[i*3+1] = h_y(i);
            flat_pos[i*3+2] = h_z(i);
            flat_vel[i*3+0] = h_vx(i);
            flat_vel[i*3+1] = h_vy(i);
            flat_vel[i*3+2] = h_vz(i);
            flat_mass[i] = h_m(i);
        }

        if (g_collisional.pn_order & 1) {
            d.E_pn1 = pn1_energy(flat_pos.data(), flat_vel.data(), flat_mass.data(),
                                 N, G, g_collisional.c_code);
        }
        d.E_total += d.E_pn1 + d.E_pn2;
    }
#endif

    return d;
}

static double vec_mag(double x, double y, double z) {
    return std::sqrt(x*x + y*y + z*z);
}

void compute_errors(const DiagnosticState& cur, const DiagnosticState& ini,
                    const Particles& p, bool is_cluster,
                    double& dE, double& dP, double& dL) {
    // When PN2.5 is active, the conserved quantity is E_total + E_gw_cumulative
    double E_cur = cur.E_total;
    double E_ini = ini.E_total;
#ifdef ENABLE_COLLISIONAL
    if (g_collisional.enabled && (g_collisional.pn_order & 4)) {
        E_cur += g_energy_books.E_gw;
    }
#endif
    dE = (E_ini != 0.0) ? std::fabs((E_cur - E_ini) / E_ini) : 0.0;

    if (is_cluster) {
        // Need total mass from host
        auto h_mass = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
        double M_total = 0;
        for (int i = 0; i < p.N; i++) M_total += h_mass(i);
        double dpx = cur.Px - ini.Px, dpy = cur.Py - ini.Py, dpz = cur.Pz - ini.Pz;
        dP = vec_mag(dpx, dpy, dpz) / M_total;
    } else {
        double P0_mag = vec_mag(ini.Px, ini.Py, ini.Pz);
        double P_mag  = vec_mag(cur.Px, cur.Py, cur.Pz);
        dP = (P0_mag > 0.0) ? std::fabs((P_mag - P0_mag) / P0_mag) : 0.0;
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

    const char* p_label = is_cluster ? "|dP|/M [km/s]" : "|dP/P|";

    // Annotate energy label with PN order when active
    const char* e_label = "|dE/E|";
#ifdef ENABLE_COLLISIONAL
    if (g_collisional.enabled && g_collisional.pn_order > 0 && g_collisional.c_code > 0.0) {
        int pno = g_collisional.pn_order;
        if (pno & 4)      e_label = "|dE/E| (PN2.5)";
        else if (pno & 2) e_label = "|dE/E| (PN2)";
        else if (pno & 1) e_label = "|dE/E| (PN1)";
    }
#endif

    std::printf("  %-6s %14s %6s   %12s %14s %12s %14s %8s\n",
                "Level", "dt [Myr]", "count", e_label, p_label, "|dL/L|", "Time [Myr]", "Step");
    std::printf("----------------------------------------------------------------------------------------------------\n");

    for (int l = 0; l < nl; l++) {
        double dt_myr = levels[l] * TIME_UNIT_MYR;
        std::printf("  L%-5d %14.6e %6d", l, dt_myr, counts[l]);
        if (l == 0 && dE >= 0.0) {
            std::printf("   %12.6e %14.6e %12.6e %14.6f %8d", dE, dP, dL, t_myr, step);
        }
        std::printf("\n");
    }
    std::printf("\n");
}

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

    // Annotate energy with PN order when active
    const char* e_tag = "";
#ifdef ENABLE_COLLISIONAL
    if (g_collisional.enabled && g_collisional.pn_order > 0 && g_collisional.c_code > 0.0) {
        int pno = g_collisional.pn_order;
        if (pno & 4)      e_tag = " (incl. PN2.5)";
        else if (pno & 2) e_tag = " (incl. PN2)";
        else if (pno & 1) e_tag = " (incl. PN1)";
    }
#endif
    std::printf("|dE/E0| = %.15e%s\n", dE, e_tag);
    if (is_cluster) {
        std::printf("|dP|/M  = %.15e km/s\n", dP);
    } else {
        std::printf("|dP/P0| = %.15e\n", dP);
    }
    std::printf("|dL/L0| = %.15e\n", dL);
#ifdef ENABLE_COLLISIONAL
    if (g_collisional.enabled && g_collisional.pn_order > 0) {
        std::printf("E_pn1   = %.15e\n", cur.E_pn1);
        if (g_collisional.pn_order & 4) {
            std::printf("E_gw    = %.15e (cumulative GW losses)\n", g_energy_books.E_gw);
        }
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

    std::printf("\nDone!\n");
}
