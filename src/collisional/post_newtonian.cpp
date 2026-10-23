#include "post_newtonian.h"
#include "constants.h"
#include "diagnostics.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstring>
#include <vector>

// ============================================================================
// Global PN1.0 kick for compact objects outside subsystems
// (Rantala et al. 2023, Section 3.7) — now using EIH formulation directly
// ============================================================================

void global_pn1_kick(Particles& p, int start, int count,
                     double G, double c_code, double m_pn, double dt) {
    if (count == 0 || c_code <= 0.0 || dt == 0.0) return;

    // Copy particle data to host
    auto h_x  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
    auto h_y  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
    auto h_z  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
    auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
    auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
    auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
    auto h_m  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
    auto h_sid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.subsystem_id);
    auto h_kw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_type);

    // Find the BHs of at least m_pn (CollisionalConfig::pn_body) in [start, start+count)
    int end = start + count;
    std::vector<int> compact;
    compact.reserve(256);
    for (int i = start; i < end; i++) {
        if (h_kw(i) == 14 && h_m(i) >= m_pn) compact.push_back(i);
    }

    int n_compact = (int)compact.size();
    if (n_compact < 2) return;

    // Accumulate PN1.0 velocity kicks on host
    std::vector<double> dvx(n_compact, 0.0), dvy(n_compact, 0.0), dvz(n_compact, 0.0);

    for (int ii = 0; ii < n_compact; ii++) {
        int i = compact[ii];
        int sid_i = h_sid(i);

        for (int jj = ii + 1; jj < n_compact; jj++) {
            int j = compact[jj];
            int sid_j = h_sid(j);

            // Skip pairs in the same subsystem (they get full PN inside LogH)
            if (sid_i >= 0 && sid_j == sid_i) continue;

            double x_i[3] = { h_x(i), h_y(i), h_z(i) };
            double v_i[3] = { h_vx(i), h_vy(i), h_vz(i) };
            double x_j[3] = { h_x(j), h_y(j), h_z(j) };
            double v_j[3] = { h_vx(j), h_vy(j), h_vz(j) };

            // EIH: get acceleration of i due to j, and j due to i, directly
            double a_pn_i[3], a_pn_j[3];
            eih_pn_acceleration(x_i, v_i, h_m(i), x_j, v_j, h_m(j),
                                G, c_code, a_pn_i, 1);
            eih_pn_acceleration(x_j, v_j, h_m(j), x_i, v_i, h_m(i),
                                G, c_code, a_pn_j, 1);

            dvx[ii] += dt * a_pn_i[0];
            dvy[ii] += dt * a_pn_i[1];
            dvz[ii] += dt * a_pn_i[2];
            dvx[jj] += dt * a_pn_j[0];
            dvy[jj] += dt * a_pn_j[1];
            dvz[jj] += dt * a_pn_j[2];
        }
    }

    // Apply velocity kicks; the Newtonian E, P, L they change go to the PN sink (diagnostics are Newtonian)
    for (int ii = 0; ii < n_compact; ii++) {
        int i = compact[ii];
        double r[3] = {h_x(i), h_y(i), h_z(i)}, v0[3] = {h_vx(i), h_vy(i), h_vz(i)};
        h_vx(i) += dvx[ii];
        h_vy(i) += dvy[ii];
        h_vz(i) += dvz[ii];
        double v1[3] = {h_vx(i), h_vy(i), h_vz(i)};
        g_energy_books.E_pn += 0.5 * h_m(i) * ((v0[0]*v0[0] + v0[1]*v0[1] + v0[2]*v0[2]) -
                                               (v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]));
        g_energy_books.sink_PL(h_m(i), r, v0);
        g_energy_books.sink_PL(h_m(i), r, v1, -1.0);
    }

    // Copy velocities back to device
    Kokkos::deep_copy(p.vx, h_vx);
    Kokkos::deep_copy(p.vy, h_vy);
    Kokkos::deep_copy(p.vz, h_vz);
    p.zero_kick_compensation();   // velocities changed outside the compensated kicks
}
