#include "particles.h"
#include <numeric>
#include "forces.h"
#include "integrator.h"
#include "timestep.h"
#include "initial_conditions.h"
#include "diagnostics.h"
#include "cluster_diag.h"
#include "tidal_field.h"
#include "constants.h"
#include "mpi_comm.h"
#include "profiling.h"
#include "params.h"
#ifdef USE_HDF5
#include "restart.h"
#include "snapshot.h"
#endif
#include "fence_stats.h"
#ifdef ENABLE_COLLISIONAL
#include "collisional_config.h"
#include "subsystem.h"
#include "logh_batch.h"
#include "kepler.h"
#include "checkpoint.h"
#include "mergers.h"
#include "merger_tree.h"
#include "runtime_sse.h"
#include "post_newtonian.h"
#include "stellar_cache.h"
CollisionalConfig g_collisional;

// Binary orbit tracker for PN precession validation
struct BinaryTracker {
    double mu;               // G*(m1+m2)
    double omega_prev;       // previous omega for unwrapping
    double omega_cumul;      // cumulative unwrapped omega
    double orbits_cumul;     // cumulative orbit count (from elapsed time)
    double omega_gr_per_orbit; // analytic GR precession rate
    double a0, e0;           // initial orbital elements
    double period0;          // initial orbital period
    bool initialized;

    // Apoapsis-based measurement (avoids intra-orbit PN oscillation aliasing)
    double rdot_prev;        // previous r_dot for sign-change detection
    int n_apo;               // number of apoapsis passages
    double omega_at_apo_prev;// omega at previous apoapsis
    double omega_cumul_apo;  // cumulative precession measured at apoapsis
    double e_at_apo;         // eccentricity at last apoapsis

    BinaryTracker() : initialized(false), rdot_prev(0), n_apo(0),
                       omega_at_apo_prev(0), omega_cumul_apo(0), e_at_apo(0) {}

    void init(Particles& p, double G) {
        if (p.N != 2) return;
        auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
        auto h_y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
        auto h_z = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
        auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
        auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
        auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
        auto h_m  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);

        mu = G * (h_m(0) + h_m(1));
        double r[3] = { h_x(1)-h_x(0), h_y(1)-h_y(0), h_z(1)-h_z(0) };
        double v[3] = { h_vx(1)-h_vx(0), h_vy(1)-h_vy(0), h_vz(1)-h_vz(0) };
        auto oe = cartesian_to_elements(r, v, mu);

        a0 = oe.a;
        e0 = oe.e;
        period0 = oe.period;
        omega_prev = oe.omega;
        omega_cumul = 0.0;
        orbits_cumul = 0.0;

        // Analytic PN1 precession: dw/orbit = 6*pi*G*M / (a*c^2*(1-e^2))
        // Sign: prograde (same direction as orbital motion).
        // Lz > 0 (CCW) → dw > 0;  Lz < 0 (CW) → dw < 0.
        double c2 = C_CODE * C_CODE;
        double Lz = r[0]*v[1] - r[1]*v[0];
        double sign = (Lz >= 0) ? 1.0 : -1.0;
        omega_gr_per_orbit = sign * 6.0 * M_PI * mu / (oe.a * c2 * (1.0 - oe.e * oe.e));

        // Initial r_dot for apoapsis detection
        double rmag = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
        rdot_prev = (r[0]*v[0] + r[1]*v[1] + r[2]*v[2]) / rmag;
        omega_at_apo_prev = oe.omega;
        omega_cumul_apo = 0.0;
        n_apo = 0;
        e_at_apo = oe.e;

        std::printf("BinaryTracker: a=%.4e pc (%.4f AU), e=%.4f, P=%.4e yr\n",
                    oe.a, oe.a / AU_TO_PC, oe.e, oe.period * TIME_UNIT_YR);
        std::printf("  PN1 precession: %.4e rad/orbit (%.4f deg/orbit)\n",
                    omega_gr_per_orbit, omega_gr_per_orbit * 180.0 / M_PI);
        std::fflush(stdout);
        initialized = true;
    }

    void update(Particles& p, double G, double t_code) {
        (void)G;
        if (!initialized || p.N != 2) return;
        auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
        auto h_y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
        auto h_z = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
        auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
        auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
        auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);

        double r[3] = { h_x(1)-h_x(0), h_y(1)-h_y(0), h_z(1)-h_z(0) };
        double v[3] = { h_vx(1)-h_vx(0), h_vy(1)-h_vy(0), h_vz(1)-h_vz(0) };
        auto oe = cartesian_to_elements(r, v, mu);

        // Count orbits from elapsed time / initial period
        orbits_cumul = t_code / period0;

        // Unwrap omega to track cumulative precession (raw, at arbitrary phase)
        double dw = oe.omega - omega_prev;
        if (dw < -M_PI) dw += 2.0 * M_PI;
        if (dw > M_PI) dw -= 2.0 * M_PI;
        omega_cumul += dw;
        omega_prev = oe.omega;

        // Apoapsis detection: r_dot changes from positive to negative
        double rmag = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
        double rdot = (r[0]*v[0] + r[1]*v[1] + r[2]*v[2]) / rmag;

        if (rdot_prev > 0.0 && rdot <= 0.0) {
            // Passed through apoapsis — measure omega here
            double dw_apo = oe.omega - omega_at_apo_prev;
            if (dw_apo < -M_PI) dw_apo += 2.0 * M_PI;
            if (dw_apo > M_PI) dw_apo -= 2.0 * M_PI;
            omega_cumul_apo += dw_apo;
            omega_at_apo_prev = oe.omega;
            e_at_apo = oe.e;
            n_apo++;
        }
        rdot_prev = rdot;

        // Report: both raw (arbitrary phase) and apoapsis-based measurements
        double omega_gr_expected = orbits_cumul * omega_gr_per_orbit;
        double ratio_raw = (std::fabs(omega_gr_expected) > 1e-30)
                           ? omega_cumul / omega_gr_expected : 0.0;

        // ratio_apo uses omega measured only at apoapsis phases (less aliasing)
        // but normalized by total orbits from elapsed time (n_apo undercounts
        // because we only sample once per outer step, missing most apoapsides)
        double ratio_apo = 0.0;
        if (n_apo > 0 && std::fabs(omega_gr_expected) > 1e-30) {
            ratio_apo = omega_cumul_apo / omega_gr_expected;
        }

        std::printf("  Binary: N_orb=%.2f  a=%.4e  e=%.6f"
                    "  ratio_raw=%.4f  ratio_apo=%.4f (n_apo=%d, e_apo=%.6f)\n",
                    orbits_cumul, oe.a, oe.e,
                    ratio_raw, ratio_apo, n_apo, e_at_apo);
        std::fflush(stdout);
    }
};

// Spin tracker for a spinning BH binary (BIFROST sec. 3.7): |chi_i| (an exact invariant of dS/dt = Omega × S),
// the conserved J = L_N + L_PN1 + L_SO + S1 + S2 (Kidder 1995 eq. 2.9), and the precession angle of S1 about
// the initial J against the orbit-averaged spin-orbit rate (2 + 3 m2 / (2 m1)) G |J| / (c^2 a^3 (1-e^2)^1.5),
// exact for simple precession (S2 = 0).
struct SpinTracker {
    double m1 = 0, m2 = 0, chi0[2] = {0, 0}, J0[3] = {0, 0, 0}, Jhat[3] = {0, 0, 0}, e1[3] = {0, 0, 0},
           e2[3] = {0, 0, 0}, rate = 0, phi_prev = 0, phi_cumul = 0;
    bool initialized = false;

    static int find_pid(Particles& p, int pid) {
        auto h_pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);
        for (int i = 0; i < p.N; i++) if (h_pid(i) == pid) return i;
        return -1;
    }
    // x = x1 - x2, v = v1 - v2, physical spins and |chi|
    bool state(Particles& p, double G, double x[3], double v[3], double S1[3], double S2[3], double chi[2]) {
        int i1 = find_pid(p, 0), i2 = find_pid(p, 1);
        if (i1 < 0 || i2 < 0) return false;
        auto hx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
        auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
        auto hz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
        auto hvx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
        auto hvy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
        auto hvz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
        auto hm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
        auto hs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.spin_vec);
        x[0] = hx(i1) - hx(i2); x[1] = hy(i1) - hy(i2); x[2] = hz(i1) - hz(i2);
        v[0] = hvx(i1) - hvx(i2); v[1] = hvy(i1) - hvy(i2); v[2] = hvz(i1) - hvz(i2);
        m1 = hm(i1); m2 = hm(i2);
        double f1 = G * m1 * m1 / C_CODE, f2 = G * m2 * m2 / C_CODE;
        for (int k = 0; k < 3; k++) { S1[k] = hs(i1, k) * f1; S2[k] = hs(i2, k) * f2; }
        chi[0] = std::sqrt(hs(i1, 0)*hs(i1, 0) + hs(i1, 1)*hs(i1, 1) + hs(i1, 2)*hs(i1, 2));
        chi[1] = std::sqrt(hs(i2, 0)*hs(i2, 0) + hs(i2, 1)*hs(i2, 1) + hs(i2, 2)*hs(i2, 2));
        return true;
    }
    double angle(const double S1[3]) const {
        double a = S1[0]*e1[0] + S1[1]*e1[1] + S1[2]*e1[2], b = S1[0]*e2[0] + S1[1]*e2[1] + S1[2]*e2[2];
        return std::atan2(b, a);
    }
    void init(Particles& p, double G) {
        if (p.N != 2 || !p.collisional) return;
        double x[3], v[3], S1[3], S2[3];
        if (!state(p, G, x, v, S1, S2, chi0) || chi0[0] + chi0[1] == 0.0) return;
        binary_total_angular_momentum(x, v, m1, m2, S1, S2, G, C_CODE, g_collisional.pn_order & 1, J0);
        double J = std::sqrt(J0[0]*J0[0] + J0[1]*J0[1] + J0[2]*J0[2]);
        for (int k = 0; k < 3; k++) Jhat[k] = J0[k] / J;
        double sJ = S1[0]*Jhat[0] + S1[1]*Jhat[1] + S1[2]*Jhat[2];
        for (int k = 0; k < 3; k++) e1[k] = S1[k] - sJ * Jhat[k];
        double ne1 = std::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
        if (ne1 > 0) for (int k = 0; k < 3; k++) e1[k] /= ne1;
        e2[0] = Jhat[1]*e1[2] - Jhat[2]*e1[1]; e2[1] = Jhat[2]*e1[0] - Jhat[0]*e1[2]; e2[2] = Jhat[0]*e1[1] - Jhat[1]*e1[0];
        auto oe = cartesian_to_elements(x, v, G * (m1 + m2));
        double ome2 = 1.0 - oe.e * oe.e;
        rate = (2.0 + 1.5 * m2 / m1) * G * J / (C_CODE * C_CODE * oe.a * oe.a * oe.a * ome2 * std::sqrt(ome2));
        phi_prev = angle(S1);
        initialized = true;
        std::printf("SpinTracker: chi1=%.4f chi2=%.4f, S1 spin-orbit precession %.4e rad/TU\n", chi0[0], chi0[1], rate);
    }
    void update(Particles& p, double G, double t_code) {
        if (!initialized) return;
        double x[3], v[3], S1[3], S2[3], chi[2], J[3];
        if (!state(p, G, x, v, S1, S2, chi)) return;
        binary_total_angular_momentum(x, v, m1, m2, S1, S2, G, C_CODE, g_collisional.pn_order & 1, J);
        double dJ = std::sqrt((J[0]-J0[0])*(J[0]-J0[0]) + (J[1]-J0[1])*(J[1]-J0[1]) + (J[2]-J0[2])*(J[2]-J0[2])) /
                    std::sqrt(J0[0]*J0[0] + J0[1]*J0[1] + J0[2]*J0[2]);
        double phi = angle(S1), d = phi - phi_prev;
        if (d > M_PI) d -= 2.0 * M_PI;
        if (d < -M_PI) d += 2.0 * M_PI;
        phi_cumul += d;
        phi_prev = phi;
        double dchi = std::max(std::fabs(chi[0] - chi0[0]), std::fabs(chi[1] - chi0[1]));
        std::printf("  Spin: dchi=%.3e  dJ/J=%.3e  phi_S1=%.6e  ratio_so=%.6f\n", dchi, dJ, phi_cumul,
                    t_code > 0 ? phi_cumul / (rate * t_code) : 0.0);
        std::fflush(stdout);
    }
};

// Triple orbit tracker for Kozai-Lidov validation
// Tracks inner binary (a, e, omega) and mutual inclination at each print step.
// Conserved Kozai integral: C_K = (1 - e_in^2) * cos^2(i_mut) = const.
struct TripleTracker {
    double mu_in;            // G*(m1+m2)
    double mu_out;           // G*(m1+m2+m3)
    double m1, m2, m3;
    double C_K0;             // initial Kozai integral
    double a_in0, e_in0;     // initial inner orbital elements
    double i_mut0;           // initial mutual inclination (rad)
    int pid_in1 = 0, pid_in2 = 1, pid_out = 2;  // inner pair = closest pair at init (ids carry no hierarchy)
    bool initialized;

    TripleTracker() : initialized(false) {}

    // Find array index of particle with given permanent ID
    static int find_by_pid(Particles& p, int target_pid) {
        auto h_pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);
        for (int i = 0; i < p.N; i++) {
            if (h_pid(i) == target_pid) return i;
        }
        return -1;
    }

    // Compute inner binary relative state and outer relative state
    // (outer body relative to inner COM) using particle_id lookup.
    void compute_elements(Particles& p, double G,
                          OrbitalElements& oe_in, OrbitalElements& oe_out,
                          double& i_mut, double& C_K) {
        (void)G;
        // Look up particles by permanent ID (survives partition_by_level reordering)
        int i0 = find_by_pid(p, pid_in1);  // inner body 1
        int i1 = find_by_pid(p, pid_in2);  // inner body 2
        int i2 = find_by_pid(p, pid_out);  // outer body
        if (i0 < 0 || i1 < 0 || i2 < 0) return;

        auto h_x  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
        auto h_y  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
        auto h_z  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
        auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
        auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
        auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
        auto h_m  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);

        // Inner binary: relative vector
        double r_in[3] = { h_x(i1)-h_x(i0),  h_y(i1)-h_y(i0),  h_z(i1)-h_z(i0)  };
        double v_in[3] = { h_vx(i1)-h_vx(i0), h_vy(i1)-h_vy(i0), h_vz(i1)-h_vz(i0) };
        oe_in = cartesian_to_elements(r_in, v_in, mu_in);

        // Inner COM
        double M_in = m1 + m2;
        double cx  = (m1*h_x(i0)  + m2*h_x(i1))  / M_in;
        double cy  = (m1*h_y(i0)  + m2*h_y(i1))  / M_in;
        double cz  = (m1*h_z(i0)  + m2*h_z(i1))  / M_in;
        double cvx = (m1*h_vx(i0) + m2*h_vx(i1)) / M_in;
        double cvy = (m1*h_vy(i0) + m2*h_vy(i1)) / M_in;
        double cvz = (m1*h_vz(i0) + m2*h_vz(i1)) / M_in;

        // Outer orbit: outer body relative to inner COM
        double r_out[3] = { h_x(i2)-cx,  h_y(i2)-cy,  h_z(i2)-cz  };
        double v_out[3] = { h_vx(i2)-cvx, h_vy(i2)-cvy, h_vz(i2)-cvz };
        oe_out = cartesian_to_elements(r_out, v_out, mu_out);

        // Angular momentum vectors
        double L_in[3]  = { r_in[1]*v_in[2]  - r_in[2]*v_in[1],
                            r_in[2]*v_in[0]  - r_in[0]*v_in[2],
                            r_in[0]*v_in[1]  - r_in[1]*v_in[0] };
        double L_out[3] = { r_out[1]*v_out[2] - r_out[2]*v_out[1],
                            r_out[2]*v_out[0] - r_out[0]*v_out[2],
                            r_out[0]*v_out[1] - r_out[1]*v_out[0] };

        double L_in_mag  = std::sqrt(L_in[0]*L_in[0]  + L_in[1]*L_in[1]  + L_in[2]*L_in[2]);
        double L_out_mag = std::sqrt(L_out[0]*L_out[0] + L_out[1]*L_out[1] + L_out[2]*L_out[2]);

        double cos_i = (L_in[0]*L_out[0] + L_in[1]*L_out[1] + L_in[2]*L_out[2])
                       / (L_in_mag * L_out_mag + 1e-30);
        cos_i = std::max(-1.0, std::min(1.0, cos_i));
        i_mut = std::acos(cos_i);
        C_K = (1.0 - oe_in.e * oe_in.e) * cos_i * cos_i;
    }

    void init(Particles& p, double G) {
        if (p.N != 3) return;
        {
            auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
            auto h_y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
            auto h_z = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
            auto h_pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);
            auto d2 = [&](int a, int b) {
                double dx = h_x(a) - h_x(b), dy = h_y(a) - h_y(b), dz = h_z(a) - h_z(b);
                return dx * dx + dy * dy + dz * dz;
            };
            int out = 2;  // the body not in the closest pair
            if (d2(0, 2) < d2(0, 1) && d2(0, 2) <= d2(1, 2)) out = 1;
            else if (d2(1, 2) < d2(0, 1) && d2(1, 2) < d2(0, 2)) out = 0;
            pid_out = h_pid(out); pid_in1 = h_pid((out + 1) % 3); pid_in2 = h_pid((out + 2) % 3);
        }
        int i0 = find_by_pid(p, pid_in1);
        int i1 = find_by_pid(p, pid_in2);
        int i2 = find_by_pid(p, pid_out);
        if (i0 < 0 || i1 < 0 || i2 < 0) return;
        auto h_m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
        m1 = h_m(i0); m2 = h_m(i1); m3 = h_m(i2);
        mu_in  = G * (m1 + m2);
        mu_out = G * (m1 + m2 + m3);

        OrbitalElements oe_in, oe_out;
        double i_mut;
        compute_elements(p, G, oe_in, oe_out, i_mut, C_K0);
        a_in0  = oe_in.a;
        e_in0  = oe_in.e;
        i_mut0 = i_mut;

        std::printf("TripleTracker: inner a=%.4e pc (%.4f AU), e=%.6f, P=%.4e yr\n",
                    oe_in.a, oe_in.a / AU_TO_PC, oe_in.e, oe_in.period * TIME_UNIT_YR);
        std::printf("  outer a=%.4e pc (%.4f AU), e=%.6f, P=%.4e yr\n",
                    oe_out.a, oe_out.a / AU_TO_PC, oe_out.e, oe_out.period * TIME_UNIT_YR);
        std::printf("  i_mut=%.2f deg, Kozai C_K=%.8f\n",
                    i_mut * 180.0 / M_PI, C_K0);
        std::fflush(stdout);
        initialized = true;
    }

    void update(Particles& p, double G, double t_code) {
        if (!initialized || p.N != 3) return;

        OrbitalElements oe_in, oe_out;
        double i_mut, C_K;
        compute_elements(p, G, oe_in, oe_out, i_mut, C_K);

        double dC = (std::fabs(C_K0) > 1e-30) ? (C_K - C_K0) / C_K0 : C_K - C_K0;
        double t_yr = t_code * TIME_UNIT_YR;

        std::printf("  Triple: t=%.4f yr  a_in=%.4e  e_in=%.6f  i_mut=%.2f deg"
                    "  a_out=%.4e  e_out=%.6f  C_K=%.8f  dC/C0=%.2e\n",
                    t_yr, oe_in.a, oe_in.e, i_mut * 180.0 / M_PI,
                    oe_out.a, oe_out.e, C_K, dC);
        std::fflush(stdout);
    }
};
#endif

#include <Kokkos_Core.hpp>
#include <type_traits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <chrono>
#include <iostream>
#include <cmath>

// ---------------------------------------------------------------------------
// NBODY_DUMP (diagnostics): the particle state at each print, rows (pid m x y z vx vy vz subsystem_id) of doubles
// after the int N, to dump_<step>.bin in the run directory (step 0: the initial state)
static void dump_particles(const Particles& p, int step) {
    if (!std::getenv("NBODY_DUMP")) return;
    char fn[64];
    std::snprintf(fn, sizeof fn, "dump_%04d.bin", step);
    auto h = [](const auto& v) { return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v); };
    auto hx = h(p.x), hy = h(p.y), hz = h(p.z), hvx = h(p.vx), hvy = h(p.vy), hvz = h(p.vz), hm = h(p.mass);
    auto hp = h(p.particle_id), hs = h(p.subsystem_id);
    FILE* f = std::fopen(fn, "wb");
    if (!f) return;
    const int n = p.N;
    std::fwrite(&n, sizeof n, 1, f);
    for (int i = 0; i < n; i++) {
        const double r[9] = {(double)hp(i), hm(i), hx(i), hy(i), hz(i), hvx(i), hvy(i), hvz(i), (double)hs(i)};
        std::fwrite(r, sizeof r, 1, f);
    }
    std::fclose(f);
}

// BINARY_CHI1 / BINARY_CHI2 = "cx,cy,cz" (or space-separated): dimensionless spin vectors of the two binary_test bodies
static void set_binary_spins(Particles& p, const Config& cfg) {
#ifdef ENABLE_COLLISIONAL
    if (!p.collisional) return;
    auto h_spin = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.spin_vec);
    bool any = false;
    for (int b = 0; b < 2; b++) {
        std::string s = cfg.get_str(b == 0 ? "BINARY_CHI1" : "BINARY_CHI2");
        double c[3] = {0, 0, 0};
        for (char& ch : s) if (ch == ',' || ch == ';') ch = ' ';
        if (s.empty() || std::sscanf(s.c_str(), "%lf %lf %lf", &c[0], &c[1], &c[2]) != 3) continue;
        for (int k = 0; k < 3; k++) h_spin(b, k) = c[k];
        any = true;
    }
    if (any) Kokkos::deep_copy(p.spin_vec, h_spin);
#else
    (void)p; (void)cfg;
#endif
}

static void print_usage(const char* prog) {
    std::printf("Usage: %s [--config <file>] [--test <case>]\n", prog);
    std::printf("  --config <file>   Config file (default: config.txt)\n");
    std::printf("  --test <case>     Override TEST_CASE from config\n");
    std::printf("  --restart <file>  Continue from a restart file (\"latest\": newest in RESTART_DIR, \"auto\": that or a fresh start)\n");
    std::printf("  -h, --help        Show this help\n");
}

static const char* outer_planet_names[] = {"Sun", "Jupiter", "Saturn", "Uranus", "Neptune"};

int main(int argc, char** argv) {
    // Initialize MPI before Kokkos (MPI may set device affinity)
    MPIComm::init(&argc, &argv);

    Kokkos::initialize(argc, argv);
    const auto prog_start = std::chrono::steady_clock::now();   // WALLTIME_LIMIT counts from here
    fence_stats::init();
    int exit_code = 0;
    {
        int my_rank = MPIComm::rank();
        int n_ranks = MPIComm::size();

        if (my_rank == 0) {
            std::printf("Kokkos configuration:\n"); std::fflush(stdout);
            Kokkos::print_configuration(std::cout);
            std::cout << std::flush;
            if (n_ranks > 1) {
                std::printf("\nMPI: %d ranks\n", n_ranks);
#if EFFECTIVE_GPU_AWARE_MPI
                std::printf("GPU-aware MPI: ENABLED\n");
#else
                std::printf("GPU-aware MPI: disabled (host-staged transfers)\n");
#endif
            }
        }
        MPIComm::barrier();

        std::string config_file = "config.txt";
        std::string test_override;
        std::string restart_arg;

        for (int i = 1; i < argc; i++) {
            if ((!std::strcmp(argv[i], "--config") || !std::strcmp(argv[i], "-c")) && i+1 < argc)
                config_file = argv[++i];
            else if (!std::strcmp(argv[i], "--test") && i+1 < argc)
                test_override = argv[++i];
            else if (!std::strcmp(argv[i], "--restart") && i+1 < argc)
                restart_arg = argv[++i];
            else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
                print_usage(argv[0]); Kokkos::finalize(); MPIComm::finalize(); return 0;
            }
            // Silently skip Kokkos args (--kokkos-*)
            else if (std::strncmp(argv[i], "--kokkos", 8) == 0) {
                // Skip value arg if present
                if (i+1 < argc && argv[i+1][0] != '-') i++;
            }
            else {
                std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
                print_usage(argv[0]); Kokkos::finalize(); MPIComm::finalize(); return 1;
            }
        }

        // Parameters (core/params.h): unknown keys and bad values in the file are errors; every key's effective
        // value and its source go to run_params.txt (rank 0)
        Config cfg;
        cfg.load(config_file, my_rank == 0);
        if (!test_override.empty()) cfg.set_override("TEST_CASE", test_override);
        if (!restart_arg.empty()) cfg.set_override("RESTART_FILE", restart_arg);
        if (my_rank == 0) cfg.write_effective("run_params.txt");

        std::string test = cfg.get_str("TEST_CASE");
        std::string integrator = cfg.get_str("INTEGRATOR");
        double t_final_myr  = cfg.get_double("TOTAL_TIME_MYR");
        double dt_max_myr   = cfg.get_double("DT_MAX_MYR");
        double dt_min_myr   = cfg.get_double("DT_MIN_MYR");
        double eta          = cfg.get_double("ETA");
        double soft_pc      = cfg.get_double("SOFTENING_PC");
        int print_interval  = cfg.get_int("PRINT_INTERVAL");
        int replica_check_interval = cfg.get_int("REPLICA_CHECK_INTERVAL");  // steps; 0 = off
        int N_plummer       = cfg.get_int("PLUMMER_N");
        double M_total_msun = cfg.get_double("PLUMMER_M_TOTAL");
        // PLUMMER_R_HALF absent or "mass_size": r_h from the SCIC mass-size relation (BIFROST App. A eq. A2-A3)
        std::string r_half_str = cfg.get_str("PLUMMER_R_HALF");
        double r_half_pc    = (r_half_str == "mass_size") ? -1.0 : Config::to_double("PLUMMER_R_HALF", r_half_str);
        int seed            = cfg.get_int("PLUMMER_SEED");
        std::string mass_dist = cfg.get_str("PLUMMER_MASS_DIST");
        double age_gyr      = cfg.get_double("PLUMMER_AGE_GYR");
        double metallicity  = cfg.get_double("PLUMMER_METALLICITY");
        bool remove_bh      = cfg.get_bool("REMOVE_BH");
        double binary_frac  = cfg.get_double("BINARY_FRAC");
        double bin_a_min_au = cfg.get_double("BINARY_A_MIN_AU");
        double bin_a_max_au = cfg.get_double("BINARY_A_MAX_AU");
        std::string binary_pop = cfg.get_str("BINARY_POP");
        bool binary_peri_check = cfg.get_bool("BINARY_PERI_CHECK");
        bool add_imbh       = cfg.get_bool("ADD_IMBH");
        double imbh_m1      = cfg.get_double("IMBH_M1");
        double imbh_m2      = cfg.get_double("IMBH_M2");
        double imbh_a_pc    = cfg.get_double("IMBH_A_PC");
        double imbh_e       = cfg.get_double("IMBH_E");
        double imbh_spin    = cfg.get_double("IMBH_SPIN");
        // Artificial-binary avoidance (BIFROST App. A): IC_R_BREAK = r_break in units of BINARY_A_MAX_AU ("r_break ~ a
        // few times a_max", Table A1); R_BREAK_AU (absolute, AU) takes precedence when > 0
        double r_break_au   = cfg.get_double("R_BREAK_AU");
        if (r_break_au <= 0.0) r_break_au = cfg.get_double("IC_R_BREAK") * bin_a_max_au;
        // Binary centre-of-mass velocity: "sampled" (one f(E) draw per binary, SCIC/mcluster) or "average" (old)
        std::string binary_com_v = cfg.get_str("BINARY_COM_V");

        // SSE model config (used by IC generator and runtime SSE)
        std::string sse_wind_str    = cfg.get_str("SSE_WIND_MODEL");
        std::string sse_remnant_str = cfg.get_str("SSE_REMNANT_MODEL");
        std::string sse_ppisn_str   = cfg.get_str("SSE_PPISN_MODEL");
        std::string sse_ecsn_str    = cfg.get_str("SSE_ECSN_MODEL");
        double sse_mxns_val         = cfg.get_double("SSE_MXNS");
        double sse_flbv_val         = cfg.get_double("SSE_FLBV");
        SSEConfig ic_sse_cfg = make_sse_config(sse_wind_str, sse_remnant_str,
                                                sse_ppisn_str, sse_ecsn_str,
                                                sse_mxns_val, sse_flbv_val);

        bool profiling_enabled = cfg.get_bool("PROFILING");
        g_profiler.enabled = profiling_enabled;   // PROFILING = false: no timers and no device fences
        g_profiler.fence = profiling_enabled && cfg.get_bool("PROFILING_FENCES");
        serial_n2_gpu = cfg.get_int("SERIAL_N2_GPU");
        serial_n2_cpu = cfg.get_int("SERIAL_N2_CPU");
        all_slow_split = cfg.get_bool("ALL_SLOW_SPLIT");
        hhs_cc = cfg.get_bool("HHS_CC");
        hhs_cc_max = cfg.get_int("HHS_CC_MAX");
        hhs_cc_min = cfg.get_int("HHS_CC_MIN");
        bool ngb_mass_scaling = false;
        double ngb_mref = 0.0;
        flyby_cm_velocity = cfg.get_bool("FLYBY_CM_VELOCITY");   // FROST-CLUSTERS II sec. 2.1
        g_eattr.on = std::getenv("NBODY_EATTR") != nullptr;   // per-print energy attribution (diagnostics.h)
        distribute_subsystems = cfg.get_bool("DISTRIBUTE_SUBSYSTEMS");   // FROST Algorithm 3
        distribute_subsystems_min = cfg.get_int("DISTRIBUTE_MIN_SUBSYSTEMS");
        use_integer_positions = cfg.get_bool("INTEGER_POSITIONS");
        double domain_halfsize = cfg.get_double("DOMAIN_HALFSIZE_PC");
        use_nonblocking_mpi = cfg.get_bool("MPI_NON_BLOCKING");
        mpi_transport = parse_mpi_transport(cfg.get_str("MPI_TRANSPORT"));   // AUDIT G row 4

        // Barnes-Hut tree configuration
        use_tree = cfg.get_bool("USE_TREE");
        tree_threshold = cfg.get_int("TREE_THRESHOLD");
        tree_theta = cfg.get_double("TREE_THETA");
        tree_n_leaf = cfg.get_int("TREE_N_LEAF");
        tree_n_group = cfg.get_int("TREE_N_GROUP");
        // Exact tree searches for the HHS time steps and neighbours (gravity/dt_tree.h); forces stay direct
        // Exact tree searches on by default above the measured break-even set sizes (AUDIT F7; results are
        // bit-identical to O(N^2), so only speed changes). Host builds: ~1000-1500 for both searches (32 threads).
        // GPUs (A100): neighbours ~25k, time steps ~30k with component velocities, ~150k with FLYBY_CM_VELOCITY
        // (fewer prunable pairs). GPU values also serve APUs until measured there. With FLYBY_CM_VELOCITY and the
        // tidal-sampling step the 200k cluster ran 4.12 s/step with the tree against 3.96 s without (verify-gpuopt,
        // 2026-10-11), so the c.o.m. threshold sits above 200k.
        constexpr bool host_build = std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>;
        const int dt_default = host_build ? 2048 : (flyby_cm_velocity ? 300000 : 65536);
        tree_dt_min_n = cfg.get_int("TREE_DT_MIN_N");        // -1: the build's default
        if (tree_dt_min_n < 0) tree_dt_min_n = dt_default;
        tree_ngb_min_n = cfg.get_int("TREE_NGB_MIN_N");
        if (tree_ngb_min_n < 0) tree_ngb_min_n = host_build ? 2048 : 32768;
        tree_dt_cpu_max = cfg.get_int("TREE_DT_CPU_MAX");
        tree_dt_vc_fac = cfg.get_double("TREE_DT_VC_FAC");

#ifdef ENABLE_COLLISIONAL
        // ETA_FF/ETA_FB always parsed (used by force kernels even without subsystems)
        g_collisional.eta_ff          = cfg.get_double("ETA_FF");
        g_collisional.eta_fb          = cfg.get_double("ETA_FB");

        // Collisional (unsoftened direct N-body) configuration
        g_collisional.enabled = cfg.get_bool("COLLISIONAL");
        if (g_collisional.enabled && use_tree) {
            // Merger compaction re-indexes particles under a built tree, and tree forces lose 4th order and
            // L conservation (force-error floor ~1e-5 at theta 0.5): not supported for collisional runs
            if (MPIComm::rank() == 0)
                std::fprintf(stderr, "ERROR: USE_TREE is not supported with COLLISIONAL = true\n");
            Kokkos::finalize();
            MPIComm::finalize();
            return 1;
        }
        if (g_collisional.enabled) {
            g_collisional.r_ngb           = cfg.get_double("R_NGB");
            g_collisional.r_ngb_max       = cfg.get_double("R_NGB_MAX");
            g_collisional.eta_grad        = cfg.get_double("ETA_GRAD");
            g_collisional.eta_jerk        = cfg.get_double("ETA_JERK");
            g_collisional.eta_va          = eta;
            g_collisional.eta_gbs         = cfg.get_double("ETA_GBS");
            g_collisional.eta_pert        = cfg.get_double("ETA_PERT");
            g_collisional.tidal_gamma     = cfg.get_double("TIDAL_GAMMA");
            g_collisional.weak_perturber_gamma = cfg.get_double("WEAK_PERTURBER_GAMMA");
            g_collisional.flyby_ngb_cm    = cfg.get_bool("FLYBY_NGB_CM");
            ngb_mass_scaling              = cfg.get_bool("NGB_MASS_SCALING");
            ngb_mref                      = cfg.get_double("NGB_MREF");   // <m> override [Msun] (tests)
            g_collisional.ngb_mass_gamma  = cfg.get_double("NGB_MASS_GAMMA");
            g_collisional.tidal_step_res  = cfg.get_double("TIDAL_STEP_RES");
            g_collisional.secular_triples = cfg.get_bool("SECULAR_TRIPLES");
            g_collisional.sdar            = cfg.get_bool("SDAR");
            if (const char* e = std::getenv("NBODY_SDAR")) g_collisional.sdar = std::atoi(e) != 0;
            g_collisional.sdar_kref       = cfg.get_double("SDAR_KREF");
            g_collisional.sdar_c          = cfg.get_double("SDAR_C");
            g_collisional.subsys_collisions = cfg.get_bool("SUBSYS_COLLISIONS");
            g_collisional.tde_accretion = cfg.get_double("TDE_ACCRETION_FRACTION");
            g_collisional.coll_mass_loss_c = cfg.get_double("COLL_MASS_LOSS_C");
            g_collisional.bh_merge_max_single = cfg.get_str("BH_MERGE_RADIUS") == "max_single";
            g_collisional.grad_fresh = cfg.get_bool("GRAD_FRESH");
            g_collisional.gbs_deuflhard = cfg.get_str("GBS_SEQUENCE") == "deuflhard";
            g_collisional.gbs_kmax = cfg.get_int("GBS_KMAX");
            g_collisional.r_ngb_wide = cfg.get_double("NGB_WIDE_FAC") * g_collisional.r_ngb;
            g_collisional.tidal_res       = cfg.get_double("TIDAL_RES");
            g_collisional.flyby_cm_eps    = cfg.get_double("FLYBY_CM_EPS");
            g_collisional.eta_restart     = cfg.get_double("ETA_RESTART");
            g_collisional.n_orb_secular   = cfg.get_int("N_ORB_SECULAR");
            g_collisional.kepler_2body    = cfg.get_bool("KEPLER_2BODY");
            g_collisional.m_pn            = cfg.get_double("M_PN");
            g_collisional.pn_order        = cfg.get_int("PN_ORDER");
            g_collisional.r_escape_rh     = cfg.get_double("R_ESCAPE_RH");
            g_collisional.r_escape_rj     = cfg.get_double("R_ESCAPE_RJ");
            g_collisional.mstar_threshold = cfg.get_int("MSTAR_THRESHOLD");
            g_collisional.sse_enabled     = cfg.get_bool("SSE_ENABLED");
            g_collisional.sse_metallicity = cfg.get_double("SSE_METALLICITY");
            g_collisional.sse_wind        = sse_wind_str;
            g_collisional.sse_remnant     = sse_remnant_str;
            g_collisional.sse_ppisn       = sse_ppisn_str;
            g_collisional.sse_ecsn        = sse_ecsn_str;
            g_collisional.sse_mxns        = sse_mxns_val;
            g_collisional.sse_flbv        = sse_flbv_val;
            g_collisional.sse_natal_kicks = cfg.get_str("SSE_NATAL_KICKS");
            g_collisional.kick_seed       = seed;
            g_collisional.compute_derived();

            // Collisional mode requires zero softening
            if (soft_pc > 0.0 && my_rank == 0) {
                std::printf("WARNING: COLLISIONAL=true but SOFTENING_PC=%.2e > 0. "
                            "Setting softening to 0.\n", soft_pc);
            }
            soft_pc = 0.0;
        }
#endif

        // Restart files and snapshots (io/restart.h, io/snapshot.h). RESTART_INTERVAL / SNAPSHOT_INTERVAL in steps
        // (0: off); a restart file is also written at the end of the run and before WALLTIME_LIMIT (hours) or on
        // SIGUSR1 (both stop the run after the step). RESTART_FILE (or --restart): continue from that file.
        const int restart_interval = cfg.get_int("RESTART_INTERVAL");
        const double walltime_limit_s = cfg.get_double("WALLTIME_LIMIT") * 3600.0;
        const int snapshot_interval = cfg.get_int("SNAPSHOT_INTERVAL");
        restart_arg = cfg.get_str("RESTART_FILE");
#ifdef USE_HDF5
        RestartOptions restart_opt;
        restart_opt.dir = cfg.get_str("RESTART_DIR");
        restart_opt.keep = cfg.get_int("RESTART_KEEP");
        SnapshotOptions snap_opt;
        snap_opt.dir = cfg.get_str("SNAPSHOT_DIR");
        snap_opt.base = cfg.get_str("SNAPSHOT_BASE");
        snap_opt.potential = cfg.get_bool("SNAPSHOT_POTENTIAL");
        // "latest": the newest restart file in RESTART_DIR (must exist); "auto": the same, or a fresh start from
        // the initial conditions if there is none (one job script for every job of a chain)
        if (restart_arg == "latest" || restart_arg == "auto") {
            const bool must = restart_arg == "latest";
            restart_arg = latest_restart(restart_opt.dir);
            if (restart_arg.empty() && must) {
                if (my_rank == 0) std::fprintf(stderr, "ERROR: no restart file in %s\n", restart_opt.dir.c_str());
                Kokkos::finalize(); MPIComm::finalize(); return 1;
            }
        }
        const bool restart_output = restart_interval > 0 || walltime_limit_s > 0.0;
        if (restart_output) install_stop_signals();
#else
        if (restart_interval > 0 || walltime_limit_s > 0.0 || snapshot_interval > 0 || !restart_arg.empty()) {
            if (my_rank == 0)
                std::fprintf(stderr, "ERROR: restart files and snapshots need a build with HDF5 (USE_HDF5)\n");
            Kokkos::finalize(); MPIComm::finalize(); return 1;
        }
#endif
        const bool resuming = !restart_arg.empty();

        double G = G_CODE;
        double soft2 = soft_pc * soft_pc;
        double dt_max = dt_max_myr / TIME_UNIT_MYR;   // Myr → code TU
#ifdef ENABLE_COLLISIONAL
        g_eattr.dt_cap = dt_max;
#endif
        double dt_min = dt_min_myr / TIME_UNIT_MYR;   // Myr → code TU
        double t_final = t_final_myr / TIME_UNIT_MYR;  // Myr → code TU

        if (use_integer_positions) {
            int_pos_scale = domain_halfsize / (double)(1LL << 62);
            int_pos_inv_scale = 1.0 / int_pos_scale;
        }

        Particles p;
        DistributedParticles dp;  // For multi-GPU mode
        const char* const* names = nullptr;
        bool use_distributed = MPIComm::is_distributed();

        if (my_rank == 0) {
            std::printf("Initializing particles...\n"); std::fflush(stdout);
        }
        auto t_init0 = std::chrono::high_resolution_clock::now();

        // Initialize particles on rank 0 (or all ranks for single-GPU)
        // Binary test parameters
        double binary_a_pc  = cfg.get_double("BINARY_A_PC");
        double binary_e     = cfg.get_double("BINARY_E");
        double binary_m1    = cfg.get_double("BINARY_M1");
        double binary_m2    = cfg.get_double("BINARY_M2");
        double binary_rot   = cfg.get_double("BINARY_ROT_DEG");

        // Triple test parameters
        double triple_a_in  = cfg.get_double("TRIPLE_A_IN_PC");
        double triple_e_in  = cfg.get_double("TRIPLE_E_IN");
        double triple_m1    = cfg.get_double("TRIPLE_M1");
        double triple_m2    = cfg.get_double("TRIPLE_M2");
        double triple_a_out = cfg.get_double("TRIPLE_A_OUT_PC");
        double triple_e_out = cfg.get_double("TRIPLE_E_OUT");
        double triple_m3    = cfg.get_double("TRIPLE_M3");
        double triple_incl  = cfg.get_double("TRIPLE_INCL_DEG");
        bool triple_outer_first = cfg.get_bool("TRIPLE_OUTER_FIRST");
        double triple_in_rot = cfg.get_double("TRIPLE_IN_ROT_DEG");
        double triple_in_start_q = cfg.get_double("TRIPLE_IN_START_Q");
        double triple_out_start_q = cfg.get_double("TRIPLE_OUT_START_Q");

#ifdef USE_HDF5
        RunState rs;   // run-loop state of a restart (written and read by io/restart.cpp)
        if (resuming) {
            // Every rank reads the file: particles, subsystems, books, merger tree, escaper log, run state
            try {
                read_restart(restart_arg, p, rs, true);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "ERROR: restart from %s: %s\n", restart_arg.c_str(), e.what());
                Kokkos::finalize(); MPIComm::finalize(); return 1;
            }
            if (rs.integrator != integrator || rs.test != test)
                std::fprintf(stderr, "WARNING: restart file was written by INTEGRATOR=%s TEST_CASE=%s\n",
                             rs.integrator.c_str(), rs.test.c_str());
            if (my_rank == 0)
                std::printf("Restart: %s, step %d, t = %.10e Myr, N = %d\n", restart_arg.c_str(), rs.step,
                            rs.t * TIME_UNIT_MYR, p.N);
        }
#endif
        if (resuming) {
            if (test == "outer_planets") names = outer_planet_names;
        } else if (my_rank == 0 || !use_distributed) {
            if (test == "outer_planets") {
                init_outer_planets(p);
                names = outer_planet_names;
            } else if (test == "plummer") {
                init_plummer(p, N_plummer, M_total_msun, r_half_pc, seed, mass_dist, age_gyr, metallicity,
                             remove_bh, binary_frac, bin_a_min_au, bin_a_max_au,
                             add_imbh, imbh_m1, imbh_m2, imbh_a_pc, imbh_e, imbh_spin,
                             r_break_au, &ic_sse_cfg, binary_pop, binary_peri_check,
                             binary_com_v == "sampled");
            } else if (test == "binary_test") {
                init_binary_test(p, binary_a_pc, binary_e, binary_m1, binary_m2, binary_rot);
                set_binary_spins(p, cfg);
            } else if (test == "binary_test_peri") {
                init_binary_test_peri(p, binary_a_pc, binary_e, binary_m1, binary_m2);
                set_binary_spins(p, cfg);
            } else if (test == "triple_test") {
                init_triple_test(p, triple_a_in, triple_e_in, triple_m1, triple_m2,
                                 triple_a_out, triple_e_out, triple_m3, triple_incl, triple_outer_first,
                                 triple_in_rot, triple_in_start_q, triple_out_start_q);
            } else {
                std::fprintf(stderr, "Unknown test case: %s\n", test.c_str());
                MPIComm::finalize();
                Kokkos::finalize();
                return 1;
            }
        }

#ifdef ENABLE_COLLISIONAL
        // Test cases (binary_test, triple_test): stellar radius of every non-BH body (collision tests; 0: none)
        {
            const double r_test = cfg.get_double("TEST_STELLAR_RADIUS_RSUN");
            if (!resuming && r_test > 0.0 && test != "plummer" && p.collisional && (my_rank == 0 || !use_distributed)) {
                auto h_st = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_type);
                auto h_sr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_radius);
                for (int i = 0; i < p.N; i++) h_sr(i) = h_st(i) < 10 ? r_test : 0.0;
                Kokkos::deep_copy(p.stellar_radius, h_sr);
                StellarPropsCache::get().invalidate();
            }
        }
        // Test cases: TEST_BOOST_KMS = "vx,vy,vz" adds a uniform velocity to every body (frame-invariance tests, e.g.
        // of the PN terms: KETJU, Mannerkoski et al. 2023, sec. 2.2)
        {
            const std::string boost = cfg.get_str("TEST_BOOST_KMS");
            if (!resuming && !boost.empty() && test != "plummer" && (my_rank == 0 || !use_distributed)) {
                double bv[3] = {0, 0, 0};
                std::sscanf(boost.c_str(), "%lf%*[, ]%lf%*[, ]%lf", &bv[0], &bv[1], &bv[2]);
                auto hvx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
                auto hvy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
                auto hvz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
                for (int i = 0; i < p.N; i++) { hvx(i) += bv[0]; hvy(i) += bv[1]; hvz(i) += bv[2]; }
                Kokkos::deep_copy(p.vx, hvx); Kokkos::deep_copy(p.vy, hvy); Kokkos::deep_copy(p.vz, hvz);
            }
        }
#endif

        // Convert initial positions to integer representation (before MPI distribution)
        if (!resuming && use_integer_positions && (my_rank == 0 || !use_distributed)) {
            auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
            auto h_y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
            auto h_z = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
            auto h_ix = Kokkos::create_mirror_view(p.ix);
            auto h_iy = Kokkos::create_mirror_view(p.iy);
            auto h_iz = Kokkos::create_mirror_view(p.iz);
            for (int i = 0; i < p.N; i++) {
                h_ix(i) = llrint(h_x(i) * int_pos_inv_scale);
                h_iy(i) = llrint(h_y(i) * int_pos_inv_scale);
                h_iz(i) = llrint(h_z(i) * int_pos_inv_scale);
                h_x(i) = (double)h_ix(i) * int_pos_scale;  // snap to integer grid
                h_y(i) = (double)h_iy(i) * int_pos_scale;
                h_z(i) = (double)h_iz(i) * int_pos_scale;
            }
            Kokkos::deep_copy(p.ix, h_ix);
            Kokkos::deep_copy(p.iy, h_iy);
            Kokkos::deep_copy(p.iz, h_iz);
            Kokkos::deep_copy(p.x, h_x);
            Kokkos::deep_copy(p.y, h_y);
            Kokkos::deep_copy(p.z, h_z);
        }

        // Distribute particles across ranks if using MPI
        if (use_distributed) {
            dp.distribute_initial(p);
#ifdef USE_HDF5
            // The broadcast carries only some fields: every rank overwrites its replica with the whole saved state
            if (resuming) {
                try {
                    read_restart(restart_arg, dp.local, rs, false);
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "ERROR: restart from %s: %s\n", restart_arg.c_str(), e.what());
                    MPIComm::finalize(); return 1;
                }
            }
#endif
            // After distribution, each rank works with dp.local
            // Rank 0's 'p' is no longer valid - use dp for all operations
        }

        auto t_init1 = std::chrono::high_resolution_clock::now();
        if (my_rank == 0) {
            std::printf("Particle init: %.3f s\n", std::chrono::duration<double>(t_init1-t_init0).count());
            std::fflush(stdout);
        }

        // Check for duplicate positions in initial conditions (sort-based O(N log N))
        if (my_rank == 0) {
            Particles& check_p = use_distributed ? dp.local : p;
            auto hx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), check_p.x);
            auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), check_p.y);
            auto hz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), check_p.z);
            std::vector<int> idx(check_p.N);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](int a, int b) {
                if (hx(a) != hx(b)) return hx(a) < hx(b);
                if (hy(a) != hy(b)) return hy(a) < hy(b);
                return hz(a) < hz(b);
            });
            int n_dup = 0;
            for (int k = 1; k < check_p.N; k++) {
                int i = idx[k-1], j = idx[k];
                if (hx(i) == hx(j) && hy(i) == hy(j) && hz(i) == hz(j)) {
                    std::printf("  IC DUPLICATE: idx %d and %d at (%.15e, %.15e, %.15e)\n",
                                i, j, hx(i), hy(i), hz(i));
                    if (++n_dup >= 20) break;
                }
            }
            if (n_dup == 0)
                std::printf("  IC check: no duplicate positions found\n");
        }

        bool is_cluster = (test == "plummer");

        // Get particle count (local or total depending on mode)
        int N_display = use_distributed ? dp.N_total : p.N;

        const char* mode_str = (integrator == "hhs") ? "hhs" : "fsi";
        if (my_rank == 0) {
            std::printf("=== Kokkos N-Body: %s, %s test, N=%d ===\n",
                        mode_str, test.c_str(), N_display);
            if (use_distributed) {
                std::printf("    Distributed mode: %d ranks, %d local particles\n",
                            n_ranks, dp.N_local);
            }
            std::fflush(stdout);
            std::printf("G=%.4e  soft=%.4e pc  dt_max=%.4e Myr  dt_min=%.4e Myr",
                        G, soft_pc, dt_max_myr, dt_min_myr);
#ifdef ENABLE_COLLISIONAL
            std::printf("  eta_ff=%.4f  eta_fb=%.4f\n",
                        g_collisional.eta_ff, g_collisional.eta_fb);
#else
            std::printf("  eta=%.4f\n", eta);
#endif
            std::printf("t_final=%.4e Myr  print_interval=%d  profiling=%s\n",
                        t_final_myr, print_interval,
                        profiling_enabled ? "true" : "false");
            if (use_integer_positions) {
                std::printf("Integer positions: domain=%.1f pc, resolution=%.2e pc\n",
                            domain_halfsize, int_pos_scale);
            }
            if (use_tree) {
                std::printf("Barnes-Hut tree: theta=%.2f, n_leaf=%d, n_group=%d, threshold=%d\n",
                            tree_theta, tree_n_leaf, tree_n_group, tree_threshold);
            }
#ifdef ENABLE_COLLISIONAL
            if (g_collisional.enabled) {
                std::printf("Collisional mode: r_ngb=%.2e, r_ngb_max=%.2e, eta_gbs=%.2e\n",
                            g_collisional.r_ngb, g_collisional.r_ngb_max, g_collisional.eta_gbs);
                std::printf("  PN order=%d, M_PN=%.1f M_sun, MSTAR threshold=%d\n",
                            g_collisional.pn_order, g_collisional.m_pn, g_collisional.mstar_threshold);
                if (g_collisional.sdar)
                    std::printf("  SDAR slow-down in LogH subsystems: k_ref=%.1e, c=%.3g\n",
                                g_collisional.sdar_kref, g_collisional.sdar_c);
                if (g_collisional.sse_enabled) {
                    std::printf("  SSE: Z=%.4f, wind=%s, remnant=%s, ppisn=%s, ecsn=%s, natal kicks=%s\n",
                                g_collisional.sse_metallicity,
                                g_collisional.sse_wind.c_str(),
                                g_collisional.sse_remnant.c_str(),
                                g_collisional.sse_ppisn.c_str(),
                                g_collisional.sse_ecsn.c_str(),
                                g_collisional.sse_natal_kicks.c_str());
                }
            }
#endif
            std::printf("\n");
            std::fflush(stdout);
            if (profiling_enabled) {
                g_profiler.print_memory_report(N_display, n_ranks);
            }
        }

        auto levels = build_levels(dt_max, dt_min);

        // Get reference to working particles (local for distributed, p for single)
        Particles& work_p = use_distributed ? dp.local : p;

#ifdef ENABLE_COLLISIONAL
        // Allocate neighbor pair buffers before initial force computation
        // so the force kernels can detect close pairs
        if (g_collisional.enabled) {
            int max_pairs = std::max(1000, work_p.N / 10);
            g_subsystem_mgr.allocate_ngb_buffers(max_pairs);
            g_subsystem_mgr.reset_ngb_count();
            if (my_rank == 0) {
                std::printf("Collisional: neighbor buffer allocated for %d pairs\n", max_pairs);
                std::fflush(stdout);
            }
        }

        // Initialize SSE context for runtime stellar evolution
        SSEContext* g_sse_ctx = nullptr;
        if (g_collisional.enabled && g_collisional.sse_enabled) {
            auto sse_cfg = make_sse_config(g_collisional.sse_wind,
                                            g_collisional.sse_remnant,
                                            g_collisional.sse_ppisn,
                                            g_collisional.sse_ecsn,
                                            g_collisional.sse_mxns,
                                            g_collisional.sse_flbv);
            g_sse_ctx = new SSEContext(g_collisional.sse_metallicity, sse_cfg);
        }
#endif

#ifdef ENABLE_COLLISIONAL
        // FLYBY_CM_EPS scale: |E0| (the energy the run's |dE/E| refers to)
        if (!resuming && g_collisional.enabled && g_collisional.flyby_cm_eps > 0.0)   // restarts restore it
            g_collisional.e_ref = std::fabs(compute_diagnostics(work_p, G, soft2).E_total);
        // NGB_MASS_SCALING: <m> of the initial particles (fixed for the run)
        // (restarts restore it: recomputing it from the evolved particles would change the neighbour radius)
        if (g_collisional.enabled && ngb_mass_scaling && work_p.N > 0) {
            if (!resuming) {
                double m_sum = 0.0;
                auto pm = work_p.mass;
                Kokkos::parallel_reduce("ngb_mref", work_p.N, KOKKOS_LAMBDA(int i, double& s) { s += pm(i); }, m_sum);
                g_collisional.ngb_inv_mref = work_p.N / m_sum;
                if (ngb_mref > 0.0) g_collisional.ngb_inv_mref = 1.0 / ngb_mref;
            }
            if (my_rank == 0) std::printf("NGB_MASS_SCALING: <m> = %.4f Msun\n", 1.0 / g_collisional.ngb_inv_mref);
        }
#endif
        // Initial forces, subsystems and levels (a restart continues from the saved ones instead)
        if (!resuming) {
            // Initial force computation + adaptive timestep
            if (my_rank == 0) {
                std::printf("Computing initial forces...\n"); std::fflush(stdout);
            }
            auto t0 = std::chrono::high_resolution_clock::now();

            if (use_distributed) {
                compute_acceleration_and_timestep_distributed(dp, G, soft2, eta);
            } else {
                compute_acceleration_and_timestep(work_p, G, soft2, eta);
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            if (my_rank == 0) {
                std::printf("Initial forces: %.3f s\n", std::chrono::duration<double>(t1-t0).count());
                std::fflush(stdout);
            }

#ifdef ENABLE_COLLISIONAL
            // Initial subsystem detection using standalone neighbor search
            if (g_collisional.enabled) {
                g_subsystem_mgr.reset_ngb_count();
                neighbor_search_subset(work_p, 0, work_p.N, G,
                                       g_collisional.r_ngb, g_collisional.r_ngb_max, dt_max);
                g_subsystem_mgr.detect_from_pairs(work_p, G, dt_max,
                                                   g_collisional.r_ngb, g_collisional.r_ngb_max,
                                                   g_collisional.n_orb_secular);
                g_subsystem_mgr.update_particle_flags(work_p);
                g_subsystem_mgr.reset_step_diagnostics();
                g_subsystem_mgr.accumulate_diagnostics();
                if (my_rank == 0) g_subsystem_mgr.print_step_summary();
            }
#endif

            assign_levels(work_p, levels);
            work_p.partition_by_level();
        }
#ifdef ENABLE_COLLISIONAL
        if (resuming && work_p.collisional) g_subsystem_mgr.rebuild_pid_to_idx(work_p);
#endif

        if (my_rank == 0) {
            print_timebin_table(work_p, levels, -1.0, 0.0, 0.0, 0.0, 0, is_cluster);
        }

        // Print dt range
        if (my_rank == 0) {
            auto h_dt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), work_p.dt_adapt);
            double mn = 1e30, mx = 0;
            for (int i = 0; i < work_p.N; i++) {
                if (h_dt(i) < mn) mn = h_dt(i);
                if (h_dt(i) > mx) mx = h_dt(i);
            }
            std::printf("Particle dt range: [%.4e, %.4e] Myr\n\n", mn*TIME_UNIT_MYR, mx*TIME_UNIT_MYR);
        }

#ifdef ENABLE_COLLISIONAL
        BinaryTracker bin_tracker;
        SpinTracker spin_tracker;
        if ((test == "binary_test" || test == "binary_test_peri") && g_collisional.enabled && my_rank == 0) {
            bin_tracker.init(work_p, G);
            spin_tracker.init(work_p, G);
        }
        TripleTracker triple_tracker;
        if (test == "triple_test" && g_collisional.enabled && my_rank == 0) {
            triple_tracker.init(work_p, G);
        }
#endif
#ifdef USE_HDF5
        // Trackers continue from their saved state (tracker output of a restarted run equals the uninterrupted one)
        auto save_run_state = [&](double t_now, int step_now) {
            rs.t = t_now; rs.step = step_now; rs.integrator = integrator; rs.test = test;
#ifdef ENABLE_COLLISIONAL
            rs.put("bin_tracker", bin_tracker); rs.put("spin_tracker", spin_tracker);
            rs.put("triple_tracker", triple_tracker);
#endif
        };
        if (resuming) {
#ifdef ENABLE_COLLISIONAL
            rs.get("bin_tracker", bin_tracker); rs.get("spin_tracker", spin_tracker);
            rs.get("triple_tracker", triple_tracker);
#endif
        }
#endif

        auto diag0 = compute_diagnostics(work_p, G, soft2);   // collective under MPI: every rank, also on a restart
#ifdef USE_HDF5
        if (resuming) diag0 = rs.diag0;
        else rs.diag0 = diag0;
#endif
        if (my_rank == 0 && !resuming) dump_particles(work_p, 0);

        if (my_rank == 0) {
            std::printf("Running...\n"); std::fflush(stdout);
        }

        // TIDAL_FIELD = none | pointmass | loghalo (gravity/tidal_field.h): host galaxy, cluster on a circular orbit
        // at GALAXY_R_PC; GALAXY_MASS_MSUN (point mass) or GALAXY_VC_KMS, GALAXY_RC_PC (logarithmic halo)
        {
            const std::string tf = cfg.get_str("TIDAL_FIELD");
            g_tidal.type = tf == "pointmass" ? TidalField::POINT_MASS : tf == "loghalo" ? TidalField::LOG_HALO
                                                                                       : TidalField::NONE;
            if (tf != "none" && !g_tidal.on()) { std::fprintf(stderr, "Unknown TIDAL_FIELD '%s'\n", tf.c_str()); return 1; }
            g_tidal.M_gal = cfg.get_double("GALAXY_MASS_MSUN");
            g_tidal.V_c = cfg.get_double("GALAXY_VC_KMS");
            g_tidal.R_c = cfg.get_double("GALAXY_RC_PC");
            if (g_tidal.on() && !resuming) g_tidal.init_circular(cfg.get_double("GALAXY_R_PC"), G);
            if (g_tidal.on() && my_rank == 0)
                std::printf("Tidal field %s: reference at (%.6g, %.6g, %.6g) pc, v = (%.6g, %.6g, %.6g) km/s\n", tf.c_str(),
                            g_tidal.X[0], g_tidal.X[1], g_tidal.X[2], g_tidal.V[0], g_tidal.V[1], g_tidal.V[2]);
        }
        auto tidal_print = [&](int step_now) {
            if (g_tidal.on() && my_rank == 0 && step_now % print_interval == 0)
                std::printf("TIDAL step %d X %.12e %.12e %.12e V %.12e %.12e %.12e E_tidal %.6e\n", step_now, g_tidal.X[0],
                            g_tidal.X[1], g_tidal.X[2], g_tidal.V[0], g_tidal.V[1], g_tidal.V[2]
#ifdef ENABLE_COLLISIONAL
                            , g_energy_books.E_tidal
#else
                            , 0.0
#endif
                            );
        };
        auto wall_start = std::chrono::high_resolution_clock::now();

        double t = 0;
        int step = 0;
        // CLUSTER_DIAG_INTERVAL (steps, 0: off): density centre, core, Lagrangian radii and bound pairs
        // (io/cluster_diag.h) appended to CLUSTER_DIAG_FILE by rank 0 (the full replica)
        const int cluster_diag_interval = cfg.get_int("CLUSTER_DIAG_INTERVAL");
        const std::string cluster_diag_file = cfg.get_str("CLUSTER_DIAG_FILE");
        const double cluster_diag_rpair = cfg.get_double("CLUSTER_DIAG_RPAIR_PC");
        auto cluster_diag_now = [&](double t_now, int step_now) {
            if (cluster_diag_interval <= 0 || step_now % cluster_diag_interval != 0 || my_rank != 0) return;
            append_cluster_diag(cluster_diag_file, step_now,
                                compute_cluster_diag(work_p, G, t_now * TIME_UNIT_MYR, cluster_diag_rpair));
        };
        if (!resuming) cluster_diag_now(t, step);
#ifdef USE_HDF5
        if (resuming) { t = rs.t; step = rs.step; }
        int last_restart_step = resuming ? step : -1, last_snapshot_step = -1;
        int last_restart_snap = resuming ? rs.snapshot_index : -1;   // a later snapshot changes the restart state
        double max_step_wall = 0.0;
        bool stop_run = false;
        // Snapshot (all ranks enter: the diagnostics are collective) and restart file of the state after step_now
        // SNAPSHOT_CLUSTER_DIAG (default on): density centre, core, Lagrangian radii, binaries in the snapshot Header
        const bool snapshot_cluster_diag = cfg.get_bool("SNAPSHOT_CLUSTER_DIAG");
        auto write_snapshot_now = [&](double t_now, int step_now) {
            if (snapshot_interval <= 0 || last_snapshot_step == step_now) return;
            SnapshotMeta sm;
            sm.t = t_now; sm.step = step_now; sm.index = rs.snapshot_index; sm.G = G; sm.soft2 = soft2;
            sm.diag = compute_diagnostics(work_p, G, soft2);
            sm.diag0 = diag0;
            sm.config_json = config_json(cfg.params);
            if (my_rank == 0) {
                if (snapshot_cluster_diag) {
                    sm.has_cluster = true;
                    sm.cluster = compute_cluster_diag(work_p, G, t_now * TIME_UNIT_MYR, cluster_diag_rpair);
                }
                std::string path = write_snapshot(snap_opt, work_p, sm);
                std::printf("Snapshot %d written: %s\n", rs.snapshot_index, path.c_str());
                std::fflush(stdout);
            }
            rs.snapshot_index++;
            last_snapshot_step = step_now;
        };
        auto write_restart_now = [&](double t_now, int step_now) {
            if (last_restart_step == step_now && last_restart_snap == rs.snapshot_index) return;
            save_run_state(t_now, step_now);
            // Replicated MPI state: every rank must hold the same state, rank 0 writes it
            bool agree = !use_distributed || dp.replicas_agree(step_now);
            agree = ranks_agree_on_run_state(rs) && agree;
            if (!agree) { exit_code = 3; stop_run = true; return; }
            if (my_rank == 0) {
                std::string path = write_restart(restart_opt, work_p, rs, cfg.text, cfg.params);
                std::printf("Restart file written: %s (step %d, t = %.10e Myr)\n", path.c_str(), step_now,
                            t_now * TIME_UNIT_MYR);
                std::fflush(stdout);
            }
            MPIComm::barrier();
            last_restart_step = step_now;
            last_restart_snap = rs.snapshot_index;
        };
        // After every step: snapshots, periodic restart files, and the wall-clock / SIGUSR1 stop
        auto after_step_io = [&](double t_now, int step_now, double step_wall) {
            if (snapshot_interval > 0 && step_now % snapshot_interval == 0) write_snapshot_now(t_now, step_now);
            if (!restart_output) return;
            max_step_wall = std::max(max_step_wall, step_wall);
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - prog_start).count();
            // Stop when the next step (taken as 1.5 x the slowest so far) plus a minute for the file may not fit
            bool stop = stop_requested() ||
                        (walltime_limit_s > 0.0 && elapsed + 1.5 * max_step_wall + 60.0 > walltime_limit_s);
            stop = any_rank(stop);
            if (stop || (restart_interval > 0 && step_now % restart_interval == 0)) write_restart_now(t_now, step_now);
            if (stop) {
                if (my_rank == 0)
                    std::printf("STOP: %s, restart file written at step %d\n",
                                stop_requested() ? "SIGUSR1" : "WALLTIME_LIMIT reached", step_now);
                stop_run = true;
            }
        };
        if (!resuming && snapshot_interval > 0) write_snapshot_now(t, step);
#endif

        auto print_progress = [&](double t_now, int step_now) {
            auto diag = compute_diagnostics(work_p, G, soft2);
            double dE, dP, dL;
            compute_errors(diag, diag0, work_p, is_cluster, dE, dP, dL);
            if (my_rank == 0) {
                print_timebin_table(work_p, levels, dE, dP, dL, t_now*TIME_UNIT_MYR, step_now, is_cluster);
#ifdef ENABLE_COLLISIONAL
                // SDAR: |dE/E| above includes the booked slow-down energy E_sd; the physical energy alone is off
                // by E_sd (bounded oscillation of the slowed binaries' two-body energies, Wang+2020 Eq. 29)
                if (g_collisional.enabled && g_collisional.sdar)
                    std::printf("SDAR step %d E_sd/E0 %.6e unbooked_dE/E0 %.6e\n", step_now,
                                g_energy_books.E_sd / std::fabs(diag0.E_total),
                                (diag.E_total + g_energy_books.total_sink() - g_energy_books.E_sd - diag0.E_total) /
                                    std::fabs(diag0.E_total));
                dump_particles(work_p, step_now);
                if (g_eattr.on) {
                    const double E0 = std::fabs(diag0.E_total);
                    std::printf("EATTR step %d signed_dE %.6e logh %.6e logh_abs %.6e n_logh %d books: gw %.6e pn %.6e "
                                "merge %.6e esc %.6e | worst %.6e n=%d type=%d pids %d %d a=%.4e e=%.4f dt=%.3e\n",
                                step_now, (diag.E_total + g_energy_books.total_sink() - diag0.E_total) / E0,
                                g_eattr.logh / E0, g_eattr.logh_abs / E0, g_eattr.n_logh, g_energy_books.E_gw / E0,
                                g_energy_books.E_pn / E0, g_energy_books.E_merge / E0, g_energy_books.E_escape / E0,
                                g_eattr.worst / E0, g_eattr.worst_n, g_eattr.worst_type, g_eattr.worst_pid0,
                                g_eattr.worst_pid1, g_eattr.worst_a, g_eattr.worst_e, g_eattr.worst_dt * TIME_UNIT_MYR);
                    std::printf("EATTR_TID step %d max_est %.6e pids %d %d m %.3f %.3f a=%.4e e=%.4f gamma=%.3e P=%.3e dt=%.3e\n",
                                step_now, g_eattr.tid_max / E0, g_eattr.tid_pid0, g_eattr.tid_pid1, g_eattr.tid_m1,
                                g_eattr.tid_m2, g_eattr.tid_a, g_eattr.tid_e, g_eattr.tid_gamma,
                                g_eattr.tid_P * TIME_UNIT_MYR, g_eattr.tid_dt * TIME_UNIT_MYR);
                    { extern double g_sec_nonpn_gw;
                      std::printf("EATTR_SEC step %d nonpn_gw_booked %.6e\n", step_now, g_sec_nonpn_gw / E0); }
                    g_eattr.reset();
                }
#endif
#ifdef ENABLE_COLLISIONAL
                if (bin_tracker.initialized) {
                    bin_tracker.update(work_p, G, t_now);
                }
                if (spin_tracker.initialized) {
                    spin_tracker.update(work_p, G, t_now);
                }
                if (triple_tracker.initialized) {
                    triple_tracker.update(work_p, G, t_now);
                }
                if (g_collisional.enabled && !g_subsystem_mgr.active.empty()) {
                    print_subsystem_diagnostics(work_p, G);
                }
#endif
            }
        };

        if (integrator == "fsi") {
            // All particles at the fastest occupied level (FROST sec. 6.2), not the floor of the
            // level table (levels.back() is DT_MIN: ~1e11 steps for a 1000 yr run)
            double dt_min_particle = 1e300;
            {
                auto h_dt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), work_p.dt_adapt);
                for (int i = 0; i < work_p.N; i++) dt_min_particle = std::min(dt_min_particle, h_dt(i));
            }
            double dt_global = levels.back();
            for (double lv : levels) {
                if (lv <= dt_min_particle) { dt_global = lv; break; }
            }
#ifdef USE_HDF5
            if (resuming) dt_global = rs.dt_global;   // fixed at the start of the run
            rs.dt_global = dt_global;
#endif
            if (my_rank == 0) std::printf("FSI global step: %.4e Myr\n", dt_global * TIME_UNIT_MYR);
            while (t < t_final) {
                auto step_t0 = std::chrono::high_resolution_clock::now();
                tidal_kick(work_p, 0.5 * dt_global, G, g_energy_books);
                // In replicated mode, work_p contains all particles on all ranks
                fsi_step(work_p, 0, work_p.N, G, soft2, dt_global);
                tidal_drift_reference(dt_global);
                tidal_kick(work_p, 0.5 * dt_global, G, g_energy_books);
#ifdef ENABLE_COLLISIONAL
                // Placeholders of bodies merged inside subsystem integrations
                if (work_p.collisional && !g_subsystem_mgr.ghost_pids.empty() &&
                    compact_zero_mass_range(work_p, 0, work_p.N) > 0) {
                    if (use_distributed) dp.sync_count();
                    g_subsystem_mgr.rebuild_pid_to_idx(work_p);
                    g_subsystem_mgr.update_particle_flags(work_p);
                }
#endif
                t += dt_global;
                step++;
                tidal_print(step);
                auto step_t1 = std::chrono::high_resolution_clock::now();
                double step_wall = std::chrono::duration<double>(step_t1 - step_t0).count();
                g_profiler.step_wall_total += step_wall;
                g_profiler.step_count++;
                if (step % print_interval == 0) {
                    compute_acceleration_and_timestep(work_p, G, soft2, eta);
                    assign_levels(work_p, levels);
                    work_p.partition_by_level();
                    print_progress(t, step);
                    if (my_rank == 0 && profiling_enabled) {
                        g_profiler.print_report(N_display, n_ranks);
                    }
                    g_profiler.reset();
                    g_subset_hist.reset();
                }
                cluster_diag_now(t, step);
#ifdef USE_HDF5
                after_step_io(t, step, step_wall);
                if (stop_run) break;
#endif
            }
        } else {
            // HHS-FSI (per paper's Algorithm 4)
            // Note: the post-step force+timestep recomputation is only needed for
            // diagnostics. HHS recomputes adaptive timesteps at the start of each step from
            // the same state, so skipping it for non-diagnostic steps saves a
            // full O(N^2) pass per step.
#ifdef ENABLE_COLLISIONAL
            Checkpoint checkpoint;
            // KE + PE of the state accepted at the end of the last step: the next step's E_step_start (AUDIT G row 3).
            // Recomputed when the particles changed in between (SSE here; escaper removal changes N)
            double E_carry = 0;
            int E_carry_N = -1;
#endif
            while (t < t_final) {
                NBODY_NVTX_PUSH("STEP");
                fence_stats::report(step, false);   // reset: count only this step
                auto step_t0 = std::chrono::high_resolution_clock::now();
#ifdef ENABLE_COLLISIONAL
                g_sim_time = t;
#endif
                tidal_kick(work_p, 0.5 * dt_max, G, g_energy_books);

#ifdef ENABLE_COLLISIONAL
                // Runtime stellar evolution before each step
                if (g_collisional.enabled && g_collisional.sse_enabled && g_sse_ctx) {
                    E_carry_N = -1;
                    double dt_sse_myr = dt_max * TIME_UNIT_MYR;
                    advance_stellar_evolution(work_p, *g_sse_ctx, dt_sse_myr,
                                              g_energy_books, G);
                    // Massless remnants (kw = 15) leave now: inside the step, compaction must only ever
                    // remove particles from the current HHS range
                    if (work_p.compact_remove_zero_mass() > 0) {
                        if (use_distributed) dp.sync_count();
                        for (int pid : work_p.removed_pid_log) { g_subsystem_mgr.removed_pids.insert(pid); g_subsystem_mgr.ghost_pids.erase(pid); }
                        g_subsystem_mgr.rebuild_pid_to_idx(work_p);
                    }
                }

                // ============================================================
                // Adaptive restart loop (BIFROST §3.19)
                // If energy error is too large, restore checkpoint, reduce all
                // eta parameters by 50%, increase r_ngb by 15%, and retry.
                // Escape valves: max retries exhausted, or error not decreasing.
                // ============================================================
                if (g_collisional.enabled && g_collisional.eta_restart > 0.0) {
                    // Per-step reference energy (BIFROST §3.19: "beginning of the interval")
                    // BIFROST Eq.: |ΔE_s/E| = |E(t) - E(t-τ)| / |E(t₀)|
                    //   numerator: per-step change (E_after - E_step_start)
                    //   denominator: initial energy E(t₀) = diag0.E_total
                    g_profiler.start();
                    double E_step_start = E_carry_N == work_p.N ? E_carry + g_energy_books.total_sink()
                                                                : compute_total_energy(work_p, g_energy_books, G, soft2);
                    g_profiler.stop(StepProfiler::ENERGY_CHECK);

                    // Save originals ONCE before the loop
                    double eta_ff_orig   = g_collisional.eta_ff;
                    double eta_fb_orig   = g_collisional.eta_fb;
                    double eta_gbs_orig  = g_collisional.eta_gbs;
                    double eta_grad_orig = g_collisional.eta_grad;
                    double eta_jerk_orig = g_collisional.eta_jerk;
                    double eta_va_orig   = g_collisional.eta_va;
                    double eta_pert_orig = g_collisional.eta_pert;
                    double r_ngb_orig    = g_collisional.r_ngb;

                    constexpr int max_restarts = 10;
                    double prev_dE = 1e30;
                    Checkpoint best;  // post-step state of the attempt with the smallest error
                    double best_dE = 1e30;
                    double best_Epart = 0;

                    for (int restart = 0; restart <= max_restarts; restart++) {
                        // Save checkpoint
                        { NvtxScope nv_("CKPT"); checkpoint.save(work_p, g_subsystem_mgr, g_energy_books); }

                        // Run HHS step
                        { NvtxScope nv_("HHS");
                        if (use_distributed) {
                            hhs_fsi_step_distributed(dp, dt_max, G, soft2, dt_min, eta, 2);
                        } else {
                            hhs_fsi_step_dynamic(p, 0, p.N, dt_max, G, soft2, dt_min, eta);
                        }
                        }

                        // Per-step error: |E(t) - E(t-τ)| / |E(t₀)|  (BIFROST §3.19)
                        g_profiler.start();
                        double E_part = compute_particle_energy(work_p, G, soft2);
                        double E_after = E_part + g_energy_books.total_sink();
                        g_profiler.stop(StepProfiler::ENERGY_CHECK);
                        double dE = std::fabs((E_after - E_step_start) / (std::fabs(diag0.E_total) + 1e-30));

                        if (dE < g_collisional.eta_restart) {
                            // Accepted
                            E_carry = E_part;
                            E_carry_N = work_p.N;
                            break;
                        }

                        bool give_up = restart == max_restarts || (restart > 0 && dE >= prev_dE);
                        if (give_up) {
                            // Max retries exhausted, or the error grew again (BIFROST: accept the
                            // attempt with the smallest error rather than the last one)
                            if (dE > best_dE) {
                                best.restore(work_p, g_subsystem_mgr, g_energy_books);
                                if (use_distributed) dp.sync_count();
                            }
                            std::printf("RESTART: %s, accepting |dE/E|=%.2e\n",
                                        restart == max_restarts ? "max retries reached" : "dE not decreasing",
                                        std::min(dE, best_dE));
                            E_carry = dE > best_dE ? best_Epart : E_part;
                            E_carry_N = work_p.N;
                            break;
                        }

                        prev_dE = dE;
                        if (dE < best_dE) {
                            best_dE = dE;
                            best_Epart = E_part;
                            best.save(work_p, g_subsystem_mgr, g_energy_books);
                        }

                        // Restore checkpoint
                        checkpoint.restore(work_p, g_subsystem_mgr, g_energy_books);
                        if (use_distributed) dp.sync_count();

                        // Reduce ALL eta parameters by 50% (BIFROST §3.19)
                        g_collisional.eta_ff   *= 0.5;
                        g_collisional.eta_fb   *= 0.5;
                        g_collisional.eta_gbs  *= 0.5;
                        g_collisional.eta_grad *= 0.5;
                        g_collisional.eta_jerk *= 0.5;
                        g_collisional.eta_va   *= 0.5;
                        g_collisional.eta_pert *= 0.5;

                        // Increase r_ngb by 15%
                        g_collisional.r_ngb *= 1.15;
                        g_collisional.r_ngb_sq = g_collisional.r_ngb * g_collisional.r_ngb;

                        std::printf("RESTART[%d]: |dE/E|=%.2e, reducing etas by 50%%, r_ngb=%.4e\n",
                                    restart + 1, dE, g_collisional.r_ngb);
                    }

                    // Restore original parameters after step (BIFROST: "original parameters are restored")
                    g_collisional.eta_ff   = eta_ff_orig;
                    g_collisional.eta_fb   = eta_fb_orig;
                    g_collisional.eta_gbs  = eta_gbs_orig;
                    g_collisional.eta_grad = eta_grad_orig;
                    g_collisional.eta_jerk = eta_jerk_orig;
                    g_collisional.eta_va   = eta_va_orig;
                    g_collisional.eta_pert = eta_pert_orig;
                    g_collisional.r_ngb    = r_ngb_orig;
                    g_collisional.r_ngb_sq = r_ngb_orig * r_ngb_orig;
                } else {
#endif
                    // Non-collisional or eta_restart disabled: single step, no restart
                    if (use_distributed) {
                        hhs_fsi_step_distributed(dp, dt_max, G, soft2, dt_min, eta, 2);
                    } else {
                        hhs_fsi_step_dynamic(p, 0, p.N, dt_max, G, soft2, dt_min, eta);
                    }
#ifdef ENABLE_COLLISIONAL
                }
#endif
                tidal_drift_reference(dt_max);
                tidal_kick(work_p, 0.5 * dt_max, G, g_energy_books);

                t += dt_max;
                step++;
                tidal_print(step);

                // Replicated MPI state must stay bitwise identical; stop instead of diverging silently
                if (use_distributed && replica_check_interval > 0 && step % replica_check_interval == 0 &&
                    !dp.replicas_agree(step)) {
                    exit_code = 3;
                    break;
                }

                auto step_t1 = std::chrono::high_resolution_clock::now();
                double step_wall = std::chrono::duration<double>(step_t1 - step_t0).count();
                g_profiler.step_wall_total += step_wall;
                g_profiler.step_count++;

                fence_stats::report(step, my_rank == 0);
                if (my_rank == 0 && step <= 5) {
                    std::printf("Step %d: %.3fs, dt=%.2e Myr\n",
                        step, step_wall, dt_max*TIME_UNIT_MYR);
                    std::fflush(stdout);
                }

                NBODY_NVTX_PUSH("DIAG");
                if (step % print_interval == 0) {
                    // Recompute forces + timestep only for diagnostics
                    if (use_distributed) {
                        compute_acceleration_and_timestep_distributed(dp, G, soft2, eta);
                    } else {
                        compute_acceleration_and_timestep(work_p, G, soft2, eta);
                    }
                    assign_levels(work_p, levels);
                    work_p.partition_by_level();
                    print_progress(t, step);
#ifdef ENABLE_COLLISIONAL
                    // Periodic escaper removal
                    if (g_collisional.enabled && is_cluster && g_collisional.r_escape_rh > 0) {
                        int n_esc = remove_escapers(work_p, G, g_collisional.r_escape_rh,
                                                     g_subsystem_mgr, g_energy_books);
                        if (n_esc > 0) {
                            g_subsystem_mgr.step_escapers += n_esc;
                            work_p.compact_remove_zero_mass();
                            if (use_distributed) dp.sync_count();
                            for (int pid : work_p.removed_pid_log) { g_subsystem_mgr.removed_pids.insert(pid); g_subsystem_mgr.ghost_pids.erase(pid); }
                        }
                    }
                    // Write event logs at snapshot intervals
                    if (my_rank == 0) {
                        g_merger_tree.write_ascii("merger_tree.dat");
                        g_escaper_log.write_ascii("escaper_log.dat");
                    }
#endif
                    if (my_rank == 0 && profiling_enabled) {
                        g_profiler.print_report(N_display, n_ranks);
                    }
                    g_profiler.reset();
                    g_subset_hist.reset();
                }
                NBODY_NVTX_POP();  // DIAG
                NBODY_NVTX_POP();  // STEP
                cluster_diag_now(t, step);
#ifdef USE_HDF5
                after_step_io(t, step, step_wall);
                if (stop_run) break;
#endif
            }
        }
#ifdef USE_HDF5
        // End of the run: a restart file (continue with a larger TOTAL_TIME_MYR) and the last snapshot, of the
        // state the loop ended with (before the final diagnostics re-sort the particles)
        // (no extra snapshot after a wall-clock stop: a chain of jobs writes the snapshots of one long run)
        if (exit_code == 0) {
            if (snapshot_interval > 0 && !stop_run) write_snapshot_now(t, step);
            if (restart_output) write_restart_now(t, step);
        }
#endif

        auto wall_end = std::chrono::high_resolution_clock::now();
        double wall_sec = std::chrono::duration<double>(wall_end - wall_start).count();

        // Final diagnostics: recompute forces + timestep if not already done
        if (step % print_interval != 0) {
            if (use_distributed) {
                compute_acceleration_and_timestep_distributed(dp, G, soft2, eta);
            } else {
                compute_acceleration_and_timestep(work_p, G, soft2, eta);
            }
            assign_levels(work_p, levels);
            work_p.partition_by_level();
        }
        print_progress(t, step);
        if (my_rank == 0 && profiling_enabled && g_profiler.step_count > 0) {
            g_profiler.print_report(N_display, n_ranks);
        }
        g_profiler.reset();
        g_subset_hist.reset();

        auto diag_final = compute_diagnostics(work_p, G, soft2);
        if (my_rank == 0) {
            print_final(diag_final, diag0, work_p, levels, is_cluster, wall_sec, step, names);
#ifdef ENABLE_COLLISIONAL
            g_merger_tree.write_ascii("merger_tree.dat");
            g_merger_tree.print_summary();
            g_escaper_log.write_ascii("escaper_log.dat");
#endif
        }

        if (use_distributed) {
            dp.local.deallocate();
        } else {
            p.deallocate();
        }
#ifdef ENABLE_COLLISIONAL
        delete g_sse_ctx;
        // Global Views must be released before Kokkos::finalize
        g_logh_batch = SubsystemBatch();
        g_subsystem_mgr = SubsystemManager();
#endif
    }
    release_integrator_globals();
    Kokkos::finalize();
    MPIComm::finalize();
    return exit_code;
}
