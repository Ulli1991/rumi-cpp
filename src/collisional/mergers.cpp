#include "mergers.h"
#include "stellar_cache.h"
#include "merger_tree.h"
#include "kepler.h"
#include "constants.h"
#include "collisional_config.h"
#include "gw_recoil.h"
#include "gw_output.h"
#include "tidal_field.h"
#include <cstdint>
#include <cstring>
#include "integrator.h"
#include "small_for.h"
#include "device_staging.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>

// ============================================================================
// Peters (1964) GW coalescence timescale (Peters Eq. 5.14)
// ============================================================================

double gw_coalescence_time(double a, double e, double m1, double m2,
                           double G, double c_code) {
    double c5 = std::pow(c_code, 5);
    double beta = 64.0/5.0 * G*G*G * m1 * m2 * (m1 + m2) / c5;

    if (e < 0.01) {
        // Circular approximation
        return a * a * a * a / (4.0 * beta);
    }

    // Full Peters integral approximation (Peters 1964 Eq. 5.14)
    double e04 = std::pow(e, 48.0/19.0);
    double c0 = a * (1.0 - e*e) / std::pow(e, 12.0/19.0) *
                std::pow(1.0 + 121.0/304.0 * e*e, -870.0/2299.0);
    // Approximate T_merge ~ (12/19) * c0^4 / (4*beta) * integral
    // Using the fit from Peters & Mathews:
    return a * a * a * a / (4.0 * beta) * std::pow(1.0 - e*e, 3.5) /
           (1.0 + 73.0/24.0 * e*e + 37.0/96.0 * e*e*e*e);
}

// ============================================================================
// Tidal disruption radius
// ============================================================================

double tidal_disruption_radius(double m_star, double R_star, double M_bh) {
    // r_t = 1.3 (M_bh / m_star)^{1/3} R_star: FROST-CLUSTERS I (Rantala et al. 2024) App. B3.3 and FROST-CLUSTERS
    // III sec. 2.1.4, the cube-root scaling of Kochanek (1992). (Was 1.3 ((m + M) / m)^{1/3} R, Rantala et al. 2023.)
    // R_star in the same units as positions (pc)
    return 1.3 * std::cbrt(M_bh / m_star) * R_star;
}

double collision_distance(int kw1, double m1, double R1, int kw2, double m2, double R2, double G, double c,
                          MergerType& type) {
    const bool bh1 = (kw1 == 14), bh2 = (kw2 == 14);
    const double r_isco = c > 0.0 ? 6.0 * G * (m1 + m2) / (c * c) : 0.0;
    // BH pairs (default): 10 Schwarzschild radii of the total mass, 20 G (m1 + m2) / c^2, the conservative
    // option of Rantala et al. 2023, sec. 3.14.1 (KETJU-GADGET4 12 R_S, Mannerkoski+2019, Rantala+2025 ~10 R_S): truncated PN fails inside,
    // 3.5PN turns radiation reaction into an energy gain at pericentre x = G M / (r c^2) ~ 0.11-0.14 (CASES.md C8).
    // BH_MERGE_RADIUS = max_single: FROST-CLUSTERS I sec. 2.6.3, r_sep < max(r_ISCO,1, r_ISCO,2), r_ISCO = 6 G m / c^2
    if (bh1 && bh2) {
        type = MergerType::ISCO;
        if (!(c > 0.0)) return 0.0;
        return g_collisional.bh_merge_max_single ? 6.0 * G * std::max(m1, m2) / (c * c) : 20.0 * G * (m1 + m2) / (c * c);
    }
    if (bh1 || bh2) {
        const double m_s = bh1 ? m2 : m1, R_s = bh1 ? R2 : R1, M_bh = bh1 ? m1 : m2;
        const double r_t = R_s > 0.0 && m_s > 0.0 ? std::max(tidal_disruption_radius(m_s, R_s, M_bh), R_s) : 0.0;
        if (r_isco >= r_t) { type = MergerType::ISCO; return r_isco; }
        type = MergerType::TIDAL_DISRUPTION;
        return r_t;
    }
    type = MergerType::STELLAR_COLLISION;
    return R1 + R2;
}

double collision_remnant_mass(MergerType type, int kw1, double m1, int kw2, double m2) {
    // TDE: the BH keeps TDE_ACCRETION_FRACTION of the star (FROST-CLUSTERS I App. B; Rizzuto+2021 0.1-1, DRAGON-II and
    // FROST-CLUSTERS 0.5); the rest leaves the cluster, booked as mass loss by the callers
    if (type == MergerType::TIDAL_DISRUPTION)
        return kw1 == 14 ? m1 + g_collisional.tde_accretion * m2 : m2 + g_collisional.tde_accretion * m1;
    // Stellar collisions: Glebbeek et al. 2013 mass loss f = C q / (1 + q)^2 of m1 + m2, q = m_small / m_large
    // (FROST-CLUSTERS III sec. 2.2.3: C = 0.3, 7.5 % for equal masses); COLL_MASS_LOSS_C = 0 (default): none
    if (type == MergerType::STELLAR_COLLISION && g_collisional.coll_mass_loss_c > 0.0) {
        const double q = std::min(m1, m2) / std::max(m1, m2);
        return (m1 + m2) * (1.0 - g_collisional.coll_mass_loss_c * q / ((1.0 + q) * (1.0 + q)));
    }
    (void)kw1; (void)kw2;
    return m1 + m2;
}

// ============================================================================
// BH-BH merger remnant: quasi-circular fits, scaled with the orbital angular momentum at the merger
// ============================================================================
// j = h / h_qc, h = |dx x dv| of the relative orbit at the decision, h_qc = chi_orb G M / (c eta) the orbital
// angular momentum per reduced mass that gives the quasi-circular remnant spin chi_orb (J conservation from the
// plunge, Buonanno, Kidder & Lehner 2008, PRD 77, 026004). j >= 1 (any inspiral reaching 10 R_S; a direct plunge
// needs h below ~2.7 G M / c): the quasi-circular fits. j -> 0: the head-on limit, E_rad / M = eta^2 (0.0110 -
// 0.0088 eta) (Sperhake et al. 2011, PRD 84, 084038, Sec. IV, NR from rest), no orbital spin, no kick (head-on
// kicks are below 5 km/s, ibid.; KETJU: none). In between E_rad and the kick go with j^2, the orbital spin with j.

void bh_bh_remnant(double m1, double m2, const double chi1[3], const double chi2[3],
                   const double L_orb[3], const double dx[3], const double dv[3], double& m_rem, double chi_rem[3],
                   double v_kick[3], double G, double c_code) {
    double M = m1 + m2;
    double eta = m1 * m2 / (M * M);
    double delta_m = std::fabs(m1 - m2) / M;

    // Quasi-circular radiated energy (Barausse, Morozova & Rezzolla 2012, ApJ 758, 63, simplified)
    const double E_qc = 0.0547 * eta * (1.0 - 0.5 * delta_m * delta_m);
    const double E_ho = eta * eta * (0.0110 - 0.0088 * eta);
    // Quasi-circular orbital spin (Barausse & Rezzolla 2009, non-spinning part)
    const double chi_orb = eta * (3.464 - 3.162*eta - 2.484*eta*eta);

    const double h[3] = {dx[1]*dv[2] - dx[2]*dv[1], dx[2]*dv[0] - dx[0]*dv[2], dx[0]*dv[1] - dx[1]*dv[0]};
    const double h_mag = std::sqrt(h[0]*h[0] + h[1]*h[1] + h[2]*h[2]);
    const double h_qc = chi_orb * G * M / (c_code * eta);
    const double j = h_qc > 0.0 ? std::min(1.0, h_mag / h_qc) : 1.0;
    const double w = j * j;

    m_rem = M * (1.0 - (E_ho + (E_qc - E_ho) * w));

    // Remnant spin: orbital part j chi_orb along L plus the BH spins, (chi1 m1^2 + chi2 m2^2) / M^2 (Barausse &
    // Rezzolla 2009); the BH spins are carried over unchanged in a head-on merger (KETJU)
    double L_hat[3] = {0.0, 0.0, 0.0};
    const double L_mag = std::sqrt(L_orb[0]*L_orb[0] + L_orb[1]*L_orb[1] + L_orb[2]*L_orb[2]);
    if (L_mag > 0.0) for (int k = 0; k < 3; k++) L_hat[k] = L_orb[k] / L_mag;
    for (int k = 0; k < 3; k++)
        chi_rem[k] = j * chi_orb * L_hat[k] + (chi1[k] * m1 * m1 + chi2[k] * m2 * m2) / (M * M);
    const double chi_mag = std::sqrt(chi_rem[0]*chi_rem[0] + chi_rem[1]*chi_rem[1] + chi_rem[2]*chi_rem[2]);
    if (chi_mag > 0.998) for (int k = 0; k < 3; k++) chi_rem[k] *= 0.998 / chi_mag;   // Kerr limit

    // GW kick with spins (gw_recoil.h; Lousto et al. 2012). The phase angle and the in-plane orientation, unknown
    // from the orbit-averaged state, come from a hash of the masses, spins and L: deterministic on every MPI rank
    // and across restarts. (Before 2026-10-12: spins ignored, and the mass term used 1 + B delta_m instead of
    // 1 + B eta: 138 instead of 157 km/s at q = 0.5)
    std::uint64_t hsh = 1469598103934665603ULL;
    auto mix = [&](double x) { std::uint64_t u; std::memcpy(&u, &x, 8); hsh = (hsh ^ u) * 1099511628211ULL; };
    mix(m1); mix(m2);
    for (int k = 0; k < 3; k++) { mix(chi1[k]); mix(chi2[k]); mix(L_orb[k]); }
    const double u1 = (double)(hsh >> 11) * 0x1.0p-53;
    hsh = (hsh ^ 0x9e3779b97f4a7c15ULL) * 1099511628211ULL;
    const double u2 = (double)(hsh >> 11) * 0x1.0p-53;
    gw_recoil_kick(m1, m2, chi1, chi2, L_orb, u1, u2, v_kick);
    for (int k = 0; k < 3; k++) v_kick[k] *= w;
}

// ============================================================================
// Detect mergers in active subsystems
// ============================================================================

std::vector<MergerEvent> detect_mergers(const SubsystemManager& mgr,
                                         const Particles& p,
                                         double G, double c_code, double dt) {
    std::vector<MergerEvent> events;
    if (mgr.active.empty()) return events;

    const auto& pid_to_idx = mgr.pid_to_idx();

    // Gather x, y, z, mass, stellar type and radius of the binaries' members only (rows 2j, 2j + 1 for the j-th
    // binary): one small transfer through persistent buffers instead of nine copies of the members' index range
    // Also the inner binary of a secular (stable hierarchical, >= N_ORB_SECULAR inner orbits per step) triple, by the
    // GW rule only and only when LogH could not follow the inspiral (C24; sub.orb holds the inner elements): its Peters
    // time is the isolated pair's, as for a binary. Before, a BH inner binary with t_GW << dt went to the secular
    // triple step (stepped past a = 0: NaN members) or to LogH, which stopped at its step ceiling ~1e5 orbits short of
    // the contact distance. A shorter inspiral is refused by the secular triple step and decided by LogH's pericentre
    // landing (FC-I App. B3), as before (coll_bh_isco, bh_merge_radius_*)
    std::vector<const Subsystem*> bins;
    std::vector<int> slots;
    std::vector<int> pair_pids;   // the two members per entry of bins
    for (const auto& sub : mgr.active) {
        int pid1, pid2;
        if (sub.n_bodies == 2) {
            pid1 = sub.particle_ids[0]; pid2 = sub.particle_ids[1];
        } else if (sub.n_bodies == 3 && sub.type == SubsystemType::TRIPLE_STABLE && sub.is_secular &&
                   sub.inner_binary_id >= 0 && sub.outer_companion_pid >= 0) {
            pid1 = sub.inner_binary_id; pid2 = -1;
            for (int pid : sub.particle_ids)
                if (pid != sub.inner_binary_id && pid != sub.outer_companion_pid) pid2 = pid;
            if (pid2 < 0) continue;
        } else {
            continue;
        }
        int idx1 = pid_to_idx[pid1];
        int idx2 = pid_to_idx[pid2];
        if (idx1 < 0 || idx2 < 0) continue;
        bins.push_back(&sub);
        slots.push_back(idx1);
        slots.push_back(idx2);
        pair_pids.push_back(pid1);
        pair_pids.push_back(pid2);
    }
    if (bins.empty()) return events;
    constexpr int NF = 6;
    const int n_rows = (int)slots.size();
    DeviceStaging& st = DeviceStaging::get<2>();
    st.reserve(n_rows, NF);
    std::copy(slots.begin(), slots.end(), st.h_idx.data());
    st.upload_idx(n_rows);
    {
        auto d_idx = raw(st.d_idx);
        auto d_buf = raw(st.d_buf);
        auto px = raw(p.x), py = raw(p.y), pz = raw(p.z), pm = raw(p.mass), psr = raw(p.stellar_radius);
        auto pst = raw(p.stellar_type);
        small_for("merger_gather", n_rows, KOKKOS_LAMBDA(int k) {
            int i = d_idx(k);
            d_buf(k * NF + 0) = px(i); d_buf(k * NF + 1) = py(i); d_buf(k * NF + 2) = pz(i);
            d_buf(k * NF + 3) = pm(i); d_buf(k * NF + 4) = (double)pst(i); d_buf(k * NF + 5) = psr(i);
        });
    }
    st.download_buf((size_t)n_rows * NF);
    auto h_buf = st.h_buf;
    auto h_x = [&](int k) { return h_buf(k * NF + 0); };
    auto h_y = [&](int k) { return h_buf(k * NF + 1); };
    auto h_z = [&](int k) { return h_buf(k * NF + 2); };
    auto h_m = [&](int k) { return h_buf(k * NF + 3); };
    auto h_stype = [&](int k) { return (int)h_buf(k * NF + 4); };
    auto h_srad = [&](int k) { return h_buf(k * NF + 5); };

    double R_sun_to_pc = 2.254e-8;  // R_sun in parsecs

    for (size_t j = 0; j < bins.size(); j++) {
        const auto& sub = *bins[j];
        const int r1 = 2 * (int)j, r2 = r1 + 1;   // rows of the two members

        double dx = h_x(r2) - h_x(r1);
        double dy = h_y(r2) - h_y(r1);
        double dz = h_z(r2) - h_z(r1);
        double r = std::sqrt(dx*dx + dy*dy + dz*dz);

        double m1 = h_m(r1), m2 = h_m(r2);
        int stype1 = h_stype(r1), stype2 = h_stype(r2);
        // A massless member is the placeholder of a body merged inside a subsystem integration
        // (finalize_subsystem_collision): already merged, removed with the next compaction of its range
        if (m1 <= 0.0 || m2 <= 0.0) continue;

        MergerEvent ev{};
        ev.type = MergerType::NONE;
        ev.pid1 = pair_pids[r1];
        ev.pid2 = pair_pids[r2];
        ev.m1 = m1;
        ev.m2 = m2;

        // Contact: ISCO (two BHs, or a BH and a body it swallows whole), tidal disruption, stellar collision
        // (FROST-CLUSTERS I App. B3 criteria, shared with the subsystem integrations: collision_distance)
        MergerType ctype;
        const double r_coll = collision_distance(stype1, m1, h_srad(r1) * R_sun_to_pc, stype2, m2,
                                                 h_srad(r2) * R_sun_to_pc, G, c_code, ctype);
        if (r < r_coll && sub.n_bodies == 2) {
            ev.type = ctype;
            ev.m_remnant = collision_remnant_mass(ctype, stype1, m1, stype2, m2);
        }

        // GW coalescence timescale: compact remnants only (WD/NS/BH, kw 10-14), and only with radiation reaction
        // (PN_ORDER 2.5PN bit); stars merge on contact above
        const bool compact_pair = stype1 >= 10 && stype1 <= 14 && stype2 >= 10 && stype2 <= 14;
        const bool gw_on = (g_collisional.pn_order & 4) && c_code > 0.0;
        if (ev.type == MergerType::NONE && sub.orb.a > 0 && compact_pair && gw_on) {
            double t_gw = gw_coalescence_time(sub.orb.a, sub.orb.e, m1, m2, G, c_code);
            // Triple: orbits to coalescence against LogH's per-call ceiling of 4e6 GBS steps at 64 per orbit
            // (logh_batch.cpp gbs_step_limit)
            const double P_in = 2.0 * M_PI * std::sqrt(sub.orb.a * sub.orb.a * sub.orb.a / (G * (m1 + m2)));
            const bool logh_cannot = sub.n_bodies == 2 || t_gw > (4.0e6 / 64.0) * P_in;
            if (t_gw < dt && logh_cannot) {
                ev.type = MergerType::GW_COALESCENCE;
                ev.m_remnant = m1 + m2;
            }
        }

        if (ev.type != MergerType::NONE) {
            events.push_back(ev);
        }
    }

    return events;
}

// ============================================================================
// Execute merger
// ============================================================================

void execute_merger(const MergerEvent& event, Particles& p,
                    SubsystemManager& mgr, EnergyBookkeeping& books,
                    double G, double c_code) {
    auto h_x  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
    auto h_y  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
    auto h_z  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
    auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
    auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
    auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
    auto h_m  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
    auto h_stype = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_type);
    auto h_spin = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.spin_vec);

    const auto& pid_to_idx = g_subsystem_mgr.pid_to_idx();

    int idx1 = pid_to_idx[event.pid1];
    int idx2 = pid_to_idx[event.pid2];
    if (idx1 < 0 || idx2 < 0) return;

    double m1 = h_m(idx1), m2 = h_m(idx2);
    double M = m1 + m2;

    // CoM position and velocity
    double cx = (m1*h_x(idx1) + m2*h_x(idx2)) / M;
    double cy = (m1*h_y(idx1) + m2*h_y(idx2)) / M;
    double cz = (m1*h_z(idx1) + m2*h_z(idx2)) / M;
    double cvx = (m1*h_vx(idx1) + m2*h_vx(idx2)) / M;
    double cvy = (m1*h_vy(idx1) + m2*h_vy(idx2)) / M;
    double cvz = (m1*h_vz(idx1) + m2*h_vz(idx2)) / M;

    double m_rem = event.m_remnant;
    double v_kick[3] = {0, 0, 0};
    double chi_rem[3] = {0, 0, 0};

    // BH-BH merger: use fitting formulae for remnant mass, spin, kick
    bool is_bh1 = (h_stype(idx1) == 14), is_bh2 = (h_stype(idx2) == 14);
    if (is_bh1 && is_bh2 && c_code > 0.0) {
        // Orbital angular momentum direction
        double dx = h_x(idx2) - h_x(idx1), dy = h_y(idx2) - h_y(idx1), dz = h_z(idx2) - h_z(idx1);
        double dvx = h_vx(idx2) - h_vx(idx1), dvy = h_vy(idx2) - h_vy(idx1), dvz = h_vz(idx2) - h_vz(idx1);
        double L_orb[3] = {dy*dvz - dz*dvy, dz*dvx - dx*dvz, dx*dvy - dy*dvx};

        double chi1[3] = {h_spin(idx1, 0), h_spin(idx1, 1), h_spin(idx1, 2)};
        double chi2[3] = {h_spin(idx2, 0), h_spin(idx2, 1), h_spin(idx2, 2)};

        const double dxr[3] = {dx, dy, dz}, dvr[3] = {dvx, dvy, dvz};
        bh_bh_remnant(m1, m2, chi1, chi2, L_orb, dxr, dvr, m_rem, chi_rem, v_kick, G, c_code);
    }

    // With integer positions the remnant sits on the grid; use that position in the books too
    if (use_integer_positions) {
        cx = (double)llrint(cx * int_pos_inv_scale) * int_pos_scale;
        cy = (double)llrint(cy * int_pos_inv_scale) * int_pos_scale;
        cz = (double)llrint(cz * int_pos_inv_scale) * int_pos_scale;
    }

    // Energy sink = E(before) - E(after) of everything the merger changes, so E_total + sinks
    // stays constant (Rantala et al. 2023, §3.16): the pair's kinetic and mutual potential energy, the
    // remnant's kinetic energy (with kick), and the potential with all other particles, where
    // m1 at x1 and m2 at x2 become m_rem at the CoM. Collisional runs are unsoftened.
    auto dist = [&](int k, double px, double py, double pz) {
        double ex = h_x(k) - px, ey = h_y(k) - py, ez = h_z(k) - pz;
        return std::sqrt(ex*ex + ey*ey + ez*ez);
    };
    double r = dist(idx2, h_x(idx1), h_y(idx1), h_z(idx1));
    double v1sq = h_vx(idx1)*h_vx(idx1) + h_vy(idx1)*h_vy(idx1) + h_vz(idx1)*h_vz(idx1);
    double v2sq = h_vx(idx2)*h_vx(idx2) + h_vy(idx2)*h_vy(idx2) + h_vz(idx2)*h_vz(idx2);
    double ux = cvx + v_kick[0], uy = cvy + v_kick[1], uz = cvz + v_kick[2];
    double E_before = 0.5 * (m1 * v1sq + m2 * v2sq) - G * m1 * m2 / (r > 0 ? r : 1e-30);
    double E_after = 0.5 * m_rem * (ux*ux + uy*uy + uz*uz);
    for (int k = 0; k < p.N; k++) {
        if (k == idx1 || k == idx2 || h_m(k) == 0.0) continue;
        double r1 = dist(k, h_x(idx1), h_y(idx1), h_z(idx1));
        double r2 = dist(k, h_x(idx2), h_y(idx2), h_z(idx2));
        double rc = dist(k, cx, cy, cz);
        E_after -= G * h_m(k) * (m_rem / rc - m1 / r1 - m2 / r2);
    }
    books.E_merge += E_before - E_after;
    {
        double r1[3] = {h_x(idx1), h_y(idx1), h_z(idx1)}, v1[3] = {h_vx(idx1), h_vy(idx1), h_vz(idx1)};
        double r2[3] = {h_x(idx2), h_y(idx2), h_z(idx2)}, v2[3] = {h_vx(idx2), h_vy(idx2), h_vz(idx2)};
        double rc[3] = {cx, cy, cz}, u[3] = {ux, uy, uz};
        books.sink_PL(m1, r1, v1);
        books.sink_PL(m2, r2, v2);
        books.sink_PL(m_rem, rc, u, -1.0);
    }

    // Merger tree, with the remnant made here (before 2026-10-21 the caller recorded the event before this call:
    // remnant mass m1 + m2, no kick, no spin)
    const Subsystem* sub = nullptr;
    for (const auto& s : mgr.active)
        if (std::find(s.particle_ids.begin(), s.particle_ids.end(), event.pid2) != s.particle_ids.end()) sub = &s;
    {
        MergerEvent ev = event;
        ev.m_remnant = m_rem;
        for (int k = 0; k < 3; k++) { ev.v_kick[k] = v_kick[k]; ev.spin_remnant[k] = chi_rem[k]; }
        if (const Subsystem* s2 = find_subsystem_for_merger(mgr, event)) {
            g_merger_tree.record(ev, *s2, g_sim_time);
        } else {   // the tree's a, e from the pair itself
            Subsystem tmp{};
            const double rr[3] = {h_x(idx2) - h_x(idx1), h_y(idx2) - h_y(idx1), h_z(idx2) - h_z(idx1)};
            const double vv[3] = {h_vx(idx2) - h_vx(idx1), h_vy(idx2) - h_vy(idx1), h_vz(idx2) - h_vz(idx1)};
            tmp.orb = cartesian_to_elements(rr, vv, G * M);
            g_merger_tree.record(ev, tmp, g_sim_time);
        }
    }

    // Gravitational-wave table (gw_output.h): a merger of two compact objects, from the state at the decision
    if (g_gw_log.on && gw_compact_pair(h_stype(idx1), h_stype(idx2))) {
        GwRow row;
        row.kind = 0; row.t = g_sim_time; row.type = (int)event.type;
        row.path = event.type == MergerType::GW_COALESCENCE ? (sub && sub->is_secular ? GW_PATH_SECULAR : GW_PATH_PETERS)
                                                            : GW_PATH_CONTACT;
        row.pid1 = event.pid1; row.pid2 = event.pid2; row.kw1 = h_stype(idx1); row.kw2 = h_stype(idx2);
        row.m1 = m1; row.m2 = m2;
        for (int k = 0; k < 3; k++) { row.chi1[k] = h_spin(idx1, k); row.chi2[k] = h_spin(idx2, k); }
        const double rr[3] = {h_x(idx2) - h_x(idx1), h_y(idx2) - h_y(idx1), h_z(idx2) - h_z(idx1)};
        const double vv[3] = {h_vx(idx2) - h_vx(idx1), h_vy(idx2) - h_vy(idx1), h_vz(idx2) - h_vz(idx1)};
        gw_fill_pair(row, rr, vv, G, c_code);
        row.m_rem = m_rem;
        for (int k = 0; k < 3; k++) row.chi_rem[k] = chi_rem[k];
        row.v_kick = std::sqrt(v_kick[0] * v_kick[0] + v_kick[1] * v_kick[1] + v_kick[2] * v_kick[2]);
        row.n_members = sub ? sub->n_bodies : 2;
        const double xc[3] = {(m1 * h_x(idx1) + m2 * h_x(idx2)) / M, (m1 * h_y(idx1) + m2 * h_y(idx2)) / M,
                              (m1 * h_z(idx1) + m2 * h_z(idx2)) / M};
        const double vc[3] = {cvx, cvy, cvz};
        gw_environment(row, p.N, h_x.data(), h_y.data(), h_z.data(), h_vx.data(), h_vy.data(), h_vz.data(), h_m.data(),
                       idx1, idx2, xc, vc, r, G);
        g_gw_log.pending.push_back(row);
    }

    // Place remnant at idx1, mark idx2 for removal
    h_x(idx1) = cx; h_y(idx1) = cy; h_z(idx1) = cz;
    h_vx(idx1) = cvx + v_kick[0];
    h_vy(idx1) = cvy + v_kick[1];
    h_vz(idx1) = cvz + v_kick[2];
    h_m(idx1) = m_rem;
    // Remnant type independent of which progenitor sits in idx1: a compact progenitor (kw >= 10)
    // dominates, the more compact one first (BH 14 > NS 13 > WDs); two stars stay a star
    {
        int k1 = h_stype(idx1), k2 = h_stype(idx2);
        h_stype(idx1) = (k1 >= 10 || k2 >= 10) ? std::max(k1, k2) : k1;
    }
    h_spin(idx1, 0) = chi_rem[0];
    h_spin(idx1, 1) = chi_rem[1];
    h_spin(idx1, 2) = chi_rem[2];

    // Mark particle 2 with zero mass (will be compacted later)
    h_m(idx2) = 0.0;
    mgr.removed_pids.insert(event.pid2);

    Kokkos::deep_copy(p.x, h_x);
    if (use_integer_positions) {   // integer positions are the source of truth for the drift
        auto h_ix = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.ix);
        auto h_iy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.iy);
        auto h_iz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.iz);
        h_ix(idx1) = llrint(cx * int_pos_inv_scale);
        h_iy(idx1) = llrint(cy * int_pos_inv_scale);
        h_iz(idx1) = llrint(cz * int_pos_inv_scale);
        Kokkos::deep_copy(p.ix, h_ix);
        Kokkos::deep_copy(p.iy, h_iy);
        Kokkos::deep_copy(p.iz, h_iz);
    }
    Kokkos::deep_copy(p.y, h_y);
    Kokkos::deep_copy(p.z, h_z);
    Kokkos::deep_copy(p.vx, h_vx);
    Kokkos::deep_copy(p.vy, h_vy);
    Kokkos::deep_copy(p.vz, h_vz);
    p.zero_kick_compensation();   // merger remnant velocity is new
    Kokkos::deep_copy(p.mass, h_m);
    Kokkos::deep_copy(p.stellar_type, h_stype);
    StellarPropsCache::get().invalidate();   // remnant type (and radius below)
    Kokkos::deep_copy(p.spin_vec, h_spin);

    // A star made by a merger must evolve as that star: runtime SSE recomputes the mass from m_zams and the
    // age, so keeping a progenitor's would shrink the remnant back to it. Simplification of BSE (Hurley et
    // al. 2002): the remnant restarts as a zero-age star of the merged mass. Compact remnants (kw >= 10)
    // are not evolved by SSE.
    if (h_stype(idx1) < 10) {
        auto h_mzams = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.m_zams);
        auto h_sage  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_age);
        h_mzams(idx1) = m_rem;
        h_sage(idx1) = 0.0;
        Kokkos::deep_copy(p.m_zams, h_mzams);
        Kokkos::deep_copy(p.stellar_age, h_sage);
    }

    // Remove the merged subsystem
    for (auto it = mgr.active.begin(); it != mgr.active.end(); ) {
        bool contains_pid2 = false;
        for (int pid : it->particle_ids) {
            if (pid == event.pid2) { contains_pid2 = true; break; }
        }
        if (contains_pid2) {
            it = mgr.active.erase(it);
        } else {
            ++it;
        }
    }

}

// ============================================================================
// Collision decided inside a subsystem integration (FROST-CLUSTERS I App. B3)
// ============================================================================

void finalize_subsystem_collision(const SubsystemCollision& c, Particles& p, SubsystemManager& mgr,
                                  EnergyBookkeeping& books, double G, double c_code) {
    auto h_x  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
    auto h_y  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
    auto h_z  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
    auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
    auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
    auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
    auto h_m  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
    auto h_stype = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_type);
    auto h_spin = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.spin_vec);

    const int ik = mgr.safe_pid_to_idx(c.pid_keep), ig = mgr.safe_pid_to_idx(c.pid_gone);
    if (ik < 0 || ig < 0) {
        std::printf("  [SUBSYS COLLISION] WARNING: pids %d / %d not found, event dropped\n", c.pid_keep, c.pid_gone);
        return;
    }
    const double M = c.m_keep + c.m_gone;
    const double xr[3] = {h_x(ik), h_y(ik), h_z(ik)}, vr[3] = {h_vx(ik), h_vy(ik), h_vz(ik)};
    double x1[3], x2[3], v1[3], v2[3];   // the pair at its frozen relative state around the remnant
    for (int k = 0; k < 3; k++) {
        x1[k] = xr[k] + c.d_keep[k]; x2[k] = xr[k] + c.d_gone[k];
        v1[k] = vr[k] + c.u_keep[k]; v2[k] = vr[k] + c.u_gone[k];
    }
    std::vector<int> mem;
    for (int b = 0; b < c.n_members; b++) mem.push_back(mgr.safe_pid_to_idx(c.member_pids[b]));
    auto is_member = [&](int k) { return std::find(mem.begin(), mem.end(), k) != mem.end(); };
    auto dist = [&](int k, const double q[3]) {
        const double ex = h_x(k) - q[0], ey = h_y(k) - q[1], ez = h_z(k) - q[2];
        return std::sqrt(ex*ex + ey*ey + ez*ez);
    };

    // 1. The merge itself (mass and momentum conserved): the subsystem's internal energy change at the collision,
    //    E_after - E_before, from the integrator (exact, with the other members at their positions then)
    books.E_merge -= c.dE_int;
    // 2. Potential with the bodies outside the subsystem: m_keep, m_gone at the frozen offsets become M at the
    //    centre of mass (quadrupole order), evaluated at the remnant's final position
    double dU_ext = 0.0;
    for (int k = 0; k < p.N; k++) {
        if (h_m(k) <= 0.0 || is_member(k)) continue;
        dU_ext -= G * h_m(k) * (M / dist(k, xr) - c.m_keep / dist(k, x1) - c.m_gone / dist(k, x2));
    }
    books.E_merge -= dU_ext;
    // 3. Orbital momentum and angular momentum of the pair (PN subsystems: booked by the caller's P / L books)
    if (!c.pl_booked) {
        books.sink_PL(c.m_keep, x1, v1);
        books.sink_PL(c.m_gone, x2, v2);
        books.sink_PL(M, xr, vr, -1.0);
    }

    // 4. Remnant physics at the remnant's final state, booked as execute_merger books it (all other bodies)
    double m_rem = collision_remnant_mass(c.type, c.kw_keep, c.m_keep, c.kw_gone, c.m_gone);
    double kick[3] = {0, 0, 0}, chi_rem[3] = {0, 0, 0};
    double L_orb[3] = {0, 0, 0};
    {
        const double* d[2] = {c.d_keep, c.d_gone};
        const double* u[2] = {c.u_keep, c.u_gone};
        const double m[2] = {c.m_keep, c.m_gone};
        for (int q = 0; q < 2; q++) {
            L_orb[0] += m[q] * (d[q][1] * u[q][2] - d[q][2] * u[q][1]);
            L_orb[1] += m[q] * (d[q][2] * u[q][0] - d[q][0] * u[q][2]);
            L_orb[2] += m[q] * (d[q][0] * u[q][1] - d[q][1] * u[q][0]);
        }
    }
    if (c.kw_keep == 14 && c.kw_gone == 14 && c_code > 0.0) {
        const double chi1[3] = {h_spin(ik, 0), h_spin(ik, 1), h_spin(ik, 2)};
        const double chi2[3] = {h_spin(ig, 0), h_spin(ig, 1), h_spin(ig, 2)};
        double dxr[3], dvr[3];
        for (int k = 0; k < 3; k++) { dxr[k] = c.d_gone[k] - c.d_keep[k]; dvr[k] = c.u_gone[k] - c.u_keep[k]; }
        bh_bh_remnant(c.m_keep, c.m_gone, chi1, chi2, L_orb, dxr, dvr, m_rem, chi_rem, kick, G, c_code);
    }
    const double u[3] = {vr[0] + kick[0], vr[1] + kick[1], vr[2] + kick[2]};
    double E_before = 0.5 * M * (vr[0]*vr[0] + vr[1]*vr[1] + vr[2]*vr[2]);
    double E_after = 0.5 * m_rem * (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
    if (m_rem != M)
        for (int k = 0; k < p.N; k++) {
            if (k == ik || k == ig || h_m(k) <= 0.0) continue;
            E_after -= G * h_m(k) * (m_rem - M) / dist(k, xr);
        }
    books.E_merge += E_before - E_after;
    books.sink_PL(M, xr, vr);
    books.sink_PL(m_rem, xr, u, -1.0);

    // Gravitational-wave table (gw_output.h): the pair at the collision, the remnant as made here
    if (g_gw_log.on && gw_compact_pair(c.kw_keep, c.kw_gone)) {
        GwRow row;
        row.kind = 0; row.t = g_sim_time; row.type = (int)c.type; row.path = GW_PATH_SUBSYS;
        row.pid1 = c.pid_keep; row.pid2 = c.pid_gone; row.kw1 = c.kw_keep; row.kw2 = c.kw_gone;
        row.m1 = c.m_keep; row.m2 = c.m_gone;
        for (int k = 0; k < 3; k++) { row.chi1[k] = h_spin(ik, k); row.chi2[k] = h_spin(ig, k); }
        double rr[3], vv[3];
        for (int k = 0; k < 3; k++) { rr[k] = c.d_gone[k] - c.d_keep[k]; vv[k] = c.u_gone[k] - c.u_keep[k]; }
        gw_fill_pair(row, rr, vv, G, c_code);
        row.m_rem = m_rem;
        for (int k = 0; k < 3; k++) row.chi_rem[k] = chi_rem[k];
        row.v_kick = std::sqrt(kick[0] * kick[0] + kick[1] * kick[1] + kick[2] * kick[2]);
        row.n_members = c.n_members;
        gw_environment(row, p.N, h_x.data(), h_y.data(), h_z.data(), h_vx.data(), h_vy.data(), h_vz.data(), h_m.data(),
                       ik, ig, xr, vr, std::sqrt(rr[0] * rr[0] + rr[1] * rr[1] + rr[2] * rr[2]), G);
        g_gw_log.pending.push_back(row);
    }

    h_m(ik) = m_rem;
    h_vx(ik) = u[0]; h_vy(ik) = u[1]; h_vz(ik) = u[2];
    h_vx(ig) = u[0]; h_vy(ig) = u[1]; h_vz(ig) = u[2];   // the massless placeholder co-moves
    h_m(ig) = 0.0;
    {
        const int k1 = c.kw_keep, k2 = c.kw_gone;
        h_stype(ik) = (k1 >= 10 || k2 >= 10) ? std::max(k1, k2) : k1;
    }
    for (int k = 0; k < 3; k++) h_spin(ik, k) = chi_rem[k];
    Kokkos::deep_copy(p.vx, h_vx); Kokkos::deep_copy(p.vy, h_vy); Kokkos::deep_copy(p.vz, h_vz);
    p.zero_kick_compensation();
    Kokkos::deep_copy(p.mass, h_m);
    Kokkos::deep_copy(p.stellar_type, h_stype);
    StellarPropsCache::get().invalidate();   // remnant type (and radius below)
    Kokkos::deep_copy(p.spin_vec, h_spin);
    // The remnant's subsystem keeps integrating with it: LogH's stored normalisation (R = G M_total / V^2) and the
    // PN auxiliary velocities belong to the old masses, so they start afresh (as strip_removed does) when the mass
    // changed; otherwise the next call runs with G_norm != 1 in effect
    if (m_rem != M)
        for (auto& sub : mgr.active)
            if (std::find(sub.particle_ids.begin(), sub.particle_ids.end(), c.pid_keep) != sub.particle_ids.end()) {
                sub.scales_set = false;
                sub.has_aux_vel = false;
            }
    if (h_stype(ik) < 10) {   // a merged star restarts as a zero-age star of its mass (as execute_merger)
        auto h_mzams = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.m_zams);
        auto h_sage  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_age);
        h_mzams(ik) = m_rem;
        h_sage(ik) = 0.0;
        Kokkos::deep_copy(p.m_zams, h_mzams);
        Kokkos::deep_copy(p.stellar_age, h_sage);
    }

    mgr.ghost_pids.insert(c.pid_gone);
    mgr.step_mergers++;

    MergerEvent ev{};
    ev.type = c.type;
    ev.pid1 = c.pid_keep; ev.pid2 = c.pid_gone;
    ev.m1 = c.m_keep; ev.m2 = c.m_gone; ev.m_remnant = m_rem;
    for (int k = 0; k < 3; k++) { ev.v_kick[k] = kick[k]; ev.spin_remnant[k] = chi_rem[k]; }
    Subsystem tmp{};
    {
        double rr[3], vv[3];
        for (int k = 0; k < 3; k++) { rr[k] = c.d_gone[k] - c.d_keep[k]; vv[k] = c.u_gone[k] - c.u_keep[k]; }
        auto oe = cartesian_to_elements(rr, vv, G * M);
        tmp.orb = oe;
    }
    g_merger_tree.record(ev, tmp, g_sim_time);
    static const char* names[] = {"GW", "ISCO", "TDE", "STELLAR", "NONE"};
    std::printf("  [SUBSYS COLLISION] %s pids %d + %d, m %.4g + %.4g -> %.4g Msun, r_sep / r_coll = %.4g, r_coll = %.6g pc, a = %.4g pc,"
                " e = %.6f, n = %d, dE_int = %.6e\n", names[(int)c.type], c.pid_keep, c.pid_gone, c.m_keep, c.m_gone,
                m_rem, c.r_coll > 0.0 ? c.r_sep / c.r_coll : -1.0, c.r_coll, tmp.orb.a, tmp.orb.e, c.n_members, c.dE_int);
}

// ============================================================================
// Escaper removal
// ============================================================================

int remove_escapers(Particles& p, double G, double r_escape_rh,
                    SubsystemManager& sm, EnergyBookkeeping& books) {
    (void)sm;
    if (p.N < 10) return 0;  // too few particles for meaningful escape detection

    // Copy to host
    auto h_x  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
    auto h_y  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
    auto h_z  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
    auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
    auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
    auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
    auto h_m  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
    auto h_pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);
    auto h_sid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.subsystem_id);

    int N = p.N;

    // Compute center-of-mass position and velocity
    double M_total = 0;
    double cm_x = 0, cm_y = 0, cm_z = 0;
    double cm_vx = 0, cm_vy = 0, cm_vz = 0;
    for (int i = 0; i < N; i++) {
        double m = h_m(i);
        M_total += m;
        cm_x += m * h_x(i); cm_y += m * h_y(i); cm_z += m * h_z(i);
        cm_vx += m * h_vx(i); cm_vy += m * h_vy(i); cm_vz += m * h_vz(i);
    }
    double inv_M = 1.0 / M_total;
    cm_x *= inv_M; cm_y *= inv_M; cm_z *= inv_M;
    cm_vx *= inv_M; cm_vy *= inv_M; cm_vz *= inv_M;

    // Compute radial distances from CM
    std::vector<double> radii(N);
    auto set_radii = [&]() {
        for (int i = 0; i < N; i++) {
            double dx = h_x(i) - cm_x, dy = h_y(i) - cm_y, dz = h_z(i) - cm_z;
            radii[i] = std::sqrt(dx*dx + dy*dy + dz*dz);
        }
    };
    set_radii();

    // Estimate half-mass radius
    std::vector<std::pair<double,double>> r_mass(N);
    for (int i = 0; i < N; i++) r_mass[i] = {radii[i], h_m(i)};
    std::sort(r_mass.begin(), r_mass.end());
    double cum_mass = 0;
    double r_half = r_mass.back().first;
    for (auto& rm : r_mass) {
        cum_mass += rm.second;
        if (cum_mass >= 0.5 * M_total) {
            r_half = rm.first;
            break;
        }
    }

    double R_escape = r_escape_rh * r_half;
    // With a tidal field (NBODY6, Aarseth 2003; Baumgardt & Makino 2003): a star is lost once it is farther than
    // R_ESCAPE_RJ (2) Jacobi radii from the cluster centre, with no energy test (the tail stars are energetically
    // unbound only in the rotating frame). r_J from the current total mass (as NBODY6; an iteration on the mass
    // inside r_J strips a supertidal cluster at once instead of over its dissolution time); the centre is the c.o.m.
    // of the stars inside r_J, iterated from the overall c.o.m. so the tails do not pull it
    double r_jac = 0.0;
    const bool jacobi = g_tidal.on() && g_collisional.r_escape_rj > 0.0;
    if (jacobi) {
        for (int it = 0; it < 10; it++) {
            const double Xg[3] = {g_tidal.X[0] + cm_x, g_tidal.X[1] + cm_y, g_tidal.X[2] + cm_z};
            r_jac = g_tidal.jacobi_radius(Xg, M_total, G);
            if (!(r_jac > 0.0)) break;
            double mm = 0, cx = 0, cy = 0, cz = 0, cvx = 0, cvy = 0, cvz = 0;
            for (int i = 0; i < N; i++) {
                if (radii[i] > r_jac) continue;
                const double m = h_m(i);
                mm += m; cx += m * h_x(i); cy += m * h_y(i); cz += m * h_z(i);
                cvx += m * h_vx(i); cvy += m * h_vy(i); cvz += m * h_vz(i);
            }
            if (mm <= 0.0) break;
            const double shift = std::sqrt((cx / mm - cm_x) * (cx / mm - cm_x) + (cy / mm - cm_y) * (cy / mm - cm_y) +
                                           (cz / mm - cm_z) * (cz / mm - cm_z));
            cm_x = cx / mm; cm_y = cy / mm; cm_z = cz / mm;
            cm_vx = cvx / mm; cm_vy = cvy / mm; cm_vz = cvz / mm;
            set_radii();
            if (shift <= 1e-6 * r_jac) break;
        }
        if (r_jac > 0.0) R_escape = g_collisional.r_escape_rj * r_jac;
    }
    if (R_escape <= 0) return 0;

    int n_escaped = 0;
    for (int i = 0; i < N; i++) {
        if (radii[i] < R_escape) continue;

        // Skip particles in subsystems (their CoM should be checked, but that's complex)
        if (h_sid(i) >= 0) continue;

        // Check if unbound and moving outward
        double dx = h_x(i) - cm_x, dy = h_y(i) - cm_y, dz = h_z(i) - cm_z;
        double dvx = h_vx(i) - cm_vx, dvy = h_vy(i) - cm_vy, dvz = h_vz(i) - cm_vz;
        double r = radii[i];
        double v2 = dvx*dvx + dvy*dvy + dvz*dvz;
        double r_dot_v = dx*dvx + dy*dvy + dz*dvz;

        // Specific energy: E_spec = 0.5*v^2 - G*M_enclosed/r
        // Approximate M_enclosed ~ M_total for particles at large r
        double E_spec = 0.5 * v2 - G * M_total / r;
        // Isolated cluster: moving outward and unbound
        if (!jacobi && (r_dot_v <= 0 || E_spec <= 0)) continue;

        // This particle is escaping
        // Energy accounting: the particle's share of the measured E = T + U (T in the frame of the diagnostics)
        double KE_esc = 0.5 * h_m(i) * (h_vx(i)*h_vx(i) + h_vy(i)*h_vy(i) + h_vz(i)*h_vz(i));
        // PE with all other particles
        double PE_esc = 0;
        for (int j = 0; j < N; j++) {
            if (j == i || h_m(j) <= 0) continue;
            double djx = h_x(j) - h_x(i), djy = h_y(j) - h_y(i), djz = h_z(j) - h_z(i);
            double rij = std::sqrt(djx*djx + djy*djy + djz*djz);
            PE_esc -= G * h_m(i) * h_m(j) / rij;
        }
        books.E_escape += KE_esc + PE_esc;
        {
            double ri[3] = {h_x(i), h_y(i), h_z(i)}, vi[3] = {h_vx(i), h_vy(i), h_vz(i)};
            books.sink_PL(h_m(i), ri, vi);
        }

        // Record escaper event before zeroing mass
        g_escaper_log.record(h_pid(i), g_sim_time, h_m(i),
                             h_x(i), h_y(i), h_z(i),
                             h_vx(i), h_vy(i), h_vz(i),
                             r, std::sqrt(v2), E_spec);

        h_m(i) = 0.0;  // mark for removal
        n_escaped++;
    }

    if (n_escaped > 0) {
        Kokkos::deep_copy(p.mass, h_m);
        std::printf("ESCAPER: removed %d particles, E_escape=%.4e, R_half=%.4e, R_escape=%.4e, r_J=%.9e, M=%.9e\n",
                    n_escaped, books.E_escape, r_half, R_escape, r_jac, M_total);
    }

    return n_escaped;
}
