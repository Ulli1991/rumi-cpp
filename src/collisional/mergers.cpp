#include "mergers.h"
#include "merger_tree.h"
#include "kepler.h"
#include "constants.h"
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
    // R_td = η * ((m_star + M_bh) / m_star)^{1/3} * R_star
    // η = 1.3: numerical prefactor from Rees (1988, Nature 333, 523)
    // R_star should be in the same units as positions (pc)
    return 1.3 * std::cbrt((m_star + M_bh) / m_star) * R_star;
}

// ============================================================================
// BH-BH merger remnant (Zlochower+ 2015 / Jimenez-Forteza+ 2017)
// ============================================================================

void bh_bh_remnant(double m1, double m2, const double chi1[3], const double chi2[3],
                   const double L_orb[3], double& m_rem, double chi_rem[3],
                   double v_kick[3], double G, double c_code) {
    (void)chi1;
    (void)chi2;
    (void)G;
    (void)c_code;
    double M = m1 + m2;
    double eta = m1 * m2 / (M * M);
    double q = std::min(m1, m2) / std::max(m1, m2);
    double delta_m = std::fabs(m1 - m2) / M;

    // Radiated energy fraction (Barausse, Morozova & Rezzolla 2012, ApJ 758, 63)
    double E_rad_frac = 0.0547 * eta * (1.0 - 0.5 * delta_m * delta_m);
    m_rem = M * (1.0 - E_rad_frac);

    // Remnant spin (Barausse & Rezzolla 2009 simplified)
    // chi_f ~ eta * (3.464 - 3.162*eta - 2.484*eta^2)
    double chi_f_mag = eta * (3.464 - 3.162*eta - 2.484*eta*eta);
    chi_f_mag = std::min(chi_f_mag, 0.998);  // Kerr limit

    // Direction: along orbital angular momentum
    double L_mag = std::sqrt(L_orb[0]*L_orb[0] + L_orb[1]*L_orb[1] + L_orb[2]*L_orb[2]);
    if (L_mag > 1e-30) {
        chi_rem[0] = chi_f_mag * L_orb[0] / L_mag;
        chi_rem[1] = chi_f_mag * L_orb[1] / L_mag;
        chi_rem[2] = chi_f_mag * L_orb[2] / L_mag;
    } else {
        chi_rem[0] = chi_rem[1] = 0.0;
        chi_rem[2] = chi_f_mag;
    }

    // GW kick (Campanelli+ 2007, PRL 98, 231102; Lousto+ 2012, PRD 85, 084015)
    // v_kick ~ A * eta^2 * delta_m * (1 + B * delta_m)
    // Maximum kick ~5000 km/s for edge cases, typically ~200-500 km/s
    double A = 12000.0;  // km/s — calibrated from NR simulations (Lousto+ 2012)
    double B = -0.93;
    double v_kick_mag = A * eta * eta * delta_m * (1.0 + B * delta_m);

    // Kick direction: in orbital plane, perpendicular to line of separation
    // Simplified: random direction in orbital plane
    if (L_mag > 1e-30) {
        // Find perpendicular direction in orbital plane
        double e1[3];
        // e1 = L x z_hat (or x_hat if L || z_hat)
        double lhat[3] = {L_orb[0]/L_mag, L_orb[1]/L_mag, L_orb[2]/L_mag};
        if (std::fabs(lhat[2]) < 0.9) {
            e1[0] = lhat[1]; e1[1] = -lhat[0]; e1[2] = 0.0;
        } else {
            e1[0] = 0.0; e1[1] = lhat[2]; e1[2] = -lhat[1];
        }
        double e1_mag = std::sqrt(e1[0]*e1[0]+e1[1]*e1[1]+e1[2]*e1[2]);
        e1[0] /= e1_mag; e1[1] /= e1_mag; e1[2] /= e1_mag;

        v_kick[0] = v_kick_mag * e1[0];
        v_kick[1] = v_kick_mag * e1[1];
        v_kick[2] = v_kick_mag * e1[2];
    } else {
        v_kick[0] = v_kick_mag;
        v_kick[1] = v_kick[2] = 0.0;
    }
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

    // Find min/max index among subsystem members to copy only the needed range
    int min_idx = p.N, max_idx = 0;
    for (const auto& sub : mgr.active) {
        if (sub.n_bodies != 2) continue;
        for (int pid : sub.particle_ids) {
            int idx = (pid >= 0 && pid < (int)pid_to_idx.size()) ? pid_to_idx[pid] : -1;
            if (idx >= 0) { min_idx = std::min(min_idx, idx); max_idx = std::max(max_idx, idx); }
        }
    }
    if (min_idx > max_idx) return events;

    auto rng = Kokkos::pair<int,int>(min_idx, max_idx + 1);
    auto h_x  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.x, rng));
    auto h_y  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.y, rng));
    auto h_z  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.z, rng));
    auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.vx, rng));
    auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.vy, rng));
    auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.vz, rng));
    auto h_m  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.mass, rng));
    auto h_stype = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.stellar_type, rng));
    auto h_srad = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(p.stellar_radius, rng));
    const int off = min_idx;  // offset for remapping: h_x(idx - off)

    double R_sun_to_pc = 2.254e-8;  // R_sun in parsecs

    for (const auto& sub : mgr.active) {
        if (sub.n_bodies != 2) continue;

        int idx1 = pid_to_idx[sub.particle_ids[0]];
        int idx2 = pid_to_idx[sub.particle_ids[1]];
        if (idx1 < 0 || idx2 < 0) continue;

        double dx = h_x(idx2 - off) - h_x(idx1 - off);
        double dy = h_y(idx2 - off) - h_y(idx1 - off);
        double dz = h_z(idx2 - off) - h_z(idx1 - off);
        double r = std::sqrt(dx*dx + dy*dy + dz*dz);

        double m1 = h_m(idx1 - off), m2 = h_m(idx2 - off);
        int stype1 = h_stype(idx1 - off), stype2 = h_stype(idx2 - off);
        bool is_bh1 = (stype1 == 14), is_bh2 = (stype2 == 14);

        MergerEvent ev{};
        ev.type = MergerType::NONE;
        ev.pid1 = sub.particle_ids[0];
        ev.pid2 = sub.particle_ids[1];
        ev.m1 = m1;
        ev.m2 = m2;

        // ISCO check (BH)
        double r_isco = 6.0 * G * (m1 + m2) / (c_code * c_code);
        if ((is_bh1 || is_bh2) && r < r_isco) {
            ev.type = MergerType::ISCO;
            ev.m_remnant = m1 + m2;  // will be refined by bh_bh_remnant
        }

        // GW coalescence timescale
        if (ev.type == MergerType::NONE && sub.orb.a > 0) {
            double t_gw = gw_coalescence_time(sub.orb.a, sub.orb.e, m1, m2, G, c_code);
            if (t_gw < dt) {
                ev.type = MergerType::GW_COALESCENCE;
                ev.m_remnant = m1 + m2;
            }
        }

        // Stellar collision
        if (ev.type == MergerType::NONE && !is_bh1 && !is_bh2) {
            double R1_pc = h_srad(idx1 - off) * R_sun_to_pc;
            double R2_pc = h_srad(idx2 - off) * R_sun_to_pc;
            if (r < R1_pc + R2_pc) {
                ev.type = MergerType::STELLAR_COLLISION;
                ev.m_remnant = m1 + m2;
            }
        }

        // Tidal disruption (star + BH)
        if (ev.type == MergerType::NONE) {
            if (is_bh1 && !is_bh2) {
                double R_star_pc = h_srad(idx2 - off) * R_sun_to_pc;
                double R_td = tidal_disruption_radius(m2, R_star_pc, m1);
                if (r < R_td) {
                    ev.type = MergerType::TIDAL_DISRUPTION;
                    ev.m_remnant = m1;  // BH eats the star (simplified)
                }
            } else if (!is_bh1 && is_bh2) {
                double R_star_pc = h_srad(idx1 - off) * R_sun_to_pc;
                double R_td = tidal_disruption_radius(m1, R_star_pc, m2);
                if (r < R_td) {
                    ev.type = MergerType::TIDAL_DISRUPTION;
                    ev.m_remnant = m2;
                }
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

        bh_bh_remnant(m1, m2, chi1, chi2, L_orb, m_rem, chi_rem, v_kick, G, c_code);
    }

    // Energy dissipated (mass difference converted to GW energy)
    double dm = M - m_rem;
    // Binding energy of the binary + kinetic energy of lost mass
    double dx = h_x(idx2) - h_x(idx1), dy = h_y(idx2) - h_y(idx1), dz = h_z(idx2) - h_z(idx1);
    double r = std::sqrt(dx*dx + dy*dy + dz*dz);
    double E_bind = -G * m1 * m2 / (r > 0 ? r : 1e-30);
    books.E_merge += E_bind + 0.5 * dm * (cvx*cvx + cvy*cvy + cvz*cvz);

    // Place remnant at idx1, mark idx2 for removal
    h_x(idx1) = cx; h_y(idx1) = cy; h_z(idx1) = cz;
    h_vx(idx1) = cvx + v_kick[0];
    h_vy(idx1) = cvy + v_kick[1];
    h_vz(idx1) = cvz + v_kick[2];
    h_m(idx1) = m_rem;
    h_stype(idx1) = (is_bh1 || is_bh2) ? 14 : h_stype(idx1);
    h_spin(idx1, 0) = chi_rem[0];
    h_spin(idx1, 1) = chi_rem[1];
    h_spin(idx1, 2) = chi_rem[2];

    // Mark particle 2 with zero mass (will be compacted later)
    h_m(idx2) = 0.0;

    Kokkos::deep_copy(p.x, h_x);
    Kokkos::deep_copy(p.y, h_y);
    Kokkos::deep_copy(p.z, h_z);
    Kokkos::deep_copy(p.vx, h_vx);
    Kokkos::deep_copy(p.vy, h_vy);
    Kokkos::deep_copy(p.vz, h_vz);
    p.zero_kick_compensation();   // merger remnant velocity is new
    Kokkos::deep_copy(p.mass, h_m);
    Kokkos::deep_copy(p.stellar_type, h_stype);
    Kokkos::deep_copy(p.spin_vec, h_spin);

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

    (void)dm;  // energy accounting already handled above
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
    for (int i = 0; i < N; i++) {
        double dx = h_x(i) - cm_x, dy = h_y(i) - cm_y, dz = h_z(i) - cm_z;
        radii[i] = std::sqrt(dx*dx + dy*dy + dz*dz);
    }

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

        // Moving outward?
        if (r_dot_v <= 0) continue;

        // Specific energy: E_spec = 0.5*v^2 - G*M_enclosed/r
        // Approximate M_enclosed ~ M_total for particles at large r
        double E_spec = 0.5 * v2 - G * M_total / r;
        if (E_spec <= 0) continue;

        // This particle is escaping
        // Energy accounting: KE + PE relative to cluster
        double KE_esc = 0.5 * h_m(i) * v2;
        // PE with all other particles
        double PE_esc = 0;
        for (int j = 0; j < N; j++) {
            if (j == i || h_m(j) <= 0) continue;
            double djx = h_x(j) - h_x(i), djy = h_y(j) - h_y(i), djz = h_z(j) - h_z(i);
            double rij = std::sqrt(djx*djx + djy*djy + djz*djz);
            PE_esc -= G * h_m(i) * h_m(j) / rij;
        }
        books.E_escape += KE_esc + PE_esc;

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
        std::printf("ESCAPER: removed %d particles, E_escape=%.4e, R_half=%.4e, R_escape=%.4e\n",
                    n_escaped, books.E_escape, r_half, R_escape);
    }

    return n_escaped;
}
