#include "mergers.h"
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
    // R_td = 1.3 * ((m_star + M_bh) / m_star)^{1/3} * R_star
    // R_star should be in the same units as positions (pc)
    return 1.3 * std::cbrt((m_star + M_bh) / m_star) * R_star;
}

// ============================================================================
// BH-BH merger remnant (Zlochower+ 2015 / Jimenez-Forteza+ 2017)
// ============================================================================

void bh_bh_remnant(double m1, double m2, const double chi1[3], const double chi2[3],
                   const double L_orb[3], double& m_rem, double chi_rem[3],
                   double v_kick[3], double G, double c_code) {
    double M = m1 + m2;
    double eta = m1 * m2 / (M * M);
    double q = std::min(m1, m2) / std::max(m1, m2);
    double delta_m = std::fabs(m1 - m2) / M;

    // Radiated energy fraction (Barausse+ 2012 fit)
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

    // GW kick (Campanelli+ 2007 / Lousto+ 2012 simplified)
    // v_kick ~ A * eta^2 * delta_m * (1 + B * delta_m)
    // Maximum kick ~5000 km/s for edge cases, typically ~200-500 km/s
    double A = 12000.0;  // km/s
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

    auto h_x  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
    auto h_y  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
    auto h_z  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
    auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
    auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
    auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
    auto h_m  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
    auto h_pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);
    auto h_stype = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_type);
    auto h_srad = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_radius);

    int max_pid = 0;
    for (int i = 0; i < p.N; i++) max_pid = std::max(max_pid, h_pid(i));
    std::vector<int> pid_to_idx(max_pid + 1, -1);
    for (int i = 0; i < p.N; i++) pid_to_idx[h_pid(i)] = i;

    double R_sun_to_pc = 2.254e-8;  // R_sun in parsecs

    for (const auto& sub : mgr.active) {
        if (sub.n_bodies != 2) continue;

        int idx1 = pid_to_idx[sub.particle_ids[0]];
        int idx2 = pid_to_idx[sub.particle_ids[1]];
        if (idx1 < 0 || idx2 < 0) continue;

        double dx = h_x(idx2) - h_x(idx1);
        double dy = h_y(idx2) - h_y(idx1);
        double dz = h_z(idx2) - h_z(idx1);
        double r = std::sqrt(dx*dx + dy*dy + dz*dz);

        double m1 = h_m(idx1), m2 = h_m(idx2);
        int stype1 = h_stype(idx1), stype2 = h_stype(idx2);
        bool is_bh1 = (stype1 == 14), is_bh2 = (stype2 == 14);

        MergerEvent ev{};
        ev.type = MergerType::NONE;
        ev.pid1 = sub.particle_ids[0];
        ev.pid2 = sub.particle_ids[1];

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
            double R1_pc = h_srad(idx1) * R_sun_to_pc;
            double R2_pc = h_srad(idx2) * R_sun_to_pc;
            if (r < R1_pc + R2_pc) {
                ev.type = MergerType::STELLAR_COLLISION;
                ev.m_remnant = m1 + m2;
            }
        }

        // Tidal disruption (star + BH)
        if (ev.type == MergerType::NONE) {
            if (is_bh1 && !is_bh2) {
                double R_star_pc = h_srad(idx2) * R_sun_to_pc;
                double R_td = tidal_disruption_radius(m2, R_star_pc, m1);
                if (r < R_td) {
                    ev.type = MergerType::TIDAL_DISRUPTION;
                    ev.m_remnant = m1;  // BH eats the star (simplified)
                }
            } else if (!is_bh1 && is_bh2) {
                double R_star_pc = h_srad(idx1) * R_sun_to_pc;
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
                    SubsystemManager& mgr, EnergyBookkeeping& books, double G) {
    auto h_x  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
    auto h_y  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
    auto h_z  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
    auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
    auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
    auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
    auto h_m  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
    auto h_pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);

    int max_pid = 0;
    for (int i = 0; i < p.N; i++) max_pid = std::max(max_pid, h_pid(i));
    std::vector<int> pid_to_idx(max_pid + 1, -1);
    for (int i = 0; i < p.N; i++) pid_to_idx[h_pid(i)] = i;

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

    // Energy dissipated
    double dm = M - event.m_remnant;
    books.E_merge += 0.5 * dm * (cvx*cvx + cvy*cvy + cvz*cvz);

    // Place remnant at idx1, mark idx2 for removal
    h_x(idx1) = cx; h_y(idx1) = cy; h_z(idx1) = cz;
    h_vx(idx1) = cvx + event.v_kick[0];
    h_vy(idx1) = cvy + event.v_kick[1];
    h_vz(idx1) = cvz + event.v_kick[2];
    h_m(idx1) = event.m_remnant;

    // Mark particle 2 with zero mass (will be compacted later)
    h_m(idx2) = 0.0;

    Kokkos::deep_copy(p.x, h_x);
    Kokkos::deep_copy(p.y, h_y);
    Kokkos::deep_copy(p.z, h_z);
    Kokkos::deep_copy(p.vx, h_vx);
    Kokkos::deep_copy(p.vy, h_vy);
    Kokkos::deep_copy(p.vz, h_vz);
    Kokkos::deep_copy(p.mass, h_m);

    std::printf("MERGER: type=%d, pid %d + %d -> remnant mass=%.4f M_sun\n",
                (int)event.type, event.pid1, event.pid2, event.m_remnant);
}

// ============================================================================
// Escaper removal
// ============================================================================

int remove_escapers(Particles& p, double G, double R_escape,
                    SubsystemManager& sm, EnergyBookkeeping& books) {
    // TODO: Implement deterministic prefix-scan compaction
    // For now, return 0 (no escapers removed)
    return 0;
}
