#include "initial_conditions.h"
#include "constants.h"
#ifdef ENABLE_COLLISIONAL
#include "collisional_config.h"
#include "kepler.h"
#define COLL_ENABLED g_collisional.enabled
#else
#define COLL_ENABLED false
#endif
#include "sse.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <random>
#include <algorithm>
#include <vector>
#include <string>
#include <unordered_map>

void init_outer_planets(Particles& p) {
    p.allocate(5, COLL_ENABLED);

    double pos_au[][3] = {
        {0.0, 0.0, 0.0},
        {-3.5023653, -3.8169847, -1.5507963},
        {9.0755314, -3.0458353, -1.6483708},
        {8.3101420, -16.2901086, -7.2521278},
        {11.4707666, -25.7294829, -10.8169456}
    };
    double vel_aud[][3] = {
        {0.0, 0.0, 0.0},
        {0.00565429, -0.00412490, -0.00190589},
        {0.00168318, 0.00483525, 0.00192462},
        {0.00354178, 0.00137102, 0.00055029},
        {0.00288930, 0.00114527, 0.00039677}
    };
    double mass_msun[] = {
        1.0, 9.54786104043e-4, 2.85583733151e-4,
        4.37273164546e-5, 5.17759138449e-5
    };

    auto h_x = Kokkos::create_mirror_view(p.x);
    auto h_y = Kokkos::create_mirror_view(p.y);
    auto h_z = Kokkos::create_mirror_view(p.z);
    auto h_vx = Kokkos::create_mirror_view(p.vx);
    auto h_vy = Kokkos::create_mirror_view(p.vy);
    auto h_vz = Kokkos::create_mirror_view(p.vz);
    auto h_mass = Kokkos::create_mirror_view(p.mass);

    // Convert: positions AU→pc, velocities AU/day→km/s, masses already in M_sun
    constexpr double AU_PER_DAY_TO_KMS = 149597870.7 / 86400.0;  // 1 AU/day ≈ 1731.457 km/s
    for (int i = 0; i < 5; i++) {
        h_x(i)  = pos_au[i][0] * AU_TO_PC;
        h_y(i)  = pos_au[i][1] * AU_TO_PC;
        h_z(i)  = pos_au[i][2] * AU_TO_PC;
        h_vx(i) = vel_aud[i][0] * AU_PER_DAY_TO_KMS;
        h_vy(i) = vel_aud[i][1] * AU_PER_DAY_TO_KMS;
        h_vz(i) = vel_aud[i][2] * AU_PER_DAY_TO_KMS;
        h_mass(i) = mass_msun[i];
    }

    Kokkos::deep_copy(p.x, h_x);
    Kokkos::deep_copy(p.y, h_y);
    Kokkos::deep_copy(p.z, h_z);
    Kokkos::deep_copy(p.vx, h_vx);
    Kokkos::deep_copy(p.vy, h_vy);
    Kokkos::deep_copy(p.vz, h_vz);
    Kokkos::deep_copy(p.mass, h_mass);
}

// =========================================================================
// Stellar evolution via SSE (Hurley et al. 2000)
// =========================================================================

static double turnoff_mass(double age_gyr, double Z, const SSEConfig* sse_cfg = nullptr) {
    // Binary search: find the mass where SSE transitions from MS to post-MS
    SSEContext sse(Z, sse_cfg ? *sse_cfg : SSEConfig{});
    double age_myr = age_gyr * 1000.0;
    double m_low = 0.1, m_high = 100.0;
    while (m_high - m_low > 0.001) {
        double m_mid = (m_low + m_high) / 2.0;
        SSEResult res = sse.evolve(m_mid, age_myr);
        if (res.kw <= 1) // still on MS
            m_low = m_mid;
        else
            m_high = m_mid;
    }
    return (m_low + m_high) / 2.0;
}

static std::vector<double> sample_kroupa_imf(int N, std::mt19937_64& rng,
                                              double m_min = 0.08, double m_max = 150.0) {
    const double m_break = 0.5, alpha1 = 1.3, alpha2 = 2.3;
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    double I1 = (m_min < m_break)
        ? (std::pow(m_break, 1-alpha1) - std::pow(m_min, 1-alpha1)) / (1-alpha1) : 0;
    double C = std::pow(m_break, alpha2 - alpha1);
    double I2 = (m_max > m_break)
        ? C * (std::pow(std::min(m_max, 150.0), 1-alpha2) - std::pow(std::max(m_break, m_min), 1-alpha2)) / (1-alpha2) : 0;
    double p1 = I1 / (I1 + I2);

    std::vector<double> masses(N);
    for (int i = 0; i < N; i++) {
        if (uni(rng) < p1) {
            double u = uni(rng);
            masses[i] = std::pow(std::pow(m_min, 1-alpha1) + u * (std::pow(m_break, 1-alpha1) - std::pow(m_min, 1-alpha1)), 1.0/(1-alpha1));
        } else {
            double u = uni(rng);
            masses[i] = std::pow(std::pow(m_break, 1-alpha2) + u * (std::pow(m_max, 1-alpha2) - std::pow(m_break, 1-alpha2)), 1.0/(1-alpha2));
        }
    }
    return masses;
}

// =========================================================================
// Plummer sphere generator
// =========================================================================

void init_plummer(Particles& p, int N, double M_total_msun, double r_half_pc, int seed,
                  const std::string& mass_distribution, double age_gyr, double metallicity,
                  bool remove_bh, double binary_frac,
                  double bin_a_min_au, double bin_a_max_au,
                  bool add_imbh, double imbh_m1, double imbh_m2,
                  double imbh_a_pc, double imbh_e, double imbh_spin,
                  double r_break_au,
                  const SSEConfig* sse_cfg) {
    // Use host-side std::vectors so we can resize for REMOVE_BH / ADD_IMBH
    double M_total = M_total_msun;
    double a_pc = r_half_pc / PLUMMER_SCALE_FACTOR;
    double a = a_pc;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    std::vector<double> hx(N), hy(N), hz(N);
    std::vector<double> hvx(N), hvy(N), hvz(N);
    std::vector<double> hmass(N);

    // Collisional fields (populated for all mass distributions)
    std::vector<int>    h_kw(N, 1);         // stellar type (default: MS)
    std::vector<double> h_mzams(N, 0.0);    // ZAMS mass
    std::vector<double> h_sage(N, 0.0);     // stellar age (Myr)
    std::vector<double> h_srad(N, 0.0);     // radius (R_sun)
    std::vector<double> h_lum(N, 0.0);      // luminosity (L_sun)
    std::vector<double> h_mcore(N, 0.0);    // core mass

    // =====================================================================
    // Generate masses
    // =====================================================================
    if (mass_distribution == "equal") {
        double m_per = M_total / N;
        for (int i = 0; i < N; i++) {
            hmass[i] = m_per;
            h_mzams[i] = m_per;
            h_srad[i] = std::pow(m_per, 0.8);  // ZAMS R ~ M^0.8
            h_lum[i] = std::pow(m_per, 3.5);   // ZAMS L ~ M^3.5
        }
        std::printf("  Equal mass: %.4f M_sun per particle\n", M_total_msun / N);
    } else if (mass_distribution == "kroupa") {
        auto masses = sample_kroupa_imf(N, rng);
        double sum = 0;
        for (int i = 0; i < N; i++) sum += masses[i];
        for (int i = 0; i < N; i++) {
            hmass[i] = masses[i] * (M_total / sum);
            h_mzams[i] = hmass[i];
            h_srad[i] = std::pow(hmass[i], 0.8);
            h_lum[i] = std::pow(hmass[i], 3.5);
        }
        double mn = *std::min_element(masses.begin(), masses.end()) * (M_total_msun / sum);
        double mx = *std::max_element(masses.begin(), masses.end()) * (M_total_msun / sum);
        std::printf("  Raw Kroupa IMF (not evolved)\n");
        std::printf("    Mass range: %.2f - %.2f M_sun\n", mn, mx);
    } else if (mass_distribution == "kroupa_evolved") {
        SSEContext sse(metallicity, sse_cfg ? *sse_cfg : SSEConfig{});
        double age_myr = age_gyr * 1000.0;
        auto m_initial = sample_kroupa_imf(N, rng);
        int n_ms = 0, n_evolved = 0, n_wd = 0, n_ns = 0, n_bh = 0;
        for (int i = 0; i < N; i++) {
            SSEResult res = sse.evolve(m_initial[i], age_myr);
            hmass[i]   = res.mass;
            h_kw[i]    = res.kw;
            h_mzams[i] = m_initial[i];
            h_sage[i]  = age_myr;
            h_srad[i]  = res.radius;
            h_lum[i]   = res.luminosity;
            h_mcore[i] = res.mass_core;
            if (res.kw <= 1) n_ms++;
            else if (res.kw >= 2 && res.kw <= 9) n_evolved++;
            else if (res.kw >= 10 && res.kw <= 12) n_wd++;
            else if (res.kw == 13) n_ns++;
            else if (res.kw == 14) n_bh++;
        }
        // Use physical SSE masses directly (no rescaling)
        M_total_msun = 0;
        for (int i = 0; i < N; i++) M_total_msun += hmass[i];
        M_total = M_total_msun;

        double m_to = turnoff_mass(age_gyr, metallicity, sse_cfg);
        std::printf("  Stellar population at %.1f Gyr (Z=%.4f, full SSE):\n", age_gyr, metallicity);
        std::printf("    Turn-off mass: %.2f M_sun\n", m_to);
        std::printf("    Total mass:    %.0f M_sun\n", M_total_msun);
        std::printf("    Main sequence: %d (%.1f%%)\n", n_ms, 100.0 * n_ms / N);
        std::printf("    Evolved:       %d (%.1f%%)\n", n_evolved, 100.0 * n_evolved / N);
        std::printf("    White dwarfs:  %d (%.1f%%)\n", n_wd, 100.0 * n_wd / N);
        std::printf("    Neutron stars: %d (%.1f%%)\n", n_ns, 100.0 * n_ns / N);
        std::printf("    Black holes:   %d (%.1f%%)\n", n_bh, 100.0 * n_bh / N);
        double mn = *std::min_element(hmass.begin(), hmass.end());
        double mx = *std::max_element(hmass.begin(), hmass.end());
        std::printf("    Mass range: %.4f - %.4f M_sun\n", mn, mx);
        double ms_mn = 1e30, ms_mx = 0;
        double wd_mn = 1e30, wd_mx = 0, ns_mn = 1e30, ns_mx = 0, bh_mn = 1e30, bh_mx = 0;
        for (int i = 0; i < N; i++) {
            if (h_kw[i] <= 1) { ms_mn = std::min(ms_mn, hmass[i]); ms_mx = std::max(ms_mx, hmass[i]); }
            else if (h_kw[i] >= 10 && h_kw[i] <= 12) { wd_mn = std::min(wd_mn, hmass[i]); wd_mx = std::max(wd_mx, hmass[i]); }
            else if (h_kw[i] == 13) { ns_mn = std::min(ns_mn, hmass[i]); ns_mx = std::max(ns_mx, hmass[i]); }
            else if (h_kw[i] == 14) { bh_mn = std::min(bh_mn, hmass[i]); bh_mx = std::max(bh_mx, hmass[i]); }
        }
        if (n_ms > 0) std::printf("    MS masses:  %.4f - %.4f M_sun\n", ms_mn, ms_mx);
        if (n_wd > 0) std::printf("    WD masses:  %.4f - %.4f M_sun\n", wd_mn, wd_mx);
        if (n_ns > 0) std::printf("    NS masses:  %.4f - %.4f M_sun\n", ns_mn, ns_mx);
        if (n_bh > 0) std::printf("    BH masses:  %.4f - %.4f M_sun\n", bh_mn, bh_mx);
    }

    // Crossing time
    double t_cr_code = std::sqrt(r_half_pc * r_half_pc * r_half_pc / (G_CODE * M_total_msun));
    double t_cr_myr = t_cr_code * TIME_UNIT_MYR;
    std::printf("  Crossing time: %.3f Myr\n", t_cr_myr);

    // =====================================================================
    // Sample Plummer positions and velocities
    // =====================================================================
    for (int i = 0; i < N; i++) {
        double U = std::clamp(uni(rng), 1e-10, 1.0 - 1e-10);
        double r = a / std::sqrt(std::pow(U, -2.0/3.0) - 1.0);

        double cos_theta = 2.0 * uni(rng) - 1.0;
        double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
        double phi = 2.0 * M_PI * uni(rng);

        hx[i] = r * sin_theta * std::cos(phi);
        hy[i] = r * sin_theta * std::sin(phi);
        hz[i] = r * cos_theta;

        double v_escape = std::sqrt(2.0 * G_CODE * M_total / std::sqrt(r*r + a*a));
        while (true) {
            double q = uni(rng);
            double g_q = q * q * std::pow(1.0 - q * q, 3.5);
            if (uni(rng) * 0.1 < g_q) {
                double v_mag = q * v_escape;
                double vcos = 2.0 * uni(rng) - 1.0;
                double vsin = std::sqrt(1.0 - vcos * vcos);
                double vphi = 2.0 * M_PI * uni(rng);
                hvx[i] = v_mag * vsin * std::cos(vphi);
                hvy[i] = v_mag * vsin * std::sin(vphi);
                hvz[i] = v_mag * vcos;
                break;
            }
        }
    }

    // Center of mass correction
    {
        double com_x = 0, com_y = 0, com_z = 0;
        double com_vx = 0, com_vy = 0, com_vz = 0;
        for (int i = 0; i < N; i++) {
            com_x += hmass[i] * hx[i];   com_y += hmass[i] * hy[i];   com_z += hmass[i] * hz[i];
            com_vx += hmass[i] * hvx[i]; com_vy += hmass[i] * hvy[i]; com_vz += hmass[i] * hvz[i];
        }
        com_x /= M_total; com_y /= M_total; com_z /= M_total;
        com_vx /= M_total; com_vy /= M_total; com_vz /= M_total;
        for (int i = 0; i < N; i++) {
            hx[i] -= com_x; hy[i] -= com_y; hz[i] -= com_z;
            hvx[i] -= com_vx; hvy[i] -= com_vy; hvz[i] -= com_vz;
        }
    }

    // =====================================================================
    // Artificial binary removal: find and break spurious close bound pairs
    // =====================================================================
    if (r_break_au > 0.0) {
        double r_break_pc = r_break_au * AU_TO_PC;
        double search_r = 2.0 * r_break_pc;  // max separation for a < r_break
        double cell_size = search_r;

        // Spatial hash cell key
        struct CellKey {
            int ix, iy, iz;
            bool operator==(const CellKey& o) const {
                return ix == o.ix && iy == o.iy && iz == o.iz;
            }
        };
        struct CellHash {
            size_t operator()(const CellKey& k) const {
                size_t h = std::hash<int>()(k.ix);
                h ^= std::hash<int>()(k.iy) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= std::hash<int>()(k.iz) + 0x9e3779b9 + (h << 6) + (h >> 2);
                return h;
            }
        };
        auto to_cell = [&](double x, double y, double z) -> CellKey {
            return { static_cast<int>(std::floor(x / cell_size)),
                     static_cast<int>(std::floor(y / cell_size)),
                     static_cast<int>(std::floor(z / cell_size)) };
        };

        constexpr int max_angular_iter = 10;
        constexpr int max_full_iter = 5;
        int total_removed = 0;

        std::printf("  Artificial binary removal (r_break=%.1f AU = %.3e pc):\n",
                    r_break_au, r_break_pc);

        for (int iter = 0; iter < max_angular_iter + max_full_iter; iter++) {
            // Build spatial hash
            std::unordered_map<CellKey, std::vector<int>, CellHash> grid;
            grid.reserve(N);
            for (int i = 0; i < N; i++) {
                grid[to_cell(hx[i], hy[i], hz[i])].push_back(i);
            }

            // Find artificial binaries
            int n_found = 0;
            std::vector<bool> flagged(N, false);
            double search_r2 = search_r * search_r;

            for (int i = 0; i < N; i++) {
                if (flagged[i]) continue;
                CellKey ci = to_cell(hx[i], hy[i], hz[i]);

                for (int di = -1; di <= 1; di++)
                for (int dj = -1; dj <= 1; dj++)
                for (int dk = -1; dk <= 1; dk++) {
                    CellKey cn = { ci.ix + di, ci.iy + dj, ci.iz + dk };
                    auto it = grid.find(cn);
                    if (it == grid.end()) continue;

                    for (int j : it->second) {
                        if (j <= i || flagged[j]) continue;
                        double dx = hx[i] - hx[j], dy = hy[i] - hy[j], dz = hz[i] - hz[j];
                        double r2 = dx*dx + dy*dy + dz*dz;
                        if (r2 > search_r2 || r2 == 0.0) continue;

                        double dvx = hvx[i] - hvx[j], dvy = hvy[i] - hvy[j], dvz = hvz[i] - hvz[j];
                        double v2 = dvx*dvx + dvy*dvy + dvz*dvz;
                        double r_sep = std::sqrt(r2);
                        double M_pair = hmass[i] + hmass[j];
                        double E_spec = 0.5 * v2 - G_CODE * M_pair / r_sep;

                        if (E_spec < 0.0) {
                            double a_pair = -G_CODE * M_pair / (2.0 * E_spec);
                            if (a_pair < r_break_pc) {
                                flagged[i] = true;
                                flagged[j] = true;
                                n_found++;
                            }
                        }
                    }
                }
            }

            if (n_found == 0) {
                std::printf("    Iter %d: clean — no artificial binaries found\n", iter + 1);
                break;
            }

            total_removed += n_found;

            // Resample flagged particles
            int n_resampled = 0;
            for (int i = 0; i < N; i++) {
                if (!flagged[i]) continue;
                n_resampled++;

                if (iter < max_angular_iter) {
                    // Angular-only: preserve |r| and |v|, resample directions
                    double r = std::sqrt(hx[i]*hx[i] + hy[i]*hy[i] + hz[i]*hz[i]);
                    double v = std::sqrt(hvx[i]*hvx[i] + hvy[i]*hvy[i] + hvz[i]*hvz[i]);

                    double ct = 2.0 * uni(rng) - 1.0;
                    double st = std::sqrt(1.0 - ct * ct);
                    double ph = 2.0 * M_PI * uni(rng);
                    hx[i] = r * st * std::cos(ph);
                    hy[i] = r * st * std::sin(ph);
                    hz[i] = r * ct;

                    double vct = 2.0 * uni(rng) - 1.0;
                    double vst = std::sqrt(1.0 - vct * vct);
                    double vph = 2.0 * M_PI * uni(rng);
                    hvx[i] = v * vst * std::cos(vph);
                    hvy[i] = v * vst * std::sin(vph);
                    hvz[i] = v * vct;
                } else {
                    // Full resampling: new Plummer position + velocity
                    double U = std::clamp(uni(rng), 1e-10, 1.0 - 1e-10);
                    double r_new = a / std::sqrt(std::pow(U, -2.0/3.0) - 1.0);

                    double ct = 2.0 * uni(rng) - 1.0;
                    double st = std::sqrt(1.0 - ct * ct);
                    double ph = 2.0 * M_PI * uni(rng);
                    hx[i] = r_new * st * std::cos(ph);
                    hy[i] = r_new * st * std::sin(ph);
                    hz[i] = r_new * ct;

                    double v_esc = std::sqrt(2.0 * G_CODE * M_total / std::sqrt(r_new*r_new + a*a));
                    while (true) {
                        double q = uni(rng);
                        double g_q = q * q * std::pow(1.0 - q * q, 3.5);
                        if (uni(rng) * 0.1 < g_q) {
                            double v_mag = q * v_esc;
                            double vct = 2.0 * uni(rng) - 1.0;
                            double vst = std::sqrt(1.0 - vct * vct);
                            double vph = 2.0 * M_PI * uni(rng);
                            hvx[i] = v_mag * vst * std::cos(vph);
                            hvy[i] = v_mag * vst * std::sin(vph);
                            hvz[i] = v_mag * vct;
                            break;
                        }
                    }
                }
            }

            std::printf("    Iter %d: %d pairs (resampled %d particles, %s)\n",
                        iter + 1, n_found, n_resampled,
                        iter < max_angular_iter ? "angular" : "full");
        }

        if (total_removed > 0) {
            // Re-center COM after resampling
            double com_x = 0, com_y = 0, com_z = 0;
            double com_vx = 0, com_vy = 0, com_vz = 0;
            for (int i = 0; i < N; i++) {
                com_x += hmass[i] * hx[i];   com_y += hmass[i] * hy[i];   com_z += hmass[i] * hz[i];
                com_vx += hmass[i] * hvx[i]; com_vy += hmass[i] * hvy[i]; com_vz += hmass[i] * hvz[i];
            }
            com_x /= M_total; com_y /= M_total; com_z /= M_total;
            com_vx /= M_total; com_vy /= M_total; com_vz /= M_total;
            for (int i = 0; i < N; i++) {
                hx[i] -= com_x; hy[i] -= com_y; hz[i] -= com_z;
                hvx[i] -= com_vx; hvy[i] -= com_vy; hvz[i] -= com_vz;
            }
        }
    }

    // =====================================================================
    // REMOVE_BH: strip black holes (kw=14) from ICs
    // =====================================================================
    if (remove_bh) {
        int n_removed = 0;
        int j = 0;
        for (int i = 0; i < N; i++) {
            if (h_kw[i] == 14) { n_removed++; continue; }
            hx[j] = hx[i]; hy[j] = hy[i]; hz[j] = hz[i];
            hvx[j] = hvx[i]; hvy[j] = hvy[i]; hvz[j] = hvz[i];
            hmass[j] = hmass[i];
            h_kw[j] = h_kw[i]; h_mzams[j] = h_mzams[i]; h_sage[j] = h_sage[i];
            h_srad[j] = h_srad[i]; h_lum[j] = h_lum[i]; h_mcore[j] = h_mcore[i];
            j++;
        }
        N = j;
        hx.resize(N); hy.resize(N); hz.resize(N);
        hvx.resize(N); hvy.resize(N); hvz.resize(N);
        hmass.resize(N);
        h_kw.resize(N); h_mzams.resize(N); h_sage.resize(N);
        h_srad.resize(N); h_lum.resize(N); h_mcore.resize(N);
        M_total = 0;
        for (int i = 0; i < N; i++) M_total += hmass[i];
        M_total_msun = M_total;
        std::printf("  REMOVE_BH: removed %d black holes, N=%d, M_total=%.0f M_sun\n",
                    n_removed, N, M_total_msun);
    }

    // =====================================================================
    // BINARY_FRAC: pair primordial binaries
    // =====================================================================
#ifdef ENABLE_COLLISIONAL
    if (binary_frac > 0.0 && N >= 2) {
        int N_bin = static_cast<int>(binary_frac * N / 2);
        if (2 * N_bin > N) N_bin = N / 2;

        // Sort indices by mass (heaviest first) for mass-segregated pairing
        std::vector<int> idx(N);
        for (int i = 0; i < N; i++) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return hmass[a] > hmass[b]; });

        double bin_a_min_pc = bin_a_min_au * AU_TO_PC;
        double bin_a_max_pc = bin_a_max_au * AU_TO_PC;
        double log_a_min = std::log(bin_a_min_pc);
        double log_a_max = std::log(bin_a_max_pc);

        std::printf("  BINARY_FRAC: pairing %d binaries (%.0f%% of %d particles)\n",
                    N_bin, 100.0 * 2.0 * N_bin / N, N);

        for (int b = 0; b < N_bin; b++) {
            int i = idx[2 * b];      // primary (heavier)
            int j = idx[2 * b + 1];  // secondary

            double m_i = hmass[i], m_j = hmass[j];
            double M_bin = m_i + m_j;
            double mu = G_CODE * M_bin;

            // Log-uniform semi-major axis
            double a_bin = std::exp(log_a_min + uni(rng) * (log_a_max - log_a_min));
            // Thermal eccentricity: e = sqrt(U)
            double e_bin = std::sqrt(uni(rng));
            // Isotropic orientation
            double inc = std::acos(2.0 * uni(rng) - 1.0);
            double Omega = 2.0 * M_PI * uni(rng);
            double omega = 2.0 * M_PI * uni(rng);
            double M_anom = 2.0 * M_PI * uni(rng);

            double r_rel[3], v_rel[3];
            elements_to_cartesian(a_bin, e_bin, inc, Omega, omega, M_anom, mu, r_rel, v_rel);

            // COM position/velocity = particle i's Plummer values
            double cx = hx[i], cy = hy[i], cz = hz[i];
            // COM velocity = average of both particles' Plummer velocities
            double cvx = (m_i * hvx[i] + m_j * hvx[j]) / M_bin;
            double cvy = (m_i * hvy[i] + m_j * hvy[j]) / M_bin;
            double cvz = (m_i * hvz[i] + m_j * hvz[j]) / M_bin;

            // Split binary around COM
            hx[i]  = cx - (m_j / M_bin) * r_rel[0];
            hy[i]  = cy - (m_j / M_bin) * r_rel[1];
            hz[i]  = cz - (m_j / M_bin) * r_rel[2];
            hx[j]  = cx + (m_i / M_bin) * r_rel[0];
            hy[j]  = cy + (m_i / M_bin) * r_rel[1];
            hz[j]  = cz + (m_i / M_bin) * r_rel[2];

            hvx[i] = cvx - (m_j / M_bin) * v_rel[0];
            hvy[i] = cvy - (m_j / M_bin) * v_rel[1];
            hvz[i] = cvz - (m_j / M_bin) * v_rel[2];
            hvx[j] = cvx + (m_i / M_bin) * v_rel[0];
            hvy[j] = cvy + (m_i / M_bin) * v_rel[1];
            hvz[j] = cvz + (m_i / M_bin) * v_rel[2];
        }

        // Re-center COM after binary placement
        double com_x = 0, com_y = 0, com_z = 0;
        double com_vx = 0, com_vy = 0, com_vz = 0;
        for (int i = 0; i < N; i++) {
            com_x += hmass[i] * hx[i];   com_y += hmass[i] * hy[i];   com_z += hmass[i] * hz[i];
            com_vx += hmass[i] * hvx[i]; com_vy += hmass[i] * hvy[i]; com_vz += hmass[i] * hvz[i];
        }
        com_x /= M_total; com_y /= M_total; com_z /= M_total;
        com_vx /= M_total; com_vy /= M_total; com_vz /= M_total;
        for (int i = 0; i < N; i++) {
            hx[i] -= com_x; hy[i] -= com_y; hz[i] -= com_z;
            hvx[i] -= com_vx; hvy[i] -= com_vy; hvz[i] -= com_vz;
        }
    }
#endif

    // =====================================================================
    // ADD_IMBH: inject IMBH binary at cluster center
    // =====================================================================
#ifdef ENABLE_COLLISIONAL
    if (add_imbh) {
        int i0 = N, i1 = N + 1;
        N += 2;
        hx.resize(N); hy.resize(N); hz.resize(N);
        hvx.resize(N); hvy.resize(N); hvz.resize(N);
        hmass.resize(N);
        h_kw.resize(N); h_mzams.resize(N); h_sage.resize(N);
        h_srad.resize(N); h_lum.resize(N); h_mcore.resize(N);

        double M_imbh = imbh_m1 + imbh_m2;
        double mu = G_CODE * M_imbh;

        // Place at apoapsis (same convention as init_binary_test)
        double r_apo = imbh_a_pc * (1.0 + imbh_e);
        double v_apo = std::sqrt(mu / imbh_a_pc * (1.0 - imbh_e) / (1.0 + imbh_e + 1e-30));

        double f1 = imbh_m2 / M_imbh;
        double f2 = imbh_m1 / M_imbh;

        hx[i0] =  f1 * r_apo; hy[i0] = 0; hz[i0] = 0;
        hx[i1] = -f2 * r_apo; hy[i1] = 0; hz[i1] = 0;
        hvx[i0] = 0; hvy[i0] =  f1 * v_apo; hvz[i0] = 0;
        hvx[i1] = 0; hvy[i1] = -f2 * v_apo; hvz[i1] = 0;
        hmass[i0] = imbh_m1;
        hmass[i1] = imbh_m2;

        h_kw[i0] = 14; h_kw[i1] = 14;  // BH
        h_mzams[i0] = imbh_m1; h_mzams[i1] = imbh_m2;
        h_sage[i0] = 0; h_sage[i1] = 0;
        h_srad[i0] = 0; h_srad[i1] = 0;  // BH has no stellar radius
        h_lum[i0] = 0; h_lum[i1] = 0;
        h_mcore[i0] = imbh_m1; h_mcore[i1] = imbh_m2;

        M_total += M_imbh;
        M_total_msun = M_total;

        double period_code = 2.0 * M_PI * std::sqrt(imbh_a_pc * imbh_a_pc * imbh_a_pc / mu);
        double period_yr = period_code * TIME_UNIT_YR;
        std::printf("  ADD_IMBH: injected %.0f+%.0f M_sun BH binary at center\n", imbh_m1, imbh_m2);
        std::printf("    a=%.4e pc (%.2f AU), e=%.4f, P=%.4f yr, spin=%.2f\n",
                    imbh_a_pc, imbh_a_pc / AU_TO_PC, imbh_e, period_yr, imbh_spin);
    }
#endif

    // =====================================================================
    // Allocate Particles and deep_copy from host vectors
    // =====================================================================
    p.allocate(N, COLL_ENABLED);

    auto dk_x  = Kokkos::create_mirror_view(p.x);
    auto dk_y  = Kokkos::create_mirror_view(p.y);
    auto dk_z  = Kokkos::create_mirror_view(p.z);
    auto dk_vx = Kokkos::create_mirror_view(p.vx);
    auto dk_vy = Kokkos::create_mirror_view(p.vy);
    auto dk_vz = Kokkos::create_mirror_view(p.vz);
    auto dk_m  = Kokkos::create_mirror_view(p.mass);

    for (int i = 0; i < N; i++) {
        dk_x(i) = hx[i]; dk_y(i) = hy[i]; dk_z(i) = hz[i];
        dk_vx(i) = hvx[i]; dk_vy(i) = hvy[i]; dk_vz(i) = hvz[i];
        dk_m(i) = hmass[i];
    }

    Kokkos::deep_copy(p.x, dk_x);   Kokkos::deep_copy(p.y, dk_y);   Kokkos::deep_copy(p.z, dk_z);
    Kokkos::deep_copy(p.vx, dk_vx); Kokkos::deep_copy(p.vy, dk_vy); Kokkos::deep_copy(p.vz, dk_vz);
    Kokkos::deep_copy(p.mass, dk_m);

    // =====================================================================
    // Collisional field initialization
    // =====================================================================
#ifdef ENABLE_COLLISIONAL
    if (COLL_ENABLED) {
        auto dk_pid   = Kokkos::create_mirror_view(p.particle_id);
        auto dk_stype = Kokkos::create_mirror_view(p.stellar_type);
        auto dk_mzams = Kokkos::create_mirror_view(p.m_zams);
        auto dk_sage  = Kokkos::create_mirror_view(p.stellar_age);
        auto dk_srad  = Kokkos::create_mirror_view(p.stellar_radius);
        auto dk_lum   = Kokkos::create_mirror_view(p.luminosity);
        auto dk_mcore = Kokkos::create_mirror_view(p.core_mass);
        for (int i = 0; i < N; i++) {
            dk_pid(i)   = i;
            dk_stype(i) = h_kw[i];
            dk_mzams(i) = h_mzams[i];
            dk_sage(i)  = h_sage[i];
            dk_srad(i)  = h_srad[i];
            dk_lum(i)   = h_lum[i];
            dk_mcore(i) = h_mcore[i];
        }
        Kokkos::deep_copy(p.particle_id, dk_pid);
        Kokkos::deep_copy(p.stellar_type, dk_stype);
        Kokkos::deep_copy(p.m_zams, dk_mzams);
        Kokkos::deep_copy(p.stellar_age, dk_sage);
        Kokkos::deep_copy(p.stellar_radius, dk_srad);
        Kokkos::deep_copy(p.luminosity, dk_lum);
        Kokkos::deep_copy(p.core_mass, dk_mcore);

        // Initialize spin vectors for IMBH if requested
        if (add_imbh && imbh_spin > 0.0) {
            auto dk_spin = Kokkos::create_mirror_view(p.spin_vec);
            auto dk_smag = Kokkos::create_mirror_view(p.spin_mag);
            // Spins aligned with z-axis (orbital angular momentum direction)
            int i0 = N - 2, i1 = N - 1;
            dk_spin(i0, 0) = 0; dk_spin(i0, 1) = 0; dk_spin(i0, 2) = imbh_spin;
            dk_spin(i1, 0) = 0; dk_spin(i1, 1) = 0; dk_spin(i1, 2) = imbh_spin;
            dk_smag(i0) = imbh_spin;
            dk_smag(i1) = imbh_spin;
            Kokkos::deep_copy(p.spin_vec, dk_spin);
            Kokkos::deep_copy(p.spin_mag, dk_smag);
        }
    }
#endif

    std::printf("  Final N=%d, M_total=%.0f M_sun\n", N, M_total_msun);
}

// =========================================================================
// Isolated binary test case
// =========================================================================

void init_binary_test(Particles& p, double a_pc, double e,
                      double m1_msun, double m2_msun, double rot_deg) {
    p.allocate(2, COLL_ENABLED);

    double mu = G_CODE * (m1_msun + m2_msun);
    double M_total = m1_msun + m2_msun;

    // Start at apoapsis (mean anomaly = π): gentler w=v bootstrap for aux_vel.
    // At periapsis, V_scale is maximal → dt_norm huge → large w=v init error.
    // At apoapsis: r_rel = (-a*(1+e), 0, 0), v_rel = (0, -sqrt(mu/a*(1-e)/(1+e)), 0)
    double r_apo_val = a_pc * (1.0 + e);
    double v_apo = std::sqrt(mu / a_pc * (1.0 - e) / (1.0 + e + 1e-30));

    // Split into two particles around COM at origin
    double f1 = m2_msun / M_total;  // particle 0 gets -f1 * r_rel
    double f2 = m1_msun / M_total;  // particle 1 gets +f2 * r_rel

    auto h_x  = Kokkos::create_mirror_view(p.x);
    auto h_y  = Kokkos::create_mirror_view(p.y);
    auto h_z  = Kokkos::create_mirror_view(p.z);
    auto h_vx = Kokkos::create_mirror_view(p.vx);
    auto h_vy = Kokkos::create_mirror_view(p.vy);
    auto h_vz = Kokkos::create_mirror_view(p.vz);
    auto h_mass = Kokkos::create_mirror_view(p.mass);

    // Apoapsis: r along -x, v along -y (CCW orbit)
    h_x(0) =  f1 * r_apo_val;  h_y(0) = 0.0;  h_z(0) = 0.0;
    h_x(1) = -f2 * r_apo_val;  h_y(1) = 0.0;  h_z(1) = 0.0;
    h_vx(0) = 0.0; h_vy(0) =  f1 * v_apo; h_vz(0) = 0.0;
    h_vx(1) = 0.0; h_vy(1) = -f2 * v_apo; h_vz(1) = 0.0;
    h_mass(0) = m1_msun;
    h_mass(1) = m2_msun;
    // Rotate the orbit in its plane by rot_deg (moves the argument of periapsis; default pi)
    if (rot_deg != 0.0) {
        const double c = std::cos(rot_deg * M_PI / 180.0), s = std::sin(rot_deg * M_PI / 180.0);
        for (int i = 0; i < 2; i++) {
            const double x = h_x(i), y = h_y(i), vx = h_vx(i), vy = h_vy(i);
            h_x(i) = c * x - s * y;   h_y(i) = s * x + c * y;
            h_vx(i) = c * vx - s * vy; h_vy(i) = s * vx + c * vy;
        }
    }

    Kokkos::deep_copy(p.x, h_x);   Kokkos::deep_copy(p.y, h_y);   Kokkos::deep_copy(p.z, h_z);
    Kokkos::deep_copy(p.vx, h_vx); Kokkos::deep_copy(p.vy, h_vy); Kokkos::deep_copy(p.vz, h_vz);
    Kokkos::deep_copy(p.mass, h_mass);

    // Set stellar types: BH (type 14) for masses above M_PN threshold (collisional fields only exist
    // in collisional mode; writing them otherwise overruns an empty View)
    if (p.collisional) {
        auto h_stype = Kokkos::create_mirror_view(p.stellar_type);
        h_stype(0) = (m1_msun >= 5.0) ? 14 : 0;
        h_stype(1) = (m2_msun >= 5.0) ? 14 : 0;
        Kokkos::deep_copy(p.stellar_type, h_stype);
    }

    // Informational output
    double period_code = 2.0 * M_PI * std::sqrt(a_pc * a_pc * a_pc / mu);
    double period_yr = period_code * TIME_UNIT_YR;
    double r_peri = a_pc * (1.0 - e);

    std::printf("Binary test: m1=%.1f m2=%.1f M_sun, a=%.4e pc (%.2f AU)\n",
                m1_msun, m2_msun, a_pc, a_pc / AU_TO_PC);
    std::printf("  e=%.4f, r_peri=%.4e pc (%.2f AU), r_apo=%.4e pc (%.2f AU)\n",
                e, r_peri, r_peri / AU_TO_PC, r_apo_val, r_apo_val / AU_TO_PC);
    std::printf("  Period = %.4f yr (%.4e Myr), starting at APOAPSIS\n", period_yr, period_yr / 1e6);
}

void init_binary_test_peri(Particles& p, double a_pc, double e,
                           double m1_msun, double m2_msun) {
    p.allocate(2, COLL_ENABLED);

    double mu = G_CODE * (m1_msun + m2_msun);
    double M_total = m1_msun + m2_msun;

    // Start at periapsis (mean anomaly = 0):
    //   r_rel = (a*(1-e), 0, 0),  v_rel = (0, sqrt(mu/a * (1+e)/(1-e)), 0)
    double r_peri = a_pc * (1.0 - e);
    double v_peri = std::sqrt(mu / a_pc * (1.0 + e) / (1.0 - e + 1e-30));

    double f1 = m2_msun / M_total;
    double f2 = m1_msun / M_total;

    auto h_x  = Kokkos::create_mirror_view(p.x);
    auto h_y  = Kokkos::create_mirror_view(p.y);
    auto h_z  = Kokkos::create_mirror_view(p.z);
    auto h_vx = Kokkos::create_mirror_view(p.vx);
    auto h_vy = Kokkos::create_mirror_view(p.vy);
    auto h_vz = Kokkos::create_mirror_view(p.vz);
    auto h_mass = Kokkos::create_mirror_view(p.mass);

    h_x(0) = -f1 * r_peri;  h_y(0) = 0.0;  h_z(0) = 0.0;
    h_x(1) =  f2 * r_peri;  h_y(1) = 0.0;  h_z(1) = 0.0;
    h_vx(0) = 0.0; h_vy(0) = -f1 * v_peri; h_vz(0) = 0.0;
    h_vx(1) = 0.0; h_vy(1) =  f2 * v_peri; h_vz(1) = 0.0;
    h_mass(0) = m1_msun;
    h_mass(1) = m2_msun;

    Kokkos::deep_copy(p.x, h_x);   Kokkos::deep_copy(p.y, h_y);   Kokkos::deep_copy(p.z, h_z);
    Kokkos::deep_copy(p.vx, h_vx); Kokkos::deep_copy(p.vy, h_vy); Kokkos::deep_copy(p.vz, h_vz);
    Kokkos::deep_copy(p.mass, h_mass);

    // Set stellar types: BH (type 14) for masses above M_PN threshold (collisional fields only exist
    // in collisional mode; writing them otherwise overruns an empty View)
    if (p.collisional) {
        auto h_stype = Kokkos::create_mirror_view(p.stellar_type);
        h_stype(0) = (m1_msun >= 5.0) ? 14 : 0;
        h_stype(1) = (m2_msun >= 5.0) ? 14 : 0;
        Kokkos::deep_copy(p.stellar_type, h_stype);
    }

    double period_code = 2.0 * M_PI * std::sqrt(a_pc * a_pc * a_pc / mu);
    double period_yr = period_code * TIME_UNIT_YR;
    double r_apo = a_pc * (1.0 + e);

    std::printf("Binary test: m1=%.1f m2=%.1f M_sun, a=%.4e pc (%.2f AU)\n",
                m1_msun, m2_msun, a_pc, a_pc / AU_TO_PC);
    std::printf("  e=%.4f, r_peri=%.4e pc (%.2f AU), r_apo=%.4e pc (%.2f AU)\n",
                e, r_peri, r_peri / AU_TO_PC, r_apo, r_apo / AU_TO_PC);
    std::printf("  Period = %.4f yr (%.4e Myr), starting at PERIAPSIS\n", period_yr, period_yr / 1e6);
}

// =========================================================================
// Hierarchical triple test case
// =========================================================================

void init_triple_test(Particles& p,
                      double a_in_pc, double e_in, double m1_msun, double m2_msun,
                      double a_out_pc, double e_out, double m3_msun,
                      double i_deg, bool outer_first) {
    p.allocate(3, COLL_ENABLED);

    double M_in  = m1_msun + m2_msun;
    double M_tot = M_in + m3_msun;
    double mu_in  = G_CODE * M_in;
    double mu_out = G_CODE * M_tot;
    double i_rad  = i_deg * M_PI / 180.0;

    // --- Inner binary at apoapsis (in x-y plane) ---
    double r_apo_in = a_in_pc * (1.0 + e_in);
    double v_apo_in = std::sqrt(mu_in / a_in_pc * (1.0 - e_in) / (1.0 + e_in + 1e-30));
    double f1 = m2_msun / M_in;   // particle 0 fraction
    double f2 = m1_msun / M_in;   // particle 1 fraction

    // Inner binary positions/velocities relative to inner COM
    // Apoapsis along +x, velocity along +y (CCW orbit, L along +z)
    double x0 =  f1 * r_apo_in,  vy0 =  f1 * v_apo_in;
    double x1 = -f2 * r_apo_in,  vy1 = -f2 * v_apo_in;

    // --- Outer orbit at apoapsis ---
    // Outer angular momentum direction: L_out = (0, sin(i), cos(i))
    // At apoapsis: position along +x, velocity along (0, cos(i), -sin(i))
    double r_apo_out = a_out_pc * (1.0 + e_out);
    double v_apo_out = std::sqrt(mu_out / a_out_pc * (1.0 - e_out) / (1.0 + e_out + 1e-30));

    // Outer relative position and velocity (m3 relative to inner COM)
    double r3_x = r_apo_out;
    double v3_y =  v_apo_out * std::cos(i_rad);
    double v3_z = -v_apo_out * std::sin(i_rad);

    // Shift to total COM frame:
    // inner COM offset = -(m3/M_tot) * r3_vec
    // m3 offset = (M_in/M_tot) * r3_vec
    double f_in  = -m3_msun / M_tot;   // inner COM shift fraction
    double f_out =  M_in / M_tot;      // outer body shift fraction

    auto h_x  = Kokkos::create_mirror_view(p.x);
    auto h_y  = Kokkos::create_mirror_view(p.y);
    auto h_z  = Kokkos::create_mirror_view(p.z);
    auto h_vx = Kokkos::create_mirror_view(p.vx);
    auto h_vy = Kokkos::create_mirror_view(p.vy);
    auto h_vz = Kokkos::create_mirror_view(p.vz);
    auto h_mass = Kokkos::create_mirror_view(p.mass);

    // Inner members m1, m2 and the outer body m3; outer_first stores m3 at index 0 (lowest particle id), so
    // code that takes the inner pair from the first two ids is caught
    const double px[3] = {x0 + f_in * r3_x, x1 + f_in * r3_x, f_out * r3_x};
    const double pvy[3] = {vy0 + f_in * v3_y, vy1 + f_in * v3_y, f_out * v3_y};
    const double pvz[3] = {f_in * v3_z, f_in * v3_z, f_out * v3_z};
    const double pm[3] = {m1_msun, m2_msun, m3_msun};
    for (int k = 0; k < 3; k++) {
        const int i = outer_first ? (k + 1) % 3 : k;
        h_x(i) = px[k];  h_y(i) = 0.0;     h_z(i) = 0.0;
        h_vx(i) = 0.0;   h_vy(i) = pvy[k]; h_vz(i) = pvz[k];
        h_mass(i) = pm[k];
    }

    Kokkos::deep_copy(p.x, h_x);   Kokkos::deep_copy(p.y, h_y);   Kokkos::deep_copy(p.z, h_z);
    Kokkos::deep_copy(p.vx, h_vx); Kokkos::deep_copy(p.vy, h_vy); Kokkos::deep_copy(p.vz, h_vz);
    Kokkos::deep_copy(p.mass, h_mass);

    // Set stellar types: BH (type 14) for masses above M_PN threshold (collisional fields only exist
    // in collisional mode; writing them otherwise overruns an empty View)
    if (p.collisional) {
        auto h_stype = Kokkos::create_mirror_view(p.stellar_type);
        for (int i = 0; i < 3; i++) h_stype(i) = (h_mass(i) >= 5.0) ? 14 : 0;
        Kokkos::deep_copy(p.stellar_type, h_stype);
    }

    // Informational output
    double P_in_code = 2.0 * M_PI * std::sqrt(a_in_pc * a_in_pc * a_in_pc / mu_in);
    double P_out_code = 2.0 * M_PI * std::sqrt(a_out_pc * a_out_pc * a_out_pc / mu_out);
    double P_in_yr = P_in_code * TIME_UNIT_YR;
    double P_out_yr = P_out_code * TIME_UNIT_YR;
    double ratio = a_out_pc / a_in_pc;

    // Kozai-Lidov timescale estimate: T_KL ~ (M_tot/m3) * (a_out/a_in)^3 * P_in / (1-e_out^2)^{3/2}
    double T_KL_yr = (M_tot / m3_msun) * ratio * ratio * ratio * P_in_yr
                     / std::pow(1.0 - e_out * e_out, 1.5);

    std::printf("Triple test (hierarchical):\n");
    std::printf("  Inner: m1=%.1f m2=%.1f M_sun, a=%.4e pc (%.4f AU), e=%.4f\n",
                m1_msun, m2_msun, a_in_pc, a_in_pc / AU_TO_PC, e_in);
    std::printf("  Outer: m3=%.1f M_sun, a=%.4e pc (%.4f AU), e=%.4f\n",
                m3_msun, a_out_pc, a_out_pc / AU_TO_PC, e_out);
    std::printf("  Mutual inclination: %.1f deg\n", i_deg);
    std::printf("  a_out/a_in = %.1f\n", ratio);
    std::printf("  P_inner = %.6f yr, P_outer = %.6f yr\n", P_in_yr, P_out_yr);
    std::printf("  Kozai-Lidov timescale ~ %.2f yr (%.4e Myr)\n", T_KL_yr, T_KL_yr / 1e6);
    std::printf("  Starting both orbits at APOAPSIS\n");
}
