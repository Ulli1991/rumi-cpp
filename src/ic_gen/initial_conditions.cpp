#include "initial_conditions.h"
#include "constants.h"
#ifdef ENABLE_COLLISIONAL
#include "collisional_config.h"
#include "kepler.h"
#include "mergers.h"
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
#include <map>
#include <utility>
#include <cstdlib>

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
// SCIC helpers (BIFROST, Rantala et al. 2023, MNRAS 522, 5180, App. A)
// =========================================================================

double scic_mass_size_r_half(double M) {
    // Eq. (A2), star clusters: r_h / pc = 2.55 (M / 1e4 M_sun)^0.24, after Brown & Gnedin (2021, MNRAS 508, 5935).
    // Eq. (A3), nuclear star clusters: log10(R_e / pc) = 1/2 below M = 10^6.5 M_sun, (1/2) log10(M / M_sun) - 11/4
    // above, "with r_h = ((1/2)^(-2/3) - 1)^(-1) R_e for the Plummer model" -- as printed that factor is 1.70, but
    // the projected half-mass radius of a Plummer sphere is a_p and r_h = (2^(2/3) - 1)^(-1/2) a_p = 1.305 a_p
    // (the paper's own relation in App. A), so we use r_h = 1.305 R_e (exponent -1/2, a typo in the paper).
    // Where SCIC switches from (A2) to (A3) is not stated; we switch at the (A3) break, 10^6.5 M_sun.
    if (M < std::pow(10.0, 6.5)) return 2.55 * std::pow(M / 1e4, 0.24);
    double log_re = 0.5 * std::log10(M) - 11.0 / 4.0;
    return PLUMMER_SCALE_FACTOR * std::pow(10.0, log_re);
}

namespace {

// BIFROST App. A, "Avoiding artificial binaries and multiplets": "impose a minimum distance r_break ~ a few times
// a_max between particles which are not members of the same binary system ... If an artificial binary is detected
// (i.e. a < r_break) the positions and velocities of the particles in question are re-sampled ... A practical
// algorithm in our SCIC code first performs the angular re-sampling method [keeping |r_i| and |v_i|] for the
// artificial binaries using up to 10 iterations before switching to the original re-sampling if artificial systems
// remain. After another 5-10 iteration rounds typically only few or zero artificial binaries remain."
// The objects are the N_s + N_b singles and binary centres of mass ("the binaries treated as point particles before
// assigning the positions and velocities of the binary components"); a binary's centre of mass is its primary's
// Plummer point (pairs[].first), the secondary's own draw is unused. The text uses both a separation (minimum
// distance) and a semi-major axis (a < r_break); we flag a pair of objects for either, r < r_break or bound with
// a < r_break (a bound pair with a < r_break has r < 2 r_break, the search radius). Both objects of a flagged pair
// are re-sampled: 10 angular rounds, then up to 10 full rounds.
template <class FR, class FD, class FS>
void avoid_artificial_binaries(const std::vector<std::pair<int,int>>& pairs,
                               std::vector<double>& hx, std::vector<double>& hy, std::vector<double>& hz,
                               std::vector<double>& hvx, std::vector<double>& hvy, std::vector<double>& hvz,
                               const std::vector<double>& hmass, double r_break_pc,
                               FR plummer_radius, FD set_direction, FS plummer_speed) {
    const int N = (int)hmass.size();
    std::vector<double> m_obj(hmass);      // object mass carried by each particle index
    std::vector<char> is_object(N, 1);
    for (auto [i, j] : pairs) { m_obj[i] = hmass[i] + hmass[j]; is_object[j] = 0; }
    std::vector<int> obj;
    obj.reserve(N);
    for (int i = 0; i < N; i++) if (is_object[i]) obj.push_back(i);
    const int n_obj = (int)obj.size();

    const double search_r = 2.0 * r_break_pc, search_r2 = search_r * search_r;
    struct Key { long long x, y, z; bool operator==(const Key& o) const { return x == o.x && y == o.y && z == o.z; } };
    struct KeyHash {
        size_t operator()(const Key& k) const {
            return (size_t)(k.x * 73856093LL) ^ (size_t)(k.y * 19349663LL) ^ (size_t)(k.z * 83492791LL);
        }
    };
    auto key = [&](int i) {
        return Key{(long long)std::floor(hx[i] / search_r), (long long)std::floor(hy[i] / search_r),
                   (long long)std::floor(hz[i] / search_r)};
    };
    std::vector<char> flagged(n_obj);
    auto detect = [&]() {
        std::unordered_map<Key, std::vector<int>, KeyHash> grid;
        grid.reserve(2 * n_obj);
        for (int k = 0; k < n_obj; k++) grid[key(obj[k])].push_back(k);
        std::fill(flagged.begin(), flagged.end(), 0);
        int n_pairs = 0;
        for (int k = 0; k < n_obj; k++) {
            int i = obj[k];
            Key c = key(i);
            for (long long dx = -1; dx <= 1; dx++)
            for (long long dy = -1; dy <= 1; dy++)
            for (long long dz = -1; dz <= 1; dz++) {
                auto it = grid.find(Key{c.x + dx, c.y + dy, c.z + dz});
                if (it == grid.end()) continue;
                for (int l : it->second) {
                    if (l <= k) continue;
                    int j = obj[l];
                    double rx = hx[i] - hx[j], ry = hy[i] - hy[j], rz = hz[i] - hz[j];
                    double r2 = rx*rx + ry*ry + rz*rz;
                    if (r2 >= search_r2) continue;
                    double r = std::sqrt(r2);
                    bool close = r < r_break_pc;
                    if (!close) {
                        double ux = hvx[i] - hvx[j], uy = hvy[i] - hvy[j], uz = hvz[i] - hvz[j];
                        double mu = G_CODE * (m_obj[i] + m_obj[j]);
                        double eps = 0.5 * (ux*ux + uy*uy + uz*uz) - mu / r;
                        close = eps < 0.0 && -mu / (2.0 * eps) < r_break_pc;
                    }
                    if (close) { flagged[k] = flagged[l] = 1; n_pairs++; }
                }
            }
        }
        return n_pairs;
    };
    constexpr int n_angular = 10, n_full = 10;
    int before = 0, after = 0, round = 0, n_resampled = 0;
    while (true) {
        int n = detect();
        if (round == 0) before = n;
        if (n == 0 || round == n_angular + n_full) { after = n; break; }
        for (int k = 0; k < n_obj; k++) {
            if (!flagged[k]) continue;
            int i = obj[k];
            n_resampled++;
            double r = std::sqrt(hx[i]*hx[i] + hy[i]*hy[i] + hz[i]*hz[i]);
            double v = std::sqrt(hvx[i]*hvx[i] + hvy[i]*hvy[i] + hvz[i]*hvz[i]);
            if (round >= n_angular) { r = plummer_radius(); v = plummer_speed(r); }
            set_direction(r, hx[i], hy[i], hz[i]);
            set_direction(v, hvx[i], hvy[i], hvz[i]);
        }
        round++;
    }
    // Keep the objects' centre of mass at rest at the origin (secondaries are placed relative to their primary)
    double M = 0, c[6] = {0, 0, 0, 0, 0, 0};
    for (int i : obj) {
        double mk = m_obj[i];
        M += mk;
        c[0] += mk * hx[i]; c[1] += mk * hy[i]; c[2] += mk * hz[i];
        c[3] += mk * hvx[i]; c[4] += mk * hvy[i]; c[5] += mk * hvz[i];
    }
    for (int i : obj) {
        hx[i] -= c[0] / M; hy[i] -= c[1] / M; hz[i] -= c[2] / M;
        hvx[i] -= c[3] / M; hvy[i] -= c[4] / M; hvz[i] -= c[5] / M;
    }
    std::printf("  IC_R_BREAK: r_break=%.4g pc, close pairs before=%d after=%d (%d objects, %d re-sampled in %d rounds:"
                " %d angular, %d full)\n", r_break_pc, before, after, n_obj, n_resampled, round,
                std::min(round, n_angular), std::max(0, round - n_angular));
}

// IC summary on objects (singles, binary centres of mass): half-mass radius of all particles about the centre of
// mass, virial ratio Q = T / |W| of the objects (W direct up to 20000 objects, else the Plummer value
// -3 pi G M^2 / (32 a)), and the binaries' internal binding energy relative to |T + W| of the objects
void print_ic_check(const std::vector<int>& partner,
                    const std::vector<double>& hx, const std::vector<double>& hy, const std::vector<double>& hz,
                    const std::vector<double>& hvx, const std::vector<double>& hvy, const std::vector<double>& hvz,
                    const std::vector<double>& hmass, double a_plummer) {
    const int N = (int)hmass.size();
    double M = 0, c[3] = {0, 0, 0};
    for (int i = 0; i < N; i++) { M += hmass[i]; c[0] += hmass[i] * hx[i]; c[1] += hmass[i] * hy[i]; c[2] += hmass[i] * hz[i]; }
    std::vector<std::pair<double,double>> rm(N);
    for (int i = 0; i < N; i++) {
        double dx = hx[i] - c[0] / M, dy = hy[i] - c[1] / M, dz = hz[i] - c[2] / M;
        rm[i] = {std::sqrt(dx*dx + dy*dy + dz*dz), hmass[i]};
    }
    std::sort(rm.begin(), rm.end());
    double cum = 0, r_h = 0;
    for (auto& [r, m] : rm) { cum += m; if (cum >= 0.5 * M) { r_h = r; break; } }

    std::vector<double> ox, oy, oz, om;
    double T = 0, E_bin = 0;
    for (int i = 0; i < N; i++) {
        int j = partner[i];
        if (j >= 0 && j < i) continue;
        if (j < 0) {
            ox.push_back(hx[i]); oy.push_back(hy[i]); oz.push_back(hz[i]); om.push_back(hmass[i]);
            T += 0.5 * hmass[i] * (hvx[i]*hvx[i] + hvy[i]*hvy[i] + hvz[i]*hvz[i]);
            continue;
        }
        double mi = hmass[i], mj = hmass[j], mb = mi + mj;
        double vx = (mi * hvx[i] + mj * hvx[j]) / mb, vy = (mi * hvy[i] + mj * hvy[j]) / mb;
        double vz = (mi * hvz[i] + mj * hvz[j]) / mb;
        ox.push_back((mi * hx[i] + mj * hx[j]) / mb); oy.push_back((mi * hy[i] + mj * hy[j]) / mb);
        oz.push_back((mi * hz[i] + mj * hz[j]) / mb); om.push_back(mb);
        T += 0.5 * mb * (vx*vx + vy*vy + vz*vz);
        double rx = hx[i] - hx[j], ry = hy[i] - hy[j], rz = hz[i] - hz[j];
        double ux = hvx[i] - hvx[j], uy = hvy[i] - hvy[j], uz = hvz[i] - hvz[j];
        E_bin += 0.5 * mi * mj / mb * (ux*ux + uy*uy + uz*uz) - G_CODE * mi * mj / std::sqrt(rx*rx + ry*ry + rz*rz);
    }
    const int n_obj = (int)om.size();
    double W = 0;
    const bool direct = n_obj <= 20000;
    if (direct) {
        for (int k = 0; k < n_obj; k++)
            for (int l = k + 1; l < n_obj; l++) {
                double dx = ox[k] - ox[l], dy = oy[k] - oy[l], dz = oz[k] - oz[l];
                W -= G_CODE * om[k] * om[l] / std::sqrt(dx*dx + dy*dy + dz*dz);
            }
    } else {
        W = -3.0 * M_PI * G_CODE * M * M / (32.0 * a_plummer);
    }
    std::printf("  IC check: N_obj=%d r_h(measured)=%.6g pc Q_obj=%.6g E_bin/|E_obj|=%.6g (W %s)\n", n_obj, r_h,
                T / std::fabs(W), std::fabs(E_bin) / std::fabs(T + W), direct ? "direct" : "Plummer");
}

}  // namespace

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
                  const SSEConfig* sse_cfg,
                  const std::string& binary_pop, bool binary_peri_check,
                  bool binary_com_sampled) {
    // Use host-side std::vectors so we can resize for REMOVE_BH / ADD_IMBH
    double M_total = M_total_msun;

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

    // Half-mass radius: user-given, or the SCIC mass-size relation of the generated cluster mass (for an evolved
    // population the present-day mass; BIFROST does not say which mass SCIC uses)
    if (r_half_pc <= 0.0) {
        r_half_pc = scic_mass_size_r_half(M_total_msun);
        std::printf("  IC r_h = %.6g pc (SCIC mass-size relation, BIFROST App. A eq. A2/A3, M = %.6g M_sun)\n",
                    r_half_pc, M_total_msun);
    } else {
        std::printf("  IC r_h = %.6g pc (PLUMMER_R_HALF)\n", r_half_pc);
    }
    const double a = r_half_pc / PLUMMER_SCALE_FACTOR;  // Plummer scale radius, r_h = (2^(2/3) - 1)^(-1/2) a_p

    // Crossing time
    double t_cr_code = std::sqrt(r_half_pc * r_half_pc * r_half_pc / (G_CODE * M_total_msun));
    double t_cr_myr = t_cr_code * TIME_UNIT_MYR;
    std::printf("  Crossing time: %.3f Myr\n", t_cr_myr);

    // =====================================================================
    // Sample Plummer positions and velocities
    // =====================================================================
    // Plummer (1911) radius from the inverted cumulative mass M(r) / M = r^3 / (r^2 + a^2)^(3/2), speed from the
    // Plummer f(E) ~ (-E)^(7/2) by von Neumann rejection (Aarseth, Henon & Wielen 1974), isotropic directions
    auto plummer_radius = [&]() {
        double U = std::clamp(uni(rng), 1e-10, 1.0 - 1e-10);
        return a / std::sqrt(std::pow(U, -2.0/3.0) - 1.0);
    };
    auto set_direction = [&](double r, double& x, double& y, double& z) {
        double ct = 2.0 * uni(rng) - 1.0;
        double st = std::sqrt(1.0 - ct * ct);
        double ph = 2.0 * M_PI * uni(rng);
        x = r * st * std::cos(ph);
        y = r * st * std::sin(ph);
        z = r * ct;
    };
    auto plummer_speed = [&](double r) {
        double v_escape = std::sqrt(2.0 * G_CODE * M_total / std::sqrt(r*r + a*a));
        while (true) {
            double q = uni(rng);
            double g_q = q * q * std::pow(1.0 - q * q, 3.5);
            if (uni(rng) * 0.1 < g_q) return q * v_escape;
        }
    };
    for (int i = 0; i < N; i++) {
        double r = plummer_radius();
        set_direction(r, hx[i], hy[i], hz[i]);
        set_direction(plummer_speed(r), hvx[i], hvy[i], hvz[i]);
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
    // ---- Pairing: list of (primary, secondary) indices ----------------------------------------------
    // Decided before the artificial-binary pass so that it acts on the N_s + N_b objects (singles and binary
    // centres of mass, BIFROST App. A); a binary's centre of mass is its primary's Plummer phase-space point.
    std::vector<std::pair<int,int>> pairs;
#ifdef ENABLE_COLLISIONAL
    if (binary_frac > 0.0 && N >= 2) {
        int N_bin = static_cast<int>(binary_frac * N / 2);
        if (2 * N_bin > N) N_bin = N / 2;
        pairs.reserve(N_bin);
        int n_massive_primaries = 0;
        if (binary_pop == "mass_rank") {
            // Old behaviour: heaviest with next heaviest (q ~ 1, mass-segregated pairing)
            std::vector<int> idx(N);
            for (int i = 0; i < N; i++) idx[i] = i;
            std::sort(idx.begin(), idx.end(), [&](int a, int b) { return hmass[a] > hmass[b]; });
            for (int b = 0; b < N_bin; b++) pairs.emplace_back(idx[2 * b], idx[2 * b + 1]);
        } else if (binary_pop == "bifrost") {
            // BIFROST (Rantala et al. 2023, MNRAS 522, 5180) App. A "Binary stars", Table A1:
            //   N = N_s + 2 N_b, N_b = f_bin N / 2.
            //   Primaries above m_pair = 5 M_sun get a secondary from a flat mass-ratio distribution
            //   q = m_2/m_1 = const, 0.1 <= q < 1.0 (Kiminki 2012, Sana 2012, Kobulnicky 2014): "beginning from
            //   the most massive star we assign secondary star masses by finding the actual closest mass to the
            //   sampled random secondary mass from the remaining stars until no primary stars above 5 M_sun
            //   remain".
            //   Below m_pair the paper gives no rule; SCIC is "functionally very similar to mcluster" and the
            //   above is mcluster's pairing = 3 (Kupper et al. 2011, MNRAS 417, 2300; main.c order(), "Uniform
            //   distribution of mass ratio (0.1<q<1.0) for m>Msort and random pairing for remaining", Kiminki &
            //   Kobulnicky 2012, Sana et al. 2012): every star above msort = 5 M_sun is a primary, the closest
            //   unused mass to q m1 its secondary, the rest are paired at random; the first N_b pairs are binaries.
            constexpr double m_pair = 5.0, q_min = 0.1, q_max = 1.0;
            std::multimap<double, int> remaining;  // mass -> index, unpaired stars
            for (int i = 0; i < N; i++) remaining.emplace(hmass[i], i);
            while ((int)pairs.size() < N_bin && remaining.size() >= 2) {
                auto top = std::prev(remaining.end());
                if (top->first <= m_pair) break;
                int i1 = top->second;
                double m1 = top->first;
                remaining.erase(top);
                double m2_target = (q_min + uni(rng) * (q_max - q_min)) * m1;
                auto it = remaining.lower_bound(m2_target);
                if (it == remaining.end()) it = std::prev(it);
                else if (it != remaining.begin()) {
                    auto lo = std::prev(it);
                    if (m2_target - lo->first < it->first - m2_target) it = lo;
                }
                pairs.emplace_back(i1, it->second);
                remaining.erase(it);
                n_massive_primaries++;
            }
            // Random pairing of the rest: draw 2 (N_b - n) stars without replacement, pair consecutively
            std::vector<int> rest;
            rest.reserve(remaining.size());
            for (const auto& kv : remaining) rest.push_back(kv.second);
            int n_more = std::min(N_bin - (int)pairs.size(), (int)rest.size() / 2);
            for (int k = 0; k < 2 * n_more; k++) {  // partial Fisher-Yates
                std::uniform_int_distribution<int> pick(k, (int)rest.size() - 1);
                std::swap(rest[k], rest[pick(rng)]);
            }
            for (int b = 0; b < n_more; b++) {
                int i = rest[2 * b], j = rest[2 * b + 1];
                if (hmass[j] > hmass[i]) std::swap(i, j);
                pairs.emplace_back(i, j);
            }
        } else {
            std::fprintf(stderr, "ERROR: BINARY_POP must be bifrost or mass_rank, got '%s'\n", binary_pop.c_str());
            std::exit(1);
        }

        std::printf("  BINARY_FRAC: pairing %d binaries (%.0f%% of %d particles), BINARY_POP=%s",
                    (int)pairs.size(), 100.0 * 2.0 * pairs.size() / N, N, binary_pop.c_str());
        if (binary_pop == "bifrost") std::printf(" (%d primaries > 5 M_sun with flat q)", n_massive_primaries);
        std::printf("\n");
    }
#endif

    // =====================================================================
    // Artificial binaries and multiplets (BIFROST App. A, "Avoiding artificial binaries and multiplets")
    // =====================================================================
    std::vector<int> partner(N, -1);  // other member of a binary (primary <-> secondary), -1 for singles
    for (auto [i, j] : pairs) { partner[i] = j; partner[j] = i; }
    if (r_break_au > 0.0)
        avoid_artificial_binaries(pairs, hx, hy, hz, hvx, hvy, hvz, hmass, r_break_au * AU_TO_PC,
                                  plummer_radius, set_direction, plummer_speed);

#ifdef ENABLE_COLLISIONAL
    if (!pairs.empty()) {
        double log_a_min = std::log(bin_a_min_au * AU_TO_PC);
        double log_a_max = std::log(bin_a_max_au * AU_TO_PC);

        // ---- Orbits ---------------------------------------------------------------------------------
        // Contact distance below which the code merges the pair at once (collisional/mergers.cpp
        // detect_mergers): R1 + R2 for two stars, the tidal-disruption radius for a star and a BH, the ISCO
        // 6 G M / c^2 for two BHs. FROST-CLUSTERS II (Rantala et al. 2025) Sec. 2.3: "We ensure that for each
        // sampled binary the pericenter distance initially fulfils r_p = a(1-e) > R_p + R_s"; BIFROST App. A
        // merges such systems in the IC generator instead (SCIC), together with compact binaries whose GW
        // inspiral takes less than 0.01 Myr. We keep the pair and redraw e (and, if even e = 0 cannot clear
        // the contact distance, a): the thermal f(e) = 2e restricted to e < e_max is e = e_max sqrt(U), which
        // is what redrawing e until r_p clears gives. The same truncated form is used by Moe & Di Stefano
        // (2017, ApJS 230, 15) Eq. 3, p_e ~ e^eta on 0 < e < e_max(P) = 1 - (P / 2 d)^(-2/3) (Roche-lobe
        // fill factor <~ 70 % at periastron); we only exclude actual contact, as FROST-CLUSTERS II does.
        // Compact pairs additionally redraw e until t_GW (Peters 1964) > 0.01 Myr.
        constexpr double R_sun_pc = 2.254e-8;      // as collisional/mergers.cpp
        constexpr double c_kms = 299792.458;
        const double t_gw_min = 1e4 / TIME_UNIT_YR;  // 0.01 Myr in code time units
        int n_e_redrawn = 0, n_a_redrawn = 0, n_unpaired = 0;
        std::vector<double> q_list, q_massive, a_list, e_list;
        q_list.reserve(pairs.size()); a_list.reserve(pairs.size()); e_list.reserve(pairs.size());

        for (auto [i, j] : pairs) {
            double m_i = hmass[i], m_j = hmass[j];
            double M_bin = m_i + m_j;
            double mu = G_CODE * M_bin;

            bool compact_i = (h_kw[i] >= 10), compact_j = (h_kw[j] >= 10);
            // Pairs with a BH from collision_distance (FC-I tidal radius); two non-BHs keep (R1 + R2) R_sun as before,
            // bit for bit (R1 R_sun + R2 R_sun rounds differently and changes the redrawn e, hence the ICs)
            MergerType contact_type;
            const double r_contact = (h_kw[i] == 14 || h_kw[j] == 14)
                ? collision_distance(h_kw[i], m_i, h_srad[i] * R_sun_pc, h_kw[j], m_j, h_srad[j] * R_sun_pc, G_CODE,
                                     c_kms, contact_type)
                : (h_srad[i] + h_srad[j]) * R_sun_pc;
            bool check_gw = binary_peri_check && compact_i && compact_j;

            // Log-uniform semi-major axis (BIFROST Table A1: f(a) flat in log a)
            double a_bin = std::exp(log_a_min + uni(rng) * (log_a_max - log_a_min));
            // Thermal eccentricity f(e) = 2e (BIFROST App. A): e = sqrt(U)
            double e_bin = std::sqrt(uni(rng));
            bool ok = true;
            if (binary_peri_check && a_bin * (1.0 - e_bin) <= r_contact) {
                n_e_redrawn++;
                int tries = 0;
                while (a_bin <= r_contact && tries < 1000) {  // no e clears it: redraw a as well
                    a_bin = std::exp(log_a_min + uni(rng) * (log_a_max - log_a_min));
                    tries++;
                }
                if (tries > 0) n_a_redrawn++;
                if (a_bin <= r_contact) ok = false;
                else e_bin = (1.0 - r_contact / a_bin) * std::sqrt(uni(rng));
            }
            if (ok && check_gw) {
                int tries = 0;
                while (gw_coalescence_time(a_bin, e_bin, m_i, m_j, G_CODE, c_kms) < t_gw_min && tries < 1000) {
                    e_bin *= std::sqrt(uni(rng));  // thermal below the current e
                    tries++;
                }
                if (gw_coalescence_time(a_bin, e_bin, m_i, m_j, G_CODE, c_kms) < t_gw_min) ok = false;
            }
            if (!ok) {  // both stay single at their own Plummer phase-space points
                n_unpaired++;
                partner[i] = partner[j] = -1;
                continue;
            }

            q_list.push_back(m_j / m_i);
            if (m_i > 5.0) q_massive.push_back(m_j / m_i);
            a_list.push_back(a_bin);
            e_list.push_back(e_bin);

            // Isotropic orientation
            double inc = std::acos(2.0 * uni(rng) - 1.0);
            double Omega = 2.0 * M_PI * uni(rng);
            double omega = 2.0 * M_PI * uni(rng);
            double M_anom = 2.0 * M_PI * uni(rng);

            double r_rel[3], v_rel[3];
            elements_to_cartesian(a_bin, e_bin, inc, Omega, omega, M_anom, mu, r_rel, v_rel);

            // Centre of mass = particle i's Plummer phase-space point: the binary is sampled from f(E) as one
            // object (BIFROST App. A; mcluster, Kupper et al. 2011). The mean of both members' independent draws
            // (BINARY_COM_V = average, the old default) shrinks the c.o.m. velocity dispersion by
            // sqrt(m_i^2 + m_j^2) / M_bin (1/sqrt(2) for q = 1): binaries start kinematically cold.
            double cx = hx[i], cy = hy[i], cz = hz[i];
            double cvx = hvx[i], cvy = hvy[i], cvz = hvz[i];
            if (!binary_com_sampled) {
                cvx = (m_i * hvx[i] + m_j * hvx[j]) / M_bin;
                cvy = (m_i * hvy[i] + m_j * hvy[j]) / M_bin;
                cvz = (m_i * hvz[i] + m_j * hvz[j]) / M_bin;
            }

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

        if (binary_peri_check)
            std::printf("    pericentre check (r_p > contact distance): e redrawn for %d, a redrawn for %d, "
                        "%d pairs left single (a_max too small)\n", n_e_redrawn, n_a_redrawn, n_unpaired);
        else
            std::printf("    BINARY_PERI_CHECK=false: no pericentre condition (contact binaries allowed)\n");
        if (n_unpaired > 0 && r_break_au > 0.0)
            std::printf("    WARNING: the %d secondaries left single were not in the IC_R_BREAK pass\n", n_unpaired);
        // Distribution summary (deciles of q, log10 a [AU], e)
        auto deciles = [](std::vector<double> v, const char* name, double scale, bool log10v) {
            if (v.empty()) return;
            std::sort(v.begin(), v.end());
            std::printf("    %s deciles:", name);
            for (int k = 0; k <= 10; k++) {
                double x = v[std::min((size_t)k * (v.size() - 1) / 10, v.size() - 1)] * scale;
                std::printf(" %.3g", log10v ? std::log10(x) : x);
            }
            std::printf("\n");
        };
        deciles(q_list, "q", 1.0, false);
        deciles(q_massive, "q (m1 > 5 M_sun)", 1.0, false);
        deciles(a_list, "log10(a/AU)", 1.0 / AU_TO_PC, true);
        deciles(e_list, "e", 1.0, false);

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

    print_ic_check(partner, hx, hy, hz, hvx, hvy, hvz, hmass, a);

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
    double v_apo = e < 1.0 ? std::sqrt(mu / a_pc * (1.0 - e) / (1.0 + e + 1e-30)) : 0.0;

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
    if (e >= 1.0) {
        // Parabolic or hyperbolic encounter: a_pc is the pericentre distance q; start inbound at r0 = 5 q (true
        // anomaly f < 0), relative state r = r0 (cos f, sin f), v = sqrt(mu / p) (-sin f, e + cos f), p = q (1 + e)
        const double q = a_pc, p_semi = q * (1.0 + e), r0 = 5.0 * q;
        const double f = -std::acos(std::max(-1.0, std::min(1.0, (p_semi / r0 - 1.0) / e)));
        const double vs = std::sqrt(mu / p_semi);
        const double rr[2] = {r0 * std::cos(f), r0 * std::sin(f)};
        const double vv[2] = {-vs * std::sin(f), vs * (e + std::cos(f))};
        h_x(0) = -f1 * rr[0]; h_y(0) = -f1 * rr[1]; h_vx(0) = -f1 * vv[0]; h_vy(0) = -f1 * vv[1];
        h_x(1) =  f2 * rr[0]; h_y(1) =  f2 * rr[1]; h_vx(1) =  f2 * vv[0]; h_vy(1) =  f2 * vv[1];
    }
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
                      double i_deg, bool outer_first, double in_rot_deg) {
    p.allocate(3, COLL_ENABLED);

    double M_in  = m1_msun + m2_msun;
    double M_tot = M_in + m3_msun;
    double mu_in  = G_CODE * M_in;
    double mu_out = G_CODE * M_tot;
    double i_rad  = i_deg * M_PI / 180.0;

    // --- Inner binary at apoapsis (in x-y plane) ---
    double r_apo_in = a_in_pc * (1.0 + e_in);
    double v_apo_in = e_in < 1.0 ? std::sqrt(mu_in / a_in_pc * (1.0 - e_in) / (1.0 + e_in + 1e-30)) : 0.0;
    double f1 = m2_msun / M_in;   // particle 0 fraction
    double f2 = m1_msun / M_in;   // particle 1 fraction

    // Inner binary positions/velocities relative to inner COM
    // Apoapsis along +x, velocity along +y (CCW orbit, L along +z)
    double x0 =  f1 * r_apo_in,  vy0 =  f1 * v_apo_in;
    double x1 = -f2 * r_apo_in,  vy1 = -f2 * v_apo_in;
    // TRIPLE_IN_ROT_DEG: inner orbit rotated about z by this angle (apoapsis direction, member velocities)
    const double cr = std::cos(in_rot_deg * M_PI / 180.0), sr = std::sin(in_rot_deg * M_PI / 180.0);
    double y0 = sr * x0, y1 = sr * x1, vx0 = -sr * vy0, vx1 = -sr * vy1;
    x0 *= cr; x1 *= cr; vy0 *= cr; vy1 *= cr;
    if (e_in >= 1.0) {
        // Parabolic / hyperbolic inner encounter (as init_binary_test): a_in_pc is the pericentre distance q,
        // start inbound at 5 q; relative vector r_1 - r_0 = r0 (cos f, sin f)
        const double q = a_in_pc, p_semi = q * (1.0 + e_in), r0 = 5.0 * q;
        const double f = -std::acos(std::max(-1.0, std::min(1.0, (p_semi / r0 - 1.0) / e_in)));
        const double vs = std::sqrt(mu_in / p_semi);
        const double rr[2] = {r0 * std::cos(f), r0 * std::sin(f)};
        const double vv[2] = {-vs * std::sin(f), vs * (e_in + std::cos(f))};
        x0 = -f1 * rr[0]; y0 = -f1 * rr[1]; vx0 = -f1 * vv[0]; vy0 = -f1 * vv[1];
        x1 =  f2 * rr[0]; y1 =  f2 * rr[1]; vx1 =  f2 * vv[0]; vy1 =  f2 * vv[1];
    }

    // --- Outer orbit at apoapsis ---
    // Outer angular momentum direction: L_out = (0, sin(i), cos(i))
    // At apoapsis: position along +x, velocity along (0, cos(i), -sin(i))
    // Outer relative position and velocity (m3 relative to inner COM) in the plane of x and
    // w = (0, cos i, -sin i). e_out > 1: hyperbolic fly-by with periapsis a_out_pc, starting inbound at 5
    // periapsis distances (true anomaly f < 0)
    double r3_x, r3_y, r3_z, v3_x, v3_y, v3_z;
    {
        double rx, rw, vx_, vw;
        if (e_out >= 1.0) {
            const double q = a_out_pc, p_semi = q * (1.0 + e_out), r0 = 5.0 * q;
            const double f = -std::acos(std::max(-1.0, std::min(1.0, (p_semi / r0 - 1.0) / e_out)));
            const double vs = std::sqrt(mu_out / p_semi);
            rx = r0 * std::cos(f);  rw = r0 * std::sin(f);
            vx_ = -vs * std::sin(f); vw = vs * (e_out + std::cos(f));
        } else {
            rx = a_out_pc * (1.0 + e_out);  rw = 0.0;
            vx_ = 0.0;  vw = std::sqrt(mu_out / a_out_pc * (1.0 - e_out) / (1.0 + e_out + 1e-30));
        }
        r3_x = rx;  r3_y = rw * std::cos(i_rad);  r3_z = -rw * std::sin(i_rad);
        v3_x = vx_; v3_y = vw * std::cos(i_rad);  v3_z = -vw * std::sin(i_rad);
    }

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
    const double py[3] = {y0 + f_in * r3_y, y1 + f_in * r3_y, f_out * r3_y};
    const double pz[3] = {f_in * r3_z, f_in * r3_z, f_out * r3_z};
    const double pvx[3] = {vx0 + f_in * v3_x, vx1 + f_in * v3_x, f_out * v3_x};
    const double pvy[3] = {vy0 + f_in * v3_y, vy1 + f_in * v3_y, f_out * v3_y};
    const double pvz[3] = {f_in * v3_z, f_in * v3_z, f_out * v3_z};
    const double pm[3] = {m1_msun, m2_msun, m3_msun};
    for (int k = 0; k < 3; k++) {
        const int i = outer_first ? (k + 1) % 3 : k;
        h_x(i) = px[k];   h_y(i) = py[k];   h_z(i) = pz[k];
        h_vx(i) = pvx[k]; h_vy(i) = pvy[k]; h_vz(i) = pvz[k];
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
