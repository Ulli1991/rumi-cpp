#include "gw_output.h"
#include "particles.h"
#include "subsystem.h"
#include "kepler.h"
#include "mergers.h"
#include "constants.h"
#include "mpi_comm.h"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <unordered_map>

GwLog g_gw_log;

namespace {

constexpr double TU_S = PC_TO_CM / 1e5;   // code time unit (pc / (km/s)) in s
constexpr double AU_PC = 4.84813681109536e-6;

const char* kind_str(int k) { return k == 0 ? "MERGER" : "BINARY"; }
const char* path_str(int p) {
    static const char* s[] = {"subsys", "contact", "peters", "secular", "census"};
    return p >= 0 && p <= 4 ? s[p] : "?";
}
const char* type_str(int t) {
    static const char* s[] = {"GW_COAL", "ISCO", "TDE", "STELLAR"};
    return t >= 0 && t <= 3 ? s[t] : "-";
}
const char* channel_str(int c) {
    static const char* s[] = {"2body", "fewbody", "capture", "ejected", "cluster"};
    return c >= 0 && c <= 4 ? s[c] : "?";
}

void write_header(FILE* f) {
    std::fprintf(f,
        "# RUMI++ gravitational-wave table (src/collisional/gw_output.h, docs/CASES.md E2). One row per compact\n"
        "# (NS kw 13 / BH kw 14) merger at the moment the code decides it (kind MERGER) and per bound compact pair in a\n"
        "# subsystem with t_gw < GW_CENSUS_TGW_MYR at every snapshot (kind BINARY). a, e: Newtonian osculating elements\n"
        "# of the relative orbit; f_gw: peak GW frequency (Wen 2003); e_10Hz, e_10mHz: e on the orbit-averaged Peters\n"
        "# (1964) track through (a, e) where f_gw = 10 Hz / 1e-2 Hz (-1: f_gw above it on the whole track); t_gw: Peters\n"
        "# merger time (Mandel 2021 fit); chi_eff, chi_p (Schmidt+2015) with L_hat = r x v; remnant columns NaN in\n"
        "# census rows. path: subsys (LogH/GBS/IRK collision at pericentre, PN), contact (ISCO at a step boundary),\n"
        "# peters (t_gw < dt), secular (t_gw < dt of a secular binary), census. channel: ejected (pair c.o.m. unbound\n"
        "# from the cluster, point-mass potential), capture (e >= 1 at the decision), fewbody (subsystem of >= 3),\n"
        "# 2body, cluster (census, bound to the cluster). tidal: max_k (m_k / d_k^3) / (M / r^3) over all other bodies.\n"
        "# r_cl, v_cl: pair c.o.m. relative to the cluster c.o.m. Units: Msun, AU, Hz, Myr, km/s, pc.\n"
        "# row kind t_myr path type channel pid1 pid2 kw1 kw2 m1 m2 chi1x chi1y chi1z chi2x chi2y chi2z a_au e f_orb_hz"
        " f_gw_hz e_10Hz e_10mHz t_gw_myr chi_eff chi_p m_rem chi_rem_x chi_rem_y chi_rem_z v_kick_kms n_members tidal"
        " r_cl_pc v_cl_kms\n");
}

void write_row(FILE* f, long long id, const GwRow& r) {
    std::fprintf(f, "%lld %s %.15e %s %s %s %d %d %d %d %.10e %.10e %.6e %.6e %.6e %.6e %.6e %.6e %.10e %.10e %.10e"
                    " %.10e %.10e %.10e %.10e %.6e %.6e %.10e %.6e %.6e %.6e %.6e %d %.6e %.10e %.10e\n",
                 id, kind_str(r.kind), r.t * TIME_UNIT_MYR, path_str(r.path), type_str(r.type), channel_str(r.channel),
                 r.pid1, r.pid2, r.kw1, r.kw2, r.m1, r.m2, r.chi1[0], r.chi1[1], r.chi1[2], r.chi2[0], r.chi2[1],
                 r.chi2[2], r.a / AU_PC, r.e, r.f_orb, r.f_gw, r.e_10hz, r.e_10mhz, r.t_gw * TIME_UNIT_MYR, r.chi_eff,
                 r.chi_p, r.m_rem, r.chi_rem[0], r.chi_rem[1], r.chi_rem[2], r.v_kick, r.n_members, r.tidal, r.r_cl,
                 r.v_cl);
}

}  // namespace

void GwLog::open(bool resume) {
    if (!on || MPIComm::rank() != 0) return;
    if (resume) {
        // Keep the header and the first n_written rows (rows after the restart point are made again)
        std::ifstream in(path);
        std::string text, line;
        long long rows = 0;
        while (in && std::getline(in, line)) {
            if (!line.empty() && line[0] != '#') {
                if (rows >= n_written) break;
                rows++;
            }
            text += line + "\n";
        }
        in.close();
        if (rows < n_written)
            std::printf("GW: WARNING: %s has %lld of the %lld rows of the restart point\n", path.c_str(), rows,
                        n_written);
        FILE* f = std::fopen(path.c_str(), "w");
        if (!f) return;
        if (text.empty()) write_header(f);
        else std::fputs(text.c_str(), f);
        std::fclose(f);
        return;
    }
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) { std::fprintf(stderr, "WARNING: cannot open %s for writing\n", path.c_str()); return; }
    write_header(f);
    std::fclose(f);
}

void GwLog::flush() {
    if (!on || pending.empty()) return;
    if (MPIComm::rank() == 0) {
        FILE* f = std::fopen(path.c_str(), "a");
        if (f) {
            for (size_t k = 0; k < pending.size(); k++) write_row(f, n_written + (long long)k, pending[k]);
            std::fclose(f);
        } else {
            std::fprintf(stderr, "WARNING: cannot append to %s\n", path.c_str());
        }
    }
    n_written += (long long)pending.size();
    pending.clear();
}

double gw_f_peak_hz(double a, double e, double M, double G) {
    const double rp = a * (1.0 - e);   // also for e > 1 (a < 0)
    if (!(rp > 0.0) || !(M > 0.0)) return 0.0;
    return std::sqrt(G * M) * std::pow(1.0 + e, 1.1954) / (M_PI * std::pow(rp, 1.5)) / TU_S;
}

double gw_e_at_f(double a, double e, double M, double G, double f_hz) {
    if (!(M > 0.0) || !(f_hz > 0.0)) return -1.0;
    if (!(e > 0.0)) return 0.0;   // circular stays circular
    constexpr double B = 121.0 / 304.0, P = 870.0 / 2299.0;
    double c0;
    if (e < 1.0) {
        if (!(a > 0.0)) return -1.0;
        c0 = a * (1.0 - e * e) * std::pow(e, -12.0 / 19.0) * std::pow(1.0 + B * e * e, -P);
    } else {
        const double rp = a * (1.0 - e);
        if (!(rp > 0.0)) return -1.0;
        c0 = 2.0 * rp / std::pow(1.0 + B, P);   // track with the same pericentre at e -> 1
    }
    // f_GW along the track, rp(e) = c0 e^(12/19) (1 + B e^2)^P / (1 + e): falls from infinity (e -> 0) to its minimum
    // at e* = 0.74971 and rises again by 1.2 % to e = 1. The root taken is the one on e < e*, where f_GW rises
    // monotonically to the merger (the last entry into the band); f below the minimum: never reached (-1)
    constexpr double E_STAR = 0.7497112117388973;
    const double K = std::sqrt(G * M) / M_PI / TU_S;
    auto f_of = [&](double ee) {
        const double rp = c0 * std::pow(ee, 12.0 / 19.0) * std::pow(1.0 + B * ee * ee, P) / (1.0 + ee);
        return K * std::pow(1.0 + ee, 1.1954) / std::pow(rp, 1.5);
    };
    if (f_hz <= f_of(E_STAR)) return -1.0;
    double lo = std::log(1e-300), hi = std::log(E_STAR);   // ln e; f(lo) > f_hz > f(hi)
    for (int it = 0; it < 200 && hi - lo > 1e-17; it++) {
        const double mid = 0.5 * (lo + hi);
        if (f_of(std::exp(mid)) > f_hz) lo = mid; else hi = mid;
    }
    return std::exp(0.5 * (lo + hi));
}

double gw_t_merge(double a, double e, double m1, double m2, double G, double c) {
    if (!(a > 0.0) || !(e < 1.0) || !(c > 0.0)) return std::numeric_limits<double>::infinity();
    const double beta = 64.0 / 5.0 * G * G * G * m1 * m2 * (m1 + m2) / std::pow(c, 5);
    const double Tc = a * a * a * a / (4.0 * beta);
    const double e2 = e * e;
    return Tc * (1.0 + 0.27 * std::pow(e, 10) + 0.33 * std::pow(e, 20) + 0.2 * std::pow(e, 1000)) *
           std::pow(1.0 - e2, 3.5);
}

void gw_chi_eff_p(double m1, double m2, const double chi1[3], const double chi2[3], const double L[3],
                  double& chi_eff, double& chi_p) {
    double Lh[3] = {0.0, 0.0, 1.0};
    const double Ln = std::sqrt(L[0] * L[0] + L[1] * L[1] + L[2] * L[2]);
    if (Ln > 0.0) for (int k = 0; k < 3; k++) Lh[k] = L[k] / Ln;
    const double c1 = chi1[0] * Lh[0] + chi1[1] * Lh[1] + chi1[2] * Lh[2];
    const double c2 = chi2[0] * Lh[0] + chi2[1] * Lh[1] + chi2[2] * Lh[2];
    chi_eff = (m1 * c1 + m2 * c2) / (m1 + m2);
    auto perp = [&](const double* ch, double cp) {
        double s = 0.0;
        for (int k = 0; k < 3; k++) { const double d = ch[k] - cp * Lh[k]; s += d * d; }
        return std::sqrt(s);
    };
    double p1 = perp(chi1, c1), p2 = perp(chi2, c2), ma = m1, mb = m2;
    if (mb > ma) { std::swap(p1, p2); std::swap(ma, mb); }
    const double q = mb / ma;
    chi_p = std::max(p1, (4.0 * q + 3.0) / (4.0 + 3.0 * q) * q * p2);
}

void gw_fill_pair(GwRow& row, const double r[3], const double v[3], double G, double c) {
    const double M = row.m1 + row.m2;
    const OrbitalElements oe = cartesian_to_elements(r, v, G * M);
    row.a = oe.a;
    row.e = oe.e;
    row.f_orb = (oe.a > 0.0 && oe.e < 1.0) ? std::sqrt(G * M / (oe.a * oe.a * oe.a)) / (2.0 * M_PI) / TU_S : 0.0;
    row.f_gw = gw_f_peak_hz(oe.a, oe.e, M, G);
    row.e_10hz = gw_e_at_f(oe.a, oe.e, M, G, 10.0);
    row.e_10mhz = gw_e_at_f(oe.a, oe.e, M, G, 1e-2);
    row.t_gw = gw_t_merge(oe.a, oe.e, row.m1, row.m2, G, c);
    const double L[3] = {r[1] * v[2] - r[2] * v[1], r[2] * v[0] - r[0] * v[2], r[0] * v[1] - r[1] * v[0]};
    gw_chi_eff_p(row.m1, row.m2, row.chi1, row.chi2, L, row.chi_eff, row.chi_p);
}

void gw_environment(GwRow& row, int N, const double* x, const double* y, const double* z, const double* vx,
                    const double* vy, const double* vz, const double* m, int skip1, int skip2, const double xc[3],
                    const double vc[3], double r_sep, double G) {
    double M_tot = 0.0, cx = 0.0, cy = 0.0, cz = 0.0, cvx = 0.0, cvy = 0.0, cvz = 0.0, tid = 0.0;
    const double M = row.m1 + row.m2;
    for (int k = 0; k < N; k++) {
        if (!(m[k] > 0.0)) continue;
        M_tot += m[k];
        cx += m[k] * x[k]; cy += m[k] * y[k]; cz += m[k] * z[k];
        cvx += m[k] * vx[k]; cvy += m[k] * vy[k]; cvz += m[k] * vz[k];
        if (k == skip1 || k == skip2) continue;
        const double dx = x[k] - xc[0], dy = y[k] - xc[1], dz = z[k] - xc[2];
        const double d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > 0.0) tid = std::max(tid, m[k] / (d2 * std::sqrt(d2)));
    }
    row.tidal = M > 0.0 ? tid * r_sep * r_sep * r_sep / M : 0.0;
    if (!(M_tot > 0.0)) return;
    const double dr[3] = {xc[0] - cx / M_tot, xc[1] - cy / M_tot, xc[2] - cz / M_tot};
    const double dv[3] = {vc[0] - cvx / M_tot, vc[1] - cvy / M_tot, vc[2] - cvz / M_tot};
    row.r_cl = std::sqrt(dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]);
    row.v_cl = std::sqrt(dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]);
    const double rv = dr[0] * dv[0] + dr[1] * dv[1] + dr[2] * dv[2];
    // Unbound from the cluster in the point-mass potential (deeper than the real one: a sufficient condition)
    const bool ejected = row.r_cl > 0.0 && rv > 0.0 && 0.5 * row.v_cl * row.v_cl - G * M_tot / row.r_cl > 0.0;
    if (row.kind == 1) row.channel = ejected ? GW_CH_EJECTED : GW_CH_CLUSTER;
    else if (ejected) row.channel = GW_CH_EJECTED;
    else if (row.e >= 1.0) row.channel = GW_CH_CAPTURE;
    else row.channel = row.n_members >= 3 ? GW_CH_FEWBODY : GW_CH_2BODY;
}

void gw_census(const Particles& p, const SubsystemManager& mgr, double G, double c, double t) {
    if (!g_gw_log.on || !(g_gw_log.census_tgw > 0.0) || !(c > 0.0) || mgr.active.empty() || !p.collisional) return;
    auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
    auto h_y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
    auto h_z = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
    auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
    auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
    auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);
    auto h_m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
    auto h_kw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_type);
    auto h_pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);
    auto h_spin = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.spin_vec);
    const int N = p.N;
    std::unordered_map<int, int> idx_of;
    for (int i = 0; i < N; i++) idx_of[h_pid(i)] = i;
    for (const auto& sub : mgr.active) {
        // compact members; each with its nearest compact member of the subsystem; mutual nearest pairs only
        std::vector<int> mem;
        for (int pid : sub.particle_ids) {
            auto it = idx_of.find(pid);
            if (it == idx_of.end()) continue;
            const int i = it->second;
            if (h_m(i) > 0.0 && gw_compact_pair(h_kw(i), h_kw(i))) mem.push_back(i);
        }
        if (mem.size() < 2) continue;
        auto d2 = [&](int i, int j) {
            const double dx = h_x(j) - h_x(i), dy = h_y(j) - h_y(i), dz = h_z(j) - h_z(i);
            return dx * dx + dy * dy + dz * dz;
        };
        auto nearest = [&](int i) {
            int b = -1;
            for (int j : mem) if (j != i && (b < 0 || d2(i, j) < d2(i, b))) b = j;
            return b;
        };
        for (int i : mem) {
            const int j = nearest(i);
            if (j < 0 || nearest(j) != i || h_pid(i) > h_pid(j)) continue;   // each mutual pair once
            GwRow row;
            row.kind = 1; row.t = t; row.path = GW_PATH_CENSUS; row.type = -1;
            row.pid1 = h_pid(i); row.pid2 = h_pid(j); row.kw1 = h_kw(i); row.kw2 = h_kw(j);
            row.m1 = h_m(i); row.m2 = h_m(j);
            for (int k = 0; k < 3; k++) { row.chi1[k] = h_spin(i, k); row.chi2[k] = h_spin(j, k); }
            const double r[3] = {h_x(j) - h_x(i), h_y(j) - h_y(i), h_z(j) - h_z(i)};
            const double v[3] = {h_vx(j) - h_vx(i), h_vy(j) - h_vy(i), h_vz(j) - h_vz(i)};
            gw_fill_pair(row, r, v, G, c);
            if (!(row.e < 1.0) || !(row.t_gw < g_gw_log.census_tgw)) continue;
            const double nan = std::numeric_limits<double>::quiet_NaN();
            row.m_rem = nan; row.v_kick = nan;
            for (double& x : row.chi_rem) x = nan;
            row.n_members = sub.n_bodies;
            const double M = row.m1 + row.m2;
            const double xc[3] = {(row.m1 * h_x(i) + row.m2 * h_x(j)) / M, (row.m1 * h_y(i) + row.m2 * h_y(j)) / M,
                                  (row.m1 * h_z(i) + row.m2 * h_z(j)) / M};
            const double vc[3] = {(row.m1 * h_vx(i) + row.m2 * h_vx(j)) / M,
                                  (row.m1 * h_vy(i) + row.m2 * h_vy(j)) / M,
                                  (row.m1 * h_vz(i) + row.m2 * h_vz(j)) / M};
            gw_environment(row, N, h_x.data(), h_y.data(), h_z.data(), h_vx.data(), h_vy.data(), h_vz.data(),
                           h_m.data(), i, j, xc, vc, std::sqrt(d2(i, j)), G);
            g_gw_log.pending.push_back(row);
        }
    }
}
