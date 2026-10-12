#include "snapshot.h"
#include "h5_util.h"
#include "restart.h"
#include "constants.h"
#ifdef ENABLE_COLLISIONAL
#include "subsystem.h"
#include "tidal_field.h"
#endif
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <unordered_map>

namespace fs = std::filesystem;

namespace {

// Direct-sum potential of every particle (own View: the particle state is not touched)
Kokkos::View<double*> potential(const Particles& p, double G, double soft2) {
    const int N = p.N;
    Kokkos::View<double*> phi("snap_phi", N);
    auto x = p.x, y = p.y, z = p.z, m = p.mass;
    Kokkos::parallel_for("snapshot_potential", N, KOKKOS_LAMBDA(int i) {
        double s = 0.0;
        const double xi = x(i), yi = y(i), zi = z(i);
        for (int j = 0; j < N; j++) {
            const double dx = x(j) - xi, dy = y(j) - yi, dz = z(j) - zi;
            const double r2 = dx * dx + dy * dy + dz * dz + soft2;
            if (j != i && r2 > 0.0) s -= m(j) / Kokkos::sqrt(r2);
        }
        phi(i) = G * s;
    });
    return phi;
}

template <class V>
auto host(const V& v) { return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v); }

// LUNA's units group (checkpoint_utils._write_units, full = True) for pc, M_sun, km/s
void write_units(hid_t f) {
    auto u = h5::create_group(f, "units");
    const double L = PC_TO_CM, M = MSUN_IN_G, V = 1e5, T = L / V;
    h5::attr(u, "UnitLength_in_cm", L);
    h5::attr(u, "UnitMass_in_g", M);
    h5::attr(u, "UnitVelocity_in_cm_per_s", V);
    h5::attr(u, "UnitTime_in_s", T);
    h5::attr(u, "UnitDensity_in_cgs", M / (L * L * L));
    h5::attr(u, "UnitPressure_in_cgs", M / (L * T * T));
    h5::attr(u, "UnitEnergy_in_cgs", M * L * L / (T * T));
}

// LUNA's grid group (write_snapshot_distributed.py) with one placeholder cell: read_snapshot() requires it
void write_cells(hid_t f, const SnapshotMeta& m, double box, double M, const double vcm[3]) {
    auto c = h5::create_group(f, "cells");
    h5::attr(c, "time", m.t);
    h5::attr(c, "step", m.step);
    h5::attr(c, "process_id", 0);
    h5::attr(c, "process_count", 1);
    h5::attr(c, "shard_idx", 0);
    h5::attr(c, "gamma", 5.0 / 3.0);
    h5::attr(c, "spatial_dimension", 3);
    h5::attr(c, "order", 0);
    h5::attr(c, "limiter", 0);
    h5::attr(c, "config_json", m.config_json);
    for (const char* k : {"Lx", "Ly", "Lz", "dx", "dy", "dz"}) h5::attr(c, k, box);
    for (const char* k : {"global_Nx", "Nx_global", "global_Ny", "Ny_global", "global_Nz", "Nz_global",
                          "local_Nx", "local_Ny", "local_Nz"}) h5::attr(c, k, 1);
    for (const char* k : {"x_start", "y_start", "z_start"}) h5::attr(c, k, 0);
    h5::attr(c, "placeholder", 1);
    h5::attr(c, "placeholder_note", "cpp-nbody is a particle code: one cell spanning the box, density = total mass / "
                                    "BoxSize^3, velocity = centre-of-mass velocity, no gas (pressure = 0)");
    const double rho = box > 0 ? M / (box * box * box) : 0.0;
    // [1,1,1] arrays as LUNA writes them for a 3-d grid
    auto cube = [&](const char* name, double v) {
        hsize_t dims[3] = {1, 1, 1};
        h5::Handle sp(H5Screate_simple(3, dims, nullptr), H5Sclose);
        h5::Handle ds(H5Dcreate2(c, name, H5T_IEEE_F64LE, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose);
        if (ds < 0 || H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v) < 0) h5::fail(name);
    };
    cube("density", rho);
    cube("velocity_x", vcm[0]);
    cube("velocity_y", vcm[1]);
    cube("velocity_z", vcm[2]);
    cube("internal_energy", 0.0);
    cube("pressure", 0.0);
    // conserved state U[5,1,1,1]: rho, rho v, E (kinetic only)
    const double v2 = vcm[0] * vcm[0] + vcm[1] * vcm[1] + vcm[2] * vcm[2];
    const double U[5] = {rho, rho * vcm[0], rho * vcm[1], rho * vcm[2], 0.5 * rho * v2};
    hsize_t dims[4] = {5, 1, 1, 1};
    h5::Handle sp(H5Screate_simple(4, dims, nullptr), H5Sclose);
    h5::Handle ds(H5Dcreate2(c, "U", H5T_IEEE_F64LE, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose);
    if (ds < 0 || H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, U) < 0) h5::fail("U");
}

}  // namespace

std::string write_snapshot(const SnapshotOptions& opt, const Particles& p, const SnapshotMeta& m) {
    const int N = p.N;
    char sub[32], name[256];
    std::snprintf(sub, sizeof sub, "snapdir_%03d", m.index);
    std::snprintf(name, sizeof name, "%s_%03d.0.hdf5", opt.base.c_str(), m.index);
    const fs::path dir = fs::path(opt.dir.empty() ? "." : opt.dir) / sub;
    std::error_code ec;
    fs::create_directories(dir, ec);
    const std::string path = (dir / name).string(), tmp = path + ".partial";

    auto hx = host(p.x), hy = host(p.y), hz = host(p.z), hvx = host(p.vx), hvy = host(p.vy), hvz = host(p.vz);
    auto hm = host(p.mass), hdt = host(p.dt_adapt);
    auto hlev = host(p.level), horig = host(p.orig_index);
    auto hphi = host(opt.potential ? potential(p, m.G, m.soft2) : Kokkos::View<double*>("snap_phi", 0));

    // Particle types: 5 = black holes, 4 = everything else
    std::vector<int> type(N, 4);
    std::vector<uint64_t> ids(N);
    for (int i = 0; i < N; i++) ids[i] = (uint64_t)horig(i);
#ifdef ENABLE_COLLISIONAL
    // (unallocated in collisionless runs: empty mirrors)
    auto hpid = host(p.particle_id), hsid = host(p.subsystem_id), hst = host(p.stellar_type);
    auto hrad = host(p.stellar_radius), hlum = host(p.luminosity), hcm = host(p.core_mass);
    auto hmz = host(p.m_zams), hage = host(p.stellar_age);
    auto hspin = host(p.spin_vec);
    std::vector<int64_t> partner(N, -1);
    if (p.collisional) {
        std::unordered_map<int, int> idx;
        idx.reserve(2 * N);
        for (int i = 0; i < N; i++) {
            ids[i] = (uint64_t)hpid(i);
            if (hst(i) == 14) type[i] = 5;
            idx[hpid(i)] = i;
        }
        // Most bound partner inside each subsystem
        for (const auto& s : g_subsystem_mgr.active) {
            std::vector<int> mem;
            for (int pid : s.particle_ids) { auto it = idx.find(pid); if (it != idx.end()) mem.push_back(it->second); }
            for (int a : mem) {
                double best = 0.0;
                for (int b : mem) {
                    if (a == b) continue;
                    const double dx = hx(b) - hx(a), dy = hy(b) - hy(a), dz = hz(b) - hz(a);
                    const double dvx = hvx(b) - hvx(a), dvy = hvy(b) - hvy(a), dvz = hvz(b) - hvz(a);
                    const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (r <= 0.0) continue;
                    const double E = 0.5 * (dvx * dvx + dvy * dvy + dvz * dvz) - m.G * (hm(a) + hm(b)) / r;
                    if (E < best) { best = E; partner[a] = hpid(b); }
                }
            }
        }
    }
#endif

    // Box, total mass, centre-of-mass velocity
    double box = 0.0, M = 0.0, vcm[3] = {0, 0, 0};
    for (int i = 0; i < N; i++) {
        box = std::max({box, std::fabs(hx(i)), std::fabs(hy(i)), std::fabs(hz(i))});
        M += hm(i);
        vcm[0] += hm(i) * hvx(i); vcm[1] += hm(i) * hvy(i); vcm[2] += hm(i) * hvz(i);
    }
    box *= 2.0;
    if (M > 0) for (double& v : vcm) v /= M;

    {
        auto f = h5::create_file(tmp);
        std::vector<long long> npart(6, 0);
        for (int i = 0; i < N; i++) npart[type[i]]++;

        auto H = h5::create_group(f, "Header");
        h5::attr(H, "BoxSize", box);
        h5::attr(H, "Time", m.t);
        h5::attr(H, "Redshift", 0.0);
        h5::attr(H, "Omega0", 0.0);
        h5::attr(H, "OmegaLambda", 0.0);
        h5::attr(H, "HubbleParam", 1.0);
        h5::attr(H, "NumPart_ThisFile", npart);
        h5::attr(H, "NumPart_Total", npart);
        h5::attr(H, "NumPart_Total_HighWord", std::vector<long long>(6, 0));
        h5::attr(H, "NumFilesPerSnapshot", 1);
        h5::attr(H, "MassTable", std::vector<double>(6, 0.0));
        h5::attr(H, "Flag_DoublePrecision", 1);
        h5::attr(H, "DoublePrecision", 1);
        h5::attr(H, "config_json", m.config_json);
        h5::attr(H, "Code", "cpp-nbody");
        h5::attr(H, "CodeVersion", code_version());
        h5::attr(H, "Time_Myr", m.t * TIME_UNIT_MYR);
        h5::attr(H, "Step", m.step);
        h5::attr(H, "SnapshotIndex", m.index);
        h5::attr(H, "N", N);
        h5::attr(H, "G", m.G);
        h5::attr(H, "Softening_pc", std::sqrt(m.soft2));
        h5::attr(H, "UnitLength_in_cm", PC_TO_CM);
        h5::attr(H, "UnitMass_in_g", MSUN_IN_G);
        h5::attr(H, "UnitVelocity_in_cm_per_s", 1e5);
        h5::attr(H, "E_kin", m.diag.KE);
        h5::attr(H, "E_pot", m.diag.PE);
        h5::attr(H, "E_total", m.diag.E_total);
        h5::attr(H, "E_total_initial", m.diag0.E_total);
        double sink = 0.0;
#ifdef ENABLE_COLLISIONAL
        const EnergyBookkeeping& b = g_energy_books;
        sink = b.total_sink();
        h5::attr(H, "E_escape", b.E_escape); h5::attr(H, "E_gw", b.E_gw); h5::attr(H, "E_merge", b.E_merge);
        h5::attr(H, "E_sse", b.E_sse); h5::attr(H, "E_pn", b.E_pn); h5::attr(H, "E_sd", b.E_sd);
        h5::attr(H, "E_tidal", b.E_tidal);
        h5::attr(H, "P_sink", std::vector<double>(b.P_sink, b.P_sink + 3));
        h5::attr(H, "L_sink", std::vector<double>(b.L_sink, b.L_sink + 3));
#endif
        if (m.has_cluster) {
            const ClusterDiag& d = m.cluster;
            h5::attr(H, "Cluster_MassTotal", d.M_total);
            h5::attr(H, "Cluster_DensityCentre", std::vector<double>(d.center, d.center + 3));
            h5::attr(H, "Cluster_DensityCentreVelocity", std::vector<double>(d.vcenter, d.vcenter + 3));
            h5::attr(H, "Cluster_CoreRadius_pc", d.r_core);
            h5::attr(H, "Cluster_CoreDensity", d.rho_core);
            h5::attr(H, "Cluster_Sigma1D_kms", d.sigma_c);
            h5::attr(H, "Cluster_LagrangianFractions",
                     std::vector<double>(ClusterDiag::LAG_FRAC, ClusterDiag::LAG_FRAC + ClusterDiag::NLAG));
            h5::attr(H, "Cluster_LagrangianRadii_pc", std::vector<double>(d.r_lag, d.r_lag + ClusterDiag::NLAG));
            h5::attr(H, "Cluster_NBinaries", d.n_bin);
            h5::attr(H, "Cluster_NBinariesHard", d.n_bin_hard);
            h5::attr(H, "Cluster_EBinaries", d.E_bin);
        }
        if (g_tidal.on()) {
            h5::attr(H, "Tidal_Field", g_tidal.type == TidalField::POINT_MASS ? "pointmass" : "loghalo");
            h5::attr(H, "Tidal_RefPosition", std::vector<double>(g_tidal.X, g_tidal.X + 3));
            h5::attr(H, "Tidal_RefVelocity", std::vector<double>(g_tidal.V, g_tidal.V + 3));
            double rj = 0.0;
            if (m.has_cluster) {
                const double Xg[3] = {g_tidal.X[0] + m.cluster.center[0], g_tidal.X[1] + m.cluster.center[1],
                                      g_tidal.X[2] + m.cluster.center[2]};
                rj = g_tidal.jacobi_radius(Xg, m.cluster.M_total, m.G);
            }
            h5::attr(H, "Tidal_JacobiRadius_pc", rj);
        }
        h5::attr(H, "dE_rel", m.diag0.E_total != 0.0
                                  ? (m.diag.E_total + sink - m.diag0.E_total) / std::fabs(m.diag0.E_total) : 0.0);

        for (int t : {4, 5}) {
            if (npart[t] == 0) continue;
            std::vector<int> sel;
            for (int i = 0; i < N; i++) if (type[i] == t) sel.push_back(i);
            const size_t n = sel.size();
            auto g = h5::create_group(f, "PartType" + std::to_string(t));
            auto vec3 = [&](const char* nm, auto& a, auto& b2, auto& c) {
                std::vector<double> v(3 * n);
                for (size_t k = 0; k < n; k++) { v[3 * k] = a(sel[k]); v[3 * k + 1] = b2(sel[k]); v[3 * k + 2] = c(sel[k]); }
                h5::write(g, nm, v, 3);
            };
            auto col = [&](const char* nm, auto&& get) {
                using T = std::decay_t<decltype(get(0))>;
                std::vector<T> v(n);
                for (size_t k = 0; k < n; k++) v[k] = get(sel[k]);
                h5::write(g, nm, v);
            };
            vec3("Coordinates", hx, hy, hz);
            vec3("Velocities", hvx, hvy, hvz);
            col("Masses", [&](int i) { return hm(i); });
            col("ParticleIDs", [&](int i) { return ids[i]; });
            if (opt.potential) col("Potential", [&](int i) { return hphi(i); });
            col("TimeStep", [&](int i) { return hdt(i); });
            col("TimeBin", [&](int i) { return (int)hlev(i); });
#ifdef ENABLE_COLLISIONAL
            if (p.collisional) {
                col("StellarType", [&](int i) { return (int)hst(i); });
                col("StellarRadius", [&](int i) { return hrad(i); });
                col("Luminosity", [&](int i) { return hlum(i); });
                col("CoreMass", [&](int i) { return hcm(i); });
                col("ZAMSMass", [&](int i) { return hmz(i); });
                col("StellarAge", [&](int i) { return hage(i); });
                std::vector<double> s(3 * n);
                for (size_t k = 0; k < n; k++) for (int c = 0; c < 3; c++) s[3 * k + c] = hspin(sel[k], c);
                h5::write(g, "Spin", s, 3);
                col("SubsystemID", [&](int i) { return (int)hsid(i); });
                col("BinaryPartnerID", [&](int i) { return partner[i]; });
            }
#endif
        }
        write_cells(f, m, box, M, vcm);
        write_units(f);
    }
    fs::rename(tmp, path);
    return path;
}
