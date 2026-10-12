// Snapshot round trip (CTest snapshot_roundtrip): reads a snapshot and the restart file of the same state and
// checks that every particle of the snapshot carries, bit for bit, the restart file's position, velocity, mass,
// stellar data and spin; that the particle types are complete; and that the LUNA groups are there.
//   h5-roundtrip <snapshot.hdf5> <restart.h5>        exit 0: identical, 1: mismatch, 2: unreadable
#include "h5_util.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s snapshot.hdf5 restart.h5\n", argv[0]); return 2; }
    try {
        auto snap = h5::open_file(argv[1]);
        auto rst = h5::open_file(argv[2]);
        auto rp = h5::open_group(rst, "particles");
        const long long N = h5::get_int(rp, "N");
        const bool coll = h5::get_int(rp, "collisional") != 0;
        std::vector<double> x, y, z, vx, vy, vz, m, rad, spin;
        std::vector<int> pid, st, orig;
        h5::read(rp, "x", x); h5::read(rp, "y", y); h5::read(rp, "z", z);
        h5::read(rp, "vx", vx); h5::read(rp, "vy", vy); h5::read(rp, "vz", vz); h5::read(rp, "mass", m);
        h5::read(rp, "orig_index", orig);
        if (coll) {
            h5::read(rp, "particle_id", pid); h5::read(rp, "stellar_type", st);
            h5::read(rp, "stellar_radius", rad); h5::read(rp, "spin_vec", spin);
        }
        std::unordered_map<long long, long long> row;
        for (long long i = 0; i < N; i++) row[coll ? pid[i] : orig[i]] = i;

        int bad = 0;
        auto H = h5::open_group(snap, "Header");
        std::vector<double> dummy;
        long long seen = 0;
        const double t_snap = h5::get_double(H, "Time"), t_rst = h5::get_double(rst, "time");
        if (std::memcmp(&t_snap, &t_rst, sizeof(double)) != 0) { std::printf("time differs\n"); bad++; }
        for (int t : {0, 1, 2, 3, 4, 5}) {
            const std::string g = "PartType" + std::to_string(t);
            if (!h5::exists(snap, g)) continue;
            auto P = h5::open_group(snap, g);
            std::vector<double> c, v, mm, r, s;
            std::vector<uint64_t> ids;
            std::vector<int> sst;
            const size_t n = h5::read(P, "Coordinates", c);
            h5::read(P, "Velocities", v); h5::read(P, "Masses", mm); h5::read(P, "ParticleIDs", ids);
            if (coll) { h5::read(P, "StellarType", sst); h5::read(P, "StellarRadius", r); h5::read(P, "Spin", s); }
            for (size_t k = 0; k < n; k++) {
                auto it = row.find((long long)ids[k]);
                if (it == row.end()) { if (bad++ < 5) std::printf("id %llu not in the restart file\n", (unsigned long long)ids[k]); continue; }
                const long long i = it->second;
                const double a[7] = {c[3 * k], c[3 * k + 1], c[3 * k + 2], v[3 * k], v[3 * k + 1], v[3 * k + 2], mm[k]};
                const double b[7] = {x[i], y[i], z[i], vx[i], vy[i], vz[i], m[i]};
                bool same = std::memcmp(a, b, sizeof a) == 0;
                if (coll) {
                    same &= sst[k] == st[i] && std::memcmp(&r[k], &rad[i], sizeof(double)) == 0 &&
                            std::memcmp(&s[3 * k], &spin[3 * i], 3 * sizeof(double)) == 0;
                    same &= (t == 5) == (st[i] == 14);
                }
                if (!same && bad++ < 5) std::printf("particle %llu differs\n", (unsigned long long)ids[k]);
            }
            seen += (long long)n;
        }
        if (seen != N) { std::printf("snapshot holds %lld of %lld particles\n", seen, N); bad++; }
        auto np = h5::get_doubles(H, "MassTable");
        if (np.size() != 6) { std::printf("MassTable has %zu entries\n", np.size()); bad++; }
        for (const char* g : {"cells", "units"})
            if (!h5::exists(snap, g)) { std::printf("missing group %s\n", g); bad++; }
        if (h5::exists(snap, "cells")) {
            auto C = h5::open_group(snap, "cells");
            for (const char* a : {"time", "step", "gamma", "spatial_dimension", "Lx", "dx", "Nx_global", "Ly", "dy",
                                  "Ny_global", "Lz", "dz", "Nz_global", "x_start", "y_start", "z_start"})
                if (!h5::has_attr(C, a)) { std::printf("cells lacks attribute %s\n", a); bad++; }
            for (const char* d : {"density", "velocity_x", "velocity_y", "velocity_z", "pressure", "internal_energy"})
                if (!h5::exists(C, d)) { std::printf("cells lacks dataset %s\n", d); bad++; }
        }
        // Header structure attributes (snapshot_cluster_header): one line each, full precision
        for (const char* a : {"Cluster_MassTotal", "Cluster_DensityCentre", "Cluster_CoreRadius_pc", "Cluster_CoreDensity",
                              "Cluster_Sigma1D_kms", "Cluster_LagrangianRadii_pc", "Cluster_NBinaries",
                              "Cluster_NBinariesHard", "Cluster_EBinaries", "Tidal_RefPosition", "Tidal_RefVelocity",
                              "Tidal_JacobiRadius_pc", "E_tidal"}) {
            if (!h5::has_attr(H, a)) continue;
            std::printf("header %s", a);
            for (double v : h5::get_doubles(H, a)) std::printf(" %.17g", v);
            std::printf("\n");
        }
        std::printf("round trip: %lld particles compared, %d mismatches\n", seen, bad);
        return bad ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 2;
    }
}
