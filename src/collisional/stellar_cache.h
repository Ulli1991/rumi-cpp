#pragma once

// Host copy of every particle's stellar type and radius, by particle id (subsystem audit #4). The collision
// distances of secular candidates (advance_subsystems_in_range) and of LogH members (SubsystemBatch::pack) need
// them; gathering them from the device per call cost ~40 us of latency each (+0.4-0.6 s per step at 200k since
// 043190d). Types and radii change only through runtime SSE, mergers (execute_merger, finalize_subsystem_collision)
// and a checkpoint restore, which call invalidate(); the next use downloads the arrays once. Index permutations,
// compaction and escaper removal leave the pid -> value map unchanged. NBODY_CHECK_STELLAR_CACHE=1 compares every
// lookup batch with a device gather and aborts on any difference.

#include "particles.h"
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct StellarPropsCache {
    std::vector<int> stype;       // by particle id; -1: no particle with this id at the last refresh
    std::vector<double> radius;   // R_sun
    bool valid = false;
    long long refreshes = 0;

    static StellarPropsCache& get() {
        static StellarPropsCache c;
        return c;
    }
    static bool check_enabled() {
        static const bool e = [] { const char* v = std::getenv("NBODY_CHECK_STELLAR_CACHE"); return v && v[0] && v[0] != '0'; }();
        return e;
    }
    void invalidate() { valid = false; }
    void ensure(const Particles& p) {
        if (valid) return;
        auto h_pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.particle_id);
        auto h_st = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_type);
        auto h_sr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_radius);
        int max_pid = -1;
        for (int i = 0; i < p.N; i++) max_pid = std::max(max_pid, h_pid(i));
        stype.assign(max_pid + 1, -1);
        radius.assign(max_pid + 1, 0.0);
        for (int i = 0; i < p.N; i++) {
            stype[h_pid(i)] = h_st(i);
            radius[h_pid(i)] = h_sr(i);
        }
        valid = true;
        refreshes++;
    }
    bool has(int pid) const { return pid >= 0 && pid < (int)stype.size() && stype[pid] >= 0; }
    // Stellar type of a particle, refreshing the cache when it does not know the id (-1: no such particle)
    int type_of(const Particles& p, int pid) {
        ensure(p);
        if (!has(pid)) { invalidate(); ensure(p); }
        return has(pid) ? stype[pid] : -1;
    }
    int type_of(int pid) const { return stype[pid]; }
    double radius_of(int pid) const { return radius[pid]; }

    // NBODY_CHECK_STELLAR_CACHE: the cached values of these particles (device indices idx, ids pid) against the device
    void verify(const Particles& p, const std::vector<int>& idx, const std::vector<int>& pid, const char* where) const {
        if (idx.empty()) return;
        auto h_st = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_type);
        auto h_sr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.stellar_radius);
        for (size_t k = 0; k < idx.size(); k++) {
            if (has(pid[k]) && type_of(pid[k]) == h_st(idx[k]) && radius_of(pid[k]) == h_sr(idx[k])) continue;
            std::fprintf(stderr, "NBODY_CHECK_STELLAR_CACHE (%s): pid %d idx %d cached kw=%d R=%.17g device kw=%d R=%.17g\n",
                         where, pid[k], idx[k], has(pid[k]) ? type_of(pid[k]) : -1, has(pid[k]) ? radius_of(pid[k]) : -1.0,
                         h_st(idx[k]), h_sr(idx[k]));
            std::fflush(stderr);
            std::abort();
        }
    }
};
