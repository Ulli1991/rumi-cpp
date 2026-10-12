#pragma once
// NBODY_FENCE_STATS=1 (gpu-idle prototype): count Kokkos fences (by name), deep copies (by direction) and kernel
// launches per step through the Kokkos Tools callbacks; rank 0 prints the table after every step.
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <execinfo.h>

namespace fence_stats {
struct Stats {
    std::map<std::string, long long> fences;
    long long dc_h2d = 0, dc_d2h = 0, dc_d2d = 0, dc_h2h = 0;
    long long pfor = 0, pred = 0, pscan = 0;
    std::map<std::string, long long> sites;   // NBODY_FENCE_STATS=2: fence name + raw return addresses
};
inline Stats& S() { static Stats s; return s; }
inline bool& on() { static bool b = false; return b; }
inline int& mode() { static int m = 0; return m; }

inline void begin_fence(const char* name, uint32_t, uint64_t*) {
    S().fences[name]++;
    if (mode() >= 2 && std::strncmp(name, "Kokkos::Tools::invoke", 21) != 0) {
        void* bt[10];
        int n = backtrace(bt, 10);
        char key[512];
        int len = std::snprintf(key, sizeof key, "%.40s |", name);
        for (int k = 2; k < n && len < 480; k++) len += std::snprintf(key + len, sizeof key - len, " %p", bt[k]);
        S().sites[key]++;
    }
}
inline void end_fence(uint64_t) {}
inline void begin_pfor(const char*, uint32_t, uint64_t*) { S().pfor++; }
inline void begin_pred(const char*, uint32_t, uint64_t*) { S().pred++; }
inline void begin_pscan(const char*, uint32_t, uint64_t*) { S().pscan++; }
inline void end_k(uint64_t) {}
inline void begin_dc(Kokkos::Tools::SpaceHandle dst, const char*, const void*, Kokkos::Tools::SpaceHandle src,
                     const char*, const void*, uint64_t) {
    bool dh = std::strcmp(dst.name, "Host") == 0, sh = std::strcmp(src.name, "Host") == 0;
    if (dh && sh) S().dc_h2h++;
    else if (dh) S().dc_d2h++;
    else if (sh) S().dc_h2d++;
    else S().dc_d2d++;
}

inline void init() {
    const char* e = std::getenv("NBODY_FENCE_STATS");
    if (!e || !e[0] || e[0] == '0') return;
    on() = true;
    mode() = std::atoi(e);
    namespace KT = Kokkos::Tools::Experimental;
    KT::set_begin_fence_callback(begin_fence);
    KT::set_end_fence_callback(end_fence);
    KT::set_begin_parallel_for_callback(begin_pfor);
    KT::set_end_parallel_for_callback(end_k);
    KT::set_begin_parallel_reduce_callback(begin_pred);
    KT::set_end_parallel_reduce_callback(end_k);
    KT::set_begin_parallel_scan_callback(begin_pscan);
    KT::set_end_parallel_scan_callback(end_k);
    KT::set_begin_deep_copy_callback(begin_dc);
}

inline void report(int step, bool print) {
    if (!on()) return;
    auto& s = S();
    if (print) {
        long long tot = 0;
        std::vector<std::pair<long long, std::string>> v;
        for (auto& kv : s.fences) { tot += kv.second; v.push_back({kv.second, kv.first}); }
        std::sort(v.rbegin(), v.rend());
        std::printf("FENCE_STATS step %d: fences %lld, deep_copy h2d %lld d2h %lld d2d %lld h2h %lld, "
                    "parallel_for %lld reduce %lld scan %lld\n", step, tot, s.dc_h2d, s.dc_d2h, s.dc_d2d, s.dc_h2h,
                    s.pfor, s.pred, s.pscan);
        for (size_t i = 0; i < v.size() && i < 25; i++)
            std::printf("FENCE_STATS   %9lld  %s\n", v[i].first, v[i].second.c_str());
        if (mode() >= 2) {
            std::vector<std::pair<long long, std::string>> w;
            for (auto& kv : s.sites) w.push_back({kv.second, kv.first});
            std::sort(w.rbegin(), w.rend());
            for (size_t i = 0; i < w.size() && i < 600; i++)
                std::printf("FENCE_SITE %9lld %s\n", w[i].first, w[i].second.c_str());
        }
        std::fflush(stdout);
    }
    s = Stats();
}
}  // namespace fence_stats
