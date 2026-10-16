#pragma once
// Stall capture (STALL_FACTOR, STALL_RESTART): what made a step slow, for cutting the case out of the run
// (scripts/cutout.py) and keeping it as an isolated test. During each step the recorder keeps the costliest LogH
// integrations (GBS steps, members) and the first set at the deepest HHS level (members); the main loop prints both
// when the step's wall time exceeds STALL_FACTOR x the median of the previous steps, and with STALL_RESTART keeps
// the restart file written before the step.
#include <algorithm>
#include <cstdio>
#include <mutex>
#include <vector>

struct StallRecorder {
    static constexpr int MAX_LOGH = 8;     // costliest LogH integrations kept per step
    static constexpr int MAX_DEEP = 32;    // members of the deepest set kept
    struct LogH {
        long gbs = 0;
        double dt_myr = 0.0;
        std::vector<int> pids;
        std::vector<double> mass;
    };
    std::mutex mu;
    std::vector<LogH> logh;
    int deep_depth = -1;
    double deep_pivot_myr = 0.0;
    std::vector<int> deep_pids;

    void reset() {
        std::lock_guard<std::mutex> lk(mu);
        logh.clear();
        deep_depth = -1;
        deep_pivot_myr = 0.0;
        deep_pids.clear();
    }
    // one LogH integration (host threads: locked; only integrations above the current 8th place are stored)
    void note_logh(long gbs, double dt_myr, const int* pids, const double* mass, int n) {
        std::lock_guard<std::mutex> lk(mu);
        if ((int)logh.size() >= MAX_LOGH && gbs <= logh.back().gbs) return;
        LogH r;
        r.gbs = gbs;
        r.dt_myr = dt_myr;
        r.pids.assign(pids, pids + n);
        r.mass.assign(mass, mass + n);
        logh.insert(std::upper_bound(logh.begin(), logh.end(), r,
                                     [](const LogH& a, const LogH& b) { return a.gbs > b.gbs; }), std::move(r));
        if ((int)logh.size() > MAX_LOGH) logh.pop_back();
    }
    bool deeper(int depth) const { return depth > deep_depth; }
    void note_deep(int depth, double pivot_myr, const std::vector<int>& pids) {
        deep_depth = depth;
        deep_pivot_myr = pivot_myr;
        deep_pids.assign(pids.begin(), pids.begin() + std::min<size_t>(pids.size(), MAX_DEEP));
    }
    void print(int step, double wall, double median) const {
        std::printf("STALL step %d: %.3f s (median of the previous steps %.3f s)\n", step, wall, median);
        std::printf("STALL deepest depth %d pivot %.3e Myr pids", deep_depth, deep_pivot_myr);
        for (int pid : deep_pids) std::printf(" %d", pid);
        std::printf("\n");
        for (const auto& r : logh) {
            std::printf("STALL logh gbs %ld dt %.3e Myr pids", r.gbs, r.dt_myr);
            for (int pid : r.pids) std::printf(" %d", pid);
            std::printf(" m[Msun]");
            for (double m : r.mass) std::printf(" %.4g", m);
            std::printf("\n");
        }
        std::fflush(stdout);
    }
};

inline StallRecorder g_stall;
