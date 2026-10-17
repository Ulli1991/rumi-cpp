#pragma once
// Stall capture (STALL_FACTOR, STALL_DE_FACTOR, STALL_RESTART): what made a step slow or put an energy jump into
// it, for cutting the case out of the run (scripts/cutout.py) and keeping it as an isolated test. During each step
// the recorder keeps the costliest LogH integrations (GBS steps, members), the LogH integrations with the largest
// energy error (|dE| / |E0| of the run, members) and the first set at the deepest HHS level (members); the main
// loop prints them when the step's wall time exceeds STALL_FACTOR x, or its energy change STALL_DE_FACTOR x, the
// median of the previous steps, and with STALL_RESTART keeps the restart file written before the step.
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
    std::vector<LogH> logh_err;            // gbs field: unused; dt_myr field: |dE| / |E0| of the run
    // binaries with the largest tidal-sampling error estimate mu |f_tidal| v dt of a step (|E0| units; the field
    // kicks sample the perturber's tide on the binary at the binary's step)
    struct Tidal { double est = 0.0, a = 0.0, e = 0.0, gamma = 0.0, P_myr = 0.0, dt_myr = 0.0; int pid0 = -1, pid1 = -1; double m0 = 0.0, m1 = 0.0; };
    std::vector<Tidal> tidal;
    int deep_depth = -1;
    double deep_pivot_myr = 0.0;
    std::vector<int> deep_pids;

    void reset() {
        std::lock_guard<std::mutex> lk(mu);
        logh.clear();
        logh_err.clear();
        tidal.clear();
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
    // one LogH integration's energy error |dE| / |E0| of the run (kept: the MAX_LOGH largest)
    void note_logh_err(double err, const int* pids, const double* mass, int n) {
        std::lock_guard<std::mutex> lk(mu);
        if ((int)logh_err.size() >= MAX_LOGH && err <= logh_err.back().dt_myr) return;
        LogH r;
        r.dt_myr = err;
        r.pids.assign(pids, pids + n);
        r.mass.assign(mass, mass + n);
        logh_err.insert(std::upper_bound(logh_err.begin(), logh_err.end(), r,
                                         [](const LogH& a, const LogH& b) { return a.dt_myr > b.dt_myr; }), std::move(r));
        if ((int)logh_err.size() > MAX_LOGH) logh_err.pop_back();
    }
    void note_tidal(const Tidal& t) {
        if (tidal.size() >= MAX_LOGH && t.est <= tidal.back().est) return;   // unlocked pre-check (serial caller)
        std::lock_guard<std::mutex> lk(mu);
        for (auto& o : tidal)   // one entry per binary: its largest estimate
            if (o.pid0 == t.pid0 && o.pid1 == t.pid1) {
                if (t.est > o.est) o = t;
                std::sort(tidal.begin(), tidal.end(), [](const Tidal& a, const Tidal& b) { return a.est > b.est; });
                return;
            }
        tidal.insert(std::upper_bound(tidal.begin(), tidal.end(), t,
                                      [](const Tidal& a, const Tidal& b) { return a.est > b.est; }), t);
        if (tidal.size() > MAX_LOGH) tidal.pop_back();
    }
    bool deeper(int depth) const { return depth > deep_depth; }
    void note_deep(int depth, double pivot_myr, const std::vector<int>& pids) {
        deep_depth = depth;
        deep_pivot_myr = pivot_myr;
        deep_pids.assign(pids.begin(), pids.begin() + std::min<size_t>(pids.size(), MAX_DEEP));
    }
    // header: "STALL step N: ..." (slow step) or "STALL energy step N: ..." (energy jump)
    void print(const char* header) const {
        std::printf("%s\n", header);
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
        for (const auto& t : tidal)
            std::printf("STALL tidal est/E0 %.3e pids %d %d m[Msun] %.4g %.4g a %.3e pc e %.4f gamma %.3e P %.3e dt %.3e Myr\n",
                        t.est, t.pid0, t.pid1, t.m0, t.m1, t.a, t.e, t.gamma, t.P_myr, t.dt_myr);
        for (const auto& r : logh_err) {
            std::printf("STALL loghE dE/E0 %.3e pids", r.dt_myr);
            for (int pid : r.pids) std::printf(" %d", pid);
            std::printf(" m[Msun]");
            for (double m : r.mass) std::printf(" %.4g", m);
            std::printf("\n");
        }
        std::fflush(stdout);
    }
};

inline StallRecorder g_stall;
