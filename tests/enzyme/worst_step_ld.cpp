// Long-double (x87: 64-bit mantissa) build of logh.cpp + logh_step.cpp for tests/enzyme/worst_step.cpp: the
// reference of a LogH + GBS step without double round-off. In double, a reference of many sub-steps carries
// ~1e-10 of round-off when a tight binary sits far from the c.o.m. (its separation is a difference of large
// coordinates), more than the error of the few-substep step it should judge.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
namespace std {   // std::max(1.0, x) and the like with a double literal and a long double x
inline long double max(double a, long double b) { return a > b ? (long double)a : b; }
inline long double max(long double a, double b) { return a > b ? a : (long double)b; }
inline long double min(double a, long double b) { return a < b ? (long double)a : b; }
inline long double min(long double a, double b) { return a < b ? a : (long double)b; }
}  // namespace std
#pragma clang diagnostic ignored "-Wformat"   // the %a of a debug print in logh.cpp
namespace ld {   // the double build in worst_step.cpp defines the same names
#define double long double
#include "../../src/collisional/logh.cpp"
#include "../../src/collisional/logh_step.cpp"
#include "../../src/collisional/post_newtonian_core.cpp"
#undef double
}  // namespace ld
using namespace ld;

// Newtonian 3-body state (G = 1, t = 0, B) over ds in nsub sub-steps of k fixed stages (Bulirsch sequence);
// y: the packed result (pos, vel, t, B) rounded to double
void gbs_reference_ld(const double* pos, const double* vel, const double* mass, double B, double ds, int nsub, int k,
                      double* y) {
    static const int SEQ[] = {2, 4, 6, 8, 12, 16, 24, 32, 48, 64};
    LogHState s{};
    s.n_bodies = 3;
    for (int i = 0; i < 9; i++) { s.pos[i] = pos[i]; s.vel[i] = vel[i]; }
    for (int i = 0; i < 3; i++) s.mass[i] = mass[i];
    s.B = B;
    GbsParams par;
    par.seq = SEQ;
    par.k_fixed = k;
    GbsTable tab;
    for (int i = 0; i < nsub; i++) {
        LogHStepStart start;
        logh_step_start(s, 1.0L, 0.0L, 0, start);
        GbsStepOut out;
        gbs_step(par, s, (long double)ds / nsub, tab, -1, &start, out);
    }
    long double yl[GBS_DIM_MAX];
    int dim;
    gbs_pack_state(s, yl, dim);
    for (int d = 0; d < dim; d++) y[d] = (double)yl[d];
}
