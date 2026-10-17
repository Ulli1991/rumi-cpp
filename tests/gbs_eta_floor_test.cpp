// GBS error estimate against its round-off floor (ctest gbs_eta_floor).
// Monte Carlo arithmetic (tests/vfc, Verificarlo random rounding) of one orbit of an e = 0.999 binary: the error
// estimate of gbs_extrapolate has no significant digit at err ~ 5e-14 (Bulirsch sequence) and ~4e-13 (Deuflhard, 8
// stages); it is the extrapolation-amplified round-off of the stage results (Lebesgue constant of the Neville weights:
// 9 for Bulirsch, 119 / 553 for Deuflhard with 8 / 10 stages) times (1 + e) / (1 - e) at the pericentre. An eta_gbs
// below that floor is never met by convergence, only by chance: the step size collapses (one orbit took 1.16e6 steps
// instead of 77 at ETA_GBS = 1e-15) and the energy error grows (1.4e-10 instead of 3e-12), with no message.
// Requirement: one orbit at eta = 1e-15 takes at most 10x the steps of eta = 1e-10, for both sequences, and its energy
// error is not worse than 3x that of eta = 1e-10 or 1e-11, whichever is larger.
#include "logh_step.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

static const int SEQ_B[] = {2, 4, 6, 8, 12, 16, 24, 32, 48, 64};
static const int SEQ_D[] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20};

struct Run { long steps; double dE; };

static Run orbit(const int* seq, double eta) {
    LogHState st{};
    st.n_bodies = 2;
    st.mass[0] = 0.6; st.mass[1] = 0.4;
    const double e = 0.999, p = 1.0 - e * e, r = p / (1.0 - e), vs = std::sqrt(1.0 / p);   // apocentre, G M = a = 1
    const double rel[3] = {-r, 0.0, 0.0}, relv[3] = {0.0, vs * (e - 1.0), 0.0};
    for (int k = 0; k < 3; k++) {
        st.pos[k] = -0.4 * rel[k];  st.pos[3 + k] = 0.6 * rel[k];
        st.vel[k] = -0.4 * relv[k]; st.vel[3 + k] = 0.6 * relv[k];
    }
    const double E0 = logh_total_energy(st, 1.0);
    st.B = -E0;
    GbsParams par;
    par.seq = seq;
    par.k_max = 10;
    par.eta = eta;
    static GbsTable tab;
    const double s_end = 2.0 * M_PI * 0.24;   // one orbit: <-U> = m1 m2 / a
    double s = 0.0, ds = s_end / 64.0;
    long steps = 0;
    while (s < s_end && steps < 2000000) {
        LogHStepStart start;
        logh_step_start(st, 1.0, 0.0, 0, start);
        GbsStepOut o;
        const double h = std::min(ds, s_end - s);
        gbs_step(par, st, h, tab, -1, &start, o);
        steps++;
        if (o.accepted) s += h;
        ds = o.ds_next;
        if (!(ds > 1e-14 * s_end)) break;
    }
    return {steps, std::fabs(logh_total_energy(st, 1.0) - E0) / std::fabs(E0)};
}

int main() {
    int fails = 0;
    for (int q = 0; q < 2; q++) {
        const int* seq = q ? SEQ_D : SEQ_B;
        const Run ref = orbit(seq, 1e-10), tight = orbit(seq, 1e-15);
        const bool ok = tight.steps <= 10 * ref.steps && tight.dE <= 3.0 * std::max(ref.dE, 1e-11);
        std::printf("%s: eta 1e-10 %ld steps |dE/E| %.2e;  eta 1e-15 %ld steps |dE/E| %.2e  %s\n",
                    q ? "deuflhard" : "bulirsch ", ref.steps, ref.dE, tight.steps, tight.dE, ok ? "ok" : "FAIL");
        if (!ok) fails++;
    }
    std::printf(fails ? "FAIL\n" : "PASS\n");
    return fails;
}
