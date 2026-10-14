// IC stellar evolution of very massive stars (src/ic_gen/sse.cpp), BIFROST Fig. 9 settings (Vink winds, Fryer 2012
// rapid, strong PPISN, Z = 2e-4, 1 Gyr):
//  - 100 Msun: BH of 41.6 Msun (BIFROST Fig. 9: most massive BH 41.6 Msun)
//  - 101-130 Msun: BHs, at most the 45 Msun of strong pulsational pair instability (Belczynski et al. 2016; Banerjee
//    et al. 2020). Hurley's clamp at 100 Msun left them in core helium burning (kw 4, 103-123 Msun, R = 1350 Rsun)
//  - 150 Msun: pair-instability supernova, no remnant (kw 15)
#include "sse.h"
#include <cstdio>

int main() {
    int fail = 0;
    auto check = [&](const char* label, double val, double lo, double hi) {
        const bool ok = val >= lo && val <= hi;
        std::printf("%s %s = %.6g (in [%.6g, %.6g])\n", ok ? "ok  " : "FAIL", label, val, lo, hi);
        if (!ok) fail++;
    };
    SSEContext s(0.0002, make_sse_config("VINK", "FRYER_RAPID", "STRONG", "ON", 2.5, 1.5));
    SSEResult r = s.evolve(100.0, 1000.0);
    check("100 Msun: kw", r.kw, 14, 14);
    check("100 Msun: BH mass [Msun]", r.mass, 41.5, 41.7);
    for (double m : {101.0, 110.0, 130.0}) {
        r = s.evolve(m, 1000.0);
        char l[64];
        std::snprintf(l, sizeof l, "%.0f Msun: kw", m);
        check(l, r.kw, 14, 14);
        std::snprintf(l, sizeof l, "%.0f Msun: BH mass [Msun]", m);
        check(l, r.mass, 41.6, 45.0 + 1e-9);
    }
    r = s.evolve(150.0, 1000.0);
    check("150 Msun: kw (no remnant)", r.kw, 15, 15);
    return fail ? 1 : 0;
}
