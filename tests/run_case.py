"""Run one cpp-nbody test case from a small config and check the printed diagnostics against bounds."""
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TIMEOUT_S = int(os.environ.get("NBODY_TEST_TIMEOUT", 120))  # seconds per run on one CPU core; GPUs need more (launch latency)
# FROST Fig. 4 binary: m1 = m2 = 10 Msun, a = 10 AU, ~1000 orbits; diagnostics every 1e-5 Myr
KEPLER = dict(TEST_CASE="binary_test", INTEGRATOR="hhs", TOTAL_TIME_MYR="7.0711e-3", DT_MAX_MYR="1.0e-5",
              DT_MIN_MYR="1.0e-14", SOFTENING_PC="0", COLLISIONAL="false", PRINT_INTERVAL="1")
# FROST Fig. 4 as published: endpoint |E(1000 P) - E(0)| / |E(0)|, the level grid aligned to the period
P_MYR = math.sqrt(1000 / 20) * 1e-6
FROST4 = dict(KEPLER, TOTAL_TIME_MYR=f"{1000 * P_MYR:.10e}", DT_MAX_MYR=f"{P_MYR:.10e}")
# FROST Fig. 5: Sun and the four giant planets (Tables A1-A2), 1000 yr
PLANETS = dict(KEPLER, TEST_CASE="outer_planets", TOTAL_TIME_MYR="1.0e-3")
# 400 Kroupa stars, half in contact binaries (a = 1-4 R_sun): ~75 stellar mergers in 20 steps, at HHS depths
# 0-10. Each merger compacts the particle arrays mid-recursion and may trigger step restarts.
MERGERS = dict(TEST_CASE="plummer", INTEGRATOR="hhs", TOTAL_TIME_MYR="2e-8", DT_MAX_MYR="1e-9",
               DT_MIN_MYR="1e-16", ETA_FF="0.2", ETA_FB="0.2", ETA_GRAD="0.1", ETA_JERK="0.1", SOFTENING_PC="0",
               COLLISIONAL="true", R_NGB="1e-3", R_NGB_MAX="0.1", ETA_PERT="0.02", ETA_GBS="1e-10",
               ETA_RESTART="1e-8", N_ORB_SECULAR="3", M_PN="5.0", PN_ORDER="0", R_ESCAPE_RH="100.0",
               SSE_ENABLED="false", PLUMMER_N="400", PLUMMER_M_TOTAL="200", PLUMMER_R_HALF="0.3", PLUMMER_SEED="7",
               PLUMMER_MASS_DIST="kroupa", BINARY_FRAC="0.5", BINARY_A_MIN_AU="0.005", BINARY_A_MAX_AU="0.02",
               PRINT_INTERVAL="1", PROFILING="false")
# 1e5 Msun BH binary, a = 2e-3 pc > r_ngb (no subsystem), e = 0.5, ~10 orbits with PN1: only the global PN1
# kick (BIFROST sec. 3.8) can precess it, 6.0e-5 rad/orbit
PN1_WIDE = dict(TEST_CASE="binary_test", INTEGRATOR="hhs", BINARY_A_PC="2e-3", BINARY_E="0.5", BINARY_M1="5e4",
                BINARY_M2="5e4", TOTAL_TIME_MYR="2.7e-4", DT_MAX_MYR="1.35e-6", DT_MIN_MYR="1e-16", SOFTENING_PC="0",
                COLLISIONAL="true", R_NGB="1e-3", R_NGB_MAX="1e-3", PN_ORDER="1", M_PN="5.0", SSE_ENABLED="false",
                PRINT_INTERVAL="20", PROFILING="false")
# 200 stars plus a 500+500 Msun BH binary (a = 2e-5 pc) with spins 0.9 along L, all PN orders
SPIN_IMBH = dict(TEST_CASE="plummer", INTEGRATOR="hhs", TOTAL_TIME_MYR="2e-3", DT_MAX_MYR="1e-4", DT_MIN_MYR="1e-16",
                 SOFTENING_PC="0", COLLISIONAL="true", PN_ORDER="31", SSE_ENABLED="false", PLUMMER_N="200",
                 PLUMMER_M_TOTAL="200", PLUMMER_R_HALF="0.5", PLUMMER_SEED="11", PLUMMER_MASS_DIST="equal",
                 ADD_IMBH="true", IMBH_M1="500", IMBH_M2="500", IMBH_A_PC="2e-5", IMBH_E="0.5", IMBH_SPIN="0.9",
                 PRINT_INTERVAL="5", PROFILING="false")
# 2000 Kroupa stars, 10 % primordial binaries (a = 0.1-100 AU, thermal e), two 1e-3 Myr steps
BINARIES = dict(TEST_CASE="plummer", INTEGRATOR="hhs", TOTAL_TIME_MYR="2e-3", DT_MAX_MYR="1e-3", DT_MIN_MYR="1e-14",
                ETA_FF="0.2", ETA_FB="0.2", ETA_GRAD="0.1", ETA_JERK="0.1", SOFTENING_PC="0", COLLISIONAL="true",
                R_NGB="1e-3", R_NGB_MAX="0.1", ETA_PERT="0.02", ETA_GBS="1e-10", ETA_RESTART="1e-8",
                N_ORB_SECULAR="3", PN_ORDER="0", SSE_ENABLED="false", PLUMMER_N="2000", PLUMMER_M_TOTAL="800",
                PLUMMER_R_HALF="1.0", PLUMMER_SEED="5", PLUMMER_MASS_DIST="kroupa", BINARY_FRAC="0.1",
                BINARY_A_MIN_AU="0.1", BINARY_A_MAX_AU="100", PRINT_INTERVAL="1", PROFILING="false")
# 500+500 Msun BH binary, a = 1e-7 pc (v/c ~ 0.02), all PN orders, steps of a fraction of an orbit (LogH)
PN_LOGH = dict(TEST_CASE="binary_test", INTEGRATOR="hhs", BINARY_A_PC="1e-7", BINARY_E="0.5", BINARY_M1="500",
               BINARY_M2="500", TOTAL_TIME_MYR="2e-9", DT_MAX_MYR="2e-10", DT_MIN_MYR="1e-18", SOFTENING_PC="0",
               COLLISIONAL="true", PN_ORDER="31", M_PN="5.0", ETA_RESTART="1e-8", SSE_ENABLED="false",
               PRINT_INTERVAL="1", PROFILING="false")
# 5000 Kroupa stars, 50 % primordial binaries (1250 pairs, more than the initial neighbour buffer of 1000)
BINARIES50 = dict(BINARIES, PLUMMER_N="5000", PLUMMER_M_TOTAL="2000", BINARY_FRAC="0.5", TOTAL_TIME_MYR="1e-3")
# Dense core (r_h = 0.05 pc), 50 % binaries of 1-20 AU: binary-single encounters form triples and multiplets
TRIPLES = dict(BINARIES, PLUMMER_N="2000", PLUMMER_M_TOTAL="800", PLUMMER_R_HALF="0.05", BINARY_FRAC="0.5",
               BINARY_A_MIN_AU="1", BINARY_A_MAX_AU="20", TOTAL_TIME_MYR="2e-4", DT_MAX_MYR="1e-4")
# 16 stars inside r_ngb: one 15-body neighbour group, more than LogH's fixed storage (LOGH_MAX_BODIES = 8)
BIG_GROUP = dict(TEST_CASE="plummer", INTEGRATOR="hhs", TOTAL_TIME_MYR="1e-5", DT_MAX_MYR="1e-6", DT_MIN_MYR="1e-16",
                 SOFTENING_PC="0", COLLISIONAL="true", R_NGB="1e-3", R_NGB_MAX="0.1", PN_ORDER="0", SSE_ENABLED="false",
                 PLUMMER_N="16", PLUMMER_M_TOTAL="16", PLUMMER_R_HALF="2e-4", PLUMMER_SEED="3", PLUMMER_MASS_DIST="equal",
                 PRINT_INTERVAL="1", PROFILING="false")


def run(exe, cfg):
    """L0 diagnostic rows (|dE/E|, |dP|/M, |dL/L|) of one run; raises on a nonzero exit."""
    return diagnostics(run_output(exe, cfg))


def run_output(exe, cfg):
    """stdout of one run; raises on a nonzero exit or a timeout."""
    with tempfile.TemporaryDirectory() as d:
        # NBODY_TEST_EXTRA="KEY=value,KEY2=value" overrides keys of every case (A/B comparisons)
        extra = dict(kv.split("=", 1) for kv in os.environ.get("NBODY_TEST_EXTRA", "").split(",") if "=" in kv)
        Path(d, "config.txt").write_text("".join(f"{k} = {v}\n" for k, v in {**cfg, **extra}.items()))
        try:
            launcher = os.environ.get("NBODY_TEST_LAUNCHER", "").split()  # e.g. "mpirun -n 2"
            out = subprocess.run(launcher + [exe], cwd=d, capture_output=True, text=True, timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise SystemExit(f"FAIL: no result within {TIMEOUT_S} s (hung or stuck in time)")
    if out.returncode != 0:
        raise SystemExit(f"FAIL: exit {out.returncode}\n{out.stdout[-2000:]}{out.stderr[-2000:]}")
    return out.stdout


def diagnostics(stdout):
    rows = [line.split() for line in stdout.splitlines() if re.match(r"^\s+L0\s", line) and len(line.split()) >= 8]
    if not rows:
        raise SystemExit("FAIL: no diagnostics printed")
    return [[float(r[3]), float(r[4]), float(r[5])] for r in rows]


def endpoint(exe, label, base, eta, bound, literature):
    """FROST's measure: the energy error at the last output (t = 1000 P), against our bound and the paper's."""
    rows = run(exe, {**base, "ETA": eta, "ETA_FF": eta, "ETA_FB": eta})
    ok = check(f"{label} eta={eta} |dE/E| at 1000 P", rows[-1][0], bound)
    return ok & check(f"{label} eta={eta} |dE/E| at 1000 P, FROST Fig. 4 upper end", rows[-1][0], literature)


def floor(exe, label, cfg, bounds):
    """Round-off floor at small eta: endpoint and max |dE/E|, max |dP|/M and |dL/L| against bounds."""
    rows = run(exe, cfg)
    vals = (rows[-1][0], max(r[0] for r in rows), max(r[1] for r in rows), max(r[2] for r in rows))
    names = ("|dE/E| at end", "max|dE/E|", "max|dP|/M [km/s]", "max|dL/L|")
    return all([check(f"{label} {n}", v, b) for n, v, b in zip(names, vals, bounds)])


def mergers(exe, label, cfg, bounds):
    """Stellar mergers inside the HHS recursion: the particle count must drop by exactly the recorded mergers
    (no duplicated or lost particles), momentum and energy including the merger sink must be conserved."""
    out = run_output(exe, cfg)
    rows = diagnostics(out)
    n0 = int(cfg["PLUMMER_N"])
    n_merged = int(re.findall(r"Merger tree written to \S+ \((\d+) events\)", out)[-1])
    last_table = out[out.rfind("Level        dt"):]
    n_final = sum(int(m) for m in re.findall(r"^\s+L\d+\s+\S+\s+(\d+)", last_table, re.M))
    ok = check(f"{label} mergers", n_merged, 50, lower=True)
    ok &= check(f"{label} |N_final - (N0 - mergers)|", abs(n_final - (n0 - n_merged)), 0)
    ok &= check(f"{label} max|dE/E| with merger sink", max(r[0] for r in rows), bounds[0])
    ok &= check(f"{label} max|dL/L| with merger sink", max(r[2] for r in rows), bounds[1])
    return ok & check(f"{label} max|dP|/M [km/s]", max(r[1] for r in rows), bounds[1])


def big_group(exe, label, cfg, bounds):
    """A group larger than LogH's storage must stay with the FSI: no crash, P conserved, E bounded."""
    out = run_output(exe, cfg)
    rows = diagnostics(out)
    ok = check(f"{label} groups left to the FSI", out.count("exceeds LOGH_MAX_BODIES"), 1, lower=True)
    ok &= check(f"{label} max|dE/E|", max(r[0] for r in rows), bounds[0])
    return ok & check(f"{label} max|dP|/M [km/s]", max(r[1] for r in rows), bounds[1])


def precession(exe, label, cfg, tol):
    """Periapsis precession measured by the binary tracker against the 1PN rate 6 pi G M / (c^2 a (1 - e^2))."""
    ratio = float(re.findall(r"ratio_raw=(\S+)", run_output(exe, cfg))[-1])
    return check(f"{label} |precession / 1PN - 1|", abs(ratio - 1.0), tol)


def spins(exe, label, cfg, chi):
    """Every subsystem member's |chi| printed by the subsystem diagnostics must equal the IC spin."""
    out = run_output(exe, cfg)
    rows = re.findall(r"^\s+\d+\s+BIN\s+2\s+(?:\S+\s+){5}(\S+)\s+(\S+)", out, re.M)
    worst = max((abs(float(c) - chi) for row in rows for c in row), default=float("inf"))
    return check(f"{label} max | |chi| - {chi} |", worst, 1e-3)


def replicas(exe, label, cfg):
    """Replicated MPI state must agree bitwise every step (the run stops with exit 3 otherwise)."""
    out = run_output(exe, cfg)
    return check(f"{label} replica mismatches", out.count("REPLICA MISMATCH"), 0) & \
        check(f"{label} steps run", len(diagnostics(out)), 20, lower=True)


def binaries(exe, label, cfg, bounds):
    """A cluster with primordial binaries must step at cluster time-steps (the binaries' internal motion is
    LogH's or the secular integrator's), finish in time and conserve P and E."""
    rows = run(exe, cfg)
    ok = check(f"{label} steps", len(rows), 2, lower=True)
    ok &= check(f"{label} max|dE/E|", max(r[0] for r in rows), bounds[0])
    ok &= check(f"{label} max|dL/L|", max(r[2] for r in rows), bounds[1])
    return ok & check(f"{label} max|dP|/M [km/s]", max(r[1] for r in rows), bounds[1])


def pn_books(exe, label, cfg, bounds):
    """A PN subsystem must not run into the energy restart (it cannot pass it) and the Newtonian diagnostics
    must close with the PN sink."""
    out = run_output(exe, cfg)
    rows = diagnostics(out)
    ok = check(f"{label} LogH give-ups", out.count("GAVE UP"), 0)
    ok &= check(f"{label} max|dE/E| with PN sink", max(r[0] for r in rows), bounds[0])
    ok &= check(f"{label} max|dL/L| with PN sink", max(r[2] for r in rows), bounds[0])
    return ok & check(f"{label} max|dP|/M [km/s]", max(r[1] for r in rows), bounds[1])


def sse_mass(exe, label, cfg, bound):
    """Runtime SSE after mergers: a remnant star must evolve as itself, so SSE removes only wind mass."""
    out = run_output(exe, cfg)
    dm = sum(float(v) for v in re.findall(r"SSE: dm_total=(\S+) M_sun", out))
    ok = check(f"{label} mergers", out.count("[MERGER #"), 20, lower=True)
    return ok & check(f"{label} mass removed by SSE [M_sun]", dm, bound)


def all_binaries(exe, label, cfg, n_bin):
    """Every primordial binary must become a subsystem (none dropped by a full neighbour buffer)."""
    out = run_output(exe, cfg)
    found = int(re.findall(r"Types:\s+bin=(\d+)", out)[0])
    rows = diagnostics(out)
    return check(f"{label} binaries detected", found, n_bin, lower=True) & \
        check(f"{label} max|dE/E|", max(r[0] for r in rows), 1e-8)


def cc_vs_plain(exe, label, cfg, ratio):
    """HHS_CC against plain HHS on the same run: max |dE/E| at most `ratio` times the plain one."""
    plain = max(r[0] for r in run(exe, dict(cfg, HHS_CC="false")))
    cc = max(r[0] for r in run(exe, dict(cfg, HHS_CC="true")))
    print(f"   plain {plain:.3e}, HHS_CC {cc:.3e}")
    return check(f"{label} max|dE/E| HHS_CC / plain", cc / plain, ratio)


def ranks_agree(exe, label, cfg, bound):
    """The same run on 1 and 2 ranks: the energy series must agree to round-off (MPI-only paths)."""
    launcher = os.environ.get("NBODY_TEST_LAUNCHER", "")
    os.environ["NBODY_TEST_LAUNCHER"] = launcher.replace("-n 2", "-n 1")
    one = run(exe, cfg)
    os.environ["NBODY_TEST_LAUNCHER"] = launcher
    two = run(exe, cfg)
    ok = check(f"{label} rows (1 vs 2 ranks)", abs(len(one) - len(two)), 0)
    return ok & check(f"{label} max |dE/E(1) - dE/E(2)|", max(abs(a[0] - b[0]) for a, b in zip(one, two)), bound)


def triples(exe, label, cfg, bound):
    """Triples among particles with large ids must be classified (inner binary, outer companion)."""
    out = run_output(exe, cfg)
    n3 = max((int(a) + int(b) for a, b in re.findall(r"tri_s=(\d+)\s+tri_u=(\d+)", out)), default=0)
    rows = diagnostics(out)
    return check(f"{label} triples formed", n3, 1, lower=True) & \
        check(f"{label} max|dE/E|", max(r[0] for r in rows), bound)


def no_fsi_groups(exe, label, cfg, bound):
    """Neighbour chains in a dense core must be split along bound pairs, not left whole to the FSI."""
    out = run_output(exe, cfg)
    rows = diagnostics(out)
    return check(f"{label} groups left to the FSI", out.count("exceeds LOGH_MAX_BODIES"), 0) & \
        check(f"{label} max|dE/E|", max(r[0] for r in rows), bound)


def check(name, value, bound, lower=False):
    ok = value >= bound if lower else value <= bound
    print(f"{'ok  ' if ok else 'FAIL'} {name} = {value:.3e} ({'>=' if lower else '<='} {bound:.3e})")
    return ok


def sweep(exe, label, base, etas, bounds, order, drift_max=None):
    """max |dE/E| per eta and its fitted order, optionally a bounded (non-drifting) energy error, P and L at
    round-off."""
    ok, maxima = True, []
    for eta, bound in zip(etas, bounds):
        rows = run(exe, {**base, "ETA": eta, "ETA_FF": eta, "ETA_FB": eta})
        dE = [r[0] for r in rows]
        n = len(dE)
        maxima.append(max(dE))
        ok &= check(f"{label} eta={eta} max|dE/E|", max(dE), bound)
        if drift_max is not None:
            ok &= check(f"{label} eta={eta} drift (last / first fifth)", max(dE[4 * n // 5:]) / max(dE[: n // 5]),
                        drift_max)
        ok &= check(f"{label} eta={eta} max|dP|/M [km/s]", max(r[1] for r in rows), 1e-14)
        ok &= check(f"{label} eta={eta} max|dL/L|", max(r[2] for r in rows), 1e-12)
    if order is not None:
        x, y = [math.log(float(v)) for v in etas], [math.log(v) for v in maxima]
        mx, my = sum(x) / len(x), sum(y) / len(y)
        slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x)
        ok &= check(f"{label} order of max|dE/E| in eta", slope, order[0], lower=True)
        ok &= check(f"{label} order of max|dE/E| in eta", slope, order[1])
    return ok


CASES = {
    # bounds ~1.3x measured with ALL_SLOW_SPLIT (FROST Alg. 3; default): 3.24e-7, 8.98e-9, 5.24e-10, order
    # 4.64 -0.1/+0.3 (without the split 5.6e-6, 1.2e-7, 8.3e-9, order 4.70); drift 1.1-1.2 (5.0 before the
    # symmetrised-step fix, which also gave 4.6e-3 at eta = 0.064)
    "kepler_e09": lambda exe: sweep(exe, "e=0.9", dict(KEPLER, BINARY_E="0.9"), ["0.064", "0.032", "0.016"],
                                    [4.2e-7, 1.2e-8, 6.8e-10], (4.54, 4.94), 1.5),
    # circular orbit: the eta^4 term cancels (constant along the orbit), the error sits at round-off
    "kepler_e0": lambda exe: sweep(exe, "e=0", dict(KEPLER, BINARY_E="0.0"), ["0.064"], [1e-12], None),
    # FROST: "minimum relative energy error 1e-12 - 1e-10 at eta ~ 0.008" for e = 0.9, 0.99. Measured with the
    # split 1.76e-13 and 1.44e-11 (gcc/x86-64; e = 0.9 now below FROST's range); without it 1.43e-12 / 6.90e-11
    # (2.37e-12 / 5.09e-11 clang/arm64). GPU builds need 1/sqrt or RSQRT_NEWTON for e = 0.99 (rsqrt: 1.7e-10).
    # e = 0.9 on the MI300A (RSQRT_NEWTON): 1.52e-12 (the endpoint samples an oscillating error); bound 1.3x that
    "frost4_e09": lambda exe: endpoint(exe, "e=0.9", dict(FROST4, BINARY_E="0.9"), "0.008", 2.0e-12, 1e-10),
    "frost4_e099": lambda exe: endpoint(exe, "e=0.99", dict(FROST4, BINARY_E="0.99"), "0.008", 1.9e-11, 1e-10),
    # Through the CPU small-subset path, which kept the pre-fix FROST Eq. 49 (no eta): 1.5e-6 before, 9.1e-13
    # after (1.33e-12 with the split)
    "frost4_e09_cpu": lambda exe: endpoint(exe, "e=0.9 cpu path", dict(FROST4, BINARY_E="0.9", SERIAL_N2_CPU="8"),
                                           "0.008", 1.8e-12, 1e-10),
    # Round-off floor with compensated kicks and integer positions (domain 1e-3 pc), eta = 0.004; 1.3x
    # measured. Binary e = 0: 1.8e-15 end, 6.1e-15 max, L 3e-15 on clang/arm64; 1.07e-14, 1.15e-14, L 5.6e-15
    # on gcc/x86-64 (bounds from the larger; plain kicks: 2.9e-13 max, L 1e-13).
    # Planets: 3.4e-14 end, 1.1e-13 max, P 7e-18, L 2e-15 (FROST Fig. 5: ~1e-13, ~1e-16, ~1e-14).
    # With the split (twice the steps at the slowest level, so about twice the round-off): 4.5e-15 end,
    # 2.3e-14 max, L 1.2e-14; planets 9.4e-14, 1.1e-13, P 3.9e-18, L 2.5e-15.
    # The binary floor is a random walk: over BINARY_E = 0, 1e-12 ... 4e-12 (same physics, other round-off)
    # the end value spreads over 2.5e-15 - 2.2e-14 and the max over 1.0e-14 - 2.4e-14 with the per-pair Kahan
    # kernels, 2.7e-15 - 3.5e-14 and 1.0e-14 - 4.0e-14 with the tile-blocked ones (L 2.0e-14 at e = 0); the
    # bounds are 1.3x the largest of these, not of the single e = 0 draw.
    "floor_binary_e0": lambda exe: floor(exe, "floor e=0", dict(FROST4, BINARY_E="0.0", ETA="0.004", ETA_FF="0.004",
                                          ETA_FB="0.004", INTEGER_POSITIONS="true", DOMAIN_HALFSIZE_PC="1e-3"),
                                         (4.5e-14, 5.2e-14, 1e-14, 2.6e-14)),
    "floor_planets": lambda exe: floor(exe, "floor planets", dict(PLANETS, ETA="0.004", ETA_FF="0.004", ETA_FB="0.004",
                                        INTEGER_POSITIONS="true", DOMAIN_HALFSIZE_PC="1e-3"),
                                       (1.25e-13, 1.5e-13, 1.0e-17, 3.3e-15)),
    # Same floors through the CPU small-subset path (SERIAL_N2_CPU covers every set), 1.3x measured. Plain
    # kicks there before: binary 1.9e-13 end, 3.5e-13 max, L 1.8e-13; planets 2.6e-13, P 2.1e-15, L 2.9e-14.
    # Planets measured 2.3e-14 end, 9.5e-14 max, P 5.3e-18, L 2.7e-15 (CPU path on FROST Eq. 49 with eta);
    # with the split 9.4e-14 end, 1.1e-13 max, L 1.9e-15.
    "floor_binary_e0_cpu": lambda exe: floor(exe, "floor e=0 cpu path", dict(FROST4, BINARY_E="0.0", ETA="0.004",
                                              ETA_FF="0.004", ETA_FB="0.004", INTEGER_POSITIONS="true",
                                              DOMAIN_HALFSIZE_PC="1e-3", SERIAL_N2_CPU="8"),
                                             (1.8e-14, 1.9e-14, 1e-14, 9.3e-15)),
    "floor_planets_cpu": lambda exe: floor(exe, "floor planets cpu path", dict(PLANETS, ETA="0.004", ETA_FF="0.004",
                                            ETA_FB="0.004", INTEGER_POSITIONS="true", DOMAIN_HALFSIZE_PC="1e-3",
                                            SERIAL_N2_CPU="8"),
                                           (1.25e-13, 1.4e-13, 1.0e-17, 3.5e-15)),
    # 1000 yr is ~6 Neptune orbits: no drift check. HHS measured 1.69e-7, 4.27e-9, 2.50e-10, order 4.70 with the
    # split (2.35e-7, 8.27e-9, 4.39e-10, order 4.53 without)
    # (FROST: ~1e-13 at eta = 0.004, which order 4 from here reaches); FSI (all at the fastest occupied
    # level) measured 7.41e-8, 4.64e-9, 2.90e-10, order 4.00 (hung before the fsi-step fix)
    "planets_hhs": lambda exe: sweep(exe, "planets hhs", PLANETS, ["0.128", "0.064", "0.032"],
                                     [2.2e-7, 5.5e-9, 3.3e-10], (4.60, 5.00)),
    "planets_fsi": lambda exe: sweep(exe, "planets fsi", dict(PLANETS, INTEGRATOR="fsi"), ["0.128", "0.064", "0.032"],
                                     [9.6e-8, 6.1e-9, 3.8e-10], (3.90, 4.30)),
    # Before the fix: segfault in step 1 (duplicated particle after compaction, LogH NaN), then mass loss on
    # step restarts and |dE/E| ~ 1 from an incomplete merger sink. Measured on the host build: 1.4e-10 /
    # 6.7e-13 km/s (P grows smoothly: round-off of the LogH reconstruction of ~600 km/s orbits). Bounds: the
    # restart threshold for E, P far below one lost binary member (~3 km/s).
    "mergers_gpu_path": lambda exe: mergers(exe, "mergers gpu path", dict(MERGERS, SERIAL_N2_CPU="0"), (1e-8, 1e-11)),
    # Before: LogH wrote a 15-body group past its 8-body arrays (bus error). Measured 9.4e-6 / 2.0e-14 km/s.
    # Before: the single-rank step skipped the global PN1 kicks (ratio 0.03). Measured 1.045 (sampled once
    # per print, so the osculating periapsis carries the intra-orbit PN oscillation).
    "pn1_global_single_rank": lambda exe: precession(exe, "wide PN1 binary", PN1_WIDE, 0.1),
    # Before: v/a, gradient and perturbation criteria read accelerations that still held the binary
    # partner (stale from the last force evaluation), so members got orbital time-steps down to level 29
    # and step 1 never finished (> 120 s; the 10 % binary production runs hung the same way). Measured
    # 2 s, 3.2e-10 / 6.0e-15 km/s (2 stellar collisions; their orbital L is not carried, so no L check).
    "binaries_timestep": lambda exe: binaries(exe, "binaries", BINARIES, (1e-8, 1e-12)),
    # Before: 40 LogH give-ups (E_N + E_1PN cannot hold eta_restart under PN), |dE/E| 4.8e-5, |dL/L| 2.3e-5
    # unbooked. Measured 0 / 0 / 2.6e-13 km/s (the whole system is the PN subsystem, so all of it is booked).
    "pn_logh_books": lambda exe: pn_books(exe, "PN LogH binary", PN_LOGH, (1e-12, 1e-11)),
    # Before: merger remnants kept a progenitor's m_zams, so SSE shrank them back (56.7 M_sun lost in 46
    # mergers). Measured 1.3e-9 M_sun (winds over 5 steps).
    "mergers_sse": lambda exe: sse_mass(exe, "mergers + SSE", dict(MERGERS, SERIAL_N2_CPU="0", SSE_ENABLED="true",
                                                                   TOTAL_TIME_MYR="5e-9"), 1e-6),
    # Before: the neighbour buffer (max(1000, N/10) pairs) truncated silently, 250 binaries stayed field
    # particles at level ~30 and step 1 never finished. Measured: all 1250 detected, 1.1 s, 1.7e-13.
    "binaries50_detected": lambda exe: all_binaries(exe, "50 % binaries", BINARIES50, 1250),
    # HHS_CC (connected components): a pair made fast only by the v/a criterion must stay one component (before the
    # dominant-partner edges: precession off by 50 %), and binaries keep their strongest perturber (before: 10x dE)
    "cc_pn1_wide": lambda exe: precession(exe, "wide PN1 binary, HHS_CC", dict(PN1_WIDE, HHS_CC="true"), 0.1),
    "cc_binaries50": lambda exe: all_binaries(exe, "50 % binaries, HHS_CC", dict(BINARIES50, HHS_CC="true"), 1250),
    # measured 5.0e-11 / 3.8e-11 = 1.3 (10 before binaries kept their perturbers)
    "cc_vs_plain_binaries50": lambda exe: cc_vs_plain(exe, "50 % binaries", BINARIES50, 3.0),
    # Triple branch of detect_from_pairs used a particle id where the union-find wanted a compact id
    # (out of bounds for ids >= the number of pair members; the tests' triples had ids 0-2)
    # Measured 9.8e-9 (dense core, at the restart threshold): bound 3e-8
    "triples_detected": lambda exe: triples(exe, "dense core", TRIPLES, 3e-8),
    # r_h = 0.03 pc: before, a 9+ body chain (linked by unbound fly-bys) went to the FSI whole, with its
    # tight binaries integrated orbit by orbit (r_h = 0.02 pc never finished a step). Measured 3.6e-8.
    "dense_core_split": lambda exe: no_fsi_groups(exe, "dense core", dict(TRIPLES, PLUMMER_R_HALF="0.03",
                                                                            TOTAL_TIME_MYR="1e-4"), 1e-7),
    "logh_big_group": lambda exe: big_group(exe, "big group", BIG_GROUP, (1.2e-5, 1e-12)),
    # MPI cases, run under NBODY_TEST_LAUNCHER (CTest sets 2 ranks). Spins were not broadcast: lost on every
    # rank of a GPU run (host backends alias the mirror, so the host build cannot fail this one).
    "spin_mpi2": lambda exe: spins(exe, "IMBH spin, 2 ranks", SPIN_IMBH, 0.9),
    # Mergers mid-recursion on 2 ranks: compaction must leave the replicas identical (MPI slices resynced)
    "mergers_mpi2": lambda exe: replicas(exe, "mergers, 2 ranks", dict(MERGERS, SERIAL_N2_CPU="0",
                                                                         REPLICA_CHECK_INTERVAL="1")),
    # CPU path with restarts off: a merger in a nested CPU call used to drop the enclosing level's subsystem
    # flags on the host (internal forces counted twice), |dE/E| 2e-5 per step, hidden by restarts. Measured
    # now 1.4e-10 / 6.7e-13 km/s, as the GPU path.
    # 2000 stars with binaries on 1 and 2 ranks: measured |difference| 4e-15 over 5 steps (summation order)
    "mpi_1v2": lambda exe: ranks_agree(exe, "1 vs 2 ranks", dict(BINARIES, TOTAL_TIME_MYR="5e-3", REPLICA_CHECK_INTERVAL="1"),
                                       1e-13),
    # The same with DISTRIBUTE_SUBSYSTEMS (each rank integrates its share, then exchanges): replicas must stay
    # identical (hash every step) and agree with 1 rank. Before mass was exchanged: REPLICA MISMATCH, exit 3.
    "mergers_mpi2_dist": lambda exe: replicas(exe, "mergers, 2 ranks, distributed", dict(MERGERS, SERIAL_N2_CPU="0",
                                              REPLICA_CHECK_INTERVAL="1", DISTRIBUTE_SUBSYSTEMS="true",
                                              DISTRIBUTE_MIN_SUBSYSTEMS="1")),
    "mpi_1v2_dist": lambda exe: ranks_agree(exe, "1 vs 2 ranks, distributed", dict(BINARIES, TOTAL_TIME_MYR="5e-3",
                                            REPLICA_CHECK_INTERVAL="1", DISTRIBUTE_SUBSYSTEMS="true",
                                            DISTRIBUTE_MIN_SUBSYSTEMS="1"), 1e-13),
    "mergers_cpu_path": lambda exe: mergers(exe, "mergers cpu path", dict(MERGERS, SERIAL_N2_CPU="64", ETA_RESTART="0"),
                                            (1e-8, 1e-11)),
}

if __name__ == "__main__":
    exe, case = sys.argv[1], sys.argv[2]
    sys.exit(0 if CASES[case](exe) else 1)
