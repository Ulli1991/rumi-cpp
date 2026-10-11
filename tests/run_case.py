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
# 0-10. Each merger compacts the particle arrays mid-recursion and may trigger step restarts. Contact binaries on
# purpose: old mass-rank pairing, no pericentre check.
MERGERS = dict(TEST_CASE="plummer", INTEGRATOR="hhs", TOTAL_TIME_MYR="2e-8", DT_MAX_MYR="1e-9",
               DT_MIN_MYR="1e-16", ETA_FF="0.2", ETA_FB="0.2", ETA_GRAD="0.1", ETA_JERK="0.1", SOFTENING_PC="0",
               COLLISIONAL="true", R_NGB="1e-3", R_NGB_MAX="0.1", ETA_PERT="0.02", ETA_GBS="1e-10",
               ETA_RESTART="1e-8", N_ORB_SECULAR="3", M_PN="5.0", PN_ORDER="0", R_ESCAPE_RH="100.0",
               SSE_ENABLED="false", PLUMMER_N="400", PLUMMER_M_TOTAL="200", PLUMMER_R_HALF="0.3", PLUMMER_SEED="7",
               PLUMMER_MASS_DIST="kroupa", BINARY_FRAC="0.5", BINARY_A_MIN_AU="0.005", BINARY_A_MAX_AU="0.02",
               BINARY_POP="mass_rank", BINARY_PERI_CHECK="false", PRINT_INTERVAL="1", PROFILING="false")
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
# Spinning BH binary in LogH (BIFROST sec. 3.7; PN_ORDER bit 32 = spin couplings), v/c ~ 0.02, ~21 orbits.
# Precession: e = 0, PN1 + spins, small chi1 perpendicular to L (simple precession about J), chi2 = 0.
SPIN_PREC = dict(PN_LOGH, BINARY_E="0.0", PN_ORDER="33", BINARY_CHI1="0.1 0 0", BINARY_CHI2="0 0 0")
# Total J = L_N + L_SO + S1 + S2: e = 0.5, two large misaligned spins, spin couplings only (no orbital PN
# terms, whose truncated L_PN1 alone leaves |dJ|/|J| ~ 5e-6 here)
SPIN_J = dict(PN_LOGH, PN_ORDER="32", BINARY_CHI1="0.6 0 0.6", BINARY_CHI2="0 0.7 -0.4")
# Same with PN1: J adds L_PN1; the error is the O((v/c)^4) truncation (5.3e-6 measured, 4.9e-6 with chi = 0)
SPIN_J_PN1 = dict(SPIN_J, PN_ORDER="33")
# 5000 Kroupa stars, 50 % primordial binaries (1250 pairs, more than the initial neighbour buffer of 1000)
BINARIES50 = dict(BINARIES, PLUMMER_N="5000", PLUMMER_M_TOTAL="2000", BINARY_FRAC="0.5", TOTAL_TIME_MYR="1e-3")
# Dense core (r_h = 0.05 pc), 50 % binaries of 1-20 AU: binary-single encounters form triples and multiplets.
# Kept on the old mass-rank pairing: its max |dE/E| sits at the restart threshold and scatters by 10x over seeds
# for either population (seeds 5-8, mass_rank 1.3e-8 - 1.8e-7, bifrost 2.7e-8 - 1.7e-7), so a population change
# would only re-roll that scatter.
TRIPLES = dict(BINARIES, PLUMMER_N="2000", PLUMMER_M_TOTAL="800", PLUMMER_R_HALF="0.05", BINARY_FRAC="0.5",
               BINARY_A_MIN_AU="1", BINARY_A_MAX_AU="20", TOTAL_TIME_MYR="2e-4", DT_MAX_MYR="1e-4",
               BINARY_POP="mass_rank")
# Stable hierarchical triple of stars (2 AU inner, 41 AU outer, 60 deg), steps of ~45 inner orbits: the
# secular-triple path. TRIPLE_OUTER_FIRST stores the outer body at the lowest particle id.
SEC_TRIPLE = dict(TEST_CASE="triple_test", INTEGRATOR="hhs", TRIPLE_A_IN_PC="1e-5", TRIPLE_E_IN="0.3", TRIPLE_M1="1.0",
                  TRIPLE_M2="0.8", TRIPLE_A_OUT_PC="2e-4", TRIPLE_E_OUT="0.2", TRIPLE_M3="0.6", TRIPLE_INCL_DEG="60",
                  TOTAL_TIME_MYR="1e-3", DT_MAX_MYR="1e-4", DT_MIN_MYR="1e-14", SOFTENING_PC="0", COLLISIONAL="true",
                  R_NGB="1e-3", R_NGB_MAX="0.1", N_ORB_SECULAR="3", PN_ORDER="0", SSE_ENABLED="false",
                  ETA_RESTART="0", PRINT_INTERVAL="1", PROFILING="false")
# 10+10 Msun BH binary, a = 2 AU, e = 0.5, rotated to omega = 1e-6 rad (apoapsis start has omega = pi), all PN
# orders, steps of ~16 orbits: the secular binary path; 1PN precession ~4e-5 rad per step
SEC_BINARY = dict(TEST_CASE="binary_test", INTEGRATOR="hhs", BINARY_A_PC="1e-5", BINARY_E="0.5", BINARY_M1="10.0",
                  BINARY_M2="10.0", BINARY_ROT_DEG="180.0000573", TOTAL_TIME_MYR="1e-4", DT_MAX_MYR="1e-5", DT_MIN_MYR="1e-14", SOFTENING_PC="0",
                  COLLISIONAL="true", R_NGB="1e-3", R_NGB_MAX="0.1", N_ORB_SECULAR="3", PN_ORDER="31", M_PN="5.0",
                  SSE_ENABLED="false", ETA_RESTART="0", PRINT_INTERVAL="1", PROFILING="false")
# 2000 Kroupa stars, 50 % binaries of 0.01-0.1 AU (2-21 R_sun) with thermal e: many pericentres inside R1 + R2
# unless the IC generator enforces r_p > R1 + R2 (AUDIT F6). One 1e-5 Myr step (hundreds of orbits).
CLOSE_BINARIES = dict(BINARIES, BINARY_FRAC="0.5", BINARY_A_MIN_AU="0.01", BINARY_A_MAX_AU="0.1",
                      TOTAL_TIME_MYR="1e-5", DT_MAX_MYR="1e-5")
# 1.0+0.8 Msun stellar binary, a = 0.02 AU, e = 0.9, all PN orders on (M_PN = 5): ~2400 orbits per half step,
# secular path. Below m_pn it is a Kepler drift (as in LogH): no PN precession sub-steps
SEC_STELLAR = dict(SEC_BINARY, BINARY_A_PC="1e-7", BINARY_E="0.9", BINARY_M1="1.0", BINARY_M2="0.8",
                   BINARY_ROT_DEG="0")
# 10+10 Msun BH binary, a = 0.01 AU, e = 0.5, all PN orders: 1.7 rad of 1PN precession per yr, t_GW ~ 1 Myr
SEC_BH_TIGHT = dict(SEC_BINARY, BINARY_A_PC="5e-8", BINARY_ROT_DEG="0")
# Tight binary (0.02 AU, P = 2.1e-3 yr) on a Mardling-unstable outer orbit (2 AU, e = 0.95, starts at
# apocentre): LogH, ~140 inner orbits per half step of 0.3 yr, integrated orbit by orbit
HIER_LOGH = dict(TEST_CASE="triple_test", INTEGRATOR="hhs", TRIPLE_A_IN_PC="1e-7", TRIPLE_E_IN="0.1", TRIPLE_M1="1.0",
                 TRIPLE_M2="1.0", TRIPLE_A_OUT_PC="1e-5", TRIPLE_E_OUT="0.95", TRIPLE_M3="1.0", TRIPLE_INCL_DEG="30",
                 TOTAL_TIME_MYR="6e-7", DT_MAX_MYR="6e-7", DT_MIN_MYR="1e-16", SOFTENING_PC="0", COLLISIONAL="true",
                 R_NGB="1e-3", R_NGB_MAX="0.1", N_ORB_SECULAR="3", PN_ORDER="0", SSE_ENABLED="false",
                 ETA_RESTART="1e-8", ETA_GBS="1e-10", PRINT_INTERVAL="1", PROFILING="false")
# The same binary and a hyperbolic fly-by (periapsis 20 AU, e = 5, from 100 AU): tidal ratio ~1e-9, ~2400 binary
# orbits per half step of 5 yr
WEAK_FLYBY = dict(HIER_LOGH, TRIPLE_A_OUT_PC="1e-4", TRIPLE_E_OUT="5.0", TOTAL_TIME_MYR="4e-5", DT_MAX_MYR="1e-5",
                  FLYBY_CM_VELOCITY="true")
# 16 stars inside r_ngb: one 15-body neighbour group, more than LogH's fixed storage (LOGH_MAX_BODIES = 8)
# Collisional FSI run of a field binary (r_ngb below a, so no subsystem): the start-of-step kernel's criteria set the step
FSI_BINARY = dict(TEST_CASE="binary_test", INTEGRATOR="fsi", TOTAL_TIME_MYR="1e-5", DT_MAX_MYR="1e-5", DT_MIN_MYR="1e-14",
                  SOFTENING_PC="0", COLLISIONAL="true", R_NGB="1e-6", R_NGB_MAX="1e-6", PN_ORDER="0",
                  SSE_ENABLED="false", BINARY_E="0.5", PRINT_INTERVAL="1", PROFILING="false")
# The same binary through the HHS with ETA_FF/FB = 10 and no jerk step: the v/a criterion (ETA) sets the step
VA_BINARY = dict(FSI_BINARY, INTEGRATOR="hhs", TOTAL_TIME_MYR="1e-4", ETA_FF="10", ETA_FB="10", ETA_JERK="0")
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


def spin_rows(exe, cfg):
    """(dchi, dJ/J, phi_S1, ratio_so) rows printed by the spin tracker."""
    rows = re.findall(r"Spin: dchi=(\S+)\s+dJ/J=(\S+)\s+phi_S1=(\S+)\s+ratio_so=(\S+)", run_output(exe, cfg))
    if not rows:
        raise SystemExit("FAIL: no spin tracker output")
    return [[float(x) for x in r] for r in rows]


def spin_precession(exe, label, cfg, tol, dchi_bound):
    """S1 precession angle about J against the orbit-averaged 1.5PN spin-orbit rate; |chi| invariant."""
    rows = spin_rows(exe, cfg)
    ok = check(f"{label} |precession / spin-orbit rate - 1|", abs(rows[-1][3] - 1.0), tol)
    return ok & check(f"{label} max | |chi| - |chi0| |", max(r[0] for r in rows), dchi_bound)


def spin_j(exe, label, cfg, dj_bound, dchi_bound):
    """Total J = L_N + L_PN1 + L_SO + S1 + S2 conserved; |chi| invariant."""
    rows = spin_rows(exe, cfg)
    ok = check(f"{label} max |dJ|/|J|", max(r[1] for r in rows), dj_bound)
    ok &= check(f"{label} S1 precessed [rad]", abs(rows[-1][2]), 1e-3, lower=True)
    return ok & check(f"{label} max | |chi| - |chi0| |", max(r[0] for r in rows), dchi_bound)


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


def paths_agree(exe, label, cfg, bound):
    """The CPU small-set path (SERIAL_N2_CPU covers the whole run) against the device path on the same run: the
    same LogH calls and GBS steps per step, |dE/E| per step equal to `bound` (relative)."""
    def stats(out):
        return [(int(a), int(b)) for a, b in re.findall(r"Integration: LogH=(\d+) \(gbs=(\d+)", out)]
    dev_out = run_output(exe, dict(cfg, SERIAL_N2_CPU="0"))
    cpu_out = run_output(exe, dict(cfg, SERIAL_N2_CPU="1000000"))
    dev, cpu = diagnostics(dev_out), diagnostics(cpu_out)
    print(f"   LogH calls, GBS steps per step: device {stats(dev_out)}, cpu {stats(cpu_out)}")
    print(f"   |dE/E| device {[f'{r[0]:.6e}' for r in dev]}, cpu {[f'{r[0]:.6e}' for r in cpu]}")
    ok = check(f"{label} steps (device vs cpu)", abs(len(dev) - len(cpu)), 0)
    ok &= check(f"{label} steps with other LogH calls or GBS steps",
                sum(a != b for a, b in zip(stats(dev_out), stats(cpu_out))), 0)
    return ok & check(f"{label} max relative |dE/E| difference",
                      max(abs(a[0] - b[0]) / max(a[0], 1e-300) for a, b in zip(dev, cpu)), bound)


def cc_consistent(exe, label, cfg, bound):
    """HHS_CC must split the fast set with the pair time steps of the time-step kernel (NBODY_CHECK_CC reports a
    particle whose pair minimum in the split lies below its dt_adapt)."""
    os.environ["NBODY_CHECK_CC"] = "1"
    out = run_output(exe, dict(cfg, HHS_CC="true"))
    del os.environ["NBODY_CHECK_CC"]
    bad = sum(int(n) for n in re.findall(r"CC_CHECK: (\d+) of", out))
    ok = check(f"{label} particles with a split pair step below dt_adapt", bad, 0)
    return ok & check(f"{label} max|dE/E|", max(r[0] for r in diagnostics(out)), bound)


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


def triple_order(exe, label, cfg, rel):
    """A triple must evolve the same whichever particle ids its members carry: the outer body at the lowest
    id must not be taken as an inner-binary member (that pair is unbound and the triple was skipped, frozen)."""
    out = {of: run_output(exe, dict(cfg, TRIPLE_OUTER_FIRST=of)) for of in ("false", "true")}
    n_sec = {of: o.count("[SEC_TRI]") for of, o in out.items()}
    e = {of: diagnostics(o)[-1][0] for of, o in out.items()}
    ok = check(f"{label} secular-triple calls, outer body last", n_sec["false"], 1, lower=True)
    ok &= check(f"{label} secular-triple calls, outer body first", n_sec["true"], n_sec["false"], lower=True)
    ok &= check(f"{label} |dE/E| outer body first (evolves)", e["true"], 1e-30, lower=True)
    return ok & check(f"{label} |dE/E| first vs last, relative difference",
                      abs(e["true"] - e["false"]) / max(e["false"], 1e-300), rel)


def secular_steps(exe, label, cfg, per_call):
    """Internal steps of the secular binary integrator per evolution: the step must not collapse when the
    argument of periapsis is near 0 (it used omega / omega-dot, and every call ran 100000 steps)."""
    out = run_output(exe, cfg)
    pairs = [(int(a), int(b)) for a, b in re.findall(r"Secular=(\d+) \(substeps=(\d+)\)", out)]
    n_ev, n_sub = sum(a for a, _ in pairs), sum(b for _, b in pairs)
    ok = check(f"{label} secular evolutions", n_ev, 1, lower=True)
    return ok & check(f"{label} internal steps per evolution", n_sub / max(n_ev, 1), per_call)


def secular_triples_off(exe, label, cfg, bound):
    """SECULAR_TRIPLES = false: a stable triple in the secular regime goes to LogH (FROST-CLUSTERS I dropped the
    double-averaged path); with the default it takes the secular path."""
    on, off = run_output(exe, cfg), run_output(exe, dict(cfg, SECULAR_TRIPLES="false"))
    ok = check(f"{label} secular-triple calls, default", on.count("[SEC_TRI]"), 1, lower=True)
    ok &= check(f"{label} secular-triple calls, SECULAR_TRIPLES = false", off.count("[SEC_TRI]"), 0)
    return ok & check(f"{label} |dE/E| in LogH", diagnostics(off)[-1][0], bound)


def logh_complete(exe, label, cfg, bound):
    """Every LogH call must reach its end time: no GBS step-limit hits, no retries, energy within bound."""
    out = run_output(exe, cfg)
    caps = sum(int(c) for c in re.findall(r"cap=(\d+)", out))
    ok = check(f"{label} LogH calls stopped by the GBS step limit", caps, 0)
    ok &= check(f"{label} LogH retries", out.count("LogH restart:"), 0)
    return ok & check(f"{label} |dE/E|", diagnostics(out)[-1][0], bound)


def weak_split(exe, label, cfg, bound):
    """WEAK_PERTURBER_GAMMA: the weakly perturbed binary of a fly-by triple is split off (secular binary)."""
    out = run_output(exe, cfg)
    n = sum(int(c) for c in re.findall(r"weak_split=(\d+)", out))
    gbs = sum(int(c) for c in re.findall(r"\(gbs=(\d+)", out))
    ok = check(f"{label} triples split", n, 1, lower=True) & check(f"{label} LogH GBS steps", gbs, 0)
    return ok & check(f"{label} max|dE/E|", max(r[0] for r in diagnostics(out)), bound)


def no_fsi_groups(exe, label, cfg, bound):
    """Neighbour chains in a dense core must be split along bound pairs, not left whole to the FSI."""
    out = run_output(exe, cfg)
    rows = diagnostics(out)
    return check(f"{label} groups left to the FSI", out.count("exceeds LOGH_MAX_BODIES"), 0) & \
        check(f"{label} max|dE/E|", max(r[0] for r in rows), bound)


def no_ic_mergers(exe, label, cfg, bound):
    """Primordial binaries must not start inside their contact distance: no mergers in the first step."""
    out = run_output(exe, cfg)
    rows = diagnostics(out)
    return check(f"{label} mergers in step 1", out.count("[MERGER #"), 0) & \
        check(f"{label} max|dE/E|", max(r[0] for r in rows), bound)

def fsi_step(exe, label, cfg, bound):
    """The FSI integrator's global step (the smallest particle time step at the start) must stay at a sane level."""
    m = re.search(r"FSI global step: (\S+) Myr", run_output(exe, cfg))
    if not m:
        raise SystemExit("FAIL: no FSI global step printed")
    return check(f"{label} FSI global step [Myr]", float(m.group(1)), bound, lower=True)


def restarts(exe, label, cfg, bound):
    """Every step that fails ETA_RESTART must pass after restarts with smaller etas (no give-up)."""
    out = run_output(exe, cfg)
    rows = diagnostics(out)
    return check(f"{label} restarts", out.count("RESTART["), 1, lower=True) & \
        check(f"{label} give-ups", out.count("accepting"), 0) & \
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
    # With the BIFROST population and the pericentre check (defaults since AUDIT F6): 6.6e-10, L 2.4e-14.
    "binaries_timestep": lambda exe: binaries(exe, "binaries", BINARIES, (1e-8, 1e-12)),
    # Before: 40 LogH give-ups (E_N + E_1PN cannot hold eta_restart under PN), |dE/E| 4.8e-5, |dL/L| 2.3e-5
    # unbooked. Measured 0 / 0 / 2.6e-13 km/s (the whole system is the PN subsystem, so all of it is booked).
    # Spins in LogH (BIFROST sec. 3.7). Before: the spins were not in the GBS state vector, so they never
    # changed (precession ratio 0, |dJ|/|J| 7.6e-6 with PN1), and the orbit had no spin-dependent terms.
    # Measured: precession ratio - 1 = 5.5e-3 (PN1 shifts the orbit; < 5e-7 with spin couplings only),
    # | |chi| - |chi0| | <= 3.0e-15 (Rodrigues rotations, norms restored after the extrapolation),
    # |dJ|/|J| 3.5e-7 (spins only; 1.1e-8 at 4x the separation: the O((v/c)^5) truncation) and 5.25e-6 with PN1
    "spin_precession": lambda exe: spin_precession(exe, "spin-orbit precession", SPIN_PREC, 7.2e-3, 4e-15),
    "spin_j": lambda exe: spin_j(exe, "spinning BH binary", SPIN_J, 4.6e-7, 4e-15),
    "spin_j_pn1": lambda exe: spin_j(exe, "spinning BH binary + PN1", SPIN_J_PN1, 6.9e-6, 4e-15),
    "pn_logh_books": lambda exe: pn_books(exe, "PN LogH binary", PN_LOGH, (1e-12, 1e-11)),
    # Before: merger remnants kept a progenitor's m_zams, so SSE shrank them back (56.7 M_sun lost in 46
    # mergers). Measured 1.3e-9 M_sun (winds over 5 steps).
    "mergers_sse": lambda exe: sse_mass(exe, "mergers + SSE", dict(MERGERS, SERIAL_N2_CPU="0", SSE_ENABLED="true",
                                                                   TOTAL_TIME_MYR="5e-9"), 1e-6),
    # Before: the neighbour buffer (max(1000, N/10) pairs) truncated silently, 250 binaries stayed field
    # particles at level ~30 and step 1 never finished. Measured: all 1250 detected, 1.1 s, 1.7e-13 (BIFROST
    # population + pericentre check: 1.05e-10; HHS_CC 8.0e-11).
    "binaries50_detected": lambda exe: all_binaries(exe, "50 % binaries", BINARIES50, 1250),
    # HHS_CC (connected components): a pair made fast only by the v/a criterion must stay one component (before the
    # dominant-partner edges: precession off by 50 %), and binaries keep their strongest perturber (before: 10x dE)
    "cc_pn1_wide": lambda exe: precession(exe, "wide PN1 binary, HHS_CC", dict(PN1_WIDE, HHS_CC="true"), 0.1),
    "cc_binaries50": lambda exe: all_binaries(exe, "50 % binaries, HHS_CC", dict(BINARIES50, HHS_CC="true"), 1250),
    # measured 5.0e-11 / 3.8e-11 = 1.3 (10 before binaries kept their perturbers); BIFROST population: 0.76
    "cc_vs_plain_binaries50": lambda exe: cc_vs_plain(exe, "50 % binaries", BINARIES50, 3.0),
    # FLYBY_CM_VELOCITY (FROST-CLUSTERS II sec. 2.1): only the device kernels used the subsystem c.o.m. velocity;
    # the CPU small-set path and the HHS_CC split kept the component velocities. Before: CPU 1.255e-10 vs device
    # 6.42e-11 after step 1 (relative 0.95), LogH calls 136 vs 116 (6.4e-6 relative with the component velocities
    # on both paths). Measured after: same LogH calls and GBS steps, 2.2e-5 relative (9.5e-6 with 2 threads).
    # HHS_CC: 331 particles had a pair step in the split below their dt_adapt; 0 after (max |dE/E| 5.7e-11)
    "cm_cpu_vs_device": lambda exe: paths_agree(exe, "c.o.m. fly-by step", dict(BINARIES, FLYBY_CM_VELOCITY="true"),
                                                2.9e-5),
    "cc_cm_consistent": lambda exe: cc_consistent(exe, "50 % binaries, HHS_CC, c.o.m. fly-by step",
                                                  dict(BINARIES50, FLYBY_CM_VELOCITY="true"), 1e-8),
    # Triple branch of detect_from_pairs used a particle id where the union-find wanted a compact id
    # (out of bounds for ids >= the number of pair members; the tests' triples had ids 0-2)
    # The secular-triple driver took the inner pair from the first two ids (fixed 2026-10-10)
    # The secular step used omega / omega-dot (fixed 2026-10-11)
    "secular_omega0_steps": lambda exe: secular_steps(exe, "secular binary at omega = 1e-6", SEC_BINARY, 10),
    # Stellar binaries got PN secular terms: 20 internal steps per evolution here (thousands at L0 of the 200k
    # cluster, 5-7 ms secular calls). Measured 1 after (no PN below m_pn).
    "secular_stellar_no_pn": lambda exe: secular_steps(exe, "stellar binary, PN on", SEC_STELLAR, 1.3),
    # The step also had a precession term (1/omega-dot), but the omega drift is exact for constant (a, e): 1028
    # internal steps per evolution here, the 100000-step cap (48 ms) for this binary at 1e-3 Myr steps.
    # Measured 1 after (eta_sec t_GW > dt).
    "secular_bh_tight_steps": lambda exe: secular_steps(exe, "BH binary at 0.01 AU", SEC_BH_TIGHT, 1.3),
    # Before: both LogH calls stopped at the 1000-step limit after ~45 % of the interval, and passed the energy
    # check (cap=4, no retry: the members lagged behind the system clock). Measured after: 0 caps, |dE/E| 8.5e-12
    "logh_hierarchy_complete": lambda exe: logh_complete(exe, "tight binary in unstable triple", HIER_LOGH, 1.1e-11),
    # Experimental split (off by default). Without: LogH, 575k GBS steps, 3.7 s, |dE/E| 1.5e-9 (max). With:
    # 0.06 s, max 3.6e-8 at periapsis, 5.3e-9 at the end
    "weak_flyby_split": lambda exe: weak_split(exe, "weak fly-by split", dict(WEAK_FLYBY, WEAK_PERTURBER_GAMMA="1e-8"),
                                               4.7e-8),
    "triple_id_order": lambda exe: triple_order(exe, "secular triple", SEC_TRIPLE, 1e-6),
    # SECULAR_TRIPLES = false (FROST-CLUSTERS I): the switch was ignored by the per-level re-classification
    # (40 secular calls before). In LogH the triple keeps |dE/E| 2.17e-11 vs 3.4e-5 secular: bound 2.9e-11
    "secular_triples_off": lambda exe: secular_triples_off(exe, "secular triple off", SEC_TRIPLE, 2.9e-11),
    # Measured 9.8e-9 (dense core, at the restart threshold); 2.35e-8 before the pericentre check, 2.68e-8 with
    # it (seed 5): bound 3.5e-8
    "triples_detected": lambda exe: triples(exe, "dense core", TRIPLES, 3.5e-8),
    # r_h = 0.03 pc: before, a 9+ body chain (linked by unbound fly-bys) went to the FSI whole, with its
    # tight binaries integrated orbit by orbit (r_h = 0.02 pc never finished a step). Measured 3.6e-8
    # (4.5e-8 with the pericentre check).
    "dense_core_split": lambda exe: no_fsi_groups(exe, "dense core", dict(TRIPLES, PLUMMER_R_HALF="0.03",
                                                                            TOTAL_TIME_MYR="1e-4"), 1e-7),
    # AUDIT F6: thermal e without r_p > R1 + R2 put pericentres inside the stars: 21 mergers in the first
    # 1e-5 Myr before, 0 after (322 of 5000 e redrawn in a 20k cluster). Measured |dE/E| 2.3e-14.
    "ic_binaries_no_mergers": lambda exe: no_ic_mergers(exe, "close primordial binaries", CLOSE_BINARIES, 3e-14),
    # ETA_JERK = 0 turns the jerk criterion off. Before: the start-of-step kernel (AccDtK) still applied it,
    # dt = 0 * |a| / |j| = 0, the run stepped at DT_MIN (1.86e-14 Myr, hung). Measured 1.5625e-7 (as with 0.1)
    "eta_jerk0_fsi": lambda exe: fsi_step(exe, "ETA_JERK = 0", dict(FSI_BINARY, ETA_JERK="0"), 1.2e-7),
    # ETA_GRAD = 0 used to switch the v/a criterion off as well (one shared branch): the binary then stepped at
    # ETA_FF/FB = 10. Measured 2.8 before, 7.36e-7 after (v/a limited, ETA = 0.1)
    "va_without_grad": lambda exe: binaries(exe, "v/a with ETA_GRAD = 0", dict(VA_BINARY, ETA_GRAD="0", ETA="0.1"),
                                            (9.6e-7, 1e-12)),
    # Restarts halve every eta (BIFROST sec. 3.19) but did not halve ETA (v/a): with v/a limiting, a retry repeated
    # the failed step, "dE not decreasing", and the step was accepted above ETA_RESTART. ETA_GRAD = 1000 keeps the
    # shared branch on without the gradient step limiting. Measured before 10 give-ups, 1.9e-5;
    # after 13 restarts, 0 give-ups, 2.87e-7
    "restart_halves_va": lambda exe: restarts(exe, "v/a restarts", dict(VA_BINARY, ETA_GRAD="1000", ETA="0.2",
                                                                       ETA_RESTART="1e-7"), 3.7e-7),
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
