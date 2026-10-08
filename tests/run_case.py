"""Run one cpp-nbody test case from a small config and check the printed diagnostics against bounds."""
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# FROST Fig. 4 binary: m1 = m2 = 10 Msun, a = 10 AU, ~1000 orbits; diagnostics every 1e-5 Myr
KEPLER = dict(TEST_CASE="binary_test", INTEGRATOR="hhs", TOTAL_TIME_MYR="7.0711e-3", DT_MAX_MYR="1.0e-5",
              DT_MIN_MYR="1.0e-14", SOFTENING_PC="0", COLLISIONAL="false", PRINT_INTERVAL="1")


def run(exe, cfg):
    """L0 diagnostic rows (|dE/E|, |dP|/M, |dL/L|) of one run; raises on a nonzero exit."""
    with tempfile.TemporaryDirectory() as d:
        Path(d, "config.txt").write_text("".join(f"{k} = {v}\n" for k, v in cfg.items()))
        out = subprocess.run([exe], cwd=d, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"FAIL: exit {out.returncode}\n{out.stdout[-2000:]}{out.stderr[-2000:]}")
    rows = [line.split() for line in out.stdout.splitlines() if re.match(r"^\s+L0\s", line) and len(line.split()) >= 8]
    if not rows:
        raise SystemExit("FAIL: no diagnostics printed")
    return [[float(r[3]), float(r[4]), float(r[5])] for r in rows]


def check(name, value, bound, lower=False):
    ok = value >= bound if lower else value <= bound
    print(f"{'ok  ' if ok else 'FAIL'} {name} = {value:.3e} ({'>=' if lower else '<='} {bound:.3e})")
    return ok


def kepler(exe, e, etas, bounds, order, drift_max):
    """max |dE/E| per eta and its fitted order, a bounded (non-drifting) energy error, P and L at round-off."""
    ok, maxima = True, []
    for eta, bound in zip(etas, bounds):
        rows = run(exe, {**KEPLER, "BINARY_E": e, "ETA": eta, "ETA_FF": eta, "ETA_FB": eta})
        dE = [r[0] for r in rows]
        n = len(dE)
        maxima.append(max(dE))
        ok &= check(f"e={e} eta={eta} max|dE/E|", max(dE), bound)
        ok &= check(f"e={e} eta={eta} drift (last / first fifth)", max(dE[4 * n // 5:]) / max(dE[: n // 5]), drift_max)
        ok &= check(f"e={e} eta={eta} max|dP|/M [km/s]", max(r[1] for r in rows), 1e-14)
        ok &= check(f"e={e} eta={eta} max|dL/L|", max(r[2] for r in rows), 1e-12)
    if order is not None:
        x, y = [math.log(float(v)) for v in etas], [math.log(v) for v in maxima]
        mx, my = sum(x) / len(x), sum(y) / len(y)
        slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x)
        ok &= check(f"e={e} order of max|dE/E| in eta", slope, order[0], lower=True)
        ok &= check(f"e={e} order of max|dE/E| in eta", slope, order[1])
    return ok


CASES = {
    # bounds ~1.3x measured (5.81e-6, 1.22e-7, 8.34e-9), order measured 4.72 -0.1/+0.3; drift 1.0-1.2
    # (5.0 before the symmetrised-step fix, which also gave 4.6e-3 at eta = 0.064)
    "kepler_e09": lambda exe: kepler(exe, "0.9", ["0.064", "0.032", "0.016"], [7.6e-6, 1.6e-7, 1.1e-8],
                                     (4.62, 5.02), 1.5),
    # circular orbit: the eta^4 term cancels (constant along the orbit), the error sits at round-off
    "kepler_e0": lambda exe: kepler(exe, "0.0", ["0.064"], [1e-12], None, 10.0),
}

if __name__ == "__main__":
    exe, case = sys.argv[1], sys.argv[2]
    sys.exit(0 if CASES[case](exe) else 1)
