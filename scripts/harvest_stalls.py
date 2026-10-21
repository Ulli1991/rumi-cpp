"""Harvest stalled steps from production runs as isolated cases (candidates for hard tests).

    python3 scripts/harvest_stalls.py RUN.log [RUN2.log ...] --out DIR [--radius 0.05] [--exe nbody-kokkos]
                                     [--timeout 600] [--env KEY=VALUE ...]

For every STALL block of a log (STALL_FACTOR: slow step, STALL_DE_FACTOR: energy jump) whose restart was kept (STALL_RESTART, "STALL restart <path>"), the
culprits (the deepest set, each costly or inaccurate LogH integration, each subsystem of 3+ bodies, by particle
ids) are cut out with scripts/cutout.py into
DIR/<run>_s<step>_<k>/. Culprit sets met before (same particle ids) are skipped. With --exe every case is replayed
in isolation on the host (one interval, OMP_NUM_THREADS=1 unless set) and the table lists wall time, LogH GBS steps
and |dE| / E_REF (the parent cluster's |E0|, the units of the run's STALL jumps; the run itself prints |dE/E0| of
the cut-out's own E0), |dL/L|: a case that still takes long reproduces the stall and becomes a test (tests/data/, run_case.py)
with the fix that makes it fast at equal or better energy. Pure Python.
"""
import argparse
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
G = 4.302194e-3   # pc (km/s)^2 / Msun (G_CODE)


def eref_scale(case):
    """|E0| of the cut-out (ic.txt) over the parent's E_REF (config.txt; 1 without E_REF)"""
    eref = None
    for line in open(case / "config.txt"):
        if "=" in line and not line.lstrip().startswith("#") and line.split("=")[0].strip() == "E_REF":
            eref = float(line.split("=", 1)[1])
    b = [[float(v) for v in line.split()[:7]] for line in open(case / "ic.txt") if not line.startswith("#")]
    T = sum(0.5 * q[0] * (q[4] ** 2 + q[5] ** 2 + q[6] ** 2) for q in b)
    U = -sum(G * b[i][0] * b[j][0] / math.dist(b[i][1:4], b[j][1:4]) for i in range(len(b)) for j in range(i + 1, len(b)))
    return abs(T + U) / eref if eref else 1.0


def stall_blocks(log):
    """[(step, wall, median, restart path or None, [(kind, [pids], info)])] of one run log"""
    blocks, cur = [], None
    for line in open(log, errors="replace"):
        if line.startswith("STALL step ") or line.startswith("STALL energy step "):
            m = re.match(r"STALL (?:energy )?step (\d+): (?:dE/E0 )?(\S+?)(?: s)? \(median of the previous steps (\S+?)(?: s)?\)",
                         line)
            step = int(m.group(1))
            if cur is not None and cur[0] == step:   # slow and energy jump in one step: one block
                cur[1] = f"{cur[1]}, {m.group(2)}"
                continue
            cur = [step, m.group(2), m.group(3), None, []]
            blocks.append(cur)
        elif cur is None:
            continue
        elif line.startswith("STALL deepest "):
            m = re.match(r"STALL deepest depth (-?\d+) pivot (\S+) Myr pids(.*)", line)
            pids = [int(v) for v in m.group(3).split()]
            if pids:
                cur[4].append(("deep", pids, f"depth {m.group(1)}"))
        elif line.startswith("STALL logh "):
            m = re.match(r"STALL logh gbs (\d+) dt (\S+) Myr pids(.*?) m\[Msun\](.*)", line)
            cur[4].append(("logh", [int(v) for v in m.group(3).split()], f"gbs {m.group(1)} m {m.group(4).strip()}"))
        elif line.startswith("STALL loghE "):
            m = re.match(r"STALL loghE dE/E0 (\S+) pids(.*?) m\[Msun\](.*)", line)
            cur[4].append(("loghE", [int(v) for v in m.group(2).split()], f"dE/E0 {m.group(1)} m {m.group(3).strip()}"))
        elif line.startswith("STALL group "):
            m = re.match(r"STALL group n (\d+) pids(.*)", line)
            cur[4].append(("group", [int(v) for v in m.group(2).split()], f"n {m.group(1)}"))
        elif line.startswith("STALL restart "):
            cur[3] = line.split()[2]
    return blocks


def replay(exe, case, timeout, env):
    e = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"), **env}
    t0 = time.time()
    try:
        out = subprocess.run([exe, "--config", "config.txt"], cwd=case, capture_output=True, text=True,
                             timeout=timeout, env=e).stdout
    except subprocess.TimeoutExpired:
        return f">{timeout}", "-", "-", "-"
    wall = time.time() - t0
    (case / "run.log").write_text(out)
    gbs = sum(int(v) for v in re.findall(r"LogH=\d+ \(gbs=(\d+)", out))
    de = re.findall(r"^\|dE/E0\| = (\S+)", out, re.M)
    dl = re.findall(r"^\|dL/L0\| = (\S+)", out, re.M)
    return f"{wall:.2f}", str(gbs), f"{float(de[-1]) * eref_scale(case):.2e}" if de else "-", f"{float(dl[-1]):.2e}" if dl else "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=float, default=0.05, help="pc")
    ap.add_argument("--exe", help="host binary for the isolated replay")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for the replay (repeatable)")
    a = ap.parse_args()
    env = dict(kv.split("=", 1) for kv in a.env)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seen, rows = set(), []
    for log in a.logs:
        run = Path(log).resolve().parent.name
        for step, wall, med, rst, culprits in stall_blocks(log):
            if rst is None:
                print(f"{log}: step {step} ({wall} vs median {med}) has no kept restart (STALL_RESTART)")
                continue
            rst_path = Path(rst) if Path(rst).is_absolute() else Path(log).resolve().parent / rst
            for k, (kind, pids, info) in enumerate(culprits):
                key = tuple(sorted(pids))
                if key in seen:
                    continue
                seen.add(key)
                case = out / f"{run}_s{step}_{k}"
                cmd = [sys.executable, str(HERE / "cutout.py"), str(rst_path), "--radius", str(a.radius),
                       "--out", str(case)]
                for p in pids[:8]:
                    cmd += ["--pid", str(p)]
                subprocess.run(cmd, check=True, capture_output=True)
                n = sum(1 for line in open(case / "ic.txt") if not line.startswith("#"))
                res = replay(a.exe, case, a.timeout, env) if a.exe else ("-", "-", "-", "-")
                rows.append((case.name, f"{wall}/{med}", kind, info, n) + res)
    print(f"{'case':<28} {'step value/median':>22} {'kind':<5} {'n':>5} {'wall':>8} {'gbs':>9} {'dE/E_REF':>10} {'dL/L':>10}  culprit")
    for name, sw, kind, info, n, w, g, de, dl in rows:
        print(f"{name:<28} {sw:>22} {kind:<5} {n:>5} {w:>8} {g:>9} {de:>10} {dl:>10}  {info}")


if __name__ == "__main__":
    sys.exit(main())
