"""Per-subsystem error events (SUBSYS_ERR lines) as isolated cases, replayed on the host.

    python3 scripts/subsys_cases.py RUN.log [RUN2.log ...] --out DIR [--exe nbody-kokkos] [--top 20] [--timeout 300]

A subsystem advance is self-contained: its members and the time interval determine it. Every SUBSYS_ERR line
(SUBSYS_ERR_TOL: an advance whose unbooked energy change exceeds the tolerance) carries the members' state before
the advance ("[m x y z vx vy vz kw R]", Msun, pc, km/s). For the largest events (--top, one per member set) this
writes DIR/<run>_s<step>_<k>/ic.txt (c.o.m. frame) and config.txt: the parent run's config (config.txt next to the
log), TEST_CASE = particle_file, one interval of the parent's DT_MAX_MYR (the path choice counts orbits per
detection-level step, so the advance's own interval would change it), SUBSYS_ERR_TOL = 1e-30, and
the parent's E_REF / NGB_MREF (from a restart file in the run directory). With --exe each case is replayed (one
thread) and the table lists the logged error, the replay's largest SUBSYS_ERR of the same path and its |dE/E| of
the whole case (in E_REF units): a reproducing case becomes a test in tests/data/. Pure Python; h5dump for E_REF.
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

LINE = re.compile(r"^SUBSYS_ERR step (\d+) (\S+) dE/E0 (\S+) type (-?\d+) n (\d+) dt (\S+) .*? pids ([\d ]+) state (.*)$")


def events(log):
    out = []
    for line in open(log, errors="replace"):
        m = LINE.match(line.strip())
        if not m:
            continue
        bodies = [[float(x) for x in b.split()] for b in re.findall(r"\[([^\]]*)\]", m.group(8))]
        out.append(dict(step=int(m.group(1)), path=m.group(2), dE=float(m.group(3)), type=int(m.group(4)),
                        n=int(m.group(5)), dt=float(m.group(6)), pids=[int(x) for x in m.group(7).split()], bodies=bodies))
    return out


def parent_refs(run_dir):
    for h5 in sorted(run_dir.glob("*.h5")) + sorted(run_dir.glob("stall_roll/*.h5")):
        vals = {}
        for name in ("e_ref", "ngb_inv_mref"):
            r = subprocess.run(["h5dump", "-m", "%.17g", "-a", f"/collisional/config/{name}", str(h5)],
                               capture_output=True, text=True)
            mm = re.search(r"\(0\): (\S+)", r.stdout)
            if mm:
                vals[name] = float(mm.group(1))
        if vals:
            return vals
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exe")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cands = []
    for log in a.logs:
        for e in events(log):
            cands.append((log, e))
    cands.sort(key=lambda c: -abs(c[1]["dE"]))
    seen, rows = set(), []
    skip = ("TEST_CASE", "IC_FILE", "DT_MAX_MYR", "TOTAL_TIME_MYR", "RESTART_FILE", "RESTART_INTERVAL", "STALL_RESTART",
            "SUBSYS_ERR_TOL", "PRINT_INTERVAL", "SNAPSHOT_INTERVAL", "SSE_ENABLED", "WALLTIME_LIMIT", "E_REF", "NGB_MREF")
    refs_cache = {}
    for log, e in cands:
        key = tuple(sorted(e["pids"]))
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > a.top:
            break
        run_dir = Path(log).resolve().parent
        name = f"{run_dir.parent.name}_{run_dir.name}_s{e['step']}_{len(seen)}"
        d = out / name
        d.mkdir(exist_ok=True)
        B = e["bodies"]
        M = sum(b[0] for b in B)
        cm = [sum(b[0] * b[k] for b in B) / M for k in range(1, 7)]
        with open(d / "ic.txt", "w") as f:
            f.write(f"# SUBSYS_ERR {e['path']} dE/E0 {e['dE']:.3e} (step {e['step']} of {run_dir}), pids {e['pids']}, "
                    f"c.o.m. frame; m x y z vx vy vz kw R\n")
            for b in B:
                kw = int(b[7]) if len(b) > 7 and b[7] >= 0 else 1
                R = b[8] if len(b) > 8 else 0.0
                f.write(" ".join([repr(b[0])] + [repr(b[k] - cm[k - 1]) for k in range(1, 7)] + [str(kw), repr(R)]) + "\n")
        base = []
        if (run_dir / "config.txt").exists():
            base = [l for l in (run_dir / "config.txt").read_text().splitlines()
                    if not ("=" in l and not l.lstrip().startswith("#") and l.split("=")[0].strip() in skip)]
        if run_dir not in refs_cache:
            refs_cache[run_dir] = parent_refs(run_dir)
        refs = refs_cache[run_dir]
        # the parent's level-0 step (the secular / Kepler / LogH choice counts orbits per detection-level step): one
        # interval, in which the subsystem advances as often as in the run
        dtmax = 1e-3
        for l in (run_dir / "config.txt").read_text().splitlines() if (run_dir / "config.txt").exists() else []:
            if l.split("=")[0].strip() == "DT_MAX_MYR":
                dtmax = float(l.split("=")[1].split("#")[0])
        extra = [f"TEST_CASE = particle_file", f"IC_FILE = {(d / 'ic.txt').resolve()}", f"DT_MAX_MYR = {dtmax:.17g}",
                 f"TOTAL_TIME_MYR = {dtmax:.17g}", "SUBSYS_ERR_TOL = 1e-30", "PRINT_INTERVAL = 1",
                 "SNAPSHOT_INTERVAL = 0", "SSE_ENABLED = false", "RESTART_INTERVAL = 0", "STALL_RESTART = false"]
        if "e_ref" in refs:
            extra.append(f"E_REF = {refs['e_ref']!r}")
        if refs.get("ngb_inv_mref", 0) > 0:
            extra.append(f"NGB_MREF = {1.0 / refs['ngb_inv_mref']!r}")
        (d / "config.txt").write_text("\n".join(base + extra) + "\n")
        res = ("-", "-", "-")
        if a.exe:
            t0 = time.time()
            try:
                o = subprocess.run([a.exe, "--config", "config.txt"], cwd=d, capture_output=True, text=True,
                                   timeout=a.timeout, env={**os.environ, "OMP_NUM_THREADS": "1"}).stdout
                (d / "run.log").write_text(o)
                same = [abs(x["dE"]) for x in events(d / "run.log") if x["path"] == e["path"]]
                de = re.findall(r"^\|dE/E0\| = (\S+)", o, re.M)
                res = (f"{max(same):.2e}" if same else "none", f"{float(de[-1]):.2e}" if de else "-",
                       f"{time.time() - t0:.1f}")
            except subprocess.TimeoutExpired:
                res = ("timeout", "-", f">{a.timeout}")
        rows.append((name, e["path"], e["n"], e["dE"]) + res)
    print(f"{'case':<40} {'path':<15} {'n':>2} {'logged':>10} {'replay':>9} {'case dE':>9} {'s':>6}")
    for r in rows:
        print(f"{r[0]:<40} {r[1]:<15} {r[2]:>2} {r[3]:>10.2e} {r[4]:>9} {r[5]:>9} {r[6]:>6}")


if __name__ == "__main__":
    sys.exit(main())
