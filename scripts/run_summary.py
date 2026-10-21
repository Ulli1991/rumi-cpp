"""One-line summary per run: time reached, steps, final |dE/E| / |dP|/M / |dL/L|, wall time, s/step statistics.

    python3 scripts/run_summary.py RUN_DIR_OR_LOG [...]

Reads run.log with grep (C speed, also for 1 GB logs): the L0 rows of the time-bin table (|dE/E|, |dP|/M, |dL/L|,
time, step), the profiler's per-report "Total wall" (s per step; from the PROFILING report, absent with
PROFILING = off), and the chain markers "=== segment n start|end <epoch>" written by the production job scripts
(total wall = sum of the segments; otherwise the RESULTS "Wall:" line). Pure Python + grep.
"""
import os
import re
import statistics as st
import subprocess
import sys


def grep(pattern, path):
    out = subprocess.run(["grep", "-E", pattern, path], capture_output=True, text=True, errors="replace",
                         env={**os.environ, "LC_ALL": "C"})
    return out.stdout.splitlines()


def summary(path):
    log = os.path.join(path, "run.log") if os.path.isdir(path) else path
    if not os.path.isfile(log):
        return f"{path}: no run.log"
    lines = grep(r"^  L0 |^  Total wall |^=== segment |^Wall: ", log)
    last, walls, seg, wall_line = None, [], {}, None
    for l in lines:
        p = l.split()
        if l.startswith("  L0 ") and len(p) >= 8:
            last = p
        elif l.startswith("  Total wall ") and len(p) >= 4:
            try:
                walls.append(float(p[2]) / float(p[3]))   # seconds over the steps of the report
            except ValueError:
                pass
        elif l.startswith("=== segment ") and len(p) >= 5:
            seg.setdefault(p[2], {})[p[3]] = int(p[4])
        elif l.startswith("Wall: "):
            wall_line = l
    s = f"{path}:"
    if last:
        s += f" t={float(last[6]):.3f} Myr step={last[7]} dE/E={float(last[3]):.2e} dP/M={float(last[4]):.1e}" \
             f" dL/L={float(last[5]):.1e}"
    done = [v["end"] - v["start"] for v in seg.values() if "start" in v and "end" in v]
    if done:
        s += f" wall={sum(done) / 3600:.2f} h ({len(done)} seg)"
    elif wall_line:
        m = re.search(r"Wall: ([\d.]+)s", wall_line)
        if m:
            s += f" wall={float(m.group(1)) / 3600:.2f} h"
    if walls:
        s += f" s/step median={st.median(walls):.2f} mean={st.mean(walls):.2f} max={max(walls):.0f} (n={len(walls)})"
    return s


if __name__ == "__main__":
    for a in sys.argv[1:] or ["."]:
        print(summary(a), flush=True)
