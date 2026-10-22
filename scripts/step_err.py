#!/usr/bin/env python3
"""Per-step energy error and wall time of cpp-nbody runs (A/B scans).

Reads the signed cumulative unbooked |E0|-relative error from the "SDAR step N ... unbooked_dE/E0 X" lines and the
"Step wall" lines of each run.log; prints per run: steps, mean / rms / max |dE/E0| per step, |final - first|, median
and mean s/step. Usage: step_err.py [--skip K] DIR [DIR ...]  (K: steps to drop at the start, default 2)
"""
import re
import statistics
import sys


def parse(path):
    err, wall = {}, []
    rx = re.compile(r"^SDAR step (\d+) E_sd/E0 \S+ unbooked_dE/E0 (\S+)")
    with open(path, errors="replace") as f:
        for line in f:
            m = rx.match(line)
            if m:
                err[int(m.group(1))] = float(m.group(2))
            elif line.startswith("  Step wall"):
                wall.append(float(line.split()[2]))
    return err, wall


def main(argv):
    skip = 2
    if argv and argv[0] == "--skip":
        skip = int(argv[1])
        argv = argv[2:]
    print(f"{'run':40s} {'steps':>5s} {'mean|dE|':>10s} {'rms':>10s} {'max':>10s} {'drift':>10s} {'med s':>7s} {'mean s':>7s}")
    for d in argv:
        err, wall = parse(d.rstrip("/") + "/run.log")
        steps = sorted(err)
        if len(steps) < skip + 2:
            print(f"{d:40s} too few steps ({len(steps)})")
            continue
        steps = steps[skip:]
        de = [err[b] - err[a] for a, b in zip(steps, steps[1:]) if b == a + 1]
        ab = [abs(x) for x in de]
        w = wall[skip + 1:] if len(wall) > skip + 1 else wall
        print(f"{d:40s} {len(de):5d} {statistics.mean(ab):10.3e} {(sum(x * x for x in de) / len(de)) ** 0.5:10.3e} "
              f"{max(ab):10.3e} {abs(err[steps[-1]] - err[steps[0]]):10.3e} "
              f"{statistics.median(w) if w else 0:7.2f} {statistics.mean(w) if w else 0:7.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
