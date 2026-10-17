#!/usr/bin/env python3.11
"""Monte Carlo arithmetic samples of tests/vfc/vfc_driver.cpp built with Verificarlo (build_vfc.sh) and the number of
significant decimal digits of every printed quantity: s = -log10(sigma / |mean|) (Stott Parker 1997; capped at the
53-bit 15.95), plus sigma itself (for dE/E: the round-off floor of the energy error).
  vfc_run.py --run "<command prefix>" --bin <vfc binary> --case 1 [--samples 30] [--jobs 8] [--precision 53]
The command prefix runs the binary (e.g. "apptainer exec -B ... vfc.sif"); VFC_BACKENDS selects random rounding."""
import argparse, math, os, shlex, subprocess, statistics
from concurrent.futures import ThreadPoolExecutor


def sample(prefix, binary, case, backend, extra=()):
    env = {**os.environ, "VFC_BACKENDS": backend, "VFC_BACKENDS_SILENT_LOAD": "TRUE", "VFC_BACKENDS_LOGGER": "FALSE"}
    out = subprocess.run(shlex.split(prefix) + [binary, str(case), *extra], capture_output=True, text=True, env=env, check=True)
    vals = {}
    for line in out.stdout.splitlines():
        p = line.split()
        if len(p) == 2:
            try:
                vals[p[0]] = float(p[1])
            except ValueError:
                pass
    return vals


def digits(xs):
    mu = statistics.fmean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    if sd == 0.0:
        return mu, sd, 15.95
    if mu == 0.0:
        return mu, sd, 0.0
    return mu, sd, max(0.0, min(15.95, -math.log10(sd / abs(mu))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="")
    ap.add_argument("--bin", required=True)
    ap.add_argument("--case", type=int, required=True)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--precision", type=int, default=53)
    ap.add_argument("--mode", default="rr")
    ap.add_argument("--arg", action="append", default=[], help="extra driver argument (b: Bulirsch sequence)")
    ap.add_argument("--only", default="", help="comma-separated quantity names to print")
    a = ap.parse_args()
    backend = f"libinterflop_mca.so --precision-binary64={a.precision} --mode={a.mode}"
    with ThreadPoolExecutor(a.jobs) as ex:
        runs = list(ex.map(lambda _: sample(a.run, a.bin, a.case, backend, a.arg), range(a.samples)))
    names = list(runs[0])
    only = set(a.only.split(",")) if a.only else None
    print(f"# case {a.case} {' '.join(a.arg)}, {a.samples} samples, MCA {a.mode} p={a.precision}, bin {os.path.basename(a.bin)}")
    print(f"{'quantity':14s} {'mean':>24s} {'sigma':>10s} {'digits':>6s}")
    for n in names:
        if only and n not in only:
            continue
        mu, sd, s = digits([r[n] for r in runs if n in r])
        print(f"{n:14s} {mu:24.16e} {sd:10.2e} {s:6.2f}")


if __name__ == "__main__":
    main()
