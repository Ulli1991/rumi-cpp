#!/usr/bin/env python3
"""Summarise the binary-cluster timing runs (make_runs.py): wall time of the first interval (the measurement of
Rantala et al. 2023, Fig. 7) and the median over the intervals run, against that figure, with |dE/E| and where the time goes.

Usage: summarize.py <run_dir>
"""
import re
import statistics
import sys
from pathlib import Path

# Rantala et al. (2023), Fig. 7, read off the figure (1 Raven node, T = 1e-3 Myr); text: 720, 67, 1.5 Myr per 24 h
# (0.120, 1.29, 57.6 s) and +25 %, +24 %, +3 % from f_bin = 10 % to 100 %
REF = {(10000, 0.1): 0.10, (10000, 0.5): 0.12, (10000, 1.0): 0.125,
           (100000, 0.1): 1.08, (100000, 0.5): 1.25, (100000, 1.0): 1.34,
           (1000000, 0.1): 57.0}


def parse(log):
    s = log.read_text(errors="replace")
    m = re.match(r"n(\d)e(\d)_f(\d+)_(cm\w+)", log.stem)
    n, f, cm = int(m.group(1)) * 10 ** int(m.group(2)), int(m.group(3)) / 100, m.group(4)
    walls = [float(x) for x in re.findall(r"^[ \t]+Total wall[ \t]+(\S+)[ \t]+\d+", s, re.M)]
    de = [float(r) for r in re.findall(r"^[ \t]+L0[ \t]+\S+[ \t]+\d+[ \t]+(\S+)[ \t]+\S+[ \t]+\S+[ \t]+\S+[ \t]+\d+", s, re.M)]
    ic = re.search(r"IC check: N_obj=(\d+) r_h\(measured\)=(\S+) pc Q_obj=(\S+) E_bin/\|E_obj\|=(\S+)", s)
    rh = re.search(r"IC r_h = (\S+) pc", s)
    # Profile tables, one per step: component -> seconds
    profs = []
    for block in re.findall(r"--- Performance.*?\n(.*?)\n\s+-{20,}", s, re.S):
        prof = {}
        for line in block.splitlines():
            mm = re.match(r"^\s{2}(\S.*?\S)\s{2,}(\d+\.\d+)\s+(\d+|-)\s", line)
            if mm and mm.group(1) != "Component":
                prof[mm.group(1)] = float(mm.group(2))
        profs.append(prof)
    return dict(name=log.stem, n=n, f=f, cm=cm, walls=walls, de=de, profs=profs,
                mergers=len(set(re.findall(r"\[MERGER #(\d+)\]", s))), restarts=s.count("RESTART["),
                ebin=float(ic.group(4)) if ic else float("nan"), rh=float(rh.group(1)) if rh else float("nan"),
                done="Done!" in s)


def main():
    d = Path(sys.argv[1])
    runs = sorted((parse(p) for p in d.glob("n*_f*_cm*.log")), key=lambda r: (r["n"], r["cm"], r["f"]))
    print("| N | f_bin | c.o.m. | r_h [pc] | E_bin/|E| | ref. [s] | step 1 [s] | x ref. | median [s] (steps) |"
          " x ref. | |dE/E| step 1 | |dE/E| end | mergers | restarts |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        ref = REF.get((r["n"], r["f"]), float("nan"))
        w = r["walls"]
        t1 = w[0] if w else float("nan")
        med = statistics.median(w) if w else float("nan")
        print(f"| {r['n']:.0e} | {r['f']:.1f} | {r['cm'][2:]} | {r['rh']:.2f} | {r['ebin']:.1f} | {ref:.3g} | {t1:.3g} |"
              f" {t1 / ref:.1f} | {med:.3g} ({len(w)}) | {med / ref:.1f} |"
              f" {r['de'][0] if r['de'] else float('nan'):.1e} | {r['de'][-1] if r['de'] else float('nan'):.1e} |"
              f" {r['mergers']} | {r['restarts']} |" + ("" if r["done"] else " (cut)"))
    print("\nWhere the time goes (sum over all steps, % of total wall; top 6 components):")
    for r in runs:
        tot = {}
        for p in r["profs"]:
            for k, v in p.items():
                tot[k] = tot.get(k, 0.0) + v
        wall = sum(r["walls"]) or 1.0
        top = sorted(((v, k) for k, v in tot.items()), reverse=True)[:6]
        print(f"  {r['name']}: " + ", ".join(f"{k} {100 * v / wall:.0f}%" for v, k in top))


if __name__ == "__main__":
    main()
