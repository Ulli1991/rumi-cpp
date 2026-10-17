"""Replay every captured energy jump with the energy ledger and sort the jumps into classes by cause.

    python3 scripts/replay_ledger.py setup RUN.log [RUN2.log ...] --out DIR --exe nbody-kokkos [--submit]
    python3 scripts/replay_ledger.py classify DIR

setup: for every "STALL energy step" block of a log whose restart was kept (STALL_RESTART), DIR/<run>_s<step>/ gets
the run's config with RESTART_FILE = that stall_<step>.h5 and TOTAL_TIME_MYR = its time + DT_MAX_MYR (one step),
and a one-rank GPU job (gpudev,gpu) running it with NBODY_ELEDGER=1 NBODY_EATTR=1; --submit sends the jobs.

classify: reads the replays and prints one row per jump: the jump, the ledger's root error (it must equal the jump:
the replay is the run's own step), and the class:
  booking:<path>  the subsystem advances leave the jump unbooked (|members' change + books| >= half the root);
                  path = secular binaries, secular triples, or other (LogH / SDAR / collisions)
  field:d<depth>  the subsystem books balance; the error is made in the FSI, largest own error at that depth,
                  with the largest tidal-sampling estimate of the original step (binary, a, e, gamma, P, dt)
  merger          a merger inside the step (the ledger does not attribute it)
Pure Python; the jobs need the slurm partition names of raven.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

TU_MYR = 0.977792


def blocks(log):
    """[(step, value, restart path, [tidal lines])] of the energy blocks of a run log"""
    out, cur = [], None
    for line in open(log, errors="replace"):
        if line.startswith("STALL energy step "):
            m = re.match(r"STALL energy step (\d+): dE/E0 (\S+)", line)
            cur = [int(m.group(1)), float(m.group(2)), None, []]
            out.append(cur)
        elif line.startswith("STALL step "):
            cur = None if (cur is None or not line.startswith(f"STALL step {cur[0]}:")) else cur
        elif cur is not None and line.startswith("STALL tidal "):
            cur[3].append(line.strip()[len("STALL tidal "):])
        elif cur is not None and line.startswith("STALL restart "):
            cur[2] = line.split()[2]
    return out


def h5_time_myr(path):
    out = subprocess.run(["h5dump", "-m", "%.17g", "-a", "/time", str(path)], capture_output=True, text=True,
                         check=True).stdout
    return float(re.search(r"\(0\): (\S+)", out).group(1)) * TU_MYR


def setup(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    skip = ("RESTART_FILE", "TOTAL_TIME_MYR", "STALL_RESTART", "PRINT_INTERVAL", "RESTART_INTERVAL")
    for log in a.logs:
        run_dir = Path(log).resolve().parent
        cfg_src = run_dir / "config.txt"
        cfg_lines = [l for l in cfg_src.read_text().splitlines()
                     if not ("=" in l and not l.lstrip().startswith("#") and l.split("=")[0].strip() in skip)]
        dt_max = 1e-3
        for l in cfg_lines:
            if l.split("=")[0].strip() == "DT_MAX_MYR":
                dt_max = float(l.split("=")[1].split("#")[0])
        for step, val, rst, tidal in blocks(log):
            if rst is None:
                continue
            rst_path = Path(rst) if Path(rst).is_absolute() else run_dir / rst
            d = out / f"{run_dir.name}_s{step}"
            d.mkdir(exist_ok=True)
            t_end = h5_time_myr(rst_path) + dt_max
            (d / "config.txt").write_text("\n".join(cfg_lines + [
                f"RESTART_FILE = {rst_path}", f"TOTAL_TIME_MYR = {t_end:.12g}", "STALL_RESTART = false",
                "PRINT_INTERVAL = 1", "RESTART_INTERVAL = 0"]) + "\n")
            (d / "jump.txt").write_text(f"{run_dir.name} {step} {val}\n" + "".join(t + "\n" for t in tidal))
            job = d / "job.sh"
            job.write_text(f"""#!/bin/bash -l
#SBATCH -J led_{run_dir.name}_{step}
#SBATCH -o {d}/%j.out
#SBATCH -D {d}
#SBATCH --partition=gpudev,gpu
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=240000
module purge; module load gcc/15 cuda/13.2 openmpi_gpu/5.0 hdf5-serial/1.14.1
export OMP_NUM_THREADS=36 OMP_PROC_BIND=spread OMP_PLACES=threads
NBODY_ELEDGER=1 NBODY_EATTR=1 timeout 14m srun -n 1 {Path(a.exe).resolve()} --config config.txt > run.log 2>&1
""")
            print(d)
            if a.submit:
                subprocess.run(["sbatch", str(job)], check=True)


def classify(a):
    rows = []
    for d in sorted(Path(a.dir).iterdir()):
        jf, lf = d / "jump.txt", d / "run.log"
        if not jf.exists():
            continue
        head = jf.read_text().splitlines()
        run, step, val = head[0].split()
        tidal = head[1] if len(head) > 1 else ""
        if not lf.exists():
            rows.append((d.name, float(val), None, "pending", ""))
            continue
        txt = lf.read_text(errors="replace")
        root = re.findall(r"^ELEDGER root dE/E0 (\S+) \(.*?\)(.*)$", txt, re.M)
        if not root:
            rows.append((d.name, float(val), None, "no ledger (failed?)", ""))
            continue
        r0 = float(root[0][0])
        merger = "merger inside" in root[0][1]
        adv = float(re.findall(r"^ELEDGER subsystem advances: members' energy change \+ books change (\S+)", txt, re.M)[0])
        pr = re.findall(r"^ELEDGER path residues: secular binaries (\S+) .*?secular triples (\S+)", txt, re.M)[0]
        sec2, sec3 = float(pr[0]), float(pr[1])
        own = re.findall(r"^ELEDGER own (\S+) total \S+ depth (\d+) n (\d+)", txt, re.M)
        if abs(adv) >= 0.5 * abs(r0):
            other = adv - sec2 - sec3
            path = max((("secular binaries", sec2), ("secular triples", sec3), ("other (LogH/SDAR/coll)", other)),
                       key=lambda kv: abs(kv[1]))[0]
            cls, detail = f"booking:{path}", f"adv {adv:.2e} sec2 {sec2:.1e} sec3 {sec3:.1e}"
        else:
            depth = own[0][1] if own else "?"
            cls = f"field:d{depth}" + (" (merger in step)" if merger else "")
            detail = f"own {own[0][0]} n {own[0][2]}" if own else ""
            if tidal:
                detail += " | tidal " + tidal
        rows.append((d.name, float(val), r0, cls, detail))
    rows.sort(key=lambda r: -abs(r[1]))
    print(f"{'case':<26} {'jump':>10} {'ledger':>10}  class / detail")
    for name, val, r0, cls, det in rows:
        print(f"{name:<26} {val:>10.2e} {('%.2e' % r0) if r0 is not None else '-':>10}  {cls}  {det[:150]}")
    counts = {}
    for r in rows:
        counts[r[3].split(" ")[0]] = counts.get(r[3].split(" ")[0], 0) + 1
    print("classes:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("logs", nargs="+")
    s.add_argument("--out", required=True)
    s.add_argument("--exe", required=True)
    s.add_argument("--submit", action="store_true")
    c = sub.add_parser("classify")
    c.add_argument("dir")
    a = ap.parse_args()
    setup(a) if a.cmd == "setup" else classify(a)


if __name__ == "__main__":
    sys.exit(main())
