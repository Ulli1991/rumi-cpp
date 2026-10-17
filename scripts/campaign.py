"""Harvesting campaign: one run per seed of a base config with stall capture on, as GPU jobs.

    python3 scripts/campaign.py BASE_CONFIG --seeds 42,7,13 --total-myr 0.3 --exe nbody-kokkos --out DIR
                                [--time 00:50:00] [--partition gpu] [--set KEY=VALUE ...] [--submit]

Writes DIR/s<seed>/config.txt (BASE_CONFIG with PLUMMER_SEED, TOTAL_TIME_MYR, STALL_RESTART = true and the --set keys
replaced or added) and DIR/s<seed>/job.sh (one node, 4 ranks), and with --submit sends them. The logs (run.log) are
the input of scripts/replay_ledger.py setup. Jobs of at most 15 minutes go to gpudev,gpu. Pure Python.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--total-myr", required=True)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time", default="00:50:00")
    ap.add_argument("--partition", default="gpu")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    h, m, s = (int(x) for x in a.time.split(":"))
    minutes = h * 60 + m + (s > 0)
    partition = "gpudev,gpu" if minutes <= 15 else a.partition
    # WALLTIME_LIMIT: the run stops before the job's end with a restart file; RESTART_FILE = auto continues from it when
    # the same job script is submitted again (a chain), and starts fresh the first time
    over = {"TOTAL_TIME_MYR": a.total_myr, "STALL_RESTART": "true", "RESTART_FILE": "auto", "RESTART_INTERVAL": "0",
            "WALLTIME_LIMIT": f"{(minutes - 5) / 60.0:.3f}"}
    for kv in a.set:
        k, v = kv.split("=", 1)
        over[k.strip()] = v.strip()
    base = Path(a.base).read_text().splitlines()
    exe = Path(a.exe).resolve()
    for seed in a.seeds.split(","):
        keys = dict(over, PLUMMER_SEED=seed)
        lines, seen = [], set()
        for l in base:
            k = l.split("=")[0].strip() if "=" in l and not l.lstrip().startswith("#") else None
            if k in keys:
                if k not in seen:
                    lines.append(f"{k} = {keys[k]}")
                seen.add(k)
            else:
                lines.append(l)
        lines += [f"{k} = {v}" for k, v in keys.items() if k not in seen]
        d = Path(a.out) / f"s{seed}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.txt").write_text("\n".join(lines) + "\n")
        job = d / "job.sh"
        job.write_text(f"""#!/bin/bash -l
#SBATCH -J camp_{Path(a.out).name}_s{seed}
#SBATCH -o {d}/%j.out
#SBATCH -D {d}
#SBATCH --partition={partition}
#SBATCH --time={a.time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=18
#SBATCH --mem=240000
module purge; module load gcc/15 cuda/13.2 openmpi_gpu/5.0 hdf5-serial/1.14.1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_PROC_BIND=spread OMP_PLACES=threads
timeout {max(minutes - 2, 1)}m srun {exe} --config config.txt --kokkos-map-device-id-by=mpi_rank >> run.log 2>&1
""")
        print(d)
        if a.submit:
            subprocess.run(["sbatch", str(job)], check=True)


if __name__ == "__main__":
    sys.exit(main())
