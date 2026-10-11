"""Read cpp-nbody snapshots with the LUNA hydro code's own I/O (CTest luna_reader; LUNA is not modified).

    <python with LUNA importable> tests/luna_read_check.py <run directory>

Uses luna.file_io.read_snapshot (read_snapshot, latest_snapshot_dir, find_snapshot_at_time) on the
snapdir_NNN/ directories, then reads the particles the way LUNA's Gadget-style tools do (Header attributes,
PartTypeN/Coordinates, Masses, ParticleIDs). Exit 0 if everything is consistent.
"""
import glob
import os
import sys

import h5py
import numpy as np
from luna.file_io.read_snapshot import find_snapshot_at_time, latest_snapshot_dir, read_snapshot


def main(run_dir):
    bad = 0

    def check(name, ok):
        nonlocal bad
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
        bad += not ok

    dirs = sorted(glob.glob(os.path.join(run_dir, "snapdir_*")))
    check(f"snapshot directories found ({len(dirs)})", len(dirs) >= 1)
    if not dirs:
        return 1
    last = latest_snapshot_dir(run_dir)
    check("latest_snapshot_dir picks the highest number", last == dirs[-1])
    for d in dirs:
        s = read_snapshot(d)                 # LUNA's reader: placeholder grid + metadata
        check(f"{os.path.basename(d)}: read_snapshot t={s['t']:.6e} step={s['step']} ndim={s['ndim']}",
              s["ndim"] == 3 and np.isfinite(s["rho"]).all() and s["rho"].shape == (1, 1, 1))
        check(f"{os.path.basename(d)}: find_snapshot_at_time(t) returns it", find_snapshot_at_time(run_dir, s["t"]) == d)
        files = sorted(glob.glob(os.path.join(d, "*.hdf5")))
        with h5py.File(files[0], "r") as f:
            H = f["Header"].attrs
            n_tot = np.asarray(H["NumPart_Total"], dtype=np.int64)
            n_file = sum(f[g]["ParticleIDs"].shape[0] for g in f if g.startswith("PartType"))
            check(f"{os.path.basename(d)}: Header NumPart_Total {n_tot.tolist()} = {n_file} particles in the file",
                  int(n_tot.sum()) == n_file == int(H["N"]))
            check(f"{os.path.basename(d)}: Header Time = cells time", float(H["Time"]) == s["t"])
            m = sum(float(np.sum(f[g]["Masses"][:])) for g in f if g.startswith("PartType"))
            rho_box = m / float(H["BoxSize"]) ** 3
            check(f"{os.path.basename(d)}: placeholder density = M / BoxSize^3 ({rho_box:.4e})",
                  np.isclose(s["rho"].ravel()[0], rho_box, rtol=1e-12))
            for g in ("PartType4", "PartType5"):
                if g in f:
                    pos = np.asarray(f[g]["Coordinates"], dtype=np.float64)
                    ids = np.asarray(f[g]["ParticleIDs"])
                    check(f"{os.path.basename(d)}: {g} Coordinates {pos.shape}, unique ParticleIDs",
                          pos.ndim == 2 and pos.shape[1] == 3 and len(np.unique(ids)) == len(ids))
            u = f["units"].attrs
            check(f"{os.path.basename(d)}: units group (UnitLength_in_cm = 1 pc)",
                  np.isclose(u["UnitLength_in_cm"], 3.08567758e18) and "UnitTime_in_s" in u)
    print(f"{bad} failures")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
