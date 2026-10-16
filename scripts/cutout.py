"""Cut a case out of a restart file: the particles within a radius of some particles, as a particle_file IC with the
parent run's configuration, for reproducing a stall or an energy jump in isolation and keeping it as a test.

    python3 scripts/cutout.py RESTART.h5 --pid 5099 [--pid 34915 ...] --radius 0.05 --out DIR [--set KEY=VALUE ...]

Writes DIR/ic.txt ("m x y z vx vy vz kw R" per line, c.o.m. frame, pc and km/s) and DIR/config.txt: the keys of the
parent run's config file (the restart's /config/config_json; other keys take their defaults; --set adds or
replaces keys) with TEST_CASE = particle_file, one interval (TOTAL_TIME_MYR =
DT_MAX_MYR), no restarts, snapshots or runtime stellar evolution, and the parent's |E0| (E_REF) and neighbour mass
scale (NGB_MREF), so that the tidal-sampling tolerance and the neighbour radii are those of the cluster. The centre
is the c.o.m. of the given particles. A stalled step's restart is RESTART_DIR/stall_<step>.h5 (STALL_RESTART), its
particle ids are on the run's STALL lines. Needs h5dump on the PATH; pure Python.
"""
import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

FIELDS = ("particle_id", "mass", "x", "y", "z", "vx", "vy", "vz", "stellar_type", "stellar_radius")


def h5_dataset(path, name):
    out = subprocess.run(["h5dump", "-m", "%.17g", "-w", "0", "-y", "-d", name, str(path)],
                         capture_output=True, text=True, check=True).stdout
    start = out.index("DATA {") + 6
    body = out[start:out.index("}", start)]
    return [v for v in body.replace("\n", " ").replace(",", " ").split() if v]


def h5_attr(path, name):
    out = subprocess.run(["h5dump", "-m", "%.17g", "-w", "0", "-a", name, str(path)],
                         capture_output=True, text=True, check=True).stdout
    m = re.search(r"\(0\): (.*)", out)
    if not m:
        raise SystemExit(f"cutout: no attribute {name} in {path}")
    return m.group(1).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("restart")
    ap.add_argument("--pid", type=int, action="append", required=True, help="centre particle id (repeatable)")
    ap.add_argument("--radius", type=float, required=True, help="pc")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], help="KEY=VALUE config override (repeatable)")
    a = ap.parse_args()

    rst = Path(a.restart).resolve()
    data = {f: h5_dataset(rst, "/particles/" + f) for f in FIELDS}
    n = len(data["mass"])
    pid = [int(v) for v in data["particle_id"]]
    m = [float(v) for v in data["mass"]]
    q = [[float(v) for v in data[f]] for f in ("x", "y", "z", "vx", "vy", "vz")]
    kw = [int(v) for v in data["stellar_type"]]
    rad = [float(v) for v in data["stellar_radius"]]
    where = {p: i for i, p in enumerate(pid)}
    missing = [p for p in a.pid if p not in where]
    if missing:
        raise SystemExit(f"cutout: particle ids {missing} not in {rst}")
    cs = [where[p] for p in a.pid]
    mc = sum(m[i] for i in cs)
    centre = [sum(m[i] * q[k][i] for i in cs) / mc for k in range(3)]
    sel = [i for i in range(n) if m[i] > 0.0 and math.dist([q[k][i] for k in range(3)], centre) < a.radius]
    msel = sum(m[i] for i in sel)
    cm = [sum(m[i] * q[k][i] for i in sel) / msel for k in range(6)]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_myr = float(h5_attr(rst, "/time")) * 0.977792
    with open(out / "ic.txt", "w") as f:
        f.write(f"# {rst} (t = {t_myr:.6g} Myr): the {len(sel)} particles within {a.radius} pc of the c.o.m. of pids "
                f"{' '.join(map(str, a.pid))}, c.o.m. frame; m x y z vx vy vz kw R\n")
        for i in sel:
            f.write(" ".join([repr(m[i])] + [repr(q[k][i] - cm[k]) for k in range(6)] + [str(kw[i]), repr(rad[i])])
                    + "\n")

    raw = h5_attr(rst, "/config/config_json")
    cfg = json.loads(raw[1:-1] if raw.startswith('"') else raw)
    cfg.update(TEST_CASE="particle_file", IC_FILE=str((out / "ic.txt").resolve()), TOTAL_TIME_MYR=cfg["DT_MAX_MYR"],
               RESTART_FILE="", RESTART_INTERVAL="0", SNAPSHOT_INTERVAL="0", SSE_ENABLED="false",
               CLUSTER_DIAG_INTERVAL="0", WALLTIME_LIMIT="0", STALL_RESTART="false", PRINT_INTERVAL="1")
    for name, key, inverse in (("e_ref", "E_REF", False), ("ngb_inv_mref", "NGB_MREF", True)):
        try:
            v = float(h5_attr(rst, "/collisional/config/" + name))
        except subprocess.CalledProcessError:
            continue
        if v > 0.0 and float(cfg.get(key, "0") or 0) == 0.0:
            cfg[key] = repr(1.0 / v if inverse else v)
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k.strip()] = v.strip()
    with open(out / "config.txt", "w") as f:
        f.write(f"# cut out of {rst} by scripts/cutout.py\n")
        for k in sorted(cfg):
            f.write(f"{k} = {cfg[k]}\n")
    print(f"{out}: {len(sel)} particles, total mass {msel:.6g} Msun, E_REF {cfg.get('E_REF')}, NGB_MREF {cfg.get('NGB_MREF')}")


if __name__ == "__main__":
    sys.exit(main())
