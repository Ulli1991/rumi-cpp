#!/usr/bin/env python3
"""RUMI++ initial conditions -> PeTar input (the same particles for a head-to-head run, PLAN item 4b).

Input: the step-0 particle dump of a RUMI++ run with NBODY_DUMP=1 (dump_0000.bin: int N, then N rows of
pid m x y z vx vy vz subsystem_id as doubles; pc, Msun, km/s).
Output: the 7-column file petar.init reads (m x y z vx vy vz), Msun / pc / pc/Myr (petar -u 1), binaries first as
adjacent pairs (petar -b n_bin). Binaries are mutual most-bound pairs with E < 0 within r_max (the binary census of
src/io/cluster_diag.cpp), so the primordial binaries of the IC generator are found whatever their storage order.

    python nbody_to_petar.py dump_0000.bin petar_ic.txt [--rmax-pc 1e-3]
Prints n_bin for petar -b. Needs numpy and scipy.
"""
import argparse
import numpy as np
from scipy.spatial import cKDTree

G = 4.300917270036279e-3            # pc (km/s)^2 / Msun (RUMI++ constants.h)
KMS_TO_PC_MYR = 1.0227122  # 1 km/s in pc/Myr


def read_dump(path):
    with open(path, "rb") as f:
        n = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        rows = np.fromfile(f, dtype=np.float64, count=9 * n).reshape(n, 9)
    return rows


def find_binaries(m, x, v, rmax):
    pairs = cKDTree(x).query_pairs(rmax, output_type="ndarray")
    best = np.full(len(m), -1)
    best_e = np.zeros(len(m))
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        r = np.linalg.norm(x[j] - x[i], axis=1)
        mu = m[i] * m[j] / (m[i] + m[j])
        e = 0.5 * mu * np.sum((v[j] - v[i]) ** 2, axis=1) - G * m[i] * m[j] / r
        for a, b, eab in zip(i, j, e):   # most bound partner of each particle (E < 0 only)
            if eab < best_e[a]: best_e[a], best[a] = eab, b
            if eab < best_e[b]: best_e[b], best[b] = eab, a
    return [(a, best[a]) for a in range(len(m)) if best[a] > a and best[best[a]] == a]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump")
    ap.add_argument("out")
    ap.add_argument("--rmax-pc", type=float, default=1e-3, help="largest binary separation searched (pc)")
    a = ap.parse_args()
    d = read_dump(a.dump)
    d = d[d[:, 1] > 0.0]                       # massless placeholders (none at step 0)
    m, x, v = d[:, 1], d[:, 2:5], d[:, 5:8]
    bins = find_binaries(m, x, v, a.rmax_pc)
    in_bin = np.zeros(len(m), bool)
    order = []
    for i, j in bins:
        order += [i, j]
        in_bin[i] = in_bin[j] = True
    order += list(np.nonzero(~in_bin)[0])
    out = np.column_stack([m, x, v * KMS_TO_PC_MYR])[order]
    np.savetxt(a.out, out, fmt="%.17g")
    print(f"N = {len(m)}, n_bin = {len(bins)} (petar -b {len(bins)} -u 1), M = {m.sum():.6g} Msun -> {a.out}")


if __name__ == "__main__":
    main()
