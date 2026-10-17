"""Adversarial search for conservation-breaking few-body configurations.

    python3 scripts/adversary.py --exe nbody-kokkos --out DIR [--family binflyby] [--workers 72] [--random 2000]
                                 [--generations 20] [--pop 72] [--bisect 200] [--eref 2.71184797e8 --nmref 0.40052]

A scenario family maps a point of the unit cube to a few-body initial condition (particle_file IC). Each evaluation
runs it on the host (one thread) embedded in a cluster setting (E_REF, NGB_MREF of a parent run) and scores it by
the worst conservation violation of the run, in |E0| units of the parent cluster: max(|E_end - E_0| / E_REF, largest
SUBSYS_ERR), with |dL/L| and the path signature (subsystem types and integrators used) recorded. The search:
  1. random points;
  2. bisection between pairs of points whose path signatures differ (secular / LogH / Kepler, member or not): the
     errors live where the code switches path, so points are pushed onto those boundaries;
  3. an evolution strategy (mu best -> Gaussian offspring, step size adapted) from the worst points.
Every evaluation is a line of DIR/evals.jsonl; the worst distinct cases are written to DIR/top/<k>/ (ic.txt,
config.txt) for scripts/replay_ledger / tests. Pure Python (multiprocessing).

Families:
  triple    hierarchical triple (m1, m2, m3 0.1-100 Msun, BH or star, a_in 1e-2-1e2 AU, e_in, a_out / a_in 2-1e4,
            e_out, mutual inclination, phases)
  binflyby  binary (m1, m2 0.1-100 Msun, BH or star above 5 Msun, a 1e-3-1e3 AU, e, orientation, phase) and a
            passing star (m3, impact parameter 1-1e5 a, v_inf 1-100 km/s), closest approach in mid-run
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

G = 4.302194e-3          # pc (km/s)^2 / Msun
AU = 4.84813681e-6       # pc
PC_KMS_MYR = 1.0227122   # pc/Myr per km/s


def loguni(u, lo, hi):
    return lo * (hi / lo) ** u


def kepler_state(m1, m2, a, e, inc, node, peri, M):
    """relative position/velocity (pc, km/s) of an elliptic orbit at mean anomaly M"""
    mu = G * (m1 + m2)
    E = M
    for _ in range(60):
        E -= (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
    cE, sE = math.cos(E), math.sin(E)
    x, y = a * (cE - e), a * math.sqrt(1 - e * e) * sE
    r = a * (1 - e * cE)
    vx, vy = -math.sqrt(mu * a) / r * sE, math.sqrt(mu * a * (1 - e * e)) / r * cE
    return rotate((x, y, 0.0), inc, node, peri), rotate((vx, vy, 0.0), inc, node, peri)


def rotate(v, inc, node, peri):
    x, y, z = v
    c, s = math.cos(peri), math.sin(peri)
    x, y = c * x - s * y, s * x + c * y
    c, s = math.cos(inc), math.sin(inc)
    y, z = c * y - s * z, s * y + c * z
    c, s = math.cos(node), math.sin(node)
    x, y = c * x - s * y, s * x + c * y
    return (x, y, z)


def family_binflyby(u, total_myr):
    """unit cube point (12) -> bodies [(m, x, y, z, vx, vy, vz, kw, R)]"""
    m1 = loguni(u[0], 0.1, 100.0)
    m2 = loguni(u[1], 0.1, 100.0)
    bh = u[2] > 0.5                       # massive members (>= 5 Msun) as BHs or stars
    a = loguni(u[3], 1e-3, 1e3) * AU
    e = min(u[4], 0.99)
    inc, node, peri, M = math.acos(1 - 2 * u[5]), 2 * math.pi * u[6], 2 * math.pi * u[7], 2 * math.pi * u[8]
    m3 = loguni(u[9], 0.1, 100.0)
    b = loguni(u[10], 1.0, 1e5) * a        # impact parameter
    vinf = loguni(u[11], 1.0, 100.0)       # km/s
    kw = lambda m: (14 if bh else 1) if m >= 5.0 else 1
    rad = lambda m, k: 0.0 if k == 14 else 2.25e-8 * m ** 0.8   # pc (R_sun * M^0.8)
    r, v = kepler_state(m1, m2, a, e, inc, node, peri, M)
    M12 = m1 + m2
    bodies = [[m1] + [-m2 / M12 * c for c in r] + [-m2 / M12 * c for c in v],
              [m2] + [m1 / M12 * c for c in r] + [m1 / M12 * c for c in v]]
    # passer: straight-line start so that closest approach falls at mid-run (gravitational focusing ignored)
    d0 = vinf * PC_KMS_MYR * 0.5 * total_myr
    bodies.append([m3, -d0, b, 0.0, vinf, 0.0, 0.0])
    Mt = M12 + m3
    cm = [sum(bd[0] * bd[k] for bd in bodies) / Mt for k in range(1, 7)]
    out = []
    for bd in bodies:
        k = kw(bd[0])
        out.append([bd[0]] + [bd[c] - cm[c - 1] for c in range(1, 7)] + [k, rad(bd[0], k)])
    desc = dict(m1=m1, m2=m2, bh=bh, a_au=a / AU, e=e, m3=m3, b_over_a=b / a, vinf=vinf)
    return out, desc


def family_triple(u, total_myr):
    """unit cube point (13) -> hierarchical triple: inner binary (m1, m2, a_in, e_in) and an outer body (m3) on an
    orbit about the inner c.o.m. (a_out = ratio x a_in, e_out), mutual inclination and phases"""
    m1, m2, m3 = loguni(u[0], 0.1, 100.0), loguni(u[1], 0.1, 100.0), loguni(u[2], 0.1, 100.0)
    bh = u[3] > 0.5
    a_in = loguni(u[4], 1e-2, 1e2) * AU
    e_in = min(u[5], 0.99)
    ratio = loguni(u[6], 2.0, 1e4)
    e_out = min(u[7], 0.95)
    inc = math.acos(1 - 2 * u[8])
    M_in, M_out, peri_in, peri_out = (2 * math.pi * u[k] for k in (9, 10, 11, 12))
    kw = lambda m: (14 if bh else 1) if m >= 5.0 else 1
    rad = lambda m, k: 0.0 if k == 14 else 2.25e-8 * m ** 0.8
    ri, vi = kepler_state(m1, m2, a_in, e_in, 0.0, 0.0, peri_in, M_in)
    ro, vo = kepler_state(m1 + m2, m3, ratio * a_in, e_out, inc, 0.0, peri_out, M_out)
    M12, Mt = m1 + m2, m1 + m2 + m3
    cin = [-m3 / Mt * c for c in ro]; vcin = [-m3 / Mt * c for c in vo]
    bodies = [[m1] + [cin[k] - m2 / M12 * ri[k] for k in range(3)] + [vcin[k] - m2 / M12 * vi[k] for k in range(3)],
              [m2] + [cin[k] + m1 / M12 * ri[k] for k in range(3)] + [vcin[k] + m1 / M12 * vi[k] for k in range(3)],
              [m3] + [M12 / Mt * c for c in ro] + [M12 / Mt * c for c in vo]]
    out = [bd + [kw(bd[0]), rad(bd[0], kw(bd[0]))] for bd in bodies]
    desc = dict(m1=m1, m2=m2, m3=m3, bh=bh, a_in_au=a_in / AU, e_in=e_in, ratio=ratio, e_out=e_out,
                inc_deg=math.degrees(inc))
    return out, desc


FAMILIES = {"binflyby": (12, family_binflyby), "triple": (13, family_triple)}


def energy(bodies):
    T = sum(0.5 * b[0] * (b[4] ** 2 + b[5] ** 2 + b[6] ** 2) for b in bodies)
    U = 0.0
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            U -= G * bodies[i][0] * bodies[j][0] / math.dist(bodies[i][1:4], bodies[j][1:4])
    return T + U


def evaluate(args):
    u, cfg = args
    fam = FAMILIES[cfg["family"]][1]
    bodies, desc = fam(u, cfg["total_myr"])
    with tempfile.TemporaryDirectory(dir=cfg["tmp"]) as d:
        with open(Path(d, "ic.txt"), "w") as f:
            f.write("# adversary\n")
            for b in bodies:
                f.write(" ".join(repr(x) for x in b[:7]) + f" {int(b[7])} {b[8]!r}\n")
        conf = cfg["base"] + [f"IC_FILE = {Path(d, 'ic.txt')}", f"TOTAL_TIME_MYR = {cfg['total_myr']}",
                              f"E_REF = {cfg['eref']!r}", f"NGB_MREF = {cfg['nmref']!r}"]
        Path(d, "config.txt").write_text("\n".join(conf) + "\n")
        try:
            o = subprocess.run([cfg["exe"], "--config", "config.txt"], cwd=d, capture_output=True, text=True,
                               timeout=cfg["timeout"], env={**os.environ, "OMP_NUM_THREADS": "1"})
            out, rc = o.stdout, o.returncode
        except subprocess.TimeoutExpired:
            return dict(u=u, desc=desc, score=None, why="timeout")
    if rc != 0:
        return dict(u=u, desc=desc, score=None, why=f"exit {rc}: {out[-300:]}")
    de = re.findall(r"^\|dE/E0\| = (\S+)", out, re.M)
    dl = re.findall(r"^\|dL/L0\| = (\S+)", out, re.M)
    sub = [abs(float(x)) for x in re.findall(r"^SUBSYS_ERR step \d+ \S+ dE/E0 (\S+)", out, re.M)]
    e0 = abs(energy(bodies))
    de_c = float(de[-1]) * e0 / cfg["eref"] if de else 0.0
    sig = re.findall(r"Types:\s+bin=(\d+)\s+fly=(\d+)\s+tri_s=(\d+)\s+tri_u=(\d+)\s+multi=(\d+)", out)
    integ = re.findall(r"Integration: LogH=(\d+).*?Secular=(\d+)", out)
    signature = (tuple(int(x) > 0 for x in sig[-1]) if sig else ()) + \
                (tuple(int(x) > 0 for x in integ[-1]) if integ else ())
    score = max(de_c, max(sub) if sub else 0.0)
    return dict(u=u, desc=desc, score=score, dE_cluster=de_c, subsys=max(sub) if sub else 0.0,
                dL=float(dl[-1]) if dl else 0.0, signature=list(signature))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", default="binflyby", choices=sorted(FAMILIES))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--random", type=int, default=2000)
    ap.add_argument("--bisect", type=int, default=200, help="boundary pairs to bisect")
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--pop", type=int, default=72)
    ap.add_argument("--total-myr", type=float, default=2e-3)
    ap.add_argument("--eref", type=float, default=2.71184797e8)    # 200k seed-13 run
    ap.add_argument("--nmref", type=float, default=0.40052478)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--legacy", action="store_true", help="binary without SUBSYS_ERR_TOL / STALL_DE_FACTOR (score: dE only)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tmp").mkdir(exist_ok=True)
    rng = random.Random(a.seed)
    dim = FAMILIES[a.family][0]
    base = ["TEST_CASE = particle_file", "INTEGRATOR = hhs", "COLLISIONAL = true", "PN_ORDER = 31",
            "SSE_ENABLED = false", "PRINT_INTERVAL = 1", "PROFILING = false", "DT_MAX_MYR = 1e-3",
            "STALL_FACTOR = 0"] + ([] if a.legacy else ["SUBSYS_ERR_TOL = 1e-30", "STALL_DE_FACTOR = 0"])
    cfg = dict(family=a.family, exe=str(Path(a.exe).resolve()), total_myr=a.total_myr, eref=a.eref, nmref=a.nmref,
               timeout=a.timeout, base=base, tmp=str(out / "tmp"))
    log = open(out / "evals.jsonl", "a")
    allr = []

    def run(points, tag):
        with mp.Pool(a.workers) as pool:
            res = pool.map(evaluate, [(p, cfg) for p in points])
        for r in res:
            r["tag"] = tag
            log.write(json.dumps(r) + "\n")
        log.flush()
        allr.extend(r for r in res if r["score"] is not None)
        best = max((r["score"] for r in res if r["score"] is not None), default=0.0)
        print(f"{tag}: {len(res)} evals, best {best:.3e}, overall {max((r['score'] for r in allr), default=0):.3e}",
              flush=True)
        return res

    # 1. random
    run([[rng.random() for _ in range(dim)] for _ in range(a.random)], "random")
    # 2. bisection across path boundaries
    pairs = []
    for _ in range(a.bisect * 20):
        p, q = rng.sample(allr, 2)
        if p["signature"] != q["signature"]:
            pairs.append((p, q))
        if len(pairs) >= a.bisect:
            break
    for it in range(8):
        mids = [[(x + y) / 2 for x, y in zip(p["u"], q["u"])] for p, q in pairs]
        res = run(mids, f"bisect{it}")
        new = []
        for (p, q), m in zip(pairs, res):
            if m["score"] is None:
                continue
            new.append((p, m) if m["signature"] != p["signature"] else (m, q))
        pairs = new
    # 3. evolution strategy from the worst
    sigma = 0.1
    for g in range(a.generations):
        elite = sorted(allr, key=lambda r: -r["score"])[:max(4, a.pop // 8)]
        kids = []
        for _ in range(a.pop):
            par = rng.choice(elite)["u"]
            kids.append([min(1.0, max(0.0, x + rng.gauss(0, sigma))) for x in par])
        res = run(kids, f"es{g}")
        frac = sum(1 for r in res if r["score"] is not None and r["score"] >= elite[-1]["score"]) / max(len(res), 1)
        sigma *= 1.2 if frac > 0.2 else 0.85
    # worst distinct cases
    top, seen = [], []
    for r in sorted(allr, key=lambda r: -r["score"]):
        if any(sum((x - y) ** 2 for x, y in zip(r["u"], s)) < 0.05 for s in seen):
            continue
        seen.append(r["u"])
        top.append(r)
        if len(top) >= 20:
            break
    for k, r in enumerate(top):
        d = out / "top" / f"{k:02d}"
        d.mkdir(parents=True, exist_ok=True)
        bodies, desc = FAMILIES[a.family][1](r["u"], a.total_myr)
        with open(d / "ic.txt", "w") as f:
            f.write(f"# adversary {a.family} score {r['score']:.3e} {json.dumps(desc)}\n")
            for b in bodies:
                f.write(" ".join(repr(x) for x in b[:7]) + f" {int(b[7])} {b[8]!r}\n")
        (d / "config.txt").write_text("\n".join(base + [f"IC_FILE = {(d / 'ic.txt').resolve()}",
                                                        f"TOTAL_TIME_MYR = {a.total_myr}", f"E_REF = {a.eref!r}",
                                                        f"NGB_MREF = {a.nmref!r}"]) + "\n")
        print(f"top {k:02d} score {r['score']:.3e} dE {r['dE_cluster']:.2e} subsys {r['subsys']:.2e} dL {r['dL']:.1e} "
              f"sig {r['signature']} {json.dumps({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in r['desc'].items()})}")


if __name__ == "__main__":
    sys.exit(main())
