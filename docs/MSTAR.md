# MSTAR subsystems (optional module)

RUMI++ integrates subsystems (tight groups found by the neighbour search) with LogH + GBS, which holds at most
`LOGH_MAX_BODIES = 8` bodies. Larger groups (dense cores, an IMBH with its cusp, runaway-collision regions) used to be
split along their bound pairs or left to the FSI. With the optional **MSTAR module** such groups are integrated as one
regularised subsystem by the KETJU integrator (MSTAR: Rantala et al. 2020, MNRAS 492, 4131; Mannerkoski et al. 2023,
MNRAS 524, 4062) or its Kokkos port, on the GPU for large groups.

## Licence

The module (`src/mstar`) is **GPLv3** (it contains the KETJU library). It is compiled only with `-DENABLE_MSTAR=ON`
(default `OFF`); **a binary built with it is distributed under the GPLv3.** The BSD-3 core builds and runs without it
and reaches it only through `src/collisional/mstar_dispatch.h` (BSD). Cite the two papers above when you use it
(`src/mstar/README.md`, `CITATION.cff`).

## Use

```
cmake .. -DENABLE_COLLISIONAL=ON -DENABLE_MSTAR=ON ...
NBODY_MSTAR=1 ./nbody-kokkos --config config.txt
```

| switch | default | meaning |
|---|---|---|
| `NBODY_MSTAR` | off | `=1`: groups of `MSTAR_THRESHOLD` .. `NBODY_MSTAR_MAX` bodies are MSTAR subsystems |
| `MSTAR_THRESHOLD` (config) | 9 | smallest group for MSTAR (groups above 8 always use MSTAR when it is on) |
| `NBODY_MSTAR_MAX` | 1024 | largest group integrated as one subsystem (larger: split as without MSTAR) |
| `NBODY_MSTAR_MAX_STEPS` | 2e6 | GBS steps after which a call gives up (then retried with subdivided steps, as LogH) |
| `NBODY_MSTAR_TEAM_N` | 96 | groups up to this size run each GBS row inside one device team (one launch per step) |
| `NBODY_MSTAR_CROSSOVER` | unset | set: the KETJU C library on the host below this size, where no collision check / PN ledger / step limit is needed |

The step summary shows `mstar=n (dev d, gbs g, nmax m)`: integrations, of them on the device, their GBS steps and the
largest group. The GBS tolerance is `ETA_GBS`.

## What the subsystem path does

* Normalisation as for LogH (centre-of-mass frame, scales fixed on the first call, G = M = 1); PN bodies
  (`pn_body`) first; spins as angular momenta S = chi G m^2 / c (evolved and written back with the spin terms).
* PN: KETJU's terms (Thorne & Hartle 1985 1PN N-body with the three-body terms; Blanchet 2014 2-3.5PN for pairs;
  spin-orbit, spin-spin, spin evolution), switched by `PN_ORDER` among the PN bodies of the group.
* Collisions: after every accepted GBS step the module tests every pair against the collision rule of
  `collision_distance` (FROST-CLUSTERS I App. B3: R1 + R2, tidal radius 1.3 (M/m)^(1/3) R or ISCO, BH pairs 10 R_S or
  `BH_MERGE_RADIUS`): separation inside it, or approaching with the osculating pericentre inside it and reached within
  one step. The call stops there; the core merges the pair at its centre of mass (momentum exact, the change of T + U
  booked as for LogH collisions, remnant physics by `finalize_subsystem_collision`) and continues for the rest of dt.
  Unlike LogH there is no pericentre landing: the decision is made at the end of the step before the pericentre.
* Energy books: the module carries B = U - T as an extra GBS variable outside the error control and returns the
  extrapolated PN work; only that goes to `E_pn`. The rest of the members' T + U change (minus collisions) is
  integration error: it is **not** booked, so it shows in |dE/E|, in `SUBSYS_ERR` (kind `unbooked-mstar`) and in the
  `NBODY_EATTR` attribution. (LogH PN subsystems book their whole Newtonian change into `E_pn`.)

## Device implementation (module)

`src/mstar/mstar_kokkos.cpp`: all GBS rows of a step run concurrently on the device in lockstep (per substep stage
one force launch over the active rows and one kick+drift launch; small groups: every row inside one team, one launch
per step), tree coordinates rebuilt by root-to-vertex path sums, deterministic reductions, one device-to-host copy per
GBS step. Agreement with the C library: round-off (bitwise for two bodies); ctest `mstar_compare`. Scaling:
`mstar-bench` (results in `docs/CASES.md` C27).

## Tests

`ketju_unit`, `mstar_compare`, `mstar_vs_logh` (module); `mstar_big_group`, `mstar_coll_group`, `mstar_imbh_cusp`,
`mstar_dense_core` (cluster code with `NBODY_MSTAR=1`). IMBH + cusp initial conditions: `ADD_IMBH` with `IMBH_CUSP_N`,
`IMBH_CUSP_GAMMA`, `IMBH_CUSP_R_PC`, `IMBH_CUSP_M`.
