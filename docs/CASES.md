# Case register: special systems that break conservation or speed

Every system found to violate energy, momentum or angular momentum conservation beyond round-off, or to stall a
step, gets an entry: what the system is, where it was found, why it violates the conservation law, how it was
fixed, the isolated test that fails before the fix, and whether the fix improves E, P and L of whole cluster runs.
Goal (2026-10-16): the smallest long-run |dE/E| with P and L at double-precision round-off; speed only within that.

## How cases are found (the loop)

1. `scripts/campaign.py`: runs per seed with `STALL_RESTART = true`.
2. In the run: a step slower than `STALL_FACTOR` x the median, or with an energy change above `STALL_DE_FACTOR` x
   the median, prints STALL lines (costliest and least accurate LogH integrations, deepest set, 3+ body groups,
   largest tidal-sampling estimates) and keeps the restart of the state before it (`stall_<step>.h5`, replay
   bit-identical, test `stall_capture`).
3. `scripts/replay_ledger.py setup`: one-step replay of each jump with the energy ledger (`NBODY_ELEDGER`): error per
   HHS call and level, subsystem advances against the books, per-path residues.
4. `scripts/replay_ledger.py classify`: `booking:<path>` (a path changes its members' energy without booking it),
   `field:d<depth>` (the FSI error, with the tidal suspect), `merger`.
5. `scripts/cutout.py` / `harvest_stalls.py`: the system cut out of the restart (parent's `E_REF`, `NGB_MREF`) as an
   isolated case; it becomes a test in `tests/data/` if it reproduces.
6. Fix, test, A/B on seed ensembles (energy rule), entry here.

Status: **closed** (fixed, test, ensemble checked), **option** (fix kept as an option, trade-off), **open**,
**rejected** (a candidate fix that broke a conservation law).

---

## C1 Unbound fly-by group: tidal-sampling step without a floor (closed)

- **System:** run B (110k, dense young cluster), a 3-4 star unbound fly-by subsystem (0.005-0.12 pc, 12-48 km/s);
  field stars near it.
- **Symptom:** slow steps (levels to L22, step 2 > 55 min); isolated |dE/E| 3.8e-9 in one interval.
- **Cause:** the FLYBY_CM_EPS tidal-sampling step had a floor (one pericentre passage) only for bound binaries; for
  unbound pairs and groups it was 0, so a star 1400 AU from the group took 4e-9 Myr steps.
- **Fix:** 3cbb12e, floor = the shortest pair pericentre passage 2 pi r_p / v_p.
- **Test:** `flyby_group_floor` (86 stars): 7.1M secular steps, 69 s, |dE/E| 6.4e-9 before; 30k, 0.6 s, 4.4e-10 after.
- **Cluster:** B step 2 > 3300 s -> 1065-1282 s (3 seeds). 200k, 61 steps, 5 seeds: |dE/E| better on 4 (median
  6.0e-11 -> 3.0e-11), worse on 1 (one encounter, in both); |dL/L| <= 3.5e-12, |dP| at round-off.

## C2 Hard BH binary with a weak distant perturber in LogH (option)

- **System:** 200k, 10 % binaries, seed 3, step 48: 19.3 + 6.2 Msun BH binary, P = 1 h, e = 0.63, with a 1.25 Msun
  star passing at 600 AU (tidal ratio ~1e-15), grouped as an unstable triple.
- **Symptom:** 6.7M GBS steps, 200 s for one step (the binary integrated orbit by orbit for the whole fly-by).
- **Cause:** the passing star is a BIFROST Eq. 4 neighbour, so the triple goes to LogH; no slow-down for PN members.
- **Fix:** 1809570, `WEAK_SPLIT_GAMMA` (default off): below that tidal ratio the binary is its own subsystem and the
  star a field particle (Mikkola & Aarseth 1996; NBODY6 unperturbed KS pairs).
- **Test:** `weak_split_bh_binary` (5 particles, with 1e-6): no GBS steps (3.3M, 216 s without), |dE/E| 9.4e-12,
  |dL/L| 1.3e-14.
- **Cluster:** that step 200 s -> 5.8 s, but its |dE/E| 1.1e-12 -> 3.4e-11 (the star's tide sampled by kicks). 200k, 61
  steps, 5 seeds (with C3): better than C3 alone on 4, worse on 1; B seed 13 worse. Kept off by default.

## C3 Massive binary with distant light members: tidal-sampling bound 400x too large (closed)

- **System:** run B, seed 42, step 4: 12.7 + 12.6 Msun binary with two light members at 0.01 pc (also 137 + 135 Msun,
  e = 0.93, with a 0.85 Msun member at 1400 AU).
- **Symptom:** nearby field stars at L18, step 545 s.
- **Cause:** the 3+ body tidal-sampling bound 2 max|dr| x sum m |dv| took the extent from a distant member and the
  speed from the tight pair.
- **Fix:** f394c8a, the exact binary-tree (Jacobi) decomposition sum_nodes mu r v (same guarantee, tighter).
- **Test:** no fast isolated case yet (0.1 pc cut-out reproduces: 1138 s -> 60 s on one core, too slow for ctest).
- **Cluster:** B step 4 544 s -> 29 s. B 4 steps x 3 seeds 778-1222 s -> 88-109 s, |dE/E| better or equal on all 3;
  200k 5 seeds better on 4, equal on 1. In the 0.1 pc cut-out |dE/E| 12-18 % worse (within the per-step tolerance);
  FLYBY_CM_EPS 5e-12 recovered it there but not in the ensemble (worse on 4 of 8), so not adopted.

## C4 Secular triple: energy of the rebuild not booked (closed)

- **System:** 200k seed 13, step 60 (and every stable hierarchical triple on the secular path): a PN triple (BH binary
  + light star) and other secular triples.
- **Symptom:** energy jump -8.0e-10 |E0| in one step (the run's long-run |dE/E| 7.5e-10); the isolated SEC_TRIPLE test
  case lost 1.4e-4 over 10 steps, unchecked until then.
- **Cause:** found with the ledger: subsystem advances -8.01e-10 unbooked, secular binaries 6e-15, secular triples
  -8.015e-10. The path booked only the inner binary's GW loss; the orbit-averaged step and the rebuild from elements
  (point-mass outer orbit, re-placed phases) change the triple's true T + U. Isolated cut-outs of the merger and of
  the PN triple were exact, which hid it (they were not this path or not this state).
- **Fix:** a3719d9, energy-exact rebuild: the difference goes into the radial component of the outer (else inner)
  relative velocity; positions unchanged, P and L exact, T changes by exactly the difference.
- **Test:** `sec_triple_energy`: max |dE/E| 1.4e-4 -> 9.3e-16, |dP|/M 4.3e-15 km/s; |dL/L| 8.4e-13 (the rebuild rotates
  the inner orbit; 3.9e-13 before, bounded).
- **Cluster:** subsystem paths book to 7e-14 per step. 200k seed 13: 7.5e-10 -> 2.7e-10 (secular triples off: 2.4e-11);
  B seeds 42 / 7 / 13: 2.0e-10 -> 1.5e-10, 4.2e-10 -> 4.3e-10, equal. Runs without secular triples bit-identical.

## C5 Perturbed binaries sampled by the field kicks (open)

- **System:** B seed 42, step 14: 25.5 + 35.3 Msun binary (a = 64 AU, e = 0.40, P = 6.6e-5 Myr) perturbed by a 1.5 Msun
  star at 640 AU (tidal ratio up to ~1e-4); 200k seed 13 step 60 after C4: a 13.5 + 13.8 Msun binary whose field step
  is 2.7 periods.
- **Symptom:** energy jumps -3.1e-9 (B) and -3.2e-10 (200k) in one step, made in the FSI (ledger: own errors in sets of
  1e4-1e5 particles at depths 3-6), subsystem books balanced.
- **Cause:** the perturber's tide on the binary is applied by kicks at the binary's field step (3-4 per orbit); the
  tidal-sampling step is floored at one pericentre passage, so FLYBY_CM_EPS cannot hold it (tighter tolerance: no
  change). The perturber does not join the subsystem (it fails the arrival-within-the-step test, not the tidal one:
  NGB_MASS_GAMMA 1e-5 / 1e-6 changed nothing).
- **Tried:** no floor: the binary's estimate 170x smaller, the step's jump -45 %, 9x the cost; floor x 0.25: -32 %.
- **Next:** count the binaries per jump with the STALL tidal lines over the campaign; literature (BIFROST sec. 6 lists
  slow-down for perturbed binaries as missing; PeTar/SDAR perturbers in the regularised step; NBODY6 KS perturber
  lists down to gamma ~1e-6), then a method proposal.

## R1 FLYBY_CM_KICK, c.o.m. kicks of weakly perturbed binaries (rejected)

- **Tried on:** the C2 step. |dL/L| 1.3e-14 -> 1.4e-9, 15 % faster. Deleted in 1483094 (the isolated tests had stayed
  at round-off: a path inconsistency in the cluster, not debugged).
