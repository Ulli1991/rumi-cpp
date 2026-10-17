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
5. `scripts/subsys_cases.py`: per-subsystem error events (SUBSYS_ERR lines) replayed in isolation;
   `scripts/adversary.py`: a search over few-body configurations for the worst violations (complements the campaigns:
   what can happen vs what does happen, see docs/ADVERSARY.md).
6. `scripts/cutout.py` / `harvest_stalls.py`: the system cut out of the restart (parent's `E_REF`, `NGB_MREF`) as an
   isolated case; it becomes a test in `tests/data/` if it reproduces.
7. Fix, test, A/B on seed ensembles (energy rule), entry here.

## Lessons (what the cases teach; check new code against these)

0. **Check formulas against each other, not only runs.** A wrong sign in the PN1 energy (C7) survived because the
   check it would have failed was switched off as "impossible". Enzyme gradients turn "equations of motion and
   energy must agree" into a direct test (tests/enzyme).

1. **Book from the states, never from a model of them.** Every path must book exactly the change of its members'
   T + U recomputed from positions and velocities. Booking from orbital elements or an approximate model (C4: only the
   inner binary's GW loss of a secular triple) hides the model's error inside the dynamics.
2. **An approximation must conserve what the real dynamics conserves.** A rebuild from averaged or approximate
   variables has to restore E, P and L explicitly (C4 energy, C6 angular momentum: a frozen outer orbit while the
   inner orbit's L changes). Put the correction where it touches no other invariant (radial velocity for E, the outer
   orbit's L vector for L).
3. **Isolated cut-outs can hide state-dependent bugs.** C4 did not show in cut-outs (fresh subsystem state, own
   c.o.m. frame, different path choice); the replay of the run's own restart with the ledger found it. Use the
   ledger on the run's state first, cut-outs second.
4. **A weakly coupled test case can pass a broken path.** The secular-triple test showed |dL/L| 8e-13, taken for
   rounding; a strongly coupled triple showed 1e-2 (C6). Tests of a path must include its strongly coupled corner;
   the adversarial search finds it.
5. **A bound that is safe can still be far too pessimistic**, and pessimism costs levels (C3: extent of one member
   times the speed of another, 400x). Derive bounds from the exact decomposition.
6. **A floor or cutoff that is right for one class breaks another.** The pericentre-passage floor of the
   tidal-sampling step fixed C1 and is the limit in C5; a missing floor caused C1. Every floor needs the class it is
   meant for written next to it.
7. **More neighbours is not more accuracy.** Pulling perturbers into subsystems by a larger radius made C5 slower and
   worse (2026-10-17).
8. **Check L and P, not only E.** C6 and R1 conserved energy and broke angular momentum.
9. **Books cannot catch a wrong force.** A force whose work is booked (PN orders into E_pn / E_gw) can have any
   sign or coefficient and |dE/E| stays at round-off (C8: radiation reaction with the wrong sign since 2026-02).
   Check every force term against the literature by a physical identity it must satisfy (flux balance, Kepler's law,
   test-mass limit, energy consistency), order by order, both mass limits.

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
- **Tried (2026-10-17):** perturbers into the subsystem: R_NGB 2e-3 / 3e-3 pc with NGB_MASS_GAMMA 1e-6: the step 6x slower,
  its error larger (-6.0e-9 / -3.6e-9 vs -2.2e-9 cumulative), and the binary still sampled by the kicks at 3e-3 pc.
  Not the answer in this form.
- **Next:** count the binaries per jump with the STALL tidal lines over the campaign; literature (BIFROST sec. 6 lists
  slow-down for perturbed binaries as missing; PeTar/SDAR perturbers in the regularised step; NBODY6 KS perturber
  lists down to gamma ~1e-6), then a method proposal.

## C6 Secular triple: angular momentum of the frozen outer orbit (closed)

- **System:** any strongly coupled stable triple on the secular path; found by `scripts/adversary.py` (family triple,
  smoke test of 8 random points): 0.12 + 0.46 Msun inner binary, a = 0.086 AU, e = 0.23, 2.06 Msun outer body at
  a_out = 12.9 a_in, e_out = 0.46, 65 deg (Kozai regime).
- **Symptom:** |dL/L| 1.26e-2 in one step (|dE/E| 4e-16 after C4, P exact); with the code before C4 also |dE/E| 5.8e-3.
  The existing test triple (weakly coupled) showed 8.4e-13, taken for the rebuild's rounding.
- **Cause:** the secular step changes the inner orbit's angular momentum (e_in, plane) but kept the outer orbit
  frozen, so mu_in r_in x v_in + mu_out r_out x v_out changed.
- **Fix:** the outer orbit carries what is left, L_out = L(before) - L_in(new): direction -> i_out, Omega_out, magnitude
  -> e_out at the unchanged a_out; then the C4 energy fix (radial, leaves L alone). E, P, L exact.
- **Test:** `adv_kozai_triple`: max |dE/E| 4.4e-16, |dL/L| 1.2e-15 (old code: 5.8e-3, 1.26e-2); `sec_triple_energy`
  |dL/L| 8.4e-13 -> 6.2e-16 (bound tightened to round-off).
- **Cluster:** pending (the next campaign; runs without secular triples are unchanged).

## C7 PN1 energy: wrong sign (B1, closed) and no N-body EIH for n >= 3 (B2, closed)

- **Found by:** Enzyme (tests/enzyme/pn_consistency.cpp), not by a conservation run: under the code's PN1 equations
  of motion the energy T + U + E_pn1 must be conserved to O(1/c^4); dE/dt = grad_x E . v + grad_v E . a with the
  exact gradients of `pn1_energy` scaled as c^-2 (doubling c divided it by 4, not 16), for 2, 3 and 4 bodies.
- **B1 (closed):** the G^2 potential-potential term of `pn1_energy` had the wrong sign (-mu (GM/r)^2 / 2 instead of +,
  Blanchet 2014 eq. 233). With the sign flipped two bodies are consistent to exactly c^-4 (ratio 16.00, residual 1e-4
  of the 1PN terms): the acceleration was right, the energy wrong. Effect: `pn1_energy` is used only in the LogH
  energy check of PN subsystems, which is then overwritten with 0 ("no energy this check can hold"), so no effect on
  the dynamics, but the check could never have worked. Test `pn1_energy_consistency` (central differences, gcc):
  ratio 16.00 for 20 binaries (4 with the old sign).
- **B2 (open):** for n >= 3 the ratio stays ~4: acceleration (LogH `compute_pn_all_pairs`) and energy are pair sums of
  the 2-body EIH formulas; the genuine three-body terms of the N-body EIH dynamics (sums over a third body C in the
  A-B interaction, in acceleration and energy, Poisson & Will 2014 sec. 9.3) are missing, so PN subsystems with 3+
  bodies follow a model without a conserved 1PN energy (documented as "EIH N-body"). Small for hierarchical triples
  (terms ~ (m_C/m_B)(r_AB/r_AC)), a full 1PN effect for BH triples with comparable separations.
- **B2 fix:** `eih1_nbody_acceleration` (Poisson & Will 2014 eq. 9.127, three-body terms included) for the PN1 part of
  LogH subsystems with 3+ bodies (higher orders stay pairwise, binaries unchanged), and the three-body G^2 terms in
  `pn1_energy` (from the EIH Lagrangian, Landau & Lifshitz sec. 106; none for n = 2). Checked with Enzyme two ways:
  energy consistency ratio 16.00 for n = 2, 3, 4 (20 configurations each), and the explicit acceleration against the
  Euler-Lagrange acceleration of the EIH Lagrangian (deviation 4-7e-4 of the PN1 part = O(v^2/c^2) at v/c ~ 0.01).
  Test `pn1_energy_consistency` now requires 16 for n = 2, 3, 4 (pair sums: ~4). BIFROST includes these three-body
  terms (sec. 3.7, conclusions); our code claimed them but had pair sums.
- **Next:** the PN energy check of PN subsystems can be switched on again for PN1 (E_N + E_1PN conserved to
  O(v^4/c^4)); higher orders need their energies (2PN) and the radiated flux (2.5PN).

## C8 PN2, PN2.5, PN3, PN3.5 of the subsystem equations of motion wrong since 2026-02 (closed; secular rates open)

- **Found by:** checking each order against Blanchet, Living Rev. Rel. 17, 2 (2014, v3 2024; on disk
  papers/pn/src/1310.1528), after Enzyme had found C7: the work of the dissipative orders on a circular binary
  against the GW flux (balance), and the conservative orders against Kepler's third law to 3PN (eq. keplerlaw) and
  the c.o.m. coefficients A_2PN, B_2PN. Not visible in any conservation run: the work of the PN forces goes into the
  books (E_pn, E_gw), so |dE/E| stayed at round-off while the binaries followed wrong equations.
- **System:** every PN subsystem integrated by LogH (BH binaries, BH triples) with PN_ORDER >= 2. PN_ORDER defaults
  to 7 (1PN + 2PN + 2.5PN); the campaigns and adversary runs use 31. The secular path (Peters rates) and the global
  PN1 kick (PN1 only) were not affected.
- **Bugs (all in `post_newtonian_core.cpp`, single-pair and pair version alike):**
  - B1 2.5PN: opposite sign. Radiation reaction **added** the Peters power (work / Peters = -1.000): LogH binaries
    widened instead of inspiralling (e = 0.5 BH binary: da/a from 2.5PN +1.3e-5, now -1.3e-5).
  - B2 3.5PN: coefficients not Blanchet's A_3.5PN, B_3.5PN; on a circular orbit +11 x the 2.5PN loss (q = 1),
    should be -(1247/336 + 35/12 nu) x of it. With PN_ORDER 31 the sum was a loss of ~10 x Peters: the ISCO test binary
    "merged" in its first step at a = 1.3e-10 pc, e = 0.98 (15 x its start).
  - B3 2PN: the G^2 terms had an extra 1/r (dimension length^-1, Gm_r * Gm_r * inv_r), and two velocity terms of the
    v-coefficient were wrong (+4 v_A^2 (n.v_A) - 5 v_B^2 (n.v_A) instead of +4 v_B^2 (n.v_A) - 5 v_B^2 (n.v_B),
    Blanchet eq. acc3PN). LogH works in units with R_scale ~ a, so the extra 1/r scaled those terms by ~a/r (O(1),
    largest at pericentre), not by 1e5-1e7 as in pc.
  - B4 3PN: incomplete coefficients (several terms missing, same 1/r slip).
- **Fix:** 2PN corrected in place (general frame, Blanchet eq. acc3PN). 2.5PN, 3PN, 3.5PN: Blanchet's c.o.m. relative
  acceleration a = -(G M / r^2)(A n + B v) (eqs. calAcoeff, calBcoeff; 3PN in modified harmonic coordinates, eq.
  ABcoeffMH, no ln r'_0), computed once per pair (`pn_rel_highorder`), body 1 gets (m_2/M) a, body 2 -(m_1/M) a:
  sum m a = 0 exactly (momentum to round-off), depends only on relative position and velocity (Galilean
  invariant, so the conservative 3PN part keeps its energy in any frame). The pair call stays bit-identical to the
  two single calls (NBODY_CHECK_EIH).
- **Test:** `pn_highorder` (tests/pn_highorder_test.cpp): 2.5PN work / Peters = 1 to 1e-15 (q = 1, 0.2); 1PN flux
  coefficient from the 2.5PN + 3.5PN work -4.44046 / -4.11639 (Blanchet -4.44048 / -4.11640); Kepler residual drops
  by 8.1 (PN1 + 2) and 16.2 (+ 3PN) when gamma halves (q = 1, 1e-6); PN2 with r' != 0 equal to A_2PN, B_2PN to 1e-13.
  The old code fails all 10 checks. `coll_bh_isco` / `bh_merge_radius_*` now use PN_ORDER 7 (see below);
  `pn_boost_logh` bound 8e-12 -> 1.6e-11 (round-off noise, not monotonic in the boost).
- **Open (a), strong field:** truncated PN at x ~ 0.1. On an eccentric orbit with pericentre at 9 G M / c^2 the 3.5PN
  term exceeds the 2.5PN one with the opposite sign (B_3.5 / B_2.5 = -12.4 x at v^2 = 2 G M / r): radiation reaction
  pumps energy at every pericentre; the ISCO test binary widened ~80 x with a star at 100 a and unbound with the star
  at 1000 a (books closed, |dE/E| 5e-16). Correct implementation of the formula outside its range. Options: 3.5PN
  only below some x, or the orbit-averaged correction; decide with the literature (what BIFROST / ARCHAIN do).
- **Open (b), secular rates (`secular.cpp`):** the 3.5PN correction of da/dt, de/dt uses Gm/(a c^2) where nu belongs
  and the flux coefficient -1247/336 where da/dt needs its own (-1751/336 - 7/4 nu for circular orbits, harmonic a;
  eccentric: 1PN-accurate orbit averages from the literature); the 2PN / 3PN periapsis advance has no nu terms.
  Size ~x << 1 of the leading terms in the secular regime.
- **Cluster effect:** all LogH BH binaries of runs with PN_ORDER >= 4 since 2026-02 had wrong inspirals (widening with
  7, ~10 x too fast with 31); runs with PN binaries (flagship A, BIFROST Fig. 12) have to be repeated. The energy
  error of the runs is unchanged (books); campaign case harvesting is unaffected.

## Verified building blocks (checked against an independent derivation, no bug found)

| Building block | Check | Result | Test |
|---|---|---|---|
| FSI gradient force (GradK, FROST Eq. 19) | correction vs (1/m_i) d/dx_i of 1/2 sum m &#124;a&#124;^2 (Enzyme, 2-32 bodies); sign by the measured order | exactly GRAD_COEFF dt^2 (12 digits); order 4.64 | `grad_force_consistency`, `kepler_e09` |
| PN1 EIH acceleration, 2 bodies | energy consistency, Euler-Lagrange of the EIH Lagrangian (Enzyme) | c^-4; O(v^2/c^2) | `pn1_energy_consistency` |

## R1 FLYBY_CM_KICK, c.o.m. kicks of weakly perturbed binaries (rejected)

- **Tried on:** the C2 step. |dL/L| 1.3e-14 -> 1.4e-9, 15 % faster. Deleted in 1483094 (the isolated tests had stayed
  at round-off: a path inconsistency in the cluster, not debugged).
