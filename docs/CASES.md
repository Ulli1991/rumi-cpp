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

## C5 Perturbed binaries sampled by the field kicks (option)

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
- **Fix (2026-10-18, option `TIDAL_FLOOR_ADAPT`):** per bound binary, from its local estimate in subsystem_timesteps:
  est = mu |T| v_p f (T the tidal acceleration of all perturbers now, f = TIDAL_STEP_RES P (1 - e)^(3/2) the floor)
  against its share FLYBY_CM_EPS |E0|; above it the floor becomes f sqrt(share / est) (below the floor the kicks
  resolve the orbit and the splitting error is the bounded dt^2 shadow term). Binaries within their share are
  untouched; no global feedback. Literature: KS-regularised perturbed binaries in NBODY6 get their perturbers' forces at
  every KS step (~30 per orbit, perturbers down to gamma ~1e-6), i.e. the perturbed orbit is resolved; pulling the
  perturber into the subsystem (SDAR/PeTar style) was slower and worse here (above). A first variant that scaled the
  floor from the pair bound k r^3 instead of the measured tide was 6x pessimistic (144x the LogH calls in the test).
- **Test:** `tidal_floor_bh_binary` (3 particles from the s13 step-56 restart: 45 + 10.8 Msun BH binary, a = 28 AU,
  e = 0.50, 1.49 Msun star at 1300 AU): the ledger reproduces the cluster's own errors to 4 digits (6.60e-11 |E0|).
  |dE/E| 2.28e-6 -> 2.36e-9, LogH calls 1024 -> 6308; global FLYBY_CM_EPS 1e-12 + RES 0.25: 7.3e-9.
- **Cluster (200k, 61 steps, 5 seeds, final |dE/E| base -> fix):** s3 7.9e-12 -> 2.5e-11, s7 5.1e-12 -> 1.8e-12,
  s13 6.6e-11 -> 2.9e-11 (largest step 6.55e-11 -> 1.3e-11), s21 3.2e-11 -> 1.5e-10, s42 3.1e-11 -> 1.7e-11; rms per
  step better on 3, median s/step within 2 %. Kept off by default (mixed ensemble). s21 worse (3.2e-11 -> 1.5e-10) by trajectory divergence from step 3: its jump
  steps 36/37 replayed from the fix run's own restarts give identical errors with the switch on and off (a 45 + 23
  Msun binary, P = 3.3e-7 Myr, field step 11 P, estimate below its share; open). |dL/L| better on s3/s13/s21.
  FLYBY_CM_VELOCITY s7 step 1 (-3.7e-10, sets of 150-360 at depth 4) is this case: the fix gives +1.9e-12; the
  FLYBY_CM_VELOCITY s21 step 38 (-1.55e-9) is a BH-BH GW coalescence ("merger inside", own errors -2.5e-12): not C5.
  With fix + FLYBY_CM_VELOCITY the remaining jumps (s3 step 26 -1.1e-10, s42 step 35 +5.2e-10) are also GW_COAL steps,
  identical with the switch on and off; the same coalescences happen without a jump in the runs without
  FLYBY_CM_VELOCITY (open: energy around a coalescence with c.o.m. fly-by velocities).

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

## C8 PN2, PN2.5, PN3, PN3.5 of the subsystem equations of motion wrong since 2026-02 (closed)

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
  **Closed (2026-10-17):** no established code caps or tapers 3.5PN. All merge BH pairs at a fixed separation and
  call PN unreliable inside it: BIFROST sec. 3.14.1 ISCO, with 10 R_S as the conservative option; KETJU-GADGET4 12 R_S;
  Mannerkoski+2019 and Rantala+2025 ~10 R_S; Brem+2013 (NBODY6 PN) 5 R_S; NBODY7 4 R_S. Orbit-averaged, 2.5 + 3.5PN
  flips sign at pericentre x_p = 0.11-0.14 (e = 0.5-0.9; 0.21 circular with the 1PN energy). The old distance
  6 G M / c^2 (x = 1/6) lies beyond that. Now `collision_distance` merges BH pairs at 10 R_S = 20 G (m1 + m2) / c^2
  (x = 0.05; BH_MERGE_RADIUS = max_single keeps FC-I's 6 G max(m) / c^2), and the secular path does the same at
  pericentre. The existing t_gw < dt merger stays. `coll_bh_isco` / `bh_merge_radius_total` start at 1.5 x 10 R_S with
  PN_ORDER 31 and merge (r_coll 1.9147e-11 pc). At 6 G M / c^2 the same run does not finish in 120 s. The max_single
  test keeps PN_ORDER 7. With 3.5PN the binary gains energy before x = 1/3, and that run also breaks the books
  (|dE/E| 1.6e-3, |dL/L| 5.7 in 8e-12 Myr): integration outside the PN range, which the 10 R_S rule now avoids.
- **Open (b), secular rates (`secular.cpp`), closed as C9:** the 3.5PN correction of da/dt, de/dt uses Gm/(a c^2) where nu belongs
  and the flux coefficient -1247/336 where da/dt needs its own (-1751/336 - 7/4 nu for circular orbits, harmonic a;
  eccentric: 1PN-accurate orbit averages from the literature); the 2PN / 3PN periapsis advance has no nu terms.
  Size ~x << 1 of the leading terms in the secular regime.
- **Cluster effect:** all LogH BH binaries of runs with PN_ORDER >= 4 since 2026-02 had wrong inspirals (widening with
  7, ~10 x too fast with 31); runs with PN binaries (flagship A, BIFROST Fig. 12) have to be repeated. The energy
  error of the runs is unchanged (books); campaign case harvesting is unaffected.

## C9 Secular PN binaries and triples: 3.5PN rates, precession and mean motion (closed)

- **Found by:** C8 open (b), then checking the secular path against the equations of motion that C8 had verified.
- **System:** isolated PN binaries evolved by `integrate_secular_binary` (BH binaries too tight for LogH) and the inner
  orbit of secular triples, any PN_ORDER with PN1; 3.5PN rates with PN_ORDER & 16 (31 in campaigns).
- **Bugs:**
  - B1 3.5PN rates: Peters times -(1247/336 + 35/12 G m / (a c^2)) G m / (a (1 - e^2) c^2): the energy-flux
    coefficient used for da/dt and de/dt alike, G m / (a c^2) where nu belongs, no eccentricity terms. In mean-motion
    a and e_t the 1PN coefficient of da/dt is +3.1 (e = 0, nu = 0.25), +10.0 (e = 0.5), +72 (e = 0.9); the old one
    -3.7, -5.0, -19.5: opposite sign, 4x too small at e = 0.9.
  - B2 the correction is only defined in variables that are constant along the PN orbit. Osculating Newtonian a, e
    of the state oscillate at O(x) around the orbit, the same size as the correction, so the old rates had an O(x)
    error from the entry phase as well.
  - B3 periastron advance: 2PN and 3PN terms of a test mass (no nu), in osculating elements.
  - B4 mean motion: the Kepler drift advanced the orbit with the Newtonian n of the osculating a; the PN n differs at
    O(x). Pericentre time and angle drifted by 1.3e-3 rad per orbit (gamma = G m / (a c^2) = 1e-3, e = 0.5, q = 0.5).
- **Why E did not show it:** the Newtonian energy change of the secular step is booked as E_gw (lesson 9).
- **Fix (`secular.cpp`):** the state gives E, J per reduced mass to 2PN (Kidder 1995 eqs. 2.7-2.8, c.o.m.
  harmonic), eps = -2E/c^2, j = -2EJ^2/(Gm)^2 (gauge invariant). The aux-variable leapfrog evolves eps, j by the
  balance mu dE/dt = -<F>, mu dJ/dt = -<G> with the orbit-averaged fluxes in x, e_t to 1PN (Blanchet LRR eq. Finst,
  Arun et al. 2009 PRD 80 124018 eq. 4.5; x, e_t from eps, j at 1PN). Drift: pericentre angle by k n, mean anomaly by
  n, n and K = 1 + k from eps, j to 3PN (LRR eqs. n3PN, K3PN). After the Kepler drift and the rotation, the orbit is
  reshaped (a1, k e, M + dM) by a fixed point so that the state's eps, j are the evolved ones. Secular triple: inner
  (a, e) are read as the mean-motion a and e_t, with Arun et al. eqs. 6.6-6.9 for dn/dt and de_t/dt. PN2.5 alone is
  Peters, as before.
- **Test:** `secular_pn` (tests/secular_pn_test.cpp), all against the integrated equations of motion:
  - (1) radial period and pericentre advance of PN1 + 2 orbits: residual / 8 when gamma halves (e 0.5, 0.6; q 1, 0.25).
  - (2) <d eps/dt>, <d j/dt> under the 2.5 + 3.5PN acceleration along 1PN orbits: the 1PN corrections agree; the
    residual drops by 4.0 when gamma halves.
  - (3) triple closed forms equal the flux chain rule to O(x^2) (ratio 4.0); PN2.5 equals Peters to 1e-15.
  - (4) secular binary vs integrated orbit after 400 orbits: pericentre offsets 9.8e-4 P, 2.6e-3 rad, bounded (to
    3200 orbits). The old code: 2.5e-2 P, 0.50 rad, growing linearly.
- **Remaining, documented:** the Newtonian orbit shape differs from the PN orbit (e_t, e_r, e_phi differ at 1PN). So
  the secular state has O(x) periodic offsets in phase (up to ~20 x rad near pericentre) and none in the long term.
  The triple reads Newtonian elements as (a_n, e_t); exact to the leading order, O(x) ambiguity in the corrections.
- **Cluster effect:** secular BH binaries are orbit-averaged and their phase only matters for later encounters, so
  this is small for cluster statistics. Inspiral times at e ~ 0.9 change by O(70 x).

## C10 Weak-split BH binary case: |dL/L| ~2e-13 without PN (open)

- **Found by:** `weak_split_bh_binary` while closing C9. Its |dL/L| bound of 1.7e-14 came from one step. With DT_MAX
  1e-4 over the same 1e-3 Myr, L grows to 1.6e-13 with PN_ORDER 0, identically at 668c025 and with C9. With PN_ORDER
  31 the old code reaches 1.9e-13, the new code 2.8e-13.
- **System:** a 1 h BH binary with a star passing at 600 AU, split off by WEAK_SPLIT_GAMMA = 1e-6; the binary is
  advanced by the Kepler/secular path. dE/E ~ 2e-11 to 6e-11.
- **Why L jumps:** not known yet. The secular path books its members' P and L change from the states, so the
  candidates are the split itself (field kicks on the binary c.o.m. versus on the members) and the FSI kicks across
  the split.
- **Next:** a per-path L ledger (as NBODY_ELEDGER does for E) on this case.

## C11 Secular triple: octupole equations not Hamiltonian, frame-dependent angles, GW tilts the planes (closed)

- **Found by:** Enzyme. Hamilton's equations of the double-averaged quadrupole + octupole Hamiltonian (Naoz et al.
  2013 App. A), differentiated exactly, against the code's rates at 2000 random states each (tests/enzyme/
  secular_triple.cpp). The quadrupole agreed to 4e-10; the octupole did not.
- **System:** every stable hierarchical triple on the secular path (`secular_evolve_triple`,
  `secular_triple_step`) with unequal inner masses (octupole C3 ~ m1 - m2), and every PN triple (GW).
- **Bugs:**
  - B1 two octupole signs, taken over from the kozai package's derivative code while the code uses Blaes et al.
    2002's equations: cos phi = cos g1 cos g2 - cos i sin g1 sin g2 instead of -cos g1 cos g2 - cos i sin g1 sin g2
    (Blaes eq. 23, Naoz), and C3 with (m2 - m1) instead of (m1 - m2). The octupole rates were off by 0.3-1.5 of the
    total rates, and the averaged energy was not conserved by them (8.5e-2).
  - B2 de2/dt missing (the outer eccentricity was frozen); G2 fixed in cos i.
  - B3 GW tilted the orbits: H (total angular momentum) was held fixed while GW shrank G1, so cos i changed by
    0.3 dG1/G1 from GW alone. Blaes eq. 37: GW leaves the planes alone.
  - B4 frame dependence: g1, g2 came from `cartesian_to_elements`, measured from the node on the simulation xy-plane,
    not from the mutual node. Rotating the coordinate frame changed the inner eccentricity vector after one step by
    0.45.
- **Why the energy did not show it:** the step books its energy change (C4), and the secular energy of the
  octupole terms is small compared with the triple's orbital energy.
- **Fix (`secular.cpp`, `secular_triple_step.cpp`):** Blaes signs; de2/dt with e_out, G2, C2, C3 refreshed every
  substep. Each substep takes cos i from the conserved H after the Kozai-Lidov change, then moves H with the GW
  change at fixed cos i. g1, g2 are measured from the mutual node n1 = L_out x L_in (n2 = -n1). The rebuild turns
  L_in about n1 to the evolved inclination against L_tot and places the pericentres at g1, g2.
- **Test:** `secular_triple` (tests/secular_triple_test.cpp, plain C++):
  - rates vs Hamilton (long-double derivatives of the Hamiltonian in tests/secular_triple_ham.h): octupole now
    4e-8 to 6e-11;
  - dH_avg/dt along the rates 3e-10;
  - frame rotation 3.7e-15;
  - GW alone d cos i 1.5e-10 of dG1/G1;
  - 1PN precession 5e-14;
  - direct three-body RK4 run over 2 outer periods: de within 19 % (was up to 4.7x).
  9 checks fail on the old code.
- **Follow-up (same day, found by `coll_bh_isco`, a BH binary at 100 a with a star on the secular-triple path):**
  with the planes fixed under GW (B3), the rebuild still assumed the internal L is conserved (C6:
  L_out = L - L_in). It pushed the radiated angular momentum into the outer orbit. For a circular outer orbit no orbit
  at a_out carries that, so the rebuild failed every call, the energy fix found no radial velocity to use, and
  1.4e-7 of E went unbooked (|dL/L| 6.6e-2).
  - Fix: the secular step accumulates the GW part of G1 (`dG1_gw`). The rebuild keeps L + dG1_gw L_in_hat and
    measures the inner plane's angle against that total. The caller books the radiated L (L_sink, as E_gw).
  - Last energy fallback: when no radial velocity can take the energy (circular orbits), the outer orbit takes it in
    a_out. The inner plane turns about the mutual node so that |L_tot - L_in| is the outer angular momentum at that
    a_out (at fixed L a circular orbit cannot shrink otherwise), iterated to the round-off of the members' energy.
  - Result: `coll_bh_isco` |dE/E| 1.3e-14, |dL/L| 3e-16.
  - Lesson: a fix that makes a book physical (GW radiates L) breaks a correction that relied on the unphysical
    conservation. Check every correction downstream of a changed invariant.
- **Remaining:** the substep integrator is forward Euler (eta 0.01), first order, so slow drift is expected.
- **Cluster effect:** octupole (eccentric Kozai-Lidov) evolution, the channel to extreme eccentricities and GW
  mergers in triples, had wrong signs, and results depended on the orientation of the coordinate frame. Triple-driven
  merger rates from runs before this fix are not reliable.

## C13 Runtime SSE: 14.5 s of a 42 s interval at N = 1e6, unprofiled (closed)

- **Found by:** the BIFROST Fig. 7 rerun (fig7-2026-10-17). The unprofiled "Other" bucket was 32 % of an interval at
  N = 1e6, f_bin = 0.1, 45 % at 2e6 on 16 GPUs and 17-26 % at 1e4/1e5. With SSE_ENABLED = false it went away.
- **System:** every run with SSE_ENABLED = true. `advance_stellar_evolution` runs once per interval.
- **Measured (3e15078, 4 A100 ranks, 1e6 f = 0.1, per call):**
  - serial `SSEContext::evolve` loop over all stars on every rank: 9.1 s (9 us per star; evolve re-integrates the
    track from age 0);
  - E_sse from PE(before) - PE(after): two full unsoftened triangle sums, replicated on every rank, 2 x 2.7 s;
  - host mirrors and copies back: 0.03 s.

  Only 4179 of 1e6 stars (0.4 %) change mass per interval. At 1e4 and 1e5 it was 0.10 s and 0.98 s, almost all in
  the loop.
- **Fix:**
  - (a) The evolve loop runs as a Kokkos host-parallel for. With MPI each rank takes a contiguous slice and the
    results are allgathered (MPI_BYTE). evolve is a pure const function, so every rank applies the same bits.
  - (b) The potential change comes from the touched stars T alone:
    dU = -G sum_{i in T} d_i sum_{j != i} mbar_j / r_ij, with d = m' - m and mbar = (m + m') / 2. This is exact,
    because m'_i m'_j - m_i m_j = d_i mbar_j + mbar_i d_j, so there is no T x T pair term. It costs |T| N pairs on
    the device. Each rank sums a slice of T; the per-star terms are allreduced (one nonzero term each, so the sum is
    exact) and added in T order, giving the same bits for any rank count. Nothing cancels between two O(|U|) sums.
  - (c) New profiler bucket "Stellar evolution".

  The physics is unchanged: every star is still evolved every interval. FROST-CLUSTERS I (arXiv 2403.10602, "Interfacing SEVN with BIFROST") instead evolves
  each star only at its own SEVN step and freezes stars below 2.2 Msun. That would change the dynamics (radii), so it
  was not adopted.
- **Test:** `sse_books` (NATAL_MIX, winds + 89 NS + 27 BH). With NBODY_SSE_CHECK (debug) the booked dU is compared
  with the difference of two full triangle sums, to 1.8e-15 |U| (the reference's own round-off). It also checks
  max |dE/E| with E_sse at 5.7e-9 and the presence of the timer. It fails before the fix (no check rows, no timer).
  `rankcount_natal_kicks` (1 vs 2 ranks) stays bit-identical.
- **Dynamics:** bit-identical to 3e15078. NATAL_MIX and NATAL_NS dumps are identical on the host at 1 and 4 threads.
  In the GPU runs the per-interval |dE/E| rows agree to the 6th digit (only E_sse round-off differs).
- **Interval wall time** (4 A100, f = 0.1, 12 intervals, median, base -> new): 1e4 0.476 -> 0.387 s, 1e5 3.94 ->
  3.03 s, 1e6 41.8 -> 27.4 s. "Stellar evolution" is 0.009 / 0.031 / 0.24 s per call; at 1e6, 0.05 s of that is
  MPI.

## C14 Merger remnant keeps its progenitors' time step (GW recoil unresolved with FLYBY_CM_VELOCITY) (closed)

- **Found by:** the C5 A/B with FLYBY_CM_VELOCITY (tidfloor-2026-10-17): energy jumps exactly on BH-BH GW_COAL
  steps, ledger "merger inside": 200k seed 21 step 38 dE/E -1.55e-9; same coalescences without a jump when cmv is off.
- **Cause:** a merger detected in the time-step block left the remnant with its progenitors' time step although its
  mass and velocity changed (GW recoil ~110 km/s). With cmv that step is the binary's c.o.m. fly-by step (slow set,
  depth 0, 1e-3 Myr): the remnant crossed 0.1 pc in one unresolved step. Without cmv the members' internal steps were
  tiny, so the remnant landed in the fast set and was re-timed by chance. The merger booking itself was exact.
- **Fix (0a3ff1d):** after a merger at a level the set's neighbours, subsystems and time steps are computed again
  (a few times per run). Test `cmv_recoil_remnant` (79 bodies cut out of s21 stall_38): |dE/E| 3.44e-9 -> 7.19e-14.
- **A/B (200k, 61 steps, final |dE/E|; cmvcoal-2026-10-18):** cmv on, main -> fix: s3 2.7e-11 -> 2.0e-11, s7 4.0e-10
  -> 4.0e-10, s13 timeout at 59 both, s21 1.6e-9 -> 2.7e-11, s42 4.2e-11 -> 5.6e-11. cmv off, main -> fix (not
  bit-identical: remnants re-timed): s3 8.6e-12 -> 3.1e-14, s7 5.2e-12 -> 7.5e-12, s13 6.6e-11 -> 6.6e-11, s21
  3.2e-11 -> 1.8e-10, s42 3.1e-11 -> 2.5e-11. Merged as a correctness fix (user decision 2026-10-18) despite s21.
- **Open:** cmv stays off: s7 (C5-type jump at step 1, 77x base) and s3/s42 (1.8-2.3x) still fail the energy rule.

## C16 kepler2_contact: bound set from one lucky round-off sample (closed)

- **Found by:** `kepler2_contact` failing on GPU builds and gcc 15 host builds: max |dE/E| 1.353e-14 against 1.4e-15.
  It passes with gcc 13 (1.025e-15).
- **No regression:** the gcc 15 host build of 311eb01, where the test was added, gives 1.72e-14 (bound 1.6e-14,
  comment "measured 1.2e-14"). 806a676 changed the Kepler drift write-back. That moved the state handed to LogH by
  a few ulps, gcc 13 then happened to give 1.0e-15, and the bound was cut to 1.4e-15 on that one sample.
- **Where the error comes from:** the two Kepler drifts give 2e-16. LogH's dE is (E_before - E0) / |E0|, the GBS
  energy error up to the collision; the merger booking (dE_int into B and E_ref) cancels exactly. Per GBS step
  (normalised units, E = -0.186), T + U + B is 6e-16, -5e-16, -2.4e-15 (gcc 15) against 9e-16, -1e-16, 0 (gcc 13).
  The third step reaches r = 0.269, near the pericentre, where |U| = 0.93 ~ 5 |E|. A 1-ulp position error per
  leapfrog substep (12 substeps at stage 4), times dU/dr ~ 3.4 and the extrapolation's Lebesgue constant 6.3
  (sequence 2, 4, 6, 8, 12), gives ~2e-15 absolute, i.e. ~1e-14 of |E|.
- **Proof that this is round-off of a correct computation:** shift the IC by i * 1e-9 (relative, i = 0..59) in e
  (gcc 13, gcc 15, gcc 15 with -ffp-contract=off) and in a (gcc 13). Each set of 60 runs scatters non-smoothly
  between 2e-16 and 2.3-3.1e-14 (medians 7-9e-15), and neighbouring ICs differ by 10-100x. Truncation error would be
  smooth over 6e-8 in e, so it lies below the smallest values (~2e-16). With FMA off, the nominal IC gives 1.5e-15
  in LogH and 0 in total. Compensated sums in the LogH chain (R3) were already tried and rejected by the cluster A/B.
- **Fix:** bound 4e-14 = 1.3 x the largest of the 240 runs (3.1e-14). The test comment records the scatter.

## C17 tree_default_on: host-only set size, so no default tree search on GPU builds (closed)

- **Found by:** GPU ctest, "FAIL: no tree searches ran". Failing since the test was added (915ddc4, 2026-10-11).
- **Cause:** a test bug. 915ddc4 switches the exact tree searches on by default above the measured break-even set
  sizes (main.cpp): 2048 on host builds, and on GPU builds 32768 for neighbours and 65536 for time steps (300000
  with FLYBY_CM_VELOCITY). The test's 5000 stars lie above the host defaults only, so a GPU build never starts a
  default tree search. The code does what it should.
- **Fix:** `tree_default` probes the build (KOKKOS_ENABLE_CUDA / HIP in the header of a 0.2 s binary run). Device
  builds run 70000 stars at the same mean mass (BINARIES50 with PLUMMER_M_TOTAL 28000) and also require a
  time-step search. GPU (A100): 1 time-step and 16 neighbour searches, all bitwise equal to O(N^2), diagnostics
  identical to the run without tree, 73 s. Host builds keep the 5000-star case.

## C19 Near-radial BH binary in the secular PN path: unbooked dE ~1e-3 E0 (closed)

- **Found by:** the FMM force-error gate (ferr-gate-2026-10-18, 1e6 f0.1 seed 42, SSE off, 4 GPUs, binary 3c7c8b9).
  Two runs whose trajectories the injected far-field errors had nudged (NBODY_FORCE_ERR cell 1e-5 and 1e-7) reached
  the same BH binary, pids 353706 + 888286 (14.69 + 28.26 Msun, kw 14), a = 1.72-1.80e-5 pc (3.6 AU), e = 0.99933-
  0.99936, r_p ~ 2800 R_S, P = 1.1 yr, and logged `SUBSYS_ERR secular_pn dE/E0` 7.3e-4 / 2.8e-3 (cell 1e-5, step 15,
  dt 3.1e-5 / 1.6e-5 Myr) and 1.4e-3 / 1.25e-3 (cell 1e-7, step 25, dt 7.8e-6 Myr). That is 2-6x the binary's
  whole binding energy (E_bin ~ 4.5e-4 |E0|), while its GW loss over such a step is ~7e-5 E_bin (t_GW 4.6e5 yr).
- **Effect:** cell 1e-7: the restart loop recovered (RESTART[1], [2], then r_ngb 1.32e-3 changed the path).
  Cell 1e-5: the retry was worse (2.8e-3), so "dE not decreasing" accepted 7.27e-4: |dE/E| 3e-10 -> 7.3e-4 in one
  step. Unrelated to the injected forces (their own effect is < 1e-10 per interval); any run can reach this state.
- **Isolated replay** (`scripts/subsys_cases.py`, the logged state, one interval): 1.5e-16, does not reproduce
  (lesson 3: state-dependent).
- **Ledger on the run's state:** rerun with STALL_RESTART = true reproduces it bit-identically and keeps
  `/raven/ptmp/uli/cpp-nbody-runs/ferr-gate-2026-10-18/runs/cap_m42_c1e-5/stall_15.h5`; the one-step replay
  without any injected error (`ledger/cap_m42_c1e-5_s15`, NBODY_ELEDGER) gives root 7.256e-4 = own error of one depth-3
  call, class **booking:secular binaries**: `sub dE_int 7.256e-4 type 0 n 2 secular 1 pids 353706 888286` (PN path),
  every other secular binary < 7e-6 and cancelling in pairs. So the secular PN advance of this binary changes its
  members' T + U by ~1.6 E_bin without booking it. Suspects: the C9 fixed-point reshape at j = sqrt(1 - e^2) = 0.036,
  or the 1PN phase offset near pericentre (v_p/c ~ 0.02, Newtonian T at pericentre ~ 3000 E_bin: an O(x) phase
  error moves the Newtonian energy by ~E_bin) with E_gw booked from elements instead of the state (lesson 1).

- **Cause:** the PN write-back of `integrate_secular_binary`. After the Kepler drift it looks for a Newtonian orbit
  (a1, e1) at the advanced phase whose eps, j (Kidder E, J) are the evolved ones, by the fixed point a1 *= eps /
  eps_target. Near the pericentre the 1PN part of eps(state) is about delta = |1 - eps(r_p, v_p) / eps_N| ~ 8 x /
  (1 - e)^2 of eps, and delta is also the contraction factor of that iteration. For this binary delta = 2.4: the
  osculating Newtonian energy of the real PN orbit changes by more than E_bin between apocentre and pericentre. The
  depth-5 call (dt 3.125e-5 Myr, second L1 half) ended 3.9e-3 AU from the centre (r_p 2.3e-3 AU). The iteration went
  to a1 < 0, `reshape_orbit` returned without changing the state, and the energy correction to -mu / (2 a1) > 0
  wrote the pair back **unbound** (1.27 E_bin). integrator.cpp booked E_gw only when a_post > 0, so the whole change
  stayed unbooked. Not the merge rule ((1 - e) = 6.4e-4 vs 20 eps = 2.3e-6), not the 1 - e cancellation, not the
  Kepler drift.
- **Why the isolated replay missed it:** the failure needs the step to end within a few pericentre times
  sqrt(r_p^3 / G m) (time fraction ~ (1 - e)^1.5 ~ 1e-5 of the orbit), so it depends on the step length. The logged
  state with the logged dt reproduces on the host (2 particles, `tests/data/c19_secular_peri.txt`, one 3.125e-5 Myr
  interval: |dE/E| of the pair 1.22, 6.7e-4 E_REF). Steps ending near the pericentre write the pair back unbound for
  delta >~ 1 (10 of 7200 in a scan over a = 0.36-36 AU, 1 - e = 0.1-3e-4).
- **Fix:**
  - PN 2-body secular candidates with delta > SECULAR_PN_PERI_SHIFT_MAX = 0.1 go to LogH
    (`secular_pn_peri_shift`, from the osculating pericentre state, which does not depend on the phase). There the
    write-back of a Newtonian orbit is not a small correction any more. The iteration contracts by at most 0.1 per
    step, against failures at delta ~ 1.
  - `integrate_secular_binary` returns -1 and leaves the pair unchanged when the write-back has not converged
    (residual > 1e-6, a1 <= 0 or eps <= 0). The caller then sends the pair to LogH in the same call.
  - E_gw is booked from the states, E_orb(before) - E_orb(after) of any sign, so no secular step can leave a change
    unbooked.
  - Counter `SecPeriLogH` in the step summary.
  - Literature: PN orbits are quasi-Keplerian with different a_r, e_r, e_t, e_phi (Damour & Deruelle 1985; Memmesheimer
    et al. 2004). Osculating Newtonian elements oscillate at O(x a / r) along such an orbit, so a Newtonian write-back
    is valid only while that is small. BIFROST and KETJU integrate PN binaries with regularised integrators.
- **Tests:**
  - `c19_secular_peri` (run_case, E_REF / NGB_MREF of the parent): before the fix 1.22, now 7.65e-16 (bound 1e-15),
    the pair in LogH (1487 GBS steps).
  - `secular_pn` (5): the C19 call is refused with the state unchanged. Steps ending within +-5 pericentre times on
    that grid leave no admitted pair unbound, and eps of the written state = the evolved eps to 3.7e-10 (bound
    4.8e-10).
  - All 141 host ctests pass (secular_pn, coll_bh_isco, sec_triple_energy, adv_kozai_triple).
- **Cost:** bound PN pairs (nearest neighbours, a < 1e-3 pc) above delta 0.1: 4 of 949 in the 1e6 run (stall_15),
  0 of 186 / 172 / 174 in camp3-200k s404 / s202 / s303. The C19 pair costs 1487 GBS steps per 3.1e-5 Myr in LogH.
- **GPU replay** of stall_15.h5 (1 node gpudev, NBODY_ELEDGER, `ledger/c19fix_s15`): ELEDGER root step 15
  8.7e-11 (before 7.26e-4), step 16 -1.4e-11, no SUBSYS_ERR, no restart; SecPeriLogH = 32 calls per step;
  subsys_logh 8.1 s vs 6.8 s of 126 s in step 16 (the step's wall 126 s vs 106 s; the trajectories differ after
  step 15).

## C20 Tight inner binaries of small PN subsystems: LogH followed every orbit (8-120 min intervals) (fixed, A/B below)

- **Found by:** the stall harvest of 2026-10-19 (`/raven/ptmp/uli/cpp-nbody-runs/harvest-2026-10-19/`, host
  replays, one interval, one thread). 6 of the 8 reproducing slow stalls of 200k bf 0.1 are 3-6 body LogH sets with
  2.5-4.5 M GBS steps: s101_s432_1 (470 s), s404_s1025_1 (748 s), s404_s1408_1 (894 s), s101_s1340_0 (7224 s),
  s404_s706_0 (4706 s) and s404_s3047_0 (4260 s). The last three were run to the end on the host (main 683263c).
- **Cause:** every set has a BH, so it is a PN subsystem, and FROST-CLUSTERS I (App. A2) keeps kappa = 1 in all of it.
  SDAR is therefore off for the whole set (NBODY_SDAR_PN was an A/B option, off by default).
  - s432_1, s1025_1, s1340_0, s3047_0: the tight pair is a BH-BH binary of 0.016-0.05 AU (P 3e-4 to 1e-3 yr, 1e5-3e6
    orbits per interval; e.g. 32.3 + 9.7 Msun, a = 7.9e-8 pc, e = 0.26). The tide on it is gamma ~ 1e-16 from stars
    at > 1000 AU. The 2-body secular path does not apply because the set has 3-6 members. dyn_cap then holds each GBS
    step to a fraction of the binary orbit.
  - s706_0 and s1408_1: the slow pair is a 0.03-0.08 AU stellar binary (300-1e4 orbits per call) next to a BH binary.
    It is slowed only with NBODY_SDAR_PN = 1.
- **Literature:** FC-I sets kappa = 1 in PN subsystems without giving a reason. Mikkola & Aarseth (1996) slowed a
  binary with a PN-type perturbation at constant kappa. A PN pair cannot be slowed at full PN strength for long:
  kappa x GM/(c^2 r_p) is the amplitude of the short-period error, which forces kappa < ~30 here. NBODY6 integrates a
  KS binary below gamma_min = 1e-6 as unperturbed (Aarseth 2003 sec. 8).
- **Fix (local criteria):**
  1. Secular inner binaries (logh_batch.cpp, NBODY_SEC_INNER = 0: off). A bound pair qualifies when it is the tightest
     pair of both members, has n_orb >= N_ORB_SECULAR orbits in the call, and satisfies gamma n_orb <= k_ref = 1e-6.
     Here gamma = (r_apo^3 / m_a m_b) sum_k m_ab m_k / r_close,k^3 (W20 Eq. 22), and r_close is the closest approach of
     perturber k within the call in its two-body orbit about the pair. This is the kappa -> n_orb limit of SDAR.
     - Such a pair becomes one LogH body at its c.o.m., carrying the total mass, momentum and auxiliary velocity.
     - At the end of the call its internal orbit is advanced by integrate_secular_binary: orbit-averaged PN for PN
       pairs, a Kepler drift otherwise.
     - Before the collapse, a dry run of that advance over the whole call must succeed. A refused PN write-back (C19)
       or a secular coalescence (r_p < 10 R_S) keeps the pair in LogH, where the collision / GW rules decide the
       merger.
     - After every GBS step a watch checks the same bound at the current distances. Above it, the binary is expanded
       and LogH continues.
     - Energy: the collapse and the expansion change T + U of the state at a fixed phase-space point. Those jumps,
       minus the carried internal energy, go into B and sd_jump (booked as E_sd). The secular PN change of the
       internal energy is E_pn, as is all PN work, with P / L sunk per member as before.
  2. NBODY_SDAR_PN default on (PLAN item 4c): SDAR slows the non-PN inner binaries of PN subsystems. PN pairs keep
     kappa = 1 (NBODY_SDAR_PN = 0: off).
- **Cut-outs (host, one interval; main -> C20):**

  | case | wall [s] | \|dE/E\| | \|dP\|/M [km/s] | \|dL/L\| |
  |---|---|---|---|---|
  | s432_1 | 470 -> 0.10 | 5.1e-10 -> 2.4e-10 | 3.5e-13 -> 1.3e-12 | 1.9e-13 -> 1.9e-13 |
  | s1025_1 | 748 -> 0.12 | 1.8e-10 -> 2.4e-10 | 2.0e-12 -> 4.3e-12 | 3.2e-13 -> 1.1e-13 |
  | s1340_0 | 7224 -> 0.03 | 4.4e-12 -> 4.4e-12 | 5.1e-13 -> 5.7e-13 | 1.1e-13 -> 1.1e-13 |
  | s3047_0 | 4260 -> 0.09 | 1.4e-12 -> 3.9e-11 | 1.7e-12 -> 1.8e-12 | 8.9e-14 -> 3.8e-13 |
  | s706_0 | 4706 -> 1.3 (37 without SDAR_PN) | 1.27e-9 -> 1.19e-9 (6.9e-10) | | |
  | s1408_1 | 894 -> 15 (428) | 1.648e-8 -> 1.648e-8 | | |

  The P and L changes are round-off (~10 ulp of the BH momenta).
  - s3047_0, NBODY_ELEDGER: the error is the own error of the FSI steps of the star + binary at depths 12-14. Per
    call it is the same in both (largest 1.6-1.9e-13, top-12 sum of |own| 1.91e-12 main vs 1.98e-12 C20). Only the
    random-walk sum differs (root 1.0e-13 vs 2.8e-12): noise, not a bias of the collapse (LogH residues 8e-15 in both).
  - s1408_1 still takes 15 s: its 12.5 AU BH binary (34 orbits per call, gamma n_orb 1e-5) is perturbed and stays in
    full LogH, as it should.
- **Mergers:** a binary about to merge is never collapsed (dry run). In the cut-outs a BH binary with t_GW < one step
  is caught first by the binary GW rule at detection (test sec_inner_merge: s1025_1 with the binary shrunk 60x, one
  merger, E / P / L at round-off).
- **Tests:** sec_inner_s432, sec_inner_s1025, sec_inner_s3047 check the wall time (< 20 s; without the fix > 120 s:
  timeout), at least 1 collapse, the GBS steps, and |dE/E|, |dP|/M, |dL/L| at 1.3x measured. sec_inner_merge is
  described above.
- **200k A/B** (bf 0.1, 60 intervals from stall_1340 (camp2 s101), stall_1025 (camp3 s404), stall_1117 (camp2 s202),
  4 A100 each; `/raven/ptmp/uli/cpp-nbody-runs/secinner-ab-2026-10-19/`): final |dE/E| (run_summary.py, w.r.t. the restart), |dP|/M [km/s], |dL/L|,
  s/step median / mean / max, main -> C20:
  s404 (84 steps; main in 2 gpudev segments): 6.6e-11 -> 1.26e-10, 1.4e-13 -> 1.1e-13, 1.8e-12 -> 1.9e-12,
  4.15 / 7.30 / 187 -> 4.11 / 4.55 / 20.
  s202 (86 steps): 4.53e-9 -> 4.55e-9, 2.3e-13 -> 2.3e-13, 2.6e-12 -> 2.1e-12, 3.99 / 4.31 / 12 -> 3.95 / 4.22 / 8.
  s101 (91 steps): C20 1.51e-9, 1.5e-13, 1.7e-11, 4.61 / 5.22 / 18. Main is still queued on gpu (job 31366161, main_s101):
  its first step is the 1096 s stall and cannot fit in gpudev.
  Energy: equal on s202, 1.9x on s404 (noise level, user decision 2026-10-21: acceptable for the speed-up).
- **Open:** a 3-body IC with the star moved to 5e-4 pc from a 60x shrunk BH binary (a = 1.3e-9 pc, v ~ 5000 km/s)
  gives NaN at the first LogH call (dt = nan) in main as well. Not followed up: the 200k runs never reached it, and
  the IC is artificial (`/raven/ptmp/uli/cpp-nbody-runs/secinner-ab-2026-10-19/nan_ic.txt`).

## Verified building blocks (checked against an independent derivation, no bug found)

| Building block | Check | Result | Test |
|---|---|---|---|
| FSI gradient force (GradK, FROST Eq. 19) | correction vs (1/m_i) d/dx_i of 1/2 sum m &#124;a&#124;^2 (Enzyme, 2-32 bodies); sign by the measured order | exactly GRAD_COEFF dt^2 (12 digits); order 4.64 | `grad_force_consistency`, `kepler_e09` |
| PN1 EIH acceleration, 2 bodies | energy consistency, Euler-Lagrange of the EIH Lagrangian (Enzyme) | c^-4; O(v^2/c^2) | `pn1_energy_consistency` |
| Spin PN (leading SO 1.5PN, SS + Kerr quadrupole 2PN, precession, Kidder 1995) | dE/dt and dJ/dt along the code's flow, precession vs Poisson bracket of the spin Hamiltonian (Enzyme, q 1 to 1e-3, random spins, bound and unbound) | leftovers c^-5 / c^-6 (ratios 32.0 / 64.0: the next spin orders the code does not include); precession 6e-16; integrated &#124;S&#124; drift 1.9e-15. Gap: spin terms stay at leading order with PN to 3.5; frozen in 3+ body and secular paths | `spin_pn_consistency` |

## R3 Compensated (Kahan) sums in the LogH leapfrog chain (rejected by the cluster A/B)

- **Motivation:** Verificarlo (Monte Carlo arithmetic) showed that the e = 0.999 LogH step is limited by the rounding
  of x += dx and v += dv, amplified by the extrapolation (Bulirsch ~9x). Compensated sums lowered the round-off
  scatter per e = 0.999 orbit from 1.2e-11 to 1.4e-12.
- **Cluster A/B (NBODY_LOGH_KAHAN):** 200k, 10 % binaries, 0.06 Myr, |dE/E| at the end. Seeds 3/7/13/21/42:
  - base 8.6e-12 / 5.2e-12 / 6.6e-11 / 3.2e-11 / 3.1e-11;
  - Kahan 2.0e-11 / 5.3e-12 / 1.2e-10 / 1.6e-10 / 1.2e-11.
  Worse on three seeds; median step time unchanged.
- **Decision:** rejected (energy rule), switch removed together with R2's. With R2, this says the 200k cluster's
  energy error (1e-11 to 1e-10) is not set by LogH round-off or by the LogH kick. The next suspects are the FSI/field
  part and the subsystem boundaries; per-path ledger on these seeds (NBODY_ELEDGER).

## R2 BIFROST's symplectic LogH kick 1/(-U) instead of 1/(T+B) (rejected by the cluster A/B; C12 open as a finding)

- **Finding (Enzyme Jacobians, M^T J M - J):** the production LogH leapfrog is not symplectic in extended phase space.
  - Since 8cd3029 (2026-02) the kick divides by the implicit Omega = T(v_mid) + B (cubic solve), not by BIFROST's
    -U(x) (Mikkola & Tanikawa 1999). That kick depends on v, so it is no Hamiltonian flow: defect O(ds), 4e-2 at
    ds = 0.01.
  - On the energy shell T + B = -U the two kicks agree, so only the off-shell part is extra; GBS extrapolation does
    not remove it.
  - BIFROST's kick is symplectic to 6e-13 (test in branch worktree-agent-ae641fc52220640f6, tests/symplectic_test.cpp).
- **A/B (NBODY_KICK_U):**
  - Isolated (agent): e = 0.999 binary 1.8e-11 vs 4.4e-9; 200-system ensemble median 1.2-4x better, but the p90/max
    of e >= 0.9 binaries 2-4x worse.
  - Cluster, 0.06 Myr, |dE/E| at the end: 200k seeds 3/7/13/21/42 base 8.6e-12 / 5.2e-12 / 6.6e-11 / 3.2e-11 /
    3.1e-11, 1/(-U) 6.8e-11 / 1.7e-12 / 4.0e-11 / 8.2e-11 / 2.3e-11 (8x and 2.6x worse on two seeds). B (110k) seeds
    7/13/42: equal to 2 digits.
- **Decision:** the energy rule (equal or better on every seed) fails, so the implicit kick stays. Open: why the
  non-symplectic kick does as well in clusters. Likely the energy error there is dominated by the FSI/field, and
  LogH steps are short.
- **Guard (2026-10-19):** ctest `logh_symplectic` (tests/symplectic_test.cpp, Enzyme versions in
  tests/enzyme/symplectic/) keeps the 1/Omega kick's symplectic defect at its measured O(ds) value (4.0e-2 / 4.9e-2
  at ds = 1e-2, bound 1.3x) and checks reversibility, P/L equivariance and the symplectic kepler_drift to round-off.
- **Retest 2026-10-21 (user request; switch NBODY_KICK_U re-added on branch worktree-agent-a43c927618cc29a31, ctest
  logh_symplectic_kick_u: defect 2.6e-12).** tests/proto_symrk/bench.cpp, production gbs_step, eta 1e-10, k_max 10,
  max |dE/E| / force evaluations per orbit, 1/Omega vs 1/(-U):
  - e = 0.9 (1e5 orbits) 1.7e-8 / 2851 vs 6.3e-9 / 2457; e = 0.99 (1e4) 3.3e-9 / 4372 vs 1.3e-8 / 3425; Kozai triple
    (2e4) 1.9e-10 / 1315 vs 4.3e-9 / 1090; 2+2 quadruple 5.0e-11 / 3407 vs 1.1e-10 / 3115; harvest 5-body 3.9e-10 / 669
    vs 2.0e-9 / 664; 1PN binary 1.87e-6 both (EIH truncation, below). Equal cost (1/(-U) at eta 1e-11): Kozai 2.8e-10 /
    2040, harvest 6.2e-10 / 810, still worse. |dL/L| equal within 2x. 1/(-U): 15-25 % fewer evaluations, 25-55 % less
    wall per orbit (no cubic), smaller end-of-run error on binaries (secular part), larger pericentre swing.
  - Plain leapfrog at fixed ds (no GBS): both bounded (slope 0, same max swing 2e-2 at 200 steps per e = 0.9 orbit),
    because both maps are time-reversible (reversible maps also keep the energy error bounded on near-integrable
    orbits); symplecticity shows only in the smaller end-of-orbit error.
  - Cluster smoke test (gpudev, 200k bf0.1 prod config, same binary, |dE/E| at the last common step):
    s101 4.3e-11 vs 2.0e-11, s202 3.6e-11 vs 4.0e-11, s303 4.3e-11 vs 6.3e-11; per-interval median 2.5/1.5/2.0e-12 vs
    2.8/1.2/2.4e-12; median s/step 5.5/10.4/5.5 vs 5.4/8.3/5.1.
  - **Reading:** GBS extrapolation is not symplectic, so the symplectic base map buys nothing once the step is
    extrapolated to eta; what remains is a different local error, better on isolated binaries and worse on the
    hierarchical and pericentre-dominated cases. PN (identical) is not involved. The cluster differences (single seeds,
    factor 1.5-8, both signs, equal per-interval medians) are the chaotic spread of two trajectories. **Decision
    unchanged:** 1/Omega stays; no 5-seed A/B proposed (switch only on the branch).
- **PN1 in LogH (same retest), no bug:** the EIH acceleration and `pn1_energy` are Poisson & Will eq. 9.127 / Blanchet
  LRR eq. 233 (test pn1_energy_consistency, C7). RK4 of the exact EIH flow, e = 0.9: swing of E_N + E_1PN per orbit
  2.26e-2 / 1.35e-3 / 8.3e-5 at c^2 = 1e3 / 4e3 / 1.6e4 (x c^4 = 2.2e4: O(1/c^4)), periodic (back to 7e-11 after one
  orbit), 1.87e-6 at r > 1.5 a: exactly the LogH+GBS value, so the 1.9e-6 is the EIH truncation, not integrator error.
  An ADM 1PN Hamiltonian (exact invariant, prototype 3.3e-13) is not adopted: no regularised code uses it (BIFROST
  sec. 3.5, KETJU/MSTAR, Mikkola & Merritt 2008, SDAR all integrate harmonic-gauge EOM with auxiliary velocities); it
  is non-separable (implicit Gauss or Pihajoki 2015 copies instead of the explicit leapfrog), its coordinates differ
  from the harmonic ones of our 2PN-3.5PN terms at O(1/c^4), and 2.5/3.5PN are dissipative, so with PN_ORDER 31 there
  is no exact invariant either way. The energy books are unaffected: paths book their members' Newtonian T + U change
  from states, exact for any PN formulation.

## R1 FLYBY_CM_KICK, c.o.m. kicks of weakly perturbed binaries (rejected)

- **Tried on:** the C2 step. |dL/L| 1.3e-14 -> 1.4e-9, 15 % faster. Deleted in 1483094 (the isolated tests had stayed
  at round-off: a path inconsistency in the cluster, not debugged).
