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

## C15 The diagnostics output changes the trajectory: results depend on PRINT_INTERVAL (closed)

- **Found by:** N = 1e6 runs whose results differed by ~10 % in one quantity between PRINT_INTERVALs; an A/B with a
  temporary NBODY_DIAG_FORCES switch (skip the force recompute of the print) changed the trajectory.
- **System:** every HHS run (collisional or not). Physics output must be read-only on the integrator state.
- **Cause (two):**
  - (a) At a print, main.cpp recomputed accelerations and time steps for the time-bin table, assigned levels and
    then called `partition_by_level()`, which re-sorts all particles by level. The next step started from another
    slot order: other summation orders in the force sums, the neighbour and pair lists, the subsystem detection.
    That is a round-off seed (host, 2000 stars with 10 % binaries, 6 steps, PRINT_INTERVAL 1 vs 3: 81 particles
    differ, up to 4e-9 relative), which the chaotic N-body dynamics grows to macroscopic differences. Restoring
    ax / dt_adapt / level alone does not remove it (tested), the slot order does; the overwritten ax / dt_adapt /
    level themselves are recomputed by the next step (dt_valid = false at the root) and did not matter.
  - (b) Escaper removal (`remove_escapers`, R_ESCAPE_RH / R_ESCAPE_RJ) ran inside the print block, i.e. every
    PRINT_INTERVAL steps: a dynamics operation on the output cadence.
- **Fix:**
  - (a) `diag_levels_begin/end` (main.cpp): the print saves ax, ay, az, dt_adapt, level, recomputes them for the
    table, and restores them afterwards. No partition: the table only counts levels. E, P, L are computed from the
    same particle state as before (unchanged); the PE reuse of the energy check is now valid at more prints (layout
    unchanged).
  - (b) Escaper removal after every step, independent of the print. Production configs print every step
    (PRINT_INTERVAL = 1), so their escaper cadence is unchanged; runs with a large PRINT_INTERVAL now check every step
    (host O(N log N), the ESCAPERS profiler bucket). No new config key.
  - The final print after the loop (after the last restart file) still recomputes and partitions; nothing follows it.
- **Test:** `print_interval_identical`: PRINT_INTERVAL 1 vs 3, final particle dumps bit-identical, for 10 % binaries
  (subsystems, restarts), non-collisional HHS, FSI, and a tidal field with 2 r_J escapers. On main: 81 and 1442
  particles differ in the binaries and escaper cases (HHS non-collisional and FSI were already identical).
- **Side effect:** `sse_books` compares the booked dU with the difference of two full triangle sums, whose round-off
  depends on the slot order. Seeds 3-8: main 1.8-2.6e-15, fix 2.1-3.0e-15 (main above the old 2.4e-15 bound at seeds 5
  and 8: the bound was one sample). Bound 3.9e-15 = 1.3 x 3.0e-15.
- **Trajectories:** every default (PRINT_INTERVAL = 1) run changes at round-off from the first print on (no re-sort).
- **GPU (c15-2026-10-21, 4 A100, gpudev):** 1e5 fig7 config (SSE, PN 31, f_bin 0.1), 4 steps, PRINT_INTERVAL 1 vs 2:
  fix bit-identical final dumps; main 1414 particles differ (max |diff| 1.5e-6).
- **Energy A/B (prod200k bf0.1 config, PRINT_INTERVAL 1, 8 steps, main a0840e5 -> fix):** max |dE/E| s101 6.97e-12 ->
  7.06e-12, s202 7.60e-12 -> 7.18e-12, s303 1.89e-11 -> 1.89e-11; final s101 1.06e-12 -> 1.29e-12, s202 6.96e-12 ->
  6.09e-12, s303 1.21e-11 -> 1.79e-11. The max is unchanged within 1 %. The final values scatter both ways, as
  round-off-perturbed trajectories do (main itself gives other values at another PRINT_INTERVAL).

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
- **Open (closed by C24):** a 3-body IC with the star moved to 5e-4 pc from a 60x shrunk BH binary (a = 1.3e-9 pc,
  v ~ 5000 km/s) gave NaN at the first LogH call (dt = nan) in main as well.

## C21 SDAR slowed a binary whose tide comes from outside its LogH group: 8e-8 E_REF per interval (fixed)

- **Found by:** the stall harvest of 2026-10-19, cut-out s3_s324_0 (camp2-B seed 3, 110k, stall_324.h5, t = 0.323
  Myr; the 100 bodies within 0.05 pc of the deepest set). Host replay, one interval: |dE/E0| 1.81e-5 of the cut-out's
  own E0 = 8.1e-8 E_REF (|E0| of the cut-out is 4.5e-3 E_REF). The parent's jump at that step was 9.8e-8 E_REF, so
  the cut-out is faithful (parent E_REF / NGB_MREF, no external field in the parent either). 119579 GBS steps, 5 s.
- **Localised (NBODY_ELEDGER):** subsystem advances balance their books (residue -7e-15; LogH residues <= 3e-14,
  secular binaries -8e-15). The error is the own error of the FSI calls at depth 11-12 (sum 1.24e-7 at d12), many
  calls of ~5e-10 with the same sign: a drift, not a single bad step. The automatic RESTART with halved etas makes it
  worse (1.3e-7 E_REF): not a step-size error. With SDAR = false: 6.9e-10 (3.1e-12 E_REF), 15098 GBS steps, faster.
- **Cause:** SDAR slows the 168 + 94 Msun binary (a ~ 13 AU, P 2.9 yr, e 0.87) of a 3-4 body LogH group by kappa ~ 11.
  kappa came only from the in-group perturber, a 0.25 Msun star. The 141 + 113 Msun binary 0.019 pc away is outside
  the group and dominates the tide. W20 (sec. 2.2, Eq. H2) slows the binary inside an external potential U. The
  members feel dv/dt = -(1/kappa) grad U_b - grad U while their positions move at v/kappa. That is what the FSI kicks
  plus a slowed LogH drift do. It conserves H_sd, but the physical energy then differs by (1 - 1/kappa) times the
  binary's energy change from the outside tide. Nothing books that change (E_sd covers only changes made inside the
  call). It stays bounded only while U is nearly constant over kappa P (W20 Eq. UieqUz), and that is the condition the
  kappa criteria enforce. The code applied them to the in-group perturbers only, so they could not see the outside
  binary. With it, kappa_org ~ 0.94: the pair should not have been slowed. PeTar likewise sums all perturbers of a
  group, members and outside neighbours, into kappa. The 2-body subsystems already keep kappa = 1 for this reason
  (logh_batch.cpp).
- **Fix (local criteria, logh.cpp logh_sd_update, subsystem.cpp subsystem_timesteps):**
  1. Subsystem::ext_tidal (3+ body subsystems, SDAR on) is the outside tidal strength, sum m_k / d_k^3. It is
     estimated as max over member pairs of |a_ext,j - a_ext,i| / (G r_ij) from the members' external accelerations
     in the time-step pass. The value is the largest since the step's depth-0 pass, because deeper sets lack the
     slower perturbers. It is carried over by the detection and stored in restarts (dataset ext_tidal; older files
     read 0). The term m_ab ext_tidal is added to every pair's perturbation in kappa_org (W20 Eq. 22).
  2. Tidal energy cap: a pair keeps kappa = 1 when the outside tide's energy swing per orbit, 2 pi mu G ext_tidal
     a^2, exceeds FLYBY_CM_EPS E_REF. That is the per-step budget the tidal-sampling steps of fly-bys already use.
     Without the cap (1 alone) the error falls only 7x (2.5e-6 own E0): other pairs remain slowed with kappa ~ 11 in
     the same tide.
  - A/B switch NBODY_SDAR_EXT=0 restores the old behaviour.
- **Cut-out (host, one interval, own E0):** main -> C21: |dE/E| 1.81e-5 -> 6.91e-10, GBS steps 119579 -> 15098,
  |dL/L| 4.6e-14 -> 4.7e-15, wall 1.36 -> 1.23 s (no RESTART). Nothing is slowed here, the same result as SDAR off.
- **Test:** c21_sdar_outside_tide (tests/data/c21_sdar_outside_tide.txt = the cut-out): |dE/E| <= 9e-10 and GBS
  steps <= 19600 (1.3x measured). Fails on main (1.8e-5, 119579). The SDAR tests (sdar_kozai_triple, sdar_f3_flyby,
  sdar_pn_kappa1, quad22_sdar) and sec_inner_* are unchanged. Host ctest: all pass (6 login-node 120 s timeouts
  rerun individually: pass, results identical with NBODY_SDAR_EXT=0).
- **200k A/B (Raven gpudev, 2026-10-21, 4 A100, ~85-91 intervals from the C20 A/B restarts, same binary nbody-c21,
  off = NBODY_SDAR_EXT=0; `/raven/ptmp/uli/cpp-nbody-runs/sdarext-ab-2026-10-21/`):**

  | seed | final \|dE/E\| off -> new | sum of \|per-step jumps\| | max jump | \|dL/L\| | s/step median/mean/max |
  |---|---|---|---|---|---|
  | s101 | 2.46e-10 -> 1.59e-9 | 3.60e-9 -> 2.84e-9 | 1.51e-9 = | 1.5e-11 -> 1.7e-11 | 4.42/5.08/16 -> 4.38/5.05/16 |
  | s202 | 4.53e-9 -> 4.55e-9 | 4.95e-9 -> 4.94e-9 | 4.35e-9 = | 3.3e-12 -> 2.6e-12 | 3.91/4.16/8 -> 3.92/4.17/8 |
  | s404 | 1.11e-10 -> 1.00e-10 | 1.04e-9 = | 3.05e-10 = | 2.5e-12 -> 1.9e-12 | 4.04/4.48/19 -> 4.03/4.50/19 |

  The largest per-step jumps are identical in both arms (the same episodes); the s101 final difference is sign
  cancellation of two episodes in the off arm (1.35e-9 -> 6.4e-10 -> 2.4e-10), while the summed jump size is smaller
  with the fix. At bf 0.1 the outside-tide case is rare; C21 is neutral there and fixes the s3_s324_0 class.

## C22 Gauss-Legendre IRK (symplectic, 8 stages) for compact Newtonian subsystems (merged, default off; 200k A/B below)

- **Motivation:** after C20/4c the remaining subsystem cost is in compact non-hierarchical 2-4 body systems (eccentric
  binaries, 2+2 quadruples). The prototype (tests/proto_symrk, branch worktree-agent-a0cc578ab96edb7bf) was 30-1000x
  more accurate than LogH+GBS at equal cost on these, with no linear energy drift (e = 0.9: 1.7e-8 -> 2.1e-11 over
  1e5 orbits). PN1 in Hamiltonian form was retested (R2) and does not help the energy books, so PN subsystems stay on
  LogH+GBS.
- **Implementation (src/collisional/gauss_irk.{h,cpp}, logh_batch.cpp):** 8-stage Gauss method on the Poincare-
  transformed Hamiltonian, fixed ds (30 steps per innermost orbit, floor 8/sqrt(1-e)), fixed-point iteration
  (HMR08/AMM17) plus simplified Newton, landing on t_end exact to 1e-14 (GBS lands to 1e-4). Classification by local
  criteria only: 2-4 bodies, no PN member, no collision candidate, no e >= 0.99 pair or hard encounter within the
  call, no stable MA hierarchy, no SDAR slow-down; the exit stats print the count per reason. If the iteration does
  not finish, the call starts again from the same state with LogH+GBS (unit test: state unchanged on failure). The
  energy books are exact (SUBSYS_ERR kind unbooked-irk). Switch NBODY_GAUSS_IRK=1, default off.
- **Isolated:** e = 0.9 binary, 2000 orbits: |dE/E| 4e-14 in one call or in 2000 calls; P exact, L 1e-13. 2+2
  quadruple 1.6e-13. Full program, 2+2 quadruple (500 orbits per interval): |dE/E| 4.0e-10 -> 2.0e-12, 21 of 22
  calls by IRK, wall 3.0 -> 4.5 s. Harvest cut-out s1744 (MA-unstable stellar triple): error per call 1e-13 to
  9e-13 vs GBS 1e-11 to 5e-11; the run's |dE/E| (2.4e-7 -> 1.4e-6) is set by the FSI error of the stars outside the
  subsystem (ledger), chaotic. Simplified Newton agrees with the fixed point to 7e-15 but never rescued a step the
  fixed point lost at a hard pericentre; the GBS fallback is the safety net.
- **Tests:** gauss_irk (unit), gauss_irk_quad22, gauss_irk_s1744; host ctest all pass (login-node 120 s timeouts
  pass alone).
- **200k A/B (gpudev, prod200k bf0.1 config from the C20 A/B restarts, ~80 steps, 2026-10-21,
  /raven/ptmp/uli/cpp-nbody-runs/girkab-2026-10-21/):** base vs NBODY_GAUSS_IRK=1, same CUDA binary (404a33f).
  Gauss-IRK calls / all LogH calls: s101 112 / 807k, s202 101 / 459k, s303 370 / 766k (16 GBS fallbacks, no
  convergence). 0 Newton steps. s/step median 5.5 / 10.4 / 5.4 unchanged.
  - final |dE/E|: s101 4.97e-11 -> 2.21e-11, s202 6.43e-11 -> 1.59e-11, s303 2.63e-11 -> 3.66e-11;
  - max |dE/E|: s101 5.07e-11 -> 2.43e-11, s202 6.71e-11 -> 2.61e-11, s303 8.26e-11 -> 8.79e-11.
  Better on two seeds, equal within noise on s303 (max) with a worse final value. Not yet decided as default: under
  the energy rule (equal or better on every seed) s303 needs a longer run or more seeds. The runs finished after the
  Raven login-node crash of 2026-10-21 and were evaluated from Viper; the exit stats of the three IRK runs are in
  their run.log.
  IRK takes 0.01-0.05 % of the LogH calls, so the seed-to-seed differences are the chaotic spread of the
  trajectories, not the integrator (R2 has the same spread).
- **Why so few calls (counters 69ac4c7):** a host run of the same config at N = 20k (r_h 1 pc, 3 intervals) had 39304
  LogH calls, and all of them were "not chosen: PN member". In the bf 0.1 cluster the stellar binaries go to the
  secular / Kepler paths, and the LogH subsystems are the groups that contain a BH. Gauss-IRK can only matter in
  production once PN subsystems are covered (2-3.5PN terms in the Poincare form, or splitting PN out of the IRK
  flow).
- **Open:** the default decision; PN subsystems.
- **W10 guards (R4 side finding):** `std::max(dmax, NaN) = dmax` hid NaN stage values in the fixed-point and Newton
  increments, so a NaN iterate could pass the HMR08 rule (increment 0). Now every stage value is tested for finiteness
  in the fixed point, the Newton iterate and the final L; an increment growing over 3 successive iterations above 1e-13
  counts as non-contracting. Either is failure (fail 2, state unchanged, GBS fallback). Test `gauss_irk` section 5
  (third body 4e-12 from a binary member): old code fail 4 after accepted steps, now fail 2; all earlier bounds unchanged.


## C24 Secular triple: a BH inner binary that coalesces within the step came back NaN (closed)

- **Found by:** C20's merger cut-out variant (`tests/data/c24_triple_gw.txt`: s101_s432_1 with the 32.3 + 9.7 Msun
  BH binary shrunk 60x, a = 1.3e-9 pc, e = 0.26, and the 0.89 Msun star moved to 5e-4 pc). On the GPU build dt = nan
  at the first LogH call; on the host build no result within 900 s.
- **Cause:** the set is a Mardling-stable hierarchical triple (a_out / a_in = 2e5), so it went to the secular triple
  step (dt = 2.6e-4 Myr). The inner binary's Peters time is ~0.08 yr, 3e-7 of the step.
  - secular_evolve_triple took its internal step from the Kozai time alone (n_steps = 1 here). One explicit Euler
    step of the PN2.5/3.5 rate da/dt took a_in below zero, the loop stopped, and secular_triple_step rebuilt the
    members from a < 0: NaN positions and velocities. The NaNs reached LogH in the next call.
  - The binary GW rule of detect_mergers (merge if t_GW < dt, Peters 1964) looked at 2-body subsystems only. A triple
    sent straight to LogH ran into the 4e6 GBS-step limit about 1e5 orbits before the contact distance, then
    restarted with halved eta (the host hang).
- **Fix:**
  1. detect_mergers also applies the binary GW rule (t_GW < dt, Peters 1964) to the inner pair of a secular
     TRIPLE_STABLE subsystem, using the inner elements the classification keeps in sub.orb. It does so only when LogH
     could not follow the inspiral in one call: t_GW > 62500 P_in, the 4e6-step ceiling at 64 GBS steps per orbit
     (logh_batch.cpp gbs_step_limit). In a Mardling-stable triple the third body is the weak perturbation that the
     2-body rule also ignores for a binary. The merger itself is the usual execute_merger (pairwise energy sink with
     all other bodies, P / L sinks, recoil, merger tree, GW table). Contact stays with LogH's pericentre landing
     (FC-I App. B3). A first version that applied contact and GW to all stable triples took the ISCO / BH_MERGE_RADIUS
     / head-on triple tests away from that path (5 ctest failures), so it was narrowed.
  2. secular_evolve_triple also takes eta_sec min(a / |da/dt|, e / |de/dt|) of the PN rates at the start of the call
     (same eta_sec = 0.01 and the same 1e5 cap). It is bit-identical when the GW time is longer than the Kozai time
     (n_gw <= n_kl).
  3. secular_triple_step refuses an inner binary that reaches a <= 0, e >= 1 or a pericentre inside 10 R_S (20 G M12
     / c^2, the coalescence test of secular_evolve_pn) within the step. The members are left unchanged and the
     triple goes to LogH, where the collision / GW rules decide (coll_bh_isco: a few orbits to contact).
  These are physical criteria, not a NaN clamp: once the step is refused before the rebuild, no NaN can arise.
- **Side effect:** coll_bh_isco (10 + 10 Msun circular, star at 100 a) now takes 4-27 secular-triple substeps per call
  instead of 1. |dE/E| is 2.43e-14 instead of 1.22e-14, and |dL/L| 7.4e-14 instead of 6e-16, both set in the first
  call. They are within the round-off acceptance of the circular-orbit fallbacks of the secular triple rebuild
  (energy fix |res| <= 1e-13 E_tri; outer e^2 = q accepted down to -1e-10 and used as 0, C11). Bound raised to 1.3x
  the new value (3.2e-14). Path and merger unchanged (ISCO contact in LogH, r_sep / r_coll 0.970 vs 0.975).
- **Result (host, cut-out, one interval):** before: no result in 900 s (GPU: NaN). After: 1 merger (GW_COAL, 41.6
  Msun remnant, 170 km/s kick) at t = 0, then a 2-body fly-by. |dE/E| with merger sink 1.16e-16, |dP|/M 4.3e-14 km/s,
  |dL/L| with L sink 1.08e-12, wall 0.02 s.
- **Tests:** c24_triple_gw (the cut-out; 1 merger exactly, wall < 20 s, E / P / L at 1.3x measured; main: timeout
  at 120 s). secular_triple (unit), new part (6): (a) the cut-out's step at dt = 2.6e-4 is refused with the members
  unchanged (main: hierarchical, NaN members); (b) at dt = t_GW / 3, one step against two half steps gives the same
  a_in to 1.44e-4 (bound 1.9e-4; main: 3.97e-3, one Euler step).
- **Host ctest** (CPU node, 2026-10-22): 156 of 156 pass (luna_reader skipped).
- **Not changed:** the 2-body secular path (secular_evolve_pn already resolves the GW time and tests coalescence),
  LogH, PN, the energy booking.

## E2 Gravitational-wave observables of compact binaries and mergers: gw_events.dat (added, on by default)

- **Why:** runs must be comparable with LIGO/Virgo/KAGRA and LISA populations. merger_tree.dat had masses, a, e and
  the remnant only (and, for mergers by detect_mergers, not even the remnant: the event was recorded before
  execute_merger made it, m_rem = m1 + m2, no kick, no spin; fixed here, the tree is now written by execute_merger).
- **Literature (what codes write, what population papers use):** CMC (Rodriguez et al. 2018 PRL 120, 151101 and PRD
  98, 123005; Kremer et al. 2020 ApJS 247, 48): every BBH merger tagged in-cluster 2-body / 3-4-body (GW capture
  during a resonant encounter) / ejected, ejected binaries with their Peters merger time, e at 10 Hz from the peak
  frequency of Wen (2003) and Peters (1964) evolution, spins and recoil (Lousto 2012, as gw_recoil.h), chi_eff.
  Samsing (2018, PRD 97, 103014): fraction with e_10Hz > 0.1 is the signature of 3-body GW captures. DRAGON-II
  (Arca Sedda et al. 2023-24, arXiv 2307.04805-07): in-cluster vs ejected mergers, remnant mass/spin/kick, repeated
  mergers. PeTar (Wang et al. 2020; src/ar_interaction.hpp "Dynamic_merge"): ids, period, a, e, pericentre distance
  and time, slow-down factor, both stars before/after. MOCCA (Askar et al. 2017, Hong et al. 2018): escaped / in-
  cluster / GW-capture channels. BIFROST / FROST-CLUSTERS (Rantala et al. 2023-2026, papers/rantala/src): masses,
  remnant and recoil per IMBH merger, no frequency-domain output. KETJU: SMBH remnant mass/spin/kick (Zlochower &
  Lousto 2015). Population inference uses chi_eff, chi_p (Schmidt et al. 2015 PRD 91, 024043), e at 10 Hz (LVK
  eccentricity searches) and, for LISA, e at 10 mHz (Breivik et al. 2016, Nishizawa et al. 2016).
- **What is written (src/collisional/gw_output.{h,cpp}; header of the file documents the 36 columns):** one table
  `gw_events.dat`.
  - MERGER row per NS/BH-NS/BH merger, at the moment the code decides it, from execute_merger (paths `contact`:
    ISCO at a step boundary, `peters`: t_GW < dt, `secular`: the same for a secular binary) and
    finalize_subsystem_collision (`subsys`: the LogH/GBS(/IRK) collision at the pericentre with PN): time, pids,
    stellar types, masses, both spin vectors, Newtonian osculating a, e, f_orb, peak GW frequency (Wen 2003), e at
    10 Hz and 10 mHz on the orbit-averaged Peters track through (a, e) (backward or forward; -1: the track never has
    that frequency, the binary formed above it), Peters merger time (Mandel 2021 fit), chi_eff, chi_p, remnant mass,
    spin vector and kick as the merger code made them, number of subsystem members, tidal strength of the strongest
    other body relative to the pair at the separation, distance and speed of the pair from the cluster c.o.m., and a
    channel: ejected (pair c.o.m. unbound from the cluster, point-mass potential: a sufficient condition) > capture
    (e >= 1 at the decision) > fewbody (>= 3 members) > 2body.
  - BINARY rows (census) at every snapshot: the mutually nearest bound NS/BH pairs of every subsystem with t_GW <
    GW_CENSUS_TGW_MYR (default 13800 Myr; 0: none), with channel ejected / cluster. Ejected binaries are never removed
    as escapers (remove_escapers skips subsystem members), so the census counts them and their t_gw dates their
    merger after the run.
  - Peters-track subtlety (found while testing): along a track the Wen peak frequency falls from infinity to a
    minimum at e = 0.7497 and rises again by 1.2 % to e = 1, so a target frequency can have two roots; the code takes
    the one below 0.7497 (the last entry into the band before the merger).
  - Not available: primordial vs dynamically assembled (the IC pairing is not stored per particle); the decision time
    is g_sim_time (start of the HHS step, as merger_tree.dat).
- **Restart-safe:** rows are buffered (part of the step-restart checkpoint, so a rejected step leaves none), appended
  at every print and before every restart / stall-roll file; the restart stores the row count (attribute gw_rows), a
  resumed run truncates the file to it (older restart files: 0, new file).
- **Read-only:** host copies only; `NBODY_GW_OUTPUT=0` turns it off.
- **Tests (failing-first: gw_merger fails on main, no file):** `gw_merger` (one MERGER row for the PN 10+10 BH binary
  of coll_bh_isco, e = 0.6 variant, and a t_GW < dt secular binary at a = 1e-8 pc, e = 0.9: masses exact, e_10Hz /
  e_10mHz against an independent RK4 time integration of Peters' da/dt, de/dt to 1.1e-11 / 3.7e-11 (bound 4.8e-11;
  with IAU G instead of the code's G_CODE the difference is 1.6e-4), -1 where the integration never reaches the
  frequency, remnant mass / kick / spin equal to merger_tree.dat), `gw_restart` (2n steps vs a run to 1.5n that wrote
  the restart at n, resumed in the same directory: 5 stale rows dropped, tables identical line for line),
  `gw_off_identical` (final particle dumps bit-identical with the output on and off, with census rows each step: BH
  inspiral and the IMBH cluster). Full host ctest (build-gw, gcc 15 + HDF5): all pass; 19 login-node 120 s timeouts passed when rerun alone
  (luna_reader skipped, as on main).
- **200k smoke run (gpudev):** prod200k bf0.1 config from the C22 A/B restart (base_s101, step 80, 0.08 Myr),
  20 steps to 0.1 Myr with SNAPSHOT_INTERVAL 5, CUDA build 10f07a7, job 31378117,
  /raven/ptmp/uli/cpp-nbody-runs/gw-smoke-2026-10-21/{on,off}: 4 snapshots, 228 census rows (57 BH-BH binaries
  with t_gw < 13.8 Gyr per snapshot, none ejected yet; the hardest at a = 0.003 AU, e = 0.20, t_gw = 1.9e-3 Myr), no
  merger in the interval. Output on vs off (NBODY_GW_OUTPUT=0): all 22 L0 rows identical; wall 130.9 vs 131.0 s,
  step wall of the last three profile reports 5.67 / 5.74 / 5.56 vs 5.67 / 5.73 / 5.54 s.
- **Open:** primordial / dynamical tag (needs the IC partner per particle); decision time at sub-step resolution;
  spins of the census binaries are those the code tracks (BH spins from SSE are 0 unless set).

## Verified building blocks (checked against an independent derivation, no bug found)

| Building block | Check | Result | Test |
|---|---|---|---|
| FSI gradient force (GradK, FROST Eq. 19) | correction vs (1/m_i) d/dx_i of 1/2 sum m &#124;a&#124;^2 (Enzyme, 2-32 bodies); sign by the measured order | exactly GRAD_COEFF dt^2 (12 digits); order 4.64 | `grad_force_consistency`, `kepler_e09` |
| PN1 EIH acceleration, 2 bodies | energy consistency, Euler-Lagrange of the EIH Lagrangian (Enzyme) | c^-4; O(v^2/c^2) | `pn1_energy_consistency` |
| Spin PN (leading SO 1.5PN, SS + Kerr quadrupole 2PN, precession, Kidder 1995) | dE/dt and dJ/dt along the code's flow, precession vs Poisson bracket of the spin Hamiltonian (Enzyme, q 1 to 1e-3, random spins, bound and unbound) | leftovers c^-5 / c^-6 (ratios 32.0 / 64.0: the next spin orders the code does not include); precession 6e-16; integrated &#124;S&#124; drift 1.9e-15. Gap: spin terms stay at leading order with PN to 3.5; frozen in 3+ body and secular paths | `spin_pn_consistency` |

## R5 Energy episodes of the 200k production runs: binary period commensurate with the HHS block step (analysis; no production change)

- **Question:** in the 200k, 5 Myr production runs (`prod200k-2026-10-17`, binary nbody-prod-0d5d511) |dE/E| grows in
  episodes: bf0.1_s101 steps 2000-2500 (net -1.3e-8 E_REF of the final 1.25e-8), bf0.1_s202 steps 4540-4860 (+2.3e-8 net).
  Outside the episodes the median per-step |dE| is 2-4e-12 E_REF.
- **Replay not possible:** RESTART_INTERVAL = 50 rolls (only `restart_00005000/5001.h5` are kept) and
  `STALL_RESTART = false`, so no state from inside an episode exists. A replay from step 0 with main
  (C15/C20/C21 changed since 0d5d511) would not follow the same trajectory. The analysis below uses the log STALL lines,
  cut-outs of the step-5000 restarts (code_version 0d5d511a7cc9-dirty; read with main 97644ec, particles only) and
  controlled host runs.
- **Largest per-step jumps (E_REF units; per-step value = difference of consecutive L0 rows; signed values from the
  SDAR "unbooked_dE" line):**
  - s101: 2086 (t = 2.086 Myr) +1.77e-9, 2134 +1.61e-9, 2453 +1.46e-9, 2228 +8.3e-10, 2076 +8.3e-10, 2153 +8.1e-10,
    2416 +7.9e-10, 2322 +7.1e-10, 2200 -6.6e-10, 2444 +6.5e-10. In all ten the STALL deepest set (depth 16-17, pivot
    1.5e-8 / 7.6e-9 Myr) is the same three particles: 45751 (12.2 Msun BH) + 160657 (45 Msun BH), a = 1.82e-6 pc
    (0.38 AU), e = 0.652, P = 3.052e-8 Myr (11 d), and 128932, a 1.25 Msun MS star bound to the pair (a_out ~ 1800 AU,
    e_out 0.95, inclination 82 deg; Mardling-Aarseth stable, r_p/a_in ~ 300). Usually on the secular-triple path,
    sometimes as a binary plus field star. 194 of the run's 211 "STALL energy" events name this triple.
    No SUBSYS_ERR or RESTART lines. Unrelated events in two windows: GW_COAL #28 (2228) and an escaper (2416).
  - s202: 4590 +1.6e-9, 4563 +1.4e-9, 4636, 4630, 4761 (-1.4e-9), 4661, 4762, 4753, 4613, 4750 (all 1.2-1.4e-9, both
    signs). The median per-step |dE| of the episode is 1-5e-10, a coherent noise of ~100x the floor rather than
    isolated jumps. Deepest set: 108897 + 117644 (29.4 + 26.6 Msun BHs, a = 2.9e-6 pc, e = 0.66-0.69,
    P = 6.10e-8 Myr) with 27182 (0.087 Msun star; a_out 257 AU, e_out 0.89, r_p/a_in ~ 49 at step 5000). No merger,
    escaper, RESTART or SUBSYS_ERR line in any of the ten windows.
- **The pattern: P_in / (DT_MAX / 2^k) ~ 1.** The inner period from the STALL tidal lines, against the HHS block step:
  - s101, level 15 (3.0518e-8 Myr):
    - steps 571-1157: P/dt 0.998-1.0017. Energy events in every 250-step window.
    - 1371-1805: 0.979-0.992. No events in 1250-1750.
    - 1865-2331: 0.998-1.0011. The main episode.
    - after the hardening at ~2585 (a_in 1.82e-6 -> 1.57e-6 pc): 0.800 = 4/5, so the level-13 step is 5 P.
      Weaker episodes in 2750-3750.
    - 0.747 and 0.704: nearly quiet.
  - s202, level 14 (6.1035e-8 Myr): steps 4550-4750 0.9988-1.0006 (the episode); 4912 1.047 and 4983 1.020 (quiet).
    At step 5000 P = 5.54e-8 (0.908).
- **Mechanism.** The binary's members get the field (here: the bound star's tide) as kicks at its block steps, and
  between kicks the pair advances by its own path (secular binary / Kepler2 / LogH). When a block step is a multiple
  of P, every kick hits the same orbital phase. The tidal work then adds up coherently instead of averaging out. This
  is the step-size resonance of kick-drift splittings (Wisdom & Holman 1991; Rauch & Holman 1999, AJ 117, 1087). The
  HHS power-of-two grid makes every coarser level an integer multiple too. FLYBY_CM_EPS bounds the energy of one kick
  (est = mu |T| v dt <= 1e-11 E_REF; the STALL est of these pairs is 2-4e-12). There are 1e3-1e5 kicks per interval at
  depths 10-17, so a coherent sum exceeds the per-kick bound by orders of magnitude. Off resonance the same kicks
  random-walk at the floor.
- **Controlled test (host, main 97644ec, PN off, NBODY_ELEDGER=1 NBODY_EATTR=1, 20 intervals, E_REF of the parent).**
  Cut-outs of the triples at step 5000 (`cutout.py`, 3 bodies within 0.05 pc; 0.3 pc gives the same), with DT_MAX
  set to P_in 2^14 (1 + x):

  | case | x = 0 | +-1e-5 | 1e-4 | 1e-3 | 3e-2 (off) | DT_MAX 1e-3 (prod) |
  |---|---|---|---|---|---|---|
  | s202 triple, sum of per-step \|dE\| | 2.6e-9 | 2.8e-9 / 5.5e-9 | 5.9e-10 | 3.3e-10 | 4.9e-11 | 4.4e-11 |
  | s202 largest step | 6.9e-10 | 1.1e-9 | | | 9.0e-12 | 7.6e-12 |
  | s101 triple, sum | 8.1e-11 | 6.0e-11 / 5.6e-11 | 1.2e-11 | 1.1e-12 | 9.8e-13 | 3.9e-11 |

  On resonance the per-step error reaches the episodes' 0.5-1.8e-9. The ledger puts all of it in the FSI own errors
  of the three-body set at depths 10-14, ~2.5e-12 per call with one sign. The subsystem books balance (path residues
  <= 1e-15). The run's own error restarts (ETA_RESTART, etas halved) do not help ("dE not decreasing"): halving the
  step keeps it commensurate. The s101 state at 5000 is off resonance at the production DT_MAX, but its sign is
  systematic: one ~-1.1e-11 kick error per outer pericentre passage.
- **Excluded:** PN (same errors with PN_ORDER = 0). The secular-triple path: its residues are 0, and SECULAR_TRIPLES =
  false gives the same order of error. TIDAL_FLOOR_ADAPT and TIDAL_STEP_RES = 0.1 / 0: bit-identical, since the
  estimate is below its share and the steps are not at the floor. The outer pericentre (r_p 10-48 a_in, synthetic
  variants): no trend at fixed DT_MAX. Mergers and escapers: none in the windows of the triple.
- **FLYBY_CM_EPS (the per-kick tolerance), on resonance s202 x = -1e-5:**
  - 1e-11: 5.5e-9.
  - 1e-12: 1.0e-9, 5.5x the wall time.
  - 1e-13: 2.1e-10, 20x the wall time.
  - Off resonance 1e-12 gives 23x less error at 8x the wall time.
  - It treats the symptom, not the commensurability.
- **Classification:** a step-criterion limit, not a missing booking. The tidal-sampling criterion bounds the error per
  kick, but in a commensurate case the errors add up linearly with the number of kicks per interval. Nothing checks
  the phase relation between the block step and P.
- **Proposed fixes (for the integrator owners; not implemented here):**
  1. Kick only the c.o.m. when the kick step is > P / 4 (it neither resolves nor averages the orbit). This is NBODY6's
     unperturbed KS treatment (Aarseth 2003, gamma < gamma_min ~ 1e-6; Mikkola & Aarseth 1996): the field sees and
     kicks the c.o.m. These pairs have gamma 1e-11 to 1e-10. The tide's secular effect then goes into the
     secular-binary step as an orbit-averaged quadrupole torque (Heggie & Rasio 1996; the secular-triple quadrupole
     already exists). PeTar's tidal-tensor treatment of binaries is the same idea for the long-range part (Wang et
     al. 2020). Caution R1: c.o.m. kicks broke L in the cluster (1.4e-9) through an undebugged path inconsistency.
     The rebuild has to keep L exact (lessons 2 and 8).
  2. Short of 1, a local resonance criterion: per binary, delta = |dt/P - round(dt/P)| (and P/dt for dt < P).
     With N kicks per interval, a coherent sum costs N x est. Take N x est instead of est against the share (a
     per-interval budget), and resolve with TIDAL_FLOOR_ADAPT (dt <= P/20) only when it is exceeded.
  3. Production runs keep stall restarts (`STALL_RESTART = true`; the rolling restart of 50 steps lost every episode
     state) so that the next episode can be replayed with the ledger.
- **Cut-out for a test (fails today):** `episodes-2026-10-22/res_s202_-1e-5` (3 bodies, 20 intervals, 3-4 s host):
  sum of per-step |dE| 5.5e-9 E_REF and largest step 1.1e-9 now. A fix should bring it to the off-resonance level
  (<= 6.4e-11 sum = 1.3x the measured 4.9e-11) at equal or better wall time.
- **Pending:** the gpudev replays of the step-5000 restarts with main (`gpu_bf0.1_s{101,202}_r5000`: restart
  compatibility and the cluster ledger) had not run when this was written. bf0.5_s202's 2965 s stall was not analysed.
- **Files:** `/raven/ptmp/uli/cpp-nbody-runs/episodes-2026-10-22/` holds cut_* (the step-5000 cut-outs and their
  variants), syn_* (synthetic outer pericentres), eps_* (FLYBY_CM_EPS scan), res_* (DT_MAX commensurability scan) and
  gpu_* (the cluster replays).

## R6 FSI error budget at 200k bf 0.1: eta scans and time-step attribution (analysis; no production change)

- **Question:** the ledger puts 99 % of the per-step error in the FSI (C22). Which criterion, level or class sets it,
  and which cheap local change reduces it most per cost?
- **Diagnostics (4f45cbb, default off, bit-identical vs main on bin10 / bin10+PN incl. the switches at neutral):**
  `NBODY_DT_SCALE=<class>:<f>` (final step of bh / sub / single / all / deep<L> / shallow<L> times f, at every depth:
  attribution by intervention), `NBODY_DT_ATTR=1` (one rank: criterion that set each step per level and class, DTATTR
  lines), ELEDGER call types and |own| per depth; `scripts/step_err.py` (per-step |dE/E0|, s/step from run.log).
- **5 Myr budget (prod200k bf0.1, 3 seeds, 5000 steps):** final |dE/E| 1.25e-8 / 1.26e-8 / 1.36e-8. Steps above 10x
  the median (3e-12) carry 1.5e-8 / 1.3e-8 / 7.5e-9 (s101: one hierarchical BH triple, pids 45751 160657 128932,
  2.07-2.46 Myr, jumps of 3e-10 to 1.8e-9 every ~9 steps; s303: 166 moderate steps of one sign); the background
  (|dE| < 10x median) sums to 2.6e-9 / -0.8e-9 / 6.1e-9 (biased, ~1e-12 per step). Ledger (led_base_s101): the
  background own errors sit in large sets (1e4-1e5 particles) at depths 2-7, signs alternating.
- **Eta scan (s101 restart step 80, 37 steps, 4 A100, nbody-fsi = 4f45cbb):**

  | run | mean \|dE/E0\| per step | rms | max | s/step (median) |
  |---|---|---|---|---|
  | base (FF FB GRAD 0.2, JERK 0) | 3.54e-12 | 4.67e-12 | 1.09e-11 | 5.46 |
  | ETA_FF 0.1 | bit-identical to base | | | 5.47 |
  | ETA_FB 0.1 | 3.03e-12 | 3.94e-12 | 1.10e-11 | 10.26 |
  | ETA_GRAD 0.1 | 3.19e-12 | 4.37e-12 | 1.09e-11 | 5.41 |
  | FF FB GRAD 0.1 | 2.83e-12 | 3.85e-12 | 1.12e-11 | 10.45 |

  ETA_FF never sets a step (pair fly-by step < free-fall step for every unbound field pair; bound pairs are
  subsystems). Halving all etas: -20 % per step at 1.9x cost; a 4th-order truncation error would drop 16x
  (BIFROST Eq. 51, its Fig. 10: 3e-7 -> 5e-9 total). The background is therefore not eta-limited FSI truncation.
  Remaining eta-independent suspects: the tidal-sampling budget FLYBY_CM_EPS (1e-11 |E0| per pair and step, the
  same scale; the EATTR_TID / STALL tidal estimates are 2-6e-12 per binary) and its TIDAL_STEP_RES floor (C5).
- **Prepared, not yet run (session limit):** 0.7x scans, ETA_JERK 0.1, ETA 0.2 (v/a); class interventions
  bh / sub / single / deep5 x 0.5; FLYBY_CM_EPS 1e-12; TIDAL_FLOOR_ADAPT; one-rank ledger + DT_ATTR
  (`/raven/ptmp/uli/cpp-nbody-runs/fsibudget-2026-10-22/*/job.sh`).
- **Decision so far:** no eta default change (no gain per cost). ETA_GRAD 0.1 costs nothing measurable (-10 %, within
  noise; BIFROST's value) and is a candidate after a 3-seed check. The 5 Myr error is set by episodes (one triple in
  s101); the next lever is per-episode (ledger replay of a stall restart), and FLYBY_CM_EPS / TIDAL_FLOOR_ADAPT for
  the background.

## R4 Gauss-IRK on the ADM PN Hamiltonian for PN binaries (research prototype, tests/proto_admpn; no production change)

- **Question (user):** C22 showed that 8-stage Gauss-IRK beats LogH+GBS on Newtonian compact subsystems, but in
  production every LogH subsystem contains a PN pair. Does an ADM Hamiltonian formulation (to 3PN, plus radiation
  reaction) let Gauss-IRK take over the PN binaries, and is it worth a production path?
- **Literature.**
  - Conservative two-body ADM H to 3PN: Damour, Jaranowski & Schaefer 2000 (PRD 62, 044024, Eqs. 3.3-3.6), with
    omega_static = 0 and omega_kinetic = 41/24 from DJS 2001 (dim. reg.). Also Schaefer & Jaranowski LRR 2018 and
    Blanchet LRR 2014 (HADMcm).
  - N-body ADM 1PN: Schaefer 1985. The 3-body 2PN H (Schaefer 1987, typos corrected by Lousto & Nakano 2008,
    CQG 25, 195019) is not implemented.
  - RR in ADM gauge: 2.5PN Hamiltonian of Schaefer 1985 and 3.5PN of Jaranowski & Schaefer 1997. The relative
    acceleration is Koenigsdoerffer, Faye & Schaefer 2003 (PRD 68, 044004) Eq. 5.3, i.e. Iyer & Will 1995 with
    alpha = 5/3, beta = 3 and KFS's delta_i. Harmonic (the production code, Blanchet LRR): alpha = -1, beta = 0.
  - ADM <-> harmonic contact transformation: Andrade, Blanchet & Faye 2001; DJS 2001.
  - Integrators: Seyrich 2013 (arXiv 1302.4580) Gauss collocation on ADM PN Hamiltonians with spins; Tsang et al.
    2015 slimplectic (non-conservative variational) with a PN binary; Wu et al. explicit splittings for PN
    Hamiltonians (2026, arXiv 2607.01596).
  - Regularised few-body codes (Mikkola & Merritt 2008, MSTAR/KETJU, BIFROST, SDAR, SpaceHub) all integrate the
    harmonic EOM with auxiliary velocities (R2).
  - Splitting with a small perturbation: Laskar & Robutel 2001 SBAB_n, error O(eps h^2n + eps^2 h^2).
  - DJS 2000 (arXiv gr-qc/9912092v1) Eq. 4.16 prints 3 i3 E^2 in k(E, j). Their own radial action Eq. 4.12, and
    Blanchet LRR (= secular_pn_k), give i3 E^2.
- **Prototype (tests/proto_admpn).**
  - adm_pn.h:
    - H_ADM as monomial tables, with hand-coded gradient and p-Hessian.
    - N-body: Newton + N-body ADM 1PN (proto_symrk) + pair 2PN/3PN in the pair's relative variables. This is the
      same c.o.m. approximation as production.
    - RR: KFS a_RR applied as a force on p, F = H_pp^-1 a_RR, so the (x, v) dynamics is exactly KFS's.
    - Poincare form K = g (H + p_t) with p_t tracking the radiated work, so Q = H + p_t is the energy-books
      residual.
    - Variant A: RR inside the Gauss collocation.
    - Variant S: Strang split (RR half kick, IRK step, RR half kick); the kick is RK4 at fixed x, and p_t books
      the exact change of H.
    - Variant S2: SBAB2.
  - derive_contact.py (sympy) derives the ADM -> modified-harmonic map through 3PN. contact_coeffs.h holds the
    result.
  - bench_admpn.cpp provides the verify, test, cons, rr and h2h modes. The GBS side is production gbs_step with
    the bench.cpp step control, k_max 10.
  - tests/enzyme/pn_rr_balance.cpp holds the RR energy balance below.
  - Not in ctest. Bounds are 1.3x the measured values; run with `admpn test`.
- **Checks (all pass).**
  - Gradient and Hessian vs FD: 2e-10 and 1e-11. Enzyme gradient of the scalar H_ADM vs the hand-coded one, per
    order: 5e-16 to 3e-14.
  - Map. The 48 equations for 13 unknowns are consistent: the DJS 3PN H and Blanchet's MH 3PN acceleration (the
    production one) are the same dynamics. Coefficients:
    - 2PN: (5/8 nu p^2 - nu/8 pr^2 + (1/4 + 3 nu) u) n - 9/4 nu pr p. The test-mass limit u/4 is the isotropic ->
      harmonic Schwarzschild radius.
    - 3PN: 9 more, including -(2773/280 + 21/32 pi^2) nu u^2.
    - Round trip 2e-16.
  - Circular E(x) vs Blanchet 3PN: residual ~x^4. Test-mass limit vs isotropic Schwarzschild: residual ratio 25.5
    per factor 1.5 in c (c^-8).
  - Periastron advance, e = 0.9 conservative. Relative residual vs DJS k(E, j) at x = 1e-3 / 5e-4 / 2.5e-4 /
    1.25e-4: 1.00e-4 / 1.26e-5 / 1.57e-6 / 1.97e-7, i.e. exactly x^3 (4PN).
    - Failing-first: without the 3PN terms the residual is -4.2e-4.
    - The production MH run (LogH+GBS, E and j through the map) gives 8.9e-5 / 1.20e-5 / 1.68e-6. Same x^3, so
      both truncations reproduce the 3PN invariants.
  - H_ADM, e = 0.9, 30 steps per orbit: max |dH/H| 3.5e-12 over 1e4 orbits and 4.4e-11 over 1e5. This is a
    round-off-level linear drift (slope 0.96), as in the Newtonian prototype (C22).
  - RR energy books: Q = H + p_t stays at 1e-13 to 3e-13 of |E0| over a whole inspiral (all variants).
  - KFS RR vs Iyer-Will(5/3, 3, delta_i): 4e-14. Production harmonic 3.5PN is Iyer-Will consistent too (unique
    delta_i, sympy).
- **Inspiral, e = 0.5, x = 0.01 -> 0.05 (965 orbits).**
  - t_insp vs the Peters + 1PN secular track: A 3.3e-4.
  - Error vs A, at 16 / 24 / 32 / 64 steps per orbit:
    - Strang (S): 1.1e-3 / 5.2e-4 / 2.9e-4 / 7.8e-5 (h^2).
    - SBAB2 (S2): 5.3e-6 / 1.3e-6 / 4.0e-7 / 2.6e-8 (h^4).
  - A is converged to 11 digits already at 16 steps per orbit, with 2600 evaluations per orbit; S2 at 16 needs 4400.
  - At e = 0.9 the splits need many more steps: the RR force is peaked at the pericentre, and the end-point kicks
    under-sample it (h2h table below).
  - **Decision: RR inside the collocation (A).**
- **The 1.1 % inspiral difference vs production (Enzyme, user request): not a bug.**
  - At full PN, t_insp of LogH+GBS (harmonic) exceeds ADM by 1.07 % (x0 = 0.01) and 0.23 % (x0 = 0.005): ratio
    4.7, i.e. x^2.
  - With 2.5PN only: production agrees with Peters to 6e-4, and ADM to 9e-6.
  - pn_rr_balance (Enzyme): orbit-averaged RR power along the conservative 3PN orbit / gauge-invariant <F> - 1, at
    c = 10 / 20 / 40:

    | dynamics | e | c = 10 | c = 20 | c = 40 | scaling |
    |---|---|---|---|---|---|
    | production harmonic | 0.5 | -5.9e-3 | -2.4e-4 | -1.3e-5 | c^-4 or faster |
    | production harmonic | 0.8 | -7.2e-2 | -1.1e-3 | -3.9e-5 | c^-4 or faster |
    | ADM | 0.5 | +1.7e-3 | +1.4e-4 | +9.4e-6 | ratios 12-15 |
    | 2.5PN only | 0.5 | | | | c^-2 (ratio 4) |

  - So both 3.5PN implementations satisfy the 1PN flux balance. Each misses the 2PN flux (4.5PN RR) with a
    gauge-dependent remainder: harmonic -0.6 % and ADM +0.2 % at x = 0.01.
  - The 1.5PN tail (4 pi x^1.5 ~ 1.3 % at x = 0.01) is missing in both.
- **Head-to-head (1000 inner orbits, PN_ORDER 31 unless noted).**
  - Errors are against each scheme's own converged reference (Gauss-A at 160 steps per orbit; GBS at eta 1e-14).
  - dphi is the pair phase at T. Evaluations are rhs calls (Gauss) or leapfrog substeps (GBS).
  - Wall times come from a loaded login node: compare ratios only.

  | case | Gauss-A spo | dphi | evals/orb | LogH+GBS eta 1e-10 dphi | evals/orb |
  |---|---|---|---|---|---|
  | e = 0.9, x = 1e-3, conservative (11) | 24 | 6e-11 | 1990 | 7.8e-8 | 3270 |
  | e = 0.9, x = 1e-3, with RR | 32 | 8.4e-10 | 5650 | 1.0e-6 | 6540 |
  | e = 0.99, x = 1e-4 | 48 (<= 32 diverge) | 4.7e-7 | 28600 | 7.3e-8 | 29200 |
  | s432 BH pair (45 + 12.2, a = 0.37 AU, e = 0.64) | 12 | 5.3e-10 | 1300 | 1.2e-8 | 670 |
  | s1025 BH pair (45 + 35.5, a = 0.05 AU, e = 0.53) | 12 | 5.6e-11 | 1330 | 2.5e-9 | 650 |
  | s3047 BH pair (27.8 + 24.7, a = 0.011 AU, e = 0.21) | 12 | 9.7e-11 | 1330 | 1.8e-9 | 610 |
  | trip432 (s432 pair + 1.25 Msun star at 0.0076 pc) | 12 | 5.2e-10 | 1290 | 4.0e-9 | 690 |

  - trip432 needs the origin at the pair's c.o.m. With the pair 1e-4 pc from the system c.o.m., the IRK fixed-point
    increments stagnate at the round-off of the absolute positions and steps diverge at random.
  - e = 0.99 Gauss (48 steps per orbit): its energy vs the reference is 8.6e-5 of |E_N| (GBS: 1.2e-5).
  - Splits at e = 0.9 with RR, phase error at 24 / 48 steps per orbit:
    - Strang: 0.3 rad / 4e-5 rad.
    - SBAB2: 8e-2 rad / 3e-6 rad.
  - Wall per orbit (same load): e = 0.9 conservative, Gauss 0.7 ms vs GBS 2.5 ms. Harvest pairs: Gauss spo 12 is
    1.1-1.4 ms vs GBS 0.56-1.2 ms.
  - Production quirk: at eta 1e-8 the GBS step control spends 13000-26000 evaluations per orbit in 3 of 6 cases
    (e = 0.9, s1025, s3047), 4-20x more than at 1e-10. The other cases use about 530. Not followed up.
  - Physics difference ADM vs MH, references mapped to MH:
    - Conservative e = 0.9: dphi 1.2e-3 rad after 1000 orbits (4PN truncation difference, k differs by 1e-5).
    - Harvest pairs (x = 1e-6 to 5e-5): 1e-9 to 2e-8 rad.
    - With RR at x = 1e-3, e = 0.9: 21 rad, the accumulated 2PN-relative flux difference above.
- **Verdict.**
  - On isolated and weakly perturbed PN binaries with e <= 0.9, ADM + Gauss-A is 10-1000x more accurate in phase
    and energy than LogH+GBS at eta 1e-10. On the harvest pairs (the production BH binaries) it needs about 2x the
    evaluations for 20-50x smaller phase errors; at e = 0.9 it is cheaper and about 1000x more accurate.
  - It does not help at e >= 0.99: the fixed-point iteration diverges at the pericentre below about 48 steps per
    orbit, and then the cost equals GBS (as in R2/C22). It would need Newton iterations or adaptive ds.
  - **A production path is not recommended now.**
    - The cluster energy error is not set by the subsystem integrator (R2, R3, C22: IRK calls already 100x more
      accurate per call left |dE/E| unchanged).
    - The production books use Newtonian T + U from states, and are exact for either formulation.
    - The ADM path would need the coordinate map at every subsystem entry and exit (positions differ by O(c^-4),
      momenta by O(c^-2)), N-body 1PN only (2PN 3-body Schaefer 1987 for triples), a separate RR gauge, and a GBS
      fallback for e >= 0.99 and collisions. The inspiral differs from the harmonic one by O(x^2) (1 % at
      x = 0.01): within the PN truncation, but a visible physics change against the current runs.
  - Worth revisiting if long-lived PN binaries in isolation (e.g. a separate few-body GR library) need phase
    accuracy, or if C22's IRK becomes default and PN subsystems dominate
    the cost.

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
- **s303 determinism check:** a base-only rerun (job 31374530) reproduces base_s303 bit for bit (final 2.63e-11), so the
  s303 gap (3.66e-11 with IRK) is caused by the 370 IRK calls changing the trajectory, not run-to-run noise. Each IRK
  call has ~100x less error than a GBS call, so the global difference is most likely chaotic divergence; the energy
  rule still fails on s303 and NBODY_GAUSS_IRK stays default off. Production value needs PN subsystems covered.
