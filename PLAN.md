# cpp-nbody: development plan to match FROST / BIFROST / FROST-CLUSTERS in accuracy and speed

## Context

Goal (user, 2026-10-10): make cpp-nbody (Kokkos; HIP on MI300A, CUDA on A100) at least as accurate and as
fast as the published Rantala codes. The viper session that held the working plan crashed; everything since
37aa7bf was built only with HIP; commit 7a966b2 (secular-triple regression fix) was never checked on a GPU;
work moves to raven (4x A100 per node — the hardware BIFROST was timed on). A newer copy of the repo with a
literature plan exists on GitHub but cannot be fetched from raven (no credentials; user pulls it).

Papers re-read from arXiv LaTeX, stored in `/raven/u/uli/papers/rantala/` (PDF + `src/<id>/manuscript.tex`):
MSTAR 2001.03180, FROST 2011.14984, BIFROST 2210.02472, FROST-CLUSTERS I 2403.10602, II 2506.04330,
III 2601.07917, letter 2503.21879.

User decisions: (1) add SDAR inside the LogH/GBS batch and A/B it against LogH + secular, switching the
default only when equal or better; (2) accuracy gates first, then speed — every speed change keeps |dE/E|
equal or better (standing rule); (3) scope includes collision upgrades, stellar-evolution upgrade, spin +
MSTAR, and far field: **tree for time-step/neighbour assignment, FMM to take the far-field force work at
large N**.

## Targets from the papers

Accuracy (eta_ff = eta_fb = 0.2, eta_grad = eta_jerk = 0.1, eta_GBS 1e-10, interval 1e-3 Myr):
- FROST: Kepler e = 0/0.9/0.99 over 1000 P follows eta^4, floors 1e-13 / 1e-12..1e-10 at eta 0.008, dP = 0;
  Sun + 4 giants (HHS) 1e-13 at eta 0.004, dL 1e-14; 1e6 softened Plummer 2e-10 after 12 Myr.
- BIFROST Fig 8: 162k equal-mass unsoftened, per interval 8e-11, linear growth, 35 Myr.
- Fig 9: 200k evolved Kroupa (357 BHs, 5.5-41.6 Msun) 3e-7 at 5 Myr; without BHs 2e-8;
  Fig 10: all eta halved -> 5e-9, per interval < 5e-11.
- Fig 11: 200k, f_bin 50 %: per interval 7e-10, total 6e-7 at 5 Myr.
- Fig 12: IMBH inspiral + recoil 1e-6. Fig 13-15: N = 20k, f_bin 10 %, core collapse: 3e-4 at 4 Myr.

Speed (one raven node, 4x A100, per 1e-3 Myr interval): BIFROST Fig 7: N = 1e4 / 1e5 / 1e6 ->
0.12 s / 1.3 s / 58 s; f_bin 10 % -> 100 % costs +25 % (1e4, 1e5), +3 % (1e6). Strong scaling to
~40 N/1e6 GPUs. (Note: memory file `rantala-papers.md` says 5.8 s for 1e6 — wrong by 10x; fix it.)

Method changes after BIFROST that the current Rantala code uses:
- FC-I A1: Omelyan gradient, ã(r) ≈ a(r + ε²/24 a) — same kernel as Newtonian forces.
- FC-I A2: SDAR (Wang+2020) replaces LogH and the secular triple; GBS k_max 8; κ from perturbers
  (k_ref 1e-6), κ_max from c = 0.1 perturbation-change estimate; κ = 1 for PN subsystems.
- FC-I B3: collisions decided inside regularisation; osculating-pericentre candidates are stepped to
  the actual pericentre with κ = 1.
- FC-II: fly-by time-step of binary members uses the binary c.o.m. velocity; octree for neighbour and
  time-step searches (forces stay direct); r_strong = 10 a; production eta 0.2 (ff/fb/grad),
  r_subsys 0.5 mpc, eta_GBS 1e-8, end-time 1e-2, interval 5e-3 Myr.
- FC-III: node-local MPI shared-memory windows; Glebbeek collision mass loss f = 0.3 q/(1+q)²;
  Vink 2018 winds above 100 Msun; Mapelli-type rejuvenation.
- BIFROST §6 lists slow-down for perturbed binaries as missing — the same physics as our open
  dense-core floor (tidal coupling of unresolved binaries sampled by FSI kicks).
- MSTAR: MST coordinates, Deuflhard sequence, polynomial extrapolation, end-time step Δs = U(Δt − τ),
  parallel force loop × substep divisions (k_fix 8).

## Where the code stands (survey, file:line in `src/`)

Matches the papers: HHS recursion and FSI with subsystems (`integrator/integrator.cpp:1044-1161,
2299-2437`), pair exclusion (`gravity/n2_driver.h:32`), symmetrised ff/fb (`core/constants.h:83-116`),
gradient/jerk/v-a/perturbation criteria, level-0 clamp, shared subsystem step, global PN1, LogH with
exact cubic Ω and aux-velocity PN, GBS polynomial extrapolation, exact energy books, restarts.

Bugs found:
1. **Secular-triple driver uses `particle_ids[0..2]` (pid order) as inner pair + outer**
   (`integrator.cpp:646-660`, write-back ~760) while classification uses `inner_binary_id` /
   `outer_companion_pid` (`collisional/subsystem.cpp:500-545`). Members are pid-sorted
   (`subsystem.cpp:342`), so in ~2/3 of triples the "inner binary" is a wide pair and the write-back
   scrambles the triple. Prime suspect for the 2.8e-9 late-energy regression; 7a966b2 only sidestepped it.
2. `AccDtK` jerk term not guarded by `eta_jerk > 0` (`gravity/n2_kernels.h:195`).
3. Secular binary step uses ω/ω̇, which goes to 0 near ω = 0 (`collisional/secular.cpp:127-140`);
   secular merger at r_p < 6GM/c² only warns (`:458-462`).
4. Spin-orbit coefficient uses r×v without μ (`collisional/post_newtonian.cpp:320-331`); spin update is
   explicit Euler (|S| drifts).
5. Gradient time-step uses a stale `grad_mag` (cross kernel overwrites self); v/a uses `ETA`, which is
   not halved on restart.
6. Triple stability test non-standard (no inclination factor, uses a_in(1+e_in))
   (`subsystem.cpp:210-221`).

Gaps vs papers: no Omelyan option; `FLYBY_CM_VELOCITY` exists but off and GPU-only; neighbour search
O(N_set²) at every level, no octree (`gravity/forces.cpp:188-232`); GBS uses the Bulirsch sequence,
k_max 10, end-time one secant rescale at 1e-4, host-thread batching only; secular triple is
Blaes/Naoz forward Euler with frozen outer orbit; no slow-down (`LogHState::ext_accel` unused);
MSTAR code never called, groups > 8 fall back to FSI; mergers checked once per level on current r,
2-body only, no collision mass loss, TDE accretion 0, recoil not Zlochower and ignores spins; runtime
SSE once per top-level step without ε_se coupling, deterministic natal kicks, BH masses 3-27 vs paper
5.5-41.6; no MPI shared windows; FMM exists (`gravity/fmm.cpp`, spherical harmonics p = 12, host
interaction lists) but not wired; `Softening` spline struct unused.

Build/test state on raven: `/u/uli/kokkos-cuda-mpi` (Kokkos 5.0.99, CUDA AMPERE80) exists; `build-mpi/`
is configured against it but its binary is from March. No HIP-only constructs found; unified-memory
paths compile out on A100, so `SERIAL_N2_CPU` is the only small-set lever there. CTest: 30+ cases in
`CMakeLists.txt:128-147`, checks in `tests/run_case.py` (bounds measured on laptop/MI300A).

## Milestones

Each step: failing test first, bounds 1.3x measured, A/B by env var where behaviour changes, multi-seed
when noisy, notes in `VIPER_NOTES.md` (rename to `NOTES.md` later), commit per step.

### M0 — Raven bring-up and reconcile
1. User pulls the GitHub version (`! git -C /raven/u/uli/cpp-nbody pull origin main` with a token, or
   add `~/.ssh/id_ed25519.pub` to GitHub). Read its development plan and merge its items into this one.
2. CUDA build in `build-mpi/` (recipe in `VIPER_NOTES.md` hand-over section); host build against
   `/u/uli/kokkos` (Serial+OpenMP) in `build-host/`. Fix nvcc compile issues (watch `n2_driver.h`,
   `n2_kernels.h`, `double4` in `constants.h:44`).
3. Host ctest, then GPU ctest in a gpu job (`NBODY_TEST_TIMEOUT` raised). Re-measure bounds per
   platform (A100 vs MI300A vs x86) and record them in `tests/run_case.py`.
4. Tune `SERIAL_N2_CPU` / `SERIAL_N2_GPU` for A100 with `n2-bench` and the profiler.

### M1 — Correctness bugs and the open energy regression (gate for everything after)
1. Failing test: stable hierarchical triple whose outer body has the lowest pid, in the secular regime;
   check |dE| and that the inner a, e are preserved. Fix bug 1 by building the triple from
   `inner_binary_id`/`outer_companion_pid`.
2. Seed A/B on raven: 200k, 10 % binaries, 30 steps, seeds 42/7/13: f8676d6 vs 7a966b2 vs bug-1 fix,
   with and without the 3-member exclusion of 7a966b2. Gate: late |dE/E| ≤ f8676d6 on every seed
   (5.4e-10 reference). If the fix alone restores it, revert the exclusion.
3. Bugs 2-6 each with a small test (jerk guard, ω ≈ 0 secular step, spin μ / |S| conservation with a
   rotation update, fresh grad_mag per kernel, restart halves `ETA`, Mardling-Aarseth with inclination).
4. Loosen `floor_planets` L bound after step 2 (round-off only).

### M2 — Paper reproduction suite (the accuracy and speed baselines)
Scripts under `tests/paper/` (ICs, config, expected numbers, plot): FROST Kepler/planets (already in
ctest), BIFROST Fig 8 (162k equal mass, first 5 Myr), Fig 9 + Fig 10 (200k with/without BHs, eta and
eta/2, 5 Myr), Fig 11 (50 % binaries), core-collapse (20k, 10 %, r_h 0.5 pc, 4 Myr), Fig 7 timing grid
(N = 1e4, 1e5, 1e6 × f_bin 10/50/100 %, one interval, mass-size relation as BIFROST App. A).
IC generator additions needed: BIFROST-style binary population (log-uniform a, thermal e, uniform q
above 5 Msun instead of mass-rank pairing), FC-II multiplicity (Moe & Di Stefano / Offner), primordial
triples. Output: a table of |dE/E| per interval and total, and s/interval, per run — the baseline every
later change is measured against.

### M3 — Subsystem integrator upgrade (SDAR, A/B vs LogH + secular)
1. GBS: Deuflhard sequence and k_max 8 as options; MSTAR end-time control (Δs = U(Δt − τ) before
   overshoot, iterate to tolerance); test FC-II tolerances (1e-8, 1e-2) vs ours (1e-10, 1e-4) for cost
   and energy on M2 cases.
2. SDAR inside `collisional/logh.cpp` / `logh_batch.cpp`: slow-down Hamiltonian H_sd (FC-I Eqs A3-A8),
   inner-binary identification in each subsystem, κ per binary from its in-subsystem perturbers
   (k_ref 1e-6), κ_max (c = 0.1), jumping-κ between GBS steps, κ = 1 for PN members and collision
   candidates. Env switch `NBODY_SDAR`. Tests: isolated triple vs LogH reference (secular-averaged
   evolution agrees, energy books exact); Kozai-Lidov triple vs MSTAR paper Sec 5.3; binary-binary.
3. Use SDAR for stable triples in place of the secular triple (A/B); keep the secular binary for
   isolated short-period binaries (BIFROST still does).
4. Dense-core floor (r_h 0.03 pc, 2000 stars, 50 % binaries; test `dense_core_split` exists): try,
   in order, (a) perturbers as subsystem members (larger r_ngb for binaries, needs > 8 bodies → M4),
   (b) tidal tensor of the field applied inside LogH/SDAR through `ext_accel`, (c) resolution criterion
   (`TIDAL_GAMMA`). Gate: per-step error converges with eta again.
5. Switch defaults only if M2 Fig 9/11 and the dense-core case are equal or better.

### M4 — MSTAR and spins
1. Regularised integrator with MST coordinates (Prim's algorithm; reuse `collisional/mstar.cpp` MST
   code), N_d = 2 rule via lowest common ancestor, polynomial GBS; lift `LOGH_MAX_BODIES` for groups
   routed to it; parallel substep divisions over host threads (k_fix 8) and the force loop on device for
   IMBH regions of ~100-1000 bodies.
2. Tests: Pythagorean 3-body, eccentric binary 1e-14, IMBH + cusp region vs plain LogH.
3. Spin evolution for every PN subsystem (not only 2-body BH), rotation-preserving update, J = L + ΣS
   conserved; spins carried through mergers (remnant spin from Zlochower fits).

### M5 — Collisions and mergers (FC-I B3, FC-III)
Collision candidates from osculating pericentre inside the regularised step, forced steps to pericentre,
decide there (TDE r_t = 1.3 (M/m)^{1/3} R, BH-BH at ISCO, star-star R1 + R2); detection for any group size;
Glebbeek mass loss (C = 0.3) as option; TDE accretion fraction (0.5 default); Zlochower 2015 recoil with
Geneva spins; Peters merger time from current elements; secular-binary merger executed, not only warned.
Tests: head-on star collision, TDE on parabolic orbit, BH binary to ISCO with recoil, exact books.

### M6 — Stellar evolution
1. Runtime SSE inside the FSI with per-star t_next and ε_se in the time-step (FC-I B2); stochastic
   natal kicks (Hobbs Maxwellian, fallback scaling); fix `SSE_ENABLED` default and the 37 % mass loss at
   first call.
2. Vink 2001/2018 winds and Fryer 2012 delayed/rapid remnants (should move BH range toward 5.5-41.6);
   VMS radii and winds above 600 Msun (FC-III table).
3. Decide later (needs the external code): SEVN library coupling with binary evolution and r_strong
   (FC-II), or BSE.

### M7 — Speed, each change under the energy rule (after M1, measured on M2 cases)
1. Profile 200k (0 / 10 / 50 % binaries) and 1e5/1e6 Fig 7 cases on 4x A100.
2. `FLYBY_CM_VELOCITY` on all paths (CPU, HHS_CC) and A/B as default (FC-II). (done; rejected as default by the production A/B and deleted 2026-10-12; restored 2026-10-13 as an option, off by default, on top of the N^2 kernel speedups)
3. Omelyan gradient as option (`GRADIENT=omelyan`), A/B accuracy on M2; makes the FMM gradient a
   second force evaluation.
4. Tree for time-step and neighbour assignment (user requirement): octree on device, exact pruning so
   that the min-reduction and neighbour lists are bit-identical to O(N²) (cell bound on ff/fb/jerk
   and on r < r_ngb_max), used above a set-size threshold at the top levels.
5. Subsystem cost: true device batching of LogH/SDAR for many small subsystems, or keep host threads
   if faster on raven (72 cores/node); BIFROST's binary fraction is nearly free — ours is not yet.
6. MPI: node-local shared-memory window for the particle copy (FC-III), fused allgathers per call,
   then multi-node scaling to ~40 N/1e6 GPUs.

### M8 — Far field (FMM) for N ≳ 1e6
GPU interaction lists for `gravity/fmm.cpp`; near field direct and unsoftened (keeps the collisional
force exact locally), far field from expansions; symmetric (Dehnen) cell-cell interactions so kicks stay
momentum-conserving; gradient term via Omelyan (M7.3) or from local expansions; used only on HHS levels
above a set-size threshold; optional Barnes randomisation of the tree frame. Gates: FMM vs direct on a
1e6 Plummer — force error, |dE/E| ~1e-10 over the FROST 12 Myr test, dP, dL; s/interval below direct.

## Verification

- Every step: host ctest + GPU ctest on raven green; new test fails on the old code.
- M1 gate: seed A/B table (3 seeds) with late |dE/E| ≤ f8676d6.
- M2 onwards: the paper-suite table (|dE/E| per interval and total, s/interval) re-run for every
  default change; no row may get worse in energy; speed rows compared to BIFROST Fig 7.
- Long runs: Fig 9 with BHs to 5 Myr (target ≤ 3e-7, eta/2 ≤ 5e-9), Fig 11 50 % binaries
  (≤ 6e-7), core collapse (~3e-4).

## Bookkeeping after approval
- Save memories: papers location + FC-II/FC-III method changes; correct the 1e6 timing in
  `rantala-papers.md`; triple-driver bug.

---

# Plan update 2026-10-11 (after METHODS_SURVEY, HARD_PROBLEMS, SCALING_DESIGN, PARALLEL_ASYNC_REVIEW, AUDIT F/G)

Source documents in the repo (each has the details and references): METHODS_SURVEY.md (methods + experiment list),
HARD_PROBLEMS.md (frontier problems, cost, gaps, flagships), SCALING_DESIGN.md (batching + task parallelism + cost model),
PARALLEL_ASYNC_REVIEW.md (parallel-in-time, async, runtimes, shared memory, MPI/NCCL), AUDIT_2026-10.md (F1-F7, G).
Only the items below are adopted; everything else in those documents is reference, and the "rejected" list says why.

## Goals (user-confirmed)
- A: BH-rich 1e6 cluster, observed multiplicity, through BH-driven core collapse, full PN BH dynamics (stepping stone 2e5).
- B: dense young cluster, full massive-star multiplicity, runaway collisions / IMBH.
- C: 1e7 through core collapse (needs FMM + multi-node).

## Standing rules (added today)
- Speed = sustained production speed (Myr/day over long runs), first-interval timings only as BIFROST footnote.
- Energy rule unchanged (3 seeds equal or better); at the 1e-11 noise floor judge by distribution over >=5 seeds.
- GPU busy fraction (nsys) and the 1-16-node scaling curve at 1e6 are standing metrics next to s/step and |dE/E|.
- Results must become rank-count independent (SCALING_DESIGN §6); new parallel stages sum in canonical order.
- Build stack: gcc/15 cuda/13.2 openmpi_gpu/5.0 (MPCDF); per-compiler test bounds.

## Status of M0-M8
- M0 done (raven builds, new stack built). M1 gate passes on 3 seeds after the seed-7/13/42 fixes.
- M2 partial: Fig 7 like-for-like done (fig7-2026-10-11); Figs 8/9/11 long runs and core collapse open.
- M3: SDAR implemented (default off; no gain at 200k now); collisions in LogH done (M5 part).
  Kepler fix + KEPLER_2BODY on branch worktree-agent-a1a708bf08cc7e185: -12..27 % but 200k energy worse on 6/9
  seeds from the universal-variable secular change -> find cause, then merge.
- M5: collisions decided at pericentre inside LogH/GBS merged; open: TDE accretion 0.5, FC-I BH-BH radius.
- M7: trees, F2 overhead, LogH/GBS bit-identical speedups, GPU audit items (fences, energy check, transport,
  fused collectives, DtK, tree dt with tidal step, fp32 pre-reject) merged; c.o.m. velocity default pending prod runs.
- New: HDF5 restarts (bit-identical) + LUNA-compatible snapshots merged (HARD_PROBLEMS G1 done).

## Adopted next work, in order
1. Verify + finish: GPU check chain (verify-gpuopt/), gcc15 test bounds, Kepler energy issue, prod runs -> c.o.m. default.
2. Long-run enablers (HARD_PROBLEMS G2/G3): Lagrangian radii, core radius/density, binary census, event logs; external
   tidal field.
3. Scaling phases (SCALING_DESIGN §8): P0 measure (CC census, nsys_busy, graph/NCCL microbenchmarks) -> P1 rank-count
   independence (global chunk/tile rule, canonical merges, deterministic ids) -> P2 deep-level batching (control block,
   device pid map, fused small-set kernels, Kokkos graphs) + host-resident small subtrees (REVIEW #2, 3 seeds) ->
   P3 TaskContext, S||F and component concurrency -> P4 owner-computes tasks across ranks, NCCL joins, node windows.
   GPU-idle remedies from the review: event polling instead of fences, LPT order of LogH (REVIEW #6), few-sink kernel
   shape (REVIEW #7, 3 seeds), parallel GBS columns for stragglers (REVIEW #10).
4. Validation ladder (HARD_PROBLEMS §6.2): equal-mass collapse 16-32k; mass-spectrum collapse; BIFROST §5.6;
   Rizzuto-type; DRAGON-II-like; Fig 9 to BH collapse; 1e6 flagship A.
   4b. Head-to-head with PeTar (user, 2026-10-12; github.com/lwang-astro/PeTar, Wang et al. 2020: P3T tree + SDAR +
   slow-down, galpy tidal field, the closest public code in scope). Same ICs written for both (cpp-nbody snapshot ->
   PeTar input), N = 1e4 / 1e5 / 2e5, without and with 10-50 % binaries, isolated and in a tidal field; compare
   |dE/E| per interval and over the run, wall time per Myr on one raven node (PeTar MPI+OpenMP(+CUDA tree)),
   and the physics (Lagrangian radii, core collapse time, escaper rate, binary census) on seed ensembles.
4c. SDAR per pair inside PN subsystems (2026-10-14): keep kappa = 1 only for pairs with a PN member, allow the
   slow-down for non-PN inner binaries of PN subsystems (needs SDAR in the PN / auxiliary-velocity LogH variant).
   Motivation: BIFROST Fig. 11 run (200k, 50 % binaries) stalls on a 2+2 quadruple, BH binary 20.7 + 14.1 Msun
   (a = 58 AU) with a 0.79 + 0.105 Msun binary (a = 0.019 AU, ~1000 orbits per interval, tidal ratio ~4e-12) bound
   to it at 0.003 pc: the whole subsystem has kappa = 1 (FROST-CLUSTERS I App. A2), LogH hits its step limit, steps
   take > 3 min. Check FC-I's reason for kappa = 1 per subsystem first; failing-first test built from this quadruple.
   Status 701b5a7: implemented behind NBODY_SDAR_PN=1 (default off). FC-I gives no reason for kappa = 1 ("in this
   study we do not use the slow-down for PN subsystems"). quad22_sdar passes (0.44 s vs no step in 120 s). Open: 200k
   A/B from a restart of the fixed-IC Fig. 11 runs (bifrost-fixed-2026-10-14) at a PN-subsystem stall, then default.
5. Physics for A/B: multiplicity ICs (Moe & Di Stefano / Offner, triples, quadruples), spin-dependent recoil
   (Zlochower-Lousto), runtime SSE upgrade (M6), MSTAR for >8 bodies (M4), BSE/SEVN.
6. Accuracy study for the far field (experiment I-1; statistical equivalence on ensembles) before M8 FMM.
7. Production practice: N <= 2e5 as ensembles (4 seeds x 1 GPU = 2x throughput); --restart auto job chains.

## Experiment backlog from METHODS_SURVEY (ranked table), with status
| id | change | status / next |
|---|---|---|
| B-1 | FC criteria set: c.o.m. velocity, jerk off, v/a off, eta_grad 0.2 | c.o.m. part rejected as default (prod runs 2026-10-12), kept as option FLYBY_CM_VELOCITY (off); jerk/v-a off ("b1") mixed at 20k -> 200k 3-seed A/B still open |
| F-1 | secular triples off -> LogH | switch SECULAR_TRIPLES merged (default on); no triples formed at 20k; decide with SDAR on 200k |
| C-1/C-2 | weak-perturber split; field sees c.o.m. of split binaries | C-1 alone 3.6x worse (WEAK_PERTURBER_GAMMA deleted 2026-10-12); C-2 not done (SDAR + collisions removed most stalls; revisit only if stalls return) |
| E-1/E-3 | eta_GBS 1e-8, end-time tol 1e-2 (FC production values) | 20k: no speed gain; repeat at 200k after Kepler path (less LogH work left) |
| D-1 | SDAR | done, default off (no gain at 200k now; keep for dense/triple-rich runs) |
| J-1/J-2 | per-depth overhead, host path for small batches | J-1 done (F2/AUDIT G); J-2 = SCALING P2 host-resident subtrees |
| I-1 | synthetic force-error injection | open; required before M8 FMM and for the far-field accuracy study |
| K-1 | FRYER_RAPID + VINK + PPISN in IC SSE (BH masses 5.5-41.6) | open, config-only test |
| A-2 | HHS_CC default with binaries | open; prerequisite for SCALING P3/P4 task volume (3 seeds) |
| C-3 | Mardling-Aarseth with inclination / Vynatheya | branch 777b0db held; merge with a test |
| E-2 | Deuflhard sequence, k_max 8, MSTAR end-time control | open (1.2-1.5x in GBS) |
| B-2 | tidal-tensor time-step criterion | open (research) |
| C-4 | r_ngb 0.5 / 0.25 mpc (FC values) | open; interacts with NGB_MASS_SCALING |
| A-1 | Omelyan gradient | open; needed for FMM gradient |
| G-1 | PN boost-invariance test | open (cheap test) |
| H-1/H-2 | exact tree search | done (merged, bit-identical, now also with the tidal step) |
| B-5 | interval 5e-3 Myr (FC production) | open; production-speed lever, energy rule per Myr |
| B-3 | Galilean-invariant v/a | open, only if v/a kept |
| J-3 | device-batched LogH/SDAR | rejected (subsystem audit #14, review §5.4) |
| K-2/K-3 | pericentre collisions; ZL15 recoil with spins | K-2 done (M5); K-3 open (needed for goal A) |
| gradmag fix 3 | fresh grad_mag per kernel (branch ab-gradmag) | better energy, 28 % slower at 200k: re-test with B-1 |

Survey corrections already applied: SSE memory note fixed; F3 understood as a dragged-in fly-by (now handled by SDAR,
collisions, FLYBY_NGB_CM); r_strong is only a BSE switch; FMM needs eps <~ 1e-8 at today's accuracy.

## Deleted options (decided A/B comparisons: winner hard-wired, losing path and key deleted)
The code before the deletions is tagged archive/pre-step2-deletions.

| key | winner (evidence) | status |
|---|---|---|
| ALL_SLOW_SPLIT | true: fsi(S, tau/2) twice (6655494: 16x lower error, same time) | deleted 2026-10-12, winner hard-wired |
| SUBSYS_COLLISIONS | true: collisions decided inside LogH/GBS at the pericentre, FC-I App. B3 (7de902c, 043190d) | deleted 2026-10-12, winner hard-wired |
| WEAK_PERTURBER_GAMMA | off: no weak-perturber split (C-1, 7c5b3ca: 3.6x worse energy); test weak_flyby_split deleted | deleted 2026-10-12, winner hard-wired |
| FLYBY_CM_KICK, FLYBY_CM_KICK_EPS | off: no c.o.m. kicks of weakly perturbed binaries (d83324d; 200k seed-3 stall step: \|dL/L\| 1.4e-9, 15 % faster); tests cm_kick_* deleted | deleted 2026-10-16, winner hard-wired |
| (weak-perturber split back) | WEAK_SPLIT_GAMMA, default off, 1809570 (option by user decision) | 2026-10-16 |
| TIDAL_GAMMA, TIDAL_RES (+ NBODY_TIDAL_DEBUG) | off: no orbit-resolving step for strongly perturbed binaries (fe207ed, 6278068: no benefit) | deleted 2026-10-12, winner hard-wired |
| MPI_TRANSPORT = device | host staging (f24a76b: 3488 vs 1028 us; no UCC/hcoll on raven); key stays for host vs nccl on USE_NCCL builds (auto) | value deleted 2026-10-12; MPI_NON_BLOCKING gathers are now always blocking (the Iallgatherv path was device-only) |
| NBODY_SECULAR_PREC_STEP | off: no 1/omega-dot limit on the secular step (4ebc740: 100000-step cap per evolution) | deleted 2026-10-12, winner hard-wired |
| NBODY_STELLAR_CACHE=0 | host stellar-property cache (a728868: device gathers cost 0.4-0.6 s/step at 200k) | deleted 2026-10-12, winner hard-wired |
| NBODY_DETECT_TOP_ONLY | finished diagnostic (fe207ed): detection at every HHS depth | deleted 2026-10-12 |
| KEPLER_2BODY | true: isolated non-PN pairs on the Kepler drift at any orbit count (9efdca7: 12-27 % faster); PN pairs stay on LogH; kepler2_* tests keep the Kepler-vs-analytic checks only | deleted 2026-10-12, winner hard-wired |
| BINARY_COM_V | sampled: one f(E) draw per binary c.o.m. (a8728fb; average gave Q_obj 0.336, cold binaries) | deleted 2026-10-12, winner hard-wired |
| DISTRIBUTE_SUBSYSTEMS, DISTRIBUTE_MIN_SUBSYSTEMS | off: every rank advances every subsystem (5eb53e1: the exchange was slower; SCALING_DESIGN: retire); tests mergers_mpi2_dist, mpi_1v2_dist, restart_mpi2_dist deleted (duplicates of the non-distributed cases) | deleted 2026-10-12, winner hard-wired |

## Pending evaluations (agents stopped mid-work)
- Task runtimes (HPX/Charm++/Legion): literature says tasks must be >= ~100 us; evaluate only at subtree granularity.
- GPU idle prototypes (uncommitted in worktree agent-a80a0259e1d880784): overlap of host LogH with device forces.
- Ask MPCDF for an Open MPI build with UCC (GPU-aware collectives); NCCL for CUDA 13.2.
- Check viper Kokkos is built with GFX942_APU.

## Rejected (with reason; do not redo)
- Parareal/PFASST/MGRIT on the whole cluster (chaos: O(1) convergence window). Optimistic PDES / Time Warp.
- General AMT rewrite at kernel granularity (tasks 1-10 us << 100 us). CUDA dynamic parallelism, Kokkos tasking.
- LogH/GBS on GPU (lockstep divergence; CPU better), SIMD across subsystems (+22 % only).
- SoA->AoS repacking, maxrregcount/launch bounds, splitting DtK, SERIAL_N2_CPU path as is, lower SERIAL_N2_GPU.
- One process per node for 4 GPUs; MPI RMA on GPU windows; Kokkos Remote Spaces as dependency; managed memory on A100.
- Device-buffer CUDA-aware Allgatherv on raven (no UCC/hcoll; host staging or NCCL instead).

## Speed plan for the flagship runs A, B, C (2026-10-15)

Baselines, one raven node (4x A100), current main, fixed ICs (ptmp abc-2026-10-14; nsys profiles in prof/):
- A (1e6, 50 % binaries, evolved 1 Gyr): 210 s/step. GPU-bound, 70 % busy: ForceK 40 %, GradK 24 %, DtK 20 % (5x the
  force cost per pair), energy-check potential 4.4 %; ~90k syncs + 255k launches/step (2-3 s).
- B (1.1e5 young dense, r_h 0.6 pc, 10 % binaries): 630 s/step. Launch-bound, GPU 13 % busy: the FLYBY_CM_EPS step
  of stars near massive binaries keeps ~5000 particles at levels 8-27, ~1M hierarchy calls/step, each ~50 launches
  + ~35 tiny copies (85 of 150 s in CUDA API calls); host LogH ~25 %. Tidal-tensor mode (branch tidal-tensor) gave
  68 s/step but energy/L not exact on 200k: parked.
- C (1e7): the IC generator did not finish in 15 min; direct O(N^2) needs FMM.

Order (gate in brackets; "exact" = bit-identical, host ctest + dump compare):
1. Fewer launches/copies/syncs per hierarchy call: batched scalar transfers, small-set fused kick/drift, one
   subsystem pull/unpack pass (exact). B 1.3-2x.
2. Cheaper energy check (diagnostics only). A, C ~4 %.
3. Nsight Compute of DtK/ForceK/GradK, FP64 kernel tuning (energy rule, 3 seeds). A 1.2-1.5x.
4. Multi-node scaling of A, 4 and 10 nodes (throughput).
5. Host-resident deep subtrees (exact). B 3-5x.
6. IC generator at 1e7 (same ICs as today's generator).
7. CPU subsystem work overlapped with GPU kernels (exact). A, B up to 1.2x.
8. 4c default, GBS Deuflhard / k_max 8 (energy rule). B LogH share.
9. FMM far field after the force-error study (statistical equivalence + energy). A, C several x.
10. Mixed precision (double-single) against the 1e-8 force-error limit (energy rule). A, C kernels 3-5x.

## Status 2026-10-21 (Raven login crash; GPU work moves to Viper) — see docs/HANDOFF_2026-10-21.md
- Merged today: C15 (diagnostics read-only), C21 (SDAR kappa with outside perturbers; 200k A/B owed), C22 (Gauss-IRK
  for compact Newtonian subsystems, default off; 200k A/B done, s303 undecided). CASES.md has the numbers.
- 4c/C20 (secular inner binaries of PN subsystems): the stall class behind 6 of 8 harvested stalls is gone (worst
  step 187 s -> 20 s at 200k); the main-branch s101 comparison is still queued on Raven (job 31366161).
- Next, in order: (1) C21 200k A/B on Viper apudev, (2) C22 default decision (longer A/B or 4th seed, "not chosen"
  counters), (3) MSTAR for the 16-18 body dense-core groups (the remaining stall class), (4) items 2-7 above.
- Rejected today, do not redo: R2 (1/(-U) kick) retest and R3 (Kahan) — energy rule; PN1 Hamiltonian for IRK.

## Energy report watertight (2026-10-21 late) — work items W1-W8
Measured: 200k bf0.1 s101, 26 steps, all 487972 PN LogH calls re-integrated at ETA_GBS x 1e-4 (job 31378360):
plain T+U difference 3.6e-7 E_REF was orbital phase of the 1PN E_N oscillation; phase-independent
Q = T+U+E_PN(conserved function) differs by sum|.| 5.0e-10 (signed 9.4e-11) vs reported 9.5e-11 -> real hidden error
~5x reported, 40 % from one poorly resolved call (12.2+45 Msun, a 0.37 AU, e 0.64, ~11 GBS steps/orbit). Cut-outs
s1408/s706: Q 1e-16..1e-17. s432 (sec-inner off): Q 6.5e-10/interval at ETA_GBS 1e-10 (real GBS error).
Work items (in order):
W1 Local tolerance rule for PN calls (ETA_GBS scan 1e-10..1e-13 on s432 / the 12+45 Msun call; cost vs Q error).
W2 Exact GBS end-time landing (today NBODY_GBS_TTOL 1e-4 of call length -> members off the step time): cost on
   Newtonian + PN calls, default if affordable (A/B, energy rule).
W3 3PN EOM consistent for unequal masses (general-frame 3PN, Blanchet LRR acc3PN): today a bounded c^-6 swing
   proportional to the mass difference; failing-first test (remainder ~c^-8).
W4 Honest PN books: conservative 1-3PN via pn_conserved_energy from states, 2.5/3.5PN via integrated RR work;
   residual visible; separate lines for PN truncation residual, N>2 mixed-order residual; spins (SO/SS) or off.
W5 Secular / C20 sec-inner paths: converge the s1340 (2.2e-7) / s1025 (2e-8) tolerance dependence; secular-GW book
   with convergence ctest.
W6 Error-injection ctest for every booking path (E, P, L).
W7 3-seed 200k A/B of the default changes, then BIFROST long-term set with honest books: Fig 9/10/11 (5 Myr),
   Fig 8 (35 Myr), core collapse (4 Myr); quote |dE/E| only after W1-W6.
W8 (research, parallel) ADM-Hamiltonian PN (to 3PN) + Gauss-IRK for PN binaries, RR as in-collocation force (A) vs
   symmetric RR-kick split (B), vs LogH+GBS; prototype only, verdict decides a production path.
Also confirmed: Gauss-IRK (C22) gains nothing in production because 99.8 % of LogH calls are PN and 98.9-99.9 % of the
visible error is FSI -> accuracy leverage is in FSI (step criteria, eta), next after W1-W7; MSTAR waits for W4.
- W9 (low priority) 1.5PN tail (and 2PN) GW flux in the secular binary path via eccentric enhancement functions (Arun+2008, Loutrel & Yunes 2017): local, ~1 % on inspiral times at x~0.01. Hereditary tails in the direct EOM -> separate GR library. (R4: production 3.5PN verified by Enzyme flux balance; 1.1 % harmonic vs ADM inspiral gap = O(x^2) truncation.)
- W10 C22 gauss_irk: guard NaN states and diverging fixed-point iterations accepted by the stopping rule (R4 side finding). W11 (low) GBS step control at eta 1e-8 costs 4-20x more evaluations than at 1e-10 in 3 of 6 R4 cases: investigate.
