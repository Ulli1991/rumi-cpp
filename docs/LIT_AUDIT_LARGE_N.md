# Literature audit: collisional N-body beyond 1e6 bodies (1e6-1e8), accurate and fast on GPUs and multi-node

Date: 2026-10-18. Literature only; no code was changed. This builds on `METHODS_SURVEY.md` (sections A, H, I, J),
`PARALLEL_ASYNC_REVIEW.md`, `SCALING_DESIGN.md` and `docs/PAIRWISE_SPLITTING.md`, and does not repeat them.
"VERIFIED" means the number was read from the paper text (local LaTeX source, ar5iv or the arXiv abstract) in this
audit. "UNVERIFIED" means it comes from memory or a secondary source and needs checking before anyone cites it.

---

## 0. Summary

**The problem at 1e7, in numbers (our estimate, from our own level populations).** At 200k the large HHS sets cost
about 8 N² pair evaluations per top step (L0-L4 populations in `PAIRWISE_SPLITTING.md` §1:
Σ 2^L (N_L/N)² ≈ 1 + 1.73 + 2.5 + 2.3 + 0.5). With about 3 O(N²) passes (force, gradient, time step plus neighbours)
that is about 24 N² per step.

- At N = 1e7 this is 2.4e15 pair evaluations per 1e-3 Myr step.
- At about 1.5e11 FP64 pairs/s per A100 (UNVERIFIED practical rate), that is 1.6e4 GPU-s per step, or about 185
  GPU-days per Myr.
- FROST and Rantala et al. 2023 scale to N_GPU ≈ 40 N/1e6 (2210.02472 §4, VERIFIED), so about 400 GPUs at 1e7. Even with
  ideal scaling that gives about 2 Myr/day.
- Core collapse of a 1e7 cluster (BH subsystem) takes a few hundred Myr to about 1 Gyr. So pure O(N²) direct
  summation needs about a year of wall time per run.
- **Goal C therefore needs a factor of 5-20 in algorithm, not in hardware.**

**What the literature says about where that factor can come from.**

1. **PeTar's speed comes from its time-step structure, not its tree.** PeTar runs our 200k cluster about 6x faster
   even with θ = 0 (`PAIRWISE_SPLITTING.md`). Distant pairs are kicked on one long shared step; only pairs inside a
   changeover radius are sub-stepped. The HHS analogue is a pairwise (HOLD-type) split, which keeps FSI's 4th order,
   pairwise kicks and exact P and L. This removes the "8" in 8 N². It is exact, so the energy books are untouched.
2. **An approximate far field (FMM) is the only route that changes the scaling.** Dehnen's error-controlled,
   mutual (momentum-conserving) FMM costs ∝ N^0.87 at relative force error 1e-7 and beats a GPU direct code above
   N ≈ 1e5 (1405.2255, VERIFIED). Inside HHS it conserves momentum to round-off (Zhu 1712.10116; GADGET-4). Energy
   and angular momentum are then conserved only to the FMM error ε, which conflicts with our "exact in the
   non-collisional parts" standard. Random force errors give an energy error per dynamical time of about
   ε√(η/N) (Dehnen 2014 App. A; our reading). That gets easier with N: at 1e7, ε ≈ 1e-7 gives about 3e-11 per
   t_dyn. **The trade is acceptable at N ≥ 1e6 only if it is booked and measured, and only on the slow (far) pairs.**
3. **No published code keeps collisional accuracy at 1e7 through core collapse with an approximate far field.**
   - PeTar: 1e7 runs at 1106/1344 s per N-body time (single stars / binaries) on 960 CPU cores, but its energy
     errors are about 1e-7 to 1e-3 per step and it slows down after core collapse (2006.16560 §7-8, VERIFIED).
   - Taichi (FMM): |dE/E| < 1e-4, "grows sharply as the simulations approach core collapse", because it has no
     regularisation (2012.02207, VERIFIED).
   - NBODY6+P3T: "not capable of post-collapse simulations due to large errors in energy conservation"
     (Arnold, Baumgardt & Wang 2110.11658, as quoted by the search tool; check the paper).
   - In every case the failure comes from close encounters and binaries, not from the far field. **We already have
     the part they lack: regularised subsystems and exact books.** That is the opening.
4. **Monte Carlo codes are the only codes that do 1e7-1e8 through core collapse today.** CMC reaches N = 1e8
   (Rodriguez+ 2022, 2106.02643, VERIFIED). The method assumes spherical symmetry, equilibrium and orbit
   averaging, so it is a validation partner, not a method to adopt. RAPID (Rodriguez+ 1511.00695) is a hybrid:
   direct-N core, Monte Carlo halo, about 1/200 of the direct cost (VERIFIED abstract). It is the only hybrid
   with published accuracy evidence; we note it but do not adopt it.

**Ranked short list (details in §6):**

1. Pairwise (HOLD/P3T-type) split inside HHS-FSI, direct far field. **Test first.**
2. Tree-pruned exact time-step and neighbour search (a prerequisite for 1).
3. FMM (mutual, error-controlled, GPU) for slow pairs only, at N ≥ 1e6, with a booked residual.
4. Rank-distributed, node-local subsystem work (SDAR/LogH/Kepler) to avoid PeTar's hierarchical-system stalls.
5. CMC/MOCCA cross-validation runs at 1e6-1e7 as the accuracy reference for core-collapse timing.

---

## 1. Force methods

### 1.1 Ahmad-Cohen (AC) neighbour scheme on GPUs: NBODY6++GPU, NBODY7

**What it is.** Each particle's force is split into an irregular part from about 50-100 neighbours (CPU, short
step) and a regular part from all particles (GPU, long step), with the regular part Hermite-extrapolated in
between. The speed-up over a pure block-step code is about N^{1/4} (PeTar 2006.16560 §3.2). GPU regular forces:
Nitadori & Aarseth 2012 (1205.1222; local `async_parallel_time/`).

**Accuracy.**
- 4th-order Hermite, not symplectic, with energy drift.
- KS plus chain regularisation keep binaries well; DRAGON-II follows binaries with periods about 10 orders of
  magnitude below t_dyn (2307.04805, VERIFIED).
- The DRAGON-II paper reports no energy-error numbers (VERIFIED absence).
- Without binaries, NBODY6++GPU has better energy conservation than PeTar for N < 1e5 (PeTar §7).

**Speed and scaling (VERIFIED unless marked).**
- NBODY6++GPU (1504.03687): 1e6 with 5% binaries takes "about an hour per half-mass crossing time" on 320 cores
  plus 32 K20X; 400-2000x faster than NBODY6.
- DRAGON (1602.00759): 16 K20m GPUs per run on MPCDF Hydra. The 12 Gyr 1e6 run took about 8600 h (about 1 year);
  the others took 4500-4700 h; 2000-4000 h per initial t_rh. "Only a few GPUs (8 to 16) are sufficient" at 1e6,
  i.e. it does not scale further.
- DRAGON-II (2307.04805): 19 runs up to 1e6 with up to 33% binaries; 2.2 million core hours over about 2 years on
  JUWELS Booster plus a workstation. The 1e6 runs reached only 194-207 Myr, below one t_rh (233 Myr).
- DRAGON-III (2510.03933, proceedings): 1e6, aiming at 10 Gyr; only 100 Myr shown; no timings.
- No 1e7 NBODY6++GPU science run was found (search 2026-10-18).

**Fit with HHS and the exact books.**
- Poor. AC regular-force extrapolation is not symplectic and does not book energy per path.
- AC does not conserve momentum (forces not pairwise between different-step particles).
- **The HHS equivalent of AC's gain is already built in:** a particle at depth d only interacts with the depth-d
  set at its fine step. What HHS lacks is that distant members of a large fast set are still sub-stepped with each
  other. That is the target of §1.2/§6-#1.

**Verdict: reject** as a scheme. NBODY6++GPU stays the reference for timings and binary physics; its scaling
ceiling (about 16 GPUs at 1e6) is a warning, not a model.

### 1.2 P3T and tree-direct hybrids: PeTar, Iwasawa+, FDPS, NBODY6+P3T

**What it is.** P3T (Oshino, Funato & Makino 2011, 1101.5504) splits each pair's force with a smooth changeover
function K(r):
- the long-range part goes to a tree with a shared leapfrog step Δt_L;
- the short-range part (r < r_out) goes to direct Hermite, plus SDAR for binaries in PeTar.

Further work:
- Iwasawa, Portegies Zwart & Makino 2015 (1506.04553): GPU tree for the long-range part.
- FDPS (Iwasawa+ 2016, 1601.03138): the parallel framework.
- PeTar (Wang+ 2020, 2006.16560): FDPS + P3T + SDAR + slow-down.
- NBODY6+P3T (Arnold, Baumgardt & Wang 2021, 2110.11658): GPU tree for distant forces in NBODY6.

**Accuracy (VERIFIED from 2006.16560 unless marked).**
- Energy errors are given only in figures. A per-step error of about 1e-3 at N = 250k is called "not small" but
  harmless for the global evolution.
- "Sudden energy jumps" occur when strong encounters are badly handled; runs were not restarted.
- Agreement with NBODY6++GPU in global structure, binaries and escapers.
- Purely Newtonian; no PN.
- P3T long-term energy error is a random walk dominated by the tree (1101.5504 abstract).
- NBODY6+P3T: "similar energy conservation" to NBODY6; faster than NBODY6 only above 3e5; about 2x at 1e6; cannot
  run post-collapse (2110.11658, partly via search snippet).

**Speed and scaling (VERIFIED).**
- At 1e6, PeTar is 5x (single stars) and 11-12x (100% binaries) faster than NBODY6++GPU on one desktop
  (Ryzen 3970X + RTX 2080Ti).
- Cray XC50, CPU only: 1e6 flattens at about 640 cores (63 s / 121 s per N-body time unit, single / binary).
- 1e7 scales to 960 cores: 1106 s / 1344 s per N-body time unit.
- For a 6e6 M_sun, r ≈ 3 pc cluster (our arithmetic): T_NB ≈ 0.03 Myr, so about 2 Myr/day at 1e7.
- Binaries cost about 2x because Δt_L halves.
- Stated limits: stable hierarchical systems cause temporary stalls in which "the benefit of parallel computing is
  lost due to a bad load balance". The P3T drift slows after core collapse because the central density rises.
- FDPS: minimum time per step of 30 ms at 1e7 and 300 ms at 1e9 on the K computer, limited by domain
  decomposition and communication (1601.03138 abstract, VERIFIED). **That latency is a floor for any distributed
  deep level.**

**Fit.**
- The tree part as PeTar does it (θ ≈ 0.3-0.5, ε ~ 1e-3) is incompatible with 1e-10 per interval (already
  concluded in `METHODS_SURVEY.md` I.3).
- **The time-step structure is compatible.** Kicking far pairs on a long step and near pairs on short steps is a
  Hamiltonian split: H = T + U_near + U_far, with U_far = Σ φ_ij (1 − K(r_ij)).
- In HHS-FSI this becomes the pairwise split of `PAIRWISE_SPLITTING.md` (pair class by pair time step, or by
  changeover radius). It keeps 4th order formally, pairwise kicks, exact P and L, and an exact energy book if U_far
  is evaluated directly.
- A smooth K(r) (PeTar) versus a hard pair-class switch (HOLD) matters for reversibility:
  - HOLD's switch depends on the pair time step, which changes along the orbit; reclassification makes the scheme
    non-symplectic, as for any adaptive step. Symmetrised steps help (Pelupessy+ 2012).
  - A position-dependent smooth K(r) keeps a fixed, smooth Hamiltonian, but needs a changeover radius r_out and
    the extra force term dK/dr. Our energy books must then use the split potential consistently.

**Verdict: test** (top of the list). Evidence: PeTar is 6x faster on our 200k cluster at θ = 0
(`PAIRWISE_SPLITTING.md`), so the gain is structural and survives a direct far field. Deciding experiment: §6-#1.

### 1.3 FMM in collisional settings

**Dehnen's falcON / griffin.** falcON (astro-ph/0003209, astro-ph/0202512): Cartesian FMM with mutual cell-cell
interactions, so momentum is conserved exactly. griffin (Dehnen 2014, 1405.2255, "A fast multipole method for
stellar dynamics"; VERIFIED abstract and local source):
- spherical-harmonic expansions to high p;
- an a-priori per-interaction error estimate that minimises cost for a target relative force error;
- "can be made as accurate as direct summation";
- at ε ≈ 1e-7: cost ∝ N^0.87, and 16 CPU cores beat a K20M GPU direct code for N ≳ 1e5 (§6.4);
- at active fractions below 1%, the tree build dominates (§7.1).

**Momentum-conserving FMM with individual steps.**
- Zhu 2017/2021 (1712.10116, VERIFIED abstract): FMM + HHS (Taichi) conserves momentum, and "restoring force
  symmetry also substantially improves angular momentum and energy conservation".
- GADGET-4 (Springel+ 2021, 2010.03567): the same HHS + FMM combination, momentum manifestly conserved.

**Collisional use (Taichi).**
- Mukherjee, Zhu+ 2021 (2012.02207, VERIFIED local source): ε = 1e-7, p = 20; per-particle force errors have a
  median below ε, a 99th percentile ≈ ε, and **tails to about 1e-3**.
- |dE/E| < 1e-4 overall, rising sharply near core collapse. More than 100x faster than direct Taichi at 1e6; more
  than 10x faster than NBODY6++GPU on 28 cores.
- No regularisation: hard binaries slow it down, and so does core collapse.
- Mukherjee+ 2023 (2205.12289): adds HHS-FSI with the Omelyan gradient and AR-chain, ε = 2e-5, p = 12,
  N = 1.32e6, and agrees with NBODY6++GPU on MBH-binary orbits.

**GPU FMM.**
- ExaFMM (Yokota & Barba; dual-tree FMM on GPUs, e.g. 1209.3516, UNVERIFIED id).
- PKDGRAV3 (Potter, Stadel & Teyssier 2017, 1609.08621, UNVERIFIED numbers): GPU FMM at about 8e12 particles,
  collisionless.
- Our `src/gravity/fmm.cpp` already exists (GPU, spherical harmonics, p ≤ 20, M2L by rotation) but is not wired in.

**Accuracy logic for us (our analysis).**
1. Mutual FMM gives exact P. L and E are conserved only to about ε per step. The tails (1e-3 in Taichi at
   ε = 1e-7) matter for the time-step criteria and the subsystem detection, so those must keep using direct
   near-field forces.
2. Random-error energy drift ≈ ε√(η/N) per t_dyn. At N = 1e7 and η = 0.2, ε = 1e-7 gives about 1.4e-11 per t_dyn.
   **Correlated** errors (the same cell geometry over many steps) can be far worse; Dehnen's estimate does not
   cover them. This is the open risk.
3. Use FMM only for the slow-pair kick at the large levels (the U_far of §1.2). Deep levels and all
   near/fast pairs stay direct, unsoftened and booked exactly.
4. The FSI gradient kick needs ∇_i|∇U_far|². With FMM this means either:
   - an Omelyan displaced-position evaluation (FC-III; rejected by us in the exact regime because it breaks L), or
   - a second FMM pass with vector sources m_j a_j plus one more kernel derivative.
   In the FMM regime L is already only ε-exact, so Omelyan becomes acceptable there.
5. The energy check must stay an independent direct O(N²) evaluation, done once per output, not per step. The
   FMM residual becomes a ledger entry: E_measured − E_booked attributed to "far-field approximation".

**Verdict: test**, gated as `METHODS_SURVEY.md` I-1/I-2 proposes (synthetic error injection first, then a
timing crossover). Adopt only as "FMM for U_far at levels with N_L ≥ N_FMM (≈ 1e5-1e6)". It is the only way to
1e8.

### 1.4 Tree codes for collisional work: Bonsai, GOTHIC, McMillan-Aarseth

**Bonsai** (Bédorf+ 2012, 1204.2280; 2014, 1412.0659, VERIFIED abstracts):
- the whole tree runs on the GPU;
- 242e9 particles on 18,600 Titan GPUs at 24.77 Pflops;
- collisionless (θ-based, softened, shared or block steps).

**GOTHIC** (Miki & Umemura 2017, 1610.07279, UNVERIFIED id): GPU tree with block time steps and an
acceleration-based opening criterion; collisionless.

**McMillan & Aarseth 1993** (ApJ 414, 200; no arXiv): an O(N log N) collisional scheme (tree + KS). Historical.

**Verdict: reject** as force solvers at our accuracy (θ-based ε ~ 1e-3-1e-4). **Adopt Bonsai's lessons:**
- fully device-resident tree builds (Morton keys, no host round trips);
- reuse of the tree across substeps.

These apply to our time-step and neighbour tree (§6-#2) and to the FMM tree. ArborX (Kokkos BVH, 1908.11807) is the
ready-made Kokkos option.

### 1.5 Monte Carlo and hybrids: CMC, MOCCA, RAPID

**CMC** (Rodriguez+ 2022, 2106.02643, VERIFIED):
- Hénon Monte Carlo with Fewbody for binary-single and binary-binary encounters (1 min wall limit per encounter;
  10 min with PN);
- 2.5PN by default when two or more BHs are present;
- N = 1e8 Plummer run to core collapse;
- limits: spherical symmetry, no rotation or triaxiality, no resonant relaxation, needs t_rh ≫ t_dyn, needs
  N ≳ 1e3 (m_max/⟨m⟩) ≈ 1e5;
- the Coulomb-log coefficient is calibrated against N-body (γ = 0.01).

**N-body comparison.** Rodriguez+ 2016 (1601.04227; local `frontier_runs/`) compared CMC with NBODY6++GPU at 1e6 and
found good agreement. Vergara+ 2025 (A&A, NBODY6++GPU + MOCCA, 1e6 stars) found differences in runaway growth that
FC-III attributes to winds, collisional mass loss and EMS radii, not to the dynamics.

**RAPID** (Rodriguez+ 1511.00695, VERIFIED abstract): direct-N core plus Monte Carlo halo; reproduces N-body core
and half-mass radii "far more accurately than" Monte Carlo at about 1/200 of the direct cost.

**SCHMoCK** (Kira + Monte Carlo, CIERA project page): no paper found.

**Fit.** Monte Carlo breaks every one of our standards (no trajectory-level energy, statistical encounters).

**Verdict: reject** as a method. **Adopt CMC/MOCCA as cross-validation**: the same 1e6-1e7 ICs (spherical, no
rotation) and core-collapse times as an external check (§6-#5).

---

## 2. Time integration at large N

| Method | What it is | Accuracy evidence | Fit | Verdict |
|---|---|---|---|---|
| HHS / HOLD / CC (Pelupessy, Jänes & Portegies Zwart 2012, 1205.5668; Jänes+ 2014, 1407.7105; HUAYNO in AMUSE) | Recursive Hamiltonian split by particle (HHS), by pair (HOLD) or by connected component (CC) | Momentum conserving; "excellent energy conservation" with symmetrised steps (VERIFIED abstract); 2nd order | FROST and Rantala et al. 2023 use the 4th-order FSI version of HHS. HOLD = pairwise split; CC = our CC task roots | **Test HOLD inside FSI** (§6-#1); CC already used |
| Saha & Tremaine 1994 (AJ 108, 1962) | Symplectic individual steps for planetary systems (Wisdom-Holman base) | Needs a dominant central mass | Not for clusters | Reject |
| Farr & Bertschinger 2007 (astro-ph/0611416, UNVERIFIED id) | Tree-structured adaptive symplectic splitting, Kepler sub-solves | Small-N tests | Idea already in HHS + Kepler drifts | Reject (nothing new) |
| Hernandez & Bertschinger 2015 (1503.02728, local) | Pairwise Kepler-drift splitting, symplectic, time-reversible | Few-body | See §3 | Reject at large N |
| GADGET-4 HHS + FMM (2010.03567) | "Kick all, undo fast-fast" form of HHS with FMM | Momentum to round-off | Shows FMM and HHS compose | Reference for §6-#3 |
| Shared step vs block | — | — | Block/HHS wins at any N with a wide range of dynamical times | Keep HHS |
| Async/event-driven, task-based | See `PARALLEL_ASYNC_REVIEW.md` §3-4 | — | Conservative concurrency of independent components is the only safe form | Already planned (SCALING_DESIGN) |
| Parareal / PFASST / MGRIT | Rejected before (chaos) | No new evidence found 2026-10-18 | — | Stays rejected |
| PN at scale: KETJU (Rantala+ 2017, 1611.07614; Mannerkoski+ 2023, 2306.04963, UNVERIFIED id), Rantala et al. 2023 | PN only inside regularised regions; tree/GADGET outside | KETJU runs about 1e7-1e8 collisionless particles with a few PN regions | Our PN lives in subsystems, so cost scales with the number of compact pairs, not N | Nothing to change |

**Point for large N.** Taichi (2205.12289) is prior art for "HHS-FSI + FMM + regularisation", on CPUs at 1.3e6.
Nothing in the literature combines HHS-FSI, a pairwise split, FMM and GPUs. That combination is the novel item, and
the goal of this code.

---

## 3. Few-body and subsystem methods that scale to many simultaneous binaries

At 1e7 with 10-50% binaries there are 1e6-5e6 binaries.

**Rantala et al. 2023 evidence (2210.02472 §4, VERIFIED):**
- the binary fraction has "negligible effect" on wall time at 1e6 and 5e6, even with 2.5e6 binaries;
- the cost is dominated by the O(N²) loops, because most binaries are secular or Kepler;
- the large-N risk is stragglers (deep hierarchies), not the count (PeTar §8; our notes of 2026-10-12).

| Method | Accuracy | Speed / batching | Fit | Verdict |
|---|---|---|---|---|
| SDAR (Wang+ 2020, 2002.07938, local) | Symplectic-like AR + slow-down; secular evolution of perturbed binaries kept | "A few thousand times" fewer steps for weakly perturbed binaries; CPU OpenMP in PeTar | Already in RUMI++ (FC-III also switched to SDAR) | Keep; distribute (§6-#4) |
| Slow-down KS (Mikkola & Aarseth 1996; NBODY6) | κ ≤ 10 cap in NBODY6 | Serial CPU | Superseded by SDAR | Reject |
| AR-CHAIN (Mikkola & Merritt 2008, 0709.3367, local) | GBS + LogH + chain; PN | Per-system serial | Our LogH/GBS is the same family | Keep as is |
| TSUNAMI (Trani & Spera 2022/23, 2206.10583, local) | Chain + LogH + BS; tides; PN to 2.5 | Fast few-body code (single systems) | Same family; worth comparing for tides | Reject for adoption; borrow tide terms if needed |
| IAS15 (Rein & Spiegel 2014, 1409.4779, local) | 15th order, Brouwer-law round-off | Very accurate for smooth orbits; adaptive; no regularisation | Weak for hard binaries at e → 1 without regularisation | Reject for subsystems; possible reference integrator in tests |
| IRK16 (Antoñana, Makazaga & Murua 2017, Numer. Algorithms; arXiv id UNVERIFIED) | Symplectic implicit RK, 16th order, round-off monitoring | Solar-system long integrations | Implicit solves, no regularisation; for smooth orbits | Reject |
| Pairwise Kepler splitting (Hernandez & Dehnen 2017, 1612.05329; Dehnen & Hernandez 2017, 1609.09375) | Symplectic, reversible; Kepler solve per pair | Cost O(N²) Kepler solves per step | A Kepler solver per pair is expensive and breaks down in hierarchies | Reject (see `PAIRWISE_SPLITTING.md` for the HOLD variant that we do test) |
| WH with close encounters: MERCURIUS (Rein+ 2019, 1903.04972), TRACE (Lu, Hernandez & Rein 2024, 2405.03800) | Time-reversible switching (TRACE) | Planetary; needs a dominant mass | Only around an IMBH/SMBH (nuclear clusters) | Reject now; revisit for goal-B nuclear models with a central BH |
| Device-batched regularisation: GENGA (1404.2324), Swarm-NG (1208.1157), MPGOS | GPU wins above about 30 independent systems; 2.4-4x from many steps per launch | Our deep-level calls have 1-30 systems; at 1e7 the count per call grows about 10x | Per-level batch sizes at 1e7 may reach the GPU-wins regime | **Test later**: re-measure J-1 at 1e6-2e6 before building J-3 |

---

## 4. Communication and scaling: what was achieved

| Code | Method | Largest N / hardware | Throughput (VERIFIED unless marked) |
|---|---|---|---|
| NBODY6++GPU (1504.03687) | AC + Hermite + KS/chain; MPI + GPU | 1e6 on 320 cores + 32 K20X | About 1 h per half-mass crossing time (5% binaries) |
| DRAGON (1602.00759) | NBODY6++GPU | 1e6 × 4 runs, 16 K20m each | 12 Gyr in about 8600 h; 2000-4000 h per t_rh |
| DRAGON-II (2307.04805-07) | NBODY6++GPU | 1e6 (2 runs), up to 33% binaries | 2.2 M core-h for 19 runs over about 2 yr; 1e6 runs reached about 200 Myr |
| PeTar (2006.16560) | P3T + SDAR; MPI + OpenMP (+ GPU tree) | 1e7 on 960 XC50 cores | 1106 / 1344 s per N-body time unit; about 2 Myr/day for a 6e6 M_sun cluster (our conversion) |
| FROST; Rantala et al. 2023 (2011.14984, 2210.02472) | HHS-FSI, direct O(N²) | 5e6 on 128 A100 (tested); 2.5e6 binaries | Scales to N_GPU ≈ 40 N/1e6; 1.5 Myr/day at 1e6 on 4 A100 (Rantala et al. 2023 Fig. 7) |
| HiGPUs (1207.2367) | 6th-order Hermite, block steps, direct, no regularisation | 8e6 on 256 GPUs (CINECA) | "Few crossing times" only |
| phi-GPU (1312.1789; Berczik+) | 4/6/8th-order Hermite, block steps | A few 1e6; large GPU clusters | About 50% of K20 peak in production |
| Bonsai (1412.0659) | GPU tree, collisionless | 2.4e11 on 18,600 GPUs | 24.77 Pflops |
| FDPS (1601.03138) | Tree framework | 1e9 on the full K computer | 30 ms/step minimum at 1e7 (latency floor) |
| CMC (2106.02643) | Hénon Monte Carlo | 1e8 to core collapse | Wall time not extracted (UNVERIFIED; reported as days to weeks) |
| FROST-CLUSTERS I-III (2403.10602, 2503.21879, 2506.04330, 2601.07917) | Rantala et al. 2023 code + SEVN | Up to 2.4e6 (hierarchical assembly); 1e6 isolated | No wall times reported (checked in source) |

**Lessons for multi-node.**
1. **Direct O(N²) scales in GPU count with N** (Rantala et al. 2023: 40 GPUs per 1e6). At 1e7 the problem is the cost per GPU,
   not the scaling.
2. **NBODY6++GPU stops at about 16 GPUs at 1e6**, because its irregular part sits on the CPU and must be
   communicated. Any neighbour/near-field part we add must stay on the device and local in its domain.
3. **FDPS' 30 ms per step floor at 1e7** means distributed tree or FMM steps cannot be used at deep levels.
   Use FMM only at the few large levels (a handful of evaluations per top step); deep levels stay replicated or
   node-local, as `SCALING_DESIGN.md` §4 already plans.
4. **Hierarchical-system stalls** (PeTar §8) dominate the long-run Myr/day. Our production-speed metric
   (sustained Myr/day) is right to focus on them.

---

## 5. Can anyone keep collisional accuracy at 1e7?

No published code does this today with direct-quality forces through core collapse.

- **HiGPUs:** only a few crossing times at 8e6.
- **PeTar:** has run 1e7 (scaling tests) at errors orders of magnitude above our target.
- **CMC:** reaches core collapse at 1e8 statistically.

The closest physically trustworthy option is PeTar-class structure with direct-quality far forces. That is the
combination ranked first below.

---

## 6. Ranked short list for RUMI++, each with the experiment that decides it

### #1 Pairwise (HOLD/P3T-type) split inside HHS-FSI, direct far field. Test.

**Why.** PeTar's 6x at θ = 0 on our 200k cluster is structural. At 200k, levels 1-4 cost 7 of the 8 N² per step;
at 1e7 the share should be similar or larger. The split keeps 4th order, pairwise kicks, exact P and L, and exact
energy books. Prior art: HOLD (1205.5668), CC (1407.7105), P3T changeover (1101.5504).

**Decisive experiment.**
1. Finish the ETA_GRAD on/off level-population test. If the gradient criterion fills L1-L4, fix the criterion
   first; the split alone would not help (`PAIRWISE_SPLITTING.md` §4).
2. Build a CPU prototype at 20k (host build): Kepler and planet floor tests to confirm 4th order; then 200k, 3 seeds
   (42, 7, 13), 10% binaries, 20-35 steps.
3. Pass: equal or better |dE/E|, P and L at round-off, and at least 2x on the large-set phase.

### #2 Exact tree-pruned time-step and neighbour search. Adopt as infrastructure.

**Why.** It is needed by #1 (pair classes need near pairs, not all pairs) and by any 1e7 run, where an O(N²)
time-step pass alone costs about 1e14 pair tests. It is bit-identical (a min-reduction), so it carries no risk to
accuracy. The tree can be device-resident (Bonsai-style) or ArborX.

**Decisive experiment.** `METHODS_SURVEY.md` H-2: time the time-step plus neighbour phase per depth at 200k, 1e6 and
2e6. Build if the phase is more than 15% at 1e6. Pass: bit-identical levels and pair sets over a whole 200k run.

### #3 Mutual, error-controlled GPU FMM for U_far only, at levels with N_L ≥ N_FMM. Test, gated.

**Why.** It is the only N-scaling lever (Dehnen 2014: N^0.87 at ε = 1e-7). The energy noise per t_dyn falls as
1/√N. FMM + HHS keeps P exact (Zhu; GADGET-4). Taichi's core-collapse failure was the missing regularisation, which
we have. Cost: L and E are exact only to ε, booked as a separate ledger term. Our `fmm.cpp` exists.

**Decisive experiments.**
1. I-1: inject synthetic relative errors (random and cell-correlated) of size ε into the U_far kicks only, at
   200k, 1e6 and 2e6, over 3 seeds. Find the largest ε that keeps |dE/E| ≈ 1e-10 per interval and L drift below
   1e-12. Expected: about 1e-8 at 200k, about 1e-7 at 2e6.
2. I-2: time `fmm_test` against direct at that ε for 1e6 and 1e7, on one A100 and one MI300A.
3. Go if FMM is at least 5x faster than direct at 1e7 and the 1e6 FROST 12 Myr test passes the energy rule with a
   direct energy check at outputs.

### #4 Distributed, node-local subsystem work. Test (already in `SCALING_DESIGN.md`).

**Why.** At 1e7 there are about 10x more simultaneous perturbed subsystems per call. PeTar and NBODY6++GPU both lose
their parallel efficiency to long-lived hierarchies. Spread subsystem integration over all ranks' host threads with
LPT ordering. Revisit device batching (GENGA/Swarm-NG style, J-3) only if per-call counts at 2e6 exceed about 30-100.

**Decisive experiment.** J-1 per-depth breakdown at 1e6 and 2e6 on 16 GPUs: subsystem share of the step and the
largest single subsystem time per call. Build if the stragglers exceed 20% of wall time.

### #5 External accuracy reference: CMC/MOCCA vs RUMI++ at 1e6, later 1e7. Adopt (cheap).

**Why.** No N-body code has a 1e7 core-collapse reference; CMC does (N up to 1e8). The comparison is valid for
spherical, non-rotating ICs: core-collapse time, BH ejection rate, binding-energy distribution. Rodriguez+ 2016 is
the template. It is also the cheapest way to see whether a #3 approximation changes the relaxation physics, as
FMM/tree papers check with Lagrangian radii.

**Decisive experiment.** Run the same 2e5 and 1e6 ICs (no primordial binaries, then 10%) in public CMC and
RUMI++ to core collapse. Pass: t_cc within the N-body seed scatter.

### Not recommended

- AC/Hermite (incompatible with symplectic exact books; scaling ceiling).
- Low-accuracy trees (Bonsai/GOTHIC/PeTar θ).
- Monte Carlo or hybrid as the integrator.
- Parareal and its relatives.
- IAS15/IRK16/pairwise-Kepler as subsystem integrators.
- WH-type schemes outside nuclear clusters with a dominant BH.
- Mixed-precision far field: at most 2x on A100/MI300A FP32 versus FP64, the same error class as FMM without the
  scaling. Consider it only if #3 fails.

---

## 7. References (arXiv IDs)

### Rantala and collaborators

| arXiv | Paper |
|---|---|
| 2001.03180 | MSTAR |
| 2011.14984 | FROST |
| 2210.02472 | Rantala et al. 2023 |
| 2403.10602 | FROST-CLUSTERS I |
| 2503.21879, 2506.04330 | FROST-CLUSTERS II |
| 2601.07917 | FROST-CLUSTERS III |
| 1611.07614 | KETJU |
| 2306.04963 | KETJU in GADGET-4 (UNVERIFIED id) |

### Force methods and codes

| arXiv | Paper |
|---|---|
| 1101.5504 | Oshino, Funato & Makino 2011, P3T |
| 1506.04553 | Iwasawa, Portegies Zwart & Makino 2015, GPU P3T |
| 1601.03138 | FDPS |
| 2006.16560 | PeTar |
| 2002.07938 | SDAR |
| 2110.11658 | Arnold, Baumgardt & Wang 2021, NBODY6+P3T |
| 1205.1222 | Nitadori & Aarseth 2012, NBODY6 on GPUs |
| 1504.03687 | NBODY6++GPU |
| 1602.00759 | DRAGON |
| 2307.04805, 2307.04806, 2307.04807 | DRAGON-II |
| 2510.03933 | DRAGON-III |
| 2105.08067 | Kamlah+ 2022 |
| 2305.11606 | Spurzem & Kamlah 2023 review (authorship UNVERIFIED by search) |
| 1405.2255 | Dehnen 2014, griffin FMM |
| astro-ph/0003209, astro-ph/0202512 | falcON |
| 1712.10116 | Zhu, momentum-conserving FMM + HHS |
| 2012.02207 | Taichi, collisional FMM |
| 2205.12289 | Taichi + FSI + AR-chain |
| 2010.03567 | GADGET-4 |
| 1204.2280 | Bonsai |
| 1412.0659 | Bonsai on Titan |
| 1610.07279 | GOTHIC (UNVERIFIED id) |
| 1609.08621 | PKDGRAV3 (UNVERIFIED) |
| 1207.2367 | HiGPUs |
| 1312.1789 | phi-GPU |
| 2106.02643 | CMC |
| 1601.04227 | CMC vs NBODY6++GPU |
| 1511.00695 | RAPID hybrid |

### Integrators and few-body methods

| arXiv | Paper |
|---|---|
| 1205.5668 | Pelupessy, Jänes & Portegies Zwart 2012, HOLD/HHS |
| 1407.7105 | Jänes+ 2014, connected components |
| astro-ph/0611416 | Farr & Bertschinger 2007 (UNVERIFIED id) |
| 1503.02728 | Hernandez & Bertschinger 2015 |
| 1612.05329 | Hernandez & Dehnen 2017 |
| 1609.09375 | Dehnen & Hernandez 2017 |
| 0709.3367 | AR-CHAIN |
| 2206.10583 | TSUNAMI |
| 1409.4779 | IAS15 |
| 1903.04972 | MERCURIUS |
| 2405.03800 | TRACE |
| 1404.2324 | GENGA |
| 1208.1157 | Swarm-NG |
| 1908.11807 | ArborX |
