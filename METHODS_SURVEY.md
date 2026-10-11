# cpp-nbody methods survey and learning guide (October 2026)

This document has two jobs:

- explain, in physical terms, each numerical method that cpp-nbody uses or could use;
- turn the literature into a ranked list of experiments we can run.

It is written for a reader fluent in astrophysical hydrodynamics (GADGET/AREPO, individual time bins, tree/FMM gravity, operator splitting, Richardson extrapolation) and less fluent in collisional N-body methods.

**Companion documents.** Plan: `/u/uli/.claude/plans/bubbly-fluttering-falcon.md` (milestones M0-M8). Measurements: `AUDIT_2026-10.md` (findings F1-F6). Papers: `/raven/u/uli/papers/<topic>/<arXiv id>.pdf`; the list is in the appendix.

**Conventions.**

- Abbreviations: FC-I, FC-II, FC-III = FROST-CLUSTERS I 2403.10602, II 2506.04330, III 2601.07917. "BIFROST" = Rantala et al. 2023, 2210.02472. "FROST" = Rantala et al. 2021, 2011.14984. "MSTAR" = 2001.03180. "SDAR" = Wang, Nitadori & Makino 2020, 2002.07938.
- Section numbers are those of the **arXiv LaTeX**. The published BIFROST numbering is shifted by one subsection: arXiv §3.5 is published §3.4, and the code comments use the published numbers.
- Equation numbers for the Rantala papers were counted from the LaTeX sources and may be off by one where there are unnumbered displays.
- **UNVERIFIED** marks a claim we could not check in the source (recalled from memory, or taken from an abstract or a citing paper).
- "Our inference" marks reasoning that is ours, not a paper's.
- Code references are `src/<file>:<line>` at commit 9310626.

**The energy rule (standing).** A change is accepted only if |dE/E| is equal or better on 3 seeds (42, 7, 13). Speed is measured in seconds per 1e-3 Myr interval, either on the BIFROST Fig. 9 setup (200k, evolved Kroupa, Z = 2e-4, r_h = 3.1 pc, with or without BINARY_FRAC = 0.1, 20-35 steps) or on the 20k host reproduction at Fig. 9 density used in the audit (F3 update).

---

## 0. How the code works: one time step from start to finish

The code advances a star cluster by one **integration interval** of 1e-3 Myr (`DT_MAX_MYR`). Inside that interval every star gets its own step, which can be up to about 2^-30 times shorter for stars in tight binaries. The machinery that makes this cheap and accurate has five layers.

1. **Energy checkpoint** (`src/main.cpp:1040-1140`). Before the interval starts, the code stores the state and the total energy E = T + U plus the energy ledgers (energy carried away by escapers, gravitational waves, mergers and stellar evolution). If the interval ends with a relative energy change above `ETA_RESTART` = 1e-8, the state is restored and the interval is redone with all η halved and r_ngb increased by 15%, up to 10 times (BIFROST §3.17 published / §3.19 in our comments). Hydro analogue: a CFL-violation retry with a smaller Courant number.

2. **Hierarchical Hamiltonian splitting, HHS** (`src/integrator/integrator.cpp:2061-2530`, `hhs_fsi_recursive`). This is the time-bin hierarchy. At the top level the code computes, for every star, a step size from pairwise criteria (§B). With pivot = 1e-3 Myr it sorts the stars into **slow** (step ≥ pivot) and **fast** (step < pivot), then does:
   - kick slow and fast stars with their *mutual* forces only, for 1/6 of the step;
   - advance the slow set by half a step with the 4th-order integrator (FSI, §A);
   - **recurse** on the fast set with pivot/2. The recursion repeats the same procedure: recompute steps from the fast stars alone, split again, and so on;
   - apply a 2/3 slow-fast kick including the "gradient" correction;
   - advance the slow set by the second half step and recurse on the fast set again;
   - apply the final 1/6 slow-fast kick.

   Every slow-fast interaction is applied pairwise and at synchronised times, so total momentum is conserved to round-off. This is the analogue of flux-conservative coupling between AMR time levels (refluxing).

   The difference from GADGET's individual time bins: a star on a deep level never sees the 200k field directly, only the slow-fast kicks at the levels above it.

3. **Subsystem detection** at every level (`src/gravity/forces.cpp:178-243` neighbour kernel; `src/collisional/subsystem.cpp:227-620` `detect_from_pairs`).
   - Before the steps are computed, every pair closer than r_ngb,max = 0.1 pc is tested. Bound pairs with a < r_ngb (= 1e-3 pc = 206 AU), and fly-bys whose pericentre falls inside r_ngb within the step, become graph edges.
   - Connected components become **subsystems**: binaries, triples and small groups of at most 8 bodies (`LOGH_MAX_BODIES`, `src/collisional/logh.h:6`).
   - Their mutual forces are removed from the N-body sum ("pair exclusion") and integrated by a dedicated few-body solver instead. Hydro analogue: a sub-grid model with its own internal clock, coupled to the grid by the external force only.

4. **Few-body solvers** (`src/integrator/integrator.cpp:431-1000`, `advance_subsystems_in_range`). In each FSI drift the subsystems are advanced by half a step, chosen as follows:
   - a binary with at least 3 orbits per step (`N_ORB_SECULAR`) is **Kepler-drifted with orbit-averaged PN/GW terms** ("secular binary", `src/collisional/secular.cpp:424`);
   - a stable triple with at least 3 inner orbits per step is evolved by **double-averaged Kozai-Lidov equations** ("secular triple", `secular.cpp:228`);
   - everything else is integrated by **LogH**, an algorithmically regularised leapfrog in a rescaled time variable, made accurate to 1e-10 by **Gragg-Bulirsch-Stoer (GBS) extrapolation** (`src/collisional/logh.cpp`, `gbs.cpp`, `logh_batch.cpp`).

   PN terms (1PN-3.5PN) act only on subsystems with a member above `M_PN` = 5 Msun. Mergers, collisions and TDEs are checked per level (`src/collisional/mergers.cpp:116`).

5. **Global PN1 kick** (`src/collisional/post_newtonian.cpp:420`, called at `integrator.cpp:1343,1358`). Massive compact objects that are not in a subsystem get a half PN1 kick before and after the interval (BIFROST §3.8).

**Cost structure today** (audit baseline, 200k, 1 node with 4x A100):

| Configuration | s/step | |dE/E| after 20 steps |
|---|---|---|
| No binaries | 1.85 | 2e-10 |
| 10% binaries | about 10-13 | 1.3e-11 |
| 10% binaries, `FLYBY_CM_VELOCITY = true` | about 6 | 1.0e-10 |

With the c.o.m. fly-by velocity on, about half of the step is subsystem integration, dominated by per-call overhead (F2). Occasional intervals stall for 50-150 s on hierarchical subsystems (F3).

---

## A. Global integrator

### A.1 Plain explanation

**Leapfrog and its error.** Write H = T(v) + U(x). Leapfrog (kick-drift-kick) is the composition e^{εT/2} e^{εU} e^{εT/2}. Its error is a modified Hamiltonian H + ε²(c₁[T,[T,U]] + c₂[U,[T,U]]) + O(ε⁴). The double commutators are the price of splitting a coupled operator, the same Strang-splitting error that hydro codes accept when they split gravity from hydro.

**Fourth order without negative steps.**

- Yoshida/Forest-Ruth fourth-order schemes cancel both commutators with negative sub-steps. Forces are then evaluated far off the trajectory, which is bad near pericentre: Chin 2007, 0704.3273 §II, Figs. 1-2, finds a 10x larger energy spike at e = 0.9.
- Chin's forward symplectic integrator (FSI) cancels only [T,[T,U]]. It moves [U,[T,U]], which depends only on positions, into an explicitly computed extra potential −(ε²/48) Σ m|a|².
- The result has only positive sub-steps (FROST Eqs. 17, 21; BIFROST Eqs. 5-10):
  - kick ε/6 with a
  - drift ε/2
  - kick 2ε/3 with the modified acceleration ã
  - drift ε/2
  - kick ε/6 with a
- The modified acceleration is ã_i = a_i + (ε²/24) Σ_j G m_j [r² a_ji − 3(a_ji·r) r]/r⁵ (FROST Eq. 28 = BIFROST Eq. 9). It is the gradient of a scalar, so the map stays symplectic.
- Hydro analogue: a 4th-order Runge-Kutta-Nyström stage scheme whose extra stage is exact because it is a potential gradient.

**HHS-FSI** (FROST Eq. 41 = BIFROST Eq. 11). Split H = H_S + H_F + U_SF by a pivot step. The slow-fast coupling U_SF gets the FSI kicks (1/6, 2/3 with gradient, 1/6), H_S gets an FSI step, and H_F is the recursive call (algorithm in §0).

- This is Pelupessy et al. 2012's "HOLD" splitting (1205.5668 §2.1, Eqs. 5-6) promoted to 4th order.
- S and F are synchronised at every S-F kick. Pelupessy §3 shows that unsynchronised inter-level kicks still conserve momentum but make the centre of mass drift.

**Omelyan approximation** (FC-I App. A1). ã_i(r) ≈ a_i(r′) with r′_i = r_i + (ε²/24) a_i: displace every particle and call the ordinary force kernel (FC-I l.772-776 of the LaTeX; Chin 2007 Eq. 2.8, citing Omelyan 2006, PRE 74, 036703).

- Why it works: the tidal tensor is symmetric, so (a·∇)a = ½∇|a|². The error is O(ε⁴).
- Chin 2007 calls the resulting scheme "nearly indistinguishable" from the symplectic one on Kepler orbits. It is time-symmetric and momentum-conserving, but not formally symplectic. Its precession error "can differ substantially" from the exact FSI.

**Hermite block steps, for contrast.** NBODY6++GPU, PeTar's short-range part, ph4 and Kira all use 4th-order Hermite (Makino & Aarseth 1992, PASJ 44, 141): a predictor-corrector with directly computed jerk on power-of-two block steps.

- It is not symplectic and not time-symmetric, so energy drifts secularly. Forces between levels are one-sided, from predicted positions, so momentum is not conserved exactly (about 1e-7 in Pelupessy Fig. 5).
- Ahmad-Cohen splits each particle's force into a neighbour part (short step) and a regular part (long step, extrapolated). The speed-up is about N^{1/4} (PeTar 2006.16560 §3.2).
- 6th/8th-order Hermite: Nitadori & Makino 2008, arXiv 0708.0738 (abstract only checked).

### A.2 What cpp-nbody does now

- **FSI step** (`src/integrator/integrator.cpp:1136-1262`, `fsi_step`). It uses kick coefficients `FSI_KICK = 1/6`, `FSI_DRIFT = 1/2`, `FSI_MOD_KICK = 2/3` (`src/core/constants.h:22-24`), and the subsystems are advanced inside each drift (BIFROST Alg. 1 ordering).
- **Gradient term.** It is computed exactly: one force pass, then a second pass over acceleration snapshots (`GradK`, `src/gravity/n2_kernels.h:36-60`; `src/gravity/forces.cpp:102-149`). This is 4 O(N²) passes per FSI cycle, as BIFROST §2.1 counts. There is no Omelyan option.
- **HHS-FSI recursion** (`integrator.cpp:2299-2530`).
  - ALL_SLOW_SPLIT: when F is empty, the code takes two fsi(τ/2) steps as in FROST Alg. 3 (`integrator.cpp:2310-2330`).
  - A set consisting of a single subsystem is not recursed (`integrator.cpp:2270-2297`).
  - Optional `HHS_CC` connected-component split (Jänes et al. 2014, 1407.7105) (`integrator.cpp:2332-2345`; `forces.cpp:277`).
- **Tree forces** exist (`src/gravity/tree_forces.cpp`, with a momentum correction at l.678) but are **refused for collisional runs** (`src/main.cpp:612-618`).

### A.3 Literature and other codes

**Measured accuracy of HHS-FSI.**

- FROST Fig. 4 (Kepler binary, 1000 periods): |ΔE/E| ∝ η⁴, floor about 1e-13 at η ≈ 0.008. Above the η⁴ line for e = 0.9-0.99 once η ≳ 0.1.
- FROST Fig. 5 (Sun plus 4 giant planets): HHS-FSI follows η⁴ down to η ≈ 0.004. Shared-step FSI is about 10x more accurate at the same η, but impractical at large N.
- FROST Fig. 6 (1e6 stars, η = 0.2, 50 N-body times): |ΔE/E| ≈ 2e-10, |ΔL/L| ≈ 1e-13, |ΔP|/M ≈ 1e-15 km/s.

**Pelupessy 2012** (Figs. 5, 7, 8).

- N = 100: HOLD conserves P, L and centre of mass to machine precision; block steps have ~1e-7 errors.
- N = 1024: at equal energy error, HOLD is about 3x faster than block steps (fewer force evaluations), 12x faster late in core collapse, and 300-1000x faster than a shared step.

**Weakness of particle-based HOLD** (Jänes+2014 §2.2). If every star is in a hard binary, all binaries land in the fastest bin, and their mutual long-range forces are evaluated at the binary step. HOLD then degenerates to a shared step.

- The connected-component split (Eqs. 19-21) is about 2x faster (Fig. 3).
- CC with analytic 2-body Kepler components stays flat through core collapse, while HOLD slows about 10x (Fig. 5).
- BIFROST avoids most of this because binaries are subsystems and enter the hierarchy as two members whose mutual force is excluded. The c.o.m. fly-by velocity (§B) is the remaining piece of the same problem.

**GADGET-4** (2010.03567, hierarchical time integration section). The same HHS, implemented as "kick all with all forces, undo the fast-fast part". Momentum is manifestly conserved even with FMM forces.

- Cost: K_hier = N₀ + 3 Σ_k 2^{k−1} N_k force evaluations, up to 3x the one-sided scheme.
- The code shortens Δt_max automatically when N₁ > N₀/3.
- Validated only on cosmological runs.

**Prior art for "HHS-FSI + FMM + regularisation"** is not a Rantala code. It is Taichi (Mukherjee et al. 2023, 2205.12289): FMM forces, Rantala's HHS-FSI with the Omelyan gradient, and AR-Chain-Sym6+ few-body integration, at N up to 1.3e6. It reports about 2x the speed of NBODY6++GPU on CPUs only, and 10x better energy conservation (appendix).

**Head-to-head with Hermite** (BIFROST §5.6, Fig. 15; N = 20k, 10% binaries, core collapse). NBODY6 at ETAI = ETAR = 0.01 matches BIFROST at η = 0.2 in energy (about 3e-4 at 4 Myr) and is "marginally faster" at this N.

**Reversibility.** HHS-FSI is time-symmetric only if the step assignment is.

- Pelupessy §2.2 (Eqs. 10-17) and FROST §4.2 (Eqs. 46-49) symmetrise each pairwise time with its analytic time derivative. Pelupessy Fig. 6: this removes the linear drift and costs about 20% more steps.
- Dehnen 2017 (1707.09069 §3, Fig. 14): exact reversibility with block steps "appears elusive". Irreversibility falls like η². Extrapolating ln τ from the previous step gives O(h³).
- FROST/BIFROST errors grow linearly, a bias, not √t (BIFROST §5.1).
- The FC papers never mention symmetrisation; whether production BIFROST keeps it is UNVERIFIED.

**Factor-2 discrepancy in FROST §2.3** (our inference, not checked further). The illustrative test-particle sentence in FROST §2.3 uses "g = 2 t_dyn⁻² a" with a prefactor 1/24. For an isolated pair Eq. 28 gives ã = a[1 + ε²/(12 t_ff²)], so the sentence looks like a factor-2 slip in the illustration only. Eq. 28, which we implement, is self-consistent.

### A.4 Numbers that matter for us

- FSI costs 4 O(N²) passes per cycle (BIFROST §2.1).
- Omelyan also needs 2 passes for the middle kick: a(r), then a(r′). **It does not save passes.** It saves register pressure, since there is no snapshot of a_j in the source tile, and it makes an FMM or tree gradient trivial (FC-I App. A1: "computational expense ... comparable").

### A.5 Experiments

**A-1 Omelyan gradient** (plan M7.3). Add `GRADIENT=omelyan`.

- Pitfall (our inference): in the HHS S-F gradient kick (FROST Eqs. 43-44), the displacement of i ∈ S must use only the acceleration from F, and vice versa. Using the total acceleration keeps momentum but silently drops to lower order.
- Test:
  1. Kepler e = 0.9 and frost4 η-sweeps (0.4, 0.2, 0.1, 0.05): the η⁴ slope must survive.
  2. 200k without binaries, 3 seeds.
- Pass: equal or better |dE/E| on all seeds at the same η, and s/step not worse.
- Expected: speed ±10% (gradient pass becomes a force pass); accuracy roughly equal, sign unknown.

**A-2 HHS_CC as default with binaries.** The code exists. Test 200k with 10% binaries, `HHS_CC=true` vs false, 3 seeds. Pass: energy rule plus lower s/step. Jänes reports about 2x in binary-rich systems.

**A-3 Symmetrisation cost.** Diagnostic: a run with the symmetrisation term zeroed (env switch) on 20k. Expect about 20% fewer steps and a worse energy trend. Run it only to learn the size of the effect, not to change the default.

**Recommendation.** Keep HHS-FSI; it is the right integrator for this accuracy target, and no production Hermite code reaches 1e-10. Do A-2 first (code exists), then A-1 when FMM work begins. **Confidence: high** that the integrator choice is right; **medium** that Omelyan is accuracy-neutral (no N-body measurement exists in the literature).

---

## B. Time-step criteria

### B.1 Plain explanation

Each particle's step is the minimum over several criteria, each multiplied by an η.

- **Free-fall** (FROST Eq. 50; BIFROST Eq. 21): τ_ff = η_ff min_j sqrt(r_ij³ / G(m_i+m_j)). The gravitational sound-crossing time of each pair; it protects slow approaches.
- **Fly-by** (FROST Eq. 51; BIFROST Eq. 22): τ_fb = η_fb min_j r_ij/|v_ij|. A pairwise signal-speed CFL condition, the analogue of GADGET-4's signal-velocity step. For a circular orbit τ_ff = τ_fb, and η = 0.2 means about 31 steps per orbit (FROST Eq. 52).
- **Symmetrisation** (FROST §4.2, Eq. 49). Each pairwise τ is replaced by a time-centred value using dτ/dt, so the step chosen at the start of a step approximately equals the one at the end. This removes the systematic one-sided error that makes adaptive steps irreversible.
- **Gradient criteria** (BIFROST §3.5.4, Eqs. 24, 26, 27):
  - τ = η|v|/|a| (Eq. 24)
  - τ = η_∇(|v|/|g′|)^{1/3} (Eq. 26)
  - τ = η_∇(|a|/|g′|)^{1/2} (Eq. 27)

  The aim is to keep the ε² gradient correction small compared with the Newtonian kick, so that the FSI expansion stays valid. Eqs. 24 and 26 use the absolute |v|, so they are frame-dependent and give τ → 0 for a particle momentarily at rest, e.g. a central IMBH.
- **Jerk** (BIFROST §3.5.5, Eq. 29): τ = η_j |a|/|ȧ|. A Hermite-style smoothness limit.
- **Perturbation** (BIFROST Eq. 30): subsystem members get τ ≤ η_pert min(|a/ȧ|, 1/|ė|) from the Gauss equations of the external tide (energy and angular momentum rates for N > 2).
- **Level-0 clamp** (BIFROST §3.5.1). On deeper levels the criteria see only the fast set. Pairwise criteria are fine with that by design, but global ones (|a|, |j|, |v|) can grow spuriously. The clamp is τ ≤ τ(level 0). GADGET-4 has the same "partial forces in deeper levels" problem.

### B.2 What cpp-nbody does now

- **ff/fb symmetrised with the harmonic form** of FROST Eq. 49: `symmetrized_timestep`, `src/core/constants.h:83-116`. It is folded into the force kernels `DtK`/`AccDtK` (`src/gravity/n2_kernels.h:107-200`).
- **Jerk** in the same kernels (`n2_kernels.h:144-147, 195`), with η_jerk = 0.1.
- **Gradient (Eqs. 26-27) and v/a (Eq. 24)** after the force pass (`integrator.cpp:2181-2208`). v/a uses `ETA` = 0.2 (`src/main.cpp:625`) and the absolute velocity.
- **Level-0 clamp** (`integrator.cpp:2210-2225`).
- **Perturbation and shared subsystem step** (`src/collisional/subsystem.cpp:743-995`).
- **Fly-by velocity of binary members.** Component velocities by default; the c.o.m. velocity with `FLYBY_CM_VELOCITY` (`main.cpp:590`, kernels `n2_kernels.h:131,181`). The CPU path (`integrator_cpu.cpp:419`) and the HHS_CC kernels still use component velocities (audit F1).
- **Configuration:** η_ff = η_fb = 0.2, η_grad = η_jerk = 0.1 (`config.txt:17-20`), as in BIFROST Fig. 9.

### B.3 Literature and other codes

**What the FC papers use.**

- FC-I App. B (LaTeX l.869): ε_dyn = min(ε_ff, ε_fb, ε_∇). **There is no jerk and no v/a term** (verified).
- FC-I parameter table (l.322-326): η_ff = η_fb = η_∇ = **0.2**, not 0.1.
- FC-III §2: "η = 0.2 for free-fall, fly-by and gradient".
- So the production BIFROST of 2024-2026 has dropped jerk and v/a and loosened η_∇. No paper shows an ablation that justifies either choice.

**FC-II §2.1** (l.138, verified): binary members in subsystems use the binary c.o.m. velocity in the fly-by criterion, because "component velocities led to overly conservative time-step estimates for particles near hard binaries". No numbers are given. Our audit F1 measured it: the L0 occupancy goes from 10.7k back to 143k and the steady step time halves.

**Grudić & Hopkins 2020** (1910.06349, Eqs. 8-11). τ = η (‖T‖²/6)^{-1/4}, where T is the tidal tensor.

- It recovers t_dyn for Kepler orbits and is Galilean-invariant.
- About 10x smaller energy error at equal force evaluations in shallow cores, compared with acceleration-based criteria.
- They argue that criteria built on |a| (jerk, v/a) fail where a → 0, e.g. at the cluster centre.
- Collisionless GIZMO tests only.
- In a direct-summation kernel T costs about the same as the jerk (our inference).

**Aarseth criterion** (as quoted in PeTar 2006.16560 §6): τ = η sqrt[(|A0||A2| + |A1|²)/(|A1||A3| + |A2|²)] with a safety term. It needs the 2nd and 3rd force derivatives from the Hermite polynomial and does not carry over to FSI (BIFROST §3.5.1 says so). The equation number in Aarseth 2003 is UNVERIFIED.

**Pham, Rein & Spiegel 2024** (2401.02849, Eqs. 15-17): a new IAS15 criterion, τ = √2 [(|j|/|a|)² + |s|/|a|]^{-1/2}, about 2.3x larger steps as e → 1. It applies to a global-step integrator; listed for completeness.

**Zemp et al. 2007** (astro-ph/0606589): enclosed-mass dynamical time. Collisionless; formula UNVERIFIED.

**Accuracy against cost in BIFROST** (§5, Figs. 8-11).

- The error has two parts: a smooth part ∝ η⁴ (Eq. 73) and stochastic jumps when a particle's step is too long in a strong encounter.
- Fig. 9: 3e-7 total at 5 Myr with 357 BHs (2e-8 without BHs).
- Fig. 10: halving all η gives 5e-9, i.e. 60x better, at roughly 2x the cost (FROST §6.3 scaling).

**Our own analysis of the criteria** (our inference).

- For an isolated pair, |a|/|g′| = 12 t_ff², so Eq. 27 gives 3.46 η_∇ t_ff. That never binds against ff at η_∇ ≥ 0.06; it binds only where the a_ji partly cancel (multi-body or tidal configurations), which is the intent.
- For a circular binary |a|/|j| = 1/ω = t_ff, so the jerk at η_j = 0.1 halves the steps compared with ff/fb at 0.2.
- For core field stars |a| nearly cancels while |j| comes from the nearest neighbour, so the jerk step collapses.
- Jerk is therefore the prime suspect for cost without proven accuracy benefit. Gradient Eq. 27 is cheap insurance.

### B.4 Experiments

**B-1 FC-style criteria** (config only: `ETA_JERK = 0`; v/a off; `ETA_GRAD = 0.2`). Do this after F1 is settled, with `FLYBY_CM_VELOCITY = true`.

- Test: 200k with 10% binaries, 35 steps, seeds 42/7/13.
- Report median and 95th percentile per-interval |dE_s/E|, the number of intervals with |dE_s/E| > 1e-9, total |dE/E|, s/interval and L0..L_max occupancy.
- Pass: energy rule. If it fails, try each change alone (B-1a jerk off, B-1b v/a off, B-1c η_∇ = 0.2) to find the one that hurts.
- Expected: 1.2-2x fewer deep-level steps; accuracy unknown. BIFROST argued the integrator-derived criteria make energy "more robust" (§3.5.4) but never quantified it.
- Note: `ETA = 0` currently disables v/a only through `eta_va`; check that the restart halving still works (test `restart_halves_va`).

**B-2 Tidal-tensor criterion replacing jerk** (Grudić & Hopkins Eq. 11). It is computed in the jerk loop, so the cost is the same. Calibrate η to give the step count of B-1, then compare energy. Pass: energy rule at equal or lower step count.

**B-3 Make v/a Galilean-invariant** (e.g. use v − v_cm,local, or drop it). Test: an IMBH at rest at the centre (`ADD_IMBH`). Pass: the IMBH's level does not change when a uniform 100 km/s boost is added to the whole cluster.

**B-4 Level-0 clamp diagnostic.** Turn it off on 20k and record any level whose occupancy inflates. Diagnostic only.

**B-5 Interval 5e-3 Myr** (the FC-I/II value), with a GADGET-4 style automatic shortening when N₁ > N₀/3. Pass: energy rule per Myr, not per interval, and s/Myr lower.

**Recommendation.** B-1 is the cheapest high-value experiment in this whole document: config only, backed by what the FC papers use. **Confidence: medium.** The FC papers dropped jerk, but they never published the energy cost of doing so.

---

## C. Subsystem identification and stability criteria

### C.1 Plain explanation

Which stars should leave the global integrator and be handed to a few-body solver? The best analogy is an AMR refinement criterion. Codes flag a pair or group by one of three kinds of test:

- **geometric**: a radius (BIFROST r_ngb, NBODY6 R_cl);
- **dynamical**: the strength γ of the tidal perturbation relative to the binary's own binding force (NBODY6 GMIN, SDAR κ);
- **temporal**: will a fly-by reach pericentre within this step (BIFROST).

The flagged group is then integrated with its internal forces only. Everything else reaches it as an external kick.

**Triple stability** decides whether a hierarchical triple can be treated as two nested Kepler orbits with slow (secular) exchange of angular momentum.

- **Mardling & Aarseth 2001** (MNRAS 321, 398; not on arXiv; the id astro-ph/9910524 sometimes quoted is an unrelated paper). The criterion is R_p,out/a_in > 2.8 [(1+q_out)(1+e_out)/(1−e_out)^{1/2}]^{2/5} (1 − 0.3 i/π), with q_out = m3/(m1+m2), as quoted in Vynatheya+2022 Eq. 2.
- **The BIFROST printing of this formula (2210.02472 Eq. 54) misplaces a bracket.** Implement the Vynatheya form.
- **Vynatheya et al. 2022** (2207.03151 §4, Eqs. 3-4). The criterion is Y = a_out(1−e_out)/[a_in(1+ẽ_in)] > 2.4 [(1+q_out)/((1+ẽ_in)(1−e_out)^{1/2})]^{2/5} × [((1 − 0.2ẽ_in + e_out)/8)(cos i − 1) + 1].
  - ẽ_in = max(e_in, 0.5 e_max²), with e_max = sqrt(1 − (5/3)cos² i).
  - Calibrated with MSTAR; stability = surviving 100 outer orbits.
  - Classification score 0.93, against 0.90 for MA01 and 0.95 for their MLP.
- **Lalande & Trani 2022** (2206.12402): a CNN on 5e6 TSUNAMI triples, ROC-AUC above 95%.
- **Hayashi, Trani & Suto 2022/23** (2207.12672, 2209.08487): MA01 is a chaos-boundary (Lyapunov) criterion, not a disruption (Lagrange) criterion. ZLK destabilises 60° < i < 150°; retrograde orbits are more stable. For "is it safe to orbit-average", a chaos boundary is the right kind of test.

### C.2 What cpp-nbody does now

**Neighbour test**, BIFROST §3.4 Eqs. 18-19 (`src/gravity/forces.cpp:189-230`):

- bound pairs with a < r_ngb or r < r_ngb;
- unbound pairs with r < r_ngb, or with r_peri < r_ngb and 0.5 r/v < pivot (C = 0.5).

Settings: r_ngb = 1e-3 pc, r_ngb,max = 0.1 pc (`config.txt:44-45`). The search is O(N_set²) at every level.

**Components** by union-find (`subsystem.cpp:310-345`). Groups above 8 bodies are split along bound pairs; whatever is still too large stays in the FSI (`subsystem.cpp:356-383, 416-426`).

**Triple classification** (`subsystem.cpp:470-610`):

- the inner pair is the tightest bound pair;
- the outer orbit is computed relative to the inner c.o.m.;
- stability uses `mardling_stability` (`subsystem.cpp:210-221`). This version has **no inclination factor** and uses a_out(1−e_out)/(a_in(1+e_in)), which is more conservative than MA01 (plan bug 6).

**Experimental weak-perturber split** (`WEAK_PERTURBER_GAMMA`, `subsystem.cpp:532-578`, off by default).

- γ = 2(m3/M_in)(a_in(1+e_in)/r_close)³, evaluated at the third body's closest approach.
- Below γ_crit, the binary becomes its own (secular/Kepler) subsystem and the third body a field star.
- Measured on the `weak_flyby_split` test: 60x faster but 3.6x worse energy.

### C.3 Literature and other codes

**BIFROST** (§3.4) has **no perturbation-strength threshold**.

- Any pair with a < r_ngb is a subsystem, and a fly-by joins as soon as its pericentre falls inside r_ngb within the step.
- Values used across the papers: r_ngb = 1e-3 pc (BIFROST); 0.25 mpc (FC-I table, verified); r_subsys = 0.5 mpc (FC-II table).
- FC-II's r_strong = 10a is a switch for **binary stellar evolution**, not for regularisation: SEVN binary evolution is suspended inside it (FC-II §2.3.2). The plan lists it under "method changes"; it matters only once binary stellar evolution exists.

**NBODY6/NBODY6++GPU** (Aarseth 2003; Wang+2015 1504.03687 §5, Eq. 2).

- KS regularisation starts for pairs closer than R_cl = 4R_h/[N(ρ_d/ρ_h)^{1/3}] with steps below Δt_cl. For 1e6 stars these were set by hand: R_cl = 1e-5, Δt_cl = 5e-7 (N-body units).
- A KS binary with relative perturbation γ < GMIN ≈ 1e-6 is **unperturbed**: it is advanced analytically, and the outside world sees its c.o.m.
- Perturbers are stars within λR, λ = [2m_p/(Mγ_min)]^{1/3} (Aarseth 2003 Eq. 8.58, as quoted in PeTar §3.3).
- Stable triples become "hierarchical mergers" via MA01 (detail UNVERIFIED).

**PeTar** (2006.16560).

- Mass-dependent changeover radii r_in,i = max(1, m_i/⟨m⟩)^{1/3} r_in,ref, with r_out/r_in = 10 (§6.1, Eq. 12).
- SDAR groups are pairs with r_ij < θ r_in (§6.2, Eq. 22).
- Neighbour radius r_out + 3|v|Δt_s, followed by a Kepler-pericentre test (§6.3.1, Eq. 24).
- Distant tides on the binary come from "artificial particles" sampling a 2nd/3rd-order tidal tensor (§6.4). This matters to us because our FSI kicks sample the binary at random phases when ε ≫ P_bin (dense-core floor, plan M3.4).

### C.4 Our F3 stall examined against the literature

Audit F3 system: 1.07 + 1.07 Msun at a = 0.01 AU, plus a 0.15 Msun star at 0.047 pc moving at 190 km/s.

1. **The perturber is unbound.** The escape speed from 2.15 Msun at 0.047 pc is about 0.6 km/s. So this is a hyperbolic fly-by, not a triple. Stability criteria and the secular triple are irrelevant to it. It joins the subsystem through BIFROST's distant fly-by test: r_peri < r_ngb and 0.5·r/v ≈ 120 yr < pivot = 1000 yr.

2. **The tide is negligible.**
   - At 0.047 pc: γ ≈ 1.6e-19.
   - At a 1 mpc pericentre: γ ≈ 1.7e-14.
   - NBODY6 would call this binary unperturbed (γ < 1e-6) and never integrate it orbit by orbit.

3. **Orbit-count inconsistency in the audit.**
   - For a = 0.01 AU and M = 2.14 Msun, P = 6.8e-4 yr, which gives **1.46e6 orbits per 1e-3 Myr, not 8.9e4**.
   - 8.9e4 orbits corresponds to a ≈ 0.065 AU.
   - Please re-check the log line. All conclusions below hold for either value.

**What the literature would do with this system.**

| Option | Cost | Status in our code |
|---|---|---|
| (a) Exclude weak perturbers (NBODY6 GMIN; our `WEAK_PERTURBER_GAMMA`) | Binary costs about 1 Kepler drift per FSI step | Exists, off by default |
| (b) SDAR slow-down (§D) | About 40 slowed orbits per interval | Not implemented (plan M3) |
| (c) Kepler drift for the unperturbed inner binary inside the subsystem (Wisdom-Holman inside AR) | — | No production collisional code does this (UNVERIFIED) |

For (a): the 3.6x energy loss in our `weak_flyby_split` test (periapsis 20 AU, γ ~ 1e-10) is **not explained by γ**. Our guess (our inference) is that it comes from the FSI integrating the *third body* against two point masses whose phase it samples at its own step: an aliased quadrupole. NBODY6 avoids this because outside the perturber radius the field sees only the KS c.o.m.

### C.5 Experiments

**C-1 γ sweep for the weak-perturber split.** Values: `WEAK_PERTURBER_GAMMA` ∈ {0, 1e-10, 1e-8, 1e-7, 1e-6}.

- Tests:
  - the audit's 20k host reproduction (the stalled seed 7), 5 steps;
  - `weak_flyby_split`;
  - 200k with 10% binaries, 3 seeds.
- Pass: energy rule on the 3 seeds, and no interval above 60 s.

**C-2 c.o.m. coupling for split binaries.** Variant of C-1, our idea, NBODY6 practice. For binaries split off by C-1, field stars outside λ·a_apo interact with the binary's c.o.m. (monopole) instead of its components; inside that radius the components are resolved.

- This removes the aliased quadrupole.
- Test: `weak_flyby_split` with C-1 on. Pass: |dE/E| ≤ the unsplit value (1.5e-9).

**C-3 Stability criterion.** Use MA01 with inclination factor and R_p,out/a_in (Vynatheya Eq. 2), or Vynatheya Eqs. 3-4.

- Test 1: count secular triples on the 200k, 10% run.
- Test 2: integrate the triples that switch class with LogH for 100 outer orbits (the MSTAR check).
- Pass: no triple classified stable disrupts within 100 outer orbits. Accuracy fix, no speed target.

**C-4 r_ngb = 0.5 mpc and 0.25 mpc** (FC values) vs 1 mpc.

- Expected: fewer and smaller subsystems, and fewer distant fly-bys pulled in, so less F3-type cost. In exchange, more binaries are integrated by the FSI when a > r_ngb.
- Pass: energy rule plus lower s/interval on 200k with 10% binaries.

**Recommendation.** Run C-1 plus C-2 before SDAR. They are cheap, and they attack the F3 stall at its cause (a γ ~ 1e-19 perturber dragged into a regularised group). SDAR remains the general solution for *bound* perturbed hierarchies. **Confidence: medium.** C-1 alone already failed the energy rule once, at γ ~ 1e-10; C-2 is our hypothesis for why.

---

## D. Regularised few-body integration

### D.1 Plain explanation

Close encounters make 1/r forces singular. A fixed-step integrator either takes tiny steps at pericentre or loses energy there. **Regularisation** removes the singularity by changing variables.

**Time transformation (LogH/TTL).**

- The integrator steps in a fictitious time s with dt = ds/(T + B) (LogH; B = −E is the binding energy) or dt = ds/W (TTL).
- In s the leapfrog is exact on the Kepler orbit's *shape*. The only error is along the track, a phase or period error of O(n⁻²) for n steps per orbit (Preto & Tremaine 1999, astro-ph/9906322; Mikkola & Tanikawa 1999, CeMDA 74, 287, UNVERIFIED pairing).
- Wang+2020 Eq. 47: in LogH, ds ∝ dE (eccentric anomaly), so a fixed Δs means **a fixed number of steps per orbit regardless of e**. That is why we see a constant 15-19 GBS steps per inner orbit.
- Analogy: a Lagrangian mesh that compresses where the solution is steep, applied to time.

**Coordinates.**

- Chains (Mikkola & Aarseth 1993) or minimum spanning trees (MSTAR §3) express each body relative to its nearest neighbour, to limit round-off in hierarchical groups.
- MSTAR adds parallel force evaluation and parallel GBS substeps; MSTAR Fig. 9 finds it more accurate and faster than AR-CHAIN.
- Velocity-dependent forces (PN) need a time-symmetric trick to keep GBS valid: the auxiliary velocity method or the generalised midpoint (Mikkola & Merritt 2006, astro-ph/0605054; 2008, 0709.3367; §G).

**Slow-down** (Mikkola & Aarseth 1996, CeMDA 64, 197; SDAR 2002.07938 §2, Eq. 1).

- H_sd = H_b/κ + (H − H_b): the binary's internal clock runs κ times slower while the tide acts at full rate. One slowed orbit absorbs κ orbits' worth of tidal torque.
- With κ fixed, the orbit-averaged Hamiltonian equals the original (§2.2.1). So secular (ZLK) evolution is exact to first order and only the orbital phase is lost.
- It is orbit averaging done by hand, with the averaging error controlled by κ. Hydro analogue: a reduced-speed-of-light trick applied to the fast mode.
- **Jumping κ** (§2.4, Eqs. 22-28): κ is kept constant during a step and changed between steps; the energy jump is booked.
  - κ = k_ref [m1m2/(m12 r_apo³)] Σ_j r_j³/m_j, with k_ref = 1e-6 (Eq. 22; FC-I Eq. A8, verified l.821).
  - κ ≤ κ_max = (c/P)|⟨r⟩_m|/|⟨v⟩_m|, with c = 0.1 and mass-weighted perturber means relative to the binary c.o.m. (Eq. 28; FC-I l.825, verified).
  - κ ≥ 1.
- **Measured** (SDAR §7):
  - Binary-single Kozai: 6x fewer steps, ZLK timescale reproduced, |ΔL|/L ≈ 1e-10.
  - Hyperbolic binary-binary: 60x fewer steps, with κ capped near 1e3.
  - **Failure:** a 1:1 mean-motion resonance became 1:2, with a drift in a of about 1e-3 after 2 t_KL.

### D.2 What cpp-nbody does now

- **LogH leapfrog with exact cubic Ω kick** (`src/collisional/logh.cpp:123-210`), and a PN variant with auxiliary velocities and spins (`logh.cpp:211-463`).
- **No chain or MST coordinates.** `src/collisional/mstar.cpp` contains MST scaffolding that is never called. Groups above 8 bodies fall back to the FSI.
- **Batch driver** (`logh_batch.cpp:383-675`): the subsystems run on **host threads** (`Kokkos::DefaultHostExecutionSpace`, dynamic schedule, l.387) after pack/copy from the device. Despite the function name `logh_integrate_batch_gpu`, they do not run on the GPU.
- **Step limit** max(1000, 64 × inner orbits), ceiling 4e6 (`logh_batch.cpp:542`; commit 3e44de1).
- **Per-subsystem restart:** retries split dt into up to 256 pieces (`integrator.cpp:947-1000`).
- **No slow-down.** `LogHState::ext_accel` is unused.

### D.3 Literature and other codes

**Rantala's own line.**

- BIFROST §3.6 called slow-down "unnecessary due to the hierarchical nature of integration".
- FC-I §2 (l.224, verified) and App. A2 then **replaced both LogH and the secular triple with SDAR + GBS**: k_max = 8, η_GBS = 5e-8, end-time tolerance 1e-2, κ = 1 for subsystems with BHs (PN), collision candidates forced to κ = 1 (App. B3, l.885).
- FC-II and FC-III keep SDAR.
- This is the largest method disagreement inside the Rantala papers. Our clone follows the 2023 version.

**PeTar.** SDAR with a Yoshida 6th-order symplectic integrator (no GBS). κ_max is tied to the tree step (§6.2). The claim is "a few thousand times" fewer integration steps.

- §8 conclusions: "the computation becomes very slow ... due to the formation of a specific type of stable hierarchical systems", in both PeTar and NBODY6++GPU, with "no universal solution". Such spikes were removed from their benchmarks (§7.1.1).
- So our F3 stall is a known open problem, not a bug unique to us.

**NBODY6/7.**

- KS regularisation with slow-down capped at κ ≤ 10 (PeTar §3.4).
- Unperturbed binaries are advanced analytically.
- NBODY7 adds AR-CHAIN with PN for compact subsystems (Aarseth 2012, MNRAS 422, 841; detail UNVERIFIED).

**TSUNAMI** (Trani & Spera, 2206.10583). Chain coordinates, a generalised time transformation, Bulirsch-Stoer, PN up to 2.5 and tides. No slow-down is mentioned. OKINAMI is its secular companion code. The full methods paper id is UNVERIFIED.

**Single-scale integrators cost about 1e3 force evaluations per inner orbit**, whatever the method: IAS15 (about 100 steps × 8 substeps × 2 iterations), LogH+GBS (15-19 steps × k(k+1) substeps) and Hermite alike. Only methods that remove the inner phase help with F3: slow-down, Kepler splitting, unperturbed treatment, or averaging.

**SDAR estimate for F3** (our calculation).

- κ from perturbation strength: about 3e12. So κ_max binds.
- κ_max = 0.1 × (r/v)/P. Here r/v = 0.047 pc / 190 km/s ≈ 242 yr, so κ_max ≈ 3.5e4 for a = 0.01 AU.
- Slowed orbits per interval = Δt·v/(c·r) ≈ 41. This is independent of P.
- That gives about 700 GBS steps instead of more than 1e6: roughly a 2e3x reduction.
- Caveat: v must be measured relative to the binary c.o.m.

### D.4 Experiments

**D-1 SDAR jumping-κ inside LogH+GBS** (plan M3.2). Parameters: k_ref = 1e-6; c ∈ {0.03, 0.1, 0.3}; κ = 1 for PN members and collision candidates.

- Unit tests:
  - the SDAR Table 1 binary-single Kozai case (t_KL reproduced);
  - the hyperbolic binary-binary case (about 60x step reduction);
  - our F3 reproduction (seed 7).
- Monitor the slow-down energy ε(H_sd) separately from the physical |dE/E|.
- Pass: energy rule on 200k with 10% binaries, 3 seeds; F3 interval below 5 s.
- Expected: removes the F3 stalls entirely; F2 per-call cost is unchanged.

**D-2 Remove the secular triple** (route stable triples to LogH, later SDAR), as FC-I did. See F-1.

**D-3 MSTAR for groups above 8 bodies** (plan M4). Needed for IMBH runs (Fig. 12); not urgent for Fig. 9/11.

**Recommendation.** D-1 is the right long-term answer and is what the Rantala code now runs. Do C-1/C-2 first (days, not weeks). **Confidence: high** that SDAR removes the F3 cost; **medium** that it passes the energy rule at 1e-10. FC-I quotes no per-interval error with SDAR at η_GBS = 5e-8.

---

## E. Extrapolation (GBS) and alternatives

### E.1 Plain explanation

GBS is **Richardson extrapolation in the step size**.

1. Integrate one big step H with n₁, n₂, ... leapfrog substeps.
2. Because the leapfrog is time-symmetric, its error contains only even powers h², h⁴, ..., so a polynomial in h² fitted through the results and evaluated at h = 0 removes one more power per extra column.
3. Stop when two successive extrapolants differ by less than η_GBS.

Sequences of n_k:

- Bulirsch: 2, 4, 6, 8, 12, 16, 24, ... (MSTAR Eq. 10).
- Deuflhard/harmonic: 2, 4, 6, 8, 10, ... (MSTAR Eq. 11).

After k columns the order is 2k, and the work is Σn_j substeps: 72 for Deuflhard at k = 8, 138 for Bulirsch at k = 8.

**End-time control.** The regularised integrator steps in s, not t, so landing exactly on the target time needs iteration. MSTAR (§2.4, Eqs. 15-16) first takes Δs = U Δt, and if τ + Δs/U would overshoot, sets Δs = U(Δt − τ).

### E.2 What cpp-nbody does now

- **Bulirsch sequence** {2, 4, 6, 8, 12, 16, 24, 32, 48, 64}, up to 10 stages (`src/collisional/gbs.h:13-14`).
- **Polynomial Neville extrapolation in h²** (`gbs.cpp:95-140`); converged when err < η_GBS = 1e-10.
- **Step growth** ds_next = 0.8 ds (η/err)^{1/2k}, at most 4x per step (`gbs.cpp:245-251`).
- **Bail-out** at stage ≥ 3 if err > 1 (`gbs.cpp:257-262`).
- **Dynamical cap** ds ≤ t_dyn Ω / 3 (`gbs.cpp:297-309`, `NBODY_GBS_DYNCAP`).
- **End time:** secant rescale on overshoot, relative tolerance 1e-4 (`gbs.cpp:312-388`, `NBODY_GBS_TTOL`).

### E.3 Literature and other codes

**Tolerances.**

| Code | η_GBS | End-time tolerance | k_max |
|---|---|---|---|
| BIFROST | 1e-10 | 1e-4 | not given |
| FC-I (verified) | 5e-8 | 1e-2 | 8 |
| FC-II | 1e-8 | 1e-2 | — |
| FC-III | 1e-8 | 1e-3 | — |
| MSTAR | — | — | k_fix = 8, Deuflhard sequence |

- The Rantala line loosened η_GBS by 100x after 2023, without published |dE/E| justification. They also moved to 5x longer intervals, where the end-time tolerance matters less.
- MSTAR §2.3: rational extrapolation "occasionally" gives spurious convergence; it uses polynomial extrapolation. MSTAR does not use the Hairer-Nørsett-Wanner (HNW) work-per-unit-step order control (Deuflhard 1983, Numer. Math. 41, 399; not checked).

**Alternatives.**

- **IAS15** (Rein & Spiegel 2015, 1409.4779): 15th-order Gauss-Radau, about 100 steps per orbit, Brouwer-law error over 1e9 orbits. Same cost class per orbit; no help for F3.
- **WHFast** (1506.01084): Wisdom-Holman Kepler drift plus interaction kick in Jacobi coordinates, symplectic correctors. It needs one dominant Kepler term, as in an inner binary.
- **MERCURIUS** (1903.04972) and **TRACE** (Lu, Hernandez & Rein 2024, 2405.03800): hybrid integrators. TRACE is 12x faster than MERCURIUS at equal accuracy, but these hybrids handle ZLK cycles poorly (TRACE discussion).
- **Hernandez & Bertschinger 2015** (1503.02728): pairwise Kepler-solver splitting, symplectic and reversible. Dehnen & Hernandez 2017 (1609.09375) give a 4th-order version.
- **SAKURA** (1402.3325): composition of 2-body Kepler solutions, good for hierarchies with primordial binaries.

None of these appears in a production collisional cluster code's subsystem solver (UNVERIFIED for TSUNAMI internals).

### E.4 Experiments

**E-1 Tolerance grid.** η_GBS ∈ {1e-10, 1e-9, 1e-8} × end-time tolerance ∈ {1e-4, 1e-3, 1e-2}, using the env vars `NBODY_GBS_TTOL` and the config key `ETA_GBS`.

- Test: 200k with 10% binaries, 3 seeds, 20 steps.
- Report s/interval, subsystem phase time, GBS steps, and the worst subsystem dE.
- Pass: energy rule. Expect a gain of about 1.3-2x in the subsystem phase. Our own NBODY_GBS_DYNCAP = 0.3 test already showed 3.2x fewer GBS steps at unchanged global energy.

**E-2 Deuflhard sequence, k_max ∈ {8, 10}**, plus MSTAR end-time control. Unit test: `kepler_e09`, `frost4_e099`, `pn_logh_books`. Record the histogram of converged column k. Pass: equal |dE| per call with fewer substeps.

**E-3 Dynamical cap revisited.** It is a self-imposed limit on top of GBS step control. Test `NBODY_GBS_DYNCAP` ∈ {3, 1, 0.3} on 3 seeds at 200k (only the cluster run has been measured so far). Pass: energy rule.

**Recommendation.** E-1/E-3 are config-level and safe to try right after B-1. E-2 is a few hours of code. **Confidence: medium-high** for speed gains. Accuracy at 1e-8 is plausible, because global errors are dominated by FSI jumps rather than subsystem error (BIFROST §5).

---

## F. Secular approximations

### F.1 Plain explanation

If a binary completes many orbits during one outer step, replace its fast orbital phase by its orbit average. Only the slow elements (a, e, ω, plus i and Ω under external torques) then evolve. This is the same idea as a Reynolds average in turbulence: averaging out the fast mode leaves equations for the mean.

- **Isolated binary.** 2.5PN radiation reaction gives Peters (1964) da/dt = −(64/5)βF(e)/a³ and de/dt = −(304/15)βe G(e)/a⁴ (BIFROST Eqs. 44-45). Conservative 1PN/2PN terms only rotate the ellipse (Eq. 46).
- **Hierarchical triple, double averaging** (average over both orbits). The third body's tidal quadrupole exchanges angular momentum, not energy, with the inner orbit: the von Zeipel-Lidov-Kozai cycles. The octupole term (∝ (a1/a2) e2/(1−e2²)) allows orbital flips and extreme e (Naoz 2016 review, 1601.07175).
- **When averaging fails.**
  - Antonini, Murray & Mikkola 2014 (1308.3674 §3): near e_max the angular momentum changes within one inner orbit if √(1−e1) ≲ 5π (m2/M_b)[a1/(a2(1−e2))]³.
  - Grishin, Perets & Fragione 2018 (1808.02030 §2): double averaging is valid if √(1−e_max²) ≥ 2π ε_SA, with ε_SA = (a_in/a_out)^{3/2}(1−e_out²)^{−3/2} m3/[(m12+m3)m12]^{1/2} (Luo, Katz & Dong 2016, 1601.04345).
  - Brown's Hamiltonian, the second-order correction (Tremaine 2023, 2304.01257), is often larger than the octupole term.
  - In a cluster, passing stars change the elements on T_coll ≈ 2e5 yr (1e6 pc⁻³/n)(AU/a2)(30 Msun/M) (Antonini 2014; printed equation number UNVERIFIED).

### F.2 What cpp-nbody does now

**Secular binary** (`src/collisional/secular.cpp:116-227, 424`; dispatch `integrator.cpp:599-705`).

- Used when n_orb = dt/P ≥ 3 (`subsystem.cpp:452-456`). That is orbits per step, the correct reading of BIFROST's printed "N_orb = P/ε", which is a typo.
- Kepler drift plus Peters rates, with PN only for compact binaries (commit 91c7177).
- Sub-step min(a/ȧ, e/ė) × η_sec.
- Since 91c7177: one sub-step per evolution, 42x faster secular phase (audit F4).

**Secular triple** (`secular.cpp:228-400`; driver `integrator.cpp:706-910`).

- Blaes et al. 2002 quadrupole + octupole Delaunay equations, with 1PN-3PN inner precession and GW decay.
- Forward Euler.
- **Outer orbit frozen** (e_out and G2 fixed).
- Inclination recovered from total-H conservation.
- Positions rebuilt from elements.
- Loses |dE/E| = 3.4e-5 in 10 steps on an isolated triple (audit F5).

### F.3 Literature and other codes

**BIFROST §3.10 (binaries).** A second-order auxiliary-variable leapfrog in (a, e), step ε_sec = η_sec min(a/ȧ, e/ė, ω/ω̇) with η_sec ≈ 0.01 (Eq. 47). The Fig. 4 validation agrees with LogH for a 2.5PN BH binary. The paper notes the binary phase "is essentially lost". FC-II still uses this path.

**BIFROST §3.12 (triples).**

- MA01-stable triples only.
- Vector (L, e) equations of Correia et al. 2016 (1608.03484, Eqs. 9-20), quadrupole + octupole.
- **The outer orbit evolves**, with dL2/dt = −dL1/dt (Eq. 56).
- Step η_X P_out with η_X ≈ 0.1.
- Random mean anomalies are drawn when converting back to Cartesian coordinates.
- **FC-I removed this path** in favour of SDAR (l.224, verified).

**Energy bookkeeping.**

- The exact double-averaged flow conserves the secular Hamiltonian. With a frozen outer orbit, total J and ⟨Φ⟩ at octupole order are not conserved (Naoz et al. 2013, 1107.2414: G2 is not constant at octupole order).
- Converting back at arbitrary phases adds an instantaneous offset Φ(phase) − ⟨Φ⟩. Relative to the triple binding energy this is about (m3/m12)(a1/a2)³(1−e2)^{−3}, i.e. about 1e-3 for a1/a2 = 0.1 (our estimate).
- BIFROST's ledger (Eq. 68) has no secular term, so this offset shows up as energy error. It plausibly explains most of our 3.4e-5.

**Production codes.** PeTar and NBODY6++GPU have no secular triples; they use slow-down and KS/AR instead. FC-I/II/III have none either.

### F.4 Experiments

**F-1 Secular triple off** (route to LogH; later SDAR). This needs a config switch.

- Test: isolated-triple test and 200k with 10% binaries, 3 seeds.
- Pass: energy rule.
- Expected: accuracy up (it removes a 3.4e-5 error source); speed down by the cost of LogH on those triples, typically few per step.

**F-2 Diagnose** (only if the secular triple is to be kept). On the isolated triple, log the secular Hamiltonian before and after each call separately from the N-body E. Halve η_sec, which halves the error if Euler is the cause. Rebuild with a deterministic phase advance instead of the current phase treatment.

**F-3 Validity gate**, if kept: MA01 (corrected) AND √(1−e_max²) ≥ 2π ε_SA AND t_KL ≪ T_coll.

**Recommendation.** F-1. The target papers do not use a secular triple, it is our largest known per-call energy error, and its fixes (outer-orbit evolution, second-order integration, phase treatment) are more work than routing to LogH/SDAR. Keep the secular binary. **Confidence: high.**

---

## G. Post-Newtonian terms and spins

### G.1 Plain explanation

- **1PN** (order v²/c²): conservative. It gives pericentre precession and, for N bodies, the Einstein-Infeld-Hoffmann (EIH) equations with genuine three-body cross terms.
- **2PN** and **3PN**: conservative corrections. 3PN is gauge-dependent in harmonic coordinates.
- **2.5PN** and **3.5PN**: radiation reaction.
- **Spin**: spin-orbit coupling enters at 1.5PN, spin-spin at 2PN, and spins precess as dS/dt = Ω × S.

Velocity-dependent forces break the explicit leapfrog. The fix is to carry an auxiliary velocity w (Hellström & Mikkola 2010, CeMDA 106, 143): kick v using forces evaluated with w, kick w using v, half/full/half (BIFROST Eq. 34). The map stays explicit and time-symmetric, which GBS requires. Pihajoki 2015 (1412.1911) adds "mixing" maps that stop v and w drifting apart.

### G.2 What cpp-nbody does now

- **EIH 1PN** (`post_newtonian.cpp:17`) for up to 3 bodies. 2PN-3.5PN are binary-only (bitmask `PN_ORDER = 31`). The aux-velocity leapfrog is at `logh.cpp:376`.
- **Spins.** Rotation updates inside the PN leapfrog (`logh.cpp:211-250`), spin norms restored after extrapolation (`gbs.cpp:239-243`), spin-orbit term (`post_newtonian.cpp:318-375`). Plan bug 4 (spin-orbit coefficient missing μ) is still open (status not re-checked).
- **Global PN1 kick** for compact objects with m ≥ 5 Msun outside subsystems (`post_newtonian.cpp:420`). It copies the whole particle set to the host every call (l.425-432). That is a cost item at 200k if BHs are present.
- **Energy diagnostic:** Newtonian T + U only, by design (memory "Key Lessons").

### G.3 Literature and other codes

- **BIFROST §3.7** (Eq. 36): PN1 from Thorne & Hartle 1985 including three-body terms; PN2-3.5 binary-only from Blanchet 2014. Spins in §3.7.3 (Eq. 38). Global PN1 in §3.8 (Eqs. 41-42).
- **KETJU** (Mannerkoski et al. 2023, 2306.04963 §2.2) states the same choice, explicitly citing "equation (203) of Blanchet 2014". That is the general-frame acceleration of each body, not the c.o.m. relative form. The 2014 equation numbering was checked only via this citation; the current arXiv version of 1310.1528 is the 2024 update with different numbering.
- **Merger radius.**
  - KETJU default: 12 R_S of the total mass. Below about 6 R_S the PN equations misbehave.
  - FC-I: BH-BH merger at 6G m_max/c².
  - Ours (BH+BH): 20GM/c² of the total mass.
- **Production codes.**
  - NBODY7: AR-chain PN with spin-orbit to 3.5PN (Aarseth 2012; detail UNVERIFIED).
  - TSUNAMI: PN up to 2.5 via the Mikkola-Merritt midpoint method.
  - Base PeTar/SDAR: no PN (later additions UNVERIFIED).

### G.4 Experiments

**G-1 Boost test.** An isolated 10 + 10 Msun binary given v_cm = 0, 30 and 300 km/s. The inspiral time and ω̇ must agree to 1e-6. This checks the general-frame vs relative-acceleration implementation. Test only; no A/B.

**G-2 Global PN1 kick on the device** (remove the host copy). Speed only. Pass: bit-identical results.

**G-3 Merger radius** 6 / 10 / 12 R_S. Physics choice; record the remnant spin and kick distributions.

**Recommendation.** G-1 as a regression test (cheap; catches a class of silent errors). G-2 if profiling shows it. **Confidence: medium.** We have not checked which PN form our code implements; this is exactly what G-1 tells us.

---

## H. Neighbour and time-step search acceleration

### H.1 Plain explanation

Three of the four per-level O(N²) passes give exact sums (forces, gradient). The other two passes, the time-step minimum and the neighbour search, only need to *find* the limiting pair, and that can be pruned with a tree. This is not an approximation, because a min-reduction does not depend on order.

- If a tree cell is so far away and so slow that even its most extreme member could not beat the current best τ, skip it.
- Bounds for a cell (centre c, radius R, velocity centre u, velocity radius V, maximum mass m_max):
  - r ≥ |x−c| − R
  - v ≤ |v−u| + V
  - τ_ff ≥ sqrt(r_lo³/G(m_i+m_max))
  - τ_fb ≥ r_lo/v_hi
- The symmetrisation factor is monotonic in τ and in dτ/dt, so the bounds carry over. The fly-by symmetrisation needs a v_lo > 0 guard (our derivation).
- Hydro analogue: GADGET's neighbour search for SPH, except that the "smoothing length" here is the current best τ times a velocity.

Gradient and jerk criteria are sums, so they stay in the force kernel, which runs anyway.

### H.2 What cpp-nbody does now

- The time step is fused into the force kernels (`DtK`/`AccDtK`).
- The neighbour search is a separate O(N_set²) tiled kernel at every level (`forces.cpp:189-243`).
- No tree is used for either. The Barnes-Hut octree (`src/gravity/tree.cpp`) exists for collisionless runs only.

### H.3 Literature and other codes

- **FC-II §2.1** (verified, l.138): "an octree ... used only for the particle neighbour searches while the particle acceleration calculations are still performed using O(N²) direct summation." 2503.21879 adds only that it speeds up "time-step and particle neighbour assignment". **No speed-up, exactness statement or level handling is reported.**
- **FROST §5.2**: GPU for sets above "a few thousand", MPI-CPU below, serial below "a few hundred". These are qualitative, not measured.
- **Gray & Moore 2000** (NIPS 13): dual-tree all-pairs algorithms. Ram et al. 2009 (NIPS) prove O(N) for dual-tree all-nearest-neighbours. Speed numbers UNVERIFIED (the PDF could not be text-extracted).
- **GADGET-4** builds a tree only for the current level's particle set (§ hierarchical integration).
- **ArborX** (Lebrun-Grandié et al., 1908.11807): a Kokkos-native BVH library with range and nearest-neighbour queries, a ready-made device tree for us. Not evaluated.

### H.4 Experiments

**H-1 Exact tree-pruned time steps and neighbour search** (plan M7.4). Two passes:

1. Exact symmetrised min over the k ≈ 16 nearest neighbours, which gives an upper bound τ_ub.
2. A range query with radius from τ_ub and the cell bounds, followed by the exact min over the candidates.

- Pass: τ_i, the level assignment and the neighbour pair set are **bit-identical** to O(N²) for a whole 200k run. The energy rule is then automatic.
- Measure time in the time-step/neighbour phase per level size; expect gains only at N_level ≳ 1e4 on an A100 (our guess, UNVERIFIED).

**H-2 Crossover measurement** first: time the brute-force passes per level from the profiler's per-depth table. If the time-step plus neighbour phase is under about 15% of the step at 200k, H-1 is not worth doing at this N.

**Recommendation.** H-2 now (zero cost), H-1 only if H-2 shows it pays, or for N ≥ 1e6. Because the min is fused into the force kernel today, the tree saves mainly the separate neighbour pass and the deep levels. **Confidence: medium-low** on the payoff at 200k.

---

## I. Far field: FMM in collisional codes

### I.1 Plain explanation

The fast multipole method (FMM) expands distant groups in multipoles and reaches O(N) cost.

- Dehnen's falcON (astro-ph/0003209, astro-ph/0202512) uses symmetric ("mutual") cell-cell interactions, which obey Newton's third law exactly.
- Inside HHS, where all kicks are pairwise, that gives momentum conservation to round-off (GADGET-4 FMM section; Zhu 1712.10116). With block-step Hermite it would not.

The catch for a collisional code is the accuracy level:

- force errors of relative size ε give an energy error of roughly ε√(η/N) per dynamical time (Dehnen 2014, 1405.2255 App. A);
- the FMM potential is only piecewise continuous, so energy is not conserved even with exact time integration.

### I.2 What cpp-nbody does now

`src/gravity/fmm.cpp` is a GPU spherical-harmonics FMM (p up to 20, Morton tree, M2L by rotation) with a test (`tests/fmm_test.cpp`). It is not wired into the integrator.

### I.3 Literature and other codes

**Dehnen 2014** (1405.2255).

- §3.2: GPU direct summation in mixed precision (sapporo) has rms force error about 1e-7. Collisional codes accept this.
- §6.4: at ε = 10^{−6.25} the optimal order is p = 10; cost ∝ N^0.87; 16 CPU cores beat a K20M GPU direct code at N ≳ 1e5.
- §7.1: for active fractions below 1% the cost is set by the tree build.

**Taichi** (Zhu 1712.10116; Mukherjee et al. 2021, 2012.02207).

- HHS + FMM: 100x faster than direct at 1e6, at ε = 1e-7, p = 20.
- No regularisation; cumulative |dE/E| about 1e-4, dominated by encounters.
- 2205.12289 adds HHS-FSI (Omelyan gradient) and AR-Chain at ε = 2e-5, p = 12.

**PeTar P3T** (Oshino et al. 2011, 1101.5504; Iwasawa et al. 2016, 1601.03138; PeTar 2006.16560).

- Tree long-range force with a shared leapfrog step; Hermite + SDAR short-range force; a changeover function joins them.
- 6.5-11.5x faster than NBODY6++GPU at 1e6.
- |dE/E| per N-body time about 1e-7 to 1e-4, read off their figure. That is **4-6 orders above our target**, so P3T is not compatible with our accuracy goal.

**Bonsai** (1106.1900, 1412.0659): collisionless GPU tree, irrelevant for accuracy.

**GENGA** (1404.2324): planetary.

**FC-III**: no 1e6-1e7 single-cluster runs, forces still O(N²), and no timings reported. **No Rantala paper uses FMM.** BIFROST §6 lists it as future work.

**Our numbers** (our inference from Dehnen App. A).

- At N = 2e5, η = 0.2, ε = 1e-7: about 1e-10 per dynamical time, comparable to our whole budget. So FMM needs ε ≲ 1e-8 (p ≈ 14-16).
- The energy diagnostic must stay direct, otherwise the 2ε/√N measurement error swamps the signal.
- At 200k on 4x A100, FMM very likely loses to direct summation (UNVERIFIED); the plan's N ≳ 1e6 threshold is right.

### I.4 Experiments

**I-1 Synthetic force-error injection** before any FMM work. Add random relative errors of size ε ∈ {1e-9, 1e-8, 1e-7, 1e-6} to the direct forces, optionally correlated as a tree would produce them, on 200k for 3 seeds.

- Result: the largest ε that passes the energy rule.
- This sets p, and therefore the FMM cost, before a line of FMM integration code is written.

**I-2 FMM vs direct timing** at N = 1e5, 1e6 at the ε from I-1, using `fmm_test`.

**I-3** If pursued: mutual expansions, FMM only on levels above a set-size threshold, near field direct and unsoftened, Omelyan gradient (A-1). Gate: momentum at machine precision and the energy rule on the FROST 12 Myr 1e6 test.

**Recommendation.** I-1 is a day of work and decides whether M8 is feasible at all at our accuracy target. **Confidence: high** that I-1 should precede M8; **medium** that FMM is ever competitive below 1e6.

---

## J. GPU execution of many small subsystems

### J.1 Plain explanation

At deep levels each HHS call integrates 1-30 subsystems over a tiny interval. The arithmetic is small; launch latency, host-device copies and thread start-up dominate.

There are two standard cures:

- keep everything on the host for small batches;
- keep everything on the device in one persistent kernel that runs many GBS steps per launch and returns only flags.

Hydro analogue: subcycling a stiff chemistry network per cell on the GPU, where batched per-cell solvers win and per-cell launches lose.

### J.2 What cpp-nbody does now

- LogH runs on host threads after a device→host pack; results are pushed back afterwards (`logh_batch.cpp:387`; `integrator.cpp:460-465, 948-965`).
- Audit F2: with F1 on, subsystems take 53% of the step (about 3 s), with 32-512 calls per step at depths 3-7. At depth 7 pack/unpack costs about 0.6 ms per call.
- At 20k on the host, mirror push/pull was below 3 ms per step (audit F4 update). The overhead may therefore be host thread start-up and GPU sync rather than copies (to be measured).

### J.3 Literature and other codes

**BIFROST** (§3.1, §3.5) integrates many small LogH systems in parallel rather than parallelising inside one.

- **It does not say explicitly whether that happens on CPU MPI ranks or on the GPU** (UNVERIFIED; the memory note "small sets on CPU" refers to the force sets, FROST §5.2).
- BIFROST §4: going from 10% to 100% binaries costs +25% (1e4), +24% (1e5), +3% (1e6). Most binaries are secular/Kepler there.

**PeTar** (§4.5): SDAR and Hermite on CPU OpenMP threads, tree on GPU; load imbalance from long-lived hierarchies with "no universal solution".

**NBODY6++GPU**: KS serial on the CPU, regular forces on the GPU (Wang+2015).

**Swarm-NG** (Dindar et al. 2013, 1208.1157).

- Whole integrations on the GPU, one thread or warp per system, forces in shared memory.
- **Hundreds of steps per kernel launch gave 2.4-4x over one step per launch.**
- GPU vs a 6-core CPU only 2.2-2.5x.

**GENGA** (1404.2324 §4).

- Close-encounter groups found by parallel union-find on the GPU.
- Each group is integrated by Bulirsch-Stoer in one thread block, grouped into power-of-two size classes on separate streams.
- GPU beats one CPU core above about 30 independent systems.

**ODE ensemble solvers** (MPGOS 1810.03931; Niemeyer & Sung 1309.2710): one system per thread, 25x over 6 CPU cores for 5e5 systems. That is far more systems than we have per call.

### J.4 Experiments

**J-1 Measure first.** Split the subsystem phase per depth into pack, host-launch, integrate, unpack and fence, on the 200k run with 10% binaries and F1 on. Without this, J-2/J-3 are guesses.

**J-2 Serial host path below a batch size** (e.g. ≤ 4 subsystems: no Kokkos host parallel launch). Persistent host mirrors updated only for member indices. Pass: bit-identical results, lower subsystem time.

**J-3 Device-resident batched LogH/GBS** (plan M7.5). One Kokkos TeamPolicy launch per call; one team per subsystem; size classes as in GENGA; the GBS loop inside the kernel; flags copied back.

- Large effort: GBS needs per-system adaptive control flow and lots of state per team.
- Pass: equal or better energy on 3 seeds; subsystem phase at least 2x faster.

**J-4 Kokkos/CUDA Graphs** for the fixed per-level kernel sequence (launch latency).

**Recommendation.** J-1 then J-2. J-3 only if J-1 shows integration on host threads (not overhead) dominates after SDAR (D-1) has removed the orbit-by-orbit cost. **Confidence: medium.** The audit's two measurements (0.6 ms per call at 200k vs < 3 ms per step at 20k) do not yet agree on where the overhead sits.

---

## K. Physics modules (brief)

**Tidal disruption.**

- BIFROST Eq. 64: r_t = 1.3 ((m⋆+M•)/m⋆)^{1/3} R⋆ (Kochanek 1992).
- FC-I App. B3.3 and FC-III §2.1.4: 1.3 (M•/m⋆)^{1/3} R⋆, with half of the star accreted (f_c = 0.5).
- Ours: `tidal_disruption_radius` (`mergers.cpp:38`); the BH keeps its mass m1, so TDE accretion is 0 (`mergers.cpp:205`).

**Collisions.**

- FC-I App. B3.1: two-stage detection. A candidate is a pair with osculating r_p < R1 + R2; it is then forced to κ = 1 and stepped to the actual pericentre before deciding (verified l.885).
- FC-III §2.2.1: mass loss f = 0.3 q/(1+q)² from both stars (Glebbeek et al. 2013, 1307.2445, Eq. 4 by count).
- Ours: r < R1 + R2 on the current separation, once per level, 2-body subsystems only (`mergers.cpp:188-196`). So collisions are missed between level checks.

**BH merger remnant.**

- BIFROST, FC-I and KETJU use Zlochower & Lousto 2015 (1503.07536). Use the corrected version: the RIT group notes errata in L_ISCO and Table XVIII (UNVERIFIED detail).
- Ours: Barausse-Rezzolla mass/spin and a non-spinning Campanelli-type kick in a random in-plane direction, ignoring spins (`mergers.cpp:49-115`).

**Stellar evolution: correction to project memory.**

- **BIFROST §3.13 does document its SSE modification.** It cites Banerjee et al. 2020 (A&A 639, A41; arXiv 1902.07718 §2.1-2.2, App. A):
  - Belczynski 2010 winds including **Vink 2001 metallicity-dependent winds** (`mdflag = 3`);
  - **Fryer 2012 rapid/delayed remnants** (`nsflag = 3/4`), with Banerjee's erratum for the a1 coefficient in Fryer Eq. 16;
  - Belczynski 2016 PPISN (`psflag = 1`);
  - ECSN, fallback-scaled kicks.
- That explains the 5.5-41.6 Msun BH range: the Fryer-rapid mass gap and the Vink-wind maximum.
- **The memory entry "Paper does not document these modifications" is wrong and should be corrected.**
- Our `ic_gen/sse.cpp` already contains VINK, FRYER_RAPID/DELAYED and PPISN options (`sse.cpp:188-213, 992-1070`). `config.txt:58-60` currently runs VINK winds with BELCZYNSKI02 remnants.
- FC-I onward use SEVN (Iorio et al. 2023, 2211.11774) with per-star t_next and ε_se in the step. FC-III adds Vink 2018 winds above 100 Msun.

**Experiments.**

- **K-1 BH mass spectrum** (no dynamics). Run the IC generator at Z = 2e-4, 1 Gyr with `SSE_REMNANT_MODEL = FRYER_RAPID`, `SSE_WIND_MODEL = VINK`, `SSE_PPISN_MODEL = STRONG`. Compare the BH count (target 357) and mass range (target 5.5-41.6 Msun) with BIFROST Fig. 9. A config run of minutes. If close, Fig. 9 becomes a like-for-like comparison.
- **K-2 FC-I pericentre-forced collision detection** inside LogH/SDAR (plan M5). Count missed collisions against a reference run with forced pericentre steps.
- **K-3 Zlochower-Lousto 2015 recoil with spins** (plan M5). Physics, not A/B.

---

## Ranked experiment table

Ranked by expected value per unit effort, with accuracy gates first. "Cost" is the implementation effort. All tests use 3 seeds (42/7/13) unless noted.

| # | id | Area | Change | Expected speed | Expected accuracy | Cost | Test setup | Pass criterion | Literature basis |
|---|---|---|---|---|---|---|---|---|---|
| 1 | B-1 (+F1) | B | `FLYBY_CM_VELOCITY=true` (all paths); jerk off, v/a off, η_∇ = 0.2 | 2x (F1 measured) plus 1.2-2x fewer deep steps | unknown; FC papers run this set | config (CPU/CC paths: hours) | 200k, 10% binaries, 35 steps | energy rule; split into B-1a/b/c if it fails | FC-I App. B l.869, Table l.322; FC-II §2.1 l.138 |
| 2 | F-1 | F | secular triple off → LogH | slightly slower | removes a 3.4e-5 error source | hours | isolated triple + 200k, 10% binaries | energy rule | FC-I §2 l.224 (removed it); BIFROST Eq. 56 (ours freezes the outer orbit) |
| 3 | C-1 + C-2 | C | weak-perturber split γ-sweep 1e-10..1e-6; field sees the c.o.m. of split binaries | removes most F3 stalls (60x on a fly-by test) | C-1 alone was 3.6x worse; C-2 targets the cause | C-1 config; C-2 days | 20k F3 reproduction (seed 7), `weak_flyby_split`, 200k, 10% binaries | energy rule; no interval > 60 s | NBODY6 GMIN (Aarseth 2003 §8; PeTar §3.3); our inference |
| 4 | E-1/E-3 | E | η_GBS 1e-10→1e-8; end-time tolerance 1e-4→1e-2; DYNCAP ∈ {3, 1, 0.3} | 1.3-3x in the subsystem phase | likely neutral (global error FSI-dominated) | config/env | 200k, 10% binaries, 20 steps | energy rule; worst subsystem dE recorded | FC-I Table l.328-329; FC-II/III tables |
| 5 | D-1 | D | SDAR jumping κ (k_ref 1e-6, c = 0.1, κ = 1 for PN/collisions) | removes F3 orbit-by-orbit cost (about 2e3x on that system) | expected neutral; ε(H_sd) tracked | weeks (plan M3) | SDAR paper tests; F3 seed 7; 200k, 10% and 50% binaries | energy rule; F3 interval < 5 s | SDAR 2002.07938 §2.4, Eqs. 22-28; FC-I App. A2 |
| 6 | J-1 → J-2 | J | measure the per-depth overhead split; serial host path for small batches | up to 2x on the subsystem phase | bit-identical | hours / days | 200k, 10% binaries, profiler | bit-identical; faster | Swarm-NG 1208.1157; GENGA 1404.2324 §4 |
| 7 | I-1 | I | synthetic force-error injection ε = 1e-9..1e-6 | — (feasibility) | finds the tolerable ε | 1 day | 200k no binaries, 20 steps | largest ε meeting the energy rule | Dehnen 2014 1405.2255 App. A |
| 8 | K-1 | K | FRYER_RAPID + VINK + PPISN in the IC SSE | — | — | config | IC generator only | 357 BHs, 5.5-41.6 Msun within ~10% | BIFROST §3.13 → Banerjee 2020 1902.07718 |
| 9 | A-2 | A | `HHS_CC=true` default with binaries | about 2x in binary-rich runs | neutral expected | config | 200k at 10% and 50% binaries | energy rule | Jänes 2014 1407.7105 Figs. 3, 5 |
| 10 | C-3 | C | MA01 with inclination / Vynatheya criterion | — | fewer wrongly averaged triples | hours | triple census + 100-P_out LogH checks | no "stable" triple disrupts | 2207.03151 §4 Eqs. 2-4 |
| 11 | E-2 | E | Deuflhard sequence, k_max 8, MSTAR end-time control | 1.2-1.5x in GBS | equal | hours | `kepler_e09`, `frost4_e099`, `pn_logh_books`, 200k | equal dE per call, fewer substeps | MSTAR 2001.03180 §2.3-2.4, Eqs. 11, 15-16 |
| 12 | B-2 | B | tidal-tensor criterion instead of jerk | equal step count by calibration | possibly better in the core | days | 200k, 10% binaries | energy rule at ≤ steps | Grudić & Hopkins 1910.06349 Eqs. 8-11 |
| 13 | C-4 | C | r_ngb 1 → 0.5 / 0.25 mpc | fewer and smaller subsystems | unknown | config | 200k, 10% binaries | energy rule | FC-I/FC-II tables |
| 14 | A-1 | A | Omelyan gradient | ±10% | η⁴ must survive | days | Kepler/frost4 η-sweeps, 200k | slope 4, energy rule | FC-I App. A1 l.761-776; Chin 2007 0704.3273 Eq. 2.8 |
| 15 | G-1 | G | PN boost test | — | catches frame errors | hours | isolated BH binary at v_cm 0/30/300 km/s | t_merge, ω̇ agree to 1e-6 | KETJU 2306.04963 §2.2; Blanchet 2014 Eq. 203 |
| 16 | H-2 → H-1 | H | measure, then exact tree-pruned τ/neighbour search | gain only if the phase is > 15% of the step | bit-identical | hours / weeks | profiler per depth; 200k, 1e6 | bit-identical τ and pairs | FC-II §2.1 (no numbers); our derivation |
| 17 | B-5 | B | interval 5e-3 Myr (+ N₁ > N₀/3 rule) | fewer top-level syncs | unknown | config | 200k, per-Myr comparison | energy rule per Myr | FC-I/II tables; GADGET-4 2010.03567 |
| 18 | B-3 | B | Galilean-invariant v/a (if kept) | — | removes frame dependence | hours | IMBH at rest ± uniform boost | same level assignment | Grudić & Hopkins §2; our analysis |
| 19 | J-3 | J | device-resident batched LogH/SDAR | 2x+ on the subsystem phase if integration-bound | equal | weeks | 200k, 10/50% binaries | energy rule; 2x faster | GENGA §4; Swarm-NG |
| 20 | K-2/K-3 | K | pericentre-forced collisions; ZL15 recoil | — | physics | days each | head-on, TDE, BH-BH tests | exact books | FC-I App. B3; 1503.07536 |

---

## Findings that contradict the current plan or project memory

1. **Memory: "BIFROST does not document its SSE modifications" — wrong.** BIFROST §3.13 cites Banerjee et al. 2020 (1902.07718): Vink winds, Fryer 2012 remnants and PPISN. Our SSE already has these switches; experiment K-1.

2. **Plan and config keep jerk and v/a time-step criteria at η = 0.1. FC-I/II/III use min(ff, fb, ∇) at η = 0.2** (FC-I App. B l.869 and Table l.322, verified). The plan's accuracy targets come from BIFROST 2023 (which used jerk). Its speed targets for the FC era assume the cheaper set. Experiment B-1.

3. **F3 is mischaracterised.**
   - The perturber is unbound (190 km/s against a 0.6 km/s escape speed), with γ ≈ 1e-19 to 1e-14. It is a fly-by dragged in by BIFROST's distant fly-by neighbour criterion, not a triple.
   - The audit's orbit count is inconsistent: a = 0.01 AU around 2.14 Msun gives 1.5e6 orbits per interval, not 8.9e4.
   - A cheap weak-perturber exclusion plus c.o.m. coupling (C-1/C-2) may fix most stalls before SDAR.

4. **The plan treats FC-II's r_strong = 10a as a dynamics change.** It is only a switch for binary stellar evolution (FC-II §2.3.2).

5. **FMM at our accuracy target needs ε ≲ 1e-8** (Dehnen 2014 App. A applied to N = 2e5). Prior art for HHS-FSI + FMM is Taichi (2205.12289), not a Rantala code; FC-III still uses O(N²). Run I-1 before committing to M8.

6. **The FC-II octree is undocumented** (one sentence, no numbers). Plan M7.4's "bit-identical" requirement is achievable (min-reductions prune exactly) but its payoff at 200k is unproven (H-2 first).

7. **BIFROST does not state that its subsystems run on CPUs.** "Small sets on CPU" in memory refers to the force sets of FROST §5.2. The F2 design should follow measurement (J-1), not that assumption.

8. **MA01 as printed in BIFROST Eq. 54 has a misplaced bracket.** Our version also lacks the inclination factor (plan bug 6). Use Vynatheya 2022 Eq. 2.

9. **Two local PDFs are mislabelled.** `timestep_criteria/0802.4245.pdf` is VINE (Wetzstein et al. 2009), not Nitadori & Makino 2008 (that is 0708.0738). `timestep_criteria/astro-ph_0505010.pdf` is GADGET-2, not Chin & Chen (that is astro-ph/0304223, now in `fsi/`).

---

## Appendix: papers on disk (`/raven/u/uli/papers/`)

| Directory | Contents |
|---|---|
| `rantala/` | 2001.03180 MSTAR, 2011.14984 FROST, 2210.02472 BIFROST, 2403.10602 FC-I, 2506.04330 FC-II, 2601.07917 FC-III, 2503.21879 letter (LaTeX in `src/`) |
| `sdar/` | 2002.07938 Wang, Nitadori & Makino 2020 |
| `fsi/` | 0704.3273 Chin 2007; astro-ph_0304223 Chin & Chen 2005; 1609.09375 Dehnen & Hernandez 2017 |
| `timestep_criteria/` | 1205.5668 Pelupessy+2012; 1407.7105 Jänes+2014; 1707.09069 Dehnen 2017; 1910.06349 Grudić & Hopkins 2020; 2401.02849 Pham+2024; 0802.4245 (= VINE); astro-ph_0505010 (= GADGET-2) |
| `tree_timestep/` | 2006.16560 PeTar; 2010.03567 GADGET-4; gray_moore_2000 |
| `subsystems/` | 2006.16560 PeTar; 1504.03687 NBODY6++GPU |
| `fewbody/` | astro-ph_9906322 Preto & Tremaine; astro-ph_0605054 and 0709.3367 Mikkola & Merritt; 2206.10583 TSUNAMI |
| `integrators/` | 1409.4779 IAS15; 1506.01084 WHFast; 1903.04972 MERCURIUS; 2405.03800 TRACE; 1503.02728 Hernandez & Bertschinger; 1402.3325 SAKURA |
| `triple_stability/` | 2207.03151 Vynatheya+; 2206.12402 Lalande & Trani; 2209.08487 Hayashi+ |
| `secular/` | 1308.3674 Antonini+2014; 1601.04345 Luo+2016; 1601.07175 Naoz 2016; 1808.02030 Grishin+2018; 2304.01257 Tremaine 2023; 1107.2414 Naoz+2013; 1608.03484 Correia+2016 |
| `pn/` | 1310.1528 Blanchet (2024 version); 2306.04963 KETJU 2023 |
| `fmm/` | 1405.2255 Dehnen 2014; 1712.10116 Zhu; 2012.02207 and 2205.12289 Mukherjee+ |
| `gpu_fewbody/` | 1208.1157 Swarm-NG; 1404.2324 GENGA |
| `stellar_evolution/` | 1902.07718 Banerjee+2020 |
| `mergers/` | 1503.07536 Zlochower & Lousto 2015 |
| `collisions/` | 1307.2445 Glebbeek+2013 |

Not on arXiv or not found: Mardling & Aarseth 2001, Mikkola & Tanikawa 1999, Mikkola & Aarseth 1996/2002, Deuflhard 1983, Aarseth 2003 (book), Hellström & Mikkola 2010, Peters 1964, Thorne & Hartle 1985.
