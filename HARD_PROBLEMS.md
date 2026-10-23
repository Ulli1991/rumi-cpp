# The hardest problems for a collisional N-body code: literature audit (October 2026)

**What this document is.** It answers six questions about cpp-nbody:

- Which problems are the frontier for direct ("collisional") N-body codes today?
- What makes each of them hard?
- What would the user's candidate cost on our code? The candidate is a core-collapsing cluster of 1e6 stars or more.
- What do the other codes manage, and what do they approximate to get there?
- What does our code lack?
- Which two or three flagship problems should define success?

This is research and writing only; no code was changed.

**Companion documents.**
- `METHODS_SURVEY.md`: the methods, explained.
- `AUDIT_2026-10.md`: our measurements.
- The plan: `/u/uli/.claude/plans/bubbly-fluttering-falcon.md`, milestones M0–M8.

**Papers.** They are under `/raven/u/uli/papers/`. This audit added four folders:
- `frontier_runs/`: the big simulations;
- `cluster_theory/`: relaxation, collapse, chaos, Monte Carlo codes, nuclei;
- `multiplicity/`, `binary_evolution/`, `gpu_scaling/`.

**Conventions.**
- Section numbers are those of the arXiv LaTeX sources.
- **UNVERIFIED** marks a number we could not find in the source text.
- "Our estimate" marks arithmetic of ours, not a paper's.
- Abbreviations: R23 = Rantala+2023 (2210.02472); FROST = Rantala+2021 (2011.14984); FC-I/II/III = FROST-CLUSTERS I (2403.10602), II (2506.04330), III (2601.07917); NB6++ = NBODY6++GPU.

---

## Summary

1. **The user's guess is the right kind of problem, but it needs to be sharpened.**

   *Who has done what at 1e6:*
   - **DRAGON (2016)** is still the only star-by-star 1e6-star cluster integrated to 12 Gyr, with 5 % binaries. It took about a year on 16 GPUs.
   - **DRAGON-II (2023)** got its 1e6 runs only to about 0.2 Gyr.
   - **FC-I/II/III** ran 1.7–2.35e6 stars, but for only 10–50 Myr. These are young, assembling clusters, where core collapse takes a few Myr.

   *What nobody has done:*
   - a direct N-body cluster of 1e7 stars;
   - 1e6 stars with a realistic (≈ 100 %) multiplicity through core collapse and into the post-collapse phase;
   - any of this at the accuracy of regularised direct summation (R23).

2. **The problem's cost does not depend on the cluster's density.** In pure gravity, the work per crossing time depends only on N. The number of crossing times to core collapse is (t_cc / t_rh) × 0.138 N / ln(γN), which also depends only on N.

   *What the crossing times add up to:*
   - A realistic stellar mass spectrum collapses in t_cc ≈ 0.07–0.2 t_rh (Gürkan+2004; Portegies Zwart & McMillan 2002).
   - For N = 1e6 that is ≈ 1200–2400 crossing times. For 1e7 it is ≈ 10,000–20,000.

   *What that costs on our code (first-interval speed, 4 A100):*
   - 1e6: about 0.5–1 year on one raven node.
   - 1e6 on ~40 GPUs, at production speed: about 2–6 months. This is the same order as DRAGON.
   - 1e7 with direct forces: about 400–800 years on one node, and still 7–15 years on 400 GPUs.

   **1e7 is out of reach without a far-field approximation (M8).** FROST's own projection agrees: about 4 weeks per 10 Myr at 1e7.

3. **Our accuracy is the expensive part, and it is not what core collapse needs.**
   - At 1e6 our code costs about 19,000–31,000 s per N-body time unit on 4 A100 (R23 Fig. 7 is in the same range).
   - NB6++ needs 1500–3000 s on 16 K20X. PeTar needs about 120 s on 960 CPU cores, with all stars in binaries.
   - Per unit of hardware, unsoftened direct summation with regularised subsystems is roughly 10–100× more expensive than these codes.
   - The global energy error through core collapse is set by chaotic strong encounters, about 3e-4 regardless of the time-step parameter (R23 §5.6).
   - Results are statistically valid if energy is conserved to better than about 1/10 (Portegies Zwart & Boekholt 2014).

   High accuracy matters where errors are systematic rather than random:
   - secular few-body evolution (von Zeipel–Lidov–Kozai cycles, explained in §1.5);
   - post-Newtonian inspirals;
   - long-lived hierarchies;
   - the energy books, which are our bug detector.

   **It does not matter in the far field.** That is the opening for an FMM far field with a looser tolerance.

4. **Proposed flagships**, explained in §6:
   - **A.** An evolved, BH-rich 1e6 cluster with observed low-mass multiplicity, run through BH-driven core collapse into balanced evolution. Black-hole dynamics are fully post-Newtonian. The science is GW captures, eccentric mergers and hierarchical mergers.
   - **B.** A dense 1e5–1e6 young cluster with full massive-star multiplicity: triples, and later quadruples. The science is runaway collisions, very massive stars (VMS) and IMBHs. This is head-to-head with FC-II/III and needs binary stellar evolution.
   - **C**, the long-term moonshot: a 1e7 cluster through core collapse with an FMM far field. It needs M8 and multi-node scaling.

5. **Biggest gaps in our code:**
   - **No restart from disk.** The rollback is in memory only. A 24 h queue limit makes any long run impossible today.
   - **No snapshot output and no Lagrangian-radius/core diagnostics.**
   - **No external tidal field.**
   - **No binary stellar evolution.**
   - **No triples or quadruples in the initial conditions.**
   - **No multi-node speed-up yet** (measured: 2 nodes do not help at 200k).
   - **MSTAR not wired.** Groups of more than 8 bodies are left to the FSI, which matters for an IMBH with its cusp and for VMS growth.
   - **O(N²) terms besides forces:** time-step sums and the energy check.
   - **GW recoil without spins.**

---

## 1. The frontier problems

This section first explains the vocabulary, then goes through the candidates one by one.

### 1.0 Concepts used throughout

**Collisional system.** A star cluster in which individual two-body encounters change the orbits over the system's lifetime.
- Hydro analogy: a gas in which the mean free path is not small, so the "viscosity" (two-body relaxation) acts on the global timescale.
- Galaxies are collisionless. Their two-body relaxation time is far longer than the Hubble time, so smooth (softened) gravity is fine for them.
- Clusters are collisional, so we cannot soften gravity. Close encounters, binaries and collisions must be integrated exactly.

**Crossing time t_cr.** The time a star needs to cross the cluster, ~ (r_h³/GM)^{1/2}. Our code prints this quantity.
- The "N-body time unit" (Hénon unit) is about 1.5 t_cr for a Plummer model.
- Hydro analogy: the sound-crossing time.

**Relaxation time t_rh.** The time after which two-body encounters have changed a typical star's velocity by order itself (Spitzer 1987, eq. 2.63, quoted as Gürkan+2004 eq. 4):

t_rh = 0.138 N / ln(γN) × (r_h³/GM)^{1/2}, so **t_rh / t_cr = 0.138 N / ln(γN)**.

- γ ≈ 0.11 for equal masses (Giersz & Heggie 1994).
- γ ≈ 0.015 for a mass spectrum (Giersz & Heggie 1996/97); Gürkan+2004 adopt 0.01.
- The textbook shorthand is t_rh ≈ N/(8 ln N) t_cr.
- Hydro analogy: the viscous (diffusion) time. Relaxation conducts heat (kinetic energy) outward, as conduction does in a self-gravitating gas.
- **This ratio is why the cost explodes.** A 1e6 cluster has t_rh ≈ 12,000 t_cr, and a 1e7 cluster has ≈ 100,000.

**Core collapse, the gravothermal catastrophe.** A self-gravitating system has negative heat capacity (Antonov 1962; Lynden-Bell & Wood 1968).
- The dense core loses heat by conduction (relaxation) to the halo. It then contracts and heats up, which makes it lose heat faster.
- The core density runs away toward infinity in finite time.
- Hydro analogy: a thermal runaway, like a star's core with no nuclear burning, only in reverse sign.

How long it takes:
- **Equal masses:** t_cc ≈ 15 t_rh.
  - Cohn 1980, quoted by Portegies Zwart & McMillan 2002 (PZM02) §2.1.
  - PZM02's own N-body runs (N = 1e4): 15.2 ± 0.1 t_rh.
  - CMC at N = 1e8: 16.8 t_rh (2106.02643 §4.1).
  - Fujii & Portegies Zwart 2014 (1304.1550 §1): 15–20.
- **A realistic mass spectrum:** heavy stars sink to the centre by dynamical friction (mass segregation) and collapse much sooner.
  - PZM02 §2.1: t_cc ≈ 0.2 t_rh; their N-body runs give 0.19 ± 0.08.
  - Gürkan, Freitag & Rasio 2004 (astro-ph/0308449): typically ~0.1 t_rh(0); 0.07 for Plummer with Salpeter 0.2–120 M☉ (Table 2).
  - Gürkan's limit for m_max/⟨m⟩ > 50 is t_cc ≈ 0.15 t_rc(0), with t_rc(0) the *central* relaxation time (eq. 16).
  - Fujii & PZ 2014 find t_cc ∝ (m_max/⟨m⟩)^{-1} in direct N-body, against Gürkan's −1.3.

**What stops collapse: binaries as the energy source.**
- A *hard* binary has binding energy above the typical stellar kinetic energy. When it meets a third star, on average it becomes harder and gives the released energy to the passer-by. This is Heggie's law (Heggie 1975).
- The core therefore has a heat source, like nuclear burning in a star. Hénon's principle (Hénon 1961/65) says the core generates exactly the energy the halo absorbs: Ė ≈ ζ|E|/t_rh, with ζ ≈ 0.09 (Breen & Heggie 2013, 1304.3401 §2).
- **Primordial binaries** are an energy reservoir from t = 0.
- **Black holes** are an even stronger heat source. They segregate to the centre, form their own collapsing subsystem and power the cluster's "balanced evolution" for ~10 t_rh (Breen & Heggie 2013).
- Soft binaries, which are wide, are destroyed instead. They act as energy sinks.

**Gravothermal oscillations.** After collapse, an equal-mass cluster with N ≳ 7000 oscillates: core re-expansion, then re-collapse.
- Theory: Sugimoto & Bettwieser 1983; Goodman 1987.
- First N-body confirmation: Makino 1996 (astro-ph/9608160), N = 32k, which took about 3 CPU-months on a quarter of GRAPE-4.

### 1.1 Million- to ten-million-body clusters through core collapse and beyond

The user's candidate.

**Done so far:**

| Run | N | Binaries | Physical time | Hardware | Wall clock |
|---|---|---|---|---|---|
| DRAGON D1, Wang+2016 (1602.00759 §2.1, §4) | 1.05e6 | 5 % | 12 Gyr | 8 nodes, 16 K20m (Hydra) | ~8600 h ≈ 1 yr |
| NB6++ code paper, Wang+2015 (1504.03687 §5) | 1e6 | 5 % | ~4500 N-body units (our conversion) | 8 nodes, 16 K20X | 1500–3000 s per N-body unit |
| DRAGON-II, Arca Sedda+2023 (2307.04805 §2.3, Table 1) | 1.2e5–1e6 | 10–33 % of stars | 0.1–2.3 Gyr; the 1e6 runs only 194/207 Myr | JUWELS Booster | 19 runs, 2.2 M core-h, "~2 yr" |
| Heggie 2014, M4 (1409.5597 §2.2) | 4.8e5 | 7 % | ≳12 Gyr | 2 C2050 GPUs, 12 cores | 2 yr 8 months |
| FC-I (2403.10602 §3.3) | 1.7–2.35e6 | none | 50 Myr | 2–4 nodes of P100/A100 | Nov 2023 – Mar 2024 |
| FC-II (2506.04330) | 1.8e6 | Offner-type binaries and triples | 10 Myr | MPCDF + JUWELS Booster | not stated |
| PeTar benchmark (2006.16560 §7.2, §8) | 1e6 / 1e7 | 100 % | start-up only, no stellar evolution | 960 CPU cores | 121 s / 1344 s per N-body unit |

What the DRAGON numbers say:
- DRAGON spent **2000–4000 h per initial half-mass relaxation time** (1602.00759 §6).
- For an r_h = 1 pc globular cluster, the authors estimate "10⁵ hours" for 12 Gyr.
- DRAGON's "core collapse" was BH-driven. It ended after 1 Gyr (R7 models) or 200 Myr (D4); the luminous core kept expanding (§4.6).

Records:
- **No 1e7 direct N-body production run exists.** We searched arXiv and the web.
- The Living Review (Spurzem & Kamlah 2023, 2305.11606 §5.3) still names DRAGON as the record for "the largest direct N-body simulation of a globular cluster with all required astrophysics".

### 1.2 Young massive clusters: runaway collisions, very massive stars, IMBHs

**The idea.** In a dense young cluster, the massive stars segregate and collide before they die (t ≲ 3 Myr). A runaway VMS then forms and may collapse to an intermediate-mass black hole (IMBH).
- This requires t_cc < 3 Myr (PZM02).
- Portegies Zwart+2004 (astro-ph/0402622):
  - N = 131k (some runs 585k) on GRAPE-6;
  - runaway masses of 800–3000 M☉ for concentration c > 2;
  - a Kroupa IMF for MGG-11 "would require 3.6 million stars, beyond our current capabilities".

**The frontier now:**
- **FC-I:** IMBHs up to ~2200 M☉ from 1.7–2.35e6 stars assembling hierarchically from ~1000 sub-clusters.
- **FC-II:** with binaries and triples, IMBHs up to ~10⁴ M☉; up to ~10 IMBHs per run; TDE rates ~5e-5 /yr.
- **FC-III:** more than 1400 isolated runs and 30 hierarchical runs across Σ_h = 4e3–4e6 M☉/pc² and Z = 0.01–1 Z☉. It provides fitting formulae for IMBH mass.
- **Rizzuto+2021** (2008.09571): 80 NB6++ runs at N = 1.1e5. VMS ~400 M☉; IMBH formation in "not more than 3/8" of realisations.
- **DRAGON-II II:** 8 IMBHs of 107–350 M☉ in 19 clusters.
- **Fujii+2024 Science** (2406.06772): SPH gas coupled to PeTar, stars integrated star by star.
  - Clusters of 4e5 M☉ form VMSs of ~10⁴ M☉.
  - The runs were halted before 1 Myr; the 1e6 M☉ low-Z case was "too computationally expensive".
- **Gieles+2018** (1804.04682) is an analytic "conveyor-belt" model, not a simulation.

**Why this is easier in compute time than §1.1:** the physical time is only 3–50 Myr.

**Why it is harder in physics:**
- collisions;
- VMS stellar evolution, with tracks above 100–600 M☉;
- massive-star multiplicity of ~100 %, with triples dominant (§1.4);
- a large IMBH that drags a cusp of hundreds of bound stars into one regularised region (MSTAR territory).

### 1.3 Nuclear star clusters with an SMBH: loss cone, TDEs, EMRIs

**Physics.**
- **Loss cone:** stars on orbits that pass inside the tidal radius of the central SMBH are disrupted, giving tidal disruption events (TDEs). Compact objects there spiral in as extreme-mass-ratio inspirals (EMRIs).
- The rate is set by how fast relaxation refills those orbits (Frank & Rees 1976; Lightman & Shapiro 1977).
- Observed or inferred TDE rates are ~1e-5 to a few 1e-4 per galaxy per year. Stone & Metzger 2016 (1410.7772 §4.1) give Ṅ ≈ 2.9e-5 (M•/1e8)^{-0.404} yr⁻¹.

**Why direct N-body cannot do it at full scale:**
- The Milky Way nuclear star cluster has ~2.5e7 M☉, i.e. ~1e7–1e8 stars, with t_rh ≈ 14 Gyr.
- The tidal radius is ~1e-6 of r_h. Amaro-Seoane's Living Review (1205.5240 §1.4.1) calls this "a problem of ~10 orders of magnitude".
- To get the loss-cone regime right needs N "well beyond the capabilities of existing or planned computers" (Harfst+2007, astro-ph/0608125 §6).

**Workarounds people use: scaled-down models.**
- **Panamarev+2019** (1805.02153): 1e6 particles, each standing for 65 stars.
  - The SMBH is an external potential.
  - The capture radius is inflated to 10³ R_S, then rescaled with loss-cone theory.
- **Baumgardt, Amaro-Seoane & Schödel 2018** (1701.03818): 50k growing to 255k stars, with r_h inflated so that t_rh matches.
  - They warn that TDEs "cannot be modelled easily in scaled simulations".

**Our position.**
- MSTAR (Rantala+2020, 2001.03180) was built for regularised SMBH regions: |dE/E| ~1e-14.
- R23 §6 names nuclear clusters with 1e7–1e8 particles as the reason for a tree or FMM.
- **Verdict: not a near-term target.** It is the natural long-term target once FMM (M8) and MSTAR (M4) work.

### 1.4 100 % primordial multiplicity: binaries, triples, quadruples

**Observations** (Moe & Di Stefano 2017, 1606.05347, Table 13 and §12): companions with q > 0.1 and 0.2 < log P[d] < 8.

| Primary | Companions per primary | Single | Binary | Triple + quadruple |
|---|---|---|---|---|
| Solar type (0.8–1.2 M☉) | 0.50 | 0.60 | 0.30 | 0.10 |
| Early B (9–16 M☉) | 1.6 | 0.16 | 0.32 | 0.52 |
| O (> 16 M☉) | 2.1 | 0.06 | 0.21 | 0.73 |

- Offner+2023 (2203.10066, Table 1) gives the same trend. The triple-or-higher fraction rises from ~2 % for late-M stars to 14 % for FGK stars and 68 % for O stars.
- Sana+2012 (1207.6397): over 70 % of O stars will exchange mass with a companion.

**Who has done it:**
- **PeTar** (Wang, Tanikawa & Fujii 2022, 2110.15555) ran f_bin = 0–1 at N = 1e5 to 7–12 Gyr. A BSE bug found later changed the BBH merger count from 1 to 21, and rerunning was "extremely computationally expensive (a few months)".
- **FC-II** is the only million-body run with triples: Offner fractions, f_trip up to 0.68 for primaries above 30 M☉, Moe & Di Stefano orbits, 10 Myr.
  - It has no quadruples.
  - It has no triple stellar evolution, as the paper says explicitly.
- **R23 App. A** warns that artificial multiplets from random pairing "may lead to spurious formation of high-velocity stars or artificially enhanced merger rates".

**Why it is hard:**
- A binary of period P costs work every orbit, so short periods dominate. log P down to 0.2 means days, against a Gyr cluster lifetime.
- Hierarchical triples need either secular methods, which fail when perturbed, or slow-down (SDAR), which loses orbital phase (2002.07938 §1).
- Binary stellar evolution must be coupled to dynamics (§2.7).

### 1.5 GW captures and hierarchical black-hole mergers with relativistic recoil

**GW capture.** In a chaotic binary–single encounter, two BHs pass so close that gravitational-wave emission binds them. They merge while still eccentric, and are visible as eccentric events in LIGO's band.
- Samsing 2018 (1711.07452): ~10 % of globular-cluster BBH mergers are captures, and up to ~5 % are eccentric at 10 Hz.
- Rodriguez+2018 (1712.04937), with Monte Carlo + PN: ~55 % of BBH mergers happen inside the cluster; ~10 % of those are captures with e > 0.1 at 10 Hz.
- Captures need post-Newtonian (PN) terms, especially the 2.5PN radiation reaction, inside the few-body integration.

**Recoil.** Merging BHs receive a kick from anisotropic GW emission.
- Non-spinning: at most ~175 km/s (Gonzalez 2007).
- Spinning: up to ~4000–5000 km/s (Campanelli+2007; Lousto & Zlochower 2011).
- Globular-cluster escape speeds are 10–100 km/s.
- So retention, and therefore *hierarchical* second-generation mergers, depends on the birth spins. Rodriguez+2019 (1906.10260): 60 % of merger products are retained at χ_birth = 0 but under 3 % at χ_birth = 0.5.

**Who does this:**
- Monte Carlo codes do it routinely: CMC with Fewbody + PN.
- DRAGON-II used only orbit-averaged Peters decay, and applied kicks largely in post-processing.
- R23 and the FC papers have PN up to 3.5 in regularised subsystems, plus Zlochower–Lousto recoil.
- **A direct-N-body measurement of capture and hierarchical-merger rates at 1e6 with full in-cluster PN is open.**

**von Zeipel–Lidov–Kozai (ZLK) cycles.** In a hierarchical triple, the outer body drives slow oscillations of the inner binary's eccentricity and inclination. They can push the inner binary to merger. This is the archetype of a *secular*, systematic effect that numerical errors can bias.

### 1.6 Globular clusters to a Hubble time

What it takes:
- Several t_rh: DRAGON covered 0.5–2 t_rh0.
- An external tidal field: escapers and dissolution (Baumgardt & Makino 2003, astro-ph/0211471, up to 131k stars to dissolution).
- Stellar evolution over 12 Gyr.
- Ideally 2nd core collapses: Heggie 2014 found M4's 2nd collapse near 11 Gyr.

Cost:
- DRAGON: one year at 1e6.
- Monte Carlo does the same model in about one day on 48 cores (Rodriguez+2016, 1601.04227 §2, §5).

**This is where Monte Carlo is the right tool, and direct N-body is the calibrator.**

### 1.7 Star-by-star dwarf galaxies

- **GRIFFIN** (Lahén+2020, 1911.05093) and **LYRA** (Gutcke+2021, 2010.07311) resolve individual massive stars, but with softened gravity: 0.1 pc in GRIFFIN.
- **Lahén+2025** (2410.01891) is the first star-by-star galaxy model with collisional dynamics.
  - KETJU regions (radius 3ε) around every star above 3 M☉, regularised inside.
  - Validated against the R23 code at 10³–10⁴ stars.
  - Softened clusters showed "practically no evolution"; regularised ones expanded 10–100×.
- R23 §7 calls full dwarf galaxies a goal once a tree or FMM exists.

**For the user (hydro background) this is the most natural bridge problem:** a collisional N-body module inside a galaxy-scale hydro code. It is a different code architecture, though: hybrid softened/regularised, not O(N²).

---

## 2. What makes these problems hard, mechanism by mechanism

### 2.1 Dynamic range in time and space

**Time.**
- A 1-day binary orbit against a 1 Gyr evolution is a ratio of 3.7e11 ≈ 2³⁸ (our estimate).
- Our top-level interval is 1e-3 Myr (365 days). Inside it, binaries are advanced analytically (Kepler/secular), regularised (LogH/SDAR), or by time steps down to `DT_MIN_MYR = 1e-14`.
- A BH binary near merger has an orbital period of milliseconds, a range of ~1e19 overall.
- Makino 1996 (§2.2) already noted KS steps of ~1/(100 N^{2.5}) of the evolution time, "about as small as the limit of double precision" at N ~ 1e5.
- Hydro analogy: individual time bins with 40–60 levels instead of GADGET's ~10–20. The deepest levels hold only a handful of particles, so they are pure latency (§2.6).

**Space.**
- A stellar radius (R☉ = 2.3e-8 pc) against a tidal radius of 50–100 pc: ~4e9.
- A BH collision radius (~300 km ≈ 1e-11 pc) against the same: ~1e13.
- Positions near the tidal radius keep only ~16 decimal digits in double precision. Relative coordinates inside subsystems are therefore essential, as are the "integer positions" of FROST.

### 2.2 Hard binaries in the core: energy generation, scatterings

Binary–single and binary–binary encounters in the core:
- generate the heat that stops collapse (§1.0);
- eject stars and BHs;
- build hierarchies;
- cause collisions.

Each encounter is a chaotic few-body problem that must be integrated to the end.
- NB6++ (1504.03687 §4.3): with 5 % binaries at 1e6, "the KS takes about half of the calculation time when N_node ≥ 8".
- Our own profile (Fig. 7 reproduction, `/raven/ptmp/uli/cpp-nbody-runs/fig7-2026-10-11/summary.md`): subsystems take 57–78 % of the wall time at N = 1e4–1e5, but 21 % at 1e6, where O(N²) work dominates.

### 2.3 The stochastic energy error in core collapse

R23 §5.6 (Figs. 13–15; N = 20k, 10 % binaries, r_h = 0.5 pc, 4 Myr, central density ×30):
- The total error reaches |ΔE/E| ≈ 3e-4, the same as NBODY6 with ETAI = ETAR = 0.01.
- A run with η = 0.025 stays below 5e-7 until 2 Myr. After that, "the stochastic error from strong encounters in the collapsing core begins to dominate".
- "The final energy error after the core collapse is essentially independent of the chosen η while it changes the elapsed wall-clock time linearly."
- PeTar §7.1.2 (N = 250k through core collapse): an error of ~1e-3, and "the behaviour of the long-term dynamical evolution is not very sensitive to it".

**Lesson.** In core collapse, extra global accuracy costs wall-clock time linearly and does not reduce the final error. The error budget comes from a few violent encounters.

### 2.4 Chaos and reproducibility

**Chaos.** N-body systems are exponentially unstable (Miller 1964).
- Goodman, Heggie & Hut 1993: e-folding time ≈ 1/10 of a crossing time for N ≳ 30.
- Integrating accurately to core collapse would need O(N) decimal places.
- Boekholt & Portegies Zwart 2015 (Brutus, 1411.6671): a converged N = 1024 run to core collapse would need a tolerance ~10⁻²⁰⁶ and ~10⁵ years. The cost of converged solutions grows roughly exponentially with N.

Consequences:
- **No direct N-body run of a cluster is a "true" trajectory.** Over 2400 crossing times, any initial error is amplified by e^{24000}.
- **What we can demand is statistical correctness.**
  - Portegies Zwart & Boekholt 2014 (1402.6713) show, for 3-body ensembles, that results are statistically indistinguishable from converged ones "so long as the energy during integration is conserved to better than 1/10".
  - The numerical solution shadows *some* true solution (Quinlan & Tremaine 1992, via 2305.11606 §5.4).

**Two practical points for us.**
1. *Ensembles.* Cluster results need several seeds (Rizzuto+2021 used 8, DRAGON-II 1–2). The cost of a science result is therefore (seeds) × (one run).
2. *Bit-reproducibility* is a debugging property, not a physics one. It is still worth a lot: restarts and MPI replicas must give identical runs (our REPLICA_CHECK; the M5 work in `AUDIT_2026-10.md`).
   - A cautionary tale: Wang+2022 §3.6 found that BSE results depend on how often `evolv2` is called.

**What accuracy buys, then.** Not "the" trajectory, but freedom from *systematic* errors:
- secular drifts in long-lived hierarchies (ZLK cycles, PN inspirals), where errors accumulate with sign instead of randomly;
- an energy ledger tight enough to catch bugs.

That is why regularised few-body accuracy is valuable for few-body physics. It is *not* needed for the far field (see §4.6).

### 2.5 Relaxation-time scaling: how the cost grows with N

**The argument** (Hut 1993, astro-ph/9303003; Harfst+2007 §6):
- Direct summation costs O(N²) per crossing time.
- A run through collapse lasts ~ (t_cc/t_rh) × 0.138 N/ln(γN) crossing times.
- Total: **cost ∝ N³ / ln N.**
- With individual time steps it is a little worse in centrally concentrated systems: Makino 1996 §2.3 gives N^{3.3}.
- Neighbour schemes (Ahmad–Cohen) gain ≳ N^{1/4} (Spurzem & Kamlah 2023 §5.2).

**Scale-freeness (our derivation).** Newtonian gravity has no length scale, so the cost per crossing time at fixed N does not depend on r_h.
- A denser cluster has shorter crossing times, so the cost per Myr is higher, but the number of crossing times to collapse is the same.
- Density matters through three things only:
  - stellar lifetimes against t_cc (runaway collisions need t_cc < 3 Myr);
  - binary hardness, since the same binary is relatively softer in a hotter cluster;
  - collision rates.

### 2.6 GPU scaling limits

**What the literature measures:**
- **R23 §4 (Fig. 6):** good strong scaling up to N_GPU ≈ 40 N/1e6.
  - 1e5 stalls at ~8 GPUs, already inefficient beyond 4.
  - 1e6 scales to 48 GPUs, "not ideal" with more.
  - 5e6 scales to 128, the largest tested.
- **FROST §6.3:** scaling stops when the particles per GPU fall below the number of concurrent threads, about 2.5e4 per GPU (our derivation).
- **NB6++ (1504.03687 §4.2–4.3, §6):** saturates at 8–16 nodes. At 16 nodes, MPI and data movement take about half the time. KS has a compute-to-communication ratio below 100 and is "inefficient" to parallelise.

**Why scaling saturates:**
- The hierarchy's deep levels have few active particles. They are latency-bound, and no number of GPUs helps.
- The far field becomes a smaller fraction of the work as N shrinks.

**Our status:**
- Our MPI design replicates all particles on every rank (`core/mpi_comm.h:53`), as in R23.
- 2 nodes gave no speed-up at 200k, because the small-set path and the subsystems are replicated (`VIPER_NOTES.md:70`).
- We have no multi-node measurement at 1e6.
- FC-III added node-local MPI shared-memory windows.

### 2.7 Coupling stellar and binary-star evolution

**SSE vs BSE.**
- Single-star evolution (SSE; Hurley+2000) changes masses and radii and makes remnants with natal kicks.
- Binary-star evolution (BSE; Hurley+2002, astro-ph/0201220) adds Roche-lobe overflow, common envelopes, tides, magnetic braking and mergers.

**How the codes couple them:**
- **NBODY6:** BSE runs inside the KS treatment of each binary (Kamlah+2022, 2105.08067 §2.1).
  - PeTar §3.6 criticises this: "the code mixes the N-body integration and the stellar evolution in a complex way".
  - Stable triples are turned into a two-body KS "merger" (Aarseth 2001, astro-ph/0110148 §2).
- **FC-II:**
  - SEVN binary evolution only for a < 200 AU and 2.2–600 M☉.
  - Positions are reset from the new (m, a, e), keeping the angles.
  - Binary evolution is switched off inside r_strong = 10 a during strong encounters, "to avoid spurious binary evolution effects from osculating orbital elements".
- **Triples:** no N-body cluster code has triple stellar evolution. MSE (Hamers+2021, 2011.04513) handles arbitrary multiples semi-secularly, and uses MSTAR for its direct mode.
- **R23 §6** names BSE coupling to subsystems with more than two bodies as "the main challenge".

**Hydro analogy:** sub-grid physics coupled by operator splitting. The danger is calling the sub-grid model on osculating, instantaneous orbital elements while the orbit is being perturbed. It is like evaluating a cooling function on a shock-heated cell mid-shock.

### 2.8 Collisions and mergers

What a code has to get right:
- Collisions must be detected at the true pericentre, not at step ends. FC-I App. B3 steps candidates to the actual pericentre; we implemented this in M5.
- Merger products need masses: Glebbeek mass loss in FC-III.
- Merger products need radii and evolution. VMS tracks above 100–600 M☉ are uncertain; Kamlah §5.2 says extrapolating SSE/BSE above 100 M☉ "should be used with caution".
- TDE accretion fractions are a free parameter: 0.1–1 in Rizzuto; 0.5 in DRAGON-II and FC.

**What decides the IMBH mass:**
- FC-III: wind prescriptions change the final IMBH mass by 4× (6000 vs 25,000 M☉).
- **The VMS physics is more uncertain than the dynamics.**

### 2.9 Post-Newtonian terms and spins

- PN terms matter only within ~1000 gravitational radii of BHs. They therefore live inside subsystems.
- **Difficulties:**
  - velocity-dependent forces break the standard symplectic split; this needs the auxiliary-velocity method (Hellström & Mikkola 2010);
  - PN conflicts with slow-down, so FC-I App. A2 sets κ = 1 for PN systems;
  - the 2.5PN dissipation makes the energy books depend on the sink being tracked;
  - spin evolution and spin-dependent recoil are needed for hierarchical mergers (§1.5).
- **Cost:** a PN BH binary at small separation is integrated orbit by orbit. Our audit (`AUDIT_2026-10.md` F2) found ~47 µs per GBS step on PN BH binaries at depths 5–10, which sets the critical path.

---

## 3. What the user's candidate costs on our code

### 3.1 Inputs

**Our production speeds** (raven, 1 node, 4 A100):

| Setup | N | M [M☉] | r_h [pc] | t_cr [Myr] | s per 1e-3 Myr |
|---|---|---|---|---|---|
| R23 Fig. 9, 10 % binaries, defaults | 2e5 | 8.33e4 | 3.1 | 0.282 | 12.5 |
| Same, c.o.m. fly-by velocity | 2e5 | 8.33e4 | 3.1 | 0.282 | 5.4 |
| R23 Fig. 7 mass–size IC, 10 % binaries, first interval | 1e6 | 4.14e5 | 6.23 | 0.360 | 36 (R23: 57.6) |

- t_cr is the code's printed (r_h³/GM)^{1/2}.
- The 1e6 row is from the Fig. 7 reproduction (`fig7-2026-10-11/summary.md`).
- There, force, time-step, gradient and energy-check work (all O(N²)) is about 64 % of the time, and subsystems 21 %.

**Cost per crossing time** (pure gravity is scale-free, §2.5):
- 2e5: 0.98 h (defaults), 0.42 h (c.o.m.).
- 1e6: 3.6 h. In N-body units this is **19,300 s per N-body time unit** (31,000 s for the 58 s of R23 Fig. 7).
- 1e7: ×100 from N² at fixed hardware, i.e. 360 h per crossing time on 4 A100.

**Correction factors:**
- **Production speed.** The 1e6 number is a first interval from fresh ICs. R23 §4 calls such numbers "the upper limit"; encounters, multiples and mergers slow later intervals. Our 200k runs show 2–3× (memory note `production-speed-metric`). We apply **×2–3**.
- **Growth through collapse.** Central density rises ×30 in R23 §5.6, and by 10³–10⁶ in deep collapse, adding levels and binary activity. This is not quantified anywhere for HHS-FSI, so it is not included below. (Our expectation is ≤ 2×, because the core holds few stars.)

**Number of crossing times to collapse:** t_rh/t_cr = 0.138 N/ln(γN). With γ = 0.11:
- 2e5: 2760.
- 1e6: 11,900.
- 1e7: 99,000.

With γ = 0.015 these are ~15 % larger.

### 3.2 Results

Our estimate (scratch script, formulas above).

| Case | t_rh | t_cc (0.2 t_rh) | t_cc in crossing times | 1 node, first-interval speed | Remarks |
|---|---|---|---|---|---|
| **2e5 Fig. 9** (r_h 3.1 pc; BHs drive collapse) | 780 Myr | 156 Myr | 550 | **22 d** (defaults), **10 d** (c.o.m.) | ≈ 1.6e5 intervals; realistic 1–2 months on one node |
| 1e6, r_h 6.23 pc (Fig. 7 IC) | 4.3 Gyr | 860 Myr | 2400 | **360 d** | DRAGON-like low density |
| 1e6, r_h 3 pc (typical GC) | 1.2 Gyr | 240 Myr | 2400 | 360 d | same work, shorter physical time |
| 1e6, r_h 1 pc (young massive cluster) | 230 Myr | 46 Myr | 2400 | 360 d | still > 3 Myr: no "classic" runaway unless more concentrated (King W₀ ≳ 9; PZ+2004) |
| 1e6, r_h 0.5 pc | 81 Myr | 16 Myr | 2400 | 360 d | |
| 1e7, r_h 3 pc | 3.1 Gyr | 630 Myr | 20,000 | **≈ 820 yr** | direct summation |
| 1e7, r_h 1 pc | 600 Myr | 120 Myr | 20,000 | ≈ 820 yr | |
| equal mass (15 t_rh), 1e6 | | | 180,000 | ≈ 73 yr | irrelevant for a realistic IMF |

With Gürkan's 0.07–0.1 t_rh instead of 0.2, all collapse costs halve.

**Turning the 1e6 row into a plan (our estimate):**
- **1 node:** 0.5–1 yr at first-interval speed; 1–3 yr at production speed.
- **10 nodes (40 A100):**
  - R23 shows scaling to ~48 GPUs at 1e6, "not ideal". Assume 50–70 % efficiency, i.e. a 5–7× speed-up over 1 node.
  - **About 1.5–6 months** of wall clock.
  - About 70k–170k A100-hours.
- **Cross-check with FROST's projection** (2011.14984 §6.3): 1e6 at η = 0.4 takes ~2 weeks per Gyr on ~40 GPUs, without binaries. That is ~2 weeks for our 860 Myr case, or ~1 month at η = 0.2.
- **Cross-check with DRAGON:** 2000–4000 h per t_rh0 on 16 K20m. Our first-interval speed gives ~43,000 h per t_rh on 4 A100 at the same density, so we are ~10–20× slower per run. A100s are roughly 8× faster than K20s in fp64 (a vendor peak figure; UNVERIFIED for this workload), so **per unit of hardware we are ~100× more expensive than NB6++**.
- That is the price of recomputing full direct forces for every active set at η = 0.2 with |dE/E| ~1e-10 per interval. NB6++ uses Ahmad–Cohen neighbour forces with rare full force updates.

**Turning the 1e7 row into a plan:**
- 400 A100 at 50–70 % efficiency: **~12–16 years** at first-interval speed.
- FROST's projection of "4 weeks per 10 Myr" at 1e7 gives ~5 years for 630 Myr.
- **Either way, infeasible with direct summation.**
- With a far field (§4.6), the O(N²) parts would shrink by ~100–1000× (Taichi: 100× at 1e6). This would bring 1e7 to the 1e6 scale: months on tens to hundreds of GPUs. That would hold only if *every* O(N²) part goes: forces, gradient, time-step sums and the energy check (13 % at 1e6 today).

### 3.3 Time-step dynamic range in these runs

- **Top level:** 1e-3 Myr.
- **Field stars in a collapsed core:** central density up to ~10⁶× the initial value, so ~1000× shorter dynamical times (our estimate). That is about 10 more HHS levels.
- **Binaries:** primordial periods from 0.25 AU (R23 Fig. 7 minimum) to a few days (Moe & Di Stefano log P = 0.2) are 1e2–1e5 orbits per interval, handled by Kepler, secular or SDAR.
- **PN BH binaries before merger:** orbit by orbit with κ = 1, 1e6–1e9 orbits in the final Myr. This is the critical path (`AUDIT_2026-10.md` F2/F3).

### 3.4 What others report, for comparison

| Code | N, binaries | Cost | Hardware | Accuracy |
|---|---|---|---|---|
| Ours (Oct 2026) | 1e6, 10 % | 19,300 s per N-body unit (first interval) | 4 A100 | ~6e-10 after 1 interval |
| R23 Fig. 7 | 1e6, 10–100 % | 31,000 s per N-body unit (58 s per interval) | 4 A100 | η = 0.2 |
| FROST §6.3 | 1e6, no binaries | ~3600 s per N-body unit | 30–40 V100 | ~1e-10 |
| NB6++ (1504.03687 §4.2, §5) | 1e6, 0 % / 5 % | 800 s / 1500–3000 s per N-body unit | 32 / 16 K20X | not reported numerically |
| PeTar (2006.16560 §7.2) | 1e6, 0 % / 100 % | 63 / 121 s per N-body unit | 960 CPU cores | ~1e-3 cumulative in N = 250k core collapse |
| DRAGON | 1e6, 5 % | 2000–4000 h per t_rh0 | 16 K20m | not reported |
| CMC (Monte Carlo; 1601.04227) | 1e6, 10 % to 12 Gyr | < 2 days | 48 cores | statistical |

The last two rows (DRAGON and CMC) were run on the same initial model.

---

## 4. The state of the art: who manages what, by approximating what

### 4.1 NBODY6++GPU (Wang+2015; Kamlah+2022; Spurzem & Kamlah 2023)

**What it is:**
- 4th-order Hermite with block time steps.
- Ahmad–Cohen neighbour scheme: GPU regular forces, CPU irregular forces.
- KS regularisation of pairs, plus chain regularisation.
- SSE/BSE "levels" A–C (Fryer 2012, PPISN, fallback kicks), tidal fields, HDF5 output (Cai+2015, 1506.07591).

**Approximations:**
- The neighbour scheme updates far forces less often.
- Stable triples are frozen into KS "mergers" (Aarseth 2001).
- Relativity is only orbit-averaged Peters decay, with BBHs merged at 100 R_Sch (DRAGON-II III §2.1.1). Full PN with chain exists in NBODY7 (no MPI) and in an experimental version.

**Limits:**
- Strong scaling stops at 8–16 nodes.
- KS is about half the time with binaries.
- DRAGON: "we cannot use a primordial binary fraction close to unity" (1602.00759 §3.2).
- Manual tuning was needed (Wang+2015 §5: KS thresholds raised by hand; restarts at "difficult triple systems").

### 4.2 PeTar / SDAR (Wang+2020ab)

**What it is:**
- P³T: a Barnes–Hut tree with a shared leapfrog step for the long-range force, plus Hermite at short range, joined by a changeover function.
- SDAR (slow-down algorithmic regularisation) for multiples.
- FDPS parallelisation.

**Approximations:**
- **The tree far field**, the main error source (2006.16560 §7.1.1). Cumulative energy errors of ~1e-3 through core collapse are accepted.
- **Slow-down**, which keeps secular evolution but loses orbital phase. The quadruple test's 1:1 resonance became 1:2 (2002.07938 §6).

**Achieved:**
- 1e6 with 100 % binaries at 12× the speed of NB6++ on one desktop GPU.
- N = 1e5 with f_bin = 1 to 12 Gyr (2110.15555).
- 1e7 only as a start-up benchmark on CPUs.

### 4.3 Rantala et al.: FROST, R23, FROST-CLUSTERS (2021–2026)

**What it is:**
- Hierarchical 4th-order forward symplectic integration (HHS-FSI), with direct O(N²) forces on every active set.
- LogH, later SDAR, regularisation with GBS extrapolation.
- PN up to 3.5 and recoil.
- SEVN stellar evolution (FC-I onward), with binary evolution in FC-II.

**Approximations:**
- the Omelyan gradient (FC-I A1);
- slow-down (FC-I A2);
- c.o.m. fly-by velocities and an octree only for time steps (FC-II);
- looser GBS tolerances in production (5e-8, end time 1e-2).

**Achieved:** the only million-body runs with full PN few-body physics and realistic multiplicity, but over 10–50 Myr.

**Limits:**
- "The maximum number of simulation particles [...] is approximately a few million" (R23 §7).
- No FMM yet (§6).

### 4.4 Monte Carlo codes: CMC and MOCCA (the contrast)

**Method.** Hénon's orbit-averaged Monte Carlo:
- spherical symmetry;
- relaxation as statistical two-body kicks;
- strong encounters integrated with Fewbody (2.5PN for BHs in CMC);
- BSE/COSMIC.

**Achieved:**
- CMC: N = 1e8 Plummer to core collapse (2106.02643 §4.1); 1e6 clusters to 12 Gyr in about a day.
- MOCCA: "several hours for N = 2×10⁶" (1112.6246 §1).
- Agreement with the 1e6 direct model was "excellent" overall (1601.04227).

**Approximations** (2106.02643 §2.10):
- molecular chaos, so no resonant relaxation;
- validity only for N ≳ 1e5 with a realistic IMF;
- virial equilibrium, so no violent relaxation;
- encounters capped at 1 CPU-minute.

**Role for us:** Monte Carlo is the *fast survey tool*. Direct N-body calibrates it where its assumptions fail:
- small-N cores;
- IMBHs with bound cusps;
- non-spherical assembly (FC-I);
- long-lived hierarchies.

**A direct code earns its cost only there.**

### 4.5 Other approaches

- **φGPU / HiGPUs.** Softened or plain Hermite. Up to 1536 GPUs and ~6–8e6 bodies, but without regularisation or stellar physics (1312.1789, 1207.2367).
- **Taichi** (Zhu 2017; Mukherjee+2021, 2012.02207; 2205.12289). HHS + FMM, later with FSI + AR-chain. 100× faster than direct at 1e6, at a cumulative |dE/E| ~1e-4. This is the prior art for M8.
- **KETJU** (GADGET + MSTAR). Regularised regions inside softened galaxy simulations (Lahén+2025).

### 4.6 The pattern: what everyone approximates

| Approximation | Who | What it costs in accuracy |
|---|---|---|
| Far-field force: tree/FMM, or neighbour scheme with infrequent far updates | PeTar, Taichi, NB6++ (AC) | energy errors of ~1e-4 to 1e-3 over core collapse; statistically harmless (§2.3–2.4) |
| Stable hierarchies frozen, secular, or slowed down | NB6 ("mergers"), R23 (secular), PeTar/FC (SDAR) | orbital phase; resonances (SDAR §6) |
| Orbit-averaged GW decay instead of PN in the equations of motion | NB6++, DRAGON-II | GW captures and eccentric mergers missed |
| Binary evolution switched off during strong encounters | FC-II (r_strong) | consistency over small timescales |
| Scaled-down N, inflated capture radii | Panamarev, Baumgardt+2018 | TDE and EMRI rates need rescaling |
| Monte Carlo relaxation | CMC, MOCCA | no geometry, no small-N cores |

**Our inference.** The defensible design for a 1e6–1e7 code that keeps regularised few-body accuracy puts its accuracy where errors are systematic:
- exact, unsoftened near field;
- regularised few-body integration with PN;
- exact energy books.

It then accepts a far field at the level that keeps core-collapse statistics unbiased. METHODS_SURVEY §I quotes Dehnen 2014: ε ≲ 1e-8 to stay at today's 1e-10 per interval.

**What the target should be.** For core-collapse science the target need not be today's 1e-10 per interval. It should be:
- statistical equivalence with the direct run on an ensemble, e.g. Lagrangian radii, t_cc, binary binding-energy distributions and escaper counts within seed scatter;
- the subsystem ledger stays exact.

This is a decision for the user. It loosens our standing energy rule ("no |dE/E| loss") for the far field only. Experiment I-1 (synthetic force errors) should measure it before any FMM coding.

---

## 5. Gap list for our code

**Code state** (commit 5b923f7, main): we checked the `src/` tree, `config.txt`, `AUDIT_2026-10.md`, `METHODS_SURVEY.md` and `VIPER_NOTES.md`.

### 5.1 Cross-cutting gaps (block every long run)

| # | Gap | Evidence | Needed for |
|---|---|---|---|
| G1 | **No restart from disk.** `collisional/checkpoint.{h,cpp}` holds the in-memory rollback for adaptive restarts only. `NBODY_DUMP` writes a diagnostic dump (pid, m, x, v, subsystem id) without the subsystem or SSE state, and it cannot be read back. | `main.cpp:464-484` | everything longer than one 24 h job. Must be bit-identical on resume (test: run N steps, or N/2 + restart + N/2, compare). Must include subsystem/LogH/SDAR state, RNG state, SSE state, merger tree and energy books. |
| G2 | **No science output.** No snapshots (FC-I §3.1 uses GADGET-like HDF5). No Lagrangian radii, density centre, core radius or central density. No binary census (a, e, E_b distribution); no escaper or collision tables beyond `merger_tree.dat` and `escaper_log.dat`. | `io/diagnostics.h` | any comparison with R23 Figs. 13–14 or DRAGON-II. Data volume: DRAGON-II wrote 35 TB for 19 runs. FC-I §4.7 notes that binary–single encounters last ~4e-5 Myr, below any snapshot cadence, so event logs are needed. |
| G3 | **No external tidal field** (isolated clusters only; escapers are removed at 100 r_h). | grep of `src/` | GCs to a Hubble time (§1.6); comparison with DRAGON, DRAGON-II, Kamlah+2022 (all tidally limited). A point-mass or log-potential tidal acceleration plus the matching energy term is small work. |
| G4 | **Multi-node scaling unproven.** Replicated particles with allgather; 2 nodes gave no gain at 200k; never measured at 1e6. FC-III added node-local shared-memory windows. | `core/mpi_comm.h:53`, `VIPER_NOTES.md:70` | 1e6 in months, not years (§3.2). Target: 40 N/1e6 GPUs (R23 §4). |
| G5 | **O(N²) outside the force.** Collisional time-step sums (`AUDIT` F7: the tree prunes only the min-search), gradient (cross and self), and the energy check (13 % at 1e6). | `fig7` summary; `AUDIT` F7 | 1e7, together with M8. |
| G6 | **Production speed at 1e4–1e5.** 3–20× slower than the published timings (R23 Fig. 7 reproduction). The cause is subsystem overhead; SDAR is now in but off by default. | `fig7-2026-10-11/summary.md` | the validation ladder (small N, many seeds). |

### 5.2 Per frontier problem

**1e6 cluster through core collapse (§1.1)**
- Needs G1–G4.
- SDAR as default after the A/B. It removes the orbit-by-orbit stalls of F3.
- Lagrangian-radius and core diagnostics (G2).
- Ensemble tooling: seeds, job chains, automatic resubmission.
- With f_bin ≥ 50 %: subsystem throughput (G6).

**Young massive clusters, runaway, VMS/IMBH (§1.2)**
- Collisions inside integrations: done (M5).
- Glebbeek mass loss and TDE accretion fraction (M5 leftovers).
- **MSTAR for more than 8 bodies.** `LOGH_MAX_BODIES = 8`; larger groups "stay field particles, integrated by the FSI directly" (`subsystem.cpp:515`). An IMBH with a bound cusp is exactly such a group (M4).
- **Runtime stellar evolution.** `SSE_ENABLED = false` by default. The natal kicks are deterministic (`runtime_sse.cpp:74`). No per-star t_next in the time step (FC-I B2). No VMS tracks or winds above 100–600 M☉ (FC-III).
- Multiplicity initial conditions (below).

**Nuclear clusters (§1.3)**
- M8 (FMM) and M4 (MSTAR around the SMBH).
- An external or live SMBH with loss-cone bookkeeping.
- 1e7–1e8 particles, so also G4 and G5.
- Long term.

**100 % multiplicity (§1.4)**
- Initial conditions have binaries only (`BINARY_POP = flatq_massive | mass_rank`).
- Needed:
  - no triples or quadruples in clusters;
  - primary-mass-dependent fractions (Offner Table 1 / FC-II Table 3);
  - Moe & Di Stefano correlated P–q–e;
  - stability-checked hierarchies (Vynatheya+2022 / Mardling–Aarseth).
- **No binary stellar evolution at all.** Options: SEVN binary module (FC-II), BSE (NB6++, PeTar), or MSE for multiples. R23 §6 calls subsystem coupling "the main challenge".

**GW captures and hierarchical mergers (§1.5)**
- PN up to 3.5 is in the subsystems, with pairwise EIH-type coefficients for groups up to 8.
- Remnant mass and spin: fits are in. **But the kick ignores spins:** Campanelli-type, non-spinning, in a random in-plane direction (`mergers.cpp:106-130`).
- Spin evolution is only partly wired.
- Needed:
  - Zlochower–Lousto 2015 with spins (M4/M5);
  - spins carried through mergers;
  - birth-spin options, which decide retention (Rodriguez+2019).

**Globular clusters to a Hubble time (§1.6)**
- G1, G2, G3.
- BSE.
- Long-run robustness: months of runtime and hundreds of restarts.

**Dwarf galaxies (§1.7)**
- A different architecture: regularised regions inside a softened hydro code (KETJU-style, Lahén+2025).
- Our `tree_forces.cpp` (collisionless, softened) is a starting point, but this is a separate project.

### 5.3 What we already have that is ahead of most codes

- Full PN (up to 3.5) inside regularised subsystems, with exact energy books.
- Collisions decided at the true pericentre inside the integration (FC-I B3).
- SDAR.
- Exact tree time-step and neighbour searches (bit-identical).
- Kokkos portability across A100 and MI300A.

NB6++ (the standard version) and DRAGON-II lack in-equation-of-motion PN.

---

## 6. Recommendations

### 6.1 Flagship problems

**Flagship A (first; fits our strengths): an evolved, BH-rich 1e6 cluster with realistic low-mass multiplicity, through BH-driven core collapse into balanced evolution, with full PN BH dynamics.**

- **Setup.** Start from a 1 Gyr evolved population, as in R23 Fig. 9.
  - Massive stars are already gone, so binary stellar evolution matters much less: low-mass main-sequence binaries rarely interact, except the closest.
  - BHs drive the collapse (DRAGON §4.6; Breen & Heggie 2013).
  - Multiplicity follows Moe & Di Stefano for solar types: ~50 % companions, 10 % triples and quadruples.
- **Science.**
  - GW capture fraction and eccentric-merger fraction in a direct, unsoftened N-body with PN in the equations of motion (versus CMC's Fewbody: Samsing 2018; Rodriguez+2018).
  - Hierarchical mergers with spin-dependent recoil.
  - Binary energy generation and core structure with high binary fractions.
- **Novelty.** No direct run of ≳ 6e5 stars has more than ~33 % of stars in binaries, and the 1e6 ones stop at ~0.2 Gyr, and none has in-equation PN for all BHs (DRAGON-II used Peters plus post-processing).
- **Cost** (§3.2):
  - 0.1–0.2 t_rh at r_h = 3 pc means 120–240 Myr.
  - ~1–3 months on 10 nodes, after G1–G4.
  - A seed ensemble multiplies that.
- **Stepping stone:** the same at 2e5 (Fig. 9 IC, ~1–2 months on one node). This is publishable on its own if multiplicity is realistic.

**Flagship B (second; head-to-head with FC-II/III): a dense young cluster (1e5–1e6 stars) with full massive-star multiplicity, through runaway collisions and IMBH formation.**

- **Novelty relative to FC-II:**
  - quadruples;
  - binary evolution of the inner binaries of triples;
  - possibly MSE-style multiple evolution, which nobody has.
- **Needs:** MSTAR, BSE/SEVN, VMS physics, multiplicity ICs.
- **Physical time:** only 3–10 Myr.
- **Risk:** the result is dominated by VMS wind and collision prescriptions (FC-III's 4× spread), not by dynamics. We should only do this with a clear physics angle.

**Flagship C (moonshot): a 1e7-star cluster through core collapse.**

- **Novelty:** nobody has done it, by any direct method.
- **Requirements:**
  - M8 (FMM) and removal of every O(N²) part;
  - multi-node scaling to hundreds of GPUs;
  - acceptance of a statistically tested far-field tolerance (§4.6).
- **Plan:** start with I-1 (force-error injection) and an FMM-vs-direct ensemble comparison at 1e6. Commit to 1e7 only after that.

### 6.2 Validation ladder

Each rung tests the physics of the flagships, compares with published numbers, and costs little.

| Rung | Run | Compare with | Tests | Our estimated cost |
|---|---|---|---|---|
| 0 | Existing paper suite: FROST Kepler/planets; R23 Figs. 8–11 | R23 | integrator, binaries | done or in progress (M2) |
| 1 | Equal-mass Plummer, N = 16k and 32k, to and past core collapse | t_cc = 15.2 ± 0.1 t_rh (PZM02); gravothermal oscillations at 32k (Makino 1996; ⟨r_c⟩ ∝ N^{-1/3}) | relaxation, dynamically formed binaries as the only heat source, long-run stability | 16k: t_rh ≈ 300 t_cr, 15 t_rh ≈ 4500 t_cr. Small N is latency-bound; hours to days per t_cr is UNVERIFIED for our code. Measure the cost per t_cr at 16k first. |
| 2 | Mass spectrum (Salpeter/Kroupa), N = 16k–64k, Plummer and King W₀ = 6 | t_cc/t_rh = 0.07–0.2 (Gürkan Table 2; PZM02 0.19 ± 0.08); Fujii & PZ 2014 m_max scaling | mass segregation, early collapse | ~100–200 t_cr per run; several seeds |
| 3 | R23 §5.6: N = 20k, 10 % binaries, r_h = 0.5 pc, Kroupa, 4 Myr | |ΔE/E| ≈ 3e-4; central density ×30; Lagrangian radii and binary a, e (Figs. 13–15), plus NBODY6 | primordial binaries in collapse, stochastic error | ~70 t_cr; already in plan M2 |
| 4 | Rizzuto+2021-type dense cluster: N = 1.1e5, 10 % binaries, R_h = 0.6 pc, Z = 2e-4, ~15 Myr; or an FC-III isolated cluster at matching Σ_h | VMS ~400 M☉, IMBH in ≤ 3/8 realisations; FC-III fitting formulae | collisions, runtime SSE, PN BHs | needs runtime SSE (M6); 8 seeds |
| 5 | DRAGON-II-like cluster: 1.2e5–3e5, R_h = 0.47–0.8 pc, 10–33 % binaries, tidal field | BH retention, VMS 365 M☉, IMBH channels (DRAGON-II II) | G3 tidal field, G1 restarts over weeks | weeks on one node |
| 6 | R23 Fig. 9 cluster (2e5) to BH core collapse, ~150 Myr; then with Moe & Di Stefano multiplicity | DRAGON BH-collapse behaviour (§4.6); CMC run on the same IC | everything in Flagship A at 1/5 scale | 1–2 months, one node |
| 7 | 1e6 Flagship A | CMC on the same IC; DRAGON | scaling, restarts, output | 1–3 months, 10 nodes |

**Cross-code check.** On rungs 6–7, run CMC (public) on the same initial model. Monte Carlo is where direct N-body earns its keep: deviations in the core, BH subsystems and hierarchies (§4.4).

### 6.3 Open, publishable targets

Our assessment from the literature above.

1. **1e6 with ≥ 50–100 % (observed) multiplicity through core collapse with in-equation PN BHs.**
   - Not done: DRAGON had 5 %; DRAGON-II's 1e6 runs had f_b = 0.05 (≈ 10 % of stars in binaries) and reached 0.2 Gyr; PeTar reached 100 % only at 1e5; FC-II had realistic multiplicity but only 10 Myr in assembling clusters.
   - Output: GW capture, eccentric and hierarchical rates from direct N-body as a cross-check of Monte Carlo predictions.
2. **A direct-vs-Monte-Carlo calibration at high binary fraction.**
   - Rodriguez+2016 did 5 % binaries; a 50–100 % version has not been done. This is cheap at 2e5.
3. **Accuracy requirements for collisional cluster statistics.**
   - A controlled ensemble study: regularised direct (R23-type) versus tree far-field (PeTar-type) accuracy, on the same core-collapse ICs, with seeds.
   - It extends Portegies Zwart & Boekholt 2014 from 3 bodies to clusters.
   - It also decides our FMM tolerance. A methods paper.
4. **Quadruples and triple binary evolution in young-cluster runaways.** No code has this (FC-II: triples, no triple stellar evolution).
5. **1e7 through core collapse** (Flagship C). A first, by any direct method, if M8 works.

### 6.4 Suggested order of work (our proposal; fits the plan)

1. **G1 restart files, G2 diagnostics (Lagrangian radii, core, binary census), G3 tidal field.** These are small, and every long run needs them.
2. **Finish M2 and M3** (SDAR default after the A/B), then rungs 1–3 of the ladder. They need nothing new and give the first published-number comparisons through core collapse.
3. **Multiplicity ICs** (Offner fractions, Moe & Di Stefano P–q–e, triples, quadruples) and **spin-dependent recoil**. Together these are what Flagship A needs beyond step 1.
4. **G4 multi-node scaling at 1e6** (shared windows, fused allgathers), then rung 6, then Flagship A.
5. **Experiment I-1 and the accuracy study** (target 3 above), before any M8 FMM code.
6. Runtime SSE upgrade (M6), MSTAR (M4) and BSE/SEVN, for Flagship B.

---

## Appendix: sources added in this audit

- **`frontier_runs/`**
  - 1602.00759 DRAGON; 2307.04805/06/07 DRAGON-II; 1504.03687 NB6++
  - 2006.16560 PeTar; 2002.07938 SDAR; 2101.09283 and 2110.15555 PeTar applications; 2109.04592
  - 2105.08067 Kamlah+2022; astro-ph_0402622 Portegies Zwart+2004; 2008.09571 Rizzuto+2021
  - 2406.06772 Fujii+2024; 1804.04682 Gieles+2018; astro-ph_0211471 Baumgardt & Makino 2003
  - 1409.5597 Heggie 2014; astro-ph_0507239 Hurley+2005; 1010.2210 Zonoozi+2011
  - 1805.02153 Panamarev+2019; 1601.04227 Rodriguez+2016; 2305.11606 Spurzem & Kamlah 2023
- **`cluster_theory/`**
  - astro-ph_0308449 Gürkan+2004; astro-ph_0201055 PZM02; 1304.1550 Fujii & PZ 2014
  - astro-ph_9608160 Makino 1996; 1304.3401 Breen & Heggie 2013; 1101.1821 Gieles+2011
  - 1402.6713 PZ & Boekholt 2014; 1411.6671 Brutus; 2002.04029 Boekholt+2020
  - 2106.02643 CMC; 1112.6246 MOCCA; 1712.04937 and 1906.10260 Rodriguez+2018/2019; 1711.07452 Samsing 2018
  - 1410.7772 Stone & Metzger 2016; 1205.5240 Amaro-Seoane; astro-ph_0608125 Harfst+2007; 1701.03818 Baumgardt+2018
  - 1911.05093 GRIFFIN; 2410.01891 and 2504.18620 Lahén+2025; astro-ph_9303003 Hut 1993
- **`multiplicity/`:** 1207.6397 Sana+2012; 2203.10066 Offner+2023. Moe & Di Stefano 2017 is already in `binaries/1606.05347.pdf`.
- **`binary_evolution/`:** astro-ph_0201220 BSE; 1711.03556 MOBSE; 2211.11774 SEVN; 1809.04605 Spera+2019; 1911.00903 COSMIC; 1612.06172 TRES; 2011.04513 MSE; 2105.08067.
- **`gpu_scaling/`:** 1207.2367 HiGPUs; 1312.1789 φGPU; 1101.5504 P³T; 1506.04553 GPU-P³T; 1601.03138 FDPS; 1506.07591 Cai+2015 (HDF5 output).
- **`subsystems/`:** astro-ph_0110148 Aarseth 2001.

**Corrections to common numbers found during the audit:**
- The GHH93 e-folding time is ~1/10 t_cr, not 1/20.
- The mass-spectrum γ is 0.015 (Giersz & Heggie 1996/97), not 0.02.
- Gürkan's saturated t_cc is 0.15 t_rc(0), the *central* relaxation time. Relative to t_rh(0) it is 0.07–0.15 for Plummer or King W₀ ≤ 4.
- DRAGON ran on Hydra (MPCDF, 16 K20m per run), not Laohu.
- Rodriguez+2016 compared CMC with the NB6++ million-body model that later became known as DRAGON.
- The single-mass "15.7 t_rh" could not be found. The sources give 15–17: PZM02 15.2 ± 0.1; CMC 16.8.

---

## Update (2026-10-11): talks and recent papers on 1e7-star runs

Search of talks, slides and 2024–2026 papers. **No completed 1e7 direct or hybrid production run exists.**

| Source | What it says (verified) |
|---|---|
| Spurzem, IAUS 398 / MODEST-25 slides (June 2025), wwwstaff.ari.uni-heidelberg.de/spurzem/talks/spurzem-modest25.pdf | "NBODY6++GPU with up to 16M particles (Benchmarks on raven at MPCDF)"; "The Future – DRAGON III 1m – 8m (16m?)"; PeTar and the Rantala et al. code named as "New competition" |
| Li & Spurzem, NbodyCP, arXiv:2511.06181 | short test runs up to 16M stars on 16 Raven nodes (64 A100), "near ideal" speed-up; "simulations of several Gyrs will take months" |
| Wu et al., DRAGON-III, arXiv:2510.03933 (MODEST-25) | 1M/4M/10M GC models for 1–10 Gyr "ongoing and planned"; HIP port for LUMI; results so far reach ~0.1 Gyr |
| Bianchini et al., ROLLIN' I, arXiv:2603.26195 | 1.5M stars to 12 Gyr with NBODY6++GPU, ~400 days on 12–48 V100: the largest full-lifetime direct run |
| Mai, Kremer & Kıroğlu, arXiv:2510.21916 | CMC Monte Carlo 1e7 to ~12 Gyr |
| Rantala talks (KICC 2024, LISA AWG 2024, MODEST-25, Stockholm 2026) | up to 2.35M stars, binaries + triples, Nvidia/AMD; no 1e7 or FMM plan mentioned |

Consequence for Flagship C: the competition for a 1e7 direct run is DRAGON-III (NBODY6++GPU, neighbour scheme, ~8–16M planned).
A 1e7 run at regularised direct-summation accuracy with in-equation PN and high multiplicity remains open.
