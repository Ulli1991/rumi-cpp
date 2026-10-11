# Parallel-in-time, asynchronous and task-based methods for cpp-nbody: a literature review

Date: 2026-10-11. Sections 5.8 (CUDA-aware MPI on raven) and 5.9 (paradigms beyond MPI) were added on request.
Code state: commit 9b8dc35 (line numbers below refer to it). Research only; nothing in the code was
changed. Papers are in `/raven/u/uli/papers/async_parallel_time/` (~110 PDFs; file names are given as `[file]`).
Every number quoted from a paper carries its section/figure; items marked **UNVERIFIED** could not be checked against
the source. Our own measurements come from `AUDIT_2026-10.md` §G (200k, 10 % binaries, 4x A100 on one raven node).

Readers: the two design agents (deep-level batching + task parallelism; GPU idle time) and the user. Hydro analogies
are given where they fit: HHS levels are time bins, connected components are "islands" of active cells, the parent
kick is the bin synchronisation point.

---

## 0. Summary

**What limits us, in one paragraph.** A 200k step (11.3 s on 4 GPUs) is ~2000 calls of `hhs_fsi_recursive`. The few
calls at depths 0-4 carry the real arithmetic. The ~2000 calls below carry 2-6000 particles each and are dominated by a
*fixed cost per call*: 60-90 kernel launches and 25-40 host syncs, 0.6-2 ms each, whatever n is. The GPU is busy
34-38 % of the step. Every rank does this replicated work, and the ranks finish it at different times, so 2/3 of the
MPI time is waiting. This is the textbook "multistepping pathology" every individual-time-step code reports
(ChaNGa: multistep gain falls from 3.3x on 1 core to 1.4-1.6x on 32-64 cores, Jetley+2008 Table 3; ChaNGa at 128K cores
48 % efficiency multistep vs 80 % single-step, Menon+2015 §8.5; SWIFT: small steps "can actually dominate the overall
simulation run time", Schaller+2024 §2.4).

**Three conclusions.**

1. **Parallel-in-time across the cluster does not work, and the literature says so clearly.** Parareal/PFASST/MGRIT
   converge over windows of O(1) Lyapunov times for chaotic systems, independent of the step (Gander & Hairer 2014,
   Thm 3). Even the solar system defeats it (Jiménez-Pérez & Laskar 2011, p. 17). Our Lipschitz constant is set by the
   hardest binary, not by the 0.1 t_cr ensemble Lyapunov time, so it is far worse. Niche use only: a straggler
   near-Kepler subsystem with a Kepler coarse solver (2-10x on that one subsystem, not bit-identical).
2. **The asynchrony we need is already in the HHS-FSI splitting; we just do not use it.** Between the three S-F kicks
   of one call, the slow set S and the fast set F are two *closed, independent* Hamiltonian systems, and so is each
   connected component of F (Pelupessy+2012 §2.1; Jänes+2014 Eqs. 26-31). In PDES terms the lookahead is the whole
   pivot step and is known in advance, so *conservative* scheduling is exact: no rollback, no change of arithmetic,
   bit-identical results, momentum and time symmetry untouched. The global barrier per level is an artifact of a
   sequential recursion with global mutable state (`g_subsystem_mgr`, one Kokkos stream, the profiler).
3. **The biggest measured wins for "many tiny kernels, GPU starved" in other codes are: remove host syncs, aggregate
   work, and run the smallest pieces on the CPU.** Octo-Tiger: ~10x from GPU work aggregation over naive per-task
   launches, 1.5x (A100) and 4.5-5.2x (MI100) from explicit kernel fusion over multi-stream concurrency (Daiß+2022
   §VI-C). Legion: 4.9x average strong-scaling gain from trace replay (Lee+2018). CUDA: per-kernel cost 9.6 µs with a
   sync after every launch, 3.8 µs with one sync per step, 3.4 µs in a graph (Gray 2019). NBODY6++GPU, PeTar, FROST and
   ChaNGa-GPU all keep the small, irregular work on CPU cores (FROST §5.2: serial CPU code below "a few hundred"
   particles; ChaNGa-GPU: GPU pays only above 250-500 targets per processor, Jetley+2010 Fig. 8).

**Ranked ideas** (detail in §6; gains are estimates at 200k unless stated, from the audit's per-depth costs):

| # | idea | where | est. gain | effort | risk | bit-identical? |
|---|---|---|---|---|---|---|
| 1 | Cut the fixed cost per call: no syncs/allocs/scalar readbacks inside the recursion, one control-block readback per call, events instead of fences | `hhs_fsi_recursive`, `partition_by_pivot`, `forces.cpp:429`, `detect_from_pairs`, `profiling.h:108` | 1.2-2 s (10-18 %); up to ~2x at 1e5 f_bin=1 | medium | low | yes |
| 2 | Host-resident deep subtrees: a connected component below ~64-256 particles runs its *whole* sub-recursion on CPU cores, one copy in and out, many components on many cores (zero-copy on MI300A) | `hhs_cc_step` → a fixed `hhs_fsi_recursive_cpu` | most of depths >= 8 (2.3 s → ~0.3 s); 47 % share at 1e5 f_bin=1 | high | medium | no (host vs device FP); deterministic and rank-count independent; 3-seed A/B |
| 3 | Run S and F, and the components of F, concurrently (conservative PDES with exact lookahead): streams/`partition_space` + task-private subsystem state, or a breadth-first batched recursion | `hhs_fsi_recursive` steps 2‖3, 5‖6; `hhs_cc_step::recurse_all` | 1-3 s (overlap of replicated F work with S work; GPU fill) | high | medium (refactor) | yes, if each task keeps its own arithmetic |
| 4 | Fix the gather: host staging now; A/B forced ring / hand-written Isend-Irecv; then padded `ncclAllGather`/RCCL (stream-ordered, graph-capturable) or UCC inside MPI; fuse paired collectives (§5.8) | `mpi_comm.cpp:411-680`, `integrator.cpp:1885, 1958, 1986` | 0.5-0.9 s (MPI collective cost 0.9-1.15 s/step) | low-medium | low | yes |
| 5 | Give ranks *different* replicated work: distribute subsystems (cost-ordered) and CC components over ranks, books summed in index order; this attacks rank skew (2/3 of MPI time) and is the only route to strong scaling of the non-scaling 8 s | `advance_subsystems_in_range` (DISTRIBUTE_SUBSYSTEMS), `hhs_cc_step` | 0.4-1 s at 4 GPUs; grows with rank count | medium | low-medium | yes if each item is computed by one owner with the same code |
| 6 | Longest-predicted-first ordering of host subsystem integrations (LPT, Graham 1969: <= 4/3 of optimal vs <= 2 unsorted) | `logh_batch.cpp:586` | tail of each subsystem phase; measure `max(gbs_per_sub)` vs mean first | low | none | yes |
| 7 | Few-sink kernel shape: several threads per sink (Sapporo2 "2D blocks") and a higher chunk cap when `ni` is small | `n2_driver.h` `chunk_count`, `run_tiled` | mid levels (n = 300-6000, 1.7 s); measure | medium | low | no (summation split) → 3 seeds |
| 8 | Fixed-topology graphs per "level template" (Kokkos::Graph) with device-side counts, after #1 | per-phase kernel sequence | 10-50 % of remaining launch cost | medium | medium (Kokkos Graph is experimental, no conditionals) | yes |
| 9 | Check that the viper Kokkos is built for the APU (`Kokkos_ARCH_AMD_GFX942_APU`), else the host small-loop path is compiled out | build | unknown; possibly large on viper | trivial | none | — |
| — | Ensembles for throughput at N <= 2e5: 4 seeds x 1 GPU instead of 1 seed x 4 GPUs (§5.9) | job scripts | 2x throughput per node (22.0 vs 11.3 s/step) | trivial | none | — |
| 10 | Parallel GBS for stragglers: extrapolation columns in parallel (parallelism across the method), and only if a single subsystem is the critical path, Parareal with a Kepler coarse step | `logh_batch.cpp`, `gbs.cpp` | removes a straggler tail; measure first | medium-high | medium | no |

**What not to do** (§7): Parareal/PFASST/MGRIT on the cluster; optimistic PDES (Time Warp); per-pair asynchronous
variational integrators; a general AMT rewrite (Charm++, HPX, Legion: their minimum efficient task is ~100 µs, ours
are 1-10 µs); CUDA dynamic parallelism; Kokkos tasking (deprecated in 4.5); distributing tiny levels over MPI ranks;
one megakernel that also contains GBS/PN/merger logic; per-step recapture of CUDA graphs; managed memory on A100.

---

## 1. What the HHS-FSI splitting really requires (the basis for everything else)

One call of `hhs_fsi_recursive` on a set P = S ∪ F (`integrator.cpp:2290-2440`) does, with K the S-F interaction kick:

```
1  K_SF(dt/6)            reads S and F           <- join
2  S: fsi_step(dt/2)     touches only S          }  disjoint variables: exp(dt/2 H_S) and exp(dt/2 H_F)
3  F: recurse(dt/2)      touches only F          }  commute exactly
4  K~_SF(2dt/3)          reads S and F           <- join
5  F: recurse(dt/2)      touches only F          }  again independent
6  S: fsi_step(dt/2)     touches only S          }
7  K_SF(dt/6)            reads S and F           <- join
```

H_S = T_S + V_SS and H_F = T_F + V_FF share no coordinates (V_SF is "held" and applied only in the kicks; Pelupessy,
Jänes & Portegies Zwart 2012, §2.1, Eqs. 5-6 [1205.5668]; BIFROST §2.2: inter-level interactions "only include
accelerations from faster hierarchy levels" and the scheme is "manifestly momentum-conserving" [2210.02472]). So steps
2 and 3 (and 5 and 6) can execute in any order or at the same time, with *identical* floating-point results, as long
as neither touches shared state. With HHS_CC (Jänes, Pelupessy & Portegies Zwart 2014, Eq. 21,
H = H_C + H_R + V_CC + V_CR; components "evolved independently", inter-component and component-remainder kicks only
at the parent level, Eqs. 26-31, Fig. 2 [1407.7105]), each component C_i of F, with its whole sub-hierarchy, is an
independent closed system between the parent kicks; `hhs_cc_step` already computes the S-F and inter-component kicks
at the parent level (`integrator.cpp:1893-1913`) and then loops over the components sequentially
(`recurse_all`, `:1918-1932`).

**Synchronisation actually required:** three joins per call (the kicks), plus a join of all component tasks before the
parent's next kick. Nothing else. In particular a deep component does not need to wait for other components at the
same depth, nor for the slow set's work at its own level.

**What blocks this in the code today** (the refactor list for idea #3):
- `g_subsystem_mgr.active` is global and rewritten by detection in the recursion; the code saves and restores it around
  every recursive call (`:2312`, `:2337-2343`, `:2390-2396`). Concurrent tasks need task-private subsystem lists
  (S keeps the parent's list, each F task gets its own) and a deterministic merge at the join.
- `pid_to_idx` is global and rebuilt after partitions (`rebuild_pid_to_idx`); with disjoint index ranges a per-task map
  or a range-local map works.
- Mergers compact the particle array as a suffix shift (`compact_remove_zero_mass`, `:2181`). F is a suffix of P, so S
  is unaffected, but component i's compaction shifts components i+1..k. Concurrent components must compact *within
  their own range* (move the dead particle to the end of the range, shrink the count) and the parent does one global
  compaction at the join; the final particle order equals the sequential one.
- Shared static scratch: `cc_cid`/`cc_snx..z` (`:1878`), `n2_detail::partial_buffer()` (`n2_driver.h:37`), the MPI
  send/receive buffers, `g_logh_batch`. Each concurrent task needs its own (pool of preallocated buffers, as Octo-Tiger
  does, Daiß+2025 §5.3).
- One Kokkos execution-space instance (one stream) plus fences everywhere: concurrency on the GPU needs per-task
  instances (`Kokkos::Experimental::partition_space`) and fences on the instance, not global `Kokkos::fence()`
  (also inside `small_for`).
- The profiler's `cur_depth` and counters are global (`DepthScope`, `g_hhs_counters`).
- Energy books: per-task partial books summed in a fixed order at the join (already the pattern in
  `advance_subsystems_in_range`, `:621-628`).

**PDES view.** Conservative PDES (Chandy & Misra 1979, DOI 10.1109/TSE.1979.230182) fails in practice when lookahead is
poor; Tang+2005 found a conservative event-driven PIC code "performed very poorly, due to poor lookahead", and Miller &
Luding 2004 (hard spheres, optimistic with rollback, ~10x at P = 128, speed-up ∝ P^0.5, Figs. 6-7 [physics_0302002])
note that in the conservative case "one process is operating and all others are idling". HHS is the opposite case:
the lookahead (next S-F kick) is a full pivot step, known exactly. So the right model is a *static fork-join task tree*,
not Time Warp. I found no published PDES application to individual-time-step gravitational N-body (searched; absence
of evidence only).

---

## 2. Parallel-in-time integration

### 2.1 Parareal, PFASST, MGRIT, RIDC, SDC: the ideas

- **Parareal** (Lions, Maday & Turinici 2001, DOI 10.1016/S0764-4442(00)01793-6). Cut [0, T] into N slices. A cheap
  coarse propagator G runs serially; an accurate fine propagator F runs on all slices at once from the previous
  iterate's start values; update u_{n+1}^k = F(u_n^{k-1}) + G(u_n^k) − G(u_n^{k-1}). After k = N iterations the result
  equals the serial F run, with no speed-up. Speed-up is at most N/K and at most cost(F)/cost(G). Hydro analogy: a
  multigrid in time whose coarse level is a low-resolution run.
- **SDC** (Dutt, Greengard & Rokhlin 2000, DOI 10.1023/A:1022338906936): iterate a low-order sweep over collocation
  nodes to raise the order. **PFASST** (Emmett & Minion 2012, DOI 10.2140/camcos.2012.7.105) interleaves SDC sweeps
  with Parareal-like coarse corrections across slices. **MGRIT** (Falgout+2014, DOI 10.1137/130944230) is full
  multigrid in time. **RIDC** (Christlieb, Macdonald & Ong 2010, DOI 10.1137/09075740X; library Ong+2016 [1408.3082])
  pipelines the correction levels of a deferred-correction method on p cores: parallelism across the *method*,
  bounded by the order, not across time.

### 2.2 Does chaos kill it for a star cluster? Yes.

1. **The convergence window is O(1) for chaotic systems, independent of the step.** Gander & Hairer 2014
   (JCAM 259, 2-13, DOI 10.1016/j.cam.2013.01.011 [gander_hairer2014_parareal_hamiltonian]) prove windows of O(ε^-1)
   for the harmonic oscillator, O(ε^-1/2) for integrable systems, O(1) for chaotic ones (ε = coarse accuracy; §1, §3).
   Thm 3 bounds the error by c_k e^{Lt}(tΔT^r)^{k+1}; on Hénon-Heiles the window "does not increase for smaller ΔT"
   (Fig. 3). L is the Lipschitz constant of the vector field, which in a collisional cluster is set by the hardest
   binary and closest encounter (1/t_orb), many orders of magnitude above the ensemble Lyapunov rate 1/(0.1 t_cr)
   (Goodman, Heggie & Hut 1993, DOI 10.1086/173196).
2. **No cheap, accurate coarse propagator exists.** One that does not resolve binaries and encounters gets them O(1)
   wrong; one that does is not cheap. With sensible high-order F and G, cost(F)/cost(G) ≈ 2-4 and "the use of the
   parareal algorithm is questionable" (Gander & Hairer 2014 §8).
3. **Structure is lost.** The Parareal map is not symplectic even with symplectic F and G, and iterates drift in energy
   (Gander & Hairer 2014 §5, Fig. 1). Symmetric Parareal is "more or less identical" (§7); Dai, Le Bris, Legoll & Maday
   2013 call it "disappointing" and need projection onto the energy (and angular momentum) to converge
   (ESAIM M2AN 47, 717, DOI 10.1051/m2an/2012046, §4.3, §5.3-5.4 [1011.6222]). Unconverged iterates would break our
   |dE/E| rule; converged ones give no speed-up.
4. **Even near-integrable systems are hard.** Jiménez-Pérez & Laskar 2011 (time-parallel method for almost integrable
   Hamiltonians, Kepler-split coarse solver [1106.3694]) work on pendulum and spin-orbit toys but find for the full
   La2010 solar system that "for j > 8 the value of C_Δt ≡ 1 which is a worse case than the sequential solution" (p. 17).
5. **Measured particle-code results are modest even for smooth problems.** Speck+2014 (PFASST on a vortex-particle
   N-body problem, N = 10^4, smooth regularised kernel [speck2014_DD21_nbody_SDC_PFASST], Fig. 2): 8 time-ranks give
   ~0.45 of the serial runtime, i.e. 2.2x for 8x the cores. Muralikrishnan & Speck 2024 (Parareal for
   particle-in-Fourier Vlasov-Poisson on up to 1536 A100s [2407.00485], §6.3): 2.4-5.8x over space-only parallelism;
   the oscillatory Penning trap "does not follow the theoretical scaling" and long windows gave "virtually no speedup".
   PFASST/MGRIT inherit the same limits for hyperbolic/oscillatory problems (Gander, Wu & Zhou 2025, Acta Numerica
   [2503.13526]; Danieli & MacLachlan 2021 [2104.09404]).
6. The plasma-turbulence successes (Samaddar, Newman & Sánchez 2010, ~10x "unexpected", DOI 10.1016/j.jcp.2010.05.012)
   and MD (Gorynina+2022 [2212.10508]) accept *statistical* rather than trajectory accuracy. For us that would mean a
   non-symplectic, energy-drifting integrator, which our accuracy rule excludes.

The one real attraction (concurrent time slices would multiply the work at the latency-bound deep levels) is moot
because the iteration does not converge.

### 2.3 Where it could work: sub-problems

- **Long regularised near-Kepler binaries or weakly perturbed hierarchical triples** (LogH/GBS fine, Kepler drift or
  Kepler + perturbation coarse). The integrable theory applies (window ∝ ε_G^-1/2). Precedents: Saha, Stadel & Tremaine
  1997 (time slices + Picard iteration with a Kepler split; n_it ≈ 6 + N/1000; ~50x estimated on 1000 processors,
  §5-6, AJ 114, 409, DOI 10.1086/118485 [astro-ph_9605016]); Gander & Hairer 2014 Fig. 4 (Kepler, Gauss order-8 coarse
  solver at ~1e-14: 1 iteration over a window of 300, 2 over > 20 000 time units); Dai+2013 §7 (outer solar system with
  a Sun-planet-only coarse model, ~66x estimated for K = 15 on 1000 processors). Expected for us: N_slices/K ≈ 2-10x on
  *that subsystem's* wall time, converged to tolerance, never bitwise. Pitfalls: LogH runs in fictitious time, so
  slices must be cut in physical time to land on the parent step; PN 2.5/3.5 dissipation spoils integrability; strongly
  interacting triples are chaotic and excluded. **Only worth it if a single subsystem is the critical path** of a level
  (the F3 "LogH retry stall" class of `AUDIT_2026-10.md`); there are thousands of independent subsystems, so parallelism
  *across* subsystems is abundant and comes first (BIFROST §3.3: "even a large collection of subsystems can be
  simultaneously integrated in parallel").
- **Cheaper alternative for a straggler: parallel extrapolation.** In GBS the substep sequences n_1, n_2, ..., n_k of one
  extrapolation are independent and can run on k threads (parallelism across the method). Ketcheson & bin Waheed 2014
  (CAMCoS 9, 175, DOI 10.2140/camcos.2014.9.175 [1305.6165]) find that in parallel, extrapolation beats the best serial
  Runge-Kutta pair (DOP853), up to ~2x at tight tolerances on an N = 400 N-body test (abstract). This keeps the GBS
  result bit-identical to the serial GBS if the column values are combined in the same order; only the wall time of
  that one subsystem changes. Use it for subsystems whose predicted cost exceeds (total cost)/(threads).
- **Secular/Kozai evolution: no.** The orbit-averaged ODE is already cheap (F4: ~16 µs per evolution). Its only role
  would be as the coarse model of a "micro-macro Parareal" (Legoll, Lelièvre & Samaey 2013, DOI 10.1137/120872681
  [1204.5926]), valid where the secular approximation holds.

---

## 3. Asynchronous and multirate time integration

- **Local time stepping / multirate.** Multirate multistep (Gear & Wells 1984, DOI 10.1007/BF01934907), energy-conserving
  leapfrog LTS for waves (Diaz & Grote 2009, DOI 10.1137/070709414; Grote & Mitkova 2013 [1205.0654]: "fully explicit
  and inherently parallel"), multirate GARK (Günther & Sandu 2016, DOI 10.1007/s00211-015-0756-z; Sarshar+2019
  [1804.07716]). HHS-FSI already *is* the gravitational multirate method; these add theory (stability, energy
  behaviour), not parallelism beyond what the level structure exposes. Hydro analogy: AMR subcycling.
- **Asynchronous variational integrators** (Lew, Marsden, Ortiz & West 2003, ARMA 167, 85, DOI
  10.1007/s00205-002-0212-y [lew2003_AVI_ARMA]; parallel version Kale & Lew 2007, DOI 10.1002/nme.1880, numbers
  **UNVERIFIED**). Each element or pair potential has its own step, processed in time order from a priority queue;
  derived from a discrete variational principle, hence multisymplectic and exactly momentum-conserving. HHS with
  pairwise CC criteria is effectively an AVI quantised to powers of two. A per-pair AVI for N² gravitational pairs
  would lose the block structure that makes GPU batching possible and has a serial priority-queue core. Not for us.
- **Event-driven simulation** (Lubachevsky 1991, "How to simulate billiards and similar systems", DOI
  10.1016/0021-9991(91)90222-7; bounded lag, Lubachevsky 1989, DOI 10.1145/63238.63247). Natural for hard spheres where
  events are pairwise and short-range. Gravity's infinite range couples every event to every particle.
- **PDES.** Conservative Chandy-Misra(-Bryant) vs optimistic Time Warp (Jefferson 1985, "Virtual time", DOI
  10.1145/3916.3988); survey Fujimoto 1990 (DOI 10.1145/84537.84545). Time Warp needs rollback; with infinite-range
  forces a late message rolls back everything, chaos plus round-off makes reverse computation impossible (full state
  checkpoints instead), and it destroys bit-reproducibility. **Conservative PDES with the HHS lookahead (§1) is the
  correct and sufficient model.**

**Can HHS levels or components run ahead without a global barrier per level?** Yes, within one parent interval:
(a) the slow set's half step and the fast set's whole sub-recursion; (b) all components of F, each to its own depth.
No, across parent kicks: the S-F kick needs both at the same time. "Running ahead" therefore means *a task tree with
three join points per call*, and the critical path of a parent step becomes the deepest component's chain instead of
the sum over all components. The main practical problem is load balance (Jänes+2014 §4.1: a naive parallel CC
implementation "does not scale well due to load-balancing issues"), which batching and size-aware scheduling address
(§6, ideas #2, #3).

---

## 4. Task-based runtimes and astrophysics codes: what they gain on deep hierarchies

| system | key idea | measured | lesson for us |
|---|---|---|---|
| **Charm++ / ChaNGa** (Kale & Krishnan 1993, DOI 10.1145/165854.165874; Jetley+2008, DOI 10.1109/IPDPS.2008.4536319; Menon+2015 [1409.1929]) | overdecomposition into migratable objects, measured per-rung load, hierarchical balancers | multistep gain 3.29x (1 proc) → 1.39-1.55x (32-64) (Jetley+2008 Table 3); heaviest tree piece 3000x the average in the fastest rung (Menon §8.1, Fig. 8); intra-node work pushing for the fastest rung 2.3 s → 0.3 s (Fig. 9); end-of-phase intra-node rebalancing 9.8 → 8.5 s (Fig. 10); 48 % efficiency at 128K cores multistep, multistepping still 2-3x faster than single-step (§8.4-8.5) | balance *inside the node*, per rung; decompose from scratch only when many particles are active |
| **ChaNGa-GPU** (Jetley+2010, DOI 10.1109/SC.2010.49) | GPU for tree forces | GPU pays only above 250-500 targets per processor; few-target steps run on the CPU ("memory transfer and GPU kernel startup overheads dominate", §VII-A, Fig. 8) | count-based CPU/GPU switch per level |
| **HPX / Octo-Tiger** (Marcello+2021 [2101.08226]; Daiß+2019 [1908.03121]; Daiß+2022 [2210.06438]; Daiß+2023 [2303.08058]; Daiß+2025 [2412.15518]) | futures; GPU events polled by the scheduler instead of fences; on-the-fly fusion of identical small kernels | FMM kernels of 8 blocks x 64 threads use 8 of 56 SMs (2019 §6.1.1); work aggregation 10.0x (CUDA), 9.35x (Kokkos-NVIDIA), 10.8x (HIP), 11.96x (Kokkos-AMD) over none; fusion over multi-stream: 1.52x CUDA, 4.49x HIP, 5.23x Kokkos-HIP (2022 §VI-C); event polling 1.11-1.15x single node (2023); 47.6 % efficiency on 6880 A100s (2025) | fuse same-type kernels across independent pieces; poll events; preallocate executors and buffers. **Kokkos-HIP fences when its fixed launch-buffer array wraps, under a mutex (2022 §VI-C) — relevant on viper** |
| **Legion / Regent** (Bauer+2012, DOI 10.1109/SC.2012.71; Slaughter+2017 control replication, DOI 10.1145/3126908.3126949; Lee+2018 dynamic tracing, DOI 10.1109/SC.2018.00037; Task Bench, Slaughter+2020 [legion_taskbench2020]) | control replication = every rank runs the control program (what we do); trace replay | control replication 99 % at 1024 nodes for ms-scale tasks; dynamic tracing 4.9x average, up to 7.0x strong scaling; Task Bench: "100 µs-long tasks are the finest granularity that any system runs efficiently" at scale, MPI 4.6 µs on 1 node; GPUs impose "a floor on task granularity" (§V-H) | keep replicated control; replay the per-level sequence (graphs); no general runtime makes 1-10 µs tasks efficient across nodes |
| **StarPU / PaRSEC / ScalFMM** (Augonnet+2011, DOI 10.1002/cpe.1631; Bosilca+2013, DOI 10.1109/MCSE.2013.98; Agullo+2014, DOI 10.1137/130915662 [1206.0115]) | per-kernel calibrated cost models; HEFT-like device choice | LU on 3 CPUs + 1 GPU: a task only 1.2x faster on GPU goes 94.7 % to CPUs, combined efficiency 101.3 % (Augonnet §4.4, Table 4); FMM 60-86 % at 160 cores, P2P on GPU and far field on CPU (Agullo §4.4, §5) | automate the per-level CPU/GPU switch; template for a future FMM near/far split |
| **SWIFT / QuickSched** (Schaller+2016 [1606.02738]; Schaller+2024 [2305.13380]; Borrow+2018 [1807.01341]; Gonnet+2016 [1601.05384]) | dependency + conflict task graph per node, asynchronous MPI, METIS on measured task costs, drift only active cells | small steps cost ~2.7x more per update than full steps (Borrow §V); cost-based decomposition cut the fixed cost of a small step ~10x by removing communication (§IV-B, Fig. 7); > 60 % efficiency 512 → 131 072 cores with "only a few hundred tasks" per node (2016 Fig. 8); QuickSched ~1 % overhead, 75 % efficiency on 64 cores (Fig. 11) | small steps must exchange no messages (we already replicate them); critical-path-first queue priorities for host subsystem tasks |
| **GADGET-4** (Springel+2021 [2010.03567]) | hierarchical (HHS) integration; multi-bin balancing on a Peano-Hilbert curve; algorithm switch for thin bins | hierarchical cost K = N_0 + 3Σ2^(k−1)N_k vs one-sided (§4.2); PM skipped in thin bins (§4.3); FMM with thin bins "more or less forces" hierarchical integration (§4.4); balance within ~18 % in all three categories (Fig. 30); refuses timing-based balancing because it breaks binary reproducibility (§6.1) | count-based (not timing-based) switches keep results reproducible; relevant for our FMM plan |
| **NBODY6++GPU** (Wang+2015 [1504.03687]) | regular force on GPU, irregular force + KS on CPU (AVX + OpenMP) | active-set selection as costly as the irregular force until a sorted time-step list (~5x faster; still ~7 % at 1e6, §3.3.1, §4.4); MPI + data movement ~half the time at 16 nodes; KS regularisation "can be inefficient" to parallelise with MPI, compute/comm ratio < 100 (§4.3, Table 2, §6) | keep few-body work in shared memory; audit O(N) work per call |
| **PeTar** (Wang+2020 [2006.16560]) | isolated short-range clusters integrated in parallel on OpenMP threads; tree on MPI+GPU | load imbalance from big clusters and stable hierarchies: "the benefit of parallel computing is lost" (§8); flattens beyond ~160 cores at 1e5 (§7.2) | cost-ordered scheduling; split the largest cluster internally |
| **FROST / BIFROST** (2011.14984 §5.2, §6.3; 2210.02472 §3.4, §4.2) | one rank per GPU; MPI-CPU loops when N is too small for GPUs, serial CPU below a few hundred; neighbour search per level on serial CPU, MPI-CPU or CUDA by count | scaling stalls at N_GPU ≈ 4(N/1e5) (FROST Eq. 55); 1e5 inefficient beyond 4 GPUs, 1e6 to 48 (BIFROST Fig. 6) | our 49 % at 4 GPUs (200k) is on that curve; the reference code itself uses the CPU for small levels |

**Kokkos options.** `Kokkos::Experimental::Graph` maps to CUDA/HIP/SYCL graphs; topology is fixed after instantiation;
no documented conditional nodes or node-parameter updates (Kokkos Programming Guide, "Graphs"). `partition_space`
returns execution-space instances with their own streams, no implied synchronisation (experimental). Kokkos tasking
(TaskScheduler) was deprecated in 4.5 as an "unused, limited implementation" (kokkos-core-wiki); with our 5.0.99 do not
build on it. Kokkos 3 overview: Trott+2022, IEEE TPDS 33, 805, DOI 10.1109/TPDS.2021.3097283 (not read).

**Net assessment.** A general asynchronous many-task runtime would not fix our problem: our deep-level tasks are 1-10 µs
of real work, an order of magnitude below the ~100 µs any runtime runs efficiently (Task Bench), and our control is
already replicated (Legion's best mode). What transfers is a set of *techniques*: work aggregation/fusion, event
polling instead of fences, trace replay, preallocated executor and buffer pools, per-level count-based device choice,
and cost-ordered intra-node scheduling.

---

## 5. GPU work aggregation, shared memory and the node

### 5.1 Launch and sync costs: our numbers are the literature's numbers

Our launch + fence of a 16-element kernel is 9.7 µs, launch alone 3.4 µs. Gray's NVIDIA blog "Getting Started with
CUDA Graphs" (2019, V100, 20 kernels/step of 2.9 µs): 9.6 µs per kernel with a sync after every launch, 3.8 µs with one
sync per step, 3.4 µs in a graph; graph instantiation ~400 µs once. **Most of the gain is from removing syncs
(9.6 → 3.8); graphs add a little (→ 3.4).** With ~200-300k fences per step we pay ~1.5-3 s of host latency per step.
GROMACS (Doijade+2025 [2509.21527] §2.2): each launch or copy 2-10 µs; ~20 launches + 30 event calls per step exceed
50 % of CPU wall time at peak rate.

### 5.2 Persistent kernels, megakernels, device task queues

- Persistent threads (Aila & Laine 2009 [aila2009hpg] §3.3-3.4: 1.5-2.2x for ray traversal; Gupta, Stuart & Owens 2012
  [gupta2012_persistent_threads]: global sync instead of relaunch 2-2.5x (Fig. 10), but "the hardware scheduler will
  always best an ad hoc software scheduler" when items are many, and debugging is "extremely challenging").
- Megakernels: Laine, Karras & Aila 2013, "Megakernels considered harmful" [laine2013hpg_megakernels]: splitting a
  megakernel into lean kernels connected by queues gave +36 to +221 % (Table 1); megakernel thread utilisation 23 % vs
  53 % (§5); causes: divergence, register hot spots, instruction-cache overrun. Matches our own rule never to fuse O(N)
  work into the O(N²) kernels.
- Device task queues: Whippletree (Steinberger+2014, ACM TOG 33(6), DOI 10.1145/2661229.2661250
  [steinberger2014_whippletree]): up to 4x over "time-sliced kernels with CPU readback of queue fill levels" (Table 3) —
  which is exactly our current structure — but occupancy is set by the heaviest procedure (§7). Tzeng, Patney & Owens
  2010: stealing/donation queues idle > 100x less than centralised queues. Mirage Persistent Kernel (Cheng+2025, OSDI'26
  [2512.22219]): 1.0-1.7x for LLM decode on A100.
- **For us:** a persistent level-loop kernel for the smallest sets is the high-ceiling, high-risk option (raw CUDA/HIP,
  cooperative launch needed for grid sync, no portable Kokkos grid barrier, warp size 32 vs 64). It competes with idea
  #2 (host-resident deep subtrees), which gives the same removal of launch latency with ordinary C++ and the existing
  host GBS. Recommend: do #1 and #2 first; consider a persistent kernel only if the GPU path must stay for the deep
  levels on raven.
- **CUDA dynamic parallelism:** average 1.21x *slowdown* in unstructured apps (Wang & Yalamanchili 2014); CDP2 (CUDA 12)
  removed device-side `cudaDeviceSynchronize`, so a parent cannot wait for a child; no HIP equivalent. Not for us.

### 5.3 CUDA/HIP graphs

Ekelund, Markidis & Peng 2025 [2501.09398]: creation ~4 µs per node plus a base cost; best batches 50-100 kernels;
pays after 2-3 replays on A100 (Fig. 6); up to 1.4-1.5x for small workloads. Van der Wijngaart (NVIDIA blog 2021,
A100): recapture + `cudaGraphExecUpdate` 16-26 %. PyGraph [2503.19779] §2.3.2: 25 % of graphs *hurt* performance
(worst +397 %). Conditional nodes (IF/WHILE from CUDA 12.4, SWITCH 12.8) and device graph launch (CUDA 12.0) exist but
have no HIP or Kokkos equivalent (HIP graphs are "in Beta"). **For us:** our launch pattern changes every call (counts,
branches), so per-call capture does not pay. What does: (a) a fixed-topology graph per phase template whose kernels
read the active count from a device View and over-launch with early exit; (b) only after idea #1 removes the host
branches on device data inside a phase. Bit-identical (arithmetic unchanged).

### 5.4 Batched small ODE problems (GPU GBS?)

DiffEqGPU (Utkarsh+2024, CMAME 419, 116591 [2304.06835]): a whole solver per thread beats one kernel per vector
operation by 48x (adaptive) to 378x (Table 1); GPU beats CPU from ~100-1000 trajectories; adaptive steps cause
divergence. MPGOS [2011.01740]: GPU saturates at ~7680 systems; "a warp runs at the speed of its slowest thread".
Niemeyer & Sung 2014 [1309.2710]: divergent adaptive steps cost 2.3-4x; GPU beats 6 CPU cores only from ~512 ODEs.
SUNDIALS many-system (Balos+2024 [2405.01713] §3.3): lockstep batches go at the speed of "the most difficult ones".
No verified GPU code for regularised (LogH/AR) few-body GBS was found (GENGA [1404.2324] runs plain Bulirsch-Stoer for
close encounters on GPU). **For us:** our subsystems (2-10 bodies, LogH cubic solve, PN up to 3.5, merger checks, GBS
step counts varying by orders of magnitude) are the worst case for lockstep. Keep GBS on the CPU and *overlap* it with
GPU work (idea #3); GPU-batch only binned 2-body cases, and only if CPU GBS becomes the critical path after #1-#3.

### 5.5 GPU on-chip shared memory, warps, clusters

- **Tiling** (Nyland, Harris & Prins 2007, GPU Gems 3 ch. 31; Kirin, Belleman+2008 [0707.0438]; Sapporo1, Gaburov+2009
  [0902.4463]): sources staged in shared-memory tiles, as `n2_driver.h` does. Below ~4096 sinks "there is not enough
  work with one thread per body"; splitting each sink's row over q threads gave +44 % at N = 1024 (GPU Gems).
  **Sapporo2** (Bédorf, Gaburov & Portegies Zwart 2015 [1510.04068] §2.1, §3.1, Figs. 1-2): "2D thread blocks" (several
  threads per sink) matter "especially when the number of active i particles is ≤ 128". NBODY6-GPU (Nitadori & Aarseth
  2012 [1205.1222] §3.4) uses j-parallelism over blocks with reduction kernels; FROST §5.3 uses the multi-thread-per-sink
  variant, "a few tens of percents especially when … GPU occupancy is low".
- **Our driver** already splits sources into chunks for few sinks, with a deterministic chunk-order second pass. But each
  team is 128 threads with only `ni` active, and `chunk_count` caps at 64 teams (`n2_driver.h:44-53`; target
  `concurrency/(2·TS)` ≈ 864 teams on A100). For a cross kick with 50 fast sinks against 170k slow sources this is
  64 blocks with 50 live threads each. Idea #7: `TeamPolicy(league, team, vector)` with `ThreadVectorRange` over
  sources and a lane reduction (warp shuffles), a higher cap for small `ni`. Changes the summation split →
  not bit-identical → 3 seeds. Measure first: at 8.5e6 pairs the ideal is ~50 µs at 161 Gpairs/s, so the gain is real
  only where these kernels are not already hidden by launch latency.
- **Shared memory/L1 carveout** (A100: 192 KB per SM, up to 164 KB shared; MI300A: 64 KB LDS): our tiles need 4-7 KB per
  team and the kernels are FP64-bound (79 % SM FP64), so there is little to gain.
- **Warp primitives** (Lin & Grover, NVIDIA blog 2018): one warp per small subsystem or sub-warp tiles per sink; HIP
  wavefronts are 64 wide, so use Kokkos vector ranges, not hard-coded 32-lane masks. `__reduce_add_sync` is integer-only.
- **Thread-block clusters / distributed shared memory** need compute capability 9.0 (Hopper); not on A100 or MI300A.
  Cooperative-groups grid sync needs a cooperative launch; not exposed portably by Kokkos (UNVERIFIED for 5.0).

### 5.6 Node level: MPI shared windows, IPC/NVLink, APUs, NUMA

- **MPI-3 shared-memory windows** (Hoefler+2013, Computing 95, 1121, DOI 10.1007/s00607-013-0324-2
  [Hoefler2013_MPI+MPI]): stencil communication 30-60 % lower than send/recv (Fig. 6a); up to 8 % compute loss from
  NUMA/false sharing unless `alloc_shared_noncontig` (Fig. 6b). FROST-CLUSTERS III (2601.07917 §2.1.2) states BIFROST
  uses node-local shared windows so ranks "access the same single copy of the particle ensemble", with no numbers. For
  us: host-side tables (subsystem results when subsystems are distributed, idea #5) can go through a node window: each
  rank writes its slot, one barrier, everyone reads. GPU-resident arrays need IPC instead.
- **CUDA/HIP IPC over NVLink** (raven: 4x A100-SXM4, all pairs NVLink3 ~100 GB/s per direction, MPCDF docs). Li+2019
  (TPDS, [1903.04611] §3.1.1, Fig. 5; V100 DGX-1): P2P start-up ~9 µs including API overhead; NCCL collective latency
  ~18 µs (§3.2.2); NCCL has no AllGatherv. Our 200k `Allgatherv` moves ~3.6 MB into each rank, ~12-40 µs on the wire,
  but takes 1.3 ms host-staged, 3.4 ms with device buffers, and 0.77 ms even host-only. A node-local gather where each
  rank writes its sink slice straight into the peers' IPC-mapped buffers (or `cudaMemcpyAsync` with peer pointers) and
  signals with an IPC event or flag should cut this by 10x or more (idea #4). On MI300A (Schieffer+2025, MEMSYS
  [2508.11298] §6.1): MPI 1.9 µs host-staged, 4.8 µs GPU-direct, RCCL ~20 µs for small messages; RCCL wins only above
  ~4 KB; disabling SDMA raises `hipMemcpy` between APUs 58 → 90 GB/s.
- **Hybrid MPI + threads vs one rank per GPU.** Rabenseifner, Hager & Jost 2009 [Rabenseifner2009_hybrid_PDP]: hybrid
  pays only with pinning and first-touch (9.6 % on BT-MZ at 4096 cores). GROMACS thread-MPI drives all GPUs from one
  process with direct peer copies and cross-stream events (Páll+2020 [2006.09167] §II). Kokkos binds one device per
  process in practice; keep one rank per GPU, add IPC and node windows.
- **APUs (viper, MI300A).** Wahlgren+2025 [2508.12743]: GPU HBM latency 333-350 ns, CPU 236-241 ns ("for latency-bound
  tasks, the CPU has a significant advantage", §4.1); `hipMalloc` 3.5-3.6 TB/s GPU bandwidth vs 1.8-2.2 with other
  allocators (§4.2); CPU+GPU contended atomics drop CPU throughput to 11-25 % (§4.4); page faults 9-18 µs (§5.2);
  `hipMalloc` ~10 µs, free up to 22x slower (§5.1) — preallocate. On an APU the deep levels can run on host cores
  directly on `hipMalloc` data, no copies (idea #2 at zero copy cost). **Check (idea #9):** our `kHostAccessibleDevice`
  path (`small_for.h:11`) is only true if Kokkos treats HIPSpace as host-accessible; Kokkos documents
  `Kokkos_ARCH_AMD_GFX942_APU` (since 4.5) for MI300A and says `GFX942` "should only be used for MI300X". The memory
  notes say the viper install is "HIP gfx942"; if that is the non-APU flag, the host path is compiled out on viper.
- **NUMA** on raven (2x Xeon 8360Y, 36 cores per socket, UPI ~67 GB/s per direction): GBS is compute/latency-bound with a
  small working set, so NUMA matters through thread drift and wrongly first-touched host mirrors. Pin each rank's 18
  threads to its GPU's socket (`--cpu-bind=cores`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`) and check the
  GPU-to-socket map with `nvidia-smi topo -m`.

### 5.7 Host threading for many small subsystems

- `logh_batch.cpp:586` uses `RangePolicy<Schedule<Dynamic>>.set_chunk_size(1)`: already a central work queue, one atomic
  per subsystem, negligible vs a GBS integration. OpenMP tasks (Ayguadé+2009, DOI 10.1109/TPDS.2008.105) or work
  stealing (Blumofe & Leiserson 1999, DOI 10.1145/324133.324234, T_1/P + O(T_∞)) add nothing for independent tasks
  unless a single subsystem is split internally.
- **Ordering matters more than the scheduler.** Graham 1969 (SIAM J. Appl. Math. 17, 416, DOI 10.1137/0117039): longest
  processing time first is within 4/3 − 1/(3m) of optimal; arbitrary order can finish ~2x late when a long triple is
  picked up last. Predicted cost is available: `gbs_per_sub[s]` from the previous call times n_bodies². If one
  subsystem costs more than total/threads, no ordering helps → parallel extrapolation (§2.3) or overlap.
- **False sharing:** per-subsystem outputs are written once at the end of a long integration; harmless. Pad any future
  per-thread accumulators (`alignas(64)`).
- **Determinism:** the host reduction sums ints and takes min/max (exact in any order); energy books are summed in
  index order after the parallel region (`integrator.cpp:621-628`). Keep that rule for any new parallel stage and
  across ranks (gather per-item values, sum in index order). Reproducible-sum alternatives cost 3-44x (Collange+2015,
  DOI 10.1016/j.parco.2015.09.001 [Collange2015_reproducible_reduction] §6) and are not needed with this pattern.

### 5.8 CUDA-aware MPI on raven: why device buffers are slower than host staging

**Measured facts (raven).** Open MPI 5.0.10, pml ucx, UCX 1.20 built with the cuda, gdrcopy and cuda_ipc modules.
Collective components available: adapt, basic, cuda, han, inter, libnbc, sm, sync, tuned. There is **no coll/ucc and
no hcoll**. `MPI_Allgatherv` of 3 doubles per particle, 4 ranks on one node, equal slices:

| n | device buffers | host staging (D2H + host MPI + H2D) | host-only MPI |
|---|---|---|---|
| 35k | 1.63 ms | 0.42 ms | 0.22 ms |
| 200k | 3.43 ms | 1.31 ms | 0.77 ms |
| 1e6 | 16.8 ms | 7.3 ms | 4.2 ms |

Forcing `UCX_TLS=sm,self,cuda_copy,cuda_ipc` made the device path slower still (9.4 ms at 200k) (AUDIT G.2). The
wire time is ~12-40 µs at 200k: each rank receives 3.6 MB over NVLink3 at ~100 GB/s per direction per pair.

**Why: the chain of causes, checked in the Open MPI v5.0.x source.**
1. **`coll/cuda` does not handle allgatherv.** `ompi/mca/coll/cuda/coll_cuda_module.c` wraps only allreduce, exscan,
   reduce, reduce_local, reduce_scatter_block and scan, by staging through host memory. It sets
   `coll_allgather = NULL; coll_allgatherv = NULL; coll_bcast = NULL`. On main the successor `coll/accelerator` adds
   allgather, alltoall and bcast, still not allgatherv. So our device-buffer call falls through to the CPU-tuned
   `coll/tuned` (or han/adapt if they win on priority).
2. **`coll/tuned` picks algorithms that build derived datatypes.** `coll_tuned_decision_fixed.c`, communicator size
   < 8, by bytes per rank: < 256 B basic (gatherv + bcast), < 4 KB neighbour exchange, < 8 KB ring, < 16 KB neighbour
   exchange, < 256 KB **Bruck**, >= 256 KB **neighbour exchange**. Our cases: 35k ≈ 210 KB per rank → Bruck;
   200k ≈ 1.2 MB and 1e6 ≈ 6 MB → neighbour exchange. Both create `ompi_datatype_create_indexed` types at every step
   (`coll_base_allgatherv.c`). Only ring sends plain contiguous blocks.
3. **Indexed datatypes switch off the GPU fast path.** `pml_ucx_datatype.c` maps contiguous types to UCX contiguous
   buffers. It maps non-contiguous types to a UCX *generic* datatype, whose pack/unpack callbacks go through the
   Open MPI convertor. For device memory, the convertor copies with `cuMemcpyAsync` on its own stream **followed by
   `cuStreamSynchronize`** (`opal/mca/accelerator/cuda/accelerator_cuda.c`). Inference: a generic datatype cannot use
   UCX's rendezvous zero-copy over cuda_ipc, so every fragment becomes a synchronous copy to the host, a host transfer
   and a copy back. That is host staging done badly, in many small pieces with a sync each.
   - The Open MPI docs say so: "Non-contiguous datatypes currently carry high overhead"
     (docs.open-mpi.org, tuning-apps/accelerators/cuda, §11.3.5.4).
   - TEMPI (Pearson+2021, arXiv 2012.14363) documents the same weakness across CUDA-aware MPIs.
   - Further per-message costs: memory-type checks (`cuPointerGetAttributes`; UCX's memtype cache) and
     `CU_POINTER_ATTRIBUTE_SYNC_MEMOPS` set on registered GPU buffers.
4. **han and adapt ignore GPU buffers.** NERSC on Perlmutter (4x A100 nodes, like raven) found HAN "does not appear
   to take GPU buffers into account leading to poor performance". Released Open MPI collectives on GPU buffers were
   "highly variable" until new allgather code reached main (Gayatri+, CUG 2025).
5. **Restricting UCX_TLS** removed gdr_copy and rc. Our hypothesis is that small device fragments and the generic pack
   path then fell back to cuda_copy (its model: 8 µs latency). This would explain the 9.4 ms.
   - ucTrace (Gencer+2026, arXiv 2602.19084 §IV-A) observed on other A100/H100 systems that intra-node rendezvous uses
     get_zcopy over cuda_ipc and eager uses gdr_copy.

**What would make CUDA-aware MPI perform** (cheapest first):
- **Force ring (no rebuild):** `OMPI_MCA_coll_tuned_use_dynamic_rules=1 OMPI_MCA_coll_tuned_allgatherv_algorithm=3`,
  and try `OMPI_MCA_coll=^han,adapt`. Ring sends contiguous blocks, so cuda_ipc zero-copy can engage: 3 steps for
  4 ranks. This is our inference, not measured.
- **Hand-written allgatherv:** 3 `MPI_Isend` + 3 `MPI_Irecv` on contiguous slices of the persistent device buffer,
  then `Waitall`. It avoids derived datatypes, independent of tuned's decision table. Keep the send/receive Views
  persistent, as `mpi_comm.cpp` already does, so UCX's IPC-handle cache (`UCX_CUDA_IPC_CACHE=y`) is hit. The first
  `cuIpcOpenMemHandle` is expensive (tens to hundreds of µs, UNVERIFIED).
- **UCC (`coll/ucc`):** needs Open MPI built `--with-ucc` (or NVIDIA HPC-X, where it is on by default), then
  `OMPI_MCA_coll_ucc_enable=1`. It is disabled if Open MPI was built with progress threads (Open MPI docs,
  tuning-apps/collectives/ucc).
  - UCC's TL/CUDA implements allgather**v** over NVLink IPC; TL/NCCL implements it with p2p or bcast variants
    (openucx/ucc GitHub).
  - UCC paper: Venkata+, Hot Interconnects 2024, claims up to 3.4x lower latency than coll/tuned (seen via a secondary
    source; no arXiv version).
  - One forum report says UCC still chose TL/UCP for CUDA allgatherv (anecdotal).
  - AMD recommends coll_ucc built with ROCm for GPU buffers (AMD Instinct cluster-networking docs).
- **UCX tuning:** `UCX_RNDV_THRESH` (HPC-X lists an Allgather/Bcast GPU slowdown with workaround
  `UCX_RNDV_THRESH=auto,inter:12kb`), `UCX_RNDV_FRAG_MEM_TYPES`, `UCX_PROTO_ENABLE=y` (default in 1.20). These are
  second-order next to point 3 above.
- **Persistent collectives (`MPI_Allgatherv_init`):** these exist in 5.0 via coll/libnbc, which builds a linear
  p−1-round schedule once (`nbc_iallgatherv.c`). That saves only the schedule construction. Progress still happens only
  inside MPI calls and the device copies are the same, so there is **no real gain for the GPU path**.
- **`MPI_Iallgatherv`:** helps only with independent work to overlap. Today there is none at the split levels; with
  idea #3 there would be (the fast recursion).
- **Other MPIs:** MVAPICH-Plus 4.x (CUDA-aware collectives, gdrcopy; MVAPICH2-GDR is no longer developed). Literature
  on small GPU allgathers: MVAPICH2-GDR Bcast up to 10-14x lower latency than NCCL 1.x for <= 8 KB (Awan+2017, arXiv
  1707.09414 §V-B). Rolinger+2018 (arXiv 1812.05964 §IV): on NVLink systems NCCL and MVAPICH-GDR allgatherv beat host
  MPI only above 16 KB per rank, and in their application CUDA-aware MPI was 1.73-1.85x *slower* than host MPI. This is
  our finding, published.

**NCCL instead of MPI for the gather.** NCCL has no allgatherv: all collectives need equal counts (NCCL user guide,
"Collectives"). Our slices differ by at most one particle, so **padding to the maximum slice + `ncclAllGather`** is
trivial.
- Calls are stream-ordered, so the host does not wait. Pass the Kokkos stream (`Kokkos::Cuda().cuda_stream()`,
  `hip_stream()`).
- NCCL ops can be captured in CUDA graphs (NCCL >= 2.9), which matches ideas #3 and #8. Capture must be uniform across
  ranks, one GPU per process.
- Latency: per-hop LL ~1 µs, LL128 ~2 µs, Simple ~6 µs (Hu+2025, "Demystifying NCCL", arXiv 2507.04786). Intra-node
  allreduce 4.7-8.9 µs captured in graphs on H200 (arXiv 2606.05951 §VII-C).
- Expect ~10-20 µs plus ~30-50 µs of bandwidth time for 4.8 MB at 200k on 4x A100 (our estimate; UNVERIFIED for this
  setup), against 1.31 ms host-staged.
- NCCL coexists with MPI (communicator created from MPI). Do not block in `cudaStreamSynchronize` while peers wait in
  MPI, and do not mix concurrent NCCL and CUDA-aware-MPI traffic between the same devices (NCCL docs, "MPI").
- On MI300A, RCCL has a ~20 µs floor: worse than MPI below 1 KB, 5-38x better above 4 KB (Schieffer+2025 §6). Our
  split sets are >= 16384 sinks (>= 100 KB per message), RCCL's good regime.

**Verdict for the code.** Keep host staging now (`USE_GPU_AWARE_MPI=OFF`). A/B the ring setting and the hand-written
Isend/Irecv; both are one-day tests. Then put padded `ncclAllGather`/`rcclAllGather` behind a compile option in
`mpi_comm.cpp` (`allgather_accelerations_subset`, `allgather_timestep_subset`, `iallgather_*`). It is portable to
viper, stream-ordered and graph-capturable, and less work than the IPC gather of idea #4, which it replaces. Keep the
IPC/NVSHMEM route for the case where the gather must be fused into the force kernel.

### 5.9 Parallel paradigms beyond MPI, weighed for our code

"Deep-level" means the replicated, latency-bound levels; "1e6-1e7" means the split levels and multi-node runs.
Maturity is for Kokkos 5 on CUDA (A100) and HIP (MI300A).

| paradigm | what it buys at deep levels | what it buys for 1e6-1e7 | maturity (Kokkos 5, CUDA / HIP) | effort | verdict |
|---|---|---|---|---|---|
| **MPI-3 RMA** (`MPI_Put/Get`, osc/ucx) | nothing (deep levels need no communication) | one-sided writes of sink slices; but UCX lists only *partial* GPU-memory support for RMA/atomics (UCX FAQ); no GPU-window performance data found | low on GPU buffers; fine for host buffers | low-medium | host-side only (subsystem tables, with `MPI_Win_allocate_shared`, §5.6) |
| **NVSHMEM / rocSHMEM** (GPU-initiated put/get, put-with-signal, symmetric heap) | allows a level to run as one graph or persistent kernel with no host in the loop, *if* deep levels were distributed (they should not be, §7) | force kernel puts its sink results straight into every peer's replica; intra-node put 1.8-2.5 µs, inter-node ~9.5 µs with IBGDA on H200 (arXiv 2606.05951 §VII); GROMACS +46 % at 45k atoms on 4 H100, up to 2x multi-node (arXiv 2509.21527) | NVSHMEM mature on CUDA, not wrapped by Kokkos (Views must come from `nvshmem_malloc`); rocSHMEM moved into ROCm/rocm-systems, RO and GDA backends "as-is, limited support" | medium-high | the long-term route for split levels on raven once NCCL is in; not first |
| **Kokkos Remote Spaces** (PGAS Views over NVSHMEM/rocSHMEM/MPI one-sided) | — | same as NVSHMEM, with Kokkos syntax | README: "experimental development stage", no releases; Kokkos 5 compatibility not stated; paper Mishler+2023, DOI 10.1109/CLUSTERWorkshops61457.2023.00028 | medium + research risk | watch, do not depend on |
| **UPC++** (memory kinds `cuda_device`/`hip_device`) | — | RPC + one-sided with GPU memory; "reference" kinds stage through the host with a "very noticeable" penalty; native kinds use GPUDirect RDMA | mature on CPU; GPU kinds since 2022.3; known Kokkos device-mismatch pitfall (UPC++ issue 578) | high | no clear gain over NCCL/NVSHMEM for replicated data |
| **Coarray Fortran** | — | — | not applicable to C++ | — | no |
| **NCCL / RCCL** | — | stream-ordered, graph-capturable allgather (padded); see §5.8 | mature on both (RCCL ~20 µs floor) | low-medium | **yes, first replacement for the device gather** |
| **UCC** (Open MPI coll/ucc; TL/CUDA, TL/NCCL) | — | GPU-aware collectives *inside* MPI, so no code change | needs an Open MPI/UCC rebuild on raven and viper; selection quirks reported | low code / medium admin | worth asking MPCDF for; an alternative to NCCL calls in our code |
| **Task runtimes as MPI replacement** (Charm++, HPX, Legion/GASNet-EX, StarPU-MPI, PaRSEC) | none: our tasks (1-10 µs) are below every runtime's efficient granularity (~100 µs, Task Bench) | Charm++ GPU-aware channels up to 10.2x lower latency than its host path (Choi+2021, arXiv 2102.12416); HPX LCI parcelport up to 5x lower latency, Octo-Tiger 1.175x vs MPI (SC23 PAW-ATM); none beats direct NCCL/NVSHMEM for a replicated-data integrator | HPX-Kokkos exists (Daiß+2021); Charm++/Legion/StarPU have no Kokkos-native integration (UNVERIFIED) | very high (rewrite) | no; borrow techniques (§4) |
| **One process per node, 4 GPUs** (threads + Kokkos instances on each device + CUDA P2P over NVLink) | one host copy of the replicated state; the subsystem work could use all 72 cores without MPI; GROMACS thread-MPI precedent (arXiv 2006.09167 §II) | peer copies over NVLink with no MPI inside the node; MPI only between nodes | Kokkos supports instances on several devices (`Kokkos::Cuda(cudaStream_t)` / `Kokkos::HIP(hipStream_t)`, ROCm >= 5.6; Kokkos "MultiGPUSupport" page), but every View and policy must carry the instance; NCCL advises one GPU per process for graph capture; MI300A has only 2 APUs per node | very high (every Kokkos call) | no; get the same benefits with IPC/NCCL + node windows + distributed subsystems (#4, #5) |
| **Ensembles** (many seeds as independent runs: Slurm job arrays, `MPI_Comm_split` ensembles; Flux/RADICAL/Parsl/Merlin as workflow tools) | perfect: no communication, each run keeps its GPU busy as well as 1 GPU allows | not a substitute for one 1e7 run; science needs seeds anyway (Rizzuto+2021: ~8 runs per model; HARD_PROBLEMS §2.4) | trivial | low | **default production mode for N <= 2e5**: 4 seeds x 1 GPU beat 1 seed x 4 GPUs (1 GPU 22.0 s/step vs 4 GPUs 11.3 s/step → 2x throughput per node); multi-GPU for N >= 1e6 |

**Net.** For the deep levels, no communication paradigm helps, because they should not communicate at all. The levers
there are §6 #1-#3. For the split levels and multi-node 1e6-1e7 runs, the order is:
1. host staging (now);
2. padded NCCL/RCCL allgather, or UCC inside MPI if MPCDF can build it;
3. NVSHMEM put-from-kernel later.

For throughput at N <= 2e5, ensembles beat any strong-scaling work we could do.

---

## 6. The ranked ideas, mapped onto the code

Baseline shares (200k, 4 GPUs, 11.3 s/step): depths 0-4 (distributed) 7.2 s, depths 5-7 1.7 s, depths 8-11 2.3 s
(2300 calls, 0.6-2 ms each). 1e5 with f_bin = 1: depths 7-22 = 47 % of 5.96 s. Non-scaling part ~8 s at 200k.

### #1 Cut the fixed cost per call (bit-identical; do first)

*Literature:* Gray 2019 (9.6 → 3.8 µs per kernel by removing per-kernel syncs); Daiß+2023 (polling events instead of
fences, 1.11-1.15x); Wahlgren+2025 (allocation/free costs: preallocate); NBODY6++GPU §3.3.1 (O(N) active-set selection
as costly as the irregular force until fixed).
*Code:* inside `hhs_fsi_recursive` and what it calls:
- one pinned host "control block" per call that the device fills with all scalars the host needs (slow count, neighbour
  pair count, clamp count, mergers flag, "is one subsystem" flag) and that is read once, instead of ~6 separate
  `deep_copy`/`parallel_reduce` round trips (`partition_by_pivot` `:1173-1228`, `forces.cpp:429`, `detect_from_pairs`,
  the clamp reduce `:2129-2138`, the isolated-set mirror `:2215`);
- no View allocation in the recursion (static pools, as `hhs_cc_step` already does);
- profiler fences only when profiling (AUDIT row 6, -4 %);
- `small_for` fences only the instance that produced the data.
*Gain:* 1.2-2 s at 200k, up to ~2x at 1e5 f_bin = 1 (AUDIT row 1). *Risk:* low. *Bit-identity:* yes (same kernels,
same expressions). *Prerequisite for* #3 and #8 (graphs and concurrency both need a recursion that does not stop the
host on every scalar).

### #2 Host-resident deep subtrees, task-parallel over components (largest gain at deep levels)

*Literature:* FROST §5.2 (serial CPU code below a few hundred particles); BIFROST §3.4 (neighbour search per level on
CPU or GPU by count); ChaNGa-GPU (CPU below 250-500 targets); NBODY6++GPU and PeTar (few-body work on CPU threads);
StarPU (device choice by cost model); GROMACS Fig. 7 (GPU kernels 5x less efficient than CPU at small inputs);
Wahlgren+2025 (APU CPU latency advantage). Audit row 14: our current `SERIAL_N2_CPU` path is +38 % because it copies
field by field *at every level* and syncs back for detection and subsystems; "a host path only pays if detection and
subsystem state also stay on the host for the whole sub-recursion".
*Design:* when `hhs_cc_step` (or the slow-count-zero branch) produces a component with n <= N_host (start with 64-256,
count-based so it is deterministic), copy that component's particles to host once, run the *entire* sub-recursion on
the host (partition, time steps, detection, subsystem advance, mergers, kicks, drifts), copy back once at the join.
Different components go to different host threads; with 18 threads per rank (or 72 per node, see #5) many components
run at once. On MI300A the copies vanish. The host n² at n = 50 is ~2500 pairs, a few µs, against 0.6-2 ms per GPU call
today.
*Gain:* most of depths >= 8 (2.3 s → a few tenths) at 200k; potentially ~40 % of the step at 1e5 f_bin = 1. Also frees
the GPU stream for the S work (synergy with #3).
*Risk:* medium. Host and device results differ in the last bits (FMA contraction, `rsqrt`/`cbrt` implementations), so
trajectories change; the result is still deterministic and independent of the rank count if the threshold is a count.
*Bit-identity:* no relative to today → 3-seed A/B under the energy rule. Mapping: `integrator_cpu.cpp`
(`hhs_fsi_recursive_cpu`, currently with per-level round trips `:40-110`, `:582-705`, `:749-858`) is the starting point;
the subsystem advance (`advance_subsystems_in_range`) already runs on host mirrors.

### #3 Concurrency inside a call: S ‖ F and component ‖ component (conservative PDES with exact lookahead)

*Literature:* Pelupessy+2012 §2.1 and Jänes+2014 Eqs. 26-31 (independence between parent kicks); Octo-Tiger (many
concurrent small kernels + fusion; executor pools); SWIFT/QuickSched (dependency task graph, critical-path priority);
Jänes+2014 §4.1 (load balance is the hard part).
*Two implementation routes:*
- (a) **Task route:** a second host thread and a second Kokkos instance; at step 2/3 (and 6/5) launch the slow set's
  `fsi_step` on instance A while the fast recursion proceeds on instance B. The slow step at depth 0-4 is large,
  throughput-bound and distributed; the fast recursion is latency-bound and replicated. Overlapping them hides the
  fast recursion's latency behind the slow set's kernels (and the slow set's host LogH behind the fast recursion's GPU
  work: this is the only overlap of host LogH with GPU work that the splitting allows without changing the
  arithmetic; inside `fsi_step` the force after a subsystem advance needs its result). MPI calls from one thread only
  (`MPI_THREAD_FUNNELED`).
- (b) **Breadth-first batched route:** replace the sequential `recurse_all` loop by a level-synchronous sweep over all
  components: each phase (time step, partition, kick, drift, self force) is one *segmented* kernel over all components
  with per-component offsets and an "active" mask, so 100 components at depth d cost one launch, not 100. This is the
  SWIFT "activate all cells of the bin" pattern and the GPU frontier-queue pattern; it is Octo-Tiger's work aggregation
  done statically. It is bit-identical if each segment runs the same expression in the same order (the n2 driver's
  per-sink Kahan blocks are per sink, so segmenting the sink range does not change any sum).
*Refactor list:* §1 ("What blocks this in the code today").
*Gain:* 1-3 s at 200k (estimate: up to min(slow work, fast work) per level, plus launch aggregation over components);
largest at f_bin = 1 where many components exist. *Risk:* medium (global state refactor). *Bit-identity:* yes.
Route (b) combines naturally with #2 (components above N_host batched on the GPU, below it on host threads).

### #4 Fix the gather; fuse collectives (bit-identical)

*Literature and source:* §5.8 (why the device-buffer `Allgatherv` is slow in Open MPI 5.0.10: no coll/cuda for
allgatherv, coll/tuned's Bruck/neighbour-exchange build indexed datatypes, UCX packs those through synchronous copies);
Rolinger+2018 (CUDA-aware MPI 1.73-1.85x slower than host MPI in an allgatherv application); NCCL docs (equal counts,
stream-ordered, graph capture); Li+2019 (NVLink P2P ~9 µs); Schieffer+2025 (MI300A: host staging 1.9 µs, RCCL floor
~20 µs but 5-38x better above 4 KB).
*Code:* `mpi_comm.cpp` `allgather_accelerations_subset`, `allgather_timestep_subset`, `iallgather_*` (`:411-680`):
1. rebuild with `USE_GPU_AWARE_MPI=OFF` (host staging, AUDIT row 4);
2. one-day A/B: `OMPI_MCA_coll_tuned_use_dynamic_rules=1 OMPI_MCA_coll_tuned_allgatherv_algorithm=3` (ring), and a
   hand-written 3x Isend + 3x Irecv on contiguous slices of the persistent device buffer;
3. padded `ncclAllGather` (RCCL on viper) on the Kokkos stream behind a compile option; or UCC in Open MPI if MPCDF
   builds it; a CUDA/HIP-IPC gather (write the slice into peers' mapped buffers, signal by IPC event) is the
   fallback if NCCL is not wanted;
4. fuse the paired collectives (`integrator.cpp:1885-1887`, `:1958-1960`, `:1986-1988`).
*Gain:* collective cost 0.9-1.15 s/step → ~0.1-0.3 s. *Risk:* low. *Bit-identity:* yes (same bytes; padding
entries are ignored).

### #5 Distribute replicated work across ranks; reduce rank skew

*Literature:* ChaNGa intra-node work pushing (2.3 → 0.3 s for the fastest rung, Menon §8.1); PeTar §6.3 (one owner per
cluster); Hoefler+2013 (node windows); GADGET-4 §6.1 (count-based, reproducible decisions); NBODY6++GPU §6 (KS must stay
in shared memory, i.e. inside the node).
*Code:* `DISTRIBUTE_SUBSYSTEMS` (`integrator.cpp:527-532`, round-robin, measured -3.4 %) → cost-ordered list assignment
(predicted cost from `gbs_per_sub`) with results shared through a node window or one small gather; extend to CC
components (each component's subtree owned by one rank, results broadcast at the join). The audit shows the replicated
host work finishing at different times per rank is 2/3 of MPI time (AUDIT G.2) and the non-scaling part is ~8 s.
*Gain:* 0.4-1 s at 4 GPUs, and this is the item that changes the *slope* of strong scaling, because it is the only one
that gives extra ranks different work at the replicated levels. *Risk:* low-medium. *Bit-identity:* yes, if each item
is computed by exactly one owner with the same code and the books are summed in index order (results then do not depend
on the rank count, which the HHS_CC comment at `:1986-1988` requires).

### #6 LPT ordering of host subsystem integrations (bit-identical, cheap)

Sort `eligible` by predicted cost, descending, before the `parallel_reduce` at `logh_batch.cpp:586`. Graham 1969 bound
4/3 vs ~2. First measure `max(gbs_per_sub)/mean` per call to see whether tails exist. Results unchanged (each subsystem
is integrated serially by one thread; books in index order).

### #7 Few-sink kernel shape (`n2_driver.h`)

Several threads per sink for `ni` <= 64-128 (Sapporo2 2D blocks via Kokkos vector lanes) and a chunk cap that grows when
`ni` is small. Mid levels (n = 300-6000, 1.7 s) and the S-F cross kicks with few fast sinks. Changes summation split →
3 seeds. Measure first; DtK small sets are latency bound (n = 256: 0.31 ms).

### #8 Graphs per phase template

After #1. Kokkos::Graph with device-resident counts and over-launch; expect 10-50 % of the remaining launch cost
(Gray 2019; Ekelund+2025 1.4x; Van der Wijngaart 16-26 %). Watch the Kokkos-HIP launch-buffer fence (Daiß+2022). No
conditional nodes in Kokkos, so the recursion itself stays on the host.

### #9 Viper build flag

Verify `Kokkos_ARCH_AMD_GFX942_APU` in the viper Kokkos install (`KokkosCore_config.h`). Trivial; potentially large on
viper because every deep-level `small_for` and small n² currently would launch kernels instead of running on the host.

### #10 Stragglers: parallel extrapolation, then (maybe) Parareal

Parallel GBS columns for subsystems with predicted cost > total/threads (Ketcheson & bin Waheed 2014); Parareal with a
Kepler coarse step only for a near-Kepler subsystem that is still the critical path afterwards (§2.3).

### Order and interplay

Ensembles for N <= 2e5 production at once (no code). Then #1 (and the cheap AUDIT G.4 items) → #9, #6 (cheap checks) → #4 → #2 and #3(b) together (the deep-level redesign: small
components on host threads, larger ones batched on the GPU) → #5 → #7, #8 → #10. Bit-identical: #1, #3, #4, #5, #6, #8.
Need 3-seed A/B: #2, #7, #10.

---

## 7. What not to do

- **Parareal/PFASST/MGRIT/RIDC on the whole cluster.** O(1) convergence window for chaotic systems (Gander & Hairer
  2014 Thm 3); no cheap accurate coarse solver; non-symplectic iterates drift in energy; fails even on the solar system
  (Jiménez-Pérez & Laskar 2011). Expected speed-up ≤ 1.
- **Parareal on secular/Kozai evolution.** Already ~16 µs per evolution; nothing to gain.
- **Optimistic PDES (Time Warp).** Infinite-range coupling → cascading rollbacks; round-off makes reverse computation
  impossible; destroys reproducibility. Even hard spheres only reach ∝ P^0.5 (Miller & Luding 2004).
- **Per-pair asynchronous variational integrators.** Serial priority queue, loses GPU batching.
- **A general AMT rewrite (Charm++, HPX, Legion, StarPU).** Minimum efficient task ~100 µs (Task Bench) vs our 1-10 µs;
  our control is already replicated. Take the techniques, not the runtime.
- **Distributing tiny levels over MPI ranks.** Collective + skew > 4x on the kernel below ~6-30k particles (AUDIT row 8);
  every code in §4 keeps small steps communication-free.
- **CUDA dynamic parallelism** (slow, CDP2 cannot wait for children, no HIP) and **Kokkos tasking** (deprecated 4.5).
- **One megakernel holding force + GBS + PN + mergers** (Laine+2013: divergence, register hot spots; Whippletree §7:
  heaviest procedure sets occupancy). Keep the O(N²) kernels lean.
- **Per-call or per-step CUDA graph recapture** (creation ≈ launch cost; PyGraph: 25 % of graphs hurt).
- **Managed/unified memory on A100** (page migration; 9-18 µs faults on MI300A, A100 cost UNVERIFIED). Unified memory is
  an MI300A tool only, and even there keep hot GPU arrays in `hipMalloc`.
- **MPI-3 RMA on GPU windows** (UCX: partial GPU-memory support for RMA) and **Kokkos Remote Spaces** as a production
  dependency (experimental, no releases).
- **A one-process-per-node, multi-GPU rewrite**: touches every Kokkos call; IPC/NCCL + node windows + distributed
  subsystems give the same benefits.
- **Persistent MPI collectives as a GPU fix** (libnbc schedule only; same device copies).
- **Timing-driven device or rank choices** that change arithmetic (GADGET-4 §6.1): make every switch count-based.
- **Atomics on doubles** in any new parallel stage (non-reproducible sums); sum per-item values in index order.

---

## 8. Open questions to measure before building

1. How many CC components per call at depths >= 7, their size distribution, and how many contain subsystems (sets the
   N_host threshold for #2 and the batch width for #3(b)). The audit's row 1a hypothesis (pairs from different binaries
   driven down together at f_bin = 1) belongs here.
2. Per-call host-path cost of a 50-particle component on one raven core and one MI300A core, including detection and
   one LogH advance.
3. `max(gbs_per_sub)/mean` per subsystem phase (#6, #10).
4. Fraction of the step in which the slow set's work at depths 0-4 could overlap with the fast recursion (upper bound for
   #3(a)): sum over calls of min(slow time, fast time).
5. IPC gather micro-benchmark on one raven node at 35k / 200k / 1e6 vs the AUDIT host-staged numbers.
6. The viper Kokkos arch flag.

---

## 9. References (verified unless marked; file names in `/raven/u/uli/papers/async_parallel_time/`)

**Parallel-in-time.** Lions, Maday & Turinici 2001, CRAS 332, 661, DOI 10.1016/S0764-4442(00)01793-6 · Gander &
Vandewalle 2007, SISC 29, 556, DOI 10.1137/05064607X · Gander & Hairer 2008, LNCSE 60, 45, DOI
10.1007/978-3-540-75199-1_4 [gander_hairer2008_nonlinear_parareal] · Gander & Hairer 2014, JCAM 259, 2, DOI
10.1016/j.cam.2013.01.011 [gander_hairer2014_parareal_hamiltonian] · Dai, Le Bris, Legoll & Maday 2013, ESAIM M2AN 47,
717, arXiv 1011.6222 · Bal & Wu 2008, LNCSE 60, 401, DOI 10.1007/978-3-540-75199-1_51 (not read) · Saha, Stadel &
Tremaine 1997, AJ 114, 409, astro-ph/9605016 · Jiménez-Pérez & Laskar 2011, arXiv 1106.3694 · Ariel, Kim & Tsai 2016,
SISC 38, A3540, arXiv 1503.02094 · Legoll, Lelièvre & Samaey 2013, SISC 35, A1951, arXiv 1204.5926 · Ruprecht 2018,
CVS 19, arXiv 1701.01359 · Gander 2015, "50 years of time parallel time integration", DOI 10.1007/978-3-319-23321-5_3
[gander2015_50years_pint] · Gander, Wu & Zhou 2025, Acta Numerica 34, arXiv 2503.13526 · Ong & Schroder 2020, CVS 23,
DOI 10.1007/s00791-020-00331-4 (not read) · Dutt, Greengard & Rokhlin 2000, BIT 40, 241, DOI 10.1023/A:1022338906936 ·
Emmett & Minion 2012, CAMCoS 7, 105, DOI 10.2140/camcos.2012.7.105 · Speck+2012, SC12, DOI 10.1109/SC.2012.6 (numbers
UNVERIFIED) · Speck+2014, LNCSE 98, 637, DOI 10.1007/978-3-319-05789-7_61 [speck2014_DD21_nbody_SDC_PFASST] ·
Muralikrishnan & Speck 2024, arXiv 2407.00485 · Falgout+2014, SISC 36, C635, DOI 10.1137/130944230 · Danieli &
MacLachlan 2021, arXiv 2104.09404 · Christlieb, Macdonald & Ong 2010, SISC 32, 818, DOI 10.1137/09075740X; Christlieb+
2015, arXiv 1310.6331; Ong, Haynes & Ladd 2016, ACM TOMS, arXiv 1408.3082 · Samaddar, Newman & Sánchez 2010, JCP 229,
6558, DOI 10.1016/j.jcp.2010.05.012 · Samaddar+2019, CPC 235, 246, DOI 10.1016/j.cpc.2018.08.007 (metadata) · Berry+2012,
JCP 231, 5945, DOI 10.1016/j.jcp.2012.05.016 · Baffico+2002, PRE 66, 057701 · Legoll, Lelièvre & Sharma 2022, SISC 44,
B146 · Gorynina+2022, arXiv 2212.10508 · Pentland+2022, stochastic parareal, arXiv 2106.10139 · Kreienbuehl+2017,
CAMCoS 12, 109, arXiv 1509.01572 · Ketcheson & bin Waheed 2014, CAMCoS 9, 175, arXiv 1305.6165.

**Asynchronous / multirate / PDES / HHS.** Gear & Wells 1984, BIT 24, 484 · Diaz & Grote 2009, SISC 31, 1985 · Grote &
Mitkova 2013, arXiv 1205.0654 · Günther & Sandu 2016, Numer. Math. 133, 497 · Sarshar, Roberts & Sandu 2019, arXiv
1804.07716 · Lew, Marsden, Ortiz & West 2003, ARMA 167, 85 [lew2003_AVI_ARMA]; 2004, IJNME 60, 153 · Kale & Lew 2007,
IJNME 70, 291 (numbers UNVERIFIED) · Vouga+2009, arXiv 0907.0706 · Chandy & Misra 1979, IEEE TSE 5, 440 · Jefferson
1985, ACM TOPLAS 7, 404 · Fujimoto 1990, CACM 33(10), 30 · Lubachevsky 1989, CACM 32, 111; 1991, JCP 94, 255 · Miller &
Luding 2004, JCP 193, 306, arXiv physics/0302002 · Tang, Perumalla, Fujimoto, Karimabadi+ 2005, PADS
[tang2005_pads_optimistic_physical_reverse] · Pelupessy, Jänes & Portegies Zwart 2012, New Astron. 17, 711, arXiv
1205.5668 · Jänes, Pelupessy & Portegies Zwart 2014, A&A 570, A20, arXiv 1407.7105 · Rantala, Naab & Springel 2021
(FROST), MNRAS 502, 5546, arXiv 2011.14984 · Rantala+2023 (BIFROST), MNRAS, arXiv 2210.02472 · Goodman, Heggie & Hut
1993, ApJ 415, 715.

**Runtimes and codes.** Kale & Krishnan 1993, OOPSLA · Acun+2014, SC14, DOI 10.1109/SC.2014.58 · Jetley+2008, IPDPS ·
Jetley+2010, SC10 · Menon+2015, arXiv 1409.1929 · Rengasamy & Vadhiyar 2020 (G-Charm), arXiv 2008.05712 · Marcello+2021,
MNRAS, arXiv 2101.08226 · Daiß+2019, SC19, arXiv 1908.03121 · Daiß+2021, IPDPSW, DOI 10.1109/IPDPSW52791.2021.00066 (not
read) · Daiß+2022, P3HPC, arXiv 2210.06438 · Daiß+2022, ESPM2, arXiv 2210.06439 · Daiß+2023, IWOCL, arXiv 2303.08058 ·
Daiß+2025, IJHPCA, arXiv 2412.15518 · Bauer+2012, SC12 · Slaughter+2015 (Regent), SC15 · Slaughter+2017, SC17 · Lee+2018,
SC18 · Slaughter+2020 (Task Bench), SC20, arXiv 1908.05790 · Yadav+2025, ASPLOS (DOI UNVERIFIED) · Augonnet+2011, CCPE
23, DOI 10.1002/cpe.1631 · Bosilca+2013, CiSE 15(6) (not read) · Agullo+2014, SISC 36, arXiv 1206.0115 · Schaller+2016,
PASC, arXiv 1606.02738 · Schaller+2024, MNRAS 530, 2378, arXiv 2305.13380 · Borrow+2018, arXiv 1807.01341 · Gonnet,
Chalk & Schaller 2016, arXiv 1601.05384 · Springel+2021 (GADGET-4), arXiv 2010.03567 · Springel 2005, astro-ph/0505010 ·
Wang+2015 (NBODY6++GPU), arXiv 1504.03687 · Wang+2020 (PeTar), arXiv 2006.16560 · Trott+2022 (Kokkos 3), IEEE TPDS 33,
805 (not read) · Blumofe & Leiserson 1999, JACM 46, 720 · Warren & Salmon 1993, SC93 · Cybenko 1989, JPDC 7.

**GPU and node.** Aila & Laine 2009, HPG [aila2009hpg] · Gupta, Stuart & Owens 2012, InPar
[gupta2012_persistent_threads] · Laine, Karras & Aila 2013, HPG [laine2013hpg_megakernels] · Steinberger+2014
(Whippletree), ACM TOG 33(6) · Tzeng, Patney & Owens 2010, HPG (abstract) · Chen+2021 (Atos), arXiv 2112.00132 ·
Cheng+2025 (MPK), arXiv 2512.22219 · Gray 2019, NVIDIA blog "Getting started with CUDA Graphs" · Ekelund, Markidis &
Peng 2025, arXiv 2501.09398 · Ghosh+2025 (PyGraph), arXiv 2503.19779 · Wang & Yalamanchili 2014, IISWC (abstract) ·
Utkarsh+2024 (DiffEqGPU), arXiv 2304.06835 · Nagy, Plavecz & Hegedűs 2022 (MPGOS), arXiv 2011.01740 · Niemeyer & Sung
2014, arXiv 1309.2710 · Balos+2021, arXiv 2011.12984; Balos+2024, arXiv 2405.01713 · Dongarra+2017 (batched BLAS)
[dongarra2017_batched_blas] · Grimm & Stadel 2014 (GENGA), arXiv 1404.2324 · Páll+2020 (GROMACS), arXiv 2006.09167 ·
Doijade+2025 (GROMACS NVSHMEM), arXiv 2509.21527 · Hamidouche+2025 (NCCL GIN), arXiv 2511.15076 · Namashivayam+2022,
arXiv 2208.04817; 2023, arXiv 2306.15773 · Bridges+2024, arXiv 2406.05594 · Unat+2024 (GPU-centric communication
survey), arXiv 2409.09874 · Wahlgren+2025 (MI300A unified memory), arXiv 2508.12743 · Schieffer+2025 (inter-APU),
arXiv 2508.11298 · Tandon+2024 (MI300A porting), arXiv 2405.00436 · Li+2019 (GPU interconnects), arXiv 1903.04611 ·
Hoefler+2013 (MPI+MPI) [Hoefler2013_MPI+MPI] · Rabenseifner, Hager & Jost 2009 [Rabenseifner2009_hybrid_PDP] ·
Nyland, Harris & Prins 2007, GPU Gems 3 ch. 31 · Hamada & Iitaka 2007, astro-ph/0703100 · Belleman, Bédorf &
Portegies Zwart 2008 (Kirin), arXiv 0707.0438 · Gaburov, Harfst & Portegies Zwart 2009 (Sapporo), arXiv 0902.4463 ·
Bédorf, Gaburov & Portegies Zwart 2015 (Sapporo2), arXiv 1510.04068 · Nitadori & Aarseth 2012, arXiv 1205.1222 ·
Ayguadé+2009, IEEE TPDS 20, 404 · Graham 1969, SIAM J. Appl. Math. 17, 416 · Collange+2015, Parallel Comput. 49, 83
[Collange2015_reproducible_reduction] · Lin & Grover 2018, NVIDIA blog "Using CUDA warp-level primitives".

**MPI, collectives, PGAS.** Open MPI v5.0.x source: `ompi/mca/coll/cuda/coll_cuda_module.c`,
`ompi/mca/coll/tuned/coll_tuned_decision_fixed.c`, `ompi/mca/coll/base/coll_base_allgatherv.c`,
`ompi/mca/pml/ucx/pml_ucx_datatype.c`, `opal/mca/accelerator/cuda/accelerator_cuda.c`, `ompi/mca/coll/libnbc/nbc_iallgatherv.c`
(github.com/open-mpi/ompi) · Open MPI docs: tuning-apps/accelerators/cuda §11.3.5.4; tuning-apps/collectives/ucc ·
UCX v1.20 `cuda_ipc_iface.c`; UCX FAQ (openucx.readthedocs.io) · UCC: github.com/openucx/ucc (TL/CUDA, TL/NCCL);
Venkata+2024, Hot Interconnects (not read) · NVIDIA HPC-X known issues (networking-docs.nvidia.com/hpcxum/225) ·
Gayatri+, CUG 2025 (Perlmutter MPI collectives) · Pearson+2021 (TEMPI), arXiv 2012.14363 · Gencer+2026 (ucTrace),
arXiv 2602.19084 · Rolinger+2018, arXiv 1812.05964 · Awan+2017, arXiv 1707.09414 · Hu+2025 (Demystifying NCCL), arXiv
2507.04786 · MSCCL++, arXiv 2504.09014 · Demystifying NVSHMEM, arXiv 2606.05951 · NCCL user guide (collectives, p2p,
CUDA graphs, MPI) · Choi+2021 (Charm++ GPU-aware communication), arXiv 2102.12416 · Kokkos Remote Spaces
(github.com/kokkos/kokkos-remote-spaces); Mishler+2023, DOI 10.1109/CLUSTERWorkshops61457.2023.00028 · Kokkos Comm
(github.com/kokkos/kokkos-comm) · Kokkos MultiGPUSupport (kokkos.org/kokkos-core-wiki/API/core/MultiGPUSupport.html) ·
rocSHMEM (github.com/ROCm/rocSHMEM) · UPC++ (bitbucket.org/berkeleylab/upcxx; Bachan+2019, IPDPS, not re-checked) ·
Rizzuto+2021, MNRAS 501, 5257.
