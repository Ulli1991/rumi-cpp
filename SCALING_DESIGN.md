# cpp-nbody scaling architecture: deep-level batching and task parallelism across ranks

Design document. Nothing in the repository was changed. Code references are to main at 4f71d62.

Status of the inputs:
- "Measured" numbers come from AUDIT_2026-10.md section G, or from the fits given in the brief.
- "Estimate" marks my own reasoning.
- UNVERIFIED marks literature claims I could not check in the source. The local PDFs could not be opened (no pdftotext or pdftoppm on the node), so the paper citations below rest on the LaTeX sources in `rantala/src/`, the arXiv HTML pages, and the summaries already in HARD_PROBLEMS.md and METHODS_SURVEY.md.

---

## 0. Summary

1. **The s term is not one thing.** The fit T = s + p/n gives s ≈ 8 s at 200k. It breaks down roughly as follows (estimate, built from the G.2 measurements):
   - about 4.0 s fixed cost of the replicated deep and mid levels (depths ≥ 5, n ≤ 6000);
   - about 1.5 s host subsystem integration (LogH, SDAR, secular);
   - about 0.6 s detection at the distributed levels;
   - about 0.9 s MPI collectives plus rank skew;
   - about 0.5 s profiler fences;
   - about 0.2 s energy check.

   At 1e6, s ≈ 15 s, and 4.7 s of that is the energy check alone. That check is replicated O(N²) work and must be split first; it is a Phase-0 item.

2. **Without HHS_CC, the deep levels are a serial chain in time, not a breadth of independent work.** The ~2000 small calls per step are mostly the 2^d repetitions of one shrinking fast set, each depending on the one before. So:
   - batching (design 1) can only cut the **cost per call**;
   - extra ranks (design 2) only help if there is independent work.

   The design uses three sources of independent work:
   - **(a) HHS_CC components.** The components of a level do not interact during their sub-steps.
   - **(b) S/F commutation.** At *every* HHS level, the slow half-step fsi(S, τ/2) and the fast recursion hhs(F, τ/2) act on disjoint particles with no kick between them. They commute exactly, so they can run concurrently with no change to any arithmetic. This holds even without CC. It is also where the host LogH work sits, because subsystems are advanced inside fsi_step.
   - **(c) The subsystem batches** inside one drift.

3. **Bit-identity with today is achievable for design 1 and for most of design 2**, with these exceptions:
   - Intra-rank and inter-rank concurrency run the same kernels and the same host code on the same data. Global side effects (energy books, collision lists, subsystem ids, compaction, counters) are merged in the serial (canonical) order.
   - The few one-time bit changes are listed explicitly (§6.3), each with its 3-seed check.
   - Host/device bit-identity is **impossible**: the device uses CUDA's approximate `rsqrt` (`constants.h:30-38`) and the host uses `1.0/std::sqrt`. So "host-resident small subtrees" are a one-time bit change. They stay deterministic because the placement depends only on set size, and they are kept behind a switch.

4. **Rank-count independence is reachable, and it is the recommended determinism target.** The present rank-count dependence most likely comes from the sink-sliced kernels. Each rank sees only its slice of ni sinks. Two things depend on ni:
   - the source-chunk count (`n2_driver.h:44`, `chunk_count(ni, nj, TS)`);
   - the tile size (`n2_driver.h:187-188`: 256 for ni ≥ 131072, otherwise 128). The tile size also moves the Kahan block boundaries.

   A sink's sum therefore depends on how many ranks share its set. Making both quantities functions of the global set makes every sink's result independent of the rank count; this hypothesis is to be confirmed in Phase 0. Together with canonical-order merging and no floating-point collectives, 1, 2, 4, …, 64 GPUs then give **bitwise identical** runs. Tested as `rankcount_*` CTest cases.

5. **Predicted step times** (§7; ranges in the text):

   | case | today | after (1) | after (1)+(2) | with FMM |
   |---|---|---|---|---|
   | 200k, 4 / 16 GPUs | 11.3 / ~9.9 s | 7.6 / 6.0 s | 6.0 / 4.1 s | — |
   | 1e6, 4 / 64 GPUs | 31.8 / ~16.9 s | 22.9 / 8.0 s | 20.0 / 4.5 s | 4.6 / 3.2 s |
   | 1e7, 64 / 256 GPUs | — | 87 / 35 s | 77 / 25 s | 8.9 / 8.8 s |

   For 1e7, "after (1)" and later assume the energy check is already split.

   Conclusions:
   - (1)+(2) gives about 3.7× at 1e6 on 16 nodes.
   - With FMM the s term is everything. FMM pays only once (1)+(2) exist.
   - 1e7 becomes feasible only with FMM plus (1)+(2): about 10 Myr/day on 16 nodes, at first-interval speed.

6. **GPU utilisation (new first-class metric).** Today the GPU is busy 34–38 % of the step (nsys). Target: ≥ 65 % at 200k on 4 GPUs (same nsys setup), by these steps:
   - (1) removes most of the 4.1 s of launch/sync gaps;
   - S/F and component concurrency hide host LogH and detection behind device work;
   - balanced tasks remove most of the rank-skew waits.

   All three are bit-identical. The idle budget and the remedies are in §5.

7. **Communication.**
   - Device data (joins, slices): NCCL on the Kokkos stream. Allgather is pure data movement, so it is bit-exact.
   - Control data: MPI on the host.
   - FC-III-style node-local MPI shared windows (`MPI_Win_allocate_shared`): host mirrors, host-staged fallback, and the task counter.
   - No floating-point MPI or NCCL reductions anywhere; all sums are done locally in canonical order.
   - NVSHMEM is deferred until measurements show that join latency dominates.

---

## 1. Background in plain terms

**HHS** (hierarchical Hamiltonian splitting) is AMR subcycling in time with refluxing built in. At level d, a set of particles is split into slow (S, step ≥ pivot) and fast (F). The sequence (FROST Alg. 4, `hhs_fsi_recursive`, `integrator.cpp:2324`) is:

```
kick_SF(τ/6) ; fsi(S, τ/2) ; hhs(F, τ/2) ; kick_SF+grad(2τ/3) ; hhs(F, τ/2) ; fsi(S, τ/2) ; kick_SF(τ/6)
```

The recursion on F repeats this with pivot τ/2. At the deepest levels, F holds 2–50 stars, typically binaries and their perturbers, recomputed 2^d times per top-level step.

**Subsystems** (binaries, triples, groups of up to 8) are sub-grid objects with their own integrator: LogH or SDAR with GBS, or secular/Kepler. They run on host threads and are advanced inside each fsi drift (`advance_subsystems_in_range`, `integrator.cpp:475`).

**HHS_CC** (Jänes+2014): if the fast set falls apart into connected components (graph edge = pair step < pivot), each component recurses on its own. The inter-component forces move to the parent's kicks (`hhs_cc_step`, `integrator.cpp:2232`; `cc_components`, `forces.cpp:455`). Today it is off by default and applies only to sets ≤ 16384 (`integrator.cpp:2358-2363`). Those sets are deliberately replicated, "so that the integration does not depend on the number of ranks".

**Why direct collisional codes stop at a few hundred GPUs:**
- **FROST §6.3:** scaling stalls at N_GPU^max ≈ 4 N/1e5, "as the particle number per GPU decreases and becomes smaller than the number of concurrent threads". Verified in the LaTeX, l.824-832.
- **Rantala et al. 2023 §4:** scaling good to ≈ 40 N/1e6. 1e5 stalls at ~8 GPUs, 1e6 is "not ideal" beyond 48, 5e6 tested to 128 (LaTeX l.797-816). Their cost model is T ≈ T_N² N² + T_bin f_bin N (l.802-810). Their replicated, sink-sliced scatter-compute-gather is what our `mpi_comm.h` copies.
- **NBODY6++GPU (1504.03687 §4, via HARD_PROBLEMS §2.6):** saturates at 8–16 nodes, with MPI and data movement about half the time at 16 nodes. The KS part has a compute/communication ratio < 100.
- **NbodyCP (2511.06181):** "near ideal" short tests to 16M stars on 16 raven nodes (via HARD_PROBLEMS update). The PDF text was not extractable, so mechanism and numbers are UNVERIFIED. It is a neighbour-scheme code, where the rarely updated regular force is the bulk.
- **PeTar (2006.16560, HTML §6.3, §6.5, §7.2, read):**
  - Short-range clusters are isolated and "can be integrated in parallel".
  - A cluster spanning MPI processes gets "one MPI process … chosen to be the host for the cluster and others send particle data to it" (owner-computes).
  - Clusters run on OpenMP inside each process.
  - Scaling flattens at ~160 cores at 1e5 and ~640 at 1e6.
- **FC-III (2601.07917, LaTeX l.158):** the Rantala et al. code uses "node-local MPI shared memory windows which enable multiple MPI ranks to access the same single copy of the ensemble on a shared memory compute node, reducing the volume of the network communication and the number of messages".
- **FDPS (1601.03138), Taichi (2012.02207, 2205.12289):** only abstracts were readable. Taichi is OpenMP + AVX with HHS + FMM. METHODS_SURVEY §I reports 100× over direct at 1e6 at ε = 1e-7, and HHS-FSI + AR-chain in 2205.12289. Details of their level parallelisation: UNVERIFIED.
- **Pelupessy+2012 / Jänes+2014 (via METHODS_SURVEY §A.3):** CC splitting is about 2× faster in binary-rich systems and stays flat through core collapse.

The common pattern: everyone parallelises the O(N²) far part well, and the few-body/deep part is either serial (NB6++ KS) or parallel over independent clusters (PeTar, Rantala et al. 2023 LogH). Our design takes the second road and extends it to whole HHS subtrees.

---

## 2. Where the time and the GPU idle go today

### 2.1 Decomposition of s (estimate from G.2)

| part | 200k, 4 GPUs (s = 8.0) | 1e6, 4 GPUs (s = 15) | scales with n? |
|---|---|---|---|
| deep + mid levels (fixed cost per call, replicated) | ~4.0 (d5-7 1.7 + d8-11 2.3) | ~4.5 (d9-20) + mid | no |
| host subsystem integration (LogH/SDAR/secular) | ~1.5 | ~3 (of 5.1) | no (replicated) |
| detection at distributed levels (NgbK replicated, host detect) | ~0.6 | ~0.8 | no |
| MPI collectives + skew | ~0.9 | ~1.4 | grows |
| profiler fences | ~0.5 | ~0.5 | no |
| energy check | 0.2 | **4.7** | no (replicated O(N²)) |

### 2.2 GPU idle budget (200k, 4 GPUs, nsys, rank 0)

nsys slows the host side by ~35 % (nsys step ≈ 15.3 s versus 11.3 s), but not kernel time. All numbers below are therefore in nsys seconds, and the metric is always taken under the same nsys settings.

| category | nsys s/step | source |
|---|---|---|
| GPU busy | ~5.5 (36 %) | DtK 3.0, ForceK 1.4, GradK 0.7, rest 0.9 |
| short gaps 10–100 µs (launch + sync between tiny kernels) | 4.1 | measured |
| long gaps > 1 ms, total | 6.2 | measured |
| ↳ MPI waits incl. skew | ~2.0–2.5 | mean per rank 2.7–3.4, collectives themselves 0.9–1.15 |
| ↳ host subsystem integration | ~2.0 | Subsystem integ 2.57 s minus device pack |
| ↳ host detection and bookkeeping (detect_from_pairs, pid maps, flag restore, mergers) | ~1.0–1.3 | detect 1.05 + bookkeeping 0.3 |
| energy check | busy, not idle, but 3/4 of it redundant at 4 GPUs | 0.21 |

In an un-instrumented step the busy fraction is about 5.5/11.3 ≈ 45–50 %, because nsys does not inflate kernel time.

Define two metrics:
- **U** = GPU busy time / step wall time (nsys, per rank).
- **U_unique** = unique kernel time summed over ranks / (n × wall). Replicated kernels count once.

At 4 GPUs today, U_unique ≈ 25 % (estimate). Design 2 raises U_unique more than U.

---

## 3. Design (1): deep-level batching (removing the fixed cost per call)

### 3.1 What a deep call costs and what is mandatory

Each `hhs_fsi_recursive` call issues 60–90 launches and 25–40 host syncs, whatever n is (G.3 row 1). The sync points include:
- neighbour-count readback, `forces.cpp:429`;
- the pair download in detection;
- the clamp `parallel_reduce`, `integrator.cpp:2499`;
- `partition_by_pivot`: fence plus 2 scalar copies, `:1514-1519`;
- the isolated-set mirror, `:2567`;
- `update_particle_flags`: upload plus a reset over **all N**, `subsystem.cpp:762-782`;
- the subsystem mirrors;
- profiler fences.

Measured costs: launch + fence 9.7 µs, scalar D2H 13.7 µs, reduce-to-host 18.5 µs. So 30 × ~15 µs + 75 × 3.4 µs + host work ≈ 0.6–2 ms per call, as measured.

Host round trips that are **mandatory** as long as detection and LogH stay on the host:
- (i) neighbour pairs to detection, only if pairs exist;
- (ii) subsystem advances in fsi_step, only if eligible subsystems exist;
- (iii) the partition count that drives the recursion.

Everything else can be folded. Floor: ~3–6 syncs + ~5 graph launches + ~50 µs of host logic ≈ **0.15–0.25 ms per call** (estimate).

### 3.2 Components (all bit-identical)

**B1. Control block.** One pinned host struct per call, written by device kernels and read with **one** D2H after each device segment. It carries:
- neighbour count, slow_count, n_clamped, ghost flag;
- the "single subsystem" test, i.e. the subsystem ids of a set ≤ 8, instead of the mirror at `:2567`;
- the CC edge count.

This replaces the ~6 scalar copies and the fences around them. Same values, so bit-identical.

**B2. Device-side pid→slot map and range-local flags.**
- `apply_permutation` and the CC permutation update `map[pid[i]] = i` on the device. This replaces the incremental host `rebuild_pid_to_idx` and its D2H.
- `update_particle_flags` resets only the range it serves, instead of an O(N) reset per call. Today that is a 1e6-element kernel thousands of times per step at 1e6.

Integers only, so bit-identical. Check mode: `NBODY_CHECK_PIDMAP`, plus a new `NBODY_CHECK_FLAGS` that compares against a full recompute.

**B3. Fused small-set kernels for n ≤ 512.** One team per set does in one launch:
- time-step pairs (the DtK expressions);
- gradient/v-a criteria and level-0 clamp;
- partition: a team scan with the same stable order as `mark_prefix_slow` plus `build_perm_deterministic`;
- the control block.

Bit-identity argument: `run_tiled` with TS = 128 and ni ≤ 512 uses one source chunk (`chunk_count` gives n = 1 because nj/(4·TS) < 1). Each sink accumulates over sources in tiles of 128, with Kahan flushes at tile boundaries. A single-team kernel looping over sink blocks with the same tile loop and flush rule reproduces each sink's sum exactly. Minima are order-free. Check mode `NBODY_CHECK_BATCH=1` runs the old path on a copy and memcmps dt, ax/ay/az, partition and flags, in the style of `NBODY_CHECK_TREE_DT`.

**B4. Kokkos Graph segments.** The installed Kokkos is 5.0.99 at `/u/uli/kokkos-cuda-mpi`. Its headers provide:
- `Kokkos::Experimental::create_graph` / `Graph`;
- `then_parallel_for`, `then_parallel_reduce` (into a View), `then_host`;
- `cuda_capture` / `hip_capture`, which record arbitrary existing Kokkos launches on an execution-space instance into the graph;
- `instantiate()`, `submit(exec)`, `native_graph()` / `native_graph_exec()`.

It has **no** `then_parallel_scan` and no parameter-update API (only native `cudaGraphExecUpdate` via `native_graph_exec()`).

Design:
- One graph per fixed segment between host points:
  - "neighbour search + count";
  - "time step + criteria + clamp + partition + control block";
  - fsi pieces: "kick, self force" / "drift field, self force, gradient, kick" / "self force, kick";
  - cross kick.
- Graphs are cached per (segment, size class: n ≤ 32/64/128/256/512).
- Kernels read (start, count, dt, pivot) from a **device descriptor**, written by a tiny H2D copy or by the previous kernel, so one instantiation serves all calls of a class.
- Scans are captured with `cuda_capture`.
- Buffers whose `realloc` would invalidate captured pointers get monotone capacity and a version number. A growth triggers re-instantiation.
- Graphs and host LogH never share a node: `then_host` nodes cannot call CUDA/Kokkos.

Literature: GTC 2020 (A100, empty kernels, 32-node graph) shows ~15–22× lower host launch cost and ~2× lower device-side time than stream launches. I found no clean per-node A100 µs figure, so a Phase-0 micro-benchmark is needed.

**B5. Persistent staging everywhere.** Extends the F2 fix: no View allocation in deep calls (24 µs each), and LogH pack from the range mirror.

**B6. Optional, not bit-identical: host-resident small subtrees** (n ≤ 32–64, switch `HOST_SUBTREE_MAX`).
- Copy the set's full record to the host once (≈ 50 × 280 B), run the whole sub-recursion including detection and LogH on one host thread, and copy back once.
- At n = 16 an O(n²) host loop costs ~5 µs, about 30× below the best device path.
- **Cannot be bit-identical** because of `rsqrt` versus `1/sqrt` (and FMA contraction). It is deterministic and rank-count independent because placement depends only on n.
- One-time 3-seed check.
- On MI300A, `small_for` and the `run_n2` host path already mix host and device arithmetic by size, so this is not a new kind of behaviour there.
- Audit row 14 (+38 %) does not contradict this: that CPU path round-trips at every level; this one does not.

**B7. Optional later: device-resident "pure-gravity" subtrees.** A one-team kernel runs the whole recursion of a set with no subsystems and no neighbour pairs, with an explicit stack and an escape to the host when detection finds pairs. Bit-identical (same device expressions). Payoff depends on how many deep sets have no subsystems; the Phase-0 census decides.

### 3.3 Expected effect

Deep fixed cost:
- 200k: from ~1 ms to ~0.2 ms per call, about −2.7 s/step.
- 1e6: 4.5 s to ~1.4 s.
- 1e5 with f_bin = 1, where deep calls are 47 % of the step: about −2 s of 5.96 s.

Plus the Phase-0 items. Short gaps 4.1 s to ≤ 1.0 s (nsys).

---

## 4. Design (2): task parallelism across ranks for the replicated levels

### 4.1 What is independent

| task type | where | independent because | data it touches |
|---|---|---|---|
| **T_S**: fsi(S, τ/2) of a level | every level | H_S and H_F act on disjoint coordinates; no kick between them | slow range; subsystems wholly inside it |
| **T_F**: hhs(F, τ/2) subtree | every level | same | fast range |
| **T_C(c)**: component c's sub-recursion in one `recurse_all` | CC levels (count ≤ cc_max) | CC: inter-component forces are moved to the parent's kicks | the component's contiguous range |
| **T_L**: LogH/SDAR batch slices, secular batches | inside every drift | subsystems are disjoint | member ranges |

T_S and T_F also exist **without** CC, but they give only 2-way parallelism per level, and T_F is the critical path.

T_S/T_F commute because neither reads the other's particles:
- `fsi_step(slow)` uses slow sources only: `self_acceleration(p, start, count)`, and only subsystems fully inside the range;
- the fast recursion permutes only inside the fast range.

The fast range is always the suffix of the parent range (comment at `integrator.cpp:2336`), so a compaction triggered in F does not move S.

### 4.2 Execution model

**TaskContext** (the refactor everything rests on). Every piece of mutable global state the recursion uses becomes per-context:
- the scratch Views that are function-local `static`s today: `cid`/`snx`/`sny`/`snz` (`:2235`), `cc_buf`/`dmin` (`forces.cpp`), `RangeMirror` buffers, `DeviceStaging::get<k>`, `g_logh_batch`, the n2 `partial_buffer`, the neighbour buffers in `g_subsystem_mgr`;
- an execution-space instance (`Kokkos::Cuda(stream)`);
- deltas of `g_energy_books`, `g_pending_collisions`, merger-tree records, `g_hhs_counters`, `g_subsys_phase`, the `step_*` statistics;
- the subsystem-manager delta (new subsystems, persistent LogH state of advanced subsystems, removed pids);
- a deterministic id allocator.

A task is a pure function: (state of its range at the wave start) → (new state of its range, plus a delta record). The same code serves serial, intra-rank-concurrent and inter-rank execution.

**Intra-rank concurrency (bit-identical).**
- 2–4 host driver threads per rank, each with a TaskContext and its own CUDA stream.
- A shared host pool for LogH/SDAR (the 18 cores per GPU, partitioned).
- At a level: T_S goes to driver B (its LogH on the pool), while driver A continues the T_F chain. The fast set's neighbour search and time step are issued right after the first kick, which is the "pipelined next-level time step".
- CC components go to drivers or streams in LPT order.
- This hides host LogH and detection behind device work, and device gaps behind host work.

**Inter-rank owner-computes.**
- A *task root* is the highest level at which task parallelism is profitable: the first CC split of a set ≤ cc_max with ≥ 2 heavy components, or a T_S whose cost passes a threshold.
- At a task root, each task is assigned to one rank (§4.5).
- The owner runs it on its replica. Below the root, **everything inside the task is local and communication-free**: nested CC levels and deep chains run as plain local recursion or intra-rank tasks.
- Non-owners skip the task, so their copy of that range is stale until the join.
- Rule: no MPI/NCCL call inside a task. Debug guard: the MPI wrappers abort if called with an active task flag.

### 4.3 Data flow at a CC task-root level (`hhs_cc_step`)

```
all ranks (replicated):  cross(false) on S∪F  -> kick(1/6)                  [inputs identical on all ranks]
wave 1:   T_S = fsi(S,τ/2)      T_C(1..K) = recurse(component c, τ/2)       [owner-computes, no communication]
join 1:   allgather of changed ranges (S if owned, each component), task deltas
          -> canonical merge: compaction, books, ids, collisions, active list   [all ranks identical again]
all ranks: restore() ; cross(true) incl. cc_inter_gradient -> kick(2/3)
wave 2:   T_C(1..K) = recurse(component c, τ/2)      T_S' = fsi(S,τ/2)
join 2:   as join 1
all ranks: cross(false) -> kick(1/6)
```

T_S moves into the wave because it commutes with the recursions. That is exact, and the side effects merge in the serial order (wave 1: T_S first, then components; wave 2: components, then T_S).

At ordinary levels (not CC), when the slow set is large enough, T_S is offloaded to another rank (or stream) while the T_F chain continues. Join at the next cross kick.

**Sink-sliced levels (count > 16384)** stay as today, with two changes:
1. **Rank subgroups.** A set of size m is sliced over g(m) = clamp(m / m_min, 1, n) ranks, with m_min ≈ 8k sinks per GPU. The G.3 row 8 break-even was ~6–30k, so this is to be tuned. The other ranks are not forced into latency-bound slices.
2. **S/F concurrency at big levels.** T_S of a big level (its fsi_step, itself sliced) runs on subgroup A while subgroup B starts the fast set's time step and recursion. This lifts the effective N_GPU^max somewhat (estimate 1.2–1.5×; not counted in §7).

Per-sink results do not depend on g once the chunk and tile rule is global (§6), so the subgroup size does not affect bits.

### 4.4 Rejoin

- **Phase-4 first version: full-record join.**
  - Each owner sends its ranges in its final order. That order matters, because the internal partitions permuted them and the next cross kick's source order must equal the serial order.
  - Each particle carries all persistent fields (pid, x, v, m, spin, cv, stellar type/radius, dt_level0, …; about 35 × 8 = 280 B).
  - Volume: ≤ 16k particles per CC set, so ≤ 4.6 MB per join. NCCL over NVLink: ~30–60 µs (estimate); inter-node at ~25 GB/s: ~0.2 ms (UNVERIFIED bandwidth).
- **Delta record per task:**
  - book increments as a *sequence*, replayed in serial order so that (B + d1) + d2 is reproduced;
  - collisions;
  - merger-tree events;
  - new and removed subsystems with their persistent LogH state (as `exchange_subsystem_results` part 2 does now);
  - counters;
  - the measured wall time (for balancing).
- **Canonical merge.** All ranks apply the deltas in task order. Ghost compaction of the CC range is done at the join, as one stable compaction in component order.

  Inside a task, merged-away bodies are moved stably to the end of the task's own range and excluded from its count. The task's sub-calls then see exactly the sequence that the serial global tail shift would give, and other tasks' ranges never move. Bit-identical to serial: to be proven with the MERGERS and coll_* tests under schedule fuzzing (§6.4).
- **Later optimisation:** positions-only joins inside the root (24 B/particle) when only the cross kick needs them, with the full sync at task-root exit. Only if measured to matter.

### 4.5 Load balancing

- **Cost model per task:**
  - components persist from step to step;
  - their measured wall time from the previous step, keyed by the component's minimum pid, is allgathered with the join metadata;
  - a priori (new components): cost ≈ κ_call · 2·pivot/dt_min(c) + κ_pair · c · Σ_i pivot/dt_i + Σ_subsys κ_LogH · n_b² · (1 + n_orb). The dt values are already on the device after `PairMinK` and the time step.
- **Assignment:** LPT (longest first to the least-loaded rank), computed identically on all ranks from the allgathered costs.
- **Work stealing / self-scheduling:**
  - Every rank holds the wave's input state, so any rank can run any task.
  - A node-local MPI shared-window counter (FC-III mechanism; atomic fetch-add, ~100 ns) lets the 4 ranks of a node self-schedule from the LPT-ordered list. Across nodes, use the static LPT, which keeps join sizes predictable.
- **Determinism:** the assignment never affects results, because task results are rank-independent and merged canonically. So timing noise and stealing are harmless. This is the key enabling property.
- **Critical path:** a wave lasts as long as its longest task (e.g. an F3-type long LogH or a PN binary at ~47 µs per GBS step). No balancing fixes that, but SDAR does (M3). The critical task is reported every step.

### 4.6 Communication choices

| data | choice | why |
|---|---|---|
| sink-slice gathers (ax/ay/az, dt), joins | **NCCL** `ncclAllGather` / grouped `ncclBroadcast` on `exec.cuda_stream()` | no host staging, stream-ordered (no host sync), hierarchical NVLink + IB rings; pure copies, so bit-exact |
| fallback | host-staged MPI (G.3 row 4: 1.31 ms vs 3.43 ms device-buffer at 200k) | already in the code (`USE_GPU_AWARE_MPI=OFF`) |
| control (counts, costs, assignment, deltas) | MPI on host | small, irregular |
| node-local host data (mirrors, LogH staging, task counter, host-staged allgather) | **MPI shared windows** (FC-III) | one host copy per node, intra-node exchange through shared memory, one inter-node message per node pair |
| NVSHMEM | deferred | one-sided puts from kernels would let owners write joins directly; latency ~µs (UNVERIFIED). Needs a symmetric heap outside Kokkos (Kokkos Remote Spaces is experimental). Only if join latency dominates after Phase 4 |

Rules:
- Never use NCCL/MPI floating-point reductions. NCCL allreduce order depends on topology and rank count.
- All ranks enqueue NCCL operations in the same order, from one communication thread per rank.

### 4.7 Asynchrony: which barriers are truly needed

| point | needed? | reason / remedy |
|---|---|---|
| after each sink-sliced pairwise evaluation | yes (all-to-all data dependence) | fuse pairs (dt + acc; slow + fast slices): 561 → ~350 per step; NCCL on stream instead of host blocking |
| join before a CC cross kick (2 per task-root call) | yes | the cross kick and `cc_inter_*` need all positions |
| between T_S and T_F of a level | no | they commute; join at the next kick |
| inside tasks / below the task root | none | communication-free by construction |
| replicated levels without tasks | none today | keep it that way |
| step end: energy check | not on the critical path | compute PE asynchronously on a snapshot during the next step; on failure roll back two checkpoints (keep two; buffer the output of the speculative step). The trajectory is unchanged unless the check fails, then behaviour is as today |
| restarts, mergers, escapers count sync | replicated decisions | already deterministic |

Ranks therefore "run ahead" naturally inside their tasks. The HHS kicks are the only true synchronisation points, and they are synchronous by construction: that is what keeps momentum conservation exact.

### 4.8 Composition with a future FMM (M8)

- **Replicated tree, sliced sinks.** Every rank holds all particles, so every rank builds the same tree (O(N); ~1 ms per 1e6 particles on an A100, estimate) and evaluates the far field for its sink slice. Communication stays the same accelerations allgather. The existing tree path already works this way (group ranges + allreduce). For determinism, replace its `allreduce` by an allgather of slices.
- FMM only at levels above a set-size threshold; near field direct and unsoftened; mutual (symmetric) interactions for exact momentum (Dehnen; METHODS_SURVEY §I).
- The O(N²) parts shrink, so p shrinks by F ≈ 15 (1e6) to ~100 (1e7) (estimate; the tolerance from experiment I-1 sets p_order). The s term then dominates even on one node. **(1)+(2) are prerequisites for FMM to pay off.**
- The tree build itself is a replicated s-term: ~0.2 s/step at 1e6, ~1.5 s at 1e7 (estimate).
- The collisional time-step sums (jerk, acceleration) and the gradient must also go through the FMM. Otherwise an O(N²) remainder stays (F7). This changes bits and needs the energy rule.

---

## 5. GPU utilisation as a first-class metric

**Metric.**
- `tools/nsys_busy.py` (new): from an nsys run, take the union of kernel intervals per rank per step, divided by step wall time. Report U, U_unique and the gap histogram (10–100 µs, 0.1–1 ms, > 1 ms), attributed to NVTX ranges.
- Kokkos regions become NVTX ranges through the Kokkos Tools nvtx connector. Add `Kokkos::Profiling::pushRegion` for detect, logh, secular, mpi, join, energy.
- Always the same nsys options, so comparisons are like with like.

**Idle sources and remedies:**

| source (nsys s/step at 200k, 4 GPUs) | remedy | bit-identical? | target exposed idle |
|---|---|---|---|
| launch/sync gaps 4.1 | B1–B5 (control block, fused small kernels, graphs, no allocations), profiler fences conditional | yes | ≤ 1.0 |
| host LogH/SDAR/secular ~2.0 | S/F concurrency + component streams (LogH of T_S on the pool while the GPU runs T_F); inter-rank distribution of T_S/T_C/T_L | yes (canonical merge) | ≤ 0.4 |
| (alternative) split force: field-field part during LogH, member terms after | needs a members-last source order and a resumable Kahan accumulator. Field sinks: field sources then member sources, the same sequence as one pass in members-last order; member sinks recomputed afterwards. Bit-identical *to the members-last baseline*, which is a one-time order change | one-time change, 3 seeds | Phase 5 only; S/F gives most of it without a bit change |
| host detection and bookkeeping ~1.0–1.3 | overlap via S/F; device pid map and range flags (B2); detection stays on host (moving it to the device risks classification flips from host/device arithmetic) | yes | ≤ 0.3 |
| MPI waits and skew ~2.0–2.5 | NCCL on stream; collective fusion; skew removal: replicated host work becomes balanced tasks, so ranks arrive together; host staging as fallback | yes | ≤ 0.6 |
| energy check (busy but 3/4 redundant; 1e6: 4.7 s) | split by rank in fixed row blocks (sum of block sums in block order, so rank-count independent); triangular loop; reuse E_after; asynchronous on a snapshot | trajectory yes; printed dE/E last bits change | ~0 on the critical path |
| LogH on GPU | **not recommended** now. Single-thread FP64 chains (GBS steps of 2–3-body systems) are slower on a GPU thread than on a CPU core; GPU only helps with throughput (thousands of systems per call), which deep calls lack. Reconsider for high-volume secular/Kepler at top levels if J-1 data show host throughput limits | no (host vs device rsqrt) | — |

**Per-phase utilisation targets (200k, 4 GPUs, nsys):**

| stage | wall (nsys) | U | U_unique (est.) |
|---|---|---|---|
| today | 15.3 | 36 % | ~25 % |
| Phase 0 | ~14 | ~39 % | ~28 % |
| + (1) batching | ~10.5 | ~50 % | ~35 % |
| + intra-rank concurrency | ~8.5 | ~62 % | ~42 % |
| + inter-rank tasks | ~7.5 | ≥ 65 % | ~55 % |

By phase of the step:
- sink-sliced levels: ≥ 85 % busy (FP64-bound kernels; gaps only at fused collectives);
- mid levels (n 300–16k): ~40 % → ≥ 70 %;
- deep levels: inherently latency-bound, target 30–40 % (the goal there is wall time, not U);
- host subsystem phases: overlapped, so they no longer appear as idle.

At 16+ GPUs, U falls again (the serial chain dominates). Report U alongside s/step, never alone.

---

## 6. Determinism strategy

### 6.1 Target: rank-count independent (Tier I)

Requirements and fixes:
1. **Chunk count and tile size from the global set.** `chunk_count` and the 256/128 tile switch must be functions of the set being sliced (global ni, nj, GPU type), not of the rank's slice. Single-GPU runs stay bit-identical; multi-GPU runs change once.
2. **No floating-point collective reductions.** Allgather only; local sums in canonical order. This includes the energy check and the tree path's allreduce.
3. **Canonical-order merges.**
   - `exchange_subsystem_results` sums book deltas in **rank** order (`integrator.cpp:271-273`); change to subsystem/task order.
   - `allgather_collisions` is in rank order (`:286`); sort to canonical order.
4. **Deterministic subsystem ids.** `next_id++` (`subsystem.cpp:544, 668`) follows execution order. Use per-task id ranges renumbered canonically at the join, or ids derived from the minimum member pid. Ids only need equality and persistence; check `subsystem.cpp:500-670` before choosing.
5. **Task results independent of the executing rank.** Same code path, no dependence on buffer capacities or thread counts. LogH per-subsystem results are already thread-independent and its stats are summed in index order; verify.
6. **Deferred, canonical compaction** (§4.4).

Not targeted: bitwise A100 versus MI300A identity (`chunk_count` depends on device concurrency; rsqrt/FMA differ by backend).

### 6.2 What stays bit-identical with today

All of design 1 except B6. Intra-rank and inter-rank task execution (given the canonical merge). Phase-0 items, except the printed dE/E bits (energy check order).

### 6.3 One-time bit changes (each with the 3-seed energy rule: seeds 42, 7, 13; 200k with 10 % binaries, 20–35 steps; plus the 20k host reproduction)

1. Chunk/tile rule: multi-GPU runs only.
2. HHS_CC on by default (experiment A-2). It is a different splitting, so a physics change. It is a prerequisite for most of the task volume.
3. Book accumulation restructured to per-task deltas, if exact replay is not used. Affects diagnostics only; the trajectory is unchanged.
4. Deferred compaction, if it turns out not to be exact.
5. B6 host subtrees (switch).
6. Split-force overlap with members-last order (Phase 5).

### 6.4 Tests (REPLICA_CHECK style)

- **`rankcount_*` CTest cases** (`NBODY_TEST_LAUNCHER` with mpirun -n 1/2/4, host-MPI build and GPU build):
  - configs: MERGERS, binaries 10 %/50 %, triples, SPIN_IMBH, coll_*, a 5000-star cluster;
  - pass: the per-step state hash (the hash used by `replicas_agree`: N, x, v, m, ids, spins) and every printed diagnostic row are bitwise equal across rank counts.
  - Phase 0 adds them as expected-fail (documenting today's dependence); Phase 1 flips them.
- **Schedule fuzzing.**
  - `NBODY_TASK_SHUFFLE=<seed>`: random task→rank assignment and random order inside a rank.
  - `NBODY_TASKS_SERIAL=1`: the task framework runs in serial order on every rank.
  - Both must be bitwise equal to the plain run. Catches hidden global state.
- **Batch check.** `NBODY_CHECK_BATCH=1`: old path on a copy, memcmp after each batched segment. Run under all existing ctest cases.
- **Join check.** In debug mode, `replicas_agree` after every join (hash of the joined range).
- **Existing invariants:** `NBODY_CHECK_SUBSYS` (every subsystem advanced exactly once), `NBODY_CHECK_PIDMAP`, `NBODY_CHECK_CC`.

---

## 7. Cost model

**Form:**

T(n) = s_chain + s_task / min(n, n_task) + p / min(n, n_max) + c(n) − overlap

**Parameters (all from the measured fits plus the estimates of §2):**
- **200k** (p = 14, n_max = 8 from the 40 N/1e6 of Rantala et al. 2023):
  - s after Phase 0 = 6.8;
  - after (1): s = 4.1;
  - after (2): s_chain = 1.7, s_task = 2.0, n_task = 8;
  - c = 0.3 / 0.4 / 0.5 s at 4 / 16 / 64 GPUs (NCCL); intra-rank overlap ≈ 1 s at n = 1.
- **1e6** (p = 67, n_max = 40):
  - Phase 0 s = 9.2 (energy split moved into p);
  - (1): s = 6.1;
  - (2): s_chain = 2.0, s_task = 3.3, n_task = 16;
  - c = 0.4–0.6 s (NCCL; host staging about 2× more);
  - collective count ~600 per step, bandwidth term per full-N gather ~1 ms inter-node.
- **1e7** (p = 4500 s, range 3000–6700 from N² × the sub-N² density trend of Rantala et al. 2023; n_max = 400):
  - Phase 0 s ≈ 40 s (the unsplit energy check alone would be ~470 s);
  - (1): s = 15;
  - (2): s_chain = 5, s_task = 10, n_task = 64;
  - c (NCCL) = 0.5 / 1 / 1.5 / 2 s at 4 / 16 / 64 / 256 GPUs. **Host staging would cost ~10 s/step** (7 ns/particle D2H+H2D × ~1.5M particles × ~1000 collectives), so NCCL is mandatory at 1e7.
- **FMM:**
  - p/F with F = 15 (1e6), 100 (1e7);
  - n_max_FMM = N/5e4;
  - tree build adds 0.2 s (1e6) and 1.5 s (1e7) to s.

Inputs used: launch + fence 9.7 µs, scalar D2H 13.7 µs, reduce 18.5 µs (per-call floor §3.1); Allgatherv 3.4 ms device versus 1.3 ms host-staged at 200k (fit: 0.15 ms + 7 ns × N_set).

**Predicted s per 1e-3 Myr step** (first-interval type; production is ×2–3 per the production-speed note):

| N | n GPUs | today | Phase 0 | after (1) | (1)+(2) | (1)+(2)+FMM |
|---|---|---|---|---|---|---|
| 2e5 | 1 | 22.0 (meas.) | 20.4 | 17.7 | 16.7 | — (direct wins) |
| | 4 | 11.3 (meas.) | 10.3 | 7.6 | 6.0 | — |
| | 16 | ~9.9 | 8.7 | 6.0 | 4.1 | — |
| | 64 | ~10.0 | 8.8 | 6.1 | 4.2 | — |
| 1e6 | 1 | 81.8 (meas.) | ~76 | 72.3 | 70.8 | 9.0 |
| | 4 | 31.8 (meas.) | 26.0 | 22.9 | 20.0 | 4.6 |
| | 16 | ~19.3 | 13.5 | 10.4 | 6.9 | 3.2 |
| | 64 | ~16.9 | 11.1 | 8.0 | **4.5** | 3.2 |
| 1e7 | 4 | — | ~1166 | 1141 | 1133 | 20.8 |
| | 16 | — | ~322 | 297 | 288 | 10.9 |
| | 64 | — | ~112 | 87 | 77 | 8.9 |
| | 256 | — | ~60 | 35 | 25 | 8.8 |

What the model says:
- **1e6 on 16 nodes:** 16.9 → 4.5 s/step, i.e. 19 Myr/day at first-interval speed. Core collapse at r_h = 3 pc (240 Myr) takes ~13 days, or ~1–1.5 months at production speed, about 3× better than HARD_PROBLEMS §3.2.
- **Beyond ~16 GPUs at 200k** nothing helps: p saturates at n_max and s_chain stays. Use 200k for ensembles (several seeds in parallel), not strong scaling.
- **With FMM,** 1e6 saturates at ~3 s/step from 16 GPUs. Its win is on one node (4.6 s versus 20 s).
- **1e7 with FMM on 16 nodes:** ~9 s/step, ≈ 10 Myr/day. Core collapse at r_h = 3 pc (630 Myr) takes ~2 months first-interval, 4–6 months production. That makes Flagship C plausible. Without FMM it stays ≥ 1 year even on 256 GPUs.

Error bars:
- s_chain/s_task split: ±50 % (the Phase-0 CC census and critical-path counters fix it).
- 1e7 p: factor 2.
- FMM F: factor 3, until I-1/I-2 run.

---

## 8. Phased implementation plan

**Phase 0: measure, plus cheap bit-identical items (1–2 weeks)**
- G.4 cheap items:
  - profiler fences conditional on `PROFILING`;
  - energy check split in fixed row blocks + triangle + reuse;
  - rebuild with `USE_GPU_AWARE_MPI=OFF`;
  - `TREE_NGB_MIN_N=32768`;
  - fuse dt+acc and slow+fast collectives.
- Instrumentation:
  - `nsys_busy.py` and NVTX regions;
  - per-step CC census (components per call, size distribution, task wall time, longest chain), on 200k with `HHS_CC=true`;
  - critical-path counter (longest LogH, longest component).
- Micro-benchmarks:
  - Kokkos Graph (10/50-node graphs, submit + fence, A100);
  - nccl-tests `all_gather_perf` 4 B–24 MB on 1–16 nodes;
  - MPI shared-window fetch-add latency, intra-node and inter-node.
- Confirm the rank-count hypothesis: a prototype global chunk/tile rule, then 1 versus 4 GPU state hash after step 1.
- Tests: the existing bitwise checks; add the `rankcount_*` cases as expected-fail.

**Phase 1: rank-count independence (days)**
- Global chunk/tile rule; canonical order in `exchange_subsystem_results` and `allgather_collisions`; block-wise PE sum.
- Gate: `rankcount_*` pass (1/2/4 GPUs, host-MPI and GPU), then standing. 3-seed rule on 4 GPUs (expected neutral).

**Phase 2: deep-level batching (bit-identical, 3–5 weeks)**
- 2a: control block, persistent pinned buffers, device pid map, range-local flags.
- 2b: fused single-team small-set kernels (n ≤ 512).
- 2c: Kokkos Graph segments with descriptors and size classes; graph cache with versioned buffers.
- Gate:
  - `NBODY_CHECK_BATCH` clean on all ctest cases, every diagnostic row identical;
  - timing: 200k Fig 9, 1e5 f_bin = 1 (Fig 7), 1e6;
  - nsys short gaps ≤ 1 s per step at 200k.
- 2d (optional): B6 host subtrees behind a switch, 3-seed rule.

**Phase 3: TaskContext and intra-rank concurrency (bit-identical, 4–6 weeks)**
- Remove the function-local statics; per-context books, collisions, counters, ids; deferred canonical compaction.
- S/F concurrency; CC components on streams; LogH pool.
- Decide HHS_CC default (A-2, 3 seeds) at the start of this phase.
- Gate: schedule fuzzing bitwise equal; MERGERS / coll_* / SPIN_IMBH identical; U ≥ 60 % at 200k on 4 GPUs.

**Phase 4: inter-rank tasks (4–6 weeks)**
- Task roots, LPT from allgathered costs, full-record NCCL joins (MPI fallback), node-local self-scheduling, rank subgroups for the sliced mid levels.
- Retire `DISTRIBUTE_SUBSYSTEMS`; it becomes T_L tasks.
- Gate:
  - `rankcount_*` bitwise on 1/2/4/8/16 GPUs (host-MPI and GPU) and in multi-node runs;
  - the **standing scaling metric**: 1e6 Fig-7 IC (10 % binaries) on 1, 2, 4, 8, 16 nodes; mean of steps 2–6 plus a ≥ 100-step production segment; s/step, Myr/day, U, U_unique, |dE/E| per interval and total;
  - 200k Fig 9 on 1–4 nodes.

**Phase 5 (optional; bits change where marked)**
- Asynchronous speculative energy check (bit-identical).
- Positions-only joins.
- Split-force overlap (members-last; 3 seeds).
- Secular/Kepler on the device if J-1 justifies it (3 seeds).
- NVSHMEM if join latency dominates.

**Phase 6: FMM (M8)**
- Run I-1 (force-error injection) first.
- Replicated tree, sliced sinks, FMM above a set-size threshold, time-step sums and gradient included.
- Gates: exact momentum, the energy rule on 3 seeds, the FROST 1e6 test.

**Standing rule:** a change that alters bits passes the 3-seed energy rule (equal or better |dE/E| on seeds 42/7/13) before it becomes a default.

---

## 9. Risks and failure modes

| risk | effect | mitigation |
|---|---|---|
| hidden global state left in the recursion (statics, `g_subsystem_mgr` fields, `Particles::layout_dirty_*`, `removed_pid_log`) | order-dependent results under concurrency | schedule fuzzing; grep audit of statics; TaskContext review |
| merger in a task shifts the global tail | corrupted ranges of other tasks | stable move-to-range-end inside tasks, compaction only at joins |
| collective called inside a task on one rank only | deadlock | no-communication rule with a runtime guard; `sync_after_removal` only at joins |
| one long task (F3-type LogH, PN binary) | wave waits; no scaling | SDAR default (M3); critical-task reporting; split independent work around it |
| the CC census shows few or skewed components at 200k with 10 % binaries | small s_task; (2) gains mostly from S/F and T_L | Phase-0 census before Phase 4; the model already caps n_task |
| HHS_CC fails the energy rule | most of the task volume disappears | S/F and T_L remain (≈ half the gain, estimate) |
| Kokkos Graph is experimental (API churn in 5.x; no scan; no update API; HIP graphs on MI300A less mature) | rework | use `cuda_capture`/`hip_capture` around existing code; plain streams as fallback; descriptors make graphs reusable |
| captured pointers invalidated by `realloc` | wrong results | monotone capacity with versioning; `NBODY_CHECK_BATCH` in CI |
| several host threads launching Kokkos kernels | Kokkos instance scratch locks; contention | one execution instance per driver thread; measure in Phase 3 |
| NCCL and MPI ordering | hangs | one communication thread; identical enqueue order |
| rank-count independence broken by a later "harmless" tuning (chunks, tiles) | silent loss of the property | standing `rankcount_*` CTest |
| speculative energy check | output written for a step that is rolled back | buffer output until confirmed |
| host/device bit-identity is impossible | B6 and LogH-on-GPU always change bits | keep them behind switches; deterministic placement by size |
| memory: two checkpoints and per-context scratch at 1e7 | ~3 GB replica + scratch per GPU | fits in 40 GB; monitor |

---

## 10. Literature notes and verification status

- **FROST §6.3** (LaTeX read): N_GPU^max ≈ 4 N/1e5. One MPI task per GPU, scatter-compute-gather. MPI-parallel CPU loops when N is small; serial CPU below a few hundred particles.
- **Rantala et al. 2023 §4** (LaTeX read): 40 N/1e6; 1e5 stalls at ~8 GPUs; 1e6 to 48; 5e6 to 128 on Raven A100; T = T_N² N² + T_bin f_bin N. Neighbour search on serial CPU, MPI-CPU or GPU by level size (l.330). Many small LogH systems integrated in parallel (l.440); whether on CPU or GPU is unstated.
- **FC-III** (LaTeX l.158): node-local MPI shared-memory windows for a single ensemble copy per node.
- **PeTar** (arXiv HTML §6.3/6.5/7.2): independent clusters with a host process per cluster; OpenMP inside each MPI process; FDPS load balance; flattening at 160/640 cores (1e5/1e6).
- **FDPS 1601.03138:** UNVERIFIED (PDF unreadable here). Domain decomposition does not apply to our replicated design.
- **Taichi 2012.02207 / 2205.12289:** OpenMP + AVX (abstract). The HHS + FMM details are taken from METHODS_SURVEY §I; their parallelisation of fast levels is UNVERIFIED.
- **NBODY6++GPU 1504.03687 §4:** via HARD_PROBLEMS §2.6.
- **NbodyCP 2511.06181:** via HARD_PROBLEMS; its PDF text was unreadable, so UNVERIFIED.
- **Pelupessy+2012, Jänes+2014:** via METHODS_SURVEY §A.3.
- **CUDA Graphs on A100:** GTC 2020 slide figures (host launch ~15–22×, device ~2× for a 32-node graph); per-node µs not found, so it must be measured.
- **NCCL and NVSHMEM latencies on raven:** UNVERIFIED; Phase-0 micro-benchmarks.

Sources (web):
- [NVIDIA GTC 2020 CUDA Graphs slides (A100 graph vs stream launch)](https://live.nvidia.cn/gtc-od/attachments/CNS20732.pdf)
- [NVIDIA blog: Getting started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs)
- [NVIDIA blog: Employing CUDA Graphs in a dynamic environment](https://developer.nvidia.com/blog/employing-cuda-graphs-in-a-dynamic-environment/)
- [PeTar, arXiv:2006.16560 (HTML)](https://arxiv.org/html/2006.16560v2)
- [NbodyCP, arXiv:2511.06181](https://arxiv.org/abs/2511.06181)
- [Taichi, arXiv:2012.02207](https://arxiv.org/abs/2012.02207)
- [FROST-CLUSTERS III, arXiv:2601.07917](https://arxiv.org/abs/2601.07917)

### Critical Files for Implementation
- /raven/u/uli/cpp-nbody/src/integrator/integrator.cpp: `hhs_fsi_recursive`, `hhs_cc_step`, `fsi_step`, `advance_subsystems_in_range`, `partition_by_pivot`, `exchange_subsystem_results`, `allgather_collisions`
- /raven/u/uli/cpp-nbody/src/gravity/n2_driver.h: `chunk_count` and the tile switch (rank-count dependence); the template for fused small-set kernels
- /raven/u/uli/cpp-nbody/src/gravity/forces.cpp: `cc_components`, `neighbor_search_subset`, the `cc_inter_*` kernels
- /raven/u/uli/cpp-nbody/src/collisional/subsystem.cpp: `next_id`, `update_particle_flags`, `rebuild_pid_to_idx`, `restore_active` / `strip_removed`
- /raven/u/uli/cpp-nbody/src/core/mpi_comm.cpp and /raven/u/uli/cpp-nbody/src/core/mpi_comm.h: slicing, allgathers, `replicas_agree`; where NCCL, rank subgroups and shared windows go
- also: /raven/u/uli/cpp-nbody/src/collisional/logh_batch.cpp (the LogH pool and staging), /raven/u/uli/cpp-nbody/src/core/constants.h (`fast_rsqrt`, the reason for host/device divergence), /raven/u/uli/cpp-nbody/tests/run_case.py (the `rankcount_*` and fuzzing cases)
