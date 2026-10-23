# Development notes, October 2026: fixes, measurements, open items

## Done (tested on CPU, Kokkos 5.0.2 Serial, ctest: 8/8 pass in ~80 s)

- Host backends run: `TILE_SIZE` is 1 on CPU (teams of 128 aborted on Serial/OpenMP), 128 on GPU.
- Built-in tests no longer write `stellar_type` outside collisional mode (segfault).
- FROST Eq. 49 applied to the actual step eta*tau (was the eta-free timescale): the secular energy
  drift on eccentric orbits is gone; e = 0.9 / 0.99 at eta = 0.008 now 2.4e-12 / 5.1e-11 at 1000 P
  (FROST Fig. 4: 1e-12 - 1e-10; before: 1.5e-6 / 2.3e-5).
- Subsystem flags skipped outside collisional mode (multi-level runs with COLLISIONAL = false segfaulted).
- `INTEGRATOR = fsi` steps at the fastest occupied level (was DT_MIN: stuck in time).
- Compensated (Kahan) velocity kicks; with INTEGER_POSITIONS (domain matched to the system) the
  round-off floor is below FROST's: planets at eta = 0.004 E 3.4e-14, P ~1e-17, L ~2e-15.
  MPI needs no change: only `distribute_initial` writes velocities; kicks are replicated.
- `tests/run_case.py` + CTest: kepler_e09, kepler_e0, frost4_e09, frost4_e099, planets_hhs,
  planets_fsi, floor_binary_e0, floor_planets. Bounds 1.3x measured, orders measured -0.1/+0.3,
  each case fails on the pre-fix code (teeth).

## Fixes of 2026-10-09 (host build with Kokkos OpenMP/Serial; GPU checks on MI300A)

- Mergers inside the HHS recursion (bug 3, part of 9): compaction shifted the arrays under the callers'
  ranges and left a stale copy in the old last slot (the duplicated pid / one-body LogH NaN / endless
  restarts of the March binary runs). Ranges are suffixes of [0, N), so the recursion now returns the
  number of removed particles and callers shorten their counts; restored subsystem lists drop merged pids;
  the pid map and the MPI slices (`dp.sync_count()`) are refreshed after every compaction (also escapers).
- Merger energy sink is now exact (pair KE + mutual PE + potential with all others, minus remnant KE):
  |dE/E| with sinks 2e-10 through 76 mergers; it used to miss 1/2 mu v_rel^2, |dE/E| ~ 1, which triggered
  restarts.
- `Checkpoint::restore` restores `p.N` (bug 4), removed pids, merger tree, grad_mag; the restart loop keeps
  the attempt with the smallest error instead of accepting the worse last one (bug 21).
- CPU small-subset path (bug 2): its partitions reordered the host copy but never the device-only fields
  (particle_id, stellar type/radius, spin, kick compensation, orig_index), so with SERIAL_N2_CPU > 0 every
  CPU-path call scrambled identities. The host copy now records the order and replays it on the device
  (`Particles::apply_permutation`, shared with the GPU partition) before every write-back; grad_mag is
  written back too.
- Global Kokkos Views (LogH batch, subsystem manager) released before `Kokkos::finalize` (every collisional
  run aborted at exit on host backends).
- Tests `mergers_gpu_path`, `mergers_cpu_path` (400 stars, 50 % contact binaries, ~75 mergers at depths
  0-10): the old code segfaults / hangs.

Also done (same day): LogH groups > 8 bodies stay with the FSI (bug 1); single-rank global PN1 (bug 5);
spins broadcast in MPI; replica hash every REPLICA_CHECK_INTERVAL steps (exit 3); compensated kicks in the
CPU path; CPU path kept dropping the enclosing level's subsystem flags after a nested merger (2e-5/step);
non-collisional build compiles again; MPI CPU build (build-host-mpi, gcc/14 + openmpi/5.0) runs the 2-rank
tests. GPU (MI300A) ctest: all accuracy checks equal the host values; 2-body cases are ~20x slower than one
CPU core (launch latency), so GPU runs need NBODY_TEST_TIMEOUT and apu (apudev caps jobs at 15 min).

Night of 2026-10-09 (speed + accuracy):
- Binary members got orbital time-steps (stale p.ax with the partner in v/a, gradient, perturbation
  criteria): fixed; PN subsystems ran into an energy restart they cannot pass (E_N + E_1PN under PN):
  now restart on GBS non-convergence, Newtonian E/P/L change booked as E_pn / P_sink / L_sink.
- 200k, 10 % binaries, 2 MI300A: never finished a step -> 113-262 s -> 16-25 s/step; |dE/E| 3.4e-11,
  dP 1.5e-15, dL 4e-16 after 5 steps. No binaries: 2.3-2.4 s/step, 6.5e-11.
- LogH batch and secular binaries run on all host threads; SSE remnants of mergers restart as ZAMS
  stars of the merged mass (SSE shrank them back); runtime SSE updates radius/luminosity; exact
  SSE/escaper/merger E, P, L sinks; ALL_SLOW_SPLIT option (FROST Alg. 3: e=0.9 endpoint 1.8e-13,
  below FROST's 1e-12, at 1.65x cost); ACCURATE_RSQRT option (GPU frost4_e099 1.7e-10 vs 6.9e-11 CPU).

Later the same night:
- ALL_SLOW_SPLIT is the default (FROST Alg. 3): free on clusters (20k, 10 % binaries: 84 s vs 83 s, same
  error), 5-16x lower error on eccentric few-body orbits, round-off floors ~2x higher; bounds re-measured.
- Per-HHS-level O(N) host work removed (it made the CPU small-set path 44 % of the 200k binary step):
  incremental pid map (dirty-range tracking, NBODY_CHECK_PIDMAP self-check on in all tests), member-only
  flag scatter, member gathers in shared/perturbation time-steps and in detection (compact union-find).
  Results bit-identical on the CPU build.
- Neighbour pair buffer grows on overflow (50 % binaries: 250 binaries were silently not detected).
- Second review: triple branch of detection used a raw id in the compact union-find (fixed, test
  triples_detected); bounds checks abort instead of UB; SSE/escaper removals registered.
- RSQRT_NEWTON / ACCURATE_RSQRT options (GPU e=0.99 endpoint 1.7e-10 with plain rsqrt vs 6.9e-11).
- 2 nodes (4 MI300A) give no speedup at 200k: small-set path and subsystems are replicated per rank.
- Dense cores: oversize neighbour chains split along bound pairs (r_h = 0.02 pc now 2 steps in 246 s);
  accuracy there does not converge below ~7e-8 per step with eta halving (strong perturbers on
  subsystems): open. Diagnosis (r_h = 0.03 pc, 2000 stars, 50 % binaries of 1-20 AU, 10 steps of
  1e-5 Myr, restarts off): per-step |dE/E| oscillates 3e-9 - 7e-8 for eta x1 and x0.5, unchanged by
  ETA_GBS 1e-13 or disabling secular evolution; without binaries 1.7e-9 -> 1.3e-11 when eta is halved.
  The floor is the perturbers' tidal coupling to binaries whose periods are far below the FSI step:
  their internal motion is exact (LogH / secular) but the perturbation arrives as kicks that sample the
  orbital phase, and the phase-dependent tidal energy oscillates. Fix candidates: tidal tensor applied
  inside LogH (PeTar, AR-CHAIN perturbers), a resolution criterion for strongly perturbed binaries, or
  perturbers inside larger subsystems (MSTAR).
- RSQRT_NEWTON default (GPU accuracy = CPU at +4 %); SERIAL_N2_CPU = 0 fastest on MI300A (13-14 s vs 15.3 s).
- DISTRIBUTE_SUBSYSTEMS (off): correct (replicas identical, tests), but on 2 nodes slower (subsystem part
  7.3 s vs 5.2 s): ~4300 calls per step each with inter-node allgathers. Next: one exchange per HHS level
  or node-local distribution.
- Long runs on 1 node: 200k no binaries 1.2 Myr, |dE/E| ~1e-9 flat, dP 2e-16, dL 4e-15 (2.2 s/step);
  10 % binaries 0.15 Myr, 1.1e-9 (Rantala et al. 2023 Fig. 9 with BHs: 3e-7 at 5 Myr).

Tree check (65k Plummer, softened, 10 steps, 2 MI300A): direct FSI |dE/E| 2.6e-10 / 1.9e-11 / 1.3e-12 at
eta 0.2 / 0.1 / 0.05 (order 3.8-3.9), 2.8 s/step. Tree (FSI with tree gradient term, not KDK) sits on its
force-error floor independent of eta: 1.4e-5 at theta 0.5, 2.8e-6 at theta 0.3, and loses L (1e-4 to 6e-4
in 10 steps; only P is corrected); 3.9-4.2 s/step, slower than direct at this N. Not useful for the
accuracy targets until higher multipoles / symmetric (FMM-type) interactions; revisit for N >~ 1e6.

Open, found on the way:
- frost4_e099 (6.9e-11 vs 6.6e-11) and floor_binary_e0 (1.1e-14 vs 2.4e-15) fail on gcc/x86 with
  identical numbers before and after these changes: laptop bounds, re-measure per platform.
- SSE_ENABLED defaults to true: its first call removed 37 % of the mass of a ZAMS Kroupa test cluster.
- With USE_TREE the global tree indices go stale after a merger compaction (tree is off in collisional runs).
- dL/L is not conserved through mergers (orbital L of the pair is dropped; stellar spin is not tracked).

## Next steps (as of 2026-10-09)

1. Build against the GPU Kokkos and run `ctest --test-dir <build>`: first check of the fixes on the
   GPU path (TILE_SIZE 128). Bounds are from the laptop; a floor-test failure means re-measure first.
2. MPI: run floor_planets and floor_binary_e0 on 1 and 2 ranks; the energy series must agree to
   round-off (replicated state).
3. Collisional mode, next tests to write (each must fail before its fix):
   - PN binary through a merger: exact energy and momentum booking (bug 9; 336 km/s appeared at
     t = 0 in binary_test with PN on) and the restart loop accepting the worse step (bug 21).
   - LogH with more than 8 bodies: refuse instead of overflowing `LOGH_MAX_BODIES` (bug 1).
   - CPU path (SERIAL_N2_CPU) write-back of particle_id, stellar fields (bug 2).
   - Ghost particle after compaction, and `dp.N_total` in MPI (bug 3).
   - `Checkpoint::restore` does not restore `p.N` (bug 4).
   - Global PN1 only runs on more than one rank (bug 5).
4. MPI robustness: spin_vec not broadcast (IMBH spin lost on >1 rank); diagnostic force includes
   intra-subsystem pairs only in the MPI version; a per-K-step state hash across ranks that stops
   the run on divergence.
5. Deviations from the papers that need a decision before changing (A/B first): LogH as DKD with
   explicit kicks (Rantala et al. 2023 Eqs. 31-33, KETJU), GBS details (MSTAR), stale data in the step criteria,
   MSTAR not wired in.
6. MPI speed, measure first with the step profiler: fuse accelerations + time steps into one
   allgather, NCCL/RCCL inside a node, MPI-4 persistent collectives, SERIAL_N2_GPU per machine.
   Profiler now prints time per HHS depth (forces / subsystems / kicks / MPI); MPI time was never
   recorded before (no timer named its component). DISTRIBUTE_MIN_SUBSYSTEMS (default 256): smaller
   calls run on every rank. Batching the exchange per level is impossible (same-depth calls are
   separated by the deeper calls that need their results). Next: fuse enforce_shared_timestep,
   apply_perturbation_timestep and detect_mergers into one gather and one parallel loop.
7. Far field for N >~ 1e6 (tree check above: force-error floor 1e-5..1e-6, L drift, slower at 65k):
   a. Barnes randomisation (Treecode2, arXiv:2602.06295): random root shift, rotation and size at
      each tree build, so P and L errors random-walk (~sqrt t) instead of drifting. Cheap on our
      octree (rebuilt per level). Test: the 65k tree case, L drift and long-run dE/E.
   b. Dehnen FMM (falcON, astro-ph/0003209): symmetric cell-cell interactions, exact P, lets the
      HHS slow<->fast kicks be pairwise symmetric (Gadget-4 HHS + FMM, Volker's trick).
   c. Marcello (arXiv:1706.06989): torque correction on the Dehnen scheme, exact L, ~2x cost.
   d. High expansion order (p ~ 8-10, estimate) and the FSI gradient term from the local
      expansions (tidal tensor). First: force error and cost vs p on a 1e6 Plummer sphere.
   e. FMM only on the large top levels, direct below a size threshold. Check that random frames
      per force evaluation do not spoil FSI time symmetry measurably.
   Phase 1 done (tools/fmm/fmm_ref.cpp: Cartesian, softened kernel, mutual dual walk, serial):
   momentum to 1e-17 at every order; error ~theta^p. N = 2e4 Plummer: rms 3.0e-10 at theta 0.2,
   p 10 (8.4e-12 at p 12); 1.6e-7 at theta 0.3, p 10. Plain Cartesian M2L is O(p^6), 27 us at p 10:
   far too slow. Next: O(p^3) rotation-based spherical-harmonic M2L (Dehnen 2014), which needs an
   exactly Newtonian far field: compact (spline/Wendland) softening or none (to be decided).

FROST is Rantala, Naab & Springel 2021, MNRAS 502, 5546 (doi 10.1093/mnras/stab057); the HHS-FSI is its Algorithm 3.

## Isolated-set stop and stable triples (2026-10-10)

- 200k, 10 % binaries, 1 MI300A node, 30 steps: old code f8676d6 14.65 s/step, max late |dE/E| 5.4e-10; new code
  6.86 s/step but 2.8e-9 (2.1x faster, 5x worse late energy); new with the old GBS cap (DYNCAP 10) 2.9e-9, so the GBS cap
  is not the cause.
- Diagnosis: old and new agree until step 27. From step 29 the new code sends stable triples to the orbit-averaged
  secular-triple solver (16/8/16 calls in steps 29-31); the old code never does. The isolated-set stop advances a set that
  is one subsystem over the whole pivot step, and the LogH/secular choice depends on the step length
  (`n_orb_per_dt = dt / P_in`), so the longer step pushes triples into the secular path, which does not conserve their
  instantaneous energy. Fix: 3-member sets are excluded from the stop and recurse as before. Host ctest 24/24 with the fix.
  Pass criterion for the GPU check (energy rule): the fixed code must match the old code's late-time |dE/E| on seeds 42,
  7 and 13. If not, bisect: small-set host paths, 256 tiles (summation order at the top levels), a build without the
  isolated-set stop.
- GPU ctest: 28/29. `floor_planets` misses the |dL/L| bound 3.3e-15 by round-off only (3.3e-15 with the host small-set
  paths, 3.9e-15 on the device path; energy identical, 8.8e-14 against 1.25e-13). Loosen the bound after the seed runs,
  not before.
- Open decisions: dense-core floor (more single-binary reference tests, or park it: findings in the dense-core item
  above); wire the spline `Softening` into the kernels, CPU path, diagnostics and FMM; build the FMM interaction lists
  on the GPU.
- Later: full 5 Myr Rantala et al. 2023 Fig. 9 run with binaries; spin evolution (§3.9) exists but is never called;
  MSTAR for groups of more than 8 bodies; far field item 7 above.

## CUDA build (A100)

- Kokkos with CUDA (AMPERE80), modules `gcc/13 cuda/12.6 openmpi_gpu/5.0` on an MPCDF-type system:
  `cmake -DCMAKE_CXX_COMPILER=<kokkos>/bin/nvcc_wrapper -DKokkos_ROOT=<kokkos> -DUSE_MPI=ON -DUSE_GPU_AWARE_MPI=ON
  -DENABLE_COLLISIONAL=ON`; run with `--kokkos-map-device-id-by=mpi_rank`, 4 x A100 per node.
- Until 37aa7bf everything was built and run with HIP on the MI300A only. On the first CUDA build check
  `n2_driver.h` / `n2_kernels.h` (templated tile sizes, nvcc_wrapper), and that the unified-memory paths
  (`kHostAccessibleDevice`, `NBODY_SMALL_FOR`, `NBODY_SMALL_N2`) are off on the A100. `RSQRT_NEWTON` is on by default
  (GPU accuracy = CPU). The CPU ctest needs a host Kokkos (OpenMP+Serial).
