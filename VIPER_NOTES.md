# Status and next steps (for the viper session)

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

## Done on viper (2026-10-09, host build: Kokkos OpenMP/Serial at ~/kokkos-host, build-host/)

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
  subsystems): open.
- RSQRT_NEWTON default (GPU accuracy = CPU at +4 %); SERIAL_N2_CPU = 0 fastest on MI300A (13-14 s vs 15.3 s).
- DISTRIBUTE_SUBSYSTEMS (off): correct (replicas identical, tests), but on 2 nodes slower (subsystem part
  7.3 s vs 5.2 s): ~4300 calls per step each with inter-node allgathers. Next: one exchange per HHS level
  or node-local distribution.
- Long runs on 1 node: 200k no binaries 1.2 Myr, |dE/E| ~1e-9 flat, dP 2e-16, dL 4e-15 (2.2 s/step);
  10 % binaries 0.15 Myr, 1.1e-9 (BIFROST Fig. 9 with BHs: 3e-7 at 5 Myr).

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

## On viper

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
   explicit kicks (BIFROST Eqs. 31-33, KETJU), GBS details (MSTAR), stale data in the step criteria,
   MSTAR not wired in.
6. MPI speed, measure first with the step profiler: fuse accelerations + time steps into one
   allgather, NCCL/RCCL inside a node, MPI-4 persistent collectives, SERIAL_N2_GPU per machine.

Full audit and review: `jax-hydro/research_notes/Rantala integrators/` (rantala_spec.md,
cpp_nbody_code_review.md; local, not in a repo). The README's FROST and BIFROST DOIs are wrong:
FROST is 10.1093/mnras/stab057, BIFROST 10.1093/mnras/stad1360; the HHS-FSI is FROST Algorithm 3.

After cpp-nbody: the standalone Python/JAX/Pallas port (`~/jax-nbody/DESIGN.md`), measured against
this code.
