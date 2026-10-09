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
