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
