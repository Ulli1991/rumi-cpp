# MSTAR module (optional, GPLv3)

C++ rewrite of the MSTAR regularised integrator (Rantala et al. 2020; public KETJU integrator, Mannerkoski et al.
2023) with Kokkos support for GPUs, by Ulrich Steinwandel.

Cite, when you publish results obtained with a build that includes this module:

* **required** (KETJU's README asks for it): A. Rantala, P. Pihajoki, M. Mannerkoski, P. H. Johansson & T. Naab,
  *MSTAR - a fast parallelised algorithmically regularised integrator with minimum spanning tree coordinates*,
  MNRAS 492, 4131 (2020), arXiv:2001.03180;
* **recommended** for the released version (PN corrections, BH mergers and recoil, softening, dynamic order
  control): M. Mannerkoski et al., *KETJU - resolving small-scale supermassive black hole dynamics in GADGET-4*,
  MNRAS 524, 4062 (2023), arXiv:2306.04963;
* the physics the code uses: L. Blanchet, Living Rev. Relativ. 17, 2 (2014) (2-3.5PN); K. G. Arun et al., PRD 77,
  064035 (2008) (3PN gauge transformation); K. S. Thorne & J. B. Hartle, PRD 31, 1815 (1985) (1PN N-body and spin
  terms); Y. Zlochower & C. O. Lousto, PRD 92, 024022 (2015) (merger remnants); E. Hairer, S. P. Norsett &
  G. Wanner, *Solving Ordinary Differential Equations I*, ch. II.9 (GBS extrapolation and order control).

## Licence

This directory is an **optional module** of RUMI++, compiled only with the CMake option `ENABLE_MSTAR=ON` (default
`OFF`). Everything in it is licensed under the **GNU General Public License v3** (or, at your option, any later
version), see [`LICENSE`](LICENSE); **a RUMI++ binary built with `ENABLE_MSTAR=ON` is distributed under the
GPLv3.** The RUMI++ core (BSD-3-Clause) builds and runs without this directory and reaches it only through the BSD
interface `src/collisional/mstar_dispatch.h`; no code from here is copied into core files.

## Origin, copyright and modifications

The integrator is derived from the KETJU integrator library (MSTAR), Copyright (C) 2019-2022 Antti Rantala, Matias
Mannerkoski, and contributors (GPLv3): the user's fork https://github.com/Ulli1991/ketju-integrator of
https://bitbucket.org/helsinkiextragalacticgroup/ketju-integrator, commit
`eb96fd0e5395cdb6ca7b8346f170cb114e4a3c2d` (2023-06-02), and for the extra forces the experimental branch commit
`b60d16e` (Mannerkoski, 2022-02-28).

**This code has been modified by Ulrich Steinwandel (2026): C++ rewrite with Kokkos/GPU support.** Every ported
file keeps the KETJU header and states the modification below it. Changes relative to KETJU eb96fd0:

* C++ (C++17, Kokkos) throughout the production path, namespaces and RAII types; no C translation unit, no
  `extern "C"` API, no MPI. The C library is kept only as a test reference (`reference/`).
* All GBS rows (substep divisions 2, 4, ..., 2k) of a step run concurrently on the device in lockstep: per substep
  stage one force launch over the active rows (one thread per row and body) and one kick+drift launch; on the host
  execution space small groups run each row inside one team. The same engine runs on the device and on the host
  (`Options::host_engine`; small groups in GPU builds).
* Vertex coordinates are rebuilt from the tree by root-to-vertex path sums in KETJU's floating-point order;
  reductions are deterministic. Results agree with the C library to round-off (bitwise for two bodies on the CPU)
  and are reproducible run to run.
* One device-to-host copy per GBS step; the extrapolated candidate state, its energies and the collision test are
  computed on the device.
* The MST is built on the host by dense O(N^2) Prim (KETJU's tree).
* BH mergers (`mstar_mergers.cpp`): C++ port of `bh_mergers.c` and the PN c.o.m. velocity of `pn.c`, with Cartesian
  separations.
* Additions: a collision test after every accepted step (RUMI++'s collision rule, `mstar_collision_distance` in
  the BSD dispatch header); B carried as an extra GBS variable outside the error control, giving the work of the
  PN and extra forces (`pn_dB`, `ext_work`), so that the energy books can separate physics from integration error.
* Extra forces (port of the branch `b60d16e`, `ExtraForces`): a linear field `a = A (x - x0)` and static softened
  point masses (velocity-independent), `a = D v` (velocity-dependent, through the auxiliary-velocity scheme as PN);
  on the device. Unlike the branch, B follows the work of the velocity-independent extra forces (`track_work`,
  default): with B blind to it (the branch) the time transformation drifts from U and a binary in a static field was
  off by 1e-2 in position after a few orbits (`tests/mstar_extra_force.cpp`).
* Not ported: star-star softening, `no_star_star_gravity`, the divide-and-conquer MST, the MPI parallelisation.

Files written for RUMI++ (Copyright (C) 2026 Ulrich Steinwandel, GPLv3): `mstar_dispatch.cpp` (the core's interface),
`reference/serial_mpi/mpi.h` (single-task MPI stand-in for the reference builds), `reference/ketju-extra-force-fwd`
(the branch patch forward-ported onto eb96fd0, a test reference; modified KETJU files carry the notice), `tests/`,
`CMakeLists.txt`.

## Build and tests

```
cmake .. -DENABLE_COLLISIONAL=ON -DENABLE_MSTAR=ON
ctest -R 'ketju_unit|mstar_'
```

* `ketju_unit`: KETJU's own unit tests (26) on the reference library, unmodified, on the shim `tests/gtest_shim`.
* `mstar_compare`: the C++ integrator against the reference (Kepler e = 0.9 with the analytic orbit, Pythagorean
  three-body, 100-body Plummer core in both kernel paths, PN binary with spins, two BHs in a 60-star cusp with all
  PN terms, PN work ledger, collisions, head-on and orbital BH mergers).
* `mstar_extra_force`, `mstar_extra_force_branch`: the extra forces against the forward-ported patch (round-off) and
  the branch (tolerance level), a static field and a tidal tensor against RK4, a drag force (aux-velocity path).
* `mstar_embedded_tide`: an embedded group, impulsive member kicks against c.o.m. kicks + continuous tide.
* `mstar_vs_logh`: MSTAR against RUMI++'s LogH + GBS on Kepler e = 0.9 and the Pythagorean problem.
* `mstar-bench [N_max] [steps] [C_N_max] [team_N]`: scaling N = 16..4096 and launch latency (not a ctest).

Use in RUMI++: `docs/MSTAR.md`.
