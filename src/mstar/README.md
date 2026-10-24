# MSTAR module (optional, GPLv3)

This directory is an **optional module** of RUMI++. It is compiled only with the CMake option
`ENABLE_MSTAR=ON` (default `OFF`). Everything in this directory is licensed under the
**GNU General Public License v3** (or, at your option, any later version), see [`LICENSE`](LICENSE);
**a RUMI++ binary built with `ENABLE_MSTAR=ON` is distributed under the GPLv3.** The RUMI++ core
(BSD-3-Clause) builds and runs without this directory and reaches it only through the BSD interface
`src/collisional/mstar_dispatch.h`; no code from here is copied into core files.

## Origin and copyright

`ketju/` is the KETJU integrator library (MSTAR), vendored unchanged from the user's fork
https://github.com/Ulli1991/ketju-integrator (upstream:
https://bitbucket.org/helsinkiextragalacticgroup/ketju-integrator), commit
`eb96fd0e5395cdb6ca7b8346f170cb114e4a3c2d` (2023-06-02): `src/`, `include/`, `unit-tests/`, `nbody/`,
`README.md`, `TODO.md`, `LICENSE.md`. Copyright (C) 2019-2022 Antti Rantala, Matias Mannerkoski, and
contributors; GPLv3. Each file keeps its original header.

Files written for RUMI++ (Copyright (C) 2026 RUMI++ authors, GPLv3):

| file | content |
|---|---|
| `mstar_kokkos.h`, `mstar_kokkos.cpp` | Kokkos port of KETJU's `integration.c`, `mst.c` (tree proximity, Prim tree) and `pn.c`; the KETJU header is kept and the modification is stated below it |
| `mstar_dispatch.cpp` | implementation of the core's BSD interface `src/collisional/mstar_dispatch.h` (host path = the C library, device path = the port) |
| `serial_mpi/mpi.h` | single-task stand-in for the MPI subset KETJU uses (builds without MPI) |
| `tests/` | GoogleTest shim + runner for KETJU's unit tests, port-vs-C-library and MSTAR-vs-LogH tests, scaling benchmark |
| `CMakeLists.txt` | module build |

### Changes of the Kokkos port relative to the C library

* All GBS rows (substep divisions `2, 4, ..., 2k`) of a step run concurrently on the device in lockstep:
  per substep stage one force launch over all active rows and one kick+drift launch; groups with
  `N <= team_rows_max_n` (default 96) run each whole row inside one team (one launch per step).
* Vertex coordinates are rebuilt from the tree by root-to-vertex path sums in the same floating-point
  order as KETJU's edge loop; reductions are deterministic team reductions. Results agree with the C
  library to round-off (bitwise for 2-body systems) and are reproducible run to run.
* One device-to-host copy per GBS step; the extrapolated candidate state is built on the device.
* The MST is built on the host by dense O(N^2) Prim (KETJU's tree: root nearest the centre of mass,
  float edge weights), every `steps_between_mst_reconstruction` steps.
* BH mergers call the C library's `ketju_do_bh_mergers` on a host mirror of the state.
* Not ported: star-star softening, `no_star_star_gravity`, divide-and-conquer MST, MPI parallelisation.

## Citation

When you publish results obtained with a build that includes this module, cite

* **required** (KETJU's README asks for it): A. Rantala, P. Pihajoki, M. Mannerkoski, P. H. Johansson &
  T. Naab, *MSTAR - a fast parallelised algorithmically regularised integrator with minimum spanning tree
  coordinates*, MNRAS 492, 4131 (2020), arXiv:2001.03180;
* **recommended** for the released version (PN corrections, BH mergers and recoil, softening, dynamic
  order control): M. Mannerkoski et al., *KETJU - resolving
  small-scale supermassive black hole dynamics in GADGET-4*, MNRAS 524, 4062 (2023), arXiv:2306.04963;

and, for the physics the code uses: L. Blanchet, *Gravitational radiation from post-Newtonian sources and
inspiralling compact binaries*, Living Rev. Relativ. 17, 2 (2014) (2-3.5PN); K. G. Arun et al., PRD 77,
064035 (2008) (3PN gauge transformation); K. S. Thorne & J. B. Hartle, PRD 31, 1815 (1985) (1PN N-body and
spin terms); Y. Zlochower & C. O. Lousto, PRD 92, 024022 (2015) (merger remnant fits); E. Hairer,
S. P. Norsett & G. Wanner, *Solving Ordinary Differential Equations I*, ch. II.9 (GBS extrapolation and
order control).

## Build and tests

```
cmake .. -DENABLE_COLLISIONAL=ON -DENABLE_MSTAR=ON [-DUSE_MPI=ON]
ctest -R 'ketju_unit|mstar_'
```

* `ketju_unit`: KETJU's own unit tests (26), unmodified, on the shim `tests/gtest_shim`.
* `mstar_compare`: the port against the C library (Kepler e = 0.9 with the analytic orbit, Pythagorean
  three-body, 100-body Plummer core in both device paths, PN binary with spins, two BHs in a 60-star cusp
  with all PN terms, BH merger).
* `mstar_vs_logh`: MSTAR against RUMI++'s LogH + GBS on Kepler e = 0.9 and the Pythagorean problem.
* `mstar-bench [N_max] [steps] [C_N_max] [team_N]`: scaling N = 16..4096 and launch latency (not a ctest).

Use in RUMI++: `docs/MSTAR.md`.
