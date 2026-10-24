# Test references (C, not part of the production integrator)

Built by `src/mstar/CMakeLists.txt` only for the comparison ctests; never linked into `rumi-mstar` or `nbody-kokkos`.
All GPLv3.

* `ketju/`: the KETJU integrator library, commit eb96fd0 (2023-06-02), Copyright (C) 2019-2022 Antti Rantala, Matias
  Mannerkoski, and contributors; unchanged (headers, unit tests, nbody example, LICENSE.md).
* `ketju-extra-force/`: the experimental extra-force branch of KETJU, commit b60d16e (Matias Mannerkoski,
  2022-02-28; `src`, `include`, `nbody/main.c`, `README.md`), unchanged. Its files predate the per-file licence
  headers; the repository is GPLv3 (see `ketju/LICENSE.md`).
* `ketju-extra-force-fwd/`: eb96fd0 with the b60d16e patch forward-ported (integration.c, pn.c, init.c,
  ketju_integrator.h modified, with the notice in each; the other files as in `ketju/`). Unlike the branch it also
  resets the auxiliary variables when only an extra velocity-dependent force is on (the branch does so only with PN).
  Modified by Ulrich Steinwandel, 2026.
* `serial_mpi/mpi.h`: single-task stand-in for the MPI subset these libraries use (builds without MPI).
