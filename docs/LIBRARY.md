# RUMI++ as a library (host-code coupling)

`src/lib/rumi_lib.h` is the interface: plain C++, host arrays, RUMI++ units (pc, Msun, km/s). The host passes G at
every call (`set_particles`, `advance`, `energy`); `G_CODE` is not used by the library path.

- **Ranks:** one host process, or the ranks of a host sub-communicator (`attach_comm`, build with `USE_MPI=ON`): RUMI++'s
  distributed path (bodies replicated, force slices per rank). Every call is then collective over that communicator;
  host arrays are read on its rank 0 (the library broadcasts them). RUMI++ never touches `MPI_COMM_WORLD` there
  (`MPIComm::attach`; only `MPI_Abort` uses it). Results agree with one rank statistically, not bit for bit
  (other summation order in the force slices).
- **State:** the library keeps the bodies between calls. The host changes them through `kick` (its work is booked in
  the `Ledger`), `shift`, `set_mass` (in place, books the exact energy change in `E_sse` as the runtime SSE does) and a
  new `set_particles` (bodies added or removed: subsystems rebuilt; the energy jump booked in `Ledger::E_set`).
  `E_bodies + E_books - W_kick - E_set` is constant up to the integration error.
- **E_ref of FLYBY_CM_EPS:** `E_REF` if the config sets it, else max(|E_int|, |PE|/2) of the set in its centre-of-mass
  frame at each `set_particles`: |E0| of a virialised (standalone, recentred) cluster, independent of the host frame,
  and not near zero while a forming set passes from unbound to bound (|E_int| alone made the budget vanish there).
- **Stalls:** `SUBSYS_ADVANCE_LIMIT` stops a run whose subsystem needs more advances in one host step than the limit,
  with the subsystem's members, a, e and dt, instead of advancing ever more slowly; SUBSYS_ERR lines are at most one
  per subsystem and step.
- **Step:** `advance(dt, G)` is one standalone HHS step (`INTEGRATOR = hhs`, `ETA_RESTART = 0`, i.e. `PRESET = cluster`).
  `ETA_RESTART > 0`, `SSE_ENABLED`, `TIDAL_FIELD`, `USE_TREE` and `INTEGER_POSITIONS` are refused.
- **Mergers:** `merger_events` returns the kept and merged-away slots and the product mass since the last call.
- **Kokkos:** `kokkos_initialize(threads, device)` after `MPI_Init`, `kokkos_finalize()` before `MPI_Finalize`; an
  already initialised Kokkos (e.g. the host's own) is left alone. All staging is explicit `deep_copy` (any backend).
- **Restarts:** `save_state`/`load_state` write and read RUMI++'s HDF5 restart format (bodies, subsystems, books,
  merger tree, ledger, slots, an opaque host blob); a continued run is bit-identical to the uninterrupted one.
- **Replays:** `dump_ic` writes the state as an `IC_FILE`; the standalone program with `G_OVERRIDE`, `DT_MAX_CODE` and
  `E_REF` repeats a one-rank coupled run bit for bit.

## Versioning

- `RUMI_LIB_API_VERSION` in `rumi_lib.h` is bumped on every change of the interface: a call added, removed or
  changed in signature, units or meaning. Internal changes that keep the calls and their meaning do not bump it.
- The library embeds `RUMI_LIB_ID api=<version> commit=<git describe>` (`build_id()`). A host checks the version at
  build time (GIZMO's Makefile reads it from `librumi.a` with `strings` and stops on a mismatch with the version its
  `rumi_coupling.cc` expects) and records the commit for provenance (GIZMO: run log and snapshot header).
- Versions: 1 = first prototype (set_particles, kick, shift, advance, get_state); 2 = attach_comm, set_mass,
  ledger, e_ref, save_state/load_state, build_id.

## Build

```
cmake -S rumi-cpp -B build -DCMAKE_BUILD_TYPE=Release -DKokkos_ROOT=<kokkos> -DENABLE_COLLISIONAL=ON \
      -DUSE_MPI=ON -DUSE_HDF5=ON -DHDF5_ROOT=$HDF5_HOME -DBUILD_RUMI_LIBRARY=ON -DCMAKE_INSTALL_PREFIX=<prefix>
cmake --build build --target rumi -j && cmake --install build
```

Build it with the host's compiler, MPI and HDF5 (GIZMO on Orion: modules gcc, openmpi, hdf5-mpi, loaded one by one;
`USE_MPI=OFF` and `USE_HDF5=OFF` give a one-rank library without restarts). `<prefix>` then holds `lib/librumi.a`, `include/rumi_lib.h` and `rumi-lib.mk` (`RUMI_LIB_INCL`, `RUMI_LIB_LIBS`: the
library, the Kokkos archives and the backend's runtime flags). A Makefile host includes `rumi-lib.mk`; the host
compiler needs no Kokkos flags (the header has no Kokkos), but must link the same libstdc++ as the Kokkos build.

| Machine | Kokkos backend | Modules (as built) | Notes |
|---|---|---|---|
| Orion (CPU) | OpenMP (+Serial) | `gcc/13 openmpi/4.1 hdf5-mpi/2.0.0 cmake/3.30` | tested, 1-16 RUMI++ ranks; `RumiNumThreads` per rank |
| Raven (A100) | CUDA, `Kokkos_ARCH_AMPERE80` | `gcc/13 cuda/12.x cmake` | not yet built: see checks below |
| Viper (MI300A) | HIP, `Kokkos_ARCH_AMD_GFX942_APU` | `amd rocm cmake` | not yet built: see checks below |

GIZMO: `STARS_INTEGRATED_BY_RUMI` in `Config.sh`, `make RUMI_LIB_DIR=<prefix>` (GIZMO `Makefile` includes
`<prefix>/rumi-lib.mk`).

## To check on the first GPU build

- `rumi-lib.mk` link line with the host's `mpicxx` (CUDA: `-lcudart -lcuda`; HIP: `-lamdhip64`); static Kokkos CUDA
  without relocatable device code links with the host compiler.
- Device selection: only the RUMI++ ranks (GIZMO tasks 0..RumiNumRanks-1) initialise Kokkos. One GPU per node: one
  RUMI++ rank per node is not yet arranged (GIZMO's first tasks fill node 0); with RumiNumRanks = 1 use
  `RumiDeviceId = 0` or bind the GPU by `CUDA_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`. Several GPUs need the RUMI++
  ranks spread one per node (a rank list instead of 0..k-1) and `RumiDeviceId` = the node-local GPU.
- MPI with device buffers: the library builds with `USE_GPU_AWARE_MPI` off (host-staged allgathers) unless set.
- Kokkos initialisation order with the host's MPI (CUDA-aware MPI may need the device set before MPI_Init).
- Per-step transfer cost: `get_state`/`kick`/`shift` copy whole arrays host<->device each step (timers `h2d`, `d2h`).
