# RUMI++ as a library (host-code coupling)

`src/lib/rumi_lib.h` is the interface: plain C++, host arrays, RUMI++ units (pc, Msun, km/s). The host passes G at
every call (`set_particles`, `advance`, `energy`); `G_CODE` is not used by the library path.

- **Process model:** one host process (MPI rank) runs the library. It makes no MPI calls: build with `USE_MPI=OFF`.
  The host gathers its bodies to that rank and scatters the results.
- **State:** the library keeps the bodies between calls. The host changes them only through `kick`, `shift` and a new
  `set_particles` (new or removed bodies, mass changes), which rebuilds the subsystems (as the start of a standalone run).
- **Step:** `advance(dt, G)` is one standalone HHS step (`INTEGRATOR = hhs`, `ETA_RESTART = 0`, i.e. `PRESET = cluster`).
  `ETA_RESTART > 0`, `SSE_ENABLED`, `TIDAL_FIELD`, `USE_TREE` and `INTEGER_POSITIONS` are refused.
- **Kokkos:** `kokkos_initialize(threads, device)` after `MPI_Init`, `kokkos_finalize()` before `MPI_Finalize`; an
  already initialised Kokkos (e.g. the host's own) is left alone. All staging is explicit `deep_copy` (any backend).
- **Replays:** `dump_ic` writes the state as an `IC_FILE`; the standalone program with `G_OVERRIDE` and `DT_MAX_CODE`
  repeats the coupled run bit for bit.

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
      -DUSE_MPI=OFF -DUSE_HDF5=OFF -DBUILD_RUMI_LIBRARY=ON -DCMAKE_INSTALL_PREFIX=<prefix>
cmake --build build --target rumi -j && cmake --install build
```

`<prefix>` then holds `lib/librumi.a`, `include/rumi_lib.h` and `rumi-lib.mk` (`RUMI_LIB_INCL`, `RUMI_LIB_LIBS`: the
library, the Kokkos archives and the backend's runtime flags). A Makefile host includes `rumi-lib.mk`; the host
compiler needs no Kokkos flags (the header has no Kokkos), but must link the same libstdc++ as the Kokkos build.

| Machine | Kokkos backend | Modules (as built) | Notes |
|---|---|---|---|
| Orion (CPU) | OpenMP (+Serial) | `gcc/13 cmake/3.30` | tested; `RumiNumThreads` = cores of the RUMI++ rank |
| Raven (A100) | CUDA, `Kokkos_ARCH_AMPERE80` | `gcc/13 cuda/12.x cmake` | not yet built: see checks below |
| Viper (MI300A) | HIP, `Kokkos_ARCH_AMD_GFX942_APU` | `amd rocm cmake` | not yet built: see checks below |

GIZMO: `STARS_INTEGRATED_BY_RUMI` in `Config.sh`, `make RUMI_LIB_DIR=<prefix>` (GIZMO `Makefile` includes
`<prefix>/rumi-lib.mk`).

## To check on the first GPU build

- `rumi-lib.mk` link line with the host's `mpicxx` (CUDA: `-lcudart -lcuda`; HIP: `-lamdhip64`); static Kokkos CUDA
  without relocatable device code links with the host compiler.
- Device selection: only the RUMI++ rank initialises Kokkos; with one GPU per node use `RumiDeviceId = 0` or leave
  `-1` and bind the GPU by `CUDA_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`. Several GPUs (one RUMI++ rank per node)
  need the multi-rank RUMI++ path (sub-communicator) and are not implemented.
- Per-step transfer cost: `get_state`/`kick`/`shift` copy whole arrays host<->device each step (timers `h2d`, `d2h`).
