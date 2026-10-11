# cpp-nbody

Performance-portable gravitational N-body code for stellar cluster simulations. Built on the **Kokkos 5.0** performance portability library, supporting CPUs as well as NVIDIA, AMD, and Intel GPUs. MPI-parallelized with GPU-aware MPI support where available.

## Physics

The integrator implements a 4th-order Hermite scheme (Forward Symplectic Integrator, FSI) inside a **Hierarchical Hamiltonian Splitting** (HHS) framework following Algorithm 4 of the [FROST paper](https://doi.org/10.1093/mnras/stac2192). Particles are recursively split into slow and fast groups by individual adaptive timesteps, giving O(N log N) time integration cost even with a wide dynamic range.

### Non-collisional mode (default)

Softened gravity suitable for large-N collisionless problems (galaxy models, cosmological halos). Includes an optional **Barnes-Hut tree** for O(N log N) force evaluation with monopole approximation and GPU-accelerated Morton-key sorting and group-based traversal.

### Collisional mode (`-DENABLE_COLLISIONAL=ON`)

Unsoftened direct N-body with close encounter handling for star cluster simulations, following the [BIFROST](https://doi.org/10.1093/mnras/stae2255) prescription:

- **Subsystem detection**: neighbor-based detection of close encounters (flybys, binaries, hierarchical triples)
- **LogH regularization**: logarithmic Hamiltonian integrator with Gragg-Bulirsch-Stoer (GBS) extrapolation for subsystems, handling arbitrarily close encounters
- **Secular binary evolution**: Gauss perturbation equations for wide, weakly-perturbed binaries
- **Post-Newtonian corrections**: PN1 through PN3.5 for compact object dynamics
- **Single stellar evolution** (SSE): C++ port of the Hurley et al. (2000) fitting formulae with modern prescriptions (Vink 2001 winds, Fryer 2012 remnant models, PPISN/PISN, ECSN)
- **Merger detection and stellar merger trees**

Zero overhead when compiled without `ENABLE_COLLISIONAL`.

### Initial conditions

Built-in IC generator supporting Plummer spheres with configurable mass functions (equal mass, Kroupa IMF, Kroupa + SSE-evolved), primordial binary populations, and IMBH injection. Also includes isolated binary, triple, and outer solar system test cases.

## Building

Requires a Kokkos 5.0 installation configured for the target backend (CUDA, HIP, SYCL, or OpenMP for CPU).

```bash
cmake -S . -B build \
  -DKokkos_ROOT=/path/to/kokkos
cmake --build build -j$(nproc)
```

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_MPI` | OFF | MPI parallelization (replicated particles, distributed forces) |
| `USE_GPU_AWARE_MPI` | OFF | Pass device pointers directly to MPI (requires CUDA/ROCm-aware MPI) |
| `MPI_TOPOLOGY_RING` | OFF | Alternative j-partitioned MPI topology using allreduce |
| `ENABLE_COLLISIONAL` | OFF | Unsoftened collisional mode with close encounter handling |
| `USE_HDF5` | ON (if found) | HDF5 restart files and snapshots (raven: `module load hdf5-serial/1.14.1` after `gcc/13`) |
| `LUNA_PYTHON` | empty | Python with LUNA importable, for the `luna_reader` test (e.g. `/u/uli/jax-env/bin/python`) |

### Example: multi-GPU build with collisional support

```bash
module load gcc/13 cuda/12.6 openmpi_gpu/5.0 hdf5-serial/1.14.1

cmake -S . -B build-mpi \
  -DKokkos_ROOT=$HOME/kokkos-cuda-mpi \
  -DUSE_MPI=ON \
  -DUSE_GPU_AWARE_MPI=ON \
  -DENABLE_COLLISIONAL=ON
cmake --build build-mpi -j$(nproc)
```

## Running

```bash
mpirun -np <ngpus> ./build-mpi/nbody-kokkos
```

Single-GPU runs use `mpirun -np 1`. All configuration is read from `config.txt` in the working directory.

### Restart files and snapshots (HDF5)

| Config key | Default | Meaning |
|------------|---------|---------|
| `RESTART_INTERVAL` | 0 | Write `restart_<step>.h5` every this many steps (0: off) and at the end of the run |
| `WALLTIME_LIMIT` | 0 | Hours; write a restart file and stop when the next step might not fit (also on `SIGUSR1`, e.g. `sbatch --signal=USR1@600`) |
| `RESTART_DIR`, `RESTART_KEEP` | `.`, 2 | Where restart files go; how many complete ones are kept |
| `RESTART_FILE` | empty | Continue from this file (same as `--restart <file>`); `latest`: newest in `RESTART_DIR`; `auto`: that, or a fresh start if there is none |
| `SNAPSHOT_INTERVAL` | 0 | Snapshot every this many steps, at step 0 and at the end (0: off) |
| `SNAPSHOT_DIR`, `SNAPSHOT_BASE` | `.`, `snapshot` | Snapshots go to `<dir>/snapdir_NNN/<base>_NNN.0.hdf5` |
| `SNAPSHOT_POTENTIAL` | true | Store the direct-sum potential of every particle (O(N^2)) |

A continuation from a restart file is bit-identical to the uninterrupted run (CTest cases `restart_*`); the config
file may change `TOTAL_TIME_MYR` and the output keys, while changed physics parameters are reported as warnings.
Restart files are written by rank 0 through a temporary file and a rename, after the ranks were checked to agree.
Snapshots use the layout of the LUNA code (Gadget-style `Header` and `PartType4` stars / `PartType5` black holes,
plus a placeholder `cells` grid and the `units` group), so LUNA's `read_snapshot` reads them; the layouts are
described in `src/io/snapshot.h` and `src/io/restart.h`.
