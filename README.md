# RUMI++

RUMI++ is a **novel, performance-portable direct N-body code** for gravitational dynamics of stellar clusters. It is built on the **Kokkos 5.0** performance portability library and runs on configured CPU and GPU backends; optional MPI supports multi-GPU and multi-node runs. Direct pairwise gravity is the high-accuracy method, while an optional Barnes–Hut approximation is available for softened, non-collisional runs.

## Methods and features

- **Adaptive, fourth-order integration.** The Forward Symplectic Integrator (FSI) is composed with **Hierarchical Hamiltonian Splitting (HHS)**, recursively advancing slow and fast particle groups on adaptive individual timesteps. This follows the FROST method [1]. Pairwise timestep criteria resolve the wide range of orbital timescales in a cluster.
- **Direct gravity, with an optional tree.** Direct summation evaluates particle-particle forces. For softened collisionless problems, an optional Kokkos Barnes–Hut tree uses monopole approximations and group-based traversal [5]. Tree-based neighbor and timestep searches are also available.
- **Close-encounter dynamics** (compile with `-DENABLE_COLLISIONAL=ON`, enable `COLLISIONAL` in the run configuration). This mode uses unsoftened direct gravity and detects close pairs and connected few-body subsystems. Subsystems are evolved with LogH algorithmic regularization and Gragg–Bulirsch–Stoer (GBS) extrapolation; slow-down of weakly perturbed binaries is available through SDAR [3]. Secular binary and hierarchical-triple evolution are available when their orbits are sufficiently separated in timescale.
- **Compact-object and stellar physics.** Optional post-Newtonian corrections through 3.5PN, including gravitational-wave radiation reaction and spin couplings, act on compact-object encounters [2, 6]. The collisional mode also supports collisions, mergers and merger trees, single-star evolution with configurable wind and remnant prescriptions, and external galactic tidal fields.
- **Initial conditions and output.** Generate Plummer clusters with equal-mass or Kroupa mass distributions, primordial binaries and an optional central IMBH binary, or use built-in few-body and outer-Solar-System test cases. HDF5 snapshots use a LUNA-readable layout; HDF5 restarts preserve the state needed for bit-identical continuation.

Collisional features are excluded from builds made without `ENABLE_COLLISIONAL`.

### Method references

1. Rantala, Naab & Springel (2021), FROST: [MNRAS paper](https://doi.org/10.1093/mnras/stab057) — HHS-FSI.
2. Rantala et al. (2023): [MNRAS paper](https://doi.org/10.1093/mnras/stad1360) — collisional cluster methods, regularization and relativistic corrections.
3. Wang, Nitadori & Makino (2020), SDAR: [arXiv:2002.07938](https://arxiv.org/abs/2002.07938).
4. Hurley, Pols & Tout (2000), single-star evolution: [MNRAS paper](https://doi.org/10.1046/j.1365-8711.2000.03426.x).
5. Barnes & Hut (1986), hierarchical force calculation: [Nature paper](https://doi.org/10.1038/324446a0).
6. Blanchet (2014), post-Newtonian theory: [Living Reviews in Relativity](https://doi.org/10.12942/lrr-2014-2).

## Building

Requires a Kokkos 5.0 installation configured for the target backend (CUDA, HIP, SYCL, or OpenMP for CPU).

```bash
git clone https://github.com/Ulli1991/rumi-cpp.git && cd rumi-cpp
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
| `LUNA_PYTHON` | empty | Python with LUNA importable, for the `luna_reader` test |

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

Single-GPU runs use `mpirun -np 1`. All configuration is read from `config.txt` in the working directory
(or `--config <file>`). `config.txt` holds the choices of a run; every key, its default and level are in
[docs/CONFIG.md](docs/CONFIG.md), generated from the parameter table `src/core/params.cpp`. An unknown key is an
error; each run writes the effective value of every key to `run_params.txt`.

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
