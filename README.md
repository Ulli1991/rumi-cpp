# cpp-nbody

Gravitational N-body integrator with adaptive individual timesteps. Implements a 4th-order Hermite integrator (FSI) inside a Hierarchical Hamiltonian Splitting (HHS) framework with FROST adaptive timestepping.

## Building on Raven

### CPU version (`src/`)

```bash
make
```

Produces `nbody`.

### GPU version (`src/`)

Load the required modules:

```bash
module load gcc/13 cuda/12.6 openmpi_gpu/5.0
```

#### Single GPU

```bash
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=$HOME/kokkos-cuda-mpi/bin/nvcc_wrapper \
  -DKokkos_ROOT=$HOME/kokkos-cuda-mpi
cmake --build build -j$(nproc)
```

#### Multi-GPU (MPI)

```bash
cmake -S . -B build-mpi \
  -DCMAKE_CXX_COMPILER=$HOME/kokkos-cuda-mpi/bin/nvcc_wrapper \
  -DKokkos_ROOT=$HOME/kokkos-cuda-mpi \
  -DUSE_MPI=ON \
  -DUSE_GPU_AWARE_MPI=ON
cmake --build build-mpi -j$(nproc)
```

Run with:

```bash
mpirun -np <nranks> ./build-mpi/nbody-kokkos
```

Both versions read `config.txt` from the working directory.
