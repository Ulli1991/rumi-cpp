# cpp-nbody

4th-order Hermite (FSI) and Hierarchical Hamiltonian Splitting (HHS) N-body integrator in C++. Includes a performance-portable Kokkos backend targeting CPU (OpenMP), NVIDIA (CUDA), AMD (HIP), and Intel (SYCL).

## Directory Structure

```
src/              # Original CPU implementation (scalar, SIMD hints)
src-kokkos/       # Kokkos port (performance-portable)
config.txt        # Shared simulation configuration
Makefile          # Build for src/ (original)
CMakeLists.txt    # Build for src-kokkos/ (Kokkos)
```

## Building the Original Code

```bash
make            # optimized build
make debug      # debug build with ASan/UBSan
```

Produces `nbody`.

## Building the Kokkos Version

### Prerequisites

- CMake >= 3.16
- C++17 compiler (GCC >= 10.4, Clang >= 14, or Intel LLVM >= 2022)
- Kokkos (installed via CMake)

### Installing Kokkos

```bash
git clone https://github.com/kokkos/kokkos.git kokkos-src

# CPU (OpenMP)
cmake -S kokkos-src -B kokkos-build \
  -DCMAKE_INSTALL_PREFIX=$HOME/kokkos \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build kokkos-build -j$(nproc)
cmake --install kokkos-build

# NVIDIA GPU (add to the cmake line above or build separately)
#  -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON

# AMD GPU
#  -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX90A=ON

# Intel GPU
#  -DKokkos_ENABLE_SYCL=ON
```

### Building nbody-kokkos

```bash
cmake -S . -B build-kokkos -DKokkos_ROOT=$HOME/kokkos
cmake --build build-kokkos -j$(nproc)
```

Produces `build-kokkos/nbody-kokkos`.

## Running

Both executables read `config.txt` from the working directory:

```bash
./nbody                              # original
./build-kokkos/nbody-kokkos          # Kokkos

# Override config file or test case
./build-kokkos/nbody-kokkos --config config.txt
./build-kokkos/nbody-kokkos --test plummer
```

For the Kokkos OpenMP backend, set thread binding for best performance:

```bash
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=8
```

## Configuration

Edit `config.txt`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TEST_CASE` | `outer_planets` | `outer_planets` (5-body) or `plummer` (N-body cluster) |
| `INTEGRATOR` | `hhs` | `fsi` (global timestep) or `hhs` (adaptive) |
| `HHS_MODE` | `recursive` | `recursive` or `stack` |
| `TOTAL_TIME_YEARS` | `1000` | Simulation duration |
| `DT_MAX_YEARS` | `1` | Coarsest timestep |
| `DT_MIN_YEARS` | `1e-7` | Finest timestep |
| `ETA` | `0.02` | FROST timestep parameter |
| `SOFTENING_AU` | `0.0` | Gravitational softening length |
| `GRAD_N3L` | `false` | Use Newton's 3rd law in gradient corrections |
| `USE_KAHAN` | `false` | Kahan summation in N3L gradient loops |
| `PLUMMER_N` | `1000` | Number of particles (plummer test) |
| `PLUMMER_M_TOTAL` | `379` | Total mass in solar masses |
| `PLUMMER_R_HALF` | `3.5` | Half-mass radius in parsecs |
| `PLUMMER_SEED` | `42` | RNG seed |
| `PRINT_INTERVAL` | `1000` | Steps between diagnostic output |

### Kokkos-only options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_N3L` | `true` (CPU) / `false` (GPU) | Newton's 3rd law in force kernels. N3L halves work but uses atomics — good for CPU, bad for GPU. Auto-detected from backend if not set. |
