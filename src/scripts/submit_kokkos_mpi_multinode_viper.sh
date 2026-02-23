#!/bin/bash -l
#SBATCH -o ./job_mpi_multi.out.%j
#SBATCH -e ./job_mpi_multi.err.%j
#SBATCH -D ./
#SBATCH -J kokkos_nbody_multi
#SBATCH --constraint="apu"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=220000
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=uli@mpa-garching.mpg.de

# Load modules (must match build)
module purge
module load gcc/14 rocm/6.3 openmpi_gpu/5.0

# Print configuration
echo "=== Multi-Node Multi-GPU N-Body Run (Viper MI300A) ==="
echo "Nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total MPI ranks: $SLURM_NTASKS"
echo "GPUs per node: 2"
echo "Total GPUs: $(($SLURM_NNODES * 2))"
echo ""

# Set OpenMP threads per MPI rank
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# UCX optimizations for ROCm-aware communication
export UCX_TLS=rc,rocm_copy,rocm_ipc,gdr_copy
export UCX_RNDV_SCHEME=put_zcopy

# Run with MPI - each rank gets one APU via Kokkos device mapping
srun ./build-hip/nbody-kokkos --config config.txt --kokkos-map-device-id-by=mpi_rank
