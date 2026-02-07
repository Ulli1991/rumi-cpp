#!/bin/bash -l
#SBATCH -o ./job_mpi_multi.out.%j
#SBATCH -e ./job_mpi_multi.err.%j
#SBATCH -D ./
#SBATCH -J kokkos_nbody_multi
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=18
#SBATCH --mem=500000
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=uli@mpa-garching.mpg.de

# Load modules for CUDA-aware MPI
module purge
module load gcc/13 cuda/12.6 openmpi_gpu/5.0

# Print configuration
echo "=== Multi-Node Multi-GPU N-Body Run ==="
echo "Nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total MPI ranks: $SLURM_NTASKS"
echo "GPUs per node: 4"
echo "Total GPUs: $(($SLURM_NNODES * 4))"
echo ""

# Set OpenMP threads per MPI rank
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Enable CUDA-aware MPI
export OMPI_MCA_mpi_cuda_support=1

# UCX optimizations for GPU-aware communication
export UCX_TLS=rc,cuda_copy,cuda_ipc,gdr_copy
export UCX_RNDV_SCHEME=put_zcopy

# Run with MPI - each rank gets one GPU via Kokkos device mapping
srun ./build-mpi/nbody-kokkos --config config.txt --kokkos-map-device-id-by=mpi_rank
