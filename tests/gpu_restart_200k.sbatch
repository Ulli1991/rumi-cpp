#!/bin/bash -l
# Restart files on raven GPUs: 200k cluster, 4 ranks x A100 and 1 rank (tests/gpu_restart_200k.py).
#   sbatch tests/gpu_restart_200k.sbatch <build dir with the CUDA build> <output dir>
#SBATCH -J nbody_restart_gpu
#SBATCH -o ./gpu_restart.out.%j
#SBATCH -e ./gpu_restart.err.%j
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=18
#SBATCH --mem=120000
#SBATCH --time=01:00:00

module purge
module load gcc/13 cuda/12.6 openmpi_gpu/5.0 hdf5-serial/1.14.1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_PROC_BIND=spread OMP_PLACES=threads
export OMPI_MCA_mpi_cuda_support=1
export KOKKOS_MAP_DEVICE_ID_BY=mpi_rank
export NBODY_TEST_TIMEOUT=3000
export NBODY_TEST_LAUNCHER="srun --ntasks=4 --cpus-per-task=$SLURM_CPUS_PER_TASK"
export ONE_RANK_LAUNCHER="srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK"
B=$(realpath "$1")
OUT=$2
export TMPDIR=${TMPDIR_RESTART:-/ptmp/$USER/nbody_restart_tmp}   # three runs' particle dumps (~15 MB per step)
mkdir -p "$TMPDIR"
nvidia-smi -L
/usr/bin/python3.11 "$(dirname "$B")/tests/gpu_restart_200k.py" "$B/nbody-kokkos" "$OUT"
