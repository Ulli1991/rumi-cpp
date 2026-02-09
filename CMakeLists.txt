cmake_minimum_required(VERSION 3.16)
project(nbody-kokkos LANGUAGES CXX)

# Option to enable MPI
option(USE_MPI "Enable MPI for multi-node support" OFF)
option(USE_GPU_AWARE_MPI "Enable GPU-aware MPI (CUDA-aware/ROCm-aware)" OFF)
option(MIXED_PRECISION "FP32 rsqrt seed + FP64 Newton-Raphson (~1e-14 precision)" OFF)

# Find Kokkos (set Kokkos_ROOT or CMAKE_PREFIX_PATH)
find_package(Kokkos REQUIRED)

# Find MPI if enabled
if(USE_MPI)
    find_package(MPI REQUIRED)
endif()

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src-kokkos)

add_executable(nbody-kokkos
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/particles.cpp
    ${SRC_DIR}/forces.cpp
    ${SRC_DIR}/integrator.cpp
    ${SRC_DIR}/timestep.cpp
    ${SRC_DIR}/initial_conditions.cpp
    ${SRC_DIR}/diagnostics.cpp
    ${SRC_DIR}/mpi_comm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/initial_conditions/sse.cpp
)

target_include_directories(nbody-kokkos PRIVATE ${SRC_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/initial_conditions)
target_link_libraries(nbody-kokkos Kokkos::kokkos)
target_compile_options(nbody-kokkos PRIVATE -O3)

# Mixed precision rsqrt (compile-time flag for zero register overhead)
if(MIXED_PRECISION)
    target_compile_definitions(nbody-kokkos PRIVATE MIXED_PRECISION)
    message(STATUS "Mixed precision rsqrt enabled (FP32 seed + FP64 Newton-Raphson)")
endif()

# Add MPI support if enabled
if(USE_MPI)
    target_compile_definitions(nbody-kokkos PRIVATE USE_MPI)
    target_link_libraries(nbody-kokkos MPI::MPI_CXX)

    # GPU-aware MPI (CUDA-aware / ROCm-aware)
    if(USE_GPU_AWARE_MPI)
        target_compile_definitions(nbody-kokkos PRIVATE USE_GPU_AWARE_MPI=1)
        message(STATUS "GPU-aware MPI enabled - ensure your MPI supports CUDA/ROCm")
    endif()
endif()
