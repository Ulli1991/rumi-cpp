cmake_minimum_required(VERSION 3.16)
project(nbody-kokkos LANGUAGES CXX)

# Option to enable MPI
option(USE_MPI "Enable MPI for multi-node support" OFF)
option(USE_GPU_AWARE_MPI "Enable GPU-aware MPI (CUDA-aware/ROCm-aware)" OFF)
option(MPI_TOPOLOGY_RING "Use ring/j-partitioned MPI topology (allreduce instead of allgatherv)" OFF)
option(ENABLE_COLLISIONAL "Compile collisional (unsoftened) support" OFF)

# Find Kokkos (set Kokkos_ROOT or CMAKE_PREFIX_PATH)
find_package(Kokkos REQUIRED)

# Find MPI if enabled
if(USE_MPI)
    find_package(MPI REQUIRED)
endif()

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(nbody-kokkos
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/core/particles.cpp
    ${SRC_DIR}/core/mpi_comm.cpp
    ${SRC_DIR}/gravity/forces.cpp
    ${SRC_DIR}/gravity/tree.cpp
    ${SRC_DIR}/gravity/tree_forces.cpp
    ${SRC_DIR}/integrator/integrator.cpp
    ${SRC_DIR}/integrator/timestep.cpp
    ${SRC_DIR}/ic_gen/initial_conditions.cpp
    ${SRC_DIR}/ic_gen/sse.cpp
    ${SRC_DIR}/io/diagnostics.cpp
)

# Collisional source files (only compiled when ENABLE_COLLISIONAL=ON)
if(ENABLE_COLLISIONAL)
    target_sources(nbody-kokkos PRIVATE
        ${SRC_DIR}/collisional/kepler.cpp
        ${SRC_DIR}/collisional/subsystem.cpp
        ${SRC_DIR}/collisional/logh.cpp
        ${SRC_DIR}/collisional/gbs.cpp
        ${SRC_DIR}/collisional/logh_batch.cpp
        ${SRC_DIR}/collisional/secular.cpp
        ${SRC_DIR}/collisional/post_newtonian.cpp
        ${SRC_DIR}/collisional/mergers.cpp
        ${SRC_DIR}/collisional/merger_tree.cpp
        ${SRC_DIR}/collisional/mstar.cpp
        ${SRC_DIR}/collisional/checkpoint.cpp
        ${SRC_DIR}/collisional/runtime_sse.cpp
    )
endif()

target_include_directories(nbody-kokkos PRIVATE
    ${SRC_DIR}/core
    ${SRC_DIR}/gravity
    ${SRC_DIR}/integrator
    ${SRC_DIR}/ic_gen
    ${SRC_DIR}/io
)
if(ENABLE_COLLISIONAL)
    target_include_directories(nbody-kokkos PRIVATE ${SRC_DIR}/collisional)
    target_compile_definitions(nbody-kokkos PRIVATE ENABLE_COLLISIONAL)
    message(STATUS "Collisional (unsoftened direct N-body) support enabled")
endif()
target_link_libraries(nbody-kokkos Kokkos::kokkos)
target_compile_features(nbody-kokkos PRIVATE cxx_std_17)
target_compile_options(nbody-kokkos PRIVATE -O3 -march=native)

# Add MPI support if enabled
if(USE_MPI)
    target_compile_definitions(nbody-kokkos PRIVATE USE_MPI)
    target_link_libraries(nbody-kokkos MPI::MPI_CXX)

    # GPU-aware MPI (CUDA-aware / ROCm-aware)
    if(USE_GPU_AWARE_MPI)
        target_compile_definitions(nbody-kokkos PRIVATE USE_GPU_AWARE_MPI=1)
        message(STATUS "GPU-aware MPI enabled - ensure your MPI supports CUDA/ROCm")
    endif()

    # Ring topology: j-partitioned force compute with MPI_Allreduce
    if(MPI_TOPOLOGY_RING)
        target_compile_definitions(nbody-kokkos PRIVATE MPI_TOPOLOGY_RING)
        message(STATUS "MPI topology: RING (j-partitioned, allreduce)")
    endif()
endif()
