cmake_minimum_required(VERSION 3.16)
project(nbody-kokkos LANGUAGES CXX)

# Option to enable MPI
option(USE_MPI "Enable MPI for multi-node support" OFF)
option(USE_GPU_AWARE_MPI "Enable GPU-aware MPI (CUDA-aware/ROCm-aware)" OFF)
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
    ${SRC_DIR}/gravity/fmm.cpp
    ${SRC_DIR}/integrator/integrator.cpp
    ${SRC_DIR}/integrator/integrator_cpu.cpp
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

# Kokkos force kernel benchmark, counterpart of tools/hip_bench (tests/n2_bench.cpp)
add_executable(n2-bench ${CMAKE_CURRENT_SOURCE_DIR}/tests/n2_bench.cpp)
target_include_directories(n2-bench PRIVATE ${SRC_DIR}/core ${SRC_DIR}/gravity)
target_link_libraries(n2-bench Kokkos::kokkos)
target_compile_features(n2-bench PRIVATE cxx_std_17)
if(RSQRT_NEWTON)
    target_compile_definitions(n2-bench PRIVATE RSQRT_NEWTON)
endif()

# Exact tree time-step / neighbour search against the O(N^2) kernels: bitwise check and timings (tests/dt_tree_bench.cpp)
add_executable(dt-tree-bench ${CMAKE_CURRENT_SOURCE_DIR}/tests/dt_tree_bench.cpp)
target_include_directories(dt-tree-bench PRIVATE ${SRC_DIR}/core ${SRC_DIR}/gravity)
target_link_libraries(dt-tree-bench Kokkos::kokkos)
target_compile_features(dt-tree-bench PRIVATE cxx_std_17)
target_compile_options(dt-tree-bench PRIVATE $<$<CONFIG:Release>:-O3 -march=native>)
if(RSQRT_NEWTON)
    target_compile_definitions(dt-tree-bench PRIVATE RSQRT_NEWTON)
endif()

# FMM check against direct summation (tests/fmm_test.cpp)
add_executable(fmm-test ${CMAKE_CURRENT_SOURCE_DIR}/tests/fmm_test.cpp ${SRC_DIR}/gravity/fmm.cpp)
target_include_directories(fmm-test PRIVATE ${SRC_DIR}/core ${SRC_DIR}/gravity)
target_link_libraries(fmm-test Kokkos::kokkos)
target_compile_features(fmm-test PRIVATE cxx_std_17)
target_compile_features(nbody-kokkos PRIVATE cxx_std_17)

# Build type: Release (default), RelWithDebInfo, Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Per-build-type optimization flags
target_compile_options(nbody-kokkos PRIVATE
    -Wall -Wextra -Wno-unknown-pragmas
    $<$<CONFIG:Release>:-O3 -march=native>
    $<$<CONFIG:RelWithDebInfo>:-O2 -g -march=native>
    $<$<CONFIG:Debug>:-g -O0>
)

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

# Disable profiling fences (removes Kokkos::fence() around profiler regions for production runs)
option(ACCURATE_RSQRT "Device kernels use 1/sqrt (correctly rounded) instead of rsqrt (~1 ulp)" OFF)
if(ACCURATE_RSQRT)
    target_compile_definitions(nbody-kokkos PRIVATE ACCURATE_RSQRT)
endif()

option(RSQRT_NEWTON "Device rsqrt refined by one Newton step (~0.5 ulp; +4 % per 200k step, GPU accuracy = CPU)" ON)
if(RSQRT_NEWTON)
    target_compile_definitions(nbody-kokkos PRIVATE RSQRT_NEWTON)
endif()

option(DISABLE_PROFILING_FENCES "Remove Kokkos::fence() around profiler regions" OFF)
if(DISABLE_PROFILING_FENCES)
    target_compile_definitions(nbody-kokkos PRIVATE DISABLE_PROFILING_FENCES)
    message(STATUS "Profiling fences disabled")
endif()

# Tests: `ctest --test-dir <build>` runs each case through tests/run_case.py
enable_testing()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
foreach(case kepler_e09 kepler_e0 frost4_e09 frost4_e099 planets_hhs planets_fsi floor_binary_e0 floor_planets
             mergers_gpu_path mergers_cpu_path logh_big_group
             pn1_global_single_rank floor_binary_e0_cpu floor_planets_cpu
             binaries_timestep frost4_e09_cpu pn_logh_books
             mergers_sse binaries50_detected triples_detected
             cc_pn1_wide cc_binaries50 cc_vs_plain_binaries50 cm_cpu_vs_device cc_cm_consistent
             cm_eps_tidal_sampling cm_eps_cpu_vs_device dense_core_split triple_id_order secular_triples_off secular_omega0_steps spin_precession spin_j spin_j_pn1 ic_binaries_no_mergers eta_jerk0_fsi va_without_grad restart_halves_va secular_stellar_no_pn secular_bh_tight_steps logh_hierarchy_complete weak_flyby_split
             sdar_kozai_triple sdar_f3_flyby sdar_pn_kappa1
             cm_eps_tidal_components flyby_member_com
             coll_headon_triple coll_contact_flyby coll_contact_flyby_logh coll_tde_parabolic coll_bh_isco
             tree_dt_binaries tree_dt_binaries_cm tree_dt_binaries50 tree_dt_triples tree_dt_mergers tree_dt_softened tree_default_on
             ic_mass_size_rh ic_binary_com_virial ic_r_break)
    add_test(NAME ${case} COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tests/run_case.py
             $<TARGET_FILE:nbody-kokkos> ${case})
    # Few-body cases: host threads only add overhead (and timeouts); every incremental pid-map update is
    # checked against a full rebuild, and every subsystem must be advanced exactly once per subsystem call
    set_tests_properties(${case} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1;NBODY_CHECK_PIDMAP=1;NBODY_CHECK_SUBSYS=1")
endforeach()
if(USE_MPI)
    foreach(case spin_mpi2 mergers_mpi2 mpi_1v2 mergers_mpi2_dist mpi_1v2_dist)
        add_test(NAME ${case} COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tests/run_case.py
                 $<TARGET_FILE:nbody-kokkos> ${case})
        set_tests_properties(${case} PROPERTIES
                             ENVIRONMENT "NBODY_TEST_LAUNCHER=${MPIEXEC_EXECUTABLE} -n 2 --oversubscribe;OMP_NUM_THREADS=1;NBODY_CHECK_PIDMAP=1;NBODY_CHECK_SUBSYS=1")
    endforeach()
endif()
