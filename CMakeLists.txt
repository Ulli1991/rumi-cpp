cmake_minimum_required(VERSION 3.16)
project(nbody-kokkos LANGUAGES CXX)

# Option to enable MPI
option(USE_MPI "Enable MPI for multi-node support" OFF)
option(USE_GPU_AWARE_MPI "Enable GPU-aware MPI (CUDA-aware/ROCm-aware)" OFF)
option(ENABLE_COLLISIONAL "Compile collisional (unsoftened) support" OFF)
# HDF5 restart files and LUNA-compatible snapshots (src/io/restart.*, src/io/snapshot.*): on when HDF5 is found
option(USE_HDF5 "HDF5 restart files and snapshots (needs the HDF5 C library, e.g. module hdf5-serial)" ON)

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
    ${SRC_DIR}/core/params.cpp
    ${SRC_DIR}/core/mpi_comm.cpp
    ${SRC_DIR}/gravity/forces.cpp
    ${SRC_DIR}/gravity/tidal_field.cpp
    ${SRC_DIR}/gravity/tree.cpp
    ${SRC_DIR}/gravity/tree_forces.cpp
    ${SRC_DIR}/gravity/fmm.cpp
    ${SRC_DIR}/integrator/integrator.cpp
    ${SRC_DIR}/integrator/timestep.cpp
    ${SRC_DIR}/ic_gen/initial_conditions.cpp
    ${SRC_DIR}/ic_gen/sse.cpp
    ${SRC_DIR}/io/diagnostics.cpp
    ${SRC_DIR}/io/cluster_diag.cpp
)

# Collisional source files (only compiled when ENABLE_COLLISIONAL=ON)
if(ENABLE_COLLISIONAL)
    target_sources(nbody-kokkos PRIVATE
        ${SRC_DIR}/collisional/kepler.cpp
        ${SRC_DIR}/collisional/subsystem.cpp
        ${SRC_DIR}/collisional/logh.cpp
        ${SRC_DIR}/collisional/gbs.cpp
        ${SRC_DIR}/collisional/logh_step.cpp
        ${SRC_DIR}/collisional/logh_batch.cpp
        ${SRC_DIR}/collisional/secular.cpp
        ${SRC_DIR}/collisional/secular_triple_step.cpp
        ${SRC_DIR}/collisional/post_newtonian_core.cpp
        ${SRC_DIR}/collisional/post_newtonian.cpp
        ${SRC_DIR}/collisional/mergers.cpp
        ${SRC_DIR}/collisional/merger_tree.cpp
        ${SRC_DIR}/collisional/mstar.cpp
        ${SRC_DIR}/collisional/checkpoint.cpp
        ${SRC_DIR}/collisional/runtime_sse.cpp
    )
endif()

if(USE_HDF5)
    find_package(HDF5 COMPONENTS C)
    if(HDF5_FOUND)
        target_sources(nbody-kokkos PRIVATE ${SRC_DIR}/io/h5_util.cpp ${SRC_DIR}/io/restart.cpp ${SRC_DIR}/io/snapshot.cpp)
        target_include_directories(nbody-kokkos PRIVATE ${HDF5_INCLUDE_DIRS})
        target_link_libraries(nbody-kokkos ${HDF5_C_LIBRARIES})
        target_compile_definitions(nbody-kokkos PRIVATE USE_HDF5 ${HDF5_DEFINITIONS})
        message(STATUS "HDF5 ${HDF5_VERSION}: restart files and snapshots enabled")
    else()
        message(WARNING "HDF5 not found: no restart files or snapshots (set HDF5_ROOT, or USE_HDF5=OFF)")
    endif()
endif()
# git description of the source tree, stored in restart files and snapshots (refreshed at every build)
add_custom_target(nbody_git_version
    COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_CURRENT_SOURCE_DIR} -DOUT=${CMAKE_BINARY_DIR}/generated/git_version.h
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/git_version.cmake
    BYPRODUCTS ${CMAKE_BINARY_DIR}/generated/git_version.h)
add_dependencies(nbody-kokkos nbody_git_version)

target_include_directories(nbody-kokkos PRIVATE
    ${CMAKE_BINARY_DIR}/generated
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

# Exact tree time-step / neighbour search against the O(N^2) kernels: bitwise check and timings (tests/dt_tree_bench.cpp)
add_executable(dt-tree-bench ${CMAKE_CURRENT_SOURCE_DIR}/tests/dt_tree_bench.cpp)
target_include_directories(dt-tree-bench PRIVATE ${SRC_DIR}/core ${SRC_DIR}/gravity)
target_link_libraries(dt-tree-bench Kokkos::kokkos)
target_compile_features(dt-tree-bench PRIVATE cxx_std_17)
target_compile_options(dt-tree-bench PRIVATE $<$<CONFIG:Release>:-O3 -march=native>)
if(RSQRT_NEWTON)
    target_compile_definitions(dt-tree-bench PRIVATE RSQRT_NEWTON)
endif()

# GW recoil formula with spins (tests/recoil_test.cpp; ctest case gw_recoil_unit)
add_executable(recoil-test ${CMAKE_CURRENT_SOURCE_DIR}/tests/recoil_test.cpp)
target_include_directories(recoil-test PRIVATE ${SRC_DIR}/collisional)
target_compile_features(recoil-test PRIVATE cxx_std_17)
# Very massive stars in the IC stellar evolution (tests/sse_test.cpp; ctest case sse_vms_unit)
add_executable(sse-test ${CMAKE_CURRENT_SOURCE_DIR}/tests/sse_test.cpp ${SRC_DIR}/ic_gen/sse.cpp)
target_include_directories(sse-test PRIVATE ${SRC_DIR}/ic_gen ${SRC_DIR})
target_compile_features(sse-test PRIVATE cxx_std_17)
# Two-body drift check (tests/kepler_test.cpp; ctest case kepler_drift_unit)
if(ENABLE_COLLISIONAL)
    add_executable(kepler-test ${CMAKE_CURRENT_SOURCE_DIR}/tests/kepler_test.cpp ${SRC_DIR}/collisional/kepler.cpp)
    # PN1 energy against the PN1 equations of motion (tests/pn_energy_test.cpp; ctest case pn1_energy_consistency)
    add_executable(pn-energy-test ${CMAKE_CURRENT_SOURCE_DIR}/tests/pn_energy_test.cpp ${SRC_DIR}/collisional/post_newtonian_core.cpp)
    target_include_directories(pn-energy-test PRIVATE ${SRC_DIR}/collisional)
    target_compile_features(pn-energy-test PRIVATE cxx_std_17)
    # PN2-3.5 against Blanchet's review: flux balance, Kepler's law, A_2PN/B_2PN (tests/pn_highorder_test.cpp; ctest pn_highorder)
    add_executable(pn-highorder-test ${CMAKE_CURRENT_SOURCE_DIR}/tests/pn_highorder_test.cpp ${SRC_DIR}/collisional/post_newtonian_core.cpp)
    target_include_directories(pn-highorder-test PRIVATE ${SRC_DIR}/collisional)
    target_compile_features(pn-highorder-test PRIVATE cxx_std_17)
    # Orbit-averaged PN two-body evolution against the equations of motion (tests/secular_pn_test.cpp; ctest secular_pn)
    add_executable(secular-pn-test ${CMAKE_CURRENT_SOURCE_DIR}/tests/secular_pn_test.cpp ${SRC_DIR}/collisional/secular.cpp
                   ${SRC_DIR}/collisional/kepler.cpp ${SRC_DIR}/collisional/post_newtonian_core.cpp)
    target_include_directories(secular-pn-test PRIVATE ${SRC_DIR}/collisional)
    target_compile_features(secular-pn-test PRIVATE cxx_std_17)
    target_include_directories(kepler-test PRIVATE ${SRC_DIR}/collisional)
    target_compile_features(kepler-test PRIVATE cxx_std_17)
endif()

# FSI gradient force against grad(1/2 sum m |a|^2) (tests/grad_force_test.cpp; ctest case grad_force_consistency)
add_executable(grad-force-test ${CMAKE_CURRENT_SOURCE_DIR}/tests/grad_force_test.cpp)
target_include_directories(grad-force-test PRIVATE ${SRC_DIR}/core ${SRC_DIR}/gravity)
target_link_libraries(grad-force-test PRIVATE Kokkos::kokkos)
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

    # NCCL allgathers (MPI_TRANSPORT = nccl, the default when built in; CUDA only). NCCL_ROOT or $NCCL_HOME
    # (raven: module load nccl/2.23.4 after cuda/12.6)
    option(USE_NCCL "Per-particle allgathers through NCCL (CUDA builds)" OFF)
    if(USE_NCCL)
        find_path(NCCL_INCLUDE_DIR nccl.h HINTS ${NCCL_ROOT} $ENV{NCCL_HOME} PATH_SUFFIXES include REQUIRED)
        find_library(NCCL_LIBRARY nccl HINTS ${NCCL_ROOT} $ENV{NCCL_HOME} PATH_SUFFIXES lib lib64 REQUIRED)
        target_include_directories(nbody-kokkos PRIVATE ${NCCL_INCLUDE_DIR})
        target_link_libraries(nbody-kokkos ${NCCL_LIBRARY})
        target_compile_definitions(nbody-kokkos PRIVATE USE_NCCL)
        message(STATUS "NCCL allgathers: ${NCCL_LIBRARY}")
    endif()

endif()

# Disable profiling fences (removes Kokkos::fence() around profiler regions for production runs)
option(ACCURATE_RSQRT "Device kernels use 1/sqrt (correctly rounded) instead of rsqrt (~1 ulp)" OFF)
if(ACCURATE_RSQRT)
    target_compile_definitions(nbody-kokkos PRIVATE ACCURATE_RSQRT)
    target_compile_definitions(n2-bench PRIVATE ACCURATE_RSQRT)   # the benchmark times the production kernels
endif()

option(RSQRT_NEWTON "Device rsqrt refined by one Newton step (~0.5 ulp; +4 % per 200k step, GPU accuracy = CPU)" ON)
if(RSQRT_NEWTON)
    target_compile_definitions(nbody-kokkos PRIVATE RSQRT_NEWTON)
    target_compile_definitions(n2-bench PRIVATE RSQRT_NEWTON)   # the benchmark times the production kernels
endif()

option(DT_FP32_PREREJECT "Device time-step kernel: single-precision conservative pre-reject of distant pairs (bit-identical)" ON)
if(DT_FP32_PREREJECT)
    target_compile_definitions(nbody-kokkos PRIVATE DT_FP32_PREREJECT)
    target_compile_definitions(n2-bench PRIVATE DT_FP32_PREREJECT)   # the benchmark times the production kernels
endif()

option(DISABLE_PROFILING_FENCES "Remove Kokkos::fence() around profiler regions" OFF)
if(DISABLE_PROFILING_FENCES)
    target_compile_definitions(nbody-kokkos PRIVATE DISABLE_PROFILING_FENCES)
    message(STATUS "Profiling fences disabled")
endif()

# Tests: `ctest --test-dir <build>` runs each case through tests/run_case.py
enable_testing()
add_test(NAME gw_recoil_unit COMMAND recoil-test)
add_test(NAME sse_vms_unit COMMAND sse-test)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
# Parameter table (src/core/params.cpp) against docs/CONFIG.md, config.txt, the cfg.get_* reads and the NBODY_* getenvs
add_test(NAME config_table COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/scripts/gen_config_md.py --check)
foreach(case kepler_e09 kepler_e0 frost4_e09 frost4_e099 planets_hhs planets_fsi floor_binary_e0 floor_planets
             mergers_gpu_path mergers_cpu_path logh_big_group
             pn1_global_single_rank
             binaries_timestep frost4_e09_cpu pn_logh_books
             mergers_sse binaries50_detected triples_detected
             cc_pn1_wide cc_binaries50 cc_min_binaries50 cc_vs_plain_binaries50 cc_consistent_binaries50 force_err_injection cm_cpu_vs_device cc_cm_consistent
             cm_eps_tidal_sampling cm_eps_cpu_vs_device pn_bh_only flyby_group_floor weak_split_bh_binary dense_core_split triple_id_order secular_triples_off sec_triple_energy adv_kozai_triple secular_omega0_steps spin_precession spin_j spin_j_pn1 ic_binaries_no_mergers eta_jerk0_fsi va_without_grad restart_halves_va secular_stellar_no_pn secular_bh_tight_steps logh_hierarchy_complete
             sdar_kozai_triple sdar_f3_flyby sdar_pn_kappa1
             cm_eps_tidal_components flyby_member_com ngb_mass_scaling
             coll_headon_triple coll_contact_flyby coll_contact_flyby_logh coll_tde_parabolic coll_tde_accretion fryer_rapid_gap pn_boost_logh pn_boost_secular wide_binaries_ngb gbs_deuflhard_pn cluster_diag tidal_ref_orbit tidal_energy_books tidal_jacobi_escapers ss_companion_flyby int_pos_flyby quad22_sdar particle_file_roundtrip grad_only_planets bh_merge_radius_fc bh_merge_radius_total coll_mass_loss coll_bh_isco natal_kicks_ns natal_kicks_fallback
             tree_dt_binaries tree_dt_binaries_host tree_dt_binaries_cm tree_dt_binaries50 tree_dt_triples tree_dt_mergers tree_dt_softened tree_default_on
             tree_dt_tidal tree_dt_tidal_host tree_dt_tidal_cm
             host_subtree_mergers host_subtree_binaries host_subtree_triples
             triple_stability ic_mass_size_rh ic_binary_com_virial ic_r_break
             secular_drift_e0 secular_drift_e1e-8
             kepler2_norb05 kepler2_norb2 kepler2_hyperbolic kepler2_contact)
    add_test(NAME ${case} COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tests/run_case.py
             $<TARGET_FILE:nbody-kokkos> ${case})
    # Few-body cases: host threads only add overhead (and timeouts); every incremental pid-map update is
    # checked against a full rebuild, every subsystem must be advanced exactly once per subsystem call, and the
    # pair EIH evaluation must equal the two single-body calls bit for bit, and every cached stellar type and radius
    # (stellar_cache.h) must equal the device value
    set_tests_properties(${case} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1;NBODY_CHECK_PIDMAP=1;NBODY_CHECK_SUBSYS=1;NBODY_CHECK_EIH=1;NBODY_CHECK_STELLAR_CACHE=1")
endforeach()
if(ENABLE_COLLISIONAL)
    add_test(NAME kepler_drift_unit COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tests/run_case.py
             $<TARGET_FILE:kepler-test> kepler_drift_unit)
    add_test(NAME pn1_energy_consistency COMMAND pn-energy-test)
    add_test(NAME pn_highorder COMMAND pn-highorder-test)
    add_test(NAME secular_pn COMMAND secular-pn-test)
endif()
add_test(NAME grad_force_consistency COMMAND grad-force-test)
if(USE_MPI)
    foreach(case spin_mpi2 mergers_mpi2 mpi_1v2 rankcount_binaries rankcount_mergers rankcount_triples rankcount_natal_kicks)
        add_test(NAME ${case} COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tests/run_case.py
                 $<TARGET_FILE:nbody-kokkos> ${case})
        set_tests_properties(${case} PROPERTIES
                             ENVIRONMENT "NBODY_TEST_LAUNCHER=${MPIEXEC_EXECUTABLE} -n 2 --oversubscribe;OMP_NUM_THREADS=1;NBODY_CHECK_PIDMAP=1;NBODY_CHECK_SUBSYS=1;NBODY_CHECK_STELLAR_CACHE=1")
    endforeach()
endif()

# Restart files and snapshots (src/io): bit-identical continuation, snapshot round trip, LUNA's reader
if(USE_HDF5 AND HDF5_FOUND)
    add_executable(h5-roundtrip ${CMAKE_CURRENT_SOURCE_DIR}/tests/h5_roundtrip.cpp ${SRC_DIR}/io/h5_util.cpp)
    target_include_directories(h5-roundtrip PRIVATE ${SRC_DIR}/io ${HDF5_INCLUDE_DIRS})
    target_link_libraries(h5-roundtrip ${HDF5_C_LIBRARIES})
    target_compile_features(h5-roundtrip PRIVATE cxx_std_17)
    add_dependencies(nbody-kokkos h5-roundtrip)
    # LUNA_PYTHON: a Python in which LUNA (/raven/u/uli/LUNA) imports, for the luna_reader case (skipped if empty)
    set(LUNA_PYTHON "" CACHE FILEPATH "Python with LUNA importable (luna_reader test)")
    foreach(case restart_binaries restart_walltime restart_triples stall_capture restart_mergers restart_ngb_mergers restart_tidal restart_mergers_sse restart_natal_kicks restart_pn_logh
                 restart_spin restart_secular restart_secular_triple restart_sdar restart_tidal_sampling restart_cm_velocity restart_spin_imbh
                 restart_fsi snapshot_roundtrip snapshot_cluster_header luna_reader)
        add_test(NAME ${case} COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tests/run_case.py
                 $<TARGET_FILE:nbody-kokkos> ${case})
        set_tests_properties(${case} PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT
            "OMP_NUM_THREADS=1;NBODY_CHECK_PIDMAP=1;NBODY_CHECK_SUBSYS=1;NBODY_CHECK_EIH=1;LUNA_PYTHON=${LUNA_PYTHON}")
    endforeach()
    if(USE_MPI)
        foreach(case restart_mpi2 restart_mpi2_mergers)
            add_test(NAME ${case} COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tests/run_case.py
                     $<TARGET_FILE:nbody-kokkos> ${case})
            set_tests_properties(${case} PROPERTIES ENVIRONMENT
                "NBODY_TEST_LAUNCHER=${MPIEXEC_EXECUTABLE} -n 2 --oversubscribe;OMP_NUM_THREADS=1;NBODY_CHECK_PIDMAP=1;NBODY_CHECK_SUBSYS=1")
        endforeach()
    endif()
endif()

