# RUMI++ MSTAR module (GPLv3, see src/mstar/LICENSE): C++ rewrite of the MSTAR regularised integrator with Kokkos
# support for GPUs, by Ulrich Steinwandel. Included by the top-level CMakeLists.txt only with ENABLE_MSTAR=ON; a
# binary that links it is distributed under the GPLv3. The BSD-3 core reaches the module only through
# src/collisional/mstar_dispatch.h (BSD), implemented here in mstar_dispatch.cpp.
#
#   rumi-mstar                the C++/Kokkos integrator (production; C++ only)
#   ketju-reference           the KETJU C library (reference/ketju, eb96fd0, unchanged): test-time reference only
#   ketju-extra-force-ref     the KETJU extra-force branch (reference/ketju-extra-force, b60d16e): test reference
#   tests                     ketju_unit, mstar_compare, mstar_vs_logh, mstar_extra_force; mstar-bench
set(MSTAR_DIR ${CMAKE_CURRENT_LIST_DIR})

add_library(rumi-mstar STATIC ${MSTAR_DIR}/mstar_kokkos.cpp ${MSTAR_DIR}/mstar_mergers.cpp
            ${MSTAR_DIR}/mstar_dispatch.cpp)
target_include_directories(rumi-mstar PUBLIC ${MSTAR_DIR} PRIVATE ${SRC_DIR}/collisional ${SRC_DIR}/core)
target_link_libraries(rumi-mstar PUBLIC Kokkos::kokkos)
target_compile_features(rumi-mstar PUBLIC cxx_std_17)

# ---- test references (C, not linked into rumi-mstar or nbody-kokkos) ----
enable_language(C)
function(ketju_reference name dir)
    add_library(${name} STATIC ${dir}/src/init.c ${dir}/src/mst.c ${dir}/src/integration.c ${dir}/src/util.c
                ${dir}/src/pn.c ${dir}/src/bh_mergers.c ${dir}/src/debug.c)
    set_target_properties(${name} PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
    target_include_directories(${name} PUBLIC ${dir}/include)
    target_link_libraries(${name} PUBLIC m)
    if(USE_MPI)
        find_package(MPI REQUIRED COMPONENTS C)
        target_link_libraries(${name} PUBLIC MPI::MPI_C)
    else()
        target_include_directories(${name} PUBLIC ${MSTAR_DIR}/reference/serial_mpi)
    endif()
endfunction()
ketju_reference(ketju-reference ${MSTAR_DIR}/reference/ketju)
ketju_reference(ketju-extra-force-ref ${MSTAR_DIR}/reference/ketju-extra-force)
ketju_reference(ketju-extra-force-fwd-ref ${MSTAR_DIR}/reference/ketju-extra-force-fwd)

# KETJU's unit tests on a GoogleTest shim (the reference library itself)
add_executable(ketju-unit-tests ${MSTAR_DIR}/tests/ketju_unit_main.cpp
               ${MSTAR_DIR}/reference/ketju/unit-tests/integration.cpp
               ${MSTAR_DIR}/reference/ketju/unit-tests/mergers.cpp
               ${MSTAR_DIR}/reference/ketju/unit-tests/mst_construction.cpp)
target_include_directories(ketju-unit-tests PRIVATE ${MSTAR_DIR}/tests/gtest_shim)
target_link_libraries(ketju-unit-tests ketju-reference)
target_compile_features(ketju-unit-tests PRIVATE cxx_std_17)
add_test(NAME ketju_unit COMMAND ketju-unit-tests)

# C++/Kokkos MSTAR against the reference library (device and host engines)
add_executable(mstar-compare ${MSTAR_DIR}/tests/mstar_compare.cpp)
target_link_libraries(mstar-compare rumi-mstar ketju-reference)
add_test(NAME mstar_compare COMMAND mstar-compare)
set_tests_properties(mstar_compare PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1;OMP_PROC_BIND=false")

# Extra forces: against the branch patch forward-ported onto eb96fd0 (round-off) and independent references, and
# against the extra-force branch itself (tolerance level)
foreach(variant fwd branch)
    if(variant STREQUAL "fwd")
        set(_ef mstar-extra-force)
        set(_lib ketju-extra-force-fwd-ref)
    else()
        set(_ef mstar-extra-force-branch)
        set(_lib ketju-extra-force-ref)
    endif()
    add_executable(${_ef} ${MSTAR_DIR}/tests/mstar_extra_force.cpp)
    target_link_libraries(${_ef} rumi-mstar ${_lib})
    target_compile_definitions(${_ef} PRIVATE MSTAR_SAMPLE_IC="${MSTAR_DIR}/reference/ketju/nbody/nbody_sample_ic.txt")
    if(variant STREQUAL "branch")
        target_compile_definitions(${_ef} PRIVATE EF_BRANCH)
    endif()
    string(REPLACE "-" "_" _t ${_ef})
    add_test(NAME ${_t} COMMAND ${_ef})
    set_tests_properties(${_t} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1;OMP_PROC_BIND=false")
endforeach()

# MSTAR against RUMI++'s LogH + GBS (Newtonian few-body)
add_executable(mstar-vs-logh ${MSTAR_DIR}/tests/mstar_vs_logh.cpp ${SRC_DIR}/collisional/gbs.cpp
               ${SRC_DIR}/collisional/logh.cpp ${SRC_DIR}/collisional/logh_step.cpp
               ${SRC_DIR}/collisional/post_newtonian_core.cpp ${SRC_DIR}/collisional/kepler.cpp)
target_include_directories(mstar-vs-logh PRIVATE ${SRC_DIR}/collisional ${SRC_DIR}/core)
target_link_libraries(mstar-vs-logh rumi-mstar ketju-reference)
add_test(NAME mstar_vs_logh COMMAND mstar-vs-logh)
set_tests_properties(mstar_vs_logh PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1;OMP_PROC_BIND=false")

# Scaling benchmark (N_sub 16..4096) and launch/sync latency; not a ctest
add_executable(mstar-bench ${MSTAR_DIR}/tests/mstar_bench.cpp)
target_link_libraries(mstar-bench rumi-mstar ketju-reference)
