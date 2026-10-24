# RUMI++ MSTAR module (GPLv3, see src/mstar/LICENSE). Included by the top-level CMakeLists.txt only with
# ENABLE_MSTAR=ON; a binary that links it is distributed under the GPLv3. The BSD-3 core reaches the module only
# through src/collisional/mstar_dispatch.h (BSD), implemented here in mstar_dispatch.cpp.
#
#   ketju-integrator  the vendored KETJU C library (src/mstar/ketju, unchanged; serial MPI stand-in without USE_MPI)
#   rumi-mstar        the Kokkos port (mstar_kokkos.cpp) + the dispatch implementation
#   tests             ketju_unit (their unit tests), mstar_compare (port vs C library), mstar_vs_logh (vs LogH)
enable_language(C)
set(MSTAR_DIR ${CMAKE_CURRENT_LIST_DIR})

add_library(ketju-integrator STATIC
    ${MSTAR_DIR}/ketju/src/init.c ${MSTAR_DIR}/ketju/src/mst.c ${MSTAR_DIR}/ketju/src/integration.c
    ${MSTAR_DIR}/ketju/src/util.c ${MSTAR_DIR}/ketju/src/pn.c ${MSTAR_DIR}/ketju/src/bh_mergers.c
    ${MSTAR_DIR}/ketju/src/debug.c)
set_target_properties(ketju-integrator PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
target_include_directories(ketju-integrator PUBLIC ${MSTAR_DIR}/ketju/include)
target_link_libraries(ketju-integrator PUBLIC m)
if(USE_MPI)
    find_package(MPI REQUIRED COMPONENTS C)
    target_link_libraries(ketju-integrator PUBLIC MPI::MPI_C)
else()
    target_include_directories(ketju-integrator PUBLIC ${MSTAR_DIR}/serial_mpi)
endif()

add_library(rumi-mstar STATIC ${MSTAR_DIR}/mstar_kokkos.cpp ${MSTAR_DIR}/mstar_dispatch.cpp)
target_include_directories(rumi-mstar PUBLIC ${MSTAR_DIR} PRIVATE ${SRC_DIR}/collisional ${SRC_DIR}/core)
target_link_libraries(rumi-mstar PUBLIC ketju-integrator Kokkos::kokkos)
target_compile_features(rumi-mstar PUBLIC cxx_std_17)

# KETJU's unit tests on a GoogleTest shim
add_executable(ketju-unit-tests ${MSTAR_DIR}/tests/ketju_unit_main.cpp ${MSTAR_DIR}/ketju/unit-tests/integration.cpp
               ${MSTAR_DIR}/ketju/unit-tests/mergers.cpp ${MSTAR_DIR}/ketju/unit-tests/mst_construction.cpp)
target_include_directories(ketju-unit-tests PRIVATE ${MSTAR_DIR}/tests/gtest_shim)
target_link_libraries(ketju-unit-tests ketju-integrator)
target_compile_features(ketju-unit-tests PRIVATE cxx_std_17)
add_test(NAME ketju_unit COMMAND ketju-unit-tests)

# Kokkos port against the C library
add_executable(mstar-compare ${MSTAR_DIR}/tests/mstar_compare.cpp)
target_link_libraries(mstar-compare rumi-mstar)
add_test(NAME mstar_compare COMMAND mstar-compare)
set_tests_properties(mstar_compare PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1;OMP_PROC_BIND=false")

# MSTAR against RUMI++'s LogH + GBS (Newtonian few-body)
add_executable(mstar-vs-logh ${MSTAR_DIR}/tests/mstar_vs_logh.cpp ${SRC_DIR}/collisional/gbs.cpp
               ${SRC_DIR}/collisional/logh.cpp ${SRC_DIR}/collisional/logh_step.cpp
               ${SRC_DIR}/collisional/post_newtonian_core.cpp ${SRC_DIR}/collisional/kepler.cpp)
target_include_directories(mstar-vs-logh PRIVATE ${SRC_DIR}/collisional ${SRC_DIR}/core)
target_link_libraries(mstar-vs-logh rumi-mstar)
add_test(NAME mstar_vs_logh COMMAND mstar-vs-logh)
set_tests_properties(mstar_vs_logh PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1;OMP_PROC_BIND=false")

# Scaling benchmark (N_sub 16..4096) and launch/sync latency; not a ctest
add_executable(mstar-bench ${MSTAR_DIR}/tests/mstar_bench.cpp)
target_link_libraries(mstar-bench rumi-mstar)
