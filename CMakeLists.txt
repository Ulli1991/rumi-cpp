cmake_minimum_required(VERSION 3.16)
project(nbody-kokkos LANGUAGES CXX)

# Find Kokkos (set Kokkos_ROOT or CMAKE_PREFIX_PATH)
find_package(Kokkos REQUIRED)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src-kokkos)

add_executable(nbody-kokkos
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/particles.cpp
    ${SRC_DIR}/forces.cpp
    ${SRC_DIR}/integrator.cpp
    ${SRC_DIR}/timestep.cpp
    ${SRC_DIR}/initial_conditions.cpp
    ${SRC_DIR}/diagnostics.cpp
)

target_include_directories(nbody-kokkos PRIVATE ${SRC_DIR})
target_link_libraries(nbody-kokkos Kokkos::kokkos)
target_compile_options(nbody-kokkos PRIVATE -O3)
