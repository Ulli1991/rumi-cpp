#pragma once

// Tiny loops without a kernel launch where the host can access device memory (host backends, and HIP on APUs
// with unified memory such as the MI300A): the deep HHS levels run thousands of kernels on a few dozen
// particles, where a launch (~10-20 us) costs more than the work. Elsewhere (discrete GPUs) a normal parallel_for.
// NBODY_SMALL_FOR = largest n run on the host (default 256; 0 disables).

#include <Kokkos_Core.hpp>
#include <cstdlib>
#include "host_exec.h"

constexpr bool kHostAccessibleDevice =
    Kokkos::SpaceAccessibility<Kokkos::HostSpace, Kokkos::DefaultExecutionSpace::memory_space>::accessible;

inline int small_for_max() {
    static const int v = std::getenv("NBODY_SMALL_FOR") ? std::atoi(std::getenv("NBODY_SMALL_FOR")) : 256;
    return v;
}

template <class F>
void small_for(const char* name, int n, const F& f) {
    if (host_exec()) {   // host subtree: the data are in host memory
        for (int i = 0; i < n; i++) f(i);
        return;
    }
    if constexpr (kHostAccessibleDevice) {
        if (n <= small_for_max()) {
            Kokkos::fence("small_for.h:23");   // device work that produced the data
            for (int i = 0; i < n; i++) f(i);
            return;
        }
    }
    Kokkos::parallel_for(name, n, f);
}
