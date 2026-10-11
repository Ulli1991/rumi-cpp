#pragma once

// Persistent device buffers (index list + double payload) with host mirrors for small gathers and scatters of
// particle data. Deep HHS levels do thousands of these per step on a few dozen particles, where a View allocation
// (with its zero-fill kernel) and a host allocation per transfer cost more than the data. Buffers only grow.
// deep_copy fences, so a buffer may be refilled right after a kernel that read it was launched.

#include <Kokkos_Core.hpp>
#include <cstddef>

struct DeviceStaging {
    Kokkos::View<int*> d_idx;
    Kokkos::View<double*> d_buf;
    Kokkos::View<int*>::host_mirror_type h_idx;
    Kokkos::View<double*>::host_mirror_type h_buf;

    void reserve(int n, int stride) {
        if ((int)d_idx.extent(0) < n) {
            d_idx = Kokkos::View<int*>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "stage_idx"), n);
            h_idx = Kokkos::create_mirror_view(d_idx);
        }
        if (d_buf.extent(0) < (size_t)n * stride) {
            d_buf = Kokkos::View<double*>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "stage_buf"),
                                          (size_t)n * stride);
            h_buf = Kokkos::create_mirror_view(d_buf);
        }
    }
    void upload_idx(int n) {
        Kokkos::deep_copy(Kokkos::subview(d_idx, Kokkos::make_pair(0, n)),
                          Kokkos::subview(h_idx, Kokkos::make_pair(0, n)));
    }
    void upload_buf(size_t n) {
        Kokkos::deep_copy(Kokkos::subview(d_buf, Kokkos::make_pair((size_t)0, n)),
                          Kokkos::subview(h_buf, Kokkos::make_pair((size_t)0, n)));
    }
    void download_buf(size_t n) {
        Kokkos::deep_copy(Kokkos::subview(h_buf, Kokkos::make_pair((size_t)0, n)),
                          Kokkos::subview(d_buf, Kokkos::make_pair((size_t)0, n)));
    }

    // One instance per call site (tag), released at Kokkos::finalize
    template <int Tag>
    static DeviceStaging& get() {
        static DeviceStaging s;
        static bool hook = (Kokkos::push_finalize_hook([] { s = DeviceStaging(); }), true);
        (void)hook;
        return s;
    }
};
