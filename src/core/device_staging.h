#pragma once

// Persistent device buffers (index list + double payload) with host mirrors for small gathers and scatters of
// particle data. Deep HHS levels do thousands of these per step on a few dozen particles, where a View allocation
// (with its zero-fill kernel) and a host allocation per transfer cost more than the data. Buffers only grow.
// deep_copy fences, so a buffer may be refilled right after a kernel that read it was launched.

#include <Kokkos_Core.hpp>
#include "lean_sync.h"
#include <cstddef>
#include <memory>
#include <vector>

struct DeviceStaging {
    Kokkos::View<int*> d_idx;
    Kokkos::View<double*> d_buf;
    Kokkos::View<int*>::host_mirror_type h_idx;
    Kokkos::View<double*>::host_mirror_type h_buf;

    bool host = false;   // host twin (host subtrees, host_exec.h): d_* and h_* share host memory
    std::shared_ptr<std::vector<int>> host_idx;
    std::shared_ptr<std::vector<double>> host_buf;

    void reserve(int n, int stride) {
        if (host) {
            if ((int)d_idx.extent(0) < n) {
                host_idx = std::make_shared<std::vector<int>>(n);
                d_idx = Kokkos::View<int*>(host_idx->data(), n);
                h_idx = decltype(h_idx)(host_idx->data(), n);
            }
            if (d_buf.extent(0) < (size_t)n * stride) {
                host_buf = std::make_shared<std::vector<double>>((size_t)n * stride);
                d_buf = Kokkos::View<double*>(host_buf->data(), (size_t)n * stride);
                h_buf = decltype(h_buf)(host_buf->data(), (size_t)n * stride);
            }
            return;
        }
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
        dcopy(Kokkos::subview(d_idx, Kokkos::make_pair(0, n)),
                          Kokkos::subview(h_idx, Kokkos::make_pair(0, n)));
    }
    void upload_buf(size_t n) {
        dcopy(Kokkos::subview(d_buf, Kokkos::make_pair((size_t)0, n)),
                          Kokkos::subview(h_buf, Kokkos::make_pair((size_t)0, n)));
    }
    void download_buf(size_t n) {
        dcopy(Kokkos::subview(h_buf, Kokkos::make_pair((size_t)0, n)),
                          Kokkos::subview(d_buf, Kokkos::make_pair((size_t)0, n)));
    }

    // One instance per call site (tag), released at Kokkos::finalize
    template <int Tag>
    static DeviceStaging& get() {
        static DeviceStaging s, sh;
        static bool hook = (Kokkos::push_finalize_hook([] { s = DeviceStaging(); sh = DeviceStaging(); }), true);
        (void)hook;
        if (host_exec()) { sh.host = true; return sh; }
        return s;
    }
};
