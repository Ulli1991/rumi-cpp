#pragma once
// NBODY_LEAN_SYNC (default on): host<->device copies without the global fences of Kokkos::deep_copy.
//  - dcopy(dst, src): the copy is enqueued on the default execution-space instance (the stream every kernel uses),
//    so it is ordered after the kernels that produced src; only a copy into host memory waits, and only for that
//    instance (stream sync instead of a device-wide fence before and after).
//  - host_copy(view): host copy of a device view into malloc'ed memory (a Kokkos HostSpace view fences the
//    device when it is freed).
//  - lean_fence(name): a fence that only profiling or the old copy semantics needed; skipped in lean mode.
// NBODY_LEAN_SYNC=0 (reference / debug mode): every fence of the original code is kept (host_copy adds one
// fence to stand in for the deallocation fence of the mirror view it replaces).
// Removing these fences changes no arithmetic: runs are bit-identical to the reference mode.
#include <Kokkos_Core.hpp>
#include "host_exec.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <source_location>
#include <string>
#include <type_traits>
#include <vector>

// On by default (bit-identical, 9-10 % per step at 200k on 4 A100, 2026-10-12); NBODY_LEAN_SYNC=0 restores every fence
inline bool lean_sync() {
    static const bool b = [] { const char* v = std::getenv("NBODY_LEAN_SYNC"); return !(v && v[0] == '0'); }();
    return b;
}

inline void lean_fence(const char* name) {
    if (!lean_sync()) Kokkos::fence(name);
}

// NBODY_FENCE_STATS: the copy fences carry the caller's file:line (fence_stats.h counts fences by name)
inline bool lean_site_labels() {
    static const bool b = [] { const char* v = std::getenv("NBODY_FENCE_STATS"); return v && v[0] && v[0] != '0'; }();
    return b;
}
inline void lean_copy_fence(const Kokkos::DefaultExecutionSpace& ex, const char* what, const std::source_location& loc) {
    if (!lean_site_labels()) { ex.fence(what); return; }
    const char* f = loc.file_name();
    for (const char* c = f; *c; c++) if (*c == '/') f = c + 1;
    ex.fence(std::string(what) + " @" + f + ":" + std::to_string(loc.line()));
}

// Host subtree (host_exec.h): every buffer is in host memory; a contiguous copy is a memcpy
template <class D, class S>
inline void host_exec_copy(const D& dst, const S& src, const std::source_location& loc) {
    if (dst.span() != src.span() || !dst.span_is_contiguous() || !src.span_is_contiguous()) {
        std::fprintf(stderr, "host_exec_copy: non-contiguous or mismatched views at %s:%u\n", loc.file_name(), loc.line());
        std::abort();
    }
    host_exec_check(dst.data(), loc);
    host_exec_check(src.data(), loc);
    if (dst.span() > 0 && (const void*)dst.data() != (const void*)src.data())
        std::memcpy((void*)dst.data(), (const void*)src.data(), dst.span() * sizeof(typename D::value_type));
}

template <class D, class S>
inline std::enable_if_t<Kokkos::is_view<D>::value && Kokkos::is_view<S>::value> dcopy(
        const D& dst, const S& src, const std::source_location loc = std::source_location::current()) {
    if (host_exec()) { host_exec_copy(dst, src, loc); return; }
    if (!lean_sync()) { Kokkos::deep_copy(dst, src); return; }
    Kokkos::DefaultExecutionSpace ex;
    Kokkos::deep_copy(ex, dst, src);
    if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename D::memory_space>::accessible)
        lean_copy_fence(ex, "lean_sync: copy to host", loc);
}

template <class T, class S>
inline std::enable_if_t<!Kokkos::is_view<T>::value && Kokkos::is_view<S>::value> dcopy(
        T& dst, const S& src, const std::source_location loc = std::source_location::current()) {
    if (host_exec()) {
        host_exec_check(src.data(), loc);
        std::memcpy(&dst, src.data(), sizeof(T));
        return;
    }
    if (!lean_sync()) { Kokkos::deep_copy(dst, src); return; }
    Kokkos::DefaultExecutionSpace ex;
    Kokkos::deep_copy(ex, dst, src);
    lean_copy_fence(ex, "lean_sync: scalar to host", loc);
}

template <class D, class T>
inline std::enable_if_t<Kokkos::is_view<D>::value && !Kokkos::is_view<T>::value> dcopy(const D& dst, const T& val) {
    if (host_exec()) {
        static_assert(D::rank == 1);
        host_exec_check(dst.data(), std::source_location::current());
        for (size_t i = 0; i < dst.extent(0); i++) dst.data()[i] = val;
        return;
    }
    if (!lean_sync() || Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename D::memory_space>::accessible) {
        Kokkos::deep_copy(dst, val);
        return;
    }
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), dst, typename D::const_value_type(val));
}

// Host copy of a rank-1 device view, in malloc'ed memory owned by the returned object
template <class T>
struct HostCopy {
    std::shared_ptr<std::vector<T>> buf;
    Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> v;
    T& operator()(size_t i) const { return v(i); }
    T* data() const { return v.data(); }
    size_t extent(int) const { return v.extent(0); }
    size_t size() const { return v.extent(0); }
};
template <class V>
inline HostCopy<typename V::non_const_value_type> host_copy(const V& d,
                                                             const std::source_location loc = std::source_location::current()) {
    using T = typename V::non_const_value_type;
    HostCopy<T> h;
    h.buf = std::make_shared<std::vector<T>>(d.extent(0));
    h.v = Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(h.buf->data(), d.extent(0));
    dcopy(h.v, d, loc);
    if (!lean_sync() && !host_exec()) Kokkos::fence("host_copy: stands in for the mirror deallocation fence");
    return h;
}
