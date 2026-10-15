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
#include <cstdlib>
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

template <class D, class S>
inline std::enable_if_t<Kokkos::is_view<D>::value && Kokkos::is_view<S>::value> dcopy(
        const D& dst, const S& src, const std::source_location loc = std::source_location::current()) {
    if (!lean_sync()) { Kokkos::deep_copy(dst, src); return; }
    Kokkos::DefaultExecutionSpace ex;
    Kokkos::deep_copy(ex, dst, src);
    if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename D::memory_space>::accessible)
        lean_copy_fence(ex, "lean_sync: copy to host", loc);
}

template <class T, class S>
inline std::enable_if_t<!Kokkos::is_view<T>::value && Kokkos::is_view<S>::value> dcopy(
        T& dst, const S& src, const std::source_location loc = std::source_location::current()) {
    if (!lean_sync()) { Kokkos::deep_copy(dst, src); return; }
    Kokkos::DefaultExecutionSpace ex;
    Kokkos::deep_copy(ex, dst, src);
    lean_copy_fence(ex, "lean_sync: scalar to host", loc);
}

template <class D, class T>
inline std::enable_if_t<Kokkos::is_view<D>::value && !Kokkos::is_view<T>::value> dcopy(const D& dst, const T& val) {
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
    if (!lean_sync()) Kokkos::fence("host_copy: stands in for the mirror deallocation fence");
    return h;
}
