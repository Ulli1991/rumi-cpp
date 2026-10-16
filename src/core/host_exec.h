#pragma once

// Host-resident HHS subtrees (HOST_SUBTREE_MAX): a small set runs its whole sub-recursion on the host, on a copy of
// its particles in host memory. The recursion's code is the device code: its kernels take raw pointers (n2_kernels.h,
// RawView) and its loops go through xfor / xreduce / xscan / small_for / run_n2, which run as plain host loops while
// host_exec() is on. Views of the default (device) memory space may then point to host memory (host_view); they are
// only ever dereferenced through raw pointers, and copies between them (dcopy, host_copy) become memcpy.
// Off (the default, and always outside a host subtree) every call is the plain Kokkos launch.

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <source_location>
#include <string>
#include <memory>
#include <type_traits>
#include <vector>

namespace host_exec_detail {
inline bool& flag() {
    static bool b = false;
    return b;
}
}  // namespace host_exec_detail

inline bool host_exec() { return host_exec_detail::flag(); }

struct HostExecScope {
    bool saved;
    explicit HostExecScope(bool on = true) : saved(host_exec_detail::flag()) { host_exec_detail::flag() = on; }
    ~HostExecScope() { host_exec_detail::flag() = saved; }
    HostExecScope(const HostExecScope&) = delete;
    HostExecScope& operator=(const HostExecScope&) = delete;
};

// NBODY_CHECK_HOSTSUB=1: abort on a device pointer met in a host subtree (a buffer without a host twin)
inline void host_exec_check(const void* ptr, const std::source_location& loc) {
    static const bool on = [] { const char* v = std::getenv("NBODY_CHECK_HOSTSUB"); return v && v[0] && v[0] != '0'; }();
    if (!on || !ptr) return;
#if defined(KOKKOS_ENABLE_CUDA)
    cudaPointerAttributes a;
    if (cudaPointerGetAttributes(&a, ptr) == cudaSuccess && a.type == cudaMemoryTypeDevice) {
        std::fprintf(stderr, "NBODY_CHECK_HOSTSUB: device pointer %p in a host subtree at %s:%u\n", ptr, loc.file_name(),
                     loc.line());
        std::abort();
    }
    cudaGetLastError();
#else
    (void)loc;
#endif
}

// Raw-pointer access with the View call syntax (rank 1, and rank 2 LayoutRight with a static second extent)
template <class T>
struct Raw1 {
    T* __restrict__ ptr;
    KOKKOS_INLINE_FUNCTION T& operator()(int i) const { return ptr[i]; }
};
template <class T, int M>
struct Raw2 {
    T* __restrict__ ptr;
    KOKKOS_INLINE_FUNCTION T& operator()(int i, int k) const { return ptr[(size_t)i * M + k]; }
};
template <class V>
inline auto raw(const V& v, const std::source_location loc = std::source_location::current()) {
    using T = typename V::value_type;
    if (host_exec()) host_exec_check(v.data(), loc);
    if constexpr (V::rank == 1) {
        return Raw1<T>{v.data()};
    } else {
        static_assert(V::rank == 2 && std::is_same_v<typename V::array_layout, Kokkos::LayoutRight>);
        return Raw2<T, (int)V::static_extent(1)>{v.data()};
    }
}

template <class F>
inline void xfor(const char* name, int n, const F& f) {
    if (host_exec()) {
        for (int i = 0; i < n; i++) f(i);
        return;
    }
    Kokkos::parallel_for(name, n, f);
}
template <class F>
inline void xfor(const char* name, int lo, int hi, const F& f) {
    if (host_exec()) {
        for (int i = lo; i < hi; i++) f(i);
        return;
    }
    Kokkos::parallel_for(name, Kokkos::RangePolicy<>(lo, hi), f);
}

// Sum (scalar result) or a Kokkos reducer (Kokkos::Min, Kokkos::Max, ...)
template <class F, class R>
inline void xreduce(const char* name, int n, const F& f, R&& result) {
    using RR = std::remove_cv_t<std::remove_reference_t<R>>;
    if (host_exec()) {
        if constexpr (Kokkos::is_reducer_v<RR>) {
            typename RR::value_type v;
            result.init(v);
            for (int i = 0; i < n; i++) f(i, v);
            result.reference() = v;
        } else {
            RR v{};
            for (int i = 0; i < n; i++) f(i, v);
            result = v;
        }
        return;
    }
    Kokkos::parallel_reduce(name, n, f, result);
}
template <class F, class R>
inline void xreduce(const char* name, int lo, int hi, const F& f, R&& result) {
    using RR = std::remove_cv_t<std::remove_reference_t<R>>;
    if (host_exec()) {
        if constexpr (Kokkos::is_reducer_v<RR>) {
            typename RR::value_type v;
            result.init(v);
            for (int i = lo; i < hi; i++) f(i, v);
            result.reference() = v;
        } else {
            RR v{};
            for (int i = lo; i < hi; i++) f(i, v);
            result = v;
        }
        return;
    }
    Kokkos::parallel_reduce(name, Kokkos::RangePolicy<>(lo, hi), f, result);
}

// Exclusive scan f(i, update, final) with the total
template <class T, class F>
inline void xscan(const char* name, int n, const F& f, T& total) {
    if (host_exec()) {
        T u{};
        for (int i = 0; i < n; i++) f(i, u, true);
        total = u;
        return;
    }
    Kokkos::parallel_scan(name, n, f, total);
}

// A view of the default memory space over host memory (kept alive by `owner`), for the host copy of a subtree
template <class V>
inline V host_view(std::vector<std::shared_ptr<void>>& owner, size_t n) {
    using T = typename V::value_type;
    constexpr size_t M = V::rank == 1 ? 1 : V::static_extent(1);
    auto buf = std::make_shared<std::vector<T>>(n * M + 1);
    owner.push_back(buf);
    return V(buf->data(), n);
}

// Device buffer with a host twin of the same type: get(n) returns the one for the current mode (grown to n)
template <class T>
struct DualBuf {
    Kokkos::View<T*> dev, host;
    std::shared_ptr<std::vector<T>> host_mem;
    const char* label;
    explicit DualBuf(const char* l) : label(l) {}
    Kokkos::View<T*>& get(size_t n) {
        if (host_exec()) {
            if (host.extent(0) < n) {
                host_mem = std::make_shared<std::vector<T>>(n);
                host = Kokkos::View<T*>(host_mem->data(), n);
            }
            return host;
        }
        if (dev.extent(0) < n) dev = Kokkos::View<T*>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)), n);
        return dev;
    }
    void release() { dev = host = Kokkos::View<T*>(); host_mem.reset(); }
};
