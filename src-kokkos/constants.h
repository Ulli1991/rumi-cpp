#pragma once

#include <cmath>
#include <Kokkos_Core.hpp>

// Code units: pc, M_sun, km/s  (Gadget-style: time unit = 1 pc / 1 km/s)
// Length: 1 pc, Mass: 1 M_sun, Velocity: 1 km/s, Time: 1 pc/(1 km/s) ≈ 0.978 Myr
constexpr double G_CODE = 4.302194e-3;               // G in pc³ M_sun⁻¹ TU⁻²
constexpr double TIME_UNIT_MYR = 0.977792;           // 1 code TU in Myr (for config/display)
constexpr double TIME_UNIT_YR  = 977792.0;           // 1 code TU in years

// Conversion factors
constexpr double AU_TO_PC  = 4.84814e-6;             // 1 AU in pc (for outer planets init)
constexpr double PC_TO_CM  = 3.08567758e18;          // 1 parsec in cm (for SSE interface)
constexpr double MSUN_IN_G = 1.989e33;               // 1 M_sun in grams (for SSE interface)
constexpr double G_CGS     = 6.67430e-8;             // G in cm³ g⁻¹ s⁻² (for SSE interface)

// Numerical stability
constexpr double EPS = 1e-30;

// FSI coefficients
constexpr double FSI_KICK = 1.0 / 6.0;
constexpr double FSI_DRIFT = 0.5;
constexpr double FSI_MOD_KICK = 2.0 / 3.0;

// Gradient correction coefficient: G * dt² / 24 (from Eq. 29 of Rantala+2020: factor 2 in g_i × 1/48 = 1/24)
constexpr double GRAD_COEFF = 1.0 / 24.0;

// Portable reciprocal square root: uses hardware rsqrt when available.
// With -DMIXED_PRECISION: FP32 seed from hardware SFU + one Newton-Raphson
// iteration in FP64. ~1e-14 relative precision, saves FP64 SFU ops per pair.
// Requires pc unit system (r2 ~ O(1-100), well within float range).
// Without -DMIXED_PRECISION: full FP64 hardware rsqrt.
KOKKOS_INLINE_FUNCTION double fast_rsqrt(double x) {
#if defined(__CUDA_ARCH__)
#ifdef MIXED_PRECISION
    float xf = (float)x;
    double y = (double)__frsqrt_rn(xf);
    return y * (1.5 - 0.5 * x * y * y);
#else
    return rsqrt(x);
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
#ifdef MIXED_PRECISION
    float xf = (float)x;
    double y = (double)__frsqrt_rn(xf);
    return y * (1.5 - 0.5 * x * y * y);
#else
    return rsqrt(x);
#endif
#else
    return 1.0 / std::sqrt(x);
#endif
}

// Portable 256-bit vectorized load: reads 4 contiguous doubles in one transaction.
// Requires 32-byte aligned source (guaranteed by LayoutRight packed views).
KOKKOS_INLINE_FUNCTION void load4(const double* __restrict__ ptr,
                                   double& a, double& b, double& c, double& d) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const double4 v = *reinterpret_cast<const double4*>(ptr);
    a = v.x; b = v.y; c = v.z; d = v.w;
#else
    a = ptr[0]; b = ptr[1]; c = ptr[2]; d = ptr[3];
#endif
}

// Raw pointer wrapper: matches Kokkos View operator() interface but uses
// __restrict__ for compiler alias analysis. Trivially copyable for lambda capture.
struct RawView {
    double* __restrict__ ptr;
    KOKKOS_INLINE_FUNCTION double& operator()(int i) const { return ptr[i]; }
};

// Tile size for shared memory tiling in all N^2 GPU kernels
constexpr int TILE_SIZE = 128;

// Plummer sphere
constexpr double PLUMMER_SCALE_FACTOR = 1.305;
