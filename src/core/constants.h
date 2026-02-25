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

// Portable reciprocal square root: hardware rsqrt on GPU, std::sqrt on CPU.
KOKKOS_INLINE_FUNCTION double fast_rsqrt(double x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return rsqrt(x);
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
constexpr int TILE_SIZE = 256;

// Harmonic symmetrization factor for timestep (FROST paper Eq. 23)
// f(d) = (d + sqrt(1 + d²) - 1) / d  for |d| > eps, else 1.0
// Uses fast_rsqrt instead of sqrt: sqrt(x) = x * rsqrt(x)
KOKKOS_INLINE_FUNCTION double harmonic_sym_factor(double d) {
    double d2 = d * d;
    if (d2 < 1e-8) return 1.0;
    double s = 1.0 + d2;
    double sqrt_s = s * fast_rsqrt(s);
    return (d + sqrt_s - 1.0) / d;
}

// Speed of light in code units (km/s)
constexpr double C_CODE = 299792.458;

// Plummer sphere
constexpr double PLUMMER_SCALE_FACTOR = 1.305;
