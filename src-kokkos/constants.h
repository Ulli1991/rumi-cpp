#pragma once

#include <cmath>
#include <Kokkos_Core.hpp>

// CGS physical constants
constexpr double G_CGS = 6.67e-8;           // cm³ g⁻¹ s⁻²

// Unit conversions
constexpr double LENGTH_UNIT = 1.495978707e13;      // 1 AU in cm
constexpr double VELOCITY_UNIT = LENGTH_UNIT / 86400.0;  // 1 AU/day in cm/s
constexpr double MASS_UNIT = 1.989e33;               // 1 M_sun in g
constexpr double TIME_UNIT = 3.154e7;                // 1 year in seconds
constexpr double PC_TO_CM = 3.08567758e18;           // 1 parsec in cm

// Gravitational constant in pc³ M_sun⁻¹ Myr⁻²
constexpr double G_PC_MSUN_MYR = 4.4983e-3;

// Numerical stability
constexpr double EPS = 1e-30;

// FSI coefficients
constexpr double FSI_KICK = 1.0 / 6.0;
constexpr double FSI_DRIFT = 0.5;
constexpr double FSI_MOD_KICK = 2.0 / 3.0;

// Gradient correction coefficient: G * dt² / 24 (from Eq. 29 of Rantala+2020: factor 2 in g_i × 1/48 = 1/24)
constexpr double GRAD_COEFF = 1.0 / 24.0;

// Portable reciprocal square root: uses hardware rsqrt when available
KOKKOS_INLINE_FUNCTION double fast_rsqrt(double x) {
#if defined(__CUDA_ARCH__)
    return rsqrt(x);
#elif defined(__HIP_DEVICE_COMPILE__)
    return rsqrt(x);
#else
    return 1.0 / std::sqrt(x);
#endif
}

// Raw pointer wrapper: matches Kokkos View operator() interface but uses
// __restrict__ for compiler alias analysis. Trivially copyable for lambda capture.
struct RawView {
    double* __restrict__ ptr;
    KOKKOS_INLINE_FUNCTION double& operator()(int i) const { return ptr[i]; }
};

// Plummer sphere
constexpr double PLUMMER_SCALE_FACTOR = 1.305;
