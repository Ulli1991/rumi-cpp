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
    static_assert(alignof(double4) <= 32, "double4 alignment exceeds expected 32 bytes");
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

// Symmetrized pairwise timestep (FROST Eq. 23)
// Computes t_ff and t_fb symmetrized by harmonic predictor.
// r2 may be softened (r2+soft2) or raw. inv_r = 1/sqrt(r2).
KOKKOS_INLINE_FUNCTION
void symmetrized_timestep(double r2, double inv_r,
                           double dx, double dy, double dz,
                           double dvx, double dvy, double dvz,
                           double G, double mass_sum,
                           double& t_sym_ff, double& t_sym_fb) {
    double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
    double inv_Gm  = 1.0 / (G * mass_sum + EPS);
    double inv_vsq = 1.0 / (v_sq + EPS);

    double r3 = r2 * r2 * inv_r;
    double t_ff_sq = r3 * inv_Gm;
    double t_fb_sq = r2 * inv_vsq;
    double t_ff = t_ff_sq * fast_rsqrt(t_ff_sq + EPS);
    double t_fb = t_fb_sq * fast_rsqrt(t_fb_sq + EPS);

    double inv_r2 = inv_r * inv_r;
    double r_dot_v = dx*dvx + dy*dvy + dz*dvz;

    double c_ff = 1.5 * r_dot_v * inv_r2;
    double c_fb = r_dot_v * inv_r2 * (1.0 + G * mass_sum * inv_r * inv_vsq);

    double ct_ff = c_ff * t_ff_sq;
    double s_ff = 1.0 + c_ff * ct_ff;
    double u_ff = s_ff * fast_rsqrt(s_ff);
    t_sym_ff = t_ff + ct_ff / (u_ff + 1.0);

    double ct_fb = c_fb * t_fb_sq;
    double s_fb = 1.0 + c_fb * ct_fb;
    double u_fb = s_fb * fast_rsqrt(s_fb);
    t_sym_fb = t_fb + ct_fb / (u_fb + 1.0);
}

// Gravity interaction: accumulate force from one source particle.
// Returns inv_r (reusable for timestep computation).
KOKKOS_INLINE_FUNCTION
double gravity_interaction(double dx, double dy, double dz,
                           double r2_soft, double G_mj,
                           double& ax, double& ay, double& az) {
    double inv_r = fast_rsqrt(r2_soft);
    double f = G_mj * inv_r * inv_r * inv_r;
    ax += f * dx;
    ay += f * dy;
    az += f * dz;
    return inv_r;
}

// Gradient correction interaction: accumulate jerk-based correction from one source.
KOKKOS_INLINE_FUNCTION
void gradient_correction_interaction(double dx, double dy, double dz,
                                      double dax, double day, double daz,
                                      double r2_soft, double coeff_mj,
                                      double& cx, double& cy, double& cz) {
    double inv_r = fast_rsqrt(r2_soft);
    double inv_r2 = inv_r * inv_r;
    double inv_r5 = inv_r2 * inv_r2 * inv_r;
    double r_dot_a = dx * dax + dy * day + dz * daz;
    double f = coeff_mj * inv_r5;
    cx += f * (r2_soft * dax - 3.0 * r_dot_a * dx);
    cy += f * (r2_soft * day - 3.0 * r_dot_a * dy);
    cz += f * (r2_soft * daz - 3.0 * r_dot_a * dz);
}

// Speed of light in code units (km/s)
constexpr double C_CODE = 299792.458;

// Plummer sphere
constexpr double PLUMMER_SCALE_FACTOR = 1.305;
