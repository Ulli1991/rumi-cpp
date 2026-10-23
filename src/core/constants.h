#pragma once

#include <cmath>
#include <Kokkos_Core.hpp>

// Code units: pc, M_sun, km/s  (Gadget-style: time unit = 1 pc / 1 km/s)
// Length: 1 pc, Mass: 1 M_sun, Velocity: 1 km/s, Time: 1 pc/(1 km/s) ≈ 0.978 Myr
constexpr double G_CODE = 4.300917270036279e-3;      // G in pc³ M_sun⁻¹ TU⁻²: IAU 2015 nominal GM_sun 1.3271244e20 m³ s⁻²
constexpr double TIME_UNIT_MYR = 0.977792;           // 1 code TU in Myr (for config/display)
constexpr double TIME_UNIT_YR  = 977792.0;           // 1 code TU in years

// Conversion factors
constexpr double AU_TO_PC  = 4.84814e-6;             // 1 AU in pc (for outer planets init)
constexpr double PC_TO_CM  = 3.08567758e18;          // 1 parsec in cm (for SSE interface)
constexpr double MSUN_IN_G = 1.9884099e33;           // 1 M_sun in grams: GM_sun / G_CGS (snapshot units)
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
#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && defined(RSQRT_NEWTON)
    double y = rsqrt(x);                 // ~1 ulp; one Newton step brings it to ~0.5 ulp at 3 FMAs
    return y * (1.5 - 0.5 * x * y * y);
#elif (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && !defined(ACCURATE_RSQRT)
    return rsqrt(x);  // ~1 ulp on the device; ACCURATE_RSQRT uses the correctly rounded 1/sqrt
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

// Tile size for shared memory tiling in all N^2 kernels; it is also their team size. Host backends
// cap teams (Serial at 1, OpenMP at the thread count), so they run the same kernels untiled.
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
constexpr int TILE_SIZE = 128;
#else
constexpr int TILE_SIZE = 1;
#endif

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
                           double G, double mass_sum, double eta_ff, double eta_fb,
                           double& t_sym_ff, double& t_sym_fb) {
    // t_ff = sqrt(r^3 / Gm) = r^2 / sqrt(r Gm), t_fb = r / |v|: reciprocal square roots, no divisions
    double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
    double r = r2 * inv_r;
    double t_ff = r2 * fast_rsqrt(r * (G * mass_sum + EPS));
    double inv_v = fast_rsqrt(v_sq + EPS);
    double t_fb = r * inv_v;
    double inv_vsq = inv_v * inv_v;
    double t_ff_sq = t_ff * t_ff;
    double t_fb_sq = t_fb * t_fb;

    double inv_r2 = inv_r * inv_r;
    double r_dot_v = dx*dvx + dy*dvy + dz*dvz;

    double c_ff = 1.5 * r_dot_v * inv_r2;
    double c_fb = r_dot_v * inv_r2 * (1.0 + G * mass_sum * inv_r * inv_vsq);

    // Eq. 49 acts on the step eta*t, whose derivative is eta*dt/dt (FROST sec. 4.2); callers multiply
    // the result by eta, so the correction term carries one factor eta and d = c*eta*t.
    double ct_ff = c_ff * t_ff_sq * eta_ff;
    double s_ff = 1.0 + c_ff * ct_ff * eta_ff;
    double u_ff = s_ff * fast_rsqrt(s_ff);
    t_sym_ff = t_ff + ct_ff / (u_ff + 1.0);

    double ct_fb = c_fb * t_fb_sq * eta_fb;
    double s_fb = 1.0 + c_fb * ct_fb * eta_fb;
    double u_fb = s_fb * fast_rsqrt(s_fb);
    t_sym_fb = t_fb + ct_fb / (u_fb + 1.0);
}

// Gravitational softening. Plummer: phi = -G m / sqrt(r^2 + eps^2). Spline (SOFTENING_KERNEL = spline): the cubic
// spline kernel of Monaghan & Lattanzio (1985) as in Gadget-2, exactly Newtonian beyond h = 2.8 eps, so that a far
// field computed with plain 1/r expansions (FMM) is exact. For a pair at separation r (raw r2 = r^2):
//   acceleration G m g(r) r_vec,  Jacobian G m (g I + k r_vec r_vec^T) with k = g'(r)/r,  potential -G m w(r)
struct Softening {
    int spline = 0;       // 0: Plummer (eps^2 = soft2; 0 = unsoftened), 1: cubic spline with support h
    double soft2 = 0.0;   // Plummer eps^2 (0 for the spline: r enters raw)
    double h = 0.0;
    // g and k for the raw r2; Plummer and unsoftened from the softened r2s = r2 + soft2 and its inverse root
    KOKKOS_INLINE_FUNCTION void spline_gk(double r2, double& g, double& k) const {
        const double r = Kokkos::sqrt(r2), hi = 1.0 / h, u = r * hi, hi3 = hi * hi * hi;
        if (u >= 1.0) { const double ir = 1.0 / r, ir3 = ir * ir * ir; g = ir3; k = -3.0 * ir3 * ir * ir; return; }
        double dgdu;
        if (u < 0.5) { g = hi3 * (10.666666666667 + u * u * (32.0 * u - 38.4)); dgdu = hi3 * (96.0 * u * u - 76.8 * u); }
        else {
            const double iu3 = 1.0 / (u * u * u);
            g = hi3 * (21.333333333333 - 48.0 * u + 38.4 * u * u - 10.666666666667 * u * u * u - 0.066666666667 * iu3);
            dgdu = hi3 * (-48.0 + 76.8 * u - 32.0 * u * u + 0.2 * iu3 / u);
        }
        k = u > 0.0 ? dgdu * hi * hi / u : 0.0;
    }
    KOKKOS_INLINE_FUNCTION double spline_w(double r2) const {
        const double r = Kokkos::sqrt(r2), hi = 1.0 / h, u = r * hi;
        if (u >= 1.0) return 1.0 / r;
        if (u < 0.5) return -hi * (-2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6)));
        return -hi * (-3.2 + 0.066666666667 / u + u * u * (10.666666666667 + u * (-16.0 + u * (9.6 - 2.133333333333 * u))));
    }
};

// Kahan (compensated) summation: reduces accumulation error from O(N*eps) to O(eps).
KOKKOS_INLINE_FUNCTION
void kahan_add(double& sum, double& comp, double val) {
    double y = val - comp;
    double t = sum + y;
    comp = (t - sum) - y;
    sum = t;
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

// Gravity interaction with Kahan summation for improved momentum conservation.
KOKKOS_INLINE_FUNCTION
double gravity_interaction_kahan(double dx, double dy, double dz,
                                  double r2_soft, double G_mj,
                                  double& ax, double& ay, double& az,
                                  double& cx, double& cy, double& cz) {
    double inv_r = fast_rsqrt(r2_soft);
    double f = G_mj * inv_r * inv_r * inv_r;
    kahan_add(ax, cx, f * dx);
    kahan_add(ay, cy, f * dy);
    kahan_add(az, cz, f * dz);
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

// Gradient correction with Kahan summation.
KOKKOS_INLINE_FUNCTION
void gradient_correction_interaction_kahan(double dx, double dy, double dz,
                                            double dax, double day, double daz,
                                            double r2_soft, double coeff_mj,
                                            double& cx, double& cy, double& cz,
                                            double& kcx, double& kcy, double& kcz) {
    double inv_r = fast_rsqrt(r2_soft);
    double inv_r2 = inv_r * inv_r;
    double inv_r5 = inv_r2 * inv_r2 * inv_r;
    double r_dot_a = dx * dax + dy * day + dz * daz;
    double f = coeff_mj * inv_r5;
    kahan_add(cx, kcx, f * (r2_soft * dax - 3.0 * r_dot_a * dx));
    kahan_add(cy, kcy, f * (r2_soft * day - 3.0 * r_dot_a * dy));
    kahan_add(cz, kcz, f * (r2_soft * daz - 3.0 * r_dot_a * dz));
}

// MAC test: improved Barnes-Hut using group bounding box.
// Returns true if cell is distant enough to use monopole approximation.
KOKKOS_INLINE_FUNCTION
bool mac_accept(double gcx, double gcy, double gcz,
                double gsx, double gsy, double gsz,
                double ccx, double ccy, double ccz,
                double size_sq_th) {
    double drx = Kokkos::fmax(Kokkos::fabs(gcx - ccx) - gsx, 0.0);
    double dry = Kokkos::fmax(Kokkos::fabs(gcy - ccy) - gsy, 0.0);
    double drz = Kokkos::fmax(Kokkos::fabs(gcz - ccz) - gsz, 0.0);
    double dist_sq = drx * drx + dry * dry + drz * drz;
    return (size_sq_th < dist_sq) && (dist_sq > 0.0);
}

// Speed of light in code units (km/s)
constexpr double C_CODE = 299792.458;

// Plummer sphere
constexpr double PLUMMER_SCALE_FACTOR = 1.305;
