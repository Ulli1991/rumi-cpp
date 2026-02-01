#include "forces.h"
#include "constants.h"
#include <cmath>
#include <cfloat>
#include <cstring>

bool grad_n3l = false;  // default: full N² gradient (better momentum conservation)

// ==========================================================================
// Kahan summation helper macros
// ==========================================================================
#define KAHAN_ADD(sum, comp, val) do { \
    double _y = (val) - (comp);        \
    double _t = (sum) + _y;            \
    (comp) = (_t - (sum)) - _y;        \
    (sum) = _t;                        \
} while(0)

// ==========================================================================
// Full N-body acceleration with N3L + Kahan summation: O(N(N-1)/2)
// ==========================================================================
void compute_acceleration(Particles& p, double G, double soft2) {
    const int N = p.N;
    double *__restrict__ px = p.x, *__restrict__ py = p.y, *__restrict__ pz = p.z;
    double *__restrict__ ax = p.ax, *__restrict__ ay = p.ay, *__restrict__ az = p.az;
    const double *__restrict__ m = p.mass;

    for (int i = 0; i < N; i++) { ax[i] = 0; ay[i] = 0; az[i] = 0; }

    for (int i = 0; i < N; i++) {
        double axi = 0, ayi = 0, azi = 0;
        #pragma omp simd
        for (int j = i + 1; j < N; j++) {
            double dx = px[j] - px[i];
            double dy = py[j] - py[i];
            double dz = pz[j] - pz[i];
            double r2 = dx*dx + dy*dy + dz*dz + soft2;
            double inv_r = 1.0 / std::sqrt(r2);
            double inv_r3 = inv_r * inv_r * inv_r;

            double fj = G * m[j] * inv_r3;
            double fi = G * m[i] * inv_r3;

            axi += fj * dx; ayi += fj * dy; azi += fj * dz;
            ax[j] -= fi * dx; ay[j] -= fi * dy; az[j] -= fi * dz;
        }
        ax[i] += axi; ay[i] += ayi; az[i] += azi;
    }
}

// ==========================================================================
// Full N² acceleration + FROST timestep (fused single pass)
// ==========================================================================
void compute_acceleration_and_frost(Particles& p, double G, double soft2, double eta) {
    const int N = p.N;
    double *__restrict__ px = p.x,  *__restrict__ py = p.y,  *__restrict__ pz = p.z;
    double *__restrict__ pvx = p.vx, *__restrict__ pvy = p.vy, *__restrict__ pvz = p.vz;
    double *__restrict__ ax = p.ax, *__restrict__ ay = p.ay, *__restrict__ az = p.az;
    const double *__restrict__ m = p.mass;
    double *__restrict__ dt_f = p.dt_frost;

    for (int i = 0; i < N; i++) { ax[i] = 0; ay[i] = 0; az[i] = 0; dt_f[i] = DBL_MAX; }

    for (int i = 0; i < N; i++) {
        double axi = 0, ayi = 0, azi = 0;
        double t_min_i = dt_f[i];

        for (int j = i + 1; j < N; j++) {
            double dx = px[j] - px[i];
            double dy = py[j] - py[i];
            double dz = pz[j] - pz[i];
            double r2 = dx*dx + dy*dy + dz*dz;
            double r2s = r2 + soft2;
            double inv_r = 1.0 / std::sqrt(r2s);
            double inv_r3 = inv_r * inv_r * inv_r;

            // Acceleration (N3L)
            double fj = G * m[j] * inv_r3;
            double fi = G * m[i] * inv_r3;
            axi += fj * dx; ayi += fj * dy; azi += fj * dz;
            ax[j] -= fi * dx; ay[j] -= fi * dy; az[j] -= fi * dz;

            // FROST (symmetric pair — applies to both i and j)
            double dvx = pvx[j] - pvx[i];
            double dvy = pvy[j] - pvy[i];
            double dvz = pvz[j] - pvz[i];
            double v_sq = dvx*dvx + dvy*dvy + dvz*dvz;
            double r_norm = std::sqrt(r2 + EPS);
            double v_norm = std::sqrt(v_sq + EPS);
            double mass_sum = m[i] + m[j];

            double t_ff = std::sqrt(r_norm * r_norm * r_norm / (G * mass_sum + EPS));
            double t_fb = r_norm / (v_norm + EPS);

            double r_dot_v = dx*dvx + dy*dvy + dz*dvz;
            double r2_safe = r2 + EPS;

            double dt_ff_dt = 1.5 * (r_dot_v / r2_safe) * t_ff;
            double factor = 1.0 + G * mass_sum / (v_sq * r_norm + EPS);
            double dt_fb_dt = (r_dot_v / r2_safe) * factor * t_fb;

            auto symmetrize = [](double t, double ddt) -> double {
                double ddt_safe = (std::fabs(ddt) > EPS) ? ddt : EPS;
                return t * (ddt + std::sqrt(1.0 + ddt * ddt) - 1.0) / ddt_safe;
            };

            double t_ff_sym = symmetrize(t_ff, dt_ff_dt);
            double t_fb_sym = symmetrize(t_fb, dt_fb_dt);
            double t_min_pair = std::fmin(t_ff_sym, t_fb_sym);

            t_min_i = std::fmin(t_min_i, t_min_pair);
            dt_f[j] = std::fmin(dt_f[j], t_min_pair);
        }
        ax[i] += axi; ay[i] += ayi; az[i] += azi;
        dt_f[i] = t_min_i;
    }

    for (int i = 0; i < N; i++) dt_f[i] *= eta;
}

// ==========================================================================
// Self-interaction: [start, start+count) with N3L + Kahan
// ==========================================================================
void self_acceleration(Particles& p, int start, int count,
                       double G, double soft2) {
    double *__restrict__ px = p.x, *__restrict__ py = p.y, *__restrict__ pz = p.z;
    double *__restrict__ ax = p.ax, *__restrict__ ay = p.ay, *__restrict__ az = p.az;
    const double *__restrict__ m = p.mass;
    const int end = start + count;

    for (int i = start; i < end; i++) { ax[i] = 0; ay[i] = 0; az[i] = 0; }

    for (int i = start; i < end; i++) {
        double axi = 0, ayi = 0, azi = 0;
        #pragma omp simd
        for (int j = i + 1; j < end; j++) {
            double dx = px[j] - px[i];
            double dy = py[j] - py[i];
            double dz = pz[j] - pz[i];
            double r2 = dx*dx + dy*dy + dz*dz + soft2;
            double inv_r = 1.0 / std::sqrt(r2);
            double inv_r3 = inv_r * inv_r * inv_r;

            double fj = G * m[j] * inv_r3;
            double fi = G * m[i] * inv_r3;

            axi += fj * dx; ayi += fj * dy; azi += fj * dz;
            ax[j] -= fi * dx; ay[j] -= fi * dy; az[j] -= fi * dz;
        }
        ax[i] += axi; ay[i] += ayi; az[i] += azi;
    }
}

// ==========================================================================
// Self gradient correction. Uses scratch buffers (no heap alloc).
// Two modes: N3L (halves work) or full N² (better momentum conservation).
// ==========================================================================
void self_gradient_correction(Particles& p, int start, int count,
                              double G, double soft2, double dt) {
    double *__restrict__ px = p.x, *__restrict__ py = p.y, *__restrict__ pz = p.z;
    double *__restrict__ ax = p.ax, *__restrict__ ay = p.ay, *__restrict__ az = p.az;
    const double *__restrict__ m = p.mass;
    const double coeff = G * dt * dt * GRAD_COEFF;

    // Snapshot accelerations into scratch buffers
    double *ax0 = p.scratch1, *ay0 = p.scratch2, *az0 = p.scratch3;
    for (int i = 0; i < count; i++) {
        ax0[i] = ax[start + i];
        ay0[i] = ay[start + i];
        az0[i] = az[start + i];
    }

    if (grad_n3l) {
        // N3L: upper triangle only, antisymmetric
        double *cx_acc = p.scratch4, *cy_acc = p.scratch5, *cz_acc = p.scratch6;
        for (int i = 0; i < count; i++) { cx_acc[i] = 0; cy_acc[i] = 0; cz_acc[i] = 0; }

        for (int ii = 0; ii < count; ii++) {
            int i = start + ii;
            double axi0 = ax0[ii], ayi0 = ay0[ii], azi0 = az0[ii];
            for (int jj = ii + 1; jj < count; jj++) {
                int j = start + jj;
                double dx = px[j] - px[i];
                double dy = py[j] - py[i];
                double dz = pz[j] - pz[i];
                double dax = ax0[jj] - axi0;
                double day = ay0[jj] - ayi0;
                double daz = az0[jj] - azi0;

                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r2 = 1.0 / r2;
                double inv_r = std::sqrt(inv_r2);
                double inv_r5 = inv_r2 * inv_r2 * inv_r;
                double r_dot_a = dx*dax + dy*day + dz*daz;

                double fj = coeff * m[j] * inv_r5;
                double term_x = r2 * dax - 3.0 * r_dot_a * dx;
                double term_y = r2 * day - 3.0 * r_dot_a * dy;
                double term_z = r2 * daz - 3.0 * r_dot_a * dz;
                cx_acc[ii] += fj * term_x;
                cy_acc[ii] += fj * term_y;
                cz_acc[ii] += fj * term_z;

                double fi = coeff * m[i] * inv_r5;
                cx_acc[jj] -= fi * term_x;
                cy_acc[jj] -= fi * term_y;
                cz_acc[jj] -= fi * term_z;
            }
        }
        for (int i = 0; i < count; i++) {
            ax[start + i] += cx_acc[i];
            ay[start + i] += cy_acc[i];
            az[start + i] += cz_acc[i];
        }
    } else {
        // Full N²: each particle independently sums over all others
        for (int ii = 0; ii < count; ii++) {
            int i = start + ii;
            double cx = 0, cy = 0, cz = 0;
            double axi0 = ax0[ii], ayi0 = ay0[ii], azi0 = az0[ii];
            for (int jj = 0; jj < count; jj++) {
                if (jj == ii) continue;
                int j = start + jj;
                double dx = px[j] - px[i];
                double dy = py[j] - py[i];
                double dz = pz[j] - pz[i];
                double dax = ax0[jj] - axi0;
                double day = ay0[jj] - ayi0;
                double daz = az0[jj] - azi0;

                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r2 = 1.0 / r2;
                double inv_r = std::sqrt(inv_r2);
                double inv_r5 = inv_r2 * inv_r2 * inv_r;
                double r_dot_a = dx*dax + dy*day + dz*daz;
                double f = coeff * m[j] * inv_r5;
                cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
            }
            ax[i] += cx; ay[i] += cy; az[i] += cz;
        }
    }
}

// ==========================================================================
// Fused self_acceleration + self_gradient_correction (A2)
// Two sub-passes within one function, reusing r2 computation context.
// ==========================================================================
void self_acceleration_and_gradient(Particles& p, int start, int count,
                                    double G, double soft2, double dt) {
    double *__restrict__ px = p.x, *__restrict__ py = p.y, *__restrict__ pz = p.z;
    double *__restrict__ ax = p.ax, *__restrict__ ay = p.ay, *__restrict__ az = p.az;
    const double *__restrict__ m = p.mass;
    const int end = start + count;
    const double coeff = G * dt * dt * GRAD_COEFF;

    // --- Pass 1: Compute accelerations (N3L, upper triangle) ---
    for (int i = start; i < end; i++) { ax[i] = 0; ay[i] = 0; az[i] = 0; }

    for (int i = start; i < end; i++) {
        double axi = 0, ayi = 0, azi = 0;
        #pragma omp simd
        for (int j = i + 1; j < end; j++) {
            double dx = px[j] - px[i];
            double dy = py[j] - py[i];
            double dz = pz[j] - pz[i];
            double r2 = dx*dx + dy*dy + dz*dz + soft2;
            double inv_r = 1.0 / std::sqrt(r2);
            double inv_r3 = inv_r * inv_r * inv_r;

            double fj = G * m[j] * inv_r3;
            double fi = G * m[i] * inv_r3;

            axi += fj * dx; ayi += fj * dy; azi += fj * dz;
            ax[j] -= fi * dx; ay[j] -= fi * dy; az[j] -= fi * dz;
        }
        ax[i] += axi; ay[i] += ayi; az[i] += azi;
    }

    // --- Snapshot accelerations ---
    double *ax0 = p.scratch1, *ay0 = p.scratch2, *az0 = p.scratch3;
    for (int i = 0; i < count; i++) {
        ax0[i] = ax[start + i];
        ay0[i] = ay[start + i];
        az0[i] = az[start + i];
    }

    // --- Pass 2: Gradient correction ---
    if (grad_n3l) {
        double *cx_acc = p.scratch4, *cy_acc = p.scratch5, *cz_acc = p.scratch6;
        for (int i = 0; i < count; i++) { cx_acc[i] = 0; cy_acc[i] = 0; cz_acc[i] = 0; }

        for (int ii = 0; ii < count; ii++) {
            int i = start + ii;
            double axi0 = ax0[ii], ayi0 = ay0[ii], azi0 = az0[ii];
            for (int jj = ii + 1; jj < count; jj++) {
                int j = start + jj;
                double dx = px[j] - px[i];
                double dy = py[j] - py[i];
                double dz = pz[j] - pz[i];
                double dax = ax0[jj] - axi0;
                double day = ay0[jj] - ayi0;
                double daz = az0[jj] - azi0;

                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r2 = 1.0 / r2;
                double inv_r = std::sqrt(inv_r2);
                double inv_r5 = inv_r2 * inv_r2 * inv_r;
                double r_dot_a = dx*dax + dy*day + dz*daz;
                double term_x = r2 * dax - 3.0 * r_dot_a * dx;
                double term_y = r2 * day - 3.0 * r_dot_a * dy;
                double term_z = r2 * daz - 3.0 * r_dot_a * dz;

                double fj = coeff * m[j] * inv_r5;
                cx_acc[ii] += fj * term_x; cy_acc[ii] += fj * term_y; cz_acc[ii] += fj * term_z;
                double fi = coeff * m[i] * inv_r5;
                cx_acc[jj] -= fi * term_x; cy_acc[jj] -= fi * term_y; cz_acc[jj] -= fi * term_z;
            }
        }
        for (int i = 0; i < count; i++) {
            ax[start + i] += cx_acc[i]; ay[start + i] += cy_acc[i]; az[start + i] += cz_acc[i];
        }
    } else {
        for (int ii = 0; ii < count; ii++) {
            int i = start + ii;
            double cx = 0, cy = 0, cz = 0;
            double axi0 = ax0[ii], ayi0 = ay0[ii], azi0 = az0[ii];
            for (int jj = 0; jj < count; jj++) {
                if (jj == ii) continue;
                int j = start + jj;
                double dx = px[j] - px[i];
                double dy = py[j] - py[i];
                double dz = pz[j] - pz[i];
                double dax = ax0[jj] - axi0;
                double day = ay0[jj] - ayi0;
                double daz = az0[jj] - azi0;

                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r2 = 1.0 / r2;
                double inv_r = std::sqrt(inv_r2);
                double inv_r5 = inv_r2 * inv_r2 * inv_r;
                double r_dot_a = dx*dax + dy*day + dz*daz;
                double f = coeff * m[j] * inv_r5;
                cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
            }
            ax[i] += cx; ay[i] += cy; az[i] += cz;
        }
    }
}

// ==========================================================================
// Cross-interaction: group A × group B, both directions in one pass + Kahan
// ==========================================================================
void cross_acceleration(Particles& p,
                        int a_start, int a_count,
                        int b_start, int b_count,
                        double G, double soft2) {
    double *__restrict__ px = p.x, *__restrict__ py = p.y, *__restrict__ pz = p.z;
    double *__restrict__ ax = p.ax, *__restrict__ ay = p.ay, *__restrict__ az = p.az;
    const double *__restrict__ m = p.mass;
    const int a_end = a_start + a_count;
    const int b_end = b_start + b_count;

    for (int i = a_start; i < a_end; i++) { ax[i] = 0; ay[i] = 0; az[i] = 0; }
    for (int i = b_start; i < b_end; i++) { ax[i] = 0; ay[i] = 0; az[i] = 0; }

    for (int i = a_start; i < a_end; i++) {
        double axi = 0, ayi = 0, azi = 0;
        #pragma omp simd
        for (int j = b_start; j < b_end; j++) {
            double dx = px[j] - px[i];
            double dy = py[j] - py[i];
            double dz = pz[j] - pz[i];
            double r2 = dx*dx + dy*dy + dz*dz + soft2;
            double inv_r = 1.0 / std::sqrt(r2);
            double inv_r3 = inv_r * inv_r * inv_r;

            double fj = G * m[j] * inv_r3;
            double fi = G * m[i] * inv_r3;

            axi += fj * dx; ayi += fj * dy; azi += fj * dz;
            ax[j] -= fi * dx; ay[j] -= fi * dy; az[j] -= fi * dz;
        }
        ax[i] += axi; ay[i] += ayi; az[i] += azi;
    }
}

// ==========================================================================
// Cross gradient correction with N3L (A4): single pass for both directions.
// Uses scratch buffers instead of heap allocation.
// ==========================================================================
void cross_gradient_correction(Particles& p,
                               int a_start, int a_count,
                               int b_start, int b_count,
                               double G, double soft2, double dt) {
    double *__restrict__ px = p.x, *__restrict__ py = p.y, *__restrict__ pz = p.z;
    double *__restrict__ ax = p.ax, *__restrict__ ay = p.ay, *__restrict__ az = p.az;
    const double *__restrict__ m = p.mass;
    const double coeff = G * dt * dt * GRAD_COEFF;

    // Snapshot accelerations into scratch buffers
    double *ax0_a = p.scratch1, *ay0_a = p.scratch2, *az0_a = p.scratch3;
    double *ax0_b = p.scratch4, *ay0_b = p.scratch5, *az0_b = p.scratch6;
    for (int i = 0; i < a_count; i++) {
        ax0_a[i] = ax[a_start + i];
        ay0_a[i] = ay[a_start + i];
        az0_a[i] = az[a_start + i];
    }
    for (int i = 0; i < b_count; i++) {
        ax0_b[i] = ax[b_start + i];
        ay0_b[i] = ay[b_start + i];
        az0_b[i] = az[b_start + i];
    }

    if (grad_n3l) {
        // N3L: single pass, antisymmetric
        for (int ii = 0; ii < a_count; ii++) {
            int i = a_start + ii;
            double cxi = 0, cyi = 0, czi = 0;
            double axi0 = ax0_a[ii], ayi0 = ay0_a[ii], azi0 = az0_a[ii];
            for (int jj = 0; jj < b_count; jj++) {
                int j = b_start + jj;
                double dx = px[j] - px[i];
                double dy = py[j] - py[i];
                double dz = pz[j] - pz[i];
                double dax = ax0_b[jj] - axi0;
                double day = ay0_b[jj] - ayi0;
                double daz = az0_b[jj] - azi0;

                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r2 = 1.0 / r2;
                double inv_r = std::sqrt(inv_r2);
                double inv_r5 = inv_r2 * inv_r2 * inv_r;
                double r_dot_a = dx*dax + dy*day + dz*daz;
                double term_x = r2 * dax - 3.0 * r_dot_a * dx;
                double term_y = r2 * day - 3.0 * r_dot_a * dy;
                double term_z = r2 * daz - 3.0 * r_dot_a * dz;

                double fj = coeff * m[j] * inv_r5;
                cxi += fj * term_x; cyi += fj * term_y; czi += fj * term_z;
                double fi = coeff * m[i] * inv_r5;
                ax[j] -= fi * term_x; ay[j] -= fi * term_y; az[j] -= fi * term_z;
            }
            ax[i] += cxi; ay[i] += cyi; az[i] += czi;
        }
    } else {
        // Full N²: A from B
        for (int ii = 0; ii < a_count; ii++) {
            int i = a_start + ii;
            double cx = 0, cy = 0, cz = 0;
            double axi0 = ax0_a[ii], ayi0 = ay0_a[ii], azi0 = az0_a[ii];
            for (int jj = 0; jj < b_count; jj++) {
                int j = b_start + jj;
                double dx = px[j] - px[i];
                double dy = py[j] - py[i];
                double dz = pz[j] - pz[i];
                double dax = ax0_b[jj] - axi0;
                double day = ay0_b[jj] - ayi0;
                double daz = az0_b[jj] - azi0;
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r2 = 1.0 / r2;
                double inv_r = std::sqrt(inv_r2);
                double inv_r5 = inv_r2 * inv_r2 * inv_r;
                double r_dot_a = dx*dax + dy*day + dz*daz;
                double f = coeff * m[j] * inv_r5;
                cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
            }
            ax[i] += cx; ay[i] += cy; az[i] += cz;
        }
        // Full N²: B from A
        for (int jj = 0; jj < b_count; jj++) {
            int j = b_start + jj;
            double cx = 0, cy = 0, cz = 0;
            double axj0 = ax0_b[jj], ayj0 = ay0_b[jj], azj0 = az0_b[jj];
            for (int ii = 0; ii < a_count; ii++) {
                int i = a_start + ii;
                double dx = px[i] - px[j];
                double dy = py[i] - py[j];
                double dz = pz[i] - pz[j];
                double dax = ax0_a[ii] - axj0;
                double day = ay0_a[ii] - ayj0;
                double daz = az0_a[ii] - azj0;
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r2 = 1.0 / r2;
                double inv_r = std::sqrt(inv_r2);
                double inv_r5 = inv_r2 * inv_r2 * inv_r;
                double r_dot_a = dx*dax + dy*day + dz*daz;
                double f = coeff * m[i] * inv_r5;
                cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
            }
            ax[j] += cx; ay[j] += cy; az[j] += cz;
        }
    }
}

// ==========================================================================
// Fused cross_acceleration + cross_gradient_correction (A2+A4)
// ==========================================================================
void cross_acceleration_and_gradient(Particles& p,
                                     int a_start, int a_count,
                                     int b_start, int b_count,
                                     double G, double soft2, double dt) {
    double *__restrict__ px = p.x, *__restrict__ py = p.y, *__restrict__ pz = p.z;
    double *__restrict__ ax = p.ax, *__restrict__ ay = p.ay, *__restrict__ az = p.az;
    const double *__restrict__ m = p.mass;
    const int a_end = a_start + a_count;
    const int b_end = b_start + b_count;
    const double coeff = G * dt * dt * GRAD_COEFF;

    // --- Pass 1: Cross acceleration (N3L) ---
    for (int i = a_start; i < a_end; i++) { ax[i] = 0; ay[i] = 0; az[i] = 0; }
    for (int i = b_start; i < b_end; i++) { ax[i] = 0; ay[i] = 0; az[i] = 0; }

    for (int i = a_start; i < a_end; i++) {
        double axi = 0, ayi = 0, azi = 0;
        #pragma omp simd
        for (int j = b_start; j < b_end; j++) {
            double dx = px[j] - px[i];
            double dy = py[j] - py[i];
            double dz = pz[j] - pz[i];
            double r2 = dx*dx + dy*dy + dz*dz + soft2;
            double inv_r = 1.0 / std::sqrt(r2);
            double inv_r3 = inv_r * inv_r * inv_r;

            double fj = G * m[j] * inv_r3;
            double fi = G * m[i] * inv_r3;

            axi += fj * dx; ayi += fj * dy; azi += fj * dz;
            ax[j] -= fi * dx; ay[j] -= fi * dy; az[j] -= fi * dz;
        }
        ax[i] += axi; ay[i] += ayi; az[i] += azi;
    }

    // --- Snapshot accelerations ---
    double *ax0_a = p.scratch1, *ay0_a = p.scratch2, *az0_a = p.scratch3;
    double *ax0_b = p.scratch4, *ay0_b = p.scratch5, *az0_b = p.scratch6;
    for (int i = 0; i < a_count; i++) {
        ax0_a[i] = ax[a_start + i];
        ay0_a[i] = ay[a_start + i];
        az0_a[i] = az[a_start + i];
    }
    for (int i = 0; i < b_count; i++) {
        ax0_b[i] = ax[b_start + i];
        ay0_b[i] = ay[b_start + i];
        az0_b[i] = az[b_start + i];
    }

    // --- Pass 2: Gradient correction ---
    if (grad_n3l) {
        for (int ii = 0; ii < a_count; ii++) {
            int i = a_start + ii;
            double gxi = 0, gyi = 0, gzi = 0;
            double axi0 = ax0_a[ii], ayi0 = ay0_a[ii], azi0 = az0_a[ii];
            for (int jj = 0; jj < b_count; jj++) {
                int j = b_start + jj;
                double dx = px[j] - px[i];
                double dy = py[j] - py[i];
                double dz = pz[j] - pz[i];
                double dax = ax0_b[jj] - axi0;
                double day = ay0_b[jj] - ayi0;
                double daz = az0_b[jj] - azi0;

                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r2 = 1.0 / r2;
                double inv_r = std::sqrt(inv_r2);
                double inv_r5 = inv_r2 * inv_r2 * inv_r;
                double r_dot_a = dx*dax + dy*day + dz*daz;
                double term_x = r2 * dax - 3.0 * r_dot_a * dx;
                double term_y = r2 * day - 3.0 * r_dot_a * dy;
                double term_z = r2 * daz - 3.0 * r_dot_a * dz;

                double fj = coeff * m[j] * inv_r5;
                gxi += fj * term_x; gyi += fj * term_y; gzi += fj * term_z;
                double fi = coeff * m[i] * inv_r5;
                ax[j] -= fi * term_x; ay[j] -= fi * term_y; az[j] -= fi * term_z;
            }
            ax[i] += gxi; ay[i] += gyi; az[i] += gzi;
        }
    } else {
        // Full N²: A from B
        for (int ii = 0; ii < a_count; ii++) {
            int i = a_start + ii;
            double cx = 0, cy = 0, cz = 0;
            double axi0 = ax0_a[ii], ayi0 = ay0_a[ii], azi0 = az0_a[ii];
            for (int jj = 0; jj < b_count; jj++) {
                int j = b_start + jj;
                double dx = px[j] - px[i];
                double dy = py[j] - py[i];
                double dz = pz[j] - pz[i];
                double dax = ax0_b[jj] - axi0;
                double day = ay0_b[jj] - ayi0;
                double daz = az0_b[jj] - azi0;
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r2 = 1.0 / r2;
                double inv_r = std::sqrt(inv_r2);
                double inv_r5 = inv_r2 * inv_r2 * inv_r;
                double r_dot_a = dx*dax + dy*day + dz*daz;
                double f = coeff * m[j] * inv_r5;
                cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
            }
            ax[i] += cx; ay[i] += cy; az[i] += cz;
        }
        // Full N²: B from A
        for (int jj = 0; jj < b_count; jj++) {
            int j = b_start + jj;
            double cx = 0, cy = 0, cz = 0;
            double axj0 = ax0_b[jj], ayj0 = ay0_b[jj], azj0 = az0_b[jj];
            for (int ii = 0; ii < a_count; ii++) {
                int i = a_start + ii;
                double dx = px[i] - px[j];
                double dy = py[i] - py[j];
                double dz = pz[i] - pz[j];
                double dax = ax0_a[ii] - axj0;
                double day = ay0_a[ii] - ayj0;
                double daz = az0_a[ii] - azj0;
                double r2 = dx*dx + dy*dy + dz*dz + soft2;
                double inv_r2 = 1.0 / r2;
                double inv_r = std::sqrt(inv_r2);
                double inv_r5 = inv_r2 * inv_r2 * inv_r;
                double r_dot_a = dx*dax + dy*day + dz*daz;
                double f = coeff * m[i] * inv_r5;
                cx += f * (r2 * dax - 3.0 * r_dot_a * dx);
                cy += f * (r2 * day - 3.0 * r_dot_a * dy);
                cz += f * (r2 * daz - 3.0 * r_dot_a * dz);
            }
            ax[j] += cx; ay[j] += cy; az[j] += cz;
        }
    }
}
