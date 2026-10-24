// Two-body element rates under a perturbing relative acceleration f, vector forms valid for any orbit
// (Rantala et al. 2023, sec. 3.4.3): Edot = v . f, adot = 2 a^2 Edot / mu, e = (v x L) / mu - r_hat,
// de/dt = (f x L + v x (r x f)) / mu, edot = e_hat . de/dt (|de/dt| at e = 0). Templated for the Enzyme tests.
#pragma once
#include <cmath>

template <class T> struct PertRates2 { T E, a, Edot, adot, e, edot, ev[3], g[3], gn; };

template <class T> inline PertRates2<T> pert_rates_2body(const T* r, const T* v, const T* f, T mu) {
    using std::sqrt;
    PertRates2<T> o;
    const T rx = r[0], ry = r[1], rz = r[2], vx = v[0], vy = v[1], vz = v[2], fx = f[0], fy = f[1], fz = f[2];
    const T r_inv = T(1) / sqrt(rx * rx + ry * ry + rz * rz);
    o.E = T(0.5) * (vx * vx + vy * vy + vz * vz) - mu * r_inv;
    o.a = -mu / (T(2) * o.E);
    const T Lx = ry * vz - rz * vy, Ly = rz * vx - rx * vz, Lz = rx * vy - ry * vx;
    o.Edot = vx * fx + vy * fy + vz * fz;
    o.adot = T(2) * o.a * o.a * o.Edot / mu;
    o.ev[0] = (vy * Lz - vz * Ly) / mu - rx * r_inv;
    o.ev[1] = (vz * Lx - vx * Lz) / mu - ry * r_inv;
    o.ev[2] = (vx * Ly - vy * Lx) / mu - rz * r_inv;
    const T rfx = ry * fz - rz * fy, rfy = rz * fx - rx * fz, rfz = rx * fy - ry * fx;
    o.g[0] = ((fy * Lz - fz * Ly) + (vy * rfz - vz * rfy)) / mu;
    o.g[1] = ((fz * Lx - fx * Lz) + (vz * rfx - vx * rfz)) / mu;
    o.g[2] = ((fx * Ly - fy * Lx) + (vx * rfy - vy * rfx)) / mu;
    o.e = sqrt(o.ev[0] * o.ev[0] + o.ev[1] * o.ev[1] + o.ev[2] * o.ev[2]);
    o.gn = sqrt(o.g[0] * o.g[0] + o.g[1] * o.g[1] + o.g[2] * o.g[2]);
    o.edot = o.e > T(0) ? (o.ev[0] * o.g[0] + o.ev[1] * o.g[1] + o.ev[2] * o.g[2]) / o.e : o.gn;
    return o;
}
