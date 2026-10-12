#include "tidal_field.h"
#include "diagnostics.h"
#include <Kokkos_Core.hpp>
#include <cmath>

TidalField g_tidal;

void TidalField::acc(const double x[3], double G, double a[3]) const {
    const double r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    double f = 0.0;
    if (type == POINT_MASS) f = r2 > 0.0 ? -G * M_gal / (r2 * std::sqrt(r2)) : 0.0;
    else if (type == LOG_HALO) f = -V_c * V_c / (R_c * R_c + r2);
    for (int k = 0; k < 3; k++) a[k] = f * x[k];
}

double TidalField::v_circ(double R, double G) const {
    if (type == POINT_MASS) return std::sqrt(G * M_gal / R);
    if (type == LOG_HALO) return V_c * R / std::sqrt(R_c * R_c + R * R);
    return 0.0;
}

double TidalField::jacobi_radius(const double Xg[3], double M, double G) const {
    const double R2 = Xg[0] * Xg[0] + Xg[1] * Xg[1] + Xg[2] * Xg[2];
    double k = 0.0;   // Omega^2 - d2Phi/dR2
    if (type == POINT_MASS) k = R2 > 0.0 ? 3.0 * G * M_gal / (R2 * std::sqrt(R2)) : 0.0;
    else if (type == LOG_HALO) k = 2.0 * V_c * V_c * R2 / ((R_c * R_c + R2) * (R_c * R_c + R2));
    return k > 0.0 && M > 0.0 ? std::cbrt(G * M / k) : 0.0;
}

void TidalField::init_circular(double R, double G) {
    X[0] = R; X[1] = 0.0; X[2] = 0.0;
    V[0] = 0.0; V[1] = v_circ(R, G); V[2] = 0.0;
}

void tidal_kick(Particles& p, double hdt, double G, EnergyBookkeeping& books) {
    if (!g_tidal.on() || p.N <= 0) return;
    const TidalField tf = g_tidal;
    double aref[3];
    tf.acc(tf.X, G, aref);
    auto x = p.x, y = p.y, z = p.z, vx = p.vx, vy = p.vy, vz = p.vz, m = p.mass;
    const double X0 = tf.X[0], X1 = tf.X[1], X2 = tf.X[2], ar0 = aref[0], ar1 = aref[1], ar2 = aref[2];
    const int type = (int)tf.type;
    const double GM = G * tf.M_gal, vc2 = tf.V_c * tf.V_c, rc2 = tf.R_c * tf.R_c;
    // dE: kinetic energy change; dP, dL: momentum and angular momentum change (about the reference point)
    double dE = 0.0, dP[3] = {0, 0, 0}, dL[3] = {0, 0, 0};
    Kokkos::parallel_reduce("tidal_kick", p.N, KOKKOS_LAMBDA(int i, double& e, double& p0, double& p1, double& p2,
                                                             double& l0, double& l1, double& l2) {
        if (!(m(i) > 0.0)) return;
        const double gx = X0 + x(i), gy = X1 + y(i), gz = X2 + z(i);
        const double r2 = gx * gx + gy * gy + gz * gz;
        double f = 0.0;
        if (type == 1) f = r2 > 0.0 ? -GM / (r2 * Kokkos::sqrt(r2)) : 0.0;
        else if (type == 2) f = -vc2 / (rc2 + r2);
        const double dvx = hdt * (f * gx - ar0), dvy = hdt * (f * gy - ar1), dvz = hdt * (f * gz - ar2);
        const double ux = vx(i), uy = vy(i), uz = vz(i);
        e += m(i) * (ux * dvx + uy * dvy + uz * dvz + 0.5 * (dvx * dvx + dvy * dvy + dvz * dvz));
        p0 += m(i) * dvx; p1 += m(i) * dvy; p2 += m(i) * dvz;
        l0 += m(i) * (y(i) * dvz - z(i) * dvy);
        l1 += m(i) * (z(i) * dvx - x(i) * dvz);
        l2 += m(i) * (x(i) * dvy - y(i) * dvx);
        vx(i) = ux + dvx; vy(i) = uy + dvy; vz(i) = uz + dvz;
    }, dE, dP[0], dP[1], dP[2], dL[0], dL[1], dL[2]);
    Kokkos::fence();
    books.E_tidal -= dE;   // E_total + total_sink() stays constant: energy put in is a negative sink
    for (int k = 0; k < 3; k++) { books.P_sink[k] -= dP[k]; books.L_sink[k] -= dL[k]; }
    for (int k = 0; k < 3; k++) g_tidal.V[k] += hdt * aref[k];
}

void tidal_drift_reference(double dt) {
    if (!g_tidal.on()) return;
    for (int k = 0; k < 3; k++) g_tidal.X[k] += dt * g_tidal.V[k];
}
