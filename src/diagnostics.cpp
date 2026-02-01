#include "diagnostics.h"
#include "constants.h"
#include <cmath>
#include <cstdio>
#include <vector>

DiagnosticState compute_diagnostics(const Particles& p, double G, double soft2) {
    DiagnosticState d{};
    const int N = p.N;

    for (int i = 0; i < N; i++) {
        double v2 = p.vx[i]*p.vx[i] + p.vy[i]*p.vy[i] + p.vz[i]*p.vz[i];
        d.KE += 0.5 * p.mass[i] * v2;
        d.Px += p.mass[i] * p.vx[i];
        d.Py += p.mass[i] * p.vy[i];
        d.Pz += p.mass[i] * p.vz[i];
        d.Lx += p.mass[i] * (p.y[i]*p.vz[i] - p.z[i]*p.vy[i]);
        d.Ly += p.mass[i] * (p.z[i]*p.vx[i] - p.x[i]*p.vz[i]);
        d.Lz += p.mass[i] * (p.x[i]*p.vy[i] - p.y[i]*p.vx[i]);
    }

    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            double dx = p.x[j] - p.x[i];
            double dy = p.y[j] - p.y[i];
            double dz = p.z[j] - p.z[i];
            double r = std::sqrt(dx*dx + dy*dy + dz*dz + soft2);
            d.PE -= G * p.mass[i] * p.mass[j] / r;
        }
    }

    d.E_total = d.KE + d.PE;
    return d;
}

static double vec_mag(double x, double y, double z) {
    return std::sqrt(x*x + y*y + z*z);
}

void compute_errors(const DiagnosticState& cur, const DiagnosticState& ini,
                           const Particles& p, bool is_cluster,
                           double& dE, double& dP, double& dL) {
    dE = (ini.E_total != 0.0)
        ? std::fabs((cur.E_total - ini.E_total) / ini.E_total) : 0.0;

    if (is_cluster) {
        double M_total = 0;
        for (int i = 0; i < p.N; i++) M_total += p.mass[i];
        double dpx = cur.Px - ini.Px, dpy = cur.Py - ini.Py, dpz = cur.Pz - ini.Pz;
        dP = vec_mag(dpx, dpy, dpz) / M_total;
    } else {
        double P0_mag = vec_mag(ini.Px, ini.Py, ini.Pz);
        double P_mag  = vec_mag(cur.Px, cur.Py, cur.Pz);
        dP = (P0_mag > 0.0) ? std::fabs((P_mag - P0_mag) / P0_mag) : 0.0;
    }

    double L0_mag = vec_mag(ini.Lx, ini.Ly, ini.Lz);
    double L_mag  = vec_mag(cur.Lx, cur.Ly, cur.Lz);
    dL = (L0_mag > 0.0) ? std::fabs((L_mag - L0_mag) / L0_mag) : 0.0;
}

void print_timebin_table(const Particles& p, const std::vector<double>& levels,
                         double dE, double dP, double dL,
                         double t_yr, int step, bool is_cluster) {
    int nl = (int)levels.size();

    // Count particles per level
    std::vector<int> counts(nl, 0);
    for (int i = 0; i < p.N; i++) {
        if (p.level[i] >= 0 && p.level[i] < nl)
            counts[p.level[i]]++;
    }

    const char* p_label = is_cluster ? "|dP|/M [cm/s]" : "|dP/P|";

    std::printf("  %-6s %14s %6s   %12s %14s %12s %14s %8s\n",
                "Level", "dt [yr]", "count", "|dE/E|", p_label, "|dL/L|", "Time [yr]", "Step");
    std::printf("----------------------------------------------------------------------------------------------------\n");

    for (int l = 0; l < nl; l++) {
        double dt_yr = levels[l] / TIME_UNIT;
        std::printf("  L%-5d %14.6e %6d", l, dt_yr, counts[l]);
        if (l == 0 && dE >= 0.0) {
            std::printf("   %12.6e %14.6e %12.6e %14.2f %8d", dE, dP, dL, t_yr, step);
        }
        std::printf("\n");
    }
    std::printf("\n");
}

void print_final(const DiagnosticState& cur, const DiagnosticState& ini,
                 const Particles& p, const std::vector<double>& /*levels*/,
                 bool is_cluster, double wall_sec, int steps,
                 const char* const* names) {
    double dE, dP, dL;
    compute_errors(cur, ini, p, is_cluster, dE, dP, dL);

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("RESULTS\n");
    std::printf("============================================================\n");

    std::printf("|dE/E0| = %.15e\n", dE);
    if (is_cluster) {
        std::printf("|dP|/M  = %.15e cm/s\n", dP);
    } else {
        std::printf("|dP/P0| = %.15e\n", dP);
    }
    std::printf("|dL/L0| = %.15e\n", dL);

    std::printf("\nWall: %.4fs, Steps: %d", wall_sec, steps);
    if (steps > 0) std::printf(", %.4fms/step", wall_sec / steps * 1000.0);
    std::printf("\n");

    // Positions (printed in original particle order)
    if (p.N <= 10 && names != nullptr) {
        std::printf("\nPositions (AU):\n");
        // Build reverse map: orig_index[i] -> current position i
        std::vector<int> cur_pos(p.N);
        for (int i = 0; i < p.N; i++) cur_pos[p.orig_index[i]] = i;
        for (int orig = 0; orig < p.N; orig++) {
            int i = cur_pos[orig];
            double px = p.x[i] / LENGTH_UNIT;
            double py = p.y[i] / LENGTH_UNIT;
            double pz = p.z[i] / LENGTH_UNIT;
            double r = std::sqrt(px*px + py*py + pz*pz);
            std::printf("  %-10s: (%9.4f,%9.4f,%9.4f) r=%.4f\n",
                        names[orig], px, py, pz, r);
        }
    } else if (p.N > 10) {
        std::printf("\nPosition statistics (AU):\n");
        double r_min = 1e30, r_max = 0, r_sum = 0;
        for (int i = 0; i < p.N; i++) {
            double px = p.x[i] / LENGTH_UNIT;
            double py = p.y[i] / LENGTH_UNIT;
            double pz = p.z[i] / LENGTH_UNIT;
            double r = std::sqrt(px*px + py*py + pz*pz);
            if (r < r_min) r_min = r;
            if (r > r_max) r_max = r;
            r_sum += r;
        }
        std::printf("  r_min=%.4f, r_max=%.4f, r_mean=%.4f\n",
                    r_min, r_max, r_sum / p.N);
    }

    std::printf("\nDone!\n");
}
