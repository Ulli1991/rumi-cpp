// Ketju-integrator (MSTAR)
// Copyright (C) 2020-2022 Antti Rantala, Matias Mannerkoski, and contributors.
// Licensed under the GPLv3 license.
// See LICENSE.md or <https://www.gnu.org/licenses/> for details.
// Modified 2026 by the RUMI++ authors: Kokkos (device-resident) port of src/mstar/ketju/src/integration.c
// (leapfrog, GBS extrapolation, step control), mst.c (tree proximity, Prim tree) and pn.c (PN accelerations);
// see mstar_kokkos.h for the differences from the C library and src/mstar/README.md.

#include "mstar_kokkos.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "ketju_integrator/internal.h"
#include "ketju_integrator/ketju_integrator.h"
#include "mstar_dispatch.h"

namespace rumi_mstar {

namespace {

constexpr int MAX_ND = 8;     // max_tree_distance supported on the device
constexpr int MAX_K = 16;     // gbs_kmax supported on the device

inline double wall() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct ErrVal {
    double e0, e1;
    int bad;
    KOKKOS_INLINE_FUNCTION ErrVal() : e0(0), e1(0), bad(0) {}
    KOKKOS_INLINE_FUNCTION ErrVal& operator+=(const ErrVal& o) {
        e0 = e0 > o.e0 ? e0 : o.e0;
        e1 = e1 > o.e1 ? e1 : o.e1;
        bad |= o.bad;
        return *this;
    }
};
struct Acc4 {
    double x, y, z, u;
    KOKKOS_INLINE_FUNCTION Acc4() : x(0), y(0), z(0), u(0) {}
    KOKKOS_INLINE_FUNCTION Acc4& operator+=(const Acc4& o) {
        x += o.x; y += o.y; z += o.z; u += o.u;
        return *this;
    }
};

}  // namespace
}  // namespace rumi_mstar

namespace Kokkos {
template <> struct reduction_identity<rumi_mstar::ErrVal> {
    KOKKOS_FORCEINLINE_FUNCTION static rumi_mstar::ErrVal sum() { return rumi_mstar::ErrVal(); }
};
template <> struct reduction_identity<rumi_mstar::Acc4> {
    KOKKOS_FORCEINLINE_FUNCTION static rumi_mstar::Acc4 sum() { return rumi_mstar::Acc4(); }
};
}  // namespace Kokkos

namespace rumi_mstar {

using Team = Kokkos::TeamPolicy<ExecSpace>::member_type;
enum { S_TIME = 0, S_B = 1, S_T = 2, S_U = 3 };
enum { R_POS = 0, R_VEL = 3, R_AUX = 6 };

// All device data of one group. Slots: GBS rows 0..kmax-1, then two state slots (current = "base" = KETJU's state
// backup, and the candidate built from the extrapolated row), whose roles swap on an accepted step.
struct Dev {
    int N = 0, E = 0, Npn = 0, Nd = 2, root = 0, nvars = 0, kmax = 10;
    unsigned flags = 0;
    double G = 1, c = 1;
    DView<double*> mass;
    DView<int*> parent, etp, level, path_off, path_edge, ehi, elo;
    DView<double**> scal, root_s;                       // (slots, 4), (slots, 9)
    DView<double***> epos, evel, eaux;                  // (slots, E, 3)
    DView<double***> pos, vel, acc;                     // (slots, N, 3)
    DView<double**> u;                                  // (slots, N)
    DView<double***> spin, aspin, accvd, sder, pnv;     // (slots, max(Npn, 1), 3)
    DView<double*> crad;                                // collision radius per body
    DView<char*> cbh;                                   // black hole flag per body
    double cc = 0;                                      // c of the collision rule
    bool cmax1 = false;
    DView<double**> data;                               // (kmax + 1, nvars + 1); row kmax: extrapolated result;
                                                        // column nvars: B (PN work ledger, outside the error control)

    // ---- tree proximity (mst.c: ketju_check_relative_proximity*) ----
    KOKKOS_INLINE_FUNCTION bool prox_nd1(int v1, int v2, int& d, int* path, int* sign) const {
        const int diff = abs(level(v1) - level(v2));
        if (diff == 1) {
            if (parent(v2) == v1) { d = 1; sign[0] = -1; path[0] = etp(v2); return true; }
            if (parent(v1) == v2) { d = 1; sign[0] = +1; path[0] = etp(v1); return true; }
        }
        return false;
    }
    KOKKOS_INLINE_FUNCTION bool prox_nd2(int v1, int v2, int& d, int* path, int* sign) const {
        const int diff = abs(level(v1) - level(v2));
        if (diff == 1) {
            if (parent(v2) == v1) { d = 1; sign[0] = -1; path[0] = etp(v2); return true; }
            if (parent(v1) == v2) { d = 1; sign[0] = +1; path[0] = etp(v1); return true; }
        } else if (diff == 2) {
            if (parent(v2) >= 0 && parent(parent(v2)) == v1) {
                d = 2; sign[0] = -1; sign[1] = -1; path[0] = etp(v2); path[1] = etp(parent(v2)); return true;
            }
            if (parent(v1) >= 0 && parent(parent(v1)) == v2) {
                d = 2; sign[0] = +1; sign[1] = +1; path[0] = etp(v1); path[1] = etp(parent(v1)); return true;
            }
        } else {
            if (parent(v1) == parent(v2)) {
                d = 2; sign[0] = -1; sign[1] = +1; path[0] = etp(v2); path[1] = etp(v1); return true;
            }
        }
        return false;
    }
    KOKKOS_INLINE_FUNCTION bool prox_general(int v1, int v2, int& d, int* path, int* sign) const {
        for (int i = 0; i < Nd; i++) { path[i] = -1000; sign[i] = 0; }
        if (level(v1) == level(v2)) {
            const int halfNd = Nd / 2;
            int pathend[MAX_ND];
            for (int i = 0; i < Nd; i++) pathend[i] = 0;
            for (int i = 0; i < halfNd; i++) {
                sign[i] = +1;
                path[i] = etp(v1);
                pathend[i] = etp(v2);
                v1 = parent(v1);
                v2 = parent(v2);
                if (v1 == v2) {
                    d = 2 * (i + 1);
                    int cc = 0;
                    for (int m = i + 1; m < d; m++) { sign[m] = -1; path[m] = pathend[i - cc]; cc++; }
                    return true;
                }
            }
        } else {
            d = abs(level(v1) - level(v2));
            if (d <= Nd) {
                int lo, hi, s;
                if (level(v1) > level(v2)) { lo = v2; hi = v1; s = +1; } else { lo = v1; hi = v2; s = -1; }
                for (int i = 0; i < d; i++) { path[i] = etp(hi); sign[i] = s; hi = parent(hi); }
                if (lo == hi) return true;
            }
        }
        return false;
    }
    KOKKOS_INLINE_FUNCTION bool prox(int v1, int v2, int& d, int* path, int* sign) const {
        switch (Nd) {
        case 0: return false;
        case 1: return prox_nd1(v1, v2, d, path, sign);
        case 2: return prox_nd2(v1, v2, d, path, sign);
        }
        return prox_general(v1, v2, d, path, sign);
    }

    // integration.c separation_and_r2: dr = x_j - x_i (tree path when close in the tree)
    KOKKOS_INLINE_FUNCTION double separation(int s, int i, int j, double* dr) const {
        int path[MAX_ND], sign[MAX_ND], d = 0;
        dr[0] = dr[1] = dr[2] = 0;
        if (abs(level(i) - level(j)) <= Nd && prox(j, i, d, path, sign)) {
            for (int l = 0; l < d; l++) {
                const double sg = sign[l];
                for (int k = 0; k < 3; k++) dr[k] += sg * epos(s, path[l], k);
            }
        } else {
            for (int k = 0; k < 3; k++) dr[k] = pos(s, j, k) - pos(s, i, k);
        }
        return dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
    }

    // pn.c particle_separation(_and_relative_velocity): r = x_i - x_j; velocities of PN particles from pnv(s),
    // tree relative velocities from eaux (aux) or evel
    KOKKOS_INLINE_FUNCTION void pn_sep(int s, int i, int j, double* rv) const {
        int path[MAX_ND], sign[MAX_ND], d = 0;
        if (prox(i, j, d, path, sign)) {
            for (int k = 0; k < 3; k++) {
                rv[k] = 0;
                for (int l = 0; l < d; l++) rv[k] += sign[l] * epos(s, path[l], k);
            }
        } else {
            for (int k = 0; k < 3; k++) rv[k] = pos(s, i, k) - pos(s, j, k);
        }
    }
    KOKKOS_INLINE_FUNCTION void pn_sep_vel(int s, int i, int j, double* rv, double* vv, bool aux) const {
        int path[MAX_ND], sign[MAX_ND], d = 0;
        if (prox(i, j, d, path, sign)) {
            for (int k = 0; k < 3; k++) {
                rv[k] = 0;
                vv[k] = 0;
                for (int l = 0; l < d; l++) {
                    rv[k] += sign[l] * epos(s, path[l], k);
                    vv[k] += sign[l] * (aux ? eaux(s, path[l], k) : evel(s, path[l], k));
                }
            }
        } else {
            for (int k = 0; k < 3; k++) {
                rv[k] = pos(s, i, k) - pos(s, j, k);
                vv[k] = pnv(s, i, k) - pnv(s, j, k);
            }
        }
    }

    KOKKOS_INLINE_FUNCTION static double dot(const double* a, const double* b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }
    KOKKOS_INLINE_FUNCTION static void cross(const double* a, const double* b, double* r) {
        r[0] = a[1] * b[2] - b[1] * a[2];
        r[1] = a[2] * b[0] - b[2] * a[0];
        r[2] = a[0] * b[1] - b[0] * a[1];
    }

    // pn.c compute_PN_acc_single for PN particle i: velocities pnv(s), spins aspin (aux) or spin
    KOKKOS_FUNCTION void pn_single(int s, int i, bool aux) const;

    // vertex coordinates from root + root-to-vertex path sums (integration.c update_pos / update_vel order)
    KOKKOS_INLINE_FUNCTION void vertex_pos(int s, int v) const {
        double x[3] = {root_s(s, R_POS), root_s(s, R_POS + 1), root_s(s, R_POS + 2)};
        for (int p = path_off(v); p < path_off(v + 1); p++) {
            const int e = path_edge(p);
            for (int k = 0; k < 3; k++) x[k] = x[k] + epos(s, e, k);
        }
        for (int k = 0; k < 3; k++) pos(s, v, k) = x[k];
    }
    // which: R_VEL (physical, into vel or pnv) or R_AUX (into pnv)
    KOKKOS_INLINE_FUNCTION void vertex_vel(int s, int v, int which, double* out) const {
        double x[3] = {root_s(s, which), root_s(s, which + 1), root_s(s, which + 2)};
        for (int p = path_off(v); p < path_off(v + 1); p++) {
            const int e = path_edge(p);
            for (int k = 0; k < 3; k++) x[k] = x[k] + (which == R_AUX ? eaux(s, e, k) : evel(s, e, k));
        }
        for (int k = 0; k < 3; k++) out[k] = x[k];
    }

    // ---- team building blocks ----
    KOKKOS_INLINE_FUNCTION void team_copy(const Team& t, int from, int to) const {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 4), [&](int q) { scal(to, q) = scal(from, q); });
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 9), [&](int q) { root_s(to, q) = root_s(from, q); });
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, E), [&](int e) {
            for (int k = 0; k < 3; k++) { epos(to, e, k) = epos(from, e, k); evel(to, e, k) = evel(from, e, k); }
        });
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, N), [&](int i) {
            for (int k = 0; k < 3; k++) { pos(to, i, k) = pos(from, i, k); vel(to, i, k) = vel(from, i, k); }
        });
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, Npn), [&](int i) {
            for (int k = 0; k < 3; k++) spin(to, i, k) = spin(from, i, k);
        });
        t.team_barrier();
    }
    // set_auxiliary_variables
    KOKKOS_INLINE_FUNCTION void team_set_aux(const Team& t, int s) const {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, E), [&](int e) {
            for (int k = 0; k < 3; k++) eaux(s, e, k) = evel(s, e, k);
        });
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 3), [&](int k) { root_s(s, R_AUX + k) = root_s(s, R_VEL + k); });
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, Npn), [&](int i) {
            for (int k = 0; k < 3; k++) aspin(s, i, k) = spin(s, i, k);
        });
        t.team_barrier();
    }
    KOKKOS_INLINE_FUNCTION double team_T(const Team& t, int s) const {
        double T = 0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, N), [&](int i, double& sum) {
            double v2 = 0;
            for (int k = 0; k < 3; k++) v2 += vel(s, i, k) * vel(s, i, k);
            sum += 0.5 * mass(i) * v2;
        }, T);
        return T;
    }
    KOKKOS_INLINE_FUNCTION void team_drift(const Team& t, int s, double ds) const {
        const double T = team_T(t, s);
        const double dt = ds / (T + scal(s, S_B));
        t.team_barrier();
        Kokkos::single(Kokkos::PerTeam(t), [&]() {
            scal(s, S_T) = T;
            scal(s, S_TIME) += dt;
            for (int k = 0; k < 3; k++) root_s(s, R_POS + k) += dt * root_s(s, R_VEL + k);
        });
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, E), [&](int e) {
            for (int k = 0; k < 3; k++) epos(s, e, k) += dt * evel(s, e, k);
        });
        t.team_barrier();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, N), [&](int v) { vertex_pos(s, v); });
        t.team_barrier();
    }
    // forces of slot s inside one team (small N): acc and u_i = sum_{j>i} m_j / r_ij, U = G sum_i m_i u_i
    KOKKOS_INLINE_FUNCTION void team_force(const Team& t, int s) const {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, N), [&](int i) {
            double a[3] = {0, 0, 0}, ui = 0;
            for (int j = 0; j < N; j++) {
                if (j == i) continue;
                double dr[3];
                const double r2 = separation(s, i, j, dr);
                const double invr = 1. / sqrt(r2);
                const double Ginvr3 = G * invr * invr * invr;
                const double mj = mass(j);
                if (j > i) ui += mj * invr;
                for (int k = 0; k < 3; k++) a[k] += mj * (dr[k] * Ginvr3);
            }
            for (int k = 0; k < 3; k++) acc(s, i, k) = a[k];
            u(s, i) = ui;
        });
        t.team_barrier();
        team_U(t, s);
    }
    KOKKOS_INLINE_FUNCTION void team_U(const Team& t, int s) const {
        double U = 0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, N), [&](int i, double& sum) {
            sum += G * mass(i) * u(s, i);
        }, U);
        t.team_barrier();
        Kokkos::single(Kokkos::PerTeam(t), [&]() { scal(s, S_U) = U; });
        t.team_barrier();
    }
    KOKKOS_INLINE_FUNCTION void team_pn_acc(const Team& t, int s, bool aux) const {
        // velocities of the PN particles (update_vel(PHYSICAL / AUXILIARY) restricted to the particles used)
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, Npn), [&](int i) {
            double v[3];
            vertex_vel(s, i, aux ? R_AUX : R_VEL, v);
            for (int k = 0; k < 3; k++) pnv(s, i, k) = v[k];
        });
        t.team_barrier();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, Npn), [&](int i) { pn_single(s, i, aux); });
        t.team_barrier();
    }
    KOKKOS_INLINE_FUNCTION double eacc_vd(int s, int e, int k) const {
        const int hi = ehi(e), lo = elo(e);
        return (hi < Npn ? accvd(s, hi, k) : 0.0) - (lo < Npn ? accvd(s, lo, k) : 0.0);
    }
    // integration.c auxiliary_kick (into eaux) / physical_kick (into evel)
    KOKKOS_INLINE_FUNCTION void team_pn_subkick(const Team& t, int s, double dt, bool physical) const {
        team_pn_acc(t, s, physical);
        if (physical) {
            Kokkos::single(Kokkos::PerTeam(t), [&]() {
                double dB = 0;
                for (int i = 0; i < Npn; i++)
                    for (int k = 0; k < 3; k++) dB += mass(i) * pnv(s, i, k) * accvd(s, i, k);
                scal(s, S_B) -= dB * dt;
            });
        }
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, E), [&](int e) {
            const int hi = ehi(e), lo = elo(e);
            for (int k = 0; k < 3; k++) {
                const double a = (acc(s, hi, k) - acc(s, lo, k)) + eacc_vd(s, e, k);
                if (physical) evel(s, e, k) += dt * a; else eaux(s, e, k) += dt * a;
            }
        });
        Kokkos::single(Kokkos::PerTeam(t), [&]() {
            for (int k = 0; k < 3; k++) {
                const double a = acc(s, root, k) + (root < Npn ? accvd(s, root, k) : 0.0);
                if (physical) root_s(s, R_VEL + k) += dt * a; else root_s(s, R_AUX + k) += dt * a;
            }
        });
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, Npn), [&](int i) {
            for (int k = 0; k < 3; k++) {
                if (physical) spin(s, i, k) += dt * sder(s, i, k); else aspin(s, i, k) += dt * sder(s, i, k);
            }
        });
        t.team_barrier();
    }
    // integration.c kick (forces of slot s already computed)
    KOKKOS_INLINE_FUNCTION void team_kick(const Team& t, int s, double ds) const {
        const double dt = ds / scal(s, S_U);
        if (flags != 0 && Npn >= 2) {
            team_pn_subkick(t, s, dt / 2, false);
            team_pn_subkick(t, s, dt, true);
            team_pn_subkick(t, s, dt / 2, false);
        } else {
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, E), [&](int e) {
                const int hi = ehi(e), lo = elo(e);
                for (int k = 0; k < 3; k++) evel(s, e, k) += dt * (acc(s, hi, k) - acc(s, lo, k));
            });
            Kokkos::single(Kokkos::PerTeam(t), [&]() {
                for (int k = 0; k < 3; k++) root_s(s, R_VEL + k) += dt * acc(s, root, k);
            });
            t.team_barrier();
        }
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, N), [&](int v) {
            double x[3];
            vertex_vel(s, v, R_VEL, x);
            for (int k = 0; k < 3; k++) vel(s, v, k) = x[k];
        });
        t.team_barrier();
    }
    // store_gbs_data for 0-based row ri (data row ri)
    KOKKOS_INLINE_FUNCTION void team_pack(const Team& t, int s, int ri) const {
        Kokkos::single(Kokkos::PerTeam(t), [&]() {
            data(ri, 0) = scal(s, S_TIME);
            for (int q = 0; q < 6; q++) data(ri, 1 + q) = root_s(s, q);
        });
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, E), [&](int e) {
            for (int k = 0; k < 3; k++) {
                data(ri, 7 + 3 * e + k) = epos(s, e, k);
                data(ri, 7 + 3 * E + 3 * e + k) = evel(s, e, k);
            }
        });
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, Npn), [&](int i) {
            for (int k = 0; k < 3; k++) data(ri, 7 + 6 * E + 3 * i + k) = spin(s, i, k);
        });
        Kokkos::single(Kokkos::PerTeam(t), [&]() { data(ri, nvars) = scal(s, S_B); });
        t.team_barrier();
    }
    // update_system_from_gbs_result: data row `row` into slot s, then into_Cartesian_from_MST
    KOKKOS_INLINE_FUNCTION void team_unpack(const Team& t, int s, int row) const {
        Kokkos::single(Kokkos::PerTeam(t), [&]() {
            scal(s, S_TIME) = data(row, 0);
            for (int q = 0; q < 6; q++) root_s(s, q) = data(row, 1 + q);
        });
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, E), [&](int e) {
            for (int k = 0; k < 3; k++) {
                epos(s, e, k) = data(row, 7 + 3 * e + k);
                evel(s, e, k) = data(row, 7 + 3 * E + 3 * e + k);
            }
        });
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, Npn), [&](int i) {
            for (int k = 0; k < 3; k++) spin(s, i, k) = data(row, 7 + 6 * E + 3 * i + k);
        });
        t.team_barrier();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, N), [&](int v) {
            vertex_pos(s, v);
            double x[3];
            vertex_vel(s, v, R_VEL, x);
            for (int k = 0; k < 3; k++) vel(s, v, k) = x[k];
        });
        t.team_barrier();
    }

    // Collision test of pair i < j in slot s (RUMI++): separation inside the collision distance, or approaching
    // with osculating pericentre inside it and the pericentre reached within dt_step. Returns true with rsep (the
    // separation, or the osculating pericentre), rc and type.
    KOKKOS_INLINE_FUNCTION bool pair_collides(int s, int i, int j, double dt_step, double& rsep, double& rc,
                                              int& type) const {
        rc = mstar_collision_distance(cbh(i) != 0, mass(i), crad(i), cbh(j) != 0, mass(j), crad(j), G, cc, cmax1, type);
        if (!(rc > 0)) return false;
        double dx[3], dv[3], r2 = 0, v2 = 0, rv = 0;
        for (int k = 0; k < 3; k++) {
            dx[k] = pos(s, j, k) - pos(s, i, k);
            dv[k] = vel(s, j, k) - vel(s, i, k);
            r2 += dx[k] * dx[k]; v2 += dv[k] * dv[k]; rv += dx[k] * dv[k];
        }
        const double r = sqrt(r2);
        if (r < rc) { rsep = r; return true; }
        if (rv >= 0 || !(dt_step > 0)) return false;
        const double mu = G * (mass(i) + mass(j));
        double h[3];
        cross(dx, dv, h);
        const double h2 = dot(h, h);
        const double eps = 0.5 * v2 - mu / r;
        const double e = sqrt(fmax(0.0, 1.0 + 2.0 * eps * h2 / (mu * mu)));
        const double rp = h2 / (mu * (1.0 + e));
        if (!(rp < rc)) return false;
        // time to the pericentre on the osculating orbit
        double tp;
        if (fabs(e - 1.0) < 1e-8) {
            tp = sqrt(2.0) / (3.0 * sqrt(mu)) * (r + 2.0 * rp) * sqrt(fmax(r - rp, 0.0));
        } else if (eps < 0) {
            const double a = -mu / (2.0 * eps), n = sqrt(mu / (a * a * a));
            const double cE = fmin(1.0, fmax(-1.0, (1.0 - r / a) / e));
            const double EE = acos(cE);
            tp = (EE - e * sin(EE)) / n;
        } else {
            const double a = -mu / (2.0 * eps), n = sqrt(mu / (-a * a * a));
            const double chF = fmax(1.0, (1.0 - r / a) / e);
            const double F = acosh(chF);
            tp = (e * sinh(F) - F) / n;
        }
        if (tp > dt_step) return false;
        rsep = rp;
        return true;
    }
};

// pn.c compute_PN_acc_single (Thorne & Hartle 1985 1PN + spin; Blanchet 2014 2-3.5PN, binary terms)
KOKKOS_FUNCTION void Dev::pn_single(int s, int i, bool aux) const {
    double a_out[3] = {0, 0, 0}, sd_out[3] = {0, 0, 0};
    const int Np = Npn;
    const double G2 = G * G, G3 = G2 * G, G4 = G2 * G2;
    const double c2 = c * c, c4 = c2 * c2, c5 = c * c4, c6 = c4 * c2, c7 = c * c6;
    const double pi2 = M_PI * M_PI;
    auto S = [&](int p, int k) { return aux ? aspin(s, p, k) : spin(s, p, k); };
    auto V = [&](int p, int k) { return pnv(s, p, k); };
    const double mi = mass(i), mi2 = mi * mi;
    double spi[3] = {S(i, 0), S(i, 1), S(i, 2)};
    double veli[3] = {V(i, 0), V(i, 1), V(i, 2)};
    const double Si2 = dot(spi, spi);
    const double vi2 = dot(veli, veli);
    for (int j = 0; j < Np; ++j) {
        if (j == i) continue;
        double rv_ij[3], nv_ij[3], vv_ij[3];
        pn_sep_vel(s, i, j, rv_ij, vv_ij, aux);
        const double rij = sqrt(dot(rv_ij, rv_ij));
        for (int k = 0; k < 3; k++) nv_ij[k] = rv_ij[k] / rij;
        const double rij2 = rij * rij, rij3 = rij2 * rij, rij4 = rij3 * rij;
        const double mj = mass(j), mj2 = mj * mj, mj3 = mj2 * mj;
        double velj[3] = {V(j, 0), V(j, 1), V(j, 2)};
        double spj[3] = {S(j, 0), S(j, 1), S(j, 2)};
        const double vivj = dot(veli, velj);
        const double vj2 = dot(velj, velj);
        const double vj4 = vj2 * vj2;
        const double vij2 = dot(vv_ij, vv_ij);
        const double vjnij = dot(velj, nv_ij);
        const double vjnij2 = vjnij * vjnij;
        const double vjnij4 = vjnij2 * vjnij2;
        const double vinij = dot(veli, nv_ij);
        const double vinij2 = vinij * vinij;
        const double vijnij = dot(vv_ij, nv_ij);

        if (flags & PN_1_0) {
            double nij_factor = vi2 + 2. * vj2 - 4. * vivj - 1.5 * vjnij2;
            if (flags & PN_THREEBODY) {
                for (int k = 0; k < Np; ++k) {
                    const double mk = mass(k);
                    if (k != i) {
                        double rv_ik[3];
                        pn_sep(s, i, k, rv_ik);
                        const double rik = sqrt(dot(rv_ik, rv_ik));
                        nij_factor -= 4. * G * mk / rik;
                    }
                    if (k != j) {
                        double rv_jk[3], nv_jk[3];
                        pn_sep(s, j, k, rv_jk);
                        const double rjk = sqrt(dot(rv_jk, rv_jk));
                        for (int l = 0; l < 3; l++) nv_jk[l] = rv_jk[l] / rjk;
                        nij_factor -= G * mk / rjk * (1. - .5 * rij / rjk * dot(nv_ij, nv_jk));
                        for (int l = 0; l < 3; ++l) a_out[l] -= 3.5 * nv_jk[l] * G2 * mj * mk / (rij * rjk * rjk * c2);
                    }
                }
            } else {
                nij_factor -= 4. * G * mj / rij;
                nij_factor -= 5. * G * mi / rij;
            }
            nij_factor *= -G * mj / (rij2 * c2);
            const double vij_factor = G * mj / (rij2 * c2) * (4. * vijnij + vjnij);
            for (int l = 0; l < 3; ++l) a_out[l] += nij_factor * nv_ij[l] + vij_factor * vv_ij[l];
        }

        if (flags & PN_SPIN_ALL) {
            const double nijSj = dot(nv_ij, spj);
            if (flags & PN_1_5_SPIN) {
                double Sjxnij[3], Sixnij[3], Sjxvij[3], Sixvij[3];
                cross(spj, nv_ij, Sjxnij);
                cross(spi, nv_ij, Sixnij);
                cross(spj, vv_ij, Sjxvij);
                cross(spi, vv_ij, Sixvij);
                const double factor = G / (c2 * mi * rij3);
                for (int l = 0; l < 3; ++l) {
                    a_out[l] += factor * (6. * nv_ij[l] * (mi * dot(Sjxnij, vv_ij) + mj * dot(Sixnij, vv_ij))
                                          + 4. * mi * Sjxvij[l] + 3. * mj * Sixvij[l]
                                          - vijnij * (6. * mi * Sjxnij[l] + 3. * mj * Sixnij[l]));
                }
            }
            if (flags & PN_2_0_SPIN) {
                const double SiSj = dot(spi, spj);
                const double nijSi = dot(nv_ij, spi);
                const double Sj2 = dot(spj, spj);
                const double factor = G / (c2 * mi * rij4);
                for (int l = 0; l < 3; ++l) {
                    a_out[l] += factor * (nv_ij[l] * (-3. * SiSj + 15. * nijSi * nijSj) - 3. * spi[l] * nijSj
                                          - 3. * spj[l] * nijSi);
                    a_out[l] += factor * (mi / mj * ((-1.5 * Sj2 + 7.5 * nijSj * nijSj) * nv_ij[l] - 3. * nijSj * spj[l])
                                          + mj / mi * ((-1.5 * Si2 + 7.5 * nijSi * nijSi) * nv_ij[l]
                                                       - 3. * nijSi * spi[l]));
                }
            }
            if (flags & PN_SPIN_EVOL) {
                const double nijSi = dot(nv_ij, spi);
                double buffer[3];
                double factor = G / (c2 * rij3);
                for (int l = 0; l < 3; ++l) buffer[l] = factor * (-spj[l] + 3. * nv_ij[l] * nijSj);
                double v_diff[3], v_diffxnij[3];
                for (int l = 0; l < 3; ++l) v_diff[l] = 2. * velj[l] - 1.5 * veli[l];
                cross(v_diff, nv_ij, v_diffxnij);
                factor = G * mj / (c2 * rij2);
                for (int l = 0; l < 3; ++l) buffer[l] += factor * v_diffxnij[l];
                factor = 3 * G * mj / (rij3 * mi * c2) * nijSi;
                for (int l = 0; l < 3; ++l) buffer[l] += factor * nv_ij[l];
                double sder_l[3];
                cross(buffer, spi, sder_l);
                for (int l = 0; l < 3; ++l) sd_out[l] += sder_l[l];
            }
        }

        double nij_factor = 0., vij_factor = 0.;
        if (flags & PN_2_0) {
            nij_factor += (-57. * G3 * mi2 * mj / (4. * rij4) - 69. * G3 * mi * mj2 / (2. * rij4) - 9. * G3 * mj3 / rij4
                           + G * mj / rij2
                                 * (vjnij2 * (-15. / 8. * vjnij2 + 1.5 * vi2 - 6. * vivj + 9. / 2. * vj2)
                                    - 2. * vivj * vivj + 4. * vivj * vj2 - 2. * vj4)
                           + G2 * mi * mj / rij3
                                 * (39. / 2. * vinij2 - 39. * vinij * vjnij + 17. / 2. * vjnij2 - 15. / 4. * vi2
                                    - 5. / 2. * vivj + 5. / 4. * vj2)
                           + G2 * mj2 / rij3
                                 * (2. * vinij2 - 4. * vinij * vjnij - 6. * vjnij2 - 8. * vivj + 4. * vj2))
                          / c4;
            vij_factor += (G2 * mj2 / rij3 * (-2. * vinij - 2. * vjnij)
                           + G2 * mi * mj / rij3 * (-63. / 4. * vinij + 55. / 4. * vjnij)
                           + G * mj / rij2
                                 * (-6. * vinij * vjnij2 + 9. / 2. * vjnij2 * vjnij + vjnij * vi2 - 4. * vinij * vivj
                                    + 4. * vjnij * vivj + 4. * vinij * vj2 - 5. * vjnij * vj2))
                          / c4;
        }
        if (flags & PN_2_5) {
            nij_factor += (208. * G3 * mi * mj2 / (15. * rij4) - 24. * G3 * mi2 * mj / (5. * rij4)
                           + 12. * G2 * mi * mj * vij2 / (5. * rij3))
                          * vijnij / c5;
            vij_factor += (8. * G3 * mi2 * mj / (5. * rij4) - 32. * G3 * mi * mj2 / (5. * rij4)
                           - 4. * G2 * mi * mj * vij2 / (5. * rij3))
                          / c5;
        }
        if (flags & PN_3_0) {
            nij_factor +=
                (G * mj / rij2
                     * (35. / 16. * vjnij4 * vjnij2 - 15. / 8. * vjnij4 * vi2 + 15. / 2. * vjnij4 * vivj
                        + 3. * vjnij2 * vivj * vivj - 15. / 2. * vjnij4 * vj2 + 1.5 * vjnij2 * vi2 * vj2
                        - 12. * vjnij2 * vivj * vj2 - 2. * vivj * vivj * vj2 + 15. / 2. * vjnij2 * vj4
                        + 4. * vivj * vj4 - 2. * vj4 * vj2)
                 + G2 * mi * mj / rij3
                       * (-171. / 8. * vinij2 * vinij2 + 171. / 2. * vinij2 * vinij * vjnij
                          - 723. / 4. * vinij2 * vjnij2 + 383. / 2. * vinij * vjnij2 * vjnij - 455. / 8. * vjnij4
                          + 229. / 4. * vinij2 * vi2 - 205. / 2. * vinij * vjnij * vi2 + 191. / 4. * vjnij2 * vi2
                          - 91. / 8. * vi2 * vi2 - 229. / 2. * vinij2 * vivj + 244. * vinij * vjnij * vivj
                          - 225. / 2. * vjnij2 * vivj + 91. / 2. * vi2 * vivj - 177. / 4. * vivj * vivj
                          + 229. / 4. * vinij2 * vj2 - 283. / 2. * vinij * vjnij * vj2 + 259. / 4. * vjnij2 * vj2
                          - 91. / 4. * vi2 * vj2 + 43. * vivj * vj2 - 81. / 8. * vj4)
                 + G2 * mj2 / rij3
                       * (-6. * vinij2 * vjnij2 + 12. * vinij * vjnij2 * vjnij + 6. * vjnij4
                          + 4. * vinij * vjnij * vivj + 12. * vjnij2 * vivj + 4. * vivj * vivj
                          - 4. * vinij * vjnij * vj2 - 12. * vjnij2 * vj2 - 8. * vivj * vj2 + 4. * vj4)
                 + G3 * mj3 / rij4 * (-vinij2 + 2. * vinij * vjnij + 43. / 2. * vjnij2 + 18. * vivj - 9. * vj2)
                 + G3 * mi * mj2 / rij4
                       * (415. / 8. * vinij2 - 375. / 4. * vinij * vjnij + 1113. / 8. * vjnij2
                          - 615. / 64. * vijnij * vijnij * pi2 + 18. * vi2 + 123. / 64. * pi2 * vij2 + 33. * vivj
                          - 33. / 2. * vj2)
                 + G3 * mi2 * mj / rij4
                       * (-45887. / 168. * vinij2 + 24025. / 42. * vinij * vjnij - 10469. / 42. * vjnij2
                          + 48197. / 840. * vi2 - 36227. / 420. * vivj + 36227. / 840. * vj2)
                 + G4 / (rij4 * rij)
                       * (16. * mj2 * mj2 + mi2 * mj2 * (175. - 41. / 16. * pi2) - 3187. / 1260. * mi2 * mi * mj
                          + mi * mj3 * (110741. / 630. - 41. / 16. * pi2))
                 + 22. / 3. * G3 * mi2 * mj
                       * (8 * vijnij * vijnij / rij4 - vij2 / rij4 + G * (mi + mj) / (rij4 * rij)))
                / c6;
            vij_factor +=
                (G * mj / rij2
                     * (15. / 2. * vinij * vjnij4 - 45. / 8. * vjnij4 * vjnij - 1.5 * vjnij2 * vjnij * vi2
                        + 6. * vinij * vjnij2 * vivj - 6. * vjnij2 * vjnij * vivj - 2. * vjnij * vivj * vivj
                        - 12. * vinij * vjnij2 * vj2 + 12. * vjnij2 * vjnij * vj2 + vjnij * vi2 * vj2
                        - 4. * vinij * vivj * vj2 + 8. * vjnij * vivj * vj2 + 4. * vinij * vj4 - 7. * vjnij * vj4)
                 + G2 * mj2 / rij3
                       * (-2. * vinij2 * vjnij + 8. * vinij * vjnij2 + 2. * vjnij2 * vjnij + 2. * vinij * vivj
                          + 4. * vjnij * vivj - 2. * vinij * vj2 - 4. * vjnij * vj2)
                 + G2 * mi * mj / rij3
                       * (-243. / 4. * vinij2 * vinij + 565. / 4. * vinij2 * vjnij - 269. / 4. * vinij * vjnij2
                          - 95. / 12. * vjnij2 * vjnij + 207. / 8. * vinij * vi2 - 137. / 8. * vjnij * vi2
                          - 36. * vinij * vivj + 27. / 4. * vjnij * vivj + 81. / 8. * vinij * vj2
                          + 83. / 8. * vjnij * vj2)
                 + G3 * mj3 / rij4 * (4. * vinij + 5. * vjnij)
                 + G3 * mi * mj2 / rij4 * (-307. / 8. * vinij + 479. / 8. * vjnij + 123. / 32. * vijnij * pi2)
                 + G3 * mi2 * mj / rij4 * (31397. / 420. * vinij - 36227. / 420. * vjnij)
                 - 44. / 3. * G3 * mi2 * mj * vijnij / rij4)
                / c6;
        }
        if (flags & PN_3_5) {
            const double vijnij4 = vijnij * vijnij * vijnij * vijnij;
            nij_factor +=
                (G4 * mi2 * mi * mj / (rij4 * rij) * (3992. / 105. * vinij - 4328. / 105. * vjnij)
                 + G4 * mi2 * mj2 / (rij4 * rij) * (-13576. / 105. * vinij + 2872. / 21. * vjnij)
                 - 3172. / 21. * G4 * mi * mj3 / (rij4 * rij) * vijnij
                 + G3 * mi2 * mj / rij4
                       * (48. * vinij2 * vinij - 696. / 5. * vinij2 * vjnij + 744. / 5. * vinij * vjnij2
                          - 288. / 5. * vjnij2 * vjnij - 4888. / 105. * vinij * vi2 + 5056. / 105. * vjnij * vi2
                          + 2056. / 21. * vinij * vivj - 2224. / 21. * vjnij * vivj - 1028. / 21. * vinij * vj2
                          + 5812. / 105. * vjnij * vj2)
                 + G3 * mi * mj2 / rij4
                       * (-582. / 5. * vinij2 * vinij + 1746. / 5. * vinij2 * vjnij - 1954. / 5. * vinij * vjnij2
                          + 158. * vjnij2 * vjnij + 3568. / 105. * vijnij * vi2 - 2864. / 35. * vinij * vivj
                          + 10048. / 105. * vjnij * vivj + 1432. / 35. * vinij * vj2 - 5752. / 105. * vjnij * vj2)
                 + G2 * mi * mj / rij3
                       * (-56. * vijnij4 * vijnij + 60. * vinij2 * vinij * vij2 - 180. * vinij2 * vjnij * vij2
                          + 174. * vinij * vjnij2 * vij2 - 54. * vjnij2 * vjnij * vij2 - 246. / 35. * vijnij * vi2 * vi2
                          + 1068. / 35. * vinij * vi2 * vivj - 984. / 35. * vjnij * vi2 * vivj
                          - 1068. / 35. * vinij * vivj * vivj + 180. / 7. * vjnij * vivj * vivj
                          - 534. / 35. * vinij * vi2 * vj2 + 90. / 7. * vjnij * vi2 * vj2
                          + 984. / 35. * vinij * vivj * vj2 - 732. / 35. * vjnij * vivj * vj2
                          - 204. / 35. * vinij * vj4 + 24. / 7. * vjnij * vj4))
                / c7;
            vij_factor +=
                (-184. / 21. * G4 * mi2 * mi * mj / (rij4 * rij) + 6224. / 105. * G4 * mi2 * mj2 / (rij4 * rij)
                 + 6388. / 105. * G4 * mi * mj3 / (rij4 * rij)
                 + G3 * mi2 * mj / rij4
                       * (52. / 15. * vinij2 - 56. / 15. * vinij * vjnij - 44. / 15. * vjnij2 - 132. / 35. * vi2
                          + 152. / 35. * vivj - 48. / 35. * vj2)
                 + G3 * mi * mj2 / rij4
                       * (454. / 15. * vinij2 - 372. / 5. * vinij * vjnij + 854. / 15. * vjnij2 - 152. / 21. * vi2
                          + 2864. / 105. * vivj - 1768. / 105. * vj2)
                 + G2 * mi * mj / rij3
                       * (60. * vijnij4 - 348. / 5. * vinij2 * vij2 + 684. / 5. * vinij * vjnij * vij2
                          - 66. * vjnij2 * vij2 + 334. / 35. * vi2 * vi2 - 1336. / 35. * vi2 * vivj
                          + 1308. / 35. * vivj * vivj + 654. / 35. * vi2 * vj2 - 1252. / 35. * vivj * vj2
                          + 292. / 35. * vj4))
                / c7;
        }
        for (int l = 0; l < 3; ++l) a_out[l] += nij_factor * nv_ij[l] + vij_factor * vv_ij[l];
    }
    for (int k = 0; k < 3; k++) { accvd(s, i, k) = a_out[k]; sder(s, i, k) = sd_out[k]; }
}

// ---------------------------------------------------------------------------------------------------------------

struct MstarKokkos::Impl {
    MstarKokkos& o;
    Dev d;
    int slots = 0, bslot = 0, cslot = 0, cap_N = -1, cap_pn = -1, cap_k = -1;
    int gbs_k = 4;
    // per-step result read back to the host: 0 T, 1 U, 2 time, 3 e0, 4 e1, 5 bad, 6 extrapolated B, 7 collision
    // pair key i N + j (-1: none), 8 r_sep, 9 r_coll, 10 type; from RES_PN the PN positions
    static constexpr int RES_PN = 12;
    DView<double*> res;
    DView<long> ckey;
    bool coll_on = false;
    Kokkos::View<double*, Kokkos::HostSpace> h_res;
    DView<double*> tmin;                // initial time step reduction
    // host tree
    std::vector<int> h_parent, h_etp, h_level, h_off, h_pe, h_ehi, h_elo;

    explicit Impl(MstarKokkos& owner) : o(owner) {}

    void allocate() {
        const int N = o.n, Npn = o.n_pn, K = o.opt.gbs_kmax;
        if (K > MAX_K || K < 4) throw std::runtime_error("MSTAR: gbs_kmax must be in 4..16");
        if (o.opt.max_tree_distance > MAX_ND) throw std::runtime_error("MSTAR: max_tree_distance > 8");
        d.N = N; d.E = N - 1; d.Npn = Npn; d.Nd = o.opt.max_tree_distance; d.kmax = K;
        d.flags = o.opt.PN_flags; d.G = o.opt.G; d.c = o.opt.c;
        d.nvars = 1 + 3 + 3 + 6 * (N - 1) + 3 * Npn;
        if (N == cap_N && Npn == cap_pn && K == cap_k) return;
        cap_N = N; cap_pn = Npn; cap_k = K;
        slots = K + 2;
        const int E = std::max(N - 1, 1), P = std::max(Npn, 1);
        d.mass = DView<double*>("mstar_mass", N);
        d.crad = DView<double*>("mstar_crad", N);
        d.cbh = DView<char*>("mstar_cbh", N);
        d.parent = DView<int*>("mstar_parent", N);
        d.etp = DView<int*>("mstar_etp", N);
        d.level = DView<int*>("mstar_level", N);
        d.path_off = DView<int*>("mstar_poff", N + 1);
        d.ehi = DView<int*>("mstar_ehi", E);
        d.elo = DView<int*>("mstar_elo", E);
        d.scal = DView<double**>("mstar_scal", slots, 4);
        d.root_s = DView<double**>("mstar_root", slots, 9);
        d.epos = DView<double***>("mstar_epos", slots, E, 3);
        d.evel = DView<double***>("mstar_evel", slots, E, 3);
        d.eaux = DView<double***>("mstar_eaux", slots, E, 3);
        d.pos = DView<double***>("mstar_pos", slots, N, 3);
        d.vel = DView<double***>("mstar_vel", slots, N, 3);
        d.acc = DView<double***>("mstar_acc", slots, N, 3);
        d.u = DView<double**>("mstar_u", slots, N);
        d.spin = DView<double***>("mstar_spin", slots, P, 3);
        d.aspin = DView<double***>("mstar_aspin", slots, P, 3);
        d.accvd = DView<double***>("mstar_accvd", slots, P, 3);
        d.sder = DView<double***>("mstar_sder", slots, P, 3);
        d.pnv = DView<double***>("mstar_pnv", slots, P, 3);
        d.data = DView<double**>("mstar_data", K + 1, d.nvars + 1);
        res = DView<double*>("mstar_res", RES_PN + 3 * P);
        h_res = Kokkos::View<double*, Kokkos::HostSpace>("mstar_hres", RES_PN + 3 * P);
        ckey = DView<long>("mstar_ckey");
        tmin = DView<double*>("mstar_tmin", 1);
        errview = DView<ErrVal>("mstar_err");
        bslot = K; cslot = K + 1;
    }

    // Prim's MST (mst.c construct_MST_Prim: root nearest the CoM, float edge weights) in dense O(N^2) form,
    // then the relative coordinates of slot s from the host state
    void build_tree(const double* pos, const double* vel, int s) {
        const double t0 = wall();
        const int N = o.n;
        h_parent.assign(N, -1); h_etp.assign(N, -1); h_level.assign(N, 0);
        h_ehi.assign(std::max(N - 1, 1), 0); h_elo.assign(std::max(N - 1, 1), 0);
        // ketju_compute_CoM
        double cp[3] = {0, 0, 0}, M = 0;
        for (int i = 0; i < N; i++) {
            M += o.mass[i];
            for (int k = 0; k < 3; k++) cp[k] += o.mass[i] * pos[3 * i + k];
        }
        for (int k = 0; k < 3; k++) cp[k] /= M;
        double rmin = DBL_MAX;
        int root = -1;
        for (int i = 0; i < N; i++) {
            double r2 = 0;
            for (int k = 0; k < 3; k++) { const double ds = pos[3 * i + k] - cp[k]; r2 += ds * ds; }
            if (rmin > std::sqrt(r2)) { rmin = std::sqrt(r2); root = i; }
        }
        std::vector<float> key(N, std::numeric_limits<float>::infinity());
        std::vector<int> par(N, -1);
        std::vector<char> in(N, 0);
        auto relax = [&](int v) {
            for (int w = 0; w < N; w++) {
                if (in[w]) continue;
                double r2 = 0;
                for (int k = 0; k < 3; k++) { const double ds = pos[3 * v + k] - pos[3 * w + k]; r2 += ds * ds; }
                const float wt = (float)std::sqrt(r2);
                if (wt < key[w]) { key[w] = wt; par[w] = v; }
            }
        };
        in[root] = 1;
        relax(root);
        std::vector<double> ep(3 * std::max(N - 1, 1)), ev(3 * std::max(N - 1, 1));
        for (int cnum = 0; cnum < N - 1; cnum++) {
            int v = -1;
            float best = std::numeric_limits<float>::infinity();
            for (int w = 0; w < N; w++)
                if (!in[w] && (v < 0 || key[w] < best)) { best = key[w]; v = w; }
            const int p = par[v];
            in[v] = 1;
            h_parent[v] = p;
            h_level[v] = h_level[p] + 1;
            h_etp[v] = cnum;
            h_ehi[cnum] = v;
            h_elo[cnum] = p;
            for (int k = 0; k < 3; k++) {
                ep[3 * cnum + k] = pos[3 * v + k] - pos[3 * p + k];
                ev[3 * cnum + k] = vel[3 * v + k] - vel[3 * p + k];
            }
            relax(v);
        }
        // root-to-vertex edge paths (edges are in Prim order: parents first)
        h_off.assign(N + 1, 0);
        for (int v = 0; v < N; v++) h_off[v + 1] = h_off[v] + h_level[v];
        h_pe.assign(std::max(h_off[N], 1), 0);
        for (int cnum = 0; cnum < N - 1; cnum++) {
            const int v = h_ehi[cnum], p = h_elo[cnum];
            for (int q = 0; q < h_level[p]; q++) h_pe[h_off[v] + q] = h_pe[h_off[p] + q];
            h_pe[h_off[v] + h_level[p]] = cnum;
        }
        d.root = root;
        // upload
        auto up_i = [](DView<int*>& dv, const std::vector<int>& hv, int n) {
            auto hm = Kokkos::create_mirror_view(dv);
            for (int q = 0; q < n; q++) hm(q) = hv[q];
            Kokkos::deep_copy(dv, hm);
        };
        up_i(d.parent, h_parent, N);
        up_i(d.etp, h_etp, N);
        up_i(d.level, h_level, N);
        up_i(d.path_off, h_off, N + 1);
        if ((int)d.path_edge.extent(0) < (int)h_pe.size()) d.path_edge = DView<int*>("mstar_pe", h_pe.size() + N);
        up_i(d.path_edge, h_pe, (int)h_pe.size());
        up_i(d.ehi, h_ehi, N - 1);
        up_i(d.elo, h_elo, N - 1);
        {
            auto se = Kokkos::subview(d.epos, s, Kokkos::ALL, Kokkos::ALL);
            auto sv = Kokkos::subview(d.evel, s, Kokkos::ALL, Kokkos::ALL);
            auto sr = Kokkos::subview(d.root_s, s, Kokkos::ALL);
            auto he = Kokkos::create_mirror_view(se);
            auto hv = Kokkos::create_mirror_view(sv);
            auto hr = Kokkos::create_mirror_view(sr);
            Kokkos::deep_copy(hr, sr);
            for (int e = 0; e < N - 1; e++)
                for (int k = 0; k < 3; k++) { he(e, k) = ep[3 * e + k]; hv(e, k) = ev[3 * e + k]; }
            for (int k = 0; k < 3; k++) { hr(R_POS + k) = pos[3 * root + k]; hr(R_VEL + k) = vel[3 * root + k]; }
            Kokkos::deep_copy(se, he);
            Kokkos::deep_copy(sv, hv);
            Kokkos::deep_copy(sr, hr);
        }
        o.perf.t_mst += wall() - t0;
    }

    // host state -> slot s (time 0), tree from it
    void upload_state(int s) {
        const int N = o.n;
        auto hm = Kokkos::create_mirror_view(d.mass);
        for (int i = 0; i < N; i++) hm(i) = o.mass[i];
        Kokkos::deep_copy(d.mass, hm);
        auto sp = Kokkos::subview(d.pos, s, Kokkos::ALL, Kokkos::ALL);
        auto sv = Kokkos::subview(d.vel, s, Kokkos::ALL, Kokkos::ALL);
        auto ss = Kokkos::subview(d.spin, s, Kokkos::ALL, Kokkos::ALL);
        auto hp = Kokkos::create_mirror_view(sp);
        auto hv = Kokkos::create_mirror_view(sv);
        auto hs = Kokkos::create_mirror_view(ss);
        for (int i = 0; i < N; i++)
            for (int k = 0; k < 3; k++) { hp(i, k) = o.pos[3 * i + k]; hv(i, k) = o.vel[3 * i + k]; }
        for (int i = 0; i < (int)hs.extent(0); i++)
            for (int k = 0; k < 3; k++) hs(i, k) = i < o.n_pn ? o.spin[3 * i + k] : 0.0;
        Kokkos::deep_copy(sp, hp);
        Kokkos::deep_copy(sv, hv);
        Kokkos::deep_copy(ss, hs);
        Kokkos::deep_copy(Kokkos::subview(d.scal, s, (int)S_TIME), 0.0);
        coll_on = (int)o.coll_R.size() == N;
        d.cc = o.coll_c;
        d.cmax1 = o.coll_bh_max_single;
        if (coll_on) {
            auto hr = Kokkos::create_mirror_view(d.crad);
            auto hb = Kokkos::create_mirror_view(d.cbh);
            for (int i = 0; i < N; i++) { hr(i) = o.coll_R[i]; hb(i) = (int)o.coll_bh.size() == N ? o.coll_bh[i] : 0; }
            Kokkos::deep_copy(d.crad, hr);
            Kokkos::deep_copy(d.cbh, hb);
        }
        build_tree(o.pos.data(), o.vel.data(), s);
    }
    // slot s -> host state (time relative to the run start)
    void download_state(int s, double& t) {
        const int N = o.n;
        auto sp = Kokkos::subview(d.pos, s, Kokkos::ALL, Kokkos::ALL);
        auto sv = Kokkos::subview(d.vel, s, Kokkos::ALL, Kokkos::ALL);
        auto ss = Kokkos::subview(d.spin, s, Kokkos::ALL, Kokkos::ALL);
        auto hp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sp);
        auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sv);
        auto hs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ss);
        for (int i = 0; i < N; i++)
            for (int k = 0; k < 3; k++) { o.pos[3 * i + k] = hp(i, k); o.vel[3 * i + k] = hv(i, k); }
        for (int i = 0; i < o.n_pn; i++)
            for (int k = 0; k < 3; k++) o.spin[3 * i + k] = hs(i, k);
        Kokkos::deep_copy(t, Kokkos::subview(d.scal, s, (int)S_TIME));
    }
    // MST rebuild from the positions of slot s (ketju_construct_MST after a step)
    void rebuild_tree(int s) {
        double t;
        download_state(s, t);
        build_tree(o.pos.data(), o.vel.data(), s);
    }

    // Forces (acc, u) of nslots consecutive slots from slot0: one team per (slot, i)
    void launch_force(int slot0, int nslots) {
        const Dev dd = d;
        const int N = d.N;
        Kokkos::parallel_for("mstar_force", Kokkos::TeamPolicy<ExecSpace>(nslots * N, Kokkos::AUTO),
                             KOKKOS_LAMBDA(const Team& t) {
            const int s = slot0 + t.league_rank() / N, i = t.league_rank() % N;
            Acc4 a;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, N), [&](int j, Acc4& sum) {
                if (j == i) return;
                double dr[3];
                const double r2 = dd.separation(s, i, j, dr);
                const double invr = 1. / sqrt(r2);
                const double Ginvr3 = dd.G * invr * invr * invr;
                const double mj = dd.mass(j);
                if (j > i) sum.u += mj * invr;
                sum.x += mj * (dr[0] * Ginvr3);
                sum.y += mj * (dr[1] * Ginvr3);
                sum.z += mj * (dr[2] * Ginvr3);
            }, Kokkos::Sum<Acc4>(a));
            Kokkos::single(Kokkos::PerTeam(t), [&]() {
                dd.acc(s, i, 0) = a.x; dd.acc(s, i, 1) = a.y; dd.acc(s, i, 2) = a.z;
                dd.u(s, i) = a.u;
            });
        });
        o.perf.launches++;
    }
    // T, U (from u), B = U - T of slot s, and the result vector
    void launch_energy(int s) {
        const Dev dd = d;
        Kokkos::parallel_for("mstar_energy", Kokkos::TeamPolicy<ExecSpace>(1, Kokkos::AUTO),
                             KOKKOS_LAMBDA(const Team& t) {
            const double T = dd.team_T(t, s);
            dd.team_U(t, s);
            Kokkos::single(Kokkos::PerTeam(t), [&]() {
                dd.scal(s, S_T) = T;
                dd.scal(s, S_B) = dd.scal(s, S_U) - T;
            });
        });
        o.perf.launches++;
    }
    // compute_total_energy for slot s (forces kernel for u, then T, U)
    void total_energy(int s) {
        launch_force(s, 1);
        launch_energy(s);
    }

    // first colliding pair (smallest i N + j) of slot s into ckey; dt_step from the times of slots s and s_prev
    void launch_collision(int s, int s_prev) {
        const Dev dd = d;
        const int N = d.N;
        const DView<long> key = ckey;
        if (!coll_on) {
            Kokkos::parallel_for("mstar_coll0", Kokkos::RangePolicy<ExecSpace>(0, 1), KOKKOS_LAMBDA(int) { key() = -1; });
            o.perf.launches++;
            return;
        }
        Kokkos::parallel_reduce("mstar_coll", Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(int i, long& mn) {
            const double dts = s_prev >= 0 ? dd.scal(s, S_TIME) - dd.scal(s_prev, S_TIME) : 0.0;
            for (int j = i + 1; j < N; j++) {
                double rsep, rc;
                int ty;
                if (dd.pair_collides(s, i, j, fabs(dts), rsep, rc, ty)) {
                    const long kq = (long)i * N + j;
                    if (kq < mn) mn = kq;
                    break;
                }
            }
        }, Kokkos::Min<long, MemSpace>(key));
        o.perf.launches++;
    }
    // collision result of slot s into res(7..10)
    void launch_collision_res(int s, int s_prev) {
        const Dev dd = d;
        const int N = d.N;
        const DView<long> key = ckey;
        const DView<double*> r = res;
        Kokkos::parallel_for("mstar_cres", Kokkos::RangePolicy<ExecSpace>(0, 1), KOKKOS_LAMBDA(int) {
            const long kq = key();
            r(7) = -1;
            if (kq < 0 || kq == Kokkos::reduction_identity<long>::min()) return;
            const int i = (int)(kq / N), j = (int)(kq % N);
            const double dts = s_prev >= 0 ? dd.scal(s, S_TIME) - dd.scal(s_prev, S_TIME) : 0.0;
            double rsep = 0, rc = 0;
            int ty = 0;
            dd.pair_collides(s, i, j, fabs(dts), rsep, rc, ty);
            r(7) = (double)kq; r(8) = rsep; r(9) = rc; r(10) = ty;
        });
        o.perf.launches++;
    }

    // estimate_initial_timestep on slot s
    double initial_timestep(int s) {
        const Dev dd = d;
        double t2min = DBL_MAX;
        const int N = d.N;
        Kokkos::parallel_reduce("mstar_dt0", Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(int i, double& mn) {
            const double mi = dd.mass(i);
            const int Li = dd.level(i);
            for (int j = i + 1; j < N; j++) {
                const double mj = dd.mass(j);
                const int Lj = dd.level(j);
                double r2 = 0, v2 = 0;
                int path[MAX_ND], sign[MAX_ND], dd_ = 0;
                bool proximity = false;
                if (abs(Li - Lj) < dd.Nd) proximity = dd.prox(j, i, dd_, path, sign);
                if (proximity) {
                    for (int k = 0; k < 3; k++) {
                        double dr = 0, dv = 0;
                        for (int l = 0; l < dd_; l++) {
                            dr += sign[l] * dd.epos(s, path[l], k);
                            dv += sign[l] * dd.evel(s, path[l], k);
                        }
                        r2 += dr * dr;
                        v2 += dv * dv;
                    }
                } else {
                    for (int k = 0; k < 3; k++) {
                        const double dr = dd.pos(s, j, k) - dd.pos(s, i, k);
                        r2 += dr * dr;
                        const double dv = dd.vel(s, j, k) - dd.vel(s, i, k);
                        v2 += dv * dv;
                    }
                }
                const double invr = 1. / sqrt(r2);
                const double t2 = 1. / (invr * (invr * v2 + dd.G * (mi + mj)));
                mn = t2 < mn ? t2 : mn;
            }
        }, Kokkos::Min<double>(t2min));
        o.perf.launches++;
        return 0.05 * std::sqrt(t2min);
    }

    // One GBS step attempt with fictitious step H: all k rows from the base slot, extrapolation, candidate state.
    // Returns after one device->host copy of res.
    void gbs_step(double H) {
        const int k = gbs_k, N = d.N, bs = bslot;
        const Dev dd = d;
        if (N <= o.opt.team_rows_max_n) {
            Kokkos::parallel_for("mstar_rows_team", Kokkos::TeamPolicy<ExecSpace>(k, Kokkos::AUTO),
                                 KOKKOS_LAMBDA(const Team& t) {
                const int ri = t.league_rank(), s = ri, nsub = 2 * (ri + 1);
                const double h = H / nsub;
                dd.team_copy(t, bs, s);
                dd.team_set_aux(t, s);
                dd.team_drift(t, s, h / 2);
                for (int st = 0; st < nsub; st++) {
                    dd.team_force(t, s);
                    dd.team_kick(t, s, h);
                    dd.team_drift(t, s, st == nsub - 1 ? h / 2 : h);
                }
                dd.team_pack(t, s, ri);
            });
            o.perf.launches++;
        } else {
            Kokkos::parallel_for("mstar_rows_start", Kokkos::TeamPolicy<ExecSpace>(k, Kokkos::AUTO),
                                 KOKKOS_LAMBDA(const Team& t) {
                const int ri = t.league_rank(), s = ri;
                const double h = H / (2 * (ri + 1));
                dd.team_copy(t, bs, s);
                dd.team_set_aux(t, s);
                dd.team_drift(t, s, h / 2);
            });
            o.perf.launches++;
            for (int st = 0; st < 2 * k; st++) {
                const int ri0 = st / 2;   // rows with 2 (ri + 1) > st
                launch_force(ri0, k - ri0);
                Kokkos::parallel_for("mstar_rows_kd", Kokkos::TeamPolicy<ExecSpace>(k - ri0, Kokkos::AUTO),
                                     KOKKOS_LAMBDA(const Team& t) {
                    const int ri = ri0 + t.league_rank(), s = ri, nsub = 2 * (ri + 1);
                    const double h = H / nsub;
                    dd.team_U(t, s);
                    dd.team_kick(t, s, h);
                    const bool last = st == nsub - 1;
                    dd.team_drift(t, s, last ? h / 2 : h);
                    if (last) dd.team_pack(t, s, ri);
                });
                o.perf.launches++;
            }
        }
        // extrapolation (gbs_extrapolate / extrapolate_single_variable)
        const DView<ErrVal> errv = errview;
        const double atol = o.opt.gbs_absolute_tolerance, otol = o.opt.output_time_absolute_tolerance;
        const int KR = d.kmax;
        const int nv = d.nvars;
        Kokkos::parallel_reduce("mstar_extrap", Kokkos::RangePolicy<ExecSpace>(0, d.nvars + 1),
                                KOKKOS_LAMBDA(int v, ErrVal& e) {
            double tt[MAX_K];
            for (int j = 0; j < k; j++) tt[j] = dd.data(j, v);
            double yscal = fabs(tt[k - 1]) + atol;
            if (v == 0) yscal += otol;
            for (int j = 0; j < k - 1; j++) {
                for (int i = k - 1; i >= j + 1; i--) {
                    const double ratio = (double)(2 * (i + 1)) / (double)(2 * (i - j));
                    const double nterm = ratio * ratio;
                    tt[i] = (nterm * tt[i] - tt[i - 1]) / (nterm - 1.0);
                }
            }
            const double result = tt[k - 1];
            if (v == nv) { dd.data(KR, v) = result; return; }   // B: extrapolated, not error controlled
            if (!isfinite(result)) { e.bad = 1; return; }
            dd.data(KR, v) = result;
            const double er = fabs(tt[k - 1] - tt[k - 2]) / yscal;
            const double el = fabs(tt[k - 2] - tt[k - 3]) / yscal;
            if (er > e.e0) e.e0 = er;
            if (el > e.e1) e.e1 = el;
        }, Kokkos::Sum<ErrVal, MemSpace>(errv));
        o.perf.launches++;
        // candidate state from the extrapolated row (computed whether or not the step converged; garbage when the
        // extrapolation was not finite, then discarded)
        const int cs = cslot;
        {
            Kokkos::parallel_for("mstar_unpack", Kokkos::TeamPolicy<ExecSpace>(1, Kokkos::AUTO),
                                 KOKKOS_LAMBDA(const Team& t) { dd.team_unpack(t, cs, KR); });
            o.perf.launches++;
            total_energy(cs);
            launch_collision(cs, bs);
            launch_collision_res(cs, bs);
            const DView<double*> r = res;
            const int Npn = d.Npn;
            Kokkos::parallel_for("mstar_res", Kokkos::RangePolicy<ExecSpace>(0, 1), KOKKOS_LAMBDA(int) {
                r(0) = dd.scal(cs, S_T);
                r(1) = dd.scal(cs, S_U);
                r(2) = dd.scal(cs, S_TIME);
                r(3) = errv().e0;
                r(4) = errv().e1;
                r(5) = errv().bad;
                r(6) = dd.data(KR, nv);
                for (int i = 0; i < Npn; i++)
                    for (int q = 0; q < 3; q++) r(RES_PN + 3 * i + q) = dd.pos(cs, i, q);
            });
            o.perf.launches++;
            const double t0 = wall();
            Kokkos::deep_copy(h_res, res);
            o.perf.t_sync += wall() - t0;
        }
        last_err.e0 = h_res(3);
        last_err.e1 = h_res(4);
        last_err.bad = h_res(5) != 0;
    }
    ErrVal last_err;
    DView<ErrVal> errview;
};

// ---------------------------------------------------------------------------------------------------------------

MstarKokkos::MstarKokkos(int num_pn, int num_other) : impl_(new Impl(*this)) { reset(num_pn, num_other); }
void MstarKokkos::reset(int num_pn, int num_other) {
    perf = Perf();
    mergers.clear();
    time = 0;
    n = num_pn + num_other;
    n_pn = num_pn;
    mass.assign(n, 0.0);
    pos.assign(3 * n, 0.0);
    vel.assign(3 * n, 0.0);
    spin.assign(3 * std::max(n_pn, 0), 0.0);
    extra_index.resize(n);
    for (int i = 0; i < n; i++) extra_index[i] = i;
}
MstarKokkos::~MstarKokkos() { delete impl_; }

static int num_gbs_substeps(int row) { return 2 * row; }
static int gbs_total_cost(int k) {
    int sum = 1;
    for (int i = 1; i <= k; ++i) sum += num_gbs_substeps(i);
    return k < 1 ? 0 : sum;
}
static double gbs_new_stepsize_fac(double scaled_err, int k) { return 0.94 * std::pow(0.65 / scaled_err, 1.0 / (2 * k - 1)); }

// BH mergers through the vendored C library (bh_mergers.c) on a host mirror; returns the number merged
// ketju_do_bh_mergers' condition on the PN positions in o.pos
static bool bh_merger_due(const MstarKokkos& o) {
    if (o.n_pn < 2) return false;
    const double fac = o.opt.bh_merger_distance_in_Rs * 2 * o.opt.G / (o.opt.c * o.opt.c);
    for (int i = 0; i < o.n_pn; ++i)
        for (int j = i + 1; j < o.n_pn; ++j) {
            double r2 = 0;
            for (int k = 0; k < 3; ++k) { const double dr = o.pos[3 * i + k] - o.pos[3 * j + k]; r2 += dr * dr; }
            if (std::sqrt(r2) < fac * (o.mass[i] + o.mass[j])) return true;
        }
    return false;
}
// The full host state (o.pos, o.vel, o.spin, o.time = absolute time) must be current.
static int host_bh_mergers(MstarKokkos& o) {
    struct ketju_system R;
    ketju_create_system(&R, o.n_pn, o.n - o.n_pn, MPI_COMM_SELF);
    R.constants->G = o.opt.G;
    R.constants->c = o.opt.c;
    R.options->PN_flags = o.opt.PN_flags;
    R.options->max_tree_distance = o.opt.max_tree_distance;
    R.options->bh_merger_distance_in_Rs = o.opt.bh_merger_distance_in_Rs;
    R.options->enable_bh_merger_kicks = o.opt.enable_bh_merger_kicks;
    R.physical_state->time = o.time;
    for (int i = 0; i < o.n; i++) {
        R.physical_state->mass[i] = o.mass[i];
        for (int k = 0; k < 3; k++) {
            R.physical_state->pos[i][k] = o.pos[3 * i + k];
            R.physical_state->vel[i][k] = o.vel[3 * i + k];
        }
    }
    for (int i = 0; i < o.n_pn; i++)
        for (int k = 0; k < 3; k++) R.physical_state->spin[i][k] = o.spin[3 * i + k];
    std::vector<int> extra(o.extra_index);
    R.particle_extra_data = extra.data();
    R.particle_extra_data_elem_size = sizeof(int);
    ketju_construct_MST(&R);
    const int merged = ketju_do_bh_mergers(&R);
    for (int m = 0; m < R.num_mergers; m++) {
        const auto& mi = R.merger_infos[m];
        o.mergers.push_back({mi.t_merger, mi.m1, mi.m2, mi.m_remnant, mi.chi1, mi.chi2, mi.chi_remnant, mi.v_kick,
                             extra[mi.extra_index1], extra[mi.extra_index2]});
    }
    // full particle list: the merged-away entries stay at the end of extra_index (KETJU's extra data rule)
    o.extra_index = extra;
    o.n = R.num_particles;
    o.n_pn = R.num_pn_particles;
    o.mass.assign(R.physical_state->mass, R.physical_state->mass + o.n);
    o.pos.resize(3 * o.n);
    o.vel.resize(3 * o.n);
    o.spin.resize(3 * o.n_pn);
    for (int i = 0; i < o.n; i++)
        for (int k = 0; k < 3; k++) {
            o.pos[3 * i + k] = R.physical_state->pos[i][k];
            o.vel[3 * i + k] = R.physical_state->vel[i][k];
        }
    for (int i = 0; i < o.n_pn; i++)
        for (int k = 0; k < 3; k++) o.spin[3 * i + k] = R.physical_state->spin[i][k];
    ketju_free_system(&R);
    return merged;
}

bool MstarKokkos::run(double time_interval) {
    const double tw0 = wall();
    if (n < 2) {
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < 3; ++k) pos[3 * i + k] += vel[3 * i + k] * time_interval;
        time += time_interval;
        perf.t_total += wall() - tw0;
        return true;
    }
    Impl& im = *impl_;
    im.allocate();
    im.gbs_k = opt.gbs_kmax >= 4 ? 4 : opt.gbs_kmax;
    const double t_start = time;
    time = 0;
    im.upload_state(im.bslot);
    int steps_since_reconstruction = 0;
    im.total_energy(im.bslot);
    auto read_scal = [&](int s, int q) {
        double v;
        Kokkos::deep_copy(v, Kokkos::subview(im.d.scal, s, q));
        return v;
    };
    double U = read_scal(im.bslot, S_U);
    const double initial_B = read_scal(im.bslot, S_B);
    coll_i = coll_j = -1;
    coll_type = 0;
    coll_rsep = coll_rcoll = 0;
    pn_dB = 0;
    auto take_collision = [&]() {
        const long kq = (long)im.h_res(7);
        coll_i = (int)(kq / n);
        coll_j = (int)(kq % n);
        coll_rsep = im.h_res(8);
        coll_rcoll = im.h_res(9);
        coll_type = (int)im.h_res(10);
    };
    if (im.coll_on) {   // a pair already inside its collision distance
        im.launch_collision(im.bslot, -1);
        im.launch_collision_res(im.bslot, -1);
        Kokkos::deep_copy(im.h_res, im.res);
        if (im.h_res(7) >= 0) {
            take_collision();
            double t;
            im.download_state(im.bslot, t);
            time = t_start;
            perf.t_total += wall() - tw0;
            return true;
        }
    }
    double dt = std::copysign(1.0, time_interval) * std::fmin(im.initial_timestep(im.bslot), std::fabs(time_interval) / 2);
    double Hstep = dt * U;
    bool finished = false, ok = true;
    double B_now = initial_B;
    do {
        if (opt.enable_bh_mergers && Hstep > 0 && bh_merger_due(*this)) {
            // pos holds the current PN positions (upload / res copy); the merger needs the full current state
            double t_now;
            im.download_state(im.bslot, t_now);
            time = t_start + t_now;            // merger times are absolute (KETJU records the run-relative time)
            const int num_merged = host_bh_mergers(*this);
            time = t_now;
            if (num_merged) {
                if (n < 2) {
                    for (int i = 0; i < n; ++i)
                        for (int k = 0; k < 3; ++k) pos[3 * i + k] += vel[3 * i + k] * (time_interval - t_now);
                    time = time_interval;
                    break;
                }
                im.allocate();
                im.upload_state(im.bslot);
                Kokkos::deep_copy(Kokkos::subview(im.d.scal, im.bslot, (int)S_TIME), t_now);
                steps_since_reconstruction = 0;
                im.total_energy(im.bslot);
                U = read_scal(im.bslot, S_U);
                B_now = read_scal(im.bslot, S_B);
            }
        }
        const int k = im.gbs_k;
        im.gbs_step(Hstep);
        const ErrVal err = im.last_err;
        double e0 = err.e0, e1 = err.e1;
        if (err.bad) e0 = opt.gbs_relative_tolerance * 0.65 / std::pow(0.1, 2 * k - 1);
        const double tol = opt.gbs_relative_tolerance;
        const double scaled_err = e0 / tol, scaled_err_lower = e1 / tol;
        const bool converged = e0 < tol;
        int new_k = k;
        double Hstep_new_k;
        if (converged) {
            double H_k = Hstep * gbs_new_stepsize_fac(scaled_err, k);
            const double H_km1 = Hstep * gbs_new_stepsize_fac(scaled_err_lower, k - 1);
            const int A_k = gbs_total_cost(k), A_km1 = gbs_total_cost(k - 1);
            const double W_k = A_k / H_k, W_km1 = A_km1 / H_km1;
            double Hnext = H_k;
            if (W_k / W_km1 < 0.9 && k + 1 <= opt.gbs_kmax) {
                new_k = k + 1;
                Hnext = (H_k * gbs_total_cost(k + 1)) / A_k;
            } else if (W_km1 / W_k < 0.9 && k > 4) {
                new_k = k - 1;
                Hnext = H_km1;
            }
            if (Hnext / Hstep > 5.0) Hnext = 5.0 * Hstep;
            if (H_k / Hstep > 5.0) H_k = 5.0 * Hstep;
            Hstep_new_k = Hnext;
            Hstep = H_k;
        } else {
            double f = gbs_new_stepsize_fac(scaled_err, k);
            if (f > 0.7) f = 0.7;
            if (f < 1e-3) f = 1e-3;
            Hstep *= f;
            Hstep_new_k = Hstep;
        }
        // step_total_work
        perf.total_work += gbs_total_cost(k) * (.5 * n * (n - 1)
                                                + ((double)n_pn * (n_pn - 1) * (opt.PN_flags & PN_THREEBODY ? n_pn - 2 : 1)));
        if (converged) {
            std::swap(im.bslot, im.cslot);
            if (opt.PN_flags != 0 && n_pn >= 2) pn_dB += im.h_res(6) - B_now;   // PN work of the step
            U = im.h_res(1);
            B_now = im.h_res(1) - im.h_res(0);
            time = im.h_res(2);
            for (int i = 0; i < n_pn; i++)
                for (int q = 0; q < 3; q++) pos[3 * i + q] = im.h_res(Impl::RES_PN + 3 * i + q);
            perf.successful_steps += 1;
            steps_since_reconstruction += 1;
            const double remaining_time = time_interval - time;
            if (im.h_res(7) >= 0) {   // collision at the end of this step: stop here
                take_collision();
                finished = true;
            } else if (std::fabs(remaining_time / time_interval) < opt.output_time_relative_tolerance
                || std::fabs(remaining_time) < opt.output_time_absolute_tolerance) {
                finished = true;
            } else {
                bool change_order = false;
                if (std::fabs(remaining_time) < std::fabs(Hstep / U) || time_interval * remaining_time < 0) {
                    Hstep = U * remaining_time;
                    if (new_k < k) {
                        change_order = true;
                        if (std::fabs(Hstep_new_k) < std::fabs(Hstep)) Hstep = std::copysign(Hstep_new_k, Hstep);
                    }
                } else if (new_k != k && std::fabs(remaining_time) > std::fabs(Hstep_new_k / U)) {
                    change_order = true;
                    Hstep = Hstep_new_k;
                }
                if (change_order) im.gbs_k = new_k;
                if (steps_since_reconstruction >= opt.steps_between_mst_reconstruction) {
                    im.rebuild_tree(im.bslot);
                    steps_since_reconstruction = 0;
                }
            }
        } else {
            perf.failed_steps += 1;
        }
        if (opt.max_step_count > 0 && perf.failed_steps + perf.successful_steps > opt.max_step_count) {
            ok = false;
            break;
        }
    } while (!finished);
    if (n >= 2) {
        double t;
        im.download_state(im.bslot, t);
        time = t;
    }
    time += t_start;
    perf.relative_energy_error = (B_now - initial_B) / initial_B;
    perf.t_total += wall() - tw0;
    return ok;
}

void measure_launch_latency(int reps, double& us_launch_fence, double& us_reduce_readback, double& us_async_launch) {
    DView<double*> v("lat", 1);
    Kokkos::fence();
    double t0 = wall();
    for (int r = 0; r < reps; r++) {
        Kokkos::parallel_for("lat_empty", Kokkos::RangePolicy<ExecSpace>(0, 1), KOKKOS_LAMBDA(int) { v(0) += 1; });
        Kokkos::fence();
    }
    us_launch_fence = (wall() - t0) / reps * 1e6;
    t0 = wall();
    for (int r = 0; r < reps; r++) {
        double s = 0;
        Kokkos::parallel_reduce("lat_red", Kokkos::RangePolicy<ExecSpace>(0, 32), KOKKOS_LAMBDA(int i, double& a) { a += v(0) + i; }, s);
    }
    us_reduce_readback = (wall() - t0) / reps * 1e6;
    Kokkos::fence();
    t0 = wall();
    for (int r = 0; r < reps; r++)
        Kokkos::parallel_for("lat_async", Kokkos::RangePolicy<ExecSpace>(0, 1), KOKKOS_LAMBDA(int) { v(0) += 1; });
    Kokkos::fence();
    us_async_launch = (wall() - t0) / reps * 1e6;
}

}  // namespace rumi_mstar
