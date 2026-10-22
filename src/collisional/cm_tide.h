#pragma once

// NBODY_CM_TIDE (CASES.md C24): field kicks of a bound 2-body subsystem whose kick spacing does not resolve its orbit
// go to its centre of mass only, and the tide of the kick's sources acts on the internal orbit as the orbit-averaged
// quadrupole (Milankovitch equations of the single-averaged tidal potential; Tremaine 2023 Eqs. 9-10, the averaged
// tidal tensor of PeTar, Wang et al. 2020b). NBODY6 moves unperturbed KS binaries the same way (c.o.m. only; Aarseth
// 2003 sec. 9.6; Mikkola & Aarseth 1996).
//
// Per kick (target range T, source range S, step s, kick weight h), for every bound 2-body subsystem with both
// members in T, the binary is in c.o.m. mode when (local criteria, hysteresis in brackets):
//   kick spacing s/2 > P/4 (stays on while s/2 > P/8),
//   gamma = 2 (a(1+e))^3 / M sum_k m_k / d_k^3 < NBODY_CM_TIDE_GAMMA (default 1e-3; stays on below 4x),
//   every source k changes the tide slowly: d_k / |v_k - V| > 2 P,
// with d_k the distance of source k from the binary's c.o.m. In c.o.m. mode:
//   1. both members get the mass-weighted mean of their accelerations (momentum exact),
//   2. every pair (binary, k) gets the correction F_avg,k - F_inst,k on the c.o.m. and its opposite on k
//      (momentum exact): the outer orbit then moves in the monopole plus the averaged quadrupole of the binary,
//   3. the internal orbit's angular momentum changes by h (tau - sum_k d_k x F_corr,k), tau the torque the c.o.m.
//      kick removed: total angular momentum exact by construction (the averaged quadrupole torque plus the torque of
//      what is not a point-mass pair force, e.g. the gradient term of the middle kick),
//   4. its eccentricity vector by the Milankovitch rate of the averaged quadrupole, kept on e.j = 0, e^2 + j^2 = 1,
//      at fixed a (the internal Kepler energy does not change); the members are rebuilt at the same mean longitude.
// Nothing is booked: the averaged flow conserves T + U up to the dropped short-period terms (bounded), and every
// integration error stays visible in the energy.
// Applies on the host (host subtrees, host builds) to sets of at most CM_TIDE_NMAX particles; elsewhere the kicks
// are the normal component kicks.

#include "particles.h"

namespace cm_tide {

constexpr int NMAX = 64;   // largest set (targets + sources) treated; = HOST_SUBTREE_MAX default

bool enabled();    // NBODY_CM_TIDE=1

// Source range and step of the next kick_velocities calls (set by the HHS kick sites; count 0: none)
struct KickSrc {
    int start = 0, count = 0;
    double step = 0.0, G = 0.0;
};
KickSrc& kick_src();
struct KickSrcScope {
    KickSrc saved;
    KickSrcScope(int start, int count, double step, double G) : saved(kick_src()) { kick_src() = {start, count, step, G}; }
    ~KickSrcScope() { kick_src() = saved; }
    KickSrcScope(const KickSrcScope&) = delete;
    KickSrcScope& operator=(const KickSrcScope&) = delete;
};

// Called by kick_velocities around its kernel. pre: decides the c.o.m. mode per binary, replaces the members'
// accelerations by their mean (returns false when nothing is to do). post: restores the accelerations, applies the
// pair corrections and the internal-orbit update.
bool pre(Particles& p, int start, int count, double h);
void post(Particles& p, double h);

}  // namespace cm_tide
