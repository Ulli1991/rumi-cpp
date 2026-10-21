#pragma once
// Gauss-Legendre implicit Runge-Kutta (Gauss-IRK) integration of a Newtonian subsystem (docs/CASES.md C22), plain C++
// host code like logh_gbs_step. s = 8 stages (order 16), symplectic and symmetric for any Hamiltonian (Hairer, Lubich &
// Wanner, GNI 2006, hereafter HLW, Thm. VI.4.2 / VI.4.3), applied with a FIXED step ds in the time-transformed variable to
// the Poincare-transformed Hamiltonian in extended phase space (HLW Sect. VIII.2; Mikkola & Tanikawa 1999; the LogH
// time transformation on the energy shell):
//   K(q, p, t, p_t) = g(q) (H(q, p) + p_t),  g = 1 / (-U(q)),  p_t = -(T + U) at the start of the call,
//   dq/ds = g p/m,  dp/ds = -(g + (H + p_t) g^2) grad U,  dt/ds = g.
// Implementation:
// - stage form Y_i = y_n + sum_j mu_ij L_j, L_j = h b_j f(Y_j), mu_ij + mu_ji = 1 imposed on the rounded coefficients,
//   compensated summation of the update (Antonana, Makazaga & Murua 2017, Numer. Algor. 76, 861, hereafter AMM17;
//   Hairer, McLachlan & Razakarivony 2008, BIT 48, 231, hereafter HMR08);
// - fixed-point iteration to round-off with the HMR08 stopping rule (increment zero, or no longer decreasing) and starting
//   values from the collocation polynomial of the previous step (HLW Sect. VIII.6.1);
// - where the fixed-point iteration does not contract (h L > 1: hard pericentres), simplified Newton with the Jacobian at
//   y_n (HLW Sect. VIII.6.2; AMM 2018, Numer. Algor. 78, 63) for that step; if that fails too, the call reports failure
//   with the state unchanged and the caller integrates it with LogH+GBS;
// - landing on t_end: fixed ds for the whole call; the step that would pass t_end is replaced by one step of the size
//   h* with t(h*) = t_end, found on the collocation polynomial of the rejected step (exact for collocation methods:
//   y_{n+1} = u(t_n + h)) and refined by Newton on t(h). Changing h once per call composes symplectic maps of different
//   step sizes, which is again symplectic; only a step size varying with the state at every step loses the long-time
//   behaviour (HLW Sect. VIII.1-2). The same end-time iteration on the regularised time is used by chain regularisation
//   (Mikkola & Aarseth 1993, Celest. Mech. 57, 439).
// P is conserved exactly (linear invariant), L = sum q x p to round-off (a quadratic invariant of the transformed system
// too: sum q_a x grad_a U = 0), so both stay at round-off like LogH.
#include "logh.h"

constexpr int GIRK_STAGES = 8;

struct GaussIrkStats {
    long long steps = 0;      // accepted steps
    long long rhs = 0;        // force evaluations (n^2 / 2 pair terms each)
    long long iters = 0;      // fixed-point iterations
    long long newton = 0;     // steps solved by simplified Newton
    long long rejected = 0;   // trial steps beyond t_end (replaced by the landing step)
};

struct GaussIrkOut {
    bool ok = false;
    int fail = 0;        // 1 step limit, 2 no convergence (fixed point and Newton), 3 collision watch, 4 not finite
    int steps = 0;
    double ds = 0.0;     // step in the time-transformed variable
};

struct GaussIrkOpts {
    double spo = 30.0;          // steps per innermost orbit (raised to 8 / sqrt(1 - e_max) for eccentric pairs)
    bool force_newton = false;  // tests: solve every step by simplified Newton
    bool spo_floor = true;      // tests: false keeps spo below the pericentre floor (the fixed point then fails)
};

// Step ds in the time-transformed variable for spo steps per innermost orbit: ds = integral of -U dt over the
// shortest orbital period of the bound pairs, with every bound pair's term at its orbit average G m_i m_j / a and the
// unbound ones at their current value (a phase-independent step: the current -U of an eccentric outer binary at its
// pericentre doubled ds for that call, 4e-10 instead of 7e-14 on the 2+2 test), / spo; without a bound pair, over
// 2 pi sqrt(r_min^3 / (G M)).
double gauss_irk_ds(const LogHState& state, double G, double spo, bool spo_floor = true);

// Integrate a Newtonian state (pos, vel, mass; use_aux_vel, spins and SDAR off) from state.t to t_end with opt.spo
// steps per innermost orbit. coll: pair collision distances [i * LOGH_MAX_BODIES + j] (nullptr: none); a pair whose
// separation or osculating pericentre falls below its distance stops the call (fail 3: the caller's GBS path decides
// collisions). On success pos, vel and t = t_end are set and B is left as it was (T + U + B is the energy error);
// on failure the state is unchanged.
void gauss_irk_integrate(LogHState& state, double t_end, double G, const GaussIrkOpts& opt, long long max_steps,
                         const double* coll, GaussIrkOut& out, GaussIrkStats* stats = nullptr);
