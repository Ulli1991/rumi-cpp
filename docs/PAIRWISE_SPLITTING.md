# Pairwise (HOLD-type) splitting inside HHS-FSI — derivation and estimate

Status: idea under evaluation (2026-10-13). Motivation: PeTar runs our 200k test cluster ~6x faster than we do even
with its tree off (θ = 0, direct far field), so the gain comes from its time-step structure, not from the tree.

## 1. Where our time goes

200k stars, 10 % binaries, 4 A100, one 1e-3 Myr step (seed 42, SDAR on): 7.7 s, of which the large HHS sets
(depths 0-4, 35k-200k particles) take 5.7 s, 4.3 s of it in the O(N²) force / time-step / gradient kernels.

Level populations (calls x mean set size): L0 1 x 200k, L1 2 x 186k, L2 4 x 158k, L3 8 x 108k, L4 16 x 35k.
At depth d, HHS recomputes **all F-F pairs** of the fast set F at every substep, e.g. at L3 108k x 108k ≈ 1.2e10
pairs, 8 times per step. Most of these pairs are far apart; their mutual force barely changes over 1.25e-4 Myr.
They are recomputed only because both members are "fast" for other reasons (their own close neighbours).

PeTar instead evaluates distant pairs once per tree step (2.4e-4 Myr) and short-steps only pairs inside the
changeover radius (~0.014 pc), on the CPU.

## 2. The FSI step at one HHS level (FROST, Rantala et al. 2021)

At a level with pivot τ the Hamiltonian is split as H = A + B with

- A = T + U_FF (the "drift" part, solved recursively at τ/2),
- B = U_SS + U_SF (the "kick" part, positions only),

and advanced with the fourth-order forward-symplectic step

    e^{τ/6 B} e^{τ/2 A} e^{2τ/3 B̃} e^{τ/2 A} e^{τ/6 B},   B̃ = B + (τ²/48) {B,{A,B}}.

It needs only (i) B depending on positions alone and (ii) a computable gradient term:
{A,B} = {T,B} + {U_FF,B} = {T,B} because U_FF and B are both position-only, so
{B,{A,B}} = {B,{T,B}} = −Σ_i |∇_i B|² / m_i (the "gradient" kick of FROST).

## 3. Pairwise splitting

Classify every pair (i,j) at the level by its own pair time step t_ij (the symmetrised free-fall / fly-by step
FROST already computes): fast if t_ij < τ, slow otherwise.

- U_fast = Σ_{fast pairs} φ_ij,  U_slow = Σ_{slow pairs} φ_ij,
- A = T + U_fast (recursion at τ/2 with the same rule applied to the fast pairs),
- B = U_slow.

Both FSI requirements hold unchanged: B is positions-only, and since U_fast commutes with B,
{B,{A,B}} = −Σ_i |∇_i U_slow|² / m_i. **The fourth-order FSI structure carries over formally.**
Kicks stay pairwise (Newton's third law per pair), so linear and angular momentum stay exact.

Work: at depth d only pairs with t_ij < τ_d enter the recursion, so the cost of deep levels scales with the number
of fast **pairs**, not with N_F². The top level still evaluates all slow pairs once per step (O(N²), as now).

Open points (literature check needed before coding):
- HOLD (Pelupessy, Jänes & Portegies Zwart 2012, New Astron. 17, 711) is the published pairwise splitting, but
  second order (leapfrog); Jänes et al. 2014 compare HOLD, HHS and connected components (CC) and report trade-offs.
  Why FROST and Rantala et al. 2023 chose particle-based HHS is to be checked.
- Error constants: the recursion makes A only approximately integrated; whether the effective order stays four
  with pair-based sets has to be measured (Kepler / planet floor tests, then the 200k energy rule on 3 seeds).
- Pair classification needs the pair steps of all pairs touching the fast set (an O(N_F x N) pass, as the
  time-step kernel already does), and a pair's class must be symmetric and consistent across the recursion.
- Subsystems (LogH/SDAR, Kepler) keep their exclusion: subsystem-internal pairs are never in U.

## 4. Estimate of the fast-pair count, and the puzzle it exposes

A pair is fast at L3 (τ = 1.25e-4 Myr) if η r / v < τ or η sqrt(r³ / G m) < τ. With σ ≈ 4-6 pc/Myr and η = 0.2
this means separations below ~1e-3 to 4e-3 pc. At 1e3-1e4 stars / pc³ a typical star has far less than one such
neighbour; only binaries, close encounters and stars near BHs do. **Pair steps alone cannot put 108k stars on L3.**

So something non-pairwise fills the deep levels: prime suspect the gradient criterion (η_grad = 0.2, computed from
a particle's total acceleration and its gradient), else the level-0 clamp or the perturbation step. PeTar has no
such criterion (everyone takes the tree step). If the gradient criterion is the cause, it is the real driver of the
large-set cost, and pairwise splitting alone would not remove it; the remedy would then be in the criterion.

Test queued 2026-10-13 (ptmp ab-2026-10-12b, grad{on,off}.log): 200k, 4 steps, ETA_GRAD = 0.2 vs 0 — level
populations and step time.

## 5. Decision path

1. Gradient on/off test → which criterion populates L1-L4.
2. Literature check (HOLD, Jänes 2014, FROST's choice).
3. If promising: CPU prototype of pairwise splitting at one level, floor tests (order), then 200k 3-seed energy rule.
