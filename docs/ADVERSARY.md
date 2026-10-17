# Adversarial search for conservation-breaking configurations

`scripts/adversary.py` searches few-body initial conditions for the worst conservation violation of the code at its
current settings. The integrator parameters stay fixed; the search maximises the violation over configurations, so
that every path that can break E, P or L is found, fixed in the code and kept as a test (docs/CASES.md). It
complements the campaigns: they show what happens in clusters and how often, the search shows what can happen.
Parameter defaults are chosen separately, offline, by sweeps over seed ensembles, never by feedback on the energy.

## Objective
Per evaluation (one isolated run of 1-3 parent steps, closest approach mid-run, one CPU thread, deterministic):
|E_end - E_0| / E_REF of a parent cluster (E_REF, NGB_MREF of that run), the largest per-path booking residue
(SUBSYS_ERR), |dP|, |dL/L|, and slowness (GBS steps, wall time; a timeout is a hit). Ranked separately.
Hit thresholds: |dE| > 1e-12 E_REF per interval, |dL/L| > 1e-13, booking residue above round-off.

## Families
- triple: hierarchical triples (stable and unstable; secular path, SDAR, LogH)
- binflyby: binary + passing star (tidal sampling, weak perturbers, subsystem membership)
- planned: binary-binary; BH binary with PN near merger (PN, GW, mergers); each as stars, BHs and mixed.
Parameter ranges from physical priors (masses 0.1-100 Msun, a 1e-3-1e3 AU, 1-100 km/s).

## Search
1. random points in the family's unit cube;
2. bisection onto path boundaries (pairs with different path signatures: subsystem types, integrators used),
   where the errors live;
3. evolution strategy from the worst points.
Budget about 1-2 node-hours per family on the CPU partition (72 evaluations at a time per node). Embarrassingly
parallel: scale with nodes.

## Acceptance
The search must find known cases with the binaries before their fixes: the secular-triple energy leak (C4) and the
angular momentum of the frozen outer orbit (C6, found by its first smoke test), the weak-perturber stall (C2).

## From hit to case
Deduplicate (parameter distance, path signature), classify with the ledger and per-path residues, shrink to a fast
case, add the test, enter it in docs/CASES.md. Order of closing: booking errors first, then severity x frequency
(frequency from the campaigns).
