# BIFROST Fig. 8: 162k equal-mass cluster, 35 Myr

Rantala et al. 2023 (MNRAS 522, 5180; arXiv:2210.02472), Sec. 5.2 "A star cluster with equal-mass particles".
Line numbers below refer to `/raven/u/uli/papers/rantala/src/2210.02472/manuscript.tex`.

| Parameter | Value | Source |
|---|---|---|
| N | 162000 equal-mass | Fig. 8 caption, l. 862. The text (l. 868) says 161000. We use 162000 (also the figure label); with 0.5 Msun each, M = 8.1e4 Msun (the paper: "chosen to resemble a median Milky Way globular cluster") |
| m_star | 0.5 Msun | l. 868 |
| Profile | Plummer, r_h = 3.1 pc | l. 868; SCIC App. A, l. 1850-1882: inversion-sampled r(M), velocities from f(E) ~ E^(7/2) by von Neumann rejection. No truncation or virial rescaling is stated, so Q = 0.5 holds only statistically. `init_plummer` does the same |
| Softening | 0 (BIFROST); 1e-3 pc in the FROST-like run | l. 870 |
| r_ngb | 1e-3 pc | l. 870 (r_ngb,max = 0.1 pc: Sec. 3.3, the same value as in tests/paper/fig7) |
| eta_ff = eta_fb | 0.2 | l. 870 |
| eta_grad = eta_jerk | 0.1 | l. 870 |
| eta_GBS | 1e-10 | l. 870 |
| LogH end-time tolerance | 1e-4 | l. 870 (`NBODY_GBS_TTOL` default) |
| Interval / max step | 1e-3 Myr | l. 870 |
| End time | 35 Myr (35000 intervals) | l. 870 |
| Binaries, PN, SSE | none | equal-mass stars only, no BHs |
| Result (BIFROST) | mean per-interval \|dE_s/E\| ~ 8e-11, occasional intervals up to ~1e-8 (close hyperbolic encounters), total error grows linearly. The softened run has a mean of ~5e-11 | l. 872 |

**FROST comparison (arXiv:2011.14984).** The softened run of Fig. 8 uses the same ICs with Plummer softening of
1e-3 pc and only the plain fourth-order forward integrator (2210.02472 l. 866, 870). FROST itself used
eps_P = 1e-3 pc and eta = 0.2 (2011.14984 l. 820). Its cluster ICs (McLuster, evolved Kroupa IMF, r_1/2 = 3.5 pc,
App. B l. 942-975) are not the Fig. 8 ICs. `config_fig8_soft.txt` is that softened run
(`COLLISIONAL = false`, `SOFTENING_PC = 1e-3`). It is optional and not part of the chain.

**Files**
- `config_fig8.txt`: the unsoftened run. PRINT_INTERVAL = 1 gives the per-interval error for `scripts/step_err.py`
  and `scripts/run_summary.py`. It also sets STALL_RESTART = true and RESTART_FILE = auto.
- `make_runs.sh RUN_DIR EXE [SEEDS]`: writes the 15-min gpudev check job and one 24 h gpu-partition chain per
  seed (101 202 303 by default). It submits nothing.

**Chain logic.** Each segment runs `--restart auto` and stops cleanly at WALLTIME_LIMIT = 23.6 h. The segment
resubmits itself only if all of these hold: the newest `restart_*.h5` is new in this segment, its step (taken with
`head -1`, because ".h5" also matches the regex) is below 35000, and fewer than 12 segments have run.

**Status 2026-10-22.** The binary is built: main 6784753, CUDA, at
`/raven/ptmp/uli/cpp-nbody-runs/fig8-2026-10-22/nbody-main-6784753`. `make_runs.sh` has written the job
directories there (`gpudev_s101`, `s101`, `s202`, `s303`). Nothing has been submitted yet. Next step: run
`gpudev_s101/job.sh` to get the s/interval and the first per-interval |dE/E|. Then the node-hour estimate is
35000 x s/interval / 3600 per seed. The reference is about 1.3 s/interval at 1e5 (BIFROST Fig. 7) and 2-5
s/interval at 2e5 (Fig. 9). That implies about 15-50 node-h per seed, or 1-3 segments of 24 h.
