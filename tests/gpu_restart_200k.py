"""GPU check of the restart files on a BIFROST Fig. 9 cluster (200k Kroupa stars evolved to 1 Gyr, 10 % binaries,
all PN orders, FLYBY_CM_EPS): 2n steps against n steps + restart + n steps, L0 rows and particle dumps bit-identical,
on NBODY_TEST_LAUNCHER (e.g. 4 ranks, one A100 each) and on one rank; then a short run with snapshots in OUT_DIR for
LUNA's reader (tests/luna_read_check.py).

    python3 tests/gpu_restart_200k.py <nbody-kokkos> <out dir>       (inside a GPU allocation; see the job script
                                                                      tests/gpu_restart_200k.sbatch)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_case as rc  # noqa: E402

FIG9 = dict(TEST_CASE="plummer", INTEGRATOR="hhs", DT_MAX_MYR="1e-3", DT_MIN_MYR="1e-14", ETA_FF="0.2", ETA_FB="0.2",
            ETA_GRAD="0.1", ETA_JERK="0.1", SOFTENING_PC="0", SERIAL_N2_GPU="16384",
            COLLISIONAL="true", R_NGB="1e-3", R_NGB_MAX="0.1", ETA_PERT="0.02", FLYBY_CM_EPS="1e-11",
            TIDAL_STEP_RES="1.0", ETA_GBS="1e-10", ETA_RESTART="1e-8", N_ORB_SECULAR="3", M_PN="5.0", PN_ORDER="31",
            R_ESCAPE_RH="100.0", SSE_ENABLED="false", SSE_METALLICITY="0.0002", SSE_WIND_MODEL="VINK",
            SSE_PPISN_MODEL="STRONG", SSE_ECSN_MODEL="ON", SSE_MXNS="2.5", PLUMMER_N="200000", PLUMMER_R_HALF="3.1",
            PLUMMER_SEED="42", PLUMMER_MASS_DIST="kroupa_evolved", PLUMMER_AGE_GYR="1.0", PLUMMER_METALLICITY="0.0002",
            R_BREAK_AU="1000.0", BINARY_FRAC="0.1", BINARY_A_MIN_AU="0.01", BINARY_A_MAX_AU="100.0",
            BINARY_POP="bifrost", PROFILING="false", REPLICA_CHECK_INTERVAL="1")


def main(exe, out_dir):
    n = int(os.environ.get("GPU_RESTART_STEPS", "3"))
    launcher = os.environ.get("NBODY_TEST_LAUNCHER", "")
    ok = rc.restart_identical(exe, f"200k, 10 % binaries, launcher '{launcher}'", FIG9, n)
    one = os.environ.get("ONE_RANK_LAUNCHER")
    if one is not None:
        os.environ["NBODY_TEST_LAUNCHER"] = one
        ok &= rc.restart_identical(exe, f"200k, 10 % binaries, launcher '{one}'", FIG9, n, stop="walltime")
        os.environ["NBODY_TEST_LAUNCHER"] = launcher
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rc.run_in(exe, dict(FIG9, TOTAL_TIME_MYR="1.5e-3", PRINT_INTERVAL="1", SNAPSHOT_INTERVAL="1", RESTART_INTERVAL="1"),
              out_dir)
    print(f"snapshots and restart files of the short run: {out_dir}")
    print("ALL OK" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
