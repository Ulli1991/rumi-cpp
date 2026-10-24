# KETJU reference cases

The physics cases of the KETJU integrator's unit tests (Rantala+20 MSTAR, ketju-integrator `unit-tests/`), with
reference results produced by running KETJU, run through RUMI++: integration and mergers through the library path
(`src/lib/rumi_lib.h`: HHS with LogH+GBS subsystems), MST construction through `build_mst` (`mstar.h`).

## Layout and licences

| Path | Content | Licence |
|---|---|---|
| `data/*.txt` | one case per file: initial conditions, settings, tolerances, KETJU's results | data produced by running KETJU |
| `mstar_reference_test.cpp`, `mstar_reference_adapter.{h,cpp}` | harness (reads a case, runs RUMI++, checks) | BSD-3 (RUMI++) |
| `ketju_generator/` | the program that produced `data/` (links ketju-integrator, copies its test cases) | **GPLv3** (`ketju_generator/LICENSE.md`); not built by RUMI++'s CMake |

The harness contains no KETJU code; RUMI++ only stores KETJU's output numbers.

## Running

On by default with `ENABLE_COLLISIONAL=ON` (`RUMI_KETJU_REFERENCE_TESTS`, default `ON`): `mstar-reference-test` links the
`rumi` library target (or an internal copy of it when `BUILD_RUMI_LIBRARY=OFF`), and
`ctest --test-dir <build> -L ketju_reference` runs 29 tests `ketju_ref_<case>` (+ `ketju_ref_binary_no_pn_one_host_step`).
Cases RUMI++ cannot express exit 77 (skipped, reason printed).

The adapter scales each case to RUMI++ units (mass unit 1e6 Msun, length G M / c^2, velocity c; PN bodies are BHs,
kw 14, the others zero-radius stars), sets `PN_ORDER` from the case (`1pn`, `1pn+threebody` -> 1, `dynamic_all` -> 31,
`all` -> 63), `ETA_GBS` = the case's GBS tolerance, `N_ORB_SECULAR` huge, `SECULAR_TRIPLES = false`, `SDAR = false`,
`NBODY_GBS_TTOL = 1e-12` (KETJU's output-time tolerance), and advances in `KETJU_REF_HOST_STEPS` (default 1000) calls.
Spins are not available through `rumi_lib` and are not compared. When RUMI++'s MSTAR exists, route the cases (or the
MST ones) through it in `mstar_reference_adapter.cpp`.

## Status

FAIL cases are registered with `WILL_FAIL` (CMakeLists.txt) until fixed.


| Case(s) | RUMI++ |
|---|---|
| `binary_no_pn`, `binary_full_pn_no_spin` | pass (against the original test's expected state) |
| `instant_*` (4), `merger_from_orbit`, `mst_*` (16) | pass |
| `binary_full_pn_spin`, `..._generic_units` | skipped: no spin input/output in `rumi_lib`; KETJU's body 1 has chi = 4 (super-extremal) |
| `binary_1pn`, `binary_1pn_threebody` | pass: LogH evaluates the PN terms with lab-frame velocities (`LogHState::v_frame`); before, with velocities relative to the Newtonian c.o.m. of each call, which moves at 0.0425 c here and drifts under EIH for unequal masses |
| `head_on_merger` | FAIL: remnant 1.97265 < 1.99; the radiated-energy fit 0.0547 eta (1 - delta^2 / 2) (mergers.cpp) ignores the orbit (head-on NR ~0.06 %, quasi-circular ~4.8 %) |
| `binary_no_pn_one_host_step` | pass (the Peters shortcut needs radiation reaction, PN_ORDER 2.5PN bit) |

## Cases

| Case | Origin (KETJU test) | Checked |
|---|---|---|
| `binary_no_pn` | binary_system_no_pn | final state vs `expected_body` (old integrator), rel. tol. 20 x 1e-12 x KETJU steps; \|dE/E\| < 1e-10 |
| `binary_full_pn_no_spin` | binary_system_full_pn_no_spin | final state vs `expected_body`, rel. tol. 10 x 1e-12 x steps |
| `binary_1pn`, `binary_1pn_threebody` | binary_system_with_and_without_threebody_terms | final state, abs. tol. 2.5e-12 x steps x (40, 0.085); the two agree |
| `binary_full_pn_spin`, `..._generic_units` | binary_system_full_pn_spin_generic_units | final state incl. spins, rel. tol. 200 x 1e-12 x steps; G = 0.0634, c = 12354 must reproduce G = c = 1 |
| `head_on_merger` | head_on_merger | one remnant, mass in [1.99, 2], at rest at the origin with zero spin (abs. 1e-8) |
| `merger_from_orbit` | merger_from_orbit | one remnant, mass in (1.8, 2.01) |
| `instant_binary_merger`, `instant_multiple_merger1..3` | binary/multiple_merger_with_extra_data* | merger count and pairs (ids, component masses; any order, as it follows the merger radius: KETJU 6 R_S, RUMI++ 10 R_S), surviving ids |
| `mst_prim_N`, `mst_dnc_N` (N = 2 ... 1000) | MST/ConstructionTest.random_stars | spanning tree; prim: exact MST length and KETJU's edge set; dnc: length <= KETJU's |

`expected_body` (two cases) is the original test's hard-coded result of the old AR-CHAIN integrator, the reference the
KETJU test itself uses; current KETJU differs from it within the tolerance (e.g. 9e-8 in one component of
`binary_full_pn_no_spin`), so a second code is compared with that reference, not with KETJU's own result.
Tolerances are KETJU's own, evaluated with KETJU's accepted step count, and are stored in each file (`tol_rel`,
`tol_abs_pos`/`tol_abs_vel`, `tol_abs`). For `tol_rel` each component is compared relative to itself, floored at 1e-3
of the largest component of that vector (KETJU's test had no floor; zero components are exact there).
Remnant masses from GR fits are only range-checked (`remnant_mass_range`) or required to lie in (0.8, 1] (m1 + m2)
and to equal the final body mass. A remnant may carry either component's id.

Notes on the KETJU side:
- PN sets (`pn`): `none`, `1pn`, `1pn+threebody`, `dynamic_all` (1, 2, 2.5, 3, 3.5PN + three-body 1PN),
  `all` (`dynamic_all` + 1.5/2PN spin terms + spin evolution); `ketju_pn_flags` gives KETJU's bitmask.
  Spins are spin angular momenta S (chi = c S / (G m^2)). BH mergers at `merger_distance_Rs` x 2 G (m1 + m2) / c^2.
- `ketju_relative_energy_error` is the Newtonian energy change; meaningful only for `pn none`.
- KETJU's divide-and-conquer MST is a heuristic (2-13 % longer than the exact MST for N >= 20). The original test
  set `KETJU_MST_PRIM` regardless of its parameter, so its "DnC" cases ran Prim; the generator uses the named algorithm.
  The MST positions (std::mt19937(0x5eed), uniform(-1, 1)) are stored, not regenerated.

## Regenerating the data

    module load gcc openmpi
    tests/mstar_reference/ketju_generator/make_reference.sh <ketju-integrator source dir> [output dir]

`data/` was produced from github.com/Ulli1991/ketju-integrator eb96fd0 (mirror of the Helsinki repository); GIZMO's
modified copy (stella_dev 50438568 + the collision_radius init fix) gives bit-identical files, since its additions
(external tide, contact collisions) are off in these cases.
