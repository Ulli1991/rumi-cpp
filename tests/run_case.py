"""Run one cpp-nbody test case from a small config and check the printed diagnostics against bounds."""
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

# Code defaults of the keys whose default became the production value of config.txt when the parameter table
# (src/core/params.cpp) replaced the per-call defaults (2026-10-12). Every case is written over these, so the cases
# run exactly as before; a case that wants the production value sets it.
OLD_DEFAULTS = dict(DT_MIN_MYR="1.0e-9", ETA="0.2", ETA_GRAD="0.1", ETA_JERK="0.1", ETA_RESTART="0", FLYBY_CM_EPS="0.0",
                    NGB_MASS_SCALING="false", SERIAL_N2_GPU="0", TREE_THRESHOLD="10000", TREE_THETA="0.5",
                    BINARY_A_PC="4.848e-5", BINARY_E="0.0", TRIPLE_A_OUT_PC="2e-7", TRIPLE_E_OUT="0.3",
                    TRIPLE_INCL_DEG="85.0", SDAR="false")
TIMEOUT_S = int(os.environ.get("NBODY_TEST_TIMEOUT", 120))  # seconds per run on one CPU core; GPUs need more (launch latency)
# FROST Fig. 4 binary: m1 = m2 = 10 Msun, a = 10 AU, ~1000 orbits; diagnostics every 1e-5 Myr
KEPLER = dict(TEST_CASE="binary_test", INTEGRATOR="hhs", TOTAL_TIME_MYR="7.0711e-3", DT_MAX_MYR="1.0e-5",
              DT_MIN_MYR="1.0e-14", SOFTENING_PC="0", COLLISIONAL="false", PRINT_INTERVAL="1")
# FROST Fig. 4 as published: endpoint |E(1000 P) - E(0)| / |E(0)|, the level grid aligned to the period
P_MYR = math.sqrt(1000 / 20) * 1e-6
FROST4 = dict(KEPLER, TOTAL_TIME_MYR=f"{1000 * P_MYR:.10e}", DT_MAX_MYR=f"{P_MYR:.10e}")
# FROST Fig. 5: Sun and the four giant planets (Tables A1-A2), 1000 yr
PLANETS = dict(KEPLER, TEST_CASE="outer_planets", TOTAL_TIME_MYR="1.0e-3")
# 400 Kroupa stars, half in contact binaries (a = 1-4 R_sun): ~75 stellar mergers in 20 steps, at HHS depths
# 0-10. Each merger compacts the particle arrays mid-recursion and may trigger step restarts. Contact binaries on
# purpose: old mass-rank pairing, no pericentre check.
MERGERS = dict(TEST_CASE="plummer", INTEGRATOR="hhs", TOTAL_TIME_MYR="2e-8", DT_MAX_MYR="1e-9",
               DT_MIN_MYR="1e-16", ETA_FF="0.2", ETA_FB="0.2", ETA_GRAD="0.1", ETA_JERK="0.1", SOFTENING_PC="0",
               COLLISIONAL="true", R_NGB="1e-3", R_NGB_MAX="0.1", ETA_PERT="0.02", ETA_GBS="1e-10",
               ETA_RESTART="1e-8", N_ORB_SECULAR="3", M_PN="5.0", PN_ORDER="0", R_ESCAPE_RH="100.0",
               SSE_ENABLED="false", PLUMMER_N="400", PLUMMER_M_TOTAL="200", PLUMMER_R_HALF="0.3", PLUMMER_SEED="7",
               PLUMMER_MASS_DIST="kroupa", BINARY_FRAC="0.5", BINARY_A_MIN_AU="0.005", BINARY_A_MAX_AU="0.02",
               BINARY_POP="mass_rank", BINARY_PERI_CHECK="false", PRINT_INTERVAL="1", PROFILING="false")
# 1e5 Msun BH binary, a = 2e-3 pc > r_ngb (no subsystem), e = 0.5, ~10 orbits with PN1: only the global PN1
# kick (BIFROST sec. 3.8) can precess it, 6.0e-5 rad/orbit
PN1_WIDE = dict(TEST_CASE="binary_test", INTEGRATOR="hhs", BINARY_A_PC="2e-3", BINARY_E="0.5", BINARY_M1="5e4",
                BINARY_M2="5e4", TOTAL_TIME_MYR="2.7e-4", DT_MAX_MYR="1.35e-6", DT_MIN_MYR="1e-16", SOFTENING_PC="0",
                COLLISIONAL="true", R_NGB="1e-3", R_NGB_MAX="1e-3", PN_ORDER="1", M_PN="5.0", SSE_ENABLED="false",
                PRINT_INTERVAL="20", PROFILING="false")
# 200 stars plus a 500+500 Msun BH binary (a = 2e-5 pc) with spins 0.9 along L, all PN orders
SPIN_IMBH = dict(TEST_CASE="plummer", INTEGRATOR="hhs", TOTAL_TIME_MYR="2e-3", DT_MAX_MYR="1e-4", DT_MIN_MYR="1e-16",
                 SOFTENING_PC="0", COLLISIONAL="true", PN_ORDER="31", SSE_ENABLED="false", PLUMMER_N="200",
                 PLUMMER_M_TOTAL="200", PLUMMER_R_HALF="0.5", PLUMMER_SEED="11", PLUMMER_MASS_DIST="equal",
                 ADD_IMBH="true", IMBH_M1="500", IMBH_M2="500", IMBH_A_PC="2e-5", IMBH_E="0.5", IMBH_SPIN="0.9",
                 PRINT_INTERVAL="5", PROFILING="false")
# 2000 Kroupa stars, 10 % primordial binaries (a = 0.1-100 AU, thermal e), two 1e-3 Myr steps
FLYBY_GROUP_STEPS = 39400  # flyby_group_floor: 30304 measured (host build)
FLYBY_GROUP_DE = 5.8e-10   # flyby_group_floor: 4.42e-10 measured (host build)
PN_BH_ONLY_DE = 7.7e-10   # pn_bh_only: 5.93e-10 measured (host build)
BINARIES = dict(TEST_CASE="plummer", INTEGRATOR="hhs", TOTAL_TIME_MYR="2e-3", DT_MAX_MYR="1e-3", DT_MIN_MYR="1e-14",
                ETA_FF="0.2", ETA_FB="0.2", ETA_GRAD="0.1", ETA_JERK="0.1", SOFTENING_PC="0", COLLISIONAL="true",
                R_NGB="1e-3", R_NGB_MAX="0.1", ETA_PERT="0.02", ETA_GBS="1e-10", ETA_RESTART="1e-8",
                N_ORB_SECULAR="3", PN_ORDER="0", SSE_ENABLED="false", PLUMMER_N="2000", PLUMMER_M_TOTAL="800",
                PLUMMER_R_HALF="1.0", PLUMMER_SEED="5", PLUMMER_MASS_DIST="kroupa", BINARY_FRAC="0.1",
                BINARY_A_MIN_AU="0.1", BINARY_A_MAX_AU="100", PRINT_INTERVAL="1", PROFILING="false")
# 500+500 Msun BH binary, a = 1e-7 pc (v/c ~ 0.02), all PN orders, steps of a fraction of an orbit (LogH)
PN_LOGH = dict(TEST_CASE="binary_test", INTEGRATOR="hhs", BINARY_A_PC="1e-7", BINARY_E="0.5", BINARY_M1="500",
               BINARY_M2="500", TOTAL_TIME_MYR="2e-9", DT_MAX_MYR="2e-10", DT_MIN_MYR="1e-18", SOFTENING_PC="0",
               COLLISIONAL="true", PN_ORDER="31", M_PN="5.0", ETA_RESTART="1e-8", SSE_ENABLED="false",
               PRINT_INTERVAL="1", PROFILING="false")
# Spinning BH binary in LogH (BIFROST sec. 3.7; PN_ORDER bit 32 = spin couplings), v/c ~ 0.02, ~21 orbits.
# Precession: e = 0, PN1 + spins, small chi1 perpendicular to L (simple precession about J), chi2 = 0.
SPIN_PREC = dict(PN_LOGH, BINARY_E="0.0", PN_ORDER="33", BINARY_CHI1="0.1 0 0", BINARY_CHI2="0 0 0")
# Total J = L_N + L_SO + S1 + S2: e = 0.5, two large misaligned spins, spin couplings only (no orbital PN
# terms, whose truncated L_PN1 alone leaves |dJ|/|J| ~ 5e-6 here)
SPIN_J = dict(PN_LOGH, PN_ORDER="32", BINARY_CHI1="0.6 0 0.6", BINARY_CHI2="0 0.7 -0.4")
# Same with PN1: J adds L_PN1; the error is the O((v/c)^4) truncation (5.3e-6 measured, 4.9e-6 with chi = 0)
SPIN_J_PN1 = dict(SPIN_J, PN_ORDER="33")
# 5000 Kroupa stars, 50 % primordial binaries (1250 pairs, more than the initial neighbour buffer of 1000)
BINARIES50 = dict(BINARIES, PLUMMER_N="5000", PLUMMER_M_TOTAL="2000", BINARY_FRAC="0.5", TOTAL_TIME_MYR="1e-3")
# Dense core (r_h = 0.05 pc), 50 % binaries of 1-20 AU: binary-single encounters form triples and multiplets.
# Kept on the old mass-rank pairing: its max |dE/E| sits at the restart threshold and scatters by 10x over seeds
# for either population (seeds 5-8, mass_rank 1.3e-8 - 1.8e-7, bifrost 2.7e-8 - 1.7e-7), so a population change
# would only re-roll that scatter.
TRIPLES = dict(BINARIES, PLUMMER_N="2000", PLUMMER_M_TOTAL="800", PLUMMER_R_HALF="0.05", BINARY_FRAC="0.5",
               BINARY_A_MIN_AU="1", BINARY_A_MAX_AU="20", TOTAL_TIME_MYR="2e-4", DT_MAX_MYR="1e-4",
               BINARY_POP="mass_rank")
# Stable hierarchical triple of stars (2 AU inner, 41 AU outer, 60 deg), steps of ~45 inner orbits: the
# secular-triple path. TRIPLE_OUTER_FIRST stores the outer body at the lowest particle id.
SEC_TRIPLE = dict(TEST_CASE="triple_test", INTEGRATOR="hhs", TRIPLE_A_IN_PC="1e-5", TRIPLE_E_IN="0.3", TRIPLE_M1="1.0",
                  TRIPLE_M2="0.8", TRIPLE_A_OUT_PC="2e-4", TRIPLE_E_OUT="0.2", TRIPLE_M3="0.6", TRIPLE_INCL_DEG="60",
                  TOTAL_TIME_MYR="1e-3", DT_MAX_MYR="1e-4", DT_MIN_MYR="1e-14", SOFTENING_PC="0", COLLISIONAL="true",
                  R_NGB="1e-3", R_NGB_MAX="0.1", N_ORB_SECULAR="3", PN_ORDER="0", SSE_ENABLED="false",
                  ETA_RESTART="0", PRINT_INTERVAL="1", PROFILING="false")
# 10+10 Msun BH binary, a = 2 AU, e = 0.5, rotated to omega = 1e-6 rad (apoapsis start has omega = pi), all PN
# orders, steps of ~16 orbits: the secular binary path; 1PN precession ~4e-5 rad per step
SEC_BINARY = dict(TEST_CASE="binary_test", INTEGRATOR="hhs", BINARY_A_PC="1e-5", BINARY_E="0.5", BINARY_M1="10.0",
                  BINARY_M2="10.0", BINARY_ROT_DEG="180.0000573", TOTAL_TIME_MYR="1e-4", DT_MAX_MYR="1e-5", DT_MIN_MYR="1e-14", SOFTENING_PC="0",
                  COLLISIONAL="true", R_NGB="1e-3", R_NGB_MAX="0.1", N_ORB_SECULAR="3", PN_ORDER="31", M_PN="5.0",
                  SSE_ENABLED="false", ETA_RESTART="0", PRINT_INTERVAL="1", PROFILING="false")
# 2000 Kroupa stars, 50 % binaries of 0.01-0.1 AU (2-21 R_sun) with thermal e: many pericentres inside R1 + R2
# unless the IC generator enforces r_p > R1 + R2 (AUDIT F6). One 1e-5 Myr step (hundreds of orbits).
CLOSE_BINARIES = dict(BINARIES, BINARY_FRAC="0.5", BINARY_A_MIN_AU="0.01", BINARY_A_MAX_AU="0.1",
                      TOTAL_TIME_MYR="1e-5", DT_MAX_MYR="1e-5")
# 1.0+0.8 Msun stellar binary, a = 0.02 AU, e = 0.9, all PN orders on (M_PN = 5): ~2400 orbits per half step,
# secular path. Below m_pn it is a Kepler drift (as in LogH): no PN precession sub-steps
SEC_STELLAR = dict(SEC_BINARY, BINARY_A_PC="1e-7", BINARY_E="0.9", BINARY_M1="1.0", BINARY_M2="0.8",
                   BINARY_ROT_DEG="0")
# 10+10 Msun BH binary, a = 0.01 AU, e = 0.5, all PN orders: 1.7 rad of 1PN precession per yr, t_GW ~ 1 Myr
SEC_BH_TIGHT = dict(SEC_BINARY, BINARY_A_PC="5e-8", BINARY_ROT_DEG="0")
# Tight binary (0.02 AU, P = 2.1e-3 yr) on a Mardling-unstable outer orbit (2 AU, e = 0.95, starts at
# apocentre): LogH, ~140 inner orbits per half step of 0.3 yr, integrated orbit by orbit
HIER_LOGH = dict(TEST_CASE="triple_test", INTEGRATOR="hhs", TRIPLE_A_IN_PC="1e-7", TRIPLE_E_IN="0.1", TRIPLE_M1="1.0",
                 TRIPLE_M2="1.0", TRIPLE_A_OUT_PC="1e-5", TRIPLE_E_OUT="0.95", TRIPLE_M3="1.0", TRIPLE_INCL_DEG="30",
                 TOTAL_TIME_MYR="6e-7", DT_MAX_MYR="6e-7", DT_MIN_MYR="1e-16", SOFTENING_PC="0", COLLISIONAL="true",
                 R_NGB="1e-3", R_NGB_MAX="0.1", N_ORB_SECULAR="3", PN_ORDER="0", SSE_ENABLED="false",
                 ETA_RESTART="1e-8", ETA_GBS="1e-10", PRINT_INTERVAL="1", PROFILING="false")
# Wang, Nitadori & Makino 2020 (SDAR) Table 1 B-S triple in physical units: inner 0.9 + 0.1 Msun at 0.02 AU,
# e = 0.9; outer 2 Msun at 20 AU, e = 0.99; 80 deg; both at apocentre. t_KL ~ 680 yr (their Eq. 53), 2.3e5 inner
# orbits; kappa ~ 52 at the outer apocentre (as in the paper). LogH path (SECULAR_TRIPLES = false).
KL_TRIPLE = dict(SEC_TRIPLE, SECULAR_TRIPLES="false", TRIPLE_A_IN_PC="1e-7", TRIPLE_E_IN="0.9", TRIPLE_M1="0.9",
                 TRIPLE_M2="0.1", TRIPLE_A_OUT_PC="1e-4", TRIPLE_E_OUT="0.99", TRIPLE_M3="2.0", TRIPLE_INCL_DEG="80.2",
                 TOTAL_TIME_MYR="5e-5", DT_MAX_MYR="1e-5", ETA_RESTART="1e-8")
# AUDIT F3 system (200k cluster): ~1 Msun binary at 0.05 AU and a 0.7 Msun star passing at 0.01 pc with v_inf = 190
# km/s (e = 31100), starting at 0.05 pc; R_NGB = 0.02 pc keeps it in the subsystem as the fly-by criterion does with
# long steps. Plain LogH integrates ~3e4 binary orbits per 1e-4 Myr orbit by orbit. (Until 2026-10-12 with
# FLYBY_CM_VELOCITY = true; the cases on it measure the same with component velocities.)
F3_FLYBY = dict(HIER_LOGH, TRIPLE_A_IN_PC="2.4e-7", TRIPLE_A_OUT_PC="0.01", TRIPLE_E_OUT="31100", TRIPLE_M3="0.7",
                R_NGB="0.02", TOTAL_TIME_MYR="2e-4", DT_MAX_MYR="1e-4")
# 10 + 10 Msun BH binary at 0.02 AU with a 1 Msun star at 20 AU (e_out = 0.3), all PN orders: SDAR must keep
# kappa = 1 (FROST-CLUSTERS I App. A2); with PN off the same triple is slowed down (control)
PN_TRIPLE = dict(HIER_LOGH, TRIPLE_M1="10.0", TRIPLE_M2="10.0", TRIPLE_A_OUT_PC="1e-4", TRIPLE_E_OUT="0.3",
                 TRIPLE_M3="1.0", TRIPLE_INCL_DEG="40", PN_ORDER="31", M_PN="5.0", TOTAL_TIME_MYR="5e-6",
                 DT_MAX_MYR="2.5e-6", ETA_RESTART="0", SECULAR_TRIPLES="false")
# Wide eccentric BH binary (30 + 5.6 Msun, a = 40 AU, e = 0.97, P = 50 yr: a subsystem, LogH) and a 0.55 Msun
# star at ~30 binary apocentres (a_out = 0.012 pc, outside r_ngb): the 20k cluster's seed-13 configuration. The
# star's pair step is ~10 binary periods, and its kicks sample the binary's orbit
CM_TIDE = dict(TEST_CASE="triple_test", INTEGRATOR="hhs", TRIPLE_A_IN_PC="2e-4", TRIPLE_E_IN="0.97", TRIPLE_M1="30.0",
               TRIPLE_M2="5.6", TRIPLE_A_OUT_PC="1.2e-2", TRIPLE_E_OUT="0.3", TRIPLE_M3="0.55", TRIPLE_INCL_DEG="40",
               TOTAL_TIME_MYR="1e-2", DT_MAX_MYR="1e-3", DT_MIN_MYR="1e-14", SOFTENING_PC="0", COLLISIONAL="true",
               R_NGB="1e-3", R_NGB_MAX="0.1", N_ORB_SECULAR="3", PN_ORDER="0", SSE_ENABLED="false", ETA_RESTART="0",
               PRINT_INTERVAL="1", PROFILING="false")
# Hard stellar binary (0.92 + 0.92 Msun, a = 0.011 AU, e = 0.86, ~1e6 orbits per 1e-3 Myr: secular) and a slow star
# passing at 0.011 pc (e = 22, from 0.057 pc): the 200k, 10 % binaries seed-13 configuration (component velocities).
# The inner orbit is turned so that at t = 0 a member's velocity points at the star (its pair "pericentre" is ~0)
# Hard 20 + 20 Msun binary (a = 1 AU, e = 0.5, ~1e4 orbits per 1e-3 Myr: Kepler drift) and a 0.05 Msun body on a
# 0.01 pc orbit (e = 0.3, one outer orbit): a weak, slowly varying tide (quadrupole energy D ~ 3e-13 |E0|). With
# c.o.m. fly-by steps the body's kicks span ~3000 binary orbits and sample the quadrupole force at random phases
WEAK_TIDE = dict(CM_TIDE, TRIPLE_A_IN_PC="5e-6", TRIPLE_E_IN="0.5", TRIPLE_M1="20.0", TRIPLE_M2="20.0",
                 TRIPLE_A_OUT_PC="0.01", TRIPLE_E_OUT="0.3", TRIPLE_M3="0.05", TOTAL_TIME_MYR="1.5e-2",
                 FLYBY_CM_VELOCITY="true", FLYBY_CM_EPS="0")
FLYBY_TRIPLE = dict(TEST_CASE="triple_test", INTEGRATOR="hhs", TRIPLE_A_IN_PC="5.54e-8", TRIPLE_E_IN="0.86",
                    TRIPLE_M1="0.923", TRIPLE_M2="0.923", TRIPLE_A_OUT_PC="0.0113", TRIPLE_E_OUT="22.0",
                    TRIPLE_M3="0.352", TRIPLE_INCL_DEG="0", TRIPLE_IN_ROT_DEG="-170.58", TOTAL_TIME_MYR="2e-3",
                    DT_MAX_MYR="1e-3",
                    DT_MIN_MYR="1e-16", SOFTENING_PC="0", COLLISIONAL="true", R_NGB="1e-3", R_NGB_MAX="0.1",
                    N_ORB_SECULAR="3", PN_ORDER="0", SSE_ENABLED="false", ETA_RESTART="1e-8", ETA_GBS="1e-10",
                    PRINT_INTERVAL="1", PROFILING="false")
# Collisions inside the regularised subsystem integration (FROST-CLUSTERS I App. B3). Stellar radii of the test
# bodies from TEST_STELLAR_RADIUS_RSUN (BHs: none). Head-on: 1+1 Msun stars of 1 R_sun on a radial inner orbit
# (a = 0.2 AU, e = 0.99999, r_p = 2e-6 R_sun) with a 1 Msun star at 2 AU: a 3-body subsystem, which the per-level
# check (2-body subsystems only) never merges.
COLL_HEADON = dict(TEST_CASE="triple_test", INTEGRATOR="hhs", TRIPLE_A_IN_PC="1e-6", TRIPLE_E_IN="0.99999",
                   TRIPLE_M1="1.0", TRIPLE_M2="1.0", TRIPLE_A_OUT_PC="1e-5", TRIPLE_E_OUT="0.2", TRIPLE_M3="1.0",
                   TRIPLE_INCL_DEG="30", TOTAL_TIME_MYR="4e-7", DT_MAX_MYR="1e-7", DT_MIN_MYR="1e-16", SOFTENING_PC="0",
                   COLLISIONAL="true", R_NGB="1e-3", R_NGB_MAX="0.1", N_ORB_SECULAR="3", PN_ORDER="0",
                   SSE_ENABLED="false", ETA_RESTART="1e-8", ETA_GBS="1e-10", PRINT_INTERVAL="1", PROFILING="false",
                   TEST_STELLAR_RADIUS_RSUN="1.0")
# The 200k step-1 stall (AUDIT F3, SDAR runs with BINARY_PERI_CHECK = false): a contact binary, 1+1 Msun at 0.05 AU,
# e = 0.6 (R = 2.1300 R_sun: r_p just inside r_coll, so the steps jump over the short stretch inside r_coll unless
# one is forced onto the pericentre; R = 2.131 was too deep for gcc 15, whose steps hit the overlap directly, and
# R <= 2.1295 misses; 2.1298-2.1302 lands on gcc 13 and gcc 15) and a 0.7 Msun star passing at 0.003 pc (190 km/s, inside
# R_NGB from the start): a 3-body subsystem with SDAR, kappa = 1 for the candidate
COLL_CONTACT = dict(F3_FLYBY, TRIPLE_E_IN="0.6", TRIPLE_A_OUT_PC="0.003", SDAR="true", TEST_STELLAR_RADIUS_RSUN="2.1300")
# TDE on a parabolic orbit: a 1 Msun star of 1 R_sun reaches q = 1e-7 pc from a 100 Msun BH (r_t = 1.3 (M/m)^{1/3}
# R = 6.03 R_sun = 1.36e-7 pc; q / r_t = 0.74), starting at 5 q; a 4 Msun star at 1e-4 pc (circular) binds the system
# (E0 != 0 for the relative energy error)
COLL_TDE = dict(COLL_HEADON, TRIPLE_A_IN_PC="1e-7", TRIPLE_E_IN="1.0", TRIPLE_M1="100.0", TRIPLE_M2="1.0",
                TRIPLE_A_OUT_PC="1e-4", TRIPLE_E_OUT="0.0", TRIPLE_M3="4.0", TOTAL_TIME_MYR="4e-9", DT_MAX_MYR="1e-9",
                DT_MIN_MYR="1e-18", M_PN="5.0")
# BH binary reaching the ISCO: 10+10 Msun at a = 1.5 r_ISCO (6 G M / c^2 = 5.74e-12 pc), circular, all PN orders: the
# 2.5PN / 3.5PN inspiral reaches r_ISCO within the first step; a 1 Msun star at 100 a makes it a 3-body subsystem
COLL_ISCO = dict(COLL_HEADON, TRIPLE_A_IN_PC="8.6e-12", TRIPLE_E_IN="0.0", TRIPLE_M1="10.0", TRIPLE_M2="10.0",
                 TRIPLE_A_OUT_PC="8.6e-10", TRIPLE_E_OUT="0.0", TRIPLE_M3="1.0", TOTAL_TIME_MYR="2e-14",
                 DT_MAX_MYR="1e-15", DT_MIN_MYR="1e-24", PN_ORDER="31", M_PN="5.0", TEST_STELLAR_RADIUS_RSUN="0")

# Wide stellar binary (0.97 + 0.97 Msun, a = 46 AU, e = 0.5, P = 220 yr) and a 46.5 Msun black hole passing at 1.6 mpc
# (outside r_ngb = 1 mpc; e = 2.5, from 8 mpc): tidal ratio ~0.1 at pericentre. The 200k, 10 % binaries seed-42
# step-19 event (there a 23+23 Msun BH binary). NGB_MREF = the cluster's <m> (0.41 Msun)
BH_FLYBY = dict(TEST_CASE="triple_test", INTEGRATOR="hhs", TRIPLE_A_IN_PC="2.2e-4", TRIPLE_E_IN="0.5", TRIPLE_M1="0.97",
                TRIPLE_M2="0.97", TRIPLE_A_OUT_PC="1.6e-3", TRIPLE_E_OUT="2.5", TRIPLE_M3="46.5", TRIPLE_INCL_DEG="30",
                TOTAL_TIME_MYR="2e-3", DT_MAX_MYR="1e-3", DT_MIN_MYR="1e-16", SOFTENING_PC="0", COLLISIONAL="true",
                R_NGB="1e-3", R_NGB_MAX="0.1", N_ORB_SECULAR="3", PN_ORDER="0", SSE_ENABLED="false",
                ETA_RESTART="1e-8", ETA_GBS="1e-10", PRINT_INTERVAL="1", PROFILING="false", NGB_MREF="0.41",
                FLYBY_CM_EPS="1e-11")
# Kepler drift of an isolated binary subsystem: 1 + 1 Msun, a = 1e-5 pc (2 AU, P = 2.0944e-6 Myr), PN off, run for
# T = 100.37 P as one step or as 16 steps of 6.27 P (both above N_ORB_SECULAR = 3: the secular binary path)
G_CODE = 4.302194e-3                 # constants.h: pc^3 Msun^-1 TU^-2
TU_MYR = 0.977792
P_SEC_MYR = 2.0 * math.pi * math.sqrt(1e-15 / (G_CODE * 2.0)) * TU_MYR
SEC_DRIFT = dict(TEST_CASE="binary_test", INTEGRATOR="hhs", BINARY_A_PC="1e-5", BINARY_E="0.0", BINARY_M1="1.0",
                 BINARY_M2="1.0", TOTAL_TIME_MYR=f"{100.37 * P_SEC_MYR:.15e}", DT_MAX_MYR=f"{100.37 * P_SEC_MYR:.15e}",
                 DT_MIN_MYR="1e-14", SOFTENING_PC="0", COLLISIONAL="true", R_NGB="1e-3", R_NGB_MAX="0.1",
                 N_ORB_SECULAR="3", PN_ORDER="0", SSE_ENABLED="false", ETA_RESTART="0", PRINT_INTERVAL="1",
                 PROFILING="false")
# Kepler path of isolated non-PN pairs: the same binary at e = 0.7 with 0.5 or 2 orbits per step (below
# N_ORB_SECULAR: LogH before 9efdca7), and a hyperbolic encounter (q = 1e-5 pc, e = 1.5, from 5 q inbound) over 20 pericentre crossing times
KEP_2B = dict(SEC_DRIFT, BINARY_E="0.7", TOTAL_TIME_MYR=f"{2.0 * P_SEC_MYR:.15e}", DT_MAX_MYR=f"{0.5 * P_SEC_MYR:.15e}",
              ETA_GBS="1e-10")
T_CROSS_MYR = math.sqrt(1e-15 / (G_CODE * 2.0)) * TU_MYR
KEP_2B_HYP = dict(KEP_2B, BINARY_E="1.5", TOTAL_TIME_MYR=f"{20.0 * T_CROSS_MYR:.15e}",
                  DT_MAX_MYR=f"{5.0 * T_CROSS_MYR:.15e}")
# Contact binary through the Kepler path: 1 + 1 Msun, a = 0.05 AU, e = 0.6, R = 2.152 R_sun (r_p = 4.3018 R_sun, r_p /
# r_coll = 0.9995: the steps would jump over the short stretch inside r_coll unless one lands on the pericentre),
# steps of 0.3 P from the apocentre: the first drift ends before the pericentre (Kepler drift), the next contains it
# (candidate: LogH decides the collision at the pericentre, FROST-CLUSTERS I App. B3)
P_CONTACT_MYR = 2.0 * math.pi * math.sqrt((0.05 * 4.84814e-6) ** 3 / (G_CODE * 2.0)) * TU_MYR
KEP_2B_CONTACT = dict(KEP_2B, BINARY_A_PC=f"{0.05 * 4.84814e-6:.10e}", BINARY_E="0.6", TEST_STELLAR_RADIUS_RSUN="2.152",
                      TOTAL_TIME_MYR=f"{1.5 * P_CONTACT_MYR:.15e}", DT_MAX_MYR=f"{0.3 * P_CONTACT_MYR:.15e}",
                      ETA_RESTART="1e-8")

# 16 stars inside r_ngb: one 15-body neighbour group, more than LogH's fixed storage (LOGH_MAX_BODIES = 8)
# Collisional FSI run of a field binary (r_ngb below a, so no subsystem): the start-of-step kernel's criteria set the step
FSI_BINARY = dict(TEST_CASE="binary_test", INTEGRATOR="fsi", TOTAL_TIME_MYR="1e-5", DT_MAX_MYR="1e-5", DT_MIN_MYR="1e-14",
                  SOFTENING_PC="0", COLLISIONAL="true", R_NGB="1e-6", R_NGB_MAX="1e-6", PN_ORDER="0",
                  SSE_ENABLED="false", BINARY_E="0.5", PRINT_INTERVAL="1", PROFILING="false")
# The same binary through the HHS with ETA_FF/FB = 10 and no jerk step: the v/a criterion (ETA) sets the step
VA_BINARY = dict(FSI_BINARY, INTEGRATOR="hhs", TOTAL_TIME_MYR="1e-4", ETA_FF="10", ETA_FB="10", ETA_JERK="0")
# IC generator only (one 1e-7 Myr step): 2000 Kroupa stars without binaries, M = 1000 M_sun
IC_ONLY = dict(BINARIES, BINARY_FRAC="0", PLUMMER_M_TOTAL="1000", TOTAL_TIME_MYR="1e-7", DT_MAX_MYR="1e-7")
# Equal masses for the r_h check: the mass-weighted median radius of 2000 Kroupa stars scatters by ~4 % over seeds
IC_EQUAL = dict(IC_ONLY, PLUMMER_MASS_DIST="equal")
BIG_GROUP = dict(TEST_CASE="plummer", INTEGRATOR="hhs", TOTAL_TIME_MYR="1e-5", DT_MAX_MYR="1e-6", DT_MIN_MYR="1e-16",
                 SOFTENING_PC="0", COLLISIONAL="true", R_NGB="1e-3", R_NGB_MAX="0.1", PN_ORDER="0", SSE_ENABLED="false",
                 PLUMMER_N="16", PLUMMER_M_TOTAL="16", PLUMMER_R_HALF="2e-4", PLUMMER_SEED="3", PLUMMER_MASS_DIST="equal",
                 PRINT_INTERVAL="1", PROFILING="false")
# Natal kicks (runtime_sse.cpp): 500 12-Msun stars at Z = 0.02 in a cluster of r_h = 1 kpc (no dynamics to speak of),
# 1 Myr steps through their core collapse at 19-21 Myr: 500 NSs (Fryer 2012 rapid, f_fb ~ 0.025)
NATAL_NS = dict(TEST_CASE="plummer", INTEGRATOR="hhs", TOTAL_TIME_MYR="22", DT_MAX_MYR="1.0", DT_MIN_MYR="1e-14",
                SOFTENING_PC="0", COLLISIONAL="true", PN_ORDER="0", SSE_ENABLED="true", SSE_METALLICITY="0.02",
                SSE_REMNANT_MODEL="FRYER_RAPID", SSE_ECSN_MODEL="OFF", SSE_PPISN_MODEL="NONE", PLUMMER_N="500",
                PLUMMER_M_TOTAL="6000", PLUMMER_R_HALF="1000", PLUMMER_SEED="3", PLUMMER_MASS_DIST="equal",
                R_ESCAPE_RH="1000", PRINT_INTERVAL="1", PROFILING="false")
# 500 Kroupa masses scaled to <m> = 8 Msun (1.1-295 Msun) at Z = 2e-4 (FRYER_RAPID, ECSN, PPISN STRONG), 60 Myr:
# 89 NSs, 21 BHs with partial fallback, 6 direct-collapse BHs, 6 ECSN NSs
NATAL_MIX = dict(NATAL_NS, TOTAL_TIME_MYR="60", PLUMMER_MASS_DIST="kroupa", PLUMMER_M_TOTAL="4000",
                 PLUMMER_R_HALF="2000", SSE_METALLICITY="0.0002", SSE_ECSN_MODEL="ON", SSE_PPISN_MODEL="STRONG")


def run(exe, cfg):
    """L0 diagnostic rows (|dE/E|, |dP|/M, |dL/L|) of one run; raises on a nonzero exit."""
    return diagnostics(run_output(exe, cfg))


def run_output(exe, cfg, env=None, dump=False):
    """stdout of one run (env: extra environment variables); raises on a nonzero exit or a timeout. dump=True: also
    the particles of the last NBODY_DUMP snapshot (main.cpp dump_particles) as rows (pid, m, x, y, z, vx, vy, vz)
    sorted by pid, returned as (stdout, rows)."""
    with tempfile.TemporaryDirectory() as d:
        if dump:
            env = {**(env or {}), "NBODY_DUMP": "1"}
        # NBODY_TEST_EXTRA="KEY=value,KEY2=value" overrides keys of every case (A/B comparisons)
        extra = dict(kv.split("=", 1) for kv in os.environ.get("NBODY_TEST_EXTRA", "").split(",") if "=" in kv)
        Path(d, "config.txt").write_text("".join(f"{k} = {v}\n" for k, v in {**OLD_DEFAULTS, **cfg, **extra}.items()))
        try:
            launcher = os.environ.get("NBODY_TEST_LAUNCHER", "").split()  # e.g. "mpirun -n 2"
            out = subprocess.run(launcher + [exe], cwd=d, capture_output=True, text=True, timeout=TIMEOUT_S,
                                 env={**os.environ, **(env or {})})
        except subprocess.TimeoutExpired:
            raise SystemExit(f"FAIL: no result within {TIMEOUT_S} s (hung or stuck in time)")
        if out.returncode != 0:
            raise SystemExit(f"FAIL: exit {out.returncode}\n{out.stdout[-2000:]}{out.stderr[-2000:]}")
        if not dump:
            return out.stdout
        files = sorted(Path(d).glob("dump_*.bin"))
        if not files:
            raise SystemExit("FAIL: no NBODY_DUMP snapshot written")
        raw = files[-1].read_bytes()
        n = struct.unpack_from("i", raw)[0]
        rows = [struct.unpack_from("9d", raw, 4 + 72 * i)[:8] for i in range(n)]
        return out.stdout, sorted(rows)


def diagnostics(stdout):
    rows = [line.split() for line in stdout.splitlines() if re.match(r"^\s+L0\s", line) and len(line.split()) >= 8]
    if not rows:
        raise SystemExit("FAIL: no diagnostics printed")
    return [[float(r[3]), float(r[4]), float(r[5])] for r in rows]


def endpoint(exe, label, base, eta, bound, literature):
    """FROST's measure: the energy error at the last output (t = 1000 P), against our bound and the paper's."""
    rows = run(exe, {**base, "ETA": eta, "ETA_FF": eta, "ETA_FB": eta})
    ok = check(f"{label} eta={eta} |dE/E| at 1000 P", rows[-1][0], bound)
    return ok & check(f"{label} eta={eta} |dE/E| at 1000 P, FROST Fig. 4 upper end", rows[-1][0], literature)


def floor(exe, label, cfg, bounds):
    """Round-off floor at small eta: endpoint and max |dE/E|, max |dP|/M and |dL/L| against bounds."""
    rows = run(exe, cfg)
    vals = (rows[-1][0], max(r[0] for r in rows), max(r[1] for r in rows), max(r[2] for r in rows))
    names = ("|dE/E| at end", "max|dE/E|", "max|dP|/M [km/s]", "max|dL/L|")
    return all([check(f"{label} {n}", v, b) for n, v, b in zip(names, vals, bounds)])


def mergers(exe, label, cfg, bounds):
    """Stellar mergers inside the HHS recursion: the particle count must drop by exactly the recorded mergers
    (no duplicated or lost particles), momentum and energy including the merger sink must be conserved."""
    out = run_output(exe, cfg)
    rows = diagnostics(out)
    n0 = int(cfg["PLUMMER_N"])
    n_merged = int(re.findall(r"Merger tree written to \S+ \((\d+) events\)", out)[-1])
    last_table = out[out.rfind("Level        dt"):]
    n_final = sum(int(m) for m in re.findall(r"^\s+L\d+\s+\S+\s+(\d+)", last_table, re.M))
    ok = check(f"{label} mergers", n_merged, 50, lower=True)
    ok &= check(f"{label} |N_final - (N0 - mergers)|", abs(n_final - (n0 - n_merged)), 0)
    ok &= check(f"{label} max|dE/E| with merger sink", max(r[0] for r in rows), bounds[0])
    ok &= check(f"{label} max|dL/L| with merger sink", max(r[2] for r in rows), bounds[1])
    return ok & check(f"{label} max|dP|/M [km/s]", max(r[1] for r in rows), bounds[1])


def big_group(exe, label, cfg, bounds):
    """A group larger than LogH's storage must stay with the FSI: no crash, P conserved, E bounded."""
    out = run_output(exe, cfg)
    rows = diagnostics(out)
    ok = check(f"{label} groups left to the FSI", out.count("exceeds LOGH_MAX_BODIES"), 1, lower=True)
    ok &= check(f"{label} max|dE/E|", max(r[0] for r in rows), bounds[0])
    return ok & check(f"{label} max|dP|/M [km/s]", max(r[1] for r in rows), bounds[1])


def precession(exe, label, cfg, tol):
    """Periapsis precession measured by the binary tracker against the 1PN rate 6 pi G M / (c^2 a (1 - e^2))."""
    ratio = float(re.findall(r"ratio_raw=(\S+)", run_output(exe, cfg))[-1])
    return check(f"{label} |precession / 1PN - 1|", abs(ratio - 1.0), tol)


def spins(exe, label, cfg, chi):
    """Every subsystem member's |chi| printed by the subsystem diagnostics must equal the IC spin."""
    out = run_output(exe, cfg)
    rows = re.findall(r"^\s+\d+\s+BIN\s+2\s+(?:\S+\s+){5}(\S+)\s+(\S+)", out, re.M)
    worst = max((abs(float(c) - chi) for row in rows for c in row), default=float("inf"))
    return check(f"{label} max | |chi| - {chi} |", worst, 1e-3)


def spin_rows(exe, cfg):
    """(dchi, dJ/J, phi_S1, ratio_so) rows printed by the spin tracker."""
    rows = re.findall(r"Spin: dchi=(\S+)\s+dJ/J=(\S+)\s+phi_S1=(\S+)\s+ratio_so=(\S+)", run_output(exe, cfg))
    if not rows:
        raise SystemExit("FAIL: no spin tracker output")
    return [[float(x) for x in r] for r in rows]


def spin_precession(exe, label, cfg, tol, dchi_bound):
    """S1 precession angle about J against the orbit-averaged 1.5PN spin-orbit rate; |chi| invariant."""
    rows = spin_rows(exe, cfg)
    ok = check(f"{label} |precession / spin-orbit rate - 1|", abs(rows[-1][3] - 1.0), tol)
    return ok & check(f"{label} max | |chi| - |chi0| |", max(r[0] for r in rows), dchi_bound)


def spin_j(exe, label, cfg, dj_bound, dchi_bound):
    """Total J = L_N + L_PN1 + L_SO + S1 + S2 conserved; |chi| invariant."""
    rows = spin_rows(exe, cfg)
    ok = check(f"{label} max |dJ|/|J|", max(r[1] for r in rows), dj_bound)
    ok &= check(f"{label} S1 precessed [rad]", abs(rows[-1][2]), 1e-3, lower=True)
    return ok & check(f"{label} max | |chi| - |chi0| |", max(r[0] for r in rows), dchi_bound)


def replicas(exe, label, cfg):
    """Replicated MPI state must agree bitwise every step (the run stops with exit 3 otherwise)."""
    out = run_output(exe, cfg)
    return check(f"{label} replica mismatches", out.count("REPLICA MISMATCH"), 0) & \
        check(f"{label} steps run", len(diagnostics(out)), 20, lower=True)


def binaries(exe, label, cfg, bounds):
    """A cluster with primordial binaries must step at cluster time-steps (the binaries' internal motion is
    LogH's or the secular integrator's), finish in time and conserve P and E."""
    rows = run(exe, cfg)
    ok = check(f"{label} steps", len(rows), 2, lower=True)
    ok &= check(f"{label} max|dE/E|", max(r[0] for r in rows), bounds[0])
    ok &= check(f"{label} max|dL/L|", max(r[2] for r in rows), bounds[1])
    return ok & check(f"{label} max|dP|/M [km/s]", max(r[1] for r in rows), bounds[1])


def pn_books(exe, label, cfg, bounds):
    """A PN subsystem must not run into the energy restart (it cannot pass it) and the Newtonian diagnostics
    must close with the PN sink."""
    out = run_output(exe, cfg)
    rows = diagnostics(out)
    ok = check(f"{label} LogH give-ups", out.count("GAVE UP"), 0)
    ok &= check(f"{label} max|dE/E| with PN sink", max(r[0] for r in rows), bounds[0])
    ok &= check(f"{label} max|dL/L| with PN sink", max(r[2] for r in rows), bounds[0])
    return ok & check(f"{label} max|dP|/M [km/s]", max(r[1] for r in rows), bounds[1])


def sse_mass(exe, label, cfg, bound):
    """Runtime SSE after mergers: a remnant star must evolve as itself, so SSE removes only wind mass."""
    out = run_output(exe, cfg)
    dm = sum(float(v) for v in re.findall(r"SSE: dm_total=(\S+) M_sun", out))
    ok = check(f"{label} mergers", out.count("[MERGER #"), 20, lower=True)
    return ok & check(f"{label} mass removed by SSE [M_sun]", dm, bound)


def all_binaries(exe, label, cfg, n_bin):
    """Every primordial binary must become a subsystem (none dropped by a full neighbour buffer)."""
    out = run_output(exe, cfg)
    found = int(re.findall(r"Types:\s+bin=(\d+)", out)[0])
    rows = diagnostics(out)
    return check(f"{label} binaries detected", found, n_bin, lower=True) & \
        check(f"{label} max|dE/E|", max(r[0] for r in rows), 1e-8)


def cc_min_groups(exe, label, cfg, ratio):
    """HHS_CC_MIN: components below the size merge into one group, so the components per split drop to a few, and
    max |dE/E| stays within `ratio` of plain HHS."""
    def comps(out):
        line = [l for l in out.splitlines() if "HHS_CC components:" in l][-1]
        return sum(float(m[0]) * float(m[1]) for m in re.findall(r"d\d+ (\d+) x ([\d.]+)", line))
    plain = max(r[0] for r in run(exe, dict(cfg, HHS_CC="false")))
    out_cc, out_min = run_output(exe, dict(cfg, HHS_CC="true")), run_output(exe, dict(cfg, HHS_CC="true", HHS_CC_MIN="64"))
    n_cc, n_min = comps(out_cc), comps(out_min)
    dmin = max(r[0] for r in diagnostics(out_min))
    print(f"   components over the run: HHS_CC {n_cc:.0f}, HHS_CC_MIN = 64 {n_min:.0f}; max|dE/E| plain {plain:.3e}, MIN {dmin:.3e}")
    ok = check(f"{label} components HHS_CC_MIN / HHS_CC", n_min / n_cc, 0.6)   # measured 0.44
    return ok & check(f"{label} max|dE/E| HHS_CC_MIN / plain", dmin / plain, ratio)


def cc_vs_plain(exe, label, cfg, ratio):
    """HHS_CC against plain HHS on the same run: max |dE/E| at most `ratio` times the plain one."""
    plain = max(r[0] for r in run(exe, dict(cfg, HHS_CC="false")))
    cc = max(r[0] for r in run(exe, dict(cfg, HHS_CC="true")))
    print(f"   plain {plain:.3e}, HHS_CC {cc:.3e}")
    return check(f"{label} max|dE/E| HHS_CC / plain", cc / plain, ratio)


def paths_agree(exe, label, cfg, bound):
    """The host-subtree path (HOST_SUBTREE_MAX covers the whole run) against the device path on the same run: the
    same LogH calls and GBS steps per step, |dE/E| per step equal to `bound` (relative)."""
    def stats(out):
        return [(int(a), int(b)) for a, b in re.findall(r"Integration: LogH=(\d+) \(gbs=(\d+)", out)]
    dev_out = run_output(exe, dict(cfg, HOST_SUBTREE_MAX="0"))
    cpu_out = run_output(exe, dict(cfg, HOST_SUBTREE_MAX="1000000"))
    dev, cpu = diagnostics(dev_out), diagnostics(cpu_out)
    print(f"   LogH calls, GBS steps per step: device {stats(dev_out)}, cpu {stats(cpu_out)}")
    print(f"   |dE/E| device {[f'{r[0]:.6e}' for r in dev]}, cpu {[f'{r[0]:.6e}' for r in cpu]}")
    ok = check(f"{label} steps (device vs cpu)", abs(len(dev) - len(cpu)), 0)
    ok &= check(f"{label} steps with other LogH calls or GBS steps",
                sum(a != b for a, b in zip(stats(dev_out), stats(cpu_out))), 0)
    return ok & check(f"{label} max relative |dE/E| difference",
                      max(abs(a[0] - b[0]) / max(a[0], 1e-300) for a, b in zip(dev, cpu)), bound)
def force_err(exe, label, cfg, ratio):
    """NBODY_FORCE_ERR (synthetic far-field force errors, plan I-1): eps = 0 leaves the run bit-identical to no
    variable (diagnostics and every particle); eps = 1e-3 on every set of >= 256 particles must raise max |dE/E| by
    at least `ratio` in every error model."""
    out0, dump0 = run_output(exe, cfg, dump=True)
    out1, dump1 = run_output(exe, cfg, env={"NBODY_FORCE_ERR": "0", "NBODY_FORCE_ERR_MIN": "256"}, dump=True)
    ok = check(f"{label} eps = 0: diagnostics differ (0 = identical)", float(diagnostics(out0) != diagnostics(out1)), 0)
    ok &= check(f"{label} eps = 0: particles differ (0 = identical)", float(dump0 != dump1), 0)
    base = max(r[0] for r in diagnostics(out0))
    for model in ("random", "frozen", "smooth"):
        out = run_output(exe, cfg, env={"NBODY_FORCE_ERR": "1e-3", "NBODY_FORCE_ERR_MIN": "256",
                                        "NBODY_FORCE_ERR_MODEL": model})
        ok &= check(f"{label} {model}: injection announced", float("NBODY_FORCE_ERR: eps" in out), 1, lower=True)
        de = max(r[0] for r in diagnostics(out))
        print(f"   {model}: max|dE/E| {de:.3e} (no errors {base:.3e})")
        ok &= check(f"{label} {model} eps = 1e-3: max|dE/E| / unperturbed", de / base, ratio, lower=True)
    return ok


def host_subtree(exe, label, cfg, de_ratio, hs_max=64):
    """HOST_SUBTREE_MAX = hs_max: sets of up to hs_max particles run their whole sub-recursion on a host copy, with the device
    code. Host builds do the same arithmetic either way: bit-identical diagnostics and particles. GPU builds (host
    vs device rounding): max |dE/E| within de_ratio of the device run. NBODY_CHECK_HOSTSUB aborts on a device pointer
    met in a host subtree."""
    out0, dump0 = run_output(exe, {**cfg, "HOST_SUBTREE_MAX": "0"}, dump=True)
    out1, dump1 = run_output(exe, {**cfg, "HOST_SUBTREE_MAX": str(hs_max)}, env={"NBODY_CHECK_HOSTSUB": "1"}, dump=True)
    m = re.search(r"HOST_SUBTREE: (\d+) calls", out1)
    ok = check(f"{label}: host subtree calls", float(m.group(1)) if m else 0.0, 1, lower=True)
    if "KOKKOS_ENABLE_CUDA: yes" in out0 or "KOKKOS_ENABLE_HIP: yes" in out0:
        de0 = max(r[0] for r in diagnostics(out0))
        de1 = max(r[0] for r in diagnostics(out1))
        print(f"   max|dE/E| device {de0:.3e}, host subtrees {de1:.3e}")
        return ok & check(f"{label}: max|dE/E| host subtrees / device", de1 / de0, de_ratio)
    ok &= check(f"{label}: diagnostics differ (0 = identical)", float(diagnostics(out0) != diagnostics(out1)), 0)
    ok &= check(f"{label}: particles differ (0 = identical)", float(dump0 != dump1), 0)
    return ok


def paths_same_steps(exe, label, cfg, min_logh, gbs_rel=0.0):
    """The CPU small-set path against the device path: the same LogH calls per step, GBS steps per step equal to
    within `gbs_rel` (relative; the two paths sum forces in other orders, and the GBS step count of a LogH call can
    flip at round-off), and at least `min_logh` LogH calls in all."""
    def stats(out):
        return [(int(a), int(b)) for a, b in re.findall(r"Integration: LogH=(\d+) \(gbs=(\d+)", out)]
    dev = stats(run_output(exe, dict(cfg, HOST_SUBTREE_MAX="0")))
    cpu = stats(run_output(exe, dict(cfg, HOST_SUBTREE_MAX="1000000")))
    print(f"   LogH calls, GBS steps per step: device {dev}, cpu {cpu}")
    ok = check(f"{label} steps with other LogH calls (device vs cpu)",
               sum(a[0] != b[0] for a, b in zip(dev, cpu)) + abs(len(dev) - len(cpu)), 0)
    ok &= check(f"{label} largest relative GBS step difference (device vs cpu)",
                max([abs(a[1] - b[1]) / max(a[1], 1) for a, b in zip(dev, cpu)] + [0.0]), gbs_rel)
    return ok & check(f"{label} LogH calls", sum(a for a, _ in dev), min_logh, lower=True)



def flyby_group_floor(exe):
    """A star near an unbound fly-by subsystem: the tidal-sampling step of FLYBY_CM_EPS has the pericentre passage
    time of the subsystem's pairs as floor (without it 7.1M secular steps, 69 s, |dE/E| 6.4e-9 in one interval)."""
    cfg = dict(TEST_CASE="particle_file", IC_FILE=str(Path(__file__).parent / "data" / "flyby_group_r050.txt"),
               TOTAL_TIME_MYR="1e-3", DT_MAX_MYR="1e-3", DT_MIN_MYR="1.0e-14", ETA="0", ETA_GRAD="0.2", ETA_JERK="0",
               ETA_RESTART="1e-8", FLYBY_CM_EPS="1e-11", NGB_MASS_SCALING="true", SERIAL_N2_GPU="16384",
               TREE_THRESHOLD="100000", TREE_THETA="0.7", SDAR="true", COLLISIONAL="true", PN_ORDER="31",
               SSE_ENABLED="false", COLL_MASS_LOSS_C="0.3", PRINT_INTERVAL="1", PROFILING="false")
    out = run_output(exe, cfg)
    steps = sum(int(a) for a in re.findall(r"Secular=(\d+) \(substeps=", out))
    ok = check("fly-by group: secular steps", steps, FLYBY_GROUP_STEPS)
    return ok & check("fly-by group: max|dE/E|", max(r[0] for r in diagnostics(out)), FLYBY_GROUP_DE)


def pn_bh_only(exe):
    """PN_BH_ONLY (default): fewer LogH calls than the mass rule (false) on the same run, energy within bound."""
    def logh(out):
        return sum(int(a) for a in re.findall(r"Integration: LogH=(\d+) \(gbs=", out))
    cfg = dict(BINARIES, PN_ORDER="31")
    base = run_output(exe, dict(cfg, PN_BH_ONLY="false"))
    bh = run_output(exe, cfg)
    print(f"   LogH calls: mass rule {logh(base)}, BH only {logh(bh)}")
    ok = check("PN_BH_ONLY LogH calls below the mass rule's", logh(bh), logh(base) - 1)
    return ok & check("PN_BH_ONLY max|dE/E|", max(r[0] for r in diagnostics(bh)), PN_BH_ONLY_DE)

def cc_consistent(exe, label, cfg, bound):
    """HHS_CC must split the fast set with the pair time steps of the time-step kernel (NBODY_CHECK_CC reports a
    particle whose pair minimum in the split lies below its dt_adapt)."""
    os.environ["NBODY_CHECK_CC"] = "1"
    out = run_output(exe, dict(cfg, HHS_CC="true"))
    del os.environ["NBODY_CHECK_CC"]
    bad = sum(int(n) for n in re.findall(r"CC_CHECK: (\d+) of", out))
    ok = check(f"{label} particles with a split pair step below dt_adapt", bad, 0)
    return ok & check(f"{label} max|dE/E|", max(r[0] for r in diagnostics(out)), bound)


def ranks_agree(exe, label, cfg, bound):
    """The same run on 1 and 2 ranks: the energy series must agree to round-off (MPI-only paths)."""
    launcher = os.environ.get("NBODY_TEST_LAUNCHER", "")
    os.environ["NBODY_TEST_LAUNCHER"] = launcher.replace("-n 2", "-n 1")
    one = run(exe, cfg)
    os.environ["NBODY_TEST_LAUNCHER"] = launcher
    two = run(exe, cfg)
    ok = check(f"{label} rows (1 vs 2 ranks)", abs(len(one) - len(two)), 0)
    return ok & check(f"{label} max |dE/E(1) - dE/E(2)|", max(abs(a[0] - b[0]) for a, b in zip(one, two)), bound)


def rank_count_identical(exe, label, cfg):
    """SCALING_DESIGN.md sec. 6.1: the same run on 1 and 2 ranks must be bitwise identical (every particle dump and
    every printed L0 row), so results do not depend on the GPU count."""
    launcher = os.environ.get("NBODY_TEST_LAUNCHER", "")
    res = []
    for lch in (launcher.replace("-n 2", "-n 1"), launcher):
        os.environ["NBODY_TEST_LAUNCHER"] = lch
        with tempfile.TemporaryDirectory() as d:
            out = run_in(exe, cfg, d, env={"NBODY_DUMP": "1"})
            res.append((l0_by_step(out), {f.name: f.read_bytes() for f in sorted(Path(d).glob("dump_*.bin"))}))
    os.environ["NBODY_TEST_LAUNCHER"] = launcher
    (l1, d1), (l2, d2) = res
    ok = check(f"{label} L0 rows differing (1 vs 2 ranks)", sum(l1[s] != l2.get(s) for s in l1), 0)
    ok &= check(f"{label} dumps", len(d1), 2, lower=True)
    return ok & check(f"{label} particle dumps differing (1 vs 2 ranks)", sum(d1[k] != d2.get(k) for k in d1), 0)


def triples(exe, label, cfg, bound):
    """Triples among particles with large ids must be classified (inner binary, outer companion)."""
    out = run_output(exe, cfg)
    n3 = max((int(a) + int(b) for a, b in re.findall(r"tri_s=(\d+)\s+tri_u=(\d+)", out)), default=0)
    rows = diagnostics(out)
    return check(f"{label} triples formed", n3, 1, lower=True) & \
        check(f"{label} max|dE/E|", max(r[0] for r in rows), bound)


def triple_order(exe, label, cfg, rel):
    """A triple must evolve the same whichever particle ids its members carry: the outer body at the lowest
    id must not be taken as an inner-binary member (that pair is unbound and the triple was skipped, frozen)."""
    out = {of: run_output(exe, dict(cfg, TRIPLE_OUTER_FIRST=of)) for of in ("false", "true")}
    n_sec = {of: o.count("[SEC_TRI]") for of, o in out.items()}
    e = {of: diagnostics(o)[-1][0] for of, o in out.items()}
    ok = check(f"{label} secular-triple calls, outer body last", n_sec["false"], 1, lower=True)
    ok &= check(f"{label} secular-triple calls, outer body first", n_sec["true"], n_sec["false"], lower=True)
    ok &= check(f"{label} |dE/E| outer body first (evolves)", e["true"], 1e-30, lower=True)
    return ok & check(f"{label} |dE/E| first vs last, relative difference",
                      abs(e["true"] - e["false"]) / max(e["false"], 1e-300), rel)


def triple_stability(exe, label, cfg, cases):
    """Mardling & Aarseth (2001) as in BIFROST Sec. 3.11: R_p = a_out (1 - e_out) against
    2.8 [(1 + q_out)(1 + e_out) / sqrt(1 - e_out)]^(2/5) (1 - 0.3 i / pi) a_in. Each case (a_out / a_in, mutual
    inclination in degrees, stable?) must be classified so at detection (first step's type counts)."""
    ok = True
    a_in = float(cfg["TRIPLE_A_IN_PC"])
    for ratio, incl, stable in cases:
        out = run_output(exe, dict(cfg, TRIPLE_A_OUT_PC=f"{ratio * a_in:.6e}", TRIPLE_INCL_DEG=str(incl)))
        m = re.search(r"tri_s=(\d+)\s+tri_u=(\d+)", out)
        if not m:
            raise SystemExit("FAIL: no subsystem type counts printed")
        n_s, n_u = int(m.group(1)), int(m.group(2))
        want, other = (n_s, n_u) if stable else (n_u, n_s)
        name = f"{label} a_out/a_in={ratio} i={incl} deg {'stable' if stable else 'unstable'}"
        ok &= check(f"{name}: detections as such", want, 1, lower=True)
        ok &= check(f"{name}: detections as the other", other, 0)
    return ok


def secular_steps(exe, label, cfg, per_call):
    """Internal steps of the secular binary integrator per evolution: the step must not collapse when the
    argument of periapsis is near 0 (it used omega / omega-dot, and every call ran 100000 steps)."""
    out = run_output(exe, cfg)
    pairs = [(int(a), int(b)) for a, b in re.findall(r"Secular=(\d+) \(substeps=(\d+)\)", out)]
    n_ev, n_sub = sum(a for a, _ in pairs), sum(b for _, b in pairs)
    ok = check(f"{label} secular evolutions", n_ev, 1, lower=True)
    return ok & check(f"{label} internal steps per evolution", n_sub / max(n_ev, 1), per_call)


def secular_triples_off(exe, label, cfg, bound):
    """SECULAR_TRIPLES = false: a stable triple in the secular regime goes to LogH (FROST-CLUSTERS I dropped the
    double-averaged path); with the default it takes the secular path."""
    on, off = run_output(exe, cfg), run_output(exe, dict(cfg, SECULAR_TRIPLES="false"))
    ok = check(f"{label} secular-triple calls, default", on.count("[SEC_TRI]"), 1, lower=True)
    ok &= check(f"{label} secular-triple calls, SECULAR_TRIPLES = false", off.count("[SEC_TRI]"), 0)
    return ok & check(f"{label} |dE/E| in LogH", diagnostics(off)[-1][0], bound)


def logh_complete(exe, label, cfg, bound):
    """Every LogH call must reach its end time: no GBS step-limit hits, no retries, energy within bound."""
    out = run_output(exe, cfg)
    caps = sum(int(c) for c in re.findall(r"cap=(\d+)", out))
    ok = check(f"{label} LogH calls stopped by the GBS step limit", caps, 0)
    ok &= check(f"{label} LogH retries", out.count("LogH restart:"), 0)
    return ok & check(f"{label} |dE/E|", diagnostics(out)[-1][0], bound)


def flyby_member(exe, label, cfg, bound):
    """A star far outside r_ngb from a hard binary must not join it in a 3-body LogH subsystem: no unstable triples,
    no LogH calls, energy within bound."""
    out = run_output(exe, cfg)
    tri = sum(int(c) for c in re.findall(r"tri_u=(\d+)", out))
    logh = sum(int(c) for c in re.findall(r"Integration: LogH=(\d+)", out))
    ok = check(f"{label} unstable triples", tri, 0) & check(f"{label} LogH calls", logh, 0)
    return ok & check(f"{label} max|dE/E|", max(r[0] for r in diagnostics(out)), bound)


def sdar_stats(out):
    """(GBS steps, LogH step-limit hits, slowed LogH calls, Triple-tracker e_in series) of one run."""
    gbs = sum(int(c) for c in re.findall(r"\(gbs=(\d+)", out))
    caps = sum(int(c) for c in re.findall(r"cap=(\d+)", out))
    slowed = sum(int(c) for c in re.findall(r"sdar=(\d+)", out))
    e_in = [float(e) for e in re.findall(r"Triple: .*? e_in=([0-9.]+)", out)]
    return gbs, caps, slowed, e_in


def sdar_vs_logh(exe, label, cfg, steps_ratio, de_in, dE_bound, plain_env=None):
    """SDAR slow-down (Wang+2020) against plain LogH on the same subsystem: at most `steps_ratio` of the GBS steps,
    no step-limit hit, |dE/E| (slowed-Hamiltonian books, Wang+2020 Eq. 25) within dE_bound and not above LogH's, the
    inner eccentricity (secular evolution) within de_in of LogH's at every print. plain_env: environment of the
    LogH reference run."""
    plain, sd = run_output(exe, dict(cfg, SDAR="false"), plain_env), run_output(exe, dict(cfg, SDAR="true"))
    (g0, _, _, e0), (g1, c1, n1, e1) = sdar_stats(plain), sdar_stats(sd)
    dE0, dE1 = max(r[0] for r in diagnostics(plain)), max(r[0] for r in diagnostics(sd))
    ok = check(f"{label} slowed LogH calls", n1, 1, lower=True)
    ok &= check(f"{label} GBS steps SDAR / LogH ({g1} / {g0})", g1 / max(g0, 1), steps_ratio)
    ok &= check(f"{label} SDAR step-limit hits", c1, 0)
    ok &= check(f"{label} SDAR max|dE/E|", dE1, dE_bound)
    ok &= check(f"{label} SDAR max|dE/E| / LogH max|dE/E| ({dE0:.2e})", dE1 / dE0, 1.0)
    if de_in is not None:
        ok &= check(f"{label} prints", min(len(e0), len(e1)), 2, lower=True)
        ok &= check(f"{label} max|e_in(SDAR) - e_in(LogH)|", max(abs(a - b) for a, b in zip(e0, e1)), de_in)
    return ok


def sdar_pn_off(exe, label, cfg):
    """SDAR keeps kappa = 1 in PN subsystems: diagnostics identical to plain LogH, no slowed call; the same triple
    without PN is slowed (control)."""
    plain, sd = run_output(exe, dict(cfg, SDAR="false")), run_output(exe, dict(cfg, SDAR="true"))
    ok = check(f"{label} slowed LogH calls with PN", sdar_stats(sd)[2], 0)
    ok &= check(f"{label} diagnostics identical to LogH (0 = yes)", float(diagnostics(sd) != diagnostics(plain)), 0)
    ok &= check(f"{label} GBS steps identical to LogH (0 = yes)", float(sdar_stats(sd)[0] != sdar_stats(plain)[0]), 0)
    ctrl = run_output(exe, dict(cfg, SDAR="true", PN_ORDER="0"))
    return ok & check(f"{label} slowed LogH calls without PN", sdar_stats(ctrl)[2], 1, lower=True)


def tree_exact(exe, label, cfg, need_dt=False):
    """Exact tree time-step and neighbour searches (gravity/dt_tree.h), forced on at every set size: with
    NBODY_CHECK_TREE_DT=1 every tree search is compared bitwise with the O(N^2) kernel (the run aborts on a
    difference), and the diagnostics must equal those of the run without the tree to the last printed digit."""
    os.environ["NBODY_CHECK_TREE_DT"] = "1"
    out = run_output(exe, cfg)
    del os.environ["NBODY_CHECK_TREE_DT"]
    m = re.search(r"TREE_DT: dt calls=(\d+) sinks=\d+ pairs=(\d+) \(N\^2 (\d+)\) ngb calls=(\d+) pairs=\d+ checked=(\d+)", out)
    if not m:
        raise SystemExit("FAIL: no tree searches ran")
    dt_calls, pairs, n2_pairs, ngb_calls, checked = (int(g) for g in m.groups())
    print(f"   {dt_calls} time-step and {ngb_calls} neighbour searches, pair evaluations {pairs} of {n2_pairs} (N^2)")
    ref = run_output(exe, dict({k: v for k, v in cfg.items() if not k.startswith("TREE_")}, TREE_DT_MIN_N="0",
                                TREE_NGB_MIN_N="0", TREE_DT_CPU_MAX="0"))
    rows = [l for l in out.splitlines() if re.match(r"^\s+L0\s", l)]
    ref_rows = [l for l in ref.splitlines() if re.match(r"^\s+L0\s", l)]
    ok = check(f"{label} searches checked against N^2", checked, dt_calls + ngb_calls, lower=True)
    ok &= check(f"{label} searches run", dt_calls + ngb_calls, 1, lower=True)
    if need_dt:
        ok &= check(f"{label} time-step searches run", dt_calls, 1, lower=True)
    ok &= check(f"{label} diagnostic rows differing from the run without tree",
                sum(a != b for a, b in zip(rows, ref_rows)) + abs(len(rows) - len(ref_rows)), 0)
    return ok


def no_fsi_groups(exe, label, cfg, bound):
    """Neighbour chains in a dense core must be split along bound pairs, not left whole to the FSI."""
    out = run_output(exe, cfg)
    rows = diagnostics(out)
    return check(f"{label} groups left to the FSI", out.count("exceeds LOGH_MAX_BODIES"), 0) & \
        check(f"{label} max|dE/E|", max(r[0] for r in rows), bound)


def no_ic_mergers(exe, label, cfg, bound):
    """Primordial binaries must not start inside their contact distance: no mergers in the first step."""
    out = run_output(exe, cfg)
    rows = diagnostics(out)
    return check(f"{label} mergers in step 1", out.count("[MERGER #"), 0) & \
        check(f"{label} max|dE/E|", max(r[0] for r in rows), bound)

def ic_check(out):
    """(N_obj, measured r_h [pc], Q = T/|W| of singles and binary centres of mass, E_bin/|E_obj|) printed by the IC
    generator."""
    m = re.search(r"IC check: N_obj=(\d+) r_h\(measured\)=(\S+) pc Q_obj=(\S+) E_bin/\|E_obj\|=(\S+)", out)
    if not m:
        raise SystemExit("FAIL: no IC check line printed")
    return int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))


def ic_r_half(out):
    m = re.search(r"IC r_h = (\S+) pc", out)
    if not m:
        raise SystemExit("FAIL: no IC r_h line printed")
    return float(m.group(1))


def ic_mass_size(exe, label, cfg, bound):
    """BIFROST App. A (SCIC) eq. A2: without a user-given r_h, r_h / pc = 2.55 (M / 1e4 M_sun)^0.24 (Brown & Gnedin
    2021); the same with PLUMMER_R_HALF absent and = mass_size, and the sampled cluster must have it."""
    m_tot = float(cfg["PLUMMER_M_TOTAL"])
    want = 2.55 * (m_tot / 1e4) ** 0.24
    ok = True
    for name, c in (("PLUMMER_R_HALF = mass_size", dict(cfg, PLUMMER_R_HALF="mass_size")),
                    ("PLUMMER_R_HALF absent", {k: v for k, v in cfg.items() if k != "PLUMMER_R_HALF"})):
        out = run_output(exe, c)
        ok &= check(f"{label}, {name}: |r_h / eq. A2 - 1|", abs(ic_r_half(out) / want - 1), 1e-5)  # printed to 6 significant figures
        ok &= check(f"{label}, {name}: |measured r_h / eq. A2 - 1|", abs(ic_check(out)[1] / want - 1), bound)
    return ok


def ic_com_virial(exe, label, cfg, bound):
    """Binary centres of mass are sampled from the Plummer f(E) as single objects (BIFROST App. A, mcluster): the
    virial ratio of singles and centres of mass must be 1/2 within sampling noise (the deleted BINARY_COM_V = average,
    the mass-weighted mean of the two members' independent draws, gave 0.336: kinematically cold)."""
    q = ic_check(run_output(exe, cfg))[2]
    print(f"   Q_obj {q:.4f}")
    return check(f"{label} |Q_obj - 1/2|", abs(q - 0.5), bound)


def ic_r_break(exe, label, cfg, bound):
    """SCIC artificial-binary avoidance (BIFROST App. A): no two objects (singles, binary centres of mass) closer
    than r_break or bound with a < r_break after the re-sampling; angular re-sampling keeps the radial profile."""
    out = run_output(exe, cfg)
    m = re.search(r"IC_R_BREAK: r_break=(\S+) pc, close pairs before=(\d+) after=(\d+)", out)
    if not m:
        raise SystemExit("FAIL: no IC_R_BREAK line printed")
    before, after = int(m.group(2)), int(m.group(3))
    rh = ic_check(out)[1]
    rh0 = ic_check(run_output(exe, dict(cfg, IC_R_BREAK="0")))[1]
    ok = check(f"{label} close pairs before re-sampling", before, 1, lower=True)
    ok &= check(f"{label} close pairs after re-sampling", after, 0)
    return ok & check(f"{label} |r_h / r_h(IC_R_BREAK = 0) - 1|", abs(rh / rh0 - 1), bound)


def fsi_step(exe, label, cfg, bound):
    """The FSI integrator's global step (the smallest particle time step at the start) must stay at a sane level."""
    m = re.search(r"FSI global step: (\S+) Myr", run_output(exe, cfg))
    if not m:
        raise SystemExit("FAIL: no FSI global step printed")
    return check(f"{label} FSI global step [Myr]", float(m.group(1)), bound, lower=True)


def restarts(exe, label, cfg, bound):
    """Every step that fails ETA_RESTART must pass after restarts with smaller etas (no give-up)."""
    out = run_output(exe, cfg)
    rows = diagnostics(out)
    return check(f"{label} restarts", out.count("RESTART["), 1, lower=True) & \
        check(f"{label} give-ups", out.count("accepting"), 0) & \
        check(f"{label} max|dE/E|", max(r[0] for r in rows), bound)


def remnant_masses(exe, label, cfg, bh_min, ns_max, m_max):
    """Remnant populations of the IC stellar evolution: lowest BH mass and highest NS mass (Fryer 2012 rapid: the
    2-5 Msun mass gap), and the highest mass of any particle (stars above Hurley's 100 Msun limit stuck in core
    helium burning at ~R = 1350 Rsun whatever the age)."""
    out = run_output(exe, cfg)
    bh = re.findall(r"BH masses:\s+(\S+) - (\S+)", out)
    ns = re.findall(r"NS masses:\s+(\S+) - (\S+)", out)
    mr = re.findall(r"Mass range:\s+(\S+) - (\S+)", out)
    ok = check(f"{label} lowest BH mass [Msun]", float(bh[0][0]) if bh else 0.0, bh_min, lower=True)
    ok &= check(f"{label} highest particle mass [Msun]", float(mr[-1][1]) if mr else 1e30, m_max)
    return ok & check(f"{label} highest NS mass [Msun]", float(ns[0][1]) if ns else 1e30, ns_max)


def boost_invariance(exe, label, cfg, boosts, bound):
    """The relative orbit of an isolated binary must not depend on a uniform velocity of the whole system (G-1; PN
    terms in the subsystem frame, KETJU: Mannerkoski et al. 2023 sec. 2.2): |dr| / |r| at the end, boosted vs not."""
    def rel(b):
        _, rows = run_output(exe, dict(cfg, TEST_BOOST_KMS=b), dump=True)
        return binary_rel(rows)
    r0, _ = rel("0,0,0")
    n = lambda a: math.sqrt(sum(x * x for x in a))
    worst = 0.0
    for b in boosts:
        r1, _ = rel(b)
        worst = max(worst, n([x - y for x, y in zip(r1, r0)]) / n(r0))
    return check(f"{label} max |dr| / |r| over boosts {boosts}", worst, bound)


def wide_pairs(exe, label, cfg, bound):
    """NGB_WIDE_FAC: wide bound binaries just above r_ngb are subsystems, so no all-fast two-single set recurses to
    the deepest levels (NBODY_SLOW0_STATS), with the energy error of the plain run or better."""
    os.environ["NBODY_SLOW0_STATS"] = "1"
    out = run_output(exe, cfg)
    del os.environ["NBODY_SLOW0_STATS"]
    calls = sum(int(m) for m in re.findall(r"SLOW0_STATS: 2 0 2 \| (\d+)", out))
    ok = check(f"{label} all-fast two-single deep sets", calls, 0)
    return ok & check(f"{label} max|dE/E|", max(r[0] for r in diagnostics(out)), bound)


def cluster_diag(exe, label, cfg, n_bin_expected):
    """CLUSTER_DIAG_INTERVAL output at step 0: Lagrangian radii of the Plummer model against r(f) = a (f^(-2/3) -
    1)^(-1/2), a = r_h (2^(2/3) - 1)^(1/2), density centre near the origin, and every primordial binary found."""
    def first_row(c):
        with tempfile.TemporaryDirectory() as d:
            run_in(exe, dict(c, CLUSTER_DIAG_INTERVAL="1"), d)
            rows = [l.split() for l in Path(d, "cluster_diag.dat").read_text().splitlines() if not l.startswith("#")]
        return [float(x) for x in rows[0]]
    # structure from an equal-mass Plummer model (a Kroupa realisation's mass shells are dominated by a few heavy
    # stars: r_lag(90 %) 2.47 vs 2.84 pc at 20k), the binary census from the Kroupa model with binaries
    r0 = first_row(dict(cfg, PLUMMER_MASS_DIST="equal", BINARY_FRAC="0"))
    rb = first_row(cfg)
    a = float(cfg["PLUMMER_R_HALF"]) * math.sqrt(2.0 ** (2.0 / 3.0) - 1.0)
    fr = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9)
    worst = max(abs(r0[9 + k] / (a / math.sqrt(f ** (-2.0 / 3.0) - 1.0)) - 1.0) for k, f in enumerate(fr) if f >= 0.1)
    ok = check(f"{label} Lagrangian radii 10-90 %: max relative deviation from Plummer", worst, 0.05)
    ok &= check(f"{label} density centre offset / a", math.sqrt(r0[3] ** 2 + r0[4] ** 2 + r0[5] ** 2) / a, 0.05)
    ok &= check(f"{label} |n_bin - primordial| / primordial", abs(rb[16] - n_bin_expected) / n_bin_expected, 0.02)
    return ok


def tidal_orbit(exe, label, cfg, bound):
    """TIDAL_FIELD: the reference point of a cluster on a circular orbit in a logarithmic halo stays on the analytic
    circle X = R (cos Omega t, sin Omega t, 0), Omega = V_c / R."""
    out = run_output(exe, cfg)
    t = [l.split() for l in out.splitlines() if l.startswith("TIDAL step")][-1]
    X = [float(v) for v in t[4:7]]
    R, Vc = float(cfg["GALAXY_R_PC"]), float(cfg["GALAXY_VC_KMS"])
    tt = int(t[2]) * float(cfg["DT_MAX_MYR"]) / TU_MYR
    ana = [R * math.cos(Vc / R * tt), R * math.sin(Vc / R * tt), 0.0]
    return check(f"{label} |X - X_circle| / R after {t[2]} steps", math.dist(X, ana) / R, bound)


def tidal_books(exe, label, cfg, bound):
    """TIDAL_FIELD: with the tidal kicks booked (E_tidal), the energy check measures the integration error: a strong
    tide (r_J ~ r_h) gives the same |dE/E| level as no field, and the book itself is nonzero."""
    out = run_output(exe, cfg)
    et = [float(l.split()[-1]) for l in out.splitlines() if l.startswith("TIDAL step")]
    ok = check(f"{label} |E_tidal| booked (> 0)", abs(et[-1]) if et else 0.0, 1e-6, lower=True)
    return ok & check(f"{label} max|dE/E| with E_tidal", max(r[0] for r in diagnostics(out)), bound)


def no_runaway_depth(exe, label, cfg, max_level, bound):
    """A configuration that drove the field recursion tens of levels deep (or hung) must finish with every particle
    on a sane level and the energy conserved."""
    out = run_output(exe, cfg)
    lv = [int(m) for m in re.findall(r"^\s+L(\d+)\s+\S+\s+[1-9]", out, re.M)]
    ok = check(f"{label} deepest occupied HHS level", max(lv) if lv else 99, max_level)
    return ok & check(f"{label} max|dE/E|", max(r[0] for r in diagnostics(out)), bound)


def int_pos_vs_double(exe, label, cfg, ratio):
    """INTEGER_POSITIONS against double positions on the same run: max |dE/E| within `ratio` of the double one."""
    dbl = max(r[0] for r in run(exe, dict(cfg, INTEGER_POSITIONS="false")))
    ip = max(r[0] for r in run(exe, dict(cfg, INTEGER_POSITIONS="true")))
    print(f"   double {dbl:.3e}, integer {ip:.3e}")
    return check(f"{label} max|dE/E| integer / double positions", ip / dbl, ratio)


def particle_file_roundtrip(exe, cfg):
    """TEST_CASE = particle_file: the initial particles of `cfg` (dump at step 0) written to an IC_FILE and integrated
    again give the same final state bit for bit (reader, units and particle order)."""
    def dumps(c, d):
        Path(d, "config.txt").write_text("".join(f"{k} = {v}\n" for k, v in {**OLD_DEFAULTS, **c}.items()))
        out = subprocess.run([exe], cwd=d, capture_output=True, text=True, timeout=TIMEOUT_S,
                             env={**os.environ, "NBODY_DUMP": "1"})
        if out.returncode != 0:
            raise SystemExit(f"FAIL: exit {out.returncode}\n{out.stdout[-2000:]}{out.stderr[-2000:]}")
        rows = []
        for f in sorted(Path(d).glob("dump_*.bin")):
            raw = f.read_bytes()
            n = struct.unpack_from("i", raw)[0]
            rows.append(sorted(struct.unpack_from("9d", raw, 4 + 72 * i)[:8] for i in range(n)))
        return rows
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        ref = dumps(cfg, d1)
        ic = Path(d2, "ic.txt")
        ic.write_text("".join("%.17g %.17g %.17g %.17g %.17g %.17g %.17g\n" % tuple(r[1:8]) for r in ref[0]))
        got = dumps(dict(cfg, TEST_CASE="particle_file", IC_FILE=str(ic)), d2)
    ok = check("particle_file: particles read", len(got[0]), len(ref[0]), lower=True)
    ok &= check("particle_file: particles read", len(got[0]), len(ref[0]))
    diff = max(abs(a - b) for ra, rb in zip(ref[-1], got[-1]) for a, b in zip(ra[1:], rb[1:]))
    return ok & check("particle_file: |final state - reference| (bit-identical)", diff, 0.0)


def kepler_ae(p, q):
    """Osculating a (pc), e of two dump rows (pid, m, x, y, z, vx, vy, vz)."""
    r = [q[2 + k] - p[2 + k] for k in range(3)]
    v = [q[5 + k] - p[5 + k] for k in range(3)]
    mu = G_CODE * (p[1] + q[1])
    eps = 0.5 * sum(x * x for x in v) - mu / math.sqrt(sum(x * x for x in r))
    h = [r[1] * v[2] - r[2] * v[1], r[2] * v[0] - r[0] * v[2], r[0] * v[1] - r[1] * v[0]]
    return -mu / (2 * eps), math.sqrt(max(0.0, 1 + 2 * eps * sum(x * x for x in h) / mu ** 2))


def quad22_sdar(exe, label, cfg, bound):
    """PLAN item 4c: the 2+2 quadruple of the BIFROST Fig. 11 stall (PN BH binary + a 0.019 AU stellar binary bound to
    it). With SDAR per pair (NBODY_SDAR_PN=1) the stellar binary is slowed although the subsystem has PN members: the
    step finishes within the test timeout (0.4 s on the login node; without it no step in 120 s), the slow-down is
    used (kappa_max > 1) and the energy stays within `bound`. Step-limit retries of the eccentric BH binary are allowed
    (shorter pieces are cheaper than one long call, PLAN item 4 note)."""
    out, rows = run_output(exe, cfg, env={"NBODY_SDAR_PN": "1"}, dump=True)
    k = [float(x) for x in re.findall(r"kappa_max=(\S+?)\)", out)]
    ok = check(f"{label} largest kappa", max(k) if k else 0.0, 1e3, lower=True)
    ok &= check(f"{label} max|dE/E|", max(r[0] for r in diagnostics(out)), bound)
    # PN is a ~1e-8 effect here: the PN path with the slowed stellar binary must give its orbit as the Newtonian SDAR
    # path does. Measured |a / a_N - 1| 1e-7, |e - e_N| 1e-7; with the slowed pair's own PN term at full strength
    # (its precession x kappa per slowed orbit) a grew by 9 % and e by 0.04
    _, rows_n = run_output(exe, dict(cfg, PN_ORDER="0"), env={"NBODY_SDAR_PN": "1"}, dump=True)
    (a, e), (a_n, e_n) = (kepler_ae(r[2], r[3]) for r in (rows, rows_n))
    ok &= check(f"{label} stellar binary |a / a_Newtonian - 1|", abs(a / a_n - 1), 1e-5)
    return ok & check(f"{label} stellar binary |e - e_Newtonian|", abs(e - e_n), 1e-5)


def tidal_escapers(exe, label, cfg, bound):
    """TIDAL_FIELD + R_ESCAPE_RJ = 2 (NBODY6, Baumgardt & Makino 2003): stars beyond 2 r_J leave, r_J from the total
    mass at the first removal matches (G M R^2 / 2 V_c^2)^(1/3) (singular logarithmic halo), the energy with the
    escaper and tidal books stays conserved; with R_ESCAPE_RJ = 0 the isolated criterion (R_ESCAPE_RH) keeps them."""
    out = run_output(exe, cfg)
    ev = re.findall(r"ESCAPER: removed (\d+) particles.*r_J=(\S+), M=(\S+)", out)
    if not ev:
        return check(f"{label} escapers removed", 0, 1, lower=True)
    ok = check(f"{label} stars removed beyond 2 r_J", sum(int(e[0]) for e in ev), 1, lower=True)
    R, Vc = float(cfg["GALAXY_R_PC"]), float(cfg["GALAXY_VC_KMS"])
    rj = (G_CODE * float(ev[0][2]) * R * R / (2 * Vc * Vc)) ** (1 / 3)
    ok &= check(f"{label} |r_J / (G M R^2 / 2 V_c^2)^(1/3) - 1| (centre off the reference point)",
                abs(float(ev[0][1]) / rj - 1), 2e-3)
    ok &= check(f"{label} max|dE/E| with E_escape, E_tidal", max(r[0] for r in diagnostics(out)), bound)
    out = run_output(exe, dict(cfg, R_ESCAPE_RJ="0"))
    return ok & check(f"{label} R_ESCAPE_RJ = 0: removals (R_ESCAPE_RH = 100)", len(re.findall("ESCAPER: removed", out)), 0)


def grad_sweep(exe, label, cfg, etas, bounds, order):
    """max |dE/E| per ETA_GRAD and its order in ETA_GRAD (4: the step is eta_grad times a function of the state)."""
    ok, maxima = True, []
    for eta, bound in zip(etas, bounds):
        rows = run(exe, {**cfg, "ETA_GRAD": eta})
        maxima.append(max(r[0] for r in rows))
        ok &= check(f"{label} ETA_GRAD={eta} max|dE/E|", maxima[-1], bound)
    slope = math.log(maxima[0] / maxima[1]) / math.log(float(etas[0]) / float(etas[1]))
    ok &= check(f"{label} order of max|dE/E| in ETA_GRAD", slope, order[0], lower=True)
    return ok & check(f"{label} order of max|dE/E| in ETA_GRAD", slope, order[1])


def bh_merge_radius(exe, label, cfg, r_expect):
    """BH_MERGE_RADIUS: the collision distance of the ISCO event (printed r_coll) and r_sep < r_coll at the decision."""
    out = run_output(exe, cfg)
    m = re.findall(r"\[SUBSYS COLLISION\] ISCO .*?r_sep / r_coll = (\S+), r_coll = (\S+) pc", out)
    if not m:
        return check(f"{label} ISCO event", 0, 1, lower=True)
    ok = check(f"{label} |r_coll / r_expected - 1|", abs(float(m[0][1]) / r_expect - 1.0), 3e-3)   # G, c rounding
    return ok & check(f"{label} r_sep / r_coll at the decision", float(m[0][0]), 1.0)


def coll_mass_loss(exe, label, cfg, bound):
    """COLL_MASS_LOSS_C = 0.3 (FROST-CLUSTERS III, Glebbeek 2013): an equal-mass stellar collision keeps 92.5 % of
    m1 + m2, and the energy with the merger sink stays conserved (the lost mass is booked)."""
    out = run_output(exe, dict(cfg, COLL_MASS_LOSS_C="0.3"))
    m = re.findall(r"\[SUBSYS COLLISION\] STELLAR pids \d+ \+ \d+, m (\S+) \+ (\S+) -> (\S+) Msun", out)
    if not m:
        return check(f"{label} stellar collision", 0, 1, lower=True)
    m1, m2, mr = (float(x) for x in m[0])
    q = min(m1, m2) / max(m1, m2)
    ok = check(f"{label} |m_rem / ((m1 + m2)(1 - 0.3 q/(1+q)^2)) - 1|", abs(mr / ((m1 + m2) * (1 - 0.3 * q / (1 + q) ** 2)) - 1), 1e-3)
    return ok & check(f"{label} max|dE/E| with merger sink", max(r[0] for r in diagnostics(out)), bound)


SIGMA_NS, SIGMA_ECSN = 265.0, 3.0   # km/s: Hobbs et al. 2005; Banerjee et al. 2020 sec. 3.2 (Gessner & Janka 2018)


def natal_kicks_printed(out):
    """(pid, new kw, m, f_fb, ecsn, dv) of every "SSE natal kick" line"""
    pat = r"SSE natal kick: pid (\d+) kw \d+ -> (\d+) m (\S+) Msun f_fb (\S+) ecsn (\d) dv (\S+) (\S+) (\S+) km/s"
    return [(int(p), int(k), float(m), float(f), e == "1", [float(a), float(b), float(c)])
            for p, k, m, f, e, a, b, c in re.findall(pat, out)]


def maxwell_mean_check(label, vs, sigma):
    """Mean of Maxwellian samples (1D dispersion sigma) against 2 sigma sqrt(2/pi), within 3 standard errors"""
    if not vs:
        return check(f"{label} kicks", 0, 1, lower=True)
    mean, se = sum(vs) / len(vs), sigma * math.sqrt(3 - 8 / math.pi) / math.sqrt(len(vs))
    return check(f"{label}: |<v> - 2 sigma sqrt(2/pi)| [km/s] (n = {len(vs)}, bound 3 SE)",
                 abs(mean - 2 * sigma * math.sqrt(2 / math.pi)), 3 * se)


def natal_kicks_ns(exe, label, cfg, bound):
    """SSE_NATAL_KICKS = hobbs: every new NS gets an isotropic Maxwellian kick of sigma_NS (1 - f_fb) (Banerjee et
    al. 2020 eq. 1). 500 NSs: <|v|/(1 - f_fb)> = 2 sigma sqrt(2/pi) = 423 km/s, each component's dispersion sigma and
    the mean kick direction zero, all within 3 standard errors; E with the E_sse sink conserved. Before 2026-10-12:
    265 km/s along the current velocity (mean 265 km/s), and E_sse missed the potential energy on host builds (|dE/E|
    2 at the supernovae). SSE_NATAL_KICKS = none: no kicks."""
    out = run_output(exe, cfg)
    k = [x for x in natal_kicks_printed(out) if x[1] == 13 and not x[4]]
    ok = check(f"{label} NSs kicked", len(k), 450, lower=True)
    if not k:
        return False
    n = len(k)
    s = [[c / (1 - f) for c in dv] for _, _, _, f, _, dv in k]
    ok &= maxwell_mean_check(label, [math.sqrt(sum(c * c for c in v)) for v in s], SIGMA_NS)
    for i, ax in enumerate("xyz"):
        disp = math.sqrt(sum(v[i] ** 2 for v in s) / n)
        ok &= check(f"{label}: |sigma_{ax} - 265| [km/s] (bound 3 SE)", abs(disp - SIGMA_NS), 3 * SIGMA_NS / math.sqrt(2 * n))
        mean_dir = sum(v[i] / math.sqrt(sum(c * c for c in v)) for v in s) / n
        ok &= check(f"{label}: |<n_{ax}>| (isotropic; bound 3 / sqrt(3 n))", abs(mean_dir), 3 / math.sqrt(3 * n))
    ok &= check(f"{label} max|dE/E| with E_sse", max(r[0] for r in diagnostics(out)), bound)
    none = run_output(exe, dict(cfg, SSE_NATAL_KICKS="none"))
    ok &= check(f"{label} SSE_NATAL_KICKS = none: kicks", len(natal_kicks_printed(none)), 0)
    return ok & check(f"{label} SSE_NATAL_KICKS = none: NSs formed", none.count(" 500 type changes"), 2, lower=True)


def natal_kicks_fallback(exe, label, cfg, bound):
    """Fallback scaling (Banerjee et al. 2020 eq. 1; Fryer et al. 2012 rapid f_fb): direct-collapse BHs (f_fb = 1)
    exactly unkicked; BHs with partial fallback kicked; <|v|/(1 - f_fb)> of all non-ECSN NSs and BHs = 423 km/s
    within 3 standard errors; ECSN NSs from the 3 km/s Maxwellian; E with the E_sse sink conserved. Before
    2026-10-12: BHs 50 km/s (1 - (1 - m_rem/m_old)), direct collapse included, ECSN NSs 265 km/s."""
    out = run_output(exe, cfg)
    k = natal_kicks_printed(out)
    dc = [x for x in k if x[1] == 14 and x[3] >= 1.0]
    part = [x for x in k if x[1] == 14 and 0.0 < x[3] < 1.0 and not x[4]]
    ec = [x for x in k if x[4]]
    mag = lambda dv: math.sqrt(sum(c * c for c in dv))
    ok = check(f"{label} direct-collapse BHs", len(dc), 3, lower=True)
    ok &= check(f"{label} direct-collapse BHs kicked", sum(mag(x[5]) != 0.0 for x in dc), 0)
    ok &= check(f"{label} BHs with partial fallback, kicked", sum(mag(x[5]) > 0.0 for x in part), 10, lower=True)
    ok &= maxwell_mean_check(f"{label} NS+BH, |v|/(1 - f_fb)",
                             [mag(x[5]) / (1 - x[3]) for x in k if x[3] < 1.0 and not x[4]], SIGMA_NS)
    ok &= check(f"{label} ECSN NSs", len(ec), 3, lower=True)
    ok &= maxwell_mean_check(f"{label} ECSN", [mag(x[5]) for x in ec], SIGMA_ECSN)
    return ok & check(f"{label} max|dE/E| with E_sse", max(r[0] for r in diagnostics(out)), bound)


def tde_accretion(exe, label, cfg):
    """TDE remnant mass: m_BH + TDE_ACCRETION_FRACTION x m_star (FROST-CLUSTERS I, DRAGON-II: 0.5) for f = 0, 0.5, 1."""
    ok = True
    for f in ("0", "0.5", "1"):
        out = run_output(exe, dict(cfg, TDE_ACCRETION_FRACTION=f))
        m = re.findall(r"\[SUBSYS COLLISION\] TDE pids \d+ \+ \d+, m (\S+) \+ (\S+) -> (\S+) Msun", out)
        if not m:
            ok &= check(f"{label} f = {f}: TDE event", 0, 1, lower=True)
            continue
        m_bh, m_st, m_rem = (float(x) for x in m[0])
        ok &= check(f"{label} f = {f}: |m_rem - (m_BH + f m_star)|", abs(m_rem - (max(m_bh, m_st) + float(f) * min(m_bh, m_st))),
                    1e-3)
    return ok


def subsys_collision(exe, label, cfg, kind, bound, min_landings=0, min_kepler2=0):

    """A collision inside a subsystem integration (FROST-CLUSTERS I App. B3): exactly one event of the given kind,
    one body less at the end (no duplicated or lost particle), no LogH step-limit hit or give-up, and the energy
    with the merger sink conserved to the integration error. min_kepler2: drifts of the pair on the Kepler path
    (non-PN pairs) before it was handed to LogH."""
    out = run_output(exe, cfg)
    rows = diagnostics(out)
    n0 = {"triple_test": 3, "binary_test": 2}[cfg["TEST_CASE"]]
    last_table = out[out.rfind("Level        dt"):]
    n_final = sum(int(m) for m in re.findall(r"^\s+L\d+\s+\S+\s+(\d+)", last_table, re.M))
    landings = sum(int(m) for m in re.findall(r"pericentre landings (\d+)", out))
    ok = check(f"{label} [SUBSYS COLLISION] {kind} events", out.count(f"[SUBSYS COLLISION] {kind} "), 1, lower=True)
    ok &= check(f"{label} collision events", out.count("[SUBSYS COLLISION]"), 1)
    ok &= check(f"{label} |N_final - (N0 - 1)|", abs(n_final - (n0 - 1)), 0)
    ok &= check(f"{label} LogH step-limit hits and give-ups", out.count("step limit") + out.count("GAVE UP"), 0)
    ok &= check(f"{label} pericentre landings", landings, min_landings, lower=True)
    if min_kepler2:
        ok &= check(f"{label} Kepler2 drifts", sum(int(m) for m in re.findall(r"Kepler2=(\d+)", out)), min_kepler2,
                    lower=True)
    return ok & check(f"{label} max|dE/E| with merger sink", max(r[0] for r in rows), bound)


def run_in(exe, cfg, d, args=(), env=None):
    """stdout of one run in directory d (kept, unlike run_output); raises on a nonzero exit or a timeout."""
    extra = dict(kv.split("=", 1) for kv in os.environ.get("NBODY_TEST_EXTRA", "").split(",") if "=" in kv)
    Path(d, "config.txt").write_text("".join(f"{k} = {v}\n" for k, v in {**OLD_DEFAULTS, **cfg, **extra}.items()))
    launcher = os.environ.get("NBODY_TEST_LAUNCHER", "").split()
    try:
        out = subprocess.run(launcher + [exe, *args], cwd=d, capture_output=True, text=True, timeout=TIMEOUT_S,
                             env={**os.environ, **(env or {})})
    except subprocess.TimeoutExpired:
        raise SystemExit(f"FAIL: no result within {TIMEOUT_S} s (hung or stuck in time)")
    if out.returncode != 0:
        raise SystemExit(f"FAIL: exit {out.returncode}\n{out.stdout[-3000:]}{out.stderr[-3000:]}")
    return out.stdout


def l0_by_step(stdout):
    """L0 diagnostic rows as printed (|dE/E| |dP| |dL/L| t step), keyed by step (the first printed row wins: the
    final summary repeats the last step after the escaper removal, which changes |dP|/M)."""
    rows = {}
    for line in stdout.splitlines():
        f = line.split()
        if re.match(r"^\s+L0\s", line) and len(f) >= 8:
            rows.setdefault(int(f[-1]), " ".join(f[3:]))
    return rows


def restart_identical(exe, label, cfg, n, stop="interval", snapshots=False):
    """A run of 2n steps against n steps + restart file + n steps (stop = "walltime": the first run stops
    after one step at WALLTIME_LIMIT and the second continues from there): every L0 row and every NBODY_DUMP
    particle dump (the full state in storage order) after the restart point must be identical, bit for bit."""
    import hashlib
    dt = float(cfg["DT_MAX_MYR"])
    base = dict(cfg, PRINT_INTERVAL="1")
    full = dict(base, TOTAL_TIME_MYR=f"{(2 * n - 0.5) * dt:.17g}")
    env = {"NBODY_DUMP": "1"}
    if snapshots:
        full["SNAPSHOT_INTERVAL"] = "1"
    with tempfile.TemporaryDirectory() as da, tempfile.TemporaryDirectory() as db:
        ref = run_in(exe, full, da, env=env)
        if stop == "walltime":
            first = dict(full, WALLTIME_LIMIT="1e-9")      # stops after step 1, writes restart_00000001.h5
            k = 1
        else:
            first = dict(full, TOTAL_TIME_MYR=f"{(n - 0.5) * dt:.17g}", RESTART_INTERVAL=str(n))
            k = n
        out1 = run_in(exe, first, db, env=env)
        files = sorted(Path(db).glob("restart_*.h5"))
        ok = check(f"{label} restart file at step {k} written", int(Path(db, f"restart_{k:08d}.h5").exists()), 1,
                   lower=True)
        if not ok:
            print(out1[-2000:])
            return False
        out2 = run_in(exe, full, db, args=("--restart", f"restart_{k:08d}.h5"), env=env)
        ra, r1, r2 = l0_by_step(ref), l0_by_step(out1), l0_by_step(out2)
        steps = sorted(ra)
        resumed = {s: r for s, r in r2.items() if s > k}
        diff = [s for s in steps if s > k and resumed.get(s) != ra[s]] + [s for s in steps if s <= k and r1.get(s) != ra[s]]
        print(f"   restart at step {k} of {max(steps)}; {len(resumed)} rows after the restart, restart files kept: "
              f"{[f.name for f in files]}")
        for s in diff[:3]:
            print(f"   step {s}: uninterrupted '{ra[s]}' restarted '{resumed.get(s, r1.get(s))}'")
        ok &= check(f"{label} steps after the restart", len(resumed), 2 * n - k, lower=True)
        ok &= check(f"{label} L0 rows differing from the uninterrupted run", len(diff), 0)
        # Run constants fixed at the start (NGB_MASS_SCALING <m>) must be restored, not recomputed from the state
        mref = lambda out: re.findall(r"NGB_MASS_SCALING: <m> = (\S+)", out)
        if mref(ref):
            ok &= check(f"{label} NGB_MASS_SCALING <m> restored (0 = yes)", int(mref(out2) != mref(ref)[:1]), 0)

        def digest(d, s):
            p = Path(d, f"dump_{s:04d}.bin")
            return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        dumps = [s for s in steps if s > k]
        bad = [s for s in dumps if digest(da, s) is None or digest(da, s) != digest(db, s)]
        ok &= check(f"{label} particle dumps (steps {k + 1}-{max(steps)}) differing", len(bad), 0)
        if snapshots:
            # snapshots of the restarted run: same numbering and identical particle data as the uninterrupted one
            sa = sorted(Path(da).glob("snapdir_*/*.hdf5"))
            sb = sorted(Path(db).glob("snapdir_*/*.hdf5"))
            ok &= check(f"{label} snapshot files (uninterrupted vs restarted)", abs(len(sa) - len(sb)), 0)
        return ok


def snapshot_roundtrip(exe, label, cfg):
    """Snapshot and restart file of the same state (the end of the run), read back by h5-roundtrip
    (tests/h5_roundtrip.cpp): positions, velocities, masses, ids, stellar types, radii and spins in the snapshot
    equal the restart file's bit for bit, every particle is in its PartType group, the LUNA groups are complete."""
    with tempfile.TemporaryDirectory() as d:
        run_in(exe, dict(cfg, SNAPSHOT_INTERVAL="1000000", RESTART_INTERVAL="1000000"), d)
        snaps = sorted(Path(d).glob("snapdir_*/snapshot_*.0.hdf5"))
        rsts = sorted(Path(d).glob("restart_*.h5"))
        ok = check(f"{label} snapshots written (start and end)", len(snaps), 2, lower=True)
        ok &= check(f"{label} restart files written", len(rsts), 1, lower=True)
        if not ok:
            return False
        tool = Path(exe).with_name("h5-roundtrip")
        res = subprocess.run([str(tool), str(snaps[-1]), str(rsts[-1])], capture_output=True, text=True)
        print("   " + res.stdout.rstrip().replace("\n", "\n   "))
        if res.returncode != 0:
            print(res.stderr[-2000:])
        return ok & check(f"{label} snapshot vs restart file (h5-roundtrip exit code)", res.returncode, 0)


def snapshot_cluster_header(exe, label, cfg):
    """The snapshot Header carries the cluster structure (SNAPSHOT_CLUSTER_DIAG) and the tidal reference: at the
    final step the Cluster_* attributes equal the CLUSTER_DIAG_FILE row of the same step, Tidal_RefPosition the
    last TIDAL line, Tidal_JacobiRadius_pc = (G M R^2 / 2 V_c^2)^(1/3) at the reference point + density centre."""
    with tempfile.TemporaryDirectory() as d:
        out = run_in(exe, dict(cfg, SNAPSHOT_INTERVAL="1000000", RESTART_INTERVAL="1000000", CLUSTER_DIAG_INTERVAL="1"), d)
        snaps = sorted(Path(d).glob("snapdir_*/snapshot_*.0.hdf5"))
        rsts = sorted(Path(d).glob("restart_*.h5"))
        rows = [l.split() for l in Path(d, "cluster_diag.dat").read_text().splitlines() if not l.startswith("#")]
        if not snaps or not rsts or not rows:
            return check(f"{label} snapshot, restart file and cluster_diag.dat written", 0, 1, lower=True)
        res = subprocess.run([str(Path(exe).with_name("h5-roundtrip")), str(snaps[-1]), str(rsts[-1])],
                             capture_output=True, text=True)
        hdr = {f[1]: [float(v) for v in f[2:]] for f in (l.split() for l in res.stdout.splitlines()) if f[0] == "header"}
    row = [float(v) for v in rows[-1]]
    need = ("Cluster_MassTotal", "Cluster_CoreRadius_pc", "Cluster_LagrangianRadii_pc", "Cluster_NBinaries",
            "Tidal_RefPosition", "Tidal_JacobiRadius_pc", "E_tidal")
    missing = [k for k in need if k not in hdr]
    ok = check(f"{label} Header attributes missing", len(missing), 0)
    if missing:
        print("   missing:", missing, res.stdout[-1500:])
        return False
    rel = lambda a, b: abs(a - b) / max(abs(b), 1e-300)
    # cluster_diag.dat: step t M cx cy cz r_core rho_core sigma_c r_lag[7] n_bin n_bin_hard E_bin (%.9e)
    dev = max([rel(hdr["Cluster_MassTotal"][0], row[2]), rel(hdr["Cluster_CoreRadius_pc"][0], row[6])] +
              [rel(a, b) for a, b in zip(hdr["Cluster_LagrangianRadii_pc"], row[9:16])])
    ok &= check(f"{label} Header vs cluster_diag.dat row of step {int(row[0])} (max rel. dev.)", dev, 1e-8)
    ok &= check(f"{label} Header vs cluster_diag.dat binaries", abs(hdr["Cluster_NBinaries"][0] - row[16]), 0)
    t = [l.split() for l in out.splitlines() if l.startswith("TIDAL step")][-1]
    X = hdr["Tidal_RefPosition"]
    ok &= check(f"{label} Tidal_RefPosition vs TIDAL line", max(rel(a, float(b)) for a, b in zip(X, t[4:7])), 1e-8)
    Xg = [X[k] + row[3 + k] for k in range(3)]
    R2, Vc = sum(v * v for v in Xg), float(cfg["GALAXY_VC_KMS"])
    rj = (G_CODE * row[2] * R2 / (2 * Vc * Vc)) ** (1 / 3)
    return ok & check(f"{label} Tidal_JacobiRadius_pc / (G M R^2 / 2 V_c^2)^(1/3) - 1", rel(hdr["Tidal_JacobiRadius_pc"][0], rj), 1e-6)


def luna_reader(exe, label, cfg):
    """A cpp-nbody snapshot read through LUNA's own read_snapshot (src/luna/file_io/read_snapshot.py) and its
    Gadget-style particle access. Needs LUNA_PYTHON (a Python with h5py where `import luna` works) or skips."""
    py = os.environ.get("LUNA_PYTHON", "")
    if not py or not Path(py).exists():
        print(f"skip {label}: set LUNA_PYTHON to a Python in which LUNA imports (e.g. /u/uli/jax-env/bin/python)")
        return "skip"
    with tempfile.TemporaryDirectory() as d:
        run_in(exe, {"SNAPSHOT_INTERVAL": "1000000", **cfg}, d)
        res = subprocess.run([py, str(Path(__file__).with_name("luna_read_check.py")), d], capture_output=True,
                             text=True, env={**os.environ, "JAX_PLATFORMS": "cpu"})
        print(res.stdout.rstrip())
        if res.returncode != 0:
            print(res.stderr[-3000:])
        return check(f"{label} LUNA reader failures", res.returncode, 0)
def binary_rel(rows):
    """Relative position and velocity (second body - first) of a two-body dump."""
    a, b = rows[0], rows[1]
    return [b[k] - a[k] for k in (2, 3, 4)], [b[k] - a[k] for k in (5, 6, 7)]


def binary_test_reference(cfg, t_myr):
    """Analytic relative position of the binary_test orbit at t: it starts at apocentre, r = (-a (1 + e), 0, 0),
    counter-clockwise, so r(t) = (a (cos E - e), a sqrt(1 - e^2) sin E, 0) with E - e sin E = pi + n t."""
    a, e = float(cfg["BINARY_A_PC"]), float(cfg["BINARY_E"])
    mu = G_CODE * (float(cfg["BINARY_M1"]) + float(cfg["BINARY_M2"]))
    M = math.fmod(math.pi + math.sqrt(mu / a ** 3) * t_myr / TU_MYR, 2.0 * math.pi)
    E = M
    for _ in range(50):
        E -= (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
    return [a * (math.cos(E) - e), a * math.sqrt(1.0 - e * e) * math.sin(E), 0.0]


def dist3(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def binary_test_reference_hyp(cfg, t_myr):
    """Analytic relative position of the hyperbolic (e > 1) binary_test encounter at t: BINARY_A_PC
    is the pericentre distance q, the start is inbound at r0 = 5 q, pericentre along +x; r = (A (e - cosh F),
    A sqrt(e^2 - 1) sinh F, 0) with e sinh F - F = M0 + n t, A = q / (e - 1), n = sqrt(mu / A^3)."""
    q, e = float(cfg["BINARY_A_PC"]), float(cfg["BINARY_E"])
    mu = G_CODE * (float(cfg["BINARY_M1"]) + float(cfg["BINARY_M2"]))
    p, r0, A = q * (1.0 + e), 5.0 * q, q / (e - 1.0)
    f0 = -math.acos(max(-1.0, min(1.0, (p / r0 - 1.0) / e)))
    F0 = 2.0 * math.atanh(math.sqrt((e - 1.0) / (e + 1.0)) * math.tan(0.5 * f0))
    M = e * math.sinh(F0) - F0 + math.sqrt(mu / A ** 3) * t_myr / TU_MYR
    F = math.asinh(M / e)
    for _ in range(100):
        F -= (e * math.sinh(F) - F - M) / (e * math.cosh(F) - 1.0)
    return [A * (e - math.cosh(F)), A * math.sqrt(e * e - 1.0) * math.sinh(F), 0.0]


def kepler_exact(exe, label, cfg, bounds):
    """An isolated non-PN 2-body subsystem below N_ORB_SECULAR orbits per step (or unbound) on the analytic Kepler
    drift: no LogH call, and the orbit agrees with the analytic one. bounds: (Kepler vs analytic, max |dE/E|)."""
    total = float(cfg["TOTAL_TIME_MYR"])
    run_cfg = dict(cfg, TOTAL_TIME_MYR=f"{total * (1 - 1e-9):.15e}")   # stop after exactly n DT_MAX
    out_k, rows = run_output(exe, run_cfg, env={"NBODY_GBS_TTOL": "1e-12"}, dump=True)
    hyp = float(cfg["BINARY_E"]) > 1.0
    ref = binary_test_reference_hyp(cfg, total) if hyp else binary_test_reference(cfg, total)
    scale = math.sqrt(sum(x * x for x in ref))
    n_kep = sum(int(m) for m in re.findall(r"Kepler2=(\d+)", out_k))
    ok = check(f"{label} Kepler run: Kepler2 drifts", n_kep, 1, lower=True)
    ok &= check(f"{label} Kepler run: LogH calls", sum(int(m) for m in re.findall(r"Integration: LogH=(\d+)", out_k)), 0)
    ok &= check(f"{label} |r_Kepler - r_analytic| / r", dist3(binary_rel(rows)[0], ref) / scale, bounds[0])
    return ok & check(f"{label} Kepler run max|dE/E|", max(x[0] for x in diagnostics(out_k)), bounds[1])


def secular_drift(exe, label, cfg, bounds):
    """An isolated binary on the secular (Kepler-drift) path: one step of T against 16 steps of T / 16 (the exact flow
    composes, so they agree to round-off), both against the analytic orbit; the two-body energy and angular momentum
    are conserved to round-off. bounds: (1 vs 16, vs analytic, max |dE/E|, max |dL/L|)."""
    total = float(cfg["TOTAL_TIME_MYR"])
    # the run stops at the first t >= TOTAL_TIME, and n additions of DT_MAX may fall short of n DT_MAX by round-off
    runs = {n: run_output(exe, dict(cfg, DT_MAX_MYR=f"{total / n:.15e}", TOTAL_TIME_MYR=f"{total * (1 - 1e-9):.15e}"),
                          dump=True) for n in (1, 16)}
    ref = binary_test_reference(cfg, total)
    r = {n: binary_rel(rows)[0] for n, (_, rows) in runs.items()}
    scale = math.sqrt(sum(x * x for x in ref))
    ok = True
    for n, (out, _) in runs.items():
        n_sec = sum(int(m) for m in re.findall(r"Secular=(\d+)", out))
        n_logh = sum(int(m) for m in re.findall(r"Integration: LogH=(\d+)", out))
        ok &= check(f"{label} {n} step(s): secular evolutions", n_sec, n, lower=True)
        ok &= check(f"{label} {n} step(s): LogH calls", n_logh, 0)
        ok &= check(f"{label} {n} step(s): |r - r_analytic| / r", dist3(r[n], ref) / scale, bounds[1])
        rows = diagnostics(out)
        ok &= check(f"{label} {n} step(s): max|dE/E|", max(x[0] for x in rows), bounds[2])
        ok &= check(f"{label} {n} step(s): max|dL/L|", max(x[2] for x in rows), bounds[3])
    return ok & check(f"{label} |r(1 step) - r(16 steps)| / r", dist3(r[1], r[16]) / scale, bounds[0])


def kepler_unit(exe, bounds):
    """tests/kepler_test.cpp: worst errors of kepler_advance per orbit class against bounds (1v16, rt, ref, dE, dL)."""
    out = subprocess.run([exe], capture_output=True, text=True, timeout=TIMEOUT_S).stdout
    print(out, end="")
    ok = True
    for cls, bnd in bounds.items():
        m = re.search(rf"KEPTEST {re.escape(cls)} 1v16=(\S+) rt=(\S+) ref=(\S+) dE=(\S+) dL=(\S+)", out)
        if not m:
            raise SystemExit(f"FAIL: no KEPTEST line for {cls}")
        for name, val, b in zip(("1v16", "rt", "ref", "dE", "dL"), m.groups(), bnd):
            if b is not None:
                ok &= check(f"{cls} {name}", float(val), b)
    return ok


def check(name, value, bound, lower=False):
    ok = value >= bound if lower else value <= bound
    print(f"{'ok  ' if ok else 'FAIL'} {name} = {value:.3e} ({'>=' if lower else '<='} {bound:.3e})")
    return ok


def sweep(exe, label, base, etas, bounds, order, drift_max=None):
    """max |dE/E| per eta and its fitted order, optionally a bounded (non-drifting) energy error, P and L at
    round-off."""
    ok, maxima = True, []
    for eta, bound in zip(etas, bounds):
        rows = run(exe, {**base, "ETA": eta, "ETA_FF": eta, "ETA_FB": eta})
        dE = [r[0] for r in rows]
        n = len(dE)
        maxima.append(max(dE))
        ok &= check(f"{label} eta={eta} max|dE/E|", max(dE), bound)
        if drift_max is not None:
            ok &= check(f"{label} eta={eta} drift (last / first fifth)", max(dE[4 * n // 5:]) / max(dE[: n // 5]),
                        drift_max)
        ok &= check(f"{label} eta={eta} max|dP|/M [km/s]", max(r[1] for r in rows), 1e-14)
        ok &= check(f"{label} eta={eta} max|dL/L|", max(r[2] for r in rows), 1e-12)
    if order is not None:
        x, y = [math.log(float(v)) for v in etas], [math.log(v) for v in maxima]
        mx, my = sum(x) / len(x), sum(y) / len(y)
        slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x)
        ok &= check(f"{label} order of max|dE/E| in eta", slope, order[0], lower=True)
        ok &= check(f"{label} order of max|dE/E| in eta", slope, order[1])
    return ok


CASES = {
    # bounds ~1.3x measured with the all-slow split (FROST Alg. 3, hard-wired): 3.24e-7, 8.98e-9, 5.24e-10, order
    # 4.64 -0.1/+0.3 (without the split 5.6e-6, 1.2e-7, 8.3e-9, order 4.70); drift 1.1-1.2 (5.0 before the
    # symmetrised-step fix, which also gave 4.6e-3 at eta = 0.064)
    "kepler_e09": lambda exe: sweep(exe, "e=0.9", dict(KEPLER, BINARY_E="0.9"), ["0.064", "0.032", "0.016"],
                                    [4.2e-7, 1.2e-8, 6.8e-10], (4.54, 4.94), 1.5),
    # circular orbit: the eta^4 term cancels (constant along the orbit), the error sits at round-off
    "kepler_e0": lambda exe: sweep(exe, "e=0", dict(KEPLER, BINARY_E="0.0"), ["0.064"], [1e-12], None),
    # FROST: "minimum relative energy error 1e-12 - 1e-10 at eta ~ 0.008" for e = 0.9, 0.99. Measured with the
    # split 1.76e-13 and 1.44e-11 (gcc/x86-64; e = 0.9 now below FROST's range); without it 1.43e-12 / 6.90e-11
    # (2.37e-12 / 5.09e-11 clang/arm64). GPU builds need 1/sqrt or RSQRT_NEWTON for e = 0.99 (rsqrt: 1.7e-10).
    # e = 0.9 on the MI300A (RSQRT_NEWTON): 1.52e-12 (the endpoint samples an oscillating error); bound 1.3x that
    "frost4_e09": lambda exe: endpoint(exe, "e=0.9", dict(FROST4, BINARY_E="0.9"), "0.008", 2.0e-12, 1e-10),
    "frost4_e099": lambda exe: endpoint(exe, "e=0.99", dict(FROST4, BINARY_E="0.99"), "0.008", 1.9e-11, 1e-10),
    # Through the CPU small-subset path, which kept the pre-fix FROST Eq. 49 (no eta): 1.5e-6 before, 9.1e-13
    # after (1.33e-12 with the split)
    "frost4_e09_cpu": lambda exe: endpoint(exe, "e=0.9 cpu path", dict(FROST4, BINARY_E="0.9", HOST_SUBTREE_MAX="8"),
                                           "0.008", 1.8e-12, 1e-10),
    # Round-off floor with compensated kicks and integer positions (domain 1e-3 pc), eta = 0.004; 1.3x
    # measured. Binary e = 0: 1.8e-15 end, 6.1e-15 max, L 3e-15 on clang/arm64; 1.07e-14, 1.15e-14, L 5.6e-15
    # on gcc/x86-64 (bounds from the larger; plain kicks: 2.9e-13 max, L 1e-13).
    # Planets: 3.4e-14 end, 1.1e-13 max, P 7e-18, L 2e-15 (FROST Fig. 5: ~1e-13, ~1e-16, ~1e-14).
    # With the split (twice the steps at the slowest level, so about twice the round-off): 4.5e-15 end,
    # 2.3e-14 max, L 1.2e-14; planets 9.4e-14, 1.1e-13, P 3.9e-18, L 2.5e-15.
    # The binary floor is a random walk: over BINARY_E = 0, 1e-12 ... 4e-12 (same physics, other round-off)
    # the end value spreads over 2.5e-15 - 2.2e-14 and the max over 1.0e-14 - 2.4e-14 with the per-pair Kahan
    # kernels, 2.7e-15 - 3.5e-14 and 1.0e-14 - 4.0e-14 with the tile-blocked ones (L 2.0e-14 at e = 0); the
    # bounds are 1.3x the largest of these, not of the single e = 0 draw.
    "floor_binary_e0": lambda exe: floor(exe, "floor e=0", dict(FROST4, BINARY_E="0.0", ETA="0.004", ETA_FF="0.004",
                                          ETA_FB="0.004", INTEGER_POSITIONS="true", DOMAIN_HALFSIZE_PC="1e-3"),
                                         (4.5e-14, 5.2e-14, 1e-14, 2.6e-14)),
    "floor_planets": lambda exe: floor(exe, "floor planets", dict(PLANETS, ETA="0.004", ETA_FF="0.004", ETA_FB="0.004",
                                        INTEGER_POSITIONS="true", DOMAIN_HALFSIZE_PC="1e-3"),
                                       (1.25e-13, 1.5e-13, 1.0e-17, 3.3e-15)),
    # 1000 yr is ~6 Neptune orbits: no drift check. HHS measured 1.69e-7, 4.27e-9, 2.50e-10, order 4.70 with the
    # split (2.35e-7, 8.27e-9, 4.39e-10, order 4.53 without)
    # (FROST: ~1e-13 at eta = 0.004, which order 4 from here reaches); FSI (all at the fastest occupied
    # level) measured 7.41e-8, 4.64e-9, 2.90e-10, order 4.00 (hung before the fsi-step fix)
    "planets_hhs": lambda exe: sweep(exe, "planets hhs", PLANETS, ["0.128", "0.064", "0.032"],
                                     [2.2e-7, 5.5e-9, 3.3e-10], (4.60, 5.00)),
    "planets_fsi": lambda exe: sweep(exe, "planets fsi", dict(PLANETS, INTEGRATOR="fsi"), ["0.128", "0.064", "0.032"],
                                     [9.6e-8, 6.1e-9, 3.8e-10], (3.90, 4.30)),
    # Before the fix: segfault in step 1 (duplicated particle after compaction, LogH NaN), then mass loss on
    # step restarts and |dE/E| ~ 1 from an incomplete merger sink. Measured on the host build: 1.4e-10 /
    # 6.7e-13 km/s (P grows smoothly: round-off of the LogH reconstruction of ~600 km/s orbits). Bounds: the
    # restart threshold for E, P far below one lost binary member (~3 km/s).
    "mergers_gpu_path": lambda exe: mergers(exe, "mergers gpu path", dict(MERGERS, HOST_SUBTREE_MAX="0"), (1e-8, 1e-11)),
    # Before: LogH wrote a 15-body group past its 8-body arrays (bus error). Measured 9.4e-6 / 2.0e-14 km/s.
    # Before: the single-rank step skipped the global PN1 kicks (ratio 0.03). Measured 1.045 (sampled once
    # per print, so the osculating periapsis carries the intra-orbit PN oscillation).
    "pn1_global_single_rank": lambda exe: precession(exe, "wide PN1 binary", PN1_WIDE, 0.1),
    # Before: v/a, gradient and perturbation criteria read accelerations that still held the binary
    # partner (stale from the last force evaluation), so members got orbital time-steps down to level 29
    # and step 1 never finished (> 120 s; the 10 % binary production runs hung the same way). Measured
    # 2 s, 3.2e-10 / 6.0e-15 km/s (2 stellar collisions; their orbital L is not carried, so no L check).
    # With the BIFROST population and the pericentre check (defaults since AUDIT F6): 6.6e-10, L 2.4e-14.
    "binaries_timestep": lambda exe: binaries(exe, "binaries", BINARIES, (1e-8, 1e-12)),
    # Before: 40 LogH give-ups (E_N + E_1PN cannot hold eta_restart under PN), |dE/E| 4.8e-5, |dL/L| 2.3e-5
    # unbooked. Measured 0 / 0 / 2.6e-13 km/s (the whole system is the PN subsystem, so all of it is booked).
    # Spins in LogH (BIFROST sec. 3.7). Before: the spins were not in the GBS state vector, so they never
    # changed (precession ratio 0, |dJ|/|J| 7.6e-6 with PN1), and the orbit had no spin-dependent terms.
    # Measured: precession ratio - 1 = 5.5e-3 (PN1 shifts the orbit; < 5e-7 with spin couplings only),
    # | |chi| - |chi0| | <= 3.0e-15 (Rodrigues rotations, norms restored after the extrapolation),
    # |dJ|/|J| 3.5e-7 (spins only; 1.1e-8 at 4x the separation: the O((v/c)^5) truncation) and 5.25e-6 with PN1
    "spin_precession": lambda exe: spin_precession(exe, "spin-orbit precession", SPIN_PREC, 7.2e-3, 4e-15),
    "spin_j": lambda exe: spin_j(exe, "spinning BH binary", SPIN_J, 4.6e-7, 4e-15),
    "spin_j_pn1": lambda exe: spin_j(exe, "spinning BH binary + PN1", SPIN_J_PN1, 6.9e-6, 4e-15),
    "pn_logh_books": lambda exe: pn_books(exe, "PN LogH binary", PN_LOGH, (1e-12, 1e-11)),
    # GBS_SEQUENCE = deuflhard, GBS_KMAX = 8 (MSTAR / FROST-CLUSTERS I): exact books as with the Bulirsch sequence
    # (measured 5.3e-16; the sequences differ from stage 4 on, 10 vs 12 substeps)
    "gbs_deuflhard_pn": lambda exe: pn_books(exe, "PN LogH binary, Deuflhard k_max 8",
                                             dict(PN_LOGH, GBS_SEQUENCE="deuflhard", GBS_KMAX="8"), (1e-12, 1e-11)),
    # Before: merger remnants kept a progenitor's m_zams, so SSE shrank them back (56.7 M_sun lost in 46
    # mergers). Measured 1.3e-9 M_sun (winds over 5 steps).
    "mergers_sse": lambda exe: sse_mass(exe, "mergers + SSE", dict(MERGERS, HOST_SUBTREE_MAX="0", SSE_ENABLED="true",
                                                                   TOTAL_TIME_MYR="5e-9"), 1e-6),
    # Before: the neighbour buffer (max(1000, N/10) pairs) truncated silently, 250 binaries stayed field
    # particles at level ~30 and step 1 never finished. Measured: all 1250 detected, 1.1 s, 1.7e-13 (BIFROST
    # population + pericentre check: 1.05e-10; HHS_CC 8.0e-11).
    "binaries50_detected": lambda exe: all_binaries(exe, "50 % binaries", BINARIES50, 1250),
    # HHS_CC (connected components): a pair made fast only by the v/a criterion must stay one component (before the
    # dominant-partner edges: precession off by 50 %), and binaries keep their strongest perturber (before: 10x dE)
    "cc_pn1_wide": lambda exe: precession(exe, "wide PN1 binary, HHS_CC", dict(PN1_WIDE, HHS_CC="true"), 0.1),
    "cc_binaries50": lambda exe: all_binaries(exe, "50 % binaries, HHS_CC", dict(BINARIES50, HHS_CC="true"), 1250),
    # measured 5.0e-11 / 3.8e-11 = 1.3 (10 before binaries kept their perturbers); BIFROST population: 0.76
    # HHS_CC_MIN = 64: 542 -> 239 components, max|dE/E| 7.7e-11 vs plain 1.0e-10 (GPU 200k: 36.5 -> 10.9 s/step, plain 9.1)
    "cc_min_binaries50": lambda exe: cc_min_groups(exe, "50 % binaries", BINARIES50, 3.0),
    "cc_vs_plain_binaries50": lambda exe: cc_vs_plain(exe, "50 % binaries", BINARIES50, 3.0),
    # 2000 Kroupa singles, 2 steps: eps = 1e-3 raised max|dE/E| from 1.5e-11 to 3.4e-8 (random), 2.5e-7 (frozen), 6.4e-8 (smooth)
    "force_err_injection": lambda exe: force_err(exe, "2000 stars", dict(BINARIES, BINARY_FRAC="0.0"), 1000.0),
    # HHS_CC must split with the pair time steps of the time-step kernel (with the c.o.m. velocities of the deleted
    # FLYBY_CM_VELOCITY, 331 particles had a pair step in the split below their dt_adapt before the fix; max |dE/E|
    # 4.9e-11 then). Measured with component velocities (host, 2026-10-12): 0 such particles, max |dE/E| 7.75e-11
    "cc_consistent_binaries50": lambda exe: cc_consistent(exe, "50 % binaries, HHS_CC", BINARIES50, 1.01e-10),
    # FLYBY_CM_VELOCITY (FROST-CLUSTERS II sec. 2.1; deleted 2026-10-12, restored 2026-10-13, off by default): only the
    # device kernels used the subsystem c.o.m. velocity; the CPU small-set path and the HHS_CC split kept the
    # component velocities. Before: CPU 1.255e-10 vs device 6.42e-11 after step 1 (relative 0.95), LogH calls 136 vs
    # 116 (6.4e-6 relative with the component velocities on both paths). Measured after: same LogH calls and GBS
    # steps, 2.2e-5 relative (9.5e-6 with 2 threads).
    # HHS_CC: 331 particles had a pair step in the split below their dt_adapt; 0 after (max |dE/E| 5.7e-11)
    "cm_cpu_vs_device": lambda exe: paths_agree(exe, "c.o.m. fly-by step", dict(BINARIES, FLYBY_CM_VELOCITY="true"),
                                                2.9e-5),
    "cc_cm_consistent": lambda exe: cc_consistent(exe, "50 % binaries, HHS_CC, c.o.m. fly-by step",
                                                  dict(BINARIES50, FLYBY_CM_VELOCITY="true"), 1e-8),
    # FLYBY_CM_EPS: a slow particle's kicks sample a subsystem's orbit at the pair step (the 200k, 10 % binaries seed-7
    # jumps, a 29+29 Msun BH binary at 55 AU and a passing star, 1.2e-8); the tidal-sampling pair step bounds the
    # energy they put in per step. Before: max |dE/E| 1.60e-5, end 9.6e-6 (c.o.m. velocities: 1.0e-5). Measured with
    # FLYBY_CM_EPS = 1e-5 (host, 1 thread): max 1.869e-7, end 9.305e-8 (1e-6: 6.9e-9, 1e-7: 5.1e-10). The c.o.m.-
    # velocity variant (cm_eps_tidal_sampling) gives the same max
    "cm_eps_tidal_components": lambda exe: floor(exe, "component fly-by step, tidal sampling",
                                                 dict(CM_TIDE, FLYBY_CM_EPS="1e-5"), (1.2e-7, 2.4e-7, 1e30, 1e30)),
    # The same with the c.o.m. velocity (FLYBY_CM_VELOCITY): before (c.o.m. velocity alone) max |dE/E| 1.0e-5;
    # measured with FLYBY_CM_EPS = 1e-5: 1.85e-7
    "cm_eps_tidal_sampling": lambda exe: floor(exe, "c.o.m. fly-by step, tidal sampling",
                                               dict(CM_TIDE, FLYBY_CM_VELOCITY="true", FLYBY_CM_EPS="1e-5"),
                                               (2.4e-7, 2.4e-7, 1e30, 1e30)),
    # The CPU small-set path takes the same tidal-sampling steps: same LogH calls and GBS steps per step as the
    # device (|dE/E| then differs at the level of the GBS errors of these many short LogH advances, ~1e-13 each).
    # Production settings (component velocities, sampled binary c.o.m. velocities, Kepler drift of non-PN pairs) with
    # PN on, so that the pairs with a member >= M_PN take LogH. Measured (host, 2026-10-12): 7824 + 7824 = 15648
    # LogH calls, GBS steps 82364 / 82309 per step on both paths (relative difference 0; bound 2e-4 kept for
    # round-off flips of a GBS step count; before 2026-10-12: c.o.m. velocities, KEPLER_2BODY = false and
    # BINARY_COM_V = average, 17240 calls at 88ed3db, difference 0). The paths differ by 1 ulp in field-star
    # velocities after step 1 (CPU small-set N3L loop vs the tiled device kernel's summation order, traced
    # 2026-10-12), which chaos amplifies
    # FLYBY_CM_KICK: a weakly perturbed binary interacts with the rest through its c.o.m. (mollified kicks), so the
    # c.o.m. fly-by steps no longer sample its orbit. Before (c.o.m. steps, no tidal-sampling step): max |dE/E|
    # 8.7e-9, end 8.6e-9 (component velocities with FLYBY_CM_EPS = 1e-11: 7.9e-11 in 2x the time). Measured with
    # FLYBY_CM_KICK (host, 1 thread): max 2.06e-12 (outer pericentre, the bound quadrupole energy), end 5.8e-14,
    # max |dP|/M 4.3e-14 km/s, max |dL/L| 4.0e-15
    "cm_kick_weak_tide": lambda exe: floor(exe, "c.o.m. kicks, weak tide",
                                           dict(WEAK_TIDE, FLYBY_CM_KICK="true", FLYBY_CM_KICK_EPS="1e-11"),
                                           (2e-13, 2.7e-12, 1e-12, 1e-13)),
    # The same in a cluster: 5000 stars, 50 % binaries, c.o.m. fly-by steps with the tidal-sampling step. Before
    # (FLYBY_CM_KICK off): max |dE/E| 2.46e-11 (component velocities 7.7e-11). Measured (host, 1 thread): 1224 of
    # 1250 binaries flagged, max |dE/E| 7.3e-12, |dP|/M 4.3e-15 km/s, |dL/L| 3.3e-15
    "cm_kick_binaries50": lambda exe: floor(exe, "c.o.m. kicks, 50 % binaries",
                                            dict(BINARIES50, FLYBY_CM_VELOCITY="true", FLYBY_CM_EPS="1e-11",
                                                 FLYBY_CM_KICK="true"),
                                            (1.5e-11, 1.5e-11, 1e-13, 1e-13)),
    # PN_BH_ONLY: binaries of massive stars that are not BHs leave the PN (kappa = 1) LogH path for the Kepler drift
    # and SDAR; the BH binaries stay PN
    "pn_bh_only": lambda exe: pn_bh_only(exe),
    # Tidal-sampling step near unbound fly-by subsystems had no floor (run B, levels to L22; fixed 2026-10-16)
    "flyby_group_floor": lambda exe: flyby_group_floor(exe),
    # PN_BH_ONLY false: the PN binaries of massive stars give the LogH calls the two paths are compared on
    "cm_eps_cpu_vs_device": lambda exe: paths_same_steps(exe, "fly-by step with tidal sampling",
                                                         dict(BINARIES, FLYBY_CM_EPS="1e-11", PN_ORDER="31",
                                                              PN_BH_ONLY="false"),
                                                         12000, 2e-4),
    # Triple branch of detect_from_pairs used a particle id where the union-find wanted a compact id
    # (out of bounds for ids >= the number of pair members; the tests' triples had ids 0-2)
    # The secular-triple driver took the inner pair from the first two ids (fixed 2026-10-10)
    # The secular step used omega / omega-dot (fixed 2026-10-11)
    "secular_omega0_steps": lambda exe: secular_steps(exe, "secular binary at omega = 1e-6", SEC_BINARY, 10),
    # Stellar binaries got PN secular terms: 20 internal steps per evolution here (thousands at L0 of the 200k
    # cluster, 5-7 ms secular calls). Measured 1 after (no PN below m_pn).
    "secular_stellar_no_pn": lambda exe: secular_steps(exe, "stellar binary, PN on", SEC_STELLAR, 1.3),
    # The step also had a precession term (1/omega-dot), but the omega drift is exact for constant (a, e): 1028
    # internal steps per evolution here, the 100000-step cap (48 ms) for this binary at 1e-3 Myr steps.
    # Measured 1 after (eta_sec t_GW > dt).
    "secular_bh_tight_steps": lambda exe: secular_steps(exe, "BH binary at 0.01 AU", SEC_BH_TIGHT, 1.3),
    # Before: both LogH calls stopped at the 1000-step limit after ~45 % of the interval, and passed the energy
    # check (cap=4, no retry: the members lagged behind the system clock). Measured after: 0 caps, |dE/E| 8.5e-12
    # The star 0.057 pc from a hard binary whose member moves straight at it passed the pair fly-by test and joined
    # the binary in a 3-body LogH subsystem with ~1e6 inner orbits per step: no result within 120 s before
    # (FLYBY_NGB_CM = false). Measured after (host): no triple, no LogH call, max |dE/E| 1.68e-12
    "flyby_member_com": lambda exe: flyby_member(exe, "star passing a hard binary", FLYBY_TRIPLE, 2.2e-12),
    # Before (fixed r_ngb): the hole stays outside the subsystem, the binary takes its tide as kicks (7256 LogH calls),
    # |dE/E| 2.38e-9. Measured with NGB_MASS_SCALING (host): one 3-body subsystem, 8 LogH calls, 4.44e-14
    "ngb_mass_scaling": lambda exe: floor(exe, "black hole passing a wide binary, mass-scaled r_ngb",
                                          dict(BH_FLYBY, NGB_MASS_SCALING="true"), (5.8e-14, 5.8e-14, 1e30, 1e30)),
    "logh_hierarchy_complete": lambda exe: logh_complete(exe, "tight binary in unstable triple", HIER_LOGH, 1.1e-11),
    "triple_id_order": lambda exe: triple_order(exe, "secular triple", SEC_TRIPLE, 1e-6),
    # SDAR (M3; Wang, Nitadori & Makino 2020; FROST-CLUSTERS I App. A2). Before (no SDAR key, plain LogH): GBS
    # step ratio 1, no slowed call. B-S triple (W20 Table 1), 0.07 t_KL: kappa_max 51.7 as in W20 sec. 7.1;
    # plain LogH (1024 steps/orbit limit, else it gives up) 4.86e6 GBS steps, 99 restarts, max|dE/E| 1.2e-7;
    # SDAR 1.56e5 steps (0.032), 2.56e-9 (books; 8e-8 physical), e_in within 2e-6 of LogH; gcc 15: 5.03e-9
    "sdar_kozai_triple": lambda exe: sdar_vs_logh(exe, "Wang+2020 B-S triple", KL_TRIPLE, 0.042, 3e-6, 6.5e-9,
                                                  {"NBODY_GBS_STEPS_PER_ORBIT": "1024"}),
    # F3 fly-by: LogH 1.52e6 GBS steps, max|dE/E| 5.6e-7 (restarts); SDAR 389 steps (2.6e-4), 3.1e-11. The SDAR
    # error depends on the binary's phase when the star arrives (after 60 Kepler drifts of ~3e4 orbits): over
    # TRIPLE_IN_ROT_DEG = 0, 10, ..., 70 it ranges 1.3e-11 - 3.2e-10 (median 1.6e-10) with 043190d's elements drift and
    # 1.5e-11 - 3.9e-10 (median 8e-11) with the new drift; this phase gives 8.0e-11 (043190d 1.3e-11)
    "sdar_f3_flyby": lambda exe: sdar_vs_logh(exe, "F3 fly-by", F3_FLYBY, 3.4e-4, None, 1.05e-10),
    # Before: the control (PN off) has no slowed call either
    "sdar_pn_kappa1": lambda exe: sdar_pn_off(exe, "PN BH binary + star", PN_TRIPLE),
    # SECULAR_TRIPLES = false (FROST-CLUSTERS I): the switch was ignored by the per-level re-classification
    # (40 secular calls before). In LogH the triple keeps |dE/E| 2.17e-11 vs 3.4e-5 secular: bound 2.9e-11
    "secular_triples_off": lambda exe: secular_triples_off(exe, "secular triple off", SEC_TRIPLE, 2.9e-11),
    # Measured 9.8e-9 (dense core, at the restart threshold); 2.35e-8 before the pericentre check, 2.68e-8 with
    # it (seed 5): bound 3.5e-8
    "triples_detected": lambda exe: triples(exe, "dense core", TRIPLES, 3.5e-8),
    # Triple stability near the MA01 boundary (q_out = 1/3, e_out = 0.2: R_p,crit / a_in = 3.53 (1 - 0.3 i/pi),
    # i.e. a_out / a_in > 4.42 coplanar, 3.98 at 60 deg, 3.53 at 120 deg). The old criterion (no inclination
    # factor, R_p / (a_in (1 + e_in)) with e_in = 0.3: a_out / a_in > 5.74) called the first and last unstable.
    "triple_stability": lambda exe: triple_stability(exe, "MA01", dict(SEC_TRIPLE, TOTAL_TIME_MYR="1e-4"),
                                                     [(5.0, 60, True), (4.0, 0, False), (4.0, 120, True)]),
    # r_h = 0.03 pc: before, a 9+ body chain (linked by unbound fly-bys) went to the FSI whole, with its
    # tight binaries integrated orbit by orbit (r_h = 0.02 pc never finished a step). Measured 3.6e-8
    # (4.5e-8 with the pericentre check).
    "dense_core_split": lambda exe: no_fsi_groups(exe, "dense core", dict(TRIPLES, PLUMMER_R_HALF="0.03",
                                                                            TOTAL_TIME_MYR="1e-4"), 1e-7),
    # AUDIT F6: thermal e without r_p > R1 + R2 put pericentres inside the stars: 21 mergers in the first
    # 1e-5 Myr before, 0 after (322 of 5000 e redrawn in a 20k cluster). Measured |dE/E| 2.3e-14.
    "ic_binaries_no_mergers": lambda exe: no_ic_mergers(exe, "close primordial binaries", CLOSE_BINARIES, 3e-14),
    # ETA_JERK = 0 turns the jerk criterion off. Before: the start-of-step kernel (AccDtK) still applied it,
    # dt = 0 * |a| / |j| = 0, the run stepped at DT_MIN (1.86e-14 Myr, hung). Measured 1.5625e-7 (as with 0.1)
    "eta_jerk0_fsi": lambda exe: fsi_step(exe, "ETA_JERK = 0", dict(FSI_BINARY, ETA_JERK="0"), 1.2e-7),
    # ETA_GRAD = 0 used to switch the v/a criterion off as well (one shared branch): the binary then stepped at
    # ETA_FF/FB = 10. Measured 2.8 before, 7.36e-7 after (v/a limited, ETA = 0.1)
    "va_without_grad": lambda exe: binaries(exe, "v/a with ETA_GRAD = 0", dict(VA_BINARY, ETA_GRAD="0", ETA="0.1"),
                                            (9.6e-7, 1e-12)),
    # Restarts halve every eta (BIFROST sec. 3.19) but did not halve ETA (v/a): with v/a limiting, a retry repeated
    # the failed step, "dE not decreasing", and the step was accepted above ETA_RESTART. ETA_GRAD = 1000 keeps the
    # shared branch on without the gradient step limiting. Measured before 10 give-ups, 1.9e-5;
    # after 13 restarts, 0 give-ups, 2.87e-7
    "restart_halves_va": lambda exe: restarts(exe, "v/a restarts", dict(VA_BINARY, ETA_GRAD="1000", ETA="0.2",
                                                                       ETA_RESTART="1e-7"), 3.7e-7),
    # SCIC initial conditions (BIFROST App. A). r_h from the mass-size relation when not given (before: 3.5 pc
    # default, PLUMMER_R_HALF = mass_size not understood): 2000 stars, 1000 M_sun -> 1.463 pc. Measured r_h 1.86 %
    # off (seed 5; seeds 1-8 scatter -6 ... +3 %, rms 2.5 %, the median of 2000 draws)
    "ic_mass_size_rh": lambda exe: ic_mass_size(exe, "mass-size r_h", IC_EQUAL, 0.025),
    # 100 % binaries: Q of the centres of mass was 0.336 (members' independent velocities averaged); measured
    # 0.5073 with one draw per binary
    "ic_binary_com_virial": lambda exe: ic_com_virial(exe, "binary c.o.m. virial ratio",
                                                      dict(IC_ONLY, BINARY_FRAC="1.0"), 9.5e-3),
    # Dense core (r_h = 0.05 pc), 50 % binaries of 1-20 AU, r_break = 10 a_max = 200 AU: 4 close object pairs,
    # 0 after one angular round; r_h changes by 2.1e-4 (re-centring only)
    "ic_r_break": lambda exe: ic_r_break(exe, "artificial binaries", dict(IC_ONLY, PLUMMER_R_HALF="0.05",
                                         BINARY_FRAC="0.5", BINARY_A_MIN_AU="1", BINARY_A_MAX_AU="20",
                                         IC_R_BREAK="10"), 2.7e-4),
    # Collisions inside the subsystem integration (FROST-CLUSTERS I App. B3; host gcc/x86-64 measurements). Before
    # (1890208; and with the deleted SUBSYS_COLLISIONS = false): head-on no merger, 9 x 8 LogH give-ups, max|dE/E|
    # 5.3e-5; contact no merger, 4.2e5-4.8e5 GBS steps per step, LogH retries and a step restart,
    # 6.3 s/step, 9.5e-9; TDE: the star passes the BH at 0.74 r_t; ISCO: the binary goes on inside the ISCO (3-body
    # subsystem). After: one merger each; max|dE/E| 1.38e-12 (head-on), 4.72e-11 (contact; the GBS error of the call,
    # the books are exact), 1.16e-12 (TDE), 6.1e-16 (ISCO, PN: all booked; round-off, bound 1e-14)
    "coll_headon_triple": lambda exe: subsys_collision(exe, "head-on in triple", COLL_HEADON, "STELLAR", 1.8e-12),
    "coll_contact_flyby": lambda exe: subsys_collision(exe, "contact binary + fly-by, SDAR", COLL_CONTACT, "STELLAR",
                                                       6.1e-11, min_landings=1),
    "coll_contact_flyby_logh": lambda exe: subsys_collision(exe, "contact binary + fly-by, LogH",
                                                            dict(COLL_CONTACT, SDAR="false"), "STELLAR", 6.1e-11,
                                                            min_landings=1),
    # Fryer et al. 2012 rapid: fallback masses were used as fractions (BHs from 2.7 Msun, 3x too many). Measured
    # after (200k evolved Kroupa, Z = 2e-4, 1 Gyr): 339 BHs 5.53-45 Msun (BIFROST Fig. 9: 357, 5.5-41.6); here 10k
    "fryer_rapid_gap": lambda exe: remnant_masses(exe, "Fryer rapid", dict(BINARIES, PLUMMER_N="10000", PLUMMER_M_TOTAL="4000",
                                                  PLUMMER_MASS_DIST="kroupa_evolved", BINARY_FRAC="0",
                                                  SSE_METALLICITY="0.0002", SSE_REMNANT_MODEL="FRYER_RAPID",
                                                  SSE_WIND_MODEL="VINK", SSE_PPISN_MODEL="STRONG",
                                                  TOTAL_TIME_MYR="1e-6", DT_MAX_MYR="1e-6"), 5.0, 2.0, 50.0),
    # G-1. LogH PN binary: frame-invariant to round-off (measured 6.0e-12 at 30 km/s, 5e-13 at 300 km/s along x,
    # 4.3e-12 along y). Secular PN binary: 1.2e-3 before (the reshaped orbit kept the rounding error of a, a phase
    # error over 1e4-1e5 orbits per step); 9.1e-8 after (absolute-coordinate rounding)
    "pn_boost_logh": lambda exe: boost_invariance(exe, "PN LogH binary", PN_LOGH, ["30,0,0", "300,0,0", "0,300,0"], 8e-12),
    "pn_boost_secular": lambda exe: boost_invariance(exe, "PN secular binary", SEC_BH_TIGHT,
                                                     ["30,0,0", "300,0,0", "0,300,0"], 1.2e-7),
    # AUDIT G row 1a: 1000 binaries of 250-450 AU (just above r_ngb = 206 AU) in 2000 stars. Plain: the pairs are FSI
    # particles stepped at a fraction of their orbit, 250 all-fast two-single deep sets, 94.5 s, max|dE/E| 9.65e-9.
    # NGB_WIDE_FAC = 2.5: subsystems on the Kepler drift, 0 such sets, 5.7 s, 9.70e-9 (host, 1 thread)
    "wide_binaries_ngb": lambda exe: wide_pairs(exe, "wide binaries, NGB_WIDE_FAC = 2.5",
                                                dict(BINARIES, BINARY_FRAC="1.0", BINARY_A_MIN_AU="250",
                                                     BINARY_A_MAX_AU="450", TOTAL_TIME_MYR="2e-3", FLYBY_CM_EPS="1e-11",
                                                     TIDAL_STEP_RES="1", FLYBY_NGB_CM="true", NGB_WIDE_FAC="2.5"),
                                                1.3e-8),
    # Cluster diagnostics (HARD_PROBLEMS G2): 10k stars, r_h 1 pc (20k: radii within 1.3 % of Plummer, centre
    # 1.1 % of a; 999 of 1000 binaries); Kroupa with 10 % of stars in binaries of 0.1-100 AU (500)
    "cluster_diag": lambda exe: cluster_diag(exe, "Plummer 10k", dict(BINARIES, PLUMMER_N="10000", PLUMMER_M_TOTAL="4000",
                                                                       TOTAL_TIME_MYR="1e-6", DT_MAX_MYR="1e-6"), 500),
    # External tidal field (HARD_PROBLEMS G3; PeTar-style reference point + differential kicks). Reference orbit in a
    # 220 km/s logarithmic halo at 8 kpc, 100 steps of 0.1 Myr: 3.6e-7 of R off the circle
    "tidal_ref_orbit": lambda exe: tidal_orbit(exe, "log halo", dict(SEC_BINARY, PN_ORDER="0", TOTAL_TIME_MYR="10",
                                                                     DT_MAX_MYR="0.1", PRINT_INTERVAL="10",
                                                                     TIDAL_FIELD="loghalo", GALAXY_VC_KMS="220",
                                                                     GALAXY_R_PC="8000"), 4.7e-7),
    # 2000 stars at 100 pc in the halo (r_J ~ 0.7 pc, r_h 1 pc): max|dE/E| 1.17e-9 (9.75e-10 without the field),
    # E_tidal -2.07 (code units, ~7e-4 of |E|); with the Jacobi escapers (R_ESCAPE_RJ = 2, default) 9.77e-10
    "tidal_energy_books": lambda exe: tidal_books(exe, "strong tide", dict(BINARIES, TOTAL_TIME_MYR="5e-3",
                                                                           TIDAL_FIELD="loghalo", GALAXY_VC_KMS="220",
                                                                           GALAXY_R_PC="100"), 1.5e-9),
    # A 0.76 Msun star bound to a 258 Msun body at 260 AU (a just above r_ngb) while a 600 km/s star passes that body
    # at 6e-4 pc: the fly-by subsystem formed without the companion (the mass-scaled filter weighed its tide only
    # against bound partners), whose pull on the subsystem then forced 1.5e-14 Myr tidal-sampling steps (35 HHS
    # levels, no step finished in 120 s). Now the companion joins (tidal ratio ~2e3 against the fly-by's separation):
    # deepest level 7, max|dE/E| 2.12e-9
    "ss_companion_flyby": lambda exe: no_runaway_depth(exe, "companion of a fly-by member", dict(
        HIER_LOGH, TRIPLE_A_IN_PC="1.26e-3", TRIPLE_E_IN="0.1", TRIPLE_M1="258", TRIPLE_M2="0.76",
        TRIPLE_A_OUT_PC="6e-4", TRIPLE_E_OUT="200", TRIPLE_OUT_START_Q="133.333333", TRIPLE_M3="0.13",
        TRIPLE_INCL_DEG="20", TOTAL_TIME_MYR="2e-3", DT_MAX_MYR="1e-3", ETA_PERT="0.02", ETA_RESTART="0",
        NGB_MASS_SCALING="true", ETA="0", ETA_JERK="0", ETA_GRAD="0.2", FLYBY_CM_EPS="1e-11", DT_MIN_MYR="1e-14"),
        12, 2.8e-9),
    # INTEGER_POSITIONS with a fly-by subsystem that forms and dissolves (1 + 1 Msun, q = 20 AU, e = 1.5, a third star
    # at 0.05 pc): the subsystem integrators moved the members' double positions but not their integer coordinates,
    # so the members snapped back to where they joined when the fly-by ended: max|dE/E| 1.75 (double positions
    # 3.92e-7; 200k, 10 % binaries on 4 GPUs: 2e-7 against 1.4e-11). Now identical to the double run
    "int_pos_flyby": lambda exe: int_pos_vs_double(exe, "fly-by subsystem dissolving", dict(
        HIER_LOGH, TRIPLE_A_IN_PC="1e-4", TRIPLE_E_IN="1.5", TRIPLE_IN_START_Q="5", TRIPLE_M1="1", TRIPLE_M2="1",
        TRIPLE_A_OUT_PC="0.05", TRIPLE_E_OUT="0.3", TRIPLE_M3="1", TOTAL_TIME_MYR="2e-2", DT_MAX_MYR="1e-3",
        ETA_RESTART="0", DOMAIN_HALFSIZE_PC="1"), 1.3),
    # 2+2 quadruple (quad22_test defaults = the BIFROST Fig. 11 stall, 200k, 50 % binaries): BH binary 20.7 + 14.1 Msun
    # (a = 58 AU) and a 0.79 + 0.105 Msun binary (a = 0.019 AU, ~1500 orbits per 1e-3 Myr, tidal ratio ~4e-12) bound to it
    # at ~0.003 pc, one PN subsystem. Without item 4c (SDAR off for whole PN subsystems): no step within 900 s on the
    # login node; with NBODY_SDAR_PN=1: 0.44 s, kappa_max 8.9e4, max|dE/E| 2.1e-16 (bound: round-off floor)
    # TEST_CASE = particle_file reproduces binary_test (Kepler e = 0.9, 10 intervals) bit for bit from its initial dump
    "particle_file_roundtrip": lambda exe: particle_file_roundtrip(exe, dict(KEPLER, TOTAL_TIME_MYR="1e-4")),
    "quad22_sdar": lambda exe: quad22_sdar(exe, "2+2 quadruple, PN + SDAR", dict(
        HIER_LOGH, TEST_CASE="quad22_test", PN_ORDER="31", M_PN="5.0", SDAR="true", R_NGB="0.01",
        TOTAL_TIME_MYR="1e-3", DT_MAX_MYR="1e-3", TEST_STELLAR_RADIUS_RSUN="0.4", ETA_RESTART="1e-8"), 1e-14),
    # Same cluster, R_ESCAPE_RJ = 2: r_J 0.708 pc at M = 800 Msun, 623 stars leave over 5 steps, max|dE/E| 9.77e-10
    "tidal_jacobi_escapers": lambda exe: tidal_escapers(exe, "2 r_J", dict(BINARIES, TOTAL_TIME_MYR="5e-3",
                                                                          TIDAL_FIELD="loghalo", GALAXY_VC_KMS="220",
                                                                          GALAXY_R_PC="100"), 1.3e-9),
    # GRAD_FRESH (fix 3): the gradient criterion read grad_mag as the last gradient kick left it (0 before the first
    # kick, the slow-fast part after a cross kick): as the only criterion the first step ran at ETA_FF/FB = 10 and
    # the planets ran away (max|dE/E| 1.03 / 1.07); with |g'| from all particles at level 0: 3.61e-6 / 1.87e-7, order 4.27
    "grad_only_planets": lambda exe: grad_sweep(exe, "gradient step only, planets, GRAD_FRESH",
                                                dict(PLANETS, COLLISIONAL="true", R_NGB="1e-9", R_NGB_MAX="1e-9",
                                                     PN_ORDER="0", SSE_ENABLED="false", PRINT_INTERVAL="10",
                                                     ETA_FF="10", ETA_FB="10", ETA_JERK="0", ETA="0", GRAD_FRESH="true"),
                                                ["0.1", "0.05"], [4.7e-6, 2.4e-7], (3.8, 4.7)),
    # BH_MERGE_RADIUS: total (default) 6 G (m1 + m2) / c^2 = 5.74e-12 pc for 10 + 10 Msun; max_single (FROST-CLUSTERS
    # I sec. 2.6.3) 6 G max(m1, m2) / c^2 = 2.87e-12 pc (the inspiral runs longer)
    "bh_merge_radius_fc": lambda exe: bh_merge_radius(exe, "max single ISCO", dict(COLL_ISCO, BH_MERGE_RADIUS="max_single",
                                                                                   TOTAL_TIME_MYR="6e-14"), 2.8692e-12),
    "bh_merge_radius_total": lambda exe: bh_merge_radius(exe, "total-mass ISCO", COLL_ISCO, 5.7384e-12),
    # Glebbeek 2013 / FROST-CLUSTERS III collision mass loss (head-on 1 + 1 Msun: remnant 1.85 Msun)
    "coll_mass_loss": lambda exe: coll_mass_loss(exe, "head-on with mass loss", COLL_HEADON, 1.8e-12),
    # Natal kicks (2026-10-12): max |dE/E| with E_sse 4.3e-9 and 9.6e-9 (round-off of the ~1e7 kick energy against
    # |E0| = 30), bounds 1.3x; the statistical bounds are 3 standard errors (measured: <|v|> at 0.45, 0.28 and 0.29
    # of them, <n_k> up to 0.86)
    "natal_kicks_ns": lambda exe: natal_kicks_ns(exe, "500 NSs", NATAL_NS, 5.6e-9),
    "natal_kicks_fallback": lambda exe: natal_kicks_fallback(exe, "rapid + ECSN", NATAL_MIX, 1.25e-8),
    "coll_tde_accretion": lambda exe: tde_accretion(exe, "TDE accretion fraction", COLL_TDE),
    "coll_tde_parabolic": lambda exe: subsys_collision(exe, "TDE, parabolic", COLL_TDE, "TDE", 1.5e-12),
    "coll_bh_isco": lambda exe: subsys_collision(exe, "BH binary to ISCO, PN", COLL_ISCO, "ISCO", 1e-14),
    # Kepler drift (kepler_drift, subsystem audit #2): universal variables below e = 0.1 and near the parabola, the
    # perifocal construction from the anomaly otherwise. Bounds 1.3x measured (raven host, 2026-10-11); 043190d's
    # elements route: bound e < 0.5 1v16 = 2.0, ref = 1.2 (phase lost at e <= 1e-8), parabolic and hyperbolic
    # dE = 1.8e9 / 1.5e8 (leapfrog fallback), thermal dE 1.5e-13. 4422c7d (universal f and g from any state): thermal
    # dE 7.8e-13 (5e-10 for random binaries at e = 0.9998). 1e5 orbits and the far hyperbolae sit at the conditioning
    # of the problem (eps n dt of phase; |r| |v| eps of L at r = 1e6 A)
    "kepler_drift_unit": lambda exe: kepler_unit(exe, {
        "bound_thermal": (7.6e-9, 4.2e-10, None, 7.4e-14, 3.4e-15),
        "bound_e<0.5": (8.6e-11, 1.5e-11, 7.8e-12, 3.2e-15, 1.6e-15),
        "bound_e>=0.5": (2.1e-12, 2.8e-12, 1.2e-12, 3.9e-15, 1e-15),
        "bound_1e5_orbits": (2.6e-10, 1.3e-9, 7.5e-10, 1.5e-15, 1e-15),
        "parabolic": (2e-15, 7.7e-10, None, 1.3e-15, 1.4e-12),
        "hyperbolic": (2.3e-14, 1.4e-11, 2.6e-14, 1.16e-13, 9.7e-11),
        "hyperbolic_far": (3.4e-11, None, 8.2e-14, 2.9e-13, None)}),
    # Secular (Kepler-drift) binary at e = 0 and 1e-8, 100.37 orbits as 1 or 16 steps. Measured: 1 vs 16 9.5e-13 /
    # 1.1e-12, vs analytic 1.1e-12 / 9.2e-13, |dE/E| 3.3e-15 / 3.0e-15, |dL/L| 0. 043190d: 1 vs 16 = 2.0 / 0.55,
    # vs analytic up to 2.0 / 1.3 (elements route)
    "secular_drift_e0": lambda exe: secular_drift(exe, "e = 0", SEC_DRIFT, (1.25e-12, 1.45e-12, 4.3e-15, 1e-15)),
    "secular_drift_e1e-8": lambda exe: secular_drift(exe, "e = 1e-8", dict(SEC_DRIFT, BINARY_E="1e-8"),
                                                     (1.4e-12, 1.2e-12, 4e-15, 1e-15)),
    # Kepler path of isolated non-PN pairs (subsystem audit #1, KEPLER_2BODY until 2026-10-12): the drift is exact
    # (vs the analytic orbit at round-off). Measured (raven host, 2026-10-11): Kepler vs analytic 1.1e-15 / 2e-32 /
    # 2.4e-15, max |dE/E| 1.2e-15 / 0 / 2.0e-15; bounds 1.3x (1e-15 floor). The deleted LogH alternative (same runs,
    # NBODY_GBS_TTOL = 1e-12) was 5.4e-13 / 7.2e-11 / 1.1e-13 off the analytic orbit. 043190d: no Kepler2 drift
    "kepler2_norb05": lambda exe: kepler_exact(exe, "0.5 orbits per step", KEP_2B, (1.4e-15, 1.6e-15)),
    "kepler2_norb2": lambda exe: kepler_exact(exe, "2 orbits per step",
                                              dict(KEP_2B, TOTAL_TIME_MYR=f"{4.0 * P_SEC_MYR:.15e}",
                                                   DT_MAX_MYR=f"{2.0 * P_SEC_MYR:.15e}"), (1e-15, 1e-15)),
    "kepler2_hyperbolic": lambda exe: kepler_exact(exe, "hyperbolic encounter", KEP_2B_HYP, (3.1e-15, 2.6e-15)),
    # Grazing contact binary: two Kepler drifts, then the drift containing the pericentre goes to LogH, whose step is
    # aimed at the pericentre and merges there (r_sep / r_coll = 0.9997); measured |dE/E| 1.0e-15 with the merger sink
    "kepler2_contact": lambda exe: subsys_collision(exe, "contact binary via the Kepler path", KEP_2B_CONTACT, "STELLAR",
                                                    1.4e-15, min_kepler2=1),
    "logh_big_group": lambda exe: big_group(exe, "big group", BIG_GROUP, (1.2e-5, 1e-12)),
    # Exact tree searches (M7.4), forced at every set size, each search checked bitwise against O(N^2)
    # Tree searches on by default above the break-even set size (host builds 2048): 5000 stars use them at the top
    # levels without any TREE_* key (no tree search ran before), bit-identical to the O(N^2) run
    # Host-resident subtrees (GPU, 2026-10-16: max |dE/E| ratio 1.000 in all three; bound 1.3x): mergers compact the host copy mid-recursion (every set of the 400 stars on the host);
    # binaries with subsystems at every level
    "host_subtree_mergers": lambda exe: host_subtree(exe, "host subtrees, mergers", MERGERS, 1.3, hs_max=512),
    "host_subtree_binaries": lambda exe: host_subtree(exe, "host subtrees, 10 % binaries", BINARIES, 1.3),
    "host_subtree_triples": lambda exe: host_subtree(exe, "host subtrees, dense core", TRIPLES, 1.3),
    "tree_default_on": lambda exe: tree_exact(exe, "tree default, 50 % binaries", BINARIES50),
    "tree_dt_binaries": lambda exe: tree_exact(exe, "tree, 10 % binaries", dict(BINARIES, TREE_DT_MIN_N="2")),
    # host-thread tree (TREE_DT_CPU_MAX; component velocities; tree_dt_binaries_cm with the c.o.m. ones)
    "tree_dt_binaries_host": lambda exe: tree_exact(exe, "host tree, 10 % binaries",
                                                    dict(BINARIES, TREE_DT_CPU_MAX="1000000", TREE_DT_MIN_N="2")),
    "tree_dt_binaries_cm": lambda exe: tree_exact(exe, "host tree, 10 % binaries, c.o.m. fly-by velocities",
                                                  dict(BINARIES, FLYBY_CM_VELOCITY="true", TREE_DT_CPU_MAX="1000000",
                                                       TREE_DT_MIN_N="2")),
    "tree_dt_binaries50": lambda exe: tree_exact(exe, "tree, 50 % binaries", dict(BINARIES50, TREE_DT_MIN_N="2")),
    "tree_dt_triples": lambda exe: tree_exact(exe, "tree, dense core", dict(TRIPLES, TREE_DT_MIN_N="2")),
    "tree_dt_mergers": lambda exe: tree_exact(exe, "tree, mergers", dict(MERGERS, HOST_SUBTREE_MAX="0", TREE_DT_MIN_N="2")),
    # FLYBY_CM_EPS tidal-sampling step in the tree bound (dt_tree.h tidal_lower_bound): before, the O(N^2) kernel took
    # every time-step search with the tidal step on (no tree searches), now the tree must give the same bits
    "tree_dt_tidal": lambda exe: tree_exact(exe, "tree, 10 % binaries, tidal-sampling step",
                                            dict(BINARIES, FLYBY_CM_EPS="1e-11", TREE_DT_MIN_N="2"), need_dt=True),
    "tree_dt_tidal_host": lambda exe: tree_exact(exe, "host tree, tidal-sampling step",
                                                 dict(BINARIES, FLYBY_CM_EPS="1e-11", TREE_DT_CPU_MAX="1000000",
                                                      TREE_DT_MIN_N="2"), need_dt=True),
    "tree_dt_tidal_cm": lambda exe: tree_exact(exe, "host tree, c.o.m. fly-by velocities, tidal-sampling step",
                                               dict(BINARIES, FLYBY_CM_VELOCITY="true", FLYBY_CM_EPS="1e-11",
                                                    TREE_DT_CPU_MAX="1000000", TREE_DT_MIN_N="2"), need_dt=True),
    # (the dense-core variant, TRIPLES with FLYBY_CM_EPS = 1e-11, passed with 150991 searches checked but takes ~460 s)
    "tree_dt_softened": lambda exe: tree_exact(exe, "tree, softened collisionless", dict(BINARIES, COLLISIONAL="false",
                                               SOFTENING_PC="1e-3", BINARY_FRAC="0", TREE_DT_MIN_N="2")),
    # MPI cases, run under NBODY_TEST_LAUNCHER (CTest sets 2 ranks). Spins were not broadcast: lost on every
    # rank of a GPU run (host backends alias the mirror, so the host build cannot fail this one).
    "spin_mpi2": lambda exe: spins(exe, "IMBH spin, 2 ranks", SPIN_IMBH, 0.9),
    # Mergers mid-recursion on 2 ranks: compaction must leave the replicas identical (MPI slices resynced)
    "mergers_mpi2": lambda exe: replicas(exe, "mergers, 2 ranks", dict(MERGERS, HOST_SUBTREE_MAX="0",
                                                                         REPLICA_CHECK_INTERVAL="1")),
    # CPU path with restarts off: a merger in a nested CPU call used to drop the enclosing level's subsystem
    # flags on the host (internal forces counted twice), |dE/E| 2e-5 per step, hidden by restarts. Measured
    # now 1.4e-10 / 6.7e-13 km/s, as the GPU path.
    # 2000 stars with binaries on 1 and 2 ranks: measured |difference| 4e-15 over 5 steps (summation order)
    # Rank-count independence (2026-10-12: trajectories already were; the PE check now sums fixed row blocks, before
    # the printed |dE/E| differed in the 7th digit). Sets above SERIAL_N2_GPU = 256 are split over the ranks
    "rankcount_binaries": lambda exe: rank_count_identical(exe, "binaries, split sets",
                                                          dict(BINARIES, TOTAL_TIME_MYR="3e-3", SERIAL_N2_GPU="256")),
    # Dense core (triples, subsystems, restarts): the distributed gradient took grad_mag from |a_after - a_before| and
    # its slow-fast gradient wrote none, so the gradient time step depended on the rank count (2026-10-12)
    "rankcount_triples": lambda exe: rank_count_identical(exe, "dense core, split sets",
                                                         dict(TRIPLES, SERIAL_N2_GPU="256")),
    "rankcount_mergers": lambda exe: rank_count_identical(exe, "mergers, split sets",
                                                         dict(MERGERS, HOST_SUBTREE_MAX="0", SERIAL_N2_GPU="64")),
    # Natal kicks: hashed from (pid, PLUMMER_SEED, kw), the same on every rank count
    "rankcount_natal_kicks": lambda exe: rank_count_identical(exe, "natal kicks, split sets",
                                                             dict(NATAL_MIX, SERIAL_N2_GPU="256")),
    "mpi_1v2": lambda exe: ranks_agree(exe, "1 vs 2 ranks", dict(BINARIES, TOTAL_TIME_MYR="5e-3", REPLICA_CHECK_INTERVAL="1"),
                                       1e-13),
    "mergers_cpu_path": lambda exe: mergers(exe, "mergers cpu path", dict(MERGERS, HOST_SUBTREE_MAX="64", ETA_RESTART="0"),
                                            (1e-8, 1e-11)),
    # Restart files (HARD_PROBLEMS.md G1): 2n steps against n steps + restart + n steps, L0 rows and particle dumps
    # bit-identical. Before (bc1cbc3): no restart files, --restart is an unknown option (exit 1).
    "restart_binaries": lambda exe: restart_identical(exe, "10 % binaries", BINARIES, 2, snapshots=True),
    "restart_walltime": lambda exe: restart_identical(exe, "10 % binaries, WALLTIME_LIMIT stop", BINARIES, 2,
                                                      stop="walltime"),
    "restart_triples": lambda exe: restart_identical(exe, "dense core, triples", TRIPLES, 1),
    "restart_mergers": lambda exe: restart_identical(exe, "mergers, step restarts", dict(MERGERS, HOST_SUBTREE_MAX="0"), 5),
    "restart_mergers_sse": lambda exe: restart_identical(exe, "mergers + SSE", dict(MERGERS, HOST_SUBTREE_MAX="0",
                                                                                   SSE_ENABLED="true"), 3),
    # Natal kicks before and after the restart at 30 Myr (draws hashed from pid, PLUMMER_SEED and kw)
    "restart_natal_kicks": lambda exe: restart_identical(exe, "natal kicks", NATAL_MIX, 30),
    # NGB_MASS_SCALING: <m> of the initial particles is a run constant (043190d..: recomputed from the merged-down
    # state on restart)
    "restart_ngb_mergers": lambda exe: restart_identical(exe, "mergers, mass-scaled r_ngb",
                                                         dict(MERGERS, HOST_SUBTREE_MAX="0", NGB_MASS_SCALING="true"), 5),
    "restart_tidal": lambda exe: restart_identical(exe, "tidal field", dict(BINARIES, TIDAL_FIELD="loghalo",
                                                                          GALAXY_VC_KMS="220", GALAXY_R_PC="100"), 2),
    "restart_pn_logh": lambda exe: restart_identical(exe, "PN LogH binary (aux. velocities)", PN_LOGH, 5),
    "restart_spin": lambda exe: restart_identical(exe, "spinning BH binary", SPIN_J_PN1, 5),
    "restart_secular": lambda exe: restart_identical(exe, "secular PN binary", SEC_BINARY, 5),
    "restart_secular_triple": lambda exe: restart_identical(exe, "secular triple", SEC_TRIPLE, 5),
    "restart_sdar": lambda exe: restart_identical(exe, "SDAR triple", dict(KL_TRIPLE, SDAR="true"), 2),
    "restart_tidal_sampling": lambda exe: restart_identical(exe, "tidal-sampling step",
                                                            dict(CM_TIDE, FLYBY_CM_EPS="1e-5"), 5),
    "restart_cm_velocity": lambda exe: restart_identical(exe, "c.o.m. fly-by velocity, tidal sampling",
                                                         dict(CM_TIDE, FLYBY_CM_VELOCITY="true", FLYBY_CM_EPS="1e-5"), 5),
    "restart_spin_imbh": lambda exe: restart_identical(exe, "cluster + spinning IMBH binary, all PN", SPIN_IMBH, 5),
    # FSI: the global step is the fastest occupied level, not DT_MAX: the first run stops after step 1
    "restart_fsi": lambda exe: restart_identical(exe, "FSI integrator", dict(FSI_BINARY, DT_MAX_MYR="1e-6"), 5,
                                                 stop="walltime"),
    "snapshot_cluster_header": lambda exe: snapshot_cluster_header(exe, "tidal cluster", dict(BINARIES, TOTAL_TIME_MYR="3e-3",
                                                                                             TIDAL_FIELD="loghalo", GALAXY_VC_KMS="220",
                                                                                             GALAXY_R_PC="100")),
    "snapshot_roundtrip": lambda exe: snapshot_roundtrip(exe, "IMBH cluster", dict(SPIN_IMBH, TOTAL_TIME_MYR="3e-4")),
    "luna_reader": lambda exe: luna_reader(exe, "IMBH cluster", dict(SPIN_IMBH, TOTAL_TIME_MYR="3e-4",
                                                                     SNAPSHOT_INTERVAL="1")),
    # MPI (2 ranks under NBODY_TEST_LAUNCHER): rank 0 writes, every rank reads; replicas checked before the write
    "restart_mpi2": lambda exe: restart_identical(exe, "10 % binaries, 2 ranks", dict(BINARIES, REPLICA_CHECK_INTERVAL="1"), 2),
    "restart_mpi2_mergers": lambda exe: restart_identical(exe, "mergers, 2 ranks", dict(MERGERS, HOST_SUBTREE_MAX="0",
                                                          REPLICA_CHECK_INTERVAL="1"), 5),
}

if __name__ == "__main__":
    exe, case = sys.argv[1], sys.argv[2]
    result = CASES[case](exe)
    sys.exit(77 if result == "skip" else 0 if result else 1)   # 77: CTest SKIP_RETURN_CODE
