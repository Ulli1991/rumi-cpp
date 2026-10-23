"""Writes <dir>/logh_ctrl.cpp: src/collisional/logh.cpp with the Newtonian kick of logh_leapfrog_steps replaced by
The dv/ds = a / Ubar, Ubar = -U(q) (Rantala et al. 2023 Eq. 3; Mikkola & Tanikawa 1999): the reference (control)
for tests/enzyme/symplectic (compile with -DCTRL -I<dir> -Isrc/collisional). Usage: python3.11 make_ctrl.py <dir>"""
import os, sys
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
s = open(os.path.join(root, "src", "collisional", "logh.cpp")).read()
new = "        { const double iU = -1.0 / logh_potential_energy(state, G); for (int i = 0; i < n3; i++) state.vel[i] += ds_half * accel[i] * iU; }\n"
for old in ("        kick_cubic(state, accel, ds_half, Omega0);\n", "        kick_cubic(state, accel, ds_half, Omega_half);\n"):
    assert s.count(old) == 1, old
    s = s.replace(old, new)
os.makedirs(sys.argv[1], exist_ok=True)
open(os.path.join(sys.argv[1], "logh_ctrl.cpp"), "w").write(s)
