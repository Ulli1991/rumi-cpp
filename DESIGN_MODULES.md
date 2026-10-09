# Module design for additional physics (SPH and beyond)

Status: design note, nothing implemented. Written before any hydro work so that it starts from a clean
architecture and does not repeat what happened in tree codes such as Gadget, where the SPH neighbour search
reused the gravity tree because it was "free".

## The mistake to avoid

Doing the SPH neighbour search on the gravity tree couples the two solvers through one data structure:

- The tree node becomes a dumping ground: `h_max`, velocity bounds for the time-step limiter, feedback and
  accretion radii, metal-mixing radii, per-physics activity flags. Every gas module adds fields to a structure
  designed for multipoles, and the gravity walk pays for them in memory and cache traffic.
- Changing the gravity solver (opening criterion, leaf size, FMM, GPU layout) silently breaks hydro, feedback
  and accretion, which depend on its structure. Optimising one physics becomes a negotiation with all others.
- The optimal shapes differ: gravity wants mass-based opening, deep trees and large leaves; SPH wants cells of
  the kernel support (~2h). A shared tree is a compromise for both.
- Active-particle logic for gravity steps and hydro sub-steps ends up in the same walk, and the time
  integration becomes impossible to reason about.

The same happened to the particle struct: one record carrying every field of every module.

## Principles

1. **Each module owns its spatial structure.** Gravity: direct sum (`gravity/n2_driver.h`) or FMM
   (`gravity/fmm.h`). Hydro: cell lists and Verlet lists (Cabana). Feedback, accretion and other local
   physics build their own small neighbour structures or use hydro's through its interface. No module adds
   fields to another module's structure.

2. **Narrow interfaces.** Modules never call each other; the integrator combines their results.
   - gravity: in positions, masses, softening (per particle if adaptive); out accelerations, potential.
     It does not know which particles are gas.
   - hydro: in positions, velocities, smoothing lengths, thermodynamic state; out accelerations, du/dt,
     time-step constraints.
   - subsystems / regularisation (existing): in member state, perturbations; out new member state, energy
     books.

3. **Component-organised particle data.** A core set (position, velocity, mass, id, time-step level) plus fields
   registered by each module (hydro: h, rho, u, P, ...; stellar evolution: type, age, radius, ...), stored as
   structure of arrays or Cabana AoSoA. A gravity kernel never pulls hydro fields through the cache, and a
   module can be compiled out without touching the others.

4. **Share orderings, not structures.** One space-filling-curve (Morton) sort can define the memory order for
   every module, and one partition along the curve can define the MPI decomposition. Communication stays per
   module: gravity exchanges far-field information (multipoles or tree cells), hydro exchanges ghost particles
   within 2h, and the two never share buffers or code paths.

5. **Time integration by operator splitting with an explicit contract.** Each module reports its time-step
   constraint and provides kick (and, where needed, drift) operators for a given particle set and interval. The
   integrator owns the level hierarchy (HHS-FSI). Coupling errors are then a property of the splitting and can be
   analysed, instead of hiding in shared active-particle logic. Candidate schemes: fold gas into the HHS levels
   with a Courant constraint, or sub-cycle hydro inside gravity steps; to be decided by tests.

6. **GPU first, backend-specific variants behind one interface.** The neighbour search is the same algorithm on
   CPU and GPU (cell lists plus Verlet lists), with variants chosen per execution space:
   - GPU: full lists (no atomics), particles sorted by cell, streaming cell-pair kernels with the 27 neighbour
     cells in team scratch instead of stored lists when memory is tight (1e7 x 200 x 4 B = 8 GB), grouping by h
     or neighbour count against divergence.
   - CPU: half lists (each pair once, symmetric updates), CSR storage, SIMD over sorted neighbours.
   Cabana's neighbour lists expose exactly these choices (full / half, CSR / 2D, linked cell / Verlet).
   Strongly varying h: multi-level cell grids binned by h, still not the gravity tree.

7. **Every module is tested on its own first.** Gravity against direct summation (`fmm-test`, the floor and
   FROST tests); hydro on Sod, Sedov, Gresho, Kelvin-Helmholtz and Evrard collapse; coupling tests only after
   each module passes alone.

8. **No hidden global state.** Modules hold their own state and buffers, released explicitly before
   `Kokkos::finalize` (cpp-nbody had to retrofit finalize hooks for static Views).

## What cpp-nbody already follows and what to clean up first

Follows: the direct-sum driver and the FMM take positions and masses through narrow interfaces and know nothing
about particle types; collisional physics is compiled separately (`ENABLE_COLLISIONAL`).

Step zero before adding hydro:
- split `Particles` (core, collisional and scratch fields in one struct) into a core set plus per-module
  field groups;
- replace global managers (`g_subsystem_mgr`, `g_collisional`, `g_energy_books`, `g_octree`) with module
  objects owned by the simulation and passed to the integrator;
- give the integrator an explicit module interface (time-step constraint, kick, drift, energy contribution)
  so that gravity, subsystems and a future hydro module plug in the same way.
