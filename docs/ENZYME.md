# Enzyme autodiff tests

`tests/enzyme/` differentiates the production kernels and integrator steps with [Enzyme](https://enzyme.mit.edu)
(LLVM plugin; forward, reverse, batched forward and nested modes) and checks the results against the code's
analytic derivatives, symplecticity and reversibility. Most tests `#include` the source they differentiate, so they
follow the code; `tests/enzyme/symplectic/symp_fsi.cpp` mirrors the FSI/HHS kernels (pair formulas verbatim).

## Plugin build (Enzyme v0.0.302, clang/LLVM 18)

```
module load clang/18 cmake
git clone https://github.com/EnzymeAD/Enzyme && cd Enzyme && git checkout v0.0.302
cmake -S enzyme -B build18 -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=$(which clang) -DCMAKE_CXX_COMPILER=$(which clang++) \
      -DLLVM_DIR=<clang-18 prefix>/lib/cmake/llvm
cmake --build build18 -j 8          # -> build18/Enzyme/ClangEnzyme-18.so
```

The plugin only loads into the clang it was built against (18.1.8 on the MPCDF systems).

## ctest

```
cmake -S . -B build ... -DRUMI_ENZYME_TESTS=ON \
      -DENZYME_CLANG=<clang-18 prefix>/bin/clang++ -DENZYME_PLUGIN=<...>/build18/Enzyme/ClangEnzyme-18.so
cmake --build build --target enzyme-tests && (cd build && ctest -R enzyme)
```

The tests are compiled by `ENZYME_CLANG` (not the project compiler) with `-fopenmp` and the Kokkos headers of the
configured Kokkos (header-only use; no Kokkos library is linked). Registered (exit status = number of failed checks):

| ctest | what |
|---|---|
| `enzyme_gradient_numerics` | spline softening g = -w'/r, k = g'/r, continuity; 2-body perturbation rates (pert_rates.h) = d/dt of the code's E, a, \|e\| for 1e-10 <= e <= 346; FSI GradK = J a; monopole kernel = Hessian, dipole kernel = third derivative of -1/s; tree cell term vs J a (dipole: first order in s/D); fwd = rev = batched fwd |
| `enzyme_grad_force` | FSI gradient force = GRAD_COEFF dt^2 x (1/m_i) d/dx_i of sum m \|a\|^2 / 2 |
| `enzyme_logh_step` | d(dE)/dx of a LogH + GBS step vs central differences (Newtonian, PN); forward vs reverse |
| `enzyme_spin_consistency`, `enzyme_pn_rr_balance` | spin PN and radiation-reaction energy balance (docs/CASES.md) |
| `enzyme_symp_fsi` | FSI and one HHS level: M^T J M = J, reversibility, P and L; map jump where a level flips (O(dt^5)); d(dE)/d(dt_k) by reverse vs forward mode, local order 5 |
| `enzyme_symp_kepler`, `enzyme_symp_kepler_parabolic` | kepler_drift Jacobian symplectic, E/L/rotation equivariance, near-parabolic scan |

The other files (`symp_logh`, `symp_gbs`, `symp_sec`, `ens`, `pnrev`, `pn_consistency`, `secular_triple`,
`worst_step`) are diagnostic tools that print tables; build lines are in their headers.

Known AD artifact: where kepler_drift's Laguerre-Conway loop leaves at F == 0 on the initial guess (e = 0, short
steps), Enzyme returns the guess's tangent (symplectic defect ~3e-8) while the value is the root (central-difference
defect ~1e-11); `symp_kepler` allows 1e-7 there.
