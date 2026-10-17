// Secular triple rates (secular_evolve_triple) against Hamilton's equations of the double-averaged quadrupole + octupole
// Hamiltonian (tests/secular_triple_ham.h, Naoz et al. 2013 app. A) with the exact gradient from Enzyme:
// dg/dt = -dH/dG, dG/dt = dH/dg (H = -energy), de/dt = -G/(L^2 e) dG/dt, d cos i/dt through cos i(G1, G2, Gtot).
// Prints the maximum difference per term over random states, relative to the sum of the rates; the plain C++ ctest
// with long double differences is tests/secular_triple_test.cpp.
//   clang++ -std=c++20 -O2 -fplugin=ClangEnzyme-18.so -Isrc/collisional tests/enzyme/secular_triple.cpp \
//           src/collisional/secular.cpp src/collisional/kepler.cpp src/collisional/post_newtonian_core.cpp
#include "secular.h"
#include "../secular_triple_ham.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

extern int enzyme_dup, enzyme_const;
template <typename... T> void __enzyme_autodiff(void*, T...);

static double ham(const double* q, const TripleHamParams* p) { return triple_hamiltonian<double>(*p, q[0], q[1], q[2], q[3]); }

int main() {
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (int oct = 0; oct < 2; oct++) {
        double mx[5] = {0, 0, 0, 0, 0};
        for (int t = 0; t < 2000; t++) {
            const double m1 = 0.1 + 2 * U(rng), m3 = 0.1 + 2 * U(rng), m2 = oct ? 0.1 + 2 * U(rng) : m1;
            const double a1 = 0.01 + 0.09 * U(rng), e1 = 0.05 + 0.9 * U(rng), e2 = 0.05 + 0.85 * U(rng);
            const double ci = -0.99 + 1.98 * U(rng), g1 = 2 * M_PI * U(rng), g2 = 2 * M_PI * U(rng);
            const double M12 = m1 + m2, Mt = M12 + m3;
            TripleHamParams p{m1, m2, m3, m1 * m2 / M12 * std::sqrt(M12 * a1), M12 * m3 / Mt * std::sqrt(Mt), 0.0, 1, 1};
            const double G1 = p.L1 * std::sqrt(1 - e1 * e1), G2 = p.L2 * std::sqrt(1 - e2 * e2);
            p.Gtot = std::sqrt(G1 * G1 + G2 * G2 + 2 * G1 * G2 * ci);
            double q[4] = {g1, G1, g2, G2}, dq[4] = {0, 0, 0, 0};
            __enzyme_autodiff((void*)ham, enzyme_dup, q, dq, enzyme_const, &p);
            const double rg1 = -dq[1], rg2 = -dq[3], rG1 = dq[0], rG2 = dq[2];
            const double re1 = -G1 / (p.L1 * p.L1 * e1) * rG1, re2 = -G2 / (p.L2 * p.L2 * e2) * rG2;
            const double rci = (-1 / G2 - ci / G1) * rG1 + (-1 / G1 - ci / G2) * rG2;
            const double scale = std::fabs(rg1) + std::fabs(rg2) + std::fabs(re1) + std::fabs(re2);
            SecularTriple tri{};
            tri.inner.a = a1; tri.inner.e = e1; tri.inner.omega = g1; tri.inner.m1 = m1; tri.inner.m2 = m2;
            tri.a_out = 1.0; tri.e_out = e2; tri.omega_out = g2; tri.m3 = m3; tri.cos_i_mut = ci;
            SecularTriple t0 = tri, t2 = tri;
            const double dt = 1e-6 / scale;
            secular_evolve_triple(tri, dt, 1.0, 1e30, 0);
            secular_evolve_triple(t2, 2 * dt, 1.0, 1e30, 0);
            const double d[5] = {(tri.inner.e - t0.inner.e) / dt - re1, (tri.inner.omega - t0.inner.omega) / dt - rg1,
                                 (tri.omega_out - t0.omega_out) / dt - rg2, (tri.e_out - t0.e_out) / dt - re2,
                                 2 * (tri.cos_i_mut - t0.cos_i_mut) / dt - (t2.cos_i_mut - t0.cos_i_mut) / (2 * dt) - rci};
            for (int k = 0; k < 5; k++) mx[k] = std::max(mx[k], std::fabs(d[k]) / scale);
        }
        std::printf("%s: max |code - Hamilton| / sum|rates|: de1 %.2e dg1 %.2e dg2 %.2e de2 %.2e dcos_i %.2e\n",
                    oct ? "octupole  " : "quadrupole", mx[0], mx[1], mx[2], mx[3], mx[4]);
    }
}
