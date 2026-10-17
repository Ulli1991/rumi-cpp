#pragma once
// Double-averaged hierarchical-triple Hamiltonian to octupole order after the elimination of the nodes (h1 - h2 = pi),
// Naoz et al. 2013 (MNRAS 431, 2155) appendix A (their eqs. C2, C3, A, B, cos phi and H(Delta h -> pi)); Blaes, Lee &
// Socrates 2002 eq. 41 is the same function with the opposite overall sign. Canonical pairs (g1, G1), (g2, G2) at
// fixed L1, L2 and total angular momentum Gtot, cos i = (Gtot^2 - G1^2 - G2^2) / (2 G1 G2). Naoz's sign: H = -energy,
// so dg/dt = -dH/dG, dG/dt = +dH/dg. Masses m1, m2 (inner), m3 (outer), G = 1 units; m1 - m2 > 0 with the inner
// eccentricity vector pointing to the pericentre of m2 relative to m1 (Naoz eq. Mj). Templated for long double central
// differences (tests/secular_triple_test.cpp) and Enzyme (tests/enzyme/secular_triple.cpp).
#include <cmath>

struct TripleHamParams {
    double m1, m2, m3;   // masses
    double L1, L2;       // Delaunay L of the inner and outer orbit
    double Gtot;         // total angular momentum
    int quad, oct;       // include the quadrupole, octupole term
};

template <class T>
T triple_hamiltonian(const TripleHamParams& p, T g1, T G1, T g2, T G2) {
    using std::cos; using std::sin; using std::sqrt; using std::pow;
    const T M12 = (T)p.m1 + (T)p.m2, Mt = M12 + (T)p.m3, m12 = (T)p.m1 * (T)p.m2, L1 = p.L1, L2 = p.L2, m3 = p.m3;
    const T C2 = T(1) / 16 * pow(M12, 7) / pow(Mt, 3) * pow(m3, 7) / pow(m12, 3) * pow(L1, 4) / (pow(L2, 3) * G2 * G2 * G2);
    const T C3 = -T(15) / 64 * pow(M12, 9) / pow(Mt, 4) * pow(m3, 9) * ((T)p.m1 - (T)p.m2) / pow(m12, 5)
                 * pow(L1, 6) / (pow(L2, 3) * pow(G2, 5));
    const T e1s = 1 - G1 * G1 / (L1 * L1), e2s = 1 - G2 * G2 / (L2 * L2);
    const T th = ((T)p.Gtot * (T)p.Gtot - G1 * G1 - G2 * G2) / (2 * G1 * G2), s2 = 1 - th * th;
    const T B = 2 + 5 * e1s - 7 * e1s * cos(2 * g1), A = 4 + 3 * e1s - T(5) / 2 * B * s2;
    const T cphi = -cos(g1) * cos(g2) - th * sin(g1) * sin(g2);
    T H = 0;
    if (p.quad) H += C2 * ((2 + 3 * e1s) * (3 * th * th - 1) + 15 * e1s * s2 * cos(2 * g1));
    if (p.oct) H += C3 * sqrt(e1s * e2s) * (A * cphi + 10 * th * s2 * (1 - e1s) * sin(g1) * sin(g2));
    return H;
}
