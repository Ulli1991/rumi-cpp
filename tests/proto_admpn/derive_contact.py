#!/usr/bin/env python3
# Derives the ADM -> modified-harmonic (MH) contact transformation of the two-body c.o.m. problem through 3PN and,
# by the same over-determined linear system, checks the transcription of both dynamics:
#   ADM: reduced Hamiltonian H(x, p) of Damour, Jaranowski & Schaefer 2000 (PRD 62, 044024, Eqs. 3.2-3.6) with
#        omega_static = 0, omega_kinetic = 41/24 (DJS 2001, dimensional regularization);
#   MH:  relative acceleration a = -(1/r^2)[(1 + A) n + B v] of Blanchet LRR 17, 2 (2014, v3 2024),
#        Eqs. (calAcoeff), (calBcoeff) at 1PN, 2PN and (ABcoeffMH) at 3PN (as src/collisional/post_newtonian_core.cpp).
# Ansatz (units G M = 1, eps = 1/c^2): x_h = x + eps^2 d2 + eps^3 d3 with
#   d2 = (a1 p^2 + a2 pr^2 + a3/r) n + a4 pr p,
#   d3 = (b1 p^4 + b2 p^2 pr^2 + b3 pr^4 + b4 p^2/r + b5 pr^2/r + b6/r^2) n + (b7 p^2 + b8 pr^2 + b9/r) pr p,
# v_h = d x_h / dt along the ADM flow, and the condition d v_h / dt = a_MH(x_h, v_h) through eps^3.
# Everything is written in the moving basis (n, p) with scalar coefficients of (r, pr, p2) (rotation covariance):
#   n' = 2 H_p2 (p - pr n) / r,  p' = -H_r n - H_pr (p - pr n) / r,
#   r' = 2 H_p2 pr + H_pr,  pr' = (2 H_p2 p2 + H_pr pr - r' pr) / r - H_r,  p2' = -2 [H_r pr + H_pr (p2 - pr^2) / r].
# The residual's n and p components are evaluated at random rational (r, pr, p2) points: 4 + 9 unknowns against
# 2 x (points) equations per order, so a solution exists only if the two dynamics are the same physics; a consistent
# solve checks every coefficient of H_3PN and of A_3PN, B_3PN (with the 1PN and 2PN ones).
# Run: python3 tests/proto_admpn/derive_contact.py   (sympy, about a minute); prints the C++ for contact_coeffs.h.
import random
import sympy as sp

eps, nu = sp.symbols('epsilon nu')
pi = sp.pi
r, pr, p2 = sp.symbols('r pr p2', positive=True)
a = sp.symbols('a1:5')
b = sp.symbols('b1:10')
u = 1 / r

H1 = sp.Rational(1, 8) * (3 * nu - 1) * p2**2 - sp.Rational(1, 2) * ((3 + nu) * p2 + nu * pr**2) * u + u**2 / 2
H2 = (sp.Rational(1, 16) * (1 - 5 * nu + 5 * nu**2) * p2**3
      + sp.Rational(1, 8) * ((5 - 20 * nu - 3 * nu**2) * p2**2 - 2 * nu**2 * pr**2 * p2 - 3 * nu**2 * pr**4) * u
      + sp.Rational(1, 2) * ((5 + 8 * nu) * p2 + 3 * nu * pr**2) * u**2
      - sp.Rational(1, 4) * (1 + 3 * nu) * u**3)
wk, ws = sp.Rational(41, 24), 0
H3 = (sp.Rational(1, 128) * (-5 + 35 * nu - 70 * nu**2 + 35 * nu**3) * p2**4
      + sp.Rational(1, 16) * ((-7 + 42 * nu - 53 * nu**2 - 5 * nu**3) * p2**3 + (2 - 3 * nu) * nu**2 * pr**2 * p2**2
                              + 3 * (1 - nu) * nu**2 * pr**4 * p2 - 5 * nu**3 * pr**6) * u
      + (sp.Rational(1, 16) * (-27 + 136 * nu + 109 * nu**2) * p2**2 + sp.Rational(1, 16) * (17 + 30 * nu) * nu * pr**2 * p2
         + sp.Rational(1, 12) * (5 + 43 * nu) * nu * pr**4) * u**2
      + ((-sp.Rational(25, 8) + (pi**2 / 64 - sp.Rational(335, 48)) * nu - sp.Rational(55, 12) * nu**2) * p2
         + (-sp.Rational(85, 16) - 3 * pi**2 / 64 + sp.Rational(27, 8) * nu) * nu * pr**2) * u**3
      + (sp.Rational(1, 8) + (sp.Rational(109, 12) - sp.Rational(21, 32) * pi**2) * nu) * u**4
      + wk * (p2 - 3 * pr**2) * nu**2 * u**3 + ws * nu * u**4)
H = p2 / 2 - u + eps * H1 + eps**2 * H2 + eps**3 * H3


def tr(e, order=3):
    e = sp.expand(e)
    return sum(e.coeff(eps, k) * eps**k for k in range(order + 1))


Hr, Hpr, Hp2 = sp.diff(H, r), sp.diff(H, pr), sp.diff(H, p2)
rdot = tr(2 * Hp2 * pr + Hpr)
p2dot = tr(-2 * (Hr * pr + Hpr * (p2 - pr**2) / r))
prdot = tr((2 * Hp2 * p2 + Hpr * pr - rdot * pr) / r - Hr)
# basis derivatives: n' = cnn n + cnp p, p' = cpn n + cpp p
cnn, cnp = tr(-2 * Hp2 * pr / r), tr(2 * Hp2 / r)
cpn, cpp = tr(-Hr + Hpr * pr / r), tr(-Hpr / r)


def ddt_scalar(f):
    return tr(sp.diff(f, r) * rdot + sp.diff(f, pr) * prdot + sp.diff(f, p2) * p2dot)


def ddt_vec(v):  # v = (alpha, beta) for alpha n + beta p
    al, be = v
    return (tr(ddt_scalar(al) + al * cnn + be * cpn), tr(ddt_scalar(be) + al * cnp + be * cpp))


d2s = a[0] * p2 + a[1] * pr**2 + a[2] * u
d3s = b[0] * p2**2 + b[1] * p2 * pr**2 + b[2] * pr**4 + b[3] * p2 * u + b[4] * pr**2 * u + b[5] * u**2
d2p = a[3] * pr
d3p = (b[6] * p2 + b[7] * pr**2 + b[8] * u) * pr
xh = (r + eps**2 * d2s + eps**3 * d3s, eps**2 * d2p + eps**3 * d3p)
print('building v_h, a_h ...', flush=True)
vh = ddt_vec(xh)
ah = ddt_vec(vh)
print('done', flush=True)


# truncated power series in eps (coefficients of eps^0..eps^3), sympy coefficients
class TP:
    def __init__(self, c):
        self.c = [sp.expand(x) for x in (list(c) + [0] * 4)[:4]]

    @staticmethod
    def of(e):
        e = sp.expand(e)
        return TP([e.coeff(eps, k) for k in range(4)])

    def __add__(self, o):
        o = o if isinstance(o, TP) else TP([o])
        return TP([x + y for x, y in zip(self.c, o.c)])
    __radd__ = __add__

    def __neg__(self):
        return TP([-x for x in self.c])

    def __sub__(self, o):
        return self + (-(o if isinstance(o, TP) else TP([o])))

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if not isinstance(o, TP):
            return TP([x * o for x in self.c])
        return TP([sum(self.c[i] * o.c[k - i] for i in range(k + 1)) for k in range(4)])
    __rmul__ = __mul__

    def pw(self, alpha):  # self^alpha, self.c[0] != 0
        c0 = self.c[0]
        w = TP([0] + [x / c0 for x in self.c[1:]])
        res, term = TP([1]), TP([1])
        for k in range(1, 4):
            term = term * w
            res = res + term * sp.binomial(alpha, k)
        return res * c0**alpha


def dot(v, w):  # (alpha n + beta p) . (gamma n + delta p)
    return v[0] * w[0] + (v[0] * w[1] + v[1] * w[0]) * pr0 + v[1] * w[1] * p20


def a_mh(x, v):
    q = dot(x, x)
    irr = q.pw(sp.Rational(-1, 2))
    v2 = dot(v, v)
    rd = dot(x, v) * irr
    m = irr
    rd2 = rd * rd
    E1, E2, E3 = TP([0, 1]), TP([0, 0, 1]), TP([0, 0, 0, 1])
    A1 = rd2 * (-sp.Rational(3, 2) * nu) + v2 * (1 + 3 * nu) - m * (4 + 2 * nu)
    B1 = rd * (-4 + 2 * nu)
    v4 = v2 * v2; rd4 = rd2 * rd2; m2 = m * m
    A2 = (rd4 * (sp.Rational(15, 8) * nu - sp.Rational(45, 8) * nu**2) + rd2 * v2 * (-sp.Rational(9, 2) * nu + 6 * nu**2)
          + v4 * (3 * nu - 4 * nu**2)
          + m * (rd2 * (-2 - 25 * nu - 2 * nu**2) + v2 * (-sp.Rational(13, 2) * nu + 2 * nu**2))
          + m2 * (9 + sp.Rational(87, 4) * nu))
    B2 = (rd2 * rd * (sp.Rational(9, 2) * nu + 3 * nu**2) + rd * v2 * (-sp.Rational(15, 2) * nu - 2 * nu**2)
          + m * rd * (2 + sp.Rational(41, 2) * nu + 4 * nu**2))
    # 3PN terms only needed at eps^0 of their argument (they enter at eps^3)
    R, V2, M = rd.c[0], v2.c[0], m.c[0]
    A3 = (-sp.Rational(35, 16) * R**6 * nu + sp.Rational(175, 16) * R**6 * nu**2 - sp.Rational(175, 16) * R**6 * nu**3
          + sp.Rational(15, 2) * R**4 * nu * V2 - sp.Rational(135, 4) * R**4 * nu**2 * V2 + sp.Rational(255, 8) * R**4 * nu**3 * V2
          - sp.Rational(15, 2) * R**2 * nu * V2**2 + sp.Rational(237, 8) * R**2 * nu**2 * V2**2 - sp.Rational(45, 2) * R**2 * nu**3 * V2**2
          + sp.Rational(11, 4) * nu * V2**3 - sp.Rational(49, 4) * nu**2 * V2**3 + 13 * nu**3 * V2**3
          + M * (79 * R**4 * nu - sp.Rational(69, 2) * R**4 * nu**2 - 30 * R**4 * nu**3 - 121 * R**2 * nu * V2
                 + 16 * R**2 * nu**2 * V2 + 20 * R**2 * nu**3 * V2 + sp.Rational(75, 4) * nu * V2**2 + 8 * nu**2 * V2**2
                 - 10 * nu**3 * V2**2)
          + M**2 * (R**2 + sp.Rational(22717, 168) * R**2 * nu + sp.Rational(11, 8) * R**2 * nu**2 - 7 * R**2 * nu**3
                    + sp.Rational(615, 64) * R**2 * nu * pi**2 - sp.Rational(20827, 840) * nu * V2 + nu**3 * V2
                    - sp.Rational(123, 64) * nu * pi**2 * V2)
          + M**3 * (-16 - sp.Rational(1399, 12) * nu - sp.Rational(71, 2) * nu**2 + sp.Rational(41, 16) * nu * pi**2))
    B3 = (-sp.Rational(45, 8) * R**5 * nu + 15 * R**5 * nu**2 + sp.Rational(15, 4) * R**5 * nu**3 + 12 * R**3 * nu * V2
          - sp.Rational(111, 4) * R**3 * nu**2 * V2 - 12 * R**3 * nu**3 * V2 - sp.Rational(65, 8) * R * nu * V2**2
          + 19 * R * nu**2 * V2**2 + 6 * R * nu**3 * V2**2
          + M * (sp.Rational(329, 6) * R**3 * nu + sp.Rational(59, 2) * R**3 * nu**2 + 18 * R**3 * nu**3 - 15 * R * nu * V2
                 - 27 * R * nu**2 * V2 - 10 * R * nu**3 * V2)
          + M**2 * (-4 * R - sp.Rational(5849, 840) * R * nu + 25 * R * nu**2 + 8 * R * nu**3
                    - sp.Rational(123, 32) * R * nu * pi**2))
    A = E1 * A1 + E2 * A2 + E3 * A3
    B = E1 * B1 + E2 * B2 + E3 * B3
    f = (irr * irr) * (-1)
    one = TP([1])
    return (f * ((one + A) * x[0] * irr + B * v[0]), f * ((one + A) * x[1] * irr + B * v[1]))


random.seed(1)
res = {2: [], 3: []}
for k in range(12):
    rv = sp.Rational(random.randint(10, 60), 10)
    prv = sp.Rational(random.randint(-8, 8), 10)
    p2v = prv**2 + sp.Rational(random.randint(1, 9), 10)
    pt = {r: rv, pr: prv, p2: p2v}
    pr0, p20 = prv, p2v
    xs = (TP.of(xh[0].subs(pt)), TP.of(xh[1].subs(pt)))
    vs = (TP.of(vh[0].subs(pt)), TP.of(vh[1].subs(pt)))
    al = (TP.of(ah[0].subs(pt)), TP.of(ah[1].subs(pt)))
    ar = a_mh(xs, vs)
    for comp in range(2):
        d = al[comp] - ar[comp]
        for order in (0, 1):
            c = sp.simplify(d.c[order])
            assert c == 0, f'order {order} residual nonzero: {c}'
        res[2].append(d.c[2])
        res[3].append(d.c[3])
    print('point', k, 'done', flush=True)

sol2 = sp.solve(res[2], a, dict=True)
print('2PN solution:', sol2)
assert sol2, '2PN system inconsistent'
s2 = sol2[0]
eq3 = [sp.expand(e.subs(s2)) for e in res[3]]
sol3 = sp.solve(eq3, b, dict=True)
print('3PN solution:', sol3)
assert sol3, '3PN system inconsistent'
s3 = sol3[0]
print('consistent: the ADM 3PN Hamiltonian and the MH 3PN acceleration are related by the contact transformation')
print('// ---- C++ (contact_coeffs.h)')
for i, key in enumerate(a):
    print(f'    K.a[{i}] = {sp.ccode(sp.expand(s2[key]))};')
for i, key in enumerate(b):
    print(f'    K.b[{i}] = {sp.ccode(sp.expand(s3[key]))};')
