// Numerical checks of the solid-harmonic identities used by fmm_sph.cpp.
//   g++ -O2 -std=c++17 -o harm_test harm_test.cpp && ./harm_test
#include "harmonics.h"
#include <cmath>
#include <cstdio>
#include <random>

static double fact(int n) { return std::tgamma(n + 1.0); }

int main() {
    const int P = 10;
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> U(-1, 1);
    std::vector<cplx> Ya(hsize(2 * P)), Yb(hsize(2 * P)), Yab(hsize(2 * P)), Ta(hsize(2 * P)), Tab(hsize(2 * P));

    // 1. recurrences vs definition (std::assoc_legendre has no Condon-Shortley phase)
    double x[3] = {0.3, -0.5, 0.7};
    regular(x, P, Ya.data());
    irregular(x, P, Ta.data());
    double r = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]), ct = x[2] / r, ph = std::atan2(x[1], x[0]);
    double e1 = 0, e2 = 0;
    for (int n = 0; n <= P; n++)
        for (int m = 0; m <= n; m++) {
            cplx e = std::polar(1.0, m * ph) * std::assoc_legendre(n, m, ct);
            cplx y = std::pow(r, n) / fact(n + m) * e, t = fact(n - m) / std::pow(r, n + 1) * e;
            e1 = std::max(e1, std::abs(y - Ya[hidx(n, m)]) / std::abs(y));
            e2 = std::max(e2, std::abs(t - Ta[hidx(n, m)]) / std::abs(t));
        }
    std::printf("1. recurrences vs definition: regular %.1e irregular %.1e\n", e1, e2);

    // 2. addition theorem 1/|x - y| = sum conj(Y_n^m(y)) T_n^m(x), |y| < |x|
    double X[3] = {1.1, 0.4, -0.8}, Yv[3] = {0.1, -0.15, 0.12};
    regular(Yv, P, Ya.data());
    irregular(X, P, Ta.data());
    cplx s = 0, s2 = 0;
    for (int n = 0; n <= P; n++)
        for (int m = -n; m <= n; m++) { s += std::conj(Ya[hidx(n, m)]) * Ta[hidx(n, m)]; s2 += Ya[hidx(n, m)] * Ta[hidx(n, m)]; }
    double d[3] = {X[0] - Yv[0], X[1] - Yv[1], X[2] - Yv[2]};
    double exact = 1.0 / std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    std::printf("2. addition: sum conj(Y)T rel err %.1e (imag %.1e) | sum Y T rel err %.1e\n",
                std::abs(s.real() - exact) / exact, s.imag(), std::abs(s2.real() - exact) / exact);

    // 3. regular translation Y_n^m(a+b) = sum_{k,l} Y_k^l(a) Y_{n-k}^{m-l}(b)
    double a[3] = {0.2, 0.1, -0.3}, b[3] = {-0.4, 0.25, 0.15}, ab[3] = {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
    regular(a, P, Ya.data());
    regular(b, P, Yb.data());
    regular(ab, P, Yab.data());
    double e3 = 0;
    for (int n = 0; n <= P; n++)
        for (int m = -n; m <= n; m++) {
            cplx v = 0;
            for (int k = 0; k <= n; k++)
                for (int l = -k; l <= k; l++) {
                    int nk = n - k, ml = m - l;
                    if (std::abs(ml) > nk) continue;
                    v += Ya[hidx(k, l)] * Yb[hidx(nk, ml)];
                }
            e3 = std::max(e3, std::abs(v - Yab[hidx(n, m)]) / (std::abs(Yab[hidx(n, m)]) + 1e-300));
        }
    std::printf("3. regular translation: max rel err %.1e\n", e3);

    // 4. irregular translation T_n^m(R + y) = sum_{k,l} c(k,l) Y_k^l(y)^(*) T_{n+k}^{m+l}(R), |y| < |R|
    double R[3] = {1.2, -0.7, 0.9}, yy[3] = {0.08, 0.05, -0.11}, Ry[3] = {R[0] + yy[0], R[1] + yy[1], R[2] + yy[2]};
    const int PP = 2 * P;
    std::vector<cplx> Yy(hsize(PP)), TR(hsize(PP)), TRy(hsize(PP));
    regular(yy, PP, Yy.data());
    irregular(R, PP, TR.data());
    irregular(Ry, PP, TRy.data());
    for (int variant = 0; variant < 4; variant++) {
        double e4 = 0;
        for (int n = 0; n <= 4; n++)
            for (int m = -n; m <= n; m++) {
                cplx v = 0;
                for (int k = 0; k + n <= PP; k++)
                    for (int l = -k; l <= k; l++) {
                        int nk = n + k, ml = m + l;
                        if (std::abs(ml) > nk) continue;
                        cplx yk = (variant & 1) ? std::conj(Yy[hidx(k, l)]) : Yy[hidx(k, l)];
                        double sg = (variant & 2) ? ((k & 1) ? -1.0 : 1.0) : 1.0;
                        v += sg * yk * TR[hidx(nk, ml)];
                    }
                e4 = std::max(e4, std::abs(v - TRy[hidx(n, m)]) / std::abs(TRy[hidx(n, m)]));
            }
        std::printf("4. irregular translation variant %d (%s, %s): max rel err %.1e\n", variant,
                    (variant & 1) ? "conj(Y)" : "Y", (variant & 2) ? "(-1)^k" : "+1", e4);
    }

    // 5. gradient of regular harmonics, finite differences
    double h = 1e-6, e5[3] = {0, 0, 0};
    double x0[3] = {0.31, -0.27, 0.44};
    std::vector<cplx> Y0(hsize(P)), Yp(hsize(P)), Ym(hsize(P));
    regular(x0, P, Y0.data());
    for (int dim = 0; dim < 3; dim++) {
        double xp[3] = {x0[0], x0[1], x0[2]}, xm[3] = {x0[0], x0[1], x0[2]};
        xp[dim] += h; xm[dim] -= h;
        regular(xp, P, Yp.data());
        regular(xm, P, Ym.data());
        for (int n = 1; n <= P; n++)
            for (int m = -n; m <= n; m++) {
                cplx fd = (Yp[hidx(n, m)] - Ym[hidx(n, m)]) / (2 * h);
                auto get = [&](int nn, int mm) { return std::abs(mm) > nn ? cplx(0) : Y0[hidx(nn, mm)]; };
                cplx an;
                if (dim == 2) an = get(n - 1, m);
                else if (dim == 0) an = 0.5 * (get(n - 1, m - 1) - get(n - 1, m + 1));
                else an = cplx(0, 0.5) * (get(n - 1, m - 1) + get(n - 1, m + 1));
                e5[dim] = std::max(e5[dim], std::abs(fd - an) / (std::abs(fd) + 1e-12));
            }
    }
    std::printf("5. gradient: dz Y_n^m = Y_{n-1}^m %.1e, dx = (Y^{m-1} - Y^{m+1})/2 %.1e, dy = i(Y^{m-1} + Y^{m+1})/2 %.1e\n",
                e5[2], e5[0], e5[1]);
    return 0;
}
