#pragma once
// Solid harmonics for the spherical FMM (tools/fmm). Normalisation (Dehnen 2014, CompAC 1:1):
//   regular   Y_n^m(x) = r^n / (n+m)!     P_n^m(cos t) e^{i m phi}
//   irregular T_n^m(x) = (n-m)! / r^{n+1} P_n^m(cos t) e^{i m phi}
// with P_n^m the associated Legendre function without the Condon-Shortley phase, extended to m < 0 by
// Y_n^{-m} = (-1)^m conj(Y_n^m) and T_n^{-m} = (-1)^m conj(T_n^m). The identities used by the FMM are
// checked numerically in harm_test.cpp; the computation uses Cartesian recurrences (no angles).
#include <complex>
#include <vector>

using cplx = std::complex<double>;

inline int hidx(int n, int m) { return n * n + n + m; }   // m in [-n, n]
inline int hsize(int p) { return (p + 1) * (p + 1); }

// Regular solid harmonics Y_n^m(x) for n <= p (all m). Recurrences for r^n/(n+m)! P_n^m e^{im phi}:
//   Y_m^m = (x + i y) / (2m) Y_{m-1}^{m-1}                         (no Condon-Shortley phase)
//   Y_{m+1}^m = z Y_m^m
//   (n+m) (n-m) Y_n^m ... from (n-m) P_n^m = (2n-1) cos P_{n-1}^m - (n+m-1) P_{n-2}^m, scaled:
//   Y_n^m = [ (2n-1) z Y_{n-1}^m - r^2 Y_{n-2}^m ] / ((n+m)(n-m))
inline void regular(const double x[3], int p, cplx* Y) {
    const double r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    const cplx w(x[0], x[1]);
    Y[hidx(0, 0)] = 1.0;
    for (int m = 1; m <= p; m++) Y[hidx(m, m)] = Y[hidx(m - 1, m - 1)] * w / (2.0 * m);
    for (int m = 0; m < p; m++) Y[hidx(m + 1, m)] = x[2] * Y[hidx(m, m)];
    for (int m = 0; m <= p; m++)
        for (int n = m + 2; n <= p; n++)
            Y[hidx(n, m)] = ((2.0 * n - 1.0) * x[2] * Y[hidx(n - 1, m)] - r2 * Y[hidx(n - 2, m)]) / ((double)(n + m) * (n - m));
    for (int n = 1; n <= p; n++)
        for (int m = 1; m <= n; m++) Y[hidx(n, -m)] = ((m & 1) ? -1.0 : 1.0) * std::conj(Y[hidx(n, m)]);
}

// Irregular solid harmonics T_n^m(x) for n <= p. With P recurrences and r^{-(n+1)} (n-m)!:
//   T_m^m = (2m-1) (x + i y) / r^2 T_{m-1}^{m-1},  T_0^0 = 1/r
//   T_{m+1}^m = (2m+1) z / r^2 T_m^m
//   T_n^m = [ (2n-1) z T_{n-1}^m - (n-1+m)(n-1-m) T_{n-2}^m ] / r^2
inline void irregular(const double x[3], int p, cplx* T) {
    const double r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    const double ir2 = 1.0 / r2;
    const cplx w(x[0], x[1]);
    T[hidx(0, 0)] = std::sqrt(ir2);
    for (int m = 1; m <= p; m++) T[hidx(m, m)] = (2.0 * m - 1.0) * w * ir2 * T[hidx(m - 1, m - 1)];
    for (int m = 0; m < p; m++) T[hidx(m + 1, m)] = (2.0 * m + 1.0) * x[2] * ir2 * T[hidx(m, m)];
    for (int m = 0; m <= p; m++)
        for (int n = m + 2; n <= p; n++)
            T[hidx(n, m)] = ((2.0 * n - 1.0) * x[2] * T[hidx(n - 1, m)] - (double)(n - 1 + m) * (n - 1 - m) * T[hidx(n - 2, m)]) * ir2;
    for (int n = 1; n <= p; n++)
        for (int m = 1; m <= n; m++) T[hidx(n, -m)] = ((m & 1) ? -1.0 : 1.0) * std::conj(T[hidx(n, m)]);
}

// Rotations of degree-n solid harmonics: Y_n(Q x) = S_n(Q) Y_n(x) (2n+1 x 2n+1). For rotations about z by
// alpha S is diagonal, e^{i m alpha}. Rotations about x by +-90 degrees are fitted once from sample points
// (convention-proof); a rotation about y by beta is Rx(-90) Rz(beta) Rx(90). Irregular harmonics rotate with
// D S D^{-1}, D = diag((n-m)! (n+m)!), since T_n^m = (n-m)!(n+m)! Y_n^m / r^{2n+1}.
struct HarmonicRotator {
    int p;
    std::vector<std::vector<cplx>> Sx90, Sxm90;   // [n][(m+n)*(2n+1) + (m'+n)]
    static void rotx(double ang, const double x[3], double y[3]) {
        double c = std::cos(ang), s = std::sin(ang);
        y[0] = x[0]; y[1] = c * x[1] - s * x[2]; y[2] = s * x[1] + c * x[2];
    }
    // S_{m m'} = int Y_n^m(Q x) conj(Y_n^{m'}(x)) dOmega / int |Y_n^{m'}|^2 dOmega on the unit sphere, by a
    // Gauss-Legendre (cos theta) x uniform (phi) product rule that is exact for these degree-2n integrands
    static std::vector<cplx> fit(int n, double ang) {
        const int d = 2 * n + 1, nmu = n + 2, nphi = 2 * n + 3;
        std::vector<double> mu(nmu), w(nmu);
        for (int i = 0; i < nmu; i++) {   // Legendre roots by Newton
            double x = std::cos(M_PI * (i + 0.75) / (nmu + 0.5)), dp = 0;
            for (int it = 0; it < 100; it++) {
                double p0 = 1, p1 = x;
                for (int k = 2; k <= nmu; k++) { double p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
                dp = nmu * (x * p1 - p0) / (x * x - 1);
                double dx = p1 / dp;
                x -= dx;
                if (std::abs(dx) < 1e-16) break;
            }
            mu[i] = x; w[i] = 2.0 / ((1 - x * x) * dp * dp);
        }
        std::vector<cplx> S(d * d, 0.0), Yx(hsize(n)), Yq(hsize(n));
        std::vector<double> norm(d, 0.0);
        for (int i = 0; i < nmu; i++)
            for (int j = 0; j < nphi; j++) {
                double st = std::sqrt(std::max(0.0, 1 - mu[i] * mu[i])), ph = 2 * M_PI * j / nphi;
                double x[3] = {st * std::cos(ph), st * std::sin(ph), mu[i]}, q[3];
                rotx(ang, x, q);
                regular(x, n, Yx.data());
                regular(q, n, Yq.data());
                double wt = w[i] * 2 * M_PI / nphi;
                for (int mp = -n; mp <= n; mp++) {
                    cplx cy = std::conj(Yx[hidx(n, mp)]) * wt;
                    norm[mp + n] += std::norm(Yx[hidx(n, mp)]) * wt;
                    for (int m = -n; m <= n; m++) S[(m + n) * d + mp + n] += Yq[hidx(n, m)] * cy;
                }
            }
        for (int m = 0; m < d; m++) for (int mp = 0; mp < d; mp++) S[m * d + mp] /= norm[mp];
        return S;
    }
    explicit HarmonicRotator(int p_) : p(p_), Sx90(p_ + 1), Sxm90(p_ + 1) {
        for (int n = 0; n <= p; n++) { Sx90[n] = fit(n, M_PI / 2); Sxm90[n] = fit(n, -M_PI / 2); }
    }
};

// Real form for expansions of real distributions (v_n^{-m} = (-1)^m conj(v_n^m)): degree n is held as 2n+1
// reals (a_0, a_1, b_1, ..., a_n, b_n), v_n^m = a_m + i b_m. A complex linear map of the coefficients that
// preserves this symmetry becomes a real (2n+1)^2 matrix, built by applying it to unit vectors; zeros (parity)
// are dropped (sparse rows).
struct RealMap {
    std::vector<std::vector<int>> col;      // [n] concatenated rows: column indices
    std::vector<std::vector<double>> val;   // [n] values
    std::vector<std::vector<int>> row_ptr;  // [n] size 2n+2
    // map(n, v_full_degree_n (2n+1 complex, index m+n)) -> out_full
    template <class F>
    void build(int p, F map) {
        col.assign(p + 1, {}); val.assign(p + 1, {}); row_ptr.assign(p + 1, {});
        for (int n = 0; n <= p; n++) {
            const int d = 2 * n + 1;
            std::vector<double> Mr(d * d, 0.0);
            for (int c = 0; c < d; c++) {
                std::vector<cplx> v(d, 0.0), out(d, 0.0);
                int m = (c + 1) / 2;
                cplx unit = (c == 0) ? cplx(1, 0) : ((c & 1) ? cplx(1, 0) : cplx(0, 1));
                v[m + n] = unit;
                if (m > 0) v[-m + n] = ((m & 1) ? -1.0 : 1.0) * std::conj(unit);
                map(n, v.data(), out.data());
                for (int r = 0; r < d; r++) {
                    int mo = (r + 1) / 2;
                    cplx o = out[mo + n];
                    Mr[r * d + c] = (r == 0) ? o.real() : ((r & 1) ? o.real() : o.imag());
                }
            }
            double mx = 0;
            for (double x : Mr) mx = std::max(mx, std::abs(x));
            row_ptr[n].push_back(0);
            for (int r = 0; r < d; r++) {
                for (int c = 0; c < d; c++)
                    if (std::abs(Mr[r * d + c]) > 1e-14 * mx) { col[n].push_back(c); val[n].push_back(Mr[r * d + c]); }
                row_ptr[n].push_back((int)col[n].size());
            }
        }
    }
    void apply(int n, const double* in, double* out) const {
        const int d = 2 * n + 1;
        const int* rp = row_ptr[n].data(); const int* ci = col[n].data(); const double* vv = val[n].data();
        for (int r = 0; r < d; r++) {
            double acc = 0;
            for (int k = rp[r]; k < rp[r + 1]; k++) acc += vv[k] * in[ci[k]];
            out[r] = acc;
        }
    }
};
