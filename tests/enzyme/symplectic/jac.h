// Jacobian helpers: Enzyme forward mode, central differences, symplectic defect, det, reversibility
#pragma once
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

extern int enzyme_dup, enzyme_const, enzyme_dupnoneed;
template <typename... T> void __enzyme_fwddiff(void*, T...);

constexpr int DMAX = 64;
typedef void (*MapFn)(const double* in, double* out);

// M[i][j] = d out_i / d in_j via Enzyme forward mode (one tangent per column)
template <MapFn F>
void jac_enzyme(const double* z, int d, double M[DMAX][DMAX]) {
    for (int j = 0; j < d; j++) {
        double dz[DMAX] = {}, out[DMAX], dout[DMAX] = {};
        dz[j] = 1.0;
        __enzyme_fwddiff((void*)F, enzyme_dup, z, dz, enzyme_dup, out, dout);
        for (int i = 0; i < d; i++) M[i][j] = dout[i];
    }
}
// central differences, 4th order (Richardson), step h_j = hrel * max(|z_j|, scale)
inline void jac_fd(MapFn F, const double* z, int d, double M[DMAX][DMAX], double hrel = 1e-4, double scale = 1.0) {
    for (int j = 0; j < d; j++) {
        double h = hrel * std::max(std::fabs(z[j]), scale);
        double zp[DMAX], o[4][DMAX];
        const double sh[4] = {h, -h, 2 * h, -2 * h};
        for (int k = 0; k < 4; k++) {
            std::memcpy(zp, z, d * sizeof(double));
            zp[j] += sh[k];
            F(zp, o[k]);
        }
        for (int i = 0; i < d; i++)
            M[i][j] = (8.0 * (o[0][i] - o[1][i]) - (o[2][i] - o[3][i])) / (12.0 * h);
    }
}
// J = [[0, I], [-I, 0]] with z = (q_1..q_m, p_1..p_m), d = 2m
inline double Jel(int i, int j, int m) {
    if (i < m && j == i + m) return 1.0;
    if (i >= m && j == i - m) return -1.0;
    return 0.0;
}
// max |(M^T J M - J)_ij| / max(1, max|M|^2)  (relative to the size of the products formed)
inline double symp_defect(const double M[DMAX][DMAX], int d, double* absolute = nullptr) {
    const int m = d / 2;
    double worst = 0.0, mmax = 0.0;
    for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) mmax = std::max(mmax, std::fabs(M[i][j]));
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++) {
            long double s = 0.0L;
            for (int k = 0; k < m; k++) s += (long double)M[k][i] * M[k + m][j] - (long double)M[k + m][i] * M[k][j];
            worst = std::max(worst, (double)std::fabs(s - Jel(i, j, m)));
        }
    if (absolute) *absolute = worst;
    return worst / std::max(1.0, mmax * mmax);
}
// defect restricted to the tangent space orthogonal to g (energy shell): W^T (M^T J M - J) W, W orthonormal basis of g-perp
inline double symp_defect_shell(const double M[DMAX][DMAX], int d, const double* g) {
    const int m = d / 2;
    static double A[DMAX][DMAX];
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++) {
            long double s = 0.0L;
            for (int k = 0; k < m; k++) s += (long double)M[k][i] * M[k + m][j] - (long double)M[k + m][i] * M[k][j];
            A[i][j] = (double)(s - Jel(i, j, m));
        }
    // basis: Gram-Schmidt of {g, e_1..e_d}, drop g
    std::vector<std::vector<double>> B;
    std::vector<double> gn(g, g + d);
    double nn = 0; for (double x : gn) nn += x * x; nn = std::sqrt(nn);
    for (double& x : gn) x /= nn;
    B.push_back(gn);
    for (int e = 0; e < d && (int)B.size() < d; e++) {
        std::vector<double> u(d, 0.0); u[e] = 1.0;
        for (int pass = 0; pass < 2; pass++)
            for (auto& b : B) { double s = 0; for (int i = 0; i < d; i++) s += b[i] * u[i]; for (int i = 0; i < d; i++) u[i] -= s * b[i]; }
        double n2 = 0; for (double x : u) n2 += x * x; n2 = std::sqrt(n2);
        if (n2 < 1e-8) continue;
        for (double& x : u) x /= n2;
        B.push_back(u);
    }
    double worst = 0.0, mmax = 0.0;
    for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) mmax = std::max(mmax, std::fabs(M[i][j]));
    for (size_t a = 1; a < B.size(); a++)
        for (size_t b = 1; b < B.size(); b++) {
            double s = 0;
            for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) s += B[a][i] * A[i][j] * B[b][j];
            worst = std::max(worst, std::fabs(s));
        }
    return worst / std::max(1.0, mmax * mmax);
}
inline double det(const double Min[DMAX][DMAX], int d) {
    static long double A[DMAX][DMAX];
    for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) A[i][j] = Min[i][j];
    long double D = 1.0L;
    for (int c = 0; c < d; c++) {
        int p = c;
        for (int r = c + 1; r < d; r++) if (std::fabs((double)A[r][c]) > std::fabs((double)A[p][c])) p = r;
        if (A[p][c] == 0.0L) return 0.0;
        if (p != c) { for (int k = 0; k < d; k++) std::swap(A[p][k], A[c][k]); D = -D; }
        D *= A[c][c];
        for (int r = c + 1; r < d; r++) {
            long double f = A[r][c] / A[c][c];
            for (int k = c; k < d; k++) A[r][k] -= f * A[c][k];
        }
    }
    return (double)D;
}
inline double max_diff(const double A[DMAX][DMAX], const double B[DMAX][DMAX], int d) {
    double w = 0, s = 0;
    for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) { w = std::max(w, std::fabs(A[i][j] - B[i][j])); s = std::max(s, std::fabs(B[i][j])); }
    return w / std::max(1.0, s);
}
// || Mb Ma - I ||_max
inline double prod_minus_I(const double Mb[DMAX][DMAX], const double Ma[DMAX][DMAX], int d) {
    double w = 0;
    for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) {
        long double s = 0; for (int k = 0; k < d; k++) s += (long double)Mb[i][k] * Ma[k][j];
        w = std::max(w, (double)std::fabs(s - (i == j ? 1.0L : 0.0L)));
    }
    return w;
}
