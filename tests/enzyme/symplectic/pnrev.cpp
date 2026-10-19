// PN aux-velocity leapfrog: reversibility and det on the full state (q, v, w, t, B); w is part of the state
// Build (from the repository root; logh_ctrl.cpp: python3.11 tests/enzyme/symplectic/make_ctrl.py <dir>):
//   clang++ -std=c++20 -O2 -fplugin=ClangEnzyme-18.so -Isrc/collisional tests/enzyme/symplectic/pnrev.cpp -o pnrev
//   (Enzyme)
#define main main_unused
#include "symp_logh.cpp"
#undef main
static int g_ord = 1;
static void pnmap(const double* z, double* out) {
    LogHState s; std::memset(&s, 0, sizeof(s));
    const int n3 = 3 * g_n; s.n_bodies = g_n; s.c_code = g_c; s.use_aux_vel = true;
    for (int i = 0; i < g_n; i++) s.mass[i] = g_mass[i];
    for (int i = 0; i < n3; i++) { s.pos[i] = z[i]; s.vel[i] = z[n3 + i]; s.aux_vel[i] = z[2 * n3 + i]; }
    s.t = z[3 * n3]; s.B = z[3 * n3 + 1];
    logh_leapfrog_steps_pn(s, g_ds, 1.0, g_c, g_ord, 1, nullptr);
    for (int i = 0; i < n3; i++) { out[i] = s.pos[i]; out[n3 + i] = s.vel[i]; out[2 * n3 + i] = s.aux_vel[i]; }
    out[3 * n3] = s.t; out[3 * n3 + 1] = s.B;
}
int main() {
    std::printf("# PN full state: sys ord c e ds | rev_val rev_jac det-1\n");
    for (int sys : {2, 3}) for (int ord : {1, 7}) for (double c : {30.0, 1000.0}) for (double e : {0.0, 0.9}) for (double ds : {1e-1, 1e-2, 1e-3}) {
        g_n = sys; g_ord = ord; g_c = c; g_ds = ds;
        double z0[DMAX], z[DMAX], zo[DMAX], zb[DMAX];
        ics(sys, e, -0.3, z0);
        const int n3 = 3 * g_n, d = 3 * n3 + 2;
        for (int i = 0; i < n3; i++) { z[i] = z0[i]; z[n3 + i] = z0[n3 + 1 + i] / g_mass[i / 3]; z[2 * n3 + i] = z[n3 + i] * (1 + 1e-3 * ((i % 3) - 1)); }
        z[3 * n3] = 0.3; z[3 * n3 + 1] = z0[2 * n3 + 1];
        pnmap(z, zo); jac_enzyme<pnmap>(z, d, M);
        g_ds = -ds; pnmap(zo, zb); jac_enzyme<pnmap>(zo, d, Mb); g_ds = ds;
        double rv = 0; for (int j = 0; j < d; j++) rv = std::max(rv, std::fabs(zb[j] - z[j]) / std::max(1.0, std::fabs(z[j])));
        std::printf("%d %d %5g %4.2f %6.0e | %8.1e %8.1e %9.2e\n", sys, ord, c, e, ds, rv, prod_minus_I(Mb, M, d), det(M, d) - 1);
    }
}
