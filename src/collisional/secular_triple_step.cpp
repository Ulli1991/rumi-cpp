#include "secular_triple_step.h"
#include "secular.h"
#include "kepler.h"
#include <algorithm>
#include <cmath>

// The secular-triple step of advance_subsystems_in_range on its own (plain C++, no Kokkos: tests/enzyme can
// differentiate it): bodies 0, 1 the inner binary (0 = inner_binary_id), 2 the outer companion; m, x, v in Msun, pc,
// km/s, in and out. The code is the integrator's, with its particle accessors bound to the three bodies.
SecularTripleStep secular_triple_step(double M_[3], double X_[3][3], double V_[3][3], const int pids[3], double dt,
                                      double G, double c_code, int pn_order) {
    SecularTripleStep out;
    auto h_m = [&](int i) -> double& { return M_[i]; };
    auto h_x = [&](int i) -> double& { return X_[i][0]; };
    auto h_y = [&](int i) -> double& { return X_[i][1]; };
    auto h_z = [&](int i) -> double& { return X_[i][2]; };
    auto h_vx = [&](int i) -> double& { return V_[i][0]; };
    auto h_vy = [&](int i) -> double& { return V_[i][1]; };
    auto h_vz = [&](int i) -> double& { return V_[i][2]; };
    const int start = 0, idx0 = 0, idx1 = 1, idx2 = 2;
    double m1 = h_m(idx0 - start), m2 = h_m(idx1 - start), m3 = h_m(idx2 - start);
    double mu_in = G * (m1 + m2);

    // Inner binary relative coords
    double r_in[3] = {h_x(idx1 - start)-h_x(idx0 - start), h_y(idx1 - start)-h_y(idx0 - start), h_z(idx1 - start)-h_z(idx0 - start)};
    double v_in[3] = {h_vx(idx1 - start)-h_vx(idx0 - start), h_vy(idx1 - start)-h_vy(idx0 - start), h_vz(idx1 - start)-h_vz(idx0 - start)};

    // Inner binary -> elements
    OrbitalElements oe_in = cartesian_to_elements(r_in, v_in, mu_in);

    // Outer orbit: CM of inner binary vs outer companion
    double M12 = m1 + m2;
    double cm_x = (m1*h_x(idx0 - start) + m2*h_x(idx1 - start)) / M12;
    double cm_y = (m1*h_y(idx0 - start) + m2*h_y(idx1 - start)) / M12;
    double cm_z = (m1*h_z(idx0 - start) + m2*h_z(idx1 - start)) / M12;
    double cm_vx = (m1*h_vx(idx0 - start) + m2*h_vx(idx1 - start)) / M12;
    double cm_vy = (m1*h_vy(idx0 - start) + m2*h_vy(idx1 - start)) / M12;
    double cm_vz = (m1*h_vz(idx0 - start) + m2*h_vz(idx1 - start)) / M12;

    double r_out[3] = {h_x(idx2 - start)-cm_x, h_y(idx2 - start)-cm_y, h_z(idx2 - start)-cm_z};
    double v_out[3] = {h_vx(idx2 - start)-cm_vx, h_vy(idx2 - start)-cm_vy, h_vz(idx2 - start)-cm_vz};
    double mu_out = G * (M12 + m3);
    OrbitalElements oe_out = cartesian_to_elements(r_out, v_out, mu_out);

    if (oe_in.a <= 0 || oe_out.a <= 0) {  // no longer hierarchical: integrate it directly
        out.hierarchical = false;
        return out;
    }

    // Mutual inclination from angular momentum vectors
    double L_in[3] = {r_in[1]*v_in[2]-r_in[2]*v_in[1],
                      r_in[2]*v_in[0]-r_in[0]*v_in[2],
                      r_in[0]*v_in[1]-r_in[1]*v_in[0]};
    double L_out[3] = {r_out[1]*v_out[2]-r_out[2]*v_out[1],
                       r_out[2]*v_out[0]-r_out[0]*v_out[2],
                       r_out[0]*v_out[1]-r_out[1]*v_out[0]};
    double L_in_mag = std::sqrt(L_in[0]*L_in[0]+L_in[1]*L_in[1]+L_in[2]*L_in[2]);
    double L_out_mag = std::sqrt(L_out[0]*L_out[0]+L_out[1]*L_out[1]+L_out[2]*L_out[2]);
    double cos_i_mut = (L_in_mag > 0 && L_out_mag > 0) ?
        (L_in[0]*L_out[0]+L_in[1]*L_out[1]+L_in[2]*L_out[2]) / (L_in_mag*L_out_mag) : 1.0;

    // Build SecularTriple and evolve
    SecularTriple tri{};
    tri.inner.a = oe_in.a; tri.inner.e = oe_in.e; tri.inner.omega = oe_in.omega;
    tri.inner.inc = oe_in.inc; tri.inner.Omega = oe_in.Omega;
    tri.inner.m1 = m1; tri.inner.m2 = m2;
    tri.inner.pid1 = pids[0]; tri.inner.pid2 = pids[1];
    tri.a_out = oe_out.a; tri.e_out = oe_out.e;
    tri.omega_out = oe_out.omega; tri.inc_out = oe_out.inc; tri.Omega_out = oe_out.Omega;
    tri.m3 = m3; tri.pid3 = pids[2];
    tri.cos_i_mut = cos_i_mut;

    out.a_in = tri.inner.a; out.e_in = tri.inner.e; out.a_out = tri.a_out; out.cos_i = cos_i_mut;

    double a_in_pre = tri.inner.a;  // save for GW bookkeeping
    double booked_tri = 0.0;   // GW energy booked for this triple (E_gw)
    auto tri_energy = [&]() {
        const int ii[3] = {idx0 - start, idx1 - start, idx2 - start};
        double E = 0.0;
        for (int b = 0; b < 3; b++) {
            const int i = ii[b];
            E += 0.5 * h_m(i) * (h_vx(i) * h_vx(i) + h_vy(i) * h_vy(i) + h_vz(i) * h_vz(i));
            for (int c = b + 1; c < 3; c++) {
                const int j = ii[c];
                const double dx = h_x(j) - h_x(i), dy = h_y(j) - h_y(i), dz = h_z(j) - h_z(i);
                E -= G * h_m(i) * h_m(j) / std::sqrt(dx * dx + dy * dy + dz * dz);
            }
        }
        return E;
    };
    const double E_tri0 = tri_energy();   // T + U of the three members before the secular step

    secular_evolve_triple(tri, dt, G, c_code, pn_order);

    // GW energy bookkeeping for inner binary inspiral
    if (tri.inner.a > 0.0 && a_in_pre > 0.0) {
        double E_orb_pre_tri = -G * m1 * m2 / (2.0 * a_in_pre);
        double E_orb_post_tri = -G * m1 * m2 / (2.0 * tri.inner.a);
        booked_tri = E_orb_pre_tri - E_orb_post_tri;
    }

    // Update inner orbital plane to reflect changed mutual inclination.
    // The secular integrator evolves cos_i_mut but not (inc, Omega).
    // Rotate L_in around the node vector (L_out × L_in) by delta_i.
    {
        double L_in_hat[3] = {L_in[0]/L_in_mag, L_in[1]/L_in_mag, L_in[2]/L_in_mag};
        double L_out_hat[3] = {L_out[0]/L_out_mag, L_out[1]/L_out_mag, L_out[2]/L_out_mag};

        // Node vector = L_out × L_in (rotation axis)
        double nd[3] = {L_out_hat[1]*L_in_hat[2] - L_out_hat[2]*L_in_hat[1],
                        L_out_hat[2]*L_in_hat[0] - L_out_hat[0]*L_in_hat[2],
                        L_out_hat[0]*L_in_hat[1] - L_out_hat[1]*L_in_hat[0]};
        double nd_mag = std::sqrt(nd[0]*nd[0] + nd[1]*nd[1] + nd[2]*nd[2]);

        if (nd_mag > 1e-30) {
            nd[0] /= nd_mag; nd[1] /= nd_mag; nd[2] /= nd_mag;

            double i_old = std::acos(std::max(-1.0, std::min(1.0, cos_i_mut)));
            double i_new = std::acos(std::max(-1.0, std::min(1.0, tri.cos_i_mut)));
            double delta_i = i_new - i_old;

            // Rodrigues' rotation: R(v,θ) = v cosθ + (k×v) sinθ + k(k·v)(1-cosθ)
            double cd = std::cos(delta_i), sd = std::sin(delta_i);
            double kdv = nd[0]*L_in_hat[0] + nd[1]*L_in_hat[1] + nd[2]*L_in_hat[2];
            double kcv[3] = {nd[1]*L_in_hat[2] - nd[2]*L_in_hat[1],
                             nd[2]*L_in_hat[0] - nd[0]*L_in_hat[2],
                             nd[0]*L_in_hat[1] - nd[1]*L_in_hat[0]};
            double L_new[3];
            for (int k = 0; k < 3; k++)
                L_new[k] = L_in_hat[k]*cd + kcv[k]*sd + nd[k]*kdv*(1.0-cd);

            // Extract new inc, Omega from rotated L_in
            double Lz = std::max(-1.0, std::min(1.0, L_new[2]));
            tri.inner.inc = std::acos(Lz);
            tri.inner.Omega = std::atan2(L_new[0], -L_new[1]);
        }
    }

    // Convert evolved elements back to Cartesian
    // Inner binary
    double n_in = std::sqrt(mu_in / (tri.inner.a * tri.inner.a * tri.inner.a));
    double M_in_new = oe_in.mean_anom + n_in * dt;
    double r_in_new[3], v_in_new[3];
    elements_to_cartesian(tri.inner.a, tri.inner.e, tri.inner.inc,
                           tri.inner.Omega, tri.inner.omega, M_in_new, mu_in,
                           r_in_new, v_in_new);

    // Outer orbit. Angular-momentum-exact: the secular step changes the inner orbit's angular momentum (e_in,
    // plane); the internal L = mu_in r_in x v_in + mu_out r_out x v_out is conserved by taking the outer orbit's
    // L as what is left, L_out = L(before) - L_in(new): its direction gives the outer plane (i_out, Omega_out),
    // its magnitude e_out at the unchanged a_out (a frozen outer orbit lost up to |dL/L| 1.3e-2 per step in a
    // strongly coupled Kozai triple, found by scripts/adversary.py). The energy fix below leaves L alone
    double e_out_rb = tri.e_out, inc_out_rb = tri.inc_out, Omega_out_rb = tri.Omega_out;
    {
        const double mu_in_red = m1 * m2 / M12, mu_out_red = M12 * m3 / (M12 + m3);
        double Lin_new[3] = {r_in_new[1] * v_in_new[2] - r_in_new[2] * v_in_new[1],
                             r_in_new[2] * v_in_new[0] - r_in_new[0] * v_in_new[2],
                             r_in_new[0] * v_in_new[1] - r_in_new[1] * v_in_new[0]};
        double Lout_t[3];
        for (int k = 0; k < 3; k++)
            Lout_t[k] = mu_in_red * L_in[k] + mu_out_red * L_out[k] - mu_in_red * Lin_new[k];
        const double Lm = std::sqrt(Lout_t[0] * Lout_t[0] + Lout_t[1] * Lout_t[1] + Lout_t[2] * Lout_t[2]);
        const double h = Lm / mu_out_red;                       // specific angular momentum of the outer orbit
        const double q = 1.0 - h * h / (mu_out * tri.a_out);     // e_out^2
        if (Lm > 0.0 && q >= 0.0) {
            e_out_rb = std::sqrt(q);
            inc_out_rb = std::acos(std::max(-1.0, std::min(1.0, Lout_t[2] / Lm)));
            Omega_out_rb = std::atan2(Lout_t[0], -Lout_t[1]);
        } else {
            out.unfixed++;   // no outer orbit at this a_out carries that L
        }
    }
    double n_out = std::sqrt(mu_out / (tri.a_out * tri.a_out * tri.a_out));
    double M_out_new = oe_out.mean_anom + n_out * dt;
    double r_out_new[3], v_out_new[3];
    elements_to_cartesian(tri.a_out, e_out_rb, inc_out_rb,
                           Omega_out_rb, tri.omega_out, M_out_new, mu_out,
                           r_out_new, v_out_new);

    // Reconstruct absolute coordinates
    // Total CM
    double M_tot = M12 + m3;
    double cm_tot[3] = {
        (M12*cm_x + m3*h_x(idx2 - start)) / M_tot,
        (M12*cm_y + m3*h_y(idx2 - start)) / M_tot,
        (M12*cm_z + m3*h_z(idx2 - start)) / M_tot
    };
    double cm_tot_v[3] = {
        (M12*cm_vx + m3*h_vx(idx2 - start)) / M_tot,
        (M12*cm_vy + m3*h_vy(idx2 - start)) / M_tot,
        (M12*cm_vz + m3*h_vz(idx2 - start)) / M_tot
    };

    // Drift total CM
    for (int k = 0; k < 3; k++) cm_tot[k] += cm_tot_v[k] * dt;

    // Outer orbit positions: CM_inner and particle 3
    double cm_in_new[3], v_cm_in_new[3];
    for (int k = 0; k < 3; k++) {
        cm_in_new[k] = cm_tot[k] - (m3/M_tot) * r_out_new[k];
        v_cm_in_new[k] = cm_tot_v[k] - (m3/M_tot) * v_out_new[k];
    }

    // Inner binary: particle 0 and 1
    h_x(idx0 - start) = cm_in_new[0] - (m2/M12)*r_in_new[0];
    h_y(idx0 - start) = cm_in_new[1] - (m2/M12)*r_in_new[1];
    h_z(idx0 - start) = cm_in_new[2] - (m2/M12)*r_in_new[2];
    h_vx(idx0 - start) = v_cm_in_new[0] - (m2/M12)*v_in_new[0];
    h_vy(idx0 - start) = v_cm_in_new[1] - (m2/M12)*v_in_new[1];
    h_vz(idx0 - start) = v_cm_in_new[2] - (m2/M12)*v_in_new[2];

    h_x(idx1 - start) = cm_in_new[0] + (m1/M12)*r_in_new[0];
    h_y(idx1 - start) = cm_in_new[1] + (m1/M12)*r_in_new[1];
    h_z(idx1 - start) = cm_in_new[2] + (m1/M12)*r_in_new[2];
    h_vx(idx1 - start) = v_cm_in_new[0] + (m1/M12)*v_in_new[0];
    h_vy(idx1 - start) = v_cm_in_new[1] + (m1/M12)*v_in_new[1];
    h_vz(idx1 - start) = v_cm_in_new[2] + (m1/M12)*v_in_new[2];

    h_x(idx2 - start) = cm_tot[0] + (M12/M_tot)*r_out_new[0];
    h_y(idx2 - start) = cm_tot[1] + (M12/M_tot)*r_out_new[1];
    h_z(idx2 - start) = cm_tot[2] + (M12/M_tot)*r_out_new[2];
    h_vx(idx2 - start) = cm_tot_v[0] + (M12/M_tot)*v_out_new[0];
    h_vy(idx2 - start) = cm_tot_v[1] + (M12/M_tot)*v_out_new[1];
    h_vz(idx2 - start) = cm_tot_v[2] + (M12/M_tot)*v_out_new[2];

    // Energy-exact rebuild: the orbit-averaged step and the rebuild from elements (point-mass outer orbit,
    // re-placed phases) change the members' true T + U by more than the booked GW energy (200k seed 3,
    // step 60: -8.0e-10 |E0| unbooked from 2888 secular-triple calls, the whole energy jump of the step).
    // The difference goes into the radial component of a relative velocity, outer orbit first, else the
    // inner binary: P (the c.o.m. velocities stay) and L (a change along r leaves r x v alone) stay exact,
    // the positions and so U do not change, and T changes by exactly the difference
    {
        const double dE = (E_tri0 - booked_tri) - tri_energy();
        auto radial_fix = [&](int ia, int ib, double ma, double mb, int ic, double mc, bool outer) {
            // relative coordinate: b against a, or (outer) c against the c.o.m. of a and b
            double r[3], u[3];
            const double M2 = ma + mb, M3 = M2 + mc;
            const double xa[3] = {h_x(ia), h_y(ia), h_z(ia)}, xb[3] = {h_x(ib), h_y(ib), h_z(ib)};
            const double va[3] = {h_vx(ia), h_vy(ia), h_vz(ia)}, vb[3] = {h_vx(ib), h_vy(ib), h_vz(ib)};
            double mu;
            if (outer) {
                const double xc[3] = {h_x(ic), h_y(ic), h_z(ic)}, vc[3] = {h_vx(ic), h_vy(ic), h_vz(ic)};
                for (int k = 0; k < 3; k++) {
                    r[k] = xc[k] - (ma * xa[k] + mb * xb[k]) / M2;
                    u[k] = vc[k] - (ma * va[k] + mb * vb[k]) / M2;
                }
                mu = mc * M2 / M3;
            } else {
                for (int k = 0; k < 3; k++) { r[k] = xb[k] - xa[k]; u[k] = vb[k] - va[k]; }
                mu = ma * mb / M2;
            }
            const double rr = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
            if (!(rr > 0.0) || !(mu > 0.0)) return false;
            const double ur = (r[0] * u[0] + r[1] * u[1] + r[2] * u[2]) / rr;
            const double q = ur * ur + 2.0 * dE / mu;
            if (q < 0.0) return false;
            const double d = (ur >= 0.0 ? 1.0 : -1.0) * std::sqrt(q) - ur;
            const double du[3] = {d * r[0] / rr, d * r[1] / rr, d * r[2] / rr};
            if (outer) {
                h_vx(ic) += (M2 / M3) * du[0]; h_vy(ic) += (M2 / M3) * du[1]; h_vz(ic) += (M2 / M3) * du[2];
                h_vx(ia) -= (mc / M3) * du[0]; h_vy(ia) -= (mc / M3) * du[1]; h_vz(ia) -= (mc / M3) * du[2];
                h_vx(ib) -= (mc / M3) * du[0]; h_vy(ib) -= (mc / M3) * du[1]; h_vz(ib) -= (mc / M3) * du[2];
            } else {
                h_vx(ia) -= (mb / M2) * du[0]; h_vy(ia) -= (mb / M2) * du[1]; h_vz(ia) -= (mb / M2) * du[2];
                h_vx(ib) += (ma / M2) * du[0]; h_vy(ib) += (ma / M2) * du[1]; h_vz(ib) += (ma / M2) * du[2];
            }
            return true;
        };
        const int i0 = idx0 - start, i1 = idx1 - start, i2 = idx2 - start;
        if (dE != 0.0 && !radial_fix(i0, i1, m1, m2, i2, m3, true) && !radial_fix(i0, i1, m1, m2, i2, m3, false))
            out.unfixed++;
    }
    out.booked_gw = booked_tri;
    return out;
}
