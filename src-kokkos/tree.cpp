#include "tree.h"
#include "constants.h"
#include "integrator.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <algorithm>
#include <vector>
#include <cstdio>

// ==========================================================================
// Morton key computation (3D interleaving of integer coordinates)
// ==========================================================================

static inline uint64_t spread_bits_21(uint64_t x) {
    x &= 0x1fffff;
    x = (x | (x << 32)) & 0x1f00000000ffff;
    x = (x | (x << 16)) & 0x1f0000ff0000ff;
    x = (x | (x << 8))  & 0x100f00f00f00f00f;
    x = (x | (x << 4))  & 0x10c30c30c30c30c3;
    x = (x | (x << 2))  & 0x1249249249249249;
    return x;
}

static inline uint64_t morton_key_3d(uint64_t ix, uint64_t iy, uint64_t iz) {
    return spread_bits_21(ix) | (spread_bits_21(iy) << 1) | (spread_bits_21(iz) << 2);
}

// ==========================================================================
// Host-side octree construction
// ==========================================================================

struct HostCell {
    double cx, cy, cz;
    double mass;
    double cvx, cvy, cvz;  // center-of-mass velocity
    double half_width;
    int children[8];
    int pbeg, pend;
    bool is_leaf;
};

// Recursive octree build on sorted particles
// sx/sy/sz/sm/svx/svy/svz/si are mutable arrays reordered in-place by octant
static void build_octree_recursive(
    std::vector<HostCell>& cells,
    int cell_idx,
    double* sx, double* sy, double* sz, double* sm,
    double* svx, double* svy, double* svz, int* si,
    int pbeg, int pend,
    double ox, double oy, double oz, double half,
    int n_leaf_max, int depth)
{
    HostCell& cell = cells[cell_idx];
    cell.pbeg = pbeg;
    cell.pend = pend;
    cell.half_width = half;

    int count = pend - pbeg;

    // Compute center of mass and center-of-mass velocity
    double totm = 0, comx = 0, comy = 0, comz = 0;
    double comvx = 0, comvy = 0, comvz = 0;
    for (int i = pbeg; i < pend; i++) {
        totm += sm[i];
        comx += sm[i] * sx[i];
        comy += sm[i] * sy[i];
        comz += sm[i] * sz[i];
        comvx += sm[i] * svx[i];
        comvy += sm[i] * svy[i];
        comvz += sm[i] * svz[i];
    }
    if (totm > 0) {
        double inv_m = 1.0 / totm;
        cell.cx = comx * inv_m;
        cell.cy = comy * inv_m;
        cell.cz = comz * inv_m;
        cell.cvx = comvx * inv_m;
        cell.cvy = comvy * inv_m;
        cell.cvz = comvz * inv_m;
    } else {
        cell.cx = ox; cell.cy = oy; cell.cz = oz;
        cell.cvx = 0; cell.cvy = 0; cell.cvz = 0;
    }
    cell.mass = totm;

    if (count <= n_leaf_max || depth >= TREE_MAX_DEPTH) {
        cell.is_leaf = true;
        for (int c = 0; c < 8; c++) cell.children[c] = -1;
        return;
    }

    cell.is_leaf = false;

    // Count particles per octant
    int octant_count[8] = {};
    std::vector<int> octant_id(count);
    for (int i = pbeg; i < pend; i++) {
        int oct = 0;
        if (sx[i] >= ox) oct |= 1;
        if (sy[i] >= oy) oct |= 2;
        if (sz[i] >= oz) oct |= 4;
        octant_id[i - pbeg] = oct;
        octant_count[oct]++;
    }

    int octant_offset[8];
    octant_offset[0] = pbeg;
    for (int c = 1; c < 8; c++)
        octant_offset[c] = octant_offset[c - 1] + octant_count[c - 1];

    // Reorder all 8 arrays (x,y,z,m,vx,vy,vz,idx) into octant order
    int n = count;
    std::vector<double> tx(n), ty(n), tz(n), tm(n), tvx(n), tvy(n), tvz(n);
    std::vector<int> ti(n);
    int fill[8];
    for (int c = 0; c < 8; c++) fill[c] = octant_offset[c];

    for (int i = pbeg; i < pend; i++) {
        int oct = octant_id[i - pbeg];
        int dst = fill[oct]++;
        int k = dst - pbeg;
        tx[k] = sx[i]; ty[k] = sy[i]; tz[k] = sz[i];
        tm[k] = sm[i]; ti[k] = si[i];
        tvx[k] = svx[i]; tvy[k] = svy[i]; tvz[k] = svz[i];
    }
    for (int i = 0; i < n; i++) {
        sx[pbeg + i] = tx[i]; sy[pbeg + i] = ty[i]; sz[pbeg + i] = tz[i];
        sm[pbeg + i] = tm[i]; si[pbeg + i] = ti[i];
        svx[pbeg + i] = tvx[i]; svy[pbeg + i] = tvy[i]; svz[pbeg + i] = tvz[i];
    }

    // Create children
    double qh = half * 0.5;
    for (int c = 0; c < 8; c++) {
        int cb = octant_offset[c];
        int ce = cb + octant_count[c];
        if (octant_count[c] == 0) {
            cells[cell_idx].children[c] = -1;
            continue;
        }
        int child_idx = (int)cells.size();
        cells.emplace_back();
        cells[cell_idx].children[c] = child_idx;

        double cox = ox + ((c & 1) ? qh : -qh);
        double coy = oy + (((c >> 1) & 1) ? qh : -qh);
        double coz = oz + (((c >> 2) & 1) ? qh : -qh);

        build_octree_recursive(cells, child_idx, sx, sy, sz, sm, svx, svy, svz, si,
                               cb, ce, cox, coy, coz, qh, n_leaf_max, depth + 1);
    }
}

// ==========================================================================
// Octree::build
// ==========================================================================
void Octree::build(Particles& p, int start, int count, double theta_param) {
    if (count <= 0) return;

    theta = theta_param;
    n_particles = count;
    double inv_theta_sq = 1.0 / (theta * theta);

    // 1. Copy positions + masses + velocities to host
    auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.x);
    auto h_y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.y);
    auto h_z = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.z);
    auto h_m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
    auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
    auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
    auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);

    // 2. Compute bounding box
    double xmin = h_x(start), xmax = h_x(start);
    double ymin = h_y(start), ymax = h_y(start);
    double zmin = h_z(start), zmax = h_z(start);
    for (int i = start + 1; i < start + count; i++) {
        xmin = std::min(xmin, h_x(i)); xmax = std::max(xmax, h_x(i));
        ymin = std::min(ymin, h_y(i)); ymax = std::max(ymax, h_y(i));
        zmin = std::min(zmin, h_z(i)); zmax = std::max(zmax, h_z(i));
    }
    box_cx = 0.5 * (xmin + xmax);
    box_cy = 0.5 * (ymin + ymax);
    box_cz = 0.5 * (zmin + zmax);
    box_half = 0.5 * std::max({xmax - xmin, ymax - ymin, zmax - zmin});
    box_half *= 1.001;

    // 3. Compute Morton keys and sort
    double inv_range = (1 << 21) / (2.0 * box_half);
    std::vector<uint64_t> keys(count);
    std::vector<int> perm(count);

    for (int i = 0; i < count; i++) {
        int pi = start + i;
        uint64_t ix = (uint64_t)((h_x(pi) - (box_cx - box_half)) * inv_range);
        uint64_t iy = (uint64_t)((h_y(pi) - (box_cy - box_half)) * inv_range);
        uint64_t iz = (uint64_t)((h_z(pi) - (box_cz - box_half)) * inv_range);
        ix = std::min(ix, (uint64_t)((1 << 21) - 1));
        iy = std::min(iy, (uint64_t)((1 << 21) - 1));
        iz = std::min(iz, (uint64_t)((1 << 21) - 1));
        keys[i] = morton_key_3d(ix, iy, iz);
        perm[i] = i;
    }

    std::sort(perm.begin(), perm.end(), [&](int a, int b) {
        return keys[a] < keys[b];
    });

    // Build sorted arrays on host (including original index tracking)
    std::vector<double> sx(count), sy(count), sz(count), sm(count);
    std::vector<double> svx(count), svy(count), svz(count);
    std::vector<int> si(count);
    for (int i = 0; i < count; i++) {
        int orig = perm[i];
        sx[i] = h_x(start + orig);
        sy[i] = h_y(start + orig);
        sz[i] = h_z(start + orig);
        sm[i] = h_m(start + orig);
        svx[i] = h_vx(start + orig);
        svy[i] = h_vy(start + orig);
        svz[i] = h_vz(start + orig);
        si[i] = start + orig;  // original particle index
    }

    // 4. Build octree on host (reorders sx/sy/sz/sm/svx/svy/svz/si by octant)
    std::vector<HostCell> cells;
    cells.reserve(count / std::max(n_leaf, 1) * 2 + 100);
    cells.emplace_back();
    build_octree_recursive(cells, 0, sx.data(), sy.data(), sz.data(), sm.data(),
                           svx.data(), svy.data(), svz.data(), si.data(),
                           0, count, box_cx, box_cy, box_cz, box_half,
                           n_leaf, 0);

    n_cells = (int)cells.size();

    // 5. Upload to device
    cell_com = Kokkos::View<double*[4], Kokkos::LayoutRight>("cell_com", n_cells);
    cell_vel = Kokkos::View<double*[3], Kokkos::LayoutRight>("cell_vel", n_cells);
    cell_size_sq_theta = Kokkos::View<double*>("cell_size_sq_theta", n_cells);
    cell_children = Kokkos::View<int*[8], Kokkos::LayoutRight>("cell_children", n_cells);
    cell_pbeg = Kokkos::View<int*>("cell_pbeg", n_cells);
    cell_pend = Kokkos::View<int*>("cell_pend", n_cells);
    cell_is_leaf = Kokkos::View<int*>("cell_is_leaf", n_cells);
    sorted_indices = Kokkos::View<int*>("sorted_indices", count);
    sorted_pos = Kokkos::View<double*[4], Kokkos::LayoutRight>("sorted_pos", count);

    auto h_cell_com = Kokkos::create_mirror_view(cell_com);
    auto h_cell_vel = Kokkos::create_mirror_view(cell_vel);
    auto h_cell_size = Kokkos::create_mirror_view(cell_size_sq_theta);
    auto h_cell_children = Kokkos::create_mirror_view(cell_children);
    auto h_cell_pbeg = Kokkos::create_mirror_view(cell_pbeg);
    auto h_cell_pend = Kokkos::create_mirror_view(cell_pend);
    auto h_cell_is_leaf = Kokkos::create_mirror_view(cell_is_leaf);

    for (int c = 0; c < n_cells; c++) {
        h_cell_com(c, 0) = cells[c].cx;
        h_cell_com(c, 1) = cells[c].cy;
        h_cell_com(c, 2) = cells[c].cz;
        h_cell_com(c, 3) = cells[c].mass;
        h_cell_vel(c, 0) = cells[c].cvx;
        h_cell_vel(c, 1) = cells[c].cvy;
        h_cell_vel(c, 2) = cells[c].cvz;
        h_cell_size(c) = cells[c].half_width * cells[c].half_width * inv_theta_sq;
        for (int ch = 0; ch < 8; ch++)
            h_cell_children(c, ch) = cells[c].children[ch];
        h_cell_pbeg(c) = cells[c].pbeg;
        h_cell_pend(c) = cells[c].pend;
        h_cell_is_leaf(c) = cells[c].is_leaf ? 1 : 0;
    }

    auto h_sorted_idx = Kokkos::create_mirror_view(sorted_indices);
    auto h_sorted_pos = Kokkos::create_mirror_view(sorted_pos);
    for (int i = 0; i < count; i++) {
        h_sorted_idx(i) = si[i];
        h_sorted_pos(i, 0) = sx[i];
        h_sorted_pos(i, 1) = sy[i];
        h_sorted_pos(i, 2) = sz[i];
        h_sorted_pos(i, 3) = sm[i];
    }

    Kokkos::deep_copy(cell_com, h_cell_com);
    Kokkos::deep_copy(cell_vel, h_cell_vel);
    Kokkos::deep_copy(cell_size_sq_theta, h_cell_size);
    Kokkos::deep_copy(cell_children, h_cell_children);
    Kokkos::deep_copy(cell_pbeg, h_cell_pbeg);
    Kokkos::deep_copy(cell_pend, h_cell_pend);
    Kokkos::deep_copy(cell_is_leaf, h_cell_is_leaf);
    Kokkos::deep_copy(sorted_indices, h_sorted_idx);
    Kokkos::deep_copy(sorted_pos, h_sorted_pos);

    // 6. Store permanent particle IDs in sorted order (for rebuild after partition)
    sorted_orig_ids = Kokkos::View<int*>("sorted_orig_ids", count);
    sorted_subset_flag = Kokkos::View<int*>("sorted_subset_flag", count);
    {
        auto si_v = sorted_indices;
        auto soid = sorted_orig_ids;
        auto porig = p.orig_index;
        Kokkos::parallel_for("store_orig_ids", count, KOKKOS_LAMBDA(int j) {
            soid(j) = porig(si_v(j));
        });
    }

    // 7. Allocate dual monopole Views
    cell_com_slow = Kokkos::View<double*[4], Kokkos::LayoutRight>("cell_com_slow", n_cells);
    cell_com_fast = Kokkos::View<double*[4], Kokkos::LayoutRight>("cell_com_fast", n_cells);

    // 8. Build traversal groups
    make_groups();
}

// ==========================================================================
// Make traversal groups from sorted particles
// ==========================================================================
void Octree::make_groups() {
    n_groups = (n_particles + n_group - 1) / n_group;

    group_start = Kokkos::View<int*>("group_start", n_groups);
    group_count = Kokkos::View<int*>("group_count", n_groups);
    group_center = Kokkos::View<double*[3]>("group_center", n_groups);
    group_size = Kokkos::View<double*[3]>("group_size", n_groups);

    auto h_gs = Kokkos::create_mirror_view(group_start);
    auto h_gc = Kokkos::create_mirror_view(group_count);
    auto h_gcen = Kokkos::create_mirror_view(group_center);
    auto h_gsz = Kokkos::create_mirror_view(group_size);

    auto h_sp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sorted_pos);

    for (int g = 0; g < n_groups; g++) {
        int gs = g * n_group;
        int ge = std::min(gs + n_group, n_particles);
        int gc = ge - gs;

        h_gs(g) = gs;
        h_gc(g) = gc;

        double xmin = h_sp(gs, 0), xmax = h_sp(gs, 0);
        double ymin = h_sp(gs, 1), ymax = h_sp(gs, 1);
        double zmin = h_sp(gs, 2), zmax = h_sp(gs, 2);
        for (int i = gs + 1; i < ge; i++) {
            xmin = std::min(xmin, h_sp(i, 0)); xmax = std::max(xmax, h_sp(i, 0));
            ymin = std::min(ymin, h_sp(i, 1)); ymax = std::max(ymax, h_sp(i, 1));
            zmin = std::min(zmin, h_sp(i, 2)); zmax = std::max(zmax, h_sp(i, 2));
        }

        h_gcen(g, 0) = 0.5 * (xmin + xmax);
        h_gcen(g, 1) = 0.5 * (ymin + ymax);
        h_gcen(g, 2) = 0.5 * (zmin + zmax);
        h_gsz(g, 0) = 0.5 * (xmax - xmin);
        h_gsz(g, 1) = 0.5 * (ymax - ymin);
        h_gsz(g, 2) = 0.5 * (zmax - zmin);
    }

    Kokkos::deep_copy(group_start, h_gs);
    Kokkos::deep_copy(group_count, h_gc);
    Kokkos::deep_copy(group_center, h_gcen);
    Kokkos::deep_copy(group_size, h_gsz);
}

// ==========================================================================
// O(N) position refresh from live particle data
// Updates sorted_pos without rebuilding tree structure
// ==========================================================================
void Octree::refresh_positions(Particles& p) {
    if (n_particles <= 0) return;
    auto si = sorted_indices;
    auto sp = sorted_pos;
    auto px = p.x;
    auto py = p.y;
    auto pz = p.z;
    Kokkos::parallel_for("tree_refresh_pos", n_particles, KOKKOS_LAMBDA(int i) {
        int orig = si(i);
        sp(i, 0) = px(orig);
        sp(i, 1) = py(orig);
        sp(i, 2) = pz(orig);
        // mass (sp(i,3)) unchanged
    });
}

// ==========================================================================
// Compute cell center-of-mass accelerations (bottom-up on host)
// Children always have higher indices than parents (DFS emplace_back order)
// so reverse iteration gives bottom-up ordering.
// ==========================================================================
void Octree::compute_cell_accelerations(Particles& p) {
    if (n_cells <= 0 || n_particles <= 0) return;

    auto h_ax = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.ax);
    auto h_ay = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.ay);
    auto h_az = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.az);
    auto h_mass = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
    auto h_si = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sorted_indices);
    auto h_children = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cell_children);
    auto h_is_leaf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cell_is_leaf);
    auto h_pbeg = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cell_pbeg);
    auto h_pend = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cell_pend);
    auto h_com = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cell_com);

    if ((int)cell_accel.extent(0) != n_cells)
        cell_accel = Kokkos::View<double*[3], Kokkos::LayoutRight>("cell_accel", n_cells);
    auto h_ca = Kokkos::create_mirror_view(cell_accel);

    // Bottom-up: highest index first
    for (int c = n_cells - 1; c >= 0; c--) {
        double cell_m = h_com(c, 3);
        if (cell_m <= 0) {
            h_ca(c, 0) = h_ca(c, 1) = h_ca(c, 2) = 0;
            continue;
        }
        double inv_m = 1.0 / cell_m;

        if (h_is_leaf(c)) {
            double max = 0, may = 0, maz = 0;
            for (int i = h_pbeg(c); i < h_pend(c); i++) {
                int orig = h_si(i);
                double m = h_mass(orig);
                max += m * h_ax(orig);
                may += m * h_ay(orig);
                maz += m * h_az(orig);
            }
            h_ca(c, 0) = max * inv_m;
            h_ca(c, 1) = may * inv_m;
            h_ca(c, 2) = maz * inv_m;
        } else {
            double max = 0, may = 0, maz = 0;
            for (int ch = 0; ch < 8; ch++) {
                int child = h_children(c, ch);
                if (child < 0) continue;
                double cm = h_com(child, 3);
                max += cm * h_ca(child, 0);
                may += cm * h_ca(child, 1);
                maz += cm * h_ca(child, 2);
            }
            h_ca(c, 0) = max * inv_m;
            h_ca(c, 1) = may * inv_m;
            h_ca(c, 2) = maz * inv_m;
        }
    }

    Kokkos::deep_copy(cell_accel, h_ca);
}

// ==========================================================================
// Rebuild sorted_indices after particle reordering (e.g. partition_by_pivot)
// Uses permanent IDs (sorted_orig_ids) + reverse map to find current indices.
// O(N) GPU kernels.
// ==========================================================================
void Octree::rebuild_sorted_indices(Particles& p) {
    if (n_particles <= 0) return;

    // Build reverse map: permanent_id -> current_particle_index
    int N = p.N;
    static Kokkos::View<int*> reverse_map("reverse_map", 0);
    if ((int)reverse_map.extent(0) < N)
        reverse_map = Kokkos::View<int*>("reverse_map", N);

    auto orig = p.orig_index;
    auto rmap = reverse_map;
    Kokkos::parallel_for("build_reverse_map", N, KOKKOS_LAMBDA(int i) {
        rmap(orig(i)) = i;
    });

    // Rebuild sorted_indices from permanent IDs
    auto si = sorted_indices;
    auto soid = sorted_orig_ids;
    int n = n_particles;
    Kokkos::parallel_for("rebuild_sorted_idx", n, KOKKOS_LAMBDA(int j) {
        si(j) = rmap(soid(j));
    });
}

// ==========================================================================
// Full update after partition: rebuild indices, refresh positions,
// mark slow/fast flags, recompute dual monopoles.
// Call once per HHS level after partition_by_pivot.
// ==========================================================================
void Octree::update_subsets(Particles& p, int slow_start, int slow_count) {
    if (n_particles <= 0) return;

    // 1. Fix sorted_indices after particle reordering
    rebuild_sorted_indices(p);

    // 2. Refresh positions from updated sorted_indices
    refresh_positions(p);

    // 3. Mark slow/fast flags per sorted particle
    auto si = sorted_indices;
    auto flag = sorted_subset_flag;
    int n = n_particles;
    int ss = slow_start, sc = slow_count;
    Kokkos::parallel_for("mark_subsets", n, KOKKOS_LAMBDA(int j) {
        int idx = si(j);
        flag(j) = (idx >= ss && idx < ss + sc) ? 0 : 1;  // 0=slow, 1=fast
    });
    Kokkos::fence();

    // 4. Recompute dual monopoles (including velocities)
    recompute_dual_monopoles(p);
}

// ==========================================================================
// Recompute dual monopoles (cell_com_slow/fast, cell_vel_slow/fast) from
// sorted_pos, sorted_subset_flag, and particle velocities. Bottom-up on host.
// ==========================================================================
void Octree::recompute_dual_monopoles(Particles& p) {
    if (n_cells <= 0 || n_particles <= 0) return;

    // Copy needed data to host
    auto h_sp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(sorted_pos, Kokkos::make_pair(0, n_particles), Kokkos::ALL));
    auto h_flag = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(sorted_subset_flag, Kokkos::make_pair(0, n_particles)));
    auto h_si = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(sorted_indices, Kokkos::make_pair(0, n_particles)));
    auto h_children = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(cell_children, Kokkos::make_pair(0, n_cells), Kokkos::ALL));
    auto h_is_leaf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(cell_is_leaf, Kokkos::make_pair(0, n_cells)));
    auto h_pbeg = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(cell_pbeg, Kokkos::make_pair(0, n_cells)));
    auto h_pend = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(cell_pend, Kokkos::make_pair(0, n_cells)));

    // Copy velocities for mass-weighted velocity computation
    auto h_vx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vx);
    auto h_vy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vy);
    auto h_vz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.vz);

    // Allocate dual monopoles if needed
    if ((int)cell_com_slow.extent(0) < n_cells) {
        cell_com_slow = Kokkos::View<double*[4], Kokkos::LayoutRight>("cell_com_slow", n_cells);
        cell_com_fast = Kokkos::View<double*[4], Kokkos::LayoutRight>("cell_com_fast", n_cells);
    }
    if ((int)cell_vel_slow.extent(0) < n_cells) {
        cell_vel_slow = Kokkos::View<double*[3], Kokkos::LayoutRight>("cell_vel_slow", n_cells);
        cell_vel_fast = Kokkos::View<double*[3], Kokkos::LayoutRight>("cell_vel_fast", n_cells);
    }

    auto h_com_slow = Kokkos::create_mirror_view(
        Kokkos::subview(cell_com_slow, Kokkos::make_pair(0, n_cells), Kokkos::ALL));
    auto h_com_fast = Kokkos::create_mirror_view(
        Kokkos::subview(cell_com_fast, Kokkos::make_pair(0, n_cells), Kokkos::ALL));
    auto h_vel_slow = Kokkos::create_mirror_view(
        Kokkos::subview(cell_vel_slow, Kokkos::make_pair(0, n_cells), Kokkos::ALL));
    auto h_vel_fast = Kokkos::create_mirror_view(
        Kokkos::subview(cell_vel_fast, Kokkos::make_pair(0, n_cells), Kokkos::ALL));

    // Bottom-up: highest index first (children always have higher index than parent)
    for (int c = n_cells - 1; c >= 0; c--) {
        double sm_s = 0, sx_s = 0, sy_s = 0, sz_s = 0;
        double sm_f = 0, sx_f = 0, sy_f = 0, sz_f = 0;
        double svx_s = 0, svy_s = 0, svz_s = 0;
        double svx_f = 0, svy_f = 0, svz_f = 0;

        if (h_is_leaf(c)) {
            // Leaf: accumulate from particles
            for (int i = h_pbeg(c); i < h_pend(c); i++) {
                double m = h_sp(i, 3);
                double x = h_sp(i, 0), y = h_sp(i, 1), z = h_sp(i, 2);
                int orig = h_si(i);
                double vx = h_vx(orig), vy = h_vy(orig), vz = h_vz(orig);
                if (h_flag(i) == 0) {  // slow
                    sm_s += m; sx_s += m * x; sy_s += m * y; sz_s += m * z;
                    svx_s += m * vx; svy_s += m * vy; svz_s += m * vz;
                } else {  // fast
                    sm_f += m; sx_f += m * x; sy_f += m * y; sz_f += m * z;
                    svx_f += m * vx; svy_f += m * vy; svz_f += m * vz;
                }
            }
        } else {
            // Internal: sum children
            for (int ch = 0; ch < 8; ch++) {
                int child = h_children(c, ch);
                if (child < 0) continue;
                double cms = h_com_slow(child, 3);
                sm_s += cms;
                sx_s += cms * h_com_slow(child, 0);
                sy_s += cms * h_com_slow(child, 1);
                sz_s += cms * h_com_slow(child, 2);
                svx_s += cms * h_vel_slow(child, 0);
                svy_s += cms * h_vel_slow(child, 1);
                svz_s += cms * h_vel_slow(child, 2);
                double cmf = h_com_fast(child, 3);
                sm_f += cmf;
                sx_f += cmf * h_com_fast(child, 0);
                sy_f += cmf * h_com_fast(child, 1);
                sz_f += cmf * h_com_fast(child, 2);
                svx_f += cmf * h_vel_fast(child, 0);
                svy_f += cmf * h_vel_fast(child, 1);
                svz_f += cmf * h_vel_fast(child, 2);
            }
        }

        // Store slow monopole
        if (sm_s > 0) {
            double inv = 1.0 / sm_s;
            h_com_slow(c, 0) = sx_s * inv;
            h_com_slow(c, 1) = sy_s * inv;
            h_com_slow(c, 2) = sz_s * inv;
            h_vel_slow(c, 0) = svx_s * inv;
            h_vel_slow(c, 1) = svy_s * inv;
            h_vel_slow(c, 2) = svz_s * inv;
        } else {
            h_com_slow(c, 0) = h_com_slow(c, 1) = h_com_slow(c, 2) = 0;
            h_vel_slow(c, 0) = h_vel_slow(c, 1) = h_vel_slow(c, 2) = 0;
        }
        h_com_slow(c, 3) = sm_s;

        // Store fast monopole
        if (sm_f > 0) {
            double inv = 1.0 / sm_f;
            h_com_fast(c, 0) = sx_f * inv;
            h_com_fast(c, 1) = sy_f * inv;
            h_com_fast(c, 2) = sz_f * inv;
            h_vel_fast(c, 0) = svx_f * inv;
            h_vel_fast(c, 1) = svy_f * inv;
            h_vel_fast(c, 2) = svz_f * inv;
        } else {
            h_com_fast(c, 0) = h_com_fast(c, 1) = h_com_fast(c, 2) = 0;
            h_vel_fast(c, 0) = h_vel_fast(c, 1) = h_vel_fast(c, 2) = 0;
        }
        h_com_fast(c, 3) = sm_f;
    }

    Kokkos::deep_copy(Kokkos::subview(cell_com_slow, Kokkos::make_pair(0, n_cells), Kokkos::ALL), h_com_slow);
    Kokkos::deep_copy(Kokkos::subview(cell_com_fast, Kokkos::make_pair(0, n_cells), Kokkos::ALL), h_com_fast);
    Kokkos::deep_copy(Kokkos::subview(cell_vel_slow, Kokkos::make_pair(0, n_cells), Kokkos::ALL), h_vel_slow);
    Kokkos::deep_copy(Kokkos::subview(cell_vel_fast, Kokkos::make_pair(0, n_cells), Kokkos::ALL), h_vel_fast);
}

// ==========================================================================
// Compute dual cell accelerations (slow/fast) bottom-up from particle data.
// Uses sorted_subset_flag to split accelerations into slow/fast.
// Call after cross-force computation (p.ax/ay/az contain cross-forces).
// ==========================================================================
void Octree::compute_dual_cell_accelerations(Particles& p) {
    if (n_cells <= 0 || n_particles <= 0) return;

    auto h_ax = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.ax);
    auto h_ay = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.ay);
    auto h_az = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.az);
    auto h_mass = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p.mass);
    auto h_si = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(sorted_indices, Kokkos::make_pair(0, n_particles)));
    auto h_flag = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(sorted_subset_flag, Kokkos::make_pair(0, n_particles)));
    auto h_children = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(cell_children, Kokkos::make_pair(0, n_cells), Kokkos::ALL));
    auto h_is_leaf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(cell_is_leaf, Kokkos::make_pair(0, n_cells)));
    auto h_pbeg = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(cell_pbeg, Kokkos::make_pair(0, n_cells)));
    auto h_pend = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(cell_pend, Kokkos::make_pair(0, n_cells)));

    // Read dual monopole masses for weighting internal cells
    auto h_com_slow = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(cell_com_slow, Kokkos::make_pair(0, n_cells), Kokkos::ALL));
    auto h_com_fast = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
        Kokkos::subview(cell_com_fast, Kokkos::make_pair(0, n_cells), Kokkos::ALL));

    if ((int)cell_accel_slow.extent(0) < n_cells) {
        cell_accel_slow = Kokkos::View<double*[3], Kokkos::LayoutRight>("cell_accel_slow", n_cells);
        cell_accel_fast = Kokkos::View<double*[3], Kokkos::LayoutRight>("cell_accel_fast", n_cells);
    }
    auto h_ca_slow = Kokkos::create_mirror_view(
        Kokkos::subview(cell_accel_slow, Kokkos::make_pair(0, n_cells), Kokkos::ALL));
    auto h_ca_fast = Kokkos::create_mirror_view(
        Kokkos::subview(cell_accel_fast, Kokkos::make_pair(0, n_cells), Kokkos::ALL));

    for (int c = n_cells - 1; c >= 0; c--) {
        double ms = h_com_slow(c, 3), mf = h_com_fast(c, 3);

        if (h_is_leaf(c)) {
            double max_s = 0, may_s = 0, maz_s = 0;
            double max_f = 0, may_f = 0, maz_f = 0;
            for (int i = h_pbeg(c); i < h_pend(c); i++) {
                int orig = h_si(i);
                double m = h_mass(orig);
                if (h_flag(i) == 0) {
                    max_s += m * h_ax(orig); may_s += m * h_ay(orig); maz_s += m * h_az(orig);
                } else {
                    max_f += m * h_ax(orig); may_f += m * h_ay(orig); maz_f += m * h_az(orig);
                }
            }
            if (ms > 0) { double inv = 1.0/ms; h_ca_slow(c,0) = max_s*inv; h_ca_slow(c,1) = may_s*inv; h_ca_slow(c,2) = maz_s*inv; }
            else { h_ca_slow(c,0) = h_ca_slow(c,1) = h_ca_slow(c,2) = 0; }
            if (mf > 0) { double inv = 1.0/mf; h_ca_fast(c,0) = max_f*inv; h_ca_fast(c,1) = may_f*inv; h_ca_fast(c,2) = maz_f*inv; }
            else { h_ca_fast(c,0) = h_ca_fast(c,1) = h_ca_fast(c,2) = 0; }
        } else {
            double max_s = 0, may_s = 0, maz_s = 0;
            double max_f = 0, may_f = 0, maz_f = 0;
            for (int ch = 0; ch < 8; ch++) {
                int child = h_children(c, ch);
                if (child < 0) continue;
                double cm_s = h_com_slow(child, 3);
                max_s += cm_s * h_ca_slow(child, 0); may_s += cm_s * h_ca_slow(child, 1); maz_s += cm_s * h_ca_slow(child, 2);
                double cm_f = h_com_fast(child, 3);
                max_f += cm_f * h_ca_fast(child, 0); may_f += cm_f * h_ca_fast(child, 1); maz_f += cm_f * h_ca_fast(child, 2);
            }
            if (ms > 0) { double inv = 1.0/ms; h_ca_slow(c,0) = max_s*inv; h_ca_slow(c,1) = may_s*inv; h_ca_slow(c,2) = maz_s*inv; }
            else { h_ca_slow(c,0) = h_ca_slow(c,1) = h_ca_slow(c,2) = 0; }
            if (mf > 0) { double inv = 1.0/mf; h_ca_fast(c,0) = max_f*inv; h_ca_fast(c,1) = may_f*inv; h_ca_fast(c,2) = maz_f*inv; }
            else { h_ca_fast(c,0) = h_ca_fast(c,1) = h_ca_fast(c,2) = 0; }
        }
    }

    Kokkos::deep_copy(Kokkos::subview(cell_accel_slow, Kokkos::make_pair(0, n_cells), Kokkos::ALL), h_ca_slow);
    Kokkos::deep_copy(Kokkos::subview(cell_accel_fast, Kokkos::make_pair(0, n_cells), Kokkos::ALL), h_ca_fast);
}

void Octree::deallocate() {
    cell_com = {};
    cell_vel = {};
    cell_accel = {};
    cell_com_slow = {};
    cell_com_fast = {};
    cell_vel_slow = {};
    cell_vel_fast = {};
    cell_accel_slow = {};
    cell_accel_fast = {};
    cell_size_sq_theta = {};
    cell_children = {};
    cell_pbeg = {};
    cell_pend = {};
    cell_is_leaf = {};
    sorted_indices = {};
    sorted_pos = {};
    sorted_orig_ids = {};
    sorted_subset_flag = {};
    group_start = {};
    group_count = {};
    group_center = {};
    group_size = {};
    n_cells = 0;
    n_groups = 0;
    n_particles = 0;
}
