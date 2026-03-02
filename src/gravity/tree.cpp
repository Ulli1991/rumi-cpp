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

    // 5. Upload to device (only reallocate if capacity insufficient)
    if ((int)cell_com.extent(0) < n_cells) {
        cell_com = Kokkos::View<double*[4], Kokkos::LayoutRight>("cell_com", n_cells);
        cell_vel = Kokkos::View<double*[3], Kokkos::LayoutRight>("cell_vel", n_cells);
        cell_size_sq_theta = Kokkos::View<double*>("cell_size_sq_theta", n_cells);
        cell_children = Kokkos::View<int*[8], Kokkos::LayoutRight>("cell_children", n_cells);
        cell_pbeg = Kokkos::View<int*>("cell_pbeg", n_cells);
        cell_pend = Kokkos::View<int*>("cell_pend", n_cells);
        cell_is_leaf = Kokkos::View<int*>("cell_is_leaf", n_cells);
    }
    if ((int)sorted_indices.extent(0) < count) {
        sorted_indices = Kokkos::View<int*>("sorted_indices", count);
        sorted_pos = Kokkos::View<double*[4], Kokkos::LayoutRight>("sorted_pos", count);
    }

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

    // Pre-allocate: only reallocate if too small
    if ((int)group_start.extent(0) < n_groups) {
        group_start = Kokkos::View<int*>("group_start", n_groups);
        group_count = Kokkos::View<int*>("group_count", n_groups);
        group_center = Kokkos::View<double*[3]>("group_center", n_groups);
        group_size = Kokkos::View<double*[3]>("group_size", n_groups);
    }

    // Compute group bounding boxes on GPU directly from sorted_pos
    auto d_gs = group_start;
    auto d_gc = group_count;
    auto d_gcen = group_center;
    auto d_gsz = group_size;
    auto d_sp = sorted_pos;
    int np = n_particles;
    int ng = n_group;

    Kokkos::parallel_for("make_groups", n_groups, KOKKOS_LAMBDA(int g) {
        int gs = g * ng;
        int ge = (gs + ng < np) ? gs + ng : np;
        int gc = ge - gs;

        d_gs(g) = gs;
        d_gc(g) = gc;

        double xmin = d_sp(gs, 0), xmax = d_sp(gs, 0);
        double ymin = d_sp(gs, 1), ymax = d_sp(gs, 1);
        double zmin = d_sp(gs, 2), zmax = d_sp(gs, 2);
        for (int i = gs + 1; i < ge; i++) {
            double xi = d_sp(i, 0), yi = d_sp(i, 1), zi = d_sp(i, 2);
            xmin = (xi < xmin) ? xi : xmin; xmax = (xi > xmax) ? xi : xmax;
            ymin = (yi < ymin) ? yi : ymin; ymax = (yi > ymax) ? yi : ymax;
            zmin = (zi < zmin) ? zi : zmin; zmax = (zi > zmax) ? zi : zmax;
        }

        d_gcen(g, 0) = 0.5 * (xmin + xmax);
        d_gcen(g, 1) = 0.5 * (ymin + ymax);
        d_gcen(g, 2) = 0.5 * (zmin + zmax);
        d_gsz(g, 0) = 0.5 * (xmax - xmin);
        d_gsz(g, 1) = 0.5 * (ymax - ymin);
        d_gsz(g, 2) = 0.5 * (zmax - zmin);
    });
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
// Compute cell center-of-mass accelerations on GPU.
// Each cell independently reduces over its particle range [pbeg, pend).
// TeamPolicy: one team per cell, threads cooperatively reduce particles.
// No bottom-up dependency — every cell is independent.
// ==========================================================================
void Octree::compute_cell_accelerations(Particles& p) {
    if (n_cells <= 0 || n_particles <= 0) return;

    if ((int)cell_accel.extent(0) != n_cells)
        cell_accel = Kokkos::View<double*[3], Kokkos::LayoutRight>("cell_accel", n_cells);

    auto ax = p.ax;
    auto ay = p.ay;
    auto az = p.az;
    auto mass = p.mass;
    auto si = sorted_indices;
    auto pbeg_v = cell_pbeg;
    auto pend_v = cell_pend;
    auto com = cell_com;
    auto ca = cell_accel;
    int nc = n_cells;

    constexpr int TEAM_SZ = 32;
    // Scratch: 3 doubles per thread for partial sums (sax, say, saz)
    size_t scratch_bytes = 3 * TEAM_SZ * sizeof(double);

    using team_policy = Kokkos::TeamPolicy<>;
    using team_member = team_policy::member_type;
    using scratch_t = Kokkos::View<double*, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                                    Kokkos::MemoryUnmanaged>;

    Kokkos::parallel_for("cell_accel_gpu",
        team_policy(nc, TEAM_SZ).set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)),
        KOKKOS_LAMBDA(const team_member& team) {
            int c = team.league_rank();
            int tid = team.team_rank();
            int tsz = team.team_size();
            scratch_t scratch(team.team_scratch(0), 3 * tsz);

            double cell_m = com(c, 3);
            int b = pbeg_v(c), e = pend_v(c);

            // Team-stride accumulation
            double lax = 0, lay = 0, laz = 0;
            for (int i = b + tid; i < e; i += tsz) {
                int orig = si(i);
                double m = mass(orig);
                lax += m * ax(orig);
                lay += m * ay(orig);
                laz += m * az(orig);
            }

            // Write partials to scratch
            scratch(tid)          = lax;
            scratch(tid + tsz)    = lay;
            scratch(tid + 2*tsz)  = laz;
            team.team_barrier();

            // Thread 0 reduces and writes result
            if (tid == 0) {
                double sax = 0, say = 0, saz = 0;
                for (int t = 0; t < tsz; t++) {
                    sax += scratch(t);
                    say += scratch(t + tsz);
                    saz += scratch(t + 2*tsz);
                }
                if (cell_m > 0) {
                    double inv = 1.0 / cell_m;
                    ca(c, 0) = sax * inv;
                    ca(c, 1) = say * inv;
                    ca(c, 2) = saz * inv;
                } else {
                    ca(c, 0) = ca(c, 1) = ca(c, 2) = 0;
                }
            }
        });
}

// ==========================================================================
// Compute cell_accel from active particles only (sflag != 2).
// For GADGET-4: cell monopole accelerations for gradient correction.
// ==========================================================================
void Octree::compute_active_cell_accelerations(Particles& p) {
    if (n_cells <= 0 || n_particles <= 0) return;

    if ((int)cell_accel.extent(0) != n_cells)
        cell_accel = Kokkos::View<double*[3], Kokkos::LayoutRight>("cell_accel", n_cells);

    auto ax = p.ax;
    auto ay = p.ay;
    auto az = p.az;
    auto mass = p.mass;
    auto si = sorted_indices;
    auto flg = sorted_subset_flag;
    auto pbeg_v = cell_pbeg;
    auto pend_v = cell_pend;
    auto sp = sorted_pos;
    auto ca = cell_accel;
    int nc = n_cells;

    constexpr int TEAM_SZ = 32;
    size_t scratch_bytes = 4 * TEAM_SZ * sizeof(double);  // m_active, max, may, maz

    using team_policy = Kokkos::TeamPolicy<>;
    using team_member = team_policy::member_type;
    using scratch_t = Kokkos::View<double*, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                                    Kokkos::MemoryUnmanaged>;

    Kokkos::parallel_for("active_cell_accel",
        team_policy(nc, TEAM_SZ).set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)),
        KOKKOS_LAMBDA(const team_member& team) {
            int c = team.league_rank();
            int tid = team.team_rank();
            int tsz = team.team_size();
            scratch_t scratch(team.team_scratch(0), 4 * tsz);

            int b = pbeg_v(c), e = pend_v(c);

            double lm = 0, lax = 0, lay = 0, laz = 0;
            for (int i = b + tid; i < e; i += tsz) {
                if (flg(i) == 2) continue;  // skip inactive particles
                int orig = si(i);
                double m = sp(i, 3);  // use sorted mass (consistent)
                lm += m;
                lax += m * ax(orig);
                lay += m * ay(orig);
                laz += m * az(orig);
            }

            scratch(tid)          = lm;
            scratch(tid + tsz)    = lax;
            scratch(tid + 2*tsz)  = lay;
            scratch(tid + 3*tsz)  = laz;
            team.team_barrier();

            if (tid == 0) {
                double sm = 0, sax = 0, say = 0, saz = 0;
                for (int t = 0; t < tsz; t++) {
                    sm  += scratch(t);
                    sax += scratch(t + tsz);
                    say += scratch(t + 2*tsz);
                    saz += scratch(t + 3*tsz);
                }
                if (sm > 0) {
                    double inv = 1.0 / sm;
                    ca(c, 0) = sax * inv;
                    ca(c, 1) = say * inv;
                    ca(c, 2) = saz * inv;
                } else {
                    ca(c, 0) = ca(c, 1) = ca(c, 2) = 0;
                }
            }
        });
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

    // 4. Recompute dual monopoles (including velocities)
    recompute_dual_monopoles(p);
}

// ==========================================================================
// Three-flag update_subsets for GADGET-4 cross-force decomposition.
// [start, start+slow_count) → flag=0 (slow)
// [start+slow_count, start+count) → flag=1 (fast)
// Everything else → flag=2 (not at this level)
// ==========================================================================
void Octree::update_subsets_3flag(Particles& p, int start, int count, int slow_count) {
    if (n_particles <= 0) return;

    // 1. Fix sorted_indices after particle reordering
    rebuild_sorted_indices(p);

    // 2. Refresh positions from updated sorted_indices
    refresh_positions(p);

    // 3. Mark three-flag system per sorted particle
    auto si = sorted_indices;
    auto flag = sorted_subset_flag;
    int n = n_particles;
    int ss = start, sc = slow_count, tc = count;
    Kokkos::parallel_for("mark_subsets_3flag", n, KOKKOS_LAMBDA(int j) {
        int idx = si(j);
        if (idx >= ss && idx < ss + sc)
            flag(j) = 0;        // slow
        else if (idx >= ss + sc && idx < ss + tc)
            flag(j) = 1;        // fast
        else
            flag(j) = 2;        // not at this level
    });

    // 4. Recompute dual monopoles (only flag=0 and flag=1 contribute)
    recompute_dual_monopoles(p);
}

// ==========================================================================
// Recompute total cell monopoles (cell_com) from sorted_pos.
// Each cell reduces over ALL its particles regardless of flag.
// Call after refresh_positions() when F_all tree walks need current cell_com.
// ==========================================================================
void Octree::recompute_total_monopoles() {
    if (n_cells <= 0 || n_particles <= 0) return;

    auto sp = sorted_pos;
    auto si = sorted_indices;
    auto pbeg_v = cell_pbeg;
    auto pend_v = cell_pend;
    auto com = cell_com;
    auto vel_v = cell_vel;
    int nc = n_cells;

    constexpr int TEAM_SZ = 32;

    using team_policy = Kokkos::TeamPolicy<>;
    using team_member = team_policy::member_type;
    using scratch_t = Kokkos::View<double*, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                                    Kokkos::MemoryUnmanaged>;

    // NOTE: We don't have velocity in sorted_pos, and we don't need cell_vel for
    // the total monopoles (F_all only uses cell_com, not cell_vel).
    // So we only recompute {cx, cy, cz, mass} in cell_com.

    Kokkos::parallel_for("recompute_total_mono",
        team_policy(nc, TEAM_SZ).set_scratch_size(0, Kokkos::PerTeam(4 * TEAM_SZ * sizeof(double))),
        KOKKOS_LAMBDA(const team_member& team) {
            int c = team.league_rank();
            int tid = team.team_rank();
            int tsz = team.team_size();
            scratch_t scratch(team.team_scratch(0), 4 * tsz);

            int b = pbeg_v(c), e = pend_v(c);

            double lm = 0, lx = 0, ly = 0, lz = 0;
            for (int i = b + tid; i < e; i += tsz) {
                double m = sp(i, 3);
                lm += m;
                lx += m * sp(i, 0);
                ly += m * sp(i, 1);
                lz += m * sp(i, 2);
            }

            scratch(tid)          = lm;
            scratch(tid + tsz)    = lx;
            scratch(tid + 2*tsz)  = ly;
            scratch(tid + 3*tsz)  = lz;
            team.team_barrier();

            if (tid == 0) {
                double sm = 0, sx = 0, sy = 0, sz = 0;
                for (int t = 0; t < tsz; t++) {
                    sm += scratch(t);
                    sx += scratch(t + tsz);
                    sy += scratch(t + 2*tsz);
                    sz += scratch(t + 3*tsz);
                }
                if (sm > 0) {
                    double inv = 1.0 / sm;
                    com(c, 0) = sx * inv;
                    com(c, 1) = sy * inv;
                    com(c, 2) = sz * inv;
                } else {
                    com(c, 0) = com(c, 1) = com(c, 2) = 0;
                }
                com(c, 3) = sm;
            }
        });
}

// ==========================================================================
// Recompute dual monopoles (cell_com_slow/fast, cell_vel_slow/fast) on GPU.
// Each cell independently reduces over its particle range [pbeg, pend).
// TeamPolicy: one team per cell, threads cooperatively reduce particles.
// No bottom-up dependency — every cell is independent.
// Only flag=0 (slow) and flag=1 (fast) particles contribute; flag=2 skipped.
// ==========================================================================
void Octree::recompute_dual_monopoles(Particles& p) {
    if (n_cells <= 0 || n_particles <= 0) return;

    // Allocate dual monopoles if needed
    if ((int)cell_com_slow.extent(0) < n_cells) {
        cell_com_slow = Kokkos::View<double*[4], Kokkos::LayoutRight>("cell_com_slow", n_cells);
        cell_com_fast = Kokkos::View<double*[4], Kokkos::LayoutRight>("cell_com_fast", n_cells);
    }
    if ((int)cell_vel_slow.extent(0) < n_cells) {
        cell_vel_slow = Kokkos::View<double*[3], Kokkos::LayoutRight>("cell_vel_slow", n_cells);
        cell_vel_fast = Kokkos::View<double*[3], Kokkos::LayoutRight>("cell_vel_fast", n_cells);
    }

    auto sp = sorted_pos;
    auto flg = sorted_subset_flag;
    auto si = sorted_indices;
    auto pvx = p.vx;
    auto pvy = p.vy;
    auto pvz = p.vz;
    auto pbeg_v = cell_pbeg;
    auto pend_v = cell_pend;
    auto com_s = cell_com_slow;
    auto com_f = cell_com_fast;
    auto vel_s = cell_vel_slow;
    auto vel_f = cell_vel_fast;
    int nc = n_cells;

    constexpr int TEAM_SZ = 32;
    // Scratch: 14 doubles per thread (7 slow + 7 fast: m,x,y,z,vx,vy,vz each)
    size_t scratch_bytes = 14 * TEAM_SZ * sizeof(double);

    using team_policy = Kokkos::TeamPolicy<>;
    using team_member = team_policy::member_type;
    using scratch_t = Kokkos::View<double*, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                                    Kokkos::MemoryUnmanaged>;

    Kokkos::parallel_for("dual_mono_gpu",
        team_policy(nc, TEAM_SZ).set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)),
        KOKKOS_LAMBDA(const team_member& team) {
            int c = team.league_rank();
            int tid = team.team_rank();
            int tsz = team.team_size();
            scratch_t scratch(team.team_scratch(0), 14 * tsz);

            int b = pbeg_v(c), e = pend_v(c);

            // Team-stride accumulation of 14 values
            double lsm_s = 0, lsx_s = 0, lsy_s = 0, lsz_s = 0;
            double lsm_f = 0, lsx_f = 0, lsy_f = 0, lsz_f = 0;
            double lsvx_s = 0, lsvy_s = 0, lsvz_s = 0;
            double lsvx_f = 0, lsvy_f = 0, lsvz_f = 0;

            for (int i = b + tid; i < e; i += tsz) {
                double m = sp(i, 3);
                double x = sp(i, 0), y = sp(i, 1), z = sp(i, 2);
                int orig = si(i);
                double vx = pvx(orig), vy = pvy(orig), vz = pvz(orig);
                int f = flg(i);
                if (f == 0) {  // slow
                    lsm_s += m; lsx_s += m * x; lsy_s += m * y; lsz_s += m * z;
                    lsvx_s += m * vx; lsvy_s += m * vy; lsvz_s += m * vz;
                } else if (f == 1) {  // fast (skip flag=2 = not at this level)
                    lsm_f += m; lsx_f += m * x; lsy_f += m * y; lsz_f += m * z;
                    lsvx_f += m * vx; lsvy_f += m * vy; lsvz_f += m * vz;
                }
            }

            // Write partials to scratch (14 values per thread)
            scratch(tid)            = lsm_s;
            scratch(tid + tsz)      = lsx_s;
            scratch(tid + 2*tsz)    = lsy_s;
            scratch(tid + 3*tsz)    = lsz_s;
            scratch(tid + 4*tsz)    = lsvx_s;
            scratch(tid + 5*tsz)    = lsvy_s;
            scratch(tid + 6*tsz)    = lsvz_s;
            scratch(tid + 7*tsz)    = lsm_f;
            scratch(tid + 8*tsz)    = lsx_f;
            scratch(tid + 9*tsz)    = lsy_f;
            scratch(tid + 10*tsz)   = lsz_f;
            scratch(tid + 11*tsz)   = lsvx_f;
            scratch(tid + 12*tsz)   = lsvy_f;
            scratch(tid + 13*tsz)   = lsvz_f;
            team.team_barrier();

            // Thread 0 reduces across team and writes result
            if (tid == 0) {
                double sm_s = 0, sx_s = 0, sy_s = 0, sz_s = 0;
                double sm_f = 0, sx_f = 0, sy_f = 0, sz_f = 0;
                double svx_s = 0, svy_s = 0, svz_s = 0;
                double svx_f = 0, svy_f = 0, svz_f = 0;
                for (int t = 0; t < tsz; t++) {
                    sm_s  += scratch(t);
                    sx_s  += scratch(t + tsz);
                    sy_s  += scratch(t + 2*tsz);
                    sz_s  += scratch(t + 3*tsz);
                    svx_s += scratch(t + 4*tsz);
                    svy_s += scratch(t + 5*tsz);
                    svz_s += scratch(t + 6*tsz);
                    sm_f  += scratch(t + 7*tsz);
                    sx_f  += scratch(t + 8*tsz);
                    sy_f  += scratch(t + 9*tsz);
                    sz_f  += scratch(t + 10*tsz);
                    svx_f += scratch(t + 11*tsz);
                    svy_f += scratch(t + 12*tsz);
                    svz_f += scratch(t + 13*tsz);
                }
                if (sm_s > 0) {
                    double inv = 1.0 / sm_s;
                    com_s(c, 0) = sx_s * inv; com_s(c, 1) = sy_s * inv; com_s(c, 2) = sz_s * inv;
                    vel_s(c, 0) = svx_s * inv; vel_s(c, 1) = svy_s * inv; vel_s(c, 2) = svz_s * inv;
                } else {
                    com_s(c, 0) = com_s(c, 1) = com_s(c, 2) = 0;
                    vel_s(c, 0) = vel_s(c, 1) = vel_s(c, 2) = 0;
                }
                com_s(c, 3) = sm_s;
                if (sm_f > 0) {
                    double inv = 1.0 / sm_f;
                    com_f(c, 0) = sx_f * inv; com_f(c, 1) = sy_f * inv; com_f(c, 2) = sz_f * inv;
                    vel_f(c, 0) = svx_f * inv; vel_f(c, 1) = svy_f * inv; vel_f(c, 2) = svz_f * inv;
                } else {
                    com_f(c, 0) = com_f(c, 1) = com_f(c, 2) = 0;
                    vel_f(c, 0) = vel_f(c, 1) = vel_f(c, 2) = 0;
                }
                com_f(c, 3) = sm_f;
            }
        });
}

// ==========================================================================
// Compute dual cell accelerations (slow/fast) on GPU.
// Each cell independently reduces over its particle range [pbeg, pend).
// TeamPolicy: one team per cell, threads cooperatively reduce particles.
// No bottom-up dependency — every cell is independent.
// ==========================================================================
void Octree::compute_dual_cell_accelerations(Particles& p) {
    if (n_cells <= 0 || n_particles <= 0) return;

    if ((int)cell_accel_slow.extent(0) < n_cells) {
        cell_accel_slow = Kokkos::View<double*[3], Kokkos::LayoutRight>("cell_accel_slow", n_cells);
        cell_accel_fast = Kokkos::View<double*[3], Kokkos::LayoutRight>("cell_accel_fast", n_cells);
    }

    auto ax = p.ax;
    auto ay = p.ay;
    auto az = p.az;
    auto mass = p.mass;
    auto si = sorted_indices;
    auto flg = sorted_subset_flag;
    auto pbeg_v = cell_pbeg;
    auto pend_v = cell_pend;
    auto com_s = cell_com_slow;
    auto com_f = cell_com_fast;
    auto ca_s = cell_accel_slow;
    auto ca_f = cell_accel_fast;
    int nc = n_cells;

    constexpr int TEAM_SZ = 32;
    // Scratch: 6 doubles per thread (3 slow + 3 fast: ax,ay,az each)
    size_t scratch_bytes = 6 * TEAM_SZ * sizeof(double);

    using team_policy = Kokkos::TeamPolicy<>;
    using team_member = team_policy::member_type;
    using scratch_t = Kokkos::View<double*, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                                    Kokkos::MemoryUnmanaged>;

    Kokkos::parallel_for("dual_accel_gpu",
        team_policy(nc, TEAM_SZ).set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)),
        KOKKOS_LAMBDA(const team_member& team) {
            int c = team.league_rank();
            int tid = team.team_rank();
            int tsz = team.team_size();
            scratch_t scratch(team.team_scratch(0), 6 * tsz);

            int b = pbeg_v(c), e = pend_v(c);
            double ms = com_s(c, 3), mf = com_f(c, 3);

            // Team-stride accumulation of 6 values
            double lax_s = 0, lay_s = 0, laz_s = 0;
            double lax_f = 0, lay_f = 0, laz_f = 0;

            for (int i = b + tid; i < e; i += tsz) {
                int orig = si(i);
                double m = mass(orig);
                int f = flg(i);
                if (f == 0) {
                    lax_s += m * ax(orig); lay_s += m * ay(orig); laz_s += m * az(orig);
                } else if (f == 1) {  // skip flag=2 (not at this level)
                    lax_f += m * ax(orig); lay_f += m * ay(orig); laz_f += m * az(orig);
                }
            }

            // Write partials to scratch
            scratch(tid)          = lax_s;
            scratch(tid + tsz)    = lay_s;
            scratch(tid + 2*tsz)  = laz_s;
            scratch(tid + 3*tsz)  = lax_f;
            scratch(tid + 4*tsz)  = lay_f;
            scratch(tid + 5*tsz)  = laz_f;
            team.team_barrier();

            // Thread 0 reduces across team and writes result
            if (tid == 0) {
                double sax_s = 0, say_s = 0, saz_s = 0;
                double sax_f = 0, say_f = 0, saz_f = 0;
                for (int t = 0; t < tsz; t++) {
                    sax_s += scratch(t);
                    say_s += scratch(t + tsz);
                    saz_s += scratch(t + 2*tsz);
                    sax_f += scratch(t + 3*tsz);
                    say_f += scratch(t + 4*tsz);
                    saz_f += scratch(t + 5*tsz);
                }
                if (ms > 0) { double inv = 1.0/ms; ca_s(c,0) = sax_s*inv; ca_s(c,1) = say_s*inv; ca_s(c,2) = saz_s*inv; }
                else { ca_s(c,0) = ca_s(c,1) = ca_s(c,2) = 0; }
                if (mf > 0) { double inv = 1.0/mf; ca_f(c,0) = sax_f*inv; ca_f(c,1) = say_f*inv; ca_f(c,2) = saz_f*inv; }
                else { ca_f(c,0) = ca_f(c,1) = ca_f(c,2) = 0; }
            }
        });
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
