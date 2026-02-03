#pragma once

#include "particles.h"
#include <vector>
#include <string>

// Runtime function pointer type for N3L dispatch
using AccFn = void(*)(Particles&, int, int, double, double);
using AccGradFn = void(*)(Particles&, int, int, double, double, double);
using CrossAccFn = void(*)(Particles&, int, int, int, int, double, double);
using CrossAccGradFn = void(*)(Particles&, int, int, int, int, double, double, double);

// Set the function pointers based on UseN3L flag
void set_force_dispatch(bool use_n3l);

// Single-level FSI step on [start, start+count)
void fsi_step(Particles& p, int start, int count,
              double G, double soft2, double dt);

// HHS-FSI recursive step with dynamic pivot-based partitioning (per paper's Algorithm 4)
// Partitions particles [start, start+count) into slow (dt >= pivot) and fast (dt < pivot)
// Recomputes FROST for the current subset at each level
void hhs_fsi_step_dynamic(Particles& p, int start, int count, double pivot_dt,
                          double G, double soft2, double dt_min,
                          double eta, bool use_n3l);

// Legacy HHS-FSI with pre-computed global levels (for comparison)
void hhs_fsi_step(Particles& p, int level, double dt,
                  double G, double soft2,
                  const std::vector<double>& level_dts,
                  double eta = 0.0, bool use_n3l = false,
                  bool reassign_timesteps = false);

enum class StackOp {
    CROSS_NEWTON  = 0,
    FSI_HALF      = 1,
    RECURSE       = 2,
    CROSS_GRADIENT= 3,
    FSI_BASE      = 4,
};

struct StackEntry {
    int op;
    int level;
    double dt;
};

void hhs_fsi_step_stack(Particles& p, int start_level, double dt,
                        double G, double soft2,
                        std::vector<StackEntry>& stack);

enum class HHSMode { Recursive, Stack };

inline HHSMode parse_hhs_mode(const std::string& s) {
    if (s == "stack") return HHSMode::Stack;
    return HHSMode::Recursive;
}
