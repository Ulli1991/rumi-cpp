#pragma once

#include "particles.h"
#include <vector>
#include <string>

// Single-level FSI step on [start, start+count)
// U(dt/6) -> T(dt/2) -> Ũ(2dt/3) -> T(dt/2) -> U(dt/6)
void fsi_step(Particles& p, int start, int count,
              double G, double soft2, double dt);

// HHS-FSI recursive step
void hhs_fsi_step(Particles& p, int level, double dt,
                  double G, double soft2,
                  const std::vector<double>& level_dts);

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

// HHS-FSI stack-based step (flat while loop, no recursion)
void hhs_fsi_step_stack(Particles& p, int start_level, double dt,
                        double G, double soft2,
                        std::vector<StackEntry>& stack);

enum class HHSMode { Recursive, Stack };

inline HHSMode parse_hhs_mode(const std::string& s) {
    if (s == "stack") return HHSMode::Stack;
    return HHSMode::Recursive;
}
