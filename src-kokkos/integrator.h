#pragma once

#include "particles.h"
#include <vector>
#include <string>

// Set force function pointers based on UseN3L flag
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
