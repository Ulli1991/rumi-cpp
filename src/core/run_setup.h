#pragma once
// Run settings shared by the standalone program (main.cpp) and the library (lib/rumi_lib.cpp): the config keys
// that set the integrator's globals (time-step and subsystem parameters, search thresholds, MPI transport)
#include "params.h"

struct RunSettings {
    double soft_pc = 0.0;          // SOFTENING_PC in, 0 for collisional runs out
    bool ngb_mass_scaling = false; // NGB_MASS_SCALING
    double ngb_mref = 0.0;         // NGB_MREF
    double e_ref_set = 0.0;        // E_REF
};

// Sets the integrator globals from cfg; returns non-zero on an invalid combination (message printed on rank 0)
int apply_run_settings(const Config& cfg, RunSettings& rs);
