#pragma once

#include "particles.h"
#include "post_newtonian_core.h"

// Global PN1.0 velocity kick for black holes of at least m_pn outside subsystems
void global_pn1_kick(Particles& p, int start, int count,
                     double G, double c_code, double m_pn, double dt);
