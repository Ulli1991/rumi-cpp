#pragma once
// Science snapshots (HDF5) in the layout of the LUNA hydro code (src/luna/file_io), so that
// LUNA's read_snapshot() and its Gadget-style particle tools read RUMI++ output directly (HARD_PROBLEMS.md G2).
//
//   <SNAPSHOT_DIR>/snapdir_NNN/<SNAPSHOT_BASE>_NNN.0.hdf5      NNN = snapshot number (3+ digits), one file
//
//   /Header      Gadget/Arepo attributes as LUNA's write_gadget.py writes them: BoxSize, Time, Redshift (0),
//                Omega0, OmegaLambda (0), HubbleParam (1), NumPart_ThisFile[6], NumPart_Total[6],
//                NumPart_Total_HighWord[6], NumFilesPerSnapshot (1), MassTable[6] (0: masses per particle),
//                config_json; plus N-body attributes: Time_Myr, Step, SnapshotIndex, N, G, Softening_pc,
//                E_kin, E_pot, E_total, E_total_initial, dE_rel (with the books), E_escape E_gw E_merge E_sse E_pn
//                E_sd E_tidal, P_sink[3], L_sink[3], Code ("cpp-nbody"), CodeVersion, DoublePrecision (1).
//                SNAPSHOT_CLUSTER_DIAG (default on; io/cluster_diag.h): Cluster_MassTotal, Cluster_DensityCentre[3]
//                (pc), Cluster_DensityCentreVelocity[3] (km/s), Cluster_CoreRadius_pc, Cluster_CoreDensity
//                (Msun/pc^3), Cluster_Sigma1D_kms, Cluster_LagrangianFractions[7], Cluster_LagrangianRadii_pc[7],
//                Cluster_NBinaries, Cluster_NBinariesHard, Cluster_EBinaries. TIDAL_FIELD runs: Tidal_Field (name),
//                Tidal_RefPosition[3] (pc from the galaxy centre), Tidal_RefVelocity[3] (km/s), Tidal_JacobiRadius_pc
//                (current total mass, at reference point + density centre; 0 without the cluster block).
//                Coordinates are about the origin of the run (-BoxSize/2 .. BoxSize/2, not periodic).
//   /PartType4   stars, white dwarfs and neutron stars (stellar types 0-13, 15; every particle of a collisionless run)
//   /PartType5   black holes (stellar type 14)
//                Coordinates [n,3] pc, Velocities [n,3] km/s, Masses [n] M_sun, ParticleIDs [n] uint64 (permanent
//                id), Potential [n] (km/s)^2 (SNAPSHOT_POTENTIAL, direct sum with the run's softening),
//                TimeStep [n] TU, TimeBin [n] (HHS level); collisional runs also StellarType, StellarRadius [R_sun],
//                Luminosity [L_sun], CoreMass, ZAMSMass [M_sun], StellarAge [Myr], Spin [n,3] (dimensionless chi),
//                SubsystemID (-1: field), BinaryPartnerID (int64: the member of the same subsystem with the most
//                negative two-body energy; -1: none)
//   /cells       LUNA grid group, a placeholder: read_snapshot() needs one. One cell spanning the box, density =
//                total mass / BoxSize^3, velocity = centre-of-mass velocity, pressure = internal energy = 0,
//                gamma = 5/3, plus the attributes read_snapshot() reads; attribute placeholder = 1
//   /units       LUNA units group: UnitLength_in_cm (1 pc), UnitMass_in_g (1 M_sun), UnitVelocity_in_cm_per_s
//                (1 km/s), UnitTime_in_s, UnitDensity_in_cgs, UnitPressure_in_cgs, UnitEnergy_in_cgs

#include "particles.h"
#include "diagnostics.h"
#include "cluster_diag.h"
#include <string>

struct SnapshotOptions {
    std::string dir = ".";          // SNAPSHOT_DIR
    std::string base = "snapshot";  // SNAPSHOT_BASE
    bool potential = true;          // SNAPSHOT_POTENTIAL: O(N^2) potential of every particle
};

struct SnapshotMeta {
    double t = 0.0;                 // code time (TU)
    int step = 0;
    int index = 0;                  // NNN
    double G = 0.0, soft2 = 0.0;
    DiagnosticState diag{}, diag0{};
    std::string config_json;
    bool has_cluster = false;       // SNAPSHOT_CLUSTER_DIAG: cluster structure of this state in the Header
    ClusterDiag cluster{};
};

// Write one snapshot (one rank; through <file>.partial and a rename). Returns the path.
std::string write_snapshot(const SnapshotOptions& opt, const Particles& p, const SnapshotMeta& m);
