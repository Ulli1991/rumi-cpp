#pragma once

// Gravitational-wave observables of compact-object binaries (docs/CASES.md E2): one ASCII table gw_events.dat with
//   MERGER rows: every merger of two compact objects (NS kw 13, BH kw 14) at the moment the code decides it
//                (masses, spins, a, e, frequencies, e at f_GW = 10 Hz and 10 mHz, chi_eff, chi_p, remnant, path,
//                environment, channel);
//   BINARY rows: at every snapshot, the bound compact pairs of the subsystems with a Peters time below
//                GW_CENSUS_TGW_MYR (the LISA-band and ejected populations: ejected binaries are never removed as
//                escapers, so the census counts them; their merger time is t_gw).
// The output only reads the state (bit-identical particles with it on or off; env NBODY_GW_OUTPUT=0 turns it off).
// Restart-safe: rows are buffered (the buffer is part of the step-restart checkpoint), appended at the print points
// and before every restart file, and the restart stores the number of rows written; a run resumed from it truncates
// the file to that many rows first (no duplicates, no gaps).
//
// Definitions (literature in docs/CASES.md E2):
//   f_GW     peak GW frequency of an eccentric orbit, Wen (2003, ApJ 598, 419) Eq. 37:
//            f = sqrt(G M) (1 + e)^1.1954 / (pi [a (1 - e)]^1.5)  (as CMC, Samsing 2018, DRAGON-II)
//   e_f      eccentricity where the orbit-averaged Peters (1964) track through the decision state (a, e),
//            a(e) = c0 e^(12/19) (1 + 121/304 e^2)^(870/2299) / (1 - e^2), has f_GW = f (backward or forward).
//            f_GW along a track has a minimum at e = 0.7497; the root is the one below it (the last entry into the
//            band, after which f_GW rises to the merger); -1: f_GW above f on the whole track (formed in the band);
//            e >= 1 at the decision: the track with the same pericentre and e -> 1
//   t_gw     Peters (1964) merger time, Mandel (2021, RNAAS 5, 223) fit: T_c (1 + 0.27 e^10 + 0.33 e^20 +
//            0.2 e^1000) (1 - e^2)^3.5, T_c = a^4 / (4 beta), beta = 64/5 G^3 m1 m2 M / c^5 (3 % accurate)
//   chi_eff  (m1 chi1 + m2 chi2) . L_hat / M;  chi_p Schmidt et al. (2015, PRD 91, 024043):
//            max(chi1_perp, (4 q + 3) / (4 + 3 q) q chi2_perp), body 1 the heavier, q = m2 / m1

#include <string>
#include <vector>

struct Particles;
struct SubsystemManager;

struct GwRow {
    int kind = 0;                 // 0 MERGER, 1 BINARY (census)
    double t = 0;                 // code time
    int path = 0;                 // GwPath
    int type = -1;                // MergerType (MERGER rows), -1 for census rows
    int channel = 0;              // GwChannel
    int pid1 = -1, pid2 = -1, kw1 = 0, kw2 = 0;
    double m1 = 0, m2 = 0;
    double chi1[3] = {}, chi2[3] = {};
    double a = 0, e = 0;          // code units (pc); Newtonian osculating elements of the relative orbit
    double f_orb = 0, f_gw = 0;   // Hz
    double e_10hz = 0, e_10mhz = 0;
    double t_gw = 0;              // code time
    double chi_eff = 0, chi_p = 0;
    double m_rem = 0, chi_rem[3] = {}, v_kick = 0;
    int n_members = 0;
    double tidal = 0;             // max over the other bodies of (m_k / d_k^3) / (M / r^3), r the pair separation
    double r_cl = 0, v_cl = 0;    // distance (pc) and speed (km/s) of the pair's c.o.m. from the cluster's c.o.m.
};

enum GwPath { GW_PATH_SUBSYS = 0, GW_PATH_CONTACT = 1, GW_PATH_PETERS = 2, GW_PATH_SECULAR = 3, GW_PATH_CENSUS = 4 };
enum GwChannel { GW_CH_2BODY = 0, GW_CH_FEWBODY = 1, GW_CH_CAPTURE = 2, GW_CH_EJECTED = 3, GW_CH_CLUSTER = 4 };

struct GwLog {
    bool on = true;               // env NBODY_GW_OUTPUT=0: off
    double census_tgw = 0.0;      // GW_CENSUS_TGW_MYR in code time (0: no census)
    std::string path = "gw_events.dat";
    std::vector<GwRow> pending;   // recorded, not yet in the file
    long long n_written = 0;      // rows in the file (all ranks count; rank 0 writes)

    // Start of a run: a new file with the header, or (resume = true) the file truncated to n_written rows
    void open(bool resume);
    // Append the pending rows (rank 0), all ranks advance n_written
    void flush();
};

extern GwLog g_gw_log;

// Pure helpers (unit-tested through the run tests)
double gw_f_peak_hz(double a, double e, double M, double G);                 // Wen (2003), code units in
double gw_e_at_f(double a, double e, double M, double G, double f_hz);      // Peters track, see above
double gw_t_merge(double a, double e, double m1, double m2, double G, double c);
void gw_chi_eff_p(double m1, double m2, const double chi1[3], const double chi2[3], const double L[3],
                  double& chi_eff, double& chi_p);

// Fill the orbital and spin columns of a row from the pair's relative state (r = x2 - x1, v = v2 - v1)
void gw_fill_pair(GwRow& row, const double r[3], const double v[3], double G, double c);

// Environment columns from the full particle set (host arrays; skip1, skip2 the pair): tidal, r_cl, v_cl, and the
// channel (row.e and row.n_members set before)
void gw_environment(GwRow& row, int N, const double* x, const double* y, const double* z, const double* vx,
                    const double* vy, const double* vz, const double* m, int skip1, int skip2, const double xc[3],
                    const double vc[3], double r_sep, double G);

// Is this pair one the table records (both NS / BH)?
inline bool gw_compact_pair(int kw1, int kw2) { return kw1 >= 13 && kw1 <= 14 && kw2 >= 13 && kw2 <= 14; }

// Census of the bound compact pairs in subsystems with t_gw < census_tgw (snapshot times)
void gw_census(const Particles& p, const SubsystemManager& mgr, double G, double c, double t);
