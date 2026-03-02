#pragma once

#include <string>

// =========================================================================
// SSE model selection enums
// =========================================================================

enum class WindModel {
    HURLEY         = 1,  // NdJ + Reimers + VW + WR + additive LBV (Hurley 2000)
    HURLEY_LBV     = 2,  // Same but flat-rate LBV (1e-4 * flbv)
    VINK           = 3,  // Vink 2001 hot-star + Z-dep WR (DRAGON-II)
    VINK_NO_BISTAB = 4,  // Vink without bi-stability jump
};

enum class RemnantModel {
    HURLEY         = 0,  // Linear Mc->Mt (Hurley 2000)
    BELCZYNSKI02   = 1,  // FeNi core + fallback (current default)
    BELCZYNSKI08   = 2,  // Updated FeNi core (Belczynski 2008)
    FRYER_RAPID    = 3,  // Rapid SN, mass gap (Fryer 2012)
    FRYER_DELAYED  = 4,  // Delayed SN, no gap (DRAGON-II default)
    ELDRIDGE_TOUT  = 5,  // Quadratic (Eldridge & Tout 2004)
};

enum class PPISNModel {
    NONE     = 0,  // No PPISN (current)
    STRONG   = 1,  // Cap at 45 Msun, PISN 65-135 (Belczynski 2016)
    MODERATE = 2,  // Piecewise cap (Leung 2019)
    WEAK     = 3,  // Higher cap (Leung 2019)
};

enum class ECSNModel { OFF = 0, ON = 1 };

// =========================================================================
// SSE configuration
// =========================================================================

struct SSEConfig {
    double neta = 0.5, bwind = 0.0, hewind = 1.0, flbv = 1.5;
    double mxns = -1.0;  // -1 = auto (1.8 for HURLEY remnant, 2.5 otherwise)
    WindModel    wind    = WindModel::HURLEY;
    RemnantModel remnant = RemnantModel::BELCZYNSKI02;
    PPISNModel   ppisn   = PPISNModel::NONE;
    ECSNModel    ecsn    = ECSNModel::OFF;
    int ifflag = 1, wdflag = 1;
    double pts1 = 0.05, pts2 = 0.01, pts3 = 0.02;

    double effective_mxns() const {
        if (mxns > 0) return mxns;
        return (remnant == RemnantModel::HURLEY) ? 1.8 : 3.0;
    }
};

// Parse string config values into SSEConfig
SSEConfig make_sse_config(const std::string& wind_str,
                          const std::string& remnant_str,
                          const std::string& ppisn_str,
                          const std::string& ecsn_str,
                          double mxns, double flbv);

// =========================================================================
// SSE result
// =========================================================================

struct SSEResult {
    int kw;           // Stellar type (0-15)
    double mass;      // Final mass (M_sun)
    double mass_core; // Core mass (M_sun)
    double radius;    // Stellar radius (R_sun)
    double luminosity;// Luminosity (L_sun)
};

// =========================================================================
// SSE context (stellar evolution engine)
// =========================================================================

struct RemnantResult {
    double mt;    // remnant mass
    int kw;       // stellar type (13=NS, 14=BH, 15=massless SN)
    double fbfac; // fallback fraction (informational)
};

class SSEContext {
public:
    explicit SSEContext(double Z, const SSEConfig& cfg = SSEConfig{});
    SSEResult evolve(double m_zams, double age_myr) const;

private:
    double Z_;
    SSEConfig cfg_;
    double zpars_[20];
    double msp_[200];
    double gbp_[200];

    void init_coefficients();

    // Fitting formulae from zfuncs.f (all use msp_/gbp_ arrays)
    double lzamsf(double m) const;
    double rzamsf(double m) const;
    double tbgbf(double m) const;
    double thookf(double m) const;
    double ltmsf(double m) const;
    double lalphf(double m) const;
    double lbetaf(double m) const;
    double lnetaf(double m) const;
    double lhookf(double m, double mhook) const;
    double rtmsf(double m) const;
    double ralphf(double m) const;
    double rbetaf(double m) const;
    double rgammf(double m) const;
    double rhookf(double m, double mhook) const;
    double lbgbf(double m) const;
    double lbagbf(double m, double mhefl) const;
    double rgbf(double m, double lum) const;
    double ragbf(double m, double lum, double mhelf) const;
    double mctmsf(double m) const;
    double mcheif(double m, double mhefl, double mchefl) const;
    double mcagbf(double m) const;
    double mbagbf(double mc) const;
    double mcgbtf(double t, double A, const double GB[10],
                  double tinf1, double tinf2, double tx) const;
    double lgbtf(double t, double A, const double GB[10],
                 double tinf1, double tinf2, double tx) const;
    double mcgbf(double lum, const double GB[10], double lx) const;
    double lmcgbf(double mc, const double GB[10]) const;
    double lheif(double m, double mhefl) const;
    double lhef(double m) const;
    double rminf(double m) const;
    double thef(double m, double mc, double mhefl) const;
    double tblf(double m, double mhefl, double mfgb) const;
    double lzahbf(double m, double mc, double mhefl) const;
    double rzahbf(double m, double mc, double mhefl) const;
    double lzhef(double m) const;
    double rzhef(double m) const;
    double themsf(double m) const;
    double rhehgf(double m, double lum, double rx, double lx) const;
    double rhegbf(double lum) const;
    double lpertf(double m, double mew) const;
    double rpertf(double m, double mew, double r, double rc) const;

    // Remnant mass computation (centralizes NS/BH/PPISN logic)
    RemnantResult compute_remnant(double mc, double mcbagb, double mt_presn,
                                  double mass0, bool from_he_star) const;

    // Main routines
    void star(int& kw, double mass, double mt, double& tm, double& tn,
              double tscls[20], double lums[10], double GB[10]) const;
    void hrdiag(double& mass, double& aj, double& mt, double& tm, double& tn,
                double tscls[20], double lums[10], double GB[10],
                double& r, double& lum, int& kw, double& mc, double& rc,
                double& menv, double& renv) const;
    double mlwind(int kw, double lum, double r, double mt, double mc,
                  double rl, double z) const;
    void deltat(int kw, double age, double tm, double tn,
                const double tscls[20], double& dt, double& dtr) const;
};
